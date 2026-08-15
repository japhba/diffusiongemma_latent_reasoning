"""
DiffusionGemma denoising-trajectory inference worker.

Loads google/diffusiongemma-26B-A4B-it once and exposes a tiny HTTP API that
samples a single canvas and returns the FULL per-denoising-step trajectory
(per-position top-k tokens, confidence, entropy, pad/EOS mass) needed to drive
the three dashboard views from "How Transparent is DiffusionGemma?"
(arXiv 2606.20560): Summary, Line Graph, Sampling Table.

Steering exposed to the UI:
  * T  -> max_denoising_steps  (number of denoising steps)
  * C  -> model.config.canvas_length (canvas size; generate() reads it fresh)
  * temperature schedule (t_max -> t_min), entropy_bound, top_k, seed,
    early-stop on/off, thinking on/off.

Runs on a Slurm GPU node; the workbench Flask app proxies to it over the
cluster LAN (see app.py / serve-diffusiongemma). One generation at a time (GPU bound).
"""

import argparse
import base64
import json
import os
import socket
import threading
import time
from pathlib import Path

import torch
from flask import Flask, jsonify, request

# Explicit module-path imports (class/file names verified against transformers 5.12.x).
from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
from transformers.models.diffusion_gemma.generation_diffusion_gemma import (
    DiffusionGemmaGenerationConfig,
    EntropyBoundSamplerConfig,
)

HERE = Path(__file__).resolve().parent
MODEL_ID = os.environ.get("DIFFUSIONGEMMA_MODEL", "google/diffusiongemma-26B-A4B-it")


# --------------------------------------------------------------------------------------
# Tracing model: capture each denoising step's processed logits without duplicating the
# (subtle) sampler code. _denoising_step RETURNS self_conditioning_logits, which is
# exactly processed_logits cast to the embedding dtype -- so we read it from the return
# value and compute top-k / entropy / pad-mass on GPU, moving only [C, k] to CPU.
# --------------------------------------------------------------------------------------
class TracingDiffusionGemma(DiffusionGemmaForBlockDiffusion):
    """Wraps _denoising_step to (a) trace per-step distributions for the viz,
    (b) track two 'basin' token sets' per-position mass for the order parameter,
    (c) capture the full per-step carried state of a *donor* rollout, and
    (d) inject a donor's state into this rollout at a chosen step (cross-rollout
    steering). The carried state is the 4-tuple
        (current_canvas[B,C], argmax_canvas[B,C], self_conditioning_logits[B,C,V], finished[B])
    that the generate() loop feeds forward; the final output is the last argmax_canvas."""

    def _begin_trace(self, top_k: int, pad_ids: list[int], basin_a_ids: list[int] | None = None,
                     basin_b_ids: list[int] | None = None, light: bool = False,
                     capture_state: bool = False, inject: dict | None = None,
                     seed_canvas=None, seed_step: int = 0,
                     clamp_canvas=None, clamp_mask=None, clamp_from_step: int = 0):
        self._trace = []
        self._trace_active = True
        self._trace_k = top_k
        self._trace_light = light          # light: skip topk/entropy (rollout sweeps only need basin mass)
        self._pad_ids = torch.tensor(sorted(set(pad_ids)), device=self.device)
        self._basin_a = torch.tensor(sorted(set(basin_a_ids)), device=self.device) if basin_a_ids else None
        self._basin_b = torch.tensor(sorted(set(basin_b_ids)), device=self.device) if basin_b_ids else None
        self._captured = [] if capture_state else None   # donor: per-step state, on CPU
        self._inject = inject                            # {step, donor_states, alpha, inject_canvas} or None
        # seed_canvas: a [1,C] LongTensor to FORCE into the canvas at denoising step `seed_step` (e.g. a
        # wrong/flawed CoT we plant, then watch denoising try to rewrite it). Self-conditioning is zeroed
        # at that step. seed_step is the NOISE-LEVEL knob: planting early (high remaining temperature) lets
        # the model refactor freely; planting late (low temperature) leaves it little room to escape.
        self._seed_canvas = seed_canvas
        self._seed_step = int(seed_step)
        # clamp: HOLD a set of canvas positions pinned to clamp_canvas at EVERY step >= clamp_from_step
        # (the model may never heal them). Unlike seed (a one-shot plant the model then rewrites), clamp
        # forces the rest of the canvas -- e.g. the answer slot -- to be denoised in the persistent context
        # of a FIXED (e.g. corrupted) CoT. This isolates the CoT->answer read channel: d(answer)/d(noise).
        self._clamp_canvas = clamp_canvas        # [1,C] LongTensor or None
        self._clamp_mask = clamp_mask            # [C] bool tensor (positions to pin) or None
        # NB: do NOT reset _no_commit here — _begin_trace runs after the request sets it
        # (same reason _s_bump is not reset here); only seed the RNG if it is missing.
        if not hasattr(self, "_nc_gen"):
            self._nc_gen = torch.Generator().manual_seed(1234)
        self._clamp_from_step = int(clamp_from_step)
        self._call_i = 0

    def _end_trace(self):
        self._trace_active = False
        return self._trace

    # ---- batched path (many rollouts per generate; for /barrier) ----
    def _begin_batch(self, basin_a_ids, basin_b_ids, capture_state=False, inject=None):
        self._batch_active = True
        self._batch = []
        self._batch_call_i = 0
        self._batch_a = torch.tensor(sorted(set(basin_a_ids)), device=self.device)
        self._batch_b = torch.tensor(sorted(set(basin_b_ids)), device=self.device)
        self._batch_captured = [] if capture_state else None
        self._batch_inject = inject

    def _end_batch(self):
        self._batch_active = False
        return self._batch, self._batch_captured

    # ---- (g) cloze probe: exact first-step distribution readout on a caller-built canvas ----
    # Standalone mode (no _begin_trace): initialize every attribute _denoising_step touches
    # unconditionally, so /cloze works on a fresh worker before any /sample ran.
    def _begin_probe(self, query_ids: list[int], top_k: int = 12):
        self._probe_active = True
        self._probe_q = torch.tensor(sorted(set(int(q) for q in query_ids)), dtype=torch.long,
                                     device=self.device) if query_ids else None
        self._probe_k = int(top_k)
        self._probe_out = []
        self._trace_active = False
        self._inject = None
        self._captured = None
        self._seed_canvas = None
        self._clamp_mask = None
        self._call_i = 0
        self._no_commit = None
        self._nc_gen = torch.Generator().manual_seed(1234)

    def _end_probe(self):
        self._probe_active = False
        return self._probe_out

    def _probe_step(self, out):
        probs = torch.softmax(out[2].float(), dim=-1)                     # [B,C,V]; t=1 -> exact model softmax
        topp, topi = probs.topk(self._probe_k, dim=-1)
        entropy = -(probs.clamp_min(1e-12).log() * probs).sum(-1)
        pm = probs.mean(0)                                                 # noise-averaged distribution [C,V]
        mtopp, mtopi = pm.topk(self._probe_k, dim=-1)
        rec = {
            "entropy": entropy.to("cpu", torch.float32),                   # [B,C]
            "topk_ids": topi.to("cpu", torch.int32),                       # [B,C,k]
            "topk_p": topp.to("cpu", torch.float32),
            "mean_topk_ids": mtopi.to("cpu", torch.int32),                 # [C,k]
            "mean_topk_p": mtopp.to("cpu", torch.float32),
        }
        if self._probe_q is not None:
            rec["qids"] = self._probe_q.to("cpu")
            rec["qp"] = probs.index_select(-1, self._probe_q).to("cpu", torch.float32)   # [B,C,Q]
        self._probe_out.append(rec)

    def _batch_step(self, out):
        si = self._batch_call_i
        if self._batch_inject is not None and si == self._batch_inject["step"]:
            cur, arg, scl, fin = out
            a = float(self._batch_inject["alpha"])
            scl = (1.0 - a) * scl + a * self._batch_inject["donor_S"].to(scl.device, scl.dtype)
            if self._batch_inject["inject_canvas"] and self._batch_inject.get("donor_canvas") is not None:
                cur = self._batch_inject["donor_canvas"].to(cur.device); arg = cur
            out = (cur, arg, scl, fin)
        if self._batch_captured is not None:
            self._batch_captured.append({
                "self_cond": out[2].detach().to("cpu", torch.bfloat16),   # [B,C,V]
                "current": out[0].detach().to("cpu"),                      # [B,C]
            })
        probs = torch.softmax(out[2].float(), dim=-1)
        self._batch.append({
            "pa": probs.index_select(-1, self._batch_a).sum(-1).to("cpu", torch.float32),  # [B,C]
            "pb": probs.index_select(-1, self._batch_b).sum(-1).to("cpu", torch.float32),  # [B,C]
            "argmax": out[1].detach().to("cpu", torch.int32),                               # [B,C]
        })
        self._batch_call_i += 1
        return out

    # ---- (f) logit-lens: per-layer residual capture across denoising steps ----
    # Register a forward hook on every decoder layer; while _lens_capturing is True (set only
    # around the real _denoising_step decoder pass) each hook appends that layer's residual
    # stream output [B,C,hidden] for the current step. The per-layer logit lens is then
    #   softcap * tanh( lm_head(decoder.norm(h_layer)) / softcap )
    # i.e. the SAME final projection the model applies after layer 30, applied to every layer.
    def _begin_lens(self, layers=None):
        self._lens_active = True
        self._lens_capturing = False
        dec = self.model.decoder
        want = set(range(len(dec.layers))) if layers is None else set(int(l) for l in layers)
        self._lens_buf = {i: [] for i in want}
        self._lens_handles = []
        def mk(i):
            def hook(_m, _inp, out):
                if getattr(self, "_lens_capturing", False):
                    h = out[0] if isinstance(out, tuple) else out
                    self._lens_buf[i].append(h.detach().to(torch.bfloat16))   # keep on GPU
            return hook
        for i, layer in enumerate(dec.layers):
            if i in want:
                self._lens_handles.append(layer.register_forward_hook(mk(i)))

    def _end_lens(self):
        for h in self._lens_handles:
            h.remove()
        self._lens_active = False
        self._lens_handles = []
        return self._lens_buf

    def _denoising_step(self, *args, **kwargs):
        if getattr(self, "_lens_active", False):
            self._lens_capturing = True
        out = super()._denoising_step(*args, **kwargs)
        if getattr(self, "_lens_active", False):
            self._lens_capturing = False
        # /energy: capture the processed logits of the (single) readout step
        if getattr(self, "_energy_active", False):
            self._energy_buf.append(out[2].detach())
        # S-mode nulls: "echo" replaces S with a one-hot of the CURRENT canvas token (the
        # recurrent pathway stays active but carries nothing beyond the canvas — the
        # memoryless-sampler null); "flat" zeroes it (uniform soft input).
        if getattr(self, "_s_mode", None):
            scl = out[2]
            if self._s_mode == "flat":
                out = (out[0], out[1], torch.zeros_like(scl), out[3])
            else:
                new = torch.full_like(scl, -30000.0)
                new.scatter_(-1, out[0].to(new.device).unsqueeze(-1), 0.0)
                out = (out[0], out[1], new, out[3])
        # S rank-ops: ON-MANIFOLD negotiation probe — manipulate the S channel's OWN candidates
        # at a position, per step (ranking re-evaluated each step): "swap12" promotes the current
        # runner-up to the lead by exchanging the two logit values; "drop" deletes the rank-r
        # candidate. Sampling and all other positions untouched.
        if getattr(self, "_s_rankops", None):
            new = None
            for op in self._s_rankops:
                a, b = op.get("steps") or (0, 10 ** 9)
                if not (a <= self._call_i < b):
                    continue
                if new is None:
                    new = out[2].clone()
                row = new[0, op["pos"]]
                vals, idx = row.topk(max(int(op.get("rank", 2)), 2))
                if op["op"] == "swap12":
                    row[idx[0]] = vals[1]
                    row[idx[1]] = vals[0]
                elif op["op"] == "drop":
                    row[idx[int(op.get("rank", 2)) - 1]] = -30000.0
            if new is not None:
                out = (out[0], out[1], new, out[3])
        # No-commit probe: hold chosen positions FLUID past the point where the EB sampler
        # would normally accept them, to test whether CANVAS COMMITMENT (rather than the S^t
        # sheet) is what makes late interventions fail. The two mechanisms are separable and
        # NOT equivalent:
        #   "logits" — flatten the returned self-conditioning logits (uniform => max entropy),
        #              so the next step reads no proposal at that position. This alone does NOT
        #              un-commit: out[0] (current_canvas) still carries the accepted token
        #              forward, so it mainly fools the `committed` proxy.
        #   "canvas" — overwrite the fed-forward canvas token with a fresh uniform-random id.
        #              THIS is what actually un-commits, since the canvas is the carrier at
        #              accepted positions (S^t is the carrier only at fluid ones).
        if getattr(self, "_no_commit", None):
            cur = scl = arg = None
            V = out[2].shape[-1]
            for nc_ in self._no_commit:
                a, b = nc_.get("steps") or (0, 10 ** 9)
                if not (a <= self._call_i < b):
                    continue
                mode = nc_.get("mode", "both")
                if mode in ("logits", "both"):
                    if scl is None:
                        scl = out[2].clone()
                    scl[0, nc_["pos"], :] = 0.0
                if mode in ("canvas", "both"):
                    if cur is None:
                        cur = out[0].clone()
                    if arg is None:
                        arg = out[1].clone()
                    rid = int(torch.randint(0, V, (1,), generator=self._nc_gen).item())
                    cur[0, nc_["pos"]] = rid
                    arg[0, nc_["pos"]] = rid
            if cur is not None or scl is not None or arg is not None:
                out = (cur if cur is not None else out[0],
                       arg if arg is not None else out[1],
                       scl if scl is not None else out[2], out[3])
        # S-bump / S-pin: negotiation probe. Additively bias (delta) or hard-pin (one-hot) the
        # S^t proposal at chosen positions, leaving sampling and all other positions free —
        # then watch whether the rest of the canvas re-negotiates around the pinned proposal.
        if getattr(self, "_s_bump", None):
            new = None
            for bmp in self._s_bump:
                a, b = bmp.get("steps") or (0, 10 ** 9)
                if not (a <= self._call_i < b):
                    continue
                if new is None:
                    new = out[2].clone()
                if bmp.get("delta") is None:      # pin: one-hot S at this position
                    new[0, bmp["pos"], :] = -30000.0
                    new[0, bmp["pos"], bmp["id"]] = 0.0
                else:
                    new[0, bmp["pos"], bmp["id"]] += float(bmp["delta"])
            if new is not None:
                out = (out[0], out[1], new, out[3])
        # S-only truncation: THIS step sampled from the full distribution; only the returned
        # self_conditioning_logits (the S^t soft channel fed to the next step) are top-k
        # truncated (non-top-k -> -3e4; next step's softmax redistributes the mass).
        # Optional scoping: positions (bool mask [C]) and step window [a,b) — mechanism probes.
        if getattr(self, "_trunc_s_only", None):
            cfg = self._trunc_s_only
            a, b = cfg.get("steps") or (0, 10 ** 9)
            if a <= self._call_i < b:
                scl = out[2]
                kth = scl.topk(int(cfg["k"]), dim=-1).values[..., -1:]
                tr = scl.masked_fill(scl < kth, -30000.0)
                if cfg.get("pos") is not None:
                    m = cfg["pos"].to(scl.device)                      # [C] bool
                    tr = torch.where(m.view(1, -1, 1), tr, scl)
                out = (out[0], out[1], tr, out[3])
        # S-channel recording: per step, top-N ids+logprobs of the S distribution per position,
        # plus exact rank & logprob of a small probe-id set (rank computed against the FULL vocab).
        if getattr(self, "_s_rec", None) is not None:
            lsm = torch.log_softmax(out[2][0].float(), dim=-1)         # [C, V]
            tp, ti = lsm.topk(int(self._s_rec["topk"]), dim=-1)
            rec = {"ids": ti.to("cpu", torch.int32), "lp": tp.to("cpu", torch.float16)}
            pids = self._s_rec.get("probe")
            if pids is not None:
                plp = lsm[:, pids]                                     # [C, P]
                prk = torch.stack([(lsm > plp[:, j].unsqueeze(-1)).sum(-1)
                                   for j in range(pids.shape[0])], -1)  # [C, P]
                rec["probe_lp"] = plp.to("cpu", torch.float16)
                rec["probe_rank"] = prk.to("cpu", torch.int32)
            self._s_rec["buf"].append(rec)
        if getattr(self, "_batch_active", False):
            return self._batch_step(out)
        if getattr(self, "_probe_active", False):
            self._probe_step(out)
            self._call_i += 1
            return out
        si = self._call_i

        # ---- (e) seed: FORCE a planted canvas (e.g. a flawed CoT) at step `seed_step`, zero self-cond ----
        if getattr(self, "_seed_canvas", None) is not None and si == getattr(self, "_seed_step", 0):
            cur, arg, scl, fin = out
            sc = self._seed_canvas.to(arg.device)
            out = (sc, sc, torch.zeros_like(scl), fin)

        # ---- (e2) clamp: re-pin masked positions to clamp_canvas EVERY step (no healing) ----
        if getattr(self, "_clamp_mask", None) is not None and si >= self._clamp_from_step:
            cur, arg, scl, fin = out
            m = self._clamp_mask.to(arg.device); cc = self._clamp_canvas.to(arg.device)
            cur = cur.clone(); arg = arg.clone(); scl = scl.clone()
            cur[:, m] = cc[:, m]; arg[:, m] = cc[:, m]
            scl[:, m, :] = 0.0   # drop self-conditioning at pinned positions so the next step reads the pin
            out = (cur, arg, scl, fin)

        # ---- (d) injection: splice the donor's step-si state into this rollout ----
        if self._inject is not None and si == self._inject["step"]:
            cur, arg, scl, fin = out
            d = self._inject["donor_states"][si]
            a = float(self._inject["alpha"])
            scl = (1.0 - a) * scl + a * d["self_cond"].to(scl.device, scl.dtype)   # blend continuous bottleneck S
            if self._inject["inject_canvas"]:
                cur = d["current_canvas"].to(cur.device)
                arg = d["argmax_canvas"].to(arg.device)
            out = (cur, arg, scl, fin)

        # ---- (c) donor capture: keep the carried state (CPU, bf16 for the big logits) ----
        if self._captured is not None:
            self._captured.append({
                "current_canvas": out[0].detach().to("cpu"),
                "argmax_canvas": out[1].detach().to("cpu"),
                "self_cond": out[2].detach().to("cpu", torch.bfloat16),
            })

        # ---- (a/b) trace for viz + basin order parameter ----
        if self._trace_active:
            argmax_canvas = out[1]
            logits = out[2].float()
            probs = torch.softmax(logits, dim=-1)
            b = 0
            rec = {"argmax": argmax_canvas[b].to("cpu", torch.int32)}
            if self._basin_a is not None:
                rec["pa"] = probs[b].index_select(-1, self._basin_a).sum(-1).to("cpu", torch.float32)  # [C]
                rec["pb"] = probs[b].index_select(-1, self._basin_b).sum(-1).to("cpu", torch.float32)  # [C]
            if not self._trace_light:
                topp, topi = probs.topk(self._trace_k, dim=-1)
                entropy = -(probs.clamp_min(1e-12).log() * probs).sum(-1)
                pad_prob = probs.index_select(-1, self._pad_ids).sum(-1)
                # EB-committed detection: the post-EB canvas holds the accepted candidate
                # (drawn from the tempered softmax -> ~always in the top-k) at accepted
                # positions and a uniform-random renoise token (in top-k w.p. ~k/|V|)
                # elsewhere. The acceptance set itself isn't in the step's return, so this
                # membership test is the tightest available proxy (FP rate ~64/262k).
                cur = out[0]
                committed = (topi[b] == cur[b].unsqueeze(-1)).any(-1) | (cur[b] == argmax_canvas[b])
                rec.update({
                    "topk_ids": topi[b].to("cpu", torch.int32),
                    "topk_p": topp[b].to("cpu", torch.float32),
                    "conf": topp[b, :, 0].to("cpu", torch.float32),
                    "entropy": entropy[b].to("cpu", torch.float32),
                    "pad_prob": pad_prob[b].to("cpu", torch.float32),
                    "committed": committed.to("cpu"),
                    "mean_entropy": float(entropy[b].mean()),
                })
            self._trace.append(rec)

        self._call_i += 1
        return out


# --------------------------------------------------------------------------------------
# App / model state
# --------------------------------------------------------------------------------------
app = Flask(__name__)
STATE = {"model": None, "processor": None, "lock": threading.Lock(), "last_request": time.time()}

# Fitted Jacobian-lens transports J_l = E[dh_L29/dh_l] (jlens_dg_fit.py), applied as
# lens_l(h) = softcap_tanh(lm_head(norm(J_l @ h))). Cached per (lens_name, layer) on GPU bf16.
LENS_DIR = Path(os.environ.get("DG_LENS_DIR", str(Path(__file__).parent)))
_JLENS_CACHE: dict = {}


def _jlens_J(lens_name: str, layer: int, device) -> torch.Tensor:
    key = (lens_name, int(layer))
    if key not in _JLENS_CACHE:
        pack_key = ("__pack__", lens_name)
        if pack_key not in _JLENS_CACHE:
            _JLENS_CACHE[pack_key] = torch.load(LENS_DIR / f"lens_{lens_name}.pt",
                                                map_location="cpu", weights_only=False)["J"]
        _JLENS_CACHE[key] = _JLENS_CACHE[pack_key][int(layer)].to(device, torch.bfloat16)
    return _JLENS_CACHE[key]


def _unembed(model, residual, softcap=True):
    """Residual [..., d] -> fp32 logits via the model's own final norm + lm_head (+ softcap).
    Matches jlens_dg_common.unembed / run_logitlens precision (matmul in head dtype)."""
    w = model.lm_head.weight
    logits = model.lm_head(model.model.decoder.norm(residual.to(w.dtype))).float()
    if softcap:
        sc = float(model.final_logit_softcapping)
        logits = sc * torch.tanh(logits / sc)
    return logits


def load_model():
    print(f"[worker] loading {MODEL_ID} ...", flush=True)
    t0 = time.time()
    # Cap our share of the GPU so we coexist with a co-resident model (the vLLM judge on Node V's
    # shared B200) without OOM. DG_MEM_FRACTION = fraction of TOTAL GPU memory this worker may
    # allocate; set it to ~0.45 on Node V (≈ equal split with the judge's --gpu-memory-utilization,
    # leaving slack). Default 0 = uncapped (dedicated cluster GPUs need no cap). See ~/.claude/CLAUDE.md.
    frac = float(os.environ.get("DG_MEM_FRACTION", "0") or 0)
    if frac > 0 and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(frac, 0)
        print(f"[worker] DG_MEM_FRACTION={frac} -> capped to ~{frac * 100:.0f}% of GPU memory", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    # Force the whole model onto the single allocated GPU. device_map="auto" silently offloads to
    # CPU when the visible GPU looks momentarily contended (seen on shared cluster nodes), which
    # makes batched generation unusably slow; pin to cuda:0 when CUDA is present.
    dev_map = {"": "cuda:0"} if torch.cuda.is_available() else "auto"
    kw = {}
    # torch<=2.8's _grouped_mm kernel is Hopper-only (exactly SM90), but transformers' capability
    # check passes on Blackwell (10,0) >= (9,0) -> RuntimeError mid-forward. Fall back to batched_mm
    # experts there; torch>=2.9/2.10 grouped kernels support SM80+ and keep the fast path.
    if (torch.cuda.is_available() and torch.cuda.get_device_capability() != (9, 0)
            and not hasattr(torch.nn.functional, "grouped_mm")):
        kw["experts_implementation"] = "batched_mm"
        print("[worker] Blackwell + torch<=2.8: experts_implementation=batched_mm", flush=True)
    model = TracingDiffusionGemma.from_pretrained(MODEL_ID, dtype="auto", device_map=dev_map, **kw)
    model.eval()
    STATE["model"], STATE["processor"] = model, processor
    print(f"[worker] loaded in {time.time() - t0:.1f}s  device={model.device}  canvas={model.config.canvas_length}", flush=True)


def _id2str(tokenizer, ids: set[int]) -> dict[str, str]:
    # raw sentencepiece pieces (keep the leading-space marker visible) -> map on the frontend
    out = {}
    for i in sorted(ids):
        piece = tokenizer.convert_ids_to_tokens(int(i))
        out[str(int(i))] = piece if piece is not None else f"<{i}>"
    return out


@app.get("/health")
def health():
    m = STATE["model"]
    return jsonify(
        ready=m is not None,
        model=MODEL_ID,
        canvas_length=(m.config.canvas_length if m is not None else None),
        device=str(m.device) if m is not None else None,
        idle_seconds=round(time.time() - STATE["last_request"], 1),
    )


@app.get("/tasks")
def tasks():
    return jsonify(json.loads((HERE / "tasks.json").read_text()))


@app.post("/embed_tokens")
def embed_tokens():
    """Input-embedding rows for a list of token ids (the model's own semantic geometry,
    for clustering top-k candidates offline). Returns L2-normalized fp16 vectors, base64."""
    STATE["last_request"] = time.time()
    req = request.get_json(force=True)
    ids = [int(i) for i in req["ids"]]
    with STATE["lock"]:
        emb = STATE["model"].get_input_embeddings().weight
        vecs = emb[torch.tensor(ids, dtype=torch.long, device=emb.device)].detach().float()
        vecs = torch.nn.functional.normalize(vecs, dim=-1).to(torch.float16).cpu()
    return jsonify({"n": len(ids), "d": vecs.shape[1],
                    "b64": base64.b64encode(vecs.numpy().tobytes()).decode()})


def sample_trajectory(model, processor, req: dict) -> dict:
    """Pure sampler: runs one canvas and returns the full trajectory JSON dict.
    Shared by the /sample route and the offline precompute script."""
    prompt = req["prompt"]
    T = int(req.get("T", 16))
    C = int(req.get("C", 128))
    top_k = int(req.get("top_k", 8))
    t_max = float(req.get("t_max", 0.8))
    t_min = float(req.get("t_min", 0.4))
    entropy_bound = float(req.get("entropy_bound", 0.1))
    seed = int(req.get("seed", 0))
    early_stop = bool(req.get("early_stop", False))
    enable_thinking = bool(req.get("enable_thinking", False))
    max_new_tokens = int(req.get("max_new_tokens", C))  # default: single canvas
    init_text = req.get("init_text")        # optional: plant this text into the canvas (forced flawed CoT)
    init_step = int(req.get("init_step", 0))  # denoising step to plant at (noise-level knob; 0 = start)
    clamp_text = req.get("clamp_text")      # optional: PIN this text into the canvas prefix every step (no heal)
    clamp_from_step = int(req.get("clamp_from_step", 0))  # start pinning at this step (0 = whole rollout)

    tok = processor.tokenizer
    pad_id = model.config.eos_token_id if isinstance(model.config.eos_token_id, int) else None
    eos_ids = model.config.eos_token_id
    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    pad_token_id = getattr(model.config, "pad_token_id", 0) or 0
    pad_ids = [pad_token_id] + list(eos_ids)

    with STATE["lock"]:
        torch.manual_seed(seed)
        model.config.canvas_length = C  # generate() reads this fresh -> steers canvas size

        messages = [{"role": "user", "content": prompt}]
        enc = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=enable_thinking,
        ).to(model.device)

        gen_config = DiffusionGemmaGenerationConfig(
            max_new_tokens=max_new_tokens,
            max_denoising_steps=T,
            sampler_config=EntropyBoundSamplerConfig(entropy_bound=entropy_bound),
            t_min=t_min,
            t_max=t_max,
            # early_stop OFF -> make both adaptive-stopping gates unreachable so all T steps run
            stability_threshold=(1 if early_stop else T + 1),
            confidence_threshold=(0.005 if early_stop else 1e-9),
            pad_token_id=pad_token_id,
            eos_token_id=eos_ids,
        )

        # donor injection: draw a NATIVE rollout from the model itself (donor_seed), capture its FINAL
        # carried state (canvas + self-conditioning), and inject it at `init_step` into this rollout.
        # donor_alpha blends the donor self-conditioning (1.0 = full native S = sticky; 0.0 = the model's
        # own S, i.e. canvas-only). This is the native-donor analogue of the hand-written init_text plant.
        donor_inject = None
        if req.get("donor_seed") is not None:
            dseed = int(req["donor_seed"]); dalpha = float(req.get("donor_alpha", 1.0))
            # the donor may be drawn from a DIFFERENT (sibling) prompt, so we can manufacture a
            # confident donor for an answer the recipient strongly disfavors (basin-imbalance probes).
            denc = enc
            if req.get("donor_prompt"):
                denc = processor.apply_chat_template(
                    [{"role": "user", "content": req["donor_prompt"]}], tokenize=True,
                    add_generation_prompt=True, return_dict=True, return_tensors="pt",
                    enable_thinking=enable_thinking).to(model.device)
            # optionally HARVEST the donor at a DIFFERENT (e.g. LOW) temperature than the recipient, so we can
            # plant a confident-WRONG converged state (a genuine local minimum) and then re-denoise it at the
            # recipient's own (possibly hotter) temperature -> does re-noising escape the wrong basin?
            dgc = gen_config
            if req.get("donor_t_max") is not None:
                dgc = DiffusionGemmaGenerationConfig(
                    max_new_tokens=max_new_tokens, max_denoising_steps=T,
                    sampler_config=EntropyBoundSamplerConfig(entropy_bound=float(req.get("donor_entropy_bound", entropy_bound))),
                    t_min=float(req.get("donor_t_min", t_min)), t_max=float(req["donor_t_max"]),
                    stability_threshold=(1 if early_stop else T + 1),
                    confidence_threshold=(0.005 if early_stop else 1e-9),
                    pad_token_id=pad_token_id, eos_token_id=eos_ids)
            torch.manual_seed(dseed)
            model._begin_trace(top_k=1, pad_ids=pad_ids, capture_state=True)
            model.generate(**denc, generation_config=dgc)
            model._end_trace()
            # donor_capture_step: take the donor's carried state AFTER step j instead of its final
            # state -> with init_step=j this is an exact mid-trajectory PROMPT SWAP (native state,
            # same schedule, only the conditioning changes).
            dcap = req.get("donor_capture_step")
            dfin = model._captured[int(dcap) if dcap is not None else -1]
            model._captured = None
            if req.get("donor_edit_text"):
                # STICKY semantic-repair probe: keep the donor's native committed self-conditioning S but
                # swap its CANVAS for an EDITED one (e.g. a non-sequitur planted into the coherent trace).
                # S "remembers" the coherent trace, the canvas carries the error -> does commitment heal it?
                eids = tok(req["donor_edit_text"], add_special_tokens=False).input_ids[:C]
                erow = torch.tensor([list(eids) + [pad_token_id] * (C - len(eids))], dtype=torch.long)
                dfin = dict(dfin); dfin["current_canvas"] = erow.clone(); dfin["argmax_canvas"] = erow.clone()
            donor_inject = {"step": init_step, "alpha": dalpha,
                            "inject_canvas": bool(req.get("donor_inject_canvas", True)),
                            "donor_states": {init_step: dfin}}
            torch.manual_seed(seed)

        seed_canvas = None
        if init_text and donor_inject is None:
            # tokenize the planted text and lay it into a [1,C] canvas (rest = pad), to FORCE into
            # the canvas at denoising step 0 (then denoising may rewrite it -> approach/answer flip).
            ids = tok(init_text, add_special_tokens=False).input_ids[:C]
            row = list(ids) + [pad_token_id] * (C - len(ids))
            seed_canvas = torch.tensor([row], dtype=torch.long, device=model.device)

        clamp_canvas = clamp_mask = None
        if req.get("clamp_ids") is not None:
            # surgical clamp: caller supplies the FULL [C] canvas + the exact positions to pin. Lets us hold
            # the CoT positions at corrupted token values while the ANSWER positions re-denoise freely.
            crow = list(req["clamp_ids"])[:C]; crow = crow + [pad_token_id] * (C - len(crow))
            clamp_canvas = torch.tensor([crow], dtype=torch.long, device=model.device)
            clamp_mask = torch.zeros(C, dtype=torch.bool, device=model.device)
            for p in req.get("clamp_positions", []):
                if 0 <= int(p) < C:
                    clamp_mask[int(p)] = True
        elif clamp_text:
            coff = int(req.get("clamp_offset", 0))
            cids = tok(clamp_text, add_special_tokens=False).input_ids[: C - coff]
            crow = [pad_token_id] * C
            crow[coff: coff + len(cids)] = cids
            clamp_canvas = torch.tensor([crow], dtype=torch.long, device=model.device)
            clamp_mask = torch.zeros(C, dtype=torch.bool, device=model.device)
            clamp_mask[coff: coff + len(cids)] = True

        # ---- interventions (constraint functional-relevance study) ----
        # ban_ids: hard-suppress these token ids in the OUTPUT logits at every denoising step
        #   (lm_head hook; the sampler's tempered softmax redistributes the mass over the rest).
        # state_ablate {layer, ids, lens}: at every denoising forward, project the layer-l residual
        #   off the (orthonormalized) lens directions of these tokens — d_t = J_l^T w_t through the
        #   fitted jlens transport; w_t directly at the last layer (identity transport).
        _iv_handles = []
        # trunc_k: hard top-k truncation of the OUTPUT distribution at every position/step
        # (all non-top-k logits -> -3e4; the sampler's tempered softmax renormalizes the mass).
        # NOTE: /sample's `top_k` only sets the trace recording depth, NOT the sampler.
        # trunc_scope: "both" (default) truncates at lm_head, upstream of BOTH the sampler and the
        # self-conditioning channel; "s_only" leaves sampling untouched and truncates only the S^t
        # soft channel between steps (see _denoising_step) — isolates WHERE the tail matters.
        model._trunc_s_only = None
        if req.get("trunc_k"):
            _tkv = int(req["trunc_k"])
            if req.get("trunc_scope", "both") == "s_only":
                _pos = None
                if req.get("trunc_s_positions") is not None:
                    _pos = torch.zeros(C, dtype=torch.bool)
                    for _p in req["trunc_s_positions"]:
                        if 0 <= int(_p) < C:
                            _pos[int(_p)] = True
                _stp = req.get("trunc_s_steps")
                model._trunc_s_only = {"k": _tkv, "pos": _pos,
                                       "steps": tuple(int(x) for x in _stp) if _stp else None}
            else:
                _tw = req.get("trunc_steps")   # [a,b): truncate only inside this step window
                _ta, _tb = (int(_tw[0]), int(_tw[1])) if _tw else (0, 10 ** 9)
                def _trunc_hook(_m, _i, out):
                    if not (_ta <= getattr(model, "_call_i", 0) < _tb):
                        return out
                    kth = out.topk(_tkv, dim=-1).values[..., -1:]
                    out.masked_fill_(out < kth, -30000.0)
                    return out
                _iv_handles.append(model.lm_head.register_forward_hook(_trunc_hook))
        model._s_mode = req.get("s_mode")   # None | "echo" | "flat"
        model._s_rankops = None
        if req.get("s_rankops"):
            model._s_rankops = [dict(pos=int(x["pos"]), op=str(x["op"]), rank=int(x.get("rank", 2)),
                                     steps=(tuple(int(v) for v in x["steps"]) if x.get("steps") else None))
                                for x in req["s_rankops"]]
        model._s_bump = None
        if req.get("s_bump"):
            model._s_bump = [dict(pos=int(x["pos"]), id=int(x["id"]),
                                  delta=(None if x.get("pin") else float(x.get("delta", 8.0))),
                                  steps=(tuple(int(v) for v in x["steps"]) if x.get("steps") else None))
                             for x in req["s_bump"]]
        model._no_commit = None
        if req.get("no_commit"):
            model._no_commit = [dict(pos=int(x["pos"]), mode=str(x.get("mode", "both")),
                                     steps=(tuple(int(v) for v in x["steps"]) if x.get("steps") else None))
                                for x in req["no_commit"]]
            model._nc_gen = torch.Generator().manual_seed(int(req.get("no_commit_seed", 1234)))
        model._s_rec = None
        if req.get("s_topk_record"):
            model._s_rec = {"topk": int(req["s_topk_record"]), "buf": [],
                            "probe": (torch.tensor([int(x) for x in req["s_probe_ids"]],
                                                   device=model.device, dtype=torch.long)
                                      if req.get("s_probe_ids") else None)}
        _ban = [int(x) for x in req.get("ban_ids") or []]
        if _ban:
            _bidx = torch.tensor(sorted(set(_ban)), device=model.device, dtype=torch.long)
            def _ban_hook(_m, _i, out):
                out.index_fill_(-1, _bidx, -30000.0)
                return out
            _iv_handles.append(model.lm_head.register_forward_hook(_ban_hook))
        _sab = req.get("state_ablate")
        if _sab:
            _sl = int(_sab["layer"])
            _sw = model.lm_head.weight[torch.tensor(sorted(set(int(x) for x in _sab["ids"])),
                                                    device=model.device)].float()      # [k, d_out]
            if _sl != len(model.model.decoder.layers) - 1:
                _sw = _sw @ _jlens_J(_sab.get("lens", "pooled"), _sl, model.device).float()  # rows = J^T w_t
            _Q = torch.linalg.qr(_sw.T)[0]                                              # [d, k'] orthonormal
            # optional scoping: steps [a,b) on the denoising-step counter, positions = canvas indices
            _sab_steps = tuple(int(v) for v in _sab["steps"]) if _sab.get("steps") else None
            _sab_pos = (torch.tensor([int(x) for x in _sab["positions"]], device=model.device,
                                     dtype=torch.long) if _sab.get("positions") else None)
            def _ab_hook(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                if h.shape[1] != C:                       # only the canvas block, not prompt prefill
                    return out
                if _sab_steps is not None:
                    ci = getattr(model, "_call_i", 0)
                    if not (_sab_steps[0] <= ci < _sab_steps[1]):
                        return out
                hf = h.float()
                if _sab_pos is not None:
                    sub = hf[:, _sab_pos]
                    hf = hf.clone()
                    hf[:, _sab_pos] = sub - (sub @ _Q) @ _Q.T
                    h2 = hf.to(h.dtype)
                else:
                    h2 = (hf - (hf @ _Q) @ _Q.T).to(h.dtype)
                return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
            _iv_handles.append(model.model.decoder.layers[_sl].register_forward_hook(_ab_hook))

        model._begin_trace(top_k=top_k, pad_ids=pad_ids, seed_canvas=seed_canvas, seed_step=init_step,
                           inject=donor_inject, clamp_canvas=clamp_canvas, clamp_mask=clamp_mask,
                           clamp_from_step=clamp_from_step)
        capture_layers = [int(l) for l in req.get("capture_layers") or []]
        jlens_layers = [int(l) for l in req.get("jlens_layers") or []]
        hook_layers = sorted(set(capture_layers) | set(jlens_layers))
        if hook_layers:
            model._begin_lens(layers=None if capture_layers else hook_layers)
        t0 = time.time()
        try:
            out = model.generate(**enc, generation_config=gen_config)
        finally:
            for _h in _iv_handles:
                _h.remove()
            model._trunc_s_only = None
            model._s_bump = None
            model._s_rankops = None
            model._s_mode = None
            model._no_commit = None
        _s_rec_json = None
        if getattr(model, "_s_rec", None) is not None:
            _s_rec_json = {"topk": model._s_rec["topk"],
                           "ids": [r["ids"].tolist() for r in model._s_rec["buf"]],
                           "lp": [[[round(float(x), 2) for x in row] for row in r["lp"].tolist()]
                                  for r in model._s_rec["buf"]]}
            if model._s_rec.get("probe") is not None:
                _s_rec_json["probe_ids"] = model._s_rec["probe"].tolist()
                _s_rec_json["probe_lp"] = [[[round(float(x), 2) for x in row] for row in r["probe_lp"].tolist()]
                                           for r in model._s_rec["buf"]]
                _s_rec_json["probe_rank"] = [r["probe_rank"].tolist() for r in model._s_rec["buf"]]
            model._s_rec = None
        dt = time.time() - t0
        steps = model._end_trace()
        acts = None
        lens_json = None
        if hook_layers:
            lens_buf = model._end_lens()
            if capture_layers:
                acts = {}
                for l in capture_layers:
                    per = lens_buf[l]                                   # T x [1, C, d] bf16 (canvas-only)
                    h = torch.stack([per[t][0] for t in range(len(per))]).to(torch.float16).cpu()
                    acts[str(l)] = base64.b64encode(h.numpy().tobytes()).decode()
                acts = {"layers": capture_layers, "d": int(h.shape[-1]), "T": int(h.shape[0]),
                        "C": int(h.shape[1]), "dtype": "float16", "order": "T,C,d", "b64": acts}
            if jlens_layers:
                # per (layer, step): jlens (J-transported) AND vanilla logit-lens top-k per position
                jk = int(req.get("jlens_topk", 10))
                lens_name = req.get("jlens_lens", "pooled")
                lens_appeared: set[int] = set()
                per_layer = {}
                # full-vocab topic-retrieval probe: for each candidate topic string, score =
                # max over positions of the summed log-softmax of its token ids (NO top-k
                # truncation -> the whole vocabulary participates via the softmax normalizer).
                topic_probe = req.get("topic_probe") or []
                topic_pos = req.get("topic_pos")   # if set, score topics at THIS position only
                probe_ids = [torch.tensor(tok(tp, add_special_tokens=False).input_ids,
                                          device=model.device, dtype=torch.long) for tp in topic_probe]
                topic_scores = {} if topic_probe else None
                with torch.no_grad():
                    for l in jlens_layers:
                        # last decoder layer: transport to L29 is identity (fit pack has no J[29]);
                        # loglens there == the model's last-layer / output distribution.
                        J = None if l == len(model.model.decoder.layers) - 1 else _jlens_J(lens_name, l, model.device)
                        steps_l = []
                        for t in range(len(lens_buf[l])):
                            h = lens_buf[l][t][0]                       # [C, d] bf16 on GPU
                            rec = {}
                            for var, hh in (("jlens", h if J is None else h @ J.T), ("loglens", h)):
                                logits = _unembed(model, hh)            # [C, V] fp32
                                sm = torch.softmax(logits, dim=-1)
                                tp, ti = sm.topk(jk, dim=-1)
                                rec[f"{var}_ids"] = ti.tolist()
                                rec[f"{var}_p"] = [[round(float(x), 4) for x in row] for row in tp.tolist()]
                                lens_appeared.update(int(x) for row in ti.tolist() for x in row)
                                if topic_probe:
                                    lsm = torch.log_softmax(logits, dim=-1)   # [C, V]
                                    scores = [round(float((lsm[int(topic_pos), ids].sum() if topic_pos is not None else lsm[:, ids].sum(-1).max())), 4) for ids in probe_ids]
                                    topic_scores.setdefault(f"{var}_L{l}", []).append(scores)
                                del logits, sm
                            steps_l.append(rec)
                        per_layer[str(l)] = steps_l
                lens_json = {"layers": jlens_layers, "topk": jk, "lens": lens_name,
                             "per_layer": per_layer, "_appeared": lens_appeared}
                if topic_probe:
                    lens_json["topic_probe"] = topic_probe
                    lens_json["topic_scores"] = topic_scores

    # ---- decode + assemble compact JSON ----
    seq = out.sequences[0]
    prompt_len = enc["input_ids"].shape[1]
    canvas_ids = seq[prompt_len : prompt_len + C].tolist()  # first canvas (single-canvas regime)
    final_text = tok.decode(
        [t for t in canvas_ids if t not in pad_ids], skip_special_tokens=False
    )
    tpf = out.tokens_per_forward
    tpf = float(tpf[0]) if hasattr(tpf, "__len__") else float(tpf)

    appeared: set[int] = set(int(x) for x in canvas_ids)
    if _s_rec_json is not None:
        appeared.update(int(x) for step in _s_rec_json["ids"] for row in step for x in row)
        appeared.update(int(x) for x in _s_rec_json.get("probe_ids", []))
    if lens_json is not None:
        appeared |= lens_json.pop("_appeared")
    steps_json = []
    for si, s in enumerate(steps):
        topk_ids = s["topk_ids"].tolist()
        appeared.update(int(x) for row in topk_ids for x in row)
        appeared.update(int(x) for x in s["argmax"].tolist())
        steps_json.append(
            {
                "step": si,
                "mean_entropy": round(s["mean_entropy"], 4),
                "argmax": s["argmax"].tolist(),
                "conf": [round(x, 4) for x in s["conf"].tolist()],
                "entropy": [round(x, 4) for x in s["entropy"].tolist()],
                "pad_prob": [round(x, 4) for x in s["pad_prob"].tolist()],
                "topk_ids": topk_ids,
                "topk_p": [[round(p, 4) for p in row] for row in s["topk_p"].tolist()],
                "committed": [bool(x) for x in s["committed"].tolist()],
            }
        )

    return {
        "params": {
            "T": T, "C": C, "top_k": top_k, "t_max": t_max, "t_min": t_min,
            "entropy_bound": entropy_bound, "seed": seed,
            "early_stop": early_stop, "enable_thinking": enable_thinking,
            "max_new_tokens": max_new_tokens,
        },
        "prompt": prompt,
        "final_text": final_text,
        "final_ids": canvas_ids,
        "tokens_per_forward": round(tpf, 3),
        "num_steps": len(steps_json),
        "canvas_length": C,
        "pad_token_id": pad_token_id,
        "eos_token_ids": eos_ids,
        "gen_seconds": round(dt, 2),
        "id2str": _id2str(tok, appeared),
        "s_rec": _s_rec_json,
        "steps": steps_json,
        "acts": acts,
        "lens": lens_json,
    }


@app.post("/sample")
def sample():
    STATE["last_request"] = time.time()
    return jsonify(sample_trajectory(STATE["model"], STATE["processor"], request.get_json(force=True)))


# ======================================================================================
# Basins & steering: bimodal rollout distribution + cross-rollout state injection.
# A "basin" is a set of answer tokens (e.g. Heads/heads vs Tails/tails). The order
# parameter m(t) = log P(A) - log P(B) at the auto-detected decision position tracks
# which basin a rollout is heading toward across denoising steps.
# ======================================================================================
def _basin_ids(tok, strings: list[str]) -> list[int]:
    ids = set()
    for s in strings:
        for variant in (s, " " + s):
            enc = tok.encode(variant, add_special_tokens=False)
            if enc:
                ids.add(int(enc[0]))
    return sorted(ids)


def _run_once(model, processor, *, prompt, T, C, seed, top_k, t_max, t_min, entropy_bound,
              early_stop, enable_thinking, max_new_tokens, basin_a_ids, basin_b_ids,
              light, capture_state, inject):
    """One rollout (no lock — callers hold STATE['lock']). Returns its trace + final canvas."""
    tok = processor.tokenizer
    eos_ids = model.config.eos_token_id
    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    pad_token_id = getattr(model.config, "pad_token_id", 0) or 0
    pad_ids = [pad_token_id] + list(eos_ids)
    torch.manual_seed(seed)
    model.config.canvas_length = C
    enc = processor.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", enable_thinking=enable_thinking,
    ).to(model.device)
    gen_config = DiffusionGemmaGenerationConfig(
        max_new_tokens=max_new_tokens, max_denoising_steps=T,
        sampler_config=EntropyBoundSamplerConfig(entropy_bound=entropy_bound),
        t_min=t_min, t_max=t_max,
        stability_threshold=(1 if early_stop else T + 1),
        confidence_threshold=(0.005 if early_stop else 1e-9),
        pad_token_id=pad_token_id, eos_token_id=eos_ids,
    )
    model._begin_trace(top_k=top_k, pad_ids=pad_ids, basin_a_ids=basin_a_ids, basin_b_ids=basin_b_ids,
                       light=light, capture_state=capture_state, inject=inject)
    out = model.generate(**enc, generation_config=gen_config)
    steps = model._end_trace()
    captured = model._captured
    prompt_len = enc["input_ids"].shape[1]
    canvas_ids = out.sequences[0][prompt_len:prompt_len + C].tolist()
    final_text = tok.decode([t for t in canvas_ids if t not in set(pad_ids)], skip_special_tokens=False)
    return {"steps": steps, "captured": captured, "canvas_ids": canvas_ids, "final_text": final_text}


def _common_params(req):
    C = int(req.get("C", 24))
    return dict(
        prompt=req["prompt"], T=int(req.get("T", 16)), C=C,
        top_k=int(req.get("top_k", 8)), t_max=float(req.get("t_max", 0.8)), t_min=float(req.get("t_min", 0.4)),
        entropy_bound=float(req.get("entropy_bound", 0.1)), early_stop=bool(req.get("early_stop", False)),
        enable_thinking=bool(req.get("enable_thinking", False)), max_new_tokens=int(req.get("max_new_tokens", C)),
    )


def _order_param(steps, dp):
    """m(t) = log P(A at dp) - log P(B at dp) over denoising steps."""
    import math
    return [math.log(max(s["pa"][dp], 1e-9)) - math.log(max(s["pb"][dp], 1e-9)) for s in steps]


def _basin_of(argmax_dp, m_last, a_set, b_set):
    if argmax_dp in a_set:
        return "A"
    if argmax_dp in b_set:
        return "B"
    return "other"


def run_rollouts(model, processor, req: dict) -> dict:
    """Sample N rollouts (one batched forward pass) and return each one's order-parameter
    trajectory + final basin, plus the decision position (pos mode) and basin counts.
    mode='pos': order param at one decision token. mode='trace': aggregate marker mass over
    the whole canvas (so the two basins can be multi-token reasoning traces)."""
    import math
    p = _common_params(req)
    N = int(req.get("N", 24)); seed0 = int(req.get("seed0", 0))
    mode = req.get("mode", "pos")
    tok = processor.tokenizer
    basin_a = req.get("basin_a") or []; basin_b = req.get("basin_b") or []
    a_ids = _basin_ids(tok, basin_a); b_ids = _basin_ids(tok, basin_b)
    if not a_ids or not b_ids:
        raise ValueError("run_rollouts needs non-empty basin_a and basin_b token lists")
    a_set, b_set = set(a_ids), set(b_ids)
    eos_ids = model.config.eos_token_id; eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    pad_set = set([getattr(model.config, "pad_token_id", 0) or 0] + list(eos_ids))

    with STATE["lock"]:
        STATE["last_request"] = time.time()
        res = _run_batch(model, processor, seed=seed0, batch=N, basin_a_ids=a_ids, basin_b_ids=b_ids,
                         capture_state=False, inject=None, **p)
    steps = res["steps"]; T = len(steps); final_argmax = steps[-1]["argmax"]   # [N,C]
    dp = (int(req["decision_pos"]) if req.get("decision_pos") is not None
          else (_pick_dp_batch(steps, a_set, b_set) if mode == "pos" else -1))

    rolls = []; counts = {"A": 0, "B": 0, "other": 0}
    for i in range(N):
        row = final_argmax[i].tolist()
        if mode == "trace":
            m = [round(math.log(float(steps[t]["pa"][i].sum()) + 1e-9)
                       - math.log(float(steps[t]["pb"][i].sum()) + 1e-9), 3) for t in range(T)]
            ca = sum(1 for x in row if x in a_set); cb = sum(1 for x in row if x in b_set)
            basin = "A" if ca > cb else ("B" if cb > ca else "other")
            ftok = f"A:{ca} B:{cb}"
        else:
            m = [round(math.log(max(float(steps[t]["pa"][i][dp]), 1e-9))
                       - math.log(max(float(steps[t]["pb"][i][dp]), 1e-9)), 3) for t in range(T)]
            amax = row[dp]; basin = "A" if amax in a_set else ("B" if amax in b_set else "other")
            ftok = tok.decode([amax])
        counts[basin] += 1
        ftext = tok.decode([t for t in res["canvas"][i].tolist() if t not in pad_set], skip_special_tokens=False)
        rolls.append({"seed": seed0 + i, "m": m, "final_basin": basin, "final_token": ftok, "final_text": ftext})
    return {"decision_pos": dp, "mode": mode, "T": T, "C": p["C"], "N": N, "counts": counts,
            "basin_a": basin_a, "basin_b": basin_b, "rollouts": rolls}


def run_steer(model, processor, req: dict) -> dict:
    """Cross-rollout steering, averaged over A/B pairs. For each recipient (basin-A) seed and
    donor (basin-B) seed, inject the donor's self-conditioning state into the recipient at each
    candidate step k and check whether the recipient flips to B. P(flip) vs k, averaged over
    pairs, traces the commitment barrier (high P(flip) early -> low/zero once committed)."""
    p = _common_params(req)
    alpha = float(req.get("alpha", 1.0)); inject_canvas = bool(int(req.get("inject_canvas", 0)))
    tok = processor.tokenizer
    basin_a = req.get("basin_a") or []; basin_b = req.get("basin_b") or []
    a_ids = _basin_ids(tok, basin_a); b_ids = _basin_ids(tok, basin_b)
    if not a_ids or not b_ids:
        raise ValueError("run_steer needs non-empty basin_a and basin_b token lists")
    a_set, b_set = set(a_ids), set(b_ids)
    T = p["T"]
    seeds_a = [int(s) for s in (req.get("seeds_a") or [int(req.get("seed_a", 0))])]
    seeds_b = [int(s) for s in (req.get("seeds_b") or [int(req.get("seed_b", 1))])]
    max_pairs = int(req.get("max_pairs", 5))
    n_pairs = min(len(seeds_a), len(seeds_b), max_pairs)
    seeds_a, seeds_b = seeds_a[:n_pairs], seeds_b[:n_pairs]
    inject_steps = req.get("inject_steps")
    if not inject_steps:
        inject_steps = list(range(T)) if T <= 24 else list(range(0, T, max(1, T // 20)))
    inject_steps = [int(k) for k in inject_steps]
    dp = int(req["decision_pos"]) if req.get("decision_pos") is not None else None

    def m_last(steps):
        return _order_param([{"pa": s["pa"].tolist(), "pb": s["pb"].tolist()} for s in steps], dp)[-1]
    def basin(steps):
        return _basin_of(steps[-1]["argmax"][dp].item(), m_last(steps), a_set, b_set)

    with STATE["lock"]:
        STATE["last_request"] = time.time()
        donors = {}   # donor seed -> _run_once result (captured states)
        for sb in dict.fromkeys(seeds_b):
            donors[sb] = _run_once(model, processor, seed=sb, basin_a_ids=a_ids, basin_b_ids=b_ids,
                                   light=False, capture_state=True, inject=None, **p)
            STATE["last_request"] = time.time()
        bases = {}    # recipient seed -> baseline result
        for sa in dict.fromkeys(seeds_a):
            bases[sa] = _run_once(model, processor, seed=sa, basin_a_ids=a_ids, basin_b_ids=b_ids,
                                  light=False, capture_state=False, inject=None, **p)
            STATE["last_request"] = time.time()
        if dp is None:
            ra0, rb0 = bases[seeds_a[0]], donors[seeds_b[0]]
            cand = [(abs(ra0["steps"][-1]["pa"][q].item() - ra0["steps"][-1]["pb"][q].item())
                     + abs(rb0["steps"][-1]["pa"][q].item() - rb0["steps"][-1]["pb"][q].item()), q)
                    for q in range(p["C"])]
            dp = max(cand)[1]
        m_a = [round(x, 3) for x in _order_param([{"pa": s["pa"].tolist(), "pb": s["pb"].tolist()}
                                                  for s in bases[seeds_a[0]]["steps"]], dp)]
        m_b = [round(x, 3) for x in _order_param([{"pa": s["pa"].tolist(), "pb": s["pb"].tolist()}
                                                  for s in donors[seeds_b[0]]["steps"]], dp)]
        a_nat = basin(bases[seeds_a[0]]["steps"]); b_nat = basin(donors[seeds_b[0]]["steps"])

        per_k = {k: [] for k in inject_steps}   # 1.0 if recipient flips to donor's basin
        for sa, sb in zip(seeds_a, seeds_b):
            a_b = basin(bases[sa]["steps"]); b_b = basin(donors[sb]["steps"])
            donor_states = donors[sb]["captured"]
            for k in inject_steps:
                rk = _run_once(model, processor, seed=sa, basin_a_ids=a_ids, basin_b_ids=b_ids,
                               light=True, capture_state=False,
                               inject={"step": k, "donor_states": donor_states, "alpha": alpha,
                                       "inject_canvas": inject_canvas}, **p)
                per_k[k].append(1.0 if (a_b != b_b and basin(rk["steps"]) == b_b) else 0.0)
                STATE["last_request"] = time.time()

    sweep = [{"step": k, "pflip": round(sum(per_k[k]) / len(per_k[k]), 3), "n": len(per_k[k])}
             for k in inject_steps]
    flip_ks = [s["step"] for s in sweep if s["pflip"] >= 0.5]
    barrier = max(flip_ks) if flip_ks else None   # largest k where injection still flips >=50%
    return {"seeds_a": seeds_a, "seeds_b": seeds_b, "n_pairs": n_pairs, "decision_pos": dp,
            "a_nat": a_nat, "b_nat": b_nat, "alpha": alpha, "inject_canvas": inject_canvas, "T": T,
            "m_a": m_a, "m_b": m_b, "sweep": sweep, "barrier_step": barrier,
            "basin_a": basin_a, "basin_b": basin_b}


# ======================================================================================
# Batched path: generate many rollouts per forward pass (B in the batch dim). Used by
# /barrier to estimate P(flip) over hundreds of samples per (alpha, k) cheaply. Supports
# two basin definitions: mode="pos" (a single decision-token, the binary case) and
# mode="trace" (a multi-token reasoning trace — classify by which marker set dominates the
# whole canvas), so the basins can be primitive reasoning traces, not just binary labels.
# ======================================================================================
def _run_batch(model, processor, *, prompt, T, C, seed, batch, top_k, t_max, t_min, entropy_bound,
               early_stop, enable_thinking, max_new_tokens, basin_a_ids, basin_b_ids, capture_state, inject):
    tok = processor.tokenizer
    eos_ids = model.config.eos_token_id
    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    pad_token_id = getattr(model.config, "pad_token_id", 0) or 0
    torch.manual_seed(seed)
    model.config.canvas_length = C
    enc = processor.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", enable_thinking=enable_thinking).to(model.device)
    enc_b = {k: (v.repeat(batch, *([1] * (v.dim() - 1))) if torch.is_tensor(v) else v) for k, v in enc.items()}
    gen_config = DiffusionGemmaGenerationConfig(
        max_new_tokens=max_new_tokens, max_denoising_steps=T,
        sampler_config=EntropyBoundSamplerConfig(entropy_bound=entropy_bound), t_min=t_min, t_max=t_max,
        stability_threshold=(1 if early_stop else T + 1), confidence_threshold=(0.005 if early_stop else 1e-9),
        pad_token_id=pad_token_id, eos_token_id=eos_ids)
    model._begin_batch(basin_a_ids, basin_b_ids, capture_state=capture_state, inject=inject)
    try:
        out = model.generate(**enc_b, generation_config=gen_config)
    finally:
        # an OOM mid-generate must not leave _batch_active=True poisoning later requests
        steps, captured = model._end_batch()
    prompt_len = enc["input_ids"].shape[1]
    canvas = out.sequences[:, prompt_len:prompt_len + C].to("cpu")     # [B, C]
    return {"steps": steps, "captured": captured, "canvas": canvas}


def _classify_rows(argmax_BC, mode, dp, a_set, b_set):
    """argmax_BC: CPU int tensor [B,C]. Returns (list of 'A'/'B'/'other', nB, nA)."""
    out = []; nB = nA = 0
    for row in argmax_BC.tolist():
        if mode == "trace":
            ca = sum(1 for x in row if x in a_set); cb = sum(1 for x in row if x in b_set)
            cls = "A" if ca > cb else ("B" if cb > ca else "other")
        else:
            x = row[dp]; cls = "A" if x in a_set else ("B" if x in b_set else "other")
        out.append(cls); nB += cls == "B"; nA += cls == "A"
    return out, nB, nA


def _pick_dp_batch(steps, a_set, b_set):
    argmax = steps[-1]["argmax"]; C = argmax.shape[1]; best = None
    for c in range(C):
        col = argmax[:, c].tolist()
        cA = sum(1 for x in col if x in a_set); cB = sum(1 for x in col if x in b_set)
        score = (min(cA, cB), cA + cB)
        if best is None or score > best[0]:
            best = (score, c)
    return best[1]


def run_barrier(model, processor, req: dict) -> dict:
    """High-statistics steering barrier (batched). For each (alpha, k) we generate `recipients`
    fresh rollouts with a basin-B donor's self-conditioning injected at step k (per-element
    donors from a captured pool), and report P(land in basin B). Same base seed across the grid
    + the no-injection baseline => steps 0..k-1 are matched, isolating the injection's effect."""
    p = _common_params(req)
    tok = processor.tokenizer
    basin_a = req.get("basin_a") or []; basin_b = req.get("basin_b") or []
    a_ids = _basin_ids(tok, basin_a); b_ids = _basin_ids(tok, basin_b)
    if not a_ids or not b_ids:
        raise ValueError("run_barrier needs non-empty basin_a and basin_b token lists")
    a_set, b_set = set(a_ids), set(b_ids)
    mode = req.get("mode", "pos")
    n_rec = int(req.get("recipients", 96)); n_pool = int(req.get("pool", 96))
    alphas = req.get("alphas", [0.25, 0.5, 0.75, 1.0])
    inject_canvas = bool(int(req.get("inject_canvas", 0)))
    T = p["T"]
    inject_steps = [int(k) for k in (req.get("inject_steps") or list(range(T)))]
    base_seed = int(req.get("base_seed", 2000)); pool_seed = int(req.get("pool_seed", 1000))

    with STATE["lock"]:
        STATE["last_request"] = time.time()
        pool = _run_batch(model, processor, seed=pool_seed, batch=n_pool, basin_a_ids=a_ids, basin_b_ids=b_ids,
                          capture_state=True, inject=None, **p)
        dp = (int(req["decision_pos"]) if req.get("decision_pos") is not None
              else (_pick_dp_batch(pool["steps"], a_set, b_set) if mode == "pos" else -1))
        cls, _, _ = _classify_rows(pool["steps"][-1]["argmax"], mode, dp, a_set, b_set)
        donor_idx = [i for i, c in enumerate(cls) if c == "B"]
        if not donor_idx:
            raise ValueError("no basin-B donors in the pool — raise `pool` or pick a more bimodal problem")
        donor_S = [pool["captured"][k]["self_cond"][donor_idx] for k in range(T)]            # [k]->[nD,C,V]
        donor_canvas = ([pool["captured"][k]["current"][donor_idx] for k in range(T)] if inject_canvas else None)
        nD = len(donor_idx); del pool
        sel = [i % nD for i in range(n_rec)]   # recipient i steered by donor sel[i]

        base = _run_batch(model, processor, seed=base_seed, batch=n_rec, basin_a_ids=a_ids, basin_b_ids=b_ids,
                          capture_state=False, inject=None, **p)
        base_cls, base_nB, base_nA = _classify_rows(base["steps"][-1]["argmax"], mode, dp, a_set, b_set)
        base_A_idx = [i for i, c in enumerate(base_cls) if c == "A"]   # recipients we try to flip A->B
        STATE["last_request"] = time.time()

        grid = []
        for alpha in alphas:
            for k in inject_steps:
                inj = {"step": k, "donor_S": donor_S[k][sel], "alpha": float(alpha), "inject_canvas": inject_canvas,
                       "donor_canvas": (donor_canvas[k][sel] if inject_canvas else None)}
                r = _run_batch(model, processor, seed=base_seed, batch=n_rec, basin_a_ids=a_ids, basin_b_ids=b_ids,
                               capture_state=False, inject=inj, **p)
                r_cls, nB, nA = _classify_rows(r["steps"][-1]["argmax"], mode, dp, a_set, b_set)
                # conditional: of the recipients that were A at baseline (same seed => matched 0..k-1), how many flip to B
                flips = sum(1 for i in base_A_idx if r_cls[i] == "B")
                grid.append({"alpha": float(alpha), "k": k, "p_b": round(nB / n_rec, 4),
                             "flip_AtoB": round(flips / max(1, len(base_A_idx)), 4), "n_A": len(base_A_idx), "n": n_rec})
                STATE["last_request"] = time.time()

    return {"mode": mode, "decision_pos": dp, "n_recipients": n_rec, "n_donors": nD,
            "baseline_p_b": round(base_nB / n_rec, 4), "baseline_p_a": round(base_nA / n_rec, 4),
            "n_baseline_A": len(base_A_idx), "alphas": [float(a) for a in alphas], "inject_steps": inject_steps,
            "grid": grid, "T": T, "inject_canvas": inject_canvas, "basin_a": basin_a, "basin_b": basin_b}


def run_resample(model, processor, req: dict) -> dict:
    """Resampling order parameter for *strategy* commitment (the answer is not enough — a strategy is a
    latent approach). Take a reference rollout, capture its full state trajectory, and at each denoising
    step k re-run the rest R times (branch: inject the reference's state at k, then continue with fresh
    randomness). Return the R completed traces per k; an external strategy-judge then turns these into
    m(k) = P(branch uses the reference's strategy | resample from step k) — rises from base-rate to 1 as
    the rollout commits to its strategy."""
    p = _common_params(req)
    tok = processor.tokenizer
    R = int(req.get("R", 24)); ref_seed = int(req.get("ref_seed", 0)); branch_seed0 = int(req.get("branch_seed0", 10000))
    T = p["T"]
    branch_steps = [int(k) for k in (req.get("branch_steps") or list(range(T)))]
    a_ids = _basin_ids(tok, req.get("basin_a") or ["a"]); b_ids = _basin_ids(tok, req.get("basin_b") or ["b"])
    eos_ids = model.config.eos_token_id; eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    pad_set = set([getattr(model.config, "pad_token_id", 0) or 0] + list(eos_ids))
    decode = lambda ids: tok.decode([t for t in ids if t not in pad_set], skip_special_tokens=False)

    with STATE["lock"]:
        STATE["last_request"] = time.time()
        ref = _run_batch(model, processor, seed=ref_seed, batch=1, basin_a_ids=a_ids, basin_b_ids=b_ids,
                         capture_state=True, inject=None, **p)
        ref_text = decode(ref["canvas"][0].tolist())
        cap = ref["captured"]   # [T] of {self_cond:[1,C,V], current:[1,C]}
        branches = {}
        for k in branch_steps:
            inj = {"step": k, "alpha": 1.0, "inject_canvas": True,
                   "donor_S": cap[k]["self_cond"].expand(R, -1, -1),     # broadcast the reference's step-k state
                   "donor_canvas": cap[k]["current"].expand(R, -1)}
            br = _run_batch(model, processor, seed=branch_seed0, batch=R, basin_a_ids=a_ids, basin_b_ids=b_ids,
                            capture_state=False, inject=inj, **p)
            branches[str(k)] = [decode(br["canvas"][i].tolist()) for i in range(R)]
            STATE["last_request"] = time.time()
    return {"ref_seed": ref_seed, "ref_text": ref_text, "T": T, "R": R,
            "branch_steps": branch_steps, "branches": branches}


def run_logitlens(model, processor, req: dict) -> dict:
    """Single-rollout LOGIT LENS across denoising. Capture every decoder layer's residual stream
    at each denoising step, project it through the model's OWN final norm + unembed + softcap
        lens_logits = softcap * tanh( lm_head(decoder.norm(h_layer)) / softcap ),
    and read the basin order parameter m = logP(A)-logP(B) + argmax at the decision position per
    (layer, step). Question: do EARLIER layers reveal the eventual basin at an EARLIER denoising
    step than the final-layer sampling logits do? Returns the full (layer x step) grid + a
    canvas-wide 'fraction of final tokens already decodable at this layer/step' metric."""
    import math
    p = _common_params(req)
    seed = int(req.get("seed", 0))
    tok = processor.tokenizer
    basin_a = req.get("basin_a") or []; basin_b = req.get("basin_b") or []
    a_ids = _basin_ids(tok, basin_a); b_ids = _basin_ids(tok, basin_b)
    if not a_ids or not b_ids:
        raise ValueError("run_logitlens needs non-empty basin_a and basin_b token lists")
    a_set, b_set = set(a_ids), set(b_ids)
    eos_ids = model.config.eos_token_id; eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    pad_token_id = getattr(model.config, "pad_token_id", 0) or 0
    pad_ids = [pad_token_id] + list(eos_ids); pad_set = set(pad_ids)
    lens_layers = req.get("lens_layers")     # optional subset of layer indices; default all
    topk = int(req.get("lens_topk", 5))

    with STATE["lock"]:
        STATE["last_request"] = time.time()
        torch.manual_seed(seed)
        model.config.canvas_length = p["C"]
        enc = processor.apply_chat_template(
            [{"role": "user", "content": p["prompt"]}], tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", enable_thinking=p["enable_thinking"]).to(model.device)
        gen_config = DiffusionGemmaGenerationConfig(
            max_new_tokens=p["max_new_tokens"], max_denoising_steps=p["T"],
            sampler_config=EntropyBoundSamplerConfig(entropy_bound=p["entropy_bound"]),
            t_min=p["t_min"], t_max=p["t_max"],
            stability_threshold=(1 if p["early_stop"] else p["T"] + 1),
            confidence_threshold=(0.005 if p["early_stop"] else 1e-9),
            pad_token_id=pad_token_id, eos_token_id=eos_ids)
        model._begin_trace(top_k=8, pad_ids=pad_ids, basin_a_ids=a_ids, basin_b_ids=b_ids, light=False)
        model._begin_lens()
        out = model.generate(**enc, generation_config=gen_config)
        steps = model._end_trace()
        lens_buf = model._end_lens()

        prompt_len = enc["input_ids"].shape[1]
        final_ids = out.sequences[0][prompt_len:prompt_len + p["C"]].tolist()
        final_canvas = torch.tensor(final_ids, device=model.device)
        T = len(steps)

        # decision position: provided, else the converged position that (a) holds a basin token and
        # (b) is most decisive (max |P(A)-P(B)|) at the final step.
        if req.get("decision_pos") is not None:
            dp = int(req["decision_pos"])
        else:
            last = steps[-1]; best = None
            for c in range(p["C"]):
                am = int(last["argmax"][c]); inset = am in a_set or am in b_set
                key = (1 if inset else 0, abs(float(last["pa"][c]) - float(last["pb"][c])))
                if best is None or key > best[0]:
                    best = (key, c)
            dp = best[1]

        norm = model.model.decoder.norm; lm_head = model.lm_head
        softcap = float(model.final_logit_softcapping)
        a_idx = torch.tensor(a_ids, device=model.device); b_idx = torch.tensor(b_ids, device=model.device)
        nonpad = torch.tensor([i for i, x in enumerate(final_ids) if x not in pad_set],
                              device=model.device, dtype=torch.long)

        lyrs = [int(l) for l in (lens_layers if lens_layers else sorted(lens_buf.keys()))]
        appeared = set(int(x) for x in final_ids)
        lens_out = []
        for l in lyrs:
            per = lens_buf[l]
            assert len(per) == T, f"layer {l}: {len(per)} captures != {T} steps"
            pa_l = []; pb_l = []; m_l = []; am_l = []; match_l = []; topk_l = []
            for t in range(T):
                logits = lm_head(norm(per[t]))[0].float()          # [C,V]
                lam = logits.argmax(-1)                            # softcap is monotonic -> argmax unaffected
                match_l.append(round(float((lam[nonpad] == final_canvas[nonpad]).float().mean()), 4)
                               if len(nonpad) else 0.0)
                row = softcap * torch.tanh(logits[dp] / softcap)   # [V], softcapped (matches model probs)
                probs = torch.softmax(row, dim=-1)
                pa = float(probs.index_select(0, a_idx).sum()); pb = float(probs.index_select(0, b_idx).sum())
                pa_l.append(round(pa, 5)); pb_l.append(round(pb, 5))
                m_l.append(round(math.log(max(pa, 1e-9)) - math.log(max(pb, 1e-9)), 4))
                am = int(row.argmax()); am_l.append(am); appeared.add(am)
                tp, ti = probs.topk(topk)
                topk_l.append({"ids": [int(x) for x in ti.tolist()], "p": [round(x, 4) for x in tp.tolist()]})
                appeared.update(int(x) for x in ti.tolist())
            lens_out.append({"layer": int(l), "pa": pa_l, "pb": pb_l, "m": m_l,
                             "argmax_id": am_l, "match_final_frac": match_l, "topk": topk_l})

        samp_pa = [round(float(s["pa"][dp]), 5) for s in steps]
        samp_pb = [round(float(s["pb"][dp]), 5) for s in steps]
        samp_m = [round(math.log(max(float(s["pa"][dp]), 1e-9)) - math.log(max(float(s["pb"][dp]), 1e-9)), 4)
                  for s in steps]
        samp_am = [int(s["argmax"][dp]) for s in steps]
        final_basin = "A" if final_ids[dp] in a_set else ("B" if final_ids[dp] in b_set else "other")
        final_text = tok.decode([t for t in final_ids if t not in pad_set], skip_special_tokens=False)

        # raw residual vectors at the decision position per step, for requested layers — so an
        # external activation-verbalizer (e.g. a gemma-4 NLA) can read DiffusionGemma activations.
        dp_vec_layers = [int(x) for x in (req.get("dp_vec_layers") or [])]
        dp_vectors = {str(L): [lens_buf[L][t][0, dp].float().tolist() for t in range(T)]
                      for L in dp_vec_layers}

    return {"prompt": p["prompt"], "basin_a": basin_a, "basin_b": basin_b, "decision_pos": dp,
            "T": T, "C": p["C"], "n_layers": len(lens_buf), "seed": seed, "softcap": softcap,
            "final_basin": final_basin, "final_text": final_text, "final_ids": final_ids,
            "a_ids": a_ids, "b_ids": b_ids, "dp_vectors": dp_vectors,
            "sampling": {"pa": samp_pa, "pb": samp_pb, "m": samp_m, "argmax_id": samp_am},
            "lens": lens_out, "id2str": _id2str(tok, appeared)}


@app.post("/logitlens")
def logitlens():
    STATE["last_request"] = time.time()
    return jsonify(run_logitlens(STATE["model"], STATE["processor"], request.get_json(force=True)))


def run_scope(model, processor, req: dict) -> dict:
    """One rollout returning (a) the full per-denoising-step trajectory (per-position argmax / entropy /
    pad-mass) for the denoise animation + entropy matshow, and (b) the FULL-CANVAS residual stream at the
    requested layer(s) for every step, so a gemma-4 NLA can verbalize what each canvas position is
    'computing' at each denoising step. Faithful read layer = 20 (the nanoNLA-gemma4 training layer).
    No basin tokens required (works for any problem, not just bistable forks).

    If `resid_out` (a path) is given, the [layers][T,C,d] residuals are written there as a float16 .npz
    (keys L<idx>) and only the path/shape is returned; otherwise they come back inline as `canvas_vectors`."""
    import numpy as np
    p = _common_params(req)
    seed = int(req.get("seed", 0))
    tok = processor.tokenizer
    scope_layers = [int(x) for x in (req.get("scope_layers") or [20])]
    resid_out = req.get("resid_out")
    eos_ids = model.config.eos_token_id; eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    pad_token_id = getattr(model.config, "pad_token_id", 0) or 0
    pad_ids = [pad_token_id] + list(eos_ids); pad_set = set(pad_ids)

    with STATE["lock"]:
        STATE["last_request"] = time.time()
        torch.manual_seed(seed)
        model.config.canvas_length = p["C"]
        enc = processor.apply_chat_template(
            [{"role": "user", "content": p["prompt"]}], tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", enable_thinking=p["enable_thinking"]).to(model.device)
        gen_config = DiffusionGemmaGenerationConfig(
            max_new_tokens=p["max_new_tokens"], max_denoising_steps=p["T"],
            sampler_config=EntropyBoundSamplerConfig(entropy_bound=p["entropy_bound"]),
            t_min=p["t_min"], t_max=p["t_max"],
            stability_threshold=(1 if p["early_stop"] else p["T"] + 1),
            confidence_threshold=(0.005 if p["early_stop"] else 1e-9),
            pad_token_id=pad_token_id, eos_token_id=eos_ids)
        model._begin_trace(top_k=int(req.get("top_k", 5)), pad_ids=pad_ids, light=False)
        model._begin_lens()
        out = model.generate(**enc, generation_config=gen_config)
        steps = model._end_trace()
        lens_buf = model._end_lens()

        prompt_len = enc["input_ids"].shape[1]
        final_ids = out.sequences[0][prompt_len:prompt_len + p["C"]].tolist()
        T = len(steps)
        appeared = set(int(x) for x in final_ids)
        steps_json = []
        for si, s in enumerate(steps):
            am = s["argmax"].tolist(); appeared.update(int(x) for x in am)
            steps_json.append({"step": si, "argmax": am,
                               "entropy": [round(x, 4) for x in s["entropy"].tolist()],
                               "pad_prob": [round(x, 4) for x in s["pad_prob"].tolist()],
                               "mean_entropy": round(s["mean_entropy"], 4)})
        # full-canvas residuals: per layer a [T, C, d] array (lens_buf[L][t] is [1, C, d])
        resid = {L: np.stack([lens_buf[L][t][0].float().cpu().numpy() for t in range(T)]).astype(np.float16)
                 for L in scope_layers}
        final_text = tok.decode([t for t in final_ids if t not in pad_set], skip_special_tokens=False)

    base = {"prompt": p["prompt"], "T": T, "C": p["C"], "seed": seed, "scope_layers": scope_layers,
            "final_text": final_text, "final_ids": final_ids, "pad_token_id": pad_token_id,
            "eos_token_ids": eos_ids, "id2str": _id2str(tok, appeared), "steps": steps_json,
            "resid_shape": [T, p["C"], int(model.config.text_config.hidden_size
                            if hasattr(model.config, "text_config") else 2816)]}
    if resid_out:
        os.makedirs(os.path.dirname(resid_out) or ".", exist_ok=True)
        np.savez(resid_out, **{f"L{L}": resid[L] for L in scope_layers})
        base["resid_path"] = resid_out
    else:
        base["canvas_vectors"] = {str(L): resid[L].tolist() for L in scope_layers}
    return base


@app.post("/scope")
def scope():
    STATE["last_request"] = time.time()
    return jsonify(run_scope(STATE["model"], STATE["processor"], request.get_json(force=True)))


@app.post("/resample")
def resample():
    STATE["last_request"] = time.time()
    return jsonify(run_resample(STATE["model"], STATE["processor"], request.get_json(force=True)))


@app.post("/barrier")
def barrier():
    STATE["last_request"] = time.time()
    return jsonify(run_barrier(STATE["model"], STATE["processor"], request.get_json(force=True)))


@app.post("/rollouts")
def rollouts():
    STATE["last_request"] = time.time()
    return jsonify(run_rollouts(STATE["model"], STATE["processor"], request.get_json(force=True)))


@app.post("/steer")
def steer():
    STATE["last_request"] = time.time()
    return jsonify(run_steer(STATE["model"], STATE["processor"], request.get_json(force=True)))


# ======================================================================================
# Cloze probe: is the output distribution a sensible (coherent) distribution over
# language? ONE denoising forward pass on a fully-specified canvas; exact softmax
# readout at chosen slot positions. DiffusionGemma has NO mask token -- the noise state
# is a uniform-random vocab token (EntropyBoundSampler.initialize_canvas), so every
# "unconditioned" quantity is averaged over n_noise random draws at the open slots
# (batched in one forward; conditions sharing `seed` are noise-paired).
# Two conditioning channels:
#   pins   -- write a token INTO the canvas at a slot (the hard, committed-token channel)
#   s_pins -- inject logit mass into self_conditioning_logits at a slot (the soft S^t
#             channel). S must then be a full tensor, so all OTHER positions get zero
#             logits = uniform soft-embedding; compare against s_base='uniform', which
#             is that same baseline WITHOUT the injection (matched control). s_base
#             ='none' (default when no s_pins) = true first-step S (skips soft-embeds).
# ======================================================================================
def run_cloze(model, processor, req: dict) -> dict:
    """req: prompt, parts (list of str | {'slot': name}; slots are 1 token wide),
    queries {slot: [token_str,..]}, pins {slot: token_str}, s_pins {slot: [token_str, scale]},
    s_base 'none'|'uniform', n_noise, seed, C, temperature, frame_noise_frac, header
    'thought'|'none', top_k, enable_thinking."""
    tok = processor.tokenizer
    vocab = model.config.text_config.vocab_size
    eos_ids = model.config.eos_token_id
    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    pad_id = getattr(model.config, "pad_token_id", 0) or 0

    def one_id(s: str) -> int:
        ids = tok.encode(s, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"{s!r} tokenizes to {ids}; cloze slots/queries need single tokens")
        return int(ids[0])

    # -- canvas frame: header + parts + <eos>, then pad fill; slots stay open (noise) --
    frame: list[int | None] = []
    if req.get("header", "thought") == "thought":
        # every sampled canvas opens with this channel header (verified on /sample outputs)
        frame += [int(tok.convert_tokens_to_ids(t)) for t in ("<|channel>", "thought", "\n", "<channel|>")]
    slot_pos: dict[str, int] = {}
    for part in req["parts"]:
        if isinstance(part, dict):
            slot_pos[part["slot"]] = len(frame)
            frame.append(None)
        else:
            frame += tok.encode(part, add_special_tokens=False)
    frame.append(eos_ids[0])
    C = int(req.get("C", len(frame)))
    if C < len(frame):
        raise ValueError(f"C={C} < frame length {len(frame)}")

    N = int(req.get("n_noise", 16))
    seed0 = int(req.get("seed", 0))
    g = torch.Generator().manual_seed(seed0)
    canvas = torch.randint(0, vocab, (N, C), generator=g)          # the model's noise state
    frame_t = torch.tensor([pad_id if t is None else t for t in frame] + [pad_id] * (C - len(frame)))
    frame_mask = torch.ones(C, dtype=torch.bool)
    for p in slot_pos.values():
        frame_mask[p] = False
    keep = frame_mask.unsqueeze(0).expand(N, C).clone()
    fnf = float(req.get("frame_noise_frac", 0.0))
    if fnf > 0:   # optionally re-noise a per-row random fraction of the FRAME (mid-denoising regime)
        keep &= ~((torch.rand((N, C), generator=g) < fnf) & frame_mask.unsqueeze(0))
    canvas = torch.where(keep, frame_t.unsqueeze(0), canvas)
    pins = req.get("pins") or {}
    for name, s in pins.items():
        canvas[:, slot_pos[name]] = one_id(s)
    canvas_cpu = canvas.clone()
    canvas = canvas.to(model.device)

    # -- self-conditioning channel --
    s_pins = req.get("s_pins") or {}
    s_base = req.get("s_base", "uniform" if s_pins else "none")
    self_cond = None
    if s_base == "uniform" or s_pins:
        emb_dtype = model.model.decoder.embed_tokens.weight.dtype
        self_cond = torch.zeros((N, C, vocab), dtype=emb_dtype, device=model.device)
        for name, (s, scale) in s_pins.items():
            self_cond[:, slot_pos[name], one_id(s)] = float(scale)

    queries = req.get("queries") or {}
    qstr = list(dict.fromkeys(s for cands in queries.values() for s in cands))
    qids = [one_id(s) for s in qstr]

    temperature = float(req.get("temperature", 1.0))
    enc = processor.apply_chat_template(
        [{"role": "user", "content": req["prompt"]}], tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
        enable_thinking=bool(req.get("enable_thinking", False)),
    ).to(model.device)
    input_ids = enc["input_ids"].expand(N, -1).contiguous()

    with STATE["lock"]:
        STATE["last_request"] = time.time()
        t0 = time.time()
        model.config.canvas_length = C
        gen_config = DiffusionGemmaGenerationConfig(
            max_new_tokens=C, max_denoising_steps=1,
            sampler_config=EntropyBoundSamplerConfig(entropy_bound=0.1),
            # validator requires t_max > t_min strictly; at T=1 the schedule evaluates to exactly
            # t_max on the single step, so the readout temperature == `temperature` regardless
            t_min=temperature - 1e-6, t_max=temperature,
            stability_threshold=2, confidence_threshold=1e-9,
            pad_token_id=pad_id, eos_token_id=eos_ids,
        )
        torch.manual_seed(seed0)
        model._begin_probe(qids, top_k=int(req.get("top_k", 12)))
        try:
            kw = {"decoder_input_ids": canvas}
            if self_cond is not None:
                kw["self_conditioning_logits"] = self_cond
            model.generate(input_ids=input_ids, generation_config=gen_config, **kw)
        finally:
            recs = model._end_probe()
        gen_seconds = time.time() - t0

    rec = recs[0]
    qcol = {q: j for j, q in enumerate(rec["qids"].tolist())} if qids else {}
    slots = {}
    for name, p in slot_pos.items():
        probs = {}
        for s, q in zip(qstr, qids):
            v = rec["qp"][:, p, qcol[q]]
            probs[s] = {"piece": tok.convert_ids_to_tokens(q), "mean": float(v.mean()),
                        "per_seed": [round(float(x), 8) for x in v]}
        slots[name] = {
            "pos": p,
            "pinned": pins.get(name),
            "canvas_tokens": [tok.convert_ids_to_tokens(int(t)) for t in canvas_cpu[:, p]],
            "probs": probs,
            "entropy_mean": float(rec["entropy"][:, p].mean()),
            "entropy_per_seed": [round(float(x), 4) for x in rec["entropy"][:, p]],
            "mean_topk": [{"piece": tok.convert_ids_to_tokens(int(i)), "p": round(float(pv), 6)}
                          for i, pv in zip(rec["mean_topk_ids"][p], rec["mean_topk_p"][p])],
        }
    return {
        "slots": slots, "slot_pos": slot_pos, "C": C, "n_noise": N, "seed": seed0,
        "temperature": temperature, "frame_noise_frac": fnf, "pins": pins, "s_pins": s_pins,
        "s_base": s_base, "header": req.get("header", "thought"),
        "frame_text": tok.decode([t for t in frame if t is not None], skip_special_tokens=False),
        "frame_len": len(frame), "gen_seconds": round(gen_seconds, 2),
    }


@app.post("/energy")
def energy():
    """Pseudo-NLL of arbitrary canvases: one readout step (temperature 1.0 by default = exact
    model softmax) over decoder_input_ids, returning per-position log p_i(C_i | C, prompt).
    Request: {prompt, canvases: [[ids...] same length], temperature?}."""
    req = request.get_json(force=True)
    model, processor = STATE["model"], STATE["processor"]
    canv = torch.tensor(req["canvases"], dtype=torch.long)
    B, C = canv.shape
    temperature = float(req.get("temperature", 1.0))
    pad_token_id = getattr(model.config, "pad_token_id", 0) or 0
    eos_ids = model.config.eos_token_id
    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    enc = processor.apply_chat_template(
        [{"role": "user", "content": req["prompt"]}], tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt").to(model.device)
    input_ids = enc["input_ids"].expand(B, -1).contiguous()
    # optional S-state sheet: sparse per-position S logits (ids+lp, e.g. a recorded top-32) —
    # energy is then a functional of the JOINT (canvas, S) state.
    sc = None
    if req.get("s_sparse"):
        s_ids = torch.tensor(req["s_sparse"]["ids"], dtype=torch.long)     # [C, K]
        s_lp = torch.tensor(req["s_sparse"]["lp"], dtype=torch.float)      # [C, K]
        V = model.lm_head.out_features
        emb_dtype = model.model.decoder.embed_tokens.weight.dtype
        sc = torch.full((1, C, V), -30000.0, dtype=emb_dtype, device=model.device)
        sc.scatter_(-1, s_ids.unsqueeze(0).to(model.device), s_lp.unsqueeze(0).to(emb_dtype).to(model.device))
        sc = sc.expand(B, -1, -1)
    with STATE["lock"]:
        STATE["last_request"] = time.time()
        model.config.canvas_length = C
        gen_config = DiffusionGemmaGenerationConfig(
            max_new_tokens=C, max_denoising_steps=1,
            sampler_config=EntropyBoundSamplerConfig(entropy_bound=0.1),
            t_min=temperature - 1e-6, t_max=temperature,
            stability_threshold=2, confidence_threshold=1e-9,
            pad_token_id=pad_token_id, eos_token_id=eos_ids)
        model._energy_buf = []
        model._energy_active = True
        model._begin_trace(top_k=1, pad_ids=[pad_token_id] + list(eos_ids), light=True)
        try:
            torch.manual_seed(int(req.get("seed", 0)))
            kw = {"decoder_input_ids": canv.to(model.device)}
            if sc is not None:
                kw["self_conditioning_logits"] = sc
            model.generate(input_ids=input_ids, generation_config=gen_config, **kw)
        finally:
            model._energy_active = False
            model._end_trace()
        scl = model._energy_buf[0]
        model._energy_buf = None
        lsm = torch.log_softmax(scl.float(), dim=-1)
        lp = lsm.gather(-1, canv.to(lsm.device).unsqueeze(-1)).squeeze(-1)   # [B, C]
        probe = None
        if req.get("probe_ids"):
            pidx = torch.tensor([int(x) for x in req["probe_ids"]], device=lsm.device, dtype=torch.long)
            probe = lsm.index_select(-1, pidx)                                # [B, C, P]
        tkk = None
        if req.get("topk"):
            _tp, _ti = lsm.exp().topk(int(req["topk"]), dim=-1)               # [B, C, k]
            tkk = (_ti.tolist(), [[[round(float(x), 4) for x in pos] for pos in row] for row in _tp.tolist()])
        ent = -(lsm.exp() * lsm).sum(-1)                                      # [B, C]
    out = {"logprob": [[round(float(x), 4) for x in row] for row in lp.tolist()],
           "entropy": [[round(float(x), 4) for x in row] for row in ent.tolist()],
           "C": C, "temperature": temperature}
    if probe is not None:
        out["probe"] = [[[round(float(x), 4) for x in pos] for pos in row] for row in probe.tolist()]
    if tkk is not None:
        out["topk_ids"], out["topk_p"] = tkk
    return jsonify(out)


@app.post("/cloze")
def cloze():
    STATE["last_request"] = time.time()
    return jsonify(run_cloze(STATE["model"], STATE["processor"], request.get_json(force=True)))


def _idle_watchdog(timeout_s: int):
    while True:
        time.sleep(30)
        if time.time() - STATE["last_request"] > timeout_s:
            print(f"[worker] idle for >{timeout_s}s -- shutting down to free the GPU.", flush=True)
            os._exit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8711)
    ap.add_argument("--addr-file", default=str(HERE / "worker_addr.txt"),
                    help="where to write '<node>:<port>' so the workbench app can find us")
    ap.add_argument("--idle-timeout", type=int, default=2400, help="exit after this many idle seconds (0=never)")
    args = ap.parse_args()

    load_model()
    STATE["last_request"] = time.time()

    node = socket.gethostname()
    Path(args.addr_file).write_text(f"{node}:{args.port}\n")
    print(f"[worker] serving on {node}:{args.port}  (addr file: {args.addr_file})", flush=True)

    if args.idle_timeout > 0:
        threading.Thread(target=_idle_watchdog, args=(args.idle_timeout,), daemon=True).start()

    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
