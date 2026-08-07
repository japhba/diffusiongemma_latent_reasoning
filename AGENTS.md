# dg_blog — DiffusionGemma blog post

Blog-post compilation directory (standalone, not a git repo). The deliverable is `post.md`
(the user's text — edit conservatively, they rewrite prose themselves) plus its HTML render
`post.html` with figures and interactive-style illustration cards.

## Math convention

**Inline math is `$...$`, display math is `$$...$$`.** Do not use `\( ... \)` (converted away
2026-08-01). `build_html.py` stashes both forms before markdown conversion and KaTeX
auto-renders them client-side; the algorithm block uses `$$ \begin{aligned} ... \end{aligned} $$`.

**Case convention (since 2026-08-02):** bold lowercase `\mathbf{s}` = a single position's
distribution (the R^|V| object: confidence `\mathbf{s}_t[x_t]`, top-k truncation, injections,
the R-definition's `\bar{\mathbf{s}}`); bold uppercase `\mathbf{S}` = the full sheet
(R^{C×|V|}: the algorithm box, "passed between steps", S-mass). Likewise lowercase `x` = a
single token, uppercase `X` = the full canvas (the V^C object: `X_t` in the algorithm box,
"attends bidirectionally"). The time index t is ALWAYS a superscript (`\mathbf{S}^t`, `\mathbf{s}^t`, `X^t`, `x^t`, `x^{\prime\,t+1}`; since 2026-08-02) — never `_t`.

## Build pipeline

Everything is a pure CPU re-render of archived study data — never launch model runs for this.
Python: system `python3` suffices (matplotlib/numpy/PIL, no scipy needed — Spearman is
hand-rolled); the former /var/tmp loracles venv is wiped on workbench reboots.

**Self-contained since 2026-08-02:** all source data is vendored under `src_data/` (~23 MB) and
every script resolves paths relative to the repo root (`Path(__file__).resolve().parent.parent`),
so the full pipeline reruns from a bare clone with numpy+matplotlib+PIL. Verified: rerunning all
12 scripts from `src_data/` reproduces every committed PNG byte-identically. Only exception:
`figs/parts/card_*.png` are pre-rendered browser screenshots (see build_html.py card recipe) —
committed as artifacts, not regenerable offline. Provenance of `src_data/`:
- `src_data/commit_ds/` ← `/workspace-vast/jbauer/exp/dg_lockin/pipe/commit_ds/` (only the
  gpqa manifests fig1 needs: acts_bench, acts_stab *_slow3, acts_psweep *_slow3).
- `src_data/saeprobes/` (+`jlens/`) ← `activation_oracles_dev/concept_probes/out/saeprobes/`.
- `src_data/ember_base_traj.json`, `src_data/ember_kill2.json` ← `diffusiongemma/exp/dg_planning/`.
- `src_data/posthoc/` ← `/workspace-vast/jbauer/exp/dg_lockin/posthoc/` (+ `lure_cots.json`,
  `steer_results.json` from `diffusiongemma/posthoc/`).
- `src_data/lockin/` ← `/workspace-vast/jbauer/exp/dg_lockin/` (clock + escape-minimum files).
- `src_data/planning/{canalysis,constr_summary,gallery}.json` ← constrained battery
  (`diffusiongemma/exp/dg_planning/` + `reports/dg-planning/data/gallery.json`).
- `src_data/symbol_arithmetic_payload.json` ← `window.__DATA__` of
  `reports/dg-planning/symbol_arithmetic.html` (builder `diffusiongemma/planning/build_superpos.py`).

- `scripts/build_html.py` — renders `post.md` → `post.html` (minimal md subset: `#`-headers,
  `**`/`__`/`*`, images, links, lists, `` ` `` code, `---`). House style: system light/dark +
  toggle (top right), no-cache meta, ~900px column. **Rebuild + push after every post.md edit.**
  It can also inject native-HTML illustration cards at `<!--ILLUST:name-->` markers from
  `data/*.json` — but since 2026-08-01 the post uses PNG renders of those cards instead
  (markers removed), so md and html show identical figures. To change a card: edit its
  generator in build_html.py, run `build_html.py card` (renders ALL four cards to card_preview.html), push, screenshot each
  `.card` on aws-static (playwright, device_scale_factor=2) → figs/parts/card_*.png,
  run `scripts/compose_figs.py`, rebuild.
- `scripts/compose_figs.py` — composes appendix figures (parts/figA*_matrix.png on top +
  parts/card_*.png below) → figs/figA*_retention.png, and promotes the letters-example card to
  figs/fig2a_example_intervention.png. Titles are intentionally absent from all plots — the
  post's italic captions carry that information; don't re-add them.
- `scripts/fig1_trunc_failures.py` — GPQA truncation stacked failure-mode bars.
  Data: `src_data/commit_ds/` (std ladder `acts_bench/gpqa/*/manifest.json`, gentle soft/k1
  `acts_stab/.../manifest.json`, gentle k2–k8 `acts_psweep/.../manifest_w*.jsonl`; schemas
  differ: `correct` vs `ok`). Failure rules per `diffusiongemma/lockin/fig2_report.py::norm_record`.
- `scripts/payload.py` — loads `src_data/symbol_arithmetic_payload.json`.
- `scripts/fig2a_transfer_map.py` — letter-arithmetic transfer matshows (headline variant:
  UPPER→UPPER, ε=0.45, payload `tmap.let`, `v=="UU"`).
- `scripts/extract_letters_example.py` — example-intervention data → `data/letters_example.json`
  (cell `UU3|hi|s0`, inject 'H'; A-slot sheet from battery state `UU3|src0|s0`).
- `scripts/fig2b_triptych.py` — parallelism, letters-only, mean ± 95% CI curves (no sinas, per
  user request 2026-08-01).
- `scripts/figA{1..4}_*.py` — appendix matrices (A1 RSA curves; A2/A3/A4 matrix-only PNGs in
  figs/parts/) + card data: `data/probe_pair.json`, `data/steer_pair.json` (cells gg+gd — both
  target models side by side), `data/jlens_layers.json` (couplet-breath-death top-5/layer for both
  read streams, from `jlens/eval_2x2.json` examples). Source data: `src_data/saeprobes/`.
  The A2 matrix is the 3×3 mode-split (probe_matrix.json modes: `headline` = DG causal,
  `declast` = DG bidirectional last-token; `decmean` only cited in prose). causal↔bidirectional
  cross-cells don't exist in the data → masked grey. Matrix scores annotated `:.2f` everywhere.
- `scripts/figA6_jlens_future.py` — future-operation card data (`data/jlens_future.json`):
  order-ops minimal pair `word-div-sub` from `src_data/jlens_future_rows.json` (extracted from
  reports/concept_probes/jlens_future.html `const rows=`, builder
  `concept_probes/analyze_jlens_future.py`); original = subtraction, counterfactual = addition,
  layers 23/21/18/12. Card `jlens_future` in build_html.py; compose_figs promotes the screenshot
  to `figs/figA6_jlens_future.png` (card-only, no matrix).
- `scripts/figA7_controllability.py` — SELF-CORRECTION (reworked 2026-08-07): word-palindrome
  task only, hot regime, margins on warped time (tau = t / run's last margin change) →
  parts/figA7_margins.png; emits `data/selfcorr_steps.json` (3 decoded canvases of the flagship
  escape, early/mid-violating/late, gallery.json frames with channel markers stripped). Card
  `selfcorr_steps`; compose_v stacks plot + card → figs/figA7_constraint_margins.png (stable
  filename kept). Both escapes = the seasonal→idiom palindrome.
- `scripts/extract_commit_{com,order}.py` — NOT bare-clone-rerunnable: read the raw cruns
  archive (7.8 GB, un-vendored) → `src_data/planning/commit_{com,order}.json`. commit_order =
  per-position final-commit step (start of the accepted-mask's terminal True suffix).
- `scripts/figA11_causality.py` — commitment ORDER DIAGRAM (reworked 2026-08-07: canvas position
  vs commit rank, diagonal = causal; replaced the CoM version, which averaged symmetric spreads
  to 0.5 and hid end-anchoring), per ttype, hot regime (def commits everything by step ~3, no
  signal). Finding: net-causal everywhere (chain-rho .57–.86) but anticausal stages — ends_with
  anchors the final word early and back-fills the middle last; all tasks back-fill the left edge
  at rank ~0.2.
- `scripts/figA8_posthoc_corr.py` — 3 post-hoc correlations (difficulty↔commit +0.37,
  difficulty↔S +0.42, commit↔S +0.66; tie-aware hand-rolled Spearman, asserts reproduce the
  report exactly) from `src_data/posthoc/{clean,suscept,difficulty}.json`; emits
  `data/posthoc_case.json` (squares_400_800 dissociation card → figA8b via compose).
- `scripts/figA9_resolution.py` — answer vs CoT region entropy per denoising step (bat_ball/monty
  vs reverse_then_add/sq1000) from `src_data/posthoc/com_posthoc_anim.json`.
- `scripts/figA10_selfrepair.py` — clock-strike self-repair: natural transient wrongs (delta-frame
  decode of `src_data/lockin/com_clock_anim.json`) + escape-vs-plant-depth
  (`src_data/lockin/com_escape_minimum.json`).
- `scripts/figA5_ember.py` — seasonal-vs-idiom PRESERVATION (regenerated 2026-08-03,
  "preservation, not flipping"): 2 panels, seed s5 — base (idiom takes over) | persistent
  idiom-kill @t2+ (dotted onset + shading → seasonal preserved). Data:
  `src_data/ember_base_traj.json` + `src_data/ember_kill2.json` (palindrome_words__3 capture,
  builder `diffusiongemma/planning/ember_preserve_fig.py`; old single-step ember_kill.json
  removed) → `figs/figA5_seasonal_ember_kill.png`.

## Serving / publishing

- One canonical file per artifact, stable paths, **no version-keyed filenames or `?v=` URLs**.
- Local: symlink `/workspace-vast/jbauer/served/dg_blog` → here;
  URL http://localhost:8095/dg_blog/post.html (Simple Browser).
- Public: symlink `activation_oracles_dev/reports/dg-blog` → here; the `reports_mirror` tmux
  rsyncs to https://reports.janbauer.cc/dg-blog/post.html every 5 min (manual push:
  `rsync -rlptL .../reports/dg-blog aws-static:reports/`). Everything here becomes public —
  no secrets.
- Visual verification: workbench has no Chromium — screenshot on aws-static
  (`~/pwenv` playwright + temp `http.server` in `~/reports`).

## Gotchas

- The user edits `post.md` live in VS Code — write collisions happen; make targeted edits and
  re-check the file after "modified since read" errors. Their prose placeholders
  (`[stub:]`, `[link]`, `[maybe give more deets]`, `see []`) are intentional.
- Current canonical saeprobes numbers: probe 2×2 = 0.826/0.793/0.792/0.820; steering = RepE
  blind-pair accuracy (gg .80 gd .79 eg .85 ed .83 cg .73 cd .70); J-Lens A-score cells
  gg .44 / gd .33 / dgc .49/.46 / dgb .51/.55. Older memories citing 0.878/... are stale.

## Repo

Private GitHub: https://github.com/japhba/dg_blog (gh account japhba). Commit + push after substantive edits.
