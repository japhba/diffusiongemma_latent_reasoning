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

**PUBLIC-RELEASE CLEANUP 2026-08-15** (user: "clean nothing-unneeded starting point"): `post.html`
+ `post.docx` untracked (gitignored build outputs — keep rebuilding them locally; the aws-static
mirror rsyncs the worktree file, unaffected). The two GDocs roundtrip .docx archives moved OUT of
the repo to `/workspace-vast/jbauer/dg_blog_archive/` (comments preserved there). REMOVED from
git (recoverable at d9bf119): all figures/scripts/data not referenced by the current post — fig2b
triptych, figA7 controllability, figA10 selfrepair (+ selfcorr card: generator stripped from
build_html.py, data/selfcorr_steps.json, parts/card_selfcorr_steps.png), figA12/A14/A15 strict+
mult triptychs, figA13 mult map, figA16 jlens-step, extract_commit_{com,order}.py,
extract_reverse_chain_order.py, src_data/lockin/, planning/{canalysis,constr_summary,gallery,
commit_com,commit_order,reverse_chain_order}.json, posthoc/{com_posthoc_anim,temp_clean,
temp_suscept,lure_cots,steer_results}.json. posthoc/counterfactual.json KEPT (figA8 script loads
it). README.md added (minimal repro instructions — user should review). Full pipeline smoke-
tested post-cleanup: every fig*.py + compose_figs + build_html (+card) runs green.

**SEMANTIC FIG NAMES 2026-08-15** (user: no numbering labels): figures, parts, and figure scripts
renamed; figure scripts now share the `plot_` prefix (README repro glob). Mapping (old → new;
historical mentions below use the old names): fig1_gpqa_trunc_failures→gpqa_trunc_failures,
fig2a_transfer_map→letters_transfer_map, fig2a_example_intervention→letters_example_intervention,
fig2c_parallel_frac→letters_parallel_frac, figA1_rsa_cosine_cka→rsa_cosine_cka,
figA2_probe_retention→probe_retention, figA3_steer_retention→steer_retention,
figA4_jlens_retention→jlens_retention, figA5_seasonal_ember_kill→seasonal_ember_kill,
figA6_jlens_future→jlens_future, figA8_posthoc_correlations→posthoc_correlations,
figA8b_posthoc_case→posthoc_case, figA9_resolution→answer_resolution,
figA11_commit_causality→commit_causality; parts figA{2,3,4}_matrix→{probe,steer,jlens}_matrix;
scripts fig*→plot_* (e.g. figA8_posthoc_corr.py→plot_posthoc_corr.py).

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
  `steer_results.json` from `diffusiongemma/posthoc/`). EXTENDED 2026-08-08 to n=40 problems:
  `scripts/extract_posthoc_ext.py` merges `ext_clean/ext_suscept.json` (captured on the DG pod,
  `/workspace/dg/posthoc_ext/`, battery `diffusiongemma/posthoc/ext_battery.py`, same GRID; the
  capture parses around the post-Aug-04 `<|channel>thought` canvas scaffold) + fresh 3-rater
  `ext_difficulty.json`.
- `src_data/gdocs_roundtrip_2026-08-13.docx` — the user's Google-Docs editing pass (base = the
  45b8dce docx export of 2026-08-02, comments dated 08-03/04, pulled back 08-13). Its text edits
  were three-way merged into post.md on 2026-08-13 (~20 wording edits + two new blocks: the
  J-lens step-accuracy figure figA16 — extracted from the docx media, exists nowhere else — and
  the "DiffusionGemma represents tokens acausally" subsection prose that replaced the figA6
  [stub:]). NOT ported: its parallelism-section edits (section removed 08-13 per user) and its
  seasonal/palindrome edits (superseded by the 08-07 repo rewrites). The five Word comments live
  only in this archived file.
- `src_data/gdocs_roundtrip_2026-08-15.docx` — SECOND GDocs pass (base = the 827b63f-era export),
  merged 2026-08-15: full restructure (new Introduction; intro→"Background on DiffusionGemma";
  "Parallel computation" DELIBERATELY REINSTATED by the user with a NEW metric — fraction of
  injection sets with min-over-members E(x_i) > 0 vs the 0.5^n chance null, E(x_i) = R(img(x_i))
  − mean R over the possible-operand image pool; its figure `figs/letters_parallel_frac.png` is
  VENDORED from the docx media (user-made upstream, k=3, n≤3 — NOT regenerable from src_data;
  second repro exception besides card PNGs). 2026-08-15: title band ("letters +3" + E formula)
  CROPPED off per user (original archived at dg_blog_archive/letters_parallel_frac_titled.png);
  E definition moved into the post caption. REPRO ATTEMPT FAILED: the payload's let/uu k=3
  n-sweeps give min-E fractions 0.40–0.50 at n=1 (either pool convention) vs the plotted 0.82 —
  the figure's source data/conventions are upstream-only; ask the user for the generating
  script/data if this must enter the repro guarantee;
  Autonomous-usage section moved to main text with the user's caption + "early and late ablation"
  claim; interp sections under "## Transfer of interpretability techniques"; Conclusion after it;
  appendix = posthoc + bidirectionality only. REMOVED from the post per this pass: the figA16
  J-lens-step block and the appendix multiplicative-map (figA13) section — scripts + figs kept.
  Corrections applied during the merge: caption steps 64→94 fixed to the true 48→96; typos
  (compute, monitorability, model's, "describes how", tasks, left-to-right, should).
- `src_data/lockin/` ← `/workspace-vast/jbauer/exp/dg_lockin/` (clock + escape-minimum files).
- `src_data/planning/{canalysis,constr_summary,gallery}.json` ← constrained battery
  (`diffusiongemma/exp/dg_planning/` + `reports/dg-planning/data/gallery.json`).
- `src_data/symbol_arithmetic_payload.json` ← `window.__DATA__` of
  `reports/dg-planning/symbol_arithmetic.html` (builder `diffusiongemma/planning/build_superpos.py`).
  Refreshed 2026-08-10 from the 2026-08-04 report build: adds doms + tmaps `mu/rf/kb/mn3/cp`
  (multiplicative ×k, reflection, QWERTY, −3, copy); the `num/let/ll/uu` subtrees were verified
  byte-identical to the 2026-08-02 extraction, so fig2a/fig2b reproduce unchanged.

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
  UPPER→UPPER, ε=0.45, payload `tmap.let`, `v=="UU"`). Since 2026-08-14 (user): restricted to k=3 ONLY (UU3 states) so every row resolves to a unique
  target letter — yticklabels are the resolved outputs D…Z (no "x'−k" arithmetic). The ja/jb
  readout exclusion is LIFTED for display: row J (= G's image, the incumbent) shows the
  displacement response (blue band). The G source column stays dropped (never injected,
  asserted). Specificity stats skip row J: diag/offdiag R = 2.191/0.732 (excess +1.46; the old
  pooled-k map was 1.697/0.731 = +0.97 — figA13's caption comparison updated accordingly).
  Pooled version = git history.
- `scripts/extract_letters_example.py` — example-intervention data → `data/letters_example.json`
  (cell `UU3|hi|s0`, inject 'H'; A-slot sheet from battery state `UU3|src0|s0`).
- `scripts/fig2b_triptych.py` — parallelism, letters-only, mean ± 95% CI curves (no sinas, per
  user request 2026-08-01). **REMOVED from the post 2026-08-13** ("remove every claim about
  parallelism"): the whole main-text "### Parallelism" section (fig2b + capacity-n≈4 claim +
  superposition-transport paragraph) and the conclusion's "(parallel)" qualifier are gone.
  Script + fig kept, unreferenced.
- `scripts/figA12_letters_strict.py` / `figA15_mult_strict.py` — the report's "strict duplicate"
  of the triptych: min-over-targets vs max-over-non-targets, `E^min = R_T^min − R_N^max`,
  `NE^min = E^min/(n·ε0)` (same cells + exclusions as the mean scripts, which they re-assert
  against the stored per-cell `E`). Both families come out `E^min < 0` at every n.
  **REMOVED from the post 2026-08-13** with the rest of the n-sweep material; scripts + figs kept.
- `scripts/figA13_mult_transfer.py` — ×k transfer matshow from payload `tmap.mu` (ε=0.45, n=1,
  k∈{2,3,4} pooled by aligning rows to the pre-image pos(x')/k; no `v` filter — all UU-style).
  Diag mean R 0.993 vs offdiag 0.803 (additive: 1.697 vs 0.731) → the "+0.19 vs +0.97" caption.
  Still in the post: the appendix "Letter arithmetic" section is now just this n=1 map (the
  former "#### Multiplication" subheading was flattened away when the n-sweep figs left).
- `scripts/figA14_mult_triptych.py` — mean triptych for `mu` (imgOf: x' = letter at k·pos(x)).
  NOTE (from the report's caveat box / memory): the mixed-reference NE for ×k is inflated by
  lattice non-specificity — placebo-corrected spec is ≈0 at ε0=0.04. **REMOVED from the post
  2026-08-13**; script + fig kept.
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
  filename kept). Both escapes = the seasonal→idiom palindrome. Briefly re-inserted 2026-08-09,
  then removed again by the user — fig + script kept, currently unreferenced in the post.
- `scripts/extract_commit_{com,order}.py` — NOT bare-clone-rerunnable: read the raw cruns
  archive (7.8 GB, un-vendored) → `src_data/planning/commit_{com,order}.json`. commit_order =
  per-position final-commit step (start of the accepted-mask's terminal True suffix).
- `scripts/figA11_causality.py` — commitment ORDER DIAGRAM (reworked 2026-08-07: canvas position
  vs commit rank, diagonal = causal; replaced the CoM version, which averaged symmetric spreads
  to 0.5 and hid end-anchoring), per ttype, hot regime (def commits everything by step ~3, no
  signal). Finding: net-causal everywhere (chain-rho .57–.86) but anticausal stages — ends_with
  anchors the final word early and back-fills the middle last; all tasks back-fill the left edge
  at rank ~0.2.
- `scripts/figA8_posthoc_corr.py` — 3 post-hoc correlations, n=40 (difficulty↔commit +0.37
  p=.02, difficulty↔S +0.28 p=.08, commit↔S +0.60 p=.0001; tie-aware hand-rolled Spearman,
  asserted; p = two-sided permutation test, 20k perms, seeded) from
  `src_data/posthoc/{clean,suscept,difficulty}.json`; emits `data/posthoc_case.json`
  (easy-vs-hard card: bat_ball | sq1000 × clean | rho=1.0 susceptibility, commit times +
  purple intervention sites → figA8b via compose). The lure-CoT counterfactual illustration
  was dropped from the card 2026-08-09 (data still in
  src_data/posthoc/{counterfactual,lure_cots}.json).
- `scripts/figA9_resolution.py` — answer vs CoT region entropy per denoising step, averaged over
  the n=40 battery grouped by S (post-hoc S<=0.1 n=29, load-bearing S>=0.3 n=7; S=0.2 middle band
  + capture-skipped months28 excluded) from `src_data/posthoc/anim_curves.json` (pod capture
  posthoc_ext/ext_anim_batch.py, seed 0 per problem) + `suscept.json` for the grouping.
  (`com_posthoc_anim.json` keeps the earlier per-case exemplar traces + viewer frames.)
- `scripts/figA10_selfrepair.py` — clock-strike self-repair: natural transient wrongs (delta-frame
  decode of `src_data/lockin/com_clock_anim.json`) + escape-vs-plant-depth
  (`src_data/lockin/com_escape_minimum.json`).
- `scripts/extract_reverse_chain_order.py` — NOT bare-clone-rerunnable: reads the thinkfast
  denoising films (exp/dg_lockin/thinkfast/films/, transparency-paper replication battery) →
  `src_data/planning/reverse_chain_order.json` (digit positions + argmax lock steps per roll).
- `scripts/extract_films_order.py` — film tasks (incl. reverse_chain, used by the bottom-right
  panel) → `src_data/planning/films_order.json`.
- `scripts/capture_bench_order.py` — FRESH POD CAPTURE (2026-08-07, runs on the DG-worker pod
  against localhost:8711): GPQA/MATH/HumanEval/WildChat × 12 rollouts, default sampler, reduced
  to per-position lock steps → `src_data/planning/bench_order.json`; poem panel via
  `capture_poem_order.py` → `poem_order.json`. figA11 is now 1×3 (2026-08-07): logically
  left-to-right (GPQA) / direction-indifferent (poem) / right-to-left (reverse_chain),
  committed-CoM over diffusion progress with L2R-filler reference. Judged overlay:
  `judge_logical_order.py` (Node V Qwen3.6-35B, thinking off; the judge decomposes into
  atoms AND states their logical derivation order as tied groups; rho_logic =
  Spearman(judged order, surface position)) → `judged_logical_order.json`
  (GPQA +0.96 inductive, poems 0.00 indifferent; JSON backslash-repair + one retry). Gotchas: NODEV_URL embeds the pod id
  (fixed to 8as4828phkjhh6 in .env 2026-08-07); the RunPod proxy 403s Python-urllib —
  send a curl User-Agent. Pod ssh
  port rotates (30013 as of 2026-08-07); worker relaunch: tmux dgworker →
  `PATH=/workspace/dgenv/bin:$PATH bash /workspace/serve_dg.sh` (hf CLI needs the venv PATH).
  figA12 was MERGED into figA11 (2026-08-07): 2×4 grid, top = benchmarks, bottom = idiosyncratic
  incl. the reverse_chain correct-vs-wrong panel; digit-level rho numbers (−0.89..−0.97) quoted
  in text come from reverse_chain_order.json.
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

Private GitHub: https://github.com/japhba/diffusiongemma_latent_reasoning (gh account japhba). Commit + push after substantive edits.
