# planning/ — symbol-arithmetic, ember, and parallel-ladder captures

Worker-driven captures (they POST to the DG worker in `../worker/` — set
`DG_WORKER`, default `http://localhost:18711`) plus CPU readers/report builders.
Capture scripts additionally need the tokenizer **google/gemma-4-26b-a4b-it** (HF-gated,
`HF_TOKEN`); readers/builders only need numpy/matplotlib/scipy.

## Dirs / env

- `$DG_PLANNING_DIR` (default `planning/exp/`) — capture JSONs (`xtask_*.json`,
  `ember_*.json`).
- `$DG_FIGS_DIR` (default `planning/figs/`) — figure PNGs from the `build_par*` scripts.
- `$DG_REPORT_OUT` (default `planning/out/`) — HTML reports (`build_superpos.py`,
  `build_seasonal.py`).
- `build_superpos.py` reads `build_seasonal.py` cwd-relative for its shared CSS, so run
  the builders from this directory: `cd planning && python build_superpos.py`.

## Capture order

1. Symbol-arithmetic cells: `xtask_compute8/9/10/12/13/14.py`, `xtask_samecase.py`,
   `xtask_mult.py`, `xtask_ops.py` (+ probes `xtask_mult_probe.py`, `xtask_mult_dose.py`,
   `xtask_ops_probe.py`).
2. N-sweeps: `xtask_samecase_nsweep.py`.
3. Parallel letter-ladders: `xtask_par2.py`, `xtask_par3.py`, then the CPU readers
   `xtask_par2_read.py` / `xtask_par3_read.py`.
4. Ember: `ember_kill2.py` (needs the worker's `no_commit` op — vendored `server.py` has
   it), `ember_preserve_fig.py` (worker needed only for the one-time
   `ember_base_traj.json` sampling; some panels also read `nego/` captures from the
   negotiation study, not vendored).

`constrained_common.py` is shared instance-selection/margin machinery; it imports
`../constrained/battery.py`.

## Builders (CPU)

`build_par_ladderfrac.py`, `build_par_min.py`, `build_par_frac.py`,
`build_par_section.py`, `build_par4_section.py`, `build_par_sina.py` -> figs;
`build_superpos.py` -> `symbol_arithmetic.html`; `build_seasonal.py` -> `seasonal.html`.

## Feeds (blog `src_data/`)

- `src_data/symbol_arithmetic_payload.json` <- the `window.__DATA__` payload of
  `build_superpos.py` (merged `xtask_compute{8,9,10,12,13,14}.json`).
- `src_data/ember_base_traj.json` <- `ember_preserve_fig.py` (first-run sampling);
  `src_data/ember_kill2.json` <- `ember_kill2.py`.
- `src_data/planning/xtask_par3.json` -> `scripts/plot_letters_parallel_frac.py`;
  `src_data/planning/xtask_samecase_nsweep.json` <- `xtask_samecase_nsweep.py`.
