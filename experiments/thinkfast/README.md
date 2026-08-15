# thinkfast/ — serial-depth battery + denoising-film capture

- `battery.py` — the deterministic task battery (instances derived from
  `MASTER_SEED = 20260630` via per-instance sha256, so every rerun regenerates identical
  problems) with **exact `check()` verifiers** per task (no LLM judging).
- `money_tasks.py` — the multiplication money-tasks (`gen_mult`), reusing `battery.last_int`.
- `grid_films.py` — companion denoising-film capture for every battery cell
  (task x depth x T rung), one JSON per cell into `$DG_FILMS_DIR`
  (default `thinkfast/films/`). Worker-driven, CPU locally:

```bash
DG_WORKER=http://localhost:18711 python grid_films.py
```

Feeds `src_data/planning/films_order.json` via `scripts/extract_films_order.py`.
