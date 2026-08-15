"""Summarize and visualize future-token surfacing by the original bidirectional DG J-Lens."""
from __future__ import annotations

import html
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(os.environ.get("DGLR_ROOT", Path(__file__).resolve().parents[1]))
OUT = REPO / "concept_probes/out/saeprobes/jlens"
REPORT = REPO / "reports/concept_probes"
WORKSPACE = tuple(range(8, 25))
SETS = ("order-ops", "multihop", "poetry", "typo", "association")
COLORS = dict(zip(SETS, plt.rcParams["axes.prop_cycle"].by_key()["color"], strict=False))


def save(fig, name: str) -> str:
    path = REPORT / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(path)
    return path.name


def control_switch(row: dict, key: str) -> int:
    return sum(row["original"][key][layer]["target_rank"] <= 20
               and row["counterfactual"][key][layer]["foil_rank"] <= 20
               for layer in WORKSPACE)


def main() -> None:
    exact = json.loads((OUT / "dg_original_future_exact.json").read_text())
    controls = json.loads((OUT / "dg_original_future_controls.json").read_text())
    rows = controls["rows"]
    for row in rows:
        row["jlens_top20_switch_layers"] = control_switch(row, "jlens")
        row["logitlens_top20_switch_layers"] = control_switch(row, "logitlens")

    fig, ax = plt.subplots(layout="constrained")
    for set_name in SETS:
        selected = [row for row in rows if row["set"] == set_name]
        if not selected:
            continue
        ax.scatter([row["logitlens_workspace_logodds_shift"] for row in selected],
                   [row["jlens_workspace_logodds_shift"] for row in selected],
                   label=set_name, color=COLORS[set_name])
    limits = ax.get_xlim()
    bounds = (min(*limits, *ax.get_ylim()), max(*limits, *ax.get_ylim()))
    ax.plot(bounds, bounds, linestyle="--", color="gray", label="equal shift")
    ax.axhline(0, color="gray", linewidth=plt.rcParams["lines.linewidth"] / 2)
    ax.axvline(0, color="gray", linewidth=plt.rcParams["lines.linewidth"] / 2)
    ax.set_xlim(bounds); ax.set_ylim(bounds)
    ax.set_xlabel("identity logit-lens future-choice log-odds shift")
    ax.set_ylabel("original J-Lens future-choice log-odds shift")
    ax.set_title("Earlier-position response to a one-token future substitution")
    ax.legend()
    shift_plot = save(fig, "saep_jlens_future_counterfactual_shift.png")

    strongest = sorted((row for row in rows if row["jlens_top20_switch_layers"]),
                       key=lambda row: (-row["jlens_top20_switch_layers"],
                                        -row["jlens_workspace_logodds_shift"]))[:4]
    fig, ax = plt.subplots(layout="constrained")
    for row in strongest:
        original = np.asarray([row["original"]["jlens"][layer]["target_minus_foil_logit"]
                               for layer in WORKSPACE])
        counterfactual = np.asarray([
            row["counterfactual"]["jlens"][layer]["target_minus_foil_logit"]
            for layer in WORKSPACE])
        ax.plot(WORKSPACE, original - counterfactual,
                label=f"{row['set']} #{row['item_index']}: {row['target_token'].strip()}→"
                      f"{row['foil_token'].strip()} (q{row['source_position']}←r{row['future_position']})")
    ax.axhline(0, color="gray", linewidth=plt.rcParams["lines.linewidth"] / 2)
    ax.set_xlabel("transformer layer")
    ax.set_ylabel("original − counterfactual target-vs-foil logit")
    ax.set_title("Original J-Lens response to the later token choice")
    ax.legend()
    layer_plot = save(fig, "saep_jlens_future_counterfactual_layers.png")

    exact_by_set = {set_name: sum(row["set"] == set_name for row in exact["candidates"])
                    for set_name in sorted({row["set"] for row in exact["candidates"]})}
    summary = {
        "protocol": controls["protocol"],
        "lens": controls["lens"],
        "workspace_layers": list(WORKSPACE),
        "n_eval_items": 551,
        "n_exact_future_top20_hits": exact["n_candidates"],
        "n_exact_future_top20_hits_by_set": exact_by_set,
        "n_exact_rank1_absent_logitlens_top20": sum(
            row["jlens_best_rank"] == 1 and row["logitlens_best_rank"] is None
            for row in exact["candidates"]),
        "n_counterfactuals": len(rows),
        "n_positive_jlens_shift": sum(row["jlens_workspace_logodds_shift"] > 0 for row in rows),
        "n_jlens_top20_switch_cases": sum(row["jlens_top20_switch_layers"] > 0 for row in rows),
        "n_logitlens_top20_switch_cases": sum(
            row["logitlens_top20_switch_layers"] > 0 for row in rows),
        "plots": {"shift": shift_plot, "layers": layer_plot},
        "strongest": [{key: row[key] for key in (
            "set", "item_index", "name", "prompt", "counterfactual_prompt", "source_position",
            "source_token", "future_position", "target_token", "foil_token",
            "jlens_workspace_logodds_shift", "logitlens_workspace_logodds_shift",
            "jlens_top20_switch_layers", "logitlens_top20_switch_layers")}
            for row in strongest],
    }
    (OUT / "dg_original_future_summary.json").write_text(json.dumps(summary, indent=1))

    embedded = json.dumps(rows).replace("</", "<\\/")
    page = f"""<!doctype html><html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="cache-control" content="no-cache, no-store, must-revalidate">
<title>Original J-Lens future-token controls</title><style>
:root{{color-scheme:light dark;--bg:#fff;--fg:#18202a;--card:#f4f6f8;--line:#cbd3dc;--hi:#dceeff}}
[data-theme=dark]{{--bg:#11161d;--fg:#e7edf5;--card:#1b222c;--line:#3b4655;--hi:#243c54}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font-family:system-ui,sans-serif}}
main{{width:min(1180px,calc(100% - 2rem));margin:1rem auto}}.top{{display:flex;justify-content:space-between;gap:1rem}}
.pill{{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:.2rem .55rem;margin:.15rem}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:.7rem;padding:1rem;margin:1rem 0}}
code,.tokens{{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}}pre{{white-space:pre-wrap;overflow:auto}}
select,button{{font:inherit}}.controls{{display:flex;gap:1rem;flex-wrap:wrap;align-items:end}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:.7rem}}
.cell{{border:1px solid var(--line);border-radius:.5rem;padding:.65rem;min-width:0}}
.tokens{{display:flex;flex-wrap:wrap;gap:.25rem}}.token{{border:1px solid var(--line);border-radius:.3rem;padding:.12rem .3rem}}
.target{{background:#b7efc5;color:#082b12}}.foil{{background:#ffc9c9;color:#3c0808}}
.prompt{{max-height:12rem}}img{{display:block;max-width:min(100%,780px);height:auto;margin:auto}}
</style></head><body><main><div class="top"><div><h1>Future-token controls</h1>
<span class="pill">original dgb_shared J-Lens</span><span class="pill">clean bidirectional DG canvas</span>
<span class="pill">one-token suffix interventions</span></div><button id="theme">theme</button></div>
<p>The later target token is replaced by one same-length tokenizer token. The earlier prefix and probe
position are unchanged. Green tokens are the original future target; red tokens are the foil.</p>
<div class="card controls"><label>case<br><select id="case"></select></label>
<label>layer<br><select id="layer">{''.join(f'<option>{x}</option>' for x in WORKSPACE)}</select></label></div>
<div id="detail"></div><div class="card"><img src="{shift_plot}"></div>
<div class="card"><img src="{layer_plot}"></div></main><script>
const rows={embedded}; const sel=document.querySelector('#case'),lay=document.querySelector('#layer');
const esc=s=>String(s).replace(/[&<>\"]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}}[c]));
rows.forEach((r,i)=>{{let o=document.createElement('option');o.value=i;o.textContent=`${{r.set}} #${{r.item_index}}: ${{r.target_token.trim()}} → ${{r.foil_token.trim()}}`;sel.append(o)}});
function toks(items,r){{return `<div class="tokens">${{items.map((x,i)=>`<span class="token ${{x===r.target_token?'target':x===r.foil_token?'foil':''}}">${{i+1}} ${{esc(x)}}</span>`).join('')}}</div>`}}
function render(){{const r=rows[+sel.value],l=+lay.value,jo=r.original.jlens[l],jc=r.counterfactual.jlens[l],lo=r.original.logitlens[l],lc=r.counterfactual.logitlens[l];document.querySelector('#detail').innerHTML=`
<div class="card"><span class="pill">read q=${{r.source_position}} (${{esc(r.source_token)}})</span><span class="pill">changed r=${{r.future_position}}</span><span class="pill">J shift ${{r.jlens_workspace_logodds_shift.toFixed(3)}}</span><span class="pill">J switch layers ${{r.jlens_top20_switch_layers}}</span><h3>Original</h3><pre class="prompt">${{esc(r.prompt)}}</pre><h3>Counterfactual</h3><pre class="prompt">${{esc(r.counterfactual_prompt)}}</pre></div>
<div class="grid"><div class="cell"><b>J-Lens · original</b>${{toks(jo.top_tokens,r)}}</div><div class="cell"><b>J-Lens · counterfactual</b>${{toks(jc.top_tokens,r)}}</div><div class="cell"><b>identity · original</b>${{toks(lo.top_tokens,r)}}</div><div class="cell"><b>identity · counterfactual</b>${{toks(lc.top_tokens,r)}}</div></div>`}}
sel.onchange=render;lay.onchange=render;render();document.querySelector('#theme').onclick=()=>{{document.documentElement.dataset.theme=document.documentElement.dataset.theme==='dark'?'light':'dark'}};
</script></body></html>"""
    (REPORT / "jlens_future.html").write_text(page)
    print(OUT / "dg_original_future_summary.json")
    print(REPORT / "jlens_future.html")


if __name__ == "__main__":
    main()
