"""Build the standalone seasonal<->idiom bistability case-study report
(reports/dg-planning/seasonal.html) from stuck_gain.json + the round-2/3 ember batteries
(ember_kill2 / ember_related / ember_autonomous3 / ember_handicap). The round-1 single-step
ember-kill (ember_kill.json) is retracted in-report and no longer sourced.
Figures shared with the main dg-planning report (same figs/ dir)."""
import os
import json
from pathlib import Path

EXP = Path(os.environ.get("DG_PLANNING_DIR", str(Path(__file__).resolve().parent / "exp")))
OUT = Path(os.environ.get("DG_REPORT_OUT", str(Path(__file__).resolve().parent / "out"))) / "seasonal.html"

STYLE = """
:root{ --bg:#ffffff; --fg:#1a1d24; --dim:#6b7280; --card:#f4f6f9; --line:#d8dee8;
  --accent:#2563eb; --gold:#b45309; --goldbg:#fef3c7; --green:#166534; --greenbg:#dcfce7;
  --err:#d7301f; --pill:#e5e7eb; --code:#0f172a; --codebg:#eef1f6; }
@media (prefers-color-scheme: dark){ :root{ --bg:#0f1420; --fg:#dfe5ef; --dim:#8b93a5; --card:#171e2e;
  --line:#2a3450; --accent:#7aa2ff; --gold:#fbbf24; --goldbg:#43350a; --green:#4ade80; --greenbg:#123a22;
  --err:#f87171; --pill:#273049; --code:#dfe5ef; --codebg:#141b2b; }}
html[data-theme=light]{ --bg:#ffffff; --fg:#1a1d24; --dim:#6b7280; --card:#f4f6f9; --line:#d8dee8;
  --accent:#2563eb; --gold:#b45309; --goldbg:#fef3c7; --green:#166534; --greenbg:#dcfce7;
  --err:#d7301f; --pill:#e5e7eb; --code:#0f172a; --codebg:#eef1f6; }
html[data-theme=dark]{ --bg:#0f1420; --fg:#dfe5ef; --dim:#8b93a5; --card:#171e2e; --line:#2a3450;
  --accent:#7aa2ff; --gold:#fbbf24; --goldbg:#43350a; --green:#4ade80; --greenbg:#123a22;
  --err:#f87171; --pill:#273049; --code:#dfe5ef; --codebg:#141b2b; }
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:100%;padding:18px 28px 80px}
h1{font-size:1.5em;margin:.2em 0 .3em} h2{font-size:1.2em;margin:1.6em 0 .4em;border-bottom:1px solid var(--line);padding-bottom:.2em}
h3{font-size:1.02em;margin:1.2em 0 .3em}
a{color:var(--accent)}
.dim{color:var(--dim);font-size:.86em}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin:10px 0}
img.fig{max-width:980px;width:100%;border-radius:8px;border:1px solid var(--line);background:#fff;display:block;margin:8px 0}
.pill{display:inline-block;background:var(--pill);border-radius:99px;padding:1px 9px;font-size:.75em;margin:0 4px 2px 0;white-space:nowrap}
code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.92em}
.gen{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.85em;background:var(--codebg);color:var(--code);
     border-radius:8px;padding:10px 12px;white-space:pre-wrap}
#themeToggle{position:fixed;top:10px;right:14px;z-index:50;background:var(--card);border:1px solid var(--line);
  color:var(--fg);border-radius:99px;padding:4px 12px;cursor:pointer;font-size:.85em}
table.rz{border-collapse:collapse;margin:8px 0;font-size:.88em}
table.rz th,table.rz td{border:1px solid var(--line);padding:3px 9px}
table.rz th{background:var(--card)}
table.rz td{vertical-align:top}
.colgrip{position:absolute;top:0;right:-3px;width:6px;height:100%;cursor:col-resize;user-select:none}
.rowgrip{position:absolute;left:0;bottom:-3px;height:6px;width:100%;cursor:row-resize;user-select:none}
ul{margin:.3em 0 .3em 1.1em;padding:0}
.controls{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:8px 0}
.controls select,.controls input[type=range]{accent-color:var(--accent);background:var(--card);color:var(--fg);
  border:1px solid var(--line);border-radius:6px;padding:3px 6px}
#stepSl{flex:1;min-width:220px}
.strip{display:flex;flex-wrap:wrap;gap:3px;font-family:ui-monospace,Menlo,monospace;font-size:.82em;
  background:var(--codebg);border:1px solid var(--line);border-radius:8px;padding:10px;margin:8px 0}
.tk{padding:1px 4px;border-radius:3px;background:var(--card);border-bottom:2px solid transparent;
  cursor:pointer;white-space:pre;display:inline-flex;align-items:center;width:76px;box-sizing:border-box}
.tkt{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:pre}
.controls button{accent-color:var(--accent);background:var(--card);color:var(--fg);
  border:1px solid var(--line);border-radius:6px;padding:3px 10px;cursor:pointer;font-size:1em}
.controls button:hover{outline:1px solid var(--fg)}
.tk.com{border-bottom-color:var(--accent)}
.tk.fin{background:var(--greenbg)}
.tk.new{color:var(--gold);font-weight:600}
.tk.slot{box-shadow:inset 0 0 0 1.5px var(--gold)}
.tk.killed{box-shadow:inset 0 0 0 2px var(--err)}
.tk.selp{outline:2px solid var(--accent)}
.tk:hover{outline:1px solid var(--fg)}
.bars{display:inline-flex;align-items:flex-end;gap:1px;margin-left:4px;height:13px}
.bars i{width:3px;display:block}
.bi{background:#2f9e44}.bs{background:#e8590c}
.srow{display:flex;align-items:center;gap:6px;padding:2px 6px;border-radius:6px;cursor:pointer;flex-wrap:wrap}
.srow.cur{background:color-mix(in srgb, var(--accent) 14%, transparent)}
.srow:hover{outline:1px solid var(--line)}
.sno{font-family:ui-monospace,Menlo,monospace;min-width:3.2em;color:var(--dim)}
.tokpill{display:inline-block;border-radius:4px;padding:0 4px;margin:0 2px 1px 0;background:var(--pill);
  font-family:ui-monospace,Menlo,monospace;font-size:.8em;white-space:pre}
.tokpill.mi{background:var(--greenbg);color:var(--green)}
.tokpill.ms{background:var(--goldbg);color:var(--gold)}
.lgwrap{overflow-x:auto;background:var(--codebg);border:1px solid var(--line);border-radius:8px;padding:8px;margin:8px 0}
.lgrid{display:grid;gap:3px;align-items:start}
.lrowh{font-family:ui-monospace,Menlo,monospace;color:var(--dim);font-size:.8em;padding-top:2px;white-space:nowrap;position:sticky;left:0;background:var(--codebg)}
.lgph{font-family:ui-monospace,Menlo,monospace;font-size:.78em;text-align:center;white-space:pre;cursor:pointer;border-radius:4px;padding:1px 2px;overflow:hidden;text-overflow:ellipsis}
.lgph.sloth{box-shadow:inset 0 0 0 1.5px var(--gold)}
.lgph.selh{outline:2px solid var(--accent)}
.lgph:hover{outline:1px solid var(--fg)}
.lstk{display:flex;flex-direction:column;gap:1px;min-height:149px}
.lch{font-family:ui-monospace,Menlo,monospace;font-size:.72em;padding:0 3px;border-radius:3px;background:var(--card);
  overflow:hidden;text-overflow:ellipsis;white-space:pre;max-width:100%;box-sizing:border-box;height:14px;line-height:14px;
  white-space:pre;overflow:hidden;max-width:8.5em;text-overflow:ellipsis}
"""


def ember_section():
    import re as _re
    k2 = json.load(open(EXP / "ember_kill2.json"))
    rel = json.load(open(EXP / "ember_related.json"))
    au3 = json.load(open(EXP / "ember_autonomous3.json"))
    hc = json.load(open(EXP / "ember_handicap.json"))
    single = [v for k, v in k2.items() if _re.fullmatch(r"s\d+\|kill@t\d+", k)]
    n_single, n_sseas = len(single), sum(v["outcome"] == "seasonal" for v in single)
    pers_seas = {t: sorted(int(k.split("|")[0][1:]) for k, v in k2.items()
                           if _re.fullmatch(rf"s\d+\|kill@t{t}\+", k) and v["outcome"] == "seasonal")
                 for t in (2, 4, 6, 8, 10)}
    ps = lambda t: ", ".join(f"s{s}" for s in pers_seas[t]) or "none"
    n_pin = sum(v["outcome"] == "seasonal" for v in rel.values() if v["tag"].startswith("pinS1@"))
    n_ramp = sum(v["cls"] == "RAMP" and v["outcome"] == "idiom" for v in au3.values() if v["tag"] == "rescue")
    csd = (2, 3, 4, 5)
    n_h8 = sum(v["outcome"] == "seasonal" for v in hc.values() if v["tag"].startswith("hcap8@") and v["seed"] in csd)
    n_h16 = sum(v["outcome"] == "seasonal" for v in hc.values() if v["tag"].startswith("hcap16@") and v["seed"] in csd)
    return f"""
<h2>The causal test: S-mass ablation &mdash; revised after replication (2026-08-03)</h2>
<p>Replay the runs with one mode's S-mass ablated at the 5 contested slots
(<span class="mono">s_bump &delta;=&minus;3&times;10<sup>4</sup></span> on that mode's 5 token ids &mdash; removes their
probability mass from the self-conditioning logits handed to the next step), at a <b>single step</b>
t<sub>abl</sub>, over a <b>window</b>, or <b>persistently</b>. An earlier version of this section reported a
single-step ember-kill window; rounds 2&ndash;3 (full 8-seed sweep, a fresh-instance replication, and a
server-revert control) retracted it, and the section below is the corrected account: what ablation
actually does, the intervention family that <em>is</em> robust, and the autonomous-transition and
eviction-prevention results built on top. Runs are bit-deterministic within a worker instance
(identical trajectory md5 on repeat), so within one environment every divergence is caused by the
intervention.</p>
<div class="card">
<b>Intervention mechanics.</b> The sampler hands each step's full output-logit sheet
S<sup>t</sup>&nbsp;&isin;&nbsp;&#8477;<sup>128&times;262144</sup> to the next denoising step as self-conditioning input. The
<span class="mono">s_bump</span> hook runs immediately after step t computes its outputs and adds
&delta;&nbsp;=&nbsp;&minus;3&times;10<sup>4</sup> to the 5 entries S<sup>t</sup>[p<sub>j</sub>, id<sub>j</sub>] (contested position p<sub>j</sub>,
target mode's token id id<sub>j</sub>) &mdash; deleting those (position,&nbsp;token) beliefs from the sheet before the
handoff; the softmax at consumption renormalizes the remaining mass at each position. Nothing else changes:
the current step's own sampling and entropy-gate acceptance, the canvas (which at unaccepted positions is
renoised to uniform anyway), the other &asymp;262k logits at those positions, and the other 123 positions are
all untouched. A <b>single-step</b> kill modifies only the sheet produced at t<sub>abl</sub> (consumed at
t<sub>abl</sub>+1); since the model recomputes S from scratch every step, it is free to rebuild the deleted mass
one step later from the rest of the state &mdash; whether it does is exactly what the experiment measures.
A <b>persistent</b> kill re-deletes the entries on every subsequent sheet.</div>
<div class="card" style="border-left:4px solid var(--err)">
<b>Retraction: the single-step ember-kill was a knife-edge fluctuation, not a critical window.</b>
The round-1 result (kills at s3@t3&ndash;6 and s4@t3 flipping the run to seasonal) does not replicate:
the full 8-seed sweep gives <b>{n_sseas}/{n_single}</b> single-step flips (t<sub>abl</sub>=1&hellip;12), the five
original success cells give 0/5 on a fresh worker instance, and a re-run with <span class="mono">server.py</span>
reverted to the exact round-1 file also gives 0/5. Runs are bit-deterministic <em>within</em> an instance
(identical trajectory md5 on repeat), but the July environment is gone (Jul-28 pod rebuild): base
trajectories now take different routes to the same finals, and the knife-edge cells resolve differently.
The clearest casualty of the same environment shift: seed&nbsp;0 &mdash; the "trapped" seed &mdash; natively
produces the <em>idiom</em> now, and a fresh 103-seed scan finds <b>0/103</b> seasonal outcomes (July: 1/24).
Working standard going forward: single-step, near-boundary causal claims require a fresh-instance re-run
before they are believed.</div>
<img class="fig" src="figs/ember_kill2.png">
<p class="dim">Round-2 grid: single-step kill outcome for every idiom seed &times; t<sub>abl</sub> (left, all black =
no flips anywhere), and the late-t no-commit pairing (right; un-committing the canvas does not restore
single-step intervenability &mdash; see phase&nbsp;D of <span class="mono">ember_kill2.py</span>).</p>
<div class="card">
<b>Scoring correction (the task's own <span class="mono">check()</span> inverts the round-1 framing).</b>
The idiom <span class="mono">"All for one and one for all."</span> is a <em>valid</em> 7-word word-level
palindrome &mdash; the correct answer, produced natively by 23/24 seeds. The seasonal
<span class="mono">"All leaves fall when leaves fall all."</span> is <em>invalid</em> (reversed:
<span class="mono">"all fall leaves when fall leaves All"</span>). Idiom&rarr;seasonal is therefore
correct&rarr;incorrect degradation, not a switch between two solutions, and the outcome worth counting for
escapes is <b>escaped-and-valid</b> (left the idiom AND still satisfies the constraint; the usual valid
escape is <span class="mono">"All say one and one say all."</span>).</div>

<h3>What ablation actually does: it preserves a live contest &mdash; it does not flip basins</h3>
<img class="fig" src="figs/ember_preserve.png">
<p class="dim">Per-seed multipanel (style of the retired round-1 figure): top = base autonomous dynamics
(black = idiom S-mass, purple = seasonal S-mass at the DIFF slots), below = persistent idiom-kill started
at t<sub>abl</sub>&isin;2,4,6,8,10 (shaded). Left, s5 (native contest peak 0.75): kills at t2/t4 preserve the
native seasonal to the final (purple border); from t6 the eviction is already underway and the run lands in
the third basin (red). Right, s1 (peak 0.09): nothing to preserve at any onset.</p>
<div class="card" style="border-left:4px solid var(--green)">
<b>Persistent idiom-kill preserves the run's <em>native</em> seasonal, gated by how much contest was there
to preserve.</b> Kills held from t2 / t4 end seasonal for exactly {ps(2)} / {ps(4)} &mdash; the seeds whose
un-intervened early seasonal S-mass peaks at &ge;0.25 (s2 0.54, s3 0.43, s4 0.25, s5 0.75; s8's
native contest was not traced, but it is also the only windowed-kill survivor besides s5). The
low-contest seeds (s1, s6: peak 0.09) have nothing to preserve and land in a third, invalid basin
(<span class="mono">"All men say what men say all."</span>-family). Onset t6+ is too late ({ps(6)}) &mdash;
the native seasonal has already been evicted. <b>Release test:</b> at full dose (&minus;3&times;10<sup>4</sup>),
windowed kills of length 3&ndash;16 steps stick in only 2/8 seeds &mdash; on release the idiom re-invades even a
fully committed seasonal canvas. <b>Non-monotonic dose is the punchline:</b> a <em>moderate</em> handicap
(&delta;=&minus;8 on the idiom ids over steps [2,8) or [2,12), then hands off) ends seasonal in
<b>{n_h8}/8</b> cells on the four contested seeds, with seasonal S-mass at 0.98 at the moment of release
and holding &asymp;0.99 to the end &mdash; while &delta;=&minus;16 sticks in only {n_h16}/8 (release mass
0.5&ndash;0.68, re-invaded within &asymp;3 steps) and &minus;3&times;10<sup>4</sup> in 2/8. To become
self-sustaining the challenger must <em>out-negotiate a present incumbent</em>; deleting the incumbent
prevents exactly the consolidation that would make the switch durable. The gentlest intervention produces
the most durable flip.</div>

<h3>Install vs remove: the robust half of the intervention family</h3>
<div class="card" style="border-left:4px solid var(--accent)">
<b>Removal-only is the fragile member; anything that installs a challenger is trivially robust.</b>
(6 idiom seeds &times; onsets t&isin;2,3,5,8.) A single-step S-<b>pin</b> of the seasonal ids
(one-hot row: kill + install in one op) flips <b>{n_pin}/24</b> &mdash; at every onset including
post-commitment t=8, where m<sub>set</sub> snaps 0.00&rarr;1.00 and latches &asymp;0.99 for the remaining
&asymp;55 intervention-free steps. Pinning the <em>incumbent</em> (control) moves 0/6. A soft
&delta;=+8 bump (incumbent untouched) flips 12/12 at t2&ndash;3, 2/6 at t5, 0/6 at t8 &mdash; a real closing
window (at t8 the saturated incumbent absorbs the 0.07 blip in one step), unlike the kill's phantom one.
On-manifold <span class="mono">swap12</span> is inert single-step (1/24) but <em>persistent</em> swap12
leaves the idiom 12/12 and lands <b>novel valid palindromes in 6/12</b>
(<span class="mono">"All one for and for one all."</span>) &mdash; the best escaped-and-valid generator found.
The law: brief <em>removal</em> of the prompt-favored incumbent fails (re-derived from the prompt); brief
<em>installation</em> of any incumbent latches (S-incumbency self-reinforces regardless of prompt support).</div>

<h3>Triggered-but-autonomous transitions: the autonomy belongs to the prompt</h3>
<img class="fig" src="figs/ember_autonomous.png">
<div class="card" style="border-left:4px solid var(--gold)">
<b>One-step triggers, then hands off: ramps exist only toward the prompt-favored mode.</b>
<b>Uphill</b> (idiom&rarr;seasonal): no autonomous regime at any dose &mdash; sub-installing displacements
always relax back, even from 0.8&ndash;0.9 momentary mass (stick-separatrix &asymp;0.85&ndash;0.9 at t2, rising
with depth), and weakening the idiom alone hands seasonal up to 0.82 of the renormalized row for free with
12/12 relaxing back. <b>Downhill</b> (pinned seasonal trap &rarr; idiom, trigger at t8): a one-step
displacement to only &asymp;0.2 is followed by a fully autonomous ramp 0.18&rarr;0.45&rarr;0.97 over three
intervention-free steps, takeover at t11 &mdash; 12/12 across both trigger types, and the purest form is the
rescue kill (remove the seasonal ids for one step, promote <em>nothing</em>): idiom regrows from 0.01,
<b>{n_ramp}/6</b>, with near-identical trajectories across seeds &mdash; the ramp is landscape-determined,
not noise-seeded. Dose gotcha: additive &delta; is not effective dose &mdash; inside a saturated trap +4 on
the challenger moves its post-trigger mass by 0.00; compare at matched post-trigger displacement.</div>
<div class="card" style="border-left:4px solid var(--gold)">
<b>Incumbent-kill, re-established on manufactured traps.</b> The July result (17/17 single-step rescues of
the natural trapped seed s0) was real in its environment, but that trap no longer exists (s0 is natively
idiom now). Current-environment version: install the trap with a pin at t2 (it holds to the end under
every noise stream tested), then a single-step kill of the seasonal ids at t6 or t10 rescues to the
correct idiom <b>12/12</b>. The trap is <em>incumbency</em> &mdash; a self-reinforcing S-loop &mdash;
not a deep prompt-side well; interrupt it once and the prompt re-derives the solution.</div>
<div class="card">
<b>Reading (all rounds together): the S sheet gates which mode the run is in; the prompt supplies the only
autonomous restoring force.</b> Installation always wins (the pin latches at any t, in either direction);
removal alone never wins uphill (the prompt re-derives the favored mode) and always wins downhill (same
reason). The natural stuck&rarr;gain ramps above, and their causal reproduction here, are prompt-pull
unfolding once the incumbent's grip is loosened &mdash; S contributes <b>persistence of modes across steps</b>
(traps and sub-threshold embers alike), not the solution-finding itself. A durable <em>uphill</em> flip
exists only via consolidation-through-competition: handicap, don't delete, the incumbent. Caveat: all
S-side ops are mode-targeted (no mass-matched neutral control); specificity evidence = the incumbent-pin
control, dose-response structure, and too-early/too-late nulls.</div>"""


EXPLORER = """
<h2>Round 2: the residual state, the canvas, and the search for a violation percept</h2>
<p>Is the commitment carried by state the logit-sheet ablation cannot reach? (These experiments predate
the retraction above; their single-step cells share its knife-edge caveat, and the durable evidence below
is the persistent-arm dissociation and the controls.) Three further handles: <b>j-space residual ablation</b>
(project the layer-21 residual off lens-transported token directions, step- and position-scoped),
<b>canvas clamps</b> (unhealable tokens re-pinned every step &mdash; immune to renoising eviction), and a
<b>full-vocab violation-percept probe</b>.</p>

<h3>J-space ablation: commitment is not in the lens-readable residual</h3>
<img class="fig" src="figs/ember_jspace.png">
<div class="card" style="border-left:4px solid var(--accent)">
<b>The seed noise is intrinsic boundary sensitivity, not under-targeting.</b>
<ul>
<li><b>Specificity control passes:</b> projecting off 100 <em>random</em> directions at the same slots/steps
never flips anything (4/4 idiom, flips &asymp; base). The kills are direction-specific.</li>
<li><b>Single-step j-space cells share the knife-edge character of the retracted S-kill</b> (July
environment; window enumerations no longer quotable) &mdash; the durable observations are that different
mode-specific perturbations of the same state land in different basins non-monotonically in strength, and
that <b>composition is not additive</b> (js100 + S-kill together flipped 0/36, including cells where a
component alone flipped).</li>
<li><b>The persistent dissociation is the punchline:</b> persistent S-kill (5 sheet entries) &rarr; seasonal 9/9;
persistent js100 (&asymp;300 directions out at L21, every step) &rarr; <b>never seasonal</b> &mdash; the model still
pursues the idiom plan and assembles degraded family variants (<span class="mono">"All for one for one for
all."</span> &mdash; itself a valid palindrome &mdash; <span class="mono">"All for one one one for all."</span>).
The S sheet carries the <em>commitment</em>; the L21 lens-readable content is downstream <em>expression
machinery</em>. Caveat: the top-100 neighborhood (minus exact seasonal ids) still contains seasonal-adjacent
tokens (▁Fall, ▁falls, ▁autumn &hellip;), so js100 scrubs both modes' neighborhoods.</li>
</ul></div>

<h3>Canvas clamps: the canvas can veto the competitor, but cannot seed it</h3>
<img class="fig" src="figs/ember_clamp.png">
<div class="card" style="border-left:4px solid var(--gold)">
<b>Prevention works with one token; implantation fails entirely; and hard commitment kills the negotiation
regardless of whose token it is.</b>
<ul>
<li><b>Prevent (s3, clamp from t4):</b> clamping the first contested slot to the incumbent's
<span class="mono">' leaves'</span> keeps the run seasonal &mdash; the ember never rises (flat 0). But the
sharpest datum: clamping the <em>challenger's own</em> <span class="mono">' for'</span> ALSO kills the escape
(final <span class="mono">"All for fall when leaves fall all."</span> &mdash; the seasonal mode absorbs the clamp
as debris). A hard-committed canvas token at a contested slot aborts the soft negotiation <em>even when it is
evidence for the eventual winner</em>: the escape must be assembled jointly across FLUID slots in S; a frozen
member cannot participate.</li>
<li><b>Implant (s0):</b> clamping <span class="mono">' for'</span> from t0 or t4 never seeds the idiom
(ember flat 0; the model accommodates around the debris). Seeding the competitor requires the S channel
(the incumbent-kill above), not canvas evidence.</li>
<li><b>Conformance modulation hits the predicted trap, in a subtler form:</b> from-t0 clamps of a mirror
pair (<span class="mono">roses&hellip;roses</span> vs <span class="mono">roses&hellip;stones</span>) do not modulate
constraint satisfaction &mdash; the model re-lengthens the sentence (8&rarr;9 words), so the clamped positions
stop being mirror partners: it evicts the <em>constraint geometry</em> instead of the tokens. All from-t0
clamp arms end constraint-violating (mm&ge;2) in a fluent accommodation family
(<span class="mono">"All X are happy because Y are men all."</span>) &mdash; DG prioritizes local fluency around
unhealable debris over the global constraint.</li>
</ul></div>

<h3>No violation percept in j-space</h3>
<img class="fig" src="figs/ember_violation.png">
<div class="card" style="border-left:4px solid var(--err)">
<b>Null, with a working positive control.</b> Full-vocab topic scores (max-over-positions log-softmax, no
top-k truncation) for a 14-word violation lexicon vs 8 matched controls, on s0 (whose final VIOLATES the
constraint) vs s1/s3 (valid): the violation&minus;control delta is statistically indistinguishable across runs
in every channel (jlens/loglens &times; L17/21/25) &mdash; task vocabulary (<span class="mono">palindrome, mirror,
same</span>) is mildly elevated in ALL runs (prompt-derived), and true violation words
(<span class="mono">wrong, error, incorrect</span>) sit at control level even in the violating run. The anchor
tokens meanwhile separate the runs exactly as expected (<span class="mono">leaves</span> tops s0,
<span class="mono">one/for</span> top s1/s3), so the probe is sensitive. The S channel's lone candidate
(<span class="mono">fail</span>, 62 cells in s0 vs &asymp;5 elsewhere) localizes to fall-dominated slots ranked
below ▁fall/▁Fall/▁leave &mdash; embedding-neighbor leakage of <span class="mono">fall</span>, not error
awareness. Consistent with the flagship-run meta-awareness null: <b>the trapped run does not "know" it is
violating in any vocabulary-readable channel.</b></div>

<h3>Graded wrongness and on-policy wrong states: eviction, repair, capitulation</h3>
<p>Beyond single-token surgery: does making the canvas <em>more obviously wrong</em> raise the competitor?
Two designs. A <b>wrongness ladder</b> of one-shot plants (S zeroed; eviction allowed and informative):
correct idiom / fluent seasonal violation / word-swapped / shuffled salad / alien fluent sentence /
unrelated word list. And <b>on-policy wrong states</b> (the harvest idea): a NATIVE donor rollout under a
sibling prompt ("autumn leaves falling" = near content, "heavy city traffic" = far content), its carried
state captured after step j and injected into a palindrome-prompt rollout at step j &mdash; with
&alpha;=1 (donor canvas + donor S) or &alpha;=0 (donor canvas, recipient's own S).</p>
<img class="fig" src="figs/ember_wrongness.png">
<div class="card" style="border-left:4px solid var(--green)">
<b>Ladder: wrongness helps exactly insofar as it prevents incumbency &mdash; and only while the schedule is hot.</b>
Planted at step 0, the fluent seasonal violation INSTALLS the trap (2/4 trapped vs 7/8 idiom natively);
everything more obviously wrong is evicted wholesale and the idiom returns at ceiling with <em>earlier</em>
crossovers (word list: x&#772;=4). Even the CORRECT idiom plant only half-sticks (2/4) &mdash; a canvas without
S backing is weak regardless of content. Planted at step 16 (cooler, sticky): the effect inverts &mdash; the
fluent trap holds 2/2, and even the alien sentence re-derives into the TRAP 2/2: low-temperature eviction
falls to the nearest fluent attractor, not to the solution.</div>
<div class="card" style="border-left:4px solid var(--accent)">
<b>On-policy donor states: inert without their S; with their S &mdash; repair, eviction, or capitulation.</b>
<ul>
<li><b>&alpha;=0 is causally inert, every cell:</b> the donor's fully-formed wrong canvas &mdash; even a
32-step-converged one &mdash; injected WITHOUT its S reproduces each seed's native outcome exactly. Renoising
evicts unbacked canvas content within a step; the strongest single confirmation that the canvas is a pure
commitment ledger.</li>
<li><b>&alpha;=1, near content, early (autumn@j4):</b> the constraint <em>repairs the incumbent's material</em>
into a novel VALID palindrome &mdash; <span class="mono">"Fall leaves as soon as leaves fall."</span> 2/2. Task
fulfilment rises, but as a third mode built from donor vocabulary: the riser is not necessarily THE
competitor.</li>
<li><b>&alpha;=1, far content, early (traffic@j4):</b> wholesale eviction &rarr; idiom 2/2 &mdash; content too
distant to repair is simply dropped, and the prompt re-derives the solution.</li>
<li><b>&alpha;=1, late (j&ge;8):</b> capitulation &mdash; the donor sentence is completed and the constraint
abandoned ("Golden leaves drift slowly toward the cooling earth."). Point of no return &asymp; j8, matching
the prompt-swap cliff in the main report.</li>
</ul>
<b>Net answer:</b> obvious wrongness elevates the competitor only via incumbency destruction under remaining
heat; a genuinely on-policy wrong state (with its S) is never <em>escaped from</em> &mdash; it is repaired (near),
evicted (far), or completed (late).</div>

<h3>Susceptibility of the subleading tokens: a one-step response function</h3>
<p>The cleanest form of the wrongness question: at a FIXED stuck state, how does the challenger's one-step
readout mass respond to a graded dose of conditioning wrongness? Two dose axes on the /energy probe (single
denoiser step, no rollout, no outcome confound): <b>k</b> = canvas wrongness (k of the 8 seasonal-phrase
tokens replaced by random vocab tokens; measurement restricted to CLEAN contested slots) and
<b>&tau;</b> = incumbency strength (the trapped run's recorded S-sheet, s0@t8 top-32, re-tempered at
temperature &tau;). The readout logits of the probed step ARE the sheet handed to the next step, so the y-axis
is literally the competitor's mass in S<sup>t+1</sup>.</p>
<img class="fig" src="figs/ember_suscept.png">
<div class="card" style="border-left:4px solid var(--accent)">
<b>&chi;(canvas wrongness) &asymp; 0 under a live incumbent &mdash; challenger susceptibility is gated by S.</b>
<ul>
<li><b>Full-strength incumbent sheet (&tau;=1):</b> the very first corrupted token gives the competitor a
one-time &asymp;7&times; nudge off baseline (3&times;10<sup>&minus;5</sup> &rarr; 2&times;10<sup>&minus;4</sup>), after which the
response SATURATES &mdash; flat at 2&ndash;5&times;10<sup>&minus;4</sup> from k&asymp;1 to k=6 &mdash; while the incumbent
holds 0.87&ndash;0.995 even with <b>6 of 8 canvas tokens randomized</b> (same probe data, not shown). Canvas
wrongness buys a single multiplicative bump in S<sup>t+1</sup>, not a dose-proportional rise.</li>
<li><b>Weakening incumbency releases the susceptibility:</b> the k=0 challenger mass rises smoothly with
&tau; (3&times;10<sup>&minus;5</sup> &rarr; 3&times;10<sup>&minus;4</sup> &rarr; 10<sup>&minus;3</sup> &rarr; 3.5&times;10<sup>&minus;3</sup>
at &tau;=8), and the canvas-dose slope &part;&chi;/&part;k grows with &tau; (at &tau;=8: 0.0035 &rarr; 0.011 by k=4).
The two doses interact: canvas evidence gets leverage only on a softened incumbent.</li>
<li><b>Even at maximum dose the challenger stays &asymp;40&times; below the incumbent</b> (&tau;=8, k=4: 0.011 vs
0.47) &mdash; one step never comes close to a crossover. This is the quantitative form of the earlier results:
escapes must proceed through ITERATED S re-encoding (the 4&ndash;5-step ramp), rescues through S-side incumbency
destruction; single-step canvas evidence &mdash; however wrong the canvas &mdash; is orders of magnitude short.</li>
<li class="dim">Bare-canvas row shown for completeness only &mdash; the no-S readout is incoherent on this task
(the earlier sanity failure), so it is not a valid reference.</li>
</ul></div>
<p>Random-token corruption confounds constraint violation with fluency destruction, so the systematic version:
x = <b>violated mirror pairs v</b> (0&ndash;3; incumbent = 2), built exclusively from fluent 1&ndash;2 token edits
within the seasonal lexical field &mdash; v=0 is a VALID palindrome made of the incumbent's own words
(<span class="mono">"All leaves fall when fall leaves all."</span>) &mdash; plus <b>edit-only controls</b> at v=2
(token changed, violation count unchanged).</p>
<img class="fig" src="figs/ember_suscept2.png" style="max-width:640px">
<div class="card" style="border-left:4px solid var(--err)">
<b>The one-step readout is blind to constraint status.</b> Competitor mass in S<sup>t+1</sup> shows NO monotone
dependence on v at any incumbency temperature &mdash; within-level spread (which specific token was edited)
is as large as any between-level difference, and the wrongness-preserving controls scatter exactly like the
wrongness-changing edits. If anything the trend is mildly INVERTED (the valid v=0 palindrome gives the
highest competitor mass at &tau;&ge;2). Retro-diagnosis: the k&ge;1 bump in the random-corruption sweep was
<em>edit detection</em> (lexical perturbation), not violation sensing. Together with the violation-percept null,
this closes the loop: <b>no single denoiser step evaluates the constraint anywhere</b> &mdash; constraint pressure
exists only as a property of the prompt-conditioned ITERATION, emerging over multiple steps of belief
re-encoding, never as a one-step force on the competitor. <span class="dim">[Partly superseded: the fidelity
correction below shows this probe condition (amputated 9-position sheets, committed mini-canvas, T=1
readout) suppresses the very dynamics being measured; the validated-condition rerun DOES find a
conflict-specific effect.]</span></div>
<p>Final refinement: measure the delta against a SEMANTICS-PRESERVING null, and intervene in S<sup>t</sup>
itself (the state) rather than the canvas. Promotion of mass &epsilon; on one token at one slot of the live
incumbent sheet (&tau;=1), one-step readout of the competitor at the untouched slots {2,3,4}. Classes:
<b>repair</b> (' leaves' at the violated slot 5 &mdash; fixes the pair), <b>break</b> (' away'/' down'/' forever'
at the satisfied slot 6), <b>neutral null</b> (' falls'/' drops'/' descend' at slot 5 &mdash; meaning preserved,
violation count unchanged), <b>sharpen</b> (the incumbent's own token), and <b>competitor</b> (' for' at
slot 5).</p>
<img class="fig" src="figs/ember_spromote.png" style="max-width:640px">
<div class="card" style="border-left:4px solid var(--err)">
<b>Every promotion class lands inside the neutral band &mdash; &Delta;(constraint-relevant &minus; neutral) &asymp; 0
in S too.</b> Promoting ANY non-incumbent token produces the same generic response: a &asymp;2&ndash;3&times; rise
at &epsilon;=0.4 whose size tracks <em>lexical proximity to the incumbent's field</em> (' falls' 2.0&times;10<sup>&minus;4</sup>
&gt; ' drops' &gt; ' descend' 0.45&times;10<sup>&minus;4</sup>, flat) and is symmetric across repair / break /
competitor directions; only sharpening the incumbent's own token stays at/below baseline. Most striking:
promoting the competitor's OWN token ' for' to 40% of slot 5's mass lifts the idiom at the OTHER slots no
more than the neutral synonyms do &mdash; <b>the joint mode structure (for&rarr;one&rarr;and) does not propagate
cross-slot within a single step</b>. One-step promotions inject hypothesis <em>diversity</em>, not direction;
the constraint &mdash; and the mode coupling itself &mdash; are enforced only by the iterated dynamics. (This is
the same lesson as the multi-step side: a persistent 5-entry deletion decides outcomes, while a one-step
40%-mass promotion barely moves the sheet &mdash; the sampler integrates small biases across steps.) <span class="dim">[Partly superseded &mdash; see the
fidelity correction below.]</span></div>

<h3>More data, done right: probe fidelity, the noise fuel, and the confirmed conflict susceptibility</h3>
<p>The natural ramp (0.03&rarr;0.73 in 4 steps) proves SOME per-step drive exists &mdash; so a one-step null is
only as good as the probe's fidelity. Validation: reconstruct single steps at FULL fidelity (all 128 sheet
positions including the tail, full-length canvas, schedule-matched temperature) and check that they
reproduce the recorded ember dynamics m(t)&rarr;m(t+1) across all 8 seeds &times; t=1&hellip;14.</p>
<img class="fig" src="figs/ember_onestep.png">
<div class="card" style="border-left:4px solid var(--green)">
<b>The one-step map is faithful &mdash; but only with the sampler's real input statistics.</b> With the
<em>realish</em> canvas (committed-proxy tokens kept, open slots renoised to uniform, exactly what the
sampler feeds itself) the probe tracks the recorded ramp closely (s3@t6: predicted 0.172 vs recorded 0.192;
s7@t6: 0.666 vs 0.751; s0 correctly flat at 0). The same sheet on a fully-COMMITTED draft canvas suppresses
the ember up to 34&times; (s3@t6: 0.005). Two consequences: (1) the earlier one-step nulls used amputated
sheets + committed mini-canvases &mdash; the exact configuration that kills the dynamics; (2) <b>the fuel is
canvas OPENNESS + fresh noise</b>: the identity of the renoise draw alone moves the one-step ember 3&ndash;5&times;
(baseline spread 0.0017&ndash;0.0090 across draws) &mdash; the ramp is fluctuation-seeded, and conflict's role is
to keep entropy high, slots unaccepted, and noise flowing. The trapped sheet (s0@t8) is one-step DEAD:
no promotion of any class moves it off 0.0000 under any draw &mdash; trapped = locally absorbing.</div>
<img class="fig" src="figs/ember_powered.png" style="max-width:700px">
<div class="card" style="border-left:4px solid var(--gold)">
<b>Powered paired rerun (48 renoise draws, same-draw pairing, &epsilon;=0.2, ramp state s3@t5): the ember IS
differentially fuelled by conflict.</b> Paired log-ratios vs the unpromoted same-draw baseline:
semantics-neutral null exactly at zero (&minus;0.02&nbsp;&plusmn;&nbsp;0.05 pooled over falls/drops/descend);
sharpening the incumbent's own token +0.12&ndash;0.15; <b>repair</b> (' leaves') +0.20&nbsp;&plusmn;&nbsp;0.08;
<b>wrongness-promotion</b> (' away'/' down'/' forever' breaking the satisfied all&hellip;all pair)
<b>+0.375&nbsp;&plusmn;&nbsp;0.054</b> &mdash; 5.6&sigma; above the neutral null and 2.5&sigma; above its own-slot sharpening
control; <b>competitor-promotion</b> (' for') +0.356&nbsp;&plusmn;&nbsp;0.090 (3.8&sigma;) &mdash; the joint mode coupling
DOES propagate cross-slot in one step, it was just invisible at low power in the unfaithful condition.
Ordering: <span class="mono">break &asymp; compet &gt; repair &gt; sharpen &gt; neutral &asymp; 0</span>.
<br><br><b>Verdict on the hypothesis:</b> the competitor is a true ember fuelled by conflict &mdash; promoting
wrongness in the state feeds the challenger about as much as promoting the challenger itself (&asymp;1.45&times;
per step at &epsilon;=0.2), while meaning-preserving substitutions do nothing. Combined with the openness
finding: conflict fuels the ember through two channels &mdash; directly (conflict content in S biases the next
sheet toward the alternative) and structurally (conflict keeps slots open, admitting the noise that seeds
the ramp). Caveats: one state (s3@t5), one &epsilon;; the break class lives at slot 6 where a semantics-neutral
promotion is impossible in principle (any new hypothesis at a satisfied pair IS wrongness), hence the
sharpening control there.</div>
<p>Stress test of that verdict: 21 further arms on the SAME 48 paired canvases &mdash; more synonyms, unrelated
nouns and function words at the violated slot, implausible and incumbent-field breakers at the satisfied slot,
and the idiom's tokens at WRONG slots (' one'/' and'@5, ' for'@6).</p>
<img class="fig" src="figs/ember_powered2.png">
<p><b><a href="suscept.html">&#9654; Interactive version</a></b> &mdash; per-arm sina violins of the per-draw
&chi; values; click any draw-dot for the base vs intervention one-step films (lensgrid style, top-20
S<sup>t+1</sup> per position, promoted entry highlighted).</p>
<div class="card" style="border-left:4px solid var(--gold)">
<b>FUD round (31 arms total): the null and the break effect survive; the "competitor coupling" gets
reinterpreted.</b>
<ul>
<li><b>Null rock-solid:</b> 7 meaning-preserving synonyms pool to &minus;0.04&nbsp;&plusmn;&nbsp;0.03. And <b>unrelated
content nouns at the violated slot actively SUPPRESS</b> the competitor (&minus;0.16&nbsp;&plusmn;&nbsp;0.04, incl.
' one'@5) &mdash; generic disruption does not fuel the ember, it drains it.</li>
<li><b>Breakers replicate across 9 diverse tokens</b> (+0.24&nbsp;&plusmn;&nbsp;0.03 pooled, &asymp;6&sigma; vs synonyms),
graded by token (+0.06&hellip;+0.49) &mdash; but <b>incumbent-field breakers score exactly zero</b>
(' fall'/' leaves'@6: +0.01/+0.02): breaking the satisfied pair with material the incumbent already carries
does nothing. The operative variable is <em>un-absorbable new hypothesis mass at a settled position</em>,
not violation count.</li>
<li><b>The "competitor" effect generalizes to ALL connectives, at either slot</b> (' of' +0.43, ' and' +0.47,
' the' +0.17, ' for' +0.36@5 / +0.35@6; pooled +0.35&nbsp;&plusmn;&nbsp;0.04) while the idiom's content token
' one'@5 suppresses &mdash; so it is NOT exact-token mode coupling: promoting the competitor's <em>syntactic
frame</em> (connective-heavy palindrome template) is what feeds it. ' for' has no privileged status over
' of'.</li>
</ul>
<b>Final form of "the ember is fuelled by conflict":</b> fuelled by (a) challenges the incumbent cannot absorb
at its settled positions and (b) promotion of the challenger's syntactic frame &mdash; not by violation count,
not by the challenger's exact tokens, and not by novelty per se (synonyms null, unrelated nouns negative).</div>
<h3>Repair deep-dive: validity doesn't matter, the gateway slot does</h3>
<p>The sharpest test of "fuelled by conflict": <b>FULL repair</b> — promote BOTH pair-fixing tokens
(' fall'@off4 + ' leaves'@off5), i.e. hand the incumbent the S-configuration of the VALID seasonal
palindrome <span class="mono">"All leaves fall when fall leaves all."</span> If conflict per se feeds the
competitor, removing it should suppress (&chi;&le;0). Plus: each pair repaired singly, a two-slot
double-breaker control, &epsilon; sweeps, and per-slot readout.</p>
<img class="fig" src="figs/ember_repair.png">
<div class="card" style="border-left:4px solid var(--accent)">
<ul>
<li><b>Full repair still ELEVATES the competitor</b> (&chi; &asymp; +0.04&ndash;0.12 across &epsilon;, above the
double-breaker at matched dose). The one-step response assigns no value to constraint satisfaction &mdash;
consistent with every constraint-blindness result. What matters is that the repaired configuration
<em>contests the incumbent's committed one</em>.</li>
<li><b>Promotions are triggers, not mass transfer:</b> &Delta;P saturates in &epsilon; (so &chi; &prop; 1/&epsilon;),
and the promoted tokens are almost entirely WIPED at their own slot in S<sup>t+1</sup> (persistence
&le;0.01 from an injected 0.2) while their influence lands elsewhere.</li>
<li><b>Where it lands: the gateway.</b> Per-slot decomposition shows the entire idiom gain concentrates at
<b>offset 1</b> (idiom ' for'), the first contested slot after the committed anchor ' All'; offsets 2&ndash;5
barely move in one step. The competitor rises specifically as <span class="mono">"All for &hellip;"</span>
gains at the phrase-entry point.</li>
<li><b>The strongest one-step amplifier found anywhere:</b> ' leaves'@off2 (&chi; = +0.19 pooled,
+0.78 at off1 alone) &mdash; bidding ' leaves' at slot 2 creates an untenable doubling
(<span class="mono">"All leaves leaves&hellip;"</span>) that force-reopens the gateway slot, where the
prompt-frame favors ' for'. Local incoherence adjacent to the gateway beats every "wrongness" op.</li>
</ul>
<b>Final refinement of the mechanism:</b> the one-step ember response is <em>gateway renegotiation</em> &mdash;
perturbations feed the competitor exactly insofar as they destabilize the phrase-entry slot, where the
prompt-favored connective frame (' for') wins reopened negotiations. This subsumes the connective-class
effect, the off1 concentration, the trigger-like dose curve, the non-additivity, and the irrelevance of
constraint validity into one statement.</div>

<h2>Cross-task: does the nascent competitor track incumbent plausibility?</h2>
<p>Everything above is one task. To separate "DG's one-step map is constraint-blind" from "the palindrome is
weird", the same machinery (hot regime, recorded S<sup>t</sup> sheets, paired one-step
<span class="mono">/energy</span> probes, rollout-level plants) was run on three more tasks with
<em>fluent</em> plausibility manipulations &mdash; no unabsorbable material anywhere:</p>
<table class="rz"><tr><th>task</th><th>natural attractor (8/8 seeds identical)</th><th>attractor valid?</th>
<th>nascent competitor</th><th>plausibility manipulation (all fluent)</th></tr>
<tr><td><span class="mono">self_count_words__7</span></td>
<td><span class="mono">"The total count of words in this specific sentence is eleven."</span></td>
<td>TRUE (11 = 11)</td><td>live number negotiation at the count slot, t2&ndash;6: ' ten'/' nine' lead at t2,
' twelve' 0.12&ndash;0.20 through t4</td>
<td>tail-appends after the count slot: <span class="mono">" overall."</span> &rarr; true = 12,
<span class="mono">" overall today."</span> &rarr; true = 13 (geometry up to the slot unchanged)
&times; incumbent &isin; {fluid, ten, eleven, twelve, banana} installed on canvas</td></tr>
<tr><td><span class="mono">self_count_word_occ__0</span></td>
<td><span class="mono">"The word the appears exactly three times in this sentence."</span></td>
<td><b>FALSE</b> (true 'the'-count = 2) &mdash; a universal WRONG attractor</td>
<td>correct ' two' nascent at 0.2&ndash;0.3 in S<sup>t</sup> &mdash; ranked below the also-wrong ' four' at
several steps</td>
<td>single-token determiner swaps: 'this'&rarr;'the' makes the trap sentence TRUE (3 'the's);
'The'&rarr;'That' makes it wronger (1) &times; incumbent &isin; {fluid, two, three, four, banana}</td></tr>
<tr><td><span class="mono">cap_au / cap_tr</span></td>
<td><span class="mono">"The capital of Australia is Canberra."</span> / <span class="mono">"&hellip;of Turkey is
Ankara."</span></td><td>TRUE</td>
<td><b>none</b> &mdash; P(correct city) = 1.0 in S from t1, 8/8 seeds; no conflict &rArr; no ember</td>
<td>installed incumbent &isin; {fluid, correct, trap (Sydney/Istanbul), foreign (Berlin), banana} &times;
scaffold appended after the city: plain '.' vs ", home of the national parliament." vs
", famous for its harbour and opera house." / ", which straddles the Bosphorus strait." &times;
sheet &isin; {recorded t1, none}</td></tr></table>
<img class="fig" src="figs/xtask_plausibility.png">
<div class="card" style="border-left:4px solid var(--accent)">
<ul>
<li><b>Counting tasks, one step: blind, with zero confound left.</b> Self-count, fluid slot: the S<sup>t+1</sup>
number distribution is IDENTICAL across true counts (t4: P(' eleven') = .57/.61/.59 for true 11/12/13;
P(' twelve') is <em>lowest</em> in the frame where twelve is correct). Installed incumbents echo by token
identity only (' twelve' gains +.07/+.04/+.04 over fluid in F11/F12/F13 &mdash; no boost where true); the
factorial interaction contrast comes out <b>&minus;0.20 where counting predicts &gt;0</b>. The occ trap
replicates it: fluent determiner swaps that move the true count 1&rarr;2&rarr;3 leave every bar in place
(P(' three') = .60/.58/.58, P(' two') = .23/.26/.27), and dissent under the installed wrong ' three' drifts
mildly ANTI-plausible. One step neither counts words nor counts 'the's.</li>
<li><b>Capitals, one step, no sheet: dramatically plausibility-SENSITIVE.</b> With a plain scaffold the wrong
incumbent gets zero deference &mdash; ' Sydney' installed on canvas &rarr; P(' Canberra') = .994 in one step.
And congruence is graded and compositional: fluid slot + opera-house scaffold &rarr; P(' Sydney') = .625 (the
scaffold alone flips the answer); ' Sydney' + parliament scaffold &rarr; P(' Canberra') collapses to .24;
' Berlin' + parliament scaffold &rarr; <b>Berlin WINS</b> (.435 vs Canberra .088) &mdash; local NP coherence
outweighs the country fact. Plausibility moves the competitor exactly when it is <em>retrievable</em>
(parametric knowledge, local coherence), not <em>computable</em> (counting, mirror-checking).</li>
<li><b>Capitals, one step, WITH the recorded sheet: P(correct) = 0.998 in ALL 30 cells.</b> Sydney installed,
Berlin installed, banana, any scaffold &mdash; nothing moves it. The strongest S-primacy demonstration in the
study: a committed sheet overrides arbitrary fluent canvas evidence in one step.</li>
<li><b>Rollout level: the pull is attractor-seeking, not truth-seeking.</b> Plants
(<span class="mono">init_text</span>, S zeroed): self-count destroys the planted VALID 12- and 13-word
solutions 8/8 and reverts to the 11-word attractor; the false "&hellip;is ten." is "repaired" to eleven 4/4
&mdash; indistinguishable from attractor-pull. The occ task is decisive because its attractor is FALSE: the
planted TRUE twin <span class="mono">"&hellip;exactly two times&hellip;"</span> is <b>EVICTED back to the false
attractor</b> 2/2 at init_step 16, and the fluent TRUE context variant ('&hellip;in the sentence') is undone
4/4. Only full-heat replants (init_step 0) occasionally land on a valid rephrasing (3/8:
<span class="mono">"This sentence contains the word the exactly two times."</span>).</li>
<li><b>S-MATCHED plants (fair incumbency) close the loop.</b> The cells above plant a COLD canvas (S zeroed)
&mdash; an unfair fight, since a natural incumbent always holds the sheet. Re-run with native (canvas, S)
pairs harvested from copy-prompt donor rollouts (capture step 24, <span class="mono">donor_alpha=1</span>,
inject at init_step 8/16; all 6 donors verbatim-verified): the TRUE occ twin now <b>SURVIVES 4/4</b>
(&alpha;=0 bridge cells reproduce the cold eviction exactly &mdash; the "truth evicted" cells were the
cold-canvas artifact) &mdash; but the FALSE <span class="mono">"&hellip;four times&hellip;"</span> ALSO
survives 4/4, never corrected. Truth-blind in both directions: whoever holds the sheet wins. The residual
selectivity tracks attractor strength + remaining heat, not validity: S-backed
<span class="mono">"&hellip;is ten."</span> is still pulled to the attractor 4/4, and the valid 12-word
sentence injected @8 degrades into an INVALID hybrid (<span class="mono">"&hellip;is twelve."</span> &mdash;
11 words stating twelve: heat reclaims the geometry, the sheet holds the number, and a true sentence is
broken into a false one). Scripts: <span class="mono">xtask_plant2.py</span>.</li>
<li><b>On-policy donors (minor prompt nudges) overturn the copy-prompt immunity &mdash; and sharpen the
verdict.</b> A copy-prompt S is OOD-sticky (a "repeat" state is hyper-committed, artificially immune to
renegotiation). Harvesting instead with the ORIGINAL task prompt plus one appended steering sentence
("Aim for &hellip; exactly two times." / "&hellip; exactly twelve words.") yields task-conditioned states
that DO renegotiate after the swap back to the unmodified prompt. Screen: nudged-three, the model
<em>made the sentence TRUE on its own</em> ("&hellip;three times in <b>the</b> sentence." &mdash; the fluent
repair); nudged-twelve produced a novel-frame FALSE sentence
(<span class="mono">"Exactly twelve words are used within this specific sentence for you."</span>, 11 words).
Injections: the FALSE four-state is <b>evicted to the attractor 4/4</b> (it survived 4/4 as a copy-state);
the TRUE 'twice' twin survives 4/4; the TRUE repaired three-state survives @16 but is <b>UN-repaired back to
the FALSE attractor @8</b> (2/2, 'the'&rarr;'this'); and both novel-frame sc donors survive 4/4 &mdash; the
FALSE twelve exactly as robustly as the TRUE ten. Renegotiation is <b>attractor-directed, not
truth-directed</b>: in-template deviations are pulled back regardless of truth value, out-of-template
sentences survive regardless of theirs. Scripts: <span class="mono">xtask_plant3.py</span>.</li>
</ul>
<b>Cross-task verdict:</b> the competitor's probability does NOT respond to how implausible the incumbent is,
whenever plausibility requires <em>computing</em> a constraint over the canvas &mdash; three tasks now
(palindrome mirror, word count, occurrence count), at one step and at rollout scale. It responds strongly,
gradedly and compositionally when plausibility is <em>retrievable</em> (capitals). At rollout scale the donor
plants give the sharpest form: faithfully-harvested (on-policy) incumbents ARE renegotiated after a prompt
swap, and the renegotiation is attractor-directed, not truth-directed &mdash; in-template deviations
reverted whether false (four&rarr;three) or true (the fluent 'the-sentence' repair un-repaired @8);
out-of-template sentences survive whether true (ten) or false (twelve); copy-prompt states are artificially
immune (OOD-sticky), and sheet-less canvases re-derive straight to the attractor. Constraint pressure exists
ONLY in the prompt-conditioned initial negotiation &mdash; and the occ task shows even that landing on a
FALSE attractor 8/8, with no corrective force afterwards. Scripts:
<span class="mono">planning/xtask_{capture,selfcount,occ,capital,plant,plant2,plant3,fig}.py</span>; data:
<span class="mono">exp/dg_planning/xtask_*.json</span>, captures <span class="mono">nego2/</span>.</div>

<h3>Computational vs correlational susceptibility: three nulls &mdash; then found in the minimal setting</h3>
<p>Distinction (raised in review): every effect above &mdash; absorbability, doubling-incoherence, gateway
routing, connective-frame coupling &mdash; is <em>distributional</em> (deep, nonlinear coherence statistics
the ground-truth text distribution genuinely contains). A <em>computational</em> susceptibility would mean:
the response to a subleading S<sup>t</sup> entry is mediated by an instance-specific algorithm whose output
cannot be read off any co-occurrence table. Three probes (<span class="mono">xtask_compute.py</span> + a
re-read of the repair data):</p>
<div class="card" style="border-left:4px solid var(--accent)"><ul>
<li><b>Mirror-copy, novel tokens (NULL):</b> inject ' velvet'/' cobalt'/' lantern'/' marble' at &epsilon;=0.3
at phrase offsets 1/2/4/5 in S<sup>t</sup> (s3@t3 and t5) and read P(token) at all 8 offsets in
S<sup>t+1</sup>. The injected token is ANNIHILATED everywhere &mdash; own slot and mirror partner alike
(&le;8&times;10<sup>&minus;4</sup> vs floor 2&times;10<sup>&minus;6</sup>). No content-independent
mirror-index copy exists in one step.</li>
<li><b>Mirror-copy, in-distribution tokens (NULL/negative):</b> from the repair battery: promoting
' leaves'@off2 does NOT reinforce its mirror partner ' leaves'@off4 (&Delta; = &minus;0.012) &mdash; it
drains the ADJACENT slot instead (leaves@off1 &minus;0.31, the doubling conflict); ' leaves'@off5 leaves its
mirror @off1 flat (+0.002). The constraint's pairing structure is not read by the map even for the model's
own vocabulary; mirror agreement in natural traces must emerge across iterations.</li>
<li><b>Word-index arithmetic (NULL):</b> self-count frame committed to "&hellip;is", inject number N
(nine&hellip;fifteen, &epsilon; 0.25/0.5) at the count slot in S<sup>t</sup>, read the period position in
S<sup>t+1</sup>. No placement-by-arithmetic: P('.') mass at +3&hellip;+7 stays &le;0.03 for every N (a
computation would put fifteen's period at +5). Only a weak monotone drift survives at the open t2 state
(P('.'@+2): eleven .198 &rarr; fifteen .350 at &epsilon;=0.25) &mdash; "bigger number word &rarr; vaguely
more continuation", a magnitude co-occurrence, and even that vanishes under the committed t4 sheet
(.96 flat).</li></ul>
<b>Interim verdict (superseded just below):</b> the global computations these tasks would need (mirror
indexing over a whole phrase, word/occurrence counting) are implemented nowhere within a step. But those
probes all injected into states whose source content was already COMMITTED ON CANVAS &mdash; the answer slot
never <em>needed</em> S. The minimal fix finds the computation.</div>

<h3>Found: one-step, prompt-parameterized computation over an S-carried operand</h3>
<p>Design fix: a task with a one-hop functional dependency between two slots, probed with BOTH slots
renoised on canvas &mdash; S<sup>t</sup> is then the only possible carrier of the operand, so any
f(x)-tracking at the target slot is necessarily S &rarr; computation &rarr; S. Tasks:
<span class="mono">"Pick any English noun and write it twice, separated by a comma."</span> (f = identity,
attractor <span class="mono">"mountain, mountain"</span> 6/6) and
<span class="mono">"Pick any number between two and nine, write it in words, then write the number
three/four greater in words, separated by a comma."</span> (f = X+k with k a PROMPT parameter; the model
solves both natively: <span class="mono">"five, eight"</span> 6/6 under +3,
<span class="mono">"six, ten"</span>/<span class="mono">"seven, eleven"</span> under +4). Intervention:
replace/promote the operand hypothesis at slot A in the recorded t2 sheet (leader &epsilon;=0.7 or
subleading &epsilon;=0.3), read the full number distribution at slot B in S<sup>t+1</sup>.</p>
<img class="fig" src="figs/xtask_compute.png">
<div class="card" style="border-left:4px solid var(--accent)"><ul>
<li><b>The target slot returns f(X) in one step.</b> '+4' prompt: three&rarr;seven (.80), four&rarr;eight
(.88), five&rarr;nine (.69), seven&rarr;eleven (.89), eight&rarr;twelve (.72), nine&rarr;thirteen (.62) &mdash;
'twelve' and 'thirteen' appear in NO capture and NO attractor; they can only arise by applying the prompt's
offset to the injected operand. '+3' prompt: the X+3 diagonal, against the 'eight' attractor column.</li>
<li><b>Cross-control kills the rest-of-sheet confound:</b> swap sheet and prompt between the two tasks.
'+3' prompt on the '+4' task's sheet: perfect X+3 diagonal 8/8 (six&rarr;nine .94, seven&rarr;ten .95);
'+4' prompt on the '+3' task's sheet: X+4 diagonal 7/8. The function parameter follows the PROMPT; the sheet
carries the operand. No unconditional co-occurrence table can produce two different functional curves for
identical injections.</li>
<li><b>Subleading entries leak through the same computation:</b> &epsilon;=0.3 injections (below the
incumbent) lift P(X+k)@B by 3&ndash;10&times; on about half the cells &mdash; the map mostly reads the
winner, but not only.</li>
<li><b>Identity transport exists but is much weaker:</b> the noun-copy task moves the injected word to the
partner slot content-independently (all 6 nouns, 100&ndash;1000&times; over the 10<sup>&minus;4</sup> floor,
max .12) &mdash; the strongly prompt-constrained arithmetic schema transports far better than free-form
copy.</li></ul>
<b>Synthesis:</b> S<sup>t</sup> IS a genuine computational interface in a single denoising step &mdash;
operand in, function value out, function selected by the prompt. The three nulls above are therefore
statements about WHICH computations the one-step map runs, not about whether S is executable: local,
prompt-specified functional dependencies between slots are computed within a step; global self-referential
constraint checks (mirror structure, counts, validity) are not &mdash; those exist only in the iterated loop,
where the pull is attractor-shaped. This also gives the ember story its final grounding: the sheet's
subleading entries are live inputs to real per-step computation, which is why S-side interventions carry
causal power everywhere in this study &mdash; while never being audited against the task constraint.</div>

<h3>Subleading entries specifically: parallel pushforward &mdash; and the register asymmetry</h3>
<p><b>Interactive explorer:</b> <a href="scompute.html">scompute.html</a> &mdash; operand&rarr;image
matrices with prompt/state/arm dropdowns, per-draw sinas, sheet rows before/after injection, the register
chart, and the pooled-stats table.</p>
<p>Reviewer's bar: the positive only counts if SUBLEADING S entries carry alternative-hypothesis content.
Battery (<span class="mono">xtask_compute3.py</span>): inject operand X at STRICTLY subleading rank
(&epsilon; &isin; {0.1, 0.3}, natural leader intact, rank asserted &ge;2), read the full number distribution
at B; plus incumbent-correctness cells where the operand is COMMITTED ON CANVAS instead (B renoised or
pinned to the attractor answer).</p>
<img class="fig" src="figs/xtask_subleading.png">
<div class="card" style="border-left:4px solid var(--accent)"><ul>
<li><b>Subleading hypotheses are computed on, in parallel and gradedly.</b> A rank-2 'four' (&epsilon;=0.3)
lifts P(seven)@B from .033 to .273 (8&times;), a rank-2 'six' lifts P(nine) from .006 to .228 (38&times;)
&mdash; while the leader image stays top-1 (.70&ndash;.93) in every cell: B's distribution is a weighted
mixture of the images of A's hypotheses. Dose-graded (&epsilon;=0.1 &rarr; 1.3&ndash;2.3&times;,
&epsilon;=0.3 &rarr; 2&ndash;38&times;), weight tracking the alternative's plausibility.</li>
<li><b>Observationally invisible:</b> natural B-output tails are dominated by attractor&plusmn;1
number-line smoothing (a correlational kernel), and a mass-weighted alignment test does NOT pick out the
true k over k&plusmn;1 nulls &mdash; the pushforward only shows once an alternative actually carries mass at
A. Parallel-hypothesis computation is real but causally hidden under the smoothing prior.</li>
<li><b>Powered replication (xtask_compute4.py): significant on every axis.</b> Third offset prompt
added (+2; fresh captures &mdash; attractor <span class="mono">"seven, nine"</span>, a third distinct natural
operand, computed correctly 6/6), sheets from 3 seeds &times; 2 steps = 12 states, all arms paired on
identical renoise draws. Pooled true-image lifts: n=80 cells, &times;8.1 geometric mean
(per prompt: +2 &times;31, +3 &times;3.2, +4 &times;4.9), Wilcoxon &gt;0 at p=4&times;10<sup>&minus;15</sup>.
Specificity: true-image lifts exceed neighbor-image (X+k&plusmn;1) lifts at p=7&times;10<sup>&minus;5</sup>
&mdash; the lifted region is CENTERED on the computed location (a &plusmn;1 smoothing kernel rides on top of
the computed image, consistent with the number-line prior). Register asymmetry under power: confirmation
boost n=36 paired contrasts, +0.025&nbsp;&plusmn;&nbsp;0.009, Wilcoxon p=8&times;10<sup>&minus;6</sup>;
computed-competitor elevation from WRONG canvas operands +0.008&nbsp;&plusmn;&nbsp;0.008 &mdash;
indistinguishable from zero, against &times;8.1 for the same operand carried in S.</li>
<li><b>The register asymmetry (answer to "does messing with the incumbent's correctness move the
competitor?"):</b> when the operand lives in S, yes, dramatically (S-leader 'four' &rarr; incumbent 'eight'
collapses .88&rarr;.35, computed-correct 'seven' takes .62). When the operand is COMMITTED ON CANVAS, no:
P(computed competitor) stays at floor (&le;.004) for every wrong operand, whether B is renoised or pinned
&mdash; the committed token never enters the computation. The only canvas effect is a CONFIRMATION boost:
the correct operand sharpens the incumbent (+.05&ndash;.07), wrong ones do nothing. One-step checking is
confirmation-asymmetric: consistent evidence reinforces, inconsistent evidence is not even read.</li></ul>
<b>This closes the arc's central paradox:</b> the incumbent-wrongness nulls (palindrome, self-count, occ)
were never just "the map can't compute those constraints" &mdash; here the map demonstrably CAN compute the
relation, and committed-but-wrong content STILL fails to elevate the computed competitor, because
commitment removes content from the negotiation's input registers. Error signals enter only through S-side
hypotheses; the canvas contributes confirmation, never contradiction. The trap phenomenology of the whole
study &mdash; incumbency self-reinforcement, veto-not-seed, attractor-directed renegotiation &mdash; follows
from this register asymmetry.</div>

<h3>Superposition: simultaneous multi-token injections</h3>
<p>Load n &isin; {1,2,3,4,6} operand hypotheses into slot A's sheet row AT ONCE (randomized mass ladder;
scheme fixedtok = per-token mass menu constant across n, scheme fixedtot = total budget 0.36 constant;
5 random subsets per cell &times; 3 prompt-states, paired draws). Read every image at B.</p>
<img class="fig" src="figs/xtask_superpos.png">
<div class="card" style="border-left:4px solid var(--accent)"><ul>
<li><b>The SET is transported, faithfully in content:</b> mean image lift stays 0.4&ndash;0.7 log10 even at
n=6 simultaneous hypotheses &mdash; every injected operand's image rises, in parallel.</li>
<li><b>The WEIGHTS are scrambled &mdash; powered re-estimate (120 cells/n, xtask_compute5b.py):</b>
own-baseline Spearman &rho;(&epsilon;<sub>i</sub>, image lift) = .44 at n=1 (the first-pass .69 on 15 cells
was small-sample flattered), decaying to .09 at n=6; scored against the FIELD (lift &minus; max non-target
lift) it is flat at .07&ndash;.14 for every n with the n=1 CI including zero. Even a SINGLE subleading
hypothesis carries only modest mass&rarr;response fidelity, and none in competitive terms &mdash;
presence-coding over magnitude-coding, now with tight CIs.</li>
<li><b>Transport is cooperative, not competitive:</b> per-unit-mass efficiency (&Sigma; image gain /
&Sigma; injected mass) RISES with n (.017&rarr;.111 fixedtok), and the attractor image's mass drops
progressively (&minus;.02&rarr;&minus;.11) &mdash; sibling hypotheses jointly destabilize the incumbent,
opening headroom for all images at once. Consistent with the trigger-like dose law (&chi;&prop;1/&epsilon;):
many small triggers beat one large one.</li></ul>
<b>Subleading check (review):</b> injection ranks were recorded per token; 476/480 landed strictly
subleading (leaders: 2/240 fixedtok at n=6, 2/240 fixedtot at n=1 &epsilon;=0.36). Restricting every
statistic to subleading-only pairs reproduces all numbers (&rho; identical at n=1&ndash;4; n=6: .196&rarr;.185)
&mdash; the superposition results are subleading-hypothesis results.<br>
<b>Reading:</b> the sheet-to-sheet computation is a parallel set-map with soft, poorly-resolved weights
&mdash; more Kalman-ish population code than exact mixture pushforward. This matches the earlier subleading
result (3&ndash;10&times; lifts regardless of exact rank) and the saturating single-token dose curves.</div>
<img class="fig" src="figs/xtask_superpos2.png">
<div class="card" style="border-left:4px solid var(--accent)">
<b>Target vs non-target decomposition (count on x, per-output lift on y):</b> target images rise with n
(&rho;=0.26, p=5&times;10<sup>&minus;5</sup>) &mdash; but so do NON-target numbers (unrelated: &rho;=0.32,
p=10<sup>&minus;7</sup>): a large share of the multi-injection response is DIFFUSE field-opening (the whole
number field rises as the incumbent weakens), and pooled means of targets and unrelated numbers nearly touch
at n=4. The pooled view is a composition artifact though &mdash; PAIRED within cells, the target-specific
edge over unrelated numbers is significant at every n (powered, 120 cells/n: +0.13/+0.13/+0.13/+0.16/<b>+0.23</b> log10 for n=1/2/3/4/6; Wilcoxon p&le;2&times;10<sup>&minus;11</sup> everywhere — the first-pass n=2 weak spot at p=.068 was sample size). Sibling mass-ORDERING however is near-chance: the higher-&epsilon; operand's image lifts more in only 58% of n=2 pairs (p=.04), 52&ndash;55% at higher n: the computed channel survives superposition, riding on
the growing diffuse component. Identity leak (the injected VALUE itself at B) behaves like the diffuse class
&mdash; consistent with the weak C1 identity copy: the map transports images, not copies.</div>

<h3>Content identification: does an injection nudge the EXPECTED output more than ANY other?</h3>
<p>Operationalization (review): per injection cell, rank ALL candidate outputs by paired lift; ask whether
the expected image X+k is strictly the most-lifted (top-1; chance 1/11), measure the selectivity margin
lift(target) &minus; max lift(other), and decode the response backwards (argmax-lifted output &minus; k =
inferred operand) &rarr; confusion matrix, accuracy, mutual information.</p>
<img class="fig" src="figs/xtask_content_id.png">
<div class="card" style="border-left:4px solid var(--accent)"><ul>
<li><b>Leader-strength content: yes, categorically.</b> Top-1 rate 14/14 = 100%
(p=3&times;10<sup>&minus;15</sup>), decoding accuracy 1.00, margin +1.27 log10 &mdash; the expected output is
nudged more than any other, every time, by over an order of magnitude.</li>
<li><b>Subleading content: yes in half the cells, far above chance.</b> Top-1 rate 38/80 = 47% vs 9% chance
(binomial p=5&times;10<sup>&minus;19</sup>); decoding accuracy 0.47 (chance .125); the response carries
<b>1.18 bits of identifiable content per subleading injection</b> (max 3). The margin distribution straddles
zero: in the other half of cells a neighbor or the diffuse field edges out the target &mdash; content is
transmitted probabilistically, not deterministically, at subleading strength.</li>
<li><b>Sets: identifiable above chance, never exactly &mdash; and the misses are structured.</b> Top-n
lifted outputs = the n expected images exactly only at n=1 (40%); Jaccard stays ~1.5&times; above the
random-set baseline at every n (0.53 vs 0.37 at n=6). This does NOT contradict the positive within-set
Spearman: &rho; is ordinal among the injected images only, while exact set match requires every image to
beat the max over ~9 non-targets (extreme-value criterion) &mdash; and the intruder census shows what wins
instead is almost entirely image&plusmn;1 neighbors and self-leaks (random 'other' intruders fall 13&rarr;1
from n=2 to n=6; same result under absolute &Delta;P ranking). The map transmits a REGION (image &plusmn;1
+ identity echo, ~1.2 bits/hypothesis); the exact argmax inside it is a coin flip.</li></ul>
<b>Summary quantification:</b> "nudges the expected logits more than any other" holds with probability
&asymp;0.5 per subleading hypothesis (1.2 bits/injection) and probability 1.0 at leader strength &mdash;
content identifiability is strong and graded by the hypothesis's mass, even though the mass itself is not
propagated as a weight.</div>
<p>Companion figure (axes per review: x = injected, y = decoded; weight-fidelity panel adjacent):</p>
<img class="fig" src="figs/xtask_content_spearman.png">
<p>Same matshow next to the mean effect on the aligned output grid (superpos paradigm): each right-panel
cell is &lang;R<sub>k,t</sub>(x&prime;|x)&rang;<sub>k,r,t</sub> &mdash; x = perturbed subleading input
token, x&prime; = output token, averaged over prompt shift k, rollout seed r and sheet step t; inside
R<sub>k,t</sub>, &lang;&middot;&rang;<sub>s</sub> averages the 12 paired renoise draws before the ratio &mdash;
the diffuse ~+0.3 background is the weakened-leader susceptibility lift; the diagonal excess above it is
the transported content:</p>
<img class="fig" src="figs/xtask_confusion_effect.png">
<div class="card"><b>Definitions (used in all specificity/superposition figures).</b> A cell c =
(prompt &tau; with offset k, sheet-state &sigma;, injected set S = {x<sub>1</sub>&hellip;x<sub>n</sub>} with
masses &epsilon;<sub>i</sub>), evaluated on D=8 PAIRED renoise draws (identical canvases arm vs base).
First average is over draws, before any ratio/difference:
<span class="mono">P&#772;<sub>c</sub><sup>arm</sup>(w) = (1/D)&Sigma;<sub>d</sub> P<sub>c,d</sub><sup>arm</sup>(w)</span>,
likewise base. Targets T(c) = {x<sub>i</sub>+k &le; 13}; attractor image j<sub>a</sub> excluded everywhere;
non-targets N(c) = W&prime;&setminus;T(c); unrelated U(c) = N(c) minus image&plusmn;1 neighbors minus
injected values (self-leak).<br>
<span class="mono">&Delta;P<sub>c</sub>(w) = P&#772;<sup>arm</sup>(w) &minus; P&#772;<sup>base</sup>(w)</span>;
total specific gain
<span class="mono">G<sub>c</sub> = &Sigma;<sub>w&isin;T</sub>&Delta;P<sub>c</sub>(w) &minus; |T|&middot;mean<sub>w&isin;N</sub>&Delta;P<sub>c</sub>(w)</span>
(per-hypothesis: G<sub>c</sub>/|T|).<br>
log-lift <span class="mono">&#8467;<sub>c</sub>(w) = log10( max(P&#772;<sup>arm</sup>(w),&delta;) /
max(P&#772;<sup>base</sup>(w),&delta;) )</span>, &delta;=10<sup>&minus;5</sup> (ratio of draw-means);
edge <span class="mono">E<sub>c</sub> = mean<sub>w&isin;T</sub>&#8467;<sub>c</sub>(w) &minus;
mean<sub>w&isin;R</sub>&#8467;<sub>c</sub>(w)</span> with R = N (vs ALL non-targets) or U (vs unrelated).
Averaging hierarchy: draws &rarr; outputs within class &rarr; cells (sina dots = per-cell E<sub>c</sub>;
mean&plusmn;SE and Wilcoxon across cells; Spearman figures instead pool per-injection pairs
(&epsilon;<sub>i</sub>, &#8467;<sub>c</sub>(x<sub>i</sub>+k)) with cell-level bootstrap).</div>
<p><b>The specificity quantity on its own (order-preservation set aside):</b> per-cell specific edge =
mean lift(target images) &minus; mean lift(reference class), sina per n over the merged 120-cells/n pool.
Flat at +0.11&ndash;0.13 for n=1&ndash;3 and RISING to +0.20&ndash;0.23 at n=6, Wilcoxon
p&le;5&times;10<sup>&minus;11</sup> everywhere, and nearly identical against unrelated-only vs ALL
non-targets (the &plusmn;1/identity spillover does not eat the edge):</p>
<img class="fig" src="figs/xtask_specific_edge.png">
<p><b>Diminishing returns? The opposite &mdash; increasing returns to the task ceiling</b>
(<span class="mono">xtask_compute6.py</span>: flat &epsilon;=0.10 per hypothesis, n = 1&hellip;7 where 7 =
ALL non-natural operands at once, 9 prompt&times;state combos):</p>
<img class="fig" src="figs/xtask_diminishing.png">
<div class="card" style="border-left:4px solid var(--accent)">
The per-hypothesis specific edge RISES with count (+0.07 &rarr; +0.24 log10 from n=1 to 7), per-hypothesis
absolute specific &Delta;P rises 21&times; (+0.0006 &rarr; +0.0128), and TOTAL specific gain grows
super-linearly (+0.0006 &rarr; +0.0896, 21&times; above linear scaling from n=1). Panel C shows why:
a lone &epsilon;=0.10 hypothesis barely dents the incumbent (attractor &Delta;P &minus;0.015), so its image
has no headroom; siblings jointly erode it (&minus;0.115 at n=7) while the DIFFUSE non-target field
saturates after n&asymp;3 &mdash; the additionally freed probability flows increasingly into the computed
images specifically. The bottleneck is incumbent suppression, not computation bandwidth: parallel hypothesis
processing shows no saturation up to &Sigma;&epsilon;=0.7, the deepest perturbation this task admits.
Diminishing returns presumably begin only where &Sigma;&epsilon;&rarr;1 (natural row annihilated) or images
collide &mdash; beyond this task's operand supply.</div>
<p><b>Majority-effect control (review: at n=7 targets were 7/11 of the narrow field):</b> new prompts
'+8'/'+11' extend the answers to <span class="mono">twenty</span> (19-word candidate field; DG solves both
natively and correctly: <span class="mono">"five, thirteen"</span> / <span class="mono">"seven,
eighteen"</span> &mdash; two more offsets for the parameterization evidence). Same flat-&epsilon; series,
targets now always &le;7/18:</p>
<img class="fig" src="figs/xtask_widefield.png">
<div class="card" style="border-left:4px solid var(--accent)">
The rise is NOT a shrinking-reference artifact &mdash; in the wide field it STEEPENS: E<sub>c</sub> climbs
monotonically +0.13 &rarr; +0.91 (n=1&rarr;7, Wilcoxon p&le;1.5&times;10<sup>&minus;11</sup> from n=3), vs
+0.07 &rarr; +0.24 narrow. And the mechanism decomposes cleanly: the diffuse non-target field saturates at
n&asymp;4 and recedes by n=7 while the specific edge keeps growing &mdash; with MINIMAL incumbent erosion
(&Delta;P(j<sub>a</sub>) only &minus;0.04 at n=7 here) the cooperative gains flow almost exclusively into the
computed images. Parallel processing of simultaneous hypotheses shows increasing, not diminishing, returns
throughout the accessible range; no saturation of computation bandwidth is observable.</div>
<p><b>Deep-n saturation probe (review: "this should definitely saturate"; more ks, more base values):</b>
a format prefix (<span class="mono">"Begin your answer with 'Numbers:'"</span>) puts the operand in spaced
token form, unlocking operands past nine (pools to 16) and n up to 14; four new offsets k=2/3/6/9. DG
solves all natively with teen operands: <span class="mono">"Numbers: twelve, fourteen"</span> /
<span class="mono">"twelve, fifteen"</span> / <span class="mono">"eight, fourteen"</span> /
<span class="mono">"seven, sixteen"</span> (8/8 seeds; nine verified offsets total).</p>
<img class="fig" src="figs/xtask_deepn.png">
<div class="card" style="border-left:4px solid var(--accent)">
<b>The computed target signal never saturates:</b> at k=3, &lang;&#8467;&rang;<sub>T</sub> grows
+0.40&rarr;+1.00 log10 (10&times;) monotonically out to n=14 simultaneous hypotheses (&Sigma;&epsilon;=0.7),
and the k=3 edge still rises (+0.35 at n=14). <b>The one genuine saturation found is channel CROWDING, not
bandwidth:</b> at k=2 the edge collapses to &asymp;0 for n&le;10 &mdash; but panel B shows its targets still
lift; the specificity is eaten because the images sit only 2 away from the dense injected cluster, so the
reference class absorbs the same leak/neighbor spillover (reference-to-operand distance &rarr;0.4 at deep
n). k=9, whose image band is far from the operand cluster, keeps a large clean edge (+0.39 at its pool
ceiling). VERDICT on the skepticism: saturation exists, but it is a RESOLUTION limit (nearby channels
blur together on the number line) &mdash; not a limit on how many parallel computations the one-step map
can run. Within everything accessible (n&le;14, &Sigma;&epsilon;&le;0.7, 9 offsets, operands 2&ndash;18),
raw parallel capacity never binds.</div>
<p><b>Smoothed, operation-diverse deep-n series</b> (xtask_compute9: +5/+7 fill the k-gaps; SUBTRACTION
&minus;3/&minus;8 with operands to twenty; MULTIPLICATION &times;10 exploiting single-token tens words
&mdash; DG natively computes all of them: <span class="mono">"Numbers: fifteen, seven"</span> (&minus;8),
<span class="mono">"Numbers: seven, seventy"</span> (&times;10); 12 verified operations total. Compound
numbers 21+ are multi-token so the contiguous field ends at twenty; deeper n comes from subtraction's
larger operand pools. 10&ndash;12 cells/point):</p>
<img class="fig" src="figs/xtask_deepn.png">
<div class="card" style="border-left:4px solid var(--accent)"><ul>
<li><b>The count-cooperativity replicates across operation type:</b> subtraction rises like addition
(&minus;3: +0.56 at n=14; &minus;8: +0.59 at n=10), and the crowding law generalizes &mdash; edge amplitude
ordering tracks |k| (2 &lt; 3&asymp;5&asymp;6 &lt; 7&asymp;9 and &minus;3 &lt; &minus;8 at matched n): the
farther the image band from the operand cluster, the cleaner the measurable specificity.</li>
<li><b>Multiplication is the outlier &mdash; the first TARGET-side saturation:</b> &times;10's edge stays
low (+0.04&hellip;+0.15) and its target lift BENDS DOWN at n=6 (+0.54&rarr;+0.43) while its reference
(mostly units) rises to meet it &mdash; despite the maximally disjoint image band. The &times;10 pushforward
is real at small n but co-activates the whole number field rather than the specific tens images as
hypotheses accumulate; parallel capacity appears operation-dependent: robust for the &plusmn;k number-line
step, shallow for &times;10.</li></ul>
Pooled per point: 2 sheet-seeds &times; 4&ndash;6 operand-subset draws (within cell: 8 paired renoise draws,
n injected operands); not pooled: operation (curves), n (x-axis).</div>
<p><b>The definitive depth test (review: keep the non-target class LARGE while pushing n; the shrinking
reference could let a domain-shaped generic response inflate the edge):</b> letter-domain case-flip tasks
&mdash; <span class="mono">"Pick any lowercase letter between a and w, write it, then write the letter
three/seven positions later in the alphabet in uppercase"</span>. Field = 52 single-token candidates (both
cases); operand band (lowercase) and image band (uppercase) fully disjoint; n to 20 at &epsilon;=0.04
(&Sigma;&epsilon;&le;0.8); and a bias-proof reference: the UNTOUCHED-CASE class (lowercase letters never
injected &mdash; typical band members that can never be targets or leak slots). DG solves both natively:
<span class="mono">"Letters: g, J"</span> / <span class="mono">"g, N"</span> (4/4) &mdash; the one-step
parameterized computation is not number-specific.</p>
<img class="fig" src="figs/xtask_letters.png">
<div class="card" style="border-left:4px solid var(--accent)">
<b>No saturation to n=20, under the honest reference.</b> The untouched-case edge rises monotonically:
L3 +0.10&rarr;+0.55, L7 +0.04&rarr;+0.96 (n=18); target lift itself reaches +1.10 log10 with the all-NT
reference class never below 31 members. The two references agree (untouched-case edge is consistently the
LARGER one &mdash; the all-NT reference was diluted by contaminated slots, so prior curves if anything
UNDERSTATED specificity). The alphabet also reproduces the |k| crowding law (L7 &gt; L3 throughout).
Cumulative verdict across 14 operations, 3 domains, n&le;20, &Sigma;&epsilon;&le;0.8: the only saturations
found are channel crowding (small |k|) and operation complexity (&times;10); parallel hypothesis COUNT never
binds in the accessible regime.</div>
<p><b>Normalization control (review: divide E by total injected mass?):</b> the flat-&epsilon; series
confound count with total mass &Sigma;&epsilon; = n&epsilon;. Decisive comparison at MATCHED &Sigma;: one
hypothesis carrying the whole mass (n=1, &epsilon;=&Sigma;) vs many sharing it, plus a
double-&epsilon; flat series &mdash; same states, canvases and bases as the deep series:</p>
<img class="fig" src="figs/xtask_dosecount.png">
<p><b>Dedicated interactive report for this result:</b> <a href="symbol_arithmetic.html">symbol_arithmetic.html</a> &mdash;
per-cell sinas of E/&Sigma;&epsilon; vs n, click any dot for the full cell inspector (base vs intervention
side by side, injected entries with individual masses and ranks, S<sup>t</sup> before/after,
S<sup>t+1</sup>, per-draw sina), with an explicit index-set caption for every average.</p>
<div class="card" style="border-left:4px solid var(--accent)"><ul>
<li><b>The series do NOT collapse onto a &Sigma;&epsilon; curve &mdash; and concentration wins, massively.</b>
At &Sigma;&epsilon;=0.8: one-big E = +3.4 (numbers) / +2.9 (letters) vs +0.34/+0.41 for ten flat hypotheses.
E is NOT a function of total mass; per-target, a concentrated hypothesis is &sim;10&times; (in log-edge)
more effective than the same mass distributed. (This REVERSES the earlier "many small triggers beat one
large" reading, which was an absolute-&Delta;P statement at small doses: in log-edge units the single-dose
response is superlinear and unsaturated to &epsilon;=0.8.)</li>
<li><b>The requested normalization, E/&Sigma;&epsilon; vs n (POWERED: 24&ndash;28 independent cells per
point &mdash; 4 sheet-seeds &times; 4&ndash;8 fresh operand-subset draws, no resampling): decays over
n=1&ndash;3 and then holds FLAT at 0.42&ndash;0.60 from n&asymp;4 to n=20, both domains, with tight
SEs.</b> So the honest restatement of the
"increasing returns": per-hypothesis edge rises with n at fixed per-hypothesis mass, but per-unit-TOTAL-mass
the process is sub-proportional &mdash; the rising E(n) curves partly reflect rising total dose. What
remains genuinely count-positive: the per-unit-mass edge stops decaying after n&asymp;4 (distributing mass
ever wider costs nothing more), and only the many-hypothesis regime computes many images at once &mdash;
concentration buys depth on ONE channel, superposition buys breadth at a fixed &asymp;0.5 edge-per-mass
rate.</li></ul>
<b>Corrected summary:</b> the one-step map offers a clean trade: one hypothesis &rarr; near-deterministic
single image (E&rarr;3+); n hypotheses &rarr; n parallel images at constant per-mass efficiency, no count
saturation to n=20. "Cooperativity" survives as non-interference plus shared incumbent erosion &mdash; not
as count outperforming mass.</div>

<h2>Intervention taxonomy: every op used in this study, with S<sup>t</sup> before &rarr; after</h2>
<p>Convention: S<sup>t</sup> = the self-conditioning logit sheet produced by step t and consumed (via softmax)
by step t+1; entries below are quoted as post-softmax probabilities at the touched position(s).
"Probe-only" ops act inside a single <span class="mono">/energy</span> readout step and never enter a rollout.</p>
<table class="rz" id="ivtab"><tr>
<th>intervention</th><th>acts on</th><th>exact operation</th><th>S<sup>t</sup> before</th><th>S<sup>t</sup> after / effect on S<sup>t+1</sup></th><th>scope</th><th>used in</th></tr>
<tr><td>S-kill (<span class="mono">s_bump</span>)</td><td>S<sup>t</sup> entries</td>
<td><span class="mono">S[p_j, w_j] += &minus;3&times;10&#8308;</span> at the 5 contested (position, mode-token) pairs, applied to the sheet AFTER step t computed it</td>
<td>ember arm: p(idiom<sub>j</sub>) &asymp; 0.02&ndash;0.03 under live incumbent; rescue arm: p(seasonal<sub>j</sub>) &asymp; 0.9+</td>
<td>p(w<sub>j</sub>) &rarr; 0; remaining entries renormalize &times;1/(1&minus;p<sub>del</sub>) at consumption</td>
<td>[t, t+1) or [t, &infin;)</td><td>ember-kill (rebound t3&ndash;6 on s3), incumbent-kill (rescue 17/17)</td></tr>
<tr><td>S rank-ops (<span class="mono">swap12</span> / <span class="mono">drop-r</span>)</td><td>S<sup>t</sup> entries</td>
<td>exchange the top-1/top-2 LOGIT values at one position (rank re-evaluated each step); or set the rank-r entry to &minus;3&times;10&#8308;</td>
<td>rank order (w&#8321;, w&#8322;, &hellip;) with masses (p&#8321; &gt; p&#8322;)</td>
<td>(w&#8322;, w&#8321;, &hellip;) &mdash; same mass multiset, leader swapped; drop: rank-r mass &rarr; 0</td>
<td>per-step window, one position (or all live slots)</td><td>negotiation slot taxonomy; beam-switch (swap12-all flips 3/3 idiom&rarr;seasonal)</td></tr>
<tr><td>S-mode nulls (<span class="mono">s_mode</span>)</td><td>whole sheet</td>
<td>echo: S := one-hot(current canvas token) per position; flat: S := 0 (uniform)</td>
<td>full belief sheet (multi-hypothesis, graded)</td>
<td>echo: exactly the canvas, no soft surplus; flat: nothing &mdash; all globally-coupled tasks die</td>
<td>every step</td><td>S-echo null (main report)</td></tr>
<tr><td>S truncation (<span class="mono">trunc_s k</span>)</td><td>sheet tail</td>
<td>keep top-k per position, rest &rarr; &minus;3&times;10&#8308; (mass renormalizes at consumption)</td>
<td>full 262k-entry tail per position</td>
<td>top-k head only; k=5 &equiv; base byte-identical (the plan fits in top-5; the tail's role is timing)</td>
<td>every step</td><td>S-plan compactness (main report)</td></tr>
<tr><td>S promotion (&epsilon;-mix)</td><td>one sheet entry</td>
<td><span class="mono">p' = (1&minus;&epsilon;)p + &epsilon;&middot;&delta;_w</span> at one (slot, token); rank-32 entry displaced if w is new</td>
<td>p(w) &asymp; 0 (break/compet/neutral) or the incumbent's own mass (sharpen)</td>
<td>p(w) = &epsilon; + (1&minus;&epsilon;)p(w); measured: competitor mass in S<sup>t+1</sup> at OTHER slots (break +0.375 log-units, 5.6&sigma; vs neutral)</td>
<td>probe-only</td><td>S-promotion, powered verdict</td></tr>
<tr><td>sheet re-tempering (&tau;)</td><td>whole sheet</td>
<td><span class="mono">lp' = log_softmax(lp/&tau;)</span> over the recorded top-32 per position</td>
<td>native sharpness: incumbent &asymp; 0.99 at contested slots (s0@t8)</td>
<td>flattened incumbency; releases the competitor's baseline 3&times;10&#8315;&#8309; &rarr; 3.5&times;10&#8315;&sup3; (&tau;=8)</td>
<td>probe-only</td><td>susceptibility &chi;(k, &tau;)</td></tr>
<tr><td>j-space ablation (<span class="mono">state_ablate</span>)</td><td>residual h<sub>21</sub> (upstream of S)</td>
<td><span class="mono">h &larr; h &minus; QQ&#7488;h</span>, Q = orth(J&#8321;&#8321;&#7488;W<sub>U</sub>[ids]); ids = 5 idiom / top-100 jlens neighborhood / 100 random</td>
<td>S<sup>t</sup> not touched directly; produced from the intact residual</td>
<td>S<sup>t</sup> computed with the ids' directions removed at the 5 positions &mdash; their logits (and lexical neighborhood) suppressed; commitment survives (persistent js100 &rarr; degraded idiom, never seasonal)</td>
<td>steps &times; positions scoped</td><td>jsN / js100 / random control</td></tr>
<tr><td>canvas clamp (<span class="mono">clamp_ids</span>)</td><td>x<sup>t</sup> (canvas)</td>
<td><span class="mono">x[p] := c_p</span> re-imposed EVERY step &ge; t&#8320; (unhealable)</td>
<td>unchanged by the op itself</td>
<td>indirect: the slot leaves the renoise pool, exits the soft negotiation; kills the escape even when c_p is the competitor's own token</td>
<td>[t&#8320;, &infin;), chosen positions</td><td>veto-not-seed clamps</td></tr>
<tr><td>canvas plant (<span class="mono">init_text</span>)</td><td>x<sup>t&#8320;</sup> + S<sup>t&#8320;</sup></td>
<td>one-shot: canvas := planted row; <b>S := 0 at the plant step</b>; eviction allowed afterwards</td>
<td>whatever the rollout had (t&#8320;=0: nothing)</td>
<td>zeroed &mdash; canvas-only evidence; fluent trap installs itself 2/4, obvious wrongness evicted &rarr; idiom</td>
<td>one step</td><td>wrongness ladder</td></tr>
<tr><td>donor injection (<span class="mono">donor_prompt, &alpha;</span>)</td><td>x + S jointly at step j</td>
<td>x := donor canvas; <span class="mono">S := (1&minus;&alpha;)S_own + &alpha;S_donor</span>, donor = native rollout under a sibling prompt captured after step j</td>
<td>recipient's native palindrome-prompt sheet</td>
<td>&alpha;=1: the donor's full foreign incumbent installed (repair / evict / capitulate by content &amp; j); &alpha;=0: own S kept &mdash; foreign canvas evicted, outcome unchanged every cell</td>
<td>one step (state splice)</td><td>on-policy prompt-swap</td></tr>
<tr><td>canvas corruption probes (k random / v systematic)</td><td>x only, probe-only</td>
<td>replace k of 8 phrase tokens with random tokens, or make fluent edits changing the violated-pair count v (plus edit-only controls)</td>
<td>reconstructed sparse sheet, UNCHANGED by the op</td>
<td>S<sup>t+1</sup> response: saturating &asymp;7&times; edit-detection bump (random), NO v-dependence (systematic) under the unfaithful probe; see fidelity correction</td>
<td>probe-only</td><td>susceptibility, systematic ladder</td></tr>
</table>
<script>
(function(){
  const tab=document.getElementById('ivtab');
  tab.querySelectorAll('tr:first-child th').forEach((th,ci)=>{
    const g=document.createElement('div');g.className='colgrip';th.style.position='relative';th.appendChild(g);
    g.addEventListener('mousedown',e=>{e.preventDefault();
      const w0=th.offsetWidth,x0=e.clientX;
      const mv=ev=>{th.style.minWidth=Math.max(40,w0+ev.clientX-x0)+'px';};
      const up=()=>{document.removeEventListener('mousemove',mv);document.removeEventListener('mouseup',up);};
      document.addEventListener('mousemove',mv);document.addEventListener('mouseup',up);});
  });
  tab.querySelectorAll('tr').forEach(tr=>{
    const td=tr.querySelector('td,th');if(!td)return;
    const g=document.createElement('div');g.className='rowgrip';td.style.position='relative';td.appendChild(g);
    g.addEventListener('mousedown',e=>{e.preventDefault();
      const h0=tr.offsetHeight,y0=e.clientY;
      const mv=ev=>{tr.style.height=Math.max(20,h0+ev.clientY-y0)+'px';};
      const up=()=>{document.removeEventListener('mousemove',mv);document.removeEventListener('mouseup',up);};
      document.addEventListener('mousemove',mv);document.addEventListener('mouseup',up);});
  });
})();
</script>

<h2>Explorer: scrub the generations</h2>
<p class="dim">Pick a rollout, scrub the step slider (or &larr;/&rarr; keys, or click a step row in the detail panel).
The strip shows the denoiser's per-position argmax draft at the current step; tiny bars inside each pill =
that position's S-mass on the idiom (green) / seasonal (orange) mode tokens (from the recorded top-10).
Underline = entropy-gate committed (proxy); green fill = matches the final canvas; gold text = changed vs
previous step; gold inset = the 5 contested slots; red inset = the kill is deleting this slot's mode token
at this step. Click a position (strip or grid header) for its full S top-10 trajectory across steps.
Below the strip: the <b>lens grid</b> at the current step &mdash; rows are the OUT channel (the S sheet's own
top-10) on top, then layers L25/L21/L17 descending (last layer on top), each showing the chosen lens's
top-10 chips per position (chip opacity &prop; &radic;p; idiom tokens green, seasonal gold); the
jlens&nbsp;/&nbsp;logit-lens choice is the dropdown.</p>
<div class="controls">
<select id="runSel"></select>
<button id="playBt" title="play / pause (space)">&#9654;</button>
<input type="range" id="stepSl" min="0" max="19" value="0">
<span id="stepLab" class="mono"></span>
</div>
<div class="dim">task prompt: <span class="mono">__TASKPROMPT__</span></div>
<div id="runMeta" class="dim"></div>
<div id="cstrip" class="strip"></div>
<div class="controls">
<span class="dim">lens grid:</span>
<select id="lensVar"><option value="j">jlens (J-transported)</option><option value="l">logit-lens (raw residual)</option></select>
<span id="lensNote" class="dim"></span>
</div>
<div class="lgwrap"><div id="lgrid"></div></div>
<div id="pdet"></div>
<script>
const $=i=>document.getElementById(i);
let IDX=[], RUN=null, sel=0, seld=null;
const esc=s=>s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
const tstr=i=>esc((RUN.id2str[i]||'?').replace(/\\u2581/g,' '));
const killAt=t=>RUN.kill&&(RUN.kill.persist?t>=RUN.kill.t:t===RUN.kill.t);
function modeMass(st,p,ids){let m=0;st.topk_ids[p].forEach((t,i)=>{if(ids.includes(t))m+=st.topk_p[p][i]});return m;}
function render(){
  const st=RUN.steps[sel], pv=sel>0?RUN.steps[sel-1]:null;
  $('stepLab').textContent=`step ${sel}/${RUN.T-1}`+(killAt(sel)?'  \\u2702 kill active':'');
  $('cstrip').innerHTML=st.argmax.map((id,p)=>{
    const cls=['tk'];
    if(st.committed[p])cls.push('com');
    if(id===RUN.final_ids[p])cls.push('fin');
    if(pv&&pv.argmax[p]!==id)cls.push('new');
    if(RUN.slots.includes(p))cls.push('slot');
    if(killAt(sel)&&RUN.slots.includes(p))cls.push('killed');
    if(p===seld)cls.push('selp');
    const mi=modeMass(st,p,RUN.diff_idt), ms=modeMass(st,p,RUN.diff_set);
    return `<span class="${cls.join(' ')}" data-p="${p}" title="p${p}: ${tstr(id)}"><span class="tkt">${tstr(id)}</span>`+
      `<span class="bars"><i class="bi" style="height:${Math.round(13*mi)}px"></i>`+
      `<i class="bs" style="height:${Math.round(13*ms)}px"></i></span></span>`;
  }).join('');
  renderDetail();
  renderLens();
}
function renderDetail(){
  if(seld===null){$('pdet').innerHTML='';return;}
  let h=`<h3>position ${seld} — final <span class="tokpill">${tstr(RUN.final_ids[seld])}</span>`+
        (RUN.slots.includes(seld)?' <span class="dim">(contested slot)</span>':'')+'</h3>';
  h+=RUN.steps.map((st,t)=>{
    const pills=st.topk_ids[seld].map((id,i)=>{
      const c=RUN.diff_idt.includes(id)?' mi':(RUN.diff_set.includes(id)?' ms':'');
      return `<span class="tokpill${c}" title="p=${st.topk_p[seld][i]}">${tstr(id)} ${st.topk_p[seld][i]}</span>`;
    }).join('');
    return `<div class="srow${t===sel?' cur':''}" data-t="${t}"><span class="sno">t${t}${killAt(t)&&RUN.slots.includes(seld)?'\\u2702':''}</span>`+
           `<span class="dim">${st.committed[seld]?'\\u25cf':'\\u25cb'}</span>${pills}</div>`;
  }).join('');
  $('pdet').innerHTML=h;
}
let LENS=null, lensVar='j';
const lstr=i=>esc((((LENS&&LENS.id2str[i])||RUN.id2str[i]||'?')).replace(/\u2581/g,' '));
function chipStack(ids,ps){return '<div class="lstk">'+ids.map((id,i)=>{
  const a=(0.15+0.85*Math.sqrt(ps[i])).toFixed(2);
  const c=RUN.diff_idt.includes(id)?'var(--greenbg)':(RUN.diff_set.includes(id)?'var(--goldbg)':'var(--card)');
  return `<span class="lch" style="background:${c};opacity:${a}" title="p=${ps[i]}">${lstr(id)}</span>`;
}).join('')+'</div>';}
function renderLens(){
  const g=$('lgrid'); if(!RUN){g.innerHTML='';return;}
  const st=RUN.steps[sel];
  let h=`<div class="lgrid" style="grid-template-columns:max-content repeat(${RUN.P},78px)">`;
  h+=`<div class="lrowh">pos &rarr;</div>`;
  for(let p=0;p<RUN.P;p++)
    h+=`<div class="lgph${RUN.slots.includes(p)?' sloth':''}${p===seld?' selh':''}" data-p="${p}">${p}\n${tstr(st.argmax[p])}</div>`;
  const rows=[['OUT (S top-10)',null]].concat(LENS?[...LENS.layers].sort((a,b)=>b-a).map(l=>['L'+l,l]):[]);
  for(const [lab,l] of rows){
    h+=`<div class="lrowh">${lab}</div>`;
    for(let p=0;p<RUN.P;p++){
      if(l===null){h+=chipStack(st.topk_ids[p],st.topk_p[p]);}
      else{const s=LENS.per_layer[String(l)][Math.min(sel,LENS.per_layer[String(l)].length-1)];
           h+=(lensVar==='j')?chipStack(s.ji[p],s.jp[p]):chipStack(s.li[p],s.lp[p]);}
    }
  }
  g.innerHTML=h+'</div>';
}
async function loadLens(tag){
  LENS=null;$('lensNote').textContent='loading\u2026';
  try{const r=await fetch('lens_ember/'+tag+'_lens.json',{cache:'no-store'});
    if(r.ok){LENS=await r.json();$('lensNote').textContent='';}
    else{$('lensNote').textContent='no lens capture (yet) for this rollout \u2014 OUT row only';}
  }catch(e){$('lensNote').textContent='no lens capture (yet) \u2014 OUT row only';}
  renderLens();
}
async function loadRun(tag){
  RUN=await (await fetch('lens_ember/'+tag+'.json',{cache:'no-store'})).json();
  sel=Math.min(sel,RUN.T-1); seld=null;
  $('stepSl').max=RUN.T-1; $('stepSl').value=sel;
  const k=RUN.kill?`kill ${RUN.kill.mode} ids at t${RUN.kill.t}${RUN.kill.persist?'+':''}`:'no intervention';
  $('runMeta').innerHTML=`<b>${RUN.tag}</b>: ${k} &rarr; <b>${RUN.outcome}</b>, draft flip `+
    `${RUN.flip===null?'&mdash;':RUN.flip}, converged t${RUN.conv} &middot; final: <span class="mono">${esc(RUN.final_text)}</span>`;
  render();
  loadLens(tag);
}
(async()=>{
  IDX=await (await fetch('lens_ember/index.json',{cache:'no-store'})).json();
  const groups={};
  IDX.forEach(r=>{(groups['s'+r.seed]=groups['s'+r.seed]||[]).push(r);});
  $('runSel').innerHTML=Object.entries(groups).map(([g,rs])=>
    `<optgroup label="${g}">`+rs.map(r=>{
      const k=r.kill?`${r.kill.mode==='idiom'?'kill':'rescue'}@t${r.kill.t}${r.kill.persist?'+':''}`:'base';
      return `<option value="${r.tag}">${g} ${k} \\u2192 ${r.outcome}${r.flip!==null?' (flip '+r.flip+')':''}</option>`;
    }).join('')+'</optgroup>').join('');
  let timer=null;
  const stopPlay=()=>{if(timer){clearInterval(timer);timer=null;$('playBt').innerHTML='&#9654;';}};
  const startPlay=()=>{if(!RUN||timer)return;
    if(sel>=RUN.T-1){sel=0;$('stepSl').value=sel;render();}
    timer=setInterval(()=>{if(sel>=RUN.T-1){stopPlay();return;}sel++;$('stepSl').value=sel;render();},400);
    $('playBt').innerHTML='&#9208;';};
  $('playBt').onclick=()=>timer?stopPlay():startPlay();
  $('runSel').onchange=e=>{stopPlay();loadRun(e.target.value);};
  $('stepSl').oninput=e=>{stopPlay();sel=+e.target.value;render();};
  $('cstrip').onclick=e=>{const el=e.target.closest('.tk');if(el){seld=+el.dataset.p;render();}};
  $('lgrid').onclick=e=>{const el=e.target.closest('.lgph');if(el){seld=+el.dataset.p;render();}};
  $('lensVar').onchange=e=>{lensVar=e.target.value;renderLens();};
  $('pdet').onclick=e=>{const el=e.target.closest('.srow');if(el){stopPlay();sel=+el.dataset.t;$('stepSl').value=sel;render();}};
  document.addEventListener('keydown',e=>{
    if(!RUN||/INPUT|SELECT/.test(e.target.tagName))return;
    if(e.key==='ArrowRight'&&sel<RUN.T-1){stopPlay();sel++;$('stepSl').value=sel;render();e.preventDefault();}
    if(e.key==='ArrowLeft'&&sel>0){stopPlay();sel--;$('stepSl').value=sel;render();e.preventDefault();}
    if(e.key===' '){timer?stopPlay():startPlay();e.preventDefault();}
  });
  loadRun(IDX[0].tag);
})();
</script>
"""


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from constrained_common import CHOSEN
    q = next(p["q"] for p in CHOSEN if p["id"] == "palindrome_words__3")
    taskprompt = (q + "\\n\\nOutput ONLY the text itself — no preamble, no quotes, no explanation."
                  ).replace("\\n", " &#8626; ")
    global EXPLORER
    EXPLORER = EXPLORER.replace("__TASKPROMPT__", taskprompt)
    sg = json.load(open(EXP / "stuck_gain.json"))
    rows = "".join(
        f"<tr><td class='mono'>s{r['seed']}</td><td>{r['mode']}</td><td>{r['seas_peak']:.2f}</td>"
        f"<td>{'—' if r['flip'] is None else r['flip']}</td><td>{'—' if r['cross'] is None else r['cross']}</td>"
        f"<td class='mono'>{r['final']}</td></tr>" for r in sg)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="cache-control" content="no-cache, no-store, must-revalidate">
<meta http-equiv="pragma" content="no-cache">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" type="image/png" href="favicon.png">
<title>DG seasonal&harr;idiom bistability: S^t as belief carrier</title>
<style>{STYLE}</style>
</head>
<body>
<button id="themeToggle" onclick="tgTheme()">&#9680; theme</button>
<div class="wrap">
<h1>The seasonal&harr;idiom bistability: a case study of S<sup>t</sup> as belief carrier</h1>
<span class="pill">google/diffusiongemma-26B-A4B-it</span>
<span class="pill">hot regime: T=64, C=128, t 1.3&rarr;0.8, entropy_bound 0.3, top_k 10</span>
<span class="pill">8 seeds</span>
<span class="pill">part of <a href="index.html">DG planning</a></span>

<h2>Setup: one prompt, two attractors</h2>
<div class="card">
<b>Task</b> (<span class="mono">palindrome_words__3</span>):
<div class="gen">Write a phrase of at least 7 words whose sequence of words reads the same forwards and backwards (a word-level palindrome).

Output ONLY the text itself — no preamble, no quotes, no explanation.</div>
The model has two attractors: the <b>idiom</b> <span class="mono">"All for one and one for all."</span> &mdash; a
valid word-level palindrome &mdash; and the <b>seasonal trap</b> <span class="mono">"All leaves fall when leaves
fall all."</span> &mdash; locally typical, palindrome-<em>flavored</em>, but <b>constraint-violating</b> (its mirror
pairs don't match). Both tokenize to 8 tokens and differ at exactly <b>5 contested slots</b>
(<span class="mono">for/one/and/one/for</span> vs <span class="mono">leaves/fall/when/leaves/fall</span>) &mdash; near-disjoint
basin vocabularies, which is what makes the mode competition legible in the S<sup>t</sup> self-conditioning
channel (per-position soft logits passed between denoising steps). Under the sampler's ground truth
(EntropyBoundSampler renoises every unaccepted position to uniform noise each step), S<sup>t</sup> is the
<em>only</em> cross-step carrier at the contested slots until they are accepted.</div>

<h2>Eight seeds, three regimes</h2>
<table class="rz"><tr><th>seed</th><th>final mode</th><th>peak seasonal S-mass</th><th>draft flip step</th><th>S crossover step</th><th>final phrase</th></tr>
{rows}</table>
<div class="card"><b>Shallow contests</b> (peak seasonal S &le; 0.45: s1, s2, s5, s6) resolve to the idiom before deep
commitment forms. <b>Deep-basin escapes</b> (peak 0.76&ndash;0.91: s3, s4, s7) sit in the seasonal basin for
several steps, then cross over. <b>One trapped run</b> (s0, peak 1.00) never escapes and emits the
constraint-violating seasonal phrase. The S crossover precedes the draft flip by &le;1 step in every
escaping run.</div>

<h2>Watching the competition in S</h2>
<p>Four probes on the S channel's basin structure (details in the <a href="index.html">main report</a>):</p>
<img class="fig" src="figs/swarm_census.png">
<div class="card">
<b>(1) Rank-layer census &mdash; the winner starts as the challenger.</b> In idiom-ending runs the seasonal basin
owns rank-1 at up to 40% of slots through t&le;4 while the idiom sits at &asymp;0; the idiom rises through the
rank layers and takes rank-1 by t6&ndash;8, the seasonal basin descending rather than vanishing.
<b>(2) Co-fluctuation:</b> cross-position correlation of per-step basin-margin changes = +0.23 vs +0.01
shuffled &mdash; a shared "which mode" variable moves all slots together. <b>(3) Ghost:</b> the losing basin
persists at rank 1&ndash;2 in S for 6+ steps after full canvas commitment, every run. <b>(4) Beam-switch
(causal):</b> promoting the rank-2 layer at every slot each step flips 3/3 idiom seeds into the seasonal
basin; promoting layer 5 yields garbage &mdash; the switchable coherent structure is top-2 shallow.</div>
<div class="card" style="border-left:4px solid var(--gold)">
<b>But the marginals are biased shadows, not posterior weights.</b> State-branching (re-inject the captured
state at step k, continue with 8 fresh seeds): branches are basin-pure 80/80; the seasonal-bound state is
genuinely bimodal at k=1&ndash;2 (&asymp;50/50), but <b>the idiom-bound state is 100% idiom-committed already at
k=1 &mdash; even while its S rank-1 layers are dominated by seasonal words</b>. The early "seasonal dominance"
in the visible ranks carries essentially zero outcome probability: an early-committing annealed sampler of a
multimodal P(answer|prompt), factorized over positions, whose commitment lives deeper in the state than any
rank snapshot.</div>

<h2>The energy landscape lives in the joint (canvas, S)</h2>
<p>The natural picture &mdash; a pseudo-NLL surface over (&alpha; = seasonal&harr;idiom mix, &rho; = corruption) with
the trajectories rolling on it &mdash; fails at the first sanity check: one-step readout of a <em>bare</em> canvas
(no S supplied) scores token-scrambled anchors BETTER than the coherent idiom (5.3&ndash;7.1 vs 15.5 mean NLL).
Expected, in hindsight: the model never operates on a coherent canvas without S. Supplying an S-sheet
(sparse reconstruction from a run's recorded top-32) restores coherence &mdash; and determines the well:</p>
<img class="fig" src="figs/swarm_landscape2.png">
<div class="card" style="border-left:4px solid var(--gold)">
Under the trapped run's S (s0@t8): seasonal anchor E=0.01, idiom 6.0 &mdash; a single valley tilted seasonal-down.
Under the idiom run's S (s1@t8): idiom E=0.04, seasonal 6.9 &mdash; the SAME canvas grid tilted the opposite way.
<b>The soft state does not roll on a fixed landscape; it is the tilt of the landscape.</b> The trap-escape is not
a thermal barrier crossing: it is the S-state swinging the surface's gradient from seasonal-down to
idiom-down, after which the canvas slides downhill (s3's canvas walks &alpha; = 0 &rarr; &frac13; &rarr; &frac23; &rarr; 1
across t6&ndash;9 while s0 stays pinned at &alpha;=0).</div>

<h2>The stuck&rarr;gain signature: sub-threshold revision</h2>
<p>Ablation-necessity is not architectural uniqueness &mdash; a masked-diffusion LM without self-conditioning
keeps accepted tokens as its cross-step carrier and samples text fine. What S<sup>t</sup> uniquely changes is the
<em>dynamics</em>: S passes <b>beliefs</b>, a canvas passes <b>decisions</b>. A decision-carrier quantizes to tokens
every step, so revision means discrete flips; a belief-carrier can accumulate evidence against the current
mode <em>below the argmax threshold</em> (the auxiliary-variable trick of MCMC: expressiveness unchanged, mixing
changed). Falsifiable signature: stuck runs should show the eventual winner's S-mass ramping gradually while
the hard channel still displays the loser &mdash; and no ramp in runs that never escape.</p>
<img class="fig" src="figs/stuck_gain.png">
<div class="card" style="border-left:4px solid var(--gold)">
<b>Present, with a clean three-regime split.</b> Deep-basin escapes ramp the idiom's S-mass gradually over 4&ndash;5
steps while drafts stay majority-seasonal (s3: 0.03 &rarr; 0.13 &rarr; 0.19 &rarr; 0.32 &rarr; 0.73 across t5&ndash;t9); the
draft flip is the lagging argmax of the ramp. The trapped run purges the idiom to exactly 0.00 (out of S's
top-32) as seasonal saturates &mdash; no gain, no escape. The razor-thin difference: through its deepest seasonal
phase s3 kept a 0.02&ndash;0.03 idiom <b>ember</b> alive in S; s0 purged it by t5&ndash;6. Since the contested slots are
unaccepted (= uniform noise in the canvas) during the ramp, this 4&ndash;5-step evidence accumulation has no
carrier other than S.</div>
{ember_section()}
{EXPLORER}
<h2>Related material in the main report</h2>
<div class="card">
<a href="index.html">DG planning report</a>: the tail-as-timing-signal mechanism, the negotiation slot
taxonomy (reporter / redundant / attractor / structural slots &mdash; the palindrome word slot moves as a
globally-coupled attractor or not at all), the S-echo null (the idiom never forms at all under echo/flat S
&mdash; S is the assembly medium, not just the escape carrier), the sampler-renoise architectural ground truth,
and the activation-oracle reads of this exact escape (the AO says idiom at +4.5 from step 0, while the
full output distribution and every logit-lens layer favor seasonal for 8 steps &mdash; the plan lives in
non-vocabulary subspaces).</div>
<p class="dim">Scripts: <span class="mono">planning/{{negotiation_capture,swarm_census,swarm_switch,swarm_branch,
swarm_landscape2,stuck_gain,ember_kill,ember_lens_capture,ember_jspace,ember_clamp,ember_violation,ember_wrongness,ember_suscept,ember_suscept2,ember_spromote,ember_onestep,ember_spromote2{{,_powered}},ember_spromote3,ember_powered_fig{{,2}},ember_repair{{,_fig}}}}.py</span> &middot; rounds 2&ndash;3 (2026-08-03):
<span class="mono">planning/{{ember_kill2{{,_read}},ember_kill_recheck3,ember_dur,ember_span,ember_rescue,ember_related{{,_read}},ember_autonomous{{,2,3,_fig}},ember_handicap,ember_pinrescue}}.py</span> &middot; data <span class="mono">exp/dg_planning/</span> &middot; built by
<span class="mono">planning/build_seasonal.py</span></p>
</div>
<script>
function tgTheme(){{
  const r = document.documentElement;
  const cur = r.dataset.theme || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark':'light');
  r.dataset.theme = cur === 'dark' ? 'light' : 'dark';
}}
</script>
</body>
</html>"""
    OUT.write_text(html)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
