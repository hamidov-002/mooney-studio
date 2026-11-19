"""
The pictures of the Runs page: the stage cards of a new run, the board of groups with their queued runs, and the
history of saved runs. Everything here only turns plain data into HTML; nothing is read from disk and nothing is run.

The three pictures are shown with gr.HTML as custom components: JS (below) turns a click on anything that carries
a data-act attribute into a Gradio click event with {act, id, group, stage}, which app.py answers.
"""
import html as _html

# A click anywhere inside the component: the nearest element with data-act says what was meant.
JS = """
element.addEventListener('click', (e) => {
  const b = e.target.closest('[data-act]');
  if (!b || !element.contains(b)) return;
  e.preventDefault(); e.stopPropagation();
  trigger('click', {act: b.dataset.act, id: b.dataset.id || '', group: b.dataset.group || '', stage: b.dataset.stage || ''});
});
"""

WAITING, RUNNING, DONE, FAILED, STOPPED, SKIPPED = "waiting", "running", "done", "failed", "stopped", "skipped"
STAGES = ["source", "mooneys", "systems", "export", "save"]
STAGE_TITLES = {"source": "Source", "mooneys": "Mooneys", "systems": "Vision systems", "export": "Export", "save": "Save"}


def esc(x):
    return _html.escape(str(x if x is not None else ""))


def act(label, what, cls="rb-a", **data):
    attrs = "".join(f' data-{k}="{esc(v)}"' for k, v in data.items())
    return f'<button type="button" class="{cls}" data-act="{what}"{attrs}>{esc(label)}</button>'


# ---------------------------------------------------------------------------- 1. the stage cards of a new run
def cards(selected, on, summaries):
    """selected: stage key that is open; on: {stage: bool}; summaries: {stage: html of its one-line summary}."""
    out = []
    for k, key in enumerate(STAGES):
        is_on = key == "source" or bool(on.get(key))
        cls = "rb-stage" + ("" if is_on else " off") + (" sel" if key == selected else "")
        toggle = "" if key == "source" else (
            f'<span class="rb-tog" data-act="toggle" data-stage="{key}" role="switch" aria-checked="{str(is_on).lower()}" '
            f'title="{"Switch this stage off" if is_on else "Switch this stage on"}"></span>')
        out.append(f'<div class="{cls}" data-act="select" data-stage="{key}" title="Show the settings of this stage">'
                   f'{toggle}<div class="rb-n">Stage {k + 1}</div><div class="rb-t">{STAGE_TITLES[key]}</div>'
                   f'<div class="rb-d">{summaries.get(key, "")}</div></div>')
        if k < len(STAGES) - 1:
            out.append('<div class="rb-arrow" aria-hidden="true">→</div>')
    return '<div class="rb-pipe">' + "".join(out) + "</div>"


def chips(labels, on=True):
    return "".join(f'<span class="rb-chip{" on" if on else ""}">{esc(l)}</span>' for l in labels)


# ---------------------------------------------------------------------------- 2. groups and queue
DOT = {DONE: "✓", RUNNING: "●", FAILED: "!", STOPPED: "■"}


def stage_chips(stages, state):
    out = []
    for s in stages:
        st, detail = s.get("state", WAITING), s.get("detail", "")
        if st == DONE:
            out.append(f'<span class="rb-st ok">✓ {esc(s["title"])}</span>')
        elif st == RUNNING and state == RUNNING:
            out.append(f'<span class="rb-st now">● {esc(s["title"])}{" · " + esc(detail) if detail else ""}</span>')
        elif st in (FAILED, STOPPED):
            out.append(f'<span class="rb-st bad">{"!" if st == FAILED else "■"} {esc(s["title"])}</span>')
        elif st == SKIPPED:
            out.append(f'<span class="rb-st skip">– {esc(s["title"])}</span>')
        else:
            out.append(f'<span class="rb-st">{esc(s["title"])}</span>')
    return "".join(out)


def eta_text(fraction, minutes, a_stage_done):
    """A rough 'about N min left', or '' while there is too little to go on. From the share done and the time it took."""
    if not (a_stage_done or fraction >= 0.10) or fraction < 0.03 or fraction >= 0.995 or minutes < 0.05:
        return ""
    left = minutes * (1 - fraction) / fraction
    if left > 600:
        return ""
    return "under a minute left" if left < 1 else f"about {round(left)} min left"


def progress_cell(it):
    """The bar of a running run: share of the whole run done, in per cent, the time so far and a rough time left."""
    frac = min(max(float(it.get("fraction") or 0), 0.0), 1.0)
    try:
        mins = float(it.get("minutes") or 0)
    except ValueError:
        mins = 0.0
    eta = eta_text(frac, mins, any(s.get("state") == DONE for s in it.get("stages") or []))
    return (f'<div class="rb-prog" title="share of the whole run that is done"><span class="rb-pbar"><i style="width:{100 * frac:.1f}%"></i></span>'
            f'<b>{100 * frac:.0f} %</b></div><div class="rb-eta">{mins:.1f} min{" · " + eta if eta else ""}</div>')


def _log_block(lines, tail=None):
    lines = [l for l in (lines or [])]
    if tail:
        lines = lines[-tail:]
    return f'<pre class="rb-log{"" if tail else " full"}">{esc(chr(10).join(lines)) or "(no log lines yet)"}</pre>'


def board(items, open_logs=(), queue_running=False, busy=None, facts=None):
    """
    items: queue items in order ({id, name, group, state, stages, fraction, minutes, message, log, config, folder?}).
    facts: {id: (settings line, source line, old_format)} worked out by the app from each run's description.
    """
    facts = facts or {}
    if not items:
        return ('<div class="rb-empty"><b>No runs yet.</b> Build a run above and press <i>Add to queue</i>. '
                "Runs of one group go one after another; each is saved under its own name.</div>")
    order = []
    for it in items:
        if it["group"] not in order:
            order.append(it["group"])
    out = []
    for g in order:
        rows = [it for it in items if it["group"] == g]
        n = {s: sum(1 for it in rows if it["state"] == s) for s in (WAITING, RUNNING, DONE, FAILED, STOPPED)}
        parts = [f"{len(rows)} run{'s' if len(rows) != 1 else ''}"] + [f"{n[s]} {s}" for s in (DONE, RUNNING, WAITING, FAILED, STOPPED) if n[s]]
        frac = sum(1.0 if it["state"] == DONE else float(it.get("fraction") or 0) if it["state"] == RUNNING else 0.0 for it in rows) / len(rows)
        if n[RUNNING]:
            button = act("Stop", "stop-now", "rb-btn danger", group=g)
        elif n[WAITING] and not queue_running and not busy:
            button = act("Run these", "run-group", "rb-btn primary", group=g)
        elif n[WAITING]:
            button = '<span class="rb-wait">waits for the job that is working</span>'
        else:
            button = ""
        out.append(f'<div class="rb-group"><div class="rb-gh"><div class="rb-gname">{esc(g)}</div>'
                   f'<div class="rb-gmeta">{" · ".join(parts)}</div>'
                   f'<div class="rb-bar" title="{frac:.0%} of this group is done"><i style="width:{100 * frac:.0f}%"></i></div>'
                   f'<div class="rb-gpct">{100 * frac:.0f} %</div>{button}</div>')
        pos = 0
        for it in rows:
            st = it["state"]
            settings, source, old = facts.get(it["id"], ("", "", False))
            if st == WAITING:
                pos += 1
            dot = DOT.get(st, str(pos))
            if st == WAITING:
                time_ = "waiting"
                acts = ([] if queue_running or busy else [act("run", "run-one", id=it["id"])]) + [
                    act("↑", "up", id=it["id"]), act("↓", "down", id=it["id"])] + (
                    [] if old else [act("edit", "edit", id=it["id"]), act("copy", "dup", id=it["id"])]) + [act("remove", "remove", id=it["id"])]
            elif st == RUNNING:
                time_ = progress_cell(it)
                acts = [act("stop", "stop-now", id=it["id"])]
            else:
                time_ = (f'{esc(it["minutes"])} min' if it.get("minutes") else st)
                acts = ([act("results", "results", id=it["id"])] if st == DONE and it.get("folder") else []) + [
                    act("log", "log", id=it["id"])] + ([] if old else [act("copy", "dup", id=it["id"])]) + [
                    act("again", "again", id=it["id"]), act("remove", "remove", id=it["id"])]
            stages = stage_chips(it.get("stages") or [], st) or '<span class="rb-st">old format</span>'
            msg = ""
            if st in (FAILED, STOPPED) and it.get("message"):
                msg = f'<div class="rb-msg bad">{esc(it["message"])}</div>'
            elif st == DONE and it.get("message"):
                msg = f'<div class="rb-msg" title="{esc(it["message"])}">{esc(it["message"].split(": ")[0].split(";")[0])}</div>'
            out.append(f'<div class="rb-job {st}"><div class="rb-dot">{dot}</div>'
                       f'<div class="rb-jn"><b>{esc(it["name"])}</b><span>{esc(settings)}</span></div>'
                       f'<div class="rb-src">{esc(source)}</div><div class="rb-stages">{stages}{msg}</div>'
                       f'<div class="rb-time">{time_}</div><div class="rb-acts">{"".join(acts)}</div></div>')
            if st == RUNNING:
                out.append(_log_block(it.get("log"), tail=12))
            elif it["id"] in open_logs:
                out.append(_log_block(it.get("log")))
        out.append("</div>")
    return '<div class="rb">' + "".join(out) + "</div>"


# ---------------------------------------------------------------------------- 3. history of saved runs
def history(runs, picked=(), armed=None, opened=None):
    """
    runs: saved runs, newest first: {folder, name, group, saved, settings, source, kpis: [(label, value)], note}.
    picked: folders chosen for the comparison; armed: folder whose deletion waits for a yes; opened: folder shown below.
    """
    if not runs:
        return ('<div class="rb-empty"><b>No saved runs yet.</b> A run with the stage <i>Save</i> switched on appears here '
                "when it has finished.</div>")
    order = []
    for r in runs:
        if r["group"] not in order:
            order.append(r["group"])
    out = []
    for g in order:
        rows = [r for r in runs if r["group"] == g]
        days = sorted({r["saved"][:10] for r in rows if r.get("saved")})
        when = days[0] if len(days) == 1 else f"{days[0]} to {days[-1]}" if days else ""
        out.append(f'<div class="rb-group"><div class="rb-gh"><div class="rb-gname">{esc(g)}</div>'
                   f'<div class="rb-gmeta">{len(rows)} saved run{"s" if len(rows) != 1 else ""}{" · " + when if when else ""}</div></div>')
        for r in rows:
            f = r["folder"]
            kp = "".join(f'<span>{esc(l)} <b>{esc(v)}</b></span>' for l, v in r.get("kpis") or []) or '<span>no result tables</span>'
            if armed == f:
                acts = ('<span class="rb-ask">Move this run to the trash?</span>' + act("yes, remove", "delete-yes", "rb-a danger", id=f)
                        + act("no", "delete-no", id=f))
            else:
                acts = (act("close" if opened == f else "open", "open", id=f)
                        + act("✓ in comparison" if f in picked else "compare", "pick", "rb-a" + (" on" if f in picked else ""), id=f)
                        + act("publish", "publish", id=f) + act("delete", "delete", id=f))
            note = f' · {esc(r["note"])}' if r.get("note") else ""
            out.append(f'<div class="rb-job done hist{" open" if opened == f else ""}"><div class="rb-dot">✓</div>'
                       f'<div class="rb-jn"><b>{esc(r["name"])}</b><span>{esc(r.get("settings", ""))}</span></div>'
                       f'<div class="rb-src">{esc(r.get("source", ""))}</div><div class="rb-kpi">{kp}</div>'
                       f'<div class="rb-time">{esc(r.get("saved", ""))}{note}</div><div class="rb-acts">{acts}</div></div>')
        out.append("</div>")
    return '<div class="rb">' + "".join(out) + "</div>"


CSS = """
.rb-pipe { display: flex; align-items: stretch; margin: 4px 0 2px; }
.rb-stage { flex: 1 1 0; min-width: 0; position: relative; background: var(--card); border: 1px solid var(--line); border-radius: 8px;
  padding: 11px 13px 12px; min-height: 104px; cursor: pointer; transition: border-color .12s, box-shadow .12s; }
.rb-stage:hover { border-color: var(--accent-soft); }
.rb-stage.sel { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-wash); }
.rb-stage.off { background: var(--paper-2); border-style: dashed; }
.rb-stage.off .rb-t, .rb-stage.off .rb-d { color: var(--muted-2); }
.rb-n { font-size: 10.5px; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
.rb-t { font-family: var(--serif); font-size: 16px; font-weight: 600; color: var(--ink); margin: 1px 0 6px; }
.rb-d { font-size: 12.5px; line-height: 1.45; color: var(--ink-2); }
.rb-d b { font-weight: 600; }
.rb-arrow { flex: 0 0 26px; display: flex; align-items: center; justify-content: center; color: var(--muted-2); font-size: 17px; }
.rb-tog { position: absolute; right: 11px; top: 11px; width: 32px; height: 18px; border-radius: 10px; background: var(--accent); cursor: pointer;
  transition: background .12s; }
.rb-tog::after { content: ""; position: absolute; right: 2px; top: 2px; width: 14px; height: 14px; border-radius: 50%; background: #fff;
  box-shadow: 0 1px 2px rgba(0,0,0,.2); }
.rb-stage.off .rb-tog { background: #c3c8d0; }
.rb-stage.off .rb-tog::after { right: auto; left: 2px; }
.rb-chip { display: inline-block; border: 1px solid var(--line); border-radius: 4px; padding: 0 7px; margin: 2px 4px 0 0; font-size: 12px;
  background: var(--card); color: var(--muted); }
.rb-chip.on { background: var(--run-bg); border-color: var(--run-line); color: var(--accent); }
@media (max-width: 1000px) { .rb-pipe { flex-wrap: wrap; gap: 8px; } .rb-arrow { display: none; } .rb-stage { flex: 1 1 30%; } }

.rb-empty { border: 1px dashed var(--line); border-radius: 8px; padding: 18px 20px; color: var(--muted); font-size: 13.5px; background: var(--paper-2); }
.rb-empty b { color: var(--ink); }
.rb-group { background: var(--card); border: 1px solid var(--line); border-radius: 8px; margin-bottom: 14px; overflow: hidden; }
.rb-gh { display: flex; align-items: center; gap: 14px; padding: 10px 16px; border-bottom: 1px solid var(--line); background: var(--paper-2); min-height: 48px; }
.rb-gname { font-family: var(--serif); font-size: 17px; font-weight: 600; color: var(--ink); }
.rb-gmeta { color: var(--muted); font-size: 12.5px; flex: 1 1 auto; }
.rb-bar { flex: 0 0 170px; height: 5px; background: var(--line); border-radius: 3px; overflow: hidden; }
.rb-bar i { display: block; height: 100%; background: var(--accent); border-radius: 3px; transition: width .4s; }
.rb-wait { font-size: 12px; color: var(--muted); }
.rb-gpct { flex: 0 0 42px; font-size: 12.5px; color: var(--muted); font-variant-numeric: tabular-nums; text-align: right; }
.rb-prog { display: flex; align-items: center; gap: 8px; }
.rb-pbar { flex: 1 1 auto; height: 6px; background: var(--line); border-radius: 3px; overflow: hidden; min-width: 60px; }
.rb-pbar i { display: block; height: 100%; background: var(--accent); border-radius: 3px; transition: width .6s; }
.rb-prog b { font-size: 12.5px; font-weight: 600; color: var(--accent); font-variant-numeric: tabular-nums; min-width: 36px; text-align: right; }
.rb-eta { font-size: 12px; color: var(--muted); margin-top: 3px; font-variant-numeric: tabular-nums; }
.rb-btn { font: inherit; font-size: 13px; font-weight: 600; border-radius: 6px; padding: 5px 14px; cursor: pointer; border: 1px solid var(--line);
  background: var(--card); color: var(--ink); }
.rb-btn.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
.rb-btn.primary:hover { background: var(--accent-2); }
.rb-btn.danger { color: var(--bad); border-color: var(--bad-line); background: var(--card); }
.rb-btn.danger:hover { background: var(--bad-bg); }
.rb-job { display: grid; grid-template-columns: 26px minmax(150px, 1.25fr) minmax(120px, .9fr) minmax(220px, 2.4fr) minmax(150px, 1fr) minmax(150px, auto);
  gap: 12px; align-items: center; padding: 10px 16px; border-bottom: 1px solid var(--line-2); font-size: 13px; }
.rb-group > .rb-job:last-child, .rb-group > .rb-log:last-child { border-bottom: none; }
.rb-dot { width: 22px; height: 22px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 11.5px;
  font-weight: 700; background: var(--wait-bg); color: var(--muted); }
.rb-job.done .rb-dot { background: var(--ok-bg); color: var(--ok); }
.rb-job.running .rb-dot { background: var(--run-bg); color: var(--accent); animation: rb-pulse 1.6s ease-in-out infinite; }
.rb-job.failed .rb-dot { background: var(--bad-bg); color: var(--bad); }
.rb-job.stopped .rb-dot { background: var(--part-bg); color: var(--part); }
@keyframes rb-pulse { 50% { opacity: .45; } }
@media (prefers-reduced-motion: reduce) { .rb-job.running .rb-dot { animation: none; } .rb-bar i { transition: none; } }
.rb-jn { min-width: 0; } .rb-jn b { font-weight: 600; color: var(--ink); display: block; overflow-wrap: anywhere; }
.rb-jn span { display: block; color: var(--muted); font-size: 12px; }
.rb-src { color: var(--muted); font-size: 12.5px; }
.rb-stages { min-width: 0; }
.rb-st { display: inline-flex; align-items: center; gap: 4px; border: 1px solid var(--line); border-radius: 12px; padding: 1px 10px; margin: 2px 5px 2px 0;
  font-size: 12px; background: var(--card); color: var(--muted); white-space: nowrap; font-variant-numeric: tabular-nums; }
.rb-st.ok { background: var(--ok-bg); border-color: var(--ok-line); color: var(--ok); }
.rb-st.now { background: var(--run-bg); border-color: var(--run-line); color: var(--accent); font-weight: 600; }
.rb-st.bad { background: var(--bad-bg); border-color: var(--bad-line); color: var(--bad); }
.rb-st.skip { border-style: dashed; }
.rb-msg { font-size: 12px; color: var(--muted); margin-top: 3px; overflow-wrap: anywhere; } .rb-msg.bad { color: var(--bad); }
.rb-time { color: var(--muted); font-size: 12.5px; font-variant-numeric: tabular-nums; }
.rb-acts { text-align: right; white-space: nowrap; }
.rb-a { font: inherit; font-size: 12.5px; color: var(--accent); background: none; border: none; padding: 2px 5px; margin-left: 3px; cursor: pointer;
  border-radius: 4px; }
.rb-a:hover { background: var(--accent-wash); }
.rb-a.danger { color: var(--bad); } .rb-a.danger:hover { background: var(--bad-bg); }
.rb-a.on { font-weight: 600; background: var(--run-bg); }
.rb-ask { font-size: 12.5px; color: var(--bad); margin-right: 4px; }
.rb-log { margin: 0; background: var(--log-bg); color: var(--log-ink); font: 12px/1.55 var(--mono); padding: 10px 16px; white-space: pre; overflow-x: auto;
  border-bottom: 1px solid var(--line-2); }
.rb-log.full { max-height: 360px; overflow-y: auto; }
.rb-job.hist { grid-template-columns: 26px minmax(150px, 1.2fr) minmax(110px, .8fr) minmax(240px, 2.6fr) minmax(110px, .9fr) minmax(240px, auto); }
.rb-job.hist.open { background: var(--accent-wash); }
.rb-kpi { display: flex; flex-wrap: wrap; gap: 2px 18px; font-size: 12.5px; color: var(--muted); }
.rb-kpi b { color: var(--ink); font-weight: 600; font-variant-numeric: tabular-nums; }
@media (max-width: 1100px) {
  .rb-job, .rb-job.hist { grid-template-columns: 26px 1fr auto; }
  .rb-src, .rb-stages, .rb-kpi { grid-column: 2 / 4; } .rb-time { grid-column: 2; } .rb-acts { grid-column: 3; }
}
"""
