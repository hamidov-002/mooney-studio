"""
A queue of experiments that run one after another without anyone watching.

The queue is a file, <project>/queue.json, so it survives a restart of the app:
    {"items": [{"id", "name", "group", "config", "state", "added", "started", "finished", "minutes", "message", "log",
                "stages": [{"title", "state", "detail"}], "fraction"}]}
state is one of: waiting, running, done, failed, stopped. Runs belong to a group; a group can be run on its own.
While a run works, its log, stages and fraction are kept in memory and written to the file every few seconds.
A run that raises StopNow (the user pressed Stop now) is marked stopped and the queue halts; the items after it stay waiting.

This module only keeps the list and runs it. What one experiment does is given by the caller as
run_one(config, log) -> message, so the queue runs exactly the same code as the Run button.
"""
import json
import os
import threading
import time
import traceback
import uuid

WAITING, RUNNING, DONE, FAILED, STOPPED = "waiting", "running", "done", "failed", "stopped"


class StopNow(BaseException):
    """Raised inside a run when the user stops it. Not an Exception, so no 'except Exception' in the work swallows it."""


_lock = threading.RLock()
_worker = {"thread": None, "stop": False}
_live = {"id": None, "log": [], "stages": [], "fraction": 0.0, "t0": 0.0, "written": 0.0}     # the run that is working now
DEFAULT_GROUP = "Queue"


def _path(project):
    return os.path.join(project, "queue.json")


def load(project):
    try:
        d = json.load(open(_path(project), encoding="utf-8"))
        d.setdefault("items", [])
    except Exception:
        d = {"items": []}
    for it in d["items"]:
        it.setdefault("group", DEFAULT_GROUP); it.setdefault("stages", []); it.setdefault("fraction", 0)
        it.setdefault("log", []); it.setdefault("minutes", ""); it.setdefault("message", "")
    # an item left 'running' by a closed app did not finish
    if not is_running():
        for it in d["items"]:
            if it.get("state") == RUNNING:
                it["state"], it["message"] = STOPPED, "the app was closed while it ran"
    return d


def snapshot(project):
    """The items as the app should show them: the file, with the run that is working now taken from memory."""
    items = load(project)["items"]
    if is_running() and _live["id"]:
        for it in items:
            if it["id"] == _live["id"]:
                it["log"], it["stages"], it["fraction"] = list(_live["log"][-300:]), [dict(s) for s in _live["stages"]], _live["fraction"]
                it["minutes"] = f"{(time.time() - _live['t0']) / 60:.1f}"
    return items


def groups(project):
    """Group names in the order they first appear in the queue."""
    out = []
    for it in load(project)["items"]:
        if it["group"] not in out:
            out.append(it["group"])
    return out


def _save(project, d):
    os.makedirs(project, exist_ok=True)
    tmp = _path(project) + ".tmp"
    json.dump(d, open(tmp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    os.replace(tmp, _path(project))


def add(project, name, config, group=None, stages=None):
    """Put one experiment at the end of the queue (after the last item of its group, if the group exists). Returns its id."""
    with _lock:
        d = load(project)
        group = str(group or DEFAULT_GROUP).strip() or DEFAULT_GROUP
        it = {"id": uuid.uuid4().hex[:8], "name": str(name).strip(), "group": group, "config": config, "state": WAITING,
              "added": time.strftime("%Y-%m-%d %H:%M"), "started": "", "finished": "", "minutes": "", "message": "", "log": [],
              "stages": [{"title": t, "state": WAITING, "detail": ""} for t in (stages or [])], "fraction": 0}
        last = max((k for k, x in enumerate(d["items"]) if x["group"] == group), default=None)
        d["items"].insert(len(d["items"]) if last is None else last + 1, it)
        _save(project, d)
        return it["id"]


def get(project, item_id):
    return next((it for it in snapshot(project) if it["id"] == item_id), None)


def replace_config(project, item_id, name, config, group=None, stages=None):
    """Change a waiting item in place (its position is kept). False if it is not waiting any more."""
    with _lock:
        d = load(project)
        for it in d["items"]:
            if it["id"] == item_id and it["state"] == WAITING:
                it.update(name=str(name).strip(), config=config, group=str(group or it["group"]).strip() or DEFAULT_GROUP,
                          stages=[{"title": t, "state": WAITING, "detail": ""} for t in (stages or [])])
                _save(project, d)
                return True
        return False


def requeue(project, item_id):
    """Set a finished, failed or stopped item back to waiting, so it can run again."""
    with _lock:
        d = load(project)
        for it in d["items"]:
            if it["id"] == item_id and it["state"] in (DONE, FAILED, STOPPED):
                it.update(state=WAITING, started="", finished="", minutes="", message="", log=[], fraction=0,
                          stages=[{"title": s["title"], "state": WAITING, "detail": ""} for s in it.get("stages", [])])
                _save(project, d)
                return True
        return False


def remove(project, item_id):
    """Take a waiting or finished item off the list. A running one cannot be removed."""
    with _lock:
        d = load(project)
        keep = [it for it in d["items"] if it["id"] != item_id or it["state"] == RUNNING]
        changed = len(keep) != len(d["items"])
        d["items"] = keep
        _save(project, d)
        return changed


def move(project, item_id, step):
    """Move a waiting item up (step -1) or down (step +1) among the waiting items of its group."""
    with _lock:
        d = load(project)
        items = d["items"]
        i = next((k for k, it in enumerate(items) if it["id"] == item_id), None)
        if i is None or items[i]["state"] != WAITING:
            return False
        j = i + step
        while 0 <= j < len(items) and (items[j]["state"] != WAITING or items[j]["group"] != items[i]["group"]):
            j += step
        if not 0 <= j < len(items):
            return False
        items[i], items[j] = items[j], items[i]
        _save(project, d)
        return True


def clear_finished(project):
    with _lock:
        d = load(project)
        d["items"] = [it for it in d["items"] if it["state"] in (WAITING, RUNNING)]
        _save(project, d)


def is_running():
    t = _worker["thread"]
    return bool(t and t.is_alive())


def stop_after_current():
    """The experiment running now finishes; the next ones stay waiting."""
    _worker["stop"] = True


def _update(project, item_id, **fields):
    with _lock:
        d = load(project)
        for it in d["items"]:
            if it["id"] == item_id:
                it.update(fields)
        _save(project, d)


def _flush(project, force=False):
    """Write the working run's log, stages and fraction to the file, at most every two seconds."""
    if _live["id"] and (force or time.time() - _live["written"] > 2):
        _live["written"] = time.time()
        _update(project, _live["id"], log=list(_live["log"][-300:]), stages=[dict(s) for s in _live["stages"]],
                fraction=round(float(_live["fraction"]), 3))


def start(project, run_one, group=None, only=None):
    """
    Run waiting items, oldest first, in a background thread: all of them, those of one group, or the single item `only`.
    run_one(config, log, report) does one run and returns the line shown for it; report(stages=None, fraction=None)
    tells the queue where the run is (stages: list of {"title", "state", "detail"}).
    Returns False if the queue already runs.
    """
    if is_running():
        return False
    _worker["stop"] = False

    def wanted(it):
        return it["state"] == WAITING and (only is None or it["id"] == only) and (group is None or it["group"] == group)

    def work():
        while not _worker["stop"]:
            nxt = next((it for it in load(project)["items"] if wanted(it)), None)
            if nxt is None:
                break
            lines = _live["log"] = []
            _live.update(id=nxt["id"], stages=[dict(s) for s in nxt.get("stages", [])], fraction=0.0, t0=time.time(), written=0.0)

            def log(msg=""):
                lines.append(f"{time.strftime('%H:%M:%S')}  {msg}" if str(msg).strip() else "")
                _flush(project)

            def report(stages=None, fraction=None):
                if stages is not None:
                    _live["stages"] = [dict(s) for s in stages]
                if fraction is not None:
                    _live["fraction"] = float(fraction)
                _flush(project)

            t0, extra = time.time(), {}
            _update(project, nxt["id"], state=RUNNING, started=time.strftime("%Y-%m-%d %H:%M"), message="", log=[], fraction=0)
            try:
                msg = run_one(nxt["config"], log, report) or "finished"
                if isinstance(msg, dict):                    # {"message": ..., "folder": ...}: extra facts are kept with the item
                    extra = {k: v for k, v in msg.items() if k != "message"}
                    msg = msg.get("message") or "finished"
                state = DONE
            except StopNow:                                  # stopped by the user: this run is marked, the rest stays waiting
                lines.extend(["", "STOPPED BY YOU. What was written so far stays as it is; the run was not saved."])
                msg, state = "stopped by you; not saved", STOPPED
                _worker["stop"] = True
            except Exception:
                tb = traceback.format_exc().strip().splitlines()
                lines.extend(["", "STOPPED WITH AN ERROR"] + tb[-12:])
                msg, state = tb[-1], FAILED
            for s in _live["stages"]:                        # a stage that was working when the run ended did not finish
                if s.get("state") == RUNNING:
                    s["state"] = STOPPED if state == STOPPED else FAILED if state == FAILED else DONE
            _update(project, nxt["id"], state=state, finished=time.strftime("%Y-%m-%d %H:%M"),
                    minutes=f"{(time.time() - t0) / 60:.1f}", message=msg, log=lines[-300:],
                    stages=[dict(s) for s in _live["stages"]], fraction=1.0 if state == DONE else round(float(_live["fraction"]), 3),
                    **extra)
            _live["id"] = None
            if only is not None:
                break

    t = threading.Thread(target=work, daemon=True)
    _worker["thread"] = t
    t.start()
    return True


def table(project):
    """Rows for the queue table in the app: position, name, state, added, started, minutes, message."""
    rows, k = [], 0
    for it in load(project)["items"]:
        if it["state"] == WAITING:
            k += 1
        rows.append([k if it["state"] == WAITING else "", it["name"], it["state"], it["added"], it["started"],
                     it["minutes"], it["message"]])
    return rows


def current_log(project):
    """The log of the item running now, or of the last one that ran."""
    items = load(project)["items"]
    it = next((x for x in items if x["state"] == RUNNING), None) or next(
        (x for x in reversed(items) if x["state"] in (DONE, FAILED, STOPPED)), None)
    return ("\n".join(it["log"]) if it else ""), (it["name"] if it else "")
