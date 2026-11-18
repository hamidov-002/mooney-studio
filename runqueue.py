"""
A queue of experiments that run one after another without anyone watching.

The queue is a file, <project>/queue.json, so it survives a restart of the app:
    {"items": [{"id", "name", "config", "state", "added", "started", "finished", "minutes", "message", "log"}]}
state is one of: waiting, running, done, failed, stopped.
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
_lock = threading.Lock()
_worker = {"thread": None, "stop": False}


def _path(project):
    return os.path.join(project, "queue.json")


def load(project):
    try:
        d = json.load(open(_path(project), encoding="utf-8"))
        d.setdefault("items", [])
    except Exception:
        d = {"items": []}
    # an item left 'running' by a closed app did not finish
    if not is_running():
        for it in d["items"]:
            if it.get("state") == RUNNING:
                it["state"], it["message"] = STOPPED, "the app was closed while it ran"
    return d


def _save(project, d):
    os.makedirs(project, exist_ok=True)
    tmp = _path(project) + ".tmp"
    json.dump(d, open(tmp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    os.replace(tmp, _path(project))


def add(project, name, config):
    """Put one experiment at the end of the queue. Returns its id."""
    with _lock:
        d = load(project)
        it = {"id": uuid.uuid4().hex[:8], "name": str(name).strip(), "config": config, "state": WAITING,
              "added": time.strftime("%Y-%m-%d %H:%M"), "started": "", "finished": "", "minutes": "", "message": "", "log": []}
        d["items"].append(it)
        _save(project, d)
        return it["id"]


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
    """Move a waiting item up (step -1) or down (step +1) among the waiting items."""
    with _lock:
        d = load(project)
        items = d["items"]
        i = next((k for k, it in enumerate(items) if it["id"] == item_id), None)
        if i is None or items[i]["state"] != WAITING:
            return False
        j = i + step
        while 0 <= j < len(items) and items[j]["state"] != WAITING:
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


def start(project, run_one):
    """Run every waiting item, oldest first, in a background thread. Returns False if the queue already runs."""
    if is_running():
        return False
    _worker["stop"] = False

    def work():
        while not _worker["stop"]:
            d = load(project)
            nxt = next((it for it in d["items"] if it["state"] == WAITING), None)
            if nxt is None:
                break
            lines = []

            def log(msg=""):
                lines.append(f"{time.strftime('%H:%M:%S')}  {msg}" if str(msg).strip() else "")
                if len(lines) % 10 == 0:                     # keep the file current without writing on every line
                    _update(project, nxt["id"], log=lines[-300:])

            t0 = time.time()
            _update(project, nxt["id"], state=RUNNING, started=time.strftime("%Y-%m-%d %H:%M"), message="", log=[])
            try:
                msg = run_one(nxt["config"], log) or "finished"
                state = DONE
            except StopNow:                                  # stopped by the user: this run is marked, the rest stays waiting
                lines.extend(["", "STOPPED BY YOU. What was written so far stays as it is; the run was not saved."])
                msg, state = "stopped by you; not saved", STOPPED
                _worker["stop"] = True
            except Exception:
                tb = traceback.format_exc().strip().splitlines()
                lines.extend(["", "STOPPED WITH AN ERROR"] + tb[-12:])
                msg, state = tb[-1], FAILED
            _update(project, nxt["id"], state=state, finished=time.strftime("%Y-%m-%d %H:%M"),
                    minutes=f"{(time.time() - t0) / 60:.1f}", message=msg, log=lines[-300:])

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
