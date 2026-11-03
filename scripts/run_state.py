"""State of a Kaggle training run, rebuilt from the progress events of kaggle/train_kernel.py.

The monitor feeds it raw log text while the run streams; build_reports.py feeds it
the downloaded events.jsonl once the run is over. Either way the result is written
to dashboard/data/training.js for the dashboard.
"""

from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime, timezone

from common import EVENT_PREFIX, write_dashboard_data

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
CONSOLE_LINES = 250
CONSOLE_LINE_CHARS = 300
WRITE_INTERVAL_S = 2.0
IMMEDIATE_EVENTS = {"hello", "stage", "train_start", "epoch", "final_val", "train_end", "eval", "done", "error"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RunState:
    def __init__(self, run: dict):
        self._lock = threading.RLock()
        self._buffers: dict[str, str] = {}
        self._last_write = 0.0
        self._urgent = False
        self.data = {
            "schema": 1,
            "run": run,
            "kaggle_status": "unknown",
            "failure_message": "",
            "monitor": {"phase": "starting", "events": 0, "updated": None, "last_event_at": None},
            "stage": "queued",
            "stage_message": "Waiting for a Kaggle GPU session",
            "stages": [],
            "started_at": None,
            "finished_at": None,
            "config": None,
            "env": None,
            "dataset": None,
            "train": None,
            "progress": None,
            "epochs": [],
            "final_val": None,
            "train_end": None,
            "eval": {},
            "done": None,
            "error": None,
            "console": [],
        }

    # ------------------------------------------------------------------ events

    def apply_event(self, ev: dict) -> None:
        with self._lock:
            d, kind, t = self.data, ev.get("type"), ev.get("t")
            d["monitor"]["events"] += 1
            d["monitor"]["last_event_at"] = t
            if kind == "hello":
                d["started_at"], d["config"] = t, ev.get("config")
            elif kind == "stage":
                d["stage"], d["stage_message"] = ev["stage"], ev.get("message", "")
                if not any(s["t"] == t and s["stage"] == ev["stage"] for s in d["stages"]):
                    d["stages"].append({"stage": ev["stage"], "t": t, "message": ev.get("message", "")})
            elif kind == "env":
                d["env"] = _without(ev, "type", "t")
            elif kind == "data":
                d["dataset"] = _without(ev, "type", "t")
            elif kind == "train_start":
                d["train"] = _without(ev, "type")
            elif kind == "epoch_start":
                batches = (d["train"] or {}).get("batches_per_epoch")
                d["progress"] = {"phase": "train", "epoch": ev["epoch"], "epochs": ev["epochs"], "batch": 0,
                                 "batches": batches, "t": t}
            elif kind == "batch":
                d["progress"] = {"phase": "train", **_without(ev, "type")}
            elif kind == "validating":
                d["progress"] = {**(d["progress"] or {}), "phase": "validate", "t": t}
            elif kind == "epoch":
                epochs = [e for e in d["epochs"] if e["epoch"] != ev["epoch"]]
                d["epochs"] = sorted(epochs + [_without(ev, "type")], key=lambda e: e["epoch"])
                d["progress"] = {**(d["progress"] or {}), "phase": "epoch_done", "epoch": ev["epoch"], "t": t}
            elif kind == "final_val":
                d["final_val"] = _without(ev, "type")
            elif kind == "train_end":
                d["train_end"] = _without(ev, "type")
            elif kind == "eval":
                d["eval"][ev["split"]] = _without(ev, "type", "split")
            elif kind == "done":
                d["done"], d["finished_at"], d["stage"] = _without(ev, "type"), t, "done"
                d["stage_message"] = "Run finished"
            elif kind == "error":
                d["error"], d["finished_at"], d["stage"] = _without(ev, "type"), t, "error"
                d["stage_message"] = ev.get("message", "Run failed")
            if kind in IMMEDIATE_EVENTS:
                self._urgent = True

    def feed(self, stream: str, text: str) -> None:
        """Consume a chunk of raw log text (it may end mid-line)."""
        with self._lock:
            *lines, rest = (self._buffers.get(stream, "") + text).split("\n")
            self._buffers[stream] = rest
            for line in lines:
                self._line(line)

    def flush(self) -> None:
        with self._lock:
            for stream, rest in self._buffers.items():
                if rest:
                    self._line(rest)
            self._buffers.clear()

    def _line(self, raw: str) -> None:
        # Progress bars redraw with "\r"; only the last redraw of a line is worth showing.
        segments = [ANSI_RE.sub("", s) for s in raw.split("\r")]
        console = None
        for seg in segments:
            i = seg.find(EVENT_PREFIX)
            if i < 0:
                if seg.strip():
                    console = seg.rstrip()
                continue
            try:
                self.apply_event(json.loads(seg[i + len(EVENT_PREFIX):]))
            except json.JSONDecodeError:
                pass
            if seg[:i].strip():
                console = seg[:i].rstrip()
        if console:
            lines = self.data["console"]
            lines.append(console[:CONSOLE_LINE_CHARS])
            del lines[:-CONSOLE_LINES]

    # ------------------------------------------------------------------ monitor bookkeeping

    def set_status(self, status: str, failure_message: str = "") -> None:
        with self._lock:
            if (status, failure_message) != (self.data["kaggle_status"], self.data["failure_message"]):
                self._urgent = True
            self.data["kaggle_status"], self.data["failure_message"] = status, failure_message

    def set_phase(self, phase: str) -> None:
        with self._lock:
            if self.data["monitor"]["phase"] != phase:
                self._urgent = True
            self.data["monitor"]["phase"] = phase

    @property
    def console(self) -> list[str]:
        return list(self.data["console"])

    def write(self, force: bool = False, **kwargs) -> None:
        """Write dashboard/data/training.js, at most every WRITE_INTERVAL_S unless something important changed."""
        with self._lock:
            now = time.time()
            if not (force or self._urgent or now - self._last_write >= WRITE_INTERVAL_S):
                return
            self.data["monitor"]["updated"] = _now_iso()
            write_dashboard_data("training", self.data, **kwargs)
            self._last_write, self._urgent = now, False


def _without(d: dict, *keys: str) -> dict:
    return {k: v for k, v in d.items() if k not in keys}
