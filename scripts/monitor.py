"""Follow the Kaggle training run and keep the dashboard data current.

Polls the kernel status, streams the live execution log while the session runs,
turns the progress events emitted by kaggle/train_kernel.py into
dashboard/data/training.js, and once the run completes downloads its outputs to
runs/kaggle/v<version>/ and builds the reports (scripts/build_reports.py).

    python scripts/monitor.py [--open] [--no-download]

Safe to restart at any point: the log stream replays from the beginning.
"""

from __future__ import annotations

import argparse
import threading
import time
import webbrowser

import requests
import urllib3.exceptions

from common import DASHBOARD_DIR, RUNS_DIR, kaggle_api, read_current_run
from run_state import RunState

QUEUED_POLL_S = 20
STATUS_POLL_S = 30
RETRY_DELAYS_S = (5, 10, 20, 30, 60)
HTTP_TIMEOUT_S = (30, 300)  # connect, read; a training session logs every few seconds
TERMINAL = {"complete", "error", "cancel_acknowledged"}
NETWORK_ERRORS = (requests.RequestException, urllib3.exceptions.HTTPError, ConnectionError, TimeoutError)


def _with_kaggle_fixes(request):
    """Two fixes for the Kaggle client's requests.

    It sends them without a timeout, so a dropped connection (network change, laptop sleep)
    used to block the log stream forever; a read timeout turns that into a retryable error.
    And the log stream is UTF-8 without a charset header, which requests decodes as ISO-8859-1.
    Besides garbling non-ASCII text, that made the client's str.splitlines() break events at
    U+0085, which comes out of the bytes of characters such as ✅, so they lost their JSON framing.
    """
    def wrapper(self, method, url, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = HTTP_TIMEOUT_S
        response = request(self, method, url, **kwargs)
        if "/kernels/logs/stream/" in url:
            response.encoding = "utf-8"
        return response
    return wrapper


requests.Session.request = _with_kaggle_fixes(requests.Session.request)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def poll_status(api, ref: str, state: RunState) -> str:
    st = api.kernels_status(ref)
    status = st.status.name.lower()
    state.set_status(status, st.failure_message or "")
    return status


def status_poller(api, ref: str, state: RunState, stop: threading.Event) -> None:
    """Keep the Kaggle status fresh while the main thread blocks on the log stream."""
    while not stop.wait(STATUS_POLL_S):
        try:
            poll_status(api, ref, state)
            state.write()
        except NETWORK_ERRORS as exc:
            log(f"status poll failed: {exc}")


def stream_logs(api, ref: str, state: RunState, seen: int) -> int:
    """Consume the log stream until the server ends it; return how many events were handled.

    On reconnect the server replays from the start, so the first `seen` events are skipped.
    """
    for index, event in enumerate(api.kernels_logs_stream(ref)):
        if index < seen:
            continue
        seen = index + 1
        if event.get("data"):
            state.feed(event.get("stream_name") or "stdout", event["data"])
        state.write()
    state.flush()
    state.write(force=True)
    return seen


def follow(api, ref: str, state: RunState) -> str:
    """Stream the run until Kaggle reports a terminal status, and return that status."""
    seen, failures, announced = 0, 0, None
    while True:
        try:
            status = poll_status(api, ref, state)
            if status != announced:
                log(f"kernel status: {status}")
                announced = status
            if status in {"queued", "new_script"}:
                state.set_phase("waiting")
                state.write(force=True)
                time.sleep(QUEUED_POLL_S)
                continue
            if status in TERMINAL and seen:
                return status
            state.set_phase("streaming")
            state.write(force=True)
            seen_before = seen
            seen = stream_logs(api, ref, state, seen)
            failures = 0 if seen > seen_before else failures + 1
            if status in TERMINAL:  # a finished session streams its persisted log once
                return status
        except NETWORK_ERRORS as exc:
            failures += 1
            state.set_phase("reconnecting")
            state.write(force=True)
            log(f"connection problem ({type(exc).__name__}); retrying")
        time.sleep(RETRY_DELAYS_S[min(failures, len(RETRY_DELAYS_S)) - 1] if failures else 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--open", action="store_true", help="open the dashboard in the default browser")
    parser.add_argument("--no-download", action="store_true", help="do not download outputs or build reports")
    args = parser.parse_args()

    run = read_current_run()
    ref = run["kernel"]
    api = kaggle_api()
    state = RunState(run)
    state.write(force=True)
    if args.open:
        webbrowser.open((DASHBOARD_DIR / "index.html").as_uri())
    log(f"following {ref} (version {run['version']}): {run['url']}")

    stop = threading.Event()
    threading.Thread(target=status_poller, args=(api, ref, state, stop), daemon=True).start()
    try:
        status = follow(api, ref, state)
    finally:
        stop.set()
    log(f"run finished with status: {status}")

    if args.no_download:
        state.set_phase("finished" if status == "complete" else "failed")
        state.write(force=True)
        return
    out_dir = RUNS_DIR / "kaggle" / f"v{run['version']}"
    state.set_phase("downloading")
    state.write(force=True)
    log(f"downloading outputs to {out_dir}")
    for attempt in range(1, 4):
        try:
            api.kernels_output(ref, path=str(out_dir), force=True, quiet=True)
            break
        except NETWORK_ERRORS as exc:
            if attempt == 3:
                raise
            log(f"download failed ({type(exc).__name__}); retrying in 30 s")
            time.sleep(30)
    if status != "complete":
        state.set_phase("failed")
        state.write(force=True)
        raise SystemExit(f"Kaggle run ended with status {status}: {state.data['failure_message'] or 'see the log'}")

    state.set_phase("reporting")
    state.write(force=True)
    import build_reports

    build_reports.build(out_dir, run, console=state.console)
    log("reports and dashboard data are up to date")


if __name__ == "__main__":
    main()
