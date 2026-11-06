"""Run both web app tabs locally with the fine-tuned model: make app (CUDA) or make app-cpu.

    python scripts/run_app.py [--device cuda|cpu] [--port 7860] [--open]
"""

from __future__ import annotations

import argparse
import os
import socket
import sys

from common import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--open", action="store_true", help="open the app in your browser")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    with socket.socket() as sock:
        # Match Uvicorn's socket settings: recent connections in TIME_WAIT must not block a normal restart.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", args.port))
        except OSError:
            parser.error(f"Port {args.port} is in use. Open the running app or choose --port.")

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Install GPU dependencies with `make setup-gpu`, "
                         "or explicitly use `make app-cpu`.")
    device = torch.cuda.get_device_name(0) if args.device == "cuda" else "CPU"
    print(f"Local inference: {device}. Photos and live video stay on this computer.", flush=True)
    from deploy_space import stage

    dest = stage(load_config(), local=True)
    os.environ.update({"BLOOD_CELLS_LOCAL": "1", "BLOOD_CELLS_DEVICE": args.device,
                       "BLOOD_CELLS_PORT": str(args.port), "BLOOD_CELLS_OPEN": "1" if args.open else "0",
                       "GRADIO_ANALYTICS_ENABLED": "False"})
    sys.stdout.flush()
    os.chdir(dest)
    os.execv(sys.executable, [sys.executable, "-u", "app.py"])


if __name__ == "__main__":
    main()
