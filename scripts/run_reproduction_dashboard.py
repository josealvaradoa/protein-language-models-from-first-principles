"""Start the local-only Week 4 reproduction dashboard control plane."""

from __future__ import annotations

import argparse
from pathlib import Path

from protein_lm.dashboard import DashboardControlPlane, create_server


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    control = DashboardControlPlane(PROJECT_ROOT)
    server = create_server(control, host=args.host, port=args.port)
    print(f"dashboard listening on http://{args.host}:{server.server_address[1]}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        control.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
