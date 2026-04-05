from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay an execution trace.")
    parser.add_argument("trace_path", type=Path, help="Path to a saved execution trace.")
    args = parser.parse_args()

    trace = json.loads(args.trace_path.read_text(encoding="utf-8"))
    print(f"trace_id={trace['trace_id']}")
    print(f"task_id={trace['task_id']}")
    for event in trace["events"]:
        payload = json.dumps(event.get("payload", {}), ensure_ascii=False)
        print(f"[{event['status']}] {event['stage']}: {event['message']} payload={payload}")


if __name__ == "__main__":
    main()
