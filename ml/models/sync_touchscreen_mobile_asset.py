"""Copy touchscreen_mobile_model/model.json to mobile/assets/models/touchscreen_model.json."""

import json
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "models" / "touchscreen_mobile_model" / "model.json"
DST = Path(__file__).resolve().parents[2] / "mobile" / "assets" / "models" / "touchscreen_model.json"


def sync_asset():
    DST.parent.mkdir(parents=True, exist_ok=True)
    with open(SRC, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(DST, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Synced {SRC} -> {DST}")


if __name__ == "__main__":
    sync_asset()
