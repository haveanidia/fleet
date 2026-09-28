import json

from fleet.validation import validate_bank

if __name__ == "__main__":
    result = validate_bank(
        "artifacts/fleet-bank"
    )
    print(json.dumps(result, indent=2))

    if not result["ok"]:
        raise SystemExit(1)
