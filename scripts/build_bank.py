from fleet.adapters import build_bank

if __name__ == "__main__":
    bank = build_bank(
        "configs/experts.json",
        "artifacts/fleet-bank",
    )
    print(
        "Built:",
        [x["name"] for x in bank["experts"]],
    )
