from fleet.semantic_build import build_semantic_router

if __name__ == "__main__":
    meta = build_semantic_router(
        bank_dir="artifacts/fleet-bank",
        anchors_path="configs/semantic_anchors.json",
        load_in_4bit=True,
    )

    print(meta)
