from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        required=True,
        help="Example: noumenon-labs/Fleet-Compiler-v0.5.1",
    )
    parser.add_argument(
        "--folder",
        default="artifacts/fleet-bank",
    )
    parser.add_argument(
        "--private",
        action="store_true",
    )
    parser.add_argument(
        "--allow-unverified-license",
        action="store_true",
        help="Explicitly override Fleet's redistribution license guard.",
    )

    args = parser.parse_args()
    folder = Path(args.folder)
    bank = json.loads(
        (folder / "fleet_bank.json").read_text(encoding="utf-8")
    )

    blocked = [
        expert
        for expert in bank.get("experts", [])
        if not expert.get("redistributable", False)
    ]

    if blocked and not args.allow_unverified_license:
        names = ", ".join(
            f"{x['name']} ({x.get('license') or 'unknown license'})"
            for x in blocked
        )
        raise SystemExit(
            "Refusing public Hub upload because some compiled derivatives "
            f"did not pass the redistribution allowlist: {names}. "
            "Remove them, verify their licenses, or explicitly pass "
            "--allow-unverified-license if you have separately confirmed rights."
        )

    api = HfApi()

    api.create_repo(
        repo_id=args.repo,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    api.upload_folder(
        repo_id=args.repo,
        repo_type="model",
        folder_path=str(folder),
        commit_message="Publish Fleet expert bank",
    )

    print(f"https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
