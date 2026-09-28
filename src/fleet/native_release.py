from __future__ import annotations

from pathlib import Path
import os

from huggingface_hub import HfApi

from .native_build import (
    build_native_checkpoint,
)
from .native_validate import (
    validate_native_checkpoint,
)
from .validation import (
    validate_bank,
)


def _resolve_token():
    token = os.environ.get(
        "HF_TOKEN"
    )

    if token:
        return token

    try:
        from google.colab import userdata  # type: ignore
        return userdata.get(
            "HF_TOKEN"
        )
    except Exception:
        return None


def release_native(
    *,
    bank_dir: str | Path,
    output_dir: str | Path,
    repo_id: str,
    private: bool = False,
    max_shard_size: str = "4GB",
    num_workers: int = 4,
) -> dict:
    validation = validate_bank(
        bank_dir
    )

    if not validation["ok"]:
        raise RuntimeError(
            "Compiled bank failed validation:\n"
            + "\n".join(
                validation[
                    "errors"
                ]
            )
        )

    build = build_native_checkpoint(
        bank_dir=bank_dir,
        output_dir=output_dir,
        dtype="float16",
        max_shard_size=(
            max_shard_size
        ),
    )

    roundtrip = (
        validate_native_checkpoint(
            output_dir
        )
    )

    if not roundtrip["ok"]:
        raise RuntimeError(
            "Native Fleet round-trip validation failed. Nothing uploaded."
        )

    token = _resolve_token()

    if not token:
        raise RuntimeError(
            "No HF_TOKEN write token available. "
            "Nothing uploaded."
        )

    api = HfApi(
        token=token
    )

    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )

    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(
            output_dir
        ),
        num_workers=int(
            num_workers
        ),
    )

    return {
        "ok": True,
        "repo_id": (
            repo_id
        ),
        "url": (
            f"https://huggingface.co/"
            f"{repo_id}"
        ),
        "build": build,
        "roundtrip": roundtrip,
    }
