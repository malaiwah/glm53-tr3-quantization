#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from release_gate import canonical_sha256


def main() -> int:
    parser = argparse.ArgumentParser(description="Flip one HF model public only with a sealed publication authorization")
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    authorization = json.loads(args.authorization.read_text())
    claimed = authorization.get("receipt_sha256")
    body = {key: value for key, value in authorization.items() if key != "receipt_sha256"}
    if (
        authorization.get("schema") != "glm53-publication-authorization/1"
        or claimed != canonical_sha256(body)
        or authorization.get("public_flip_authorized") is not True
        or authorization.get("hf_repo") != args.repo
    ):
        raise SystemExit("publication authorization is invalid or repo-mismatched")
    from huggingface_hub import HfApi

    token = args.token_file.read_text().strip()
    api = HfApi(token=token)
    api.update_repo_settings(repo_id=args.repo, repo_type="model", private=False)
    info = api.model_info(args.repo)
    if info.private:
        raise RuntimeError("Hugging Face repo remained private after update")
    result_body = {
        "schema": "glm53-hf-publication/1",
        "repo": args.repo,
        "source_revision": authorization["source_revision"],
        "authorization_sha256": claimed,
        "hub_revision": info.sha,
        "private": False,
        "published_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    result = {**result_body, "receipt_sha256": canonical_sha256(result_body)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"repo": args.repo, "revision": info.sha, "public": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
