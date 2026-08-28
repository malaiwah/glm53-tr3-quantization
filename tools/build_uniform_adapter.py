#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ENCODER_SHA256 = "e9a85a47e165c8d8644354cef611efbb81dfd9ba88544ca59f0c80ee6bc75032"
ADAPTER_SHA256 = "f378817b212dc9f4a8c9dc049803542e7c91748283f6e8ec1ebe0427be96aaf1"
CORPUS_SHA256 = "cf247acc7c5da9f0600c7d6ab3b7c2fcfc54ec30b794e3b6047559285fa44df4"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}: {old!r}")
    return text.replace(old, new, 1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a hash-pinned two-pass shared-H K3/K4/K5/K6 adapter from Brandon's v31 bundle"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source, out = args.source.resolve(), args.out.resolve()
    encoder_source = source / "encode_tr3_v31.py"
    adapter_source = source / "encode_b300.py"
    corpus_source = source / "calibration" / "reap_recall_calib.jsonl"
    if sha256_file(encoder_source) != ENCODER_SHA256:
        raise SystemExit("production encoder SHA-256 differs")
    if sha256_file(adapter_source) != ADAPTER_SHA256:
        raise SystemExit("B300 adapter SHA-256 differs")
    if sha256_file(corpus_source) != CORPUS_SHA256:
        raise SystemExit("calibration corpus SHA-256 differs")
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(source, out)
    overlay_source = Path(__file__).resolve().with_name("shared_h_overlay.py")
    overlay_path = out / "shared_h_overlay.py"
    shutil.copy2(overlay_source, overlay_path)
    overlay_sha = sha256_file(overlay_path)
    sign_template_source = (
        Path(__file__).resolve().parents[1]
        / "baselines"
        / "willfalco-3.42"
        / "shared_h_sign_template.json"
    )
    sign_template = json.loads(sign_template_source.read_text())
    sign_template_body = {
        key: value for key, value in sign_template.items() if key != "receipt_sha256"
    }
    sign_template_seal = hashlib.sha256(
        json.dumps(sign_template_body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        sign_template.get("schema") != "glm53-shared-h-sign-template/1"
        or sign_template.get("receipt_sha256") != sign_template_seal
        or sorted(map(int, sign_template.get("layers", {}))) != list(range(3, 79))
    ):
        raise SystemExit("shared-H sign template contract differs")
    sign_template_path = out / "shared_h_sign_template.json"
    shutil.copy2(sign_template_source, sign_template_path)
    sign_template_sha = sha256_file(sign_template_path)

    encoder_path = out / "encode_tr3_v31.py"
    encoder = encoder_path.read_text()
    encoder = replace_once(
        encoder,
        "BITS = 3                      # owner-pinned: EXACTLY 3.0 bpw",
        "BITS = int(os.environ.get(\"TR3_BITS\", \"3\"))  # uniform campaign rate",
        "encoder bits",
    )
    encoder_path.write_text(encoder)
    patched_encoder_sha = sha256_file(encoder_path)

    adapter_path = out / "encode_b300.py"
    adapter = adapter_path.read_text()
    replacements = [
        (
            f'BASE_ENCODER_SHA256 = "{ENCODER_SHA256}"',
            f'BASE_ENCODER_SHA256 = "{patched_encoder_sha}"',
            "encoder pin",
        ),
        ('ADAPTER_VERSION = "1"', 'ADAPTER_VERSION = "shared-h-uniform-k-v1"', "adapter version"),
        (
            "EXPECTED_LAYER_TENSORS = 256 * 3 * 4 * 4",
            "EXPECTED_LAYER_TENSORS = 9_228",
            "shared-H layer census",
        ),
        (
            "if ACTIVE_SOURCE_HASHER is not None:",
            "if ACTIVE_SOURCE_HASHER is not None and not getattr(BASE, \"SHARED_H_PROFILE_PASS\", False):",
            "single source audit pass",
        ),
        (
            "    module.layer_done = layer_done\n    return module",
            """    module.layer_done = layer_done
    from shared_h_overlay import install as install_shared_h
    install_shared_h(module)
    return module""",
            "shared-H overlay install",
        ),
        (
            """        \"adapter_version\": ADAPTER_VERSION,
        \"exllamav3\": \"0.0.43\",""",
            f"""        \"adapter_version\": ADAPTER_VERSION,
        \"rotation_layout\": \"shared_h_v1\",
        \"shared_h_overlay_sha256\": \"{overlay_sha}\",
        \"shared_h_sign_template_sha256\": \"{sign_template_sha}\",
        \"exllamav3\": \"0.0.43\",""",
            "recipe shared-H binding",
        ),
        (
            """        \"producer_version\": ADAPTER_VERSION,
        \"source_format\": \"BF16\",""",
            f"""        \"producer_version\": ADAPTER_VERSION,
        \"rotation_layout\": \"shared_h_v1\",
        \"shared_h_tensor_schema\": \"model.layers.{{L}}.mlp.experts.shared_h.{{proj}}.rank{{r}}.{{suh|svh}}\",
        \"shared_h_sign_template_sha256\": \"{sign_template_sha}\",
        \"source_format\": \"BF16\",""",
            "config shared-H metadata",
        ),
        (
            '        \"tensor_schema\": \"model.layers.{L}.mlp.experts.{E}.{proj}.rank{r}.{trellis|suh|svh|mcg}\",',
            '        \"tensor_schema\": \"expert-local trellis+mcg; gate/up svh; down suh; layer-shared H-side rotations\",',
            "config expert-local schema",
        ),
        ("and module.BITS == 3", "and module.BITS in (3, 4, 5, 6)", "bits admission"),
        ('"bits": 3,', '"bits": base.BITS,', "recipe bits"),
        ('("I16", (k // 16, n // 16, 48))', '("I16", (k // 16, n // 16, 16 * base.BITS))', "trellis shape"),
        ('int(done.get("bits", -1)) == 3', 'int(done.get("bits", -1)) == base.BITS', "done bits"),
        ('f"workers={args.workers}, GPUs={args.gpus}, bits=3, keep=0, tail=256, "',
         'f"workers={args.workers}, GPUs={args.gpus}, bits={BASE.BITS}, keep=0, tail=256, "',
         "log bits"),
        ('"bits": 3.0,', '"bits": float(BASE.BITS),', "config bits"),
        ('"expert_rel_rt_mse": dones[layer]["expert_rel_rt_mse"],',
         '"expert_rel_rt_mse": dones[layer]["expert_rel_rt_mse"],\n            "k": [BASE.BITS] * 256,',
         "tier bitmap bits"),
        (
            "CURRENT_EXPECTED_RECIPE = None",
            """CURRENT_EXPECTED_RECIPE = None
SMOKE_FIXTURE = os.environ.get(\"TR3_SMOKE_FIXTURE\", \"0\") == \"1\"""",
            "smoke fixture flag",
        ),
        (
            """    if plan.get(\"schema\") != CAPTURE_PLAN_SCHEMA:
        raise RuntimeError(f\"unexpected capture plan schema: {plan.get('schema')!r}\")""",
            """    allowed_schemas = {CAPTURE_PLAN_SCHEMA}
    if SMOKE_FIXTURE:
        allowed_schemas.add(\"glm52-smoke-capture-plan-v1\")
    if plan.get(\"schema\") not in allowed_schemas:
        raise RuntimeError(f\"unexpected capture plan schema: {plan.get('schema')!r}\")""",
            "smoke plan schema",
        ),
        (
            "    if plan.get(\"corpus_sha256\") != CORPUS_SHA256:",
            "    if not SMOKE_FIXTURE and plan.get(\"corpus_sha256\") != CORPUS_SHA256:",
            "smoke corpus boundary",
        ),
        (
            "    if plan.get(\"selection_policy\") != \"owner-corpus-axis-separated-luke-multipass-no-repeat-v1\":",
            "    if not SMOKE_FIXTURE and plan.get(\"selection_policy\") != \"owner-corpus-axis-separated-luke-multipass-no-repeat-v1\":",
            "smoke selection policy",
        ),
        (
            "    if plan.get(\"owner_corpus_only\") is not True:",
            "    if not SMOKE_FIXTURE and plan.get(\"owner_corpus_only\") is not True:",
            "smoke owner corpus flag",
        ),
        (
            "    if plan.get(\"calibration_baseline\") is not True:",
            "    if not SMOKE_FIXTURE and plan.get(\"calibration_baseline\") is not True:",
            "smoke calibration flag",
        ),
    ]
    for old, new, label in replacements:
        adapter = replace_once(adapter, old, new, label)
    adapter = replace_once(
        adapter,
        '''def _expected_layer_entries(layer: int) -> dict[str, tuple[str, tuple[int, ...]]]:
    base = BASE
    assert base is not None
    expected = {}
    for expert in range(base.NUM_EXPERTS):
        for proj in base.PROJS:
            for rank in range(base.TP):
                prefix = f"model.layers.{layer}.mlp.experts.{expert}.{proj}.rank{rank}"
                if proj == "down_proj":
                    k, n = base.SLICE, base.HIDDEN
                else:
                    k, n = base.HIDDEN, base.SLICE
                expected[f"{prefix}.suh"] = ("F16", (k,))
                expected[f"{prefix}.svh"] = ("F16", (n,))
                expected[f"{prefix}.trellis"] = ("I16", (k // 16, n // 16, 16 * base.BITS))
                expected[f"{prefix}.mcg"] = ("I32", ())
    return expected''',
        '''def _expected_layer_entries(layer: int) -> dict[str, tuple[str, tuple[int, ...]]]:
    base = BASE
    assert base is not None
    expected = {}
    for proj in base.PROJS:
        side = "svh" if proj == "down_proj" else "suh"
        for rank in range(base.TP):
            expected[f"model.layers.{layer}.mlp.experts.shared_h.{proj}.rank{rank}.{side}"] = (
                "F16", (base.HIDDEN,)
            )
    for expert in range(base.NUM_EXPERTS):
        for proj in base.PROJS:
            for rank in range(base.TP):
                prefix = f"model.layers.{layer}.mlp.experts.{expert}.{proj}.rank{rank}"
                if proj == "down_proj":
                    k, n = base.SLICE, base.HIDDEN
                    expected[f"{prefix}.suh"] = ("F16", (k,))
                else:
                    k, n = base.HIDDEN, base.SLICE
                    expected[f"{prefix}.svh"] = ("F16", (n,))
                expected[f"{prefix}.trellis"] = ("I16", (k // 16, n // 16, 16 * base.BITS))
                expected[f"{prefix}.mcg"] = ("I32", ())
    return expected''',
        "shared-H tensor schema",
    )
    adapter_path.write_text(adapter)

    bootstrap_path = out / "bootstrap_ext_b300.py"
    bootstrap = bootstrap_path.read_text()
    bootstrap = replace_once(
        bootstrap,
        'ARCH_LIST = "10.0"',
        'ARCH_LIST = os.environ.get("EXL3_ARCH_LIST", "10.0")',
        "portable arch",
    )
    bootstrap = replace_once(
        bootstrap,
        "    roots: list[Path] = []\n",
        "    roots: list[Path] = []\n"
        "    if explicit := os.environ.get(\"EXLLAMAV3_EXT_SOURCE\"):\n"
        "        roots.append(Path(explicit))\n",
        "explicit extension source",
    )
    bootstrap = replace_once(
        bootstrap,
        "import importlib.metadata",
        "import importlib.metadata\nimport importlib.util",
        "prebuilt import support",
    )
    bootstrap = replace_once(
        bootstrap,
        '''    \"\"\"Return the guarded sm_100 module and register it as ``exllamav3_ext``.\"\"\"
    existing = sys.modules.get(\"exllamav3_ext\")''',
        '''    \"\"\"Return the guarded extension and register it as ``exllamav3_ext``.\"\"\"
    prebuilt = os.environ.get(\"EXLLAMAV3_EXT_PREBUILT\")
    if prebuilt:
        prebuilt_path = Path(prebuilt).expanduser().resolve()
        if not prebuilt_path.is_file():
            raise RuntimeError(f\"prebuilt extension missing: {prebuilt_path}\")
        spec = importlib.util.spec_from_file_location(\"exllamav3_ext\", prebuilt_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f\"cannot load prebuilt extension: {prebuilt_path}\")
        ext = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ext)
        actual = _assert_ops(ext)
        ext.__dict__.setdefault(\"_b300_bootstrap\", {})
        ext.__dict__[\"_b300_bootstrap\"].update({
            \"arch_list\": ARCH_LIST,
            \"exllamav3\": REQUIRED_VERSION,
            \"prebuilt\": str(prebuilt_path),
            \"observed_ops\": actual,
        })
        sys.modules[\"exllamav3_ext\"] = ext
        return ext
    existing = sys.modules.get(\"exllamav3_ext\")''',
        "prebuilt extension load",
    )
    bootstrap_path.write_text(bootstrap)

    receipt = out / "UNIFORM_ADAPTER.txt"
    receipt.write_text(
        "source_encoder_sha256=" + ENCODER_SHA256 + "\n"
        "source_adapter_sha256=" + ADAPTER_SHA256 + "\n"
        "patched_encoder_sha256=" + patched_encoder_sha + "\n"
        "calibration_corpus_sha256=" + CORPUS_SHA256 + "\n"
        "shared_h_overlay_sha256=" + overlay_sha + "\n"
        "shared_h_sign_template_sha256=" + sign_template_sha + "\n"
        "supported_bits=3,4,5,6\n"
        "rotation_layout=shared_h_v1\n"
        "representation=two-pass shared-H uniform parts\n"
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
