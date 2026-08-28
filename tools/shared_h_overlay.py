"""Two-pass shared-H overlay for the hash-pinned Brandon v3.1 encoder.

Implements the public ``shared_h_v1`` contract documented by
local-inference-lab/rtx6kpro. The original reviewed kquant patch is no longer
publicly retrievable, so this overlay is independently reviewable and must pass
the physical 9,228-tensor census, real-weight smoke, KLD, and runtime gates.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import struct
import time
from collections import defaultdict
from pathlib import Path

ROTATION_LAYOUT = "shared_h_v1"
EXPECTED_LAYER_TENSORS = 9_228


def _signed_geometric_mean(base, rows):
    torch, _ = base._lazy_torch()
    stack = torch.stack([row.flatten().float().cpu() for row in rows], dim=0).double()
    magnitudes = stack.abs().clamp_min(torch.finfo(torch.float64).tiny)
    signs = stack[0].sign()
    signs[signs == 0] = 1
    profile = signs * magnitudes.log().mean(dim=0).exp()
    # The serialized representation is F16. Force exactly that representable
    # profile during the second pass so every expert sees runtime-identical H.
    return profile.half().float().contiguous(), 0

def _forced_finalize_factory(base, original_finalize):
    def finalize(H_data, quant_args, verbose=False):
        forced_sign = H_data.get("forced_su_sign")
        if forced_sign is None:
            return original_finalize(H_data, quant_args, verbose)
        torch, _ = base._lazy_torch()
        with base.finalize_capture_H_mutex:
            if H_data["H"].is_meta:
                H_data["L"] = None
                H_data["finalized"] = True
                H_data["diag"] = None
                H_data["q_fallback"] = True
                hessian = H_data["H"]
                k = hessian.shape[0]
                torch.randn(k, device=H_data["device"])
                su = forced_sign.to(H_data["device"], dtype=torch.float32).view(k, 1)
                H_data["su"] = su
                return True, None, None, su, None
            if "H_swap_device" in H_data:
                H_data["H"] = H_data["H"].to(H_data["H_swap_device"])
                del H_data["H_swap_device"]
            hessian = H_data["H"]
            if H_data["finalized"]:
                return (
                    H_data["q_fallback"],
                    hessian,
                    H_data["L"],
                    H_data["su"],
                    H_data["diag"],
                )
            count = H_data["count"]
            if count == 0:
                q_fallback, diag_mean = True, 0.0
            else:
                hessian /= count
                diag_mean = torch.diag(hessian).mean()
                q_fallback = diag_mean.item() < 1e-20
            hessian.diagonal().add_(quant_args.get("sigma_reg", 0.025) * diag_mean)
            diagonal = hessian.diagonal().clone()
            k = hessian.shape[0]
            # Preserve the reference RNG stream so the following expert-local
            # SV draw is identical even though SU is forced.
            torch.randn(k, device=hessian.device)
            su = forced_sign.to(hessian.device, dtype=torch.float32).view(k, 1)
            H_data["su"] = su
            hessian *= su.T
            base.blockwise_preapply_had_r_(hessian, base.HAD_K)
            hessian *= su
            base.blockwise_preapply_had_l_(hessian, base.HAD_K)
            if q_fallback:
                lower = None
            else:
                lower, hessian = base.block_ldl(hessian, 16, quant_args, verbose)
                diagonal_indices = torch.arange(k)
                lower[diagonal_indices, diagonal_indices] = 0
            H_data["L"] = lower
            H_data["H"] = hessian.cpu()
            H_data["finalized"] = True
            H_data["diag"] = diagonal
            H_data["q_fallback"] = q_fallback
            return q_fallback, H_data["H"], lower, su, diagonal

    return finalize


def _shared_signs(base, layer, device):
    del device
    torch, _ = base._lazy_torch()
    template_path = os.environ.get("SHARED_H_SIGN_TEMPLATE")
    bundled_template = Path(__file__).resolve().with_name("shared_h_sign_template.json")
    if not template_path and bundled_template.is_file():
        template_path = str(bundled_template)
    if not template_path:
        raise RuntimeError("sealed shared-H sign template is required")
    receipt = json.loads(Path(template_path).read_text())
    claimed = receipt.get("receipt_sha256")
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if receipt.get("schema") != "glm53-shared-h-sign-template/1" or claimed != actual:
        raise RuntimeError("shared-H sign template seal differs")
    row = receipt.get("layers", {}).get(str(layer))
    if row is None:
        raise RuntimeError(f"shared-H sign template has no layer {layer}")

    def decode(item):
        payload = base64.b64decode(item["base64_u8_signbit"], validate=True)
        if (
            len(payload) != base.HIDDEN
            or hashlib.sha256(payload).hexdigest() != item["sha256"]
        ):
            raise RuntimeError("shared-H sign row differs")
        return torch.tensor(
            [1.0 - 2.0 * value for value in payload], dtype=torch.float32
        )

    signs = {"gate_up": decode(row["gate_up"])}
    signs.update(
        {("down", rank): decode(row["down"][str(rank)]) for rank in range(base.TP)}
    )
    return signs, claimed


def _force_h_side(base, ctx, profile, side):
    torch, _ = base._lazy_torch()
    weight = ctx["weight_r"]
    if side == "su":
        shared = profile.to(device=weight.device, dtype=torch.float32).view(ctx["k"], 1)
        local = ctx["su"]
        unrotated = base.preapply_had_l(weight, base.HAD_K)
        unrotated *= local
        unrotated /= shared
        ctx["weight_r"] = base.preapply_had_l(unrotated, base.HAD_K)
        ctx["su"] = shared
    elif side == "sv":
        shared = profile.to(device=weight.device, dtype=torch.float32).view(1, ctx["n"])
        local_su, local_sv = ctx["su"], ctx["sv"]
        unrotated = base.preapply_had_l(weight, base.HAD_K)
        unrotated *= local_su
        unrotated = base.preapply_had_r(unrotated, base.HAD_N)
        unrotated *= local_sv
        unrotated /= shared
        unrotated = base.preapply_had_r(unrotated, base.HAD_N)
        unrotated /= local_su
        ctx["weight_r"] = base.preapply_had_l(unrotated, base.HAD_K)
        ctx["sv"] = shared
    else:
        raise ValueError(side)
    ctx["shared_h_side"] = side
    return ctx


def _post_shared(base, ctx, logfile):
    ctx = base.encode_slice_prologue_post(ctx, logfile=logfile, self_check=True)
    if ctx["shared_h_side"] == "su":
        # Legacy post places g_scale in SU. SU is layer-shared here, so move the
        # algebraically equivalent reciprocal scalar into expert-local SV.
        ctx["su"] *= ctx["g_scale"]
        ctx["sv"] /= ctx["g_scale"]
    return ctx


def _profile_pass(base, src, calib, layer, out_scales, min_routed, logfile, shared_signs):
    torch, _ = base._lazy_torch()
    samples = defaultdict(list)
    routed_count = [0] * base.NUM_EXPERTS
    fallback_experts = []
    started = time.time()
    for expert in range(base.NUM_EXPERTS):
        weights = {
            proj: base.load_expert_bf16(src, layer, expert, proj, "cuda:0", logfile)
            for proj in base.PROJS
        }
        hd_gu, hd_down, hmeta = base.build_expert_hessians(
            calib, expert, weights["gate_proj"], weights["up_proj"], "cuda:0", min_routed
        )
        hd_gu["forced_su_sign"] = shared_signs["gate_up"].view(base.HIDDEN, 1)
        routed_count[expert] = hmeta["routed"]
        if hmeta["h_fallback"]:
            fallback_experts.append(expert)
        for projection_index, projection in enumerate(base.PROJS):
            for rank, weight_slice in base.expert_slices(weights[projection], projection):
                h_data = hd_gu if projection != "down_proj" else hd_down[rank]
                ctx = base.encode_slice_prologue_pre(
                    weight_slice,
                    h_data,
                    base.slice_seed(layer, expert, projection_index, rank),
                    out_scales,
                    logfile=logfile,
                )
                side = "sv" if projection == "down_proj" else "su"
                if side == "sv":
                    forced_sv = (
                        shared_signs[("down", rank)].to(ctx["sv"].device).view(1, base.HIDDEN)
                        * ctx["sv"].abs()
                    )
                    _force_h_side(base, ctx, forced_sv, "sv")
                samples[(projection, rank, side)].append(ctx[side].flatten().float().cpu())
                ctx.clear()
        base.free_H_data(hd_gu, *hd_down)
        del weights, hd_gu, hd_down
        if expert % 16 == 15:
            base.log(
                f"layer {layer} shared-H profile pass expert {expert:3d} "
                f"({(time.time()-started)/(expert+1):.1f}s/expert)", logfile
            )
    profiles, ties = {}, {}
    for key in sorted(samples):
        profiles[key], ties[key] = _signed_geometric_mean(base, samples[key])
    gate_sign = profiles[("gate_proj", 0, "su")].sign()
    for projection in ("gate_proj", "up_proj"):
        for rank in range(base.TP):
            if not torch.equal(profiles[(projection, rank, "su")].sign(), gate_sign):
                raise RuntimeError("gate/up shared-H profiles do not share one Hessian sign")
    del samples
    torch.cuda.empty_cache()
    return profiles, ties, routed_count, fallback_experts


def process_layer_shared(base, src, work, layer, out_scales, capture_dir, min_routed, logfile, lockstep_n=None):
    torch, _ = base._lazy_torch()
    started = time.time()
    device = "cuda:0"
    st_path, done_path = base.layer_paths(work, layer)
    lockstep_n = lockstep_n or 3 * base.TP
    calib = base.LayerCalib(capture_dir, layer, logfile)
    calib_tokens = calib.tokens
    calib_sha_x = calib.manifest.get("sha256_x")
    shared_signs, sign_template_sha256 = _shared_signs(base, layer, device)

    base.SHARED_H_PROFILE_PASS = True
    try:
        profiles, profile_ties, routed_count, h_fallback_experts = _profile_pass(
            base, src, calib, layer, out_scales, min_routed, logfile, shared_signs
        )
    finally:
        base.SHARED_H_PROFILE_PASS = False
    base.log(
        f"layer {layer}: shared-H profiles ready; second calibrated pass starting", logfile
    )

    stash = {}
    err_num = [0.0] * base.NUM_EXPERTS
    err_den = [0.0] * base.NUM_EXPERTS
    slice_nmse, slice_proxy = {}, {}
    q_fallback_slices = []
    out_scales_on = 0
    group_size = max(1, (lockstep_n + 3 * base.TP - 1) // (3 * base.TP))

    for expert_start in range(0, base.NUM_EXPERTS, group_size):
        experts = list(range(expert_start, min(expert_start + group_size, base.NUM_EXPERTS)))
        group_started = time.time()
        group_h_data, pre_list, walk_contexts = [], [], []
        for expert in experts:
            weights = {
                proj: base.load_expert_bf16(src, layer, expert, proj, device, logfile)
                for proj in base.PROJS
            }
            hd_gu, hd_down, hmeta = base.build_expert_hessians(
                calib, expert, weights["gate_proj"], weights["up_proj"], device, min_routed
            )
            if hmeta["routed"] != routed_count[expert]:
                raise RuntimeError(f"expert {expert}: routed count changed between shared-H passes")
            hd_gu["forced_su_sign"] = shared_signs["gate_up"].view(base.HIDDEN, 1)
            for projection_index, projection in enumerate(base.PROJS):
                for rank, weight_slice in base.expert_slices(weights[projection], projection):
                    h_data = hd_gu if projection != "down_proj" else hd_down[rank]
                    ctx = base.encode_slice_prologue_pre(
                        weight_slice,
                        h_data,
                        base.slice_seed(layer, expert, projection_index, rank),
                        out_scales,
                        logfile=logfile,
                    )
                    side = "sv" if projection == "down_proj" else "su"
                    _force_h_side(base, ctx, profiles[(projection, rank, side)], side)
                    pre_list.append((expert, projection, rank, ctx))
            del weights
            group_h_data.append((hd_gu, hd_down))

        base.g_scale_gss_lockstep(
            [ctx for _, _, _, ctx in pre_list], logfile, self_check=True
        )
        ordered = []
        for expert, projection, rank, ctx in pre_list:
            ctx = _post_shared(base, ctx, logfile)
            ordered.append((expert, projection, rank, ctx))
            if ctx["wq_reg"] is None:
                walk_contexts.append(ctx)
        del pre_list
        walks = [base.LDLQWalk(ctx) for ctx in walk_contexts]
        base.lockstep_ldlq(walks, lockstep_n, logfile)
        if walks and not base._self_checked_lockstep:
            base.lockstep_self_check(walk_contexts, logfile)
        del walks

        for expert, projection, rank, ctx in ordered:
            output, stats = base.encode_slice_epilogue(ctx)
            stash[(expert, projection, rank)] = output
            err_num[expert] += stats["sse"]
            err_den[expert] += stats["ss"]
            key = f"{expert}.{projection}.rank{rank}"
            slice_nmse[key] = stats["nmse"]
            slice_proxy[key] = stats["proxy_err"]
            if stats["q_fallback"]:
                q_fallback_slices.append(key)
            if stats["apply_out_scales"]:
                out_scales_on += 1
            ctx.clear()
        for hd_gu, hd_down in group_h_data:
            base.free_H_data(hd_gu, *hd_down)
        del ordered, walk_contexts, group_h_data
        seconds_per_expert = (time.time() - group_started) / len(experts)
        for expert in experts:
            if expert % 16 == 15:
                relative = err_num[expert] / max(err_den[expert], 1e-30)
                base.log(
                    f"layer {layer} shared-H expert {expert:3d} done "
                    f"({seconds_per_expert:.1f}s/expert, routed {routed_count[expert]}, "
                    f"last nmse {relative:.3e})", logfile
                )

    del calib
    if q_fallback_slices:
        base.log(
            f"layer {layer} WARNING: {len(q_fallback_slices)} q_fallback slices", logfile
        )
    relative_error = [
        err_num[expert] / max(err_den[expert], 1e-30)
        for expert in range(base.NUM_EXPERTS)
    ]
    order = sorted(range(base.NUM_EXPERTS), key=lambda expert: relative_error[expert], reverse=True)
    keep = sorted(order[:base.KEEP_NVFP4])
    tail = [expert for expert in range(base.NUM_EXPERTS) if expert not in set(keep)]

    entries = []
    profile_hasher = hashlib.sha256()
    for projection in base.PROJS:
        side = "sv" if projection == "down_proj" else "su"
        suffix = "svh" if side == "sv" else "suh"
        for rank in range(base.TP):
            tensor = profiles[(projection, rank, side)].half().contiguous()
            name = f"model.layers.{layer}.mlp.experts.shared_h.{projection}.rank{rank}.{suffix}"
            payload = tensor.numpy().tobytes()
            profile_hasher.update(name.encode())
            profile_hasher.update(b"\0")
            profile_hasher.update(payload)
            entries.append((name, "F16", tuple(tensor.shape), payload))

    mcg_bytes = struct.pack("<I", base.MCG_MULT)
    for expert in tail:
        for projection in base.PROJS:
            for rank in range(base.TP):
                tensor_set = stash[(expert, projection, rank)]
                prefix = f"model.layers.{layer}.mlp.experts.{expert}.{projection}.rank{rank}"
                trellis = tensor_set["trellis"]
                if projection == "down_proj":
                    tensor = tensor_set["suh"]
                    entries.append((f"{prefix}.suh", "F16", tuple(tensor.shape), tensor.numpy().tobytes()))
                else:
                    tensor = tensor_set["svh"]
                    entries.append((f"{prefix}.svh", "F16", tuple(tensor.shape), tensor.numpy().tobytes()))
                entries.append((f"{prefix}.trellis", "I16", tuple(trellis.shape), trellis.numpy().tobytes()))
                entries.append((f"{prefix}.mcg", "I32", (), mcg_bytes))
    if len(entries) != EXPECTED_LAYER_TENSORS:
        raise RuntimeError(f"shared-H layer tensor count {len(entries)} != {EXPECTED_LAYER_TENSORS}")

    _, file_sha = base.write_safetensors(st_path, entries, metadata={"format": "pt"})
    done = {
        "layer": layer,
        "bits": base.BITS,
        "codebook": "mcg",
        "hessian": "ldlq-calibrated",
        "sigma_reg": base.SIGMA_REG,
        "min_routed": min_routed,
        "capture": {"dir": capture_dir, "tokens": calib_tokens, "sha256_x": calib_sha_x},
        "expert_routed_count": routed_count,
        "experts_layer_h_fallback": h_fallback_experts,
        "q_fallback_slices": q_fallback_slices,
        "slices_with_out_scales": out_scales_on,
        "rotation_layout": ROTATION_LAYOUT,
        "shared_h_tensor_schema": "model.layers.{L}.mlp.experts.shared_h.{proj}.rank{r}.{suh|svh}",
        "shared_h_profile_sha256": profile_hasher.hexdigest(),
        "shared_h_sign_ties": {f"{proj}.rank{rank}.{side}": profile_ties[(proj, rank, side)] for proj, rank, side in sorted(profile_ties)},
        "shared_h_profile_policy": "sealed GLM-5.2 shared-H signs, geometric-mean(abs), F16 closure",
        "shared_h_sign_template_sha256": sign_template_sha256,
        "tp": base.TP,
        "keep_nvfp4": keep,
        "tail_tr3": tail,
        "expert_rel_rt_mse": relative_error,
        "slice_nmse": slice_nmse,
        "slice_proxy_err": slice_proxy,
        "seed_base": base.SEED_BASE,
        "lockstep": lockstep_n,
        "gss_lockstep": True,
        "exllamav3": "0.0.43",
        "tensor_count": len(entries),
        "file": os.path.basename(st_path),
        "file_sha256": file_sha,
        "encode_seconds": round(time.time() - started, 1),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    temporary = done_path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(done, handle, indent=1)
    os.replace(temporary, done_path)
    base.log(
        f"layer {layer} shared-H COMPLETE in {done['encode_seconds']}s; tensors={len(entries)}", logfile
    )


def install(base):
    base.finalize_capture_H = _forced_finalize_factory(base, base.finalize_capture_H)
    base.process_layer = lambda src, work, layer, out_scales, capture_dir, min_routed, logfile, lockstep_n=None: process_layer_shared(
        base, src, work, layer, out_scales, capture_dir, min_routed, logfile, lockstep_n
    )
    return base
