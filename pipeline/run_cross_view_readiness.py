"""Record licence, checkpoint and hardware readiness for cross-view backends.

This checkpoint performs no model inference and downloads nothing.  It keeps
public-deployment eligibility separate from research availability so a model is
never presented as usable merely because its source repository is public.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BACKENDS: dict[str, dict[str, Any]] = {
    "vggt": {
        "display_name": "VGGT",
        "source": "https://github.com/facebookresearch/vggt",
        "source_revision": "a288dd0f14786c93483e45524328726ab7b1b4ce",
        "license_sha256": "b94e46746792d1adfa0ebdbccada5d4576d244ae230be9de8a46b5868fff001b",
        "license": "VGGT License v1",
        "public_product_eligible": True,
        "checkpoint_policy": "VGGT-1B-Commercial only; gated acceptance required",
        "role": "camera_pose_depth_point_map_and_tracks",
    },
    "vggt_omega": {
        "display_name": "VGGT-Omega",
        "source": "https://github.com/facebookresearch/vggt-omega",
        "source_revision": "48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6",
        "license_sha256": "27696e51930daf5d24222e2d9d65339f529153e853de0e2eb457133a1a6f5239",
        "license": "FAIR Noncommercial Research License v1",
        "public_product_eligible": False,
        "official_single_frame_peak_gpu_gib": 6.02,
        "checkpoint_policy": "gated noncommercial research checkpoint",
        "role": "camera_pose_depth_and_point_cloud",
    },
    "dust3r": {
        "display_name": "DUSt3R",
        "source": "https://github.com/naver/dust3r",
        "source_revision": "4c24a6ebf04809f2cfe59915e51779c8984aaa40",
        "license_sha256": "f06b394a6b4804e68c4f8ff4298aec5cc981fba8934fc56525c91cc64a7d1d58",
        "license": "CC BY-NC-SA 4.0",
        "public_product_eligible": False,
        "checkpoint_policy": "noncommercial use only",
        "role": "pairwise_dense_3d_and_camera_geometry",
    },
    "mast3r": {
        "display_name": "MASt3R",
        "source": "https://github.com/naver/mast3r",
        "source_revision": "f5209afc300cec36239a7ac992263f36847bbba0",
        "license_sha256": "949b1a3b437ed0b38ce51dae24babcca3395c9adae55163311ba84d44766b8ec",
        "license": "CC BY-NC-SA 4.0",
        "public_product_eligible": False,
        "checkpoint_policy": "noncommercial use only",
        "role": "learned_feature_matching_and_metric_point_maps",
    },
}

VGGT_LOCAL_GPU_POLICY_GIB = 8.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(
            ("git", "-C", str(root), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def runtime_probe() -> dict[str, Any]:
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            gpu_name = torch.cuda.get_device_name(0)
        else:
            free_bytes, total_bytes, gpu_name = 0, 0, None
        return {
            "torch_version": torch.__version__,
            "cuda_available": cuda_available,
            "gpu_name": gpu_name,
            "gpu_free_gib": round(free_bytes / 1024**3, 3),
            "gpu_total_gib": round(total_bytes / 1024**3, 3),
        }
    except (ImportError, RuntimeError):
        return {
            "torch_version": None,
            "cuda_available": False,
            "gpu_name": None,
            "gpu_free_gib": 0.0,
            "gpu_total_gib": 0.0,
        }


def build_report(
    root: Path,
    vggt_checkpoint: Path | None = None,
    vggt_license_accepted: bool = False,
    code_commit: str | None = None,
) -> dict[str, Any]:
    runtime = runtime_probe()
    checkpoint = vggt_checkpoint.resolve() if vggt_checkpoint else None
    checkpoint_present = bool(checkpoint and checkpoint.exists())
    gpu_ready = runtime["gpu_total_gib"] >= VGGT_LOCAL_GPU_POLICY_GIB

    candidates: dict[str, dict[str, Any]] = {}
    for backend_id, definition in BACKENDS.items():
        candidate = dict(definition)
        blockers: list[str] = []
        if not candidate["public_product_eligible"]:
            blockers.append("noncommercial_license_blocks_public_product_integration")
        if backend_id == "vggt":
            if not checkpoint_present:
                blockers.append("commercial_checkpoint_not_supplied")
            if not vggt_license_accepted:
                blockers.append("commercial_checkpoint_license_not_accepted")
            if not gpu_ready:
                blockers.append("local_gpu_below_8_gib_policy")
            candidate["local_checkpoint"] = str(checkpoint) if checkpoint else None
            candidate["local_checkpoint_present"] = checkpoint_present
            candidate["license_accepted_by_operator"] = bool(vggt_license_accepted)
            candidate["local_gpu_policy_gib"] = VGGT_LOCAL_GPU_POLICY_GIB
        elif backend_id == "vggt_omega" and runtime["gpu_total_gib"] < 6.02:
            blockers.append("local_gpu_below_official_single_frame_peak")
        candidate["blockers"] = blockers
        candidate["status"] = "ready" if not blockers else "blocked"
        candidates[backend_id] = candidate

    locally_ready = [name for name, item in candidates.items() if item["status"] == "ready"]
    public_candidates = [
        name for name, item in candidates.items() if item["public_product_eligible"]
    ]
    return {
        "schema_version": "cross_view_backend_readiness_v1",
        "created_at_utc": _utc_now(),
        "code_commit": code_commit or _git_commit(root),
        "execution_profile": "public_team_review_images_only",
        "runtime": runtime,
        "candidates": candidates,
        "summary": {
            "candidate_count": len(candidates),
            "public_product_candidate_count": len(public_candidates),
            "public_product_candidates": public_candidates,
            "locally_ready_count": len(locally_ready),
            "locally_ready": locally_ready,
        },
        "claims": {
            "cross_view_inference_executed": False,
            "camera_pose_accuracy_validated": False,
            "point_cloud_accuracy_validated": False,
            "cad_generated": False,
        },
        "next_action": "obtain_and_accept_VGGT_1B_Commercial_checkpoint_and_run_on_gpu_with_at_least_8_GiB",
        "decision": "blocked_pending_checkpoint_license_acceptance_and_compatible_hardware",
        "notice": "Engineering licence audit only; not legal advice.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/phase_runs/v1/cross_view_backend_readiness_v1/report.json"),
    )
    parser.add_argument("--vggt-commercial-checkpoint", type=Path)
    parser.add_argument("--vggt-license-accepted", action="store_true")
    parser.add_argument(
        "--code-commit",
        help="Explicit source commit for minimal containers that do not contain Git",
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(
        root,
        args.vggt_commercial_checkpoint,
        args.vggt_license_accepted,
        args.code_commit,
    )
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output), "summary": report["summary"], "decision": report["decision"]}, indent=2))


if __name__ == "__main__":
    main()
