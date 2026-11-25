"""Flask application for browsing STL-1 and its retained phase evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from uuid import uuid4

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_file

from studio.dataset_service import (
    AssetNotFoundError,
    DatasetError,
    DatasetRepository,
    ObjectNotFoundError,
)


def create_app(repo_root: str | Path | None = None) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024
    repository = DatasetRepository(repo_root)
    app.config["DATASET_REPOSITORY"] = repository
    generated_root = repository.phase_root / "studio_image_to_3d"

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "dataset": "STL-1",
                "objects": len(repository.records),
                "read_only": False,
                "write_scope": "identified_phase1_and_phase4_image_space_review_decisions_plus_corrected_silhouettes",
            }
        )

    @app.get("/api/summary")
    def summary():
        return jsonify(repository.summary())

    @app.get("/api/showcase-models")
    def showcase_models():
        return jsonify({"models": repository.showcase_models()})

    @app.get("/api/showcase-models/<model_id>/<kind>")
    def showcase_model_asset(model_id: str, kind: str):
        path = repository.showcase_asset(model_id, kind)
        return send_file(
            path,
            as_attachment=kind == "model" and request.args.get("download") == "1",
            conditional=True,
        )

    @app.post("/api/reconstruct/image-to-3d")
    def reconstruct_image_to_3d():
        required_views = ("front", "top", "lsv", "rsv")
        missing = [view for view in required_views if request.files.get(view) is None]
        if missing:
            raise DatasetError(
                "Four labelled views are required: Front, Top, Left and Right"
            )

        job_id = uuid4().hex
        job_root = generated_root / job_id
        input_root = job_root / "inputs"
        input_root.mkdir(parents=True, exist_ok=False)
        image_paths: dict[str, Path] = {}
        for view in required_views:
            upload = request.files[view]
            payload = upload.read()
            if not payload:
                raise DatasetError(f"The {view} image is empty")
            decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                raise DatasetError(f"The {view} upload is not a valid image")
            height, width = decoded.shape[:2]
            if width > 6000 or height > 6000 or width * height > 25_000_000:
                raise DatasetError(f"The {view} image is too large")
            path = input_root / f"{view}.png"
            if not cv2.imwrite(str(path), decoded):
                raise DatasetError(f"The {view} image could not be prepared")
            image_paths[view] = path

        try:
            # Keep the geometry stack lazy so opening Studio does not load Torch,
            # scikit-image and trimesh until the user starts reconstruction.
            from pipeline.run_saraswati_blind_benchmark import reconstruct_from_images

            model_path, record_path = reconstruct_from_images(
                image_paths,
                job_root / "reconstruction",
                resolution=256,
                object_id=f"studio_upload_{job_id}",
                dataset_name="Studio upload",
                proposal_filename="image_only_proposal.obj",
            )
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            raise DatasetError(f"Image-only reconstruction failed: {exc}") from exc

        proposal = record["proposal"]
        return jsonify(
            {
                "job_id": job_id,
                "model": {
                    "id": f"generated-{job_id}",
                    "title": "Generated image-only model",
                    "format": "OBJ",
                    "filename": model_path.name,
                    "size_bytes": model_path.stat().st_size,
                    "vertices": proposal["vertices"],
                    "triangles": proposal["faces"],
                    "watertight": proposal["watertight"],
                    "model_url": f"/api/generated-models/{job_id}/model",
                    "download_url": f"/api/generated-models/{job_id}/model?download=1",
                },
                "claims": record["claims"],
            }
        )

    @app.get("/api/generated-models/<job_id>/model")
    def generated_model(job_id: str):
        if re.fullmatch(r"[0-9a-f]{32}", job_id) is None:
            raise AssetNotFoundError("Unknown generated model")
        path = (
            generated_root
            / job_id
            / "reconstruction"
            / "image_only_proposal.obj"
        ).resolve()
        if generated_root.resolve() not in path.parents or not path.is_file():
            raise AssetNotFoundError("Unknown generated model")
        return send_file(
            path,
            as_attachment=request.args.get("download") == "1",
            conditional=True,
        )

    @app.get("/api/reviewer-worklist")
    def reviewer_worklist():
        return jsonify(repository.reviewer_worklist())

    @app.get("/api/capabilities")
    def capabilities():
        return jsonify(repository.capabilities())

    @app.get("/api/phase4/graphs/<graph_name>")
    def phase4_graph(graph_name: str):
        return jsonify(repository.phase4_graph(graph_name))

    @app.post("/api/phase4/graphs/<graph_name>/edit-preview")
    def phase4_edit_preview(graph_name: str):
        return jsonify(
            repository.preview_phase4_edit(
                graph_name, request.get_json(silent=True) or {}
            )
        )

    @app.get("/api/phase4/stl1-candidates")
    def phase4_candidate_review_summary():
        return jsonify(repository.phase4_candidate_review_summary())

    @app.get("/api/phase4/stl1-candidates/<object_id>")
    def phase4_candidate_review_object(object_id: str):
        return jsonify(repository.phase4_candidate_review_object(object_id))

    @app.get("/api/phase4/stl1-candidates/<object_id>/<view>/<kind>")
    def phase4_candidate_asset(object_id: str, view: str, kind: str):
        return send_file(
            repository.phase4_candidate_asset(
                object_id, view, kind, request.args.get("candidate_id")
            ),
            conditional=True,
        )

    @app.post("/api/phase4/stl1-candidates/review")
    def phase4_candidate_review():
        return jsonify(
            repository.review_phase4_candidate(
                request.get_json(silent=True) or {}
            )
        )

    @app.post("/api/phase4/stl1-candidates/comment")
    def phase4_candidate_comment():
        return jsonify(repository.comment_phase4_candidate(request.get_json(silent=True) or {}))

    @app.get("/api/phase1/review")
    def phase1_review():
        return jsonify(repository.phase1_review_manifest())

    @app.get("/api/phase1/assets/<view>/<kind>")
    def phase1_asset(view: str, kind: str):
        return send_file(
            repository.phase1_review_asset(view, kind, request.args.get("label")),
            conditional=True,
        )

    @app.post("/api/phase1/reviews/label")
    def phase1_label_review():
        return jsonify(repository.review_phase1_label(request.get_json(silent=True) or {}))

    @app.post("/api/phase1/reviews/identity")
    def phase1_identity_review():
        return jsonify(
            repository.review_phase1_identity(request.get_json(silent=True) or {})
        )

    @app.post("/api/phase1/reviews/camera")
    def phase1_camera_review():
        return jsonify(
            repository.review_phase1_camera(request.get_json(silent=True) or {})
        )

    @app.post("/api/phase1/reviews/comment")
    def phase1_review_comment():
        return jsonify(repository.comment_phase1_review(request.get_json(silent=True) or {}))

    @app.get("/api/phase1/stl1/review")
    def stl1_phase1_review_summary():
        return jsonify(repository.stl1_phase1_review_summary())

    @app.get("/api/phase1/stl1/review/<object_id>")
    def stl1_phase1_review_object(object_id: str):
        return jsonify(repository.stl1_phase1_review_object(object_id))

    @app.get("/api/phase1/stl1/assets/<object_id>/<view>/<evidence_type>")
    def stl1_phase1_review_asset(
        object_id: str, view: str, evidence_type: str
    ):
        return send_file(
            repository.stl1_phase1_review_asset(
                object_id,
                view,
                evidence_type,
                request.args.get("variant", "active"),
            ),
            conditional=True,
        )

    @app.post("/api/phase1/stl1/reviews/evidence")
    def stl1_phase1_evidence_review():
        return jsonify(
            repository.review_stl1_phase1_evidence(
                request.get_json(silent=True) or {}
            )
        )

    @app.post("/api/phase1/stl1/reviews/object")
    def stl1_phase1_object_review():
        return jsonify(
            repository.review_stl1_phase1_object(
                request.get_json(silent=True) or {}
            )
        )

    @app.post("/api/phase1/stl1/reviews/comment")
    def stl1_phase1_review_comment():
        return jsonify(repository.comment_stl1_phase1_review(request.get_json(silent=True) or {}))

    @app.post("/api/phase1/stl1/reviews/view")
    def stl1_phase1_view_review():
        return jsonify(
            repository.review_stl1_phase1_view(
                request.get_json(silent=True) or {}
            )
        )

    @app.post("/api/phase1/stl1/corrections/silhouette")
    def stl1_phase1_silhouette_correction():
        upload = request.files.get("mask")
        if upload is None:
            raise DatasetError("Corrected silhouette mask is required")
        return jsonify(
            repository.correct_stl1_silhouette(
                request.form.get("object_id", ""),
                request.form.get("view", ""),
                upload.read(),
                request.form.get("reviewer", ""),
                request.form.get("note"),
            )
        )

    @app.get("/api/objects")
    def objects():
        return jsonify({"objects": repository.list_objects(request.args.get("split"))})

    @app.get("/api/objects/<object_id>")
    def object_detail(object_id: str):
        return jsonify(repository.object_detail(object_id))

    @app.get("/api/objects/<object_id>/report")
    def object_report(object_id: str):
        payload = json.dumps(repository.export_object_report(object_id), indent=2)
        return Response(
            payload,
            mimetype="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="{object_id}_evidence_report.json"'
            },
        )

    @app.get("/api/objects/<object_id>/assets/<kind>")
    def object_asset(object_id: str, kind: str):
        path = repository.asset_path(object_id, kind, request.args.get("view"))
        as_attachment = kind in {"phase3_stl", "phase3_3mf"}
        return send_file(path, as_attachment=as_attachment, conditional=True)

    @app.get("/api/artifacts/<name>")
    def global_asset(name: str):
        path = repository.global_asset_path(name)
        mimetype = "text/markdown" if path.suffix.lower() == ".md" else None
        return send_file(path, conditional=True, mimetype=mimetype)

    @app.errorhandler(ObjectNotFoundError)
    @app.errorhandler(AssetNotFoundError)
    def not_found(exc: DatasetError):
        return jsonify({"error": str(exc)}), 404

    @app.errorhandler(DatasetError)
    def bad_request(exc: DatasetError):
        return jsonify({"error": str(exc)}), 400

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host=os.environ.get("IMAGE2CAD_STUDIO_HOST", "127.0.0.1"),
        port=int(os.environ.get("IMAGE2CAD_STUDIO_PORT", "8501")),
        debug=False,
    )
