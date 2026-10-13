const state = {
  summary: null,
  objects: [],
  selected: null,
  layer: "source",
  phase1Review: null,
  stl1ReviewSummary: null,
  stl1ReviewObject: null,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function score(value, digits = 3) {
  return typeof value === "number" ? value.toFixed(digits) : "—";
}

function percent(value) {
  return typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "—";
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}

function showError(error) {
  const banner = $("#error-banner");
  banner.textContent = error.message || String(error);
  banner.classList.remove("hidden");
}

function clearError() { $("#error-banner").classList.add("hidden"); }

function renderSummary(summary) {
  const featureContract = summary.phase1.complete_features?.feature_contract || {};
  const metricDepth = summary.phase2.metric_depth_consensus || {};
  $("#dataset-metrics").innerHTML = `
    <article class="metric"><span>Source images</span><strong>${summary.dataset.image_count}</strong><small>${summary.dataset.object_count} objects · immutable image evidence</small></article>
    <article class="metric"><span>Synchronized views</span><strong>${summary.dataset.views?.length ?? "—"}</strong><small>Front · top · isometric · left · right</small></article>
    <article class="metric"><span>Feature matrices</span><strong>${featureContract.stored_array_count_per_view ?? "—"}</strong><small>Stored per view for pixel-level inspection</small></article>
    <article class="metric"><span>Depth backends</span><strong>${metricDepth.available_backends?.length ?? 0}</strong><small>Consensus and disagreement evidence</small></article>`;
  const governance = summary.governance;
  $("#governance-summary").innerHTML = `
    <article class="metric"><span>Capture contract</span><strong>Tier ${escapeHtml(governance.capture_tier)}</strong><small>Five-view exploratory research</small></article>
    <article class="metric"><span>Evidence class</span><strong class="compact-value">${escapeHtml(governance.evidence_class)}</strong><small>Implemented and measured in-project</small></article>
    <article class="metric"><span>Decision</span><strong class="compact-value">Research only</strong><small>No metric or production claim</small></article>
    <article class="metric"><span>Authoritative output</span><strong class="compact-value">STEP / B-rep</strong><small>Not available for STL-1 objects</small></article>`;

  const phase0 = summary.phase0;
  const commit = phase0.source?.commit || "unknown";
  const torch = phase0.runtime?.torch || {};
  $("#phase0-summary").innerHTML = `
    <article class="metric"><span>Phase 0 status</span><strong class="compact-value">${escapeHtml(humanize(phase0.status))}</strong><small>CPU/software reproducibility freeze</small></article>
    <article class="metric"><span>Regression suite</span><strong>${phase0.regression?.test_count ?? "—"}</strong><small>${escapeHtml(phase0.regression?.result || "unknown")} in pinned container</small></article>
    <article class="metric"><span>Source commit</span><strong class="compact-value mono">${escapeHtml(commit.slice(0, 12))}</strong><small>Canonical branch: ${escapeHtml(phase0.source?.canonical_branch || "—")}</small></article>
    <article class="metric"><span>Validation runtime</span><strong class="compact-value">PyTorch ${escapeHtml(torch.version || "—")}</strong><small>CUDA available: ${torch.cuda_available ? "yes" : "no"} · ${phase0.dependency_package_count} packages inventoried</small></article>`;
  $("#phase0-limits").innerHTML = (phase0.known_limits || [])
    .map((item) => `<li>${escapeHtml(item)}</li>`).join("");

  const groundTruth = summary.phase1_ground_truth;
  const gate = groundTruth.exit_gate;
  $("#phase1-summary").innerHTML = `
    <article class="metric"><span>Phase 1 gate</span><strong class="compact-value">${escapeHtml(humanize(gate.status))}</strong><small>Human GT complete: ${gate.human_ground_truth_complete ? "yes" : "no"}</small></article>
    <article class="metric"><span>Mask reviews pending</span><strong>${gate.pending_label_review_count}</strong><small>Five views · seven labels per view</small></article>
    <article class="metric"><span>Component IDs pending</span><strong>${gate.pending_component_identity_count}</strong><small>${groundTruth.component_count} stable identities proposed</small></article>
    <article class="metric"><span>Scale and cameras</span><strong class="compact-value">${gate.metric_scale_available ? "Metric scale recorded" : "Relative image scale"}</strong><small>${gate.pending_camera_record_count} view semantics pending · metric scale ${gate.physical_scale_required ? "required" : "optional"}</small></article>`;
  $("#phase1-authority").textContent = groundTruth.authority_policy;
  $("#phase1-depth-status").innerHTML = `
    <div class="key-value"><span>Model</span><strong>Depth Anything V2 Small</strong></div>
    <div class="key-value"><span>Status</span><strong class="bad">${escapeHtml(humanize(groundTruth.depth.status || "not run"))}</strong></div>
    <div class="key-value"><span>Metric depth</span><strong class="bad">${groundTruth.depth.metric ? "Yes" : "No"}</strong></div>
    <div class="key-value"><span>Role</span><strong>Proposal and uncertainty evidence</strong></div>`;

  const validation = groundTruth.depth_validation;
  const pixels = groundTruth.pixel_features;
  $("#phase1-deep-summary").innerHTML = `
    <article class="metric"><span>Repeated depth runs</span><strong>${validation.runtime?.repeat_count ?? "—"}</strong><small>All identical: ${validation.checks?.all_three_repeats_identical ? "yes" : "no"}</small></article>
    <article class="metric"><span>Front object bias</span><strong>${score(Math.abs(validation.front_raw_object_top_bottom_delta), 4)}</strong><small>Regularized: ${score(Math.abs(validation.front_regularized_object_top_bottom_delta), 4)}</small></article>
    <article class="metric"><span>Front stone bias</span><strong>${score(Math.abs(validation.front_raw_stone_top_bottom_delta), 4)}</strong><small>Regularized: ${score(Math.abs(validation.front_regularized_stone_top_bottom_delta), 4)}</small></article>
    <article class="metric"><span>Pixel reconstruction</span><strong class="compact-value">${pixels.reconstruction_contract?.pixel_exact_roundtrip ? "Exact" : "Failed"}</strong><small>${pixels.view_count} views · metric depth ${pixels.depth_contract?.metric_depth_available ? "available" : "blocked"}</small></article>`;

  const datasetDepth = summary.phase1.depth_anything_v2;
  const depthRuntime = datasetDepth.runtime || {};
  const depthCoverage = datasetDepth.coverage || {};
  const depthOverall = datasetDepth.metrics?.overall || {};
  $("#dataset-depth-summary").innerHTML = `
    <article class="metric"><span>Coverage</span><strong>${depthCoverage.view_count ?? "—"}</strong><small>${depthCoverage.object_count ?? "—"} objects · five views each</small></article>
    <article class="metric"><span>GPU runtime</span><strong>${score(depthRuntime.total_seconds, 2)}s</strong><small>${score(depthRuntime.mean_seconds_per_view, 3)} seconds per view</small></article>
    <article class="metric"><span>Mean flip error</span><strong>${score(depthOverall.mean, 4)}</strong><small>Diagnostic only · lower is more self-consistent</small></article>
    <article class="metric"><span>Phase 1 gate</span><strong class="compact-value">${escapeHtml(humanize(datasetDepth.phase1_gate?.status || "blocked"))}</strong><small>Metric depth: no · human GT: no</small></article>`;

  const complete = summary.phase1.complete_features || {};
  const completeCoverage = complete.coverage || {};
  const completeRuntime = complete.runtime || {};
  const completeChecks = complete.checks || {};
  const completeContract = complete.feature_contract || {};
  const reconstruction = complete.reconstruction_contract || {};
  const bias = complete.depth_validation?.top_bottom_absolute_bias_by_view?.front || {};
  $("#dataset-complete-summary").innerHTML = `
    <article class="metric"><span>Complete feature matrices</span><strong>${completeContract.stored_array_count_per_view ?? "—"}</strong><small>Per view · ${completeCoverage.view_count ?? "—"} views processed</small></article>
    <article class="metric"><span>Repeated depth runs</span><strong>${completeRuntime.repeat_count ?? "—"}</strong><small>All identical: ${completeChecks.all_three_repeats_identical ? "yes" : "no"}</small></article>
    <article class="metric"><span>Source reconstruction</span><strong class="compact-value">${reconstruction.source_pixel_roundtrip_exact ? "Pixel exact" : "Failed"}</strong><small>Normalized pixels retained: ${reconstruction.normalized_pixels_retained_exactly ? "yes" : "no"}</small></article>
    <article class="metric"><span>Front vertical depth delta</span><strong>${score(bias.raw?.mean, 4)}</strong><small>Flip ensemble: ${score(bias.regularized?.mean, 4)} · diagnostic, not correction</small></article>`;

  const stl1Review = summary.stl1_phase1_review || {};
  const reviewGate = stl1Review.gate || {};
  $("#stl1-review-summary").innerHTML = `
    <article class="metric"><span>STL-1 review gate</span><strong class="compact-value">${escapeHtml(humanize(reviewGate.status || "blocked"))}</strong><small>${reviewGate.object_count ?? "—"} rings · ${reviewGate.view_count ?? "—"} views</small></article>
    <article class="metric"><span>Evidence approved</span><strong>${reviewGate.approved_evidence_count ?? 0}</strong><small>of ${reviewGate.evidence_review_count ?? "—"} human decisions</small></article>
    <article class="metric"><span>Evidence pending</span><strong>${reviewGate.pending_evidence_count ?? "—"}</strong><small>Seven review categories per view</small></article>
    <article class="metric"><span>Object inventories pending</span><strong>${reviewGate.pending_object_review_count ?? "—"}</strong><small>Cross-view stones, prongs and relief</small></article>`;

  const external = summary.external_ring_datasets || {};
  const externalMetrics = external.metrics?.syr_phase2_refined_vs_human_mask || {};
  $("#external-ring-summary").innerHTML = `
    <article class="metric"><span>External images</span><strong>${external.sample_count ?? "—"}</strong><small>${external.source_count ?? "—"} provenance-tracked sources</small></article>
    <article class="metric"><span>Human-mask baseline</span><strong>${score(externalMetrics.iou)}</strong><small>Mean refined IoU on ${external.ground_truth_mask_count ?? 0} labelled images</small></article>
    <article class="metric"><span>Strict masks approved</span><strong>${external.approved_mask_count ?? 0}</strong><small>IoU ≥ ${score(external.mask_exit_gate?.iou, 2)} and boundary F1 ≥ ${score(external.mask_exit_gate?.boundary_f1_2px, 2)}</small></article>
    <article class="metric"><span>Review required</span><strong>${external.review_required_mask_count ?? "—"}</strong><small>${escapeHtml(humanize(external.status || "unavailable"))} · no metric/CAD claim</small></article>`;

  const phase2 = summary.phase2 || {};
  const phase2Method = phase2.method || {};
  const phase2Machine = phase2.machine_review || {};
  const phase2Claims = phase2Machine.claims || {};
  const carryover = phase2.ring01_carryover || {};
  const openIssues = carryover.open_issues || [];
  $("#phase2-paired-summary").innerHTML = `
    <article class="metric"><span>Image-only Phase 2</span><strong class="compact-value">${escapeHtml(humanize(phase2.status || "not started"))}</strong><small>No CAD, scans, dimensions or calibration used</small></article>
    <article class="metric"><span>Machine audit</span><strong class="compact-value">${phase2Claims.machine_evidence_validated ? "Validated" : "Blocked"}</strong><small>Human review performed: ${phase2Claims.human_review_performed ? "yes" : "no"}</small></article>
    <article class="metric"><span>Corrected pixels</span><strong>${phase2Method.correction_pixel_count ?? 0}</strong><small>Topology-inferred · pending human review</small></article>
    <article class="metric"><span>Inner-boundary consistency</span><strong>${score(phase2Method.corrected_opening_ellipse_iou, 4)}</strong><small>Before: ${score(phase2Method.baseline_opening_ellipse_iou, 4)}</small></article>
    <article class="metric"><span>Ring01 carry-over</span><strong>${openIssues.length}</strong><small>${openIssues.length ? "Correction proposed" : "No recorded issue"}</small></article>`;
  const metricConsensus = phase2.metric_depth_consensus || {};
  const metricSummary = metricConsensus.summary || {};
  const metricAlignment = metricConsensus.geometry_alignment || {};
  const completeMetricBackends = Object.values(metricConsensus.backend_coverage || {})
    .filter((item) => item.coverage_complete).length;
  $("#metric-depth-consensus-summary").innerHTML = `
    <article class="metric"><span>Completed backends</span><strong>${completeMetricBackends}</strong><small>${metricConsensus.available_backends?.length ?? 0} available · four requested</small></article>
    <article class="metric"><span>Selected views</span><strong>${metricConsensus.sample_count ?? "—"}</strong><small>Ring01 plus train, validation and test objects</small></article>
    <article class="metric"><span>Mean stability</span><strong>${score(metricSummary.mean_cross_model_stability, 4)}</strong><small>Cross-model agreement only · not depth accuracy</small></article>
    <article class="metric"><span>Cross-view alignment</span><strong class="compact-value">${metricAlignment.cross_view_metric_alignment_complete ? "Complete" : "Blocked"}</strong><small>VGGT / MASt3R evidence not available</small></article>`;
  const depthLegend = metricConsensus.visualization?.depth_color_legend || {};
  const disagreementLegend = metricConsensus.visualization?.disagreement_color_legend || {};
  $("#metric-depth-color-convention").textContent =
    `Depth and consensus: red = ${depthLegend.red || "near"}, blue = ${depthLegend.blue || "far"}. ` +
    `Disagreement: blue = ${(disagreementLegend.blue || "low disagreement").replaceAll("_", " ")}, ` +
    `red = ${(disagreementLegend.red || "high disagreement").replaceAll("_", " ")}.`;
  const normalEvidence = phase2.surface_normal_evidence || {};
  const normalSummary = normalEvidence.summary || {};
  const dsine = normalEvidence.backends?.dsine || {};
  const dsineNote = dsine.status === "blocked_license_public_deployment"
    ? "Official licence blocks public/commercial integration without separate permission"
    : "No normal ground truth or calibrated cross-view frame";
  $("#surface-normal-summary").innerHTML = `
    <article class="metric"><span>Normal-map views</span><strong>${normalEvidence.sample_count ?? "—"}</strong><small>${normalEvidence.object_count ?? "—"} objects · direct and depth-derived evidence</small></article>
    <article class="metric"><span>Depth-proxy disagreement</span><strong>${score(normalSummary.mean_depth_proxy_disagreement_deg, 2)}°</strong><small>Mean angular disagreement · consistency only</small></article>
    <article class="metric"><span>Direct / depth delta</span><strong>${score(normalSummary.mean_direct_vs_depth_consensus_deg, 2)}°</strong><small>Metric3D direct normal versus depth consensus</small></article>
    <article class="metric"><span>Independent comparator</span><strong class="compact-value">DSINE ${escapeHtml(humanize(dsine.status || "not run"))}</strong><small>${escapeHtml(dsineNote)}</small></article>`;
  $("#phase2-blockers").innerHTML = (phase2.limitations || [])
    .map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  $("#ring01-carryover-note").textContent = carryover.overall_observation || "No carry-over note recorded.";
}

const PHASE1_VIEWS = ["front", "side", "top", "angled", "back"];
const PHASE1_LABELS = ["jewelry", "metal", "shank", "stone_visible", "setting", "prongs", "negative_space"];

function phase1AssetUrl(view, kind, label = null) {
  const query = label ? `?label=${encodeURIComponent(label)}` : "";
  return `/api/phase1/assets/${encodeURIComponent(view)}/${encodeURIComponent(kind)}${query}`;
}

function selectedPhase1Review() {
  const manifest = state.phase1Review;
  if (!manifest) return;
  const view = $("#phase1-review-view").value;
  const label = $("#phase1-review-label").value;
  const componentId = $("#phase1-component").value;
  const cameraView = $("#phase1-camera-view").value;
  $("#phase1-source-preview").src = phase1AssetUrl(view, "source");
  $("#phase1-proposal-preview").src = phase1AssetUrl(view, "proposal", label);
  const labelReview = manifest.views[view].reviews[label];
  $("#phase1-label-current").textContent = `Current: ${humanize(labelReview.decision)}${labelReview.reviewer ? ` · ${labelReview.reviewer}` : ""}`;
  const component = manifest.component_catalog.find((item) => item.component_id === componentId);
  $("#phase1-component-current").textContent = `Current: ${humanize(component.review.decision)} · ${humanize(component.class)}`;
  const camera = manifest.camera_records[cameraView];
  $("#phase1-camera-current").textContent = `Current: ${humanize(camera.review.decision)} · ${humanize(camera.projection_model)}`;
}

function renderPhase1Review(manifest) {
  state.phase1Review = manifest;
  const populate = (selector, values, labeler = humanize) => {
    const select = $(selector);
    const previous = select.value;
    select.innerHTML = values.map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(labeler(value))}</option>`).join("");
    if (values.includes(previous)) select.value = previous;
  };
  populate("#phase1-review-view", PHASE1_VIEWS);
  populate("#phase1-camera-view", PHASE1_VIEWS);
  populate("#phase1-review-label", PHASE1_LABELS);
  populate("#phase1-component", manifest.component_catalog.map((item) => item.component_id), (value) => value);
  const gate = manifest.exit_gate;
  const pending = gate.pending_label_reviews.length + gate.pending_component_identities.length + gate.pending_camera_records.length;
  const status = $("#phase1-review-status");
  status.textContent = gate.status === "pass" ? "Image GT complete" : `${pending} reviews pending`;
  status.className = `pill ${gate.status === "pass" ? "pass" : "warning"}`;
  selectedPhase1Review();
}

async function loadPhase1Review() {
  renderPhase1Review(await api("/api/phase1/review"));
}

function phase1ReviewerPayload() {
  const reviewer = $("#phase1-reviewer").value.trim();
  if (!reviewer) throw new Error("Enter the reviewer name before recording a decision.");
  return { reviewer, note: $("#phase1-review-note").value.trim() || null };
}

async function submitPhase1(path, payload) {
  clearError();
  await api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  state.summary = await api("/api/summary");
  renderSummary(state.summary);
  await loadPhase1Review();
}

const STL1_VIEWS = ["front", "top", "iso", "lsv", "rsv"];

function stl1ReviewAssetUrl(objectId, view, evidenceType, variant = "active") {
  return `/api/phase1/stl1/assets/${encodeURIComponent(objectId)}/${encodeURIComponent(view)}/${encodeURIComponent(evidenceType)}?variant=${encodeURIComponent(variant)}`;
}

function stl1ReviewerPayload() {
  const reviewer = $("#stl1-reviewer").value.trim();
  if (!reviewer) throw new Error("Enter your reviewer name before recording a decision.");
  return { reviewer, note: $("#stl1-review-note").value.trim() || null };
}

function renderStl1ReviewObject(payload) {
  state.stl1ReviewObject = payload;
  const record = payload.object;
  const definitions = payload.evidence_types;
  const populate = (selector, values, labeler = humanize) => {
    const select = $(selector);
    const previous = select.value;
    select.innerHTML = values.map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(labeler(value))}</option>`).join("");
    if (values.includes(previous)) select.value = previous;
  };
  populate("#stl1-review-view", STL1_VIEWS);
  populate("#stl1-review-evidence", Object.keys(definitions), (value) => definitions[value].title);
  $("#stl1-review-title").textContent = `${record.source_name} · ${record.object_id}`;
  $("#stl1-review-authority").textContent = payload.authority_policy;
  const gate = record.gate;
  const status = $("#stl1-object-review-status");
  status.textContent = gate.status === "pass" ? "Review complete" : `${gate.pending_evidence_count + Number(gate.object_review_pending)} items pending`;
  status.className = `pill ${gate.status === "pass" ? "pass" : "warning"}`;

  const objectReview = record.object_review;
  const inventory = objectReview.component_inventory || {};
  $("#stl1-has-stones").value = inventory.has_stones || "unknown";
  $("#stl1-stone-count").value = inventory.estimated_visible_stone_count ?? "";
  $("#stl1-has-prongs").value = inventory.has_prongs || "unknown";
  $("#stl1-has-relief").value = inventory.has_sculptural_relief || "unknown";
  $("#stl1-cross-view").value = inventory.cross_view_identity_consistent || "unknown";
  $("#stl1-inventory-current").textContent = `Current: ${humanize(objectReview.decision)}${objectReview.reviewer ? ` · ${objectReview.reviewer}` : ""}`;
  const inventoryStatus = $("#stl1-inventory-status");
  inventoryStatus.textContent = humanize(objectReview.decision);
  inventoryStatus.className = `pill ${objectReview.decision === "approved" ? "pass" : "warning"}`;
  updateStl1ReviewSelection();
}

function updateStl1ReviewSelection() {
  const payload = state.stl1ReviewObject;
  if (!payload) return;
  const record = payload.object;
  const view = $("#stl1-review-view").value;
  const evidenceType = $("#stl1-review-evidence").value;
  const review = record.views[view].evidence[evidenceType];
  const definition = payload.evidence_types[evidenceType];
  $("#stl1-review-source").src = stl1ReviewAssetUrl(record.object_id, view, "normalization");
  $("#stl1-review-evidence-image").src = stl1ReviewAssetUrl(record.object_id, view, evidenceType);
  $("#stl1-review-question").textContent = definition.question;
  $("#stl1-review-evidence-caption").textContent = review.corrected_path ? "Corrected active evidence" : "Machine proposal";
  $("#stl1-review-current").textContent = `Current: ${humanize(review.decision)}${review.reviewer ? ` · ${review.reviewer}` : ""}${review.corrected_path ? " · corrected mask" : ""}`;
  $("#stl1-mask-correction").classList.toggle("hidden", evidenceType !== "silhouette");
}

async function loadStl1ReviewSummary() {
  state.stl1ReviewSummary = await api("/api/phase1/stl1/review");
}

async function loadStl1ReviewObject(objectId) {
  renderStl1ReviewObject(await api(`/api/phase1/stl1/review/${encodeURIComponent(objectId)}`));
}

async function submitStl1Evidence(decision) {
  const common = stl1ReviewerPayload();
  const payload = await api("/api/phase1/stl1/reviews/evidence", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ...common,
      object_id: state.selected.object_id,
      view: $("#stl1-review-view").value,
      evidence_type: $("#stl1-review-evidence").value,
      decision,
    }),
  });
  renderStl1ReviewObject(payload);
  await refreshStl1ReviewState();
}

async function approveStl1View() {
  const common = stl1ReviewerPayload();
  const payload = await api("/api/phase1/stl1/reviews/view", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ...common,
      object_id: state.selected.object_id,
      view: $("#stl1-review-view").value,
      decision: "approved",
    }),
  });
  renderStl1ReviewObject(payload);
  await refreshStl1ReviewState();
}

async function submitStl1ObjectReview(decision) {
  const common = stl1ReviewerPayload();
  const stoneCount = $("#stl1-stone-count").value.trim();
  const payload = await api("/api/phase1/stl1/reviews/object", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ...common,
      object_id: state.selected.object_id,
      decision,
      component_inventory: {
        has_stones: $("#stl1-has-stones").value,
        estimated_visible_stone_count: stoneCount === "" ? null : Number(stoneCount),
        has_prongs: $("#stl1-has-prongs").value,
        has_sculptural_relief: $("#stl1-has-relief").value,
        cross_view_identity_consistent: $("#stl1-cross-view").value,
      },
    }),
  });
  renderStl1ReviewObject(payload);
  await refreshStl1ReviewState();
}

async function uploadStl1CorrectedMask() {
  const common = stl1ReviewerPayload();
  const file = $("#stl1-corrected-mask").files[0];
  if (!file) throw new Error("Choose a corrected binary PNG first.");
  const form = new FormData();
  form.append("mask", file);
  form.append("object_id", state.selected.object_id);
  form.append("view", $("#stl1-review-view").value);
  form.append("reviewer", common.reviewer);
  if (common.note) form.append("note", common.note);
  const payload = await api("/api/phase1/stl1/corrections/silhouette", {
    method: "POST",
    body: form,
  });
  $("#stl1-corrected-mask").value = "";
  renderStl1ReviewObject(payload);
  await refreshStl1ReviewState();
}

async function refreshStl1ReviewState() {
  await loadStl1ReviewSummary();
  state.summary = await api("/api/summary");
  renderSummary(state.summary);
  await loadObjects($("#split-filter").value, state.selected?.object_id);
}

async function selectNextStl1Pending() {
  const payload = state.stl1ReviewObject;
  if (!payload) return;
  const currentView = $("#stl1-review-view").value;
  const currentEvidence = $("#stl1-review-evidence").value;
  const kinds = Object.keys(payload.evidence_types);
  const tasks = STL1_VIEWS.flatMap((view) => kinds.map((evidenceType) => ({ view, evidenceType })));
  const currentIndex = tasks.findIndex((item) => item.view === currentView && item.evidenceType === currentEvidence);
  const ordered = tasks.slice(currentIndex + 1).concat(tasks.slice(0, currentIndex + 1));
  const next = ordered.find((item) => payload.object.views[item.view].evidence[item.evidenceType].decision !== "approved");
  if (next) {
    $("#stl1-review-view").value = next.view;
    $("#stl1-review-evidence").value = next.evidenceType;
    updateStl1ReviewSelection();
    return;
  }
  await loadStl1ReviewSummary();
  const nextObject = state.stl1ReviewSummary.objects.find((item) => item.pending_evidence_count > 0 || item.object_review_pending);
  if (nextObject) await selectObject(nextObject.object_id);
}

function renderObjectList() {
  $("#object-count").textContent = state.objects.length;
  $("#object-list").innerHTML = state.objects.map((item) => `
    <button class="object-button ${state.selected?.object_id === item.object_id ? "active" : ""}" data-object-id="${escapeHtml(item.object_id)}">
      <strong>${escapeHtml(item.source_name)}</strong><span>${escapeHtml(item.split)} · ${item.view_count} views</span>
      <span class="object-score">${item.phase1_pending_review_count ?? "—"} pending</span>
    </button>`).join("");
  $$(".object-button").forEach((button) => button.addEventListener("click", () => selectObject(button.dataset.objectId)));
}

function assetUrl(objectId, kind, view = null) {
  const query = view ? `?view=${encodeURIComponent(view)}` : "";
  return `/api/objects/${encodeURIComponent(objectId)}/assets/${kind}${query}`;
}

function renderViews(detail) {
  $("#view-grid").innerHTML = detail.view_order.map((view) => {
    const data = detail.views[view];
    const complete = data.complete_features || {};
    return `<article class="view-card">
      <img src="${assetUrl(detail.object_id, state.layer, view)}" alt="${escapeHtml(data.label)} ${escapeHtml(state.layer)}" loading="lazy">
      <div class="view-card-footer"><strong>${escapeHtml(data.label)}</strong><span>Flip H ${score(complete.horizontal_flip_error_over_depth_span, 4)} · V ${score(complete.vertical_flip_error_over_depth_span, 4)} · Pseudo IoU ${score(data.metrics?.iou)}</span></div>
    </article>`;
  }).join("");
}

function renderMetadata(detail) {
  $("#metadata-rows").innerHTML = detail.view_order.map((view) => {
    const data = detail.views[view];
    const dimensions = data.source_size_wh ? `${data.source_size_wh[0]} × ${data.source_size_wh[1]}` : "—";
    return `<tr><td>${escapeHtml(data.label)}</td><td>${dimensions}</td><td>${score(data.mask_occupancy)}</td><td>${score(data.metrics?.iou)}</td><td>${score(data.metrics?.boundary_f1_2px)}</td></tr>`;
  }).join("");

  const supervisionLabels = {
    jewelry_silhouette: "Jewellery silhouette",
    edge_map: "Edge map",
    individual_stones: "Individual stones",
    metal_components: "Metal components",
    prongs: "Prongs",
    stone_seats: "Stone seats",
    cavities: "Cavities",
    camera_intrinsics: "Camera intrinsics",
    camera_extrinsics: "Camera extrinsics",
    physical_scale: "Physical scale",
    mesh_target: "Mesh target",
    cad_target: "CAD target",
  };
  $("#supervision-list").innerHTML = Object.entries(supervisionLabels).map(([key, label]) => {
    const value = detail.supervision[key];
    const available = value !== null && value !== undefined;
    const display = available ? String(value).replaceAll("_", " ") : "Unavailable";
    return `<div class="key-value"><span>${label}</span><strong class="${available ? "good" : "bad"}">${escapeHtml(display)}</strong></div>`;
  }).join("");

  $("#feature-rows").innerHTML = detail.view_order.map((view) => {
    const data = detail.views[view];
    const features = data.features || {};
    return `<tr>
      <td>${escapeHtml(data.label)}</td>
      <td>${percent(features.foreground_fraction)}</td>
      <td>${percent(features.edge_fraction)}</td>
      <td>${features.contour_count ?? "—"}</td>
      <td>${features.hole_count ?? "—"}</td>
      <td>${score(features.horizontal_symmetry_iou)}</td>
    </tr>`;
  }).join("");

  $("#depth-rows").innerHTML = detail.view_order.map((view) => {
    const data = detail.views[view];
    const depth = data.depth || {};
    return `<tr>
      <td>${escapeHtml(data.label)}</td>
      <td>${score(depth.mean_flip_error_over_depth_span, 4)}</td>
      <td>${score(depth.p95_flip_error_over_depth_span, 4)}</td>
      <td>${depth.metric ? "Yes" : "No"}</td>
      <td>${depth.cross_view_aligned ? "Yes" : "No"}</td>
    </tr>`;
  }).join("");
}

function humanize(value) {
  return String(value).replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function renderBible(detail) {
  const bible = detail.bible;
  $("#bible-gates").innerHTML = Object.entries(bible.gates).map(([name, passed]) => `
    <div class="key-value"><span>${escapeHtml(humanize(name))}</span><strong class="${passed ? "good" : "bad"}">${passed ? "Pass" : "Blocked"}</strong></div>`).join("");
  $("#evidence-records").innerHTML = bible.evidence_records.map((record) => `
    <div class="key-value evidence-row">
      <span>${escapeHtml(record.stage)}</span>
      <strong>${escapeHtml(record.state)}</strong>
      <small>${escapeHtml(record.evidence_class)}</small>
    </div>`).join("");
  $("#artifact-authority").textContent = bible.artifact_authority;
}

function renderPhase3(detail) {
  const phase3 = detail.phase3;
  const preview = $("#phase3-preview");
  const metrics = $("#phase3-metrics");
  const geometry = detail.geometry_evidence || {};
  const database = geometry.database || {};
  const globalGeometry = state.summary?.phase3?.geometry_benchmark || {};
  const vggt = globalGeometry.backends?.vggt || {};
  const readiness = globalGeometry.cross_view_readiness || {};
  const readinessSummary = readiness.summary || {};
  const runtime = readiness.runtime || {};
  const candidates = readiness.candidates || {};
  const experiment = globalGeometry.ring01_vggt_low_vram || {};
  const experimentRun = experiment.experiment || {};
  const experimentReview = experiment.machine_visual_review || {};
  const multiring = globalGeometry.vggt_multiring_low_vram || {};
  const multiringObjects = multiring.objects || [];
  const multiringRefinement = globalGeometry.vggt_multiring_refinement || {};
  const refinementCohort = multiringRefinement.cohort || {};
  const semanticPriors = globalGeometry.semantic_camera_priors || {};
  const semanticPriorCohort = semanticPriors.cohort || {};
  const semanticRefinement = globalGeometry.semantic_camera_refinement || {};
  const semanticRefinementCohort = semanticRefinement.cohort || {};
  $("#phase3-geometry").innerHTML = `
    <div class="key-value"><span>COLMAP object status</span><strong class="${geometry.status === "completed_with_sparse_model" ? "good" : "bad"}">${escapeHtml(humanize(geometry.status || "not run"))}</strong></div>
    <div class="key-value"><span>SIFT keypoints</span><strong>${database.keypoints_rows?.toLocaleString() ?? "—"}</strong></div>
    <div class="key-value"><span>Verified cross-view pairs</span><strong class="${database.two_view_geometries_nonempty_records ? "good" : "bad"}">${database.two_view_geometries_nonempty_records ?? "—"}</strong></div>
    <div class="key-value"><span>Sparse reconstruction</span><strong class="${geometry.sparse_model ? "good" : "bad"}">${geometry.sparse_model ? `${geometry.sparse_model.registered_images} / 5 views` : "Not formed"}</strong></div>
    <div class="key-value"><span>VGGT</span><strong class="${vggt.status === "ready" ? "good" : "bad"}">${escapeHtml(humanize(vggt.status || "not run"))}</strong></div>
    <div class="key-value"><span>Bible gate</span><strong class="bad">Research only</strong></div>`;
  $("#cross-view-readiness").innerHTML = `
    <div class="key-value"><span>Public-product candidates</span><strong>${readinessSummary.public_product_candidate_count ?? "—"} / ${readinessSummary.candidate_count ?? "—"}</strong><small>${escapeHtml((readinessSummary.public_product_candidates || []).map(humanize).join(", ") || "None")}</small></div>
    <div class="key-value"><span>Locally ready</span><strong class="${readinessSummary.locally_ready_count ? "good" : "bad"}">${readinessSummary.locally_ready_count ?? "—"}</strong></div>
    <div class="key-value"><span>Local GPU</span><strong class="${(runtime.gpu_total_gib || 0) >= 8 ? "good" : "bad"}">${escapeHtml(runtime.gpu_name || "Unavailable")} · ${score(runtime.gpu_total_gib, 2)} GiB</strong></div>
    <div class="key-value"><span>VGGT commercial checkpoint</span><strong class="${candidates.vggt?.local_checkpoint_present ? "good" : "bad"}">${candidates.vggt?.local_checkpoint_present ? "Present" : "Required"}</strong></div>
    <div class="key-value"><span>DUSt3R / MASt3R</span><strong class="bad">Research-only licences</strong></div>
    <div class="key-value"><span>Decision</span><strong class="bad">${escapeHtml(humanize(readiness.decision || "blocked"))}</strong></div>`;
  $("#ring01-vggt-status").innerHTML = `
    <div class="key-value"><span>Execution</span><strong class="${experiment.checks?.run_completed ? "good" : "bad"}">${experiment.checks?.run_completed ? "Completed" : "Unavailable"}</strong></div>
    <div class="key-value"><span>Five-view dense output</span><strong class="${experiment.checks?.dense_shapes_valid ? "good" : "bad"}">${experiment.checks?.dense_shapes_valid ? "Generated" : "Failed"}</strong></div>
    <div class="key-value"><span>Resolution</span><strong>${experimentRun.image_size || "—"} px · ${escapeHtml(experimentRun.dtype || "—")}</strong></div>
    <div class="key-value"><span>Peak CUDA memory</span><strong>${experimentRun.peak_cuda_bytes ? `${score(experimentRun.peak_cuda_bytes / (1024 ** 3), 2)} GiB` : "—"}</strong></div>
    <div class="key-value"><span>Fusion inspection</span><strong class="bad">${escapeHtml(humanize(experimentReview.decision || "pending"))}</strong></div>
    <div class="key-value"><span>Authority</span><strong class="bad">Research hypothesis only</strong></div>`;
  const multiringIous = multiringObjects.map((item) => item.mean_leave_one_view_out_iou);
  const meanMultiringIou = multiringIous.length
    ? multiringIous.reduce((total, value) => total + value, 0) / multiringIous.length
    : null;
  $("#vggt-multiring-status").innerHTML = `
    <div class="key-value"><span>Objects tested</span><strong>${multiring.object_count ?? "—"}</strong><small>1 train · 2 held-out test</small></div>
    <div class="key-value"><span>Surface-completion mean IoU</span><strong class="bad">${score(meanMultiringIou, 3)}</strong><small>Target view excluded; measures camera alignment and missing-surface completion</small></div>
    <div class="key-value"><span>Cross-view gate accepted</span><strong class="bad">${multiring.accepted_object_count ?? "—"} / ${multiring.object_count ?? "—"}</strong></div>
    <div class="key-value"><span>Object-centric refinement</span><strong class="bad">${score(refinementCohort.delta, 3)} IoU</strong><small>${escapeHtml(humanize(multiringRefinement.decision || "not run"))}</small></div>
    <div class="key-value"><span>Decision</span><strong class="bad">Rejected as reconstruction geometry</strong></div>
    <div class="key-value"><span>External real-image sets</span><strong class="bad">Single-view only</strong><small>Used for Phase 1/2, not false multi-view fusion</small></div>`;
  $("#semantic-camera-status").innerHTML = `
    <div class="key-value"><span>VGGT optical-axis error</span><strong>${score(semanticPriorCohort.mean_vggt_pairwise_angle_error_deg, 2)}°</strong><small>Mean pairwise error against named-view directions</small></div>
    <div class="key-value"><span>Semantic-axis holdout IoU</span><strong>${score(semanticPriorCohort.mean_semantic_holdout_iou, 3)}</strong><small>Minimum ${score(semanticPriorCohort.minimum_semantic_holdout_iou, 3)}</small></div>
    <div class="key-value"><span>Naive camera snap delta</span><strong class="bad">${score(semanticRefinementCohort.mean_iou_delta, 3)}</strong><small>Original VGGT cameras retained</small></div>
    <div class="key-value"><span>Accepted camera initializations</span><strong class="bad">${semanticPriorCohort.accepted_initialization_count ?? "—"} / ${semanticPriors.object_count ?? "—"}</strong></div>
    <div class="key-value"><span>Interpretation</span><strong class="bad">Visible surfaces incomplete</strong><small>Camera layout is broadly plausible, but it does not complete hidden geometry</small></div>
    <div class="key-value"><span>Authority</span><strong class="bad">Diagnostic only</strong></div>`;
  if (!phase3) {
    preview.removeAttribute("src");
    metrics.innerHTML = "<p>No experimental Phase 3 artifact is available.</p>";
    return;
  }
  preview.src = assetUrl(detail.object_id, "phase3_preview");
  const mesh = phase3.mesh || {};
  const reprojection = phase3.reprojection || {};
  metrics.innerHTML = `
    <div class="phase3-score"><strong>${score(reprojection.mean_iou)}</strong><span>Mean reprojection IoU</span></div>
    <div class="key-value-list">
      <div class="key-value"><span>Watertight</span><strong class="${mesh.watertight ? "good" : "bad"}">${mesh.watertight ? "Pass" : "Fail"}</strong></div>
      <div class="key-value"><span>Connected bodies</span><strong>${mesh.body_count ?? "—"}</strong></div>
      <div class="key-value"><span>Vertices</span><strong>${mesh.vertices?.toLocaleString() ?? "—"}</strong></div>
      <div class="key-value"><span>Faces</span><strong>${mesh.faces?.toLocaleString() ?? "—"}</strong></div>
      <div class="key-value"><span>Camera calibrated</span><strong class="bad">${phase3.camera_calibrated ? "Yes" : "No"}</strong></div>
      <div class="key-value"><span>Physical scale</span><strong class="bad">${phase3.scale?.calibrated ? "Calibrated" : "Display scale only"}</strong></div>
      <div class="key-value"><span>Artifact authority</span><strong class="bad">Derived only</strong></div>
      <div class="key-value"><span>Decision</span><strong class="bad">Research only</strong></div>
    </div>`;
  $("#download-stl").href = assetUrl(detail.object_id, "phase3_stl");
  $("#download-3mf").href = assetUrl(detail.object_id, "phase3_3mf");
}

function renderObject(detail) {
  state.selected = detail;
  $("#object-workspace").classList.remove("hidden");
  $("#object-title").textContent = detail.source_name;
  $("#object-path").textContent = `STL-1 / ${detail.object_id}`;
  const reviewPassed = detail.phase1_review?.gate?.status === "pass";
  $("#object-badges").innerHTML = `<span class="pill ${detail.split === "test" ? "test" : "pass"}">${escapeHtml(detail.split)} split</span><span class="pill ${reviewPassed ? "pass" : "warning"}">Phase 1 ${reviewPassed ? "reviewed" : "pending"}</span><span class="pill warning">Tier A</span><span class="pill warning">Research only</span>`;
  $("#report-download").href = `/api/objects/${encodeURIComponent(detail.object_id)}/report`;
  $("#report-download").classList.remove("disabled");
  $("#phase-sheet").src = assetUrl(detail.object_id, "phase_sheet");
  $("#phase1-depth-review").src = assetUrl(detail.object_id, "phase1_depth_review");
  $("#phase1-complete-audit").src = assetUrl(detail.object_id, "phase1_complete_audit");

  const heldOut = detail.artifacts.held_out_review;
  $("#held-out-card").classList.toggle("hidden", !heldOut);
  if (heldOut) $("#held-out-review").src = assetUrl(detail.object_id, "held_out_review");

  renderViews(detail);
  renderMetadata(detail);
  renderPhase3(detail);
  renderBible(detail);
  renderObjectList();
}

async function selectObject(objectId) {
  clearError();
  try {
    renderObject(await api(`/api/objects/${encodeURIComponent(objectId)}`));
    await loadStl1ReviewObject(objectId);
  } catch (error) { showError(error); }
}

async function loadObjects(split = "all", preferredId = null) {
  const result = await api(`/api/objects?split=${encodeURIComponent(split)}`);
  state.objects = result.objects;
  renderObjectList();
  const target = state.objects.find((item) => item.object_id === preferredId) || state.objects[0];
  if (target) await selectObject(target.object_id);
}

function bindControls() {
  ["#phase1-review-view", "#phase1-review-label", "#phase1-component", "#phase1-camera-view"].forEach((selector) => {
    $(selector).addEventListener("change", selectedPhase1Review);
  });
  $$(".phase1-label-decision").forEach((button) => button.addEventListener("click", () => {
    const common = phase1ReviewerPayload();
    submitPhase1("/api/phase1/reviews/label", {
      ...common,
      view: $("#phase1-review-view").value,
      label: $("#phase1-review-label").value,
      decision: button.dataset.decision,
    }).catch(showError);
  }));
  $$(".phase1-identity-decision").forEach((button) => button.addEventListener("click", () => {
    const common = phase1ReviewerPayload();
    submitPhase1("/api/phase1/reviews/identity", {
      ...common,
      component_id: $("#phase1-component").value,
      decision: button.dataset.decision,
    }).catch(showError);
  }));
  $$(".phase1-camera-decision").forEach((button) => button.addEventListener("click", () => {
    const common = phase1ReviewerPayload();
    submitPhase1("/api/phase1/reviews/camera", {
      ...common,
      view: $("#phase1-camera-view").value,
      projection_model: $("#phase1-projection").value,
      uncertainty: $("#phase1-camera-uncertainty").value.trim(),
      decision: button.dataset.decision,
    }).catch(showError);
  }));
  ["#stl1-review-view", "#stl1-review-evidence"].forEach((selector) => {
    $(selector).addEventListener("change", updateStl1ReviewSelection);
  });
  $$(".stl1-evidence-decision").forEach((button) => button.addEventListener("click", () => {
    clearError();
    submitStl1Evidence(button.dataset.decision).catch(showError);
  }));
  $$(".stl1-object-decision").forEach((button) => button.addEventListener("click", () => {
    clearError();
    submitStl1ObjectReview(button.dataset.decision).catch(showError);
  }));
  $("#stl1-upload-mask").addEventListener("click", () => {
    clearError();
    uploadStl1CorrectedMask().catch(showError);
  });
  $("#stl1-approve-view").addEventListener("click", () => {
    clearError();
    approveStl1View().catch(showError);
  });
  $("#stl1-next-pending").addEventListener("click", () => {
    clearError();
    selectNextStl1Pending().catch(showError);
  });
  $("#split-filter").addEventListener("change", (event) => loadObjects(event.target.value).catch(showError));
  $$("#layer-switcher button").forEach((button) => button.addEventListener("click", () => {
    state.layer = button.dataset.layer;
    $$("#layer-switcher button").forEach((item) => item.classList.toggle("active", item === button));
    if (state.selected) renderViews(state.selected);
  }));
  $$(".tab").forEach((button) => button.addEventListener("click", () => {
    $$(".tab").forEach((item) => item.classList.toggle("active", item === button));
    $$(".tab-panel").forEach((panel) => panel.classList.toggle("active", panel.id === `panel-${button.dataset.panel}`));
  }));
}

async function start() {
  bindControls();
  try {
    state.summary = await api("/api/summary");
    renderSummary(state.summary);
    await loadPhase1Review();
    await loadStl1ReviewSummary();
    await loadObjects("all", "ring_001");
  } catch (error) { showError(error); }
}

document.addEventListener("DOMContentLoaded", start);
