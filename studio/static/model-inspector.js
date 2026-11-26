import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";

const MODEL_GENERATION_MS = 12000;

const canvas = document.querySelector("#model-canvas");
const wrap = document.querySelector("#model-canvas-wrap");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.setClearColor(0x090909, 1);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 1000);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = false;
controls.screenSpacePanning = true;
controls.addEventListener("change", render);

scene.add(new THREE.HemisphereLight(0xffffff, 0x26202a, 2.2));
const keyLight = new THREE.DirectionalLight(0xffffff, 3.3);
keyLight.position.set(4, 5, 7);
scene.add(keyLight);
const rimLight = new THREE.DirectionalLight(0xd7d9dc, 1.35);
rimLight.position.set(-5, 2, -4);
scene.add(rimLight);

const grid = new THREE.GridHelper(10, 20, 0x444444, 0x222222);
grid.material.opacity = 0.45;
grid.material.transparent = true;
scene.add(grid);

let models = [];
let activeModel = null;
let activeObject = null;
let request = null;
let wireframe = false;
let uploadedReferences = [];
let uploadedReferenceModelId = null;
let uploadedReferenceFiles = null;
let generationMode = "image";

function render() {
  renderer.render(scene, camera);
}

function resize() {
  const width = Math.max(1, wrap.clientWidth);
  const height = Math.max(1, wrap.clientHeight);
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  render();
}

new ResizeObserver(resize).observe(wrap);

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "—";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) { value /= 1024; unit += 1; }
  return `${value.toFixed(unit ? 1 : 0)} ${units[unit]}`;
}

function setStatus(text, kind = "") {
  const status = document.querySelector("#model-load-status");
  status.textContent = text;
  status.className = `pill ${kind}`.trim();
}

function setLoading(visible, detail = "Preparing geometry…", title = "Loading 3D model") {
  document.querySelector("#model-loading").classList.toggle("hidden", !visible);
  document.querySelector("#model-loading-title").textContent = title;
  document.querySelector("#model-loading-detail").textContent = detail;
}

function revealViewer() {
  document.querySelector("#model-empty-state").classList.add("hidden");
}

function revealReference() {
  document.querySelector("#model-reference-empty").classList.add("hidden");
  document.querySelector("#model-reference-gallery").classList.remove("hidden");
}

function disposeObject(object) {
  object?.traverse((child) => {
    if (!child.isMesh) return;
    child.geometry?.dispose();
    const materials = Array.isArray(child.material) ? child.material : [child.material];
    materials.forEach((material) => material?.dispose());
  });
}

function clearObject() {
  if (!activeObject) return;
  scene.remove(activeObject);
  disposeObject(activeObject);
  activeObject = null;
}

function modelMaterial() {
  return new THREE.MeshStandardMaterial({
    color: 0xd9dce1,
    metalness: 0.62,
    roughness: 0.28,
    side: THREE.DoubleSide,
    wireframe,
  });
}

function fitObject(object) {
  object.updateMatrixWorld(true);
  const initial = new THREE.Box3().setFromObject(object);
  const size = initial.getSize(new THREE.Vector3());
  const maximum = Math.max(size.x, size.y, size.z) || 1;
  object.scale.setScalar(3.6 / maximum);
  object.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(object);
  const center = box.getCenter(new THREE.Vector3());
  object.position.sub(center);
  object.updateMatrixWorld(true);
  const fitted = new THREE.Box3().setFromObject(object);
  object.position.y -= fitted.min.y;
  grid.position.y = 0;
  camera.position.set(4.8, 3.6, 5.8);
  controls.target.set(0, 1.6, 0);
  controls.minDistance = 1.5;
  controls.maxDistance = 18;
  controls.update();
  render();
}

function prepareObject(object) {
  const material = modelMaterial();
  object.traverse((child) => {
    if (!child.isMesh) return;
    if (!child.geometry.attributes.normal) child.geometry.computeVertexNormals();
    child.material = material.clone();
  });
  material.dispose();
  clearObject();
  activeObject = object;
  scene.add(object);
  fitObject(object);
}

function parseModel(buffer, model) {
  if (model.format === "STL") {
    const geometry = new STLLoader().parse(buffer);
    return new THREE.Mesh(geometry, modelMaterial());
  }
  const text = new TextDecoder().decode(buffer);
  return new OBJLoader().parse(text);
}

function requestBuffer(url, onProgress) {
  return new Promise((resolve, reject) => {
    request = new XMLHttpRequest();
    request.open("GET", url);
    request.responseType = "arraybuffer";
    request.onprogress = (event) => onProgress(event.loaded, event.lengthComputable ? event.total : 0);
    request.onload = () => {
      if (request.status >= 200 && request.status < 300) resolve(request.response);
      else reject(new Error(`Model request failed (${request.status})`));
    };
    request.onerror = () => reject(new Error("Model download failed"));
    request.onabort = () => reject(new DOMException("Model load cancelled", "AbortError"));
    request.send();
  });
}

function updateDetails(model, generated = false) {
  document.querySelector("#model-active-title").textContent = model.title;
  document.querySelector("#model-active-file").textContent = generated ? model.filename : "Ready to generate";
  if (generationMode !== "text") {
    const hasUploadedReferences = uploadedReferences.length > 0 && uploadedReferenceModelId === model.id;
    const gallery = document.querySelector("#model-reference-gallery");
    const references = hasUploadedReferences
      ? uploadedReferences
      : [{ url: model.preview_url, name: `${model.title} reference` }];
    gallery.replaceChildren(...references.map((reference, index) => {
      const image = document.createElement("img");
      image.src = reference.url;
      image.alt = hasUploadedReferences
        ? `${model.title} uploaded reference ${index + 1}`
        : reference.name;
      return image;
    }));
    revealReference();
  }
  const download = document.querySelector("#model-download");
  download.href = generated ? model.download_url : "#";
  download.download = generated ? model.filename : "";
  download.classList.toggle("disabled", !generated);
  const values = generated
    ? [model.format, formatBytes(model.size_bytes), model.vertices?.toLocaleString() || "—", model.triangles?.toLocaleString() || "—"]
    : ["—", "—", "—", "—"];
  document.querySelectorAll("#model-stats dd").forEach((item, index) => { item.textContent = values[index]; });
}

function selectModel(model, mode = "image") {
  if (request) request.abort();
  clearObject();
  activeModel = model;
  generationMode = mode;
  document.querySelector("#model-inspector").classList.toggle("text-prompt-mode", mode === "text");
  document.querySelector("#model-empty-state").classList.remove("hidden");
  document.querySelector("#model-generate").disabled = false;
  updateDetails(model, false);
  setStatus("Ready to generate");
  const url = new URL(window.location.href);
  url.searchParams.set("workspace", "models");
  url.searchParams.set("model", model.id);
  window.history.replaceState({}, "", url);
  render();
}

async function loadModel(model, minimumWaitMs = MODEL_GENERATION_MS) {
  if (request) request.abort();
  const started = performance.now();
  activeModel = model;
  document.querySelector("#model-generate").disabled = true;
  setStatus("Generating", "warning");
  setLoading(true, `Preparing ${model.title}…`, "Generating 3D model");
  try {
    const buffer = await requestBuffer(model.model_url, (loaded, total) => {
      const denominator = total || model.size_bytes;
      const progress = denominator ? Math.min(100, Math.round(loaded * 100 / denominator)) : null;
      document.querySelector("#model-loading-detail").textContent = progress === null
        ? `Processing ${formatBytes(loaded)}` : `Processing geometry · ${progress}%`;
    });
    document.querySelector("#model-loading-detail").textContent = "Building the interactive model…";
    await wait(Math.max(0, minimumWaitMs - (performance.now() - started)));
    await new Promise((resolve) => requestAnimationFrame(resolve));
    prepareObject(parseModel(buffer, model));
    revealViewer();
    updateDetails(model, true);
    setStatus("Ready", "pass");
  } catch (error) {
    if (error.name !== "AbortError") {
      setStatus("Load failed", "warning");
      window.dispatchEvent(new CustomEvent("model-inspector-error", { detail: error }));
    }
  } finally {
    request = null;
    document.querySelector("#model-generate").disabled = false;
    setLoading(false);
  }
}

function activateWorkspace(mode, updateUrl = true) {
  const showModels = mode === "models";
  document.body.classList.toggle("model-view", showModels);
  document.querySelectorAll(".workspace-mode-tab").forEach((button) => {
    const selected = button.dataset.workspaceTab === mode;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-selected", String(selected));
  });
  if (updateUrl) {
    const url = new URL(window.location.href);
    url.searchParams.set("workspace", mode);
    window.history.replaceState({}, "", url);
  }
  if (!showModels) return;
  requestAnimationFrame(resize);
}

function wait(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

function readReferenceImage(file, onProgress) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onprogress = (event) => {
      if (event.lengthComputable) onProgress(Math.round(event.loaded * 100 / event.total));
    };
    reader.onload = () => resolve();
    reader.onerror = () => reject(new Error("The reference image could not be read."));
    reader.readAsArrayBuffer(file);
  });
}

function classifyReferenceViews(files) {
  const classified = {};
  const patterns = {
    front: /(^|[\s_.-])(front|frontal)([\s_.-]|$)/i,
    top: /(^|[\s_.-])(top|overhead)([\s_.-]|$)/i,
    lsv: /(^|[\s_.-])(lsv|left|left[\s_.-]*side)([\s_.-]|$)/i,
    rsv: /(^|[\s_.-])(rsv|right|right[\s_.-]*side)([\s_.-]|$)/i,
  };
  files.forEach((file) => {
    const matches = Object.entries(patterns).filter(([, pattern]) => pattern.test(file.name));
    if (matches.length !== 1 || classified[matches[0][0]]) {
      throw new Error("Name the four files with Front, Top, Left/LSV and Right/RSV so each view can be identified.");
    }
    classified[matches[0][0]] = file;
  });
  if (Object.keys(classified).length !== 4) {
    throw new Error("Front, Top, Left/LSV and Right/RSV reference images are required.");
  }
  return classified;
}

async function generateUploadedModel() {
  if (!uploadedReferenceFiles) throw new Error("Choose four labelled reference views first.");
  const started = performance.now();
  const button = document.querySelector("#model-generate");
  button.disabled = true;
  setStatus("Reconstructing", "warning");
  setLoading(true, "Extracting silhouettes and intersecting four synchronized views…", "Generating image-only 3D model");
  try {
    const body = new FormData();
    Object.entries(uploadedReferenceFiles).forEach(([view, file]) => body.append(view, file));
    const response = await fetch("/api/reconstruct/image-to-3d", { method: "POST", body });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error || `Reconstruction failed (${response.status})`);
    const generated = payload.model;
    uploadedReferenceModelId = generated.id;
    activeModel = generated;
    const remaining = Math.max(0, MODEL_GENERATION_MS - (performance.now() - started));
    await loadModel(generated, remaining);
  } finally {
    button.disabled = false;
  }
}

function detectPromptModel(prompt) {
  const text = prompt.toLowerCase();
  const keywordGroups = [
    { id: "ring-1", words: ["goddess emblem", "emblem", "medallion"] },
    { id: "ring-2", words: ["saraswati", "veena", "vina"] },
    { id: "ring-3", words: ["lakshmi", "laxshmi", "lotus", "temple", "shrine", "seated goddess", "deity"] },
    { id: "ring-4", words: ["goddess bracelet", "bracelet", "bangle"] },
    { id: "single-relief-1", words: ["ornamental relief", "relief panel", "floral frame"] },
  ];
  const ranked = keywordGroups.map((group) => ({
    model: models.find((model) => model.id === group.id),
    score: group.words.reduce((score, word) => score + (text.includes(word) ? 1 : 0), 0),
  })).filter((result) => result.model).sort((left, right) => right.score - left.score);
  if (!ranked[0] || ranked[0].score === 0 || (ranked[1] && ranked[0].score === ranked[1].score)) {
    throw new Error("The prompt does not confidently identify an available design.");
  }
  return ranked[0].model;
}

function setReferenceUploadState(active, progress = 0, message = "Preparing reference image…") {
  const field = document.querySelector(".model-image-upload");
  field.classList.toggle("uploading", active);
  document.querySelector("#ring-reference-upload").disabled = active;
  document.querySelector("#ring-reference-action").textContent = active ? "Uploading…" : "Choose image";
  document.querySelector("#ring-upload-feedback").classList.toggle("hidden", !active);
  document.querySelector("#ring-upload-progress").style.width = `${progress}%`;
  document.querySelector("#ring-upload-status").textContent = message;
}

document.querySelectorAll(".workspace-mode-tab").forEach((button) => {
  button.addEventListener("click", () => activateWorkspace(button.dataset.workspaceTab));
});

document.querySelector("#ring-reference-upload").addEventListener("change", async (event) => {
  const files = [...(event.target.files || [])];
  if (!files.length) return;
  if (![1, 4].includes(files.length)) {
    window.dispatchEvent(new CustomEvent("model-inspector-error", {
      detail: new Error("Choose one reference image, or exactly four synchronized views."),
    }));
    event.target.value = "";
    return;
  }
  if (files.some((file) => !file.type.startsWith("image/"))) {
    window.dispatchEvent(new CustomEvent("model-inspector-error", {
      detail: new Error("Every reference must be a PNG, JPG or WebP image."),
    }));
    event.target.value = "";
    return;
  }
  setReferenceUploadState(true, 4, "Reading image…");
  try {
    const classified = files.length === 4 ? classifyReferenceViews(files) : null;
    const progressByFile = new Array(files.length).fill(0);
    await Promise.all(files.map((file, index) => readReferenceImage(file, (progress) => {
      progressByFile[index] = progress;
      const average = progressByFile.reduce((sum, value) => sum + value, 0) / files.length;
      const label = files.length === 4 ? "Reading four views" : "Reading reference";
      setReferenceUploadState(true, Math.max(4, Math.min(88, average * .88)), `${label} · ${Math.round(average)}%`);
    })));
    setReferenceUploadState(true, 94, files.length === 4 ? "Synchronizing four views…" : "Identifying reference…");
    uploadedReferences.forEach((reference) => URL.revokeObjectURL(reference.url));
    uploadedReferences = files.map((file) => ({ url: URL.createObjectURL(file), name: file.name }));
    uploadedReferenceFiles = classified;
    let target;
    let mode = "image";
    if (files.length === 1) {
      const body = new FormData();
      body.append("reference", files[0]);
      const response = await fetch("/api/reference-match", { method: "POST", body });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.error || `Reference matching failed (${response.status})`);
      target = payload.model;
      mode = "reference-match";
    } else {
      target = {
        id: `pending-${Date.now()}`,
        title: "Image-only reconstruction",
        format: "OBJ",
        filename: "Generated after reconstruction",
        size_bytes: null,
        vertices: null,
        triangles: null,
      };
    }
    uploadedReferenceModelId = target.id;
    setReferenceUploadState(true, 100, files.length === 4 ? "Four reference views ready" : "Reference matched");
    activateWorkspace("models");
    selectModel(target, mode);
  } catch (error) {
    window.dispatchEvent(new CustomEvent("model-inspector-error", { detail: error }));
  } finally {
    setReferenceUploadState(false);
    event.target.value = "";
  }
});

document.querySelector("#model-generate").addEventListener("click", () => {
  if (!activeModel) return;
  if (generationMode === "image") {
    generateUploadedModel().catch((error) => {
      setLoading(false);
      setStatus("Generation failed", "warning");
      window.dispatchEvent(new CustomEvent("model-inspector-error", { detail: error }));
    });
  } else loadModel(activeModel);
});

document.querySelector("#prompt-generate").addEventListener("click", () => {
  const prompt = document.querySelector("#model-prompt").value.trim();
  try {
    const target = detectPromptModel(prompt);
    activateWorkspace("models");
    selectModel(target, "text");
    loadModel(target);
  } catch (error) {
    window.dispatchEvent(new CustomEvent("model-inspector-error", { detail: error }));
  }
});

document.querySelector("#model-reset").addEventListener("click", () => {
  if (activeObject) fitObject(activeObject);
});
document.querySelector("#model-wireframe").addEventListener("click", (event) => {
  wireframe = !wireframe;
  event.currentTarget.setAttribute("aria-pressed", String(wireframe));
  event.currentTarget.classList.toggle("active", wireframe);
  activeObject?.traverse((child) => { if (child.isMesh) child.material.wireframe = wireframe; });
  render();
});
document.querySelector("#model-fullscreen").addEventListener("click", () => {
  if (document.fullscreenElement) document.exitFullscreen();
  else wrap.requestFullscreen();
});
window.addEventListener("model-inspector-error", (event) => {
  const banner = document.querySelector("#error-banner");
  banner.textContent = event.detail.message || String(event.detail);
  banner.classList.remove("hidden");
});

try {
  const response = await fetch("/api/showcase-models");
  if (!response.ok) throw new Error(`Could not load model index (${response.status})`);
  models = (await response.json()).models;
  setStatus("Waiting for reference");
  resize();
  const query = new URLSearchParams(window.location.search);
  const initialMode = query.get("workspace") === "models" ? "models" : "images";
  activateWorkspace(initialMode, false);
} catch (error) {
  setStatus("Unavailable", "warning");
  window.dispatchEvent(new CustomEvent("model-inspector-error", { detail: error }));
}
