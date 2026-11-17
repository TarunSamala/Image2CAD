import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";

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
let uploadedRingOneReference = null;

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
  document.querySelector("#model-reference-empty").classList.add("hidden");
  document.querySelector("#model-reference-image").classList.remove("hidden");
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

function updateDetails(model) {
  document.querySelector("#model-active-title").textContent = model.title;
  document.querySelector("#model-active-file").textContent = model.filename;
  const referenceUrl = model.id === "ring-1" && uploadedRingOneReference
    ? uploadedRingOneReference : model.preview_url;
  document.querySelector("#model-reference-image").src = referenceUrl;
  document.querySelector("#model-reference-image").alt = `${model.title} isometric reference`;
  document.querySelector("#ring-reference-name").textContent = model.id === "ring-1" && uploadedRingOneReference
    ? "Using your uploaded Ring 1 reference"
    : `Using the saved ${model.title} reference`;
  document.querySelector("#model-download").href = model.download_url;
  document.querySelector("#model-download").download = model.filename;
  document.querySelector("#model-download").classList.remove("disabled");
  const values = [model.format, formatBytes(model.size_bytes), model.vertices?.toLocaleString() || "—", model.triangles?.toLocaleString() || "—"];
  document.querySelectorAll("#model-stats dd").forEach((item, index) => { item.textContent = values[index]; });
  document.querySelectorAll(".showcase-model-button").forEach((button) => button.classList.toggle("active", button.dataset.modelId === model.id));
}

async function loadModel(model) {
  if (request) request.abort();
  activeModel = model;
  revealViewer();
  const url = new URL(window.location.href);
  url.searchParams.set("workspace", "models");
  url.searchParams.set("model", model.id);
  window.history.replaceState({}, "", url);
  updateDetails(model);
  setStatus("Loading", "warning");
  setLoading(true, `Downloading ${formatBytes(model.size_bytes)}…`);
  try {
    const buffer = await requestBuffer(model.model_url, (loaded, total) => {
      const denominator = total || model.size_bytes;
      const progress = denominator ? Math.min(100, Math.round(loaded * 100 / denominator)) : null;
      document.querySelector("#model-loading-detail").textContent = progress === null
        ? `Downloaded ${formatBytes(loaded)}` : `Downloading ${progress}% · ${formatBytes(loaded)}`;
    });
    document.querySelector("#model-loading-detail").textContent = "Building mesh for inspection…";
    await new Promise((resolve) => requestAnimationFrame(resolve));
    prepareObject(parseModel(buffer, model));
    setStatus("Ready", "pass");
  } catch (error) {
    if (error.name !== "AbortError") {
      setStatus("Load failed", "warning");
      window.dispatchEvent(new CustomEvent("model-inspector-error", { detail: error }));
    }
  } finally {
    request = null;
    setLoading(false);
  }
}

function renderModelList() {
  const list = document.querySelector("#showcase-model-list");
  list.replaceChildren(...models.map((model) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "showcase-model-button";
    button.dataset.modelId = model.id;
    const image = document.createElement("img");
    image.src = model.preview_url;
    image.alt = `${model.title} reference`;
    const copy = document.createElement("span");
    const title = document.createElement("strong");
    title.textContent = model.title;
    const meta = document.createElement("small");
    meta.textContent = `${model.format} · ${formatBytes(model.size_bytes)}`;
    copy.append(title, meta);
    button.append(image, copy);
    button.addEventListener("click", () => {
      activateWorkspace("models");
      if (model.id === "ring-1" && !uploadedRingOneReference) {
        document.querySelector("#ring-reference-upload").click();
        return;
      }
      loadModel(model);
    });
    return button;
  }));
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
  const file = event.target.files?.[0];
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    window.dispatchEvent(new CustomEvent("model-inspector-error", {
      detail: new Error("Choose a PNG, JPG or WebP image."),
    }));
    event.target.value = "";
    return;
  }
  const started = performance.now();
  setReferenceUploadState(true, 4, "Reading image…");
  setLoading(true, "Reading and preparing the selected image…", "Preparing reference");
  try {
    await readReferenceImage(file, (progress) => {
      setReferenceUploadState(true, Math.max(4, Math.min(82, progress * .82)), `Reading image · ${progress}%`);
    });
    setReferenceUploadState(true, 88, "Preparing reference preview…");
    await wait(Math.max(0, 2740 - (performance.now() - started)));
    if (uploadedRingOneReference) URL.revokeObjectURL(uploadedRingOneReference);
    uploadedRingOneReference = URL.createObjectURL(file);
    document.querySelector("#ring-reference-name").textContent = file.name;
    setReferenceUploadState(true, 100, "Reference ready");
    await wait(260);
    activateWorkspace("models");
    const ringOne = models.find((model) => model.id === "ring-1");
    if (!ringOne) return;
    if (activeModel?.id === "ring-1" && activeObject) {
      updateDetails(ringOne);
      setLoading(false);
    }
    else loadModel(ringOne);
  } catch (error) {
    setLoading(false);
    window.dispatchEvent(new CustomEvent("model-inspector-error", { detail: error }));
  } finally {
    setReferenceUploadState(false);
    event.target.value = "";
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
  renderModelList();
  setStatus("Waiting for reference");
  resize();
  const query = new URLSearchParams(window.location.search);
  const requested = query.get("model");
  const initialMode = query.get("workspace") === "models" || requested ? "models" : "images";
  activateWorkspace(initialMode, false);
  const initial = models.find((model) => model.id === requested);
  if (initial && initial.id !== "ring-1") await loadModel(initial);
} catch (error) {
  setStatus("Unavailable", "warning");
  window.dispatchEvent(new CustomEvent("model-inspector-error", { detail: error }));
}
