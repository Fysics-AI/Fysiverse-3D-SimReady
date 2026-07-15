import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { KTX2Loader } from "three/addons/loaders/KTX2Loader.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { MeshoptDecoder } from "meshoptimizer";

const DEFAULT_MANIFEST = "";
const CAMERA_OVERRIDE_URL = "./camera_override.json";
const CAMERA_STORAGE_KEY = "fysiverse.pipelineAnimation.camera.v2";
const WORKFLOW_MARKER = "/OOD_workflow/";
const BASIS_TRANSCODER_PATH = "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/libs/basis/";
const GLTF_VISUAL_TO_BLENDER_Z_UP = new THREE.Matrix4().makeRotationX(Math.PI / 2);

const canvas = document.getElementById("viewer");
const sessionSelect = document.getElementById("session-select");
const playButton = document.getElementById("play-button");
const cameraMenuButton = document.getElementById("camera-menu-button");
const speedSelect = document.getElementById("speed-select");
const statusEl = document.getElementById("status");
const frameSlider = document.getElementById("frame-slider");
const frameReadout = document.getElementById("frame-readout");
const stageStrip = document.getElementById("stage-strip");
const cameraPanel = document.getElementById("camera-panel");
const cameraJsonEl = document.getElementById("camera-json");
const closeCameraPanelButton = document.getElementById("close-camera-panel");
const cameraLockInput = document.getElementById("camera-lock");
const applyCameraButton = document.getElementById("apply-camera-button");
const syncCameraButton = document.getElementById("sync-camera-button");
const fitCameraButton = document.getElementById("fit-camera-button");
const copyCameraJsonButton = document.getElementById("copy-camera-json-button");
const cameraFields = {
  posX: document.getElementById("cam-pos-x"),
  posY: document.getElementById("cam-pos-y"),
  posZ: document.getElementById("cam-pos-z"),
  targetX: document.getElementById("cam-target-x"),
  targetY: document.getElementById("cam-target-y"),
  targetZ: document.getElementById("cam-target-z"),
  upX: document.getElementById("cam-up-x"),
  upY: document.getElementById("cam-up-y"),
  upZ: document.getElementById("cam-up-z"),
  fov: document.getElementById("cam-fov"),
  near: document.getElementById("cam-near"),
  far: document.getElementById("cam-far"),
  width: document.getElementById("render-width"),
  height: document.getElementById("render-height"),
  pixelRatio: document.getElementById("render-pixel-ratio"),
};

const defaultPixelRatio = Math.min(window.devicePixelRatio || 1, 2);
let renderSettings = {
  width: 0,
  height: 0,
  pixelRatio: defaultPixelRatio,
};
let cameraLocked = true;
let applyingCameraControls = false;

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
renderer.setPixelRatio(renderSettings.pixelRatio);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xdfe5ea);

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
camera.up.set(0, 0, 1);
camera.position.set(2.6, -3.2, 2.0);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, 0, 0.45);
controls.update();

const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
pmrem.dispose();

scene.add(new THREE.HemisphereLight(0xffffff, 0x72808c, 2.2));
const sun = new THREE.DirectionalLight(0xffffff, 2.0);
sun.position.set(2.2, -3.0, 4.0);
scene.add(sun);

const grid = new THREE.GridHelper(8, 16, 0x9aa6b2, 0xc8d0d8);
grid.rotation.x = Math.PI / 2;
grid.position.z = 0.001;
scene.add(grid);

const loader = new GLTFLoader();
loader.setMeshoptDecoder(MeshoptDecoder);
const ktx2Loader = new KTX2Loader().setTranscoderPath(BASIS_TRANSCODER_PATH);
ktx2Loader.detectSupport(renderer);
loader.setKTX2Loader(ktx2Loader);

let manifest = null;
let manifestUrl = "";
let records = [];
let currentFrame = 0;
let playing = true;
let lastTime = 0;
let activeStageKey = "";
let selectedManifestPath = new URLSearchParams(window.location.search).get("manifest") || DEFAULT_MANIFEST;
let loadGeneration = 0;

sessionSelect.addEventListener("change", () => {
  if (!sessionSelect.value) {
    return;
  }
  saveCameraState();
  selectedManifestPath = sessionSelect.value;
  loadManifest(sessionSelect.value);
});
playButton.addEventListener("click", () => {
  setPlaying(!playing);
});
cameraMenuButton?.addEventListener("click", () => {
  syncCameraPanelFromCamera();
  if (cameraPanel) cameraPanel.hidden = !cameraPanel.hidden;
});
closeCameraPanelButton?.addEventListener("click", () => {
  if (cameraPanel) cameraPanel.hidden = true;
});
cameraLockInput?.addEventListener("change", () => {
  cameraLocked = Boolean(cameraLockInput.checked);
  saveCameraState();
});
applyCameraButton?.addEventListener("click", () => {
  applyCameraFromPanel();
});
syncCameraButton?.addEventListener("click", () => {
  syncCameraPanelFromCamera();
  saveCameraState();
});
fitCameraButton?.addEventListener("click", () => {
  fitCameraToScene();
  syncCameraPanelFromCamera();
  saveCameraState();
  setStatus("Camera fit to scene");
});
copyCameraJsonButton?.addEventListener("click", () => {
  copyCurrentCamera();
});
for (const field of Object.values(cameraFields)) {
  field?.addEventListener("change", () => applyCameraFromPanel());
}
controls.addEventListener("end", () => {
  syncCameraPanelFromCamera();
  saveCameraState();
});
frameSlider.addEventListener("input", () => {
  setPlaying(false);
  currentFrame = Number(frameSlider.value) || 0;
  updateSceneAtFrame(currentFrame);
});
window.addEventListener("keydown", (event) => {
  if (event.code !== "Space" || event.target?.tagName === "INPUT" || event.target?.tagName === "SELECT") {
    return;
  }
  event.preventDefault();
  setPlaying(!playing);
});
window.addEventListener("resize", resize);

restoreCameraState();
syncCameraPanelFromCamera();
initializeSessionSelect().finally(() => {
  if (selectedManifestPath) {
    loadManifest(selectedManifestPath);
  } else {
    setPlaying(false);
    setStatus("Choose a scene or pass ?manifest=...");
  }
});
requestAnimationFrame(frame);

async function loadManifest(url) {
  const generation = ++loadGeneration;
  try {
    if (!url) {
      setPlaying(false);
      setStatus("Choose a scene or pass ?manifest=...");
      return;
    }
    setStatus("Loading manifest");
    setPlaying(false);
    clearRecords();
    manifestUrl = new URL(url, window.location.href).href;
    const response = await fetch(manifestUrl, { cache: "no-cache" });
    if (!response.ok) {
      throw new Error(`Manifest HTTP ${response.status}`);
    }
    const nextManifest = await response.json();
    if (generation !== loadGeneration) {
      return;
    }
    manifest = await attachCameraMetadata(nextManifest);
    if (generation !== loadGeneration) {
      return;
    }
    frameSlider.max = String(Math.max(0, (manifest.total_frames || 1) - 1));
    frameSlider.value = "0";
    currentFrame = 0;
    setPlaying(false);
    buildStageStrip(manifest);
    setStatus(`Loading ${manifest.objects?.length || 0} objects`);
    const loadedRecords = await loadObjects(manifest);
    if (generation !== loadGeneration) {
      loadedRecords.forEach((record) => disposeObject(record.root));
      return;
    }
    records = loadedRecords;
    for (const record of records) {
      scene.add(record.root);
    }
    updateSceneAtFrame(0);
    for (const record of records) {
      record.root.visible = true;
    }
    const cameraSource = await applyInitialCamera(manifest);
    if (!cameraSource) {
      fitCameraToScene();
    }
    syncCameraPanelFromCamera();
    saveCameraState();
    setPlaying(true);
    syncSessionSelect(manifestUrl);
    setStatus(cameraSource ? `${records.length} objects · ${cameraSource} camera` : `${records.length} objects`);
  } catch (error) {
    setStatus(`Load failed: ${shortError(error)}`);
    console.error(error);
  }
}

async function applyInitialCamera(data) {
  if (cameraLocked) {
    const stored = loadStoredCameraState();
    if (stored && applyCameraPayload(stored)) {
      return "locked";
    }
    const override = await loadCameraOverride();
    if (override && applyCameraPayload(override)) {
      return "override";
    }
    return "locked";
  }
  const override = await loadCameraOverride();
  if (override && applyCameraPayload(override)) {
    return "override";
  }
  if (applyManifestCamera(data)) {
    return "manifest";
  }
  return "";
}

async function loadCameraOverride() {
  const params = new URLSearchParams(window.location.search);
  if (/^(0|false|off|manifest)$/i.test(params.get("cameraOverride") || "")) {
    return null;
  }
  try {
    const response = await fetch(CAMERA_OVERRIDE_URL, { cache: "no-cache" });
    if (!response.ok) {
      return null;
    }
    const payload = await response.json();
    if (payload?.enabled === false) {
      return null;
    }
    return payload;
  } catch {
    return null;
  }
}

async function initializeSessionSelect() {
  const params = new URLSearchParams(window.location.search);
  const explicitSession = params.get("session");
  const explicitManifest = params.get("manifest");
  if (explicitManifest) {
    selectedManifestPath = explicitManifest;
  } else if (explicitSession) {
    selectedManifestPath = manifestPathForSession(explicitSession);
  }
  populateManifestSelect(selectedManifestPath);
  if (selectedManifestPath) {
    syncSessionSelect(new URL(selectedManifestPath, window.location.href).href);
  }
}

async function attachCameraMetadata(data) {
  const embedded = normalizeCameraPayload(data.camera);
  if (embedded) {
    return { ...data, camera: embedded };
  }
  const baseUrl = new URL(manifestUrl, window.location.href);
  const candidates = [
    "../sam3d_moge_separated_original_input_camera.json",
    "../sam3d_moge_optimized_original_input_camera.json",
    "../sam3d_moge_differentiable_camera_pose_optimization.json",
  ];
  for (const candidate of candidates) {
    try {
      const url = new URL(candidate, baseUrl);
      const response = await fetch(url.href, { cache: "no-cache" });
      if (!response.ok) {
        continue;
      }
      const payload = normalizeCameraPayload(await response.json(), url.href);
      if (payload) {
        return { ...data, camera: payload };
      }
    } catch (error) {
      console.warn("Camera metadata unavailable:", candidate, error);
    }
  }
  return data;
}

function normalizeCameraPayload(raw, sourceUrl = "") {
  if (!raw || typeof raw !== "object") {
    return null;
  }
  const cameraToWorld = raw.camera_to_world || raw.optimized_camera_to_world;
  if (!Array.isArray(cameraToWorld) || cameraToWorld.length < 4) {
    return null;
  }
  return {
    source_json: raw.source_json || sourceUrl || "",
    camera_source: raw.camera_source || (raw.optimized_camera_to_world ? "nvdiffrast_optimized" : "original_input_camera"),
    camera_to_world: cameraToWorld,
    intrinsics_normalized: raw.intrinsics_normalized || null,
    image_size: raw.image_size || null,
    fov_degrees: raw.fov_degrees || null,
    blender_camera: raw.blender_camera || null,
  };
}

function populateManifestSelect(manifestPath) {
  sessionSelect.innerHTML = "";
  const option = document.createElement("option");
  option.value = manifestPath || "";
  option.textContent = manifestPath ? labelForManifestPath(manifestPath) : "Pass ?manifest=...";
  option.dataset.session = sessionFromManifestPath(manifestPath);
  sessionSelect.appendChild(option);
  sessionSelect.value = option.value;
  sessionSelect.disabled = !manifestPath;
}

function syncSessionSelect(url) {
  const resolved = new URL(url, window.location.href);
  populateManifestSelect(resolved.href);
}

function manifestPathForSession(session) {
  return `../../sessions/${session}/results/release_package/web/animation_manifest.json`;
}

function labelForManifestPath(value) {
  const session = sessionFromManifestPath(value);
  if (session) {
    return `Session ${session}`;
  }
  const pathname = new URL(value, window.location.href).pathname;
  return pathname.split("/").filter(Boolean).slice(-2).join("/") || "Manifest";
}

function sessionFromManifestPath(value) {
  const match = String(value || "").match(/sessions\/([^/]+)\/results\/(?:release_package\/web|web_pipeline_animation)\/animation_manifest\.json/);
  return match ? match[1] : "";
}

async function loadObjects(data) {
  const baseUrl = new URL(manifestUrl, window.location.href);
  const loaded = [];
  for (const object of data.objects || []) {
    if (!object.url || !object.base_matrix_world) {
      continue;
    }
    const objectUrl = resolveAssetUrl(object.url, baseUrl);
    const gltf = await loader.loadAsync(objectUrl);
    const content = gltf.scene || new THREE.Group();
    content.applyMatrix4(GLTF_VISUAL_TO_BLENDER_Z_UP);
    content.traverse((child) => {
      if (!child.isMesh || !child.material) {
        return;
      }
      const materials = Array.isArray(child.material) ? child.material : [child.material];
      for (const material of materials) {
        material.side = THREE.DoubleSide;
        if ("roughness" in material) material.roughness = 1.0;
        if ("metalness" in material) material.metalness = 0.0;
      }
    });

    const root = new THREE.Group();
    root.name = object.name;
    root.matrixAutoUpdate = false;
    root.visible = false;
    root.add(content);

    const baseMatrix = matrixFromRows(object.base_matrix_world);
    loaded.push({
      root,
      label: object.label || object.name,
      baseInverse: baseMatrix.clone().invert(),
      stageMatrices: matrixMap(object.stage_matrices || {}),
      gravityKeyframes: (object.gravity_keyframes || [])
        .map((item) => ({ frame: Number(item.source_frame), matrix: matrixFromRows(item.matrix_world) }))
        .filter((item) => Number.isFinite(item.frame))
        .sort((a, b) => a.frame - b.frame),
      lastDisplayMatrix: new THREE.Matrix4(),
    });
  }
  return loaded;
}

function updateSceneAtFrame(frame) {
  if (!manifest) {
    return;
  }
  const displayFrame = Math.round(frame);
  const segment = segmentForFrame(manifest.timeline || [], displayFrame);
  const stageKey = stageKeyForSegment(segment, displayFrame);
  if (stageKey !== activeStageKey) {
    activeStageKey = stageKey;
    updateStageStrip(stageKey);
  }
  for (const record of records) {
    const target = matrixForRecord(record, segment, displayFrame);
    const display = target.clone().multiply(record.baseInverse);
    if (matrixIsFinite(display)) {
      record.lastDisplayMatrix.copy(display);
      record.root.matrix.copy(display);
    } else {
      record.root.matrix.copy(record.lastDisplayMatrix);
    }
    record.root.matrixWorldNeedsUpdate = true;
  }
  frameSlider.value = String(displayFrame);
  frameReadout.textContent = `${displayFrame} / ${Math.max(0, (manifest.total_frames || 1) - 1)}`;
}

function matrixForRecord(record, segment, frame) {
  if (!segment) {
    return firstAvailableMatrix(record);
  }
  if (segment.kind === "transition") {
    const a = record.stageMatrices.get(segment.from_stage) || firstAvailableMatrix(record);
    const b = record.stageMatrices.get(segment.to_stage) || a;
    const t = normalizedFrame(segment, frame);
    return interpolateMatrices(a, b, easeInOut(t));
  }
  if (segment.kind === "gravity") {
    const sourceStart = Number(segment.source_frame_start) || 0;
    const sourceEnd = Number(segment.source_frame_end) || sourceStart;
    const sourceFrame = Math.round(THREE.MathUtils.lerp(sourceStart, sourceEnd, normalizedFrame(segment, frame)));
    return gravityMatrixAt(record.gravityKeyframes, sourceFrame)
      || record.stageMatrices.get("separated")
      || firstAvailableMatrix(record);
  }
  return record.stageMatrices.get(segment.stage) || firstAvailableMatrix(record);
}

function gravityMatrixAt(keyframes, sourceFrame) {
  if (!keyframes.length) {
    return null;
  }
  const target = Math.round(sourceFrame);
  let best = keyframes[0];
  let bestDistance = Math.abs(best.frame - target);
  for (let i = 1; i < keyframes.length; i += 1) {
    const item = keyframes[i];
    const distance = Math.abs(item.frame - target);
    if (distance < bestDistance) {
      best = item;
      bestDistance = distance;
    }
  }
  return best.matrix;
}

function firstAvailableMatrix(record) {
  return record.stageMatrices.values().next().value || new THREE.Matrix4();
}

function segmentForFrame(timeline, frame) {
  if (!timeline.length) {
    return null;
  }
  const displayFrame = Math.round(frame);
  return timeline.find((segment) => displayFrame >= segment.frame_start && displayFrame <= segment.frame_end)
    || timeline.find((segment) => displayFrame < segment.frame_start)
    || timeline[timeline.length - 1];
}

function normalizedFrame(segment, frame) {
  const span = Math.max(1, Number(segment.frame_end) - Number(segment.frame_start));
  return THREE.MathUtils.clamp((frame - Number(segment.frame_start)) / span, 0, 1);
}

function stageKeyForSegment(segment, frame = 0) {
  if (!segment) return "";
  if (segment.kind === "transition") {
    return normalizedFrame(segment, frame) < 0.5 ? segment.from_stage : segment.to_stage;
  }
  return segment.stage || "";
}

function interpolateMatrices(a, b, t) {
  const da = decomposeMatrix(a);
  const db = decomposeMatrix(b);
  const position = da.position.lerp(db.position, t);
  const rotation = da.rotation.slerp(db.rotation, t);
  const scale = da.scale.lerp(db.scale, t);
  return new THREE.Matrix4().compose(position, rotation, scale);
}

function decomposeMatrix(matrix) {
  const position = new THREE.Vector3();
  const rotation = new THREE.Quaternion();
  const scale = new THREE.Vector3();
  matrix.decompose(position, rotation, scale);
  rotation.normalize();
  return { position, rotation, scale };
}

function matrixMap(raw) {
  const map = new Map();
  for (const [key, rows] of Object.entries(raw)) {
    map.set(key, matrixFromRows(rows));
  }
  return map;
}

function matrixFromRows(rows) {
  if (!Array.isArray(rows) || rows.length < 4) {
    return new THREE.Matrix4();
  }
  return new THREE.Matrix4().set(
    rows[0][0], rows[0][1], rows[0][2], rows[0][3],
    rows[1][0], rows[1][1], rows[1][2], rows[1][3],
    rows[2][0], rows[2][1], rows[2][2], rows[2][3],
    rows[3][0], rows[3][1], rows[3][2], rows[3][3],
  );
}

function matrixIsFinite(matrix) {
  return matrix.elements.every(Number.isFinite);
}

function easeInOut(t) {
  return t * t * (3 - 2 * t);
}

function resolveAssetUrl(path, baseUrl) {
  const value = String(path || "");
  const markerIndex = value.indexOf(WORKFLOW_MARKER);
  if (markerIndex >= 0) {
    return new URL(`/${value.slice(markerIndex + WORKFLOW_MARKER.length)}`, window.location.href).href;
  }
  return new URL(value, baseUrl).href;
}

function buildStageStrip(data) {
  stageStrip.innerHTML = "";
  const stages = data.stages || [];
  for (const stage of stages) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "stage-chip";
    chip.dataset.stage = stage.key;
    chip.textContent = stage.title || stage.key;
    chip.addEventListener("click", () => {
      const frame = firstFrameForStage(stage.key);
      if (Number.isFinite(frame)) {
        setPlaying(false);
        currentFrame = frame;
        updateSceneAtFrame(currentFrame);
      }
    });
    stageStrip.appendChild(chip);
  }
}

function updateStageStrip(activeKey) {
  for (const chip of stageStrip.querySelectorAll(".stage-chip")) {
    chip.classList.toggle("active", chip.dataset.stage === activeKey);
  }
}

function sceneRecordBounds() {
  const box = new THREE.Box3();
  for (const record of records) {
    box.expandByObject(record.root);
  }
  return box;
}

function applyManifestCamera(data) {
  const cameraData = normalizeCameraPayload(data?.camera);
  if (!cameraData) {
    return false;
  }
  const matrix = matrixFromRows(cameraData.camera_to_world);
  if (!matrixIsFinite(matrix)) {
    return false;
  }
  const position = new THREE.Vector3();
  const rotation = new THREE.Quaternion();
  const scale = new THREE.Vector3();
  matrix.decompose(position, rotation, scale);
  if (!position.toArray().every(Number.isFinite) || !rotation.toArray().every(Number.isFinite)) {
    return false;
  }

  const box = sceneRecordBounds();
  const center = new THREE.Vector3(0, 0, 0);
  const size = new THREE.Vector3(1, 1, 1);
  if (!box.isEmpty()) {
    box.getCenter(center);
    box.getSize(size);
  }
  const radius = Math.max(size.x, size.y, size.z, 0.5);
  const fovY = cameraFovY(cameraData);
  if (Number.isFinite(fovY)) {
    camera.fov = THREE.MathUtils.clamp(fovY, 5, 120);
  }
  camera.near = Math.max(0.001, radius / 1000);
  camera.far = Math.max(100, position.distanceTo(center) + radius * 20);
  camera.clearViewOffset();
  camera.position.copy(position);
  camera.quaternion.copy(rotation.normalize());
  camera.up.copy(new THREE.Vector3(0, 1, 0).applyQuaternion(camera.quaternion).normalize());

  const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion).normalize();
  const distanceToCenter = center.clone().sub(camera.position).dot(forward);
  const targetDistance = Number.isFinite(distanceToCenter) && distanceToCenter > radius * 0.1
    ? distanceToCenter
    : Math.max(radius * 2, 1);
  controls.target.copy(camera.position).addScaledVector(forward, targetDistance);
  camera.updateProjectionMatrix();
  controls.update();
  return true;
}

function applyViewerCameraPose(cameraData) {
  if (!cameraData || typeof cameraData !== "object") {
    return false;
  }
  const position = vector3FromArray(cameraData.position);
  const target = vector3FromArray(cameraData.target || cameraData.look_at);
  if (!position || !target) {
    if (cameraData.camera_to_world) {
      return applyManifestCamera({ camera: cameraData });
    }
    return false;
  }
  const up = vector3FromArray(cameraData.up);
  if (up && up.lengthSq() > 1e-12) {
    camera.up.copy(up.normalize());
  } else {
    camera.up.set(0, 0, 1);
  }
  camera.position.copy(position);
  controls.target.copy(target);
  const fov = Number(cameraData.fov);
  if (Number.isFinite(fov) && fov > 0) {
    camera.fov = THREE.MathUtils.clamp(fov, 5, 120);
  }
  const near = Number(cameraData.near);
  const far = Number(cameraData.far);
  if (Number.isFinite(near) && near > 0) camera.near = near;
  if (Number.isFinite(far) && far > camera.near) camera.far = far;
  camera.updateProjectionMatrix();
  camera.lookAt(controls.target);
  controls.update();
  return true;
}

function applyCameraPayload(payload) {
  if (!payload || typeof payload !== "object") {
    return false;
  }
  if (payload.lock_camera !== undefined) {
    cameraLocked = Boolean(payload.lock_camera);
    if (cameraLockInput) cameraLockInput.checked = cameraLocked;
  }
  applyRenderSettings(payload.render || payload.resolution || {});
  return applyViewerCameraPose(payload.camera || payload);
}

function restoreCameraState() {
  const stored = loadStoredCameraState();
  if (!stored) {
    if (cameraLockInput) cameraLockInput.checked = cameraLocked;
    return;
  }
  applyCameraPayload(stored);
}

function loadStoredCameraState() {
  try {
    const raw = window.localStorage?.getItem(CAMERA_STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function saveCameraState() {
  const payload = currentCameraPayload();
  try {
    window.localStorage?.setItem(CAMERA_STORAGE_KEY, JSON.stringify(payload));
  } catch {
    // Storage can be disabled in embedded browsers; the live camera still works.
  }
  updateCameraJson(payload);
}

function applyCameraFromPanel() {
  const payload = cameraPayloadFromPanel();
  if (!payload) {
    setStatus("Invalid camera values");
    return;
  }
  if (applyCameraPayload(payload)) {
    saveCameraState();
    setStatus("Camera applied");
  }
}

function cameraPayloadFromPanel() {
  const position = vectorFromFields("pos", camera.position);
  const target = vectorFromFields("target", controls.target);
  const up = vectorFromFields("up", camera.up);
  if (!position || !target || !up) {
    return null;
  }
  const fov = numberFromField(cameraFields.fov, camera.fov);
  const near = numberFromField(cameraFields.near, camera.near);
  const far = numberFromField(cameraFields.far, camera.far);
  return {
    enabled: true,
    lock_camera: Boolean(cameraLockInput?.checked ?? cameraLocked),
    camera: {
      position,
      target,
      up,
      fov: THREE.MathUtils.clamp(fov, 5, 120),
      near: Math.max(near, 0.0001),
      far: Math.max(far, near + 0.001),
    },
    render: {
      width: Math.max(0, Math.round(numberFromField(cameraFields.width, renderSettings.width))),
      height: Math.max(0, Math.round(numberFromField(cameraFields.height, renderSettings.height))),
      pixelRatio: THREE.MathUtils.clamp(numberFromField(cameraFields.pixelRatio, renderSettings.pixelRatio), 0.5, 3),
    },
  };
}

function vectorFromFields(prefix, fallback) {
  const ids = prefix === "pos"
    ? ["posX", "posY", "posZ"]
    : prefix === "target"
      ? ["targetX", "targetY", "targetZ"]
      : ["upX", "upY", "upZ"];
  const values = ids.map((id, index) => numberFromField(cameraFields[id], fallback.getComponent(index)));
  return values.every(Number.isFinite) ? values : null;
}

function numberFromField(field, fallback) {
  const value = Number(field?.value);
  return Number.isFinite(value) ? value : fallback;
}

function syncCameraPanelFromCamera() {
  if (applyingCameraControls) {
    return;
  }
  applyingCameraControls = true;
  if (cameraLockInput) cameraLockInput.checked = cameraLocked;
  setField(cameraFields.posX, camera.position.x);
  setField(cameraFields.posY, camera.position.y);
  setField(cameraFields.posZ, camera.position.z);
  setField(cameraFields.targetX, controls.target.x);
  setField(cameraFields.targetY, controls.target.y);
  setField(cameraFields.targetZ, controls.target.z);
  setField(cameraFields.upX, camera.up.x);
  setField(cameraFields.upY, camera.up.y);
  setField(cameraFields.upZ, camera.up.z);
  setField(cameraFields.fov, camera.fov, 2);
  setField(cameraFields.near, camera.near, 4);
  setField(cameraFields.far, camera.far, 2);
  cameraFields.width.value = renderSettings.width > 0 ? String(renderSettings.width) : "";
  cameraFields.height.value = renderSettings.height > 0 ? String(renderSettings.height) : "";
  setField(cameraFields.pixelRatio, renderSettings.pixelRatio, 2);
  updateCameraJson(currentCameraPayload());
  applyingCameraControls = false;
}

function setField(field, value, digits = 4) {
  if (!field) {
    return;
  }
  field.value = Number.isFinite(value) ? Number(value).toFixed(digits).replace(/\.?0+$/, "") : "";
}

function updateCameraJson(payload) {
  if (cameraJsonEl) {
    cameraJsonEl.value = JSON.stringify(payload, null, 2);
  }
}

function applyRenderSettings(settings) {
  if (!settings || typeof settings !== "object") {
    return;
  }
  const width = Number(settings.width);
  const height = Number(settings.height);
  const pixelRatio = Number(settings.pixelRatio ?? settings.pixel_ratio);
  if (Number.isFinite(width)) renderSettings.width = Math.max(0, Math.round(width));
  if (Number.isFinite(height)) renderSettings.height = Math.max(0, Math.round(height));
  if (Number.isFinite(pixelRatio)) renderSettings.pixelRatio = THREE.MathUtils.clamp(pixelRatio, 0.5, 3);
  resize();
}

function cameraFovY(cameraData) {
  const direct = Number(cameraData.fov_degrees?.y);
  if (Number.isFinite(direct) && direct > 0) {
    return direct;
  }
  const fy = Number(cameraData.intrinsics_normalized?.[1]?.[1]);
  if (Number.isFinite(fy) && fy > 0) {
    return THREE.MathUtils.radToDeg(2 * Math.atan(0.5 / fy));
  }
  return NaN;
}

function fitCameraToScene() {
  const box = sceneRecordBounds();
  if (box.isEmpty()) {
    return;
  }
  const center = new THREE.Vector3();
  const size = new THREE.Vector3();
  box.getCenter(center);
  box.getSize(size);
  const radius = Math.max(size.x, size.y, size.z, 0.5);
  camera.up.set(0, 0, 1);
  controls.target.copy(center);
  camera.position.copy(center).add(new THREE.Vector3(radius * 1.3, -radius * 1.9, radius * 1.1));
  camera.lookAt(center);
  camera.near = Math.max(0.01, radius / 200);
  camera.far = Math.max(20, radius * 20);
  camera.updateProjectionMatrix();
  controls.update();
}

function vector3FromArray(value) {
  if (!Array.isArray(value) || value.length < 3) {
    return null;
  }
  const out = new THREE.Vector3(Number(value[0]), Number(value[1]), Number(value[2]));
  return out.toArray().every(Number.isFinite) ? out : null;
}

function rowMajorMatrixWorld(cameraObject) {
  cameraObject.updateMatrixWorld(true);
  const e = cameraObject.matrixWorld.elements;
  return [
    [e[0], e[4], e[8], e[12]],
    [e[1], e[5], e[9], e[13]],
    [e[2], e[6], e[10], e[14]],
    [e[3], e[7], e[11], e[15]],
  ];
}

function currentCameraPayload() {
  return {
    enabled: true,
    lock_camera: cameraLocked,
    camera: {
      position: camera.position.toArray(),
      target: controls.target.toArray(),
      up: camera.up.toArray(),
      fov: camera.fov,
      near: camera.near,
      far: camera.far,
      quaternion_xyzw: camera.quaternion.toArray(),
      camera_to_world: rowMajorMatrixWorld(camera),
    },
    render: {
      width: renderSettings.width,
      height: renderSettings.height,
      pixelRatio: renderSettings.pixelRatio,
    },
  };
}

async function copyCurrentCamera() {
  const text = JSON.stringify(currentCameraPayload(), null, 2);
  showCameraJson(text);
  try {
    if (!navigator.clipboard?.writeText) {
      throw new Error("clipboard API unavailable");
    }
    await navigator.clipboard.writeText(text);
    setStatus("Camera copied");
  } catch (error) {
    console.log("Camera override JSON:", text);
    cameraJsonEl?.focus();
    cameraJsonEl?.select();
    setStatus(`Camera shown, select text manually`);
  }
}

function showCameraJson(text) {
  if (cameraJsonEl) {
    cameraJsonEl.value = text;
  }
  if (cameraPanel) {
    cameraPanel.hidden = false;
  }
}

function firstFrameForStage(stageKey) {
  if (!manifest?.timeline) {
    return NaN;
  }
  const direct = manifest.timeline.find((segment) => segment.stage === stageKey && (segment.kind === "hold" || segment.kind === "gravity"));
  if (direct) {
    return Number(direct.frame_start) || 0;
  }
  const transition = manifest.timeline.find((segment) => segment.to_stage === stageKey || segment.from_stage === stageKey);
  return transition ? Number(transition.frame_start) || 0 : NaN;
}

function setPlaying(value) {
  playing = Boolean(value);
  playButton.textContent = playing ? "Pause" : "Play";
}

function clearRecords() {
  for (const record of records) {
    scene.remove(record.root);
    disposeObject(record.root);
  }
  records = [];
}

function disposeObject(root) {
  root.traverse((object) => {
    if (object.geometry) object.geometry.dispose();
    if (object.material) {
      const materials = Array.isArray(object.material) ? object.material : [object.material];
      for (const material of materials) {
        for (const value of Object.values(material)) {
          if (value?.isTexture) value.dispose();
        }
        material.dispose?.();
      }
    }
  });
}

function resize() {
  const parent = canvas.parentElement;
  const width = Math.max(1, renderSettings.width || parent.clientWidth);
  const height = Math.max(1, renderSettings.height || parent.clientHeight);
  renderer.setPixelRatio(renderSettings.pixelRatio);
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

function frame(time) {
  resize();
  const dt = lastTime ? (time - lastTime) / 1000 : 0;
  lastTime = time;
  if (playing && manifest) {
    const fps = Number(manifest.fps) || 24;
    const speed = Number(speedSelect.value) || 1;
    currentFrame = (currentFrame + dt * fps * speed) % Math.max(1, manifest.total_frames || 1);
    updateSceneAtFrame(currentFrame);
  }
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}

function setStatus(text) {
  statusEl.textContent = text;
}

function shortError(error) {
  return String(error?.message || error || "unknown error").split("\n")[0];
}
