import * as THREE from "three";

const BASIS_TRANSCODER_PATH = "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/libs/basis/";
const DEFAULT_CAMERA_POSITION = [1.4, -1.8, 1.1];
const DEFAULT_CAMERA_LOOK_AT = [0, 0, 0.35];
const GLTF_VISUAL_TO_BLENDER_Z_UP = new THREE.Matrix4().makeRotationX(Math.PI / 2);
const DEFAULT_GS_ALIGNMENT_MODE = "none";

let sharedKtx2Loader = null;
let libsPromise = null;
let GLTFLoader, KTX2Loader, MeshoptDecoder, RoomEnvironment, GaussianSplats3D;

function ensureHybridLibs() {
  if (!libsPromise) {
    libsPromise = Promise.all([
      import("three/addons/loaders/GLTFLoader.js"),
      import("three/addons/loaders/KTX2Loader.js"),
      import("three/addons/libs/meshopt_decoder.module.js"),
      import("three/addons/environments/RoomEnvironment.js"),
      import("@mkkellogg/gaussian-splats-3d"),
    ]).then(([gltf, ktx2, meshopt, room, splats]) => {
      GLTFLoader = gltf.GLTFLoader;
      KTX2Loader = ktx2.KTX2Loader;
      MeshoptDecoder = meshopt.MeshoptDecoder;
      RoomEnvironment = room.RoomEnvironment;
      GaussianSplats3D = splats;
    });
  }
  return libsPromise;
}

class HybridSceneViewer {
  constructor(container, options = {}) {
    this.container = container;
    this.manifestUrl = options.manifestUrl || container.dataset.manifest || "";
    this.viewer = null;
    this.meshRoots = [];
    this.disposed = false;
    this.loadingEl = null;
    this.gsAlignmentMode = container.dataset.gsAlign || DEFAULT_GS_ALIGNMENT_MODE;
    this.gsManualZOffset = Number(container.dataset.gsZOffset) || 0;
  }

  async load(manifestUrl = this.manifestUrl) {
    this.dispose();
    this.disposed = false;
    this.manifestUrl = manifestUrl;
    this.container.classList.add("hybrid-viewer");
    this.container.innerHTML = "";
    this.loadingEl = createLoading(this.container, "Loading hybrid viewer");

    if (!manifestUrl) {
      this.loadingEl.textContent = "Set data-manifest to a 3DGS scene.json";
      return null;
    }

    let manifest;
    try {
      manifest = await fetchJson(manifestUrl);
      manifest = await enrichHybridManifest(manifest, manifestUrl);
    } catch (error) {
      this.loadingEl.textContent = `Failed to load manifest: ${shortError(error)}`;
      return null;
    }

    if (!manifest.splat?.url) {
      this.loadingEl.textContent = "Manifest has no splat.url";
      return null;
    }

    try {
      this.loadingEl.textContent = "Loading 3DGS engine";
      await ensureHybridLibs();
    } catch (error) {
      this.loadingEl.textContent = `Failed to load 3DGS engine: ${shortError(error)}`;
      return null;
    }

    const baseUrl = new URL(manifestUrl, window.location.href);
    const splatUrl = resolveAssetUrl(manifest.splat.url, baseUrl);
    const gsAlignment = computeGsAlignment(
      manifest,
      this.gsAlignmentMode,
      this.gsManualZOffset,
    );
    const splatTransform = composeTransformSpecs(gsAlignment.transform, manifest.world_transform, manifest.splat);
    const worldUp = manifest.world_up || [0, 0, 1];
    const cameraPosition = manifest.camera?.position || DEFAULT_CAMERA_POSITION;
    const cameraLookAt = manifest.camera?.look_at || lookAtFromQuaternion(manifest.camera) || DEFAULT_CAMERA_LOOK_AT;

    this.loadingEl.textContent = "Initializing 3DGS background";
    this.viewer = new GaussianSplats3D.Viewer({
      rootElement: this.container,
      cameraUp: worldUp,
      initialCameraPosition: cameraPosition,
      initialCameraLookAt: cameraLookAt,
      sharedMemoryForWorkers: false,
      gpuAcceleratedSort: false,
      sphericalHarmonicsDegree: manifest.splat.spherical_harmonics_degree ?? 1,
      useBuiltInControls: true,
      antialiased: false,
      showLoadingUI: false,
      devicePixelRatio: Math.min(window.devicePixelRatio || 1, 1.5),
    });
    applyViewerCameraProjection(this.viewer, manifest.camera);

    const splatFormat = getSplatFormat(manifest.splat, splatUrl);
    try {
      await this.viewer.addSplatScene(splatUrl, {
        format: splatFormat,
        position: splatTransform.position.toArray(),
        rotation: splatTransform.quaternion_xyzw,
        scale: splatTransform.scale.toArray(),
        splatAlphaRemovalThreshold: manifest.splat.alpha_removal_threshold ?? 20,
        progressiveLoad: false,
        showLoadingUI: false,
      });
    } catch (error) {
      this.loadingEl.textContent = `Failed to load splat: ${shortError(error)}`;
      return null;
    }

    if (this.disposed) {
      return null;
    }

    this.configureThreeScene();
    this.viewer.start();
    this.applyControlLimits(manifest);
    this.removeLoading();

    await this.loadMeshes(manifest, baseUrl);
    return this;
  }

  configureThreeScene() {
    const renderer = this.viewer?.renderer;
    if (!renderer) {
      return;
    }

    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;

    const pmrem = new THREE.PMREMGenerator(renderer);
    const envTexture = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    this.viewer.threeScene.environment = envTexture;
    pmrem.dispose();

    this.viewer.threeScene.add(new THREE.AmbientLight(0xffffff, 0.42));
    const sun = new THREE.DirectionalLight(0xffffff, 1.5);
    sun.position.set(1.5, -2.0, 3.0);
    this.viewer.threeScene.add(sun);
  }

  applyControlLimits(manifest) {
    const controls = this.viewer?.controls || this.viewer?.orbitControls || this.viewer?.cameraControls;
    if (!controls || !manifest.controls) {
      return;
    }

    const config = manifest.controls;
    if (config.enablePan !== undefined) controls.enablePan = Boolean(config.enablePan);
    if (Number.isFinite(config.minDistance)) controls.minDistance = config.minDistance;
    if (Number.isFinite(config.maxDistance)) controls.maxDistance = config.maxDistance;
    if (Number.isFinite(config.minPolarAngle)) controls.minPolarAngle = config.minPolarAngle;
    if (Number.isFinite(config.maxPolarAngle)) controls.maxPolarAngle = config.maxPolarAngle;
    if (Number.isFinite(config.minAzimuthAngle)) controls.minAzimuthAngle = config.minAzimuthAngle;
    if (Number.isFinite(config.maxAzimuthAngle)) controls.maxAzimuthAngle = config.maxAzimuthAngle;
    controls.update?.();
  }

  async loadMeshes(manifest, baseUrl) {
    const renderer = this.viewer?.renderer;
    const scene = this.viewer?.threeScene;
    if (!renderer || !scene) {
      return;
    }

    const loader = makeGltfLoader(renderer);
    for (const object of manifest.objects || []) {
      if (!object.url || this.disposed) {
        continue;
      }

      const objectUrl = resolveAssetUrl(object.url, baseUrl);
      try {
        const gltf = await loader.loadAsync(objectUrl);
        if (this.disposed) {
          disposeObject3D(gltf.scene);
          return;
        }
        const visualRoot = gltf.scene;
        let root = visualRoot;
        if (object.gltf_visual_to_blender_z_up) {
          root = new THREE.Group();
          root.add(visualRoot);
          visualRoot.applyMatrix4(GLTF_VISUAL_TO_BLENDER_Z_UP);
        }
        applyObjectTransform(root, object, manifest.world_transform);
        makeTexturedMaterialsMatte(root);
        scene.add(root);
        this.meshRoots.push(root);
      } catch (error) {
        console.warn("Hybrid mesh load failed:", objectUrl, error);
      }
    }
  }

  removeLoading() {
    if (this.loadingEl?.parentNode === this.container) {
      this.container.removeChild(this.loadingEl);
    }
    this.loadingEl = null;
  }

  dispose() {
    this.disposed = true;
    this.meshRoots.forEach(disposeObject3D);
    this.meshRoots = [];
    if (this.viewer) {
      try {
        this.viewer.dispose();
      } catch (error) {
        console.warn("Hybrid viewer dispose failed:", error);
      }
      this.viewer = null;
    }
    this.removeLoading();
  }
}

function makeGltfLoader(renderer) {
  if (!sharedKtx2Loader) {
    sharedKtx2Loader = new KTX2Loader().setTranscoderPath(BASIS_TRANSCODER_PATH);
  }
  sharedKtx2Loader.detectSupport(renderer);
  return new GLTFLoader().setMeshoptDecoder(MeshoptDecoder).setKTX2Loader(sharedKtx2Loader);
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-cache" });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  return response.json();
}

async function enrichHybridManifest(manifest, manifestUrl) {
  const enriched = {
    ...manifest,
    world_transform: manifest.world_transform ?? null,
    point_cloud: manifest.point_cloud ?? manifest.export_metadata?.point_cloud ?? null,
    splat: {
      ...manifest.splat,
      position: vectorArray(manifest.splat?.position, [0, 0, 0]),
      quaternion_xyzw: quaternionArray(manifest.splat?.quaternion_xyzw, [0, 0, 0, 1]),
      scale: vectorArray(manifest.splat?.scale, [1, 1, 1]),
    },
    camera: manifest.camera ? { ...manifest.camera } : null,
  };

  if (!hasCameraIntrinsics(enriched.camera)) {
    try {
      const companionUrl = new URL("manifest.json", new URL(manifestUrl, window.location.href)).href;
      if (companionUrl !== new URL(manifestUrl, window.location.href).href) {
        const companion = await fetchJson(companionUrl);
        if (companion.camera) {
          enriched.camera = { ...companion.camera, ...(enriched.camera ?? {}) };
        }
        enriched.point_cloud = enriched.point_cloud
          ?? companion.point_cloud
          ?? companion.export_metadata?.point_cloud
          ?? null;
      }
    } catch (error) {
      console.debug("3DGS companion manifest not available:", error);
    }
  }

  return enriched;
}

function createLoading(container, text) {
  const el = document.createElement("div");
  el.className = "hybrid-viewer-loading";
  el.textContent = text;
  container.appendChild(el);
  return el;
}

function getSplatFormat(splat, url) {
  const formats = {
    ply: GaussianSplats3D.SceneFormat.Ply,
    ksplat: GaussianSplats3D.SceneFormat.KSplat,
    splat: GaussianSplats3D.SceneFormat.Splat,
  };
  return formats[String(splat.format || "").toLowerCase()] || GaussianSplats3D.LoaderUtils.sceneFormatFromPath(url);
}

function resolveAssetUrl(path, manifestBaseUrl) {
  return new URL(path, manifestBaseUrl).toString();
}

function lookAtFromQuaternion(camera) {
  if (!camera?.position || !camera?.quaternion_xyzw) {
    return null;
  }
  const q = new THREE.Quaternion(
    camera.quaternion_xyzw[0],
    camera.quaternion_xyzw[1],
    camera.quaternion_xyzw[2],
    camera.quaternion_xyzw[3],
  );
  const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(q);
  return new THREE.Vector3().fromArray(camera.position).add(forward).toArray();
}

function applyObjectTransform(root, object, worldTransform = null) {
  const transform = composeTransformSpecs(worldTransform, object);
  root.position.copy(transform.position);
  root.quaternion.copy(transform.rotation);
  root.scale.copy(transform.scale);
}

function computeGsAlignment(manifest, mode, manualZOffset = 0) {
  let zOffset = manualZOffset;
  const shouldAutoAlign = /^floor$/i.test(mode)
    && !hasNonIdentityTransform(manifest.world_transform);

  if (shouldAutoAlign) {
    const splatFloorZ = splatFloorZFromManifest(manifest);
    if (Number.isFinite(splatFloorZ)) {
      zOffset -= splatFloorZ;
    }
  }

  return {
    zOffset,
    transform: new THREE.Matrix4().makeTranslation(0, 0, zOffset),
  };
}

function applyViewerCameraProjection(viewer, camera) {
  const fov = verticalFovDegreesFromCamera(camera);
  if (!Number.isFinite(fov)) {
    return;
  }

  const cameras = [viewer?.camera, viewer?.perspectiveCamera]
    .filter((value, index, list) => value && list.indexOf(value) === index);
  for (const activeCamera of cameras) {
    activeCamera.fov = THREE.MathUtils.clamp(fov, 10, 100);
    activeCamera.updateProjectionMatrix();
  }
}

function composeTransformSpecs(...specs) {
  const matrix = new THREE.Matrix4();
  for (const spec of specs) {
    matrix.multiply(matrixFromTransformSpec(spec));
  }
  return decomposeTransformMatrix(matrix);
}

function hasNonIdentityTransform(spec) {
  if (!spec) {
    return false;
  }

  const transform = decomposeTransformMatrix(matrixFromTransformSpec(spec));
  return transform.position.lengthSq() > 1e-12
    || Math.abs(transform.rotation.x) > 1e-8
    || Math.abs(transform.rotation.y) > 1e-8
    || Math.abs(transform.rotation.z) > 1e-8
    || Math.abs(transform.rotation.w - 1) > 1e-8
    || Math.abs(transform.scale.x - 1) > 1e-8
    || Math.abs(transform.scale.y - 1) > 1e-8
    || Math.abs(transform.scale.z - 1) > 1e-8;
}

function splatFloorZFromManifest(manifest) {
  return finiteOptionalNumber(
    manifest.alignment?.splat_floor_z
      ?? manifest.alignment?.point_cloud_floor_z
      ?? manifest.point_cloud?.bbox_min?.[2]
      ?? manifest.export_metadata?.point_cloud?.bbox_min?.[2],
  );
}

function matrixFromTransformSpec(spec) {
  if (!spec) {
    return new THREE.Matrix4();
  }
  if (spec.isMatrix4) {
    return spec.clone();
  }

  const matrixRows = spec.matrix_rows ?? spec.matrix ?? spec.transform_matrix ?? spec.pose_matrix;
  if (matrixRows) {
    return matrixFromRowMajor(matrixRows);
  }

  const position = new THREE.Vector3().fromArray(vectorArray(spec.position, [0, 0, 0]));
  const rotation = new THREE.Quaternion().fromArray(
    quaternionArray(spec.quaternion_xyzw ?? spec.quaternion ?? spec.rotation, [0, 0, 0, 1]),
  );
  const scale = new THREE.Vector3().fromArray(vectorArray(spec.scale, [1, 1, 1]));
  return new THREE.Matrix4().compose(position, rotation.normalize(), scale);
}

function matrixFromRowMajor(rows) {
  if (!Array.isArray(rows)) {
    return new THREE.Matrix4();
  }
  if (rows.length >= 4 && Array.isArray(rows[0])) {
    return new THREE.Matrix4().set(
      rows[0][0], rows[0][1], rows[0][2], rows[0][3],
      rows[1][0], rows[1][1], rows[1][2], rows[1][3],
      rows[2][0], rows[2][1], rows[2][2], rows[2][3],
      rows[3][0], rows[3][1], rows[3][2], rows[3][3],
    );
  }
  if (rows.length >= 16) {
    return new THREE.Matrix4().set(
      rows[0], rows[1], rows[2], rows[3],
      rows[4], rows[5], rows[6], rows[7],
      rows[8], rows[9], rows[10], rows[11],
      rows[12], rows[13], rows[14], rows[15],
    );
  }
  return new THREE.Matrix4();
}

function decomposeTransformMatrix(matrix) {
  const position = new THREE.Vector3();
  const rotation = new THREE.Quaternion();
  const scale = new THREE.Vector3();
  matrix.decompose(position, rotation, scale);
  rotation.normalize();
  return { matrix: matrix.clone(), position, rotation, scale, quaternion_xyzw: rotation.toArray() };
}

function vectorArray(value, fallback) {
  const source = Array.isArray(value) ? value : fallback;
  return [
    finiteNumber(source[0], fallback[0]),
    finiteNumber(source[1], fallback[1]),
    finiteNumber(source[2], fallback[2]),
  ];
}

function quaternionArray(value, fallback) {
  const source = Array.isArray(value) ? value : fallback;
  const result = [
    finiteNumber(source[0], fallback[0]),
    finiteNumber(source[1], fallback[1]),
    finiteNumber(source[2], fallback[2]),
    finiteNumber(source[3], fallback[3]),
  ];
  const length = Math.hypot(result[0], result[1], result[2], result[3]);
  if (length < 1e-10) {
    return fallback.slice();
  }
  return result.map((component) => component / length);
}

function finiteNumber(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function finiteOptionalNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function hasCameraIntrinsics(camera) {
  return Boolean(
    camera?.intrinsics_pixels
      || camera?.intrinsics_normalized
      || Number.isFinite(camera?.fov_y_degrees)
      || Number.isFinite(camera?.fov_degrees),
  );
}

function verticalFovDegreesFromCamera(camera) {
  const explicitFov = finiteNumber(camera?.fov_y_degrees ?? camera?.fov_degrees, NaN);
  if (Number.isFinite(explicitFov)) {
    return explicitFov;
  }

  const imageSize = camera?.image_size ?? camera?.source_image_size;
  const imageHeight = Array.isArray(imageSize) ? Number(imageSize[1]) : NaN;
  const fyPixels = Number(camera?.intrinsics_pixels?.[1]?.[1]);
  if (Number.isFinite(imageHeight) && imageHeight > 0 && Number.isFinite(fyPixels) && fyPixels > 0) {
    return THREE.MathUtils.radToDeg(2 * Math.atan(imageHeight / (2 * fyPixels)));
  }

  const fyNormalized = Number(camera?.intrinsics_normalized?.[1]?.[1]);
  if (Number.isFinite(fyNormalized) && fyNormalized > 0) {
    return THREE.MathUtils.radToDeg(2 * Math.atan(0.5 / fyNormalized));
  }

  return NaN;
}

function makeTexturedMaterialsMatte(root) {
  root.traverse((object) => {
    if (!object.isMesh || !object.material) {
      return;
    }
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    materials.forEach((material) => {
      if (!material.map) {
        return;
      }
      if ("metalness" in material) material.metalness = 0;
      if ("roughness" in material) material.roughness = 1;
      if ("specularIntensity" in material) material.specularIntensity = 0;
      material.needsUpdate = true;
    });
  });
}

function disposeObject3D(root) {
  root.traverse((object) => {
    if (!object.isMesh) {
      return;
    }
    object.geometry?.dispose?.();
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    materials.forEach((material) => material?.dispose?.());
  });
}

function shortError(error) {
  return String(error?.message || error || "unknown error").slice(0, 100);
}

function bootHybridViewers() {
  const viewers = [];
  document.querySelectorAll("[data-hybrid-viewer]").forEach((container) => {
    const viewer = new HybridSceneViewer(container);
    viewers.push(viewer);
    viewer.load();
  });
  window.HybridSceneViewer = HybridSceneViewer;
  window.hybridSceneViewers = viewers;
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", bootHybridViewers);
} else {
  bootHybridViewers();
}

export { HybridSceneViewer };
