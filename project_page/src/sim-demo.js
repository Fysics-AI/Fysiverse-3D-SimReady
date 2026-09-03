import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { MeshoptDecoder } from "meshoptimizer";
import { ColladaLoader } from "three/addons/loaders/ColladaLoader.js";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
import { FBXLoader } from "three/addons/loaders/FBXLoader.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { KTX2Loader } from "three/addons/loaders/KTX2Loader.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";
import * as RAPIER from "@dimforge/rapier3d-compat";

const FIXED_STEP = 1 / 60;
const RELEASE_SPEED_LIMIT = 18;
const TMP_VEC = new THREE.Vector3();
const ZERO_VEC = { x: 0, y: 0, z: 0 };
const Z_UP = new THREE.Vector3(0, 0, 1);
const RAD_TO_DEG = 180 / Math.PI;
const DEG_TO_RAD = Math.PI / 180;
const URDF_FETCH_TIMEOUT_MS = 2500;
const SHADOWS_ENABLED = false;
const SCENE_BOUNDARY_SCALE = 1.2;
const SCENE_BOUNDARY_WALL_THICKNESS = 0.08;
const SCENE_BOUNDARY_MIN_SIZE = 2.0;
const SCENE_BOUNDARY_MIN_HEIGHT = 1.2;
const DEFAULT_SCENE_MANIFEST = "";
const FRANKA_URDF_URL = "./assets/urdf/franka/franka.urdf";
const CHARACTER_WALK_FBX_URL = "./assets/character/Walking.fbx";
const CHARACTER_KICK_FBX_URL = "./assets/character/Kick.fbx";
const CHARACTER_SCALE = 0.2;
const CHARACTER_HEIGHT = 1.75 * CHARACTER_SCALE;
const CHARACTER_CAPSULE_RADIUS = 0.28 * CHARACTER_SCALE;
const CHARACTER_CAPSULE_HALF_HEIGHT = (CHARACTER_HEIGHT - CHARACTER_CAPSULE_RADIUS * 2) / 2;
const CHARACTER_CAPSULE_CENTER_Z = CHARACTER_HEIGHT / 2;
const CHARACTER_SPEED = 1.45 * CHARACTER_SCALE;
const CHARACTER_KICK_FALLBACK_DURATION = 1.05;
const CHARACTER_KICK_HEIGHT = 0.48 * CHARACTER_SCALE;
const CHARACTER_KICK_START_DISTANCE = 0.34 * CHARACTER_SCALE;
const CHARACTER_KICK_MAX_REACH = 1.05 * CHARACTER_SCALE;
const CHARACTER_KICK_MIN_RADIUS = 0.12 * CHARACTER_SCALE;
const CHARACTER_KICK_MAX_RADIUS = 0.28 * CHARACTER_SCALE;
const CHARACTER_KICK_IMPULSE = 1.2 * CHARACTER_SCALE;
const CHARACTER_KICK_UP_IMPULSE = 0.12 * CHARACTER_SCALE;
const CHARACTER_KICK_MIN_FORWARD_DISTANCE = 0.2 * CHARACTER_SCALE;
const CHARACTER_KICK_FORWARD_PADDING = 0.45 * CHARACTER_SCALE;
const CHARACTER_KICK_LOWER_PADDING = 0.1 * CHARACTER_SCALE;
const CHARACTER_KICK_LATERAL_PADDING = 0.42 * CHARACTER_SCALE;
const CHARACTER_START_POSITION = new THREE.Vector3(0, 0, 0);
const CHARACTER_MOVE_KEYS = new Set(["KeyW", "KeyA", "KeyS", "KeyD"]);
const THREE_EXAMPLES_BASE = "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/";
const GLTF_VISUAL_TO_BLENDER_Z_UP = new THREE.Matrix4().makeRotationX(Math.PI / 2);
const URDF_COACD_MANIFEST_NAME = "manifest.json";
const VISUAL_FRAME_ACTOR = "actor";
const VISUAL_FRAME_WORLD = "world";

await RAPIER.init();

class InteractiveRigidBodyDemo {
  constructor(container) {
    this.container = container;
    this.statusEl = document.getElementById("status");
    this.resetButton = document.getElementById("reset-button");
    this.physicsButton = document.getElementById("physics-button");
    this.debugButton = document.getElementById("debug-button");
    this.sceneSelect = document.getElementById("scene-select");
    this.loadSceneButton = document.getElementById("load-scene-button");
    this.clearSceneButton = document.getElementById("clear-scene-button");
    this.generatedStatusEl = document.getElementById("generated-status");
    this.robotSelect = document.getElementById("robot-select");
    this.loadRobotButton = document.getElementById("load-robot-button");
    this.clearRobotButton = document.getElementById("clear-robot-button");
    this.robotStatusEl = document.getElementById("robot-status");
    this.characterSelect = document.getElementById("character-select");
    this.loadCharacterButton = document.getElementById("load-character-button");
    this.clearCharacterButton = document.getElementById("clear-character-button");
    this.characterStatusEl = document.getElementById("character-status");
    this.robotBaseControlsEl = document.getElementById("robot-base-controls");
    this.jointControlsEl = document.getElementById("joint-controls");

    this.clockAccumulator = 0;
    this.lastFrameTime = 0;
    this.physicsEnabled = false;
    this.debugEnabled = false;
    this.drag = null;

    this.physicsObjects = [];
    this.sampleRecords = [];
    this.generatedRecords = [];
    this.generatedBoundaryRecords = [];
    this.bodyToRecord = new Map();
    this.colliderToRecord = new Map();
    this.joints = [];
    this.robotJointControls = [];
    this.robots = [];

    this.pointer = new THREE.Vector2();
    this.raycaster = new THREE.Raycaster();
    this.gltfLoader = null;
    this.fbxLoader = null;
    this.colladaLoader = null;
    this.stlLoader = null;
    this.generatedSceneBounds = null;
    this.generatedSceneGroundZ = 0;
    this.webSceneTransform = new THREE.Matrix4();
    this.allowOriginalVisualFallback = false;
    this.character = null;
    this.characterKeys = new Set();
    this.characterMoveDirection = new THREE.Vector3();

    this.initRenderer();
    this.initAssetLoaders();
    this.initScene();
    this.initPhysics();
    this.createWorldObjects();
    this.bindEvents();
    this.applyUrlOptions();

    this.setStatus("Ready");
    requestAnimationFrame((time) => this.frame(time));
  }

  initRenderer() {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.domElement.className = "sim-canvas";
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    this.renderer.shadowMap.enabled = SHADOWS_ENABLED;
    if (SHADOWS_ENABLED) {
      this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    }
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.container.appendChild(this.renderer.domElement);
  }

  initAssetLoaders() {
    this.gltfLoader = new GLTFLoader();
    this.gltfLoader.setMeshoptDecoder(MeshoptDecoder);

    this.dracoLoader = new DRACOLoader();
    this.dracoLoader.setDecoderPath(`${THREE_EXAMPLES_BASE}libs/draco/gltf/`);
    this.gltfLoader.setDRACOLoader(this.dracoLoader);

    this.ktx2Loader = new KTX2Loader();
    this.ktx2Loader.setTranscoderPath(`${THREE_EXAMPLES_BASE}libs/basis/`);
    this.ktx2Loader.detectSupport(this.renderer);
    this.gltfLoader.setKTX2Loader(this.ktx2Loader);

    this.colladaLoader = new ColladaLoader();
    this.fbxLoader = new FBXLoader();
    this.stlLoader = new STLLoader();
  }

  initScene() {
    this.scene = new THREE.Scene();
    this.scene.background = null;

    this.camera = new THREE.PerspectiveCamera(
      46,
      this.container.clientWidth / this.container.clientHeight,
      0.1,
      100,
    );
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(3.7, -5.4, 3.2);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0.25, -0.75, 0.55);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.mouseButtons.LEFT = null;
    this.controls.mouseButtons.MIDDLE = THREE.MOUSE.ROTATE;
    this.controls.mouseButtons.RIGHT = THREE.MOUSE.PAN;
    this.controls.update();

    const hemi = new THREE.HemisphereLight(0xffffff, 0x6f7782, 2.4);
    this.scene.add(hemi);

    const sun = new THREE.DirectionalLight(0xffffff, 2.2);
    sun.position.set(-3.5, 7, 4);
    sun.castShadow = SHADOWS_ENABLED;
    if (SHADOWS_ENABLED) {
      sun.shadow.mapSize.set(2048, 2048);
      sun.shadow.camera.near = 0.5;
      sun.shadow.camera.far = 20;
      sun.shadow.camera.left = -8;
      sun.shadow.camera.right = 8;
      sun.shadow.camera.top = 8;
      sun.shadow.camera.bottom = -8;
    }
    this.scene.add(sun);

    const grid = new THREE.GridHelper(14, 28, 0xaab4c0, 0xd8dee5);
    grid.rotation.x = Math.PI / 2;
    grid.position.z = 0.002;
    this.scene.add(grid);
    this.grid = grid;

    const groundMesh = new THREE.Mesh(
      new THREE.BoxGeometry(14, 10, 0.1),
      new THREE.MeshStandardMaterial({
        color: 0xdfe5ea,
        roughness: 0.9,
        metalness: 0.0,
      }),
    );
    groundMesh.position.set(0, 0, -0.05);
    groundMesh.receiveShadow = SHADOWS_ENABLED;
    this.scene.add(groundMesh);
    this.groundMesh = groundMesh;

    this.debugGeometry = new THREE.BufferGeometry();
    this.debugLines = new THREE.LineSegments(
      this.debugGeometry,
      new THREE.LineBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0.72,
        depthTest: true,
      }),
    );
    this.scene.add(this.debugLines);
  }

  initPhysics() {
    this.world = new RAPIER.World({ x: 0, y: 0, z: -9.81 });
    this.world.timestep = FIXED_STEP;
  }

  createWorldObjects() {
    const groundBody = this.world.createRigidBody(
      RAPIER.RigidBodyDesc.fixed().setTranslation(0, 0, -0.05),
    );
    this.world.createCollider(RAPIER.ColliderDesc.cuboid(7, 5, 0.05), groundBody);

    this.sampleRecords.push(this.createRigidBox({
      name: "free-box-cuboid",
      size: new THREE.Vector3(0.82, 0.82, 0.82),
      position: new THREE.Vector3(-0.75, -0.15, 2.4),
      color: 0x2871cc,
      collider: "cuboid",
    }));

    this.sampleRecords.push(this.createRigidBox({
      name: "free-box-convex-hull",
      size: new THREE.Vector3(0.72, 0.72, 0.72),
      position: new THREE.Vector3(0.1, 0.2, 3.6),
      color: 0xe05a47,
      collider: "convexHull",
    }));
  }

  getSceneGroundCenterPosition(target = new THREE.Vector3()) {
    const groundZ = Number.isFinite(this.generatedSceneGroundZ) ? this.generatedSceneGroundZ : 0;
    if (this.generatedSceneBounds && !this.generatedSceneBounds.isEmpty()) {
      this.generatedSceneBounds.getCenter(target);
      target.z = groundZ;
      return target;
    }

    target.copy(CHARACTER_START_POSITION);
    target.z = groundZ;
    return target;
  }

  async loadWalkCharacter() {
    const fbx = await this.fbxLoader.loadAsync(CHARACTER_WALK_FBX_URL);
    prepareCharacterVisual(fbx);
    const spawnPosition = this.getSceneGroundCenterPosition();

    const group = new THREE.Group();
    group.name = "walk-character-controller";
    group.position.copy(spawnPosition);

    const frame = new THREE.Group();
    frame.name = "walk-character-z-up-frame";
    frame.rotation.x = Math.PI / 2;
    frame.add(fbx);
    group.add(frame);
    this.scene.add(group);
    normalizeCharacterVisual(frame, CHARACTER_HEIGHT);

    const walkClip = fbx.animations?.[0];
    if (!walkClip) {
      throw new Error("Walking.fbx does not contain an animation clip.");
    }
    stabilizeMixamoRootMotion(walkClip);
    const kickClip = await this.loadCharacterKickClip();

    const mixer = new THREE.AnimationMixer(fbx);
    const walkAction = mixer.clipAction(walkClip);
    walkAction.setLoop(THREE.LoopRepeat, Infinity);
    walkAction.play();
    walkAction.paused = true;

    const kickAction = kickClip ? mixer.clipAction(kickClip) : null;
    if (kickAction) {
      kickAction.setLoop(THREE.LoopOnce, 1);
      kickAction.clampWhenFinished = false;
      kickAction.enabled = false;
      mixer.addEventListener("finished", (event) => {
        if (event.action === kickAction) {
          this.finishCharacterKick();
        }
      });
    }

    const desc = RAPIER.RigidBodyDesc.kinematicPositionBased
      ? RAPIER.RigidBodyDesc.kinematicPositionBased()
      : RAPIER.RigidBodyDesc.dynamic();
    desc.setCanSleep?.(false);
    desc.setTranslation(spawnPosition.x, spawnPosition.y, spawnPosition.z);
    const body = this.world.createRigidBody(desc);

    const colliderDesc = RAPIER.ColliderDesc.capsule
      ? RAPIER.ColliderDesc.capsule(CHARACTER_CAPSULE_HALF_HEIGHT, CHARACTER_CAPSULE_RADIUS)
      : RAPIER.ColliderDesc.cuboid(CHARACTER_CAPSULE_RADIUS, CHARACTER_CAPSULE_RADIUS, CHARACTER_HEIGHT / 2);
    colliderDesc
      .setTranslation(0, 0, CHARACTER_CAPSULE_CENTER_Z)
      .setFriction(1.2)
      .setRestitution(0.0);
    if (RAPIER.ColliderDesc.capsule) {
      colliderDesc.setRotation(
        quaternionToRapier(new THREE.Quaternion().setFromEuler(new THREE.Euler(Math.PI / 2, 0, 0))),
      );
    }
    const collider = this.world.createCollider(colliderDesc, body);
    const kickBodyDesc = RAPIER.RigidBodyDesc.kinematicPositionBased
      ? RAPIER.RigidBodyDesc.kinematicPositionBased()
      : RAPIER.RigidBodyDesc.dynamic();
    kickBodyDesc.setCanSleep?.(false);
    kickBodyDesc.setTranslation(
      spawnPosition.x,
      spawnPosition.y,
      spawnPosition.z + CHARACTER_KICK_HEIGHT,
    );
    const kickBody = this.world.createRigidBody(kickBodyDesc);

    this.character = {
      group,
      frame,
      visual: fbx,
      mixer,
      action: walkAction,
      walkAction,
      kickAction,
      body,
      collider,
      kickBody,
      kickCollider: null,
      kickElapsed: 0,
      kickDuration: kickClip?.duration ?? CHARACTER_KICK_FALLBACK_DURATION,
      kickActive: false,
      kickHitRecords: new Set(),
      footPosition: spawnPosition.clone(),
      initialFootPosition: spawnPosition.clone(),
      moving: false,
    };
  }

  async loadSelectedCharacter() {
    const selection = this.characterSelect?.value ?? "mixamo-walk-kick";
    if (selection !== "mixamo-walk-kick") {
      this.setCharacterStatus(`Unsupported character: ${selection}`);
      return;
    }

    this.loadCharacterButton.disabled = true;
    try {
      this.clearCharacter({ quiet: true });
      this.setCharacterStatus("Loading Mixamo character");
      await this.loadWalkCharacter();
      this.setCharacterStatus("Loaded. Use WASD to walk, Space to kick.");
    } catch (error) {
      console.error(error);
      this.clearCharacter({ quiet: true });
      this.setCharacterStatus(`Character failed: ${shortError(error)}`);
    } finally {
      this.loadCharacterButton.disabled = false;
    }
  }

  clearCharacter({ quiet = false } = {}) {
    if (!this.character) {
      this.characterKeys.clear();
      if (!quiet) {
        this.setCharacterStatus("No character loaded.");
      }
      return;
    }

    const character = this.character;
    this.removeCharacterKickCollider();
    character.mixer?.stopAllAction?.();
    this.scene.remove(character.group);
    disposeObject3D(character.group);
    if (character.body) {
      this.world.removeRigidBody?.(character.body);
    }
    if (character.kickBody) {
      this.world.removeRigidBody?.(character.kickBody);
    }
    this.character = null;
    this.characterKeys.clear();
    if (!quiet) {
      this.setCharacterStatus("Character cleared.");
    }
  }

  async loadCharacterKickClip() {
    try {
      const fbx = await this.fbxLoader.loadAsync(CHARACTER_KICK_FBX_URL);
      const clip = fbx.animations?.[0];
      if (!clip) {
        throw new Error("Kick.fbx does not contain an animation clip.");
      }
      stabilizeMixamoRootMotion(clip);
      return clip;
    } catch (error) {
      console.warn(`Kick animation unavailable: ${shortError(error)}`);
      return null;
    }
  }

  createRigidBox({ name, size, position, color, collider, type = "dynamic", draggable = true }) {
    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(size.x, size.y, size.z),
      new THREE.MeshStandardMaterial({
        color,
        roughness: 0.62,
        metalness: 0.03,
        emissive: 0x000000,
      }),
    );
    mesh.name = name;
    mesh.castShadow = SHADOWS_ENABLED;
    mesh.receiveShadow = SHADOWS_ENABLED;

    const desc = type === "fixed" ? RAPIER.RigidBodyDesc.fixed() : RAPIER.RigidBodyDesc.dynamic();
    desc.setCanSleep?.(false);
    desc.setTranslation(position.x, position.y, position.z);
    const body = this.world.createRigidBody(desc);

    const colliderDesc =
      collider === "convexHull"
        ? this.createConvexHullColliderDescFromMesh(mesh)
        : RAPIER.ColliderDesc.cuboid(size.x / 2, size.y / 2, size.z / 2);

    colliderDesc.setDensity(1.0).setFriction(0.72).setRestitution(0.08);
    const rapierCollider = this.world.createCollider(colliderDesc, body);

    this.scene.add(mesh);

    const record = {
      name,
      mesh,
      body,
      draggable,
      type,
      initial: {
        type: type === "fixed" ? RAPIER.RigidBodyType.Fixed : RAPIER.RigidBodyType.Dynamic,
        position: position.clone(),
        rotation: new THREE.Quaternion(),
      },
    };
    record.collider = rapierCollider;
    mesh.userData.physicsRecord = record;
    this.physicsObjects.push(record);
    this.bodyToRecord.set(body.handle, record);
    this.colliderToRecord.set(rapierCollider.handle, record);

    return record;
  }

  createRigidMeshBody(mesh, options = {}) {
    const position = (options.position ?? mesh.position ?? new THREE.Vector3()).clone();
    const rotation = (options.rotation ?? mesh.quaternion ?? new THREE.Quaternion()).clone();
    const type = options.type ?? "dynamic";
    const draggable = options.draggable ?? type !== "fixed";

    mesh.name = options.name ?? mesh.name ?? "rigid-mesh";
    mesh.castShadow = options.castShadow ?? SHADOWS_ENABLED;
    mesh.receiveShadow = options.receiveShadow ?? SHADOWS_ENABLED;

    const desc = type === "fixed" ? RAPIER.RigidBodyDesc.fixed() : RAPIER.RigidBodyDesc.dynamic();
    desc.setCanSleep?.(false);
    desc.setTranslation(position.x, position.y, position.z);
    desc.setRotation(quaternionToRapier(rotation));

    const body = this.world.createRigidBody(desc);
    const colliderDesc =
      options.collider === "convexMesh"
        ? this.createConvexMeshColliderDescFromMesh(mesh)
        : this.createConvexHullColliderDescFromMesh(mesh);

    colliderDesc
      .setDensity(options.density ?? 1.0)
      .setFriction(options.friction ?? 0.72)
      .setRestitution(options.restitution ?? 0.08);
    const rapierCollider = this.world.createCollider(colliderDesc, body);

    mesh.position.copy(position);
    mesh.quaternion.copy(rotation);
    this.scene.add(mesh);

    const record = {
      name: mesh.name,
      mesh,
      body,
      collider: rapierCollider,
      draggable,
      type,
      initial: {
        type: type === "fixed" ? RAPIER.RigidBodyType.Fixed : RAPIER.RigidBodyType.Dynamic,
        position: position.clone(),
        rotation: rotation.clone(),
      },
    };
    mesh.userData.physicsRecord = record;
    this.physicsObjects.push(record);
    this.bodyToRecord.set(body.handle, record);
    this.colliderToRecord.set(rapierCollider.handle, record);

    return record;
  }

  async loadGeneratedSceneFromInput() {
    const manifestUrl = this.sceneSelect.value.trim() || DEFAULT_SCENE_MANIFEST;
    if (!manifestUrl) {
      this.setGeneratedStatus("No scene manifest selected");
      return;
    }
    await this.loadGeneratedScene(manifestUrl);
  }

  async loadGeneratedScene(manifestUrl) {
    this.loadSceneButton.disabled = true;
    this.setGeneratedStatus("Fetching manifest");
    this.setStatus("Loading generated scene");

    try {
      const manifestResponse = await fetch(manifestUrl);
      if (!manifestResponse.ok) {
        throw new Error(`Manifest HTTP ${manifestResponse.status}`);
      }

      const resolvedManifestUrl = manifestResponse.url || new URL(manifestUrl, window.location.href).href;
      const rawManifest = await manifestResponse.json();
      const generatedScene = normalizeGeneratedSceneManifest(rawManifest);
      const webAssetInfo = await this.applyWebVisualAssets(generatedScene, rawManifest, resolvedManifestUrl);
      const generatedGroundZ = generatedScene.groundZ ?? 0;
      if (!generatedScene.objects.length) {
        throw new Error("Manifest does not contain loadable SAPIEN export objects.");
      }

      this.clearGeneratedScene();
      this.generatedSceneGroundZ = generatedGroundZ;
      this.removeRecords(this.sampleRecords);
      this.sampleRecords = [];

      const bounds = new THREE.Box3();
      for (let i = 0; i < generatedScene.objects.length; i += 1) {
        const sceneObject = generatedScene.objects[i];
        this.setGeneratedStatus(`Loading ${i + 1}/${generatedScene.objects.length}: ${sceneObject.name}`);
        const record = await this.createGeneratedSceneObject(sceneObject, resolvedManifestUrl);
        this.generatedRecords.push(record);
        bounds.expandByObject(record.mesh);
      }

      this.refreshGeneratedObjectList();
      this.generatedSceneBounds = bounds.isEmpty() ? null : bounds.clone();
      this.createGeneratedSceneBoundary(this.generatedSceneBounds);
      if (!bounds.isEmpty()) {
        this.fitCameraToBox(bounds);
      }

      const poseText = generatedScene.poseMode === "settled" ? "settled poses" : "initial poses";
      const colliderCount = this.generatedRecords.reduce((sum, record) => sum + record.colliders.length, 0);
      const colliderStats = summarizeColliderSources(this.generatedRecords);
      const colliderText = `${colliderCount} colliders: ${formatColliderStats(colliderStats)}`;
      const loadedWebAssetCount = this.generatedRecords.filter((record) => record.visualSource === "web_assets").length;
      const visualText = loadedWebAssetCount
        ? `, ${loadedWebAssetCount}/${generatedScene.objects.length} compressed GLB`
        : "";
      this.setGeneratedStatus(
        `${generatedScene.objects.length} objects${visualText}, ${colliderText}, ${poseText}`,
      );
      this.setStatus("Generated scene ready");
      return true;
    } catch (error) {
      console.error(error);
      this.generatedSceneBounds = null;
      this.generatedSceneGroundZ = 0;
      this.setGeneratedStatus(`Load failed: ${shortError(error)}`);
      this.setStatus("Scene load failed");
      return false;
    } finally {
      this.loadSceneButton.disabled = false;
    }
  }

  async applyWebVisualAssets(generatedScene, rawManifest, manifestUrl) {
    let used = generatedScene.objects.filter((object) => object.visualSource === "web_assets").length;
    if (used === generatedScene.objects.length) {
      return { used, manifestLoaded: false };
    }

    const webManifestResult = await this.loadGeneratedWebAssetsManifest(rawManifest, manifestUrl);
    const webManifest = webManifestResult?.manifest;
    if (!webManifest) {
      return { used, manifestLoaded: false };
    }

    const maps = webAssetObjectMaps(webManifest);
    const visualFrame = webAssetsVisualFrame(webManifest);
    for (const sceneObject of generatedScene.objects) {
      if (sceneObject.visualSource === "web_assets") {
        continue;
      }

      const webObject = webAssetForSceneObject(sceneObject, maps);
      const webVisualPath = webObject?.web_glb ?? webObject?.web_asset_glb ?? webObject?.visual_path;
      if (!webVisualPath) {
        continue;
      }

      sceneObject.originalVisualPath = sceneObject.visualPath;
      sceneObject.visualPath = webVisualPath;
      sceneObject.visualFrame = visualFrame;
      sceneObject.visualSource = "web_assets";
      sceneObject.webAsset = webObject;
    }

    used = generatedScene.objects.filter((object) => object.visualSource === "web_assets").length;
    return { used, manifestLoaded: true, url: webManifestResult.url };
  }

  async loadGeneratedWebAssetsManifest(rawManifest, manifestUrl) {
    const candidates = [];
    const explicitPath = rawManifest.outputs?.web_assets_manifest
      ?? rawManifest.web_assets_manifest
      ?? rawManifest.web_assets?.manifest;
    if (explicitPath) {
      try {
        candidates.push(resolveAssetUrl(explicitPath, manifestUrl));
      } catch (error) {
        console.debug("Ignoring invalid web_assets manifest path:", error);
      }
    }

    const inferredUrl = webAssetsManifestUrlFromSceneManifestUrl(manifestUrl);
    if (inferredUrl) {
      candidates.push(inferredUrl);
    }

    for (const url of uniqueStrings(candidates)) {
      try {
        const response = await fetch(url, { cache: "no-cache" });
        if (!response.ok) {
          continue;
        }
        return { manifest: await response.json(), url: response.url || url };
      } catch (error) {
        console.debug(`web_assets manifest unavailable: ${url}`, error);
      }
    }

    return null;
  }

  async createGeneratedSceneObject(sceneObject, manifestUrl) {
    const rawPose = transformFromMatrixRows(sceneObject.poseMatrix);
    const pose = decomposeTransformMatrix(
      new THREE.Matrix4().copy(this.webSceneTransform).multiply(rawPose.matrix),
    );
    const { visualGroup, visualSource } = await this.loadGeneratedVisualGroup(sceneObject, manifestUrl, pose);

    visualGroup.name = sceneObject.name;
    visualGroup.position.copy(pose.position);
    visualGroup.quaternion.copy(pose.rotation);
    visualGroup.updateMatrixWorld(true);

    const desc = RAPIER.RigidBodyDesc.dynamic();
    desc.setCanSleep?.(false);
    desc.setTranslation(pose.position.x, pose.position.y, pose.position.z);
    desc.setRotation(quaternionToRapier(pose.rotation));
    desc.setLinearDamping?.(10.0);
    desc.setAngularDamping?.(30.0);
    const body = this.world.createRigidBody(desc);

    const colliders = [];
    const colliderSources = [];
    const collisionPaths = sceneObject.collisionPaths.length
      ? sceneObject.collisionPaths
      : [sceneObject.collisionPath].filter(Boolean);

    for (const collisionPath of collisionPaths) {
      const collisionUrl = resolveAssetUrl(collisionPath, manifestUrl);
      try {
        const colliderResult = await this.loadConvexColliderDesc(collisionUrl, sceneObject.name);
        colliderResult.desc.setDensity(2.5).setFriction(4.5).setRestitution(0.0);
        colliders.push(this.world.createCollider(colliderResult.desc, body));
        colliderSources.push(colliderResult.source);
      } catch (error) {
        console.warn(`Skipping collider ${collisionPath}: ${shortError(error)}`);
      }
    }

    if (!colliders.length) {
      const fallbackColliderDesc = this.createLocalBoundingBoxColliderDesc(visualGroup);
      fallbackColliderDesc.setDensity(2.5).setFriction(4.5).setRestitution(0.0);
      colliders.push(this.world.createCollider(fallbackColliderDesc, body));
      colliderSources.push("bbox");
    }

    this.scene.add(visualGroup);

    const record = {
      name: sceneObject.name,
      mesh: visualGroup,
      body,
      collider: colliders[0],
      colliders,
      draggable: true,
      type: "dynamic",
      generatedSceneObject: true,
      colliderSources,
      metadata: sceneObject,
      visualSource,
      initial: {
        type: RAPIER.RigidBodyType.Dynamic,
        position: pose.position.clone(),
        rotation: pose.rotation.clone(),
      },
    };
    visualGroup.userData.physicsRecord = record;
    this.physicsObjects.push(record);
    this.bodyToRecord.set(body.handle, record);
    for (const collider of colliders) {
      this.colliderToRecord.set(collider.handle, record);
    }
    return record;
  }

  async loadGeneratedVisualGroup(sceneObject, manifestUrl, pose) {
    const attempts = [
      {
        path: sceneObject.visualPath,
        visualFrame: sceneObject.visualFrame,
        source: sceneObject.visualSource,
      },
    ];

    if (
      this.allowOriginalVisualFallback
      && sceneObject.visualSource === "web_assets"
      && sceneObject.originalVisualPath
      && sceneObject.originalVisualPath !== sceneObject.visualPath
    ) {
      attempts.push({
        path: sceneObject.originalVisualPath,
        visualFrame: VISUAL_FRAME_ACTOR,
        source: "sapien_export",
      });
    }

    let lastError = null;
    for (const attempt of attempts) {
      try {
        const visualUrl = resolveAssetUrl(attempt.path, manifestUrl);
        const visualGroup = await this.loadVisualGroup(visualUrl, sceneObject.name, {
          actorPoseMatrix: pose.matrix,
          visualFrame: attempt.visualFrame,
        });
        if (attempt.source !== sceneObject.visualSource) {
          console.warn(
            `Falling back to original visual for ${sceneObject.name}: ${shortError(lastError)}`,
          );
        }
        return { visualGroup, visualSource: attempt.source };
      } catch (error) {
        lastError = error;
        console.warn(`Visual load failed (${attempt.path}): ${shortError(error)}`);
      }
    }

    throw lastError ?? new Error(`Visual load failed: ${sceneObject.name}`);
  }

  async loadVisualGroup(url, name, options = {}) {
    const gltf = await this.loadGltfAsset(url);
    const content = gltf.scene || new THREE.Group();
    content.applyMatrix4(GLTF_VISUAL_TO_BLENDER_Z_UP);
    if (options.visualFrame === VISUAL_FRAME_WORLD) {
      content.applyMatrix4(this.webSceneTransform);
      const inverseActorPose = new THREE.Matrix4()
        .copy(options.actorPoseMatrix ?? new THREE.Matrix4())
        .invert();
      content.applyMatrix4(inverseActorPose);
    }

    const group = new THREE.Group();
    group.name = name;
    group.add(content);
    group.traverse((child) => {
      if (!child.isMesh) {
        return;
      }
      child.castShadow = SHADOWS_ENABLED;
      child.receiveShadow = SHADOWS_ENABLED;
      const materials = Array.isArray(child.material) ? child.material : [child.material];
      for (const material of materials) {
        if (material) {
          material.side = THREE.DoubleSide;
        }
      }
    });
    return group;
  }

  async loadGltfAsset(url) {
    const compatUrl = await this.createMeshoptCompatObjectUrl(url).catch((compatError) => {
      console.debug(`Meshopt compatibility patch unavailable for ${url}:`, compatError);
      return "";
    });

    if (!compatUrl) {
      return await this.gltfLoader.loadAsync(url);
    }

    try {
      console.warn(`Loading GLB with EXT_meshopt compatibility: ${url}`);
      return await this.gltfLoader.loadAsync(compatUrl);
    } catch (error) {
      throw new Error(`EXT_meshopt compatibility load failed: ${shortError(error)}`);
    } finally {
      URL.revokeObjectURL(compatUrl);
    }
  }

  async createMeshoptCompatObjectUrl(url) {
    if (!/\.glb(?:[?#].*)?$/i.test(new URL(url, window.location.href).pathname)) {
      return "";
    }

    const response = await fetch(url, { cache: "no-cache" });
    if (!response.ok) {
      throw new Error(`GLB HTTP ${response.status}`);
    }

    const buffer = await response.arrayBuffer();
    const bytes = new Uint8Array(buffer);
    if (bytes.byteLength < 20 || readAscii(bytes, 0, 4) !== "glTF") {
      return "";
    }

    let offset = 12;
    const decoder = new TextDecoder();
    const encoder = new TextEncoder();
    while (offset + 8 <= bytes.byteLength) {
      const chunkLength = readUint32LE(bytes, offset);
      const chunkType = readAscii(bytes, offset + 4, 4);
      const chunkStart = offset + 8;
      const chunkEnd = chunkStart + chunkLength;
      if (chunkEnd > bytes.byteLength) {
        return "";
      }

      if (chunkType === "JSON") {
        const jsonText = decoder.decode(bytes.subarray(chunkStart, chunkEnd));
        if (!jsonText.includes("KHR_meshopt_compression")) {
          return "";
        }

        const patchedJson = jsonText.replaceAll("KHR_meshopt_compression", "EXT_meshopt_compression");
        const patchedBytes = encoder.encode(patchedJson);
        if (patchedBytes.byteLength > chunkLength) {
          throw new Error("Patched GLB JSON is longer than the original chunk.");
        }
        bytes.fill(0x20, chunkStart, chunkEnd);
        bytes.set(patchedBytes, chunkStart);
        return URL.createObjectURL(new Blob([bytes], { type: "model/gltf-binary" }));
      }

      offset = chunkEnd;
    }

    return "";
  }

  async loadConvexColliderDesc(url, name) {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Collision mesh HTTP ${response.status}: ${url}`);
    }

    const { vertices, indices } = parseObjMeshData(await response.text());
    if (vertices.length < 9) {
      throw new Error(`Collision mesh has too few vertices: ${name}`);
    }

    let desc = null;
    if (indices.length >= 3) {
      desc = RAPIER.ColliderDesc.convexMesh(vertices, indices);
      if (desc) {
        return { desc, source: "convexMesh" };
      }
    }

    desc = RAPIER.ColliderDesc.convexHull(vertices);
    if (!desc) {
      throw new Error(`Rapier failed to build convex collider: ${name}`);
    }
    return { desc, source: "convexHull" };
  }

  createLocalBoundingBoxColliderDesc(object) {
    const box = new THREE.Box3().setFromObject(object);
    const center = new THREE.Vector3();
    const size = new THREE.Vector3();
    box.getCenter(center);
    box.getSize(size);

    object.worldToLocal(center);
    const desc = RAPIER.ColliderDesc.cuboid(
      Math.max(size.x / 2, 0.01),
      Math.max(size.y / 2, 0.01),
      Math.max(size.z / 2, 0.01),
    );
    return desc.setTranslation(center.x, center.y, center.z);
  }

  createGeneratedSceneBoundary(bounds) {
    this.clearGeneratedSceneBoundary();
    if (!bounds || bounds.isEmpty()) {
      return;
    }

    const center = new THREE.Vector3();
    const size = new THREE.Vector3();
    bounds.getCenter(center);
    bounds.getSize(size);

    const width = Math.max(size.x * SCENE_BOUNDARY_SCALE, SCENE_BOUNDARY_MIN_SIZE);
    const depth = Math.max(size.y * SCENE_BOUNDARY_SCALE, SCENE_BOUNDARY_MIN_SIZE);
    const height = Math.max(
      Math.max(bounds.max.z, size.z) * SCENE_BOUNDARY_SCALE,
      SCENE_BOUNDARY_MIN_HEIGHT,
    );
    const halfWidth = width / 2;
    const halfDepth = depth / 2;
    const wallHalf = SCENE_BOUNDARY_WALL_THICKNESS / 2;
    const wallZ = height / 2;

    const walls = [
      {
        name: "air-wall-left",
        position: new THREE.Vector3(center.x - halfWidth - wallHalf, center.y, wallZ),
        halfExtents: new THREE.Vector3(wallHalf, halfDepth + SCENE_BOUNDARY_WALL_THICKNESS, wallZ),
      },
      {
        name: "air-wall-right",
        position: new THREE.Vector3(center.x + halfWidth + wallHalf, center.y, wallZ),
        halfExtents: new THREE.Vector3(wallHalf, halfDepth + SCENE_BOUNDARY_WALL_THICKNESS, wallZ),
      },
      {
        name: "air-wall-front",
        position: new THREE.Vector3(center.x, center.y - halfDepth - wallHalf, wallZ),
        halfExtents: new THREE.Vector3(halfWidth + SCENE_BOUNDARY_WALL_THICKNESS, wallHalf, wallZ),
      },
      {
        name: "air-wall-back",
        position: new THREE.Vector3(center.x, center.y + halfDepth + wallHalf, wallZ),
        halfExtents: new THREE.Vector3(halfWidth + SCENE_BOUNDARY_WALL_THICKNESS, wallHalf, wallZ),
      },
      {
        name: "air-wall-ceiling",
        position: new THREE.Vector3(center.x, center.y, height + wallHalf),
        halfExtents: new THREE.Vector3(
          halfWidth + SCENE_BOUNDARY_WALL_THICKNESS,
          halfDepth + SCENE_BOUNDARY_WALL_THICKNESS,
          wallHalf,
        ),
      },
    ];

    this.generatedBoundaryRecords = walls.map((wall) => this.createFixedBoundaryCollider(wall));
  }

  createFixedBoundaryCollider({ name, position, halfExtents }) {
    const desc = RAPIER.RigidBodyDesc.fixed().setTranslation(position.x, position.y, position.z);
    const body = this.world.createRigidBody(desc);
    const colliderDesc = RAPIER.ColliderDesc
      .cuboid(halfExtents.x, halfExtents.y, halfExtents.z)
      .setFriction(4.5)
      .setRestitution(0.0);
    const collider = this.world.createCollider(colliderDesc, body);
    const mesh = new THREE.Object3D();
    mesh.name = name;

    const record = {
      name,
      mesh,
      body,
      collider,
      colliders: [collider],
      draggable: false,
      type: "fixed",
      sceneBoundary: true,
      initial: {
        type: RAPIER.RigidBodyType.Fixed,
        position: position.clone(),
        rotation: new THREE.Quaternion(),
      },
    };
    this.physicsObjects.push(record);
    this.bodyToRecord.set(body.handle, record);
    this.colliderToRecord.set(collider.handle, record);
    return record;
  }

  clearGeneratedSceneBoundary() {
    this.removeRecords(this.generatedBoundaryRecords);
    this.generatedBoundaryRecords = [];
  }

  clearLoadedScene() {
    this.clearGeneratedScene();
    this.setStatus("Scene cleared");
  }

  clearGeneratedScene() {
    if (this.drag && (
      this.generatedRecords.includes(this.drag.record)
      || this.generatedBoundaryRecords.includes(this.drag.record)
    )) {
      this.clearActiveDrag();
    }
    this.clearGeneratedSceneBoundary();
    this.removeRecords(this.generatedRecords);
    this.generatedRecords = [];
    this.generatedSceneBounds = null;
    this.generatedSceneGroundZ = 0;
    this.refreshGeneratedObjectList();
    this.setGeneratedStatus("Generated scene cleared");
  }

  removeRecords(records) {
    if (!records.length) {
      return;
    }

    const removeSet = new Set(records);
    this.physicsObjects = this.physicsObjects.filter((record) => !removeSet.has(record));

    for (const record of records) {
      this.bodyToRecord.delete(record.body.handle);
      for (const collider of record.colliders ?? [record.collider].filter(Boolean)) {
        this.colliderToRecord.delete(collider.handle);
      }
      this.scene.remove(record.mesh);
      disposeObject3D(record.mesh);
      this.world.removeRigidBody?.(record.body);
    }
  }

  refreshGeneratedObjectList() {
    // The scene panel intentionally shows only the scene selector and summary.
  }

  fitCameraToBox(box) {
    const center = new THREE.Vector3();
    const size = new THREE.Vector3();
    box.getCenter(center);
    box.getSize(size);

    const radius = Math.max(size.length() * 0.55, 0.8);
    const fov = THREE.MathUtils.degToRad(this.camera.fov);
    const distance = Math.max(radius / Math.sin(fov / 2), 2.4);
    const direction = new THREE.Vector3(1.2, -1.7, 0.95).normalize();

    this.controls.target.copy(center);
    this.camera.position.copy(center).addScaledVector(direction, distance);
    this.camera.near = Math.max(distance / 100, 0.01);
    this.camera.far = Math.max(distance * 8, 20);
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  async loadUrdfRobot(url, options = {}) {
    this.setRobotStatus(`Fetching ${url.split("/").pop()}`);
    const urdfText = await this.loadUrdfText(url, options.fallbackText);
    this.setRobotStatus("Parsing URDF");
    const parsed = this.parseUrdfRobot(urdfText);
    this.setRobotStatus("Creating robot");
    const robot = await this.createRapierRobotFromUrdf(parsed, { ...options, urdfUrl: url });
    this.robots.push(robot);
    this.buildRobotBasePanel(robot);
    this.buildJointPanel(robot);
    this.setRobotStatus(`${robot.name}: ${robot.jointControls.length} controllable joints`);
    return robot;
  }

  async loadSelectedRobot() {
    const selection = this.robotSelect?.value ?? "franka";
    if (selection !== "franka") {
      this.setRobotStatus(`Unsupported robot: ${selection}`);
      return;
    }

    this.loadRobotButton.disabled = true;
    try {
      this.clearRobot({ quiet: true });
      await this.loadUrdfRobot(FRANKA_URDF_URL, {
        position: this.getSceneGroundCenterPosition(),
        packageMap: {
          franka_description: "./assets/urdf/franka/franka_description/",
        },
      });
    } catch (error) {
      console.error(error);
      this.setRobotStatus(`URDF failed: ${shortError(error)}`);
    } finally {
      this.loadRobotButton.disabled = false;
    }
  }

  clearRobot({ quiet = false } = {}) {
    if (!this.robots.length) {
      this.robotBaseControlsEl.replaceChildren();
      this.jointControlsEl.replaceChildren();
      if (!quiet) {
        this.setRobotStatus("No robot loaded.");
      }
      return;
    }

    const records = this.robots.flatMap((robot) => [...robot.links.values()]);
    if (this.drag && records.includes(this.drag.record)) {
      this.clearActiveDrag();
    }
    this.removeRecords(records);
    this.robots = [];
    this.robotJointControls = [];
    this.robotBaseControlsEl.replaceChildren();
    this.jointControlsEl.replaceChildren();
    if (!quiet) {
      this.setRobotStatus("Robot cleared.");
    }
  }

  async loadUrdfText(url, fallbackText) {
    try {
      const response = await fetchWithTimeout(url, URDF_FETCH_TIMEOUT_MS);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      return await response.text();
    } catch (error) {
      if (!fallbackText) {
        throw error;
      }
      console.warn(`URDF fetch failed, using embedded fallback: ${shortError(error)}`);
      this.setRobotStatus(`Using embedded sample (${shortError(error)})`);
      return fallbackText;
    }
  }

  parseUrdfRobot(urdfText) {
    const document = new DOMParser().parseFromString(urdfText, "application/xml");
    const error = document.querySelector("parsererror");
    if (error) {
      throw new Error(error.textContent.trim());
    }

    const robotNode = document.querySelector("robot");
    if (!robotNode) {
      throw new Error("URDF does not contain a <robot> root.");
    }

    const links = new Map();
    for (const linkNode of childElements(robotNode, "link")) {
      const name = linkNode.getAttribute("name");
      if (!name) {
        continue;
      }

      links.set(name, {
        name,
        visuals: this.parseUrdfGeometryBlocks(linkNode, "visual"),
        collisions: this.parseUrdfGeometryBlocks(linkNode, "collision"),
      });
    }

    const joints = [];
    for (const jointNode of childElements(robotNode, "joint")) {
      const parent = firstChildElement(jointNode, "parent")?.getAttribute("link");
      const child = firstChildElement(jointNode, "child")?.getAttribute("link");
      if (!parent || !child) {
        continue;
      }

      joints.push({
        name: jointNode.getAttribute("name") || `${parent}_to_${child}`,
        type: jointNode.getAttribute("type") || "fixed",
        parent,
        child,
        origin: this.parseUrdfOrigin(firstChildElement(jointNode, "origin")),
        axis: parseVector3(firstChildElement(jointNode, "axis")?.getAttribute("xyz"), [1, 0, 0]),
        limit: this.parseUrdfLimit(firstChildElement(jointNode, "limit")),
        mimic: this.parseUrdfMimic(firstChildElement(jointNode, "mimic")),
      });
    }

    return {
      name: robotNode.getAttribute("name") || "urdf_robot",
      links,
      joints,
    };
  }

  parseUrdfGeometryBlocks(linkNode, tagName) {
    const blocks = [];
    for (const blockNode of childElements(linkNode, tagName)) {
      const geometryNode = firstChildElement(blockNode, "geometry");
      if (!geometryNode) {
        continue;
      }

      const boxNode = firstChildElement(geometryNode, "box");
      const cylinderNode = firstChildElement(geometryNode, "cylinder");
      const sphereNode = firstChildElement(geometryNode, "sphere");
      const meshNode = firstChildElement(geometryNode, "mesh");

      let geometry = null;
      if (boxNode) {
        geometry = {
          type: "box",
          size: parseVector3(boxNode.getAttribute("size"), [0.2, 0.2, 0.2]),
        };
      } else if (cylinderNode) {
        geometry = {
          type: "cylinder",
          radius: parseNumber(cylinderNode.getAttribute("radius"), 0.1),
          length: parseNumber(cylinderNode.getAttribute("length"), 0.4),
        };
      } else if (sphereNode) {
        geometry = {
          type: "sphere",
          radius: parseNumber(sphereNode.getAttribute("radius"), 0.1),
        };
      } else if (meshNode) {
        geometry = {
          type: "mesh",
          filename: meshNode.getAttribute("filename"),
          scale: parseVector3(meshNode.getAttribute("scale"), [1, 1, 1]),
        };
      }

      if (geometry) {
        blocks.push({
          origin: this.parseUrdfOrigin(firstChildElement(blockNode, "origin")),
          geometry,
        });
      }
    }
    return blocks;
  }

  parseUrdfOrigin(originNode) {
    const xyz = parseVector3(originNode?.getAttribute("xyz"), [0, 0, 0]);
    const rpy = parseVector3(originNode?.getAttribute("rpy"), [0, 0, 0]);
    return {
      xyz,
      rpy,
      quaternion: quaternionFromRpy(rpy.x, rpy.y, rpy.z),
    };
  }

  parseUrdfLimit(limitNode) {
    return {
      lower: parseNumber(limitNode?.getAttribute("lower"), -Math.PI),
      upper: parseNumber(limitNode?.getAttribute("upper"), Math.PI),
      effort: parseNumber(limitNode?.getAttribute("effort"), 30),
      velocity: parseNumber(limitNode?.getAttribute("velocity"), 4),
    };
  }

  parseUrdfMimic(mimicNode) {
    if (!mimicNode?.getAttribute("joint")) {
      return null;
    }
    return {
      joint: mimicNode.getAttribute("joint"),
      multiplier: parseNumber(mimicNode.getAttribute("multiplier"), 1),
      offset: parseNumber(mimicNode.getAttribute("offset"), 0),
    };
  }

  async createRapierRobotFromUrdf(parsed, options = {}) {
    const rootPosition = options.position?.clone?.() ?? new THREE.Vector3();
    const rootRotation = options.rotation?.clone?.() ?? new THREE.Quaternion();
    const rootNames = findRootLinks(parsed);
    const childJoints = groupJointsByParent(parsed.joints);
    const robot = {
      name: parsed.name,
      basePosition: rootPosition.clone(),
      baseRotation: rootRotation.clone(),
      initialBasePosition: rootPosition.clone(),
      initialBaseRotation: rootRotation.clone(),
      packageMap: options.packageMap ?? {},
      urdfUrl: options.urdfUrl ?? "",
      rootNames,
      childJoints,
      links: new Map(),
      rootRecords: [],
      joints: new Map(),
      jointControls: [],
      controlsByJointName: new Map(),
      markers: [],
    };

    const createLinkTree = async (linkName, transform, isRoot = false) => {
      const link = parsed.links.get(linkName);
      if (!link) {
        return null;
      }

      const record = await this.createUrdfLinkBody({
        link,
        position: transform.position,
        rotation: transform.rotation,
        fixed: isRoot,
        robot,
      });
      robot.links.set(linkName, record);
      if (isRoot) {
        robot.rootRecords.push(record);
      }

      for (const jointDef of childJoints.get(linkName) ?? []) {
        const childTransform = composeTransform(transform, jointDef.origin);
        const childRecord = await createLinkTree(jointDef.child, childTransform, false);
        if (!childRecord) {
          continue;
        }

        const parentRecord = robot.links.get(jointDef.parent);
        const jointRecord = this.createUrdfJoint({
          jointDef,
          parentRecord,
          childRecord,
          robot,
        });
        robot.joints.set(jointDef.name, jointRecord);
        if (jointRecord.control) {
          robot.jointControls.push(jointRecord.control);
          robot.controlsByJointName.set(jointDef.name, jointRecord.control);
          this.robotJointControls.push(jointRecord.control);
        }
      }

      return record;
    };

    for (const rootName of rootNames) {
      await createLinkTree(
        rootName,
        {
          position: rootPosition,
          rotation: rootRotation,
        },
        true,
      );
    }

    this.updateRobotForwardKinematics(robot, { updateInitial: true });
    return robot;
  }

  async createUrdfLinkBody({ link, position, rotation, fixed, robot }) {
    const group = new THREE.Group();
    group.name = `${robot.name}:${link.name}`;
    group.position.copy(position);
    group.quaternion.copy(rotation);

    const sourceBlocks = link.collisions.length ? link.collisions : link.visuals;
    const visualBlocks = link.visuals.length ? link.visuals : sourceBlocks;
    const desc = RAPIER.RigidBodyDesc.kinematicPositionBased
      ? RAPIER.RigidBodyDesc.kinematicPositionBased()
      : (fixed ? RAPIER.RigidBodyDesc.fixed() : RAPIER.RigidBodyDesc.dynamic());
    desc.setCanSleep?.(false);
    desc.setTranslation(position.x, position.y, position.z);
    desc.setRotation(quaternionToRapier(rotation));
    desc.setLinearDamping?.(0.15);
    desc.setAngularDamping?.(0.25);
    const body = this.world.createRigidBody(desc);

    const colliders = [];
    const colliderSources = [];

    for (const block of visualBlocks) {
      const visual = await this.createUrdfVisualMesh(block, link.name, robot);
      if (visual) {
        group.add(visual);
      }
    }

    for (const block of sourceBlocks) {
      const colliderResults = await this.createUrdfColliderDescs(block, robot);
      if (!colliderResults.length) {
        continue;
      }
      for (const colliderResult of colliderResults) {
        colliderResult.desc.setFriction(0.72).setRestitution(0.03);
        colliders.push(this.world.createCollider(colliderResult.desc, body));
        colliderSources.push(colliderResult.source);
      }
    }

    this.scene.add(group);

    const record = {
      name: link.name,
      mesh: group,
      body,
      collider: colliders[0],
      colliders,
      draggable: false,
      type: "kinematic",
      robotLink: true,
      colliderSources,
      initial: {
        type: RAPIER.RigidBodyType.KinematicPositionBased,
        position: position.clone(),
        rotation: rotation.clone(),
      },
    };
    group.userData.physicsRecord = record;
    this.physicsObjects.push(record);
    this.bodyToRecord.set(body.handle, record);
    for (const collider of colliders) {
      this.colliderToRecord.set(collider.handle, record);
    }
    return record;
  }

  async createUrdfVisualMesh(block, linkName, robot) {
    const material = new THREE.MeshStandardMaterial({
      color: colorForName(linkName),
      roughness: 0.62,
      metalness: 0.02,
    });
    let geometry = null;
    let extraRotation = new THREE.Quaternion();

    if (block.geometry.type === "box") {
      const size = block.geometry.size;
      geometry = new THREE.BoxGeometry(size.x, size.y, size.z);
    } else if (block.geometry.type === "cylinder") {
      geometry = new THREE.CylinderGeometry(block.geometry.radius, block.geometry.radius, block.geometry.length, 28);
      extraRotation = new THREE.Quaternion().setFromEuler(new THREE.Euler(Math.PI / 2, 0, 0));
    } else if (block.geometry.type === "sphere") {
      geometry = new THREE.SphereGeometry(block.geometry.radius, 28, 16);
    } else if (block.geometry.type === "mesh") {
      try {
        const object = await this.loadUrdfMeshObject(block, robot, material);
        object.position.copy(block.origin.xyz);
        object.quaternion.copy(block.origin.quaternion);
        object.scale.copy(block.geometry.scale);
        return object;
      } catch (error) {
        console.warn(`URDF visual mesh failed (${block.geometry.filename}): ${shortError(error)}`);
        return this.createMissingMeshPlaceholder(block);
      }
    } else {
      return this.createMissingMeshPlaceholder(block);
    }

    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.copy(block.origin.xyz);
    mesh.quaternion.copy(block.origin.quaternion).multiply(extraRotation);
    mesh.castShadow = SHADOWS_ENABLED;
    mesh.receiveShadow = SHADOWS_ENABLED;
    return mesh;
  }

  createMissingMeshPlaceholder(block) {
    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(0.18, 0.18, 0.18),
      new THREE.MeshStandardMaterial({ color: 0x8b96a5, roughness: 0.8 }),
    );
    mesh.position.copy(block.origin.xyz);
    mesh.quaternion.copy(block.origin.quaternion);
    mesh.castShadow = SHADOWS_ENABLED;
    mesh.receiveShadow = SHADOWS_ENABLED;
    return mesh;
  }

  async loadUrdfMeshObject(block, robot, fallbackMaterial) {
    const url = this.resolveUrdfMeshUrl(block.geometry.filename, robot);
    const extension = getUrlExtension(url);
    let object = null;

    if (extension === "dae") {
      const collada = await this.colladaLoader.loadAsync(url);
      object = collada.scene ?? new THREE.Group();
    } else if (extension === "stl") {
      const geometry = await this.stlLoader.loadAsync(url);
      geometry.computeVertexNormals();
      object = new THREE.Mesh(geometry, fallbackMaterial);
    } else if (extension === "glb" || extension === "gltf") {
      const gltf = await this.gltfLoader.loadAsync(url);
      object = gltf.scene ?? new THREE.Group();
    } else if (extension === "obj") {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`OBJ HTTP ${response.status}`);
      }
      const meshData = parseObjMeshData(await response.text());
      const geometry = geometryFromMeshData(meshData);
      object = new THREE.Mesh(geometry, fallbackMaterial);
    } else {
      throw new Error(`Unsupported URDF mesh format: ${extension || "unknown"}`);
    }

    object.traverse((child) => {
      if (!child.isMesh) {
        return;
      }
      child.castShadow = SHADOWS_ENABLED;
      child.receiveShadow = SHADOWS_ENABLED;
      if (!child.material) {
        child.material = fallbackMaterial;
      }
      const materials = Array.isArray(child.material) ? child.material : [child.material];
      for (const material of materials) {
        material.side = THREE.DoubleSide;
      }
    });
    return object;
  }

  resolveUrdfMeshUrl(filename, robot) {
    const raw = String(filename ?? "").trim();
    if (!raw) {
      throw new Error("URDF mesh filename is empty.");
    }
    if (/^(https?:|data:|blob:)/i.test(raw)) {
      return raw;
    }

    const normalized = raw.replace(/\\/g, "/");
    if (normalized.startsWith("package://")) {
      const packagePath = normalized.slice("package://".length);
      const slashIndex = packagePath.indexOf("/");
      const packageName = slashIndex >= 0 ? packagePath.slice(0, slashIndex) : packagePath;
      const relativePath = slashIndex >= 0 ? packagePath.slice(slashIndex + 1) : "";
      const mappedBase = robot.packageMap?.[packageName];
      if (!mappedBase) {
        throw new Error(`No package map for ${packageName}`);
      }
      return new URL(relativePath, ensureTrailingSlash(new URL(mappedBase, window.location.href).href)).href;
    }

    if (normalized.startsWith("/")) {
      return new URL(normalized, window.location.origin).href;
    }

    return new URL(normalized, new URL(robot.urdfUrl, window.location.href)).href;
  }

  async createUrdfColliderDescs(block, robot) {
    let desc = null;
    let rotation = block.origin.quaternion;

    if (block.geometry.type === "box") {
      const size = block.geometry.size;
      desc = RAPIER.ColliderDesc.cuboid(size.x / 2, size.y / 2, size.z / 2);
    } else if (block.geometry.type === "cylinder") {
      desc = RAPIER.ColliderDesc.cylinder(block.geometry.length / 2, block.geometry.radius);
      rotation = block.origin.quaternion
        .clone()
        .multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(Math.PI / 2, 0, 0)));
    } else if (block.geometry.type === "sphere") {
      desc = RAPIER.ColliderDesc.ball(block.geometry.radius);
    } else if (block.geometry.type === "mesh") {
      const coacdResults = await this.loadUrdfCoacdColliderDescs(block, robot);
      if (coacdResults.length) {
        return coacdResults;
      }

      try {
        const meshData = await this.loadUrdfCollisionMeshData(block, robot);
        desc = this.createConvexColliderDescFromMeshData(meshData.vertices, meshData.indices);
      } catch (error) {
        console.warn(`URDF collision mesh failed (${block.geometry.filename}): ${shortError(error)}`);
        return [];
      }
    } else {
      return [];
    }

    if (!desc) {
      return [];
    }

    desc.setTranslation(block.origin.xyz.x, block.origin.xyz.y, block.origin.xyz.z);
    desc.setRotation(quaternionToRapier(rotation));
    return [{ desc, source: block.geometry.type }];
  }

  async loadUrdfCoacdColliderDescs(block, robot) {
    const sourceUrl = this.resolveUrdfMeshUrl(block.geometry.filename, robot);
    const manifestUrl = this.resolveUrdfCoacdManifestUrl(sourceUrl);
    if (!manifestUrl) {
      return [];
    }

    try {
      const response = await fetch(manifestUrl);
      if (!response.ok) {
        return [];
      }

      const manifest = await response.json();
      const partPaths = Array.isArray(manifest.parts) ? manifest.parts : [];
      const results = [];
      for (const partPath of partPaths) {
        const partUrl = new URL(partPath, manifestUrl).href;
        const partResponse = await fetch(partUrl);
        if (!partResponse.ok) {
          console.warn(`URDF CoACD part failed (${partPath}): HTTP ${partResponse.status}`);
          continue;
        }

        const meshData = scaleMeshData(parseObjMeshData(await partResponse.text()), block.geometry.scale);
        const desc = this.createConvexColliderDescFromMeshData(meshData.vertices, meshData.indices);
        desc.setTranslation(block.origin.xyz.x, block.origin.xyz.y, block.origin.xyz.z);
        desc.setRotation(quaternionToRapier(block.origin.quaternion));
        results.push({ desc, source: "coacd" });
      }
      return results;
    } catch (error) {
      console.warn(`URDF CoACD manifest failed (${block.geometry.filename}): ${shortError(error)}`);
      return [];
    }
  }

  resolveUrdfCoacdManifestUrl(sourceUrl) {
    const url = new URL(sourceUrl, window.location.href);
    const path = url.pathname;
    const marker = "/meshes/collision/";
    const index = path.lastIndexOf(marker);
    if (index < 0) {
      return null;
    }

    const fileName = decodeURIComponent(path.slice(index + marker.length).split("/").pop() ?? "");
    const stem = fileName.replace(/\.[^.]+$/, "");
    if (!stem) {
      return null;
    }

    const basePath = path.slice(0, index + "/meshes/".length);
    url.pathname = `${basePath}collision_coacd/${encodeURIComponent(stem)}/${URDF_COACD_MANIFEST_NAME}`;
    url.search = "";
    url.hash = "";
    return url.href;
  }

  async loadUrdfCollisionMeshData(block, robot) {
    const url = this.resolveUrdfMeshUrl(block.geometry.filename, robot);
    const extension = getUrlExtension(url);

    if (extension === "obj") {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`OBJ HTTP ${response.status}`);
      }
      const meshData = parseObjMeshData(await response.text());
      return scaleMeshData(meshData, block.geometry.scale);
    }

    if (extension === "stl") {
      const geometry = await this.stlLoader.loadAsync(url);
      return {
        vertices: this.extractGeometryVertices(geometry, block.geometry.scale),
        indices: this.extractGeometryIndices(geometry),
      };
    }

    const object = await this.loadUrdfMeshObject(block, robot, new THREE.MeshStandardMaterial());
    const meshData = extractObjectGeometryData(object);
    return scaleMeshData(meshData, block.geometry.scale);
  }

  createConvexColliderDescFromMeshData(vertices, indices) {
    if (!vertices?.length || vertices.length < 9) {
      throw new Error("Collision mesh has too few vertices.");
    }

    let desc = null;
    if (indices?.length >= 3) {
      desc = RAPIER.ColliderDesc.convexMesh(vertices, indices);
    }
    desc = desc ?? RAPIER.ColliderDesc.convexHull(vertices);
    if (!desc) {
      throw new Error("Rapier failed to build a convex collider.");
    }
    return desc;
  }

  createUrdfJoint({ jointDef, parentRecord, childRecord, robot }) {
    if (jointDef.type !== "revolute" && jointDef.type !== "continuous" && jointDef.type !== "prismatic") {
      return { joint: null, jointDef, control: null };
    }

    const control = {
      name: jointDef.name,
      joint: null,
      jointDef,
      robot,
      target: 0,
      initialTarget: 0,
      lower: jointDef.type === "continuous" ? -Math.PI : jointDef.limit.lower,
      upper: jointDef.type === "continuous" ? Math.PI : jointDef.limit.upper,
      kind: jointDef.type === "prismatic" ? "linear" : "angular",
      stiffness: jointDef.type === "prismatic"
        ? Math.max(80, jointDef.limit.effort * 20)
        : Math.max(24, jointDef.limit.effort * 3),
      damping: jointDef.type === "prismatic" ? 18 : 10,
      velocity: jointDef.limit.velocity,
      slider: null,
      valueLabel: null,
    };

    this.setUrdfJointTarget(control, 0);
    return { joint: null, jointDef, control };
  }

  addJointMarker(parentRecord, localPosition, controllable, robot) {
    return null;
  }

  buildRobotBasePanel(robot) {
    this.robotBaseControlsEl.replaceChildren();

    const title = document.createElement("div");
    title.className = "base-control-title";
    title.textContent = "Base xyz";

    const grid = document.createElement("div");
    grid.className = "base-control-grid";

    const axes = [
      ["x", "X"],
      ["y", "Y"],
      ["z", "Z"],
    ];
    const inputs = {};

    for (const [axis, labelText] of axes) {
      const label = document.createElement("label");
      label.className = "base-axis-label";
      label.textContent = labelText;

      const input = document.createElement("input");
      input.type = "number";
      input.step = "0.05";
      input.value = robot.basePosition[axis].toFixed(2);
      inputs[axis] = input;

      input.addEventListener("input", () => {
        const nextPosition = new THREE.Vector3(
          parseNumber(inputs.x.value, robot.basePosition.x),
          parseNumber(inputs.y.value, robot.basePosition.y),
          parseNumber(inputs.z.value, robot.basePosition.z),
        );
        this.setRobotBasePosition(robot, nextPosition);
      });

      label.appendChild(input);
      grid.appendChild(label);
    }

    robot.baseInputs = inputs;
    this.robotBaseControlsEl.append(title, grid);
  }

  setRobotBasePosition(robot, nextPosition) {
    if (nextPosition.distanceToSquared(robot.basePosition) < 1e-12) {
      return;
    }

    robot.basePosition.copy(nextPosition);
    this.updateRobotForwardKinematics(robot);
  }

  updateRobotForwardKinematics(robot, { updateInitial = false } = {}) {
    const visit = (linkName, transform) => {
      const record = robot.links.get(linkName);
      if (!record) {
        return;
      }

      this.setRobotLinkTransform(record, transform.position, transform.rotation, updateInitial);
      for (const jointDef of robot.childJoints.get(linkName) ?? []) {
        visit(jointDef.child, this.composeRobotJointTransform(transform, jointDef, robot));
      }
    };

    for (const rootName of robot.rootNames) {
      visit(rootName, {
        position: robot.basePosition.clone(),
        rotation: robot.baseRotation.clone(),
      });
    }
  }

  composeRobotJointTransform(parentTransform, jointDef, robot) {
    const jointPosition = parentTransform.position
      .clone()
      .add(jointDef.origin.xyz.clone().applyQuaternion(parentTransform.rotation));
    const jointRotation = parentTransform.rotation.clone().multiply(jointDef.origin.quaternion);
    const target = this.getRobotJointTarget(robot, jointDef);

    if (jointDef.type === "revolute" || jointDef.type === "continuous") {
      const axis = normalizedOrDefault(jointDef.axis, new THREE.Vector3(1, 0, 0));
      const motion = new THREE.Quaternion().setFromAxisAngle(axis, target);
      return {
        position: jointPosition,
        rotation: jointRotation.multiply(motion),
      };
    }

    if (jointDef.type === "prismatic") {
      const axisWorld = normalizedOrDefault(jointDef.axis, new THREE.Vector3(1, 0, 0))
        .applyQuaternion(jointRotation);
      return {
        position: jointPosition.addScaledVector(axisWorld, target),
        rotation: jointRotation,
      };
    }

    return {
      position: jointPosition,
      rotation: jointRotation,
    };
  }

  getRobotJointTarget(robot, jointDef, visited = new Set()) {
    const directControl = robot.controlsByJointName.get(jointDef.name);
    if (directControl) {
      return directControl.target;
    }

    if (jointDef.mimic && !visited.has(jointDef.name)) {
      visited.add(jointDef.name);
      const sourceJoint = robot.joints.get(jointDef.mimic.joint)?.jointDef;
      const sourceTarget = sourceJoint
        ? this.getRobotJointTarget(robot, sourceJoint, visited)
        : robot.controlsByJointName.get(jointDef.mimic.joint)?.target ?? 0;
      return sourceTarget * jointDef.mimic.multiplier + jointDef.mimic.offset;
    }

    return robot.controlsByJointName.get(jointDef.name)?.target ?? 0;
  }

  setRobotLinkTransform(record, position, rotation, updateInitial = false) {
    record.body.setBodyType?.(RAPIER.RigidBodyType.KinematicPositionBased, true);
    record.body.setNextKinematicTranslation?.(vectorToRapier(position));
    record.body.setNextKinematicRotation?.(quaternionToRapier(rotation));
    record.body.setTranslation(vectorToRapier(position), true);
    record.body.setRotation(quaternionToRapier(rotation), true);
    record.body.setLinvel(ZERO_VEC, true);
    record.body.setAngvel(ZERO_VEC, true);
    record.mesh.position.copy(position);
    record.mesh.quaternion.copy(rotation);
    if (updateInitial) {
      record.initial.position.copy(position);
      record.initial.rotation.copy(rotation);
    }
    if (this.physicsEnabled) {
      record.body.wakeUp();
    }
  }

  buildJointPanel(robot) {
    this.jointControlsEl.replaceChildren();

    if (!robot.jointControls.length) {
      const empty = document.createElement("div");
      empty.className = "robot-status";
      empty.textContent = "No revolute joints found";
      this.jointControlsEl.appendChild(empty);
      return;
    }

    for (const control of robot.jointControls) {
      const row = document.createElement("label");
      row.className = "joint-row";

      const label = document.createElement("div");
      label.className = "joint-label";

      const name = document.createElement("span");
      name.className = "joint-name";
      name.textContent = control.name;

      const value = document.createElement("span");
      value.className = "joint-value";

      const slider = document.createElement("input");
      slider.type = "range";
      const sliderScale = control.kind === "linear" ? 1 : RAD_TO_DEG;
      slider.min = String(control.lower * sliderScale);
      slider.max = String(control.upper * sliderScale);
      slider.step = control.kind === "linear" ? "0.001" : "1";
      slider.value = String(control.target * sliderScale);

      control.slider = slider;
      control.valueLabel = value;
      this.updateJointControlLabel(control);

      slider.addEventListener("input", () => {
        const sliderValue = Number(slider.value);
        this.setUrdfJointTarget(control, control.kind === "linear" ? sliderValue : sliderValue * DEG_TO_RAD);
      });

      label.append(name, value);
      row.append(label, slider);
      this.jointControlsEl.appendChild(row);
    }
  }

  setUrdfJointTarget(control, target) {
    const clamped = THREE.MathUtils.clamp(target, control.lower, control.upper);
    control.target = clamped;
    control.joint?.configureMotorPosition?.(clamped, control.stiffness, control.damping);
    if (control.robot) {
      this.updateRobotForwardKinematics(control.robot);
    }
    this.updateJointControlLabel(control);
  }

  updateJointControlLabel(control) {
    if (!control.valueLabel) {
      return;
    }
    control.valueLabel.textContent = control.kind === "linear"
      ? `${control.target.toFixed(3)} m`
      : `${(control.target * RAD_TO_DEG).toFixed(0)} deg`;
  }

  applyRobotJointTargets() {
    for (const robot of this.robots) {
      this.updateRobotForwardKinematics(robot);
    }
  }

  createConvexHullColliderDescFromMesh(mesh) {
    const vertices = this.extractGeometryVertices(mesh.geometry, mesh.scale);
    const desc = RAPIER.ColliderDesc.convexHull(vertices);
    if (!desc) {
      throw new Error(`Rapier failed to build a convex hull for mesh: ${mesh.name || "unnamed"}`);
    }
    return desc;
  }

  createConvexMeshColliderDescFromMesh(mesh) {
    const vertices = this.extractGeometryVertices(mesh.geometry, mesh.scale);
    const indices = this.extractGeometryIndices(mesh.geometry);
    const desc = RAPIER.ColliderDesc.convexMesh(vertices, indices);
    if (!desc) {
      throw new Error(`Rapier failed to build a convex mesh for mesh: ${mesh.name || "unnamed"}`);
    }
    return desc;
  }

  extractGeometryVertices(geometry, scale = new THREE.Vector3(1, 1, 1)) {
    const position = geometry.attributes.position;
    const vertices = new Float32Array(position.count * 3);
    for (let i = 0; i < position.count; i += 1) {
      vertices[i * 3] = position.getX(i) * scale.x;
      vertices[i * 3 + 1] = position.getY(i) * scale.y;
      vertices[i * 3 + 2] = position.getZ(i) * scale.z;
    }
    return vertices;
  }

  extractGeometryIndices(geometry) {
    if (!geometry.index) {
      const position = geometry.attributes.position;
      const indices = new Uint32Array(position.count);
      for (let i = 0; i < position.count; i += 1) {
        indices[i] = i;
      }
      return indices;
    }
    return new Uint32Array(geometry.index.array);
  }

  bindEvents() {
    this.resetButton.addEventListener("click", () => this.reset());
    this.physicsButton.addEventListener("click", () => this.togglePhysics());
    this.debugButton.addEventListener("click", () => this.toggleDebug());
    this.loadRobotButton.addEventListener("click", () => this.loadSelectedRobot());
    this.clearRobotButton.addEventListener("click", () => this.clearRobot());
    this.loadCharacterButton.addEventListener("click", () => this.loadSelectedCharacter());
    this.clearCharacterButton.addEventListener("click", () => this.clearCharacter());
    this.loadSceneButton.addEventListener("click", () => {
      this.loadGeneratedSceneFromInput();
    });
    this.clearSceneButton.addEventListener("click", () => this.clearLoadedScene());
    this.bindPanelToggles();

    const canvas = this.renderer.domElement;
    canvas.addEventListener("pointerdown", (event) => this.onPointerDown(event));
    canvas.addEventListener("pointermove", (event) => this.onPointerMove(event));
    canvas.addEventListener("pointerup", (event) => this.onPointerUp(event));
    canvas.addEventListener("pointercancel", (event) => this.onPointerUp(event));
    canvas.addEventListener("contextmenu", (event) => event.preventDefault());
    window.addEventListener("keydown", (event) => this.onKeyDown(event));
    window.addEventListener("keyup", (event) => this.onKeyUp(event));
    window.addEventListener("resize", () => this.resize());
  }

  bindPanelToggles() {
    for (const button of this.container.querySelectorAll("[data-panel-toggle]")) {
      const panel = button.closest("[data-collapsible-panel]");
      if (!panel) {
        continue;
      }

      const update = () => {
        const collapsed = panel.classList.contains("is-collapsed");
        button.textContent = collapsed
          ? button.dataset.collapsedSymbol || ">"
          : button.dataset.expandedSymbol || "<";
        const label = collapsed
          ? button.dataset.expandLabel || "Show panel"
          : button.dataset.collapseLabel || "Collapse panel";
        button.setAttribute("aria-label", label);
        button.setAttribute("aria-expanded", String(!collapsed));
        button.title = label;
      };

      button.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        panel.classList.toggle("is-collapsed");
        update();
      });
      update();
    }
  }

  applyUrlOptions() {
    const params = new URLSearchParams(window.location.search);
    const manifest = params.get("manifest");
    if (manifest) {
      this.selectSceneManifest(manifest);
    }

    this.allowOriginalVisualFallback = params.get("fallbackVisual") === "1"
      || params.has("fallbackVisual");

    const shouldLoadScene = params.get("autoload") === "1" || params.has("loadScene");
    if (shouldLoadScene) {
      window.setTimeout(async () => {
        await this.loadGeneratedSceneFromInput();
      }, 0);
    }
  }

  clearActiveDrag() {
    if (!this.drag) {
      return;
    }

    setObjectEmissive(this.drag.record.mesh, 0x000000);
    try {
      this.renderer.domElement.releasePointerCapture(this.drag.pointerId);
    } catch {
      // Pointer capture may already be released by the browser.
    }
    this.drag = null;
    this.container.style.cursor = "";
  }

  selectSceneManifest(manifestUrl) {
    const existing = [...this.sceneSelect.options].find((option) => option.value === manifestUrl);
    if (existing) {
      this.sceneSelect.value = manifestUrl;
      return;
    }

    const option = document.createElement("option");
    option.value = manifestUrl;
    option.textContent = manifestUrl.split("/").slice(-3).join("/");
    this.sceneSelect.appendChild(option);
    this.sceneSelect.value = manifestUrl;
  }

  onPointerDown(event) {
    if (event.button !== 0 || this.drag) {
      return;
    }

    const hit = this.pick(event);
    if (!hit) {
      return;
    }

    event.preventDefault();
    this.renderer.domElement.setPointerCapture(event.pointerId);

    const record = hit.record;
    const cameraNormal = this.camera.getWorldDirection(new THREE.Vector3()).normalize();
    const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(cameraNormal, hit.point);
    const bodyPosition = new THREE.Vector3().copy(record.mesh.position);
    const offset = bodyPosition.sub(hit.point);

    record.body.setBodyType(RAPIER.RigidBodyType.KinematicPositionBased, true);
    record.body.setLinvel(ZERO_VEC, true);
    record.body.setAngvel(ZERO_VEC, true);

    setObjectEmissive(record.mesh, 0x17324d);

    this.drag = {
      pointerId: event.pointerId,
      record,
      plane,
      offset,
      target: record.mesh.position.clone(),
      samples: [],
    };
    this.sampleDragTarget();
    this.container.style.cursor = "grabbing";
  }

  onPointerMove(event) {
    if (!this.drag || event.pointerId !== this.drag.pointerId) {
      return;
    }
    event.preventDefault();
    this.updateDragTarget(event);
  }

  onPointerUp(event) {
    if (!this.drag || event.pointerId !== this.drag.pointerId) {
      return;
    }

    event.preventDefault();
    this.updateDragTarget(event);

    const { record, target } = this.drag;
    const releaseVelocity = this.physicsEnabled ? this.estimateReleaseVelocity() : new THREE.Vector3();

    record.body.setTranslation(vectorToRapier(target), true);
    record.body.setBodyType(
      this.physicsEnabled ? RAPIER.RigidBodyType.Dynamic : RAPIER.RigidBodyType.Fixed,
      true,
    );
    record.body.setLinvel(vectorToRapier(releaseVelocity), true);
    record.body.setAngvel(ZERO_VEC, true);
    if (this.physicsEnabled) {
      record.body.wakeUp();
    }
    setObjectEmissive(record.mesh, 0x000000);

    this.renderer.domElement.releasePointerCapture(event.pointerId);
    this.drag = null;
    this.container.style.cursor = "";
  }

  pick(event) {
    this.updatePointer(event);
    this.raycaster.setFromCamera(this.pointer, this.camera);

    const ray = new RAPIER.Ray(
      vectorToRapier(this.raycaster.ray.origin),
      vectorToRapier(this.raycaster.ray.direction),
    );
    const hit = this.world.castRay(
      ray,
      100,
      true,
      undefined,
      undefined,
      undefined,
      undefined,
      (collider) => {
        const record = this.colliderToRecord.get(collider.handle);
        return Boolean(record?.draggable && (record.body.isDynamic() || record.generatedSceneObject));
      },
    );

    if (!hit) {
      return null;
    }

    const record = this.colliderToRecord.get(hit.collider.handle);
    if (!record) {
      return null;
    }

    const toi = hit.timeOfImpact ?? hit.toi ?? 0;
    return {
      record,
      point: this.raycaster.ray.at(toi, new THREE.Vector3()),
    };
  }

  updatePointer(event) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  }

  updateDragTarget(event) {
    this.updatePointer(event);
    this.raycaster.setFromCamera(this.pointer, this.camera);

    const point = new THREE.Vector3();
    if (!this.raycaster.ray.intersectPlane(this.drag.plane, point)) {
      return;
    }

    point.add(this.drag.offset);
    point.z = Math.max(point.z, 0.08);
    this.drag.target.copy(point);
    this.sampleDragTarget();
  }

  sampleDragTarget() {
    if (!this.drag) {
      return;
    }

    const now = performance.now() / 1000;
    this.drag.samples.push({ time: now, point: this.drag.target.clone() });
    while (this.drag.samples.length > 2 && now - this.drag.samples[0].time > 0.18) {
      this.drag.samples.shift();
    }
  }

  estimateReleaseVelocity() {
    const samples = this.drag?.samples ?? [];
    if (samples.length < 2) {
      return new THREE.Vector3();
    }

    const first = samples[0];
    const last = samples[samples.length - 1];
    const dt = Math.max(last.time - first.time, 1 / 240);
    const velocity = last.point.clone().sub(first.point).divideScalar(dt);

    if (velocity.length() > RELEASE_SPEED_LIMIT) {
      velocity.setLength(RELEASE_SPEED_LIMIT);
    }
    return velocity;
  }

  onKeyDown(event) {
    if (isTypingTarget(event.target)) {
      return;
    }

    if (event.code === "Space") {
      if (!event.repeat) {
        this.triggerCharacterKick();
      }
      event.preventDefault();
      return;
    }

    if (CHARACTER_MOVE_KEYS.has(event.code)) {
      this.characterKeys.add(event.code);
      event.preventDefault();
    }
  }

  onKeyUp(event) {
    if (!CHARACTER_MOVE_KEYS.has(event.code)) {
      return;
    }

    this.characterKeys.delete(event.code);
    event.preventDefault();
  }

  updateCharacterController(delta, { physicsStep = false } = {}) {
    if (!this.character) {
      return;
    }

    const direction = this.getCharacterMoveDirection(this.characterMoveDirection);
    const isMoving = direction.lengthSq() > 1e-8;
    this.character.moving = isMoving;
    if (isMoving) {
      this.character.footPosition.addScaledVector(direction, CHARACTER_SPEED * delta);
      this.character.footPosition.z = this.generatedSceneGroundZ ?? 0;
      this.faceCharacterDirection(direction);
    }

    this.character.group.position.copy(this.character.footPosition);
    if (this.character.kickActive) {
      this.updateCharacterKickHitbox(delta, { physicsStep });
    } else {
      this.removeCharacterKickCollider();
    }

    if (physicsStep) {
      this.character.body.setNextKinematicTranslation(vectorToRapier(this.character.footPosition));
    } else {
      this.character.body.setTranslation(vectorToRapier(this.character.footPosition), true);
      this.character.body.setLinvel(ZERO_VEC, true);
      this.character.body.setAngvel(ZERO_VEC, true);
    }
  }

  updateCharacterAnimation(delta) {
    if (!this.character?.mixer) {
      return;
    }

    if (this.character.walkAction && !this.character.kickActive) {
      this.character.walkAction.paused = !this.character.moving;
    }
    this.character.mixer.update(delta);
  }

  triggerCharacterKick() {
    if (!this.character?.kickAction || this.character.kickActive) {
      return;
    }

    this.character.kickActive = true;
    this.character.kickElapsed = 0;
    this.character.kickHitRecords.clear();
    this.character.walkAction?.setEffectiveWeight(0);
    if (this.character.walkAction) {
      this.character.walkAction.paused = true;
    }
    this.character.kickAction.enabled = true;
    this.character.kickAction.reset();
    this.character.kickAction.setEffectiveTimeScale(1);
    this.character.kickAction.setEffectiveWeight(1);
    this.character.kickAction.play();
  }

  finishCharacterKick() {
    if (!this.character) {
      return;
    }

    this.character.kickActive = false;
    this.character.kickElapsed = 0;
    this.removeCharacterKickCollider();
    if (this.character.kickAction) {
      this.character.kickAction.stop();
      this.character.kickAction.enabled = false;
    }
    if (this.character.walkAction) {
      this.character.walkAction.enabled = true;
      this.character.walkAction.setEffectiveWeight(1);
      this.character.walkAction.play();
      this.character.walkAction.paused = !this.character.moving;
    }
  }

  updateCharacterKickHitbox(delta, { physicsStep = false } = {}) {
    const character = this.character;
    if (!character?.kickBody) {
      return;
    }

    character.kickElapsed += delta;
    const duration = Math.max(character.kickDuration, 0.2);
    const normalizedTime = THREE.MathUtils.clamp(character.kickElapsed / duration, 0, 1);
    const power = kickPhaseEnvelope(normalizedTime);
    if (power <= 0.001) {
      this.removeCharacterKickCollider();
      if (normalizedTime >= 1) {
        this.finishCharacterKick();
      }
      return;
    }

    const forward = this.getCharacterForward(new THREE.Vector3());
    const reach = CHARACTER_KICK_MAX_REACH * power;
    const radius = THREE.MathUtils.lerp(CHARACTER_KICK_MIN_RADIUS, CHARACTER_KICK_MAX_RADIUS, power);
    const halfHeight = Math.max(reach * 0.5, 0.05);
    const centerDistance = CHARACTER_KICK_START_DISTANCE + halfHeight;
    const center = character.footPosition
      .clone()
      .addScaledVector(forward, centerDistance);
    center.z += CHARACTER_KICK_HEIGHT;
    const rotation = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), forward);

    character.kickBody.setNextKinematicTranslation?.(vectorToRapier(center));
    character.kickBody.setNextKinematicRotation?.(quaternionToRapier(rotation));
    character.kickBody.setTranslation(vectorToRapier(center), true);
    character.kickBody.setRotation(quaternionToRapier(rotation), true);
    character.kickBody.setLinvel(ZERO_VEC, true);
    character.kickBody.setAngvel(ZERO_VEC, true);

    this.rebuildCharacterKickCollider(halfHeight, radius);
    if (physicsStep && power > 0.35) {
      this.applyCharacterKickImpulse(forward, power);
    }
  }

  rebuildCharacterKickCollider(halfHeight, radius) {
    this.removeCharacterKickCollider();
    if (!this.character?.kickBody) {
      return;
    }

    const desc = RAPIER.ColliderDesc.capsule
      ? RAPIER.ColliderDesc.capsule(halfHeight, radius)
      : RAPIER.ColliderDesc.cuboid(radius, halfHeight, radius);
    desc.setFriction(1.6).setRestitution(0.0);
    this.character.kickCollider = this.world.createCollider(desc, this.character.kickBody);
  }

  removeCharacterKickCollider() {
    if (!this.character?.kickCollider) {
      return;
    }

    this.world.removeCollider?.(this.character.kickCollider, true);
    this.character.kickCollider = null;
  }

  applyCharacterKickImpulse(forward, power) {
    const character = this.character;
    if (!character) {
      return;
    }

    for (const record of this.physicsObjects) {
      if (!record.body?.isDynamic?.() || record.robotLink) {
        continue;
      }
      if (character.kickHitRecords.has(record)) {
        continue;
      }

      const position = record.body.translation();
      const offset = new THREE.Vector3(
        position.x - character.footPosition.x,
        position.y - character.footPosition.y,
        position.z - character.footPosition.z,
      );
      const forwardDistance = offset.dot(forward);
      if (
        forwardDistance < CHARACTER_KICK_MIN_FORWARD_DISTANCE
        || forwardDistance > CHARACTER_KICK_START_DISTANCE + CHARACTER_KICK_MAX_REACH + CHARACTER_KICK_FORWARD_PADDING
      ) {
        continue;
      }

      const lateral = offset.clone().addScaledVector(forward, -forwardDistance);
      lateral.z = 0;
      const heightOk = position.z > character.footPosition.z - CHARACTER_KICK_LOWER_PADDING
        && position.z < character.footPosition.z + CHARACTER_HEIGHT * 0.95;
      if (!heightOk || lateral.length() > CHARACTER_KICK_MAX_RADIUS + CHARACTER_KICK_LATERAL_PADDING) {
        continue;
      }

      const impulse = forward
        .clone()
        .multiplyScalar(CHARACTER_KICK_IMPULSE * power)
        .addScaledVector(Z_UP, CHARACTER_KICK_UP_IMPULSE * power);
      record.body.applyImpulse?.(vectorToRapier(impulse), true);
      character.kickHitRecords.add(record);
    }
  }

  getCharacterMoveDirection(target) {
    target.set(0, 0, 0);
    if (!this.characterKeys.size) {
      return target;
    }

    const forward = this.camera.getWorldDirection(new THREE.Vector3());
    forward.z = 0;
    if (forward.lengthSq() < 1e-8) {
      forward.set(0, 1, 0);
    } else {
      forward.normalize();
    }

    const right = new THREE.Vector3().crossVectors(forward, Z_UP);
    right.z = 0;
    if (right.lengthSq() < 1e-8) {
      right.set(1, 0, 0);
    } else {
      right.normalize();
    }

    if (this.characterKeys.has("KeyW")) {
      target.add(forward);
    }
    if (this.characterKeys.has("KeyS")) {
      target.sub(forward);
    }
    if (this.characterKeys.has("KeyD")) {
      target.add(right);
    }
    if (this.characterKeys.has("KeyA")) {
      target.sub(right);
    }
    if (target.lengthSq() > 1e-8) {
      target.normalize();
    }
    return target;
  }

  faceCharacterDirection(direction) {
    const yaw = Math.atan2(direction.x, -direction.y);
    this.character.group.quaternion.setFromAxisAngle(Z_UP, yaw);
  }

  getCharacterForward(target) {
    target.set(0, -1, 0);
    if (!this.character) {
      return target;
    }

    return target.applyQuaternion(this.character.group.quaternion).setZ(0).normalize();
  }

  frame(timeMs) {
    const now = timeMs / 1000;
    const delta = this.lastFrameTime ? Math.min(now - this.lastFrameTime, 0.1) : FIXED_STEP;
    this.lastFrameTime = now;
    this.clockAccumulator += delta;

    if (this.physicsEnabled) {
      while (this.clockAccumulator >= FIXED_STEP) {
        if (this.drag) {
          this.drag.record.body.setNextKinematicTranslation(vectorToRapier(this.drag.target));
        }
        this.updateCharacterController(FIXED_STEP, { physicsStep: true });
        this.applyRobotJointTargets();
        this.world.step();
        this.clockAccumulator -= FIXED_STEP;
      }
    } else {
      this.clockAccumulator = 0;
      this.updateCharacterController(delta);
      if (this.drag) {
        this.drag.record.body.setTranslation(vectorToRapier(this.drag.target), true);
        this.drag.record.body.setLinvel(ZERO_VEC, true);
        this.drag.record.body.setAngvel(ZERO_VEC, true);
      }
    }

    this.updateCharacterAnimation(delta);
    this.syncMeshesFromPhysics();
    this.controls.update();
    this.updateDebugRender();
    this.renderer.render(this.scene, this.camera);
    requestAnimationFrame((time) => this.frame(time));
  }

  syncMeshesFromPhysics() {
    for (const record of this.physicsObjects) {
      const position = record.body.translation();
      const rotation = record.body.rotation();
      record.mesh.position.set(position.x, position.y, position.z);
      record.mesh.quaternion.set(rotation.x, rotation.y, rotation.z, rotation.w);
    }
  }

  updateDebugRender() {
    this.debugLines.visible = this.debugEnabled;
    if (!this.debugEnabled) {
      return;
    }

    const { vertices, colors } = this.world.debugRender();
    const rgb = new Float32Array((colors.length / 4) * 3);
    for (let i = 0, j = 0; i < colors.length; i += 4, j += 3) {
      rgb[j] = colors[i];
      rgb[j + 1] = colors[i + 1];
      rgb[j + 2] = colors[i + 2];
    }

    this.debugGeometry.setAttribute("position", new THREE.BufferAttribute(vertices, 3));
    this.debugGeometry.setAttribute("color", new THREE.BufferAttribute(rgb, 3));
    this.debugGeometry.computeBoundingSphere();
  }

  reset() {
    this.clearActiveDrag();
    this.clockAccumulator = 0;

    for (const robot of this.robots) {
      robot.basePosition.copy(robot.initialBasePosition);
      robot.baseRotation.copy(robot.initialBaseRotation);
      this.syncRobotBaseInputs(robot);
    }

    for (const record of this.physicsObjects) {
      if (!record.robotLink) {
        this.resetRecordToInitial(record);
      }
    }

    for (const control of this.robotJointControls) {
      this.setUrdfJointTarget(control, control.initialTarget);
      if (control.slider) {
        const sliderScale = control.kind === "linear" ? 1 : RAD_TO_DEG;
        control.slider.value = String(control.initialTarget * sliderScale);
      }
    }

    for (const robot of this.robots) {
      this.updateRobotForwardKinematics(robot);
    }
    this.resetCharacter();

    this.syncMeshesFromPhysics();
    this.updateDebugRender();
    this.setStatus("Scene reset");
  }

  resetRecordToInitial(record) {
    record.body.setBodyType(record.initial.type, true);
    record.body.setTranslation(vectorToRapier(record.initial.position), true);
    record.body.setRotation(quaternionToRapier(record.initial.rotation), true);
    record.body.setLinvel(ZERO_VEC, true);
    record.body.setAngvel(ZERO_VEC, true);
    if (this.physicsEnabled && record.initial.type !== RAPIER.RigidBodyType.Fixed) {
      record.body.wakeUp();
    }
    record.mesh.position.copy(record.initial.position);
    record.mesh.quaternion.copy(record.initial.rotation);
  }

  syncRobotBaseInputs(robot) {
    if (!robot.baseInputs) {
      return;
    }

    for (const axis of ["x", "y", "z"]) {
      if (robot.baseInputs[axis]) {
        robot.baseInputs[axis].value = robot.basePosition[axis].toFixed(2);
      }
    }
  }

  resetCharacter() {
    if (!this.character) {
      return;
    }

    this.character.footPosition.copy(this.character.initialFootPosition);
    this.character.group.position.copy(this.character.footPosition);
    this.character.group.quaternion.identity();
    this.character.body.setTranslation(vectorToRapier(this.character.footPosition), true);
    this.character.body.setLinvel(ZERO_VEC, true);
    this.character.body.setAngvel(ZERO_VEC, true);
    this.removeCharacterKickCollider();
    this.character.kickActive = false;
    this.character.kickElapsed = 0;
    this.character.kickHitRecords?.clear();
    if (this.character.kickAction) {
      this.character.kickAction.stop();
      this.character.kickAction.enabled = false;
    }
    this.character.walkAction?.reset();
    if (this.character.walkAction) {
      this.character.walkAction.enabled = true;
      this.character.walkAction.setEffectiveWeight(1);
      this.character.walkAction.play();
      this.character.walkAction.paused = true;
    }
    this.character.moving = false;
  }

  toggleDebug() {
    this.debugEnabled = !this.debugEnabled;
    this.debugButton.setAttribute("aria-pressed", String(this.debugEnabled));
    this.debugButton.textContent = this.debugEnabled ? "Colliders On" : "Colliders Off";
  }

  togglePhysics() {
    this.physicsEnabled = !this.physicsEnabled;
    this.physicsButton.setAttribute("aria-pressed", String(this.physicsEnabled));
    this.physicsButton.textContent = this.physicsEnabled ? "Physics On" : "Physics Off";
    this.clockAccumulator = 0;

    for (const record of this.physicsObjects) {
      record.body.setLinvel(ZERO_VEC, true);
      record.body.setAngvel(ZERO_VEC, true);
      if (this.physicsEnabled) {
        if (record.generatedSceneObject && !record.body.isDynamic()) {
          record.body.setBodyType(RAPIER.RigidBodyType.Dynamic, true);
        }
        record.body.wakeUp();
      }
    }

    this.setStatus(this.physicsEnabled ? "Physics running" : "Physics paused");
  }

  resize() {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  setStatus(text) {
    this.statusEl.textContent = text;
  }

  setGeneratedStatus(text) {
    this.generatedStatusEl.textContent = text;
  }

  setRobotStatus(text) {
    this.robotStatusEl.textContent = text;
  }

  setCharacterStatus(text) {
    this.characterStatusEl.textContent = text;
  }
}

function normalizeGeneratedSceneManifest(manifest) {
  const objects = [];
  let settledPoseCount = 0;

  for (const rawObject of manifest.objects ?? []) {
    const sapienExport = rawObject.sapien_export ?? rawObject;
    const webVisualPath = rawObject.web_asset_glb
      ?? rawObject.web_glb
      ?? rawObject.web_visual_path
      ?? rawObject.web_asset?.glb;
    const visualPath = webVisualPath || sapienExport.visual_path;
    if (!visualPath) {
      continue;
    }

    const finalPose = rawObject.sapien_final_pose?.final_pose;
    const poseMatrix = finalPose ?? sapienExport.initial_pose ?? rawObject.initial_pose;
    if (!poseMatrix) {
      continue;
    }

    if (finalPose) {
      settledPoseCount += 1;
    }

    objects.push({
      name: rawObject.final_3d_object_name ?? sapienExport.name ?? rawObject.name ?? `object_${objects.length}`,
      label: rawObject.semantic_label ?? rawObject.description ?? "",
      visualPath,
      originalVisualPath: webVisualPath ? sapienExport.visual_path ?? "" : "",
      visualFrame: webVisualPath ? VISUAL_FRAME_WORLD : VISUAL_FRAME_ACTOR,
      visualSource: webVisualPath ? "web_assets" : "sapien_export",
      collisionPath: sapienExport.collision_path ?? "",
      collisionPaths: Array.isArray(sapienExport.collision_paths)
        ? sapienExport.collision_paths.filter(Boolean)
        : [],
      collisionPartCount: sapienExport.collision_part_count ?? sapienExport.collision_paths?.length ?? 0,
      poseMatrix,
      poseSource: finalPose ? "settled" : "initial",
      maskId: rawObject.mask_id ?? null,
      bboxMin: sapienExport.bbox_min ?? rawObject.bbox_min ?? null,
      bboxMax: sapienExport.bbox_max ?? rawObject.bbox_max ?? null,
    });
  }

  return {
    objects,
    poseMode: settledPoseCount === objects.length && objects.length ? "settled" : "initial",
    groundZ: finiteOptionalNumber(
      manifest.ground_z
        ?? manifest.support?.ground_z
        ?? manifest.physics?.ground_z
        ?? manifest.sapien?.ground_z,
    ),
  };
}

function webAssetObjectMaps(webManifest) {
  const byMask = new Map();
  const byName = new Map();
  for (const item of webManifest.objects ?? []) {
    if (item?.mask_id !== null && item?.mask_id !== undefined) {
      byMask.set(Number(item.mask_id), item);
    }
    if (item?.name) {
      byName.set(String(item.name), item);
    }
    if (item?.final_3d_object_name) {
      byName.set(String(item.final_3d_object_name), item);
    }
  }
  return { byMask, byName };
}

function webAssetForSceneObject(sceneObject, maps) {
  if (sceneObject.maskId !== null && sceneObject.maskId !== undefined) {
    const byMask = maps.byMask.get(Number(sceneObject.maskId));
    if (byMask) {
      return byMask;
    }
  }
  return maps.byName.get(sceneObject.name) ?? null;
}

function webAssetsVisualFrame(webManifest) {
  const explicit = String(webManifest.visual_frame ?? webManifest.visualFrame ?? "").toLowerCase();
  if (/actor|local/.test(explicit)) {
    return VISUAL_FRAME_ACTOR;
  }
  return VISUAL_FRAME_WORLD;
}

function webAssetsManifestUrlFromSceneManifestUrl(manifestUrl) {
  const value = String(manifestUrl ?? "").trim();
  if (!value) {
    return "";
  }

  try {
    const url = new URL(value, window.location.href);
    const marker = "/results/";
    const markerIndex = url.pathname.indexOf(marker);
    if (markerIndex >= 0) {
      url.pathname = `${url.pathname.slice(0, markerIndex + marker.length)}web_assets/manifest.json`;
      url.search = "";
      url.hash = "";
      return url.href;
    }
    return new URL("web_assets/manifest.json", url).href;
  } catch {
    return "";
  }
}

function uniqueStrings(values) {
  return [...new Set(values.filter(Boolean).map((value) => String(value)))];
}

function resolveAssetUrl(assetPath, manifestUrl) {
  const rawPath = String(assetPath ?? "").trim();
  if (!rawPath) {
    throw new Error("Empty asset path");
  }
  if (/^(https?:|data:|blob:)/i.test(rawPath)) {
    return rawPath;
  }

  const normalized = rawPath.replace(/\\/g, "/");
  const sessionsIndex = normalized.indexOf("/sessions/");
  if (sessionsIndex >= 0) {
    return new URL(normalized.slice(sessionsIndex), window.location.origin).href;
  }

  const projectIndex = normalized.indexOf("/project_page/");
  if (projectIndex >= 0) {
    const relativePath = normalized.slice(projectIndex + "/project_page/".length);
    return new URL(relativePath, new URL("./", window.location.href)).href;
  }

  if (normalized.startsWith("/")) {
    return new URL(normalized, window.location.origin).href;
  }

  return new URL(normalized, manifestUrl).href;
}

function ensureTrailingSlash(value) {
  return value.endsWith("/") ? value : `${value}/`;
}

function getUrlExtension(url) {
  const pathname = new URL(url, window.location.href).pathname;
  const fileName = pathname.split("/").pop() ?? "";
  const dotIndex = fileName.lastIndexOf(".");
  return dotIndex >= 0 ? fileName.slice(dotIndex + 1).toLowerCase() : "";
}

function transformFromMatrixRows(rows) {
  return decomposeTransformMatrix(matrixFromRowMajor(rows));
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

function finiteOptionalNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function extractObjectGeometryData(object) {
  const vertices = [];
  const indices = [];
  const vertex = new THREE.Vector3();
  let vertexOffset = 0;

  object.updateMatrixWorld(true);
  object.traverse((child) => {
    if (!child.isMesh || !child.geometry?.attributes?.position) {
      return;
    }

    const geometry = child.geometry;
    const position = geometry.attributes.position;
    for (let i = 0; i < position.count; i += 1) {
      vertex.fromBufferAttribute(position, i).applyMatrix4(child.matrixWorld);
      vertices.push(vertex.x, vertex.y, vertex.z);
    }

    if (geometry.index) {
      for (let i = 0; i < geometry.index.count; i += 1) {
        indices.push(geometry.index.getX(i) + vertexOffset);
      }
    } else {
      for (let i = 0; i < position.count; i += 1) {
        indices.push(vertexOffset + i);
      }
    }
    vertexOffset += position.count;
  });

  return {
    vertices: new Float32Array(vertices),
    indices: new Uint32Array(indices),
  };
}

function geometryFromMeshData(meshData) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(meshData.vertices, 3));
  if (meshData.indices?.length) {
    geometry.setIndex(new THREE.BufferAttribute(meshData.indices, 1));
  }
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();
  return geometry;
}

function scaleMeshData(meshData, scale) {
  const scaled = new Float32Array(meshData.vertices.length);
  for (let i = 0; i < meshData.vertices.length; i += 3) {
    scaled[i] = meshData.vertices[i] * scale.x;
    scaled[i + 1] = meshData.vertices[i + 1] * scale.y;
    scaled[i + 2] = meshData.vertices[i + 2] * scale.z;
  }
  return {
    vertices: scaled,
    indices: meshData.indices,
  };
}

function parseObjMeshData(text) {
  const vertices = [];
  const indices = [];

  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) {
      continue;
    }

    const parts = line.split(/\s+/);
    if (parts[0] === "v" && parts.length >= 4) {
      const x = Number(parts[1]);
      const y = Number(parts[2]);
      const z = Number(parts[3]);
      if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
        vertices.push(x, y, z);
      }
    } else if (parts[0] === "f" && parts.length >= 4) {
      const face = [];
      for (let i = 1; i < parts.length; i += 1) {
        const token = parts[i].split("/")[0];
        const objIndex = Number(token);
        if (!Number.isInteger(objIndex) || objIndex === 0) {
          continue;
        }
        const zeroBasedIndex = objIndex > 0 ? objIndex - 1 : vertices.length / 3 + objIndex;
        if (zeroBasedIndex >= 0) {
          face.push(zeroBasedIndex);
        }
      }
      for (let i = 1; i + 1 < face.length; i += 1) {
        indices.push(face[0], face[i], face[i + 1]);
      }
    }
  }

  return {
    vertices: new Float32Array(vertices),
    indices: new Uint32Array(indices),
  };
}

function setObjectEmissive(object, color) {
  object.traverse((child) => {
    if (!child.isMesh) {
      return;
    }
    const materials = Array.isArray(child.material) ? child.material : [child.material];
    for (const material of materials) {
      material?.emissive?.setHex(color);
    }
  });
}

function summarizeColliderSources(records) {
  const counts = new Map();
  for (const record of records) {
    for (const source of record.colliderSources ?? []) {
      counts.set(source, (counts.get(source) ?? 0) + 1);
    }
  }
  return counts;
}

function formatColliderStats(counts) {
  const parts = [];
  for (const source of ["convexMesh", "convexHull", "bbox"]) {
    const count = counts.get(source) ?? 0;
    if (count) {
      parts.push(`${count} ${source}`);
    }
  }
  return parts.length ? parts.join(", ") : "none";
}

function summarizeRecordColliderSources(record) {
  const counts = summarizeColliderSources([record]);
  return formatColliderStats(counts);
}

function disposeObject3D(object) {
  object.traverse((child) => {
    if (!child.isMesh) {
      return;
    }
    child.geometry?.dispose?.();
    const materials = Array.isArray(child.material) ? child.material : [child.material];
    for (const material of materials) {
      material?.dispose?.();
    }
  });
}

function prepareCharacterVisual(object) {
  object.traverse((child) => {
    if (!child.isMesh) {
      return;
    }
    child.castShadow = SHADOWS_ENABLED;
    child.receiveShadow = SHADOWS_ENABLED;
    const materials = Array.isArray(child.material) ? child.material : [child.material];
    for (const material of materials) {
      if (material) {
        material.side = THREE.DoubleSide;
      }
    }
  });
}

function normalizeCharacterVisual(object, targetHeight) {
  object.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(object);
  const size = new THREE.Vector3();
  box.getSize(size);
  if (!Number.isFinite(size.z) || size.z <= 1e-6) {
    return;
  }

  object.scale.multiplyScalar(targetHeight / size.z);
  object.updateMatrixWorld(true);
  box.setFromObject(object);
  object.position.z -= box.min.z;
  object.updateMatrixWorld(true);
}

function stabilizeMixamoRootMotion(clip) {
  for (const track of clip.tracks ?? []) {
    const trackName = String(track.name ?? "").toLowerCase();
    if (!trackName.endsWith(".position") || !trackName.includes("hips")) {
      continue;
    }

    const values = track.values;
    if (!values || values.length < 3) {
      continue;
    }

    const baseX = values[0];
    const baseZ = values[2];
    for (let i = 0; i + 2 < values.length; i += 3) {
      values[i] = baseX;
      values[i + 2] = baseZ;
    }
  }
}

function kickPhaseEnvelope(t) {
  if (t < 0.18 || t > 0.86) {
    return 0;
  }

  if (t < 0.42) {
    return THREE.MathUtils.smoothstep(t, 0.18, 0.42);
  }

  if (t > 0.66) {
    return 1 - THREE.MathUtils.smoothstep(t, 0.66, 0.86);
  }

  return 1;
}

function readUint32LE(bytes, offset) {
  return bytes[offset]
    | (bytes[offset + 1] << 8)
    | (bytes[offset + 2] << 16)
    | (bytes[offset + 3] << 24);
}

function readAscii(bytes, offset, length) {
  let result = "";
  for (let i = 0; i < length; i += 1) {
    result += String.fromCharCode(bytes[offset + i]);
  }
  return result;
}

function vectorToRapier(vector) {
  return { x: vector.x, y: vector.y, z: vector.z };
}

function vectorLength(vector) {
  const x = vector?.x ?? 0;
  const y = vector?.y ?? 0;
  const z = vector?.z ?? 0;
  return Math.hypot(x, y, z);
}

function normalizedOrDefault(vector, fallback) {
  const result = vector.clone();
  if (result.lengthSq() < 1e-10) {
    return fallback.clone().normalize();
  }
  return result.normalize();
}

function quaternionToRapier(quaternion) {
  return { x: quaternion.x, y: quaternion.y, z: quaternion.z, w: quaternion.w };
}

async function fetchWithTimeout(url, timeoutMs) {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { signal: controller.signal });
  } finally {
    window.clearTimeout(timer);
  }
}

function shortError(error) {
  if (error?.name === "AbortError") {
    return "request timed out";
  }
  return String(error?.message || error || "unknown error").slice(0, 80);
}

function parseNumber(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function isTypingTarget(target) {
  const tagName = target?.tagName?.toLowerCase?.();
  return Boolean(target?.isContentEditable || ["input", "textarea", "select"].includes(tagName));
}

function childElements(parent, tagName) {
  return [...parent.children].filter((child) => child.localName === tagName);
}

function firstChildElement(parent, tagName) {
  return childElements(parent, tagName)[0] ?? null;
}

function parseVector3(value, fallback) {
  if (value == null || String(value).trim() === "") {
    return new THREE.Vector3(fallback[0], fallback[1], fallback[2]);
  }

  const parts = String(value)
    .trim()
    .split(/\s+/)
    .map(Number)
    .filter(Number.isFinite);
  return new THREE.Vector3(
    parts[0] ?? fallback[0],
    parts[1] ?? fallback[1],
    parts[2] ?? fallback[2],
  );
}

function quaternionFromRpy(roll, pitch, yaw) {
  return new THREE.Quaternion().setFromEuler(new THREE.Euler(roll, pitch, yaw, "XYZ"));
}

function composeTransform(parent, localOrigin) {
  return {
    position: parent.position
      .clone()
      .add(localOrigin.xyz.clone().applyQuaternion(parent.rotation)),
    rotation: parent.rotation.clone().multiply(localOrigin.quaternion),
  };
}

function findRootLinks(parsed) {
  const children = new Set(parsed.joints.map((joint) => joint.child));
  const roots = [...parsed.links.keys()].filter((linkName) => !children.has(linkName));
  return roots.length ? roots : [...parsed.links.keys()].slice(0, 1);
}

function groupJointsByParent(joints) {
  const grouped = new Map();
  for (const joint of joints) {
    if (!grouped.has(joint.parent)) {
      grouped.set(joint.parent, []);
    }
    grouped.get(joint.parent).push(joint);
  }
  return grouped;
}

function colorForName(name) {
  const palette = [0x39424e, 0x39a275, 0xf0a13a, 0x2871cc, 0xe05a47, 0x8b62d9];
  let hash = 0;
  for (let i = 0; i < name.length; i += 1) {
    hash = (hash * 31 + name.charCodeAt(i)) >>> 0;
  }
  return palette[hash % palette.length];
}

function boot() {
  const root = document.getElementById("sim-root");
  const app = new InteractiveRigidBodyDemo(root);

  window.RapierRigidBodyDemo = {
    app,
    THREE,
    RAPIER,
    addConvexHullBody(mesh, options = {}) {
      const position = options.position ?? mesh.position ?? TMP_VEC;
      const clone = mesh.clone();
      if (mesh.geometry) {
        clone.geometry = mesh.geometry.clone();
      }
      return app.createRigidMeshBody(clone, {
        name: options.name ?? mesh.name ?? "convex-body",
        position: position.clone ? position.clone() : new THREE.Vector3(position.x, position.y, position.z),
        collider: "convexHull",
        type: options.type ?? "dynamic",
        draggable: options.draggable ?? true,
      });
    },
    createRigidMeshBody: (mesh, options = {}) => app.createRigidMeshBody(mesh, options),
    createConvexHullColliderDescFromMesh: (mesh) => app.createConvexHullColliderDescFromMesh(mesh),
    createConvexMeshColliderDescFromMesh: (mesh) => app.createConvexMeshColliderDescFromMesh(mesh),
    loadGeneratedScene: (manifestUrl) => app.loadGeneratedScene(manifestUrl),
    clearGeneratedScene: () => app.clearGeneratedScene(),
  };
}

boot();
