// OCTOBER — procedural empty-space atmosphere
//
// This scene is built from separate p5.js layers rather than using the
// reference images directly. The layers are intentionally exposed in the
// code so the visual language can be tuned one part at a time.

const STYLE_INTERVAL = 20000;
const WHITE_RESIDUE_COUNT = 900;
const FINE_GRAIN_COUNT = 1100;
const MICRO_DOT_COUNT = 1800;
const PRINT_STROKE_COUNT = 1800;
const PRINT_SPECK_COUNT = 2200;
const PENCIL_HATCH_COUNT = 900;
const GLOBAL_PENCIL_STROKE_COUNT = 2600;
const GLOBAL_PENCIL_SPECK_COUNT = 2600;
const GLOBAL_STIPPLE_COUNT = 7200;
const GLOBAL_PENCIL_OPACITY = 0.82;
const FLOATING_DOT_COUNT = 360;
// The display runs at 30fps; semantic segmentation only needs to produce a
// new target often enough for motion to feel continuous.
const VISION_SEGMENT_INTERVAL = 100;
const VISION_CORE_BLUR = 1.5;
const VISION_OUTER_BLUR = 2;
const VISION_NEAREST_DISTANCE_BLUR = 1;
const VISION_MID_DISTANCE_BLUR = 3;
const VISION_FAR_DISTANCE_BLUR = 6;
const VISION_PENCIL_OPACITY = 0.16;
const VISION_FACE_INTERVAL = 500;
const VISION_CALIBRATION_FACE_INTERVAL = 220;
const VISION_CALIBRATION_DURATION = 3000;
// Rebuilding the full-resolution print texture is much more expensive than
// drawing the cached result, so leave the original atmosphere intact while
// giving the vision pipeline more uninterrupted frames.
const BACKGROUND_RENDER_INTERVAL = 1800;
const CAMERA_CONNECTION_GRACE_PERIOD = 7000;
const VISION_PACKAGE =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/+esm";
const VISION_WASM =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm";
const VISION_SEGMENT_MODEL =
  "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite";
const VISION_FACE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite";
const VISION_SKIN = [248, 196, 166];
const VISION_HAIR = [13, 11, 16];
const VISION_CLOTHES = [
  [250, 91, 113],
  [49, 184, 211],
  [255, 187, 61],
  [143, 101, 221],
  [51, 176, 119],
  [235, 111, 177],
  [246, 132, 59],
];

let activeStyle = 0;
let lastStyleChange = 0;
let whiteResidues = [];
let fineGrain = [];
let microDots = [];
let floatingDots = [];
let printStrokes = [];
let printSpecks = [];
let pencilHatches = [];
let printTextureLayer;
let globalIllustrationTexture;
let globalStippleTexture;
let globalIllustrationComposite;
let sunMarks = [];
let floorMarks = [];
let visionCamera;
let visionStatus = "인물 인식 모델을 준비하는 중";
let lastLoggedVisionStatus = "";
let visionReady = false;
let visionCameraStarted = false;
let visionCameraRequesting = false;
let visionCameraRecoveryOffered = false;
let visionCameraDelayTimer;
let visionSegmenter;
let visionFaceDetector;
let visionLayer;
let visionLayerContext;
let visionPencilLayer;
let visionPencilContext;
let visionPersonLayers = new Map();
let visionTargetImageData;
let visionCameraFrame;
let visionCameraContext;
let visionLastSegmentationAt = 0;
let visionLastFaceAt = 0;
let visionLastVideoTime = -1;
let visionClosestFace;
let visionDetectedFaces = [];
let visionFaceTracks = [];
let visionPersonTracks = [];
let visionNextClothesColor = 0;
let visionNextTrackId = 0;
let visionBfsQueue;
let visionPeopleCount = 0;
let visionFaceCount = 0;
let visionCalibrating = false;
let visionCalibrationStartedAt = 0;
let visionCalibrationSamples = [];
let visionCalibratedFaceArea = 0;
let visionCalibrationReady = false;
let visionRuntimeError = "";
let cachedBackground;
let lastBackgroundRender = -Infinity;
let backgroundNeedsRefresh = true;

const STYLES = [
  {
    skyTop: [245, 161, 178],
    skyBottom: [255, 220, 177],
    horizon: [255, 247, 231],
    floorBottom: [248, 184, 204],
    sun: [255, 226, 142],
    residue: [255, 255, 238],
    clothes: [
      [250, 91, 113], [255, 132, 94], [255, 183, 74], [239, 91, 151],
      [245, 119, 177], [255, 151, 112], [220, 82, 111],
    ],
  },
  {
    skyTop: [35, 78, 172],
    skyBottom: [128, 168, 231],
    horizon: [247, 246, 237],
    floorBottom: [166, 187, 239],
    sun: [255, 246, 209],
    residue: [255, 255, 255],
    clothes: [
      [44, 145, 224], [52, 190, 214], [91, 112, 226], [255, 205, 72],
      [109, 213, 173], [235, 111, 177], [238, 132, 83],
    ],
  },
  {
    skyTop: [139, 152, 220],
    skyBottom: [247, 193, 210],
    horizon: [255, 247, 223],
    floorBottom: [185, 167, 222],
    sun: [255, 231, 178],
    residue: [255, 244, 255],
    clothes: [
      [157, 111, 224], [211, 114, 205], [247, 117, 173], [255, 177, 111],
      [130, 177, 226], [187, 139, 229], [113, 199, 173],
    ],
  },
];

function setup() {
  window.__OCTOBER_SKETCH_BOOTED = true;
  console.info("[OCTOBER] 카메라 연결 시작");
  const canvas = createCanvas(windowWidth, windowHeight);
  canvas.parent("canvas-container");
  pixelDensity(1);
  noSmooth();
  frameRate(30);
  noStroke();

  // Preserve the original OCTOBER atmosphere exactly as its own set of
  // animated layers. The camera portrait is composited only after this.
  printTextureLayer = createGraphics(width, height);
  printTextureLayer.pixelDensity(1);
  createGlobalIllustrationTexture();
  createWhiteResidues();
  createFineGrain();
  createMicroDots();
  createFloatingDots();
  createPrintTexture();
  createSunMarks();
  createFloorMarks();

  lastStyleChange = millis();
  // Ask for the camera before loading the ML models. On lower-powered
  // exhibition PCs this keeps the browser's camera handshake out of the
  // model-loading bottleneck.
  startVisionCamera();
  initialiseVision();
}

function draw() {
  if (millis() - lastStyleChange >= STYLE_INTERVAL) {
    activeStyle = (activeStyle + 1) % STYLES.length;
    lastStyleChange = millis();
    backgroundNeedsRefresh = true;
  }

  const style = STYLES[activeStyle];
  drawCachedBackground(style);

  if (
    visionCameraStarted &&
    visionCamera instanceof HTMLVideoElement &&
    visionCamera.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
    visionCamera.videoWidth > 0
  ) {
    updateVisionSegmentation();
    drawVisionSegmentation();
    updateVisionCalibration();
    if (visionCalibrating) drawVisionCalibrationGuide();
  }

  drawGlobalIllustrationTexture();
  drawVisionStatus();
}

function createGlobalIllustrationTexture() {
  globalIllustrationTexture = createGraphics(width, height);
  globalIllustrationTexture.pixelDensity(1);
  globalIllustrationTexture.clear();
  globalIllustrationTexture.strokeCap(ROUND);

  // Very light overlapping strokes unify the background and the live figure
  // as one colored-pencil illustration without hiding either one.
  for (let i = 0; i < GLOBAL_PENCIL_STROKE_COUNT; i++) {
    const angle = i % 3 === 0 ? 0.72 : -0.38;
    const length = random(9, 38);
    const x = random(-width * 0.05, width * 1.05);
    const y = random(-height * 0.05, height * 1.05);
    const shade = i % 4 === 0 ? [70, 58, 92] : [255, 245, 224];
    globalIllustrationTexture.stroke(
      shade[0],
      shade[1],
      shade[2],
      random(18, 42),
    );
    globalIllustrationTexture.strokeWeight(random(0.55, 1.35));
    globalIllustrationTexture.line(
      x,
      y,
      x + cos(angle) * length,
      y + sin(angle) * length,
    );
  }

  for (let i = 0; i < GLOBAL_PENCIL_SPECK_COUNT; i++) {
    const shade = i % 3 === 0 ? [45, 38, 70] : [255, 248, 230];
    globalIllustrationTexture.noStroke();
    globalIllustrationTexture.fill(shade[0], shade[1], shade[2], random(8, 24));
    const size = random(0.45, 2.1);
    globalIllustrationTexture.ellipse(random(width), random(height), size, size);
  }

  // The reference texture is primarily granular rather than line-based:
  // thousands of tiny pigment deposits create the soft printed/stippled
  // surface, especially where the background transitions between tones.
  for (let i = 0; i < GLOBAL_STIPPLE_COUNT; i++) {
    const x = random(width);
    const y = random(height);
    const density = 0.35 + noise(x * 0.0028, y * 0.0028) * 0.65;
    if (random() > density) continue;
    const dark = i % 5 !== 0;
    const color = dark ? [36, 30, 62] : [255, 250, 230];
    const alpha = dark ? random(10, 28) : random(8, 22);
    globalIllustrationTexture.noStroke();
    globalIllustrationTexture.fill(color[0], color[1], color[2], alpha);
    const size = random(0.35, 1.35);
    globalIllustrationTexture.ellipse(x, y, size, size * random(0.65, 1.4));
  }

  // A half-resolution regular dot field gives the dense printed/powdery
  // material seen in the reference image, without creating a huge full-size
  // particle simulation for every frame.
  const stippleWidth = Math.max(320, Math.floor(width * 0.5));
  const stippleHeight = Math.max(180, Math.floor(height * 0.5));
  globalStippleTexture = createGraphics(stippleWidth, stippleHeight);
  globalStippleTexture.pixelDensity(1);
  globalStippleTexture.clear();
  globalStippleTexture.noStroke();
  const spacing = 2.8;
  for (let y = 0; y < stippleHeight; y += spacing) {
    for (let x = 0; x < stippleWidth; x += spacing) {
      const localDensity =
        0.58 + noise(x * 0.012, y * 0.012, 7) * 0.42;
      if (random() > localDensity) continue;
      const dark = (Math.floor(x / spacing) + Math.floor(y / spacing)) % 7 !== 0;
      const dotColor = dark ? [32, 26, 55] : [255, 248, 224];
      const dotAlpha = dark ? random(30, 72) : random(20, 54);
      globalStippleTexture.fill(dotColor[0], dotColor[1], dotColor[2], dotAlpha);
      const dotSize = random(0.65, 1.45);
      globalStippleTexture.ellipse(
        x + random(-0.35, 0.35),
        y + random(-0.35, 0.35),
        dotSize,
        dotSize * random(0.72, 1.25),
      );
    }
  }

  // Compose the two static texture passes once. The live draw loop now makes
  // one cached image draw instead of blending two full-screen layers every
  // frame.
  globalIllustrationComposite = createGraphics(width, height);
  globalIllustrationComposite.pixelDensity(1);
  globalIllustrationComposite.clear();
  globalIllustrationComposite.push();
  globalIllustrationComposite.blendMode(MULTIPLY);
  globalIllustrationComposite.tint(255, 112 * GLOBAL_PENCIL_OPACITY);
  globalIllustrationComposite.image(globalIllustrationTexture, 0, 0, width, height);
  globalIllustrationComposite.blendMode(SCREEN);
  globalIllustrationComposite.tint(255, 30 * GLOBAL_PENCIL_OPACITY);
  globalIllustrationComposite.image(globalIllustrationTexture, 0, 0, width, height);
  globalIllustrationComposite.blendMode(MULTIPLY);
  globalIllustrationComposite.tint(255, 82 * GLOBAL_PENCIL_OPACITY);
  globalIllustrationComposite.image(globalStippleTexture, 0, 0, width, height);
  globalIllustrationComposite.blendMode(SCREEN);
  globalIllustrationComposite.tint(255, 22 * GLOBAL_PENCIL_OPACITY);
  globalIllustrationComposite.image(globalStippleTexture, 0, 0, width, height);
  globalIllustrationComposite.pop();
}

function drawGlobalIllustrationTexture() {
  if (!globalIllustrationComposite) return;
  push();
  image(globalIllustrationComposite, 0, 0, width, height);
  tint(255, 255);
  blendMode(BLEND);
  pop();
}

// The original atmosphere contains thousands of particles and brush marks.
// Updating it at a calm cadence keeps the visible density intact while leaving
// the browser enough time to open the camera and run segmentation.
function drawCachedBackground(style) {
  if (
    backgroundNeedsRefresh ||
    !cachedBackground ||
    millis() - lastBackgroundRender >= BACKGROUND_RENDER_INTERVAL
  ) {
    drawWhiteBase();
    drawSkyLayer(style);
    drawHorizonLayer(style);
    drawFloorLayer(style);
    drawMicroPixelLayer(style);
    drawSunLayer(style);
    drawFloorResidueLayer(style);
    drawWhiteResidueLayer(style);
    drawFloatingDots(style);
    drawFineGrainLayer(style);
    drawPrintTextureLayer(style);
    cachedBackground = get();
    lastBackgroundRender = millis();
    backgroundNeedsRefresh = false;
    return;
  }

  image(cachedBackground, 0, 0, width, height);
}

// Layer 1 — a bright, almost white base.
function drawWhiteBase() {
  background(250, 249, 242);
}

// Layer 2 — the independent sky field above the horizon.
function drawSkyLayer(style) {
  const horizonY = height * 0.49;

  for (let y = 0; y < horizonY; y += 3) {
    const amount = y / horizonY;
    const gradientColor = lerpColor(
      colorFrom(style.skyTop),
      colorFrom(style.skyBottom),
      amount,
    );

    fill(red(gradientColor), green(gradientColor), blue(gradientColor), 245);
    rect(0, y, width, 4);
  }

  blendMode(SCREEN);
  for (let i = 0; i < 13; i++) {
    const x = noise(i * 0.8, frameCount * 0.002 + i) * width;
    const y = noise(i * 2.1, 20 + frameCount * 0.001 + i) * horizonY;
    const size = width * randomSeeded(i + activeStyle * 20, 0.13, 0.28);

    fill(style.residue[0], style.residue[1], style.residue[2], 10);
    ellipse(x, y, size, size * 0.42);
  }
  blendMode(BLEND);
}

// Layer 3 — the white atmospheric opening around the horizon.
function drawHorizonLayer(style) {
  const horizonY = height * 0.49;

  push();
  drawingContext.filter = "blur(28px)";
  fill(style.horizon[0], style.horizon[1], style.horizon[2], 225);
  ellipse(width * 0.5, horizonY, width * 1.16, height * 0.24);
  drawingContext.filter = "none";
  pop();

  fill(style.horizon[0], style.horizon[1], style.horizon[2], 70);
  rect(0, horizonY - height * 0.035, width, height * 0.09);
}

// Layer 4 — pale ground with slow horizontal traces.
function drawFloorLayer(style) {
  const horizonY = height * 0.49;

  for (let y = horizonY; y < height; y += 3) {
    const amount = (y - horizonY) / (height - horizonY);
    const gradientColor = lerpColor(
      colorFrom(style.horizon),
      colorFrom(style.floorBottom),
      amount,
    );

    fill(red(gradientColor), green(gradientColor), blue(gradientColor), 215);
    rect(0, y, width, 4);
  }

  blendMode(SCREEN);
  for (let i = 0; i < 12; i++) {
    const y = horizonY + height * (0.08 + i * 0.075);
    const drift = sin(frameCount * 0.006 + i) * width * 0.035;

    push();
    drawingContext.filter = "blur(16px)";
    fill(style.horizon[0], style.horizon[1], style.horizon[2], 22);
    ellipse(width * 0.5 + drift, y, width * (0.75 + i * 0.06), height * 0.035);
    drawingContext.filter = "none";
    pop();
  }
  blendMode(BLEND);
}

// Small horizontal traces that sit specifically in the floor area.
function drawFloorResidueLayer(style) {
  blendMode(SCREEN);

  push();
  drawingContext.filter = "blur(1.8px)";
  for (const mark of floorMarks) {
    const x = mark.x + sin(frameCount * mark.speed + mark.phase) * mark.motion;
    const y =
      mark.y +
      cos(frameCount * mark.speed * 0.7 + mark.phase) * mark.motion * 0.35;
    stroke(
      style.residue[0],
      style.residue[1],
      style.residue[2],
      mark.alpha * 0.3,
    );
    strokeWeight(mark.weight * 2.2);
    line(x, y, x + mark.length, y + mark.slope);
  }
  drawingContext.filter = "none";
  pop();

  for (const mark of floorMarks) {
    const x = mark.x + sin(frameCount * mark.speed + mark.phase) * mark.motion;
    const y =
      mark.y +
      cos(frameCount * mark.speed * 0.7 + mark.phase) * mark.motion * 0.35;
    stroke(style.residue[0], style.residue[1], style.residue[2], mark.alpha);
    strokeWeight(mark.weight);
    line(x, y, x + mark.length, y + mark.slope);
  }

  noStroke();
  blendMode(BLEND);
}

// Layer 5 — a cluster of slowly moving sunlight afterimages.
function drawSunLayer(style) {
  blendMode(SCREEN);

  for (const mark of sunMarks) {
    const x = mark.x + sin(frameCount * mark.speed + mark.phase) * mark.drift;
    const y =
      mark.y +
      cos(frameCount * mark.speed * 0.8 + mark.phase) * mark.drift * 0.35;

    push();
    drawingContext.filter = `blur(${mark.blur}px)`;
    for (const pixel of mark.pixels) {
      const px = x + pixel.x;
      const py = y + pixel.y;
      fill(style.sun[0], style.sun[1], style.sun[2], pixel.alpha * 0.22);
      ellipse(px, py, pixel.size * 2.5, pixel.size * 2.5);
    }
    drawingContext.filter = "none";

    for (const pixel of mark.pixels) {
      const px = x + pixel.x;
      const py = y + pixel.y;
      fill(style.sun[0], style.sun[1], style.sun[2], pixel.alpha);
      ellipse(px, py, pixel.size, pixel.size);
    }
    pop();
  }

  blendMode(BLEND);
}

// Small pixel clusters create the dense, powdery surface instead of relying
// on large smooth shapes. Their slow drift keeps the image alive.
function drawMicroPixelLayer(style) {
  blendMode(SCREEN);

  push();
  drawingContext.filter = "blur(1.6px)";
  for (const dot of microDots) {
    const x = dot.x + sin(frameCount * dot.speed + dot.phase) * dot.motion;
    const y =
      dot.y + cos(frameCount * dot.speed * 0.75 + dot.phase) * dot.motion;
    fill(
      style.residue[0],
      style.residue[1],
      style.residue[2],
      dot.alpha * 0.24,
    );
    rect(x, y, dot.size * 2.2, dot.size * 2.2);
  }
  drawingContext.filter = "none";
  pop();

  for (const dot of microDots) {
    const x = dot.x + sin(frameCount * dot.speed + dot.phase) * dot.motion;
    const y =
      dot.y + cos(frameCount * dot.speed * 0.75 + dot.phase) * dot.motion;
    fill(style.residue[0], style.residue[1], style.residue[2], dot.alpha);
    rect(x, y, dot.size, dot.size);
  }

  blendMode(BLEND);
}

// Layer 6 — larger white residues floating across the floor and sky.
function drawWhiteResidueLayer(style) {
  blendMode(SCREEN);

  for (const residue of whiteResidues) {
    const x =
      residue.x +
      sin(frameCount * residue.speed + residue.phase) * residue.motion;
    const y =
      residue.y +
      cos(frameCount * residue.speed * 0.72 + residue.phase) *
        residue.motion *
        0.5;
    const pulse = 0.72 + sin(frameCount * residue.pulse + residue.phase) * 0.28;

    fill(
      style.residue[0],
      style.residue[1],
      style.residue[2],
      residue.alpha * pulse,
    );
    ellipse(x, y, residue.size, residue.size * residue.ratio);
  }

  blendMode(BLEND);
}

// A separate, quiet layer of white dots. Each dot rises very slowly and
// sways slightly, so the atmosphere feels alive without looking like rain.
function drawFloatingDots(style) {
  blendMode(SCREEN);

  for (const dot of floatingDots) {
    dot.y -= dot.speed;
    dot.x += sin(frameCount * dot.swaySpeed + dot.phase) * dot.sway;

    if (dot.y < -dot.size * 3) {
      dot.y = height + dot.size * 3;
      dot.x = random(width);
    }

    const breathing =
      0.82 + sin(frameCount * dot.twinkleSpeed + dot.phase) * 0.18;
    const alpha = dot.alpha * breathing;

    if (dot.soft) {
      push();
      drawingContext.filter = "blur(2px)";
      fill(style.residue[0], style.residue[1], style.residue[2], alpha * 0.28);
      ellipse(dot.x, dot.y, dot.size * 3.2, dot.size * 3.2);
      drawingContext.filter = "none";
      pop();
    }

    fill(style.residue[0], style.residue[1], style.residue[2], alpha);
    ellipse(dot.x, dot.y, dot.size, dot.size);
  }

  blendMode(BLEND);
}

// Layer 7 — fine random grain, continuously regenerated like a moving print.
function drawFineGrainLayer(style) {
  blendMode(SCREEN);
  for (const grain of fineGrain) {
    const x =
      grain.x + sin(frameCount * grain.speed + grain.phase) * grain.motion;
    const y =
      grain.y +
      cos(frameCount * grain.speed * 0.8 + grain.phase) * grain.motion;
    const alpha =
      grain.alpha * (0.75 + sin(frameCount * 0.02 + grain.phase) * 0.25);

    fill(style.residue[0], style.residue[1], style.residue[2], alpha);
    ellipse(x, y, grain.size, grain.size * grain.ratio);
  }
  blendMode(BLEND);
}

// Layer 8 — dense dry-brush / printmaking texture.
// The texture is rendered into its own layer so the directional fibers can
// remain dense without interfering with the other atmosphere layers.
function drawPrintTextureLayer(style) {
  // The print layer changes slowly, so it does not need to be rebuilt every
  // frame. This removes most of the CPU load while keeping the texture alive.
  if (frameCount % 6 === 0) {
    printTextureLayer.clear();
    const context = printTextureLayer.drawingContext;
    context.save();
    context.lineCap = "round";

    // A blurred under-print creates the soft, clogged areas of ink.
    context.filter = "blur(1.2px)";
    context.globalCompositeOperation = "screen";
    for (const stroke of printStrokes) {
      const density = textureDensity(stroke.x, stroke.y);
      if (density < 0.18) continue;

      const point = animatedStrokePoint(stroke);
      const alpha = stroke.alpha * density * 0.3;
      context.strokeStyle = rgba(style.residue, alpha);
      context.lineWidth = stroke.width * 2.6;
      context.beginPath();
      context.moveTo(point.x1, point.y1);
      context.lineTo(point.x2, point.y2);
      context.stroke();
    }

    context.filter = "none";

    // Sharp fibers sit over the blurred under-print.
    for (const stroke of printStrokes) {
      const density = textureDensity(stroke.x, stroke.y);
      if (density < 0.12) continue;

      const point = animatedStrokePoint(stroke);
      const darkFiber = stroke.index % 7 === 0;
      const fiberColor = darkFiber ? style.skyTop : style.residue;
      const alpha = stroke.alpha * density * (darkFiber ? 0.62 : 0.88);
      context.strokeStyle = rgba(fiberColor, alpha);
      context.lineWidth = stroke.width;
      context.beginPath();
      context.moveTo(point.x1, point.y1);
      context.lineTo(point.x2, point.y2);
      context.stroke();
    }

    // Layered colored-pencil hatching: translucent strokes in two directions
    // create the pressure variation and rubbed pigment of a hand-drawn paper.
    context.globalCompositeOperation = "multiply";
    for (const hatch of pencilHatches) {
      const density = textureDensity(hatch.x, hatch.y);
      if (density < 0.14) continue;
      const drift = animatedStrokePoint(hatch);
      const alpha = hatch.alpha * density;
      context.strokeStyle = rgba(hatch.cross ? style.skyTop : style.residue, alpha);
      context.lineWidth = hatch.width;
      context.beginPath();
      context.moveTo(drift.x1, drift.y1);
      context.lineTo(drift.x2, drift.y2);
      context.stroke();
    }

    // Tiny ink breaks and dry-brush specks interrupt the lines.
    context.globalCompositeOperation = "screen";
    for (const speck of printSpecks) {
      const x =
        speck.x + sin(frameCount * speck.speed + speck.phase) * speck.motion;
      const y =
        speck.y +
        cos(frameCount * speck.speed * 0.8 + speck.phase) * speck.motion;
      const density = textureDensity(speck.x, speck.y);
      if (density < 0.2) continue;

      context.fillStyle = rgba(style.residue, speck.alpha * density);
      context.fillRect(x, y, speck.size, speck.size * speck.ratio);
    }

    context.restore();
  }

  blendMode(SCREEN);
  image(printTextureLayer, 0, 0, width, height);
  blendMode(BLEND);
}

function createPrintTexture() {
  printStrokes = [];
  printSpecks = [];
  pencilHatches = [];

  for (let i = 0; i < PRINT_STROKE_COUNT; i++) {
    const mainAngle = -0.34;
    const crossAngle = 0.82;
    const angle = i % 9 === 0 ? crossAngle : mainAngle + random(-0.08, 0.08);
    const length = random(5, 42);

    printStrokes.push({
      index: i,
      x: random(-width * 0.1, width * 1.1),
      y: random(-height * 0.1, height * 1.1),
      x1: cos(angle) * length * -0.5,
      y1: sin(angle) * length * -0.5,
      x2: cos(angle) * length * 0.5,
      y2: sin(angle) * length * 0.5,
      width: random(0.25, 1.35),
      alpha: random(18, 105),
      motion: random(0.15, 1.7),
      speed: random(0.001, 0.008),
      phase: random(TWO_PI),
    });
  }

  for (let i = 0; i < PRINT_SPECK_COUNT; i++) {
    printSpecks.push({
      x: random(width),
      y: random(height),
      size: random(0.3, 2.3),
      ratio: random(0.35, 2.2),
      alpha: random(20, 110),
      motion: random(0.1, 1.5),
      speed: random(0.001, 0.012),
      phase: random(TWO_PI),
    });
  }

  for (let i = 0; i < PENCIL_HATCH_COUNT; i++) {
    const cross = i % 2 === 0;
    const angle = cross ? 0.72 : -0.38;
    const length = random(10, 34);
    pencilHatches.push({
      index: i,
      x: random(-width * 0.08, width * 1.08),
      y: random(-height * 0.08, height * 1.08),
      x1: cos(angle) * length * -0.5,
      y1: sin(angle) * length * -0.5,
      x2: cos(angle) * length * 0.5,
      y2: sin(angle) * length * 0.5,
      width: random(0.35, 1.05),
      alpha: random(16, 42),
      cross,
      motion: random(0.12, 0.9),
      speed: random(0.001, 0.006),
      phase: random(TWO_PI),
    });
  }
}

function animatedStrokePoint(stroke) {
  const driftX = sin(frameCount * stroke.speed + stroke.phase) * stroke.motion;
  const driftY =
    cos(frameCount * stroke.speed * 0.7 + stroke.phase) * stroke.motion;

  return {
    x1: stroke.x + stroke.x1 + driftX,
    y1: stroke.y + stroke.y1 + driftY,
    x2: stroke.x + stroke.x2 + driftX,
    y2: stroke.y + stroke.y2 + driftY,
  };
}

function textureDensity(x, y) {
  const broad = noise(x * 0.0025, y * 0.0025, frameCount * 0.0008);
  const small = noise(x * 0.012, y * 0.012, 20 + frameCount * 0.0012);
  return constrain(broad * 0.65 + small * 0.55, 0, 1);
}

function rgba(rgb, alpha) {
  return `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${constrain(alpha, 0, 255) / 255})`;
}

function createWhiteResidues() {
  whiteResidues = [];

  for (let i = 0; i < WHITE_RESIDUE_COUNT; i++) {
    whiteResidues.push({
      x: random(width),
      y: random(height),
      size: random(0.5, 3.2),
      ratio: random(0.65, 1.5),
      alpha: random(16, 82),
      motion: random(0.3, 4.5),
      speed: random(0.0015, 0.014),
      pulse: random(0.012, 0.055),
      phase: random(TWO_PI),
    });
  }
}

function createFineGrain() {
  fineGrain = [];

  for (let i = 0; i < FINE_GRAIN_COUNT; i++) {
    fineGrain.push({
      x: random(width),
      y: random(height),
      size: random(0.25, 1.25),
      ratio: random(0.75, 1.35),
      alpha: random(12, 48),
      motion: random(0.15, 1.6),
      speed: random(0.003, 0.025),
      phase: random(TWO_PI),
    });
  }
}

function createMicroDots() {
  microDots = [];
  const centers = [];

  for (let i = 0; i < 34; i++) {
    centers.push({
      x: random(width),
      y: random(height * 0.12, height * 0.94),
      spreadX: random(width * 0.025, width * 0.16),
      spreadY: random(height * 0.015, height * 0.09),
    });
  }

  for (let i = 0; i < MICRO_DOT_COUNT; i++) {
    const center = random(centers);
    microDots.push({
      x: center.x + randomGaussian() * center.spreadX,
      y: center.y + randomGaussian() * center.spreadY,
      size: random(0.45, 1.8),
      alpha: random(14, 66),
      motion: random(0.2, 2.4),
      speed: random(0.0015, 0.012),
      phase: random(TWO_PI),
    });
  }
}

function createFloatingDots() {
  floatingDots = [];

  for (let i = 0; i < FLOATING_DOT_COUNT; i++) {
    const inLightCluster = i < FLOATING_DOT_COUNT * 0.28;
    floatingDots.push({
      x: inLightCluster ? random(width * 0.62, width * 0.9) : random(width),
      y: inLightCluster ? random(height * 0.05, height * 0.48) : random(height),
      size: random(0.7, 2.8),
      alpha: random(35, 130),
      speed: random(0.05, 0.22),
      sway: random(0.05, 0.32),
      swaySpeed: random(0.006, 0.018),
      twinkleSpeed: random(0.012, 0.04),
      phase: random(TWO_PI),
      soft: random() < 0.22,
    });
  }
}

function createSunMarks() {
  sunMarks = [
    {
      x: width * 0.82,
      y: height * 0.16,
      blur: 5,
      alpha: 120,
      drift: 12,
      speed: 0.006,
      phase: 0.4,
    },
    {
      x: width * 0.72,
      y: height * 0.24,
      blur: 8,
      alpha: 58,
      drift: 22,
      speed: 0.004,
      phase: 1.7,
    },
    {
      x: width * 0.18,
      y: height * 0.72,
      blur: 10,
      alpha: 42,
      drift: 18,
      speed: 0.005,
      phase: 2.9,
    },
  ];

  for (const mark of sunMarks) {
    mark.pixels = [];
    for (let i = 0; i < 260; i++) {
      const angle = random(TWO_PI);
      const distance = abs(randomGaussian()) * (i < 80 ? 38 : 115);
      mark.pixels.push({
        x: cos(angle) * distance,
        y: sin(angle) * distance,
        size: random(0.6, 3.4),
        alpha: random(18, 100),
      });
    }
  }
}

function createFloorMarks() {
  floorMarks = [];

  for (let i = 0; i < 400; i++) {
    floorMarks.push({
      x: random(-width * 0.1, width),
      y: random(height * 0.52, height * 0.98),
      length: random(4, 42),
      slope: random(-1.5, 1.5),
      weight: random(0.25, 1.2),
      alpha: random(10, 54),
      motion: random(0.2, 2.2),
      speed: random(0.001, 0.01),
      phase: random(TWO_PI),
    });
  }
}

function colorFrom(values) {
  return color(values[0], values[1], values[2]);
}

function randomSeeded(seed, minimum, maximum) {
  const value = abs(sin(seed * 12.9898) * 43758.5453) % 1;
  return lerp(minimum, maximum, value);
}

// --- Live semantic portrait mode -----------------------------------------

async function initialiseVision() {
  try {
    const vision = await import(VISION_PACKAGE);
    const fileset = await vision.FilesetResolver.forVisionTasks(VISION_WASM);
    visionSegmenter = await createVisionSegmenter(vision, fileset, "GPU");
    visionFaceDetector = await createVisionFaceDetector(vision, fileset, "GPU");
    visionReady = true;
    beginVisionCalibrationWhenReady();
  } catch (gpuError) {
    // The installation should still run on exhibition computers with no GPU
    // delegate, even though its update rate will be lower.
    try {
      const vision = await import(VISION_PACKAGE);
      const fileset = await vision.FilesetResolver.forVisionTasks(VISION_WASM);
      visionSegmenter = await createVisionSegmenter(vision, fileset, "CPU");
      visionFaceDetector = await createVisionFaceDetector(
        vision,
        fileset,
        "CPU",
      );
      visionReady = true;
      beginVisionCalibrationWhenReady();
    } catch (error) {
      console.error(error);
      visionStatus =
        "인식 모델을 불러오지 못했습니다. 인터넷 연결을 확인하세요.";
    }
  }
}

function createVisionSegmenter(vision, fileset, delegate) {
  return vision.ImageSegmenter.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: VISION_SEGMENT_MODEL, delegate },
    runningMode: "VIDEO",
    outputCategoryMask: true,
    outputConfidenceMasks: false,
  });
}

function createVisionFaceDetector(vision, fileset, delegate) {
  return vision.FaceDetector.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: VISION_FACE_MODEL, delegate },
    runningMode: "VIDEO",
    minDetectionConfidence: 0.55,
  });
}

async function startVisionCamera() {
  if (visionCameraStarted || visionCameraRequesting) return;

  if (window.location.protocol === "file:") {
    visionStatus =
      "파일을 직접 열면 카메라를 쓸 수 없습니다. http://127.0.0.1:5500/OCTOBER/index.html 로 여세요.";
    return;
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    visionStatus =
      "이 브라우저는 카메라를 지원하지 않습니다. Chrome 또는 Edge에서 여세요.";
    return;
  }

  visionCameraRequesting = true;
  visionStatus = "카메라를 시작하는 중 — 권한 요청이 뜨면 허용하세요";
  visionCameraDelayTimer = window.setTimeout(
    offerVisionCameraRecovery,
    CAMERA_CONNECTION_GRACE_PERIOD,
  );

  try {
    // `video: true` deliberately avoids device-specific facing-mode and size
    // constraints. Those optional constraints can leave some Windows camera
    // drivers waiting forever even after Chrome has permission.
    const stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
    });

    if (visionCameraStarted || window.__OCTOBER_CAMERA_CONNECTED) {
      stream.getTracks().forEach((track) => track.stop());
      return;
    }

    window.clearTimeout(visionCameraDelayTimer);
    visionCamera = document.createElement("video");
    visionCamera.autoplay = true;
    visionCamera.muted = true;
    visionCamera.playsInline = true;
    visionCamera.setAttribute("aria-hidden", "true");
    visionCamera.style.cssText =
      "position:fixed;width:1px;height:1px;opacity:0;pointer-events:none;";
    visionCamera.srcObject = stream;
    // Keeping the video in the document makes Chrome decode the camera frames
    // reliably. Do not wait for play() here: some camera drivers resolve it
    // late even though the stream has already been granted.
    document.body.appendChild(visionCamera);
    visionCamera.play().catch((error) => console.warn(error));
    visionCameraStarted = true;
    visionCameraRequesting = false;
    window.__OCTOBER_CAMERA_CONNECTED = true;
    if (visionReady) {
      startVisionCalibration();
    } else {
      visionStatus = "카메라 연결됨 · 인식 모델을 불러오는 중";
    }
  } catch (error) {
    window.clearTimeout(visionCameraDelayTimer);
    visionCameraRequesting = false;
    console.error(error);
    if (error?.name === "NotAllowedError" || error?.name === "SecurityError") {
      visionStatus = "카메라 권한이 차단됐습니다. Chrome/Edge에서 허용하세요.";
    } else if (error?.name === "NotFoundError") {
      visionStatus = "연결된 카메라를 찾지 못했습니다.";
    } else {
      visionStatus =
        "카메라를 시작하지 못했습니다. Chrome 또는 Edge에서 다시 시도하세요.";
    }
  }
}

function beginVisionCalibrationWhenReady() {
  if (!visionCameraStarted) {
    visionStatus = "인식 모델 준비 완료 · 카메라에 연결하는 중";
    return;
  }
  startVisionCalibration();
}

function offerVisionCameraRecovery() {
  if (visionCameraStarted || visionCameraRecoveryOffered) return;

  // A few Chrome/Windows combinations defer an automatic camera request until
  // the first user gesture. Keep the exhibition view button-free; touching
  // anywhere on the artwork retries the same request with that gesture.
  visionCameraRecoveryOffered = true;
  visionCameraRequesting = false;
  visionStatus =
    "카메라 연결이 지연되고 있습니다 — 화면 아무 곳이나 한 번 클릭하세요";
  window.addEventListener("pointerdown", resumeVisionCameraFromGesture, {
    once: true,
  });
}

function resumeVisionCameraFromGesture() {
  visionCameraRecoveryOffered = false;
  startVisionCamera();
}

function startVisionCalibration() {
  visionCalibrating = true;
  visionCalibrationReady = false;
  visionCalibrationSamples = [];
  visionCalibrationStartedAt = performance.now();
  visionStatus = "캘리브레이션 중 — 화면 중앙에 서주세요";
}

function updateVisionSegmentation() {
  const now = performance.now();
  const faceInterval = visionCalibrating
    ? VISION_CALIBRATION_FACE_INTERVAL
    : VISION_FACE_INTERVAL;
  if (
    !visionSegmenter ||
    now - visionLastSegmentationAt < VISION_SEGMENT_INTERVAL ||
    visionCamera.currentTime === visionLastVideoTime
  ) {
    return;
  }

  visionLastSegmentationAt = now;
  visionLastVideoTime = visionCamera.currentTime;

  try {
    const result = visionSegmenter.segmentForVideo(visionCamera, now);
    const mask = result.categoryMask;
    if (!mask) return;

    ensureVisionSurfaces(mask.width, mask.height);
    if (now - visionLastFaceAt >= faceInterval && visionFaceDetector) {
      visionLastFaceAt = now;
      try {
        updateVisionClosestFace(now);
      } catch (faceError) {
        // Face tracking only chooses the foreground person. It must never
        // prevent the core hair/skin/clothing segmentation from appearing.
        console.warn(faceError);
        visionFaceDetector = null;
        visionFaceCount = 0;
      }
    }

    renderVisionMask(mask.getAsUint8Array(), mask.width, mask.height);
    visionRuntimeError = "";
    if (!visionCalibrating) {
      visionStatus = visionPeopleCount
        ? `카메라 연결됨 · 인물 ${visionPeopleCount}명 · 얼굴 ${visionFaceCount}개 감지`
        : "카메라 연결됨 · 아직 인물을 찾지 못함";
    }
  } catch (error) {
    console.error(error);
    visionRuntimeError = error?.message || error?.name || "unknown error";
    visionStatus = `인식 오류: ${visionRuntimeError}`;
  }
}

function updateVisionClosestFace(now) {
  const result = visionFaceDetector.detectForVideo(visionCamera, now);
  const faces = result?.detections || [];
  const detected = faces
    .map((face) => {
      const box = face.boundingBox;
      if (!box) return null;
      return {
        x: box.originX + box.width * 0.5,
        y: box.originY + box.height * 0.5,
        width: box.width,
        height: box.height,
        area: box.width * box.height,
      };
    })
    .filter(Boolean);

  const nextTracks = [];
  const matchedTracks = new Set();
  for (const face of detected) {
    let match;
    let nearestDistance = Infinity;
    for (const previous of visionFaceTracks) {
      if (matchedTracks.has(previous.trackId)) continue;
      const distance = sq(face.x - previous.x) + sq(face.y - previous.y);
      if (distance < nearestDistance && distance < 110 * 110) {
        match = previous;
        nearestDistance = distance;
      }
    }
    if (match) matchedTracks.add(match.trackId);
    nextTracks.push({
      ...face,
      trackId: match?.trackId ?? visionNextTrackId++,
      colorIndex: match
        ? match.colorIndex
        : visionNextClothesColor++ %
          (STYLES[activeStyle].clothes?.length || VISION_CLOTHES.length),
      upperSample: match?.upperSample || null,
      lowerSample: match?.lowerSample || null,
      lowerColorIndex: match?.lowerColorIndex ?? null,
    });
  }

  visionFaceTracks = nextTracks;
  const largestFace = nextTracks.reduce(
    (largest, face) => (!largest || face.area > largest.area ? face : largest),
    null,
  );
  visionClosestFace = largestFace;
  visionDetectedFaces = nextTracks.map((face) => ({
    ...face,
    foreground: face.trackId === largestFace?.trackId,
  }));
  visionFaceCount = faces.length;
  if (visionCalibrating && largestFace) {
    visionCalibrationSamples.push(largestFace.area);
  }
}

function updateVisionCalibration() {
  if (!visionCalibrating) return;

  const elapsed = performance.now() - visionCalibrationStartedAt;
  if (elapsed < VISION_CALIBRATION_DURATION) {
    const remaining = Math.max(
      1,
      Math.ceil((VISION_CALIBRATION_DURATION - elapsed) / 1000),
    );
    visionStatus = `캘리브레이션 중 — 화면 중앙에 서주세요 (${remaining})`;
    return;
  }

  visionCalibrating = false;
  if (visionCalibrationSamples.length) {
    visionCalibratedFaceArea = medianVisionValue(visionCalibrationSamples);
    visionCalibrationReady = true;
    visionStatus = "캘리브레이션 완료 · 인물 인식을 시작합니다";
  } else {
    visionCalibrationReady = false;
    visionStatus = "캘리브레이션 실패 · 얼굴이 보이도록 화면 중앙에 서주세요";
  }
}

function medianVisionValue(values) {
  const sorted = [...values].sort((first, second) => first - second);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) * 0.5;
}

function ensureVisionSurfaces(maskWidth, maskHeight) {
  if (visionLayer?.width === maskWidth && visionLayer.height === maskHeight)
    return;

  visionLayer = document.createElement("canvas");
  visionLayer.width = maskWidth;
  visionLayer.height = maskHeight;
  visionLayerContext = visionLayer.getContext("2d", {
    willReadFrequently: true,
  });
  visionLayerContext.imageSmoothingEnabled = false;

  // A sparse diagonal pencil hatch is clipped to the person later. Keeping
  // this as a cached layer makes the texture inexpensive during live video.
  visionPencilLayer = document.createElement("canvas");
  visionPencilLayer.width = maskWidth;
  visionPencilLayer.height = maskHeight;
  visionPencilContext = visionPencilLayer.getContext("2d");
  visionPencilContext.clearRect(0, 0, maskWidth, maskHeight);
  visionPencilContext.lineCap = "round";
  const pencilStrokeCount = Math.floor((maskWidth * maskHeight) / 1050);
  for (let stroke = 0; stroke < pencilStrokeCount; stroke++) {
    const x = Math.random() * maskWidth;
    const y = Math.random() * maskHeight;
    const length = 4 + Math.random() * 16;
    const angle = -0.38 + (Math.random() - 0.5) * 0.18;
    const alpha = 0.06 + Math.random() * 0.14;
    const shade = Math.random() > 0.5 ? 18 : 255;
    visionPencilContext.strokeStyle = `rgba(${shade}, ${shade}, ${shade}, ${alpha})`;
    visionPencilContext.lineWidth = 0.35 + Math.random() * 0.8;
    visionPencilContext.beginPath();
    visionPencilContext.moveTo(x, y);
    visionPencilContext.lineTo(
      x + Math.cos(angle) * length,
      y + Math.sin(angle) * length,
    );
    visionPencilContext.stroke();
  }

  visionCameraFrame = document.createElement("canvas");
  visionCameraFrame.width = maskWidth;
  visionCameraFrame.height = maskHeight;
  visionCameraContext = visionCameraFrame.getContext("2d", {
    willReadFrequently: true,
  });
  visionBfsQueue = new Int32Array(maskWidth * maskHeight);
  visionTargetImageData = null;
}

function renderVisionMask(labels, maskWidth, maskHeight) {
  const cameraCrop = getVisionCoverCrop(
    visionCamera.videoWidth,
    visionCamera.videoHeight,
    maskWidth,
    maskHeight,
  );
  visionCameraContext.clearRect(0, 0, maskWidth, maskHeight);
  visionCameraContext.drawImage(
    visionCamera,
    cameraCrop.x,
    cameraCrop.y,
    cameraCrop.width,
    cameraCrop.height,
    0,
    0,
    maskWidth,
    maskHeight,
  );
  const source = visionCameraContext.getImageData(
    0,
    0,
    maskWidth,
    maskHeight,
  ).data;
  // Face detections already provide stable person anchors. When faces exist,
  // skip the expensive full-mask connected-component flood fill and assign
  // pixels directly to the nearest face. Keep the flood fill only as a
  // fallback for bodies whose faces are not visible.
  let componentIds;
  let people = new Map();
  if (!visionDetectedFaces.length) {
    const componentsResult = findVisionPeople(labels, maskWidth, maskHeight);
    componentIds = componentsResult.componentIds;
    const foregroundId = findVisionForeground(
      componentIds,
      componentsResult.components,
      maskWidth,
      maskHeight,
    );
    people = trackVisionPeople(componentsResult.components, foregroundId);
  }
  visionPeopleCount = visionDetectedFaces.length || people.size;
  const output = visionLayerContext.createImageData(maskWidth, maskHeight);
  const splitPersonLayers = visionDetectedFaces.length > 1;
  const personOutputs = new Map();
  if (splitPersonLayers) {
    for (const person of visionDetectedFaces) {
      personOutputs.set(person.trackId, {
        person,
        imageData: visionLayerContext.createImageData(maskWidth, maskHeight),
      });
    }
  }

  // Estimate the original garment colors above and below each face. The
  // segmentation model exposes clothing as one category, so this lightweight
  // sampling step separates a differently colored top and bottom without
  // adding another heavy model.
  for (const person of visionDetectedFaces) {
    person._upperSample = { r: 0, g: 0, b: 0, count: 0 };
    person._lowerSample = { r: 0, g: 0, b: 0, count: 0 };
  }
  for (let pixel = 0; pixel < labels.length; pixel++) {
    if (labels[pixel] !== 4 && labels[pixel] !== 5) continue;
    const person = findVisionClothingFaceForPixel(pixel, maskWidth, maskHeight);
    if (!person) continue;
    const section = getVisionClothingSection(person, pixel, maskWidth, maskHeight);
    const sample = section === "lower" ? person._lowerSample : person._upperSample;
    const sourceIndex = pixel * 4;
    sample.r += source[sourceIndex];
    sample.g += source[sourceIndex + 1];
    sample.b += source[sourceIndex + 2];
    sample.count += 1;
  }
  for (const person of visionDetectedFaces) {
    updateVisionGarmentColors(person);
  }

  for (let pixel = 0; pixel < labels.length; pixel++) {
    const category = labels[pixel];
    if (category === 0) continue;

    const facePerson =
      category === 4 || category === 5
        ? findVisionClothingFaceForPixel(pixel, maskWidth, maskHeight)
        : findVisionFaceForPixel(pixel, maskWidth, maskHeight);
    const person = facePerson || (componentIds ? people.get(componentIds[pixel]) : null);
    const foreground = person?.foreground ?? false;
    const outputIndex = pixel * 4;
    const garmentColorIndex =
      (category === 4 || category === 5) && person
        ? getVisionGarmentColorIndex(person, pixel, maskWidth, maskHeight)
        : person?.colorIndex ?? 0;
    let color = visionColorForCategory(category, garmentColorIndex);

    const edgePixel = isVisionToonEdge(pixel, labels, maskWidth, maskHeight);
    if (!foreground) {
      color = edgePixel
        ? [20, 18, 27]
        : visionToonShade(
            color,
            source[outputIndex],
            source[outputIndex + 1],
            source[outputIndex + 2],
          );
    }

    output.data[outputIndex] = color[0];
    output.data[outputIndex + 1] = color[1];
    output.data[outputIndex + 2] = color[2];
    // Partially transparent boundary pixels let the enlarged mask blend into
    // the background instead of exposing the hard 256px stair-step contour.
    output.data[outputIndex + 3] = edgePixel ? 218 : 255;
    if (splitPersonLayers && person?.trackId != null) {
      const personOutput = personOutputs.get(person.trackId)?.imageData;
      if (personOutput) {
        personOutput.data[outputIndex] = color[0];
        personOutput.data[outputIndex + 1] = color[1];
        personOutput.data[outputIndex + 2] = color[2];
        personOutput.data[outputIndex + 3] = edgePixel ? 218 : 255;
      }
    }
  }

  // Replace the displayed mask as one complete frame. Keeping only the
  // latest result prevents previous silhouettes from remaining as afterimages.
  visionTargetImageData = output;
  visionLayerContext.putImageData(output, 0, 0);
  visionLayerContext.save();
  visionLayerContext.globalCompositeOperation = "source-atop";
  visionLayerContext.globalAlpha = VISION_PENCIL_OPACITY;
  visionLayerContext.drawImage(visionPencilLayer, 0, 0);
  visionLayerContext.restore();

  // Keep a separate transparent layer for each detected person so distance
  // based blur can be applied independently during compositing.
  const activeTrackIds = new Set(personOutputs.keys());
  for (const trackId of visionPersonLayers.keys()) {
    if (!activeTrackIds.has(trackId)) visionPersonLayers.delete(trackId);
  }
  for (const [trackId, entry] of personOutputs) {
    let layer = visionPersonLayers.get(trackId)?.canvas;
    if (!layer || layer.width !== maskWidth || layer.height !== maskHeight) {
      layer = document.createElement("canvas");
      layer.width = maskWidth;
      layer.height = maskHeight;
    }
    layer.getContext("2d").putImageData(entry.imageData, 0, 0);
    visionPersonLayers.set(trackId, { canvas: layer, face: entry.person });
  }
}

function getVisionCoverCrop(sourceWidth, sourceHeight, targetWidth, targetHeight) {
  const sourceAspect = sourceWidth / sourceHeight;
  const targetAspect = targetWidth / targetHeight;

  if (sourceAspect > targetAspect) {
    const width = sourceHeight * targetAspect;
    return {
      x: (sourceWidth - width) * 0.5,
      y: 0,
      width,
      height: sourceHeight,
    };
  }

  const height = sourceWidth / targetAspect;
  return {
    x: 0,
    y: (sourceHeight - height) * 0.5,
    width: sourceWidth,
    height,
  };
}

function getVisionCoverDestination(sourceWidth, sourceHeight, targetWidth, targetHeight) {
  const scale = Math.max(targetWidth / sourceWidth, targetHeight / sourceHeight);
  const width = sourceWidth * scale;
  const height = sourceHeight * scale;
  return {
    x: (targetWidth - width) * 0.5,
    y: (targetHeight - height) * 0.5,
    width,
    height,
  };
}

function findVisionFaceForPixel(pixel, maskWidth, maskHeight) {
  if (!visionDetectedFaces.length || !visionCamera?.videoWidth) return null;

  const x = pixel % maskWidth;
  const y = Math.floor(pixel / maskWidth);
  let nearestFace = visionDetectedFaces[0];
  let nearestDistance = Infinity;

  for (const face of visionDetectedFaces) {
    // Horizontal distance is weighted more strongly so adjoining torsos are
    // divided by the nearest face even when their clothing touches.
    const facePoint = getVisionFaceMaskPoint(face, maskWidth, maskHeight);
    const distance =
      Math.abs(x - facePoint.x) * 1.6 + Math.abs(y - facePoint.y) * 0.15;
    if (distance < nearestDistance) {
      nearestFace = face;
      nearestDistance = distance;
    }
  }

  return nearestFace;
}

function findVisionClothingFaceForPixel(pixel, maskWidth, maskHeight) {
  if (!visionDetectedFaces.length || !visionCamera?.videoWidth) return null;

  const x = pixel % maskWidth;
  let nearestFace = visionDetectedFaces[0];
  let nearestDistance = Infinity;

  // Clothing is broad and often touches another person's clothing. Use the
  // horizontal face position as a stable divider so adjacent shirts keep
  // their own person's fixed color instead of blending together.
  for (const face of visionDetectedFaces) {
    const facePoint = getVisionFaceMaskPoint(face, maskWidth, maskHeight);
    const distance = Math.abs(x - facePoint.x);
    if (distance < nearestDistance) {
      nearestFace = face;
      nearestDistance = distance;
    }
  }

  return nearestFace;
}

function getVisionFaceMaskPoint(face, maskWidth, maskHeight) {
  const crop = getVisionCoverCrop(
    visionCamera.videoWidth,
    visionCamera.videoHeight,
    maskWidth,
    maskHeight,
  );
  return {
    x: ((face.x - crop.x) / crop.width) * maskWidth,
    y: ((face.y - crop.y) / crop.height) * maskHeight,
  };
}

function getVisionClothingSection(face, pixel, maskWidth, maskHeight) {
  const facePoint = getVisionFaceMaskPoint(face, maskWidth, maskHeight);
  const crop = getVisionCoverCrop(
    visionCamera.videoWidth,
    visionCamera.videoHeight,
    maskWidth,
    maskHeight,
  );
  const faceHeight = (face.height / crop.height) * maskHeight;
  const y = Math.floor(pixel / maskWidth);
  return y > facePoint.y + faceHeight * 1.55 ? "lower" : "upper";
}

function updateVisionGarmentColors(person) {
  const upper = person._upperSample;
  const lower = person._lowerSample;
  if (!upper?.count || !lower?.count) return;

  const upperAverage = [
    upper.r / upper.count,
    upper.g / upper.count,
    upper.b / upper.count,
  ];
  const lowerAverage = [
    lower.r / lower.count,
    lower.g / lower.count,
    lower.b / lower.count,
  ];
  const distance = Math.sqrt(
    sq(upperAverage[0] - lowerAverage[0]) +
      sq(upperAverage[1] - lowerAverage[1]) +
      sq(upperAverage[2] - lowerAverage[2]),
  );
  const palette = STYLES[activeStyle].clothes || VISION_CLOTHES;
  // Similar garments share one color. Clearly different garments use the
  // next stable palette color and remain fixed for this tracked person.
  if (distance > 48 && person.lowerColorIndex === null) {
    person.lowerColorIndex = (person.colorIndex + 1) % palette.length;
  } else if (distance <= 34) {
    person.lowerColorIndex = person.colorIndex;
  }
  const track = visionFaceTracks.find((item) => item.trackId === person.trackId);
  if (track) {
    track.upperSample = upperAverage;
    track.lowerSample = lowerAverage;
    track.lowerColorIndex = person.lowerColorIndex;
  }
}

function getVisionGarmentColorIndex(person, pixel, maskWidth, maskHeight) {
  if (getVisionClothingSection(person, pixel, maskWidth, maskHeight) !== "lower") {
    return person.colorIndex;
  }
  return person.lowerColorIndex ?? person.colorIndex;
}

function visionColorForCategory(category, colorIndex) {
  if (category === 1) return VISION_HAIR;
  if (category === 2 || category === 3) return VISION_SKIN;
  // Accessories share their nearest person's clothing color so that bags,
  // hats, and glasses stay in the same visual language as the garment.
  const palette = STYLES[activeStyle].clothes || VISION_CLOTHES;
  return palette[colorIndex % palette.length];
}

function visionToonShade(color, redValue, greenValue, blueValue) {
  const luminance =
    redValue * 0.2126 + greenValue * 0.7152 + blueValue * 0.0722;
  // Posterize the light into three broad bands instead of preserving the
  // camera's continuous shading. This gives background people the requested
  // illustrated/toon-shaded appearance while the closest person stays clean.
  const factor =
    luminance < 58 ? 0.48 : luminance < 118 ? 0.68 : luminance < 190 ? 0.88 : 1;
  return color.map((channel) => Math.round(channel * factor));
}

function isVisionToonEdge(pixel, labels, maskWidth, maskHeight) {
  const category = labels[pixel];
  const x = pixel % maskWidth;
  const y = Math.floor(pixel / maskWidth);
  return (
    (x > 0 && labels[pixel - 1] !== category) ||
    (x < maskWidth - 1 && labels[pixel + 1] !== category) ||
    (y > 0 && labels[pixel - maskWidth] !== category) ||
    (y < maskHeight - 1 && labels[pixel + maskWidth] !== category)
  );
}

function findVisionPeople(labels, maskWidth, maskHeight) {
  const componentIds = new Int32Array(labels.length);
  componentIds.fill(-1);
  const components = [];

  for (let start = 0; start < labels.length; start++) {
    if (labels[start] === 0 || componentIds[start] !== -1) continue;

    const id = components.length;
    let head = 0;
    let tail = 0;
    let count = 0;
    let sumX = 0;
    let sumY = 0;
    visionBfsQueue[tail++] = start;
    componentIds[start] = id;

    while (head < tail) {
      const pixel = visionBfsQueue[head++];
      const x = pixel % maskWidth;
      const y = Math.floor(pixel / maskWidth);
      count++;
      sumX += x;
      sumY += y;

      for (let yOffset = -1; yOffset <= 1; yOffset++) {
        for (let xOffset = -1; xOffset <= 1; xOffset++) {
          if (xOffset === 0 && yOffset === 0) continue;
          const nextX = x + xOffset;
          const nextY = y + yOffset;
          if (
            nextX < 0 ||
            nextX >= maskWidth ||
            nextY < 0 ||
            nextY >= maskHeight
          )
            continue;

          const neighbour = nextY * maskWidth + nextX;
          if (labels[neighbour] === 0 || componentIds[neighbour] !== -1)
            continue;
          componentIds[neighbour] = id;
          visionBfsQueue[tail++] = neighbour;
        }
      }
    }

    components.push({ id, count, x: sumX / count, y: sumY / count });
  }

  return { componentIds, components };
}

function findVisionForeground(componentIds, components, maskWidth, maskHeight) {
  const significant = components.filter((component) => component.count > 80);
  if (!significant.length) return -1;

  // Ignore tiny distant faces after calibration. This prevents a person far
  // behind the intended interaction zone from stealing the foreground role.
  const usableFace =
    visionClosestFace &&
    (!visionCalibrationReady ||
      visionClosestFace.area >= visionCalibratedFaceArea * 0.18);

  if (usableFace && visionCamera?.videoWidth && visionCamera?.videoHeight) {
    const x = constrain(
      Math.floor((usableFace.x / visionCamera.videoWidth) * maskWidth),
      0,
      maskWidth - 1,
    );
    const y = constrain(
      Math.floor((usableFace.y / visionCamera.videoHeight) * maskHeight),
      0,
      maskHeight - 1,
    );
    const directHit = componentIds[y * maskWidth + x];
    if (directHit >= 0) return directHit;

    let nearest = significant[0];
    let nearestDistance = Infinity;
    for (const component of significant) {
      const distance = sq(component.x - x) + sq(component.y - y);
      if (distance < nearestDistance) {
        nearest = component;
        nearestDistance = distance;
      }
    }
    return nearest.id;
  }

  return significant.reduce((largest, component) =>
    component.count > largest.count ? component : largest,
  ).id;
}

function trackVisionPeople(components, foregroundId) {
  const nextTracks = [];
  const people = new Map();
  const matchedPrevious = new Set();

  for (const component of components.filter((item) => item.count > 80)) {
    let match;
    let matchDistance = 42 * 42;
    for (const previous of visionPersonTracks) {
      if (matchedPrevious.has(previous.trackId)) continue;
      const distance =
        sq(component.x - previous.x) + sq(component.y - previous.y);
      if (distance < matchDistance) {
        match = previous;
        matchDistance = distance;
      }
    }
    if (match) matchedPrevious.add(match.trackId);

    const person = {
      ...component,
      trackId: match?.trackId ?? visionNextTrackId++,
      colorIndex: match
        ? match.colorIndex
        : visionNextClothesColor++ % VISION_CLOTHES.length,
      foreground: component.id === foregroundId,
    };
    nextTracks.push(person);
    people.set(component.id, person);
  }

  visionPersonTracks = nextTracks;
  return people;
}

function drawVisionSegmentation() {
  if (!(visionLayer instanceof HTMLCanvasElement) || !visionTargetImageData) return;

  // Draw only the current generated mask. There is intentionally no blurred
  // history pass: the artwork should show one current person layer at a time.
  const context = drawingContext;
  context.save();
  context.translate(width, 0);
  context.scale(-1, 1);
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  const destination = getVisionCoverDestination(
    visionLayer.width,
    visionLayer.height,
    width,
    height,
  );
  const personEntries = [...visionPersonLayers.values()];
  if (personEntries.length) {
    for (const entry of personEntries) {
      const coreBlur = getVisionPersonBlur(entry.face, visionDetectedFaces);
      const outerBlur = coreBlur <= VISION_CORE_BLUR
        ? VISION_OUTER_BLUR
        : coreBlur + 1;
      drawVisionCanvasLayer(
        context,
        entry.canvas,
        destination,
        coreBlur,
        outerBlur,
      );
    }
  } else {
    drawVisionCanvasLayer(
      context,
      visionLayer,
      destination,
      VISION_CORE_BLUR,
      VISION_OUTER_BLUR,
    );
  }
  context.filter = "none";
  context.restore();
}

function drawVisionCanvasLayer(context, layer, destination, coreBlur, outerBlur) {
  // A low-alpha soft under-print creates the requested 2px outer blend while
  // the main pass keeps the person readable at its distance-based blur.
  context.globalAlpha = 0.22;
  context.filter = `blur(${outerBlur}px)`;
  context.drawImage(
    layer,
    0,
    0,
    layer.width,
    layer.height,
    destination.x,
    destination.y,
    destination.width,
    destination.height,
  );
  context.globalAlpha = 0.88;
  context.filter = `blur(${coreBlur}px)`;
  context.drawImage(
    layer,
    0,
    0,
    layer.width,
    layer.height,
    destination.x,
    destination.y,
    destination.width,
    destination.height,
  );
}

function getVisionPersonBlur(face, faces) {
  if (!face || !faces.length) return VISION_CORE_BLUR;
  const sorted = [...faces].sort((a, b) => b.area - a.area);
  const largestArea = Math.max(sorted[0].area, 1);
  const rank = sorted.findIndex((item) => item.trackId === face.trackId);
  const relativeArea = face.area / largestArea;

  // Face area is a practical monocular distance estimate: a smaller face in
  // the same camera view is treated as farther away and receives more blur.
  if (rank === 0) return VISION_NEAREST_DISTANCE_BLUR;
  if (relativeArea >= 0.55) return VISION_MID_DISTANCE_BLUR;
  return VISION_FAR_DISTANCE_BLUR;
}

function drawVisionCalibrationGuide() {
  const elapsed = performance.now() - visionCalibrationStartedAt;
  const progress = constrain(elapsed / VISION_CALIBRATION_DURATION, 0, 1);
  const guideWidth = Math.min(width * 0.34, 310);
  const guideHeight = Math.min(height * 0.6, 420);

  push();
  noFill();
  stroke(255, 246);
  strokeWeight(2);
  drawingContext.setLineDash([8, 10]);
  ellipse(width * 0.5, height * 0.49, guideWidth, guideHeight);
  drawingContext.setLineDash([]);

  noStroke();
  fill(22, 16, 30, 175);
  rect(width * 0.5 - 128, height * 0.12, 256, 48, 24);
  fill(255);
  textAlign(CENTER, CENTER);
  textSize(14);
  text("화면 중앙에 서주세요", width * 0.5, height * 0.12 + 18);
  textSize(11);
  text(
    `${Math.max(0, Math.ceil((VISION_CALIBRATION_DURATION - elapsed) / 1000))}초`,
    width * 0.5,
    height * 0.12 + 35,
  );

  fill(255, 220);
  rect(width * 0.5 - 100, height * 0.12 + 56, 200 * progress, 3, 2);
  pop();
}

function drawVisionStatus() {
  if (window.__OCTOBER_FALLBACK_VIDEO) visionStatus = "안정 모드 · 카메라 연결됨";
  const message = `[OCTOBER] ${visionStatus}`;
  if (message !== lastLoggedVisionStatus) {
    console.info(message);
    lastLoggedVisionStatus = message;
  }
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
  createWhiteResidues();
  createFineGrain();
  createMicroDots();
  createFloatingDots();
  createPrintTexture();
  if (printTextureLayer)
    printTextureLayer.resizeCanvas(windowWidth, windowHeight);
  createGlobalIllustrationTexture();
  createSunMarks();
  createFloorMarks();
  cachedBackground = null;
  backgroundNeedsRefresh = true;
}
