// OCTOBER — camera-based painterly person study
//
// The reference images are used as visual direction, not as overlays.
// ml5.js separates the person from the camera background, and p5.js redraws
// the person as a soft, colorful, grainy figure that responds to movement.

let camera;
let segmenter;
let personMask;
let personLayer;
let modelReady = false;
let modelStatus = "loading person model...";

let activeStyle = 0;
let lastStyleChange = 0;
const STYLE_INTERVAL = 20000;

let personCenter = null;
let previousCenter = null;
let movementAmount = 0;
let particles = [];

const STYLES = [
  {
    name: "warm",
    background: [255, 188, 174],
    person: [255, 92, 116],
    light: [255, 221, 126],
    accent: [83, 146, 221],
  },
  {
    name: "blue",
    background: [48, 88, 170],
    person: [91, 154, 236],
    light: [226, 238, 255],
    accent: [255, 164, 113],
  },
  {
    name: "green",
    background: [67, 132, 109],
    person: [255, 134, 87],
    light: [255, 237, 164],
    accent: [73, 142, 230],
  },
  {
    name: "dream",
    background: [190, 167, 221],
    person: [249, 135, 190],
    light: [255, 232, 177],
    accent: [88, 111, 219],
  },
  {
    name: "rainbow",
    background: [119, 180, 218],
    person: [247, 120, 100],
    light: [255, 230, 97],
    accent: [145, 104, 234],
  },
];

async function setup() {
  const canvas = createCanvas(windowWidth, windowHeight);
  canvas.parent("canvas-container");
  pixelDensity(1);

  camera = createCapture(VIDEO);
  camera.size(width, height);
  camera.hide();

  personLayer = createGraphics(width, height);
  personLayer.pixelDensity(1);

  // Full-body segmentation is needed here; pose keypoints alone would not
  // be enough to create the painterly silhouette in the references.
  try {
    segmenter = await ml5.bodySegmentation("SelfieSegmentation", {
      maskType: "person",
      runtime: "tfjs",
    });

    // ml5 expects the underlying HTMLVideoElement. Passing camera.elt makes
    // the connection reliable across different p5.js versions.
    segmenter.detectStart(camera.elt || camera, gotSegmentation);
    modelReady = true;
    modelStatus = "person tracking ready";
  } catch (error) {
    modelStatus = "person model could not load";
    console.error(error);
  }

  lastStyleChange = millis();
}

function draw() {
  const style = STYLES[activeStyle];

  background(style.background[0], style.background[1], style.background[2]);
  drawCameraAtmosphere(style);

  if (millis() - lastStyleChange >= STYLE_INTERVAL) {
    activeStyle = (activeStyle + 1) % STYLES.length;
    lastStyleChange = millis();
  }

  if (personMask) {
    personCenter = findMaskCenter(personMask);
    updateMovement();
    drawPainterlyPerson(style);
    drawMovementTrails(style);
    emitParticles(style);
  }

  updateAndDrawParticles();
  drawGrain(style);

  // Useful while developing; remove this text for the final exhibition build.
  fill(255, 170);
  noStroke();
  textSize(12);
  text(modelStatus, 16, height - 18);
}

function gotSegmentation(results) {
  const result = Array.isArray(results) ? results[0] : results;
  if (result && result.mask) {
    personMask = result.mask;
    modelStatus = "person segmentation active";
  }
}

function drawCameraAtmosphere(style) {
  // Keep the real space faintly visible while shifting it toward a dreamlike
  // color atmosphere.
  push();
  tint(255, 70);
  image(camera, 0, 0, width, height);
  blendMode(SCREEN);
  noStroke();
  fill(style.light[0], style.light[1], style.light[2], 42);
  rect(0, 0, width, height);
  blendMode(BLEND);
  noTint();
  pop();
}

function drawPainterlyPerson(style) {
  personLayer.clear();

  // Tint the live person before applying the segmentation mask.
  personLayer.push();
  personLayer.tint(255, 180);
  personLayer.image(camera, 0, 0, width, height);
  personLayer.blendMode(SCREEN);
  personLayer.noStroke();
  personLayer.fill(style.person[0], style.person[1], style.person[2], 190);
  personLayer.rect(0, 0, width, height);
  personLayer.fill(style.light[0], style.light[1], style.light[2], 65);
  personLayer.rect(0, 0, width, height);
  personLayer.pop();

  // Clip the colored layer to the detected person silhouette.
  const maskSource = getMaskSource(personMask);
  if (maskSource) {
    const context = personLayer.canvas.getContext("2d");
    context.save();
    context.globalCompositeOperation = "destination-in";
    context.drawImage(maskSource, 0, 0, width, height);
    context.restore();
  }

  image(personLayer, 0, 0, width, height);

  // A soft halo prevents the result from feeling like a hard cutout.
  if (personCenter) {
    noStroke();
    blendMode(SCREEN);
    fill(style.light[0], style.light[1], style.light[2], 35);
    ellipse(personCenter.x, personCenter.y, 190 + movementAmount * 3);
    blendMode(BLEND);
  }
}

function drawMovementTrails(style) {
  if (!personCenter || !previousCenter) return;

  const distanceMoved = dist(
    personCenter.x,
    personCenter.y,
    previousCenter.x,
    previousCenter.y,
  );
  if (distanceMoved < 2) return;

  noStroke();
  blendMode(SCREEN);
  for (let i = 0; i < 7; i++) {
    const amount = i / 7;
    const x = lerp(previousCenter.x, personCenter.x, amount);
    const y = lerp(previousCenter.y, personCenter.y, amount);
    fill(style.accent[0], style.accent[1], style.accent[2], 28 - i * 3);
    ellipse(x, y, 80 + i * 18, 120 + i * 20);
  }
  blendMode(BLEND);
}

function updateMovement() {
  if (!personCenter) return;

  if (previousCenter) {
    const currentDistance = dist(
      personCenter.x,
      personCenter.y,
      previousCenter.x,
      previousCenter.y,
    );
    movementAmount = lerp(movementAmount, currentDistance, 0.18);
  }

  previousCenter = { x: personCenter.x, y: personCenter.y };
}

function emitParticles(style) {
  if (!personCenter) return;

  const previousX = previousCenter ? previousCenter.x : personCenter.x;
  const amount = constrain(2 + movementAmount * 0.12, 2, 14);
  for (let i = 0; i < amount; i++) {
    particles.push({
      x: personCenter.x + random(-55, 55),
      y: personCenter.y + random(-90, 90),
      vx: random(-0.7, 0.7) + (personCenter.x - previousX) * 0.02,
      vy: random(-1.1, 0.4),
      size: random(1, 6),
      life: 255,
      color: random() > 0.5 ? style.light : style.accent,
    });
  }
}

function updateAndDrawParticles() {
  noStroke();
  blendMode(SCREEN);

  for (let i = particles.length - 1; i >= 0; i--) {
    const particle = particles[i];
    particle.x += particle.vx;
    particle.y += particle.vy;
    particle.vy -= 0.01;
    particle.life -= 2.5;

    fill(
      particle.color[0],
      particle.color[1],
      particle.color[2],
      particle.life,
    );
    circle(particle.x, particle.y, particle.size);

    if (particle.life <= 0) particles.splice(i, 1);
  }

  if (particles.length > 900) particles.splice(0, particles.length - 900);
  blendMode(BLEND);
}

function drawGrain(style) {
  stroke(style.light[0], style.light[1], style.light[2], 55);
  for (let i = 0; i < 1300; i++) point(random(width), random(height));
}

function findMaskCenter(mask) {
  const source = getMaskSource(mask);
  if (!source) return null;

  const maskWidth = source.width || width;
  const maskHeight = source.height || height;
  const scratch = document.createElement("canvas");
  scratch.width = maskWidth;
  scratch.height = maskHeight;
  const context = scratch.getContext("2d");
  context.drawImage(source, 0, 0);
  const pixels = context.getImageData(0, 0, maskWidth, maskHeight).data;

  let totalX = 0;
  let totalY = 0;
  let count = 0;
  const step = 8;

  for (let y = 0; y < maskHeight; y += step) {
    for (let x = 0; x < maskWidth; x += step) {
      const index = (y * maskWidth + x) * 4;
      const brightness = pixels[index] + pixels[index + 1] + pixels[index + 2];
      if (brightness > 330) {
        totalX += x;
        totalY += y;
        count++;
      }
    }
  }

  if (!count) return null;
  return {
    x: (totalX / count / maskWidth) * width,
    y: (totalY / count / maskHeight) * height,
  };
}

function getMaskSource(mask) {
  if (!mask) return null;
  return mask.canvas || mask.elt || mask;
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
  if (camera) camera.size(windowWidth, windowHeight);
  if (personLayer) personLayer.resizeCanvas(windowWidth, windowHeight);
}
