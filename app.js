/* FC 24 Position Predictor (HTML/CSS/JS + TensorFlow.js)
 * Assumes the provided tfjs model (my-model.json & my-model.weights.bin) is in the same directory.
 * The model expects 3 numeric inputs (e.g., Pace, Shooting, Passing).
 * Position labels are a best-guess set of 16 roles — edit as needed to match your training.
 */
let model = null;

// EDIT THESE IF YOUR LABEL ORDER DIFFERs:
const POSITION_LABELS = [
  "GK","RB","RWB","CB","LB","LWB","CDM","CM",
  "CAM","RM","LM","RW","LW","CF","ST","SW"
];

// Rough pitch coordinates for where the position plays (0..1 in X/Y for our pitch box)
const POSITION_ANCHORS = {
  GK:  [0.1, 0.5],
  RB:  [0.3, 0.82],
  RWB: [0.35,0.85],
  CB:  [0.28,0.5],
  LB:  [0.3, 0.18],
  LWB: [0.35,0.15],
  CDM: [0.46,0.5],
  CM:  [0.55,0.5],
  CAM: [0.65,0.5],
  RM:  [0.6, 0.8],
  LM:  [0.6, 0.2],
  RW:  [0.78,0.78],
  LW:  [0.78,0.22],
  CF:  [0.82,0.5],
  ST:  [0.9, 0.5],
  SW:  [0.2, 0.5]
};

const el = (id) => document.getElementById(id);

async function loadModel() {
  try {
    el("modelStatus").textContent = "Loading model…";
    model = await tf.loadLayersModel("my-model.json");
    // Warm up
    tf.tidy(() => model.predict(tf.tensor2d([[70,70,70]])));
    el("modelStatus").textContent = "Model loaded ✓";
    el("predictBtn").disabled = false;
  } catch (err) {
    console.error(err);
    el("modelStatus").textContent = "Failed to load model. Check files are alongside index.html";
    el("predictBtn").disabled = true;
  }
}

function readInputs() {
  const pace = Number(el("pace").value);
  const shooting = Number(el("shooting").value);
  const passing = Number(el("passing").value);
  return [pace, shooting, passing];
}

function argmax(arr) {
  return arr.indexOf(Math.max(...arr));
}

function softmaxToTopK(probs, k=3) {
  const entries = probs.map((p,i)=>({i, p, label: POSITION_LABELS[i] ?? `Class${i}`}));
  entries.sort((a,b)=>b.p-a.p);
  return entries.slice(0,k);
}

function updateBadge(bestLabel) {
  el("badge").textContent = bestLabel;
}

function updateTop3(top3) {
  const lines = top3.map(x => `${x.label}: ${(x.p*100).toFixed(1)}%`);
  el("top3").innerHTML = lines.join("<br>");
}

function updateMarker(bestLabel) {
  const marker = el("marker");
  const field = document.querySelector(".field");
  const rect = field.getBoundingClientRect();
  const [px, py] = POSITION_ANCHORS[bestLabel] || [0.5,0.5];
  // position within the field
  marker.style.left = `calc(${px*100}% - 6px)`;
  marker.style.top  = `calc(${py*100}% - 6px)`;
}

async function predict() {
  if (!model) return;
  const x = readInputs();
  // NOTE: If you trained on scaled inputs, scale here the same way.
  const pred = tf.tidy(() => model.predict(tf.tensor2d([x])));
  const probs = Array.from(await pred.data());
  pred.dispose();

  const k = softmaxToTopK(probs, 3);
  const best = k[0];
  updateBadge(best.label);
  updateTop3(k);
  updateMarker(best.label);
}

function wireSliders() {
  const sync = (id, span) => {
    el(id).addEventListener("input", () => el(span).textContent = el(id).value);
  };
  sync("pace","paceVal");
  sync("shooting","shootVal");
  sync("passing","passVal");
}

window.addEventListener("DOMContentLoaded", () => {
  wireSliders();
  loadModel();
  el("predictBtn").addEventListener("click", predict);
});
