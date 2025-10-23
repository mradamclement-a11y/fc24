/* FC 24 Position Predictor — Raw inputs version (no scaling) */
let model = null;

// Adjust to your training class order:
const POSITION_LABELS = [
  "GK","RB","RWB","CB","LB","LWB","CDM","CM",
  "CAM","RM","LM","RW","LW","CF","ST","SW"
];

const POSITION_ANCHORS = {
  GK:[0.1,0.5], RB:[0.3,0.82], RWB:[0.35,0.85], CB:[0.28,0.5], LB:[0.3,0.18],
  LWB:[0.35,0.15], CDM:[0.46,0.5], CM:[0.55,0.5], CAM:[0.65,0.5],
  RM:[0.6,0.8], LM:[0.6,0.2], RW:[0.78,0.78], LW:[0.78,0.22],
  CF:[0.82,0.5], ST:[0.9,0.5], SW:[0.2,0.5]
};

const el = (id) => document.getElementById(id);
const debug = (msg) => { const d = el("debug"); d.textContent = String(msg || ""); };

async function loadModel(){
  el("modelStatus").textContent = "Loading model from ./my-model.json…";
  model = await tf.loadLayersModel("./my-model.json");
  tf.tidy(()=> model.predict(tf.tensor2d([[70,70,70]]))); // warm up with raw values
  el("modelStatus").textContent = "Model loaded ✓";
  el("predictBtn").disabled = false;
}

function readInputs(){
  return [Number(el("pace").value), Number(el("shooting").value), Number(el("passing").value)];
}

function softmax(arr){
  const m = Math.max(...arr);
  const exps = arr.map(v => Math.exp(v - m));
  const s = exps.reduce((a,b)=>a+b,0) || 1;
  return exps.map(v => v / s);
}

function ensureProbabilities(raw){
  const sum = raw.reduce((a,b)=>a+b,0);
  const inRange = raw.every(v => v >= 0 && v <= 1);
  if (inRange && Math.abs(sum - 1) < 0.05) return raw;
  return softmax(raw);
}

function topK(probs, k=3){
  const entries = probs.map((p,i)=>({label: POSITION_LABELS[i] ?? `Class${i}`, p}));
  entries.sort((a,b)=>b.p-a.p);
  return entries.slice(0,k);
}

function updateUI(best, top3, probs){
  el("badge").textContent = best.label;
  el("top3").innerHTML = top3.map(x => `${x.label}: ${(x.p*100).toFixed(1)}%`).join("<br>");
  const [px,py] = (POSITION_ANCHORS[best.label] || [0.5,0.5]);
  const marker = el("marker");
  marker.style.left = `calc(${px*100}% - 6px)`;
  marker.style.top  = `calc(${py*100}% - 6px)`;
  debug(`Sum(probs) ≈ ${(probs.reduce((a,b)=>a+b,0)).toFixed(3)}`);
}

async function predict(){
  if(!model) return;
  const x = readInputs(); // RAW values
  const pred = tf.tidy(()=> model.predict(tf.tensor2d([x])));
  let raw = Array.from(await pred.data());
  pred.dispose();

  if (!raw.length){
    debug("Model returned empty output. Check final layer size.");
    return;
  }
  if (raw.some(n => !Number.isFinite(n))){
    debug("Model produced NaN/Inf. Check inputs and model.");
    raw = raw.map(v => Number.isFinite(v) ? v : 0);
  }
  const probs = ensureProbabilities(raw);
  const top3 = topK(probs, 3);
  updateUI(top3[0], top3, probs);
}

function wireUI(){
  const sync = (id, span) => el(id).addEventListener("input", () => el(span).textContent = el(id).value);
  sync("pace","paceVal"); sync("shooting","shootVal"); sync("passing","passVal");
  el("predictBtn").addEventListener("click", predict);
}

window.addEventListener("DOMContentLoaded", async () => {
  wireUI();
  try{
    await loadModel();
  }catch(e){
    console.error(e);
    el("modelStatus").textContent = "Failed to load model. Ensure files are in the same folder.";
  }
});
