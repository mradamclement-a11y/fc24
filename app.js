/* FC 24 Position Predictor — 4 features (pace, shooting, passing, defending), raw inputs, with guards */
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
const debug = (msg) => { const d = el("debug"); if (d) d.textContent = String(msg || ""); };

function must(id){
  const n = el(id);
  if(!n) throw new Error(`Missing element with id="#${id}". Check index.html and cache.`);
  return n;
}

async function loadModel(){
  const status = el("modelStatus");
  if (status) status.textContent = "Loading model from ./my-model.json…";
  model = await tf.loadLayersModel("./my-model.json");
  tf.tidy(()=> model.predict(tf.tensor2d([[70,70,70,70]]))); // warm up with 4 raw values
  if (status) status.textContent = "Model loaded ✓";
  const predictBtn = el("predictBtn");
  if (predictBtn) predictBtn.disabled = false;
}

function readInputs(){
  const pace = Number(must("pace").value);
  const shooting = Number(must("shooting").value);
  const passing = Number(must("passing").value);
  const defending = Number(must("defending").value);
  return [pace, shooting, passing, defending];
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
  const badge = el("badge");
  if (badge) badge.textContent = best.label;
  const top3Div = el("top3");
  if (top3Div) top3Div.innerHTML = top3.map(x => `${x.label}: ${(x.p*100).toFixed(1)}%`).join("<br>");
  const [px,py] = (POSITION_ANCHORS[best.label] || [0.5,0.5]);
  const marker = el("marker");
  if (marker){
    marker.style.left = `calc(${px*100}% - 6px)`;
    marker.style.top  = `calc(${py*100}% - 6px)`;
  }
  debug(`Sum(probs) ≈ ${(probs.reduce((a,b)=>a+b,0)).toFixed(3)}`);
}

async function predict(){
  if(!model) return;
  const x = readInputs(); // RAW values
  const pred = tf.tidy(()=> model.predict(tf.tensor2d([x])));
  let raw = Array.from(await pred.data());
  pred.dispose();
  if (!raw.length){ debug("Model returned empty output."); return; }
  if (raw.some(n => !Number.isFinite(n))){
    debug("Model produced NaN/Inf."); raw = raw.map(v => Number.isFinite(v) ? v : 0);
  }
  const probs = ensureProbabilities(raw);
  const top3 = topK(probs, 3);
  updateUI(top3[0], top3, probs);
}

function wireUI(){
  try{
    const pace = must("pace");
    const shooting = must("shooting");
    const passing = must("passing");
    const defending = must("defending");
    const predictBtn = must("predictBtn");

    const sync = (input, spanId) => {
      const span = must(spanId);
      input.addEventListener("input", () => { span.textContent = input.value; });
    };
    sync(pace, "paceVal");
    sync(shooting, "shootVal");
    sync(passing, "passVal");
    sync(defending, "defVal");

    predictBtn.addEventListener("click", predict);
  }catch(err){
    console.error(err);
    debug(err.message || String(err));
  }
}

window.addEventListener("DOMContentLoaded", async () => {
  wireUI();
  try{
    await loadModel();
  }catch(e){
    console.error(e);
    const status = el("modelStatus");
    if (status) status.textContent = "Failed to load model. Ensure files are in the same folder.";
  }
});
