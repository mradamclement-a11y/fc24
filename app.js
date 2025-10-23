/* FC 24 Position Predictor — File-Safe version
 * Adds support for loading a TFJS model via <input type="file"> to avoid CORS errors.
 */
let model = null;

// EDIT THESE IF YOUR LABEL ORDER DIFFERs:
const POSITION_LABELS = [
  "GK","RB","RWB","CB","LB","LWB","CDM","CM",
  "CAM","RM","LM","RW","LW","CF","ST","SW"
];

// Rough pitch anchors
const POSITION_ANCHORS = {
  GK:[0.1,0.5], RB:[0.3,0.82], RWB:[0.35,0.85], CB:[0.28,0.5], LB:[0.3,0.18],
  LWB:[0.35,0.15], CDM:[0.46,0.5], CM:[0.55,0.5], CAM:[0.65,0.5],
  RM:[0.6,0.8], LM:[0.6,0.2], RW:[0.78,0.78], LW:[0.78,0.22],
  CF:[0.82,0.5], ST:[0.9,0.5], SW:[0.2,0.5]
};

const el = (id) => document.getElementById(id);

async function tryLoadModelFromHttp() {
  el("modelStatus").textContent = "Loading model (HTTP)…";
  try {
    const m = await tf.loadLayersModel("my-model.json"); // requires HTTP(s)
    tf.tidy(()=> m.predict(tf.tensor2d([[70,70,70]])));
    model = m;
    el("modelStatus").textContent = "Model loaded ✓";
    el("predictBtn").disabled = false;
  } catch (err) {
    console.warn("HTTP load failed (expected on file://):", err);
    el("modelStatus").textContent = "Local file context detected. Use 'Load Model (Files)' below.";
    // reveal file panel
    document.getElementById("fileLoadPanel").hidden = false;
  }
}

async function loadModelFromFiles(files) {
  el("fileStatus").textContent = "Checking files…";
  // Accept 2 files in any order: one .json, one .bin
  const arr = Array.from(files || []);
  if (arr.length < 2) {
    el("fileStatus").textContent = "Please select both the JSON and the BIN files.";
    return;
  }
  const jsonFile = arr.find(f => f.name.endsWith(".json"));
  const binFile  = arr.find(f => f.name.endsWith(".bin"));
  if (!jsonFile || !binFile) {
    el("fileStatus").textContent = "Need a .json and a .bin file.";
    return;
  }
  try {
    el("fileStatus").textContent = "Loading model from files…";
    const m = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, binFile]));
    tf.tidy(()=> m.predict(tf.tensor2d([[70,70,70]])));
    model = m;
    el("fileStatus").textContent = "Model loaded ✓";
    el("modelStatus").textContent = "Model loaded ✓";
    el("predictBtn").disabled = false;
  } catch (e) {
    console.error(e);
    el("fileStatus").textContent = "Failed to load model from files.";
  }
}

function readInputs(){
  return [Number(el("pace").value), Number(el("shooting").value), Number(el("passing").value)];
}
function softmaxTopK(probs, k=3){
  const entries = probs.map((p,i)=>({label: POSITION_LABELS[i] ?? `Class${i}`, p}));
  entries.sort((a,b)=>b.p-a.p);
  return entries.slice(0,k);
}
function updateUI(best, top3){
  el("badge").textContent = best.label;
  el("top3").innerHTML = top3.map(x => `${x.label}: ${(x.p*100).toFixed(1)}%`).join("<br>");
  const [px,py] = (POSITION_ANCHORS[best.label] || [0.5,0.5]);
  const marker = el("marker");
  marker.style.left = `calc(${px*100}% - 6px)`;
  marker.style.top  = `calc(${py*100}% - 6px)`;
}

async function predict(){
  if(!model) return;
  const pred = tf.tidy(()=> model.predict(tf.tensor2d([readInputs()])));
  const probs = Array.from(await pred.data());
  pred.dispose();
  const top3 = softmaxTopK(probs,3);
  updateUI(top3[0], top3);
}

function wireUI(){
  const sync = (id, span) => el(id).addEventListener("input", () => el(span).textContent = el(id).value);
  sync("pace","paceVal"); sync("shooting","shootVal"); sync("passing","passVal");
  el("predictBtn").addEventListener("click", predict);
  el("loadLocalBtn").addEventListener("click", ()=> { document.getElementById("fileLoadPanel").hidden = false; });
  el("useFilesBtn").addEventListener("click", ()=> loadModelFromFiles(el("filePicker").files));
}

window.addEventListener("DOMContentLoaded", ()=>{
  wireUI();
  tryLoadModelFromHttp();
});
