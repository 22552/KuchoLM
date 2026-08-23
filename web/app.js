import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.min.mjs';

const MODEL_URL = 'https://huggingface.co/h6e/KuchoLM-NIDA-10M/resolve/main/model.onnx';
const TOKENIZER_URL = 'https://huggingface.co/h6e/KuchoLM-NIDA-10M/resolve/main/kucholm_spm.model';

const statusEl = document.querySelector('#status');
const convertBtn = document.querySelector('#convert');
const inputEl = document.querySelector('#input');
const outputEl = document.querySelector('#output');

let session = null;
let tokenizerBytes = null;

async function init() {
  try {
    statusEl.textContent = 'モデルを読み込んでいます…';
    ort.env.wasm.numThreads = Math.min(navigator.hardwareConcurrency || 1, 4);
    session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });

    // SentencePiece は次段階でWASM tokenizer実装へ接続する。
    // 今はモデル配布物が揃っているかを先に検証する。
    const tokRes = await fetch(TOKENIZER_URL);
    if (!tokRes.ok) throw new Error('SentencePiece model not found on Hugging Face');
    tokenizerBytes = new Uint8Array(await tokRes.arrayBuffer());

    statusEl.textContent = 'ONNXモデル読み込み完了。Tokenizer接続待ち。';
    convertBtn.textContent = 'Convert';
    convertBtn.disabled = true;
  } catch (err) {
    console.error(err);
    statusEl.textContent = `読み込み失敗: ${err.message}`;
    convertBtn.textContent = 'Model unavailable';
  }
}

convertBtn.addEventListener('click', async () => {
  if (!session || !tokenizerBytes) return;
  outputEl.value = 'Tokenizer実装を接続すると、ここで完全ブラウザ推論できます。';
});

init();
