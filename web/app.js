import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.min.mjs';
import { getSentencePieceTokenizer } from 'https://esm.sh/ai-token-estimator';

const MODEL_URL = 'https://huggingface.co/h6e/KuchoLM-NIDA-10M/resolve/main/model.onnx';
const TOKENIZER_URL = 'https://huggingface.co/h6e/KuchoLM-NIDA-10M/resolve/main/kucholm_spm.model';

const PAD = 0;
const BOS = 2;
const EOS = 3;
const MAX_LEN = 128;
const STYLE_PREFIX = '<NIDA_FICTION> ';

const statusEl = document.querySelector('#status');
const convertBtn = document.querySelector('#convert');
const inputEl = document.querySelector('#input');
const outputEl = document.querySelector('#output');

let session = null;
let tokenizer = null;

function toInt64Tensor(ids) {
  return new ort.Tensor('int64', BigInt64Array.from(ids, (x) => BigInt(x)), [1, ids.length]);
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return '';
  const units = ['B', 'KB', 'MB', 'GB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit++;
  }
  return `${value.toFixed(value >= 10 || unit === 0 ? 0 : 1)} ${units[unit]}`;
}

async function downloadWithProgress(url, label, overallStart = 0, overallSpan = 100) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${label} download failed (${response.status})`);
  if (!response.body) return new Uint8Array(await response.arrayBuffer());

  const total = Number(response.headers.get('content-length')) || 0;
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.byteLength;

    if (total > 0) {
      const localPercent = Math.min(100, Math.round((received / total) * 100));
      const overallPercent = Math.min(100, Math.round(overallStart + (localPercent / 100) * overallSpan));
      statusEl.textContent = `${label}をダウンロード中… ${localPercent}% (${formatBytes(received)} / ${formatBytes(total)}) — 全体 ${overallPercent}%`;
    } else {
      statusEl.textContent = `${label}をダウンロード中… ${formatBytes(received)}`;
    }
  }

  const result = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return result;
}

function argmaxLastToken(logits, generated, repetitionPenalty = 1.15) {
  const dims = logits.dims;
  const vocab = dims[dims.length - 1];
  const seqLen = dims[dims.length - 2];
  const offset = (seqLen - 1) * vocab;
  const seen = new Set(generated);

  let bestId = 0;
  let bestScore = -Infinity;
  for (let id = 0; id < vocab; id++) {
    let score = logits.data[offset + id];
    if (seen.has(id) && id !== BOS && id !== EOS) {
      score = score >= 0 ? score / repetitionPenalty : score * repetitionPenalty;
    }
    if (score > bestScore) {
      bestScore = score;
      bestId = id;
    }
  }
  return bestId;
}

async function init() {
  try {
    convertBtn.disabled = true;
    convertBtn.textContent = 'Loading…';

    ort.env.wasm.numThreads = globalThis.crossOriginIsolated
      ? Math.min(navigator.hardwareConcurrency || 1, 4)
      : 1;

    const modelBytes = await downloadWithProgress(MODEL_URL, 'モデル', 0, 90);
    statusEl.textContent = 'モデルを初期化中… 90%';
    session = await ort.InferenceSession.create(modelBytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });

    const tokenizerBytes = await downloadWithProgress(TOKENIZER_URL, 'Tokenizer', 90, 10);
    statusEl.textContent = 'Tokenizerを初期化中… 100%';
    tokenizer = getSentencePieceTokenizer({ modelData: tokenizerBytes });

    statusEl.textContent = '準備完了。ダウンロード 100%。推論はこの端末内だけで実行されます。';
    convertBtn.textContent = 'Convert';
    convertBtn.disabled = false;
  } catch (err) {
    console.error(err);
    statusEl.textContent = `読み込み失敗: ${err.message}`;
    convertBtn.textContent = 'Model unavailable';
    convertBtn.disabled = true;
  }
}

async function generate(text) {
  const encodedRaw = tokenizer.encode(STYLE_PREFIX + text);
  const encoded = Array.from(encodedRaw, Number);
  const srcIds = [BOS, ...encoded.slice(0, MAX_LEN - 2), EOS];
  const generated = [BOS];

  for (let step = 0; step < MAX_LEN - 1; step++) {
    const feeds = {
      src: toInt64Tensor(srcIds),
      tgt_in: toInt64Tensor(generated),
    };
    const result = await session.run(feeds);
    const logits = result.logits ?? result[session.outputNames[0]];
    const nextId = argmaxLastToken(logits, generated);

    if (nextId === EOS) break;
    if (nextId !== PAD && nextId !== BOS) generated.push(nextId);
  }

  return tokenizer.decode(Uint32Array.from(generated.slice(1)));
}

convertBtn.addEventListener('click', async () => {
  const text = inputEl.value.trim();
  if (!text || !session || !tokenizer) return;

  convertBtn.disabled = true;
  convertBtn.textContent = 'Converting…';
  statusEl.textContent = '端末内で推論中…';
  outputEl.value = '';

  try {
    outputEl.value = await generate(text);
    statusEl.textContent = '完了。入力文はサーバーへ送信されていません。';
  } catch (err) {
    console.error(err);
    outputEl.value = '';
    statusEl.textContent = `推論失敗: ${err.message}`;
  } finally {
    convertBtn.disabled = false;
    convertBtn.textContent = 'Convert';
  }
});

init();
