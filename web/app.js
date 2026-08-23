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

async function loadTokenizer() {
  const response = await fetch(TOKENIZER_URL);
  if (!response.ok) {
    throw new Error(`SentencePiece model download failed (${response.status})`);
  }

  const modelData = new Uint8Array(await response.arrayBuffer());
  tokenizer = getSentencePieceTokenizer({ modelData });
}

async function init() {
  try {
    statusEl.textContent = 'モデルを読み込んでいます…';

    ort.env.wasm.numThreads = globalThis.crossOriginIsolated
      ? Math.min(navigator.hardwareConcurrency || 1, 4)
      : 1;

    [session] = await Promise.all([
      ort.InferenceSession.create(MODEL_URL, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      }),
      loadTokenizer(),
    ]);

    statusEl.textContent = '準備完了。推論はこの端末内だけで実行されます。';
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
