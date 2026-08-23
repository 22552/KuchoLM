import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.min.mjs';

const MODEL_URL = 'https://huggingface.co/h6e/KuchoLM-NIDA-10M/resolve/main/model.onnx';
const TOKENIZER_URL = 'https://huggingface.co/h6e/KuchoLM-NIDA-10M/resolve/main/kucholm_spm.model';

const PAD = 0;
const UNK = 1;
const BOS = 2;
const EOS = 3;
const MAX_LEN = 128;
const STYLE_PREFIX = '<NIDA_FICTION> ';
const SPIECE_UNDERLINE = '▁';

const statusEl = document.querySelector('#status');
const convertBtn = document.querySelector('#convert');
const inputEl = document.querySelector('#input');
const outputEl = document.querySelector('#output');

let session = null;
let tokenizer = null;

function setStatus(message) {
  statusEl.textContent = message;
}

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
  setStatus(`${label}へ接続中…`);
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) throw new Error(`${label} download failed (${response.status})`);

  if (!response.body) {
    setStatus(`${label}をダウンロード中…`);
    return new Uint8Array(await response.arrayBuffer());
  }

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
      setStatus(`${label}をダウンロード中… ${localPercent}% (${formatBytes(received)} / ${formatBytes(total)}) — 全体 ${overallPercent}%`);
    } else {
      setStatus(`${label}をダウンロード中… ${formatBytes(received)}`);
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

// Minimal protobuf reader for SentencePiece ModelProto. KuchoLM uses BPE, so we
// only need ModelProto.pieces and each SentencePiece { piece, score, type }.
function readVarint(bytes, state) {
  let value = 0;
  let shift = 0;
  while (state.i < bytes.length) {
    const b = bytes[state.i++];
    value += (b & 0x7f) * 2 ** shift;
    if ((b & 0x80) === 0) return value;
    shift += 7;
    if (shift > 49) throw new Error('Tokenizer protobuf varint is too large');
  }
  throw new Error('Unexpected end of tokenizer protobuf');
}

function skipField(bytes, state, wire) {
  if (wire === 0) {
    readVarint(bytes, state);
  } else if (wire === 1) {
    state.i += 8;
  } else if (wire === 2) {
    state.i += readVarint(bytes, state);
  } else if (wire === 5) {
    state.i += 4;
  } else {
    throw new Error(`Unsupported protobuf wire type ${wire}`);
  }
  if (state.i > bytes.length) throw new Error('Tokenizer protobuf is truncated');
}

function parseSentencePieceMessage(bytes, start, end) {
  const state = { i: start };
  let piece = '';
  let score = 0;
  let type = 1; // NORMAL
  const decoder = new TextDecoder('utf-8');

  while (state.i < end) {
    const key = readVarint(bytes, state);
    const field = key >>> 3;
    const wire = key & 7;

    if (field === 1 && wire === 2) {
      const len = readVarint(bytes, state);
      piece = decoder.decode(bytes.subarray(state.i, state.i + len));
      state.i += len;
    } else if (field === 2 && wire === 5) {
      score = new DataView(bytes.buffer, bytes.byteOffset + state.i, 4).getFloat32(0, true);
      state.i += 4;
    } else if (field === 3 && wire === 0) {
      type = readVarint(bytes, state);
    } else {
      skipField(bytes, state, wire);
    }
  }

  return { piece, score, type };
}

function parseSentencePieceModel(bytes) {
  const state = { i: 0 };
  const pieces = [];

  while (state.i < bytes.length) {
    const key = readVarint(bytes, state);
    const field = key >>> 3;
    const wire = key & 7;

    if (field === 1 && wire === 2) {
      const len = readVarint(bytes, state);
      const start = state.i;
      const end = start + len;
      if (end > bytes.length) throw new Error('Tokenizer piece is truncated');
      pieces.push(parseSentencePieceMessage(bytes, start, end));
      state.i = end;
    } else {
      skipField(bytes, state, wire);
    }
  }

  if (pieces.length < 4) throw new Error('SentencePiece vocabulary could not be parsed');
  return pieces;
}

class BrowserSentencePieceBPE {
  constructor(modelBytes) {
    this.pieces = parseSentencePieceModel(modelBytes);
    this.pieceToId = new Map();
    this.userDefined = [];

    for (let id = 0; id < this.pieces.length; id++) {
      const p = this.pieces[id];
      this.pieceToId.set(p.piece, id);
      if (p.type === 4 && p.piece) this.userDefined.push(p.piece);
    }

    // Match longer user-defined symbols first.
    this.userDefined.sort((a, b) => b.length - a.length);
  }

  normalize(text) {
    // SentencePiece's default nmt_nfkc behavior is close to Unicode NFKC for
    // KuchoLM's Japanese corpus. Match its default whitespace handling too.
    const clean = text.normalize('NFKC').replace(/\s+/gu, ' ').trim();
    return SPIECE_UNDERLINE + clean.replace(/ /g, SPIECE_UNDERLINE);
  }

  initialSymbols(normalized) {
    const symbols = [];
    let i = 0;

    while (i < normalized.length) {
      let matched = null;
      for (const token of this.userDefined) {
        if (normalized.startsWith(token, i)) {
          matched = token;
          break;
        }
      }

      if (matched) {
        const id = this.pieceToId.get(matched);
        symbols.push({ text: matched, id, locked: true });
        i += matched.length;
        continue;
      }

      const cp = normalized.codePointAt(i);
      const ch = String.fromCodePoint(cp);
      const id = this.pieceToId.has(ch) ? this.pieceToId.get(ch) : UNK;
      symbols.push({ text: ch, id, locked: false });
      i += ch.length;
    }

    return symbols;
  }

  encodeIds(text) {
    const normalized = this.normalize(text);
    const symbols = this.initialSymbols(normalized);

    // SentencePiece BPE repeatedly applies the highest-scoring available merge.
    while (symbols.length > 1) {
      let bestIndex = -1;
      let bestScore = -Infinity;
      let bestId = -1;

      for (let i = 0; i < symbols.length - 1; i++) {
        if (symbols[i].locked || symbols[i + 1].locked) continue;
        const merged = symbols[i].text + symbols[i + 1].text;
        const id = this.pieceToId.get(merged);
        if (id === undefined) continue;

        const p = this.pieces[id];
        // NORMAL and USER_DEFINED are usable pieces; user-defined pieces are
        // already atomized above, so only NORMAL should normally reach here.
        if (p.type !== 1 && p.type !== 4) continue;
        if (p.score > bestScore) {
          bestScore = p.score;
          bestIndex = i;
          bestId = id;
        }
      }

      if (bestIndex < 0) break;
      const mergedText = symbols[bestIndex].text + symbols[bestIndex + 1].text;
      symbols.splice(bestIndex, 2, { text: mergedText, id: bestId, locked: false });
    }

    return symbols.map((s) => s.id ?? UNK);
  }

  decodeIds(ids) {
    let text = '';
    for (const rawId of ids) {
      const id = Number(rawId);
      const p = this.pieces[id];
      if (!p) continue;
      // CONTROL pieces (BOS/EOS/PAD) are not emitted as text.
      if (p.type === 3) continue;
      if (p.type === 2) {
        text += '⁇';
      } else {
        text += p.piece;
      }
    }
    return text.replaceAll(SPIECE_UNDERLINE, ' ').replace(/^ /, '');
  }
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
    setStatus('起動しました。モデルを取得します…');

    ort.env.wasm.numThreads = globalThis.crossOriginIsolated
      ? Math.min(navigator.hardwareConcurrency || 1, 4)
      : 1;

    const modelBytes = await downloadWithProgress(MODEL_URL, 'モデル', 0, 90);
    setStatus('モデルを初期化中… 90%');
    session = await ort.InferenceSession.create(modelBytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });

    const tokenizerBytes = await downloadWithProgress(TOKENIZER_URL, 'Tokenizer', 90, 10);
    setStatus('Tokenizerを解析中… 100%');
    tokenizer = new BrowserSentencePieceBPE(tokenizerBytes);

    if (tokenizer.pieces.length !== 8000) {
      throw new Error(`Tokenizer vocab mismatch: expected 8000, got ${tokenizer.pieces.length}`);
    }

    setStatus('準備完了。ダウンロード 100%。推論はこの端末内だけで実行されます。');
    convertBtn.textContent = 'Convert';
    convertBtn.disabled = false;
  } catch (err) {
    console.error(err);
    setStatus(`読み込み失敗: ${err?.message || err}`);
    convertBtn.textContent = 'Model unavailable';
    convertBtn.disabled = true;
  }
}

async function generate(text) {
  const encoded = tokenizer.encodeIds(STYLE_PREFIX + text);
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

  return tokenizer.decodeIds(generated.slice(1));
}

convertBtn.addEventListener('click', async () => {
  const text = inputEl.value.trim();
  if (!text || !session || !tokenizer) return;

  convertBtn.disabled = true;
  convertBtn.textContent = 'Converting…';
  setStatus('端末内で推論中…');
  outputEl.value = '';

  try {
    outputEl.value = await generate(text);
    setStatus('完了。入力文はサーバーへ送信されていません。');
  } catch (err) {
    console.error(err);
    outputEl.value = '';
    setStatus(`推論失敗: ${err?.message || err}`);
  } finally {
    convertBtn.disabled = false;
    convertBtn.textContent = 'Convert';
  }
});

init();
