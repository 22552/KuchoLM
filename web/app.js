import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.min.mjs';

const MODEL_URL = 'https://huggingface.co/h6e/KuchoLM-NIDA-10M/resolve/main/model.onnx';
const TOKENIZER_URL = 'https://huggingface.co/h6e/KuchoLM-NIDA-10M/resolve/main/kucholm_spm.model';

const PAD = 0;
const UNK = 1;
const BOS = 2;
const EOS = 3;
const VOCAB_SIZE = 8000;
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

function errorText(err) {
  if (err instanceof Error && err.message) return err.message;
  try { return String(err); } catch { return 'unknown error'; }
}

function validateIds(ids, label) {
  if (!Array.isArray(ids) || ids.length === 0) throw new Error(`${label}: empty token list`);
  for (let i = 0; i < ids.length; i++) {
    const id = Number(ids[i]);
    if (!Number.isInteger(id) || id < 0 || id >= VOCAB_SIZE) {
      throw new Error(`${label}: token id out of range at ${i}: ${id}`);
    }
  }
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

// Only the tiny tokenizer is downloaded manually so we can show progress.
// The ONNX model is passed to ORT by URL directly to avoid keeping another
// full model copy in JS memory before ORT copies/loads it into WASM memory.
async function downloadTokenizerWithProgress(url) {
  setStatus('Tokenizerへ接続中…');
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) throw new Error(`Tokenizer download failed (${response.status})`);

  if (!response.body) {
    setStatus('Tokenizerをダウンロード中…');
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
      const percent = Math.min(100, Math.round((received / total) * 100));
      setStatus(`Tokenizerをダウンロード中… ${percent}% (${formatBytes(received)} / ${formatBytes(total)})`);
    } else {
      setStatus(`Tokenizerをダウンロード中… ${formatBytes(received)}`);
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

function readVarint(bytes, state, end = bytes.length) {
  let value = 0;
  let shift = 0;
  while (state.i < end) {
    const b = bytes[state.i++];
    value += (b & 0x7f) * 2 ** shift;
    if ((b & 0x80) === 0) return value;
    shift += 7;
    if (shift > 49) throw new Error('Tokenizer protobuf varint is too large');
  }
  throw new Error('Unexpected end of tokenizer protobuf');
}

function skipField(bytes, state, wire, end = bytes.length) {
  if (wire === 0) {
    readVarint(bytes, state, end);
  } else if (wire === 1) {
    state.i += 8;
  } else if (wire === 2) {
    state.i += readVarint(bytes, state, end);
  } else if (wire === 3) {
    while (state.i < end) {
      const key = readVarint(bytes, state, end);
      const nestedWire = key & 7;
      if (nestedWire === 4) return;
      skipField(bytes, state, nestedWire, end);
    }
  } else if (wire === 4) {
    return;
  } else if (wire === 5) {
    state.i += 4;
  } else {
    throw new Error(`Unsupported protobuf wire type ${wire}`);
  }
  if (state.i > end) throw new Error('Tokenizer protobuf is truncated');
}

function parseSentencePieceMessage(bytes, start, end) {
  const state = { i: start };
  let piece = '';
  let score = 0;
  let type = 1;
  const decoder = new TextDecoder('utf-8');

  while (state.i < end) {
    const key = readVarint(bytes, state, end);
    const field = Math.floor(key / 8);
    const wire = key & 7;

    if (field === 1 && wire === 2) {
      const len = readVarint(bytes, state, end);
      const fieldEnd = state.i + len;
      if (fieldEnd > end) throw new Error('Tokenizer piece string is truncated');
      piece = decoder.decode(bytes.subarray(state.i, fieldEnd));
      state.i = fieldEnd;
    } else if (field === 2 && wire === 5) {
      if (state.i + 4 > end) throw new Error('Tokenizer piece score is truncated');
      score = new DataView(bytes.buffer, bytes.byteOffset + state.i, 4).getFloat32(0, true);
      state.i += 4;
    } else if (field === 3 && wire === 0) {
      type = readVarint(bytes, state, end);
    } else {
      skipField(bytes, state, wire, end);
    }
  }

  return { piece, score, type };
}

function parseSentencePieceModel(bytes) {
  const state = { i: 0 };
  const pieces = [];

  while (state.i < bytes.length) {
    const keyStart = state.i;
    const key = readVarint(bytes, state);
    const field = Math.floor(key / 8);
    const wire = key & 7;

    if (field === 1 && wire === 2) {
      const len = readVarint(bytes, state);
      const start = state.i;
      const end = start + len;
      if (end > bytes.length) throw new Error('Tokenizer piece is truncated');
      pieces.push(parseSentencePieceMessage(bytes, start, end));
      state.i = end;
      continue;
    }

    if (pieces.length > 0) {
      state.i = keyStart;
      break;
    }

    skipField(bytes, state, wire);
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
    this.userDefined.sort((a, b) => b.length - a.length);
  }

  normalize(text) {
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
        symbols.push({ text: matched, id: this.pieceToId.get(matched), locked: true });
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
    const symbols = this.initialSymbols(this.normalize(text));

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
      if (p.type === 3) continue;
      if (p.type === 2) text += '⁇';
      else text += p.piece;
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

async function smokeTestOnnx() {
  setStatus('ONNXをテスト中…');
  try {
    const result = await session.run({
      src: toInt64Tensor([BOS, EOS]),
      tgt_in: toInt64Tensor([BOS]),
    });
    const logits = result.logits ?? result[session.outputNames[0]];
    if (!logits || logits.dims.at(-1) !== VOCAB_SIZE) {
      throw new Error(`unexpected logits shape: ${logits ? logits.dims.join('x') : 'none'}`);
    }
  } catch (err) {
    throw new Error(`ONNXスモークテスト失敗: ${errorText(err)} / inputs=${session.inputNames.join(',')} outputs=${session.outputNames.join(',')}`);
  }
}

async function init() {
  try {
    convertBtn.disabled = true;
    convertBtn.textContent = 'Loading…';

    ort.env.wasm.numThreads = 1;

    // Important on iPhone/iOS Safari: let ORT fetch the model directly. Do not
    // materialize the whole ONNX file as a JS Uint8Array first.
    setStatus('モデルを読み込み中…');
    session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'basic',
      enableCpuMemArena: false,
      enableMemPattern: false,
      executionMode: 'sequential',
    });

    await smokeTestOnnx();

    const tokenizerBytes = await downloadTokenizerWithProgress(TOKENIZER_URL);
    setStatus('Tokenizerを解析中…');
    tokenizer = new BrowserSentencePieceBPE(tokenizerBytes);

    if (tokenizer.pieces.length !== VOCAB_SIZE) {
      throw new Error(`Tokenizer vocab mismatch: expected ${VOCAB_SIZE}, got ${tokenizer.pieces.length}`);
    }

    const probe = tokenizer.encodeIds(`${STYLE_PREFIX}今日は学校です。`);
    validateIds(probe, 'Tokenizer self-test');

    setStatus('準備完了。ONNX/TokenizerテストOK。');
    convertBtn.textContent = 'Convert';
    convertBtn.disabled = false;
  } catch (err) {
    console.error(err);
    setStatus(`読み込み失敗: ${errorText(err)}`);
    convertBtn.textContent = 'Model unavailable';
    convertBtn.disabled = true;
  }
}

async function generate(text) {
  const encoded = tokenizer.encodeIds(STYLE_PREFIX + text);
  validateIds(encoded, 'Tokenizer');
  const srcIds = [BOS, ...encoded.slice(0, MAX_LEN - 2), EOS];
  validateIds(srcIds, 'src');
  const generated = [BOS];

  for (let step = 0; step < MAX_LEN - 1; step++) {
    validateIds(generated, 'tgt');
    setStatus(`端末内で推論中… ${step + 1}/${MAX_LEN - 1}`);

    let result;
    try {
      result = await session.run({
        src: toInt64Tensor(srcIds),
        tgt_in: toInt64Tensor(generated),
      });
    } catch (err) {
      const minSrc = Math.min(...srcIds);
      const maxSrc = Math.max(...srcIds);
      const minTgt = Math.min(...generated);
      const maxTgt = Math.max(...generated);
      throw new Error(`ONNX step ${step + 1}: ${errorText(err)} / srcLen=${srcIds.length} srcId=${minSrc}-${maxSrc} tgtLen=${generated.length} tgtId=${minTgt}-${maxTgt}`);
    }

    const logits = result.logits ?? result[session.outputNames[0]];
    const nextId = argmaxLastToken(logits, generated);
    if (!Number.isInteger(nextId) || nextId < 0 || nextId >= VOCAB_SIZE) {
      throw new Error(`invalid output token id: ${nextId}`);
    }

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
  outputEl.value = '';

  try {
    outputEl.value = await generate(text);
    setStatus('完了。入力文はサーバーへ送信されていません。');
  } catch (err) {
    console.error(err);
    outputEl.value = '';
    setStatus(`推論失敗: ${errorText(err)}`);
  } finally {
    convertBtn.disabled = false;
    convertBtn.textContent = 'Convert';
  }
});

init();