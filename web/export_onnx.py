from pathlib import Path
import argparse
import math
import torch
from torch import nn
from huggingface_hub import hf_hub_download

parser = argparse.ArgumentParser(description='Download a KuchoLM .pt checkpoint from Hugging Face and export a browser-friendly ONNX model.')
parser.add_argument('--repo', default='h6e/KuchoLM-NIDA-10M', help='Hugging Face model repo ID')
parser.add_argument('--pt', default='KuchoLM-NIDA-10M.pt', help='Checkpoint filename inside the Hugging Face repo')
parser.add_argument('--out', default='/content/model.onnx', help='Output ONNX path')
args = parser.parse_args()

print('repo:', args.repo)
print('checkpoint:', args.pt)
checkpoint_path = Path(hf_hub_download(repo_id=args.repo, filename=args.pt, repo_type='model'))
out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
print('downloaded:', checkpoint_path)

ckpt = torch.load(checkpoint_path, map_location='cpu')
state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
cfg = ckpt.get('config', {}) if isinstance(ckpt, dict) else {}

if 'embed.weight' in state:
    FORMAT = 'current'
    EMB_KEY = 'embed.weight'
    TF_PREFIX = 'tf'
elif 'emb.weight' in state:
    FORMAT = 'legacy'
    EMB_KEY = 'emb.weight'
    TF_PREFIX = 'tr'
else:
    raise RuntimeError('Unknown KuchoLM checkpoint format: embedding weight not found')

VOCAB = int(cfg.get('vocab', state[EMB_KEY].shape[0]))
D_MODEL = int(cfg.get('d_model', state[EMB_KEY].shape[1]))
NHEAD = int(cfg.get('nhead', 8))
ENC_LAYERS = int(cfg.get('enc_layers', len({k.split('.')[3] for k in state if k.startswith(f'{TF_PREFIX}.encoder.layers.')})))
DEC_LAYERS = int(cfg.get('dec_layers', len({k.split('.')[3] for k in state if k.startswith(f'{TF_PREFIX}.decoder.layers.')})))
FF_KEY = f'{TF_PREFIX}.encoder.layers.0.linear1.weight'
FF = int(cfg.get('ff', state[FF_KEY].shape[0] if FF_KEY in state else 1024))
MAX_LEN = int(cfg.get('max_len', state['pos.weight'].shape[0] if 'pos.weight' in state else 128))
PAD = 0

print('format:', FORMAT)
print(f'config: vocab={VOCAB} d_model={D_MODEL} heads={NHEAD} enc={ENC_LAYERS} dec={DEC_LAYERS} ff={FF} max_len={MAX_LEN}')

# Fixed causal mask. Keeping it as a buffer avoids dynamic shape/arange logic in
# the ONNX graph, which has been unreliable in ONNX Runtime Web/WASM on iOS.
CAUSAL_MASK = torch.triu(torch.ones(MAX_LEN, MAX_LEN, dtype=torch.bool), diagonal=1)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :MAX_LEN]


class LegacyKuchoTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D_MODEL, padding_idx=PAD)
        self.pos = SinusoidalPositionalEncoding(D_MODEL, MAX_LEN)
        self.tr = nn.Transformer(
            d_model=D_MODEL, nhead=NHEAD,
            num_encoder_layers=ENC_LAYERS, num_decoder_layers=DEC_LAYERS,
            dim_feedforward=FF, dropout=0.0,
            batch_first=True, norm_first=True,
        )
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)
        self.head.weight = self.emb.weight
        self.register_buffer('causal_mask', CAUSAL_MASK)

    def forward(self, src, tgt_in):
        src_pad = src.eq(PAD)
        tgt_pad = tgt_in.eq(PAD)
        src_h = self.pos(self.emb(src) * math.sqrt(D_MODEL))
        tgt_h = self.pos(self.emb(tgt_in) * math.sqrt(D_MODEL))
        h = self.tr(
            src_h, tgt_h,
            tgt_mask=self.causal_mask,
            src_key_padding_mask=src_pad,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=src_pad,
        )
        return self.head(h)


class CurrentKuchoTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL, padding_idx=PAD)
        self.pos = nn.Embedding(MAX_LEN, D_MODEL)
        self.tf = nn.Transformer(
            d_model=D_MODEL, nhead=NHEAD,
            num_encoder_layers=ENC_LAYERS, num_decoder_layers=DEC_LAYERS,
            dim_feedforward=FF, dropout=0.0,
            batch_first=True, norm_first=True,
        )
        self.lm_head = nn.Linear(D_MODEL, VOCAB, bias=False)
        self.lm_head.weight = self.embed.weight
        self.register_buffer('positions', torch.arange(MAX_LEN, dtype=torch.long).unsqueeze(0))
        self.register_buffer('causal_mask', CAUSAL_MASK)

    def add_pos(self, x):
        return self.embed(x) * math.sqrt(D_MODEL) + self.pos(self.positions)

    def forward(self, src, tgt_in):
        src_pad = src.eq(PAD)
        tgt_pad = tgt_in.eq(PAD)
        h = self.tf(
            self.add_pos(src), self.add_pos(tgt_in),
            tgt_mask=self.causal_mask,
            src_key_padding_mask=src_pad,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=src_pad,
        )
        return self.lm_head(h)


model = CurrentKuchoTransformer() if FORMAT == 'current' else LegacyKuchoTransformer()
state_to_load = dict(state)

if FORMAT == 'legacy':
    checkpoint_pe = state_to_load.pop('pos.pe', None)
    if checkpoint_pe is not None and hasattr(model.pos, 'pe'):
        print('checkpoint pos.pe:', tuple(checkpoint_pe.shape), '-> regenerated:', tuple(model.pos.pe.shape))

# Browser-only constant buffers are generated by the exporter and do not exist
# in training checkpoints.
missing, unexpected = model.load_state_dict(state_to_load, strict=False)
allowed_missing = {'positions', 'causal_mask'}
if FORMAT == 'legacy':
    allowed_missing.add('pos.pe')
real_missing = [k for k in missing if k not in allowed_missing]
if real_missing or unexpected:
    raise RuntimeError(f'state_dict mismatch: missing={real_missing}, unexpected={unexpected}')

model.eval()
params = sum(p.numel() for p in model.parameters())
print(f'parameters: {params / 1e6:.2f}M')

fastpath_was_enabled = torch.backends.mha.get_fastpath_enabled()
torch.backends.mha.set_fastpath_enabled(False)
print('mha fastpath: disabled for ONNX export')
print(f'browser export: FIXED batch=1 src={MAX_LEN} tgt={MAX_LEN}, explicit PAD masks')

src = torch.full((1, MAX_LEN), PAD, dtype=torch.long)
src[0, :4] = torch.tensor([2, 10, 11, 3])
tgt = torch.full((1, MAX_LEN), PAD, dtype=torch.long)
tgt[0, :2] = torch.tensor([2, 10])

try:
    with torch.no_grad():
        smoke = model(src, tgt)
        print('pytorch smoke:', tuple(smoke.shape), 'finite=', bool(torch.isfinite(smoke).all()))

        torch.onnx.export(
            model,
            (src, tgt),
            out_path,
            input_names=['src', 'tgt_in'],
            output_names=['logits'],
            # No dynamic_axes: fixed shapes are intentional for Web/WASM.
            opset_version=18,
            do_constant_folding=True,
            dynamo=False,
        )
finally:
    torch.backends.mha.set_fastpath_enabled(fastpath_was_enabled)

print('saved:', out_path)
print('expected inputs: src=[1,128], tgt_in=[1,128]')
print('source:', f'https://huggingface.co/{args.repo}/blob/main/{args.pt}')
