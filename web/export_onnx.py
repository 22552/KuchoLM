from pathlib import Path
import argparse
import math
import torch
from torch import nn
from huggingface_hub import hf_hub_download

parser = argparse.ArgumentParser(description='Download a KuchoLM .pt checkpoint from Hugging Face and export it to ONNX.')
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

VOCAB = int(cfg.get('vocab', state['emb.weight'].shape[0] if 'emb.weight' in state else 8000))
D_MODEL = int(cfg.get('d_model', state['emb.weight'].shape[1] if 'emb.weight' in state else 256))
NHEAD = int(cfg.get('nhead', 8))
ENC_LAYERS = int(cfg.get('enc_layers', len({k.split('.')[3] for k in state if k.startswith('tr.encoder.layers.')})))
DEC_LAYERS = int(cfg.get('dec_layers', len({k.split('.')[3] for k in state if k.startswith('tr.decoder.layers.')})))
FF = int(cfg.get('ff', state['tr.encoder.layers.0.linear1.weight'].shape[0] if 'tr.encoder.layers.0.linear1.weight' in state else 1024))
MAX_LEN = int(cfg.get('max_len', 128))
PAD = 0

print(f'config: vocab={VOCAB} d_model={D_MODEL} heads={NHEAD} enc={ENC_LAYERS} dec={DEC_LAYERS} ff={FF} max_len={MAX_LEN}')

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
        return x + self.pe[:, :x.size(1)]

class KuchoTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D_MODEL, padding_idx=PAD)
        self.pos = SinusoidalPositionalEncoding(D_MODEL, MAX_LEN)
        self.tr = nn.Transformer(
            d_model=D_MODEL,
            nhead=NHEAD,
            num_encoder_layers=ENC_LAYERS,
            num_decoder_layers=DEC_LAYERS,
            dim_feedforward=FF,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, src, tgt_in):
        src_pad = src.eq(PAD)
        tgt_pad = tgt_in.eq(PAD)
        mask = nn.Transformer.generate_square_subsequent_mask(tgt_in.size(1), device=tgt_in.device)
        src_h = self.pos(self.emb(src) * math.sqrt(D_MODEL))
        tgt_h = self.pos(self.emb(tgt_in) * math.sqrt(D_MODEL))
        h = self.tr(
            src_h,
            tgt_h,
            tgt_mask=mask,
            src_key_padding_mask=src_pad,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=src_pad,
        )
        return self.head(h)

model = KuchoTransformer()

# pos.pe is deterministic sinusoidal data, so ignore the checkpoint copy if its length differs.
state_to_load = dict(state)
checkpoint_pe = state_to_load.pop('pos.pe', None)
if checkpoint_pe is not None:
    print('checkpoint pos.pe:', tuple(checkpoint_pe.shape), '-> regenerated:', tuple(model.pos.pe.shape))

missing, unexpected = model.load_state_dict(state_to_load, strict=False)
real_missing = [k for k in missing if k != 'pos.pe']
if real_missing or unexpected:
    raise RuntimeError(f'state_dict mismatch: missing={real_missing}, unexpected={unexpected}')

model.eval()
params = sum(p.numel() for p in model.parameters())
print(f'parameters: {params / 1e6:.2f}M')

src = torch.tensor([[2, 10, 11, 3]], dtype=torch.long)
tgt = torch.tensor([[2, 10]], dtype=torch.long)

with torch.no_grad():
    # PyTorch 2.9+ defaults to the Dynamo exporter, which currently fails on
    # nn.Transformer with dynamic sequence lengths. Use the mature TorchScript
    # exporter instead; ONNX Runtime Web supports the resulting graph.
    torch.onnx.export(
        model,
        (src, tgt),
        out_path,
        input_names=['src', 'tgt_in'],
        output_names=['logits'],
        dynamic_axes={
            'src': {0: 'batch', 1: 'src_len'},
            'tgt_in': {0: 'batch', 1: 'tgt_len'},
            'logits': {0: 'batch', 1: 'tgt_len'},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

print('saved:', out_path)
print('source:', f'https://huggingface.co/{args.repo}/blob/main/{args.pt}')
