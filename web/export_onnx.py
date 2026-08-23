from pathlib import Path
import argparse
import math
import torch
from torch import nn
from huggingface_hub import hf_hub_download

parser = argparse.ArgumentParser(description='Download a KuchoLM .pt checkpoint from Hugging Face and export it to ONNX.')
parser.add_argument('--repo', default='h6e/KuchoLM-NIDA-10M', help='Hugging Face model repo ID')
parser.add_argument('--pt', default='KuchoLM-NIDA.pt', help='Checkpoint filename inside the Hugging Face repo')
parser.add_argument('--out', default='/content/model.onnx', help='Output ONNX path')
args = parser.parse_args()

print('repo:', args.repo)
print('checkpoint:', args.pt)

checkpoint_path = Path(hf_hub_download(repo_id=args.repo, filename=args.pt, repo_type='model'))
out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)

print('downloaded:', checkpoint_path)

ckpt = torch.load(checkpoint_path, map_location='cpu')
cfg = ckpt.get('config', {}) if isinstance(ckpt, dict) else {}

VOCAB = int(cfg.get('vocab', 8000))
D_MODEL = int(cfg.get('d_model', 256))
NHEAD = int(cfg.get('nhead', 8))
ENC_LAYERS = int(cfg.get('enc_layers', 4))
DEC_LAYERS = int(cfg.get('dec_layers', 4))
FF = int(cfg.get('ff', 1024))
MAX_LEN = int(cfg.get('max_len', 128))
PAD = 0

print(f'config: vocab={VOCAB} d_model={D_MODEL} heads={NHEAD} enc={ENC_LAYERS} dec={DEC_LAYERS} ff={FF} max_len={MAX_LEN}')

class KuchoTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL, padding_idx=PAD)
        self.pos = nn.Embedding(MAX_LEN, D_MODEL)
        self.tf = nn.Transformer(
            d_model=D_MODEL,
            nhead=NHEAD,
            num_encoder_layers=ENC_LAYERS,
            num_decoder_layers=DEC_LAYERS,
            dim_feedforward=FF,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.lm_head = nn.Linear(D_MODEL, VOCAB, bias=False)
        self.lm_head.weight = self.embed.weight

    def add_pos(self, x):
        p = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.embed(x) * math.sqrt(D_MODEL) + self.pos(p)

    def forward(self, src, tgt_in):
        src_pad = src.eq(PAD)
        tgt_pad = tgt_in.eq(PAD)
        mask = nn.Transformer.generate_square_subsequent_mask(tgt_in.size(1), device=tgt_in.device)
        h = self.tf(
            self.add_pos(src),
            self.add_pos(tgt_in),
            tgt_mask=mask,
            src_key_padding_mask=src_pad,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=src_pad,
        )
        return self.lm_head(h)

model = KuchoTransformer()
state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
model.load_state_dict(state)
model.eval()

params = sum(p.numel() for p in model.parameters())
print(f'parameters: {params / 1e6:.2f}M')

src = torch.tensor([[2, 10, 11, 3]], dtype=torch.long)
tgt = torch.tensor([[2, 10]], dtype=torch.long)

with torch.no_grad():
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
    )

print('saved:', out_path)
print('source:', f'https://huggingface.co/{args.repo}/blob/main/{args.pt}')
