from pathlib import Path
import math
import torch
from torch import nn

CHECKPOINT = Path('/content/kucholm_work/KuchoLM-NIDA-10M.pt')
OUT = Path('/content/model.onnx')

ckpt = torch.load(CHECKPOINT, map_location='cpu')
cfg = ckpt.get('config', {})
VOCAB = int(cfg.get('vocab', 8000))
D_MODEL = int(cfg.get('d_model', 256))
NHEAD = int(cfg.get('nhead', 8))
ENC_LAYERS = int(cfg.get('enc_layers', 4))
DEC_LAYERS = int(cfg.get('dec_layers', 4))
FF = int(cfg.get('ff', 1024))
MAX_LEN = int(cfg.get('max_len', 128))
PAD = 0

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
model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
model.eval()

src = torch.tensor([[2, 10, 11, 3]], dtype=torch.long)
tgt = torch.tensor([[2, 10]], dtype=torch.long)

with torch.no_grad():
    torch.onnx.export(
        model,
        (src, tgt),
        OUT,
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

print('saved:', OUT)
print('Upload model.onnx to: https://huggingface.co/h6e/KuchoLM-NIDA-10M')
