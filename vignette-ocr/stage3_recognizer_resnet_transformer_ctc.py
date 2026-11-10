"""
TAHAP 3 - Pengenal teks (recognizer): ResNet18 -> collapse tinggi -> Transformer Encoder -> CTC
Mengikuti diagram workflow yang kamu bagikan.

Bentuk tensor (input 64 x 256, stride lebar total 4):
  gambar        (B, 3, 64, 256)
  CNN           (B, 512, 2, 64)    tinggi 64 -> 2, lebar 256 -> 64
  collapse H    (B, 512, 64)       rata-rata sepanjang tinggi
  urutan        (B, 64, 512)       64 "langkah waktu" (T) dari kiri ke kanan
  proyeksi      (B, 64, 256)       d_model
  Transformer   (B, 64, 256)
  logits        (B, 64, C)         C = jumlah karakter + 1 (blank)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Vocab:
    """Indeks 0 dicadangkan untuk token 'blank' milik CTC."""

    def __init__(self, texts):
        self.chars = sorted(set("".join(texts)))
        self.c2i = {c: i + 1 for i, c in enumerate(self.chars)}
        self.size = len(self.chars) + 1

    def encode(self, s):
        return [self.c2i[c] for c in s]

    def greedy_decode(self, logits):
        """logits: (B, T, C) -> list[str]. Ambil argmax, gabungkan ulangan, buang blank."""
        out = []
        for seq in logits.argmax(-1).tolist():
            prev, chars = 0, []
            for k in seq:
                if k != prev and k != 0:
                    chars.append(self.chars[k - 1])
                prev = k
            out.append("".join(chars))
        return out


class PositionalEncoding(nn.Module):
    """Sinusoidal: memberi tahu Transformer posisi tiap kolom (Transformer tidak peka urutan)."""

    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):                       # x: (B, T, d_model)
        return x + self.pe[:, : x.size(1)]


class Recognizer(nn.Module):
    def __init__(self, num_classes, d_model=256, nhead=8, num_layers=3, pretrained=True):
        super().__init__()
        r = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        # "Modified strides (2,1)": layer2-4 hanya mengecilkan TINGGI, lebar dipertahankan
        # agar karakter yang rapat tetap punya langkah waktu sendiri-sendiri.
        for layer in (r.layer2, r.layer3, r.layer4):
            layer[0].conv1.stride = (2, 1)
            layer[0].downsample[0].stride = (2, 1)
        self.cnn = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool,
                                 r.layer1, r.layer2, r.layer3, r.layer4)
        self.proj = nn.Linear(512, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, 4 * d_model, 0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):                       # x: (B, 3, H, W)
        f = self.cnn(x)                         # (B, 512, H', W')
        f = f.mean(dim=2).permute(0, 2, 1)      # collapse H -> (B, W', 512)
        f = self.encoder(self.pos(self.proj(f)))
        return self.head(f)                     # logits (B, T, C)


def ctc_step(model, images, targets, vocab, criterion):
    """Satu langkah loss CTC. targets: list[str]."""
    logits = model(images)                                   # (B, T, C)
    log_probs = F.log_softmax(logits, -1).permute(1, 0, 2)   # CTC minta (T, B, C)
    enc = [torch.tensor(vocab.encode(t)) for t in targets]
    tgt = torch.cat(enc)                                     # semua label disambung jadi 1 vektor
    in_len = torch.full((len(targets),), log_probs.size(0), dtype=torch.long)
    tgt_len = torch.tensor([len(e) for e in enc], dtype=torch.long)
    return criterion(log_probs, tgt, in_len, tgt_len), logits


if __name__ == "__main__":
    # Smoke test: pastikan bentuk benar dan model bisa menghafal 4 gambar teks sintetis.
    from PIL import Image, ImageDraw

    texts = ["EXP : 06 / 2022", "FAB : 07 / 2019", "N°LOT : 9275", "EXP : 11 / 2024"]
    vocab = Vocab(texts)

    def render(t):
        im = Image.new("RGB", (256, 64), "white")
        ImageDraw.Draw(im).text((8, 24), t, fill="black")
        a = torch.from_numpy(__import__("numpy").array(im)).permute(2, 0, 1).float() / 255
        return (a - 0.5) / 0.5

    x = torch.stack([render(t) for t in texts])
    model = Recognizer(vocab.size, pretrained=False)
    print("logits:", tuple(model(x).shape), "(B, T, C)")

    crit, opt = nn.CTCLoss(blank=0, zero_infinity=True), None
    opt = torch.optim.AdamW(model.parameters(), 1e-3)
    for step in range(301):
        loss, logits = ctc_step(model, x, texts, vocab, crit)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            print(f"step {step:3d} loss {loss.item():.3f}")
    model.eval()
    with torch.no_grad():
        print(vocab.greedy_decode(model(x)))
