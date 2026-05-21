"""
字符级 Causal Transformer 语言模型训练脚本，含 PPL 计算与 top-p 采样续写测试。

用法示例:
    # 训练一个 6 层、512 hidden 的小型自回归语言模型，并在训练后续写
    python transformer_language_model.py --corpus "*.txt" --epochs 20 --prompt "从前有一座山"

    # 只加载已训练模型做续写
    python transformer_language_model.py --generate_only --load best_transformer_lm.pt --prompt "人工智能的未来"

说明:
    - 模型为字符级语言模型。
    - Transformer 使用 causal mask；逻辑上允许当前位置看见自己及以前 token，
      即下三角矩阵为可见区域。由于 PyTorch 的 bool mask 中 True 表示“禁止关注”，
      代码中先构造下三角 allowed_mask，再取反传入 Transformer。
"""

import math
import argparse
import glob
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────── 工具 ───────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─────────────────────────── 数据 ───────────────────────────

def load_corpus(pattern="*.txt"):
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text):
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    def __init__(self, text, char2idx, seq_len):
        self.seq_len = seq_len
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


# ─────────────────────────── 模型 ───────────────────────────

class LM(nn.Module):
    """
    字符级 Causal Transformer LM。

    关键配置:
        num_layers = 6
        hidden_dim = 512
        causal mask = 下三角可见，禁止关注未来 token
    """

    def __init__(
        self,
        vocab_size,
        hidden_dim=512,
        num_layers=6,
        nhead=8,
        dropout=0.1,
        max_seq_len=128,
        dim_feedforward=None,
        tie_weights=True,
    ):
        super().__init__()
        if hidden_dim % nhead != 0:
            raise ValueError(f"hidden_dim={hidden_dim} 必须能被 nhead={nhead} 整除。")

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        if dim_feedforward is None:
            dim_feedforward = hidden_dim * 4

        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)
        self.drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.ln_f = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size, bias=False)

        # 常见 LM 技巧：输入 embedding 与输出投影共享权重，节省参数并通常提升效果
        if tie_weights:
            self.fc.weight = self.token_embed.weight

    @staticmethod
    def build_causal_mask(seq_len, device):
        """
        构造自回归 causal mask。

        allowed_mask 为下三角矩阵:
            1 0 0
            1 1 0
            1 1 1

        PyTorch Transformer 的 bool mask 语义是 True = 禁止关注，
        所以返回 ~allowed_mask，即上三角未来位置为 True。
        """
        allowed_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        )
        return ~allowed_mask

    def forward(self, x):
        # x: (B, T)
        batch_size, seq_len = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"输入长度 {seq_len} 超过 max_seq_len={self.max_seq_len}，"
                f"请调大 --seq_len 或截断输入。"
            )

        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, T)
        h = self.token_embed(x) + self.pos_embed(pos)
        h = self.drop(h)

        causal_mask = self.build_causal_mask(seq_len, x.device)
        h = self.transformer(h, mask=causal_mask)

        h = self.ln_f(h)
        logits = self.fc(h)  # (B, T, V)
        return logits


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True, grad_clip=1.0):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    if total_tokens == 0:
        return float("inf"), float("inf")

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return avg_loss, ppl


# ─────────────────────────── top-p 采样 / 续写 ───────────────────────────

def top_p_sample(logits, top_p=0.9, temperature=1.0):
    """
    nucleus sampling / top-p 采样。
    从累计概率不超过 top_p 的最小候选集合中采样。
    """
    if temperature <= 0:
        raise ValueError("temperature 必须大于 0。")
    if not (0 < top_p <= 1):
        raise ValueError("top_p 必须在 (0, 1] 范围内。")

    logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # 删除累计概率超过 top_p 的 token，但保留第一个超过阈值的 token
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
    sorted_indices_to_remove[0] = False

    sorted_probs = sorted_probs.masked_fill(sorted_indices_to_remove, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum()

    sampled_pos = torch.multinomial(sorted_probs, num_samples=1)
    sampled_token = sorted_indices[sampled_pos]
    return sampled_token.item()


@torch.no_grad()
def generate(model, prompt, char2idx, idx2char, device, gen_len=200, top_p=0.9, temperature=0.8):
    model.eval()

    ids = [char2idx[c] for c in prompt if c in char2idx]
    if not ids:
        raise ValueError("prompt 中没有任何字符出现在词表中，请换一个提示词。")

    generated = ids[:]
    for _ in range(gen_len):
        context = generated[-model.max_seq_len:]
        x = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(x)[0, -1]  # 最后一个位置预测下一个字符
        next_id = top_p_sample(logits, top_p=top_p, temperature=temperature)
        generated.append(next_id)

    return "".join(idx2char[i] for i in generated)


def save_checkpoint(path, model, char2idx, idx2char, args):
    torch.save(
        {
            "model_state": model.state_dict(),
            "char2idx": char2idx,
            "idx2char": idx2char,
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(path, device):
    return torch.load(path, map_location=device)


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",       type=int,   default=20)
    parser.add_argument("--seq_len",      type=int,   default=128)
    parser.add_argument("--batch_size",   type=int,   default=64)

    # 神经元/hidden 设置为 512
    parser.add_argument("--hidden_dim",   type=int,   default=512)
    parser.add_argument("--num_layers",   type=int,   default=6)

    parser.add_argument("--nhead",        type=int,   default=8)
    parser.add_argument("--dropout",      type=float, default=0.1)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip",    type=float, default=1.0)
    parser.add_argument("--val_ratio",    type=float, default=0.05)
    parser.add_argument("--corpus",       default="*.txt")
    parser.add_argument("--save",         default="best_transformer_lm.pt")
    parser.add_argument("--load",         default="")
    parser.add_argument("--seed",         type=int, default=42)

    # 续写测试参数
    parser.add_argument("--prompt",       default="股市")
    parser.add_argument("--gen_len",      type=int,   default=200)
    parser.add_argument("--top_p",        type=float, default=0.9)
    parser.add_argument("--temperature",  type=float, default=0.8)
    parser.add_argument("--generate_only", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: CAUSAL TRANSFORMER")
    print(
        f"config: layers={args.num_layers}, hidden_dim={args.hidden_dim}, "
        f"nhead={args.nhead}, seq_len={args.seq_len}"
    )

    if args.generate_only:
        if not args.load:
            raise ValueError("--generate_only 模式必须提供 --load checkpoint 路径。")
        ckpt = load_checkpoint(args.load, device)
        ckpt_args = ckpt["args"]
        char2idx = ckpt["char2idx"]
        idx2char = ckpt["idx2char"]

        model = LM(
            vocab_size=len(char2idx),
            hidden_dim=ckpt_args.get("hidden_dim", 512),
            num_layers=ckpt_args.get("num_layers", 6),
            nhead=ckpt_args.get("nhead", 8),
            dropout=ckpt_args.get("dropout", 0.1),
            max_seq_len=ckpt_args.get("seq_len", 128),
        ).to(device)
        model.load_state_dict(ckpt["model_state"])

        completion = generate(
            model=model,
            prompt=args.prompt,
            char2idx=char2idx,
            idx2char=idx2char,
            device=device,
            gen_len=args.gen_len,
            top_p=args.top_p,
            temperature=args.temperature,
        )
        print("\n续写结果:")
        print(completion)
        return

    # 数据准备
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件，请确认 --corpus 路径正确。")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) >= 2:
        random.shuffle(lines)
        split = max(1, int(len(lines) * (1 - args.val_ratio)))
        train_text = "\n".join(lines[:split])
        val_text = "\n".join(lines[split:])
    else:
        # 极小语料兜底：按字符切分
        split = max(1, int(len(text) * (1 - args.val_ratio)))
        train_text = text[:split]
        val_text = text[split:]

    if len(val_text) <= args.seq_len:
        # 验证集过小时，从训练文本末尾切一小段出来，避免 val_loader 为空
        val_chars = max(args.seq_len + 1, int(len(text) * args.val_ratio))
        val_text = text[-val_chars:]
        train_text = text[:-val_chars] if len(text) > val_chars else text

    train_ds = CharDataset(train_text, char2idx, args.seq_len)
    val_ds = CharDataset(val_text, char2idx, args.seq_len)

    if len(train_ds) == 0:
        raise ValueError("训练语料太短，长度必须大于 --seq_len。")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # 模型
    model = LM(
        vocab_size=vocab_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        nhead=args.nhead,
        dropout=args.dropout,
        max_seq_len=args.seq_len,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(
            model, train_loader, criterion, optimizer, device,
            train=True, grad_clip=args.grad_clip,
        )

        with torch.no_grad():
            va_loss, va_ppl = run_epoch(
                model, val_loader, criterion, optimizer, device,
                train=False, grad_clip=args.grad_clip,
            )

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            save_checkpoint(args.save, model, char2idx, idx2char, args)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")

    # 用最佳模型进行句子续写测试
    if Path(args.save).exists() and args.prompt:
        ckpt = load_checkpoint(args.save, device)
        model.load_state_dict(ckpt["model_state"])

        completion = generate(
            model=model,
            prompt=args.prompt,
            char2idx=char2idx,
            idx2char=idx2char,
            device=device,
            gen_len=args.gen_len,
            top_p=args.top_p,
            temperature=args.temperature,
        )
        print("\n续写测试:")
        print(f"prompt: {args.prompt}")
        print(f"top_p={args.top_p}, temperature={args.temperature}, gen_len={args.gen_len}")
        print("-" * 56)
        print(completion)


if __name__ == "__main__":
    main()
