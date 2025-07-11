import argparse
import json
import pathlib
from typing import List, Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

import os, json, torch
from datetime import datetime

def log_token_entropy(
    tokenizer,
    responses: torch.Tensor,      # (bs, resp_len)  int64
    entropys: torch.Tensor,       # (bs, resp_len)  float32
    masks: torch.Tensor,          # (bs, resp_len)  bool / 0-1
    global_step: int,
    save_dir: str = "entropy_logs",
    print_first_n: int = 1,
):
    """
    把 <token, entropy> 对应关系保存成 JSONL，格式：
    {"step": 3, "sample_idx": 0, "tokens": ["▁Hello", "," ...], "entropy": [2.33, 0.18, ...]}
    """
    os.makedirs(save_dir, exist_ok=True)
    bs, resp_len = responses.size()
    records = []

    for i in range(bs):
        valid_len    = masks[i].sum().item()
        token_ids    = responses[i, :valid_len].tolist()
        token_texts  = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=False)
        entropy_vals = entropys[i, :valid_len].tolist()

        records.append(
            {
                "step": global_step,
                "sample_idx": i,
                "tokens": token_texts,
                "entropy": entropy_vals,
            }
        )

        # 只打印前 print_first_n 个样本，方便在 terminal 快速查看
        if i < print_first_n:
            pretty = " | ".join(
                f"{t}:{e:.2f}" for t, e in zip(token_texts, entropy_vals)
            )
            print(f"[entropy] step={global_step} sample={i}: {pretty}")

    # 写 JSON Lines（每行一个样本，便于追加）
    fname = f"{save_dir}/entropy_step_{global_step:07d}.jsonl"
    with open(fname, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def load_jsonl(path: str | pathlib.Path) -> List[Dict]:
    """Load a .jsonl file written by ``log_token_entropy``.

    Each line is expected to be a JSON object with at least the keys::

        {
            "step": 42,
            "sample_idx": 0,
            "tokens": ["▁Hello", ",", "world"],
            "entropy": [2.1, 0.3, 4.7]
        }

    Returns
    -------
    records : list[dict]
        A list where each item is the parsed JSON for a single sample.
    """
    path = pathlib.Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    records: List[Dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def pick_sample(records: Sequence[Dict], sample_idx: int = 0) -> Dict:
    """Select the first record whose ``sample_idx`` matches ``sample_idx``."""
    for rec in records:
        if rec.get("sample_idx") == sample_idx:
            return rec
    raise ValueError(f"sample_idx={sample_idx} not found in file – available: {[r['sample_idx'] for r in records]}")


###############################################
# Plot helpers
###############################################


def _plot_heatmap(tokens: List[str], entropy: List[float], cmap: str = "coolwarm", vmin: Optional[float] = None, vmax: Optional[float] = None, annotate: bool = False) -> plt.Figure:
    """Draw a horizontal token–entropy heatmap.

    Parameters
    ----------
    tokens : list[str]
        Token strings (already BPE-split, e.g. "▁Hello").
    entropy : list[float]
        Entropy value per token, same length as ``tokens``.
    cmap : str, default "coolwarm"
        Matplotlib colormap name.
    vmin, vmax : float | None
        Color scale limits. If ``None`` they are inferred from ``entropy``.
    annotate : bool, default False
        If True, text annotations (entropy value) will be rendered on the heatmap.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure instance for further manipulation / saving.
    """
    entropy_arr = np.array(entropy)[None, :]  # shape (1, T)

    fig, ax = plt.subplots(figsize=(max(4, len(tokens) * 0.35), 2))
    im = ax.imshow(entropy_arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    # Token labels
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=7)
    ax.set_yticks([])

    # Color bar
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.01)
    cbar.set_label("Token Entropy")

    # Optional text annotations
    if annotate:
        for j, e in enumerate(entropy):
            ax.text(j, 0, f"{e:.2f}", ha="center", va="center", fontsize=6, color="black")

    fig.tight_layout()
    return fig


def plot_from_jsonl(jsonl_path: str | pathlib.Path, sample_idx: int = 0, *, cmap: str = "coolwarm", save_path: str | None = None, show: bool = True, annotate: bool = False, vmin: Optional[float] = None, vmax: Optional[float] = None) -> plt.Figure:
    """High‑level wrapper: load JSONL, pick sample, draw heatmap.

    All kwargs are forwarded to :func:`_plot_heatmap`.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    records = load_jsonl(jsonl_path)
    rec = pick_sample(records, sample_idx)

    tokens = rec["tokens"]
    entropy = rec["entropy"]

    fig = _plot_heatmap(tokens, entropy, cmap=cmap, vmin=vmin, vmax=vmax, annotate=annotate)

    if save_path is not None:
        save_path = pathlib.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        print(f"[plot_entropy] Saved figure to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


###############################################
# Command‑line interface
###############################################


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Visualize token‑level entropy heatmaps from JSONL logs.")
    p.add_argument("jsonl", help="Path to entropy_step_xxx.jsonl file")
    p.add_argument("--sample-idx", type=int, default=0, help="Which sample_idx inside the JSONL to plot (default: 0)")
    p.add_argument("--cmap", default="coolwarm", help="Matplotlib colormap (default: coolwarm)")
    p.add_argument("--save", metavar="PATH", help="If set, save the figure to this path instead of / in addition to showing it")
    p.add_argument("--no-show", action="store_true", help="Do not call plt.show(); useful in headless environments")
    p.add_argument("--annotate", action="store_true", help="Draw entropy value text on each token cell")
    p.add_argument("--vmin", type=float, default=None, help="Colorbar minimum (default: min(entropy))")
    p.add_argument("--vmax", type=float, default=None, help="Colorbar maximum (default: max(entropy))")
    return p


def cli(argv: Optional[Sequence[str]] = None):
    args = _build_arg_parser().parse_args(argv)

    plot_from_jsonl(
        jsonl_path=args.jsonl,
        sample_idx=args.sample_idx,
        cmap=args.cmap,
        save_path=args.save,
        show=not args.no_show,
        annotate=args.annotate,
        vmin=args.vmin,
        vmax=args.vmax,
    )


if __name__ == "__main__":
    cli()
