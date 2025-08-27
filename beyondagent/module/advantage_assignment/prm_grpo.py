# -*- coding: utf-8 -*-
# PRM step → (optional) group-level standardization on steps → per-trajectory projection to target sum → suffix-sum on steps → broadcast to tokens
from __future__ import annotations
from typing import List, Dict, Optional
from dataclasses import dataclass
import torch
import math

# =========================
# Hyper & small utilities
# =========================

@dataclass
class PRMHyper:
    # 权重：一致性步的权重大，不一致性步的权重小（用于 allocation / allocation_c）
    consistent_scale: float = 1.0
    pos_unconsistent_scale: float = 0.2   # 成功轨迹里的 BAD 步权重
    neg_unconsistent_scale: float = 0.2   # 失败轨迹里的 GOOD 步权重
    eps: float = 1e-8
    do_batch_zscore: bool = True          # 是否做组内 z-score（按 step 级，allocation_c/decouple 会用到）
    traj_equal_zscore: bool = True        # True=每条轨迹等权；False=把所有 step 拉平成一个大样本
    fix_base: float = 0.2                 # fix 方案的基础幅度（good=+base, bad=-base）

def _ensure_tensor(x, device, dtype=None):
    if torch.is_tensor(x):
        t = x.to(device=device)
        if dtype is not None:
            t = t.to(dtype)
        return t
    return torch.as_tensor(x, device=device, dtype=dtype)

def _num_steps_from_step_ids(step_ids_row: torch.Tensor) -> int:
    """step_ids: shape (L,) with -1 for non-response tokens; contiguous step ids starting at 0."""
    if step_ids_row.numel() == 0:
        return 0
    m = torch.amax(step_ids_row)
    return int(m.item() + 1) if m.item() >= 0 else 0

def _align_flags(flags: List[bool], K: int, is_success: bool) -> List[bool]:
    if len(flags) == K:
        return list(flags)
    default_flag = True if is_success else False
    if len(flags) < K:
        return list(flags) + [default_flag] * (K - len(flags))
    else:
        return list(flags[:K])

# =========================
# Z-score helpers (group-wise, step-level)
# =========================

def _group_zscore_on_steps(
    step_rewards_raw: List[List[float]],
    group_ids: torch.Tensor,
    hyper: PRMHyper,
) -> List[List[float]]:
    """对 step 奖励做“组内”减均值/除方差标准化。
    - traj_equal_zscore=True: 每条轨迹等权；组均值 = 轨迹均值的均值；
      组方差 = 轨迹内相对组均值的均方差的均值（second-moment around group mean）
    - traj_equal_zscore=False: 拉平本组所有 step 一起算
    """
    if not hyper.do_batch_zscore:
        return [list(r) for r in step_rewards_raw]

    B = len(step_rewards_raw)
    gids = group_ids.view(-1).tolist()
    g2idx: Dict[int, List[int]] = {}
    for i, g in enumerate(gids):
        g2idx.setdefault(int(g), []).append(i)

    step_rewards_std: List[List[float]] = [[] for _ in range(B)]
    for _, idxs in g2idx.items():
        if hyper.traj_equal_zscore:
            # 1) 组均值：轨迹均值的等权平均
            traj_means = []
            for i in idxs:
                ri = step_rewards_raw[i]
                if ri:
                    traj_means.append(sum(ri) / len(ri))
            if len(traj_means) == 0:
                mu_g, sd_g = 0.0, 1.0
            else:
                mu_g = float(sum(traj_means) / len(traj_means))
                # 2) 组方差：先对每条轨迹围绕 mu_g 求均方差，再对轨迹等权平均
                second_moments = []
                for i in idxs:
                    ri = step_rewards_raw[i]
                    if not ri:
                        continue
                    second_moments.append(sum((x - mu_g) * (x - mu_g) for x in ri) / len(ri))
                var_g = float(sum(second_moments) / len(second_moments)) if second_moments else 0.0
                sd_g = float(math.sqrt(var_g + hyper.eps))
        else:
            # 拉平：把本组所有 step 拼在一起
            flat = []
            for i in idxs:
                flat.extend(step_rewards_raw[i])
            if len(flat) == 0:
                mu_g, sd_g = 0.0, 1.0
            else:
                t = torch.tensor(flat, dtype=torch.float32)
                mu_g = float(t.mean().item())
                sd_g = float(max(t.std(unbiased=False).item(), hyper.eps))

        inv = 1.0 / (sd_g + 1e-12)
        for i in idxs:
            ri = step_rewards_raw[i]
            if not ri:
                step_rewards_std[i] = []
            else:
                step_rewards_std[i] = [float((x - mu_g) * inv) for x in ri]
    return step_rewards_std

def _per_traj_scale_to_target_sum(
    r_std: List[float],
    target_sum: float,
    eps: float,
) -> List[float]:
    """把一条轨迹的 step 列表按比例缩放，使其总和=target_sum。退化时均分。"""
    if len(r_std) == 0:
        return []
    cur = sum(r_std)
    if abs(cur) <= eps:
        return [target_sum / len(r_std) for _ in r_std]
    scale = target_sum / cur
    return [float(x * scale) for x in r_std]

# =========================
# Builders for 4 schemes
# =========================

def _build_fix(
    orms_sign: torch.Tensor,
    step_flags: List[List[bool]],
    step_ids: torch.Tensor,
    hyper: PRMHyper,
) -> List[List[float]]:
    """方案1：fix —— 固定基数（±base），最后一步吃掉剩余量以满足 ∑=±1。"""
    B = step_ids.size(0)
    out: List[List[float]] = []
    base = float(hyper.fix_base)
    for i in range(B):
        K = _num_steps_from_step_ids(step_ids[i])
        if K == 0:
            out.append([]); continue
        is_success = bool(orms_sign[i].item() > 0)
        flags = _align_flags(step_flags[i] if i < len(step_flags) else [], K, is_success)
        r = [(+base if f else -base) for f in flags]
        # 让总和等于 ±1（按 orms_sign）
        need = float(orms_sign[i].item()) - sum(r)
        r[-1] += need
        out.append([float(x) for x in r])
    return out

def _build_allocation(
    orms_sign: torch.Tensor,
    step_flags: List[List[bool]],
    step_ids: torch.Tensor,
    hyper: PRMHyper,
) -> List[List[float]]:
    """方案2：allocation —— 一致性瓜分（同号），不做标准化；逐轨迹 ∑=±1。"""
    B = step_ids.size(0)
    out: List[List[float]] = []
    for i in range(B):
        K = _num_steps_from_step_ids(step_ids[i])
        if K == 0:
            out.append([]); continue
        is_success = bool(orms_sign[i].item() > 0)
        flags = _align_flags(step_flags[i] if i < len(step_flags) else [], K, is_success)
        n_g = sum(1 for f in flags if f); n_b = K - n_g
        if is_success:
            w_g, w_b = hyper.consistent_scale, hyper.pos_unconsistent_scale
            sgn = +1.0
        else:
            w_g, w_b = hyper.neg_unconsistent_scale, hyper.consistent_scale
            sgn = -1.0
        total_w = n_g * w_g + n_b * w_b
        unit = 0.0 if total_w <= hyper.eps else (1.0 / total_w)
        # 同号瓜分：good 和 bad 都随 orms_sign 同号，仅幅度不同；确保 sum = ±1
        r = [sgn * (w_g * unit) if f else sgn * (w_b * unit) for f in flags]
        out.append([float(x) for x in r])
    return out

def _build_allocation_c(
    orms_sign: torch.Tensor,
    step_flags: List[List[bool]],
    step_ids: torch.Tensor,
    group_ids: torch.Tensor,
    hyper: PRMHyper,
) -> List[List[float]]:
    """方案3：allocation_c —— 一致性瓜分（同号） → 组内 z-score → 按比例缩放投影（∑=±1）。"""
    B = step_ids.size(0)
    # 1) raw（逐轨迹 ∑=±1）
    step_rewards_raw: List[List[float]] = []
    for i in range(B):
        K = _num_steps_from_step_ids(step_ids[i])
        if K == 0:
            step_rewards_raw.append([]); continue
        is_success = bool(orms_sign[i].item() > 0)
        flags = _align_flags(step_flags[i] if i < len(step_flags) else [], K, is_success)
        n_g = sum(1 for f in flags if f); n_b = K - n_g
        if is_success:
            w_g, w_b = hyper.consistent_scale, hyper.pos_unconsistent_scale
            sgn = +1.0
        else:
            w_g, w_b = hyper.neg_unconsistent_scale, hyper.consistent_scale
            sgn = -1.0
        total_w = n_g * w_g + n_b * w_b
        unit = 0.0 if total_w <= hyper.eps else (1.0 / total_w)
        r_raw = [sgn * (w_g * unit) if f else sgn * (w_b * unit) for f in flags]
        step_rewards_raw.append([float(x) for x in r_raw])
    # 2) group z-score
    r_std = _group_zscore_on_steps(step_rewards_raw, group_ids, hyper)
    # 3) 按比例缩放投影（逐轨迹 ∑=±1）
    out: List[List[float]] = []
    for i in range(B):
        out.append(_per_traj_scale_to_target_sum(r_std[i], float(orms_sign[i].item()), eps=hyper.eps))
    return out

def _build_decouple(
    orms_sign: torch.Tensor,
    step_flags: List[List[bool]],
    step_ids: torch.Tensor,
    group_ids: torch.Tensor,
    hyper: PRMHyper,
) -> List[List[float]]:
    """方案4：decouple —— PRM 和 ORM 解耦。
    - PRM：只用 flags 造一个“形状”向量（good=+1，bad=-1），与 ORM 无关；
    - 标准化：对 PRM 形状在组内做 z-score；
    - 投影：按比例缩放到目标总和（±1），用 ORM_sign 仅作为“总量”来源。
    """
    B = step_ids.size(0)
    # 1) 形状（与 ORM 无关）
    shape_raw: List[List[float]] = []
    for i in range(B):
        K = _num_steps_from_step_ids(step_ids[i])
        if K == 0:
            shape_raw.append([]); continue
        # 对齐后：good=+1, bad=-1 （最终符号由 orms_sign 控制总量）
        flags = _align_flags(step_flags[i] if i < len(step_flags) else [], K, is_success=True)
        shape_raw.append([1.0 if f else -1.0 for f in flags])
    # 2) group z-score（仅对“形状”做）
    shape_std = _group_zscore_on_steps(shape_raw, group_ids, hyper)
    # 3) 按比例缩放到 ±1（ORM_sign）
    out: List[List[float]] = []
    for i in range(B):
        out.append(_per_traj_scale_to_target_sum(shape_std[i], float(orms_sign[i].item()), eps=hyper.eps))
    return out

# =========================
# Step → Token broadcast + suffix-sum
# =========================

def suffix_sum_on_steps(step_rewards: List[List[float]]) -> List[List[float]]:
    """对每个样本的 step 回报做后缀和，输出同形状的 step-adv。"""
    adv: List[List[float]] = []
    for r in step_rewards:
        if not r:
            adv.append([]); continue
        t = torch.tensor(r, dtype=torch.float32)
        s = torch.flip(torch.cumsum(torch.flip(t, dims=[0]), dim=0), dims=[0])
        adv.append([float(x) for x in s])
    return adv

def broadcast_step_adv_to_tokens(
    step_adv: List[List[float]],
    step_ids: torch.Tensor,
) -> torch.Tensor:
    """把 step-adv 按 step_ids 广播到 token 上。step_ids 为 -1 的位置填 0。"""
    device = step_ids.device
    B, L = step_ids.shape
    out = torch.zeros((B, L), device=device, dtype=torch.float32)
    for i in range(B):
        if not step_adv[i]:
            continue
        adv_i = torch.tensor(step_adv[i], device=device, dtype=torch.float32)
        sid_row = step_ids[i]
        valid = sid_row >= 0
        if torch.any(valid):
            sids = sid_row[valid]
            out[i, valid] = adv_i[sids]
    return out

# =========================
# Entry
# =========================

def compute_prm_grpo_advantages(
    batch,                          # DataProto 或兼容结构：batch.batch[...] 可索引
    step_flags: List[List[bool]],   # 每条轨迹的 GOOD/BAD 标志
    hyper: Optional[PRMHyper] = None,
    scheme: str = "allocation_c",   # "fix" | "allocation" | "allocation_c" | "decouple"
) -> dict:
    """
    统一入口：
      - 先把 ORM 压成 ±1：orms_sign = sign(sum(token_level_rewards)) （== +1 if sum>0 else -1）
      - 根据 scheme 构造 step-level 奖励（见各 builder），得到 step_rewards
      - step 后缀和 → step_adv
      - 广播到 token → advantages (B, L)
    返回：
      - advantages: (B, L) token-level advantages
      - orm_scalar: (B,) 逐条轨迹的 ±1
    """
    if hyper is None:
        hyper = PRMHyper()

    # ---- 取必要字段 ----
    responses = batch.batch["responses"]
    device = responses.device if torch.is_tensor(responses) else torch.as_tensor(responses).device

    step_ids = _ensure_tensor(batch.batch["step_ids"], device=device, dtype=torch.long)      # (B, L_resp) with -1 for non-response
    group_ids = _ensure_tensor(batch.batch["group_ids"], device=device, dtype=torch.long).view(-1)

    # 取 token-level reward（可能字段名不同，做兜底）
    token_keys_try = ["token_level_rewards", "response_token_level_rewards", "token_rewards"]
    token_level_rewards = None
    for k in token_keys_try:
        if k in batch.batch:
            token_level_rewards = _ensure_tensor(batch.batch[k], device=device, dtype=torch.float32)
            break
    if token_level_rewards is None:
        raise KeyError("token-level rewards not found in batch (tried keys: token_level_rewards / response_token_level_rewards / token_rewards)")

    # ---- ORM_sign = ±1（你要求保持 sum>0 → +1；sum<=0 → -1）----
    orm_sum = token_level_rewards.sum(dim=1)   # (B,)
    orms_sign = torch.where(orm_sum > 0, torch.ones_like(orm_sum), -torch.ones_like(orm_sum)).to(dtype=torch.float32)

    # ---- Build step rewards by scheme ----
    scheme = scheme.lower()
    if scheme == "fix":
        step_rewards = _build_fix(orms_sign, step_flags, step_ids, hyper)
    elif scheme == "allocation":
        step_rewards = _build_allocation(orms_sign, step_flags, step_ids, hyper)
    elif scheme == "allocation_c":
        step_rewards = _build_allocation_c(orms_sign, step_flags, step_ids, group_ids, hyper)
    elif scheme == "decouple":
        step_rewards = _build_decouple(orms_sign, step_flags, step_ids, group_ids, hyper)
    else:
        raise ValueError(f"Unknown PRM scheme: {scheme} (expected one of: fix | allocation | allocation_c | decouple)")

    # ---- Step → token advantages ----
    step_adv = suffix_sum_on_steps(step_rewards)
    advantages = broadcast_step_adv_to_tokens(step_adv, step_ids)

    return {
        "advantages": advantages,        # (B, L_resp)
        "orm_scalar": orms_sign,         # (B,)
    }
