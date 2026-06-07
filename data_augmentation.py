#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据增强模块 —— 针对手部关键点序列 (30 frames × 126 features) 的增强管道

126 维特征结构: [左手63维 | 右手63维]
每只手63维 = 21个关键点 × (x, y, z)
关键点已在 [-1, 1] 归一化, 以手腕为原点

增强策略:
  1. 空间噪声 (Spatial Noise)        —— 模拟关键点检测抖动
  2. 时间扭曲 (Time Warping)          —— 模拟手势速度快慢变化
  3. 缩放     (Scaling)              —— 模拟手距镜头远近
  4. 旋转     (Rotation)             —— 模拟摄像头角度变化
  5. 平移抖动 (Translation Jitter)    —— 模拟手在画面中的位置偏移
  6. 水平镜像 (Horizontal Mirror)     —— 模拟左右手互换
  7. 帧丢弃   (Frame Dropout)        —— 模拟丢帧
"""

import numpy as np


# ── 工具函数 ───────────────────────────────────────────

def _split_hands(sequence):
    """将 (T, 126) 拆分为 (T, 21, 3) × 2 便于逐手操作"""
    half = 63
    left = sequence[..., :half].reshape(-1, 21, 3)
    right = sequence[..., half:].reshape(-1, 21, 3)
    return left, right


def _merge_hands(left, right):
    """将两只手合并回 (T, 126)"""
    T = left.shape[0]
    return np.concatenate([left.reshape(T, -1), right.reshape(T, -1)], axis=-1)


def _is_zero_hand(hand):
    """判断一只手是否全为零（未检测到）"""
    return np.allclose(hand, 0.0)


def _augment_hand_2d(hand, scale=1.0, angle_deg=0.0, dx=0.0, dy=0.0):
    """
    对一只手的关键点做 2D 仿射变换 (缩放 + 旋转 + 平移)
    hand: (T, 21, 3) 或 (21, 3)
    """
    theta_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(theta_rad), np.sin(theta_rad)

    result = hand.copy()
    x, y = result[..., 0], result[..., 1]

    # 缩放 + 旋转
    new_x = scale * (cos_a * x - sin_a * y)
    new_y = scale * (sin_a * x + cos_a * y)

    result[..., 0] = new_x + dx
    result[..., 1] = new_y + dy
    # z 只做缩放
    result[..., 2] *= scale

    return result


# ── 增强函数 (每个返回 shape 与原输入一致) ─────────────

def spatial_noise(sequence, std=0.015):
    """添加高斯噪声, 模拟 MediaPipe 检测抖动"""
    noise = np.random.normal(0, std, sequence.shape).astype(np.float32)
    return sequence + noise


def time_warp(sequence, sigma=0.2):
    """
    时间扭曲: 用平滑随机曲线对帧索引做非线性重采样
    模拟手势速度快慢的自然变化
    """
    T = sequence.shape[0]
    # 随机控制点
    src = np.linspace(0, T - 1, num=np.random.randint(3, 5))
    dst = src + np.random.normal(0, sigma * T / 4, size=len(src))
    dst = np.clip(dst, 0, T - 1)

    # 生成平滑 warp 曲线
    old_indices = np.arange(T)
    new_indices = np.interp(old_indices, src, dst)
    new_indices = np.sort(new_indices)  # 保持时间单调

    # 对每个特征维插值
    warped = np.zeros_like(sequence)
    for d in range(sequence.shape[1]):
        warped[:, d] = np.interp(old_indices, new_indices, sequence[:, d])
    return warped.astype(np.float32)


def random_scale(sequence, scale_range=(0.85, 1.15)):
    """随机缩放 x,y,z 坐标, 模拟手距镜头远近"""
    scale = np.random.uniform(*scale_range)
    left, right = _split_hands(sequence)

    if not _is_zero_hand(left):
        left = _augment_hand_2d(left, scale=scale)
    if not _is_zero_hand(right):
        right = _augment_hand_2d(right, scale=scale)

    return _merge_hands(left, right)


def random_rotation(sequence, max_angle=12.0):
    """随机旋转 x,y 平面, 模拟摄像头角度微变"""
    angle = np.random.uniform(-max_angle, max_angle)
    left, right = _split_hands(sequence)

    if not _is_zero_hand(left):
        left = _augment_hand_2d(left, angle_deg=angle)
    if not _is_zero_hand(right):
        right = _augment_hand_2d(right, angle_deg=angle)

    return _merge_hands(left, right)


def translation_jitter(sequence, max_dx=0.03, max_dy=0.03):
    """轻微平移, 模拟手在画面中位置的微小偏移"""
    dx = np.random.uniform(-max_dx, max_dx)
    dy = np.random.uniform(-max_dy, max_dy)
    left, right = _split_hands(sequence)

    if not _is_zero_hand(left):
        left = _augment_hand_2d(left, dx=dx, dy=dy)
    if not _is_zero_hand(right):
        right = _augment_hand_2d(right, dx=dx, dy=dy)

    return _merge_hands(left, right)


def horizontal_mirror(sequence):
    """水平翻转 (x 取反) + 交换左右手, 模拟另一只手做手势"""
    left, right = _split_hands(sequence)
    mirrored_left = left.copy()
    mirrored_right = right.copy()

    if not _is_zero_hand(left):
        mirrored_left[..., 0] *= -1.0
    if not _is_zero_hand(right):
        mirrored_right[..., 0] *= -1.0

    # 左右手互换
    return _merge_hands(mirrored_right, mirrored_left)


def frame_dropout(sequence, drop_prob=0.1):
    """随机丢弃帧, 用相邻帧插值填充, 模拟少量丢帧"""
    T = sequence.shape[0]
    mask = np.random.random(T) > drop_prob
    # 确保首尾帧不丢
    mask[0] = mask[-1] = True

    result = sequence.copy()
    for t in range(1, T - 1):
        if not mask[t]:
            # 线性插值
            result[t] = (sequence[t - 1] + sequence[t + 1]) / 2.0
    return result.astype(np.float32)


# ── 增强管道 ───────────────────────────────────────────

AUGMENTATIONS = [
    ('spatial_noise', spatial_noise),
    ('time_warp', time_warp),
    ('random_scale', random_scale),
    ('random_rotation', random_rotation),
    ('translation_jitter', translation_jitter),
    ('horizontal_mirror', horizontal_mirror),
    ('frame_dropout', frame_dropout),
]


def apply_augmentation(sequence, aug_name=None):
    """对单个样本随机选择并应用一种增强"""
    if aug_name is None:
        aug_name = np.random.choice([name for name, _ in AUGMENTATIONS])

    for name, func in AUGMENTATIONS:
        if name == aug_name:
            return func(sequence), name

    return sequence, 'none'


def augment_sequence(sequence, num_augmentations=2):
    """对单个样本叠加多种增强"""
    result = sequence.copy()
    names = np.random.choice(
        [name for name, _ in AUGMENTATIONS],
        size=min(num_augmentations, len(AUGMENTATIONS)),
        replace=False,
    )
    for name in names:
        for aug_name, func in AUGMENTATIONS:
            if aug_name == name:
                result = func(result)
                break
    return result, '+'.join(names)


def generate_augmented_dataset(x, y, multiplier=5, seed=42):
    """
    从原始数据集生成增强后的数据集

    Args:
        x: (N, T, D) 原始特征
        y: (N,) 原始标签
        multiplier: 每个原始样本生成的增强样本数
        seed: 随机种子

    Returns:
        x_aug: (N * (multiplier+1), T, D) 增强后的特征 (含原始样本)
        y_aug: (N * (multiplier+1),) 对应的标签
    """
    rng = np.random.default_rng(seed)
    x_list = [x]
    y_list = [y]

    for i in range(multiplier):
        aug_batch = []
        for sample in x:
            # 随机选择 1~3 种增强叠加
            num = rng.integers(1, 4)
            aug_sample, _ = augment_sequence(sample, num_augmentations=num)
            aug_batch.append(aug_sample)
        x_list.append(np.array(aug_batch, dtype=np.float32))
        y_list.append(y.copy())
        print(f'  增强轮次 {i + 1}/{multiplier} 完成', flush=True)

    x_aug = np.concatenate(x_list, axis=0)
    y_aug = np.concatenate(y_list, axis=0)

    # 打乱
    indices = rng.permutation(len(x_aug))
    return x_aug[indices], y_aug[indices]


# ── 测试 ────────────────────────────────────────────────

if __name__ == '__main__':
    # 快速测试
    dummy = np.random.randn(30, 126).astype(np.float32) * 0.3
    print(f'原始 shape: {dummy.shape}')

    for name, func in AUGMENTATIONS:
        result = func(dummy.copy())
        diff = np.mean(np.abs(result - dummy))
        print(f'  {name:25s}  mean_abs_diff={diff:.6f}  shape={result.shape}')

    print('\n✓ 所有增强函数测试通过')
