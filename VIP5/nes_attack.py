#!/usr/bin/env python3
"""
VIP5 黑盒对抗攻击 - NES Attack (Natural Evolution Strategy)
============================================================
自然进化策略攻击 - 完全零阶优化方法

原理：
    NES是一种完全不需要梯度的黑盒优化方法。
    通过在当前解周围随机采样，根据采样点的得分估计梯度方向，
    然后沿着估计的梯度方向更新扰动。

    这是一种**真正的黑盒攻击**：
    - 不需要访问任何模型的内部结构
    - 不需要计算任何梯度
    - 只需要查询模型获得"得分"（这里用CLIP特征相似度作为代理）

核心算法：
    1. 在当前扰动周围随机采样多个噪声
    2. 对每个噪声方向，计算正向和负向扰动的得分
    3. 使用得分差异加权噪声，估计梯度
    4. 沿估计梯度方向更新扰动

使用方法：
    python nes_attack.py --split toys --num_images 100
    python nes_attack.py --split toys --num_images 100 --epsilon 0.08 --max_iter 30

作者: 对抗攻击研究
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json
import random
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

import torch
import torchvision.transforms as transforms

# 导入CLIP
import clip

SCRIPT_DIR = Path(__file__).resolve().parent


class NESAttacker:
    """
    自然进化策略 (Natural Evolution Strategy) 攻击器

    这是一种完全黑盒的攻击方法，不需要任何梯度信息。

    关键改进：
    - 在CLIP预处理后的tensor空间中计算得分（避免重复预处理损失）
    - 保存为PNG无损格式（避免JPEG压缩丢失扰动）
    - 直接保存224x224（避免resize来回插值损失）
    - 使用动量稳定梯度估计
    - 使用batch方式加速噪声采样的得分计算
    """

    def __init__(self, device='cuda', epsilon=0.3, max_iter=200,
                 sigma=0.05, n_samples=100, step_size=0.05, momentum=0.9):
        """
        初始化NES攻击器

        Args:
            device: 计算设备 ('cuda' 或 'cpu')
            epsilon: 最大扰动幅度 (0-1范围, 建议0.2-0.5)
            max_iter: 迭代次数
            sigma: 采样标准差
            n_samples: 每次迭代的采样数量
            step_size: 更新步长
            momentum: 动量系数
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.sigma = sigma
        self.n_samples = n_samples
        self.step_size = step_size
        self.momentum = momentum
        self.target_features = None

        # 加载CLIP模型（仅用于提取特征，不使用梯度）
        print(f"Loading CLIP model on {self.device}...")
        self.clip_model, self.clip_preprocess = clip.load('ViT-B/32', device=self.device)
        self.clip_model.eval()

        # 获取CLIP的normalize参数，用于直接在tensor空间操作
        self.clip_normalize = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
        self.clip_resize = transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC)
        self.clip_center_crop = transforms.CenterCrop(224)

        print("CLIP model loaded!")

    def extract_feature(self, image: Image.Image) -> np.ndarray:
        """提取CLIP特征（无梯度）"""
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.clip_model.encode_image(image_input)
        return features.cpu().numpy().squeeze()

    def extract_feature_from_tensor(self, img_tensor: torch.Tensor) -> np.ndarray:
        """从已预处理的tensor提取CLIP特征（无梯度）

        Args:
            img_tensor: [0,1]范围的tensor (3, 224, 224)
        """
        # 应用CLIP的normalize
        normalized = self.clip_normalize(img_tensor).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.clip_model.encode_image(normalized)
        return features.cpu().numpy().squeeze()

    def compute_score_tensor(self, img_tensor: torch.Tensor) -> float:
        """从tensor计算得分"""
        feat = self.extract_feature_from_tensor(img_tensor)
        if self.target_features is not None:
            score = np.dot(feat, self.target_features) / (
                np.linalg.norm(feat) * np.linalg.norm(self.target_features) + 1e-8
            )
        else:
            score = np.linalg.norm(feat)
        return float(score)

    def set_target_features(self, features: np.ndarray):
        """设置目标特征（热门商品的平均特征）"""
        self.target_features = features
        print(f"Target features set, shape: {features.shape}")

    def compute_score(self, image: Image.Image) -> float:
        """
        计算图像的"得分"

        使用与目标特征的余弦相似度作为得分。
        """
        feat = self.extract_feature(image)

        if self.target_features is not None:
            score = np.dot(feat, self.target_features) / (
                np.linalg.norm(feat) * np.linalg.norm(self.target_features) + 1e-8
            )
        else:
            score = np.linalg.norm(feat)

        return float(score)

    def attack_single(self, image: Image.Image) -> Image.Image:
        """
        对单张图像执行NES攻击

        关键改进：
        - 在[0,1] tensor空间中操作，避免uint8量化损失
        - 使用动量稳定梯度方向
        - 自适应步长
        """
        # 转换为224x224的tensor (直接在CLIP输入尺寸上操作)
        img_resized = image.resize((224, 224), Image.BICUBIC)
        img_array = np.array(img_resized).astype(np.float32) / 255.0
        # (H, W, C) -> (C, H, W) tensor
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()

        # 初始化扰动
        delta = torch.zeros_like(img_tensor)

        # 动量缓冲
        velocity = torch.zeros_like(img_tensor)

        # 记录最佳结果
        best_delta = delta.clone()
        best_score = self.compute_score_tensor(img_tensor)
        initial_score = best_score

        # 自适应步长
        step_size = self.step_size
        no_improve_count = 0

        for iteration in range(self.max_iter):
            grad_estimate = torch.zeros_like(img_tensor)

            # antithetic sampling
            for _ in range(self.n_samples // 2):
                noise = torch.randn_like(img_tensor) * self.sigma

                # 正向扰动
                delta_pos = torch.clamp(delta + noise, -self.epsilon, self.epsilon)
                perturbed_pos = torch.clamp(img_tensor + delta_pos, 0, 1)
                score_pos = self.compute_score_tensor(perturbed_pos)

                # 负向扰动
                delta_neg = torch.clamp(delta - noise, -self.epsilon, self.epsilon)
                perturbed_neg = torch.clamp(img_tensor + delta_neg, 0, 1)
                score_neg = self.compute_score_tensor(perturbed_neg)

                # 梯度估计
                grad_estimate += (score_pos - score_neg) * noise

                # 更新最佳结果
                if score_pos > best_score:
                    best_score = score_pos
                    best_delta = delta_pos.clone()
                    no_improve_count = 0
                if score_neg > best_score:
                    best_score = score_neg
                    best_delta = delta_neg.clone()
                    no_improve_count = 0

            # 归一化梯度
            grad_estimate /= (self.n_samples * self.sigma)

            # 动量更新
            velocity = self.momentum * velocity + (1 - self.momentum) * grad_estimate

            # 沿梯度方向更新（梯度上升）
            delta = delta + step_size * torch.sign(velocity)

            # 投影到epsilon球内
            delta = torch.clamp(delta, -self.epsilon, self.epsilon)

            # 自适应步长
            no_improve_count += 1
            if no_improve_count > 15:
                step_size = min(step_size * 1.3, self.epsilon * 0.5)
                no_improve_count = 0

        # 使用最佳扰动
        attacked_tensor = torch.clamp(img_tensor + best_delta, 0, 1)

        # 转换回PIL图像 (保持224x224，不resize回去，避免插值损失)
        attacked_array = (attacked_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        attacked_img = Image.fromarray(attacked_array)

        return attacked_img, float(best_score - initial_score)

    def attack_batch(self, image_dir: Path, output_dir: Path,
                     num_images: Optional[int] = None) -> Dict:
        """批量攻击图像"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 获取所有图像文件
        image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))

        if num_images and num_images < len(image_files):
            random.seed(42)
            image_files = random.sample(image_files, num_images)

        print(f"\n{'='*60}")
        print(f"开始NES攻击 {len(image_files)} 张图像")
        print(f"{'='*60}")
        print(f"Epsilon: {self.epsilon}")
        print(f"迭代次数: {self.max_iter}")
        print(f"采样数: {self.n_samples}")
        print(f"Sigma: {self.sigma}")
        print(f"步长: {self.step_size}")
        print(f"动量: {self.momentum}")
        print(f"保存格式: PNG (无损)")
        print(f"{'='*60}\n")

        results = {
            'success': 0,
            'failed': 0,
            'details': []
        }

        for img_path in tqdm(image_files, desc="NES Attacking"):
            try:
                original_img = Image.open(img_path).convert('RGB')
                original_feat = self.extract_feature(original_img)
                original_score = self.compute_score(original_img)

                attacked_img, score_improvement = self.attack_single(original_img)
                attacked_feat = self.extract_feature(attacked_img)
                attacked_score = self.compute_score(attacked_img)

                # 保存为PNG无损格式（关键改进：避免JPEG压缩丢失扰动）
                # 保持原始文件名的stem，但改为.png
                output_stem = img_path.stem
                output_path = output_dir / f"{output_stem}.png"
                attacked_img.save(output_path)

                # 计算特征变化
                feat_diff = np.linalg.norm(attacked_feat - original_feat)
                cosine_sim = np.dot(original_feat, attacked_feat) / (
                    np.linalg.norm(original_feat) * np.linalg.norm(attacked_feat) + 1e-8
                )

                results['success'] += 1
                results['details'].append({
                    'image': img_path.name,
                    'feature_diff': float(feat_diff),
                    'cosine_sim': float(cosine_sim),
                    'original_score': float(original_score),
                    'attacked_score': float(attacked_score),
                    'score_improvement': float(attacked_score - original_score)
                })

            except Exception as e:
                print(f"\nError processing {img_path.name}: {e}")
                results['failed'] += 1

        return results


def get_popular_items(split: str, top_k: int = 50) -> List[str]:
    """从sequential_data.txt获取热门商品列表"""
    data_dir = SCRIPT_DIR / 'data' / split
    sequential_path = data_dir / 'sequential_data.txt'

    if not sequential_path.exists():
        print(f"Warning: {sequential_path} not found")
        return []

    from collections import Counter
    item_counts = Counter()

    with open(sequential_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) > 1:
                items = parts[1:]
                for item in items:
                    item_counts[item] += 1

    popular = [item for item, _ in item_counts.most_common(top_k)]
    print(f"Found {len(popular)} popular items")
    return popular


def load_target_features(feature_dir: Path, popular_items: List[str]) -> Optional[np.ndarray]:
    """加载热门商品的平均特征作为攻击目标"""
    features = []

    for item_id in popular_items:
        feat_path = feature_dir / f"{item_id}.npy"
        if feat_path.exists():
            features.append(np.load(feat_path))

    if len(features) == 0:
        print("Warning: No target features found")
        return None

    target_feature = np.mean(features, axis=0)
    print(f"Loaded target feature from {len(features)} popular items")
    return target_feature


def main():
    parser = argparse.ArgumentParser(
        description='VIP5 黑盒对抗攻击 - NES Attack',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本用法 (推荐参数)
  python nes_attack.py --split toys --num_images 100

  # 更强攻击
  python nes_attack.py --split toys --num_images 100 --epsilon 0.4 --max_iter 300

参数建议:
  - epsilon: 0.2-0.5 (CLIP对像素扰动鲁棒,需要较大扰动)
  - max_iter: 100-300 (越多效果越好)
  - n_samples: 50-200 (越多梯度估计越准)
  - sigma: 0.03-0.1 (探索范围)
  - target_topk: 1-5 (越小目标越集中)
        """
    )

    parser.add_argument('--split', type=str, default='toys',
                        help='数据集名称 (默认: toys)')
    parser.add_argument('--num_images', type=int, default=None,
                        help='攻击图像数量 (默认: 全部)')
    parser.add_argument('--epsilon', type=float, default=0.3,
                        help='最大扰动幅度 (默认: 0.3)')
    parser.add_argument('--max_iter', type=int, default=200,
                        help='迭代次数 (默认: 200)')
    parser.add_argument('--sigma', type=float, default=0.05,
                        help='采样标准差 (默认: 0.05)')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='每次迭代采样数 (默认: 100)')
    parser.add_argument('--step_size', type=float, default=0.05,
                        help='更新步长 (默认: 0.05)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='动量系数 (默认: 0.9)')
    parser.add_argument('--target_topk', type=int, default=3,
                        help='使用top-k热门商品的特征作为目标 (默认: 3)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备 (默认: cuda)')

    args = parser.parse_args()

    # 设置路径
    image_dir = SCRIPT_DIR / args.split
    output_dir = SCRIPT_DIR / f'{args.split}2'
    feature_dir = SCRIPT_DIR / 'features' / 'vitb32_features' / f'{args.split}_original'

    # 检查输入目录
    if not image_dir.exists():
        print(f"Error: Image directory not found: {image_dir}")
        sys.exit(1)

    print("\n" + "="*60)
    print("VIP5 黑盒对抗攻击 - NES Attack (改进版)")
    print("="*60)
    print(f"输入目录: {image_dir}")
    print(f"输出目录: {output_dir}")

    # 创建攻击器
    attacker = NESAttacker(
        device=args.device,
        epsilon=args.epsilon,
        max_iter=args.max_iter,
        sigma=args.sigma,
        n_samples=args.n_samples,
        step_size=args.step_size,
        momentum=args.momentum
    )

    # 加载目标特征 - 使用少量热门商品获得更集中的目标
    if feature_dir.exists():
        popular_items = get_popular_items(args.split)
        if popular_items:
            target_feat = load_target_features(feature_dir, popular_items[:args.target_topk])
            if target_feat is not None:
                attacker.set_target_features(target_feat)
    else:
        print(f"Warning: Feature directory not found: {feature_dir}")
        print("Running without target features (using feature norm as score)")

    # 执行攻击
    results = attacker.attack_batch(image_dir, output_dir, args.num_images)

    # 打印结果
    print("\n" + "="*60)
    print("攻击完成!")
    print("="*60)
    print(f"成功: {results['success']}")
    print(f"失败: {results['failed']}")

    if results['details']:
        feat_diffs = [d['feature_diff'] for d in results['details']]
        cosine_sims = [d['cosine_sim'] for d in results['details']]
        score_improvements = [d['score_improvement'] for d in results['details']]

        print(f"\n特征变化统计:")
        print(f"  平均L2距离: {np.mean(feat_diffs):.4f}")
        print(f"  平均余弦相似度(原始vs攻击): {np.mean(cosine_sims):.4f}")

        print(f"\n得分变化统计:")
        print(f"  平均得分提升: {np.mean(score_improvements):+.4f}")
        print(f"  最大得分提升: {np.max(score_improvements):+.4f}")
        print(f"  得分提升比例: {sum(1 for s in score_improvements if s > 0) / len(score_improvements) * 100:.1f}%")

    # 保存结果
    result_path = output_dir / 'attack_results.json'
    with open(result_path, 'w') as f:
        json.dump({
            'method': 'nes',
            'epsilon': args.epsilon,
            'max_iter': args.max_iter,
            'sigma': args.sigma,
            'n_samples': args.n_samples,
            'step_size': args.step_size,
            'momentum': args.momentum,
            'target_topk': args.target_topk,
            'results': results
        }, f, indent=2)

    print(f"\n结果已保存到: {result_path}")

    # 打印下一步指令
    print("\n" + "="*60)
    print("下一步操作:")
    print("="*60)
    print(f"1. 提取攻击后图片的CLIP特征:")
    print(f"   python evaluate_attack.py --mode extract --split {args.split}")
    print(f"\n2. 在VIP5模型上评估攻击效果:")
    print(f"   python evaluate_attack_vip5.py --split {args.split} --num_samples 500")


if __name__ == '__main__':
    main()
