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

# 导入CLIP
import clip

SCRIPT_DIR = Path(__file__).resolve().parent


class NESAttacker:
    """
    自然进化策略 (Natural Evolution Strategy) 攻击器
    
    这是一种完全黑盒的攻击方法，不需要任何梯度信息。
    
    参数说明：
    - epsilon: 最大扰动幅度，范围[0,1]，建议0.03-0.1
    - max_iter: 迭代次数，建议20-100
    - sigma: 采样标准差，控制探索范围，建议0.01-0.05
    - n_samples: 每次迭代的采样数，建议10-50
    - step_size: 更新步长，建议0.01-0.05
    """
    
    def __init__(self, device='cuda', epsilon=0.05, max_iter=50,
                 sigma=0.01, n_samples=20, step_size=0.01):
        """
        初始化NES攻击器
        
        Args:
            device: 计算设备 ('cuda' 或 'cpu')
            epsilon: 最大扰动幅度 (0-1)
            max_iter: 迭代次数
            sigma: 采样标准差
            n_samples: 每次迭代的采样数量
            step_size: 更新步长
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.sigma = sigma
        self.n_samples = n_samples
        self.step_size = step_size
        self.target_features = None
        
        # 加载CLIP模型（仅用于提取特征，不使用梯度）
        print(f"Loading CLIP model on {self.device}...")
        self.clip_model, self.clip_preprocess = clip.load('ViT-B/32', device=self.device)
        self.clip_model.eval()
        print("CLIP model loaded!")
    
    def extract_feature(self, image: Image.Image) -> np.ndarray:
        """提取CLIP特征（无梯度）"""
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.clip_model.encode_image(image_input)
        return features.cpu().numpy().squeeze()
    
    def set_target_features(self, features: np.ndarray):
        """设置目标特征（热门商品的平均特征）"""
        self.target_features = features
        print(f"Target features set, shape: {features.shape}")
    
    def compute_score(self, image: Image.Image) -> float:
        """
        计算图像的"得分"
        
        这是NES的核心：我们需要一个可以查询的得分函数。
        这里使用与目标特征的余弦相似度作为得分。
        
        在真实黑盒场景中，这个得分可以是：
        - 推荐系统返回的排名
        - 点击率预测值
        - 任何可以查询的指标
        
        Args:
            image: PIL图像
            
        Returns:
            score: 得分值（越高越好）
        """
        feat = self.extract_feature(image)
        
        if self.target_features is not None:
            # 与目标特征的余弦相似度（越高越好）
            score = np.dot(feat, self.target_features) / (
                np.linalg.norm(feat) * np.linalg.norm(self.target_features) + 1e-8
            )
        else:
            # 如果没有目标特征，使用特征L2范数作为代理
            score = np.linalg.norm(feat)
        
        return float(score)
    
    def attack_single(self, image: Image.Image) -> Image.Image:
        """
        对单张图像执行NES攻击
        
        算法流程：
        1. 初始化扰动为零
        2. 循环max_iter次：
           a. 生成n_samples个随机噪声
           b. 对每个噪声，计算正向和负向扰动的得分
           c. 使用得分差异估计梯度
           d. 沿梯度方向更新扰动
           e. 将扰动投影到epsilon球内
        3. 返回扰动后的图像
        
        Args:
            image: 原始PIL图像
            
        Returns:
            attacked_image: 攻击后的PIL图像
        """
        # 转换为numpy数组 (224x224是CLIP的输入大小)
        img_array = np.array(image.resize((224, 224))).astype(np.float32) / 255.0
        
        # 初始化扰动
        delta = np.zeros_like(img_array)
        
        # 记录最佳结果
        best_delta = delta.copy()
        best_score = self.compute_score(image)
        
        for iteration in range(self.max_iter):
            # 估计梯度
            grad_estimate = np.zeros_like(img_array)
            
            # 使用antithetic sampling（对称采样）减少方差
            for _ in range(self.n_samples // 2):
                # 生成随机噪声
                noise = np.random.randn(*img_array.shape) * self.sigma
                
                # 正向扰动
                delta_pos = np.clip(delta + noise, -self.epsilon, self.epsilon)
                perturbed_pos = np.clip(img_array + delta_pos, 0, 1)
                img_pos = Image.fromarray((perturbed_pos * 255).astype(np.uint8))
                score_pos = self.compute_score(img_pos)
                
                # 负向扰动
                delta_neg = np.clip(delta - noise, -self.epsilon, self.epsilon)
                perturbed_neg = np.clip(img_array + delta_neg, 0, 1)
                img_neg = Image.fromarray((perturbed_neg * 255).astype(np.uint8))
                score_neg = self.compute_score(img_neg)
                
                # 梯度估计：使用得分差异加权噪声
                grad_estimate += (score_pos - score_neg) * noise
                
                # 更新最佳结果
                if score_pos > best_score:
                    best_score = score_pos
                    best_delta = delta_pos.copy()
                if score_neg > best_score:
                    best_score = score_neg
                    best_delta = delta_neg.copy()
            
            # 归一化梯度估计
            grad_estimate /= (self.n_samples * self.sigma)
            
            # 沿梯度方向更新扰动（梯度上升以最大化得分）
            delta += self.step_size * np.sign(grad_estimate)
            
            # 投影到epsilon球内
            delta = np.clip(delta, -self.epsilon, self.epsilon)
        
        # 使用最佳扰动生成最终图像
        attacked_array = np.clip(img_array + best_delta, 0, 1)
        attacked_img = Image.fromarray((attacked_array * 255).astype(np.uint8))
        
        # 恢复原始大小
        attacked_img = attacked_img.resize(image.size, Image.LANCZOS)
        
        return attacked_img
    
    def attack_batch(self, image_dir: Path, output_dir: Path,
                     num_images: Optional[int] = None) -> Dict:
        """
        批量攻击图像
        
        Args:
            image_dir: 原始图像目录
            output_dir: 攻击后图像保存目录
            num_images: 攻击图像数量（None表示全部）
            
        Returns:
            results: 包含攻击统计信息的字典
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取所有图像文件
        image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
        
        # 随机采样
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
        print(f"{'='*60}\n")
        
        results = {
            'success': 0,
            'failed': 0,
            'details': []
        }
        
        for img_path in tqdm(image_files, desc="NES Attacking"):
            try:
                # 加载原始图像
                original_img = Image.open(img_path).convert('RGB')
                original_feat = self.extract_feature(original_img)
                original_score = self.compute_score(original_img)
                
                # 执行攻击
                attacked_img = self.attack_single(original_img)
                attacked_feat = self.extract_feature(attacked_img)
                attacked_score = self.compute_score(attacked_img)
                
                # 保存攻击后的图像
                output_path = output_dir / img_path.name
                attacked_img.save(output_path, quality=95)
                
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
  # 基本用法
  python nes_attack.py --split toys --num_images 100
  
  # 调整参数
  python nes_attack.py --split toys --num_images 100 --epsilon 0.08 --max_iter 30
  
  # 更多采样（更准确但更慢）
  python nes_attack.py --split toys --num_images 100 --n_samples 50

参数建议:
  - epsilon: 0.03-0.1 (越大扰动越明显)
  - max_iter: 20-100 (越多效果越好，但越慢)
  - n_samples: 10-50 (越多梯度估计越准确)
  - sigma: 0.01-0.05 (探索范围)
        """
    )
    
    parser.add_argument('--split', type=str, default='toys',
                        help='数据集名称 (默认: toys)')
    parser.add_argument('--num_images', type=int, default=None,
                        help='攻击图像数量 (默认: 全部)')
    parser.add_argument('--epsilon', type=float, default=0.05,
                        help='最大扰动幅度 (默认: 0.05)')
    parser.add_argument('--max_iter', type=int, default=50,
                        help='迭代次数 (默认: 50)')
    parser.add_argument('--sigma', type=float, default=0.01,
                        help='采样标准差 (默认: 0.01)')
    parser.add_argument('--n_samples', type=int, default=20,
                        help='每次迭代采样数 (默认: 20)')
    parser.add_argument('--step_size', type=float, default=0.01,
                        help='更新步长 (默认: 0.01)')
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
    print("VIP5 黑盒对抗攻击 - NES Attack")
    print("="*60)
    print(f"输入目录: {image_dir}")
    print(f"输出目录: {output_dir}")
    print(f"Epsilon: {args.epsilon}")
    print(f"迭代次数: {args.max_iter}")
    print(f"采样数: {args.n_samples}")
    
    # 创建攻击器
    attacker = NESAttacker(
        device=args.device,
        epsilon=args.epsilon,
        max_iter=args.max_iter,
        sigma=args.sigma,
        n_samples=args.n_samples,
        step_size=args.step_size
    )
    
    # 加载目标特征
    if feature_dir.exists():
        popular_items = get_popular_items(args.split)
        if popular_items:
            target_feat = load_target_features(feature_dir, popular_items[:20])
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
        print(f"  平均余弦相似度: {np.mean(cosine_sims):.4f}")
        
        print(f"\n得分变化统计:")
        print(f"  平均得分提升: {np.mean(score_improvements):+.4f}")
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