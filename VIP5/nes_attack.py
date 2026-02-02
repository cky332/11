#!/usr/bin/env python3
"""
VIP5 黑盒对抗攻击 - NES Attack (Natural Evolution Strategy)
============================================================
自然进化策略攻击 - 完全零阶优化方法

改进版：支持使用VIP5模型输出作为直接反馈（真正的黑盒攻击）

两种模式：
    1. CLIP代理模式（默认）：使用CLIP特征相似度作为代理得分
    2. VIP5引导模式（--use_vip5）：直接使用VIP5推荐排名作为得分
       - 完全黑盒：不需要知道热门商品
       - 两阶段攻击：CLIP热身 + VIP5精细调优
       - 多用户场景平均：更稳定的梯度估计

使用方法：
    # CLIP代理模式（快速，但可能不精准）
    python nes_attack.py --split toys --num_images 100

    # VIP5引导模式（较慢，但直接优化真实目标）
    python nes_attack.py --split toys --num_images 100 --use_vip5

    # VIP5引导 + 自定义参数
    python nes_attack.py --split toys --num_images 100 --use_vip5 --vip5_n_samples 30 --vip5_max_iter 40

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
import pickle
import random
import re
from typing import List, Dict, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import torch
import torchvision.transforms as transforms

# 导入CLIP
import clip

SCRIPT_DIR = Path(__file__).resolve().parent


# ============================================================
# 工具函数（复用自evaluate_attack_vip5.py）
# ============================================================

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def ReadLineFromFile(path):
    lines = []
    with open(path, 'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines


class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


# ============================================================
# VIP5 模型加载（用于VIP5引导模式）
# ============================================================

def create_vip5_args(split='toys', checkpoint_path=None):
    """创建VIP5模型所需的参数"""
    args = DotDict()
    args.distributed = False
    args.multiGPU = False
    args.fp16 = True
    args.split = split
    args.train = split
    args.valid = split
    args.test = split
    args.batch_size = 1
    args.optim = 'adamw'
    args.warmup_ratio = 0.1
    args.lr = 1e-3
    args.num_workers = 0
    args.clip_grad_norm = 5.0
    args.losses = 'sequential,direct,explanation'
    args.backbone = '/home/mlsnrs/data/cky/5-main/t5-small-local'

    args.image_feature_type = 'vitb32'
    args.image_feature_size_ratio = 2
    args.image_feature_dim = 512

    args.use_adapter = True
    args.reduction_factor = 8
    args.use_single_adapter = True
    args.use_vis_layer_norm = True
    args.add_adapter_cross_attn = True
    args.use_lm_head_adapter = False

    args.epoch = 20
    args.local_rank = 0
    args.dropout = 0.1
    args.tokenizer = 'p5'
    args.max_text_length = 1024
    args.gen_max_length = 64
    args.do_lower_case = False
    args.weight_decay = 0.01
    args.adam_eps = 1e-6
    args.gradient_accumulation_steps = 1
    args.seed = 2022
    args.whole_word_embed = True
    args.category_embed = True

    args.LOSSES_NAME = ['sequential_loss', 'direct_loss', 'explanation_loss', 'total_loss']
    args.gpu = 0
    args.rank = 0

    if checkpoint_path:
        args.checkpoint = checkpoint_path
    else:
        args.checkpoint = str(SCRIPT_DIR / 'snap' / 'toys-vitb32-2-8-20' / 'BEST_EVAL_LOSS.pth')

    return args


def create_vip5_config(args):
    """创建VIP5模型配置"""
    sys.path.insert(0, str(SCRIPT_DIR / 'src'))
    from adapters import AdapterConfig
    from transformers import T5Config

    config = T5Config.from_pretrained(args.backbone)

    for k, v in vars(args).items():
        setattr(config, k, v)

    image_feature_dim_dict = {
        'vitb32': 512, 'vitb16': 512, 'vitl14': 768, 'rn50': 1024, 'rn101': 512
    }
    config.feat_dim = image_feature_dim_dict[args.image_feature_type]
    config.n_vis_tokens = args.image_feature_size_ratio
    config.use_vis_layer_norm = args.use_vis_layer_norm
    config.reduction_factor = args.reduction_factor

    config.use_adapter = args.use_adapter
    config.add_adapter_cross_attn = args.add_adapter_cross_attn
    config.use_lm_head_adapter = args.use_lm_head_adapter
    config.use_single_adapter = args.use_single_adapter

    config.dropout_rate = args.dropout
    config.dropout = args.dropout
    config.attention_dropout = args.dropout
    config.activation_dropout = args.dropout

    config.losses = args.losses
    tasks = re.split("[, ]+", args.losses)

    if args.use_adapter:
        config.adapter_config = AdapterConfig()
        config.adapter_config.tasks = tasks
        config.adapter_config.d_model = config.d_model
        config.adapter_config.use_single_adapter = args.use_single_adapter
        config.adapter_config.reduction_factor = args.reduction_factor
        config.adapter_config.track_z = False
    else:
        config.adapter_config = None

    return config


class VIP5Scorer:
    """
    VIP5模型评分器 - 用于在攻击循环中直接查询VIP5获取推荐排名

    核心思想：不需要知道热门商品，直接用VIP5的推荐输出作为反馈
    """

    def __init__(self, split='toys', device='cuda', checkpoint_path=None, num_users=5, num_candidates=20):
        """
        Args:
            split: 数据集
            device: 设备
            checkpoint_path: 检查点路径
            num_users: 每次评分时使用的用户数（取平均，更稳定）
            num_candidates: 候选商品数
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.split = split
        self.num_users = num_users
        self.num_candidates = num_candidates

        sys.path.insert(0, str(SCRIPT_DIR / 'src'))
        from tokenization import P5Tokenizer
        from model import VIP5Tuning
        from utils import load_state_dict

        # 加载数据映射
        data_dir = SCRIPT_DIR / 'data' / split
        self.sequential_data = ReadLineFromFile(str(data_dir / 'sequential_data.txt'))
        self.user_items = {}
        item_count = defaultdict(int)
        for line in self.sequential_data:
            user, items = line.strip().split(' ', 1)
            items_list = [int(item) for item in items.split(' ')]
            self.user_items[user] = items_list
            for item in items_list:
                item_count[item] += 1
        self.all_items = list(item_count.keys())

        datamaps = load_json(str(data_dir / 'datamaps.json'))
        self.id2item = datamaps['id2item']
        self.item2id = datamaps['item2id']

        item2img_path = data_dir / 'item2img_dict.pkl'
        if item2img_path.exists():
            self.item2img_dict = load_pickle(str(item2img_path))
        else:
            self.item2img_dict = {}

        # 预加载所有原始特征
        feat_dir = SCRIPT_DIR / 'features' / 'vitb32_features' / f'{split}_original'
        self.original_features = {}
        if feat_dir.exists():
            for f in feat_dir.glob('*.npy'):
                self.original_features[f.stem] = np.load(f)

        # 建立 item_id -> img_filename 映射
        self.item_id_to_img = {}
        for item_id in self.id2item:
            asin = self.id2item[str(item_id)]
            if asin in self.item2img_dict:
                img_info = self.item2img_dict[asin]
                if isinstance(img_info, str):
                    img_filename = img_info
                elif isinstance(img_info, list) and len(img_info) > 0:
                    img_filename = img_info[0]
                elif isinstance(img_info, dict) and 'image' in img_info:
                    img_filename = img_info['image']
                else:
                    continue
                if '/' in img_filename:
                    img_filename = img_filename.split('/')[-1]
                if img_filename.endswith(('.jpg', '.png', '.jpeg')):
                    img_filename = img_filename.rsplit('.', 1)[0]
                self.item_id_to_img[int(item_id)] = img_filename

        # 找出有特征的商品
        self.available_items = [
            item_id for item_id, img_fn in self.item_id_to_img.items()
            if img_fn in self.original_features
        ]
        print(f"VIP5Scorer: {len(self.available_items)} available items with features")

        # 加载VIP5模型
        print("Loading VIP5 model for attack scoring...")
        vip5_args = create_vip5_args(split, checkpoint_path)
        config = create_vip5_config(vip5_args)

        self.tokenizer = P5Tokenizer.from_pretrained(
            vip5_args.backbone,
            max_length=vip5_args.max_text_length,
            do_lower_case=vip5_args.do_lower_case
        )

        self.model = VIP5Tuning.from_pretrained(vip5_args.backbone, config=config)
        self.model.to(self.device)
        self.model.resize_token_embeddings(self.tokenizer.vocab_size)
        self.model.tokenizer = self.tokenizer

        if hasattr(vip5_args, 'checkpoint') and os.path.exists(vip5_args.checkpoint):
            state_dict = load_state_dict(vip5_args.checkpoint, 'cpu')
            results = self.model.load_state_dict(state_dict, strict=False)
            print(f"VIP5 checkpoint loaded: {results}")

        self.model.eval()
        self.vip5_args = vip5_args
        self.image_feature_size_ratio = vip5_args.image_feature_size_ratio
        self.image_feature_dim = vip5_args.image_feature_dim

        # 预选一组固定的评估用户和候选集（保持一致性）
        random.seed(42)
        self._prepare_eval_scenarios()
        print("VIP5Scorer ready!")

    def _prepare_eval_scenarios(self):
        """预先准备多个评估场景，攻击时重复使用"""
        self.eval_scenarios = []
        users = list(self.user_items.keys())
        random.shuffle(users)

        count = 0
        for user_id in users:
            if count >= self.num_users:
                break
            user_seq = self.user_items[user_id]
            if len(user_seq) < 2:
                continue

            # 候选商品池（排除用户历史）
            negative_pool = [item for item in self.available_items if item not in user_seq]
            if len(negative_pool) < self.num_candidates - 1:
                continue

            negatives = random.sample(negative_pool, self.num_candidates - 1)
            self.eval_scenarios.append({
                'user_id': user_id,
                'negative_ids': negatives,
            })
            count += 1

        print(f"Prepared {len(self.eval_scenarios)} evaluation scenarios")

    def _calculate_whole_word_ids(self, tokenized_text, input_ids):
        whole_word_ids = []
        curr = 0
        for i in range(len(tokenized_text)):
            if tokenized_text[i].startswith('▁') or tokenized_text[i] == '<extra_id_0>':
                curr += 1
            whole_word_ids.append(curr)
        return whole_word_ids[:len(input_ids) - 1] + [0]

    def _get_item_feature(self, item_id):
        """获取商品的原始特征"""
        img_fn = self.item_id_to_img.get(item_id)
        if img_fn and img_fn in self.original_features:
            return self.original_features[img_fn]
        return np.zeros(self.image_feature_dim, dtype=np.float32)

    @torch.no_grad()
    def _query_vip5(self, user_id, target_item_id, candidate_ids, target_feature):
        """
        查询VIP5获取目标商品的排名

        Args:
            user_id: 用户ID
            target_item_id: 目标商品ID
            candidate_ids: 候选商品ID列表
            target_feature: 目标商品的（可能被扰动的）CLIP特征

        Returns:
            rank: 目标商品在推荐列表中的排名（1-based, 越小越好）
        """
        # 构造输入文本
        candidates_text = ' {}, '.format(
            '<extra_id_0> ' * self.image_feature_size_ratio
        ).join(
            [str(c) for c in candidate_ids]
        ) + ' <extra_id_0>' * self.image_feature_size_ratio

        source_text = f"We want to make recommendation for user_{user_id} .  Select the best item from these candidates : \n {candidates_text}"

        # 加载视觉特征
        feats = np.zeros((len(candidate_ids), self.image_feature_dim), dtype=np.float32)
        for i, cand_id in enumerate(candidate_ids):
            if cand_id == target_item_id:
                feats[i] = target_feature
            else:
                feats[i] = self._get_item_feature(cand_id)

        # Tokenize
        input_ids = self.tokenizer.encode(
            source_text, padding=True, truncation=True,
            max_length=self.vip5_args.max_text_length
        )
        tokenized_text = self.tokenizer.tokenize(source_text)
        whole_word_ids = self._calculate_whole_word_ids(tokenized_text, input_ids)
        category_ids = [1 if token_id == 32099 else 0 for token_id in input_ids]

        input_ids_t = torch.LongTensor(input_ids).unsqueeze(0).to(self.device)
        whole_word_ids_t = torch.LongTensor(whole_word_ids).unsqueeze(0).to(self.device)
        category_ids_t = torch.LongTensor(category_ids).unsqueeze(0).to(self.device)
        vis_feats_t = torch.from_numpy(feats).unsqueeze(0).to(self.device)

        # Beam search
        beam_outputs = self.model.generate(
            input_ids=input_ids_t,
            whole_word_ids=whole_word_ids_t,
            category_ids=category_ids_t,
            vis_feats=vis_feats_t,
            task='direct',
            max_length=50,
            num_beams=20,
            no_repeat_ngram_size=0,
            num_return_sequences=20,
            early_stopping=True
        )

        generated_sents = self.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)

        recommended_items = []
        for sent in generated_sents:
            try:
                item_id = int(sent.strip())
                if item_id not in recommended_items:
                    recommended_items.append(item_id)
            except:
                pass

        target_id = int(target_item_id)
        try:
            rank = recommended_items.index(target_id) + 1
        except ValueError:
            rank = len(recommended_items) + 1

        return rank

    def score_feature(self, target_item_id, target_feature):
        """
        使用VIP5评估一个特征向量的"好坏"

        对多个用户场景取平均排名，转化为得分（排名越低得分越高）

        Args:
            target_item_id: 目标商品的数字ID
            target_feature: 目标商品的CLIP特征 (512,)

        Returns:
            score: float, 越高越好
        """
        ranks = []
        for scenario in self.eval_scenarios:
            candidate_ids = scenario['negative_ids'] + [target_item_id]
            # 不shuffle以保持一致性（评估场景固定）
            rank = self._query_vip5(
                scenario['user_id'],
                target_item_id,
                candidate_ids,
                target_feature
            )
            ranks.append(rank)

        avg_rank = np.mean(ranks)
        # 转化为得分：排名越低（越好），得分越高
        # 使用倒数形式，使得排名变化在高排名时更敏感
        score = 1.0 / avg_rank
        return score


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

        # VIP5评分器（可选）
        self.vip5_scorer = None

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

    def _nes_optimize(self, img_tensor, score_fn, max_iter, n_samples,
                      sigma, step_size, momentum, epsilon, init_delta=None):
        """
        通用NES优化核心

        Args:
            img_tensor: 原始图像tensor (3, 224, 224) [0,1]
            score_fn: 评分函数，输入tensor输出float
            max_iter: 迭代次数
            n_samples: 每次采样数
            sigma: 采样标准差
            step_size: 步长
            momentum: 动量系数
            epsilon: 最大扰动
            init_delta: 初始扰动（用于第二阶段）

        Returns:
            best_delta: 最优扰动
            best_score: 最优得分
            initial_score: 初始得分
        """
        if init_delta is not None:
            delta = init_delta.clone()
        else:
            delta = torch.zeros_like(img_tensor)

        velocity = torch.zeros_like(img_tensor)

        best_delta = delta.clone()
        best_score = score_fn(torch.clamp(img_tensor + delta, 0, 1))
        initial_score = best_score

        current_step_size = step_size
        no_improve_count = 0

        for iteration in range(max_iter):
            grad_estimate = torch.zeros_like(img_tensor)

            for _ in range(n_samples // 2):
                noise = torch.randn_like(img_tensor) * sigma

                # 正向扰动
                delta_pos = torch.clamp(delta + noise, -epsilon, epsilon)
                perturbed_pos = torch.clamp(img_tensor + delta_pos, 0, 1)
                score_pos = score_fn(perturbed_pos)

                # 负向扰动
                delta_neg = torch.clamp(delta - noise, -epsilon, epsilon)
                perturbed_neg = torch.clamp(img_tensor + delta_neg, 0, 1)
                score_neg = score_fn(perturbed_neg)

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

            grad_estimate /= (n_samples * sigma)
            velocity = momentum * velocity + (1 - momentum) * grad_estimate
            delta = delta + current_step_size * torch.sign(velocity)
            delta = torch.clamp(delta, -epsilon, epsilon)

            no_improve_count += 1
            if no_improve_count > 15:
                current_step_size = min(current_step_size * 1.3, epsilon * 0.5)
                no_improve_count = 0

        return best_delta, best_score, initial_score

    def attack_single(self, image: Image.Image, target_item_id=None) -> tuple:
        """
        对单张图像执行NES攻击

        如果设置了vip5_scorer，使用两阶段攻击：
            阶段1: CLIP代理热身（快速收敛到好的初始点）
            阶段2: VIP5引导精调（直接优化推荐排名）

        否则使用纯CLIP代理攻击。

        Args:
            image: 原始图像
            target_item_id: 目标商品ID（VIP5模式需要）

        Returns:
            attacked_img: 攻击后的图像
            score_improvement: 得分提升
        """
        # 转换为224x224的tensor
        img_resized = image.resize((224, 224), Image.BICUBIC)
        img_array = np.array(img_resized).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()

        if self.vip5_scorer is not None and target_item_id is not None:
            # ===== 两阶段攻击 =====

            # 阶段1: CLIP热身（使用大部分迭代次数的40%）
            clip_iters = max(int(self.max_iter * 0.4), 20)
            clip_samples = self.n_samples

            best_delta, _, clip_initial = self._nes_optimize(
                img_tensor,
                score_fn=self.compute_score_tensor,
                max_iter=clip_iters,
                n_samples=clip_samples,
                sigma=self.sigma,
                step_size=self.step_size,
                momentum=self.momentum,
                epsilon=self.epsilon,
            )

            # 阶段2: VIP5引导精调
            vip5_iters = max(int(self.max_iter * 0.3), 15)
            vip5_samples = min(self.n_samples, 30)  # VIP5查询较慢，减少采样

            def vip5_score_fn(perturbed_tensor):
                """VIP5评分：提取CLIP特征 -> 查询VIP5 -> 返回排名得分"""
                feat = self.extract_feature_from_tensor(perturbed_tensor)
                return self.vip5_scorer.score_feature(target_item_id, feat)

            best_delta, best_score, vip5_initial = self._nes_optimize(
                img_tensor,
                score_fn=vip5_score_fn,
                max_iter=vip5_iters,
                n_samples=vip5_samples,
                sigma=self.sigma * 0.8,  # 精调阶段用更小的探索范围
                step_size=self.step_size * 0.8,
                momentum=self.momentum,
                epsilon=self.epsilon,
                init_delta=best_delta,  # 从CLIP热身结果开始
            )

            # 阶段3: 用VIP5选出最终最优（在CLIP最优和VIP5最优之间选择）
            score_improvement = best_score - vip5_initial

        else:
            # ===== 纯CLIP代理攻击 =====
            best_delta, best_score, initial_score = self._nes_optimize(
                img_tensor,
                score_fn=self.compute_score_tensor,
                max_iter=self.max_iter,
                n_samples=self.n_samples,
                sigma=self.sigma,
                step_size=self.step_size,
                momentum=self.momentum,
                epsilon=self.epsilon,
            )
            score_improvement = best_score - initial_score

        # 生成攻击图像
        attacked_tensor = torch.clamp(img_tensor + best_delta, 0, 1)
        attacked_array = (attacked_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        attacked_img = Image.fromarray(attacked_array)

        return attacked_img, float(score_improvement)

    def attack_batch(self, image_dir: Path, output_dir: Path,
                     num_images: Optional[int] = None,
                     item_id_map: Optional[Dict[str, int]] = None) -> Dict:
        """
        批量攻击图像

        Args:
            image_dir: 原始图像目录
            output_dir: 输出目录
            num_images: 攻击图像数量
            item_id_map: 图片文件名(stem) -> 数字item_id 的映射（VIP5模式需要）
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))

        if num_images and num_images < len(image_files):
            random.seed(42)
            image_files = random.sample(image_files, num_images)

        use_vip5 = self.vip5_scorer is not None

        print(f"\n{'='*60}")
        print(f"开始NES攻击 {len(image_files)} 张图像")
        print(f"{'='*60}")
        print(f"攻击模式: {'VIP5引导 (两阶段)' if use_vip5 else 'CLIP代理'}")
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

                # 获取target_item_id（VIP5模式需要）
                target_item_id = None
                if use_vip5 and item_id_map:
                    target_item_id = item_id_map.get(img_path.stem)

                attacked_img, score_improvement = self.attack_single(
                    original_img, target_item_id=target_item_id
                )
                attacked_feat = self.extract_feature(attacked_img)
                attacked_score = self.compute_score(attacked_img)

                # 保存为PNG无损格式
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
                import traceback
                traceback.print_exc()
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


def build_img_to_itemid_map(split: str) -> Dict[str, int]:
    """
    构建图片文件名(stem) -> 数字item_id的映射

    这样在攻击时可以知道每张图片对应哪个商品ID，用于VIP5查询
    """
    data_dir = SCRIPT_DIR / 'data' / split
    datamaps = load_json(str(data_dir / 'datamaps.json'))
    id2item = datamaps['id2item']

    item2img_path = data_dir / 'item2img_dict.pkl'
    if not item2img_path.exists():
        return {}

    item2img_dict = load_pickle(str(item2img_path))

    img_to_itemid = {}
    for item_id, asin in id2item.items():
        if asin in item2img_dict:
            img_info = item2img_dict[asin]
            if isinstance(img_info, str):
                img_filename = img_info
            elif isinstance(img_info, list) and len(img_info) > 0:
                img_filename = img_info[0]
            elif isinstance(img_info, dict) and 'image' in img_info:
                img_filename = img_info['image']
            else:
                continue
            if '/' in img_filename:
                img_filename = img_filename.split('/')[-1]
            if img_filename.endswith(('.jpg', '.png', '.jpeg')):
                img_filename = img_filename.rsplit('.', 1)[0]
            img_to_itemid[img_filename] = int(item_id)

    print(f"Built img->itemid map with {len(img_to_itemid)} entries")
    return img_to_itemid


def main():
    parser = argparse.ArgumentParser(
        description='VIP5 黑盒对抗攻击 - NES Attack',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # CLIP代理模式 (快速)
  python nes_attack.py --split toys --num_images 100

  # VIP5引导模式 (较慢但更精准 - 真正的黑盒攻击)
  python nes_attack.py --split toys --num_images 100 --use_vip5

  # VIP5引导 + 自定义参数
  python nes_attack.py --split toys --num_images 50 --use_vip5 --vip5_num_users 8 --vip5_num_candidates 30

参数建议:
  CLIP模式:
    - epsilon: 0.2-0.5
    - max_iter: 100-300
    - n_samples: 50-200

  VIP5模式:
    - epsilon: 0.2-0.5
    - max_iter: 100-200 (会自动分配给两个阶段)
    - n_samples: 50-100
    - vip5_num_users: 3-10 (更多用户=更稳定但更慢)
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
                        help='使用top-k热门商品的特征作为目标 (默认: 3, 仅CLIP模式)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备 (默认: cuda)')

    # VIP5引导模式参数
    parser.add_argument('--use_vip5', action='store_true',
                        help='启用VIP5引导模式（直接用VIP5输出作为攻击反馈）')
    parser.add_argument('--vip5_num_users', type=int, default=5,
                        help='VIP5评分时使用的用户场景数 (默认: 5)')
    parser.add_argument('--vip5_num_candidates', type=int, default=20,
                        help='VIP5评分时的候选商品数 (默认: 20)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='VIP5模型检查点路径')

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
    if args.use_vip5:
        print("VIP5 黑盒对抗攻击 - NES Attack (VIP5引导模式)")
    else:
        print("VIP5 黑盒对抗攻击 - NES Attack (CLIP代理模式)")
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

    # 构建映射
    item_id_map = None

    if args.use_vip5:
        # VIP5引导模式：加载VIP5模型作为评分器
        print("\n--- 初始化VIP5评分器 ---")
        attacker.vip5_scorer = VIP5Scorer(
            split=args.split,
            device=args.device,
            checkpoint_path=args.checkpoint,
            num_users=args.vip5_num_users,
            num_candidates=args.vip5_num_candidates,
        )

        # 同时也设置CLIP target features（用于阶段1热身）
        if feature_dir.exists():
            popular_items = get_popular_items(args.split)
            if popular_items:
                target_feat = load_target_features(feature_dir, popular_items[:args.target_topk])
                if target_feat is not None:
                    attacker.set_target_features(target_feat)

        # 构建图片文件名 -> item_id 映射
        item_id_map = build_img_to_itemid_map(args.split)

    else:
        # CLIP代理模式：使用热门商品特征
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
    results = attacker.attack_batch(image_dir, output_dir, args.num_images,
                                     item_id_map=item_id_map)

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
            'mode': 'vip5_guided' if args.use_vip5 else 'clip_proxy',
            'epsilon': args.epsilon,
            'max_iter': args.max_iter,
            'sigma': args.sigma,
            'n_samples': args.n_samples,
            'step_size': args.step_size,
            'momentum': args.momentum,
            'target_topk': args.target_topk,
            'use_vip5': args.use_vip5,
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
