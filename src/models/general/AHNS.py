# ReChorus/src/models/general/AHNS.py
# -*- coding: UTF-8 -*-
"""
AHNS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.BaseModel import GeneralModel
from models.general.BPRMF import BPRMFBase


class AHNS(GeneralModel, BPRMFBase):
    """
    AHNS模型
    """
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'alpha', 'beta', 'p', 'candidate_M']
    
    @staticmethod
    def parse_model_args(parser):
        """解析参数"""
        parser = BPRMFBase.parse_model_args(parser)
        
        # AHNS参数
        parser.add_argument('--alpha', type=float, default=0.5)
        parser.add_argument('--beta', type=float, default=1.0)
        parser.add_argument('--p', type=float, default=-2.0)
        parser.add_argument('--candidate_M', type=int, default=16)  # 重要：必须与num_neg匹配
        
        return GeneralModel.parse_model_args(parser)
    
    def __init__(self, args, corpus):
        """初始化"""
        GeneralModel.__init__(self, args, corpus)
        BPRMFBase._base_init(self, args, corpus)
        
        self.alpha = args.alpha
        self.beta = args.beta
        self.p = args.p
        self.candidate_M = args.candidate_M
    
    def forward(self, feed_dict):
        """
        前向传播
        """
        out_dict = BPRMFBase.forward(self, feed_dict)
        predictions = out_dict['prediction']  # [batch_size, 1+num_neg]
        
        # 如果是训练阶段，执行AHNS算法
        if feed_dict['phase'] == 'train' and predictions.shape[1] > 1:
            batch_size = predictions.shape[0]
            
            # 正样本分数
            pos_scores = predictions[:, 0]
            
            # 负样本分数
            neg_scores = predictions[:, 1:]
            
            # AHNS算法：选择最合适的负样本
            adjusted_pos = pos_scores + self.alpha
            adjusted_pos = torch.clamp(adjusted_pos, min=1e-8)  # 防止除零
            target_scores = self.beta * torch.pow(adjusted_pos, self.p + 1)
            
            # 找到最接近目标分数的负样本
            diff = torch.abs(neg_scores - target_scores.unsqueeze(1))
            _, selected_idx = torch.min(diff, dim=1)
            
            # 存储选择信息
            out_dict['selected_neg_idx'] = selected_idx
            
            # 重要：保持predictions不变，BaseRunner需要所有预测分数
            # 我们只需要在loss函数中使用选中的负样本
        
        return out_dict
    
    def loss(self, out_dict: dict) -> torch.Tensor:
        """损失函数 - 使用AHNS选择的负样本"""
        predictions = out_dict['prediction']
        
        # 确保形状正确
        if predictions.dim() != 2 or predictions.shape[1] < 2:
            raise ValueError(f"预测形状错误: {predictions.shape}，期望[batch_size, 1+num_neg]")
        
        # 如果有选中的负样本索引，使用它
        if 'selected_neg_idx' in out_dict:
            selected_idx = out_dict['selected_neg_idx']
            
            # 正样本分数
            pos_pred = predictions[:, 0]
            
            # 确保索引在有效范围内
            max_idx = predictions.shape[1] - 2  # 减去正样本
            selected_idx = torch.clamp(selected_idx, 0, max_idx)
            
            # 选中的负样本分数（索引+1，因为第一个是正样本）
            neg_pred = predictions[torch.arange(predictions.size(0)), selected_idx + 1]
            
            # 计算BPR损失
            bpr_loss = -F.logsigmoid(pos_pred - neg_pred)
            return bpr_loss.mean()
        else:
            # 没有选中索引时，使用第一个负样本
            pos_pred = predictions[:, 0]
            neg_pred = predictions[:, 1]
            return -F.logsigmoid(pos_pred - neg_pred).mean()
    
    class Dataset(GeneralModel.Dataset):
        """AHNS数据集"""
        
        def __init__(self, model, corpus, phase):
            super().__init__(model, corpus, phase)
            self.candidate_M = model.candidate_M
            
        def _get_feed_dict(self, index):
            """获取单个样本的feed dict"""
            feed_dict = super()._get_feed_dict(index)
            
            if self.phase == 'train':
                target_item = feed_dict['item_id'][0]  # 正样本
                user_id = feed_dict['user_id']
                
                # 获取用户的点击集合
                clicked_set = self.corpus.train_clicked_set[user_id] if user_id in self.corpus.train_clicked_set else set()
                
                # 采样多个负样本
                neg_items = []
                attempts = 0
                
                # 采样不重复的负样本
                while len(neg_items) < self.candidate_M:
                    neg_item = np.random.randint(1, self.corpus.n_items)
                    if neg_item not in clicked_set and neg_item != target_item:
                        neg_items.append(neg_item)
                    
                    attempts += 1
                    if attempts > 1000:  # 防止无限循环
                        break
                
                # 如果采不够，用随机值填充
                while len(neg_items) < self.candidate_M:
                    neg_item = np.random.randint(1, self.corpus.n_items)
                    if neg_item != target_item:
                        neg_items.append(neg_item)
                    else:
                        # 避免正样本作为负样本
                        neg_item = (neg_item + 1) % self.corpus.n_items
                        if neg_item == 0:
                            neg_item = 1
                        neg_items.append(neg_item)
                
                # 组合：正样本 + 负样本
                item_ids = np.array([target_item] + neg_items[:self.candidate_M], dtype=np.int64)
                feed_dict['item_id'] = item_ids
            
            return feed_dict