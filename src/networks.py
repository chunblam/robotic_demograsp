import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DemoGraspConfig

class PointNetEncoder(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # 共享MLP用于点特征提取
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ELU(),
            nn.Linear(64, 128), 
            nn.ELU(),
            nn.Linear(128, cfg.POINTNET_FEAT_DIM),
            nn.ELU()
        )
        
    def forward(self, point_cloud):
        """
        论文原理：通过最大池化获取点云的全局特征
        
        Args:
            point_cloud: [B, N, 3] 点云张量
            
        Returns:
            global_features: [B, feat_dim] 全局特征
        """
        batch_size, num_points, _ = point_cloud.shape
        
        # 提取点级特征
        point_features = self.point_mlp(point_cloud)  # [B, N, feat_dim]
        
        # 全局最大池化
        global_features, _ = torch.max(point_features, dim=1)  # [B, feat_dim]
        
        return global_features

class PolicyNetwork(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # 点云编码器
        self.pointnet = PointNetEncoder(cfg)
        
        # 策略网络层
        layers = []
        input_dim = cfg.POINTNET_FEAT_DIM + cfg.EE_POSE_DIM + cfg.OBJ_POSE_DIM + cfg.HAND_POSE_DIM
        
        for hidden_dim in cfg.HIDDEN_DIMS:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ELU()
            ])
            input_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # 输出层 - 均值和log标准差
        self.mean_layer = nn.Linear(input_dim, cfg.ACTION_DIM)
        self.log_std_layer = nn.Linear(input_dim, cfg.ACTION_DIM)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """正交初始化策略网络权重"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.01)  # 小增益防止初始动作过大
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state_vector):
        """
        Args:
            state_vector: [B, state_dim] 状态向量
            
        Returns:
            action_mean: [B, action_dim] 动作均值
            action_log_std: [B, action_dim] 动作对数标准差
        """
        features = self.feature_extractor(state_vector)
        
        action_mean = self.mean_layer(features)
        action_log_std = self.log_std_layer(features)
        
        # 限制log std的范围确保数值稳定性
        action_log_std = torch.clamp(action_log_std, -20, 2)
        
        return action_mean, action_log_std
    
    def sample_action(self, state_vector):
        """
        从策略分布中采样动作
        
        论文原理：使用重参数化技巧进行可微分的动作采样
        
        Returns:
            action: 采样的动作
            log_prob: 动作的对数概率
        """
        action_mean, action_log_std = self.forward(state_vector)
        action_std = torch.exp(action_log_std)
        
        # 创建正态分布
        normal = torch.distributions.Normal(action_mean, action_std)
        
        # 重参数化采样
        action = normal.rsample()
        
        # 计算对数概率
        log_prob = normal.log_prob(action).sum(dim=-1)
        
        return action, log_prob
    
    def evaluate_actions(self, state_vector, actions):
        """
        评估给定动作的对数概率
        
        Args:
            state_vector: 状态向量
            actions: 要评估的动作
            
        Returns:
            log_prob: 动作的对数概率
            entropy: 策略分布的熵
        """
        action_mean, action_log_std = self.forward(state_vector)
        action_std = torch.exp(action_log_std)
        
        normal = torch.distributions.Normal(action_mean, action_std)
        
        log_prob = normal.log_prob(actions).sum(dim=-1)
        entropy = normal.entropy().sum(dim=-1)
        
        return log_prob, entropy

class ValueNetwork(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # 点云编码器（与策略网络共享结构）
        self.pointnet = PointNetEncoder(cfg)
        
        # 价值网络层
        layers = []
        input_dim = cfg.POINTNET_FEAT_DIM + cfg.EE_POSE_DIM + cfg.OBJ_POSE_DIM + cfg.HAND_POSE_DIM
        
        for hidden_dim in cfg.HIDDEN_DIMS:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ELU()
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        
        self.value_network = nn.Sequential(*layers)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """初始化价值网络权重"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)  # 较大增益
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state_vector):
        """
        估计状态价值
        
        论文原理：价值网络用于计算优势函数
        
        Args:
            state_vector: [B, state_dim] 状态向量
            
        Returns:
            value: [B] 状态价值估计
        """
        value = self.value_network(state_vector)
        return value.squeeze(-1)