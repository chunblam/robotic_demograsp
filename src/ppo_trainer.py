import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import time
from networks import PolicyNetwork, ValueNetwork

class PPOTrainer:
    def __init__(self, mdp_env, cfg):
        """
        初始化PPO训练器
        
        论文原理：PPO通过裁剪确保策略更新不会过大
        
        Args:
            mdp_env: 单步MDP环境
            cfg: 配置参数
        """
        self.mdp_env = mdp_env
        self.cfg = cfg
        self.device = cfg.SIM_DEVICE
        
        # 创建网络
        self.policy_net = PolicyNetwork(cfg).to(self.device)
        self.value_net = ValueNetwork(cfg).to(self.device)
        
        # 优化器
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.LEARNING_RATE)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=cfg.LEARNING_RATE)
        
        # 训练状态
        self.episode_rewards = deque(maxlen=100)
        self.best_mean_reward = -float('inf')
        self.iteration = 0
        
        # 动作边界
        self.action_low, self.action_high = mdp_env.get_action_bounds()
        
        print("PPO训练器初始化完成")
        print(f"  策略网络参数: {sum(p.numel() for p in self.policy_net.parameters()):,}")
        print(f"  价值网络参数: {sum(p.numel() for p in self.value_net.parameters()):,}")
    
    def train(self, num_iterations):
        """
        主训练循环
        
        论文原理：在多个物体上并行训练通用抓取策略
        """
        print(f"开始PPO训练，总迭代次数: {num_iterations}")
        start_time = time.time()
        
        for iteration in range(num_iterations):
            self.iteration = iteration
            
            # 收集经验
            rollouts = self._collect_rollouts()
            
            # 更新网络
            policy_loss, value_loss, entropy = self._update_networks(rollouts)
            
            # 记录训练进度
            self._log_training_progress(iteration, policy_loss, value_loss, entropy, 
                                      rollouts['rewards'], start_time)
            
            # 定期保存模型
            if iteration % 100 == 0:
                self._save_checkpoint(iteration)
        
        print("训练完成!")
        self._save_checkpoint(num_iterations, is_final=True)
    
    def _collect_rollouts(self):
        """
        收集经验数据
        
        论文原理：在单步MDP中收集(s, a, r)三元组
        """
        # 重置环境
        states = self.mdp_env.reset()
        
        # 将状态转换为向量
        state_vectors = self.mdp_env.get_state_vector(states)
        
        # 采样动作
        with torch.no_grad():
            actions, log_probs = self.policy_net.sample_action(state_vectors)
            values = self.value_net(state_vectors)
        
        # 限制动作范围
        actions = torch.clamp(actions, self.action_low, self.action_high)
        
        # 执行动作
        next_states, rewards, dones, infos = self.mdp_env.step(actions)
        
        # 存储经验
        rollouts = {
            'states': states,
            'state_vectors': state_vectors,
            'actions': actions,
            'rewards': rewards,
            'values': values,
            'log_probs': log_probs,
            'dones': dones,
            'infos': infos
        }
        
        return rollouts
    
    def _update_networks(self, rollouts):
        """
        更新策略和价值网络
        
        论文原理：使用PPO裁剪目标函数进行多轮小批量更新
        """
        # 计算优势函数和回报
        advantages, returns = self._compute_advantages_and_returns(rollouts)
        
        # 多轮更新
        policy_losses = []
        value_losses = [] 
        entropies = []
        
        for epoch in range(5):  # PPO更新轮数
            # 创建随机小批量
            batch_indices = torch.randperm(self.cfg.NUM_ENVS)
            
            for start_idx in range(0, self.cfg.NUM_ENVS, self.cfg.BATCH_SIZE):
                end_idx = min(start_idx + self.cfg.BATCH_SIZE, self.cfg.NUM_ENVS)
                batch_idx = batch_indices[start_idx:end_idx]
                
                # 获取小批量数据
                batch_states = {k: v[batch_idx] for k, v in rollouts['states'].items()}
                batch_state_vectors = rollouts['state_vectors'][batch_idx]
                batch_actions = rollouts['actions'][batch_idx]
                batch_old_log_probs = rollouts['log_probs'][batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                
                # 更新策略网络
                policy_loss, entropy = self._update_policy(
                    batch_state_vectors, batch_actions, batch_old_log_probs, batch_advantages)
                
                # 更新价值网络
                value_loss = self._update_value_net(batch_state_vectors, batch_returns)
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())
        
        return np.mean(policy_losses), np.mean(value_losses), np.mean(entropies)
    
    def _compute_advantages_and_returns(self, rollouts):
        """
        计算优势函数和回报
        
        论文原理：在单步MDP中，优势函数简化为 A_t = r_t - V(s_t)
        """
        rewards = rollouts['rewards']
        values = rollouts['values']
        dones = rollouts['dones']
        
        # 单步MDP的优势函数计算
        advantages = rewards - values
        
        # 回报就是即时奖励（单步MDP，γ=1）
        returns = rewards.clone()
        
        # 标准化优势函数
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def _update_policy(self, state_vectors, actions, old_log_probs, advantages):
        """
        更新策略网络使用PPO裁剪
        
        论文原理：L^CLIP(θ) = E[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]
        """
        # 计算新策略的概率
        new_log_probs, entropy = self.policy_net.evaluate_actions(state_vectors, actions)
        
        # 概率比
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # 裁剪的目标函数
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.cfg.CLIP_EPS, 1 + self.cfg.CLIP_EPS) * advantages
        
        # PPO裁剪损失
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 总损失 = 策略损失 - 熵奖励
        total_loss = policy_loss - self.cfg.ENTROPY_COEF * entropy.mean()
        
        # 更新策略网络
        self.policy_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.cfg.MAX_GRAD_NORM)
        self.policy_optimizer.step()
        
        return policy_loss, entropy.mean()
    
    def _update_value_net(self, state_vectors, returns):
        """
        更新价值网络
        
        论文原理：L^VF(φ) = E[(V_φ(s_t) - V_t^target)^2]
        """
        # 价值网络预测
        values = self.value_net(state_vectors)
        
        # 价值损失
        value_loss = F.mse_loss(values, returns)
        
        # 更新价值网络
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.cfg.MAX_GRAD_NORM)
        self.value_optimizer.step()
        
        return value_loss
    
    def _log_training_progress(self, iteration, policy_loss, value_loss, entropy, rewards, start_time):
        """记录训练进度"""
        mean_reward = rewards.mean().item()
        success_rate = (rewards == self.cfg.SUCCESS_REWARD).float().mean().item()
        collision_rate = (rewards == self.cfg.COLLISION_PENALTY).float().mean().item()
        
        self.episode_rewards.append(mean_reward)
        
        # 更新最佳模型
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self._save_checkpoint(iteration, is_best=True)
        
        # 打印进度
        if iteration % 10 == 0:
            elapsed_time = time.time() - start_time
            avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
            
            print(f"Iter {iteration:4d} | "
                  f"Reward: {mean_reward:6.3f} | "
                  f"Success: {success_rate:5.1%} | "
                  f"Collision: {collision_rate:5.1%} | "
                  f"Policy Loss: {policy_loss:7.4f} | "
                  f"Value Loss: {value_loss:7.4f} | "
                  f"Entropy: {entropy:6.3f} | "
                  f"Time: {elapsed_time:6.1f}s")
    
    def _save_checkpoint(self, iteration, is_best=False, is_final=False):
        """保存模型检查点"""
        if is_best:
            filename = f"best_model_iter_{iteration}.pth"
        elif is_final:
            filename = "final_model.pth"
        else:
            filename = f"checkpoint_iter_{iteration}.pth"
        
        checkpoint = {
            'iteration': iteration,
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'best_mean_reward': self.best_mean_reward,
            'cfg': self.cfg
        }
        
        torch.save(checkpoint, filename)
        
        if is_best:
            print(f"新的最佳模型已保存: {filename} (奖励: {self.best_mean_reward:.3f})")
    
    def load_checkpoint(self, filename):
        """加载模型检查点"""
        checkpoint = torch.load(filename, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.best_mean_reward = checkpoint['best_mean_reward']
        
        print(f"模型已从 {filename} 加载 (迭代: {checkpoint['iteration']})")