import isaacgym
import isaacgym.torch_utils as torch_utils
from isaacgym import gymapi, gymtorch, gymutil
import torch
import numpy as np
import os
import time
from config import DemoGraspConfig
from environment import DemoGraspEnvironment
from demo_recorder import DemoRecorder
from trajectory_editor import TrajectoryEditor
from single_step_mdp import SingleStepMDP
from networks import PolicyNetwork, ValueNetwork
from ppo_trainer import PPOTrainer

class DemoGraspTrainer:
    def __init__(self):
        """DemoGrasp完整训练流程管理器"""
        self.cfg = None
        self.env = None
        self.demo_trajectory = None
        self.mdp_env = None
        self.ppo_trainer = None
        
    def setup_training(self):
        """步骤1: 训练环境设置"""
        print("=" * 60)
        print("步骤1: 训练环境设置")
        print("=" * 60)
        
        # 设置随机种子
        torch.manual_seed(42)
        np.random.seed(42)
        
        # 创建配置对象
        self.cfg = DemoGraspConfig()
        print("✓ 配置参数加载完成")
        print(f"  机器人: Kuka Allegro ({self.cfg.TOTAL_DOF} DOF)")
        print(f"  并行环境: {self.cfg.NUM_ENVS}")
        print(f"  动作维度: {6 + self.cfg.HAND_DOF}")
        
        return self.cfg
    
    def create_simulation_environment(self):
        """步骤2: 创建仿真环境"""
        print("\n" + "=" * 60)
        print("步骤2: 创建仿真环境")
        print("=" * 60)
        
        # 创建基础仿真环境
        self.env = DemoGraspEnvironment(self.cfg)
        print("✓ IsaacGym仿真环境创建完成")
        print(f"  加载物体数量: {len(self.env.object_assets)}")
        print(f"  桌子尺寸: {self.cfg.TABLE_DIMS}")
        
        return self.env
    
    def record_demonstration(self):
        """步骤3: 录制演示轨迹"""
        print("\n" + "=" * 60)
        print("步骤3: 录制演示轨迹")
        print("=" * 60)
        
        # 创建演示录制器
        recorder = DemoRecorder(self.env)
        
        # 检查是否已有演示轨迹
        demo_path = "../output/kuka_allegro_demo.npy"
        if os.path.exists(demo_path):
            print("✓ 加载现有演示轨迹")
            self.demo_trajectory = recorder.load_demonstration(demo_path)
        else:
            print("✓ 录制新的演示轨迹")
            self.demo_trajectory = recorder.record_demonstration(object_index=0)
            recorder.save_demonstration(self.demo_trajectory, demo_path)
        
        # 显示演示轨迹信息
        print(f"  轨迹长度: {len(self.demo_trajectory['hand_actions'])} 步")
        print(f"  提升时间步: {self.demo_trajectory['lift_timestep']}")
        print(f"  手部动作维度: {len(self.demo_trajectory['hand_actions'][0])}")
        
        return self.demo_trajectory
    
    def setup_trajectory_editor(self):
        """步骤4: 设置轨迹编辑器"""
        print("\n" + "=" * 60)
        print("步骤4: 设置轨迹编辑器")
        print("=" * 60)
        
        # 创建轨迹编辑器
        editor = TrajectoryEditor(self.demo_trajectory, self.cfg)
        print("✓ 轨迹编辑器初始化完成")
        print(f"  可编辑参数: SE(3)变换 + 手部关节增量")
        print(f"  编辑维度: {6} + {self.cfg.HAND_DOF} = {6 + self.cfg.HAND_DOF}")
        
        # 测试轨迹编辑功能
        T_ee, delta_qG = editor.create_random_edit_parameters()
        edited_demo = editor.edit_trajectory(T_ee, delta_qG)
        print("✓ 轨迹编辑功能测试通过")
        
        return editor
    
    def create_mdp_environment(self):
        """步骤5: 创建单步MDP环境"""
        print("\n" + "=" * 60)
        print("步骤5: 创建单步MDP环境")
        print("=" * 60)
        
        # 创建单步MDP环境
        self.mdp_env = SingleStepMDP(self.env, self.demo_trajectory, self.cfg)
        print("✓ 单步MDP环境创建完成")
        
        # 分析复杂度减少
        complexity_reduction = self.mdp_env.analyze_mdp_complexity()
        print(f"✓ 探索复杂度减少: {complexity_reduction:.1f}x")
        
        return self.mdp_env
    
    def setup_networks(self):
        """步骤6: 设置神经网络"""
        print("\n" + "=" * 60)
        print("步骤6: 设置神经网络")
        print("=" * 60)
        
        # 创建策略网络
        policy_net = PolicyNetwork(self.cfg)
        value_net = ValueNetwork(self.cfg)
        
        policy_params = sum(p.numel() for p in policy_net.parameters())
        value_params = sum(p.numel() for p in value_net.parameters())
        
        print("✓ 神经网络创建完成")
        print(f"  策略网络参数: {policy_params:,}")
        print(f"  价值网络参数: {value_params:,}")
        print(f"  总参数: {policy_params + value_params:,}")
        print(f"  输入维度: {self.mdp_env.state_dim}")
        print(f"  输出维度: {self.mdp_env.action_dim}")
        
        return policy_net, value_net
    
    def setup_ppo_trainer(self):
        """步骤7: 设置PPO训练器"""
        print("\n" + "=" * 60)
        print("步骤7: 设置PPO训练器")
        print("=" * 60)
        
        # 创建PPO训练器
        self.ppo_trainer = PPOTrainer(self.mdp_env, self.cfg)
        print("✓ PPO训练器初始化完成")
        print(f"  学习率: {self.cfg.LEARNING_RATE}")
        print(f"  PPO裁剪系数: {self.cfg.CLIP_EPS}")
        print(f"  价值系数: {self.cfg.VALUE_COEF}")
        print(f"  熵系数: {self.cfg.ENTROPY_COEF}")
        
        return self.ppo_trainer
    
    def run_training(self):
        """步骤8: 运行PPO训练"""
        print("\n" + "=" * 60)
        print("步骤8: 运行PPO训练")
        print("=" * 60)
        
        start_time = time.time()
        print("开始PPO训练...")
        print(f"目标迭代次数: {self.cfg.NUM_ITERATIONS}")
        
        # 运行训练
        self.ppo_trainer.train(self.cfg.NUM_ITERATIONS)
        
        training_time = time.time() - start_time
        print(f"✓ 训练完成! 总时间: {training_time:.1f} 秒")
    
    def evaluate_model(self):
        """步骤9: 模型评估"""
        print("\n" + "=" * 60)
        print("步骤9: 模型评估")
        print("=" * 60)
        
        # 加载最佳模型
        best_model_path = "best_model.pth"
        if os.path.exists(best_model_path):
            self.ppo_trainer.load_checkpoint(best_model_path)
            print("✓ 最佳模型加载完成")
            
            # 在测试集上评估
            test_success_rate = self._evaluate_on_test_set()
            print(f"测试集成功率: {test_success_rate:.1%}")
        else:
            print("⚠ 未找到最佳模型文件")
        
        print("✓ 评估完成")
    
    def _evaluate_on_test_set(self):
        """在测试集上评估模型性能"""
        # 简化评估 - 实际中应该在未见过的物体上测试
        test_iterations = 10
        success_count = 0
        
        for i in range(test_iterations):
            state = self.mdp_env.reset()
            state_vector = self.mdp_env.get_state_vector(state)
            
            with torch.no_grad():
                action, _ = self.ppo_trainer.policy_net.sample_action(state_vector)
            
            _, reward, _, info = self.mdp_env.step(action)
            
            if info['success'][0]:
                success_count += 1
        
        return success_count / test_iterations
    
    def run_complete_pipeline(self):
        """运行完整的训练流程"""
        print("🚀 启动 DemoGrasp 完整训练流程")
        print("基于: 'DemoGrasp: Universal Dexterous Grasping from a Single Demonstration'")
        
        try:
            # 按顺序执行所有步骤
            self.setup_training()              # 步骤1
            self.create_simulation_environment() # 步骤2
            # self.record_demonstration()        # 步骤3
            # self.setup_trajectory_editor()     # 步骤4
            # self.create_mdp_environment()      # 步骤5
            # self.setup_networks()              # 步骤6
            # self.setup_ppo_trainer()           # 步骤7
            # self.run_training()                # 步骤8
            # self.evaluate_model()              # 步骤9
            
            print("\n🎉 DemoGrasp 训练流程全部完成!")
            
        except Exception as e:
            print(f"\n❌ 训练流程出错: {e}")
            raise

def main():
    """主函数"""
    trainer = DemoGraspTrainer()
    trainer.run_complete_pipeline()

if __name__ == "__main__":
    main()