import torch
import numpy as np
from environment import DemoGraspEnvironment
from trajectory_editor import TrajectoryEditor
from config import DemoGraspConfig

class SingleStepMDP:
    def __init__(self, env, demo_trajectory, cfg):
        """
        初始化单步MDP环境
        
        论文原理：将复杂的抓取任务重新表述为单步决策问题
        
        Args:
            env: 基础环境实例
            demo_trajectory: 录制的演示轨迹
            cfg: 配置参数
        """
        self.env = env
        self.demo = demo_trajectory
        self.cfg = cfg
        self.trajectory_editor = TrajectoryEditor(demo_trajectory, cfg)
        
        # MDP状态维度
        self.state_dim = self._compute_state_dim()
        self.action_dim = self._compute_action_dim()
        
        print(f"单步MDP初始化完成:")
        print(f"  状态维度: {self.state_dim}")
        print(f"  动作维度: {self.action_dim}")
        print(f"  传统方法动作维度: {self.cfg.TOTAL_DOF + 6} (对比减少 {(self.cfg.TOTAL_DOF + 6 - self.action_dim) / (self.cfg.TOTAL_DOF + 6) * 100:.1f}%)")
    
    def _compute_state_dim(self):
        """计算状态空间维度"""
        # 状态包含: 点云特征 + 末端位姿 + 物体位姿 + 手部姿态
        state_dim = (self.cfg.POINTNET_FEAT_DIM +      # 点云特征
                    self.cfg.EE_POSE_DIM +            # 末端执行器位姿
                    self.cfg.OBJ_POSE_DIM +           # 物体位姿
                    self.cfg.HAND_POSE_DIM)           # 手部关节角度
        return state_dim
    
    def _compute_action_dim(self):
        """计算动作空间维度"""
        # 动作包含: SE(3)变换参数 + 手部关节增量
        # SE(3): 3个平移 + 3个旋转 = 6个参数
        # 手部增量: 16个关节
        action_dim = 6 + self.cfg.HAND_DOF
        return action_dim
    
    def reset(self, env_ids=None):
        """
        重置MDP环境
        
        论文原理：每次重置后，策略在新的初始状态下做出单步决策
        
        Returns:
            state: 初始状态观测
        """
        # 重置基础环境
        observations = self.env.reset(env_ids)
        
        # 提取单步MDP所需的状态信息
        state = self._extract_mdp_state(observations)
        
        # 保存初始状态用于轨迹执行
        self.initial_observations = observations
        
        return state
    
    def _extract_mdp_state(self, observations):
        """
        从环境观测中提取MDP状态
        
        论文原理：状态包含初始时刻的所有必要信息
        s = (p_0_ee, p_0_obj, c_0_obj, q_0_hand)
        """
        state = {
            'ee_pose_0': observations['ee_pose'],      # 初始末端位姿
            'obj_pose_0': observations['obj_pose'],    # 初始物体位姿
            'point_cloud_0': observations['point_cloud'],  # 物体点云
            'hand_pose_0': observations['hand_pose']   # 初始手部姿态
        }
        return state
    
    def step(self, actions):
        """
        执行单步MDP
        
        论文原理：
        1. 策略输出编辑参数 (T_ee, Δq^G)
        2. 编辑演示轨迹
        3. 执行编辑后的轨迹
        4. 计算奖励并终止
        
        Args:
            actions: 策略输出的动作 [batch_size, action_dim]
            
        Returns:
            next_state: 终止状态（实际不使用）
            rewards: 单步奖励
            dones: 完成标志（总是True）
            info: 额外信息
        """
        batch_size = actions.shape[0] if torch.is_tensor(actions) else len(actions)
        
        print(f"执行单步MDP，批量大小: {batch_size}")
        
        # 解析动作参数
        T_ee_params, delta_qG_params = self._parse_actions(actions)
        
        rewards = []
        successes = []
        collisions = []
        
        # 对每个环境单独处理
        for i in range(batch_size):
            # 编辑演示轨迹
            edited_demo = self._edit_and_execute_demo(i, T_ee_params[i], delta_qG_params[i])
            
            # 计算奖励
            reward, success, collision = self._compute_episode_reward()
            rewards.append(reward)
            successes.append(success)
            collisions.append(collision)
        
        # 转换为张量
        rewards = torch.tensor(rewards, device=self.env.device, dtype=torch.float32)
        dones = torch.ones(batch_size, device=self.env.device, dtype=torch.bool)
        
        # 构建信息字典
        info = {
            'success': torch.tensor(successes, device=self.env.device),
            'collision': torch.tensor(collisions, device=self.env.device),
            'T_ee_params': T_ee_params,
            'delta_qG_params': delta_qG_params
        }
        
        # 单步MDP在动作执行后终止
        next_state = self._get_terminal_state(batch_size)
        
        return next_state, rewards, dones, info
    
    def _parse_actions(self, actions):
        """
        解析策略输出的动作参数
        
        论文原理：动作空间被设计为紧凑的编辑参数表示
        
        Args:
            actions: [batch_size, action_dim] 策略输出
            
        Returns:
            T_ee_params: SE(3)变换参数列表
            delta_qG_params: 手部关节增量列表
        """
        if torch.is_tensor(actions):
            actions = actions.cpu().numpy()
        
        batch_size = actions.shape[0]
        T_ee_params = []
        delta_qG_params = []
        
        for i in range(batch_size):
            action = actions[i]
            
            # 前6个参数: SE(3)变换
            translation = action[:3] * self.cfg.MAX_TRANSLATION
            rotation_euler = action[3:6] * self.cfg.MAX_ROTATION
            
            # 构建SE(3)变换矩阵
            T_ee = self._build_se3_matrix(translation, rotation_euler)
            
            # 后16个参数: 手部关节增量
            delta_qG = action[6:] * self.cfg.MAX_HAND_DELTA
            
            T_ee_params.append(T_ee)
            delta_qG_params.append(delta_qG)
        
        return T_ee_params, delta_qG_params
    
    def _build_se3_matrix(self, translation, rotation_euler):
        """
        从平移和欧拉角构建SE(3)矩阵
        
        数学原理：SE(3) = R^3 × SO(3)
        """
        T = np.eye(4)
        
        # 设置平移
        T[0:3, 3] = translation
        
        # 从欧拉角计算旋转矩阵
        roll, pitch, yaw = rotation_euler
        
        # 绕x轴旋转
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        # 绕y轴旋转
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        # 绕z轴旋转
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # 组合旋转: R = Rz * Ry * Rx
        R = Rz @ Ry @ Rx
        T[0:3, 0:3] = R
        
        return T
    
    def _edit_and_execute_demo(self, env_id, T_ee, delta_qG):
        """
        编辑并执行演示轨迹
        
        论文原理：这是单步MDP的核心 - 将编辑参数转换为完整的抓取轨迹
        """
        # 编辑演示轨迹
        edited_demo = self.trajectory_editor.edit_trajectory(T_ee, delta_qG)
        
        # 执行编辑后的轨迹
        self._execute_edited_trajectory(env_id, edited_demo)
        
        return edited_demo
    
    def _execute_edited_trajectory(self, env_id, edited_demo):
        """执行编辑后的轨迹 - 确保使用IK"""
        env_ptr = self.env.envs[env_id]
        robot_handle = self.env.robot_handles[env_id]
        
        # 获取初始物体位姿（用于坐标系转换）
        initial_obj_pose = edited_demo['object_pose_0']
        T_world_to_obj = self._get_world_to_object_transform(initial_obj_pose)
        
        # 执行轨迹中的每个时间步
        for t in range(len(edited_demo['hand_actions'])):
            hand_action = edited_demo['hand_actions'][t]
            ee_pose_obj = edited_demo['ee_poses_obj_frame'][t]
            
            # 使用改进的执行方法（包含IK）
            self.env._execute_demo_action_with_ik(env_id, hand_action, ee_pose_obj, T_world_to_obj)
            
            # 每10步刷新一次状态张量（提高性能）
            if t % 10 == 0:
                self.env.gym.refresh_actor_root_state_tensor(self.env.sim)
                self.env.gym.refresh_dof_state_tensor(self.env.sim)
        
        # 最终刷新状态张量
        self.env.gym.refresh_actor_root_state_tensor(self.env.sim)
        self.env.gym.refresh_dof_state_tensor(self.env.sim)
        self.env.gym.refresh_rigid_body_state_tensor(self.env.sim)
    
    def _get_world_to_object_transform(self, obj_pose):
        """计算世界坐标系到物体坐标系的变换"""
        pos = obj_pose[:3]
        quat = obj_pose[3:]
        
        T = np.eye(4)
        T[:3, :3] = self.trajectory_editor.quaternion_to_rotation_matrix(quat)
        T[:3, 3] = pos
        
        return np.linalg.inv(T)
    
    def _transform_pose_to_world(self, pose_obj, T_world_to_obj):
        """将物体坐标系下的位姿转换到世界坐标系"""
        pos_obj = pose_obj[:3]
        quat_obj = pose_obj[3:]
        
        # 位置转换
        pos_obj_homo = np.array([*pos_obj, 1.0])
        pos_world_homo = np.linalg.inv(T_world_to_obj) @ pos_obj_homo
        pos_world = pos_world_homo[:3]
        
        # 旋转转换
        R_obj = self.trajectory_editor.quaternion_to_rotation_matrix(quat_obj)
        R_world_to_obj = T_world_to_obj[:3, :3]
        R_world = np.linalg.inv(R_world_to_obj) @ R_obj
        
        quat_world = self.trajectory_editor.rotation_matrix_to_quaternion(R_world)
        
        return np.concatenate([pos_world, quat_world])
    
    def _compute_episode_reward(self):
        """
        计算单步MDP的奖励
        
        论文原理：极其稀疏的二元奖励函数
        r = 1[success] · 1[no collision during execution]
        """
        # 使用环境的奖励计算函数
        reward_tensor = self.env.compute_reward()
        
        # 对于单步MDP，我们只关心第一个环境的奖励（演示环境）
        reward = reward_tensor[0].item() if torch.is_tensor(reward_tensor) else reward_tensor
        
        # 解析成功和碰撞信息
        success = reward == self.cfg.SUCCESS_REWARD
        collision = reward == self.cfg.COLLISION_PENALTY
        
        return reward, success, collision
    
    def _get_terminal_state(self, batch_size):
        """
        获取终止状态
        
        论文原理：单步MDP在执行动作后立即终止
        """
        # 返回一个占位符状态，实际在单步MDP中不使用
        terminal_state = {
            'ee_pose_0': torch.zeros((batch_size, self.cfg.EE_POSE_DIM), device=self.env.device),
            'obj_pose_0': torch.zeros((batch_size, self.cfg.OBJ_POSE_DIM), device=self.env.device),
            'point_cloud_0': torch.zeros((batch_size, self.cfg.POINT_CLOUD_SIZE, 3), device=self.env.device),
            'hand_pose_0': torch.zeros((batch_size, self.cfg.HAND_POSE_DIM), device=self.env.device)
        }
        return terminal_state
    
    def get_state_vector(self, state):
        """
        将状态字典转换为向量形式，用于神经网络输入
        
        论文原理：状态特征被拼接成一个向量输入到策略网络
        """
        # 点云特征（使用简化特征）
        point_cloud_feat = torch.randn(state['point_cloud_0'].shape[0], 
                                     self.cfg.POINTNET_FEAT_DIM, 
                                     device=self.env.device)
        
        # 拼接所有状态特征
        state_vector = torch.cat([
            point_cloud_feat,                    # [batch_size, 128]
            state['ee_pose_0'],                  # [batch_size, 7]
            state['obj_pose_0'],                 # [batch_size, 7]  
            state['hand_pose_0']                 # [batch_size, 16]
        ], dim=-1)                               # 总维度: 128 + 7 + 7 + 16 = 158
        
        return state_vector
    
    def get_action_bounds(self):
        """
        获取动作空间的边界
        
        论文原理：动作参数被归一化到[-1, 1]范围
        """
        # 所有动作参数都在[-1, 1]范围内
        low = -torch.ones(self.action_dim, device=self.env.device)
        high = torch.ones(self.action_dim, device=self.env.device)
        
        return low, high
    
    def analyze_mdp_complexity(self):
        """
        分析MDP复杂度减少的效果
        
        论文原理：单步MDP显著减少了探索复杂度
        """
        traditional_action_dim = self.cfg.TOTAL_DOF + 6  # 关节控制 + 末端位姿
        traditional_episode_length = self.cfg.DEMO_LENGTH  # 完整轨迹长度
        
        demo_grasp_action_dim = self.action_dim
        demo_grasp_episode_length = 1  # 单步决策
        
        complexity_reduction = (traditional_action_dim * traditional_episode_length) / \
                             (demo_grasp_action_dim * demo_grasp_episode_length)
        
        print(f"\n=== MDP复杂度分析 ===")
        print(f"传统方法:")
        print(f"  动作维度: {traditional_action_dim}")
        print(f"  回合长度: {traditional_episode_length}")
        print(f"  总决策数: {traditional_action_dim * traditional_episode_length}")
        
        print(f"DemoGrasp单步MDP:")
        print(f"  动作维度: {demo_grasp_action_dim}")
        print(f"  回合长度: {demo_grasp_episode_length}")
        print(f"  总决策数: {demo_grasp_action_dim * demo_grasp_episode_length}")
        
        print(f"复杂度减少: {complexity_reduction:.1f}x")
        
        return complexity_reduction