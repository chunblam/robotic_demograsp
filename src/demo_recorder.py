import torch
import numpy as np
import time
import math_utils as utils
from environment import DemoGraspEnvironment
from config import DemoGraspConfig

class DemoRecorder:
    def __init__(self, env, utils):
        """
        初始化演示录制器
        
        Args:
            env: DemoGraspEnvironment实例
        """
        self.env = env
        self.cfg = env.cfg
        self.utils = utils
        
    def record_demonstration(self, object_index=0, max_attempts=5):
        """
        录制一个成功的抓取演示轨迹
        
        Args:
            object_index: 要抓取的物体索引
            max_attempts: 最大尝试次数
            
        Returns:
            demo_trajectory: 录制的演示轨迹字典
        """
        print(f"开始录制演示轨迹，物体索引: {object_index}")
        
        # 重置环境到特定物体
        self._setup_demo_environment(object_index)
        
        # 尝试录制成功的演示
        for attempt in range(max_attempts):
            print(f"尝试 {attempt + 1}/{max_attempts}")
            
            # 录制轨迹
            demo_trajectory = self._record_trajectory_attempt()
            
            # 检查是否成功
            if self._check_demo_success(demo_trajectory):
                print("演示录制成功!")
                return demo_trajectory
            else:
                print("演示失败，重新尝试...")
                self._setup_demo_environment(object_index)  # 重新设置环境
        
        print("达到最大尝试次数，演示录制失败")
        return None
    
    def _setup_demo_environment(self, object_index):
        """
        设置演示环境 - 使用特定物体和固定位置
        
        论文原理：在物体坐标系下录制轨迹，便于后续的轨迹编辑
        """
        # 重置所有环境
        self.env.reset()
        
        # 对于演示，我们只使用第一个环境
        demo_env_id = 0
        
        # 设置物体到固定位置（便于录制）
        self._set_object_fixed_pose(demo_env_id, object_index)
        
        # 设置机器人到初始位置
        self._set_robot_initial_pose(demo_env_id)
        
        # 刷新环境状态
        self.env.gym.refresh_actor_root_state_tensor(self.env.sim)
        self.env.gym.refresh_dof_state_tensor(self.env.sim)
        
        print("演示环境设置完成")
    
    def _set_object_fixed_pose(self, env_id, object_index):
        """
        设置物体到固定位置（桌子中心）
        
        论文原理：在固定的物体坐标系下录制，便于后续的SE(3)变换
        """
        # 获取物体根状态
        obj_state_index = 3 + env_id * self.env.num_actors_per_env
        
        # 设置物体到桌子中心，稍微高于桌面
        fixed_pose = torch.zeros(13, device=self.env.device)
        fixed_pose[0:3] = torch.tensor([0.0, 0.0, self.env.table_dims.z + 0.02])  # 位置
        fixed_pose[3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])  # 无旋转的四元数
        fixed_pose[7:13] = 0.0  # 零速度
        
        # 应用物体状态
        self.env.root_state_tensor[env_id, 3] = fixed_pose
        
        # 更新到仿真中
        obj_indices = torch.tensor([obj_state_index], device=self.env.device, dtype=torch.int32)
        self.env.gym.set_actor_root_state_tensor_indexed(
            self.env.sim,
            self.env.gymtorch.unwrap_tensor(self.env.root_state_tensor),
            self.env.gymtorch.unwrap_tensor(obj_indices),
            1
        )
    
    def _set_robot_initial_pose(self, env_id):
        """
        设置机器人到初始姿态（张开手，准备接近）
        
        论文原理：初始状态应该是手部张开，准备接近物体
        """
        env_ptr = self.env.envs[env_id]
        robot_handle = self.env.robot_handles[env_id]
        
        # 设置初始关节位置
        initial_dof_pos = np.zeros(self.cfg.TOTAL_DOF, dtype=np.float32)
        
        # 机械臂初始位置 - 让末端执行器在物体上方
        initial_dof_pos[:self.cfg.ARM_DOF] = [0.0, -0.4, 0.0, -1.2, 0.0, 1.0, 0.0]
        
        # 手部完全张开
        initial_dof_pos[self.cfg.ARM_DOF:] = self._get_open_hand_pose()
        
        # 应用DOF状态
        self.env.gym.set_actor_dof_states(env_ptr, robot_handle, initial_dof_pos, gymapi.STATE_ALL)
        self.env.gym.set_actor_dof_position_targets(env_ptr, robot_handle, initial_dof_pos)
    
    def _get_open_hand_pose(self):
        """
        获取手部张开姿态
        
        论文原理：初始手部应该是张开的，便于接近物体
        """
        open_hand_pose = np.zeros(self.cfg.HAND_DOF)
        
        # 设置每个手指的关节位置为接近最小值（张开状态）
        for i in range(self.cfg.HAND_DOF):
            low, high = self.cfg.HAND_JOINT_LIMITS[i]
            # 设置为接近下限的位置（张开）
            open_hand_pose[i] = low + 0.1 * (high - low)
        
        return open_hand_pose
    
    def _get_closed_hand_pose(self, object_size=0.05):
        """
        获取手部闭合姿态（根据物体大小调整）
        
        论文原理：抓取姿态应该根据物体几何形状调整
        """
        closed_hand_pose = np.zeros(self.cfg.HAND_DOF)
        
        # 根据物体大小设置抓取力度
        grasp_strength = min(1.0, object_size / 0.1)  # 标准化到[0,1]
        
        for i in range(self.cfg.HAND_DOF):
            low, high = self.cfg.HAND_JOINT_LIMITS[i]
            # 设置为根据物体大小调整的闭合位置
            closed_hand_pose[i] = low + (0.3 + 0.5 * grasp_strength) * (high - low)
        
        return closed_hand_pose
    
    def _record_trajectory_attempt(self):
        """
        录制一次抓取尝试的完整轨迹
        
        论文原理：录制完整的抓取轨迹，包括接近、抓取、提升三个阶段
        """
        demo_env_id = 0  # 只在第一个环境录制演示
        trajectory = {
            'hand_actions': [],           # 手部关节角度序列
            'ee_poses_obj_frame': [],     # 在物体坐标系下的末端执行器位姿
            'timesteps': [],              # 时间步
            'object_pose_0': None,        # 初始物体位姿（用于建立物体坐标系）
            'lift_timestep': None         # 提升开始的时间步（论文中的T_lift）
        }
        
        # 获取初始状态
        initial_obs = self.env.get_observation()
        initial_obj_pose = initial_obs['obj_pose'][demo_env_id]  # 初始物体位姿
        trajectory['object_pose_0'] = initial_obj_pose.cpu().numpy()
        
        # 计算从世界坐标系到物体坐标系的变换矩阵
        T_world_to_obj = self._get_world_to_object_transform(initial_obj_pose)
        
        print("开始录制轨迹...")
        
        # 轨迹参数
        approach_steps = 20    # 接近阶段步数
        grasp_steps = 10       # 抓取阶段步数  
        lift_steps = 10        # 提升阶段步数
        total_steps = approach_steps + grasp_steps + lift_steps
        
        lift_timestep = approach_steps + grasp_steps  # 提升开始时间步
        trajectory['lift_timestep'] = lift_timestep
        
        for t in range(total_steps):
            # 根据当前阶段选择动作
            if t < approach_steps:
                # 阶段1：接近物体
                hand_action, ee_pose_obj = self._get_approach_action(t, approach_steps, T_world_to_obj)
            elif t < lift_timestep:
                # 阶段2：闭合手部抓取物体
                hand_action, ee_pose_obj = self._get_grasp_action(t - approach_steps, grasp_steps)
            else:
                # 阶段3：提升物体
                hand_action, ee_pose_obj = self._get_lift_action(t - lift_timestep, lift_steps)
            
            # 记录当前状态
            trajectory['hand_actions'].append(hand_action.copy())
            trajectory['ee_poses_obj_frame'].append(ee_pose_obj.copy())
            trajectory['timesteps'].append(t)
            
            # 执行动作（只在演示环境执行）
            self._execute_demo_action(demo_env_id, hand_action, ee_pose_obj, T_world_to_obj)
            
            # 小延迟以便观察
            time.sleep(0.05)
            
            # 每10步打印进度
            if t % 10 == 0:
                print(f"轨迹录制进度: {t}/{total_steps}")
        
        print("轨迹录制完成")
        return trajectory
    
    def _get_world_to_object_transform(self, obj_pose):
        """
        计算从世界坐标系到物体坐标系的变换矩阵
        
        论文原理：在物体坐标系下表示轨迹，便于后续的轨迹编辑
        """
        # obj_pose: [x, y, z, qx, qy, qz, qw]
        pos = obj_pose[:3].cpu().numpy()
        quat = obj_pose[3:].cpu().numpy()
        
        # 构建齐次变换矩阵
        T = np.eye(4)
        T[:3, :3] = self.utils.quaternion_to_rotation_matrix(quat)
        T[:3, 3] = pos
        
        # 返回逆变换（世界到物体）
        return np.linalg.inv(T)
      
    def _get_approach_action(self, t, total_steps, T_world_to_obj):
        """
        生成接近阶段的动作
        
        论文原理：末端执行器应该平滑地接近物体中心
        """
        # 手部保持张开
        hand_action = self._get_open_hand_pose()
        
        # 末端执行器从当前位置平滑移动到物体上方
        # 初始位置（当前末端位置）
        current_ee_pose = self.env._get_ee_pose()[0].cpu().numpy()  # 第一个环境
        start_pos = current_ee_pose[:3]
        
        # 目标位置：物体上方10cm
        target_pos_obj = np.array([0.0, 0.0, 0.1])  # 在物体坐标系中
        
        # 将目标位置转换到世界坐标系
        target_pos_world_homo = T_world_to_obj @ np.array([*target_pos_obj, 1.0])
        target_pos_world = target_pos_world_homo[:3]
        
        # 线性插值
        alpha = t / total_steps
        current_pos = start_pos * (1 - alpha) + target_pos_world * alpha
        
        # 保持朝向指向物体
        target_quat = np.array([0.0, 0.0, 0.0, 1.0])  # 无旋转
        
        # 构建末端执行器位姿（在物体坐标系中）
        ee_pose_world = np.concatenate([current_pos, target_quat])
        ee_pose_obj_homo = T_world_to_obj @ self.utils.pose_to_homogeneous(ee_pose_world)
        ee_pose_obj = self.utils.homogeneous_to_pose(ee_pose_obj_homo)
        
        return hand_action, ee_pose_obj
    
    def _get_grasp_action(self, t, total_steps):
        """
        生成抓取阶段的动作
        
        论文原理：手部应该从张开状态平滑过渡到闭合状态
        """
        # 手部从张开到闭合
        alpha = t / total_steps
        open_pose = self._get_open_hand_pose()
        closed_pose = self._get_closed_hand_pose()
        hand_action = open_pose * (1 - alpha) + closed_pose * alpha
        
        # 末端执行器保持在同一位置（在物体坐标系中）
        current_ee_pose_obj = np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0])  # 物体上方10cm
        
        return hand_action, current_ee_pose_obj
    
    def _get_lift_action(self, t, total_steps):
        """
        生成提升阶段的动作
        
        论文原理：抓取后垂直提升物体
        """
        # 手部保持闭合
        hand_action = self._get_closed_hand_pose()
        
        # 末端执行器垂直提升
        lift_height = 0.15  # 提升15cm
        current_height = 0.1 + (t / total_steps) * lift_height
        
        # 在物体坐标系中，只是z坐标增加
        ee_pose_obj = np.array([0.0, 0.0, current_height, 0.0, 0.0, 0.0, 1.0])
        
        return hand_action, ee_pose_obj
    
    def _execute_demo_action(self, env_id, hand_action, ee_pose_obj, T_world_to_obj):
        """
        在演示环境中执行动作 - 使用逆运动学确保手腕达到目标位置
        """
        self.env._execute_demo_action_with_ik(env_id, hand_action, ee_pose_obj, T_world_to_obj)
    
    def _check_demo_success(self, trajectory):
        """
        检查演示是否成功
        
        论文原理：成功的演示应该将物体提升到足够高度并保持抓取
        """
        # 检查最终状态
        final_obs = self.env.get_observation()
        obj_pose = final_obs['obj_pose'][0]  # 第一个环境
        ee_pose = final_obs['ee_pose'][0]
        
        # 检查物体是否被抬起
        lifted = obj_pose[2].cpu().numpy() > self.env.table_dims.z + self.cfg.LIFT_HEIGHT
        
        # 检查手和物体的距离
        hand_obj_dist = np.linalg.norm(ee_pose[:3].cpu().numpy() - obj_pose[:3].cpu().numpy())
        holding = hand_obj_dist < self.cfg.HOLD_DISTANCE
        
        success = lifted and holding
        
        print(f"演示结果 - 抬起: {lifted}, 抓持: {holding}, 成功: {success}")
        
        return success
    
    def save_demonstration(self, trajectory, filename="../output/demo_trajectory.npy"):
        """保存演示轨迹到文件"""
        # 转换为numpy数组以便保存
        demo_data = {
            'hand_actions': np.array(trajectory['hand_actions']),
            'ee_poses_obj_frame': np.array(trajectory['ee_poses_obj_frame']),
            'object_pose_0': trajectory['object_pose_0'],
            'lift_timestep': trajectory['lift_timestep'],
            'timesteps': np.array(trajectory['timesteps'])
        }
        
        np.save(filename, demo_data)
        print(f"演示轨迹已保存到: {filename}")
    
    def load_demonstration(self, filename="demo_trajectory.npy"):
        """从文件加载演示轨迹"""
        demo_data = np.load(filename, allow_pickle=True).item()
        
        trajectory = {
            'hand_actions': demo_data['hand_actions'].tolist(),
            'ee_poses_obj_frame': demo_data['ee_poses_obj_frame'].tolist(),
            'object_pose_0': demo_data['object_pose_0'],
            'lift_timestep': demo_data['lift_timestep'],
            'timesteps': demo_data['timesteps'].tolist()
        }
        
        print(f"演示轨迹已从 {filename} 加载")
        return trajectory