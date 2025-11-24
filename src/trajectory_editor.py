import torch
import numpy as np
from config import DemoGraspConfig

class TrajectoryEditor:
    def __init__(self, demo_trajectory, cfg):
        """
        初始化轨迹编辑器
        
        Args:
            demo_trajectory: 录制的演示轨迹
            cfg: 配置参数
        """
        self.demo = demo_trajectory
        self.cfg = cfg
        self.lift_timestep = demo_trajectory['lift_timestep']
        
        print(f"轨迹编辑器初始化完成，提升时间步: {self.lift_timestep}")
    
    def edit_trajectory(self, T_ee, delta_qG):
        """
        编辑演示轨迹 - 核心函数
        
        论文原理：通过SE(3)变换修改末端轨迹，通过关节增量修改手部姿态
        
        Args:
            T_ee: SE(3)变换矩阵 [4x4]，定义在哪里抓取
            delta_qG: 手部关节角度增量 [16]，定义如何抓取
            
        Returns:
            edited_demo: 编辑后的轨迹
        """
        print("开始编辑轨迹...")
        
        edited_demo = {
            'hand_actions': [],
            'ee_poses_obj_frame': [],
            'timesteps': self.demo['timesteps'].copy(),
            'object_pose_0': self.demo['object_pose_0'].copy(),
            'lift_timestep': self.lift_timestep,
            'edit_parameters': {
                'T_ee': T_ee.copy(),
                'delta_qG': delta_qG.copy()
            }
        }
        
        # 编辑末端执行器轨迹
        print("编辑末端执行器轨迹...")
        edited_ee_trajectory = self._edit_ee_trajectory(T_ee)
        
        # 编辑手部关节轨迹
        print("编辑手部关节轨迹...")
        edited_hand_trajectory = self._edit_hand_trajectory(delta_qG)
        
        # 组合编辑后的轨迹
        for t in range(len(self.demo['ee_poses_obj_frame'])):
            edited_demo['hand_actions'].append(edited_hand_trajectory[t])
            edited_demo['ee_poses_obj_frame'].append(edited_ee_trajectory[t])
        
        print("轨迹编辑完成")
        return edited_demo
    
    def _edit_ee_trajectory(self, T_ee):
        """
        编辑末端执行器轨迹 - 实现论文公式1
        
        论文原理：
        - t ≤ T_lift: 应用相同的相对变换接近目标
        - t > T_lift: 纯粹垂直提升
        
        Args:
            T_ee: SE(3)变换矩阵
        """
        edited_trajectory = []
        
        for t, ee_pose_obj in enumerate(self.demo['ee_poses_obj_frame']):
            if t <= self.lift_timestep:
                # 阶段1：应用SE(3)变换修改接近路径
                # 论文公式: p'_t_ee-obj = T_ee * p*_t_ee-obj
                edited_pose = self._apply_se3_transform(ee_pose_obj, T_ee)
            else:
                # 阶段2：纯粹垂直提升
                # 论文公式: p'_t_ee-obj = [I Δz] * p'_T_lift_ee-obj
                lift_pose = edited_trajectory[self.lift_timestep].copy()
                
                # 计算从提升时刻到当前时刻的高度增量
                original_lift_z = self.demo['ee_poses_obj_frame'][self.lift_timestep][2]
                current_z = ee_pose_obj[2]
                delta_z = current_z - original_lift_z
                
                # 应用垂直提升
                lift_pose[2] += delta_z
                edited_pose = lift_pose
            
            edited_trajectory.append(edited_pose)
        
        return edited_trajectory
    
    def _apply_se3_transform(self, pose, T):
        """
        应用SE(3)变换到位姿
        
        数学原理：
        齐次坐标变换: P' = T * P
        其中 P = [x, y, z, 1]^T 是齐次坐标
        
        Args:
            pose: [x, y, z, qx, qy, qz, qw] 位姿
            T: [4x4] SE(3)变换矩阵
        """
        # 提取位置和四元数
        pos = pose[:3]
        quat = pose[3:]
        
        # 将位置转换为齐次坐标
        pos_homo = np.array([pos[0], pos[1], pos[2], 1.0])
        
        # 应用变换: P' = T * P
        transformed_pos_homo = T @ pos_homo
        transformed_pos = transformed_pos_homo[:3]
        
        # 对于旋转部分，变换矩阵的旋转部分作用于四元数
        # R' = R_T * R_original
        R_original = self.quaternion_to_rotation_matrix(quat)
        R_T = T[:3, :3]
        R_transformed = R_T @ R_original
        
        # 转换回四元数
        transformed_quat = self.rotation_matrix_to_quaternion(R_transformed)
        
        return np.concatenate([transformed_pos, transformed_quat])
    
    def _edit_hand_trajectory(self, delta_qG):
        """
        编辑手部关节轨迹 - 实现论文公式2
        
        论文原理：
        - t ≤ T_lift: 从初始状态到修改后的抓取姿态的线性插值
        - t > T_lift: 保持修改后的抓取姿态
        
        Args:
            delta_qG: 手部关节角度增量
        """
        edited_trajectory = []
        
        # 获取关键手部姿态
        q0 = np.array(self.demo['hand_actions'][0])           # 初始手部姿态
        qT_lift = np.array(self.demo['hand_actions'][self.lift_timestep])  # 原始抓取姿态
        
        # 修改后的抓取姿态
        qT_lift_modified = qT_lift + delta_qG
        
        # 应用关节限制
        qT_lift_modified = self._apply_joint_limits(qT_lift_modified)
        
        print(f"手部姿态编辑: 原始抓取姿态 {qT_lift[:4]} -> 修改后 {qT_lift_modified[:4]}")
        
        for t, hand_action in enumerate(self.demo['hand_actions']):
            if t <= self.lift_timestep:
                # 阶段1：线性插值到修改后的抓取姿态
                # 论文公式的逐元素实现
                current_q = np.array(hand_action)
                
                # 计算当前时刻相对于初始时刻的变化量
                delta_current = current_q - q0
                
                # 计算缩放因子：(新变化量)/(原始变化量)
                # 论文中的 (q*_T_lift_hand + Δq^G - q*_0_hand) / (q*_T_lift_hand - q*_0_hand)
                scale_factors = np.divide(
                    qT_lift_modified - q0,
                    qT_lift - q0,
                    out=np.ones_like(q0),  # 除零时输出1
                    where=(qT_lift - q0) != 0
                )
                
                # 应用缩放：q*_t_hand = q*_0_hand + (q*_t_hand - q*_0_hand) * scale_factors
                edited_action = q0 + delta_current * scale_factors
                
            else:
                # 阶段2：保持修改后的抓取姿态
                edited_action = qT_lift_modified.copy()
            
            # 应用关节限制
            edited_action = self._apply_joint_limits(edited_action)
            edited_trajectory.append(edited_action.tolist())
        
        return edited_trajectory
    
    def _apply_joint_limits(self, joint_angles):
        """应用关节角度限制"""
        limited_angles = joint_angles.copy()
        
        # 应用机械臂关节限制
        for i in range(self.cfg.ARM_DOF):
            low, high = self.cfg.ARM_JOINT_LIMITS[i]
            limited_angles[i] = np.clip(limited_angles[i], low, high)
        
        # 应用手部关节限制
        for i in range(self.cfg.HAND_DOF):
            low, high = self.cfg.HAND_JOINT_LIMITS[i]
            limited_angles[self.cfg.ARM_DOF + i] = np.clip(
                limited_angles[self.cfg.ARM_DOF + i], low, high
            )
        
        return limited_angles
    
    def quaternion_to_rotation_matrix(self, quat):
        """将四元数转换为旋转矩阵"""
        qx, qy, qz, qw = quat
        
        # 单位化四元数
        norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
        
        # 计算旋转矩阵
        R = np.array([
            [1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
            [2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)],
            [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2)]
        ])
        
        return R
    
    def rotation_matrix_to_quaternion(self, R):
        """将旋转矩阵转换为四元数"""
        # 更稳定的转换方法
        trace = R[0,0] + R[1,1] + R[2,2]
        
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2,1] - R[1,2]) / S
            qy = (R[0,2] - R[2,0]) / S 
            qz = (R[1,0] - R[0,1]) / S
        elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            qw = (R[2,1] - R[1,2]) / S
            qx = 0.25 * S
            qy = (R[0,1] + R[1,0]) / S
            qz = (R[0,2] + R[2,0]) / S
        elif R[1,1] > R[2,2]:
            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
            qw = (R[0,2] - R[2,0]) / S
            qx = (R[0,1] + R[1,0]) / S
            qy = 0.25 * S
            qz = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
            qw = (R[1,0] - R[0,1]) / S
            qx = (R[0,2] + R[2,0]) / S
            qy = (R[1,2] + R[2,1]) / S
            qz = 0.25 * S
        
        quat = np.array([qx, qy, qz, qw])
        return quat / np.linalg.norm(quat)
    
    def create_random_edit_parameters(self):
        """
        创建随机的轨迹编辑参数
        
        论文原理：RL策略输出这些参数来适应不同的物体和姿态
        """
        # 末端执行器变换参数
        T_ee = self._create_random_se3_transform()
        
        # 手部关节增量参数
        delta_qG = self._create_random_hand_delta()
        
        return T_ee, delta_qG
    
    def _create_random_se3_transform(self):
        """创建随机的SE(3)变换矩阵"""
        T = np.eye(4)
        
        # 随机平移 (-5cm 到 +5cm)
        translation = np.random.uniform(-0.05, 0.05, 3)
        T[0:3, 3] = translation
        
        # 随机旋转 (小角度)
        # 使用欧拉角创建小旋转
        euler_angles = np.random.uniform(-0.3, 0.3, 3)  # 约±17度
        R = self._euler_to_rotation_matrix(euler_angles)
        T[0:3, 0:3] = R
        
        return T
    
    def _euler_to_rotation_matrix(self, euler_angles):
        """将欧拉角转换为旋转矩阵"""
        roll, pitch, yaw = euler_angles
        
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
        return Rz @ Ry @ Rx
    
    def _create_random_hand_delta(self):
        """创建随机的手部关节增量"""
        # 小范围的关节角度变化
        delta = np.random.uniform(-0.2, 0.2, self.cfg.HAND_DOF)
        return delta
    
    def visualize_edit_effect(self, original_demo, edited_demo, timestep=0):
        """可视化编辑效果"""
        print(f"\n=== 轨迹编辑效果对比 (时间步 {timestep}) ===")
        
        orig_ee = original_demo['ee_poses_obj_frame'][timestep]
        edit_ee = edited_demo['ee_poses_obj_frame'][timestep]
        
        orig_hand = original_demo['hand_actions'][timestep]
        edit_hand = edited_demo['hand_actions'][timestep]
        
        print("末端执行器位姿:")
        print(f"  原始: pos({orig_ee[0]:.3f}, {orig_ee[1]:.3f}, {orig_ee[2]:.3f})")
        print(f"  编辑: pos({edit_ee[0]:.3f}, {edit_ee[1]:.3f}, {edit_ee[2]:.3f})")
        
        print("手部关节角度 (前4个):")
        print(f"  原始: {np.array(orig_hand[:4])}")
        print(f"  编辑: {np.array(edit_hand[:4])}")
        
        ee_pos_change = np.linalg.norm(np.array(edit_ee[:3]) - np.array(orig_ee[:3]))
        hand_change = np.linalg.norm(np.array(edit_hand) - np.array(orig_hand))
        
        print(f"末端位置变化: {ee_pos_change:.3f} m")
        print(f"手部姿态变化: {hand_change:.3f} rad")