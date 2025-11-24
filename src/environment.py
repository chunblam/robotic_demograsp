import isaacgym
import isaacgym.torch_utils as torch_utils
from isaacgym import gymapi, gymtorch, gymutil
import torch
import numpy as np
import os
import glob
import math_utils as utils
from config import DemoGraspConfig

class DemoGraspEnvironment:
    def __init__(self, cfg, utils):
        self.cfg = cfg
        self.device = cfg.SIM_DEVICE
        self.utils = utils
        
        print("Initializing DemoGrasp Environment...")
        
        # Initialize gym
        self.gym = gymapi.acquire_gym()
        self.sim = self._create_sim()
        
        # Create ground plane
        self._create_ground()
        
        # Create table
        self.table_dims = gymapi.Vec3(*cfg.TABLE_DIMS)
        self.table_pose = gymapi.Transform()
        self.table_pose.p = gymapi.Vec3(*cfg.TABLE_POSITION)
        
        # Load assets
        print("Loading robot asset...")
        self.robot_asset = self._load_robot()
        
        print("Loading table asset...")
        self.table_asset = self._create_table_asset()
        
        print("Loading object assets...")
        self.object_assets = self._load_object_assets()
        
        if not self.object_assets:
            raise Exception("No object assets loaded!")
        
        print(f"Loaded {len(self.object_assets)} object assets")
        
        # Set up environments
        self.envs = []
        self.robot_handles = []
        self.object_handles = []
        self._create_environments()
        
        # 设置逆运动学求解器
        self._setup_ik_solver()
        print(f"逆运动学: {'启用' if cfg.USE_INVERSE_KINEMATICS else '禁用'}")
        
        # Initialize buffers for efficient GPU access
        self._init_buffers()
        
        print("Environment initialization completed!")
    
    def _create_sim(self):
        """创建物理仿真实例"""
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / self.cfg.LOW_LEVEL_FREQ
        sim_params.substeps = self.cfg.SIM_SUBSTEPS
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.01
        sim_params.physx.rest_offset = 0.0
        
        # 使用GPU进行物理计算
        sim_params.use_gpu_pipeline = True
        sim_params.physx.use_gpu = True
        
        sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        
        if sim is None:
            raise Exception("Failed to create sim")
            
        return sim
    
    def _create_ground(self):
        """创建地面平面"""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        plane_params.distance = 0
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        plane_params.restitution = 0.0
        self.gym.add_ground(self.sim, plane_params)
    
    def _load_robot(self):
        """加载Kuka Allegro机器人"""
        robot_asset_options = gymapi.AssetOptions()
        robot_asset_options.fix_base_link = True
        robot_asset_options.flip_visual_attachments = False
        robot_asset_options.collapse_fixed_joints = False
        robot_asset_options.disable_gravity = False
        robot_asset_options.use_mesh_materials = True
        
        # 设置默认的关节驱动模式
        robot_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        
        print(f"Loading robot from: {self.cfg.ROBOT_URDF_PATH}")
        
        if not os.path.exists(self.cfg.ROBOT_URDF_PATH):
            raise FileNotFoundError(f"Robot URDF not found at: {self.cfg.ROBOT_URDF_PATH}")
        
        robot_asset = self.gym.load_asset(self.sim, "", self.cfg.ROBOT_URDF_PATH, robot_asset_options)
        
        # 配置机器人关节属性
        self._configure_robot_dof_props(robot_asset)
        
        return robot_asset
    
    def _configure_robot_dof_props(self, robot_asset):
        """配置机器人关节属性"""
        dof_props = self.gym.get_asset_dof_properties(robot_asset)
        
        # 设置机械臂关节属性
        for i in range(self.cfg.ARM_DOF):
            dof_props['stiffness'][i] = 1000.0  # 降低刚度便于控制
            dof_props['damping'][i] = 100.0
            dof_props['effort'][i] = 300.0
            dof_props['velocity'][i] = 2.0  # 降低速度限制
            
        # 设置灵巧手关节属性  
        for i in range(self.cfg.ARM_DOF, self.cfg.TOTAL_DOF):
            dof_props['stiffness'][i] = 200.0
            dof_props['damping'][i] = 10.0
            dof_props['effort'][i] = 0.35  # Allegro手的力矩限制
            dof_props['velocity'][i] = 2.0
            
        self.gym.set_asset_dof_properties(robot_asset, dof_props)
    
    def _create_table_asset(self):
        """创建桌子资产"""
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, 
                                        self.table_dims.x, 
                                        self.table_dims.y, 
                                        self.table_dims.z, 
                                        table_asset_options)
        return table_asset
    
    def _load_object_assets(self):
        """加载YCB物体资产"""
        object_assets = []
        
        # 检查YCB目录是否存在
        # if not os.path.exists(self.cfg.OBJECT_ASSET_DIR):
        #     print(f"Warning: YCB directory not found at {self.cfg.OBJECT_ASSET_DIR}")
        #     # 创建一些基本几何形状作为备选
        #     return self._create_basic_objects()
        
        # 加载预定义的物体
        for obj_path in self.cfg.OBJECT_URDF_PATHS:
            full_path = os.path.join(self.cfg.OBJECT_ASSET_DIR, obj_path)
            if os.path.exists(full_path):
                try:
                    asset_options = gymapi.AssetOptions()
                    asset_options.disable_gravity = False
                    asset_options.use_mesh_materials = True
                    
                    obj_asset = self.gym.load_asset(self.sim, "", full_path, asset_options)
                    object_assets.append(obj_asset)
                    print(f"Loaded: {obj_path}")
                except Exception as e:
                    print(f"Failed to load {obj_path}: {e}")
            else:
                print(f"Object not found: {full_path}")
        
        # 如果没加载到任何物体，创建基本几何形状
        if not object_assets:
            print("No YCB objects found, creating basic objects...")
            object_assets = self._create_basic_objects()
        
        return object_assets
    
    def _create_basic_objects(self):
        """创建基本几何形状作为备选"""
        object_assets = []
        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = False
        
        # 创建立方体
        box_asset = self.gym.create_box(self.sim, 0.05, 0.05, 0.05, asset_options)
        object_assets.append(box_asset)
        
        # 创建球体
        sphere_asset = self.gym.create_sphere(self.sim, 0.04, asset_options)
        object_assets.append(sphere_asset)
        
        # 创建圆柱体
        cylinder_asset = self.gym.create_cylinder(self.sim, 0.03, 0.06, asset_options)
        object_assets.append(cylinder_asset)
        
        return object_assets
    
    def _create_environments(self):
        """创建多个并行环境"""
        env_spacing = 2.0  # 环境间距
        env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        
        print(f"Creating {self.cfg.NUM_ENVS} environments...")
        
        for i in range(self.cfg.NUM_ENVS):
            # 创建环境
            env = self.gym.create_env(self.sim, env_lower, env_upper, 1)
            self.envs.append(env)
            
            # 添加桌子
            table_handle = self.gym.create_actor(env, self.table_asset, self.table_pose, "table", i, 0)
            
            # 添加机器人
            robot_pose = gymapi.Transform()
            robot_pose.p = gymapi.Vec3(0.0, 0.0, 0.15)  # 高于桌子
            robot_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            
            robot_handle = self.gym.create_actor(env, self.robot_asset, robot_pose, "robot", i, 0)
            self.robot_handles.append(robot_handle)
            
            # 设置机器人初始姿态
            self._set_robot_initial_pose(env, robot_handle)
            
            # 添加物体 - 随机选择物体类型
            obj_idx = i % len(self.object_assets)
            obj_asset = self.object_assets[obj_idx]
            obj_pose = self._get_random_object_pose()
            
            object_handle = self.gym.create_actor(env, obj_asset, obj_pose, f"object_{i}", i, 0)
            self.object_handles.append(object_handle)
            
            # 设置物体颜色以便区分
            color = gymapi.Vec3(np.random.random(), np.random.random(), np.random.random())
            self.gym.set_rigid_body_color(env, object_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
    
    def _set_robot_initial_pose(self, env, robot_handle):
        """设置机器人初始姿态"""
        # 设置初始关节位置
        dof_states = np.zeros(self.cfg.TOTAL_DOF, dtype=np.float32)
        
        # 设置机械臂初始位置
        dof_states[:self.cfg.ARM_DOF] = self.cfg.INITIAL_ARM_POSE
        
        # 设置手部初始位置 (张开)
        dof_states[self.cfg.ARM_DOF:] = self.cfg.INITIAL_HAND_POSE
        
        # 应用DOF状态
        self.gym.set_actor_dof_states(env, robot_handle, dof_states, gymapi.STATE_ALL)
        
        # 设置DOF目标位置
        self.gym.set_actor_dof_position_targets(env, robot_handle, dof_states)
    
    def _get_random_object_pose(self):
        """生成随机的物体初始位姿"""
        pose = gymapi.Transform()
        
        # 在50cm x 50cm区域内随机位置
        x = np.random.uniform(-self.cfg.RESET_REGION[0]/2, self.cfg.RESET_REGION[0]/2)
        y = np.random.uniform(-self.cfg.RESET_REGION[1]/2, self.cfg.RESET_REGION[1]/2)
        z = self.table_dims.z + 0.02  # 稍微高于桌子
        
        pose.p = gymapi.Vec3(x, y, z)
        
        # 随机旋转
        random_rotation = gymapi.Quat.from_euler_zyx(
            np.random.uniform(0, 2*np.pi),  # yaw
            np.random.uniform(-0.5, 0.5),   # pitch (限制范围避免物体倒下)
            np.random.uniform(0, 2*np.pi)    # roll
        )
        pose.r = random_rotation
        
        return pose
    
    def _init_buffers(self):
        """初始化GPU张量缓冲区"""
        # 获取状态张量
        self._root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self._dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self._rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        
        # 包装为PyTorch张量
        self.root_state_tensor = gymtorch.wrap_tensor(self._root_state_tensor)
        self.dof_state_tensor = gymtorch.wrap_tensor(self._dof_state_tensor)
        self.rigid_body_state_tensor = gymtorch.wrap_tensor(self._rigid_body_state_tensor)
        
        # 计算每个环境的actor数量 (地面 + 桌子 + 机器人 + 物体 = 4)
        self.num_actors_per_env = 4
        self.root_state_tensor = self.root_state_tensor.view(self.cfg.NUM_ENVS, self.num_actors_per_env, 13)
        
        # 找到末端执行器索引 (使用机械臂最后一个链接)
        self.ee_body_index = self._find_end_effector_index()
        
        print("GPU buffers initialized successfully")
    
    def _setup_ik_solver(self):
        """设置逆运动学求解器"""
        if not self.cfg.USE_INVERSE_KINEMATICS:
            return None
        
        # 获取末端执行器索引
        self.ee_body_index = self._find_end_effector_index()
        if self.ee_body_index == -1:
            print("❌ 无法找到末端执行器链接，禁用IK")
            return None
        
        print(f"✓ IK求解器配置完成，末端执行器索引: {self.ee_body_index}")
        return self.ee_body_index

    def _find_end_effector_index(self):
        """查找末端执行器刚体索引"""
        env_ptr = self.envs[0]
        robot_handle = self.robot_handles[0]
        
        ee_index = self.gym.find_actor_rigid_body_index(
            env_ptr, robot_handle, self.cfg.EE_LINK_NAME, gymapi.DOMAIN_ENV
        )
        
        if ee_index == -1:
            print(f"❌ 找不到末端执行器链接: {self.cfg.EE_LINK_NAME}")
            # 尝试备选方案
            alternatives = ["iiwa7_link_7", "allegro_mount", "palm_link"]
            for alt_name in alternatives:
                ee_index = self.gym.find_actor_rigid_body_index(
                    env_ptr, robot_handle, alt_name, gymapi.DOMAIN_ENV
                )
                if ee_index != -1:
                    print(f"✓ 使用备选末端执行器: {alt_name}")
                    break
        
        return ee_index

    def _calculate_ik(self, env_ptr, robot_handle, target_ee_pose):
        """使用IsaacGym内置IK求解器计算逆运动学"""
        if not hasattr(self, 'ee_body_index') or self.ee_body_index == -1:
            # 回退到当前关节位置
            current_dof_states = self.gym.get_actor_dof_states(env_ptr, robot_handle, gymapi.STATE_ALL)
            return current_dof_states['pos'][:self.cfg.ARM_DOF]
        
        # 创建目标变换
        target_transform = gymapi.Transform()
        target_transform.p = gymapi.Vec3(target_ee_pose[0], target_ee_pose[1], target_ee_pose[2])
        target_transform.r = gymapi.Quat(target_ee_pose[3], target_ee_pose[4], target_ee_pose[5], target_ee_pose[6])
        
        try:
            # 调用内置IK求解器
            dof_positions = self.gym.calculate_inverse_kinematics(
                env_ptr, robot_handle, self.ee_body_index, target_transform
            )
            
            # 应用关节限制
            for i in range(self.cfg.ARM_DOF):
                low, high = self.cfg.ARM_JOINT_LIMITS[i]
                dof_positions[i] = np.clip(dof_positions[i], low, high)
            
            return dof_positions[:self.cfg.ARM_DOF]
            
        except Exception as e:
            print(f"❌ IK求解失败: {e}")
            # 回退到当前关节位置
            current_dof_states = self.gym.get_actor_dof_states(env_ptr, robot_handle, gymapi.STATE_ALL)
            return current_dof_states['pos'][:self.cfg.ARM_DOF]
        
    def _execute_demo_action_with_ik(self, env_id, hand_action, ee_pose_obj, T_world_to_obj):
        """
        执行动作的公共接口 - 供demo_recorder和single_step_mdp调用
        """
        env_ptr = self.envs[env_id]
        robot_handle = self.robot_handles[env_id]
        
        # 1. 将物体坐标系下的末端位姿转换到世界坐标系
        ee_pose_world_homo = np.linalg.inv(T_world_to_obj) @ self.utils.pose_to_homogeneous(ee_pose_obj)
        ee_pose_world = self.utils.homogeneous_to_pose(ee_pose_world_homo)
        
        # 2. 设置手部关节（直接控制）
        target_dof_pos = np.zeros(self.cfg.TOTAL_DOF)
        target_dof_pos[self.cfg.ARM_DOF:] = hand_action
        
        # 3. 使用逆运动学计算机械臂关节角度
        if self.cfg.USE_INVERSE_KINEMATICS:
            arm_joint_positions = self._calculate_ik(env_ptr, robot_handle, ee_pose_world)
            target_dof_pos[:self.cfg.ARM_DOF] = arm_joint_positions
        else:
            # 回退到简化方法
            current_dof_states = self.gym.get_actor_dof_states(env_ptr, robot_handle, gymapi.STATE_ALL)
            target_dof_pos[:self.cfg.ARM_DOF] = current_dof_states['pos'][:self.cfg.ARM_DOF]
        
        # 4. 应用目标位置并执行仿真
        self.gym.set_actor_dof_position_targets(env_ptr, robot_handle, target_dof_pos)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        
        # 5. 刷新状态
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
            
    def reset(self, env_ids=None):
        """重置环境"""
        if env_ids is None:
            env_ids = torch.arange(self.cfg.NUM_ENVS, device=self.device)
        
        # 重置物体位置
        self._reset_objects(env_ids)
        
        # 重置机器人位置
        self._reset_robot(env_ids)
        
        # 刷新张量
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        return self.get_observation()
    
    def _reset_objects(self, env_ids):
        """重置物体位置"""
        num_envs = len(env_ids)
        
        # 物体是每个环境的第4个actor (索引3)
        object_indices = 3 + env_ids * self.num_actors_per_env
        
        # 随机位置
        random_positions = torch.zeros((num_envs, 3), device=self.device)
        random_positions[:, 0] = torch.rand(num_envs, device=self.device) * self.cfg.RESET_REGION[0] - self.cfg.RESET_REGION[0]/2
        random_positions[:, 1] = torch.rand(num_envs, device=self.device) * self.cfg.RESET_REGION[1] - self.cfg.RESET_REGION[1]/2
        random_positions[:, 2] = self.table_dims.z + 0.02
        
        # 随机旋转
        random_rotations = torch.rand((num_envs, 4), device=self.device) - 0.5
        random_rotations = random_rotations / torch.norm(random_rotations, dim=1, keepdim=True)
        
        # 设置物体状态
        self.root_state_tensor[env_ids, 3, 0:3] = random_positions
        self.root_state_tensor[env_ids, 3, 3:7] = random_rotations
        self.root_state_tensor[env_ids, 3, 7:13] = 0.0  # 重置速度和角速度
        
        # 应用状态
        self.gym.set_actor_root_state_tensor_indexed(self.sim, 
                                                   gymtorch.unwrap_tensor(self.root_state_tensor),
                                                   gymtorch.unwrap_tensor(object_indices), 
                                                   len(env_ids))
    
    def _reset_robot(self, env_ids):
        """重置机器人位置"""
        # 设置到初始关节位置
        initial_dof_pos = torch.zeros((len(env_ids), self.cfg.TOTAL_DOF), device=self.device)
        
        # 设置机械臂初始位置
        initial_dof_pos[:, :self.cfg.ARM_DOF] = torch.tensor(self.cfg.INITIAL_ARM_POSE, device=self.device)
        
        # 设置手部为张开状态
        initial_dof_pos[:, self.cfg.ARM_DOF:] = torch.tensor(self.cfg.INITIAL_HAND_POSE, device=self.device)
        
        # 应用DOF状态
        for i, env_id in enumerate(env_ids):
            env_ptr = self.envs[env_id]
            robot_handle = self.robot_handles[env_id]
            self.gym.set_actor_dof_states(env_ptr, robot_handle, initial_dof_pos[i].cpu().numpy(), gymapi.STATE_POS)
            self.gym.set_actor_dof_position_targets(env_ptr, robot_handle, initial_dof_pos[i].cpu().numpy())
    
    def step(self, actions):
        """执行环境步进 - 直接控制关节位置"""
        # 将动作转换为关节位置
        target_positions = self._actions_to_joint_positions(actions)
        
        # 应用目标位置到所有环境
        for i in range(self.cfg.NUM_ENVS):
            env_ptr = self.envs[i]
            robot_handle = self.robot_handles[i]
            self.gym.set_actor_dof_position_targets(env_ptr, robot_handle, target_positions[i].cpu().numpy())
        
        # 执行仿真
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        
        # 刷新张量
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        # 计算奖励和完成标志
        rewards = self.compute_reward()
        dones = self._check_done()
        
        # 获取新观测
        observations = self.get_observation()
        
        return observations, rewards, dones, {}
    
    def _actions_to_joint_positions(self, actions):
        """将动作转换为关节位置"""
        # 动作已经是关节位置的增量，直接加到当前位置上
        current_positions = self.dof_state_tensor.view(self.cfg.NUM_ENVS, -1, 2)[:, :, 0]  # 位置是第一个元素
        
        # 限制动作范围
        actions = torch.clamp(actions, -0.1, 0.1)  # 限制每次变化量
        
        # 计算新位置
        new_positions = current_positions + actions
        
        # 应用关节限制
        for i in range(self.cfg.ARM_DOF):
            low, high = self.cfg.ARM_JOINT_LIMITS[i]
            new_positions[:, i] = torch.clamp(new_positions[:, i], low, high)
            
        for i in range(self.cfg.HAND_DOF):
            low, high = self.cfg.HAND_JOINT_LIMITS[i]
            new_positions[:, self.cfg.ARM_DOF + i] = torch.clamp(new_positions[:, self.cfg.ARM_DOF + i], low, high)
        
        return new_positions
    
    def get_observation(self):
        """获取环境观测"""
        # 刷新张量以确保数据最新
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        
        observations = {
            'ee_pose': self._get_ee_pose(),
            'obj_pose': self._get_obj_pose(),
            'hand_pose': self._get_hand_pose(),
            'point_cloud': self._get_point_cloud()
        }
        
        return observations
    
    def _get_ee_pose(self):
        """获取末端执行器位姿"""
        ee_states = self.rigid_body_state_tensor.view(self.cfg.NUM_ENVS, -1, 13)[:, self.ee_body_index]
        
        ee_poses = torch.zeros((self.cfg.NUM_ENVS, 7), device=self.device)
        ee_poses[:, 0:3] = ee_states[:, 0:3]  # 位置
        ee_poses[:, 3:7] = ee_states[:, 3:7]  # 四元数旋转
        
        return ee_poses
    
    def _get_obj_pose(self):
        """获取物体位姿"""
        obj_states = self.root_state_tensor[:, 3]  # 物体是第四个actor
        
        obj_poses = torch.zeros((self.cfg.NUM_ENVS, 7), device=self.device)
        obj_poses[:, 0:3] = obj_states[:, 0:3]  # 位置
        obj_poses[:, 3:7] = obj_states[:, 3:7]  # 四元数旋转
        
        return obj_poses
    
    def _get_hand_pose(self):
        """获取手部关节角度"""
        dof_positions = self.dof_state_tensor.view(self.cfg.NUM_ENVS, -1, 2)[:, :, 0]
        hand_poses = dof_positions[:, self.cfg.ARM_DOF:]  # 只取手部关节
        
        return hand_poses
    
    def _get_point_cloud(self):
        """获取物体点云 - 简化版本，在物体周围生成点"""
        obj_poses = self._get_obj_pose()
        batch_size = obj_poses.shape[0]
        
        # 在单位球体内生成随机点
        points = torch.randn((batch_size, self.cfg.POINT_CLOUD_SIZE, 3), device=self.device)
        points = points / torch.norm(points, dim=2, keepdim=True)
        
        # 缩放到合适大小并移到物体位置
        points = points * 0.05  # 5cm半径
        points = points + obj_poses[:, None, 0:3]  # 移到物体位置
        
        return points
    
    def compute_reward(self):
        """计算奖励"""
        success = self._check_success()
        collision = self._check_collision()
        
        rewards = torch.zeros(self.cfg.NUM_ENVS, device=self.device)
        
        # 根据论文的奖励设计
        rewards[success & ~collision] = self.cfg.SUCCESS_REWARD
        rewards[success & collision] = self.cfg.COLLISION_PENALTY
        rewards[~success] = self.cfg.FAILURE_REWARD
        
        return rewards
    
    def _check_success(self):
        """按照论文方法检测抓取是否成功"""
        obj_poses = self._get_obj_pose()
        
        # 条件1: 检查物体是否被抬起至少10cm（论文标准）
        lifted = self._check_object_lifted(obj_poses)
        # 条件2: 检查手部关键点与物体的平均距离是否小于12cm（论文标准）
        holding = self._check_hand_object_proximity(obj_poses)
        # 必须同时满足两个条件
        success = lifted & holding
        
        return success

    def _check_object_lifted(self, obj_poses):
        """检查物体是否被抬起至少10cm（论文高度条件）"""
        # 记录初始高度（在第一次检测时）
        if not hasattr(self, 'initial_object_heights'):
            self.initial_object_heights = obj_poses[:, 2].clone()
        
        # 计算抬起高度：当前高度 - 初始高度
        lift_height = obj_poses[:, 2] - self.initial_object_heights
        
        # 检查是否达到10cm抬起要求
        lifted = lift_height >= self.cfg.LIFT_HEIGHT
        
        return lifted

    def _check_hand_object_proximity(self, obj_poses):
        """检查手部关键点与物体的平均距离是否小于12cm（论文距离条件）"""
        # 获取所有手部关键点（指尖）位置
        fingertip_positions = self._get_fingertip_positions()  # [batch_size, 4, 3]
        
        # 获取物体位置
        obj_positions = obj_poses[:, :3]  # [batch_size, 3]
        
        # 计算每个关键点到物体的距离
        distances = torch.norm(fingertip_positions - obj_positions.unsqueeze(1), dim=2)
        # distances形状: [batch_size, 4]
        # 计算平均距离（论文方法）
        avg_distances = torch.mean(distances, dim=1)
        # 检查平均距离是否小于12cm
        holding = avg_distances < self.cfg.HOLD_DISTANCE
        
        return holding

    def _get_fingertip_positions(self):
        """计算各个指尖的3D位置（基于手掌坐标系）"""
        # 获取手掌（末端执行器）位姿
        ee_poses = self._get_ee_pose()  # [batch_size, 7]
        batch_size = ee_poses.shape[0]
        
        # 初始化指尖位置张量
        fingertip_positions = torch.zeros((batch_size, 4, 3), device=self.device)
        
        for env_idx in range(batch_size):
            palm_pos = ee_poses[env_idx, :3]
            palm_quat = ee_poses[env_idx, 3:7]
            
            # 计算各个指尖的世界坐标系位置
            # 拇指
            thumb_offset = torch.tensor(self.cfg.FINGERTIP_OFFSETS['thumb'], 
                                    device=self.device)
            fingertip_positions[env_idx, 0] = palm_pos + self._apply_quaternion_rotation(thumb_offset, palm_quat)
            
            # 食指
            index_offset = torch.tensor(self.cfg.FINGERTIP_OFFSETS['index'], 
                                    device=self.device)
            fingertip_positions[env_idx, 1] = palm_pos + self._apply_quaternion_rotation(index_offset, palm_quat)
            
            # 中指
            middle_offset = torch.tensor(self.cfg.FINGERTIP_OFFSETS['middle'], 
                                        device=self.device)
            fingertip_positions[env_idx, 2] = palm_pos + self._apply_quaternion_rotation(middle_offset, palm_quat)
            
            # 无名指
            ring_offset = torch.tensor(self.cfg.FINGERTIP_OFFSETS['ring'], 
                                    device=self.device)
            fingertip_positions[env_idx, 3] = palm_pos + self._apply_quaternion_rotation(ring_offset, palm_quat)
        
        return fingertip_positions

    def _apply_quaternion_rotation(self, vector, quaternion):
        """应用四元数旋转到向量 - 精确实现"""
        # 将四元数转换为旋转矩阵
        R = self._quaternion_to_rotation_matrix(quaternion)
        # 应用旋转矩阵
        if isinstance(vector, torch.Tensor):
            vector_np = vector.cpu().numpy()
        else:
            vector_np = vector
            
        rotated_vector_np = R @ vector_np
        
        if isinstance(vector, torch.Tensor):
            return torch.tensor(rotated_vector_np, device=vector.device)
        else:
            return rotated_vector_np

    def _quaternion_to_rotation_matrix(self, quat):
        """四元数转旋转矩阵"""
        if isinstance(quat, torch.Tensor):
            q = quat.cpu().numpy()
        else:
            q = quat
            
        qx, qy, qz, qw = q[0], q[1], q[2], q[3]
        # 单位化
        norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
        
        # 计算旋转矩阵
        R = np.array([
            [1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
            [2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)],
            [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2)]
        ])
        
        return R

    def _check_collision(self):
        """检查与桌子的碰撞 - 基于多个指尖的检测"""
        fingertip_positions = self._get_fingertip_positions()
        
        batch_size = fingertip_positions.shape[0]
        collision_detected = torch.zeros(batch_size, device=self.device, dtype=torch.bool)
        
        for env_idx in range(batch_size):
            # 检查每个指尖是否穿透桌面
            for finger_idx in range(4):  # 4个手指
                fingertip_z = fingertip_positions[env_idx, finger_idx, 2]
                if fingertip_z < self.table_dims.z:  # 指尖低于桌面
                    collision_detected[env_idx] = True
                    break
        
        return collision_detected
    
    def _check_done(self):
        """检查episode是否结束"""
        # 单步MDP，每次执行后都结束
        return torch.ones(self.cfg.NUM_ENVS, device=self.device, dtype=torch.bool)
    
    def close(self):
        """关闭环境"""
        self.gym.destroy_sim(self.sim)