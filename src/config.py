import math

class DemoGraspConfig:
    # Simulation
    SIM_DEVICE = "cuda:0"
    SIM_SUBSTEPS = 2
    CONTROL_FREQ = 3  # Hz
    LOW_LEVEL_FREQ = 60  # Hz
    
    # Robot
    ROBOT_URDF_PATH = "../isaacgym/assets/urdf/kuka_allegro_description/kuka_allegro.urdf"
    
    # Robot DOF configuration (从URDF分析得出)
    ARM_DOF = 7   # Kuka iiwa7机械臂
    HAND_DOF = 16 # Allegro手 (4手指 × 4关节)
    TOTAL_DOF = ARM_DOF + HAND_DOF
    
    # Action Space (简化：直接控制所有关节)
    ACTION_DIM = TOTAL_DOF  # 直接控制23个关节
    
    # 使用isaacgym内置IK求解器
    USE_INVERSE_KINEMATICS = True
    EE_LINK_NAME = "iiwa7_link_ee"          # 末端执行器链接名称
    IK_DAMPING = 0.05                       # 阻尼系数
    IK_MAX_FORCE = 500.0                    # 最大力
    IK_POSITION_GAIN = 2.0                  # 位置增益
    IK_ORIENTATION_GAIN = 0.5               # 朝向增益
    
    # 关节限制 (从URDF中的limit标签获取)
    ARM_JOINT_LIMITS = [
        (-2.967, 2.967), (-2.094, 2.094), (-2.967, 2.967), 
        (-2.094, 2.094), (-2.967, 2.967), (-2.094, 2.094), (-3.054, 3.054)
    ]
    HAND_JOINT_LIMITS = [
        # 食指
        (-0.558, 0.558), (-0.279, 1.727), (-0.279, 1.727), (-0.279, 1.727),
        # 中指  
        (-0.558, 0.558), (-0.279, 1.727), (-0.279, 1.727), (-0.279, 1.727),
        # 无名指
        (-0.558, 0.558), (-0.279, 1.727), (-0.279, 1.727), (-0.279, 1.727),
        # 拇指
        (0.279, 1.570), (-0.331, 1.151), (-0.279, 1.727), (-0.279, 1.762)
    ]
    
    # Observation Space
    POINT_CLOUD_SIZE = 512
    EE_POSE_DIM = 7  # 末端执行器位姿 (位置3 + 四元数4)
    OBJ_POSE_DIM = 7  # 物体位姿
    HAND_POSE_DIM = HAND_DOF  # 手部关节角度
    OBS_DIM = 128 + EE_POSE_DIM + OBJ_POSE_DIM + HAND_POSE_DIM  # 点云特征 + 位姿信息
    
    # Environment
    NUM_ENVS = 16  # 大幅减少并行环境数量
    EPISODE_LENGTH = 1  # 单步MDP
    DEMO_LENGTH = 40    # 演示轨迹长度
    
    # Object settings
    OBJECT_ASSET_DIR = "../isaacgym/assets/urdf/ycb"
    OBJECT_SCALE = 1.0
    
    # 预定义的YCB物体列表 
    OBJECT_URDF_PATHS = [
        "010_potted_meat_can/010_potted_meat_can.urdf",
        "011_banana/011_banana.urdf", 
        "025_mug/025_mug.urdf",
        "061_foam_brick/061_foam_brick.urdf"
        
    ]
    
    # Table settings
    TABLE_DIMS = [1.0, 1.0, 0.1]  # x, y, z
    TABLE_POSITION = [0.0, 0.0, 0.05]  # 桌子高度
    
    # Reset region (论文中的50cm x 50cm区域)
    RESET_REGION = [0.5, 0.5]  # x, y范围
    
    # Success criteria (论文标准)
    LIFT_HEIGHT = 0.1  # 10cm
    HOLD_DISTANCE = 0.12  # 12cm
    
   # 指尖关键点配置
    FINGERTIP_OFFSETS = {
        'thumb': [0.08, 0.0, 0.02],   # 拇指指尖偏移 [x, y, z]
        'index': [0.07, 0.0, 0.0],    # 食指指尖偏移
        'middle': [0.07, 0.0, 0.0],   # 中指指尖偏移  
        'ring': [0.07, 0.0, 0.0]      # 无名指指尖偏移
    }
    
    # Reward
    SUCCESS_REWARD = 1.0
    COLLISION_PENALTY = 0.5
    FAILURE_REWARD = 0.0
    
    # Network
    HIDDEN_DIMS = [512, 256, 128]  # 简化网络结构
    POINTNET_FEAT_DIM = 128
    
    # PPO
    LEARNING_RATE = 3e-4
    GAMMA = 0.99
    CLIP_EPS = 0.2
    VALUE_COEF = 0.5
    ENTROPY_COEF = 0.01
    
    # Training
    NUM_ITERATIONS = 1000
    BATCH_SIZE = 64
    
    # 初始手部姿态 (张开状态)
    INITIAL_HAND_POSE = [0.0] * HAND_DOF
    
    # 初始机械臂姿态
    INITIAL_ARM_POSE = [0.0, -0.5, 0.0, -1.0, 0.0, 1.0, 0.0]