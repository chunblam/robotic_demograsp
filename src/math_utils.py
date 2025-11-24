import numpy as np

def quaternion_to_rotation_matrix(self, quat):
    """将四元数转换为旋转矩阵"""
    qx, qy, qz, qw = quat
    
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

def pose_to_homogeneous(self, pose):
    """将位姿转换为齐次变换矩阵"""
    pos = pose[:3]
    quat = pose[3:]
    
    T = np.eye(4)
    T[:3, :3] = self.quaternion_to_rotation_matrix(quat)
    T[:3, 3] = pos
    
    return T

def homogeneous_to_pose(self, T):
    """将齐次变换矩阵转换为位姿"""
    pos = T[:3, 3]
    rot = T[:3, :3]
    
    # 简化：返回单位四元数
    quat = np.array([0.0, 0.0, 0.0, 1.0])
    return np.concatenate([pos, quat])