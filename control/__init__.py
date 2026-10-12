"""control 包：UR5e 运动学 + 笛卡尔轨迹 + 导纳控制 + 运动执行器。

对外接口（见 INTERFACES.md v1）：
    Kinematics      fk / jacobian / ik / clamp
    CartesianTrajectory, min_jerk, slerp, quat_from_rotvec, rotvec_from_quat, quat_mul
    AdmittanceController  reset / update / set_params / state
    MotionRunner    move_to_pose / follow / follow_joint_path / hold
"""
from .admittance import AdmittanceController
from .kinematics import Kinematics
from .motion import MotionRunner
from .trajectory import (CartesianTrajectory, quat_from_rotvec, quat_mul, min_jerk,
                         rotvec_from_quat, slerp)

__all__ = [
    "Kinematics",
    "AdmittanceController",
    "MotionRunner",
    "CartesianTrajectory",
    "min_jerk",
    "slerp",
    "quat_from_rotvec",
    "rotvec_from_quat",
    "quat_mul",
]
