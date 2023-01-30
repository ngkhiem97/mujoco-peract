# From: https://github.com/stepjam/RLBench/blob/master/rlbench/backend/observation.py

import numpy as np


class Observation(object):
    """Storage for both visual and low-dimensional observations."""

    def __init__(self,
                 front_rgb: np.ndarray = None,
                 front_depth: np.ndarray = None,
                 front_mask: np.ndarray = None,
                 front_point_cloud: np.ndarray = None,
                 top_rgb: np.ndarray = None,
                 top_depth: np.ndarray = None,
                 top_mask: np.ndarray = None,
                 top_point_cloud: np.ndarray = None,
                 side_1_rgb: np.ndarray = None,
                 side_1_depth: np.ndarray = None,
                 side_1_mask: np.ndarray = None,
                 side_1_point_cloud: np.ndarray = None,
                 side_2_rgb: np.ndarray = None,
                 side_2_depth: np.ndarray = None,
                 side_2_mask: np.ndarray = None,
                 side_2_point_cloud: np.ndarray = None,
                #  joint_velocities: np.ndarray = None,
                #  joint_positions: np.ndarray = None,
                #  joint_forces: np.ndarray = None,
                #  gripper_open: float = 0,
                #  gripper_pose: np.ndarray = None,
                #  gripper_matrix: np.ndarray = None,
                #  gripper_joint_positions: np.ndarray = None,
                #  gripper_touch_forces: np.ndarray = None,
                #  task_low_dim_state: np.ndarray = None,
                #  ignore_collisions: np.ndarray = None,
                 misc: dict = None):
        self.front_rgb = front_rgb
        self.front_depth = front_depth
        self.front_mask = front_mask
        self.front_point_cloud = front_point_cloud
        self.top_rgb = top_rgb
        self.top_depth = top_depth
        self.top_mask = top_mask
        self.top_point_cloud = top_point_cloud
        self.side_1_rgb = side_1_rgb
        self.side_1_depth = side_1_depth
        self.side_1_mask = side_1_mask
        self.side_1_point_cloud = side_1_point_cloud
        self.side_2_rgb = side_2_rgb
        self.side_2_depth = side_2_depth
        self.side_2_mask = side_2_mask
        self.side_2_point_cloud = side_2_point_cloud
        # self.joint_velocities = joint_velocities
        # self.joint_positions = joint_positions
        # self.joint_forces = joint_forces
        # self.gripper_open = gripper_open
        # self.gripper_pose = gripper_pose
        # self.gripper_matrix = gripper_matrix
        # self.gripper_joint_positions = gripper_joint_positions
        # self.gripper_touch_forces = gripper_touch_forces
        # self.task_low_dim_state = task_low_dim_state
        # self.ignore_collisions = ignore_collisions
        self.misc = misc

    def get_low_dim_data(self) -> np.ndarray:
        """Gets a 1D array of all the low-dimensional obseervations.

        :return: 1D array of observations.
        """
        low_dim_data = [] if self.gripper_open is None else [[self.gripper_open]]
        for data in [self.joint_velocities, self.joint_positions,
                     self.joint_forces,
                     self.gripper_pose, self.gripper_joint_positions,
                     self.gripper_touch_forces, self.task_low_dim_state]:
            if data is not None:
                low_dim_data.append(data)
        return np.concatenate(low_dim_data) if len(low_dim_data) > 0 else np.array([])
