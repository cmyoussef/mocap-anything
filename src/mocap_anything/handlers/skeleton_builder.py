from QuadCap.configs._base_.datasets.animalpose import dataset_info

# ------------------------------------------------------------------------
# DATASET / KEYPOINT SETUP
# ------------------------------------------------------------------------
keypoint_info = dataset_info["keypoint_info"]
max_kpt_index = max(keypoint_info.keys())

# Build a list of keypoint dicts in ascending index order:
dataset_keypoint_metadata = [
    keypoint_info[i] for i in range(max_kpt_index + 1)
]

# Also retrieve the "skeleton_info" if you want to use it.
dataset_skeleton_info = dataset_info["skeleton_info"]

# Custom parent-child hierarchy:
# (child, parent).
animal_skeleton_hierarchy = [
    ['Withers', 'TailBase'],
    ['R_B_Elbow', 'TailBase'],
    ['R_B_Knee', 'R_B_Elbow'],
    ['R_B_Paw', 'R_B_Knee'],
    ['L_B_Elbow', 'TailBase'],
    ['L_B_Knee', 'L_B_Elbow'],
    ['L_B_Paw', 'L_B_Knee'],
    ['Throat', 'Withers'],
    ['L_F_Elbow', 'Withers'],
    ['L_F_Knee', 'L_F_Elbow'],
    ['L_F_Paw', 'L_F_Knee'],
    ['R_F_Elbow', 'Withers'],
    ['R_F_Knee', 'R_F_Elbow'],
    ['R_F_Paw', 'R_F_Knee'],
    ['Nose', 'Throat'],
    ['L_Eye', 'Nose'],
    ['R_Eye', 'Nose'],
    ['L_EarBase', 'L_Eye'],
    ['R_EarBase', 'R_Eye'],
]


# ------------------------------------------------------------------------
# 1) BASE CLASS (No-Op methods if Maya is not available)
# ------------------------------------------------------------------------
class SkeletonBuilderBase:
    """
    A base skeleton builder with no-op methods. This avoids errors
    if Maya is not installed or available in the environment.
    """

    def __init__(self, keypoint_metadata=None, skeleton_hierarchy=None, callback_func=None):
        """
        Stash references if you need them. Otherwise, do nothing.
        """
        self.keypoint_metadata = keypoint_metadata or dataset_keypoint_metadata
        self.skeleton_hierarchy = skeleton_hierarchy or animal_skeleton_hierarchy
        print("[SkeletonBuilderBase] Initialized. Maya not available? => No-op methods.")

    def create_maya_skeleton(self, default_pos=(0, 0, 0)):
        """No-op. Returns empty list."""
        print("[SkeletonBuilderBase] create_maya_skeleton: No-op.")
        return []

    def update_maya_skeleton_pose(
            self,
            pose_data,
            hierarchy_joints=None,
            pose_data_2d=None,
            joint_z_thresholds=None,
            body_depth_ratio=0.5
    ):
        """No-op. Does nothing."""
        print("[SkeletonBuilderBase] update_maya_skeleton_pose: No-op.")
        return

    def build_maya_skeleton_from_pose_data(self, pose_data,
                                           keypoint_metadata=None,
                                           skeleton_hierarchy=None):
        """
        No-op. Does nothing, returns empty list.
        """
        print("[SkeletonBuilderBase] build_maya_skeleton_from_pose_data: No-op.")
        return []

    def set_timeline_range(self, start_frame, end_frame):
        """Set the playback range for the timeline"""
        pass

    def update_current_frame(self, frame):
        """Update the current timeline frame"""
        pass

    def on_time_change(self, new_time):
        pass
