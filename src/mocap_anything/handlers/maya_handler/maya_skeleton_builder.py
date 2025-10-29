import traceback

import maya.cmds as cmds
from PyServerManager.core.logger import logger

from QuadCap.configs._base_.datasets.animalpose import dataset_info
from QuadCap.handlers.maya_handler.cleanup import cleanup_spiky_joints
from QuadCap.handlers.skeleton_builder import SkeletonBuilderBase

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
    ['Neck', 'Root_of_tail'],
    ['R_Hip', 'Root_of_tail'],
    ['R_Knee', 'R_Hip'],
    ['R_B_Paw', 'R_Knee'],
    ['L_Hip', 'Root_of_tail'],
    ['L_Knee', 'L_Hip'],
    ['L_B_Paw', 'L_Knee'],
    ['L_Shoulder', 'Neck'],
    ['L_Elbow', 'L_Shoulder'],
    ['L_F_Paw', 'L_Elbow'],
    ['R_Shoulder', 'Neck'],
    ['R_Elbow', 'R_Shoulder'],
    ['R_F_Paw', 'R_Elbow'],
    ['Nose', 'Neck'],
    ['L_Eye', 'Nose'],
    ['R_Eye', 'Nose']
]

import maya.api.OpenMaya as om


class TimelineManager:
    def __init__(self, callback_func=None, callback_args=None):
        """Initialize timeline manager and connect to timeChanged event"""
        self.callback_id = None
        self._register_time_changed_callback()
        self.callback_func = callback_func

    def _register_time_changed_callback(self):
        """Connect the Maya timeChanged event to our dummy method"""
        self.callback_id = om.MEventMessage.addEventCallback(
            'timeChanged', self.time_changed_callback
        )

    def time_changed_callback(self, new_time, client_data=None):
        """Dummy method that receives timeline change notifications"""
        print(f"Timeline changed to frame: {new_time}, {client_data}")
        try:
            if self.callback_func:
                self.callback_func(new_time)
        except Exception as e:
            traceback.print_exc()
            print("Error in timeline callback:", e)

    def set_timeline_range(self, start_frame, end_frame):
        """Set the playback range for the timeline"""
        cmds.playbackOptions(minTime=start_frame, maxTime=end_frame)
        cmds.playbackOptions(animationStartTime=start_frame, animationEndTime=end_frame)

    def update_current_frame(self, frame):
        """Update the current timeline frame"""
        cmds.currentTime(frame)

    def __del__(self):
        """Cleanup callback when instance is deleted"""
        if self.callback_id:
            om.MEventMessage.removeCallback(self.callback_id)


# ------------------------------------------------------------------------
# 2) MAYA-SPECIFIC CLASS
# ------------------------------------------------------------------------
class MayaSkeletonBuilder(SkeletonBuilderBase):
    """
    Encapsulates creation and updating of a Maya joint hierarchy using
    metadata from a keypoint-based dataset (e.g., AnimalPose).
    """

    def __init__(self, keypoint_metadata=None, skeleton_hierarchy=None, callback_func=None):
        """
        If Maya is available, call the base class init for references.
        """
        super().__init__(keypoint_metadata, skeleton_hierarchy, callback_func)
        print("[MayaSkeletonBuilder] Initialized. Maya is available? => True.")
        self.timeline_manager = TimelineManager(callback_func)

    @staticmethod
    def create_maya_skeleton(
            keypoint_metadata=None,
            skeleton_hierarchy=None,
            default_pos=(0, 0, 0)
    ):
        """
        Creates or updates joints in Maya from your keypoint metadata (names, colors),
        sets side/type attributes, and parents them according to 'skeleton_hierarchy'.
        Does NOT position them using 3D data.

        Args:
            keypoint_metadata (list): Each entry is a dict like:
               {"name": "Eye_Left", "color": [R,G,B], ...}
            skeleton_hierarchy (list of lists or tuples): E.g. [ [child, parent], ... ]
            default_pos (tuple): (x,y,z) position for newly created joints.

        Returns:
            List of all joints in the resulting hierarchy (the "main" parent plus all DAG children).
            If Maya not available, returns an empty list.
        """
        if skeleton_hierarchy is None:
            skeleton_hierarchy = animal_skeleton_hierarchy
        if keypoint_metadata is None:
            keypoint_metadata = dataset_keypoint_metadata

        created_joints = {}
        for kp_idx, kpt_info in enumerate(keypoint_metadata):
            joint_name = kpt_info["name"].replace(' ', '_')

            if cmds.objExists(joint_name):
                # print(f"[create_maya_skeleton] Joint '{joint_name}' exists, skipping creation.")
                continue

            # Create a new joint at default_pos
            cmds.select(d=True)
            new_jnt = cmds.joint(name=joint_name, position=default_pos)
            print(f"[create_maya_skeleton] Created joint '{new_jnt}' at {default_pos}.")

            # Set color from [R,G,B]
            color_rgb = kpt_info.get("color", [255, 255, 255])
            color_index = int(sum(color_rgb) % 31)
            cmds.setAttr(new_jnt + ".overrideEnabled", 1)
            cmds.setAttr(new_jnt + ".overrideColor", color_index)

            # Side attribute (0=Center, 1=Left, 2=Right, 3=None)
            side_val = 3
            j_lower = joint_name.lower()
            if "l_" in j_lower:
                side_val = 1
            elif "r_" in j_lower:
                side_val = 2

            if cmds.attributeQuery('side', node=new_jnt, exists=True):
                cmds.setAttr(new_jnt + ".side", side_val)

            # Type attribute from name
            if cmds.attributeQuery('type', node=new_jnt, exists=True):
                enum_list = cmds.attributeQuery('type', node=new_jnt, listEnum=True)
                if enum_list:
                    enum_values = enum_list[0].split(':')
                    type_val = 0
                    for i, candidate in enumerate(enum_values):
                        if candidate.lower() in j_lower:
                            type_val = i
                            break
                    cmds.setAttr(new_jnt + ".type", type_val)
            cmds.setAttr(new_jnt + ".radius", .08)
            # cmds.transformLimits(new_jnt, tz=[-0.5, .5], etz=[1, 1])

            created_joints[kp_idx] = new_jnt

        # Build the hierarchy
        for child_name, parent_name in skeleton_hierarchy:
            print(child_name, cmds.objExists(child_name), parent_name, cmds.objExists(parent_name))
            if not cmds.objExists(child_name) or not cmds.objExists(parent_name):
                continue
            curr_parent = cmds.listRelatives(child_name, parent=True)
            if not curr_parent or curr_parent[0] != parent_name:
                try:
                    cmds.parent(child_name, parent_name)
                    print(f"[create_maya_skeleton] parented '{child_name}' under '{parent_name}'.")
                except Exception as e:
                    print(f"[WARNING] Could not parent '{child_name}' -> '{parent_name}': {e}")

        # Return the main parent plus all DAG children
        if skeleton_hierarchy:
            main_parent = skeleton_hierarchy[0][1]  # e.g. 'Root_of_tail' in your example
            if cmds.objExists(main_parent):
                hierarchy_joints = [main_parent] + cmds.ls(main_parent, dag=True)
                return hierarchy_joints
        return []

    @staticmethod
    def update_maya_skeleton_pose(
            pose_data,  # your existing structure
            keypoint_metadata=None,
            hierarchy_joints=None,
            pose_data_2d=None,  # 2D coords (list of [x, y]) if available
            skeleton_hierarchy=None,
            joint_z_thresholds=None,
            body_depth_ratio=0.5
    ):
        """
        1. Extract 3D coords from pose_data.
        2. Build a dict of {joint_name: (x, y, z)} => target_positions.
        3. Call cleanup_spiky_joints(...) to remove outliers and fix Throat.
        4. Move Maya joints to final positions.

        We also pass in 2D coords to compute bounding box if we want.
        If user doesn't pass 2D, we default bounding box in the cleanup function.
        """

        if keypoint_metadata is None:
            keypoint_metadata = dataset_keypoint_metadata
        if hierarchy_joints is None:
            hierarchy_joints = []
        if skeleton_hierarchy is None:
            skeleton_hierarchy = animal_skeleton_hierarchy

        # 1) Parse the 3D points
        points_3d = pose_data.get("3d_points", [])
        if not (points_3d and points_3d[0] and points_3d[0]):
            print("[WARNING] No 3D points found or structure mismatch.")
            return

        # single batch, single instance => points_3d[0][0]
        instance_3d = points_3d[0]

        # 2) Build target_positions for each joint
        target_positions = {}
        for kp_idx, coords_3d in enumerate(instance_3d):
            if kp_idx >= len(keypoint_metadata):
                continue
            joint_name = keypoint_metadata[kp_idx]["name"].replace(' ', '_')
            if cmds.objExists(joint_name):
                x, y, z = coords_3d
                target_positions[joint_name] = (x, y, z)

        # 3) Cleanup spiky joints
        target_positions = cleanup_spiky_joints(
            target_positions=target_positions,
            skeleton_hierarchy=skeleton_hierarchy,
            pose_data_2d=pose_data_2d,  # optional 2D data
            joint_z_thresholds=joint_z_thresholds,
            body_depth_ratio=body_depth_ratio,
            fix_throat=True
        )
        current_time = cmds.currentTime(q=True)  # e.g., 1, 2, etc.

        # 4) Apply final positions to Maya
        print('-' * 30, current_time, '-' * 30)
        for jnt in hierarchy_joints:
            if jnt in target_positions:
                cmds.setKeyframe(jnt, time=current_time, attribute='translate')

                x, y, z = target_positions[jnt]
                # Example: you might invert X,Y if your coordinate system requires it
                cmds.move(x * -1, y * -1, z, f'{jnt}.scalePivot', f'{jnt}.rotatePivot', absolute=True)
                # cmds.move(x * -1, y * -1, z, jnt, absolute=True)
                cmds.setKeyframe(jnt, time=current_time, attribute='translate')
                print(f'{jnt} moved to {x}, {y}, {z} at frame {current_time}')
            else:
                print(f"[WARNING] Missing {jnt} in target_positions after cleanup")
        print('-' * (62 + len(str(current_time))))

    def build_maya_skeleton_from_pose_data(self, pose_data,
                                           keypoint_metadata=None,
                                           skeleton_hierarchy=None):
        """
        Convenience method that:
          1) create_maya_skeleton(...)
          2) update_maya_skeleton_pose(...)
        in one go.

        Args:
            pose_data (dict): Contains "3d_points" to position the joints.
            keypoint_metadata (list): The metadata describing each keypoint (name, color).
            skeleton_hierarchy (list): Each entry is (child, parent) for Maya parenting.

        Returns:
            A list of joints in the final hierarchy (or empty if no Maya).
        """
        logger.debug(f"[build_maya_skeleton_from_pose_data] pose_data: {pose_data}")
        # Step 1: Create skeleton
        if skeleton_hierarchy is None:
            skeleton_hierarchy = animal_skeleton_hierarchy
        if keypoint_metadata is None:
            keypoint_metadata = dataset_keypoint_metadata
        hierarchy_joints = self.create_maya_skeleton(keypoint_metadata, skeleton_hierarchy)
        if not hierarchy_joints:
            return []

        # Step 2: Update pose with 3D coords
        self.update_maya_skeleton_pose(pose_data, keypoint_metadata, hierarchy_joints)
        return hierarchy_joints

    def set_timeline_range(self, start_frame, end_frame):
        self.timeline_manager.set_timeline_range(start_frame, end_frame)

    def update_current_frame(self, frame):
        self.timeline_manager.update_current_frame(frame)

    def __del__(self):
        self.timeline_manager.__del__()
