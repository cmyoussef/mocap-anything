from collections import defaultdict


def build_adjacency_list(skeleton_hierarchy):
    """
    Given a list of (child, parent) pairs,
    produce adjacency dict: adjacency[joint] = [neighbors...]
    """
    adjacency = defaultdict(list)
    for child, parent in skeleton_hierarchy:
        adjacency[child].append(parent)
        adjacency[parent].append(child)
    return adjacency


def compute_2d_bbox(pose_data_2d):
    """
    Given a list of 2D keypoints, compute the bounding box (min_x, max_x, min_y, max_y).
    pose_data_2d is something like [[x0, y0], [x1, y1], ...].
    Returns: (min_x, max_x, min_y, max_y)
    """
    xs = [kp[0] for kp in pose_data_2d]
    ys = [kp[1] for kp in pose_data_2d]
    return min(xs), max(xs), min(ys), max(ys)


def derive_default_threshold_for_joint(bounding_box, body_depth_ratio=0.1):
    """
    For a given joint, return the default Z-threshold based on bounding box size
    and a user-defined ratio for the 'typical' front-to-back depth.

    bounding_box = (min_x, max_x, min_y, max_y)
    body_depth_ratio = e.g. 0.5 -> half of the bounding box dimension.

    We'll define a bounding_box_size, for example as the average of width & height,
    or some other measure. For a large bounding box, we get a larger threshold.
    """
    (min_x, max_x, min_y, max_y) = bounding_box
    width = max_x - min_x
    height = max_y - min_y

    # For simplicity, let's take the average of width & height:
    bbox_size = (width + height) / 2.0

    # Multiply by some ratio that represents how "deep" the animal is
    # compared to its 2D bounding box.
    # For a horse with a big chest, you might set body_depth_ratio ~ 0.8 or 1.0
    # (adjust to taste)
    threshold = bbox_size * body_depth_ratio
    return threshold


def detect_spikes_z(
        target_positions,
        adjacency,
        bounding_box,
        joint_z_thresholds: dict = None,
        body_depth_ratio=0.1
):
    """
    Detect "spiky" joints whose Z is too far from the average of neighbors' Z.

    If a user threshold for joint 'jnt' is in joint_z_thresholds[jnt],
    we use that; otherwise, we compute a default from bounding_box * body_depth_ratio.
    """
    if joint_z_thresholds is None:
        joint_z_thresholds = {}
        for jnt in target_positions:
            joint_z_thresholds[jnt] = derive_default_threshold_for_joint(
                bounding_box, body_depth_ratio=body_depth_ratio
            )

    spiky_joints = set()

    for jnt, neighbors in adjacency.items():
        if jnt not in target_positions:
            continue
        jnt_z = target_positions[jnt][2]

        neighbor_z_list = []
        for nbr in neighbors:
            if nbr in target_positions:
                neighbor_z_list.append(target_positions[nbr][2])
        if not neighbor_z_list:
            continue

        # average neighbor Z
        mean_neighbor_z = sum(neighbor_z_list) / len(neighbor_z_list)

        # get the threshold for this joint
        z_threshold = joint_z_thresholds.get(jnt)

        if abs(abs(jnt_z) - abs(mean_neighbor_z)) > z_threshold:
            spiky_joints.add(jnt)

    return spiky_joints


def fix_spikes_z(target_positions, adjacency, spiky_joints):
    """
    Fix spiky joint Z by averaging it with the neighbors' Z (that aren't spiky).
    """
    for jnt in spiky_joints:
        if jnt not in target_positions:
            continue

        neighbors = adjacency[jnt]
        valid_neighbor_z = []
        for nbr in neighbors:
            if nbr in target_positions and nbr not in spiky_joints:
                valid_neighbor_z.append(target_positions[nbr][2])

        if valid_neighbor_z:
            x, y, _ = target_positions[jnt]
            avg_z = sum(valid_neighbor_z) / len(valid_neighbor_z)
            target_positions[jnt] = (x, y, avg_z)

    return target_positions


def fix_throat_position(target_positions):
    """
    Example: Force the Throat's Z to be halfway between the Withers and Nose Z.
    """
    if 'Withers' not in target_positions or 'Nose' not in target_positions or 'Throat' not in target_positions:
        return target_positions  # skip if missing

    withers_x, withers_y, withers_z = target_positions['Withers']
    nose_x, nose_y, nose_z = target_positions['Nose']
    throat_x, throat_y, throat_z = target_positions['Throat']

    dist_z = (withers_z - nose_z) * 0.5
    new_throat_z = withers_z - dist_z

    target_positions['Throat'] = (throat_x, throat_y, new_throat_z)
    return target_positions


def cleanup_spiky_joints(
        target_positions,
        skeleton_hierarchy,
        pose_data_2d=None,
        joint_z_thresholds=None,
        body_depth_ratio=0.1,
        fix_throat=True
):
    """
    1) Build adjacency.
    2) If pose_data_2d is provided, compute bounding box. Otherwise use a dummy bounding box.
    3) Detect spiky joints, using user overrides or derived thresholds.
    4) Fix them by averaging with neighbors.
    5) Optionally fix Throat.

    'pose_data_2d' is a list of 2D coords (x, y) for each keypoint.
    'joint_z_thresholds' is a dict: { "Nose": 50.0, "TailBase": 80.0, ... }
        If not provided for a joint, we'll derive from bounding box * body_depth_ratio.

    'body_depth_ratio' sets how big the default threshold is relative to 2D bounding box size.
    'fix_throat': boolean to apply the special Throat fix.
    """
    adjacency = build_adjacency_list(skeleton_hierarchy)

    # 2) Compute bounding box from 2D data (or fallback).
    if pose_data_2d is None:
        pose_data_2d = [(t[0], t[1]) for t in target_positions.values()]
    bounding_box = compute_2d_bbox(pose_data_2d)

    print('-' * 10)
    print("bounding_box", bounding_box)
    print('-' * 10)
    # 3) Detect spiky joints
    spiky_joints = detect_spikes_z(
        target_positions=target_positions,
        adjacency=adjacency,
        bounding_box=bounding_box,
        joint_z_thresholds=joint_z_thresholds,
        body_depth_ratio=body_depth_ratio
    )

    # 4) Fix spiky joints
    target_positions = fix_spikes_z(target_positions, adjacency, spiky_joints)

    # 5) Optional Throat fix
    if fix_throat:
        target_positions = fix_throat_position(target_positions)

    return target_positions
