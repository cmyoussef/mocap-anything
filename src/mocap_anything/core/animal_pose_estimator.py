import functools
import os
import time

import cv2
import numpy as np
from mmengine.registry import init_default_scope

from mmpose.apis import inference_topdown, init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples

try:
    from mmdet.apis import inference_detector, init_detector

    HAS_MMDET = True
except (ImportError, ModuleNotFoundError):
    HAS_MMDET = False
    raise ImportError('MMDetection is required for detection but is not installed.')
from QuadCap.core.pose_visualizer import PoseVisualizer

# Standard COCO class names, index=0..79
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def log_time(func):
    @functools.wraps(func)  # Preserves original function name, docstring, etc.
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time

        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        milliseconds = int((elapsed_time - int(elapsed_time)) * 1000)

        print(
            f"Function '{func.__name__}' took: "
            f"{minutes} minutes, {seconds} seconds, {milliseconds} ms"
        )

        return result

    return wrapper


class AnimalPoseEstimator:
    """
    A minimal class to detect animal poses using MMDetection (for bounding boxes)
    and MMPose (for keypoints).

    - det_config / det_checkpoint: The detection model config & checkpoint
      (currently a COCO-based RTMDet).
    - pose_config / pose_checkpoint: The MMPose RTMPose AP-10K config & checkpoint.
    """

    def __init__(
            self,
            config_dir: str,
            model_dir: str,
            device: str = 'cuda:0',
            bbox_thr: float = 0.3,
            nms_thr: float = 0.3
    ):
        # -----------------------------
        # 1) Detection model (COCO-based RTMDet).
        #    NOTE: This does NOT have a deer category in COCO, so it may fail to detect deer.
        # -----------------------------
        self.det_config = os.path.join(config_dir, 'demo/mmdetection_cfg/rtmdet_m_8xb32-300e_coco.py')
        self.det_checkpoint = os.path.join(model_dir, 'rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth')

        # -----------------------------
        # 2) Pose model (RTMPose) on AP-10K
        # -----------------------------
        self.pose_config = os.path.join(
            config_dir,
            'animal_2d_keypoint/rtmpose/ap10k/rtmpose-m_8xb64-210e_ap10k-256x256.py'
        )
        self.pose_checkpoint = os.path.join(
            model_dir, 'rtmpose-m_simcc-ap10k_pt-aic-coco_210e-256x256-7a041aa1_20230206.pth'
        )
        self.device = device

        # Default thresholds (override per call if needed)
        self.bbox_thr = bbox_thr
        self.nms_thr = nms_thr

        # If you want the model to output heatmaps, set test_cfg accordingly:
        self.cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))

        # -----------------------------
        # Initialize the detection model
        # -----------------------------
        self.detector = init_detector(
            self.det_config,
            self.det_checkpoint,
            device=self.device,
        )

        # -----------------------------
        # Initialize the pose estimator
        # -----------------------------
        self.pose_estimator = init_pose_estimator(
            self.pose_config,
            self.pose_checkpoint,
            device=self.device,
            cfg_options=self.cfg_options
        )

        print("Using device:", self.device)
        self.visualizer = PoseVisualizer(self.pose_estimator)

        # Ensure default scope is correct for mmdet
        scope = self.detector.cfg.get('default_scope', 'mmdet')
        if scope is not None:
            init_default_scope(scope)

    @log_time
    def detect_pose(
            self,
            img,
            text_prompt: str = None,
            custom_entities: bool = False,
            bbox_thr: float = None,
            nms_thr: float = None
    ):
        """
        Detect bounding boxes and pose keypoints from a single image.

        Returns:
            data_samples: merged data samples (pose estimation results)
            bboxes: final bounding boxes (x1,y1,x2,y2)
            pose_results: raw list of PoseDataSample (one per bounding box)
        """

        # Override default thresholds if provided
        if bbox_thr is None:
            bbox_thr = self.bbox_thr
        if nms_thr is None:
            nms_thr = self.nms_thr

        # 1) Load image if path
        if isinstance(img, str):
            if not os.path.exists(img):
                raise FileNotFoundError(f"Image file does not exist: {img}")
            np_img = cv2.imread(img)
            if np_img is None:
                raise ValueError(f"Failed to read image from path: {img}")
        else:
            np_img = img

        # 2) Run object detection
        det_result = inference_detector(
            self.detector,
            np_img,
            text_prompt=text_prompt,
            custom_entities=custom_entities
        )

        # Convert to numpy
        pred_instance = det_result.pred_instances.cpu().numpy()
        # shape: [N,4] for bboxes, [N] for scores
        bboxes_with_scores = np.concatenate(
            (pred_instance.bboxes, pred_instance.scores[:, None]),
            axis=1
        )

        # 3) Filter by bbox confidence
        keep_mask = pred_instance.scores > bbox_thr
        bboxes_with_scores = bboxes_with_scores[keep_mask]
        kept_labels = pred_instance.labels[keep_mask]

        # 4) NMS
        if len(bboxes_with_scores) > 0:
            nms_idx = nms(bboxes_with_scores, thr=nms_thr)
            bboxes_with_scores = bboxes_with_scores[nms_idx]
            kept_labels = kept_labels[nms_idx]

        # 5) Print classes detected (if any)
        if len(bboxes_with_scores) == 0:
            print("No bounding boxes found!")
        else:
            for lbl in kept_labels:
                class_name = COCO_CLASSES[lbl] if lbl < len(COCO_CLASSES) else f"cls_{lbl}"
                print(f"Detected: {class_name}")

        # Drop the scores column for pose estimation
        bboxes = bboxes_with_scores[:, :4]

        # 6) Pose estimation
        pose_results = inference_topdown(self.pose_estimator, np_img, bboxes)

        # Merge data samples
        data_samples = merge_data_samples(pose_results)

        return data_samples, bboxes, pose_results

    def visualize_pose(self, img_path, data_samples, show=False, out_file=None):
        """
        A helper method to call the visualizer's `visualize_pose` for convenience.

        Args:
            img_path (str): Path to the image.
            data_samples: The merged data samples or list of pose results from `detect_pose`.
            show (bool): Whether to show the image in a window. Defaults to False.
            out_file (str): Optional path to save the annotated image.

        Returns:
            np.ndarray: The annotated image in BGR format.
        """
        return self.visualizer.visualize_pose(img_path, data_samples, show=show, out_file=out_file)


if __name__ == "__main__":

    import glob

    # Directories for configs and checkpoints
    base_dir = os.path.dirname(os.path.dirname(__file__))
    parent_dir = os.path.dirname(os.path.dirname(base_dir))
    config_dir = os.path.join(base_dir, 'configs')
    model_dir = os.path.join(parent_dir, 'checkpoints')
    # model_dir = r"E:\ai_projects\poseEstimation\checkpoints"

    # Initialize the pose estimator
    pose_estimator = AnimalPoseEstimator(config_dir, model_dir, device='cuda:0', bbox_thr=0.3, nms_thr=0.3)

    # Get a list of images
    img_pattern = r"E:\ai_projects\poseEstimation\quad_cap\assets\deers\*.png"
    # img_pattern = r"E:\ai_projects\poseEstimation\quad_cap\assets\deer\*.png"
    img_pattern = r"P:\Bambi\incoming\mustang horses playing copy.jpg"
    image_files = glob.glob(img_pattern)

    # Process one image (or loop over them)
    if image_files:
        img_path = image_files[0]
        data_samples, bboxes, pose_results = pose_estimator.detect_pose(img_path, text_prompt='Deer standing',
                                                                        bbox_thr=0.3, nms_thr=0.3)

        print("Detected bounding boxes:", bboxes)

        # Visualize the pose
        vis_img = pose_estimator.visualize_pose(img_path, data_samples, show=True, out_file='deer_pose.png')
        print("Visualization saved to deer_pose.png")
