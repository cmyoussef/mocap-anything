# server_pose_executor.py
import time
from scipy.spatial import cKDTree
import numpy as np
import cv2
# from PyServerManager.executors.acync_server_executor import AsyncPickleServer
from PyServerManager.executors.acync_server_executor import ServerExecutor

from QuadCap.core.animal_pose_estimator import AnimalPoseEstimator
from QuadCap.core.moge import MoGeInference


_pose_estimator = None
_moge_inference = None  # Global for MoGe in each worker


def sample_3d_keypoints(points_3d, pose_results):
    """
    For each 2D keypoint (x, y), look up the 3D coordinate points_3d[y, x].
    Return a nested list shape: [#instances, #joints, 3].
    """
    all_3d_kpts = []
    H, W, _ = points_3d.shape

    for single_result in pose_results:
        if not hasattr(single_result, 'pred_instances'):
            all_3d_kpts.append([])
            continue
        pred_inst = single_result.pred_instances
        if not hasattr(pred_inst, 'keypoints'):
            all_3d_kpts.append([])
            continue

        kpts_2d = pred_inst.keypoints  # shape (N, K, 2) or (K, 2)
        # If shape is (1, K, 2), flatten it:
        if len(kpts_2d.shape) == 3 and kpts_2d.shape[0] == 1:
            kpts_2d = kpts_2d[0]

        inst_3d = []
        for (x_f, y_f) in kpts_2d:
            x_i = int(round(x_f))
            y_i = int(round(y_f))
            if 0 <= x_i < W and 0 <= y_i < H:
                xyz = points_3d[y_i, x_i]  # shape (3,)
                inst_3d.append(xyz.tolist())
            else:
                inst_3d.append([None, None, None])
        all_3d_kpts.append(inst_3d)

    return all_3d_kpts


class PoseEstimationServerExecutor(ServerExecutor):
    """
    A ServerExecutor subclass that spins up a SocketServer and, on each request,
    runs both AnimalPoseEstimator (2D keypoints) and MoGe (3D reconstruction).
    Then returns bounding boxes, 2D joints, and 3D keypoints in a JSON-friendly dict.

    IMPORTANT:
     - We do NOT instantiate these models in __init__ (main process),
       to avoid pickle errors in Windows. Instead, we store config/paths
       and create them in worker_init_fn (child processes).
    """

    def __init__(self, *args, **kwargs):
        print("PoseEstimationServerExecutor initializing.......")

        super().__init__(*args, **kwargs)
        # Immediately after super().__init__, update self.args_dict
        if kwargs:
            self.args_dict.update(kwargs)

        # ---- Pull config/model paths for MMPose
        self.config_dir = self.args_dict.get("config_dir", None)
        self.model_dir = self.args_dict.get("model_dir", None)
        self.device = self.args_dict.get("device", "cuda:0")
        self.bbox_thr = float(self.args_dict.get("bbox_thr", 0.3))
        self.nms_thr = float(self.args_dict.get("nms_thr", 0.3))
        self.num_workers = int(self.args_dict.get("num_workers", 1))

        # ---- MoGe paths (we won't create the model here!)
        self.moge_model_path = self.args_dict.get("moge_model_path", None)
        self.moge_root_folder = self.args_dict.get("moge_root_folder", ".")

        self.logger.info("PoseEstimationServerExecutor initialized.")
        self.setup_server(data_workers=self.num_workers)
        # NOTE: do NOT call self.server.serve_forever(runner=None)!

    def worker_init_fn(self, *args, **kwargs):
        """
        Called once in each worker process. We set up the MMPose model AND the MoGe model
        so each worker has its own copy (avoiding pickle issues).
        """
        from mmpose.utils import register_all_modules
        register_all_modules()

        global _pose_estimator
        global _moge_inference

        # 1) Create the MMPose AnimalPoseEstimator in this worker
        _pose_estimator = AnimalPoseEstimator(
            config_dir=self.config_dir,
            model_dir=self.model_dir,
            device=self.device,
            bbox_thr=self.bbox_thr,
            nms_thr=self.nms_thr
        )

        # 2) Create the MoGe model if paths are set
        _moge_inference = None
        if self.moge_model_path:
            try:
                _moge_inference = MoGeInference(
                    model_path=self.moge_model_path,
                    root_folder=self.moge_root_folder,
                    device=self.device
                )
                self.logger.info(f"[INFO] Creating MoGeInference {MoGeInference} in worker_init_fn. \n\t{self.moge_model_path}")
            except Exception as e:
                raise BrokenPipeError(f"[ERROR] Failed to init MoGeInference in worker_init_fn: {e}")
                _moge_inference = None

    def run_server_forever(self):
        """
        A separate method you call *after* __init__ is done.
        """
        self.logger.info("Starting server loop now.")
        self.server.serve_forever(runner=None)

    def on_data_handler(self, *args, **kwargs):
        self.worker_init_fn()  # ensure each request sees correct model init
        global _pose_estimator
        global _moge_inference
        data = args[0]
        erode = data.get("erode", 40)
        np_image = data.get("img", None)

        self.logger.info(f"[INFO] Erode kernel size: {erode}")
        # 1) Parse the request data for the NumPy image
        if np_image is None:
            return "No image data provided"

        # 2) Separate color vs alpha
        if np_image.ndim == 3 and np_image.shape[2] == 4:
            color_image = np_image[..., :3]  # keep first 3 channels
            alpha = np_image[..., 3]
            # Erode alpha with a ~5-pixel elliptical kernel
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode, erode))
            alpha = cv2.erode(alpha, kernel, iterations=1)
        else:
            color_image = np_image
            alpha = None

        start_time = time.time()

        # 3) Pose detection
        data_samples, bboxes, pose_results = _pose_estimator.detect_pose(
            color_image, bbox_thr=self.bbox_thr, nms_thr=self.nms_thr
        )
        pose_time = time.time() - start_time
        print(f"[INFO] Pose detection took {pose_time:.3f}s.")

        # 4) If alpha exists, build KD‐Tree of alpha>0 coords & snap keypoints
        if alpha is not None:
            alpha_coords = np.argwhere(alpha > 0.1)  # shape (N, 2) => [y, x]
            self.logger.info(f"Found {len(alpha_coords)} alpha>0 pixels.")
            if alpha_coords.size > 0:
                tree = cKDTree(alpha_coords)
                for single_result in pose_results:
                    if not hasattr(single_result, 'pred_instances'):
                        continue
                    pred_inst = single_result.pred_instances
                    if not hasattr(pred_inst, 'keypoints'):
                        continue

                    # keypoints shape can be:
                    #  - (N, K, 2) for multi-instance
                    #  - (K, 2) if for some reason MMPose flattened it
                    kpts_2d = pred_inst.keypoints

                    if len(kpts_2d.shape) == 3:
                        # shape (N, K, 2). Snap each instance, each joint.
                        num_inst = kpts_2d.shape[0]
                        num_joints = kpts_2d.shape[1]
                        for i in range(num_inst):
                            for j in range(num_joints):
                                x_f, y_f = kpts_2d[i, j]
                                dist, idx = tree.query([y_f, x_f], k=1)
                                nearest_y, nearest_x = alpha_coords[idx]
                                kpts_2d[i, j, 0] = nearest_x
                                kpts_2d[i, j, 1] = nearest_y
                    else:
                        # shape (K, 2). Single instance but no leading dimension
                        num_joints = kpts_2d.shape[0]
                        for j in range(num_joints):
                            x_f, y_f = kpts_2d[j]
                            dist, idx = tree.query([y_f, x_f], k=1)
                            nearest_y, nearest_x = alpha_coords[idx]
                            kpts_2d[j, 0] = nearest_x
                            kpts_2d[j, 1] = nearest_y

                    # Assign back with the same shape
                    pred_inst.keypoints = kpts_2d

        # Convert bounding boxes + keypoints for JSON
        results_dict = {
            "bboxes": bboxes.tolist(),
            "pose_results": [],
            "3d_points": []
        }

        # Dump 2D keypoints to lists
        for single_result in pose_results:
            if hasattr(single_result.pred_instances, "keypoints"):
                kpts = single_result.pred_instances.keypoints.tolist()
                results_dict["pose_results"].append(kpts)
            else:
                results_dict["pose_results"].append([])

        # 5) Optionally run MoGe
        if _moge_inference:
            try:
                moge_start = time.time()
                moge_output = _moge_inference.infer(
                    color_image, resolution_level=9, apply_mask=False
                )

                points_3d = moge_output.points  # (H, W, 3)
                # sample 3D coords at each 2D joint
                kpts_3d = sample_3d_keypoints(points_3d, pose_results)
                results_dict["3d_points"] = kpts_3d

                moge_time = time.time() - moge_start
                print(f"[INFO] Pose detection took {pose_time:.3f}s, MoGe took {moge_time:.3f}s.")
            except Exception as e:
                print("[ERROR] MoGe inference failed:", e)
                results_dict["3d_points"] = []
        else:
            raise BrokenPipeError(
                "[ERROR] MoGe was not initialized in worker. "
                "Please check your MoGe config and paths.")
            print("[INFO] MoGe was not initialized in worker.")
            results_dict["3d_points"] = []

        elapsed_time = time.time() - start_time
        print(f"[INFO] Total on_data_handler time: {elapsed_time:.3f}s")

        return results_dict

    def run_server(self, *args, **kwargs):
        self.server.run_forever(*args, **kwargs)


if __name__ == '__main__':
    executor = PoseEstimationServerExecutor()
    # Optionally adjust logger level
    lvl = executor.args_dict.get('logger_level', 20)
    executor.logger.setLevel(int(lvl))
    executor.run_server_forever()
    print('Server should be running. Press Ctrl+C to quit.')
    try:
        while executor.server._server_running:
            pass
    except KeyboardInterrupt:
        pass
    executor.server.stop_server()
    print('Server script is now exiting.')
