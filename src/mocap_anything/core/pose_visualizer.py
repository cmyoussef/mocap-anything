import mmcv
from mmpose.registry import VISUALIZERS

class PoseVisualizer:
    """
    A helper class to visualize pose estimation results.
    """

    def __init__(self, pose_estimator):
        """
        Build and configure the internal visualizer using the given pose_estimator.

        Args:
            pose_estimator: An initialized MMPose model from which to get config and metadata.
        """
        # Configure some default drawing params (adjust as needed)
        pose_estimator.cfg.visualizer.radius = 3
        pose_estimator.cfg.visualizer.line_width = 1

        # Build the visualizer from the pose estimator's config
        self.visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
        self.visualizer.set_dataset_meta(pose_estimator.dataset_meta)

    def visualize_pose(self, img_path, data_samples, show=False, out_file=None):
        """
        Visualize the predicted keypoints (and optionally bounding boxes) on the image.

        Args:
            img_path (str): Path to the input image.
            data_samples (DataSample or merged DataSample):
                Data sample(s) containing the pose predictions.
            show (bool, optional): Whether to show the image in a window. Defaults to False.
            out_file (str, optional): Output file path to save the visualization. Defaults to None.

        Returns:
            np.ndarray: The resulting annotated image in BGR format.
        """
        # Read the image in BGR format
        img = mmcv.imread(img_path, channel_order='bgr')

        # Add the data sample to the visualizer
        self.visualizer.add_datasample(
            'pose_result',
            img,
            data_sample=data_samples,
            draw_gt=True,
            draw_heatmap=True,
            draw_bbox=True,
            show=show,
            wait_time=0,
            out_file=None,   # we don't save inside the visualizer itself
            kpt_thr=0.0      # Adjust keypoint score threshold if desired
        )

        # Get the rendered image
        vis_img = self.visualizer.get_image()

        # If requested, save the rendered image
        if out_file:
            mmcv.imwrite(vis_img, out_file)

        return vis_img
