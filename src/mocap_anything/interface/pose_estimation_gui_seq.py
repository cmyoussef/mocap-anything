import datetime
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton,
    QLineEdit, QVBoxLayout, QHBoxLayout, QFileDialog, QLabel, QSizePolicy,
    QSlider, QSpinBox, QCheckBox
)

from PyServerManager.core.logger import logger
from PyServerManager.async_server.base_async_pickle import BaseAsyncPickle

# MMPose dataset info
from QuadCap.configs._base_.datasets.ap10k import dataset_info
from QuadCap.handlers import SkeletonBuilder

# A tiny helper to guess if a file is a sequence vs single:
from QuadCap.utils.file_utils import get_image_pattern_and_frame_range

# The "server_pose_executor.py" defines PoseEstimationUserServerExecutor
from QuadCap.interface.pose_estimation_server_executor import PoseEstimationUserServerExecutor

logger.setLevel("INFO")

# ---------------------------------------------------
# Metadata
# ---------------------------------------------------
keypoint_info = dataset_info["keypoint_info"]
max_kpt_index = max(keypoint_info.keys())
keypoint_metadata = [keypoint_info[i] for i in range(max_kpt_index + 1)]
name_to_id = {v["name"]: k for k, v in keypoint_info.items()}
skeleton_info = dataset_info["skeleton_info"]


# ---------------------------------------------------
# A single label that supports drop and drag
# ---------------------------------------------------
class ImageDropLabel(QLabel):
    """
    A QLabel that can accept drag-and-drop of image files,
    display them scaled while preserving aspect ratio,
    and store both raw bytes and a NumPy array version.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.setText("Drag & Drop an image here")
        self.setAcceptDrops(True)

        self.parent_ = parent
        self._originalPixmap = None
        self._image_data = None
        self._np_image = None

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return
        file_path = urls[0].toLocalFile()
        if file_path and self.parent_:
            self.parent_.on_new_image_loaded(file_path)

    def load_image(self, file_path: str, max_width=None):
        if not os.path.isfile(file_path):
            return
        with open(file_path, 'rb') as f:
            raw = f.read()
        self._image_data = raw

        arr = np.frombuffer(raw, np.uint8)
        # IMREAD_UNCHANGED to keep alpha if it exists
        self._np_image = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

        # Optional downscale
        if max_width is not None and self._np_image is not None:
            h, w = self._np_image.shape[:2]
            if w > max_width:
                scale_factor = max_width / float(w)
                new_w = max_width
                new_h = int(h * scale_factor)
                self._np_image = cv2.resize(
                    self._np_image, (new_w, new_h), interpolation=cv2.INTER_AREA
                )

        # For display, we’ll just use the raw bytes
        pixmap = QPixmap()
        if pixmap.loadFromData(raw) and not pixmap.isNull():
            self._originalPixmap = pixmap
            self.updateScaledPixmap()
        else:
            self._originalPixmap = None
            self.setText("Failed to load image")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.updateScaledPixmap()

    def updateScaledPixmap(self):
        if not self._originalPixmap:
            return
        scaled = self._originalPixmap.scaled(
            self.size() * 0.99,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.setPixmap(scaled)

    def get_image_as_numpy(self):
        """Return the raw NumPy array, possibly BGRA if alpha is present."""
        return self._np_image

    def overlay_pose_points(self, pose_data):
        """
        Draw skeleton lines and keypoints on top of the original image.
        If the original has alpha, we also draw RGBA lines with alpha=255.
        """
        if self._image_data is None:
            print("[ImageDropLabel] No image data to overlay.")
            return

        arr = np.frombuffer(self._image_data, np.uint8)
        np_image = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

        has_alpha = (np_image.ndim == 3 and np_image.shape[2] == 4)
        # If the image has alpha, we’ll draw lines and text with alpha=255
        # We'll unify color as RGBA if has_alpha, otherwise BGR

        pose_results = pose_data.get('pose_results', [])
        points_3d = pose_data.get('3d_points', [])

        # (A) Draw skeleton lines
        for batch_idx, batch_instances in enumerate(pose_results):
            for inst_idx, instance_poses in enumerate(batch_instances):
                for _, skeleton_item in skeleton_info.items():
                    link = skeleton_item["link"]  # (nameA, nameB)
                    color_rgb = skeleton_item["color"]  # [R, G, B]
                    nameA, nameB = link

                    if nameA not in name_to_id or nameB not in name_to_id:
                        continue
                    idxA = name_to_id[nameA]
                    idxB = name_to_id[nameB]
                    if idxA < len(instance_poses) and idxB < len(instance_poses):
                        xA, yA = instance_poses[idxA]
                        xB, yB = instance_poses[idxB]

                        if has_alpha:
                            color_draw = (color_rgb[2], color_rgb[1], color_rgb[0], 255)
                        else:
                            color_draw = (color_rgb[2], color_rgb[1], color_rgb[0])

                        cv2.line(
                            np_image,
                            (int(xA), int(yA)),
                            (int(xB), int(yB)),
                            color_draw, 2
                        )

        # (B) Draw keypoints
        for batch_idx, batch_instances in enumerate(pose_results):
            for inst_idx, instance_poses in enumerate(batch_instances):
                for kp_idx, (x, y) in enumerate(instance_poses):
                    if 0 <= kp_idx < len(keypoint_metadata):
                        kpt_info = keypoint_metadata[kp_idx]
                        kpt_name = kpt_info["name"]
                        color_rgb = kpt_info["color"]
                    else:
                        kpt_name = f"joint_{kp_idx}"
                        color_rgb = [0, 255, 0]

                    if has_alpha:
                        color_draw = (color_rgb[2], color_rgb[1], color_rgb[0], 255)
                    else:
                        color_draw = (color_rgb[2], color_rgb[1], color_rgb[0])

                    cv2.circle(np_image, (int(x), int(y)), 5, color_draw, -1)
                    cv2.putText(
                        np_image,
                        kpt_name,
                        (int(x) + 5, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color_draw, 2
                    )

                # (Optional) printing 3D coords
                if batch_idx < len(points_3d):
                    batch_3d = points_3d[batch_idx]
                    if inst_idx < len(batch_3d):
                        inst_3d = batch_3d[inst_idx]
                        print(f"[DEBUG] 3D coords for instance {inst_idx}: {inst_3d}")

        # (C) Convert updated image to PNG for display
        ret, buffer = cv2.imencode('.png', np_image)
        if not ret:
            print("[ImageDropLabel] Failed to encode overlay image.")
            return

        updated_pixmap = QPixmap()
        if updated_pixmap.loadFromData(buffer.tobytes()):
            self._originalPixmap = updated_pixmap
            self.updateScaledPixmap()
        else:
            self.setText("[ImageDropLabel] Failed to display updated overlay.")


# ---------------------------------------------------
# A single worker that always loops start_frame..end_frame
# ---------------------------------------------------
class PoseEstimationWorker(QThread):
    frameResult = Signal(dict)
    progressUpdate = Signal(int, int)
    finishedAll = Signal()

    def __init__(self, server_exec, sequence_pattern, start_frame, end_frame,
                 skip_exist, erode, pose_cache=None, parent=None):
        super().__init__(parent)
        self.server_exec = server_exec
        self.sequence_pattern = sequence_pattern
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.skip_exist = skip_exist
        self.erode = erode
        self.pose_cache = pose_cache if pose_cache else {}

        self._cancel = False  # internal flag to indicate we should stop

    def cancel(self):
        """Called by the GUI to tell the worker to stop early."""
        self._cancel = True

    def run(self):
        total = self.end_frame - self.start_frame + 1
        for i, fr in enumerate(range(self.start_frame, self.end_frame + 1), 1):
            # Check if user canceled:
            if self._cancel:
                print("[PoseEstimationWorker] Canceling loop by user request.")
                break

            # If skip is on, check cache
            if self.skip_exist and (fr in self.pose_cache):
                response = {"frame_number": fr, "skipped": True}
                self.frameResult.emit(response)
                self.progressUpdate.emit(fr, total)
                continue

            # Build path
            try:
                path = self.sequence_pattern % fr
            except TypeError:
                # single-file
                path = self.sequence_pattern

            if not os.path.exists(path):
                print(f"[PoseEstimationWorker] Missing file: {path}, skipping.")
                self.progressUpdate.emit(fr, total)
                continue

            with open(path, 'rb') as f:
                raw = f.read()
            arr = np.frombuffer(raw, np.uint8)
            np_image = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if np_image is None:
                print(f"[PoseEstimationWorker] Failed to decode {path}, skipping.")
                self.progressUpdate.emit(fr, total)
                continue

            # Send to server
            response = self.server_exec.send_data_to_server({
                'img': np_image,
                'erode': self.erode
            })
            response["frame_number"] = fr

            self.frameResult.emit(response)
            self.progressUpdate.emit(fr, total)

        # Once we exit (normal or canceled), emit finished
        self.finishedAll.emit()



# ---------------------------------------------------
# Main GUI
# ---------------------------------------------------
class PoseEstimationGUI(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MocapAnything GUI")
        self.is_processing_frames = False  # track if worker is running

        # -- Layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(self)
        central_widget.setLayout(main_layout)

        # Row: host/port
        row_host_port = QHBoxLayout(self)
        self.host_edit = QLineEdit("127.0.0.1")
        self.port_edit = QLineEdit("50716")
        btn_find_port = QPushButton("Find Available Port")
        btn_find_port.clicked.connect(self.on_find_port)
        row_host_port.addWidget(self.host_edit)
        row_host_port.addWidget(self.port_edit)
        row_host_port.addWidget(btn_find_port)
        main_layout.addLayout(row_host_port)

        # Buttons to start/connect
        self.btn_start_server = QPushButton("Start Server")
        self.btn_start_server.clicked.connect(self.on_start_server)
        self.btn_connect_client = QPushButton("Connect to Server")
        self.btn_connect_client.clicked.connect(self.on_connect_to_server)
        main_layout.addWidget(self.btn_start_server)
        main_layout.addWidget(self.btn_connect_client)

        # Image drop label
        self.image_label = ImageDropLabel(self)
        main_layout.addWidget(self.image_label)

        # Browse
        self.btn_browse = QPushButton("Browse Image")
        self.btn_browse.clicked.connect(self.on_browse_image)
        main_layout.addWidget(self.btn_browse)

        # Single-frame
        self.btn_estimate_pose = QPushButton("Estimate Pose (Single Frame)")
        self.btn_estimate_pose.clicked.connect(self.on_estimate_pose)
        self.btn_estimate_pose.setEnabled(False)
        main_layout.addWidget(self.btn_estimate_pose)

        # Erode spin
        row_erode = QHBoxLayout()
        row_erode.addWidget(QLabel("Erode (px):"))
        self.pixel_spin_box = QSpinBox()
        self.pixel_spin_box.setValue(5)
        row_erode.addWidget(self.pixel_spin_box)
        main_layout.addLayout(row_erode)

        # Skip existing
        self.skip_checkbox = QCheckBox("Skip Existing")
        self.skip_checkbox.setChecked(True)
        main_layout.addWidget(self.skip_checkbox)

        # Sequence slider
        seq_layout = QHBoxLayout()
        seq_layout.addWidget(QLabel("Timeline:"))
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(1, 1)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.on_slider_value_changed)
        seq_layout.addWidget(self.frame_slider)

        self.frame_label = QLabel("Frame: -")
        seq_layout.addWidget(self.frame_label)
        main_layout.addLayout(seq_layout)

        # Process all
        self.btn_process_all = QPushButton("Process All Frames")
        self.btn_process_all.setEnabled(False)
        self.btn_process_all.clicked.connect(self.on_process_all_clicked)
        main_layout.addWidget(self.btn_process_all)

        # JSON
        main_layout.addWidget(QLabel("JSON Output Path:"))
        self.json_path_edit = QLineEdit()
        self.json_path_edit.setPlaceholderText("Optional path for JSON output")
        main_layout.addWidget(self.json_path_edit)

        # Setup server stuff
        self.server_exec = PoseEstimationUserServerExecutor(logger=logger)
        self._server_running = False

        self.skeleton_builder = SkeletonBuilder(callback_func=self.update_slider)

        # Sequence info
        self.sequence_pattern = None
        self.sequence_start = None
        self.sequence_end = None
        self.current_frame = None

        # Pose cache & JSON
        self.pose_cache = {}
        self.pose_json_dict = {}

        self.pose_worker = None

    # -----------
    # server
    # -----------
    def on_find_port(self):
        port = BaseAsyncPickle.find_available_port()
        self.port_edit.setText(str(port))

    def on_start_server(self):
        host = self.host_edit.text()
        port_str = self.port_edit.text()
        try:
            port = int(port_str)
        except ValueError:
            port = 5050

        base_dir = os.path.dirname(os.path.dirname(__file__))
        parent_dir = os.path.dirname(os.path.dirname(base_dir))
        config_dir = os.path.join(base_dir, 'configs')
        model_dir = os.path.join(parent_dir, 'checkpoints')

        moge_model_path = r"/checkpoints/model.pt"
        moge_root_folder = r"E:\ai_projects\dust3r_project\vision-forge\test\output_geo"

        logger.info(f"Starting server on {host}:{port}")
        self.server_exec.run_pose_server(
            host=host,
            port=port,
            open_new_terminal=True,
            config_dir=config_dir,
            model_dir=model_dir,
            num_workers=1,
            device="cuda:0",
            moge_model_path=moge_model_path,
            moge_root_folder=moge_root_folder,
        )
        self._server_running = True
        self.btn_estimate_pose.setEnabled(True)
        self.btn_start_server.setEnabled(False)

    def on_connect_to_server(self):
        self.server_exec.host = self.host_edit.text()
        self.server_exec.port = int(self.port_edit.text())
        self.server_exec.connect_client(start_sleep=2, retry_delay=2, max_retries=10)
        self.btn_estimate_pose.setEnabled(True)
        self.btn_start_server.setEnabled(False)
        self._server_running = True

    # -----------
    # load images
    # -----------
    def on_browse_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.on_new_image_loaded(file_path)

    def on_new_image_loaded(self, file_path: str):
        # Clear caches
        self.pose_cache.clear()
        self.pose_json_dict.clear()

        pattern, fr_range = get_image_pattern_and_frame_range(file_path)
        if fr_range is not None:
            self.sequence_pattern = pattern
            self.sequence_start, self.sequence_end = fr_range
            self.frame_slider.setRange(self.sequence_start, self.sequence_end)
            self.frame_slider.setValue(self.sequence_start)
            self.frame_slider.setEnabled(True)
            self.btn_process_all.setEnabled(True)
            self.current_frame = self.sequence_start
            self.frame_label.setText(f"Frame: {self.current_frame}")
            self.load_frame(self.current_frame)
            self.skeleton_builder.set_timeline_range(self.sequence_start, self.sequence_end)
        else:
            # Single file => treat it as 1-frame sequence
            self.sequence_pattern = file_path  # no '%d'
            self.sequence_start = 1
            self.sequence_end = 1
            self.frame_slider.setRange(1, 1)
            self.frame_slider.setEnabled(False)
            self.btn_process_all.setEnabled(False)
            self.current_frame = 1
            self.frame_label.setText("Frame: 1")

            # Actually display
            self.image_label.load_image(file_path)

    def load_frame(self, frame_number: int):
        # We attempt to build path:
        try:
            path = self.sequence_pattern % frame_number
        except TypeError:
            path = self.sequence_pattern  # single
        if not os.path.exists(path):
            print(f"File does not exist: {path}")
            return
        self.image_label.load_image(path)
        self.frame_label.setText(f"Frame: {frame_number}")
        self.current_frame = frame_number

        # If in cache, overlay
        if frame_number in self.pose_cache:
            self.image_label.overlay_pose_points(self.pose_cache[frame_number])

    def on_slider_value_changed(self, value: int):
        self.load_frame(value)
        self.skeleton_builder.update_current_frame(value)

    # -----------
    # Single frame
    # -----------
    def on_estimate_pose(self):
        if not self._server_running:
            print("Server not running!")
            return
        if self.sequence_pattern is None:
            print("No image loaded!")
            return

        skip_exist = self.skip_checkbox.isChecked()
        erode_value = self.pixel_spin_box.value()

        self.btn_estimate_pose.setEnabled(False)

        # We'll treat single frame as [start_frame=1, end_frame=1], same approach
        worker = PoseEstimationWorker(
            server_exec=self.server_exec,
            sequence_pattern=self.sequence_pattern,
            start_frame=self.current_frame,   # or just 1
            end_frame=self.current_frame,     # or just 1
            skip_exist=skip_exist,
            erode=erode_value,
            pose_cache=self.pose_cache
        )
        self.pose_worker = worker
        worker.frameResult.connect(self.handle_estimation_result)
        worker.finishedAll.connect(self.on_single_frame_finished)
        worker.start()

    def on_single_frame_finished(self):
        self.btn_estimate_pose.setEnabled(True)
        if self.pose_worker:
            self.pose_worker.deleteLater()
            self.pose_worker = None

    # -----------
    # Multi frames
    # -----------
    def on_process_all_clicked(self):
        """
        The same button handles starting or stopping processing.
        """
        if not self._server_running:
            print("Server not running!")
            return

        # If we're already processing => user wants to cancel
        if self.is_processing_frames:
            print("[GUI] User requested to cancel processing.")
            if self.pose_worker:
                self.pose_worker.cancel()
            return

        # Otherwise, let's start processing
        if not self.sequence_pattern or self.sequence_start is None or self.sequence_end is None:
            print("No sequence recognized (or single).")
            return

        skip_exist = self.skip_checkbox.isChecked()
        erode_value = self.pixel_spin_box.value()

        # Create and start the worker
        self.pose_worker = PoseEstimationWorker(
            server_exec=self.server_exec,
            sequence_pattern=self.sequence_pattern,
            start_frame=self.sequence_start,
            end_frame=self.sequence_end,
            skip_exist=skip_exist,
            erode=erode_value,
            pose_cache=self.pose_cache
        )
        self.pose_worker.frameResult.connect(self.handle_estimation_result)
        self.pose_worker.progressUpdate.connect(self.on_sequence_progress)
        self.pose_worker.finishedAll.connect(self.on_sequence_finished)

        print("[GUI] Starting multi-frame worker...")
        self.is_processing_frames = True
        self.btn_process_all.setText("Stop Processing")  # toggle button text

        self.pose_worker.start()

    def on_sequence_progress(self, current_frame, total_frames):
        print(f"[GUI] Processed frame {current_frame}/{total_frames}")

    def on_sequence_finished(self):
        print("[GUI] Sequence finished (or canceled).")
        self.is_processing_frames = False
        self.btn_process_all.setText("Process All Frames")

        if self.pose_worker:
            self.pose_worker.deleteLater()
            self.pose_worker = None

    # -----------
    # handle results
    # -----------
    def handle_estimation_result(self, response: dict):
        """
        Called on the main thread for each processed frame.
        We'll:
          - store in cache
          - overlay
          - build Maya skeleton
          - save JSON
        """
        frame_num = response.get("frame_number", None)
        skipped = response.get("skipped", False)
        if not skipped:
            # store
            self.pose_cache[frame_num] = response
            self.skeleton_builder.update_current_frame(frame_num)
            # reload
            self.load_frame(frame_num)
            # overlay
            self.image_label.overlay_pose_points(response)
            # build Maya skeleton
            self.skeleton_builder.build_maya_skeleton_from_pose_data(response, keypoint_metadata)
            # store in JSON
            self.pose_json_dict[frame_num] = response
            self.save_json()
        else:
            print(f"[GUI] Skipped frame {frame_num}")

    # -----------
    # JSON
    # -----------
    def save_json(self):
        user_json_path = self.json_path_edit.text().strip()
        if user_json_path == "":
            home_dir = os.path.expanduser("~")
            day_str = datetime.datetime.now().strftime("%Y%m%d")
            if '%' in (self.sequence_pattern or ''):
                base_name = Path(self.sequence_pattern).stem
                base_name = base_name.replace("%", "").replace("0", "").replace("d", "")
                base_name = base_name.rstrip(".")
            else:
                base_name = Path(self.sequence_pattern or 'single_image').stem
            folder = os.path.join(home_dir, "mocapAnything", day_str)
            os.makedirs(folder, exist_ok=True)
            user_json_path = os.path.join(folder, f"{base_name}.json")

        try:
            with open(user_json_path, "w") as f:
                json.dump(self.pose_json_dict, f, indent=2)
            print(f"[GUI] JSON updated => {user_json_path}")
        except Exception as e:
            print(f"[GUI] Failed to save JSON: {e}")

    def update_slider(self, value):
        """If the skeleton builder calls back with a new frame number."""
        if value is None:
            return
        self.blockSignals(True)
        self.frame_slider.setValue(value)
        self.load_frame(value)
        self.blockSignals(False)


def main():
    app = QApplication(sys.argv)
    win = PoseEstimationGUI()

    win.resize(900, 600)
    win.on_new_image_loaded(r'E:\ai_projects\poseEstimation\quad_cap\assets\deerA\init_img.0120.png')
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
