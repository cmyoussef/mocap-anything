import os
import sys

paths = [r'E:\stable-diffusion\stable-diffusion-integrator', r'E:\track_anything_project',
         r'E:\ai_projects\dust3r_project\vision-forge', r'E:\nuke-bridge', 'E:/ai_projects/ai_portal',
         r'E:\ai_projects\dust3r_project\MoGo', r'E:\ai_projects\dust3r_project\MoGo\MoGe']
for p in paths:
    if not p in sys.path:
        sys.path.append(p)

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image
from typing import *
from MoGe import utils3d
from MoGe.moge.model import MoGeModel
from nukebridge.utils.image_io import get_image_io

# Enable EXR support in OpenCV
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'


@dataclass
class MoGeOutput:
    """
    Data class to store the output of the MoGe model inference.
    """
    points: np.ndarray  # Shape: (H, W, 3)
    depth: np.ndarray  # Shape: (H, W)
    mask: Optional[np.ndarray]  # Shape: (H, W)
    intrinsics: np.ndarray  # Shape: (3, 3)
    image: np.ndarray  # Original input image (H, W, 3)
    root_folder: Path  # Root folder for saving outputs
    sensor_width_mm: float = 36.0  # Default to full-frame sensor width
    sensor_height_mm: float = 24.0  # Default to full-frame sensor height
    shift: float = 0  # Add shift value
    imageIO = get_image_io()

    def estimate_focal_length_from_frustum(self):
        """
        Estimates the focal length based on the frustum dimensions.

        Returns:
        - focal_length_x_px: Estimated focal length along the X-axis in pixels.
        - focal_length_y_px: Estimated focal length along the Y-axis in pixels.
        """
        # Get image dimensions
        image_height, image_width = self.image.shape[:2]

        # Get sensor dimensions
        sensor_width_mm = self.sensor_width_mm
        sensor_height_mm = self.sensor_height_mm

        # Get the near face vertices
        _, near_face_vertices = self.get_camera_frustum_faces()

        # Compute the width and height of the near face
        W_near = np.linalg.norm(near_face_vertices[0, :2] - near_face_vertices[3, :2])
        H_near = np.linalg.norm(near_face_vertices[0, :2] - near_face_vertices[1, :2])

        # Distance from camera to near plane (since camera is at z = -shift)
        z_near = near_face_vertices[0, 2] + self.shift

        # Compute field of view angles
        theta_x = 2 * np.arctan((W_near / 2) / z_near)
        theta_y = 2 * np.arctan((H_near / 2) / z_near)

        # Compute focal lengths in millimeters
        focal_length_x_mm = (sensor_width_mm / 2) / np.tan(theta_x / 2)
        focal_length_y_mm = (sensor_height_mm / 2) / np.tan(theta_y / 2)

        # Convert focal lengths to pixels
        focal_length_x_px = focal_length_x_mm * (image_width / sensor_width_mm)
        focal_length_y_px = focal_length_y_mm * (image_height / sensor_height_mm)

        print(f"Estimated focal lengths: fx = {focal_length_x_mm:.2f} mm, fy = {focal_length_y_mm:.2f} mm")
        print(f"Estimated focal lengths: fx = {focal_length_x_px:.2f} px, fy = {focal_length_y_px:.2f} px")

        return focal_length_x_px, focal_length_y_px

    def get_face_at_depth(self, depth_percentage=100.0, depth_tolerance=5.0):
        """
        Computes the face of the point cloud in the XY plane at the specified depth percentage,
        considering a depth tolerance to create a slice.

        Parameters:
        - depth_percentage (float): A value between 0 and 100 representing the depth percentage.
        - depth_tolerance (float): A value between 0 and 100 representing the tolerance percentage.

        Returns:
        - face_vertices (np.ndarray): An array of four vertices defining the face at the specified depth.
        """
        if not (0 <= depth_percentage <= 100):
            raise ValueError("depth_percentage must be between 0 and 100.")

        if not (0 <= depth_tolerance <= 100):
            raise ValueError("depth_tolerance must be between 0 and 100.")

        # Compute the minimum and maximum Z values
        z_min = np.min(self.points[:, :, 2])  # Nearest point
        z_max = np.max(self.points[:, :, 2])  # Furthest point

        # Calculate the depth Z value at the specified percentage
        depth_z = z_min + (depth_percentage / 100.0) * (z_max - z_min)

        # Calculate the depth tolerance in Z units
        delta_z = (depth_tolerance / 100.0) * (z_max - z_min)

        # Define the depth range
        depth_min = depth_z - delta_z
        depth_max = depth_z + delta_z

        # Flatten points and select those within the depth range
        points_flat = self.points.reshape(-1, 3)
        within_depth = (points_flat[:, 2] >= depth_min) & (points_flat[:, 2] <= depth_max)

        depth_points = points_flat[within_depth]

        if depth_points.size == 0:
            raise ValueError("No points found within the specified depth range.")

        # Compute the maximum absolute X and Y values within the depth slice
        max_abs_x = np.max(np.abs(depth_points[:, 0]))
        max_abs_y = np.max(np.abs(depth_points[:, 1]))

        # Construct the face vertices symmetrically around the origin
        face_vertices = np.array([
            [max_abs_x, max_abs_y, depth_z],
            [max_abs_x, -max_abs_y, depth_z],
            [-max_abs_x, -max_abs_y, depth_z],
            [-max_abs_x, max_abs_y, depth_z],
        ])

        return face_vertices

    def get_camera_frustum_faces(self, step=2, depth_tolerance=0.5):
        """
        Generates the near and far faces of the camera frustum based on the point cloud.

        Parameters:
        - step (int): The step size for depth percentages (default is 2).
        - depth_tolerance (float): The tolerance percentage for depth slicing (default is 0.5).

        Returns:
        - far_face_vertices (np.ndarray): Vertices of the far face of the frustum.
        - near_face_vertices (np.ndarray): Vertices of the near face of the frustum.
        """
        # Get absolute Z values
        abs_z = np.abs(self.points[:, :, 2])
        z_min, z_max = np.min(abs_z), np.max(abs_z)

        # Initialize lists to store scaled vertices
        scaled_vertices_list = []

        # Loop through depth percentages
        for depth_percentage in range(0, 101, step):
            try:
                # Get the face vertices at the specified depth
                face_vertices = self.get_face_at_depth(depth_percentage, depth_tolerance)

                # Scale the vertices based on the depth percentage
                plan_depth = np.abs(face_vertices[0, 2])
                scale_factor = z_max / plan_depth if plan_depth != 0 else 1.0

                # Scale the vertices from the origin
                scaled_vertices = face_vertices.copy()
                scaled_vertices[:, 0] *= scale_factor
                scaled_vertices[:, 1] *= scale_factor

                # Collect the scaled vertices
                scaled_vertices_list.append(scaled_vertices)

            except ValueError as e:
                # Handle cases where no points are found at the depth
                continue

        if not scaled_vertices_list:
            raise ValueError("No valid faces were generated for the frustum.")

        # Combine all scaled vertices into one array
        all_scaled_vertices = np.vstack(scaled_vertices_list)

        # Compute the maximum absolute X and Y values among all scaled vertices
        max_abs_x = np.max(np.abs(all_scaled_vertices[:, 0]))
        max_abs_y = np.max(np.abs(all_scaled_vertices[:, 1]))

        # Use the furthest Z value (z_max) for the far plane
        z_far = z_max

        # Construct the far face vertices
        far_face_vertices = np.array([
            [max_abs_x, max_abs_y, z_far],
            [max_abs_x, -max_abs_y, z_far],
            [-max_abs_x, -max_abs_y, z_far],
            [-max_abs_x, max_abs_y, z_far],
        ])

        # Construct the near face vertices using the ratio of z_min to z_max
        z_near = z_min if z_min != 0 else 0.001  # Avoid division by zero
        scale_ratio = z_near / z_far
        near_face_vertices = far_face_vertices.copy()
        near_face_vertices[:, 0] *= scale_ratio
        near_face_vertices[:, 1] *= scale_ratio
        near_face_vertices[:, 2] = z_near

        return far_face_vertices, near_face_vertices

    def save_frustum_as_obj(self, output_path):
        """
        Generates the camera frustum mesh and saves it as an OBJ file.

        Parameters:
        - output_path (str or Path): Path to save the OBJ file.
        """
        # Get the near and far face vertices
        far_face_vertices, near_face_vertices = self.get_camera_frustum_faces()

        # Combine vertices
        vertices = np.vstack((near_face_vertices, far_face_vertices))

        # Flip the Z-axis to correct orientation
        vertices[:, 2] *= -1

        # Define faces (using zero-based indexing)
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # Near face
            [4, 5, 6], [4, 6, 7],  # Far face
            [0, 1, 5], [0, 5, 4],  # Side faces
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ])

        # Reverse the face vertex order to correct normals
        faces = faces[:, ::-1]

        # Create a Trimesh object
        frustum_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        # Save the frustum as OBJ
        output_path = self.root_folder / output_path if not os.path.isabs(output_path) else Path(output_path)
        output_path = output_path.with_suffix('.obj')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frustum_mesh.export(str(output_path))

        print(f"Frustum saved to {output_path}")

    def get_camera_parameters(self):
        """
        Compute camera parameters including focal lengths in mm and adjusted position and rotation.

        Returns:
        - camera_params (dict): Dictionary containing focal lengths in mm, position, rotation, and sensor size.
        """
        # Estimate focal lengths using the frustum
        focal_length_x_px, focal_length_y_px = self.estimate_focal_length_from_frustum()

        # Get image dimensions
        image_height, image_width = self.image.shape[:2]
        c_x = image_width / 2
        c_y = image_height / 2

        # Update intrinsics matrix
        self.intrinsics = np.array([
            [focal_length_x_px, 0, c_x],
            [0, focal_length_y_px, c_y],
            [0, 0, 1]
        ])

        # Convert focal lengths to millimeters
        focal_length_x_mm = focal_length_x_px * (self.sensor_width_mm / image_width)
        focal_length_y_mm = focal_length_y_px * (self.sensor_height_mm / image_height)

        # Camera position (no rotation needed)
        position = np.array([0.0, 0.0, 0.0])
        rotation = np.array([0.0, 0.0, 0.0])  # No rotation

        camera_params = {
            'focal_length_x_mm': focal_length_x_mm,
            'focal_length_y_mm': focal_length_y_mm,
            'position': position,
            'rotation': rotation,
            'sensor_width_mm': self.sensor_width_mm,
            'sensor_height_mm': self.sensor_height_mm
        }
        return camera_params

    def get_camera_data(self):
        """
        Extract camera data such as focal lengths and field of view from intrinsics.

        Returns:
        - camera_data (dict): Dictionary containing focal lengths and FOV.
        """
        fov_x, fov_y = utils3d.numpy.intrinsics_to_fov(self.intrinsics)
        focal_length_x = self.intrinsics[0, 0]
        focal_length_y = self.intrinsics[1, 1]

        camera_data = {
            'fov_x': np.rad2deg(fov_x),
            'fov_y': np.rad2deg(fov_y),
            'focal_length_x': focal_length_x,
            'focal_length_y': focal_length_y,
        }
        return camera_data

    def save_mesh(self, output_path, file_format='ply', remove_edge=True, rtol=0.02):
        """
        Create a mesh from points and save it as a PLY or OBJ file.

        Parameters:
        - output_path (str or Path): Path to save the mesh file (relative to root_folder or absolute).
        - file_format (str): File format for the mesh ('ply' or 'obj').
        - remove_edge (bool): Whether to remove edge points based on depth discontinuities.
        - rtol (float): Relative tolerance for edge removal.
        """
        points = self.points
        image = self.image
        mask = self.mask

        height, width = image.shape[:2]

        # Resolve output path
        output_path = self.root_folder / output_path if not os.path.isabs(output_path) else Path(output_path)
        output_path = output_path.with_suffix(f'.{file_format}')

        # Prepare mask
        if mask is None:
            mask = np.ones((height, width), dtype=bool)

        # Remove edges if requested
        if remove_edge:
            depth = points[..., 2]
            edge_mask = ~utils3d.numpy.depth_edge(depth, mask=mask, rtol=rtol)
            mask = mask & edge_mask

        # Generate mesh
        faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.image_mesh(
            points,
            image.astype(np.float32) / 255.0,
            utils3d.numpy.image_uv(width=width, height=height),
            mask=mask,
            tri=True
        )

        # Adjust coordinates and UVs
        vertices, vertex_uvs = vertices * [1, -1, -1], vertex_uvs * [1, -1] + [0, 1]

        # Create Trimesh object
        if file_format.lower() == 'ply':
            mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                vertex_colors=vertex_colors,
                process=False
            )
        else:
            # For OBJ or other formats, use texture mapping
            mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                visual=trimesh.visual.TextureVisuals(
                    uv=vertex_uvs,
                    image=Image.fromarray(image)
                ),
                process=False
            )

        # Save mesh
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(output_path))

    def save_depth_map(self, output_path):
        """
        Save the raw depth map as an EXR file.

        Parameters:
        - output_path (str or Path): Path to save the depth file (relative to root_folder or absolute).
        """
        depth = self.depth
        # Resolve output path
        output_path = self.root_folder / output_path if not os.path.isabs(output_path) else Path(output_path)
        exr_path = str(output_path.with_suffix('.exr'))
        # depth_exr = depth.astype(np.float32)
        self.imageIO.write_image(depth, exr_path, image_format='exr')

        # Ensure depth is single-channel float32

        # Save as EXR
        # Ensure OpenCV has EXR support enabled
        # if not cv2.haveImageWriter(exr_path):
        #     raise RuntimeError(
        #         "OpenCV was not built with EXR support. Please ensure OpenCV is installed with OpenEXR enabled.")

        # Save the depth map
        # cv2.imwrite(exr_path, depth_exr, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])

    def save_point_cloud(self, output_path):
        """
        Save the point cloud as a PLY file.

        Parameters:
        - output_path (str or Path): Path to save the point cloud file (relative to root_folder or absolute).
        """
        points = self.points
        mask = self.mask

        # Resolve output path
        output_path = self.root_folder / output_path if not os.path.isabs(output_path) else Path(output_path)
        output_path = output_path.with_suffix('.ply')

        # Flatten points and apply mask
        valid_points = points[mask] if mask is not None else points.reshape(-1, 3)

        # Create Trimesh PointCloud
        point_cloud = trimesh.PointCloud(valid_points)

        # Save point cloud
        output_path.parent.mkdir(parents=True, exist_ok=True)
        point_cloud.export(str(output_path))

    def save_camera_nuke(self, output_path):
        """
        Save camera parameters to a Nuke script file (.nk)

        Parameters:
        - output_path (str or Path): Path to save the Nuke script (relative to root_folder or absolute).
        """
        camera_params = self.get_camera_parameters()
        focal_length_mm = camera_params['focal_length_x_mm']
        haperture_mm = camera_params['sensor_width_mm']
        vaperture_mm = camera_params['sensor_height_mm']
        position = camera_params['position']
        rotation = camera_params['rotation']

        camera_nk_content = f"""
Camera2 {{
 inputs 0
 name Camera1
 focal {focal_length_mm}
 haperture {haperture_mm}
 vaperture {vaperture_mm}
 translate {{ {position[0]} {position[1]} {position[2]} }}
 rotate {{ {rotation[0]} {rotation[1]} {rotation[2]} }}
}}
"""
        # Resolve output path
        output_path = self.root_folder / output_path if not os.path.isabs(output_path) else Path(output_path)
        output_path = output_path.with_suffix('.nk')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(camera_nk_content)


class MoGeInference:
    def __init__(self, model_path, root_folder='.', device='cuda', sensor_width_mm=36.0, sensor_height_mm=24.0):
        # Existing initialization...
        self.sensor_width_mm = sensor_width_mm
        self.sensor_height_mm = sensor_height_mm
        """
        Initialize the MoGeInference class.

        Parameters:
        - model_path (str or Path): Path to the pre-trained MoGe model.
        - root_folder (str or Path): Root folder for saving outputs.
        - device (str): Device to load the model on ('cuda' or 'cpu').
        """
        self.device = torch.device(device)
        self.model = MoGeModel.from_pretrained(model_path).to(self.device).eval()
        self.root_folder = Path(root_folder)

    def infer(self, image, resolution_level=9, apply_mask=True):
        """
        Run the model inference on an input image.

        Parameters:
        - image (np.ndarray): Input image as a NumPy array (H, W, 3).
        - resolution_level (int): Resolution level for the inference (0-9).
        - apply_mask (bool): Whether to apply the output mask.

        Returns:
        - output (MoGeOutput): An object containing all inference outputs.
        """
        # Convert image to tensor
        image_tensor = torch.tensor(image / 255.0, dtype=torch.float32, device=self.device).permute(2, 0, 1)

        # Inference
        with torch.no_grad():
            output = self.model.infer(image_tensor, resolution_level=resolution_level, apply_mask=apply_mask)

        # Retrieve outputs
        points = output['points'].cpu().numpy()
        depth = output['depth'].cpu().numpy()
        mask = output['mask'].cpu().numpy() if 'mask' in output else None
        intrinsics = output['intrinsics'].cpu().numpy()
        shift = 0

        # Create MoGeOutput object
        inference_output = MoGeOutput(
            points=points,
            depth=depth,
            mask=mask,
            intrinsics=intrinsics,
            image=image,
            root_folder=self.root_folder,
            sensor_width_mm=self.sensor_width_mm,
            sensor_height_mm=self.sensor_height_mm,
            shift=shift  # Pass the shift value
        )

        return inference_output


if __name__ == '__main__':
    # Example usage
    model_path = r'/checkpoints/model.pt'
    root_folder = r'E:\ai_projects\dust3r_project\vision-forge\test\output_geo'
    inference = MoGeInference(model_path, root_folder=root_folder)
    img = r'E:\ai_projects\dust3r_project\MoGo\MoGe\example_images\BooksCorridor.png'
    img = r'C:\Users\Femto7000\Downloads\340045955Best-Indoor-Plants-Main.jpg'
    img = r'C:/Users/Femto7000/nukesd/SD_Txt2Img/crystalClearXL_ccxl.safetensors/20241116_115144/20241116_115144_1_1.0001.png'
    # img = r'I:\Recovered data 09-04 13_11_59\Actors\Jill Taylor\cover.jpg'
    image = Image.open(img)
    image_np = np.array(image.convert('RGB'))

    # camera_params = processor.get_camera_pose_and_parameters(result["intrinsics"])
    output = inference.infer(image_np, resolution_level=9, apply_mask=False)
    # print("camera_params:", result["intrinsics"])
    # Access camera data from the output
    camera_data = output.get_camera_data()
    print("Camera Data:", camera_data)

    # Save outputs directly from the output object
    output.save_mesh('meshes/mesh_filename', file_format='obj', remove_edge=True, rtol=0.02)
    output.save_camera_nuke('nuke_camera/camera')
    output.save_depth_map('depth_map/depth_map')
    # Save the frustum as OBJ
    # output.save_frustum_as_obj('frustum/frustum_mesh')

    # Get camera parameters
    camera_params = output.get_camera_parameters()

    # Save camera data for Nuke directly from the output object
    output.save_camera_nuke('nuke_camera/camera')

    print("Camera Parameters:")
    print(f"Focal Length X (mm): {camera_params['focal_length_x_mm']}")
    print(f"Focal Length Y (mm): {camera_params['focal_length_y_mm']}")
    print(f"Camera Position: {camera_params['position']}")
    print(f"Camera Rotation: {camera_params['rotation']}")
