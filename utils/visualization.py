from functools import partial
import os
import glob
import time
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np

WIDTH = 1000
HEIGHT = 1000
POINTSIZE = 1.5
SLEEPTIME = 0.3


def get_car_model(filename):
    """Car model for visualization

    Args:
        filename (str): filename of mesh

    Returns:
        mesh: open3D mesh
    """
    mesh = o3d.io.read_triangle_mesh(filename)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    return mesh


def get_filename(path, idx):
    filenames = sorted(glob.glob(path + "*.ply"))
    return int(filenames[idx].split(".")[0].split("/")[-1])


def last_file(path):
    return get_filename(path, -1)


def first_file(path):
    return get_filename(path, 0)

class Visualization:
    """Visualization of point cloud predictions with open3D"""

    def __init__(
        self,
        path,
        sequence,
        start,
        end,
        capture=False,
        path_to_car_model=None,
        sleep_time=5e-3,
    ):
        """Init

        Args:
            path (str): path to data should be
              .
              ├── sequence
              │   ├── gt
              |   |   ├──frame.ply
              │   ├─── pred
              |   |   ├── frame
              |   │   |   ├─── (frame+1).ply

            sequence (int): Sequence to visualize
            start (int): Start at specific frame
            end (int): End at specific frame
            capture (bool, optional): Save to file at each frame. Defaults to False.
            path_to_car_model (str, optional): Path to car model. Defaults to None.
            sleep_time (float, optional): Sleep time between frames. Defaults to 5e-3.
        """
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(width=WIDTH, height=HEIGHT)
        self.render_options = self.vis.get_render_option()
        self.render_options.point_size = POINTSIZE
        self.capture = capture

        # Load car model
        if path_to_car_model:
            self.car_mesh = get_car_model(path_to_car_model)
        else:
            self.car_mesh = None

        # Path and sequence to visualize
        self.path = path
        self.sequence = sequence

        # Frames to visualize
        self.start = start
        self.end = end

        # Init
        self.current_frame = self.start
        self.current_step = 1
        self.n_pred_steps = 5 

        # Save last view
        self.ctr = self.vis.get_view_control()
        self.camera = self.ctr.convert_to_pinhole_camera_parameters()
        self.viewpoint_path = os.path.join(self.path, "viewpoint.json")

        self.print_help()
        self.update(self.vis)

        # Continuous time plot
        self.stop = False
        self.sleep_time = sleep_time

        # Initialize the default callbacks
        self._register_key_callbacks()

        self.last_time_key_pressed = time.time()
    
    def prev_frame(self, vis):
        if time.time() - self.last_time_key_pressed > SLEEPTIME:
            self.last_time_key_pressed = time.time()
            self.current_frame = max(self.start, self.current_frame - 1)
            self.update(vis)
        return False

    def next_frame(self, vis):
        if time.time() - self.last_time_key_pressed > SLEEPTIME:
            self.last_time_key_pressed = time.time()
            self.current_frame = min(self.end, self.current_frame + 1)
            self.update(vis)
        return False
    
    def prev_prediction_step(self, vis):
        if time.time() - self.last_time_key_pressed > SLEEPTIME: 
            self.last_time_key_pressed = time.time()
            self.current_step = max(1, self.current_step - 1)
            self.update(vis)
        return False

    def next_prediction_step(self, vis):
        if time.time() - self.last_time_key_pressed > SLEEPTIME:
            self.last_time_key_pressed = time.time()
            self.current_step = min(self.n_pred_steps, self.current_step + 1)
            self.update(vis)
        return False
    
    def play_sequence(self, vis):
        pass
