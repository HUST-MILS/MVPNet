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