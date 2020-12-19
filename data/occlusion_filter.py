import numpy as np
import cv2


def filter_occlusions_with_parameters(raw_depth, distance_threshold=1.0, kernel_size=5):  # SWF original with parameters
    """
    Filter background LiDAR points at occlusions.

    :param distance_threshold: minimum distance between the fore and the background
    :param raw_depth: raw depth, {np.array}, shape=[H, W]
    :param kernel_size: the size of the kernel to filter: the max distance in the pixel space between a foreground and an occluded background, {int}
    :return: filtered raw depth, {np.array}, shape=[H, W]
    """
    assert distance_threshold > 0
    assert len(raw_depth.shape) == 2

    inverse_raw = np.where(raw_depth <= 0, np.ones_like(raw_depth) * 200, raw_depth)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # use erode to supress background LiDAR points at occlusions
    eroded_raw = np.array(cv2.erode(inverse_raw, kernel))
    raw_depth[np.abs(raw_depth - eroded_raw) > distance_threshold] = 0

    return raw_depth
