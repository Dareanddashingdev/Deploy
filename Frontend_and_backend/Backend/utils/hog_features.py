import cv2
import numpy as np
from skimage.feature import hog
def extract_hog_features_from_image(image, resize_dim=(32, 32), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    """
    Extract HOG features from an image.
    Assumes image is in BGR format (like from cv2.imread).
    """
    # Resize image
    resized = cv2.resize(image, resize_dim)

    # Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Extract HOG features
    hog_features = hog(
        gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm='L2-Hys',
        visualize=False
    )
    return hog_features
def extract_hog_features_from_path(image_path, resize_dim=(32, 32)):
    """
    Load image from path and extract HOG features.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    return extract_hog_features_from_image(image, resize_dim=resize_dim)
