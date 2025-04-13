import cv2
import numpy as np
def extract_color_histogram_from_image(image, bins=8):
    """
    Extract 24D color histogram feature vector from a given RGB image.
    """
    # Compute histograms for R, G, B channels
    hist_r = cv2.calcHist([image], [0], None, [bins], [0, 256]).flatten()
    hist_g = cv2.calcHist([image], [1], None, [bins], [0, 256]).flatten()
    hist_b = cv2.calcHist([image], [2], None, [bins], [0, 256]).flatten()

    # Normalize histograms
    hist_r /= hist_r.sum() if hist_r.sum() > 0 else 1
    hist_g /= hist_g.sum() if hist_g.sum() > 0 else 1
    hist_b /= hist_b.sum() if hist_b.sum() > 0 else 1

    # Concatenate features
    features = np.concatenate([hist_r, hist_g, hist_b])
    return features
def extract_color_histogram_from_path(image_path, bins=8):
    """
    Load image from path, convert to RGB, resize, and extract histogram.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.resize(image, (100, 100))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return extract_color_histogram_from_image(image, bins=bins)
