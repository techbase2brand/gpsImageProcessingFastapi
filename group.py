import cv2
import numpy as np
import rasterio
from sklearn.cluster import DBSCAN

# Try to load thinning module for better line detection
try:
    from cv2 import ximgproc
    USE_THINNING = True
except ImportError:
    USE_THINNING = False

def run_group_detection(tif_path, selection_poly):
    """
    Analyzes a GeoTIFF, detects parking stripes, and groups them into rows.
    Returns a list of polygons (each polygon is a list of [x, y] coordinates).
    """
    # 1. Load and Preprocess Image
    with rasterio.open(tif_path) as src:
        # Read RGB bands and transpose to (H, W, C)
        img_bgr = src.read([1, 2, 3]).transpose((1, 2, 0))
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to highlight white/yellow lines
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    if USE_THINNING:
        # Reduce lines to 1-pixel thickness for cleaner geometry
        thin = ximgproc.thinning(255 - binary, thinningType=cv2.ximgproc.THINNING_GUOHALL)
        _, bw_final = cv2.threshold(thin, 1, 255, cv2.THRESH_BINARY_INV)
    else:
        _, bw_final = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # 2. Apply Spatial Masking
    mask = np.zeros(bw_final.shape, dtype=np.uint8)
    # Convert input list of tuples to a numpy array for OpenCV
    poly_array = np.array(selection_poly, dtype=np.int32)
    cv2.fillPoly(mask, [poly_array], 255)
    
    # Isolate the lines within the user-defined polygon
    target = cv2.bitwise_and(255 - bw_final, mask)

    # 3. Morphological Filtering
    # Use a vertical kernel to bridge small gaps in parking lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
    refined = cv2.morphologyEx(target, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    stripes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50: continue  # Filter out noise

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect
        
        # Ensure we are capturing elongated shapes (stripes)
        actual_h = max(w, h)
        if actual_h > 20:
            stripes.append({'pts': cnt, 'cx': cx, 'cy': cy, 'angle': angle})

    if len(stripes) < 3:
        return []

    # 4. Smart Projection and Clustering (DBSCAN)
    # Use median angle to align the projection
    angles = [s['angle'] for s in stripes]
    avg_angle = np.median(angles)
    
    theta = np.radians(avg_angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    coords = np.array([[s['cx'], s['cy']] for s in stripes])
    
    # Project coordinates perpendicular to the stripe angle to group them into rows
    dist_perp = -coords[:, 0] * sin_t + coords[:, 1] * cos_t
    
    # eps=50 is the pixel distance between different parking rows
    y_cluster = DBSCAN(eps=50, min_samples=2).fit(dist_perp.reshape(-1, 1))

    final_group_polygons = []
    for lbl in np.unique(y_cluster.labels_):
        if lbl == -1: continue # Ignore noise
        
        row_pts = []
        for i, label in enumerate(y_cluster.labels_):
            if label == lbl:
                row_pts.extend(stripes[i]['pts'].reshape(-1, 2))
        
        if row_pts:
            row_pts = np.array(row_pts, np.float32)
            # Create a bounding box around the entire detected row
            hull = cv2.convexHull(row_pts)
            rect = cv2.minAreaRect(hull)
            box = cv2.boxPoints(rect)
            # Convert to list of lists for JSON serializability
            final_group_polygons.append(box.astype(int).tolist())

    return final_group_polygons