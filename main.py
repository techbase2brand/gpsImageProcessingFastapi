# from fastapi import FastAPI, HTTPException
# from fastapi.responses import JSONResponse
# import cv2
# import numpy as np
# import rasterio
# import os
# from sklearn.cluster import DBSCAN
# from raster import download_and_mask_arcgis

# app = FastAPI()

# # Check for OpenCV thinning support
# try:
#     from cv2 import ximgproc
#     USE_THINNING = True
# except ImportError:
#     USE_THINNING = False

# def run_group_detection(tif_path, selection_poly):
#     """Detects parking rows inside the selected pixel polygon."""
#     with rasterio.open(tif_path) as src:
#         img_bgr = src.read([1, 2, 3]).transpose((1, 2, 0))
#         gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

#     binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

#     if USE_THINNING:
#         thin = ximgproc.thinning(255 - binary, thinningType=cv2.ximgproc.THINNING_GUOHALL)
#         _, bw_final = cv2.threshold(thin, 1, 255, cv2.THRESH_BINARY_INV)
#     else:
#         _, bw_final = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

#     mask = np.zeros(bw_final.shape, dtype=np.uint8)
#     cv2.fillPoly(mask, [np.array(selection_poly, dtype=np.int32)], 255)
#     target = cv2.bitwise_and(255 - bw_final, mask)

#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
#     refined = cv2.morphologyEx(target, cv2.MORPH_CLOSE, kernel)
#     contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     stripes = []
#     for cnt in contours:
#         if cv2.contourArea(cnt) < 50: continue
#         rect = cv2.minAreaRect(cnt)
#         (cx, cy), (w, h), angle = rect
#         if max(w, h) > 20:
#             stripes.append({'pts': cnt, 'cx': cx, 'cy': cy, 'angle': angle})

#     if len(stripes) < 3: return []

#     angles = [s['angle'] for s in stripes]
#     avg_angle = np.median(angles)
#     theta = np.radians(avg_angle)
#     coords = np.array([[s['cx'], s['cy']] for s in stripes])
#     dist_perp = -coords[:, 0] * np.sin(theta) + coords[:, 1] * np.cos(theta)
    
#     y_cluster = DBSCAN(eps=50, min_samples=2).fit(dist_perp.reshape(-1, 1))

#     final_group_polygons = []
#     for lbl in np.unique(y_cluster.labels_):
#         if lbl == -1: continue
#         row_pts = []
#         for i, label in enumerate(y_cluster.labels_):
#             if label == lbl:
#                 row_pts.extend(stripes[i]['pts'].reshape(-1, 2))
        
#         if row_pts:
#             hull = cv2.convexHull(np.array(row_pts, np.float32))
#             rect = cv2.minAreaRect(hull)
#             box = cv2.boxPoints(rect).astype(int).tolist()
#             final_group_polygons.append(box)

#     return final_group_polygons

# @app.get("/coordinates")
# async def get_coordinate(p1: str, p2: str, p3: str, p4: str):
#     try:
#         def parse_pt(pt_str):
#             lon, lat = map(float, pt_str.split(","))
#             return (round(lon, 6), round(lat, 6))

#         # 1. Parse Input
#         gps_coords = [parse_pt(p1), parse_pt(p2), parse_pt(p3), parse_pt(p4)]
#         filename = "parking_final.tif"
        
#         # 2. Download Image
#         if not download_and_mask_arcgis(gps_coords, filename):
#             raise HTTPException(status_code=500, detail="ArcGIS Download Failed")

#         # 3. Get Pixel Mask & Transform
#         pixel_poly = []
#         with rasterio.open(filename) as src:
#             transform = src.transform  # Save the transform for back-conversion
#             for lon, lat in gps_coords:
#                 row, col = src.index(lon, lat)
#                 pixel_poly.append((col, row))

#         # 4. Run Grouping Logic (returns pixels)
#         detected_pixel_groups = run_group_detection(filename, pixel_poly)

#         # 5. CONVERT PIXELS BACK TO GPS
#         final_gps_polygons = []
#         with rasterio.open(filename) as src:
#             for group in detected_pixel_groups:
#                 gps_row = []
#                 for x, y in group:
#                     # src.xy(row, col) converts pixels to GPS
#                     lon, lat = src.xy(y, x) 
#                     gps_row.append([round(lon, 6), round(lat, 6)])
#                 final_gps_polygons.append(gps_row)

#         return JSONResponse(status_code=200, content={
#             "status": "success",
#             "message": f"Detected {len(final_gps_polygons)} parking rows",
#             "gps_input": gps_coords,
#             "detected_polygons": final_gps_polygons, # Now returns [[lon, lat], ...]
#             "file_path": os.path.abspath(filename)
#         })

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)



# from fastapi import FastAPI, HTTPException
# from fastapi.responses import JSONResponse
# import cv2
# import numpy as np
# import rasterio
# import os
# from sklearn.cluster import DBSCAN
# from raster import download_and_mask_arcgis

# app = FastAPI()

# try:
#     from cv2 import ximgproc
#     USE_THINNING = True
# except ImportError:
#     USE_THINNING = False

# def run_group_detection(tif_path, selection_poly):
#     with rasterio.open(tif_path) as src:
#         img_bgr = src.read([1, 2, 3]).transpose((1, 2, 0))
#         gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
#         img_w = gray.shape[1]

#     # --- DYNAMIC PARAMETERS BASED ON IMAGE SCALE ---
#     # 1.5% of width is usually the gap between rows
#     dynamic_eps = img_w * 0.018 
#     # Kernel height relative to image size
#     k_height = max(3, int(img_w * 0.006)) 
#     # Minimum stripe length
#     min_stripe_len = img_w * 0.02 

#     binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
#     if USE_THINNING:
#         thin = ximgproc.thinning(255 - binary, thinningType=cv2.ximgproc.THINNING_GUOHALL)
#         _, bw_final = cv2.threshold(thin, 1, 255, cv2.THRESH_BINARY_INV)
#     else:
#         _, bw_final = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

#     mask = np.zeros(bw_final.shape, dtype=np.uint8)
#     cv2.fillPoly(mask, [np.array(selection_poly, dtype=np.int32)], 255)
#     target = cv2.bitwise_and(255 - bw_final, mask)

#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, k_height))
#     refined = cv2.morphologyEx(target, cv2.MORPH_CLOSE, kernel)
#     contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     stripes = []
#     for cnt in contours:
#         if cv2.arcLength(cnt, True) < min_stripe_len: continue
#         rect = cv2.minAreaRect(cnt)
#         (cx, cy), (w, h), angle = rect
#         stripes.append({'pts': cnt, 'cx': cx, 'cy': cy, 'angle': angle})

#     if len(stripes) < 2: return []

#     avg_angle = np.median([s['angle'] for s in stripes])
#     rad = np.radians(avg_angle)
#     coords = np.array([[s['cx'], s['cy']] for s in stripes])
#     dist_perp = -coords[:, 0] * np.sin(rad) + coords[:, 1] * np.cos(rad)
    
#     clusters = DBSCAN(eps=dynamic_eps, min_samples=2).fit(dist_perp.reshape(-1, 1))

#     final_gps_polygons = []
#     with rasterio.open(tif_path) as src:
#         for lbl in np.unique(clusters.labels_):
#             if lbl == -1: continue
#             row_pts = np.concatenate([stripes[i]['pts'] for i, l in enumerate(clusters.labels_) if l == lbl])
#             hull = cv2.convexHull(row_pts.reshape(-1, 2).astype(np.float32))
#             rect = cv2.minAreaRect(hull)
#             box_pixels = cv2.boxPoints(rect)
            
#             # Convert each pixel box point back to GPS
#             gps_box = []
#             for px_x, px_y in box_pixels:
#                 lon, lat = src.xy(px_y, px_x)
#                 gps_box.append([round(lon, 6), round(lat, 6)])
#             final_gps_polygons.append(gps_box)

#     return final_gps_polygons

# @app.get("/coordinates")
# async def get_coordinate(p1: str, p2: str, p3: str, p4: str):
#     try:
#         def parse_pt(pt_str):
#             lon, lat = map(float, pt_str.split(","))
#             return (round(lon, 6), round(lat, 6))

#         gps_coords = [parse_pt(p1), parse_pt(p2), parse_pt(p3), parse_pt(p4)]
#         filename = "parking_final.tif"
        
#         if not download_and_mask_arcgis(gps_coords, filename):
#             raise HTTPException(status_code=500, detail="Download Failed")

#         pixel_poly = []
#         with rasterio.open(filename) as src:
#             for lon, lat in gps_coords:
#                 row, col = src.index(lon, lat)
#                 pixel_poly.append((col, row))

#         detected_gps_rows = run_group_detection(filename, pixel_poly)

#         return JSONResponse(status_code=200, content={
#             "status": "success",
#             "detected_polygons": detected_gps_rows,
#             "file_path": os.path.abspath(filename)
#         })
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": str(e)})

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)



from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import rasterio
import os
from sklearn.cluster import DBSCAN
from raster import download_and_mask_arcgis
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://unaxiomatic-harder-lakeesha.ngrok-free.app" # Remove trailing slash
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

try:
    from cv2 import ximgproc
    USE_THINNING = True
except ImportError:
    USE_THINNING = False

def run_group_detection(tif_path, selection_poly):
    """
    Detects long parking rows using PCA for a consistent Master Angle.
    Returns GPS polygons for each detected row.
    """
    with rasterio.open(tif_path) as src:
        img_bgr = src.read([1, 2, 3]).transpose((1, 2, 0))
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_w = gray.shape[1]
        img_transform = src.transform

    # 1. Preprocessing and Thresholding
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    if USE_THINNING:
        thin = ximgproc.thinning(255 - binary, thinningType=cv2.ximgproc.THINNING_GUOHALL)
        _, bw_final = cv2.threshold(thin, 1, 255, cv2.THRESH_BINARY_INV)
    else:
        _, bw_final = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # 2. Masking to User Selection
    mask = np.zeros(bw_final.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(selection_poly, dtype=np.int32)], 255)
    target = cv2.bitwise_and(255 - bw_final, mask)

    # 3. Line Reinforcement and Contour Detection
    k_height = max(3, int(img_w * 0.005)) 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, k_height))
    refined = cv2.morphologyEx(target, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    stripes = []
    min_len = img_w * 0.015
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        if max(rect[1]) > min_len:
            stripes.append(cnt)

    if len(stripes) < 2: 
        return []

    # 4. PCA for Master Angle (Fixed for modern OpenCV unpacking)
    all_pts = np.vstack(stripes).reshape(-1, 2).astype(np.float32)
    pca_results = cv2.PCACompute(all_pts, mean=None)
    eigenvectors = pca_results[1]
    
    # Calculate Master Angle of the entire lot
    angle_rad = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])
    avg_angle = np.degrees(angle_rad)

    # 5. Projection and Row Grouping
    rad = np.radians(avg_angle)
    stripe_data = []
    for cnt in stripes:
        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        # Project center perpendicular to row direction
        proj = -cx * np.sin(rad) + cy * np.cos(rad)
        stripe_data.append({'cnt': cnt, 'proj': proj})

    stripe_data.sort(key=lambda x: x['proj'])

    rows = []
    if stripe_data:
        row_gap_threshold = img_w * 0.02 # Strict gap to avoid overlapping rows
        current_row = [stripe_data[0]['cnt']]
        for i in range(1, len(stripe_data)):
            if abs(stripe_data[i]['proj'] - stripe_data[i-1]['proj']) < row_gap_threshold:
                current_row.append(stripe_data[i]['cnt'])
            else:
                if len(current_row) >= 2: rows.append(current_row)
                current_row = [stripe_data[i]['cnt']]
        if len(current_row) >= 2: rows.append(current_row)

    # 6. Generate Parallel GPS Bounding Boxes
    final_gps_polygons = []
    with rasterio.open(tif_path) as src:
        for r in rows:
            row_pts = np.vstack(r).reshape(-1, 2).astype(np.float32)
            hull = cv2.convexHull(row_pts)
            rect = cv2.minAreaRect(hull)
            (center, size, _) = rect
            
            # Create pixel box using the Master Angle
            box_pixels = cv2.boxPoints((center, size, avg_angle))
            
            # Convert pixel box to GPS coordinates
            gps_box = []
            for px_x, px_y in box_pixels:
                lon, lat = src.xy(px_y, px_x)
                gps_box.append([round(lon, 6), round(lat, 6)])
            final_gps_polygons.append(gps_box)

    return final_gps_polygons

@app.get("/coordinates")
async def get_coordinate(p1: str, p2: str, p3: str, p4: str):
    try:
        def parse_pt(pt_str):
            lon, lat = map(float, pt_str.split(","))
            return (round(lon, 6), round(lat, 6))

        gps_coords = [parse_pt(p1), parse_pt(p2), parse_pt(p3), parse_pt(p4)]
        filename = "parking_final.tif"
        
        # Download and mask the imagery from your raster script
        if not download_and_mask_arcgis(gps_coords, filename):
            raise HTTPException(status_code=500, detail="ArcGIS Download Failed")

        # Convert GPS input to pixel coordinates for masking
        pixel_poly = []
        with rasterio.open(filename) as src:
            for lon, lat in gps_coords:
                row, col = src.index(lon, lat)
                pixel_poly.append((col, row))

        # Run the updated detection logic
        detected_gps_rows = run_group_detection(filename, pixel_poly)

        return JSONResponse(status_code=200, content={
            "status": "success",
            "detected_polygons": detected_gps_rows,
            "file_path": os.path.abspath(filename)
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)