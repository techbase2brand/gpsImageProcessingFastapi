from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import rasterio
import os
from sklearn.cluster import DBSCAN
from raster import download_and_mask_arcgis

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://unaxiomatic-harder-lakeesha.ngrok-free.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    from cv2 import ximgproc
    USE_THINNING = True
except ImportError:
    USE_THINNING = False

# =========================
# HELPER: GEOMETRIC SPLITTING
# =========================
def split_row_into_vertical_groups(box, num_groups=5):
    """
    Subdivides a long row bounding box into equal vertical segments.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = box.sum(axis=1)
    rect[0] = box[np.argmin(s)]    # Top-left
    rect[2] = box[np.argmax(s)]    # Bottom-right
    diff = np.diff(box, axis=1)
    rect[1] = box[np.argmin(diff)] # Top-right
    rect[3] = box[np.argmax(diff)] # Bottom-left

    tl, tr, br, bl = rect
    sub_groups = []

    for i in range(num_groups):
        t1 = i / num_groups
        t2 = (i + 1) / num_groups

        # Interpolate points along the top and bottom edges
        p1 = tl + t1 * (tr - tl)
        p2 = tl + t2 * (tr - tl)
        p3 = bl + t2 * (br - bl)
        p4 = bl + t1 * (br - bl)

        sub_groups.append(np.array([p1, p2, p3, p4], dtype=np.float32))
    
    return sub_groups

# =========================
# DETECTION LOGIC
# =========================
def run_group_detection(tif_path, selection_poly):
    with rasterio.open(tif_path) as src:
        img_bgr = src.read([1, 2, 3]).transpose((1, 2, 0))
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_w = gray.shape[1]

    # 1. Thresholding
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    if USE_THINNING:
        thin = ximgproc.thinning(255 - binary, thinningType=cv2.ximgproc.THINNING_GUOHALL)
        _, bw_final = cv2.threshold(thin, 1, 255, cv2.THRESH_BINARY_INV)
    else:
        _, bw_final = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # 2. Masking
    mask = np.zeros(bw_final.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(selection_poly, dtype=np.int32)], 255)
    target = cv2.bitwise_and(255 - bw_final, mask)

    # 3. Reinforce Lines
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

    if len(stripes) < 2: return []

    # 4. PCA for Master Angle
    all_pts = np.vstack(stripes).reshape(-1, 2).astype(np.float32)
    pca_results = cv2.PCACompute(all_pts, mean=None)
    eigenvectors = pca_results[1]
    angle_rad = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])
    avg_angle = np.degrees(angle_rad)

    # 5. Projection & Grouping
    rad = np.radians(avg_angle)
    stripe_data = []
    for cnt in stripes:
        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        proj = -cx * np.sin(rad) + cy * np.cos(rad)
        stripe_data.append({'cnt': cnt, 'proj': proj})

    stripe_data.sort(key=lambda x: x['proj'])

    rows = []
    if stripe_data:
        row_gap_threshold = img_w * 0.02 
        current_row = [stripe_data[0]['cnt']]
        for i in range(1, len(stripe_data)):
            if abs(stripe_data[i]['proj'] - stripe_data[i-1]['proj']) < row_gap_threshold:
                current_row.append(stripe_data[i]['cnt'])
            else:
                if len(current_row) >= 2: rows.append(current_row)
                current_row = [stripe_data[i]['cnt']]
        if len(current_row) >= 2: rows.append(current_row)

    # 6. Generate Sub-divided GPS Polygons
    final_gps_polygons = []
    with rasterio.open(tif_path) as src:
        for r in rows:
            row_pts = np.vstack(r).reshape(-1, 2).astype(np.float32)
            hull = cv2.convexHull(row_pts)
            rect = cv2.minAreaRect(hull)
            (center, size, _) = rect
            
            # Create the long row bounding box in pixel space
            long_box = cv2.boxPoints((center, size, avg_angle))
            
            # Split into 5 vertical segments
            vertical_segments = split_row_into_vertical_groups(long_box, num_groups=5)
            
            for segment in vertical_segments:
                gps_segment = []
                for px_x, px_y in segment:
                    lon, lat = src.xy(px_y, px_x)
                    gps_segment.append([round(lon, 6), round(lat, 6)])
                final_gps_polygons.append(gps_segment)

    return final_gps_polygons

@app.get("/")
def welcome():
     return JSONResponse(status_code=200, content={
            "status": "success",
            
        })

@app.get("/coordinates")
async def get_coordinate(p1: str, p2: str, p3: str, p4: str):
    try:
        def parse_pt(pt_str):
            lon, lat = map(float, pt_str.split(","))
            return (round(lon, 6), round(lat, 6))

        gps_coords = [parse_pt(p1), parse_pt(p2), parse_pt(p3), parse_pt(p4)]
        filename = "parking_final.tif"
        
        if not download_and_mask_arcgis(gps_coords, filename):
            raise HTTPException(status_code=500, detail="Download Failed")

        pixel_poly = []
        with rasterio.open(filename) as src:
            for lon, lat in gps_coords:
                row, col = src.index(lon, lat)
                pixel_poly.append((col, row))

        detected_gps_rows = run_group_detection(filename, pixel_poly)

        return JSONResponse(status_code=200, content={
            "status": "success",
            "detected_polygons": detected_gps_rows,
            "file_path": os.path.abspath(filename)
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    

@app.get("/downloard")
async def downloard(p1: str, p2: str, p3: str, p4: str):
    try:
        def parse_pt(pt_str):
            # Coordinates split karke float banaunda hai
            parts = pt_str.split(",")
            return (float(parts[0].strip()), float(parts[1].strip()))

        # 4 Points di list
        gps_coords = [parse_pt(p1), parse_pt(p2), parse_pt(p3), parse_pt(p4)]
        
        filename = "parking_final.tif"

        # Function call
        success = download_and_mask_arcgis(gps_coords, filename)
        
        if success:
            # File return karni zaruri hai, nahi tan browser 'null' dikhayega
            if os.path.exists(filename):
                return FileResponse(
                    path=filename, 
                    filename=filename, 
                    media_type='image/tiff'
                )
            else:
                raise HTTPException(status_code=500, detail="File saved but not found on disk")
        else:
            return JSONResponse(status_code=500, content={"status": "error", "message": "Download failed"})

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})    


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)