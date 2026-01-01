import requests
import rasterio
from rasterio.transform import from_bounds
from rasterio.mask import mask # Nava library
from shapely.geometry import Polygon # Polygon banon layi
import numpy as np
from io import BytesIO
from PIL import Image
import urllib3
import os

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def download_and_mask_arcgis(poly_coords, filename="parking_masked.tif"):
    """
    poly_coords: List of 4 tuples [(lon, lat), (lon, lat), (lon, lat), (lon, lat)]
    """
    # 1. Bounding Box kaddo (Download karan layi)
    lons = [p[0] for p in poly_coords]
    lats = [p[1] for p in poly_coords]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)

    print(f"Starting ArcGIS Download for BBox...")
    
    width = 2000
    aspect_ratio = (max_lat - min_lat) / (max_lon - min_lon)
    height = int(width * aspect_ratio)
    bbox = f"{min_lon},{min_lat},{max_lon},{max_lat}"
    
    url = (
        f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?"
        f"bbox={bbox}&bboxSR=4326&size={width},{height}&imageSR=4326&format=jpg&f=image"
    )

    temp_file = "temp_raw.tif"

    try:
        # --- DOWNLOAD STEP ---
        response = requests.get(url, verify=False, timeout=230)
        response.raise_for_status()

        img = Image.open(BytesIO(response.content)).convert('RGB')
        data = np.array(img).transpose(2, 0, 1)

        transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)

        # Pehla temporary rectangle file save karo
        with rasterio.open(
            temp_file, 'w', driver='GTiff',
            height=height, width=width, count=3,
            dtype=data.dtype, crs='EPSG:4326', transform=transform,
        ) as dst:
            dst.write(data)

        # --- MASKING STEP (Sirf 4 points wala area rakho) ---
        print("Masking image to your 4 coordinates...")
        with rasterio.open(temp_file) as src:
            # Shapely polygon banao
            geo_poly = [Polygon(poly_coords)]
            
            # crop=True baki faltu area nu kadd denda hai
            # nodata=0 bahar de pixels nu black kar denda hai
            out_image, out_transform = mask(src, geo_poly, crop=True, nodata=0)
            out_meta = src.meta.copy()

        # Meta update karo navi dimensions layi
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        # Final file save karo
        with rasterio.open(filename, "w", **out_meta) as dest:
            dest.write(out_image)

        # Temporary file delete karo
        if os.path.exists(temp_file):
            os.remove(temp_file)

        print(f"SUCCESS: Polygon masked file saved as {filename}")
        return True

    except Exception as e:
        print(f"Error: {e}")
        return False

