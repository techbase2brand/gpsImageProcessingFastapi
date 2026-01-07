# import requests
# import rasterio
# from rasterio.transform import from_bounds
# import numpy as np
# from io import BytesIO
# from PIL import Image
# import urllib3
# import os

# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# def download_and_mask_arcgis(poly_coords, filename="parking_final.tif"):
#     """
#     Downloads ArcGIS imagery based on GPS coordinates.
#     poly_coords: List of (lon, lat) tuples.
#     """
#     lons = [p[0] for p in poly_coords]
#     lats = [p[1] for p in poly_coords]
#     min_lon, max_lon = min(lons), max(lons)
#     min_lat, max_lat = min(lats), max(lats)

#     width = 2000
#     aspect_ratio = (max_lat - min_lat) / (max_lon - min_lon)
#     height = int(width * aspect_ratio)
#     bbox = f"{min_lon},{min_lat},{max_lon},{max_lat}"
    
#     url = (
#         f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?"
#         f"bbox={bbox}&bboxSR=4326&size={width},{height}&imageSR=4326&format=jpg&f=image"
#     )

#     try:
#         response = requests.get(url, verify=False, timeout=230)
#         response.raise_for_status()

#         img = Image.open(BytesIO(response.content)).convert('RGB')
#         data = np.array(img).transpose(2, 0, 1)
#         transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)

#         with rasterio.open(
#             filename, 'w', driver='GTiff',
#             height=height, width=width, count=3,
#             dtype=data.dtype, crs='EPSG:4326', transform=transform,
#         ) as dst:
#             dst.write(data)
        
#         return True
#     except Exception as e:
#         print(f"Download Error: {e}")
#         return False


import requests
import rasterio
from rasterio.transform import from_bounds
import numpy as np
from io import BytesIO
from PIL import Image
import urllib3
import os

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def download_and_mask_arcgis(poly_coords, filename="parking_final.tif"):
    """
    Downloads ArcGIS imagery and locks the GPS transform 
    to the actual image size to prevent coordinate shifting.
    """
    lons = [p[0] for p in poly_coords]
    lats = [p[1] for p in poly_coords]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)

    # Set a target width; height is calculated by ArcGIS
    target_width = 1280 
    aspect_ratio = (max_lat - min_lat) / (max_lon - min_lon)
    target_height = int(target_width * aspect_ratio)
    
    bbox = f"{min_lon},{min_lat},{max_lon},{max_lat}"
    url = (
        f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?"
        f"bbox={bbox}&bboxSR=4326&size={target_width},{target_height}&imageSR=4326&format=jpg&f=image"
    )

    try:
        response = requests.get(url, verify=False, timeout=230)
        response.raise_for_status()
        
        img = Image.open(BytesIO(response.content)).convert('RGB')
        
        # USE ACTUAL DIMENSIONS FROM THE DOWNLOADED IMAGE
        actual_w, actual_h = img.size 
        data = np.array(img).transpose(2, 0, 1)

        # Create transform based on EXACT pixel counts received
        transform = from_bounds(min_lon, min_lat, max_lon, max_lat, actual_w, actual_h)

        with rasterio.open(
            filename, 'w', driver='GTiff',
            height=actual_h, width=actual_w,
            count=3, dtype='uint8', crs='EPSG:4326',
            transform=transform
        ) as dst:
            dst.write(data)
        
        return True
    except Exception as e:
        print(f"Download Error: {e}")
        return False