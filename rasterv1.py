import requests
import rasterio
from rasterio.transform import from_bounds
import numpy as np
from io import BytesIO
from PIL import Image
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def download_arcgis_4326(min_lon, min_lat, max_lon, max_lat, filename="parking_final.tif"):
    print("Starting ArcGIS Download (WGS84 - EPSG:4326)...")
    
    # 1. Image size nu BBOX de ratio mutabik rakho taan jo stretch na hoye
    # Is naal pixel perfect rahengi
    width = 2000
    aspect_ratio = (max_lat - min_lat) / (max_lon - min_lon)
    height = int(width * aspect_ratio)
    
    bbox = f"{min_lon},{min_lat},{max_lon},{max_lat}"
    
    # URL vich clear ditta hai ki 4326 hi chahide
    url = (
        f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?"
        f"bbox={bbox}&bboxSR=4326&size={width},{height}&imageSR=4326&format=jpg&f=image"
    )

    try:
        response = requests.get(url, verify=False, timeout=230)
        response.raise_for_status()

        img = Image.open(BytesIO(response.content)).convert('RGB')
        data = np.array(img).transpose(2, 0, 1)

        # 2. Transform Degrees (4326) de hisab naal
        transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)

        with rasterio.open(
            filename, 'w',
            driver='GTiff',
            height=height, width=width,
            count=3,
            dtype=data.dtype,
            crs='EPSG:4326', 
            transform=transform,
        ) as dst:
            dst.write(data)
        
        print(f"SUCCESS: File saved {filename}")
        print(f"Resolution: {width}x{height}")

    except Exception as e:
        print(f"Error: {e}")
        return False
    return True




    
    