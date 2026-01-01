from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from raster import download_and_mask_arcgis
from group import run_group_detection
import os
import uvicorn


app = FastAPI()

# @app.get("/coordinates")
# async def get_coordinate(p1: str, p2: str, p3: str, p4: str):
#     """
#     Endpoint to download and mask ArcGIS imagery based on 4 coordinates.
#     URL Example: /coordinates?p1=-86.49,35.97&p2=-86.49,35.97&p3=-86.49,35.96&p4=-86.49,35.96
#     """
#     try:
#         # 1. Helper function for String to Float conversion with error check
#         def parse_pt(pt_str):
#             try:
#                 lon, lat = map(float, pt_str.split(","))
#                 return (lon, lat)
#             except (ValueError, IndexError):
#                 raise ValueError(f"Invalid format: {pt_str}. Format must be 'longitude,latitude'.")

#         # Parse the 4 corners into a list of tuples
#         poly_coords = [parse_pt(p1), parse_pt(p2), parse_pt(p3), parse_pt(p4)]

#         # 2. Logic for downloading and masking the raster
#         filename = "parking_final.tif"
        
#         # Call the processing function inside the try-except block
#         success = download_and_mask_arcgis(poly_coords, filename)

#         if success:
#             return JSONResponse(
#                 status_code=200,
#                 content={
#                     "status": "success",
#                     "message": "File downloaded and processed successfully.",
#                     "file_path": os.path.abspath(filename)
#                 }
#             )
#         else:
#             # Handle case where the function returns False without raising an exception
#             raise HTTPException(
#                 status_code=500, 
#                 detail="ArcGIS download or masking process failed."
#             )

#     except ValueError as ve:
#         # Handle coordinate parsing errors (400 Bad Request)
#         raise HTTPException(status_code=400, detail=str(ve))
    
#     except Exception as e:
#         # Catch-all for unexpected errors (e.g., Network, File System, Library errors)
#         print(f"SYSTEM ERROR: {e}")
#         raise HTTPException(
#             status_code=500, 
#             detail=f"Internal Server Error: {str(e)}"
#         )

@app.get("/coordinates")
async def get_coordinate(p1: str, p2: str, p3: str, p4: str):
    try:
        # 1. Parse GPS strings to (Lon, Lat) tuples
        def parse_pt(pt_str):
            lon, lat = map(float, pt_str.split(","))
            return (lon, lat)

        gps_coords = [parse_pt(p1), parse_pt(p2), parse_pt(p3), parse_pt(p4)]
        filename = "parking_final.tif"
        
        # 2. Download and save the GeoTIFF
        success = download_and_mask_arcgis(gps_coords, filename)

        if success:
            # 3. CONVERT GPS TO PIXELS
            # We open the file we just created to get its coordinate system (transform)
            pixel_coords = []
            with rasterio.open(filename) as src:
                for lon, lat in gps_coords:
                    # row = Y pixel, col = X pixel
                    row, col = src.index(lon, lat)
                    pixel_coords.append((col, row)) 

            # 4. RUN DETECTION
            # Now we pass the actual pixel points into the grouping function
            detected_groups = run_group_detection(filename, pixel_coords)

            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "file_path": os.path.abspath(filename),
                    "detected_polygons": detected_groups # List of [[x,y],[x,y]...]
                }
            )
        else:
            raise HTTPException(status_code=500, detail="Download failed")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)