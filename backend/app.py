import base64

from fastapi import FastAPI, File, UploadFile, Form, Response
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from scipy import ndimage
import io
from PIL import Image
import PIL.ExifTags
from geopy.geocoders import Nominatim #allows for processing and translating geolocation data

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Location"] #added this 
)

# coordinate conversion scale
def convert(value):
    d, m, s = value
    return float(d) + float(m)/60 + float(s)/3600

# process geolocation information from image
def img_EXIF_extract(img):
    exif_data = img.getexif()  # Extract EXIF metadata from the image
    if exif_data:
        exif = {
            PIL.ExifTags.TAGS[k]: v
            for k, v in img._getexif().items()
            if k in PIL.ExifTags.TAGS
        }
        gps = exif.get('GPSInfo')
        if not gps: 
            return "No GPS info"

        # Extract raw GPS data for latitude (north) and longitude (east)
        north = gps[2]
        north_ref = gps[1]
        east = gps[4]
        east_ref = gps[3]

        print(north)
        print(east)

        # Convert latitude and longitude from degrees-minutes-seconds to decimal format
        lat = convert(north)
        if north_ref == 'S':
            lat = -lat
        long = convert(east)
        
        if east_ref == 'W':
            long = -long 

        # Convert to float for precise calculations
        lat, long = float(lat), float(long)
        lat = round(lat, 2)
        long = round(long, 2)

        print(lat)
        print(long)

        # Use Geopy's Nominatim geocoder to retrieve the address for the coordinates
        try:
            geoLoc = Nominatim(user_agent="caballeros_coloring_v15_18974297590349289783", timeout=60)    # Initialize the geocoder
            locname = geoLoc.reverse(f"{lat}, {long}", language="en")  # Perform reverse geocoding
            address = locname.raw['address']
            city = address.get('city') or address.get('town')
            state = address.get('state')
            country = address.get('country')
            return f"{city}, {state}, {country}"  # return the address of the location
        except Exception as e:
            print(f"Geocoding failed: {e}")
            # raw coords fallback
            return f"Coords: {lat}, {long}"

    else:
        return "No EXIF data for photo!"

# SAM setup
model_type = "vit_b"
sam_checkpoint = "sam_vit_b_01ec64.pth"
model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
model.to(device="cpu")
mask_generator = SamAutomaticMaskGenerator(model)

# Heavily inspired by https://github.com/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb
def show_anns(anns, image, mode):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    # ADD IF
    if mode == "detailed_scan":
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ann_image = cv2.adaptiveThreshold(
            grayscale_image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        ann_image = cv2.cvtColor(ann_image, cv2.COLOR_GRAY2BGR)
    else:
        ann_image = np.full(image.shape, 255, dtype=np.uint8) # see if this makes an all white image?
    # finish adding 
    white = [255,255,255]
    black = [0,0,0]
    for ann in sorted_anns:
        m = ann['segmentation']
        # use binary erosion to show only edges
        inside = ndimage.binary_erosion(m, iterations=5)
        edge = (m ^ inside).astype(m.dtype)
        if mode == "simple_scan":
            ann_image[edge] = black
            ann_image[inside] = white
        else:
            color_mask = np.concatenate([np.random.random(3)])
            ann_image[edge] = color_mask
        

    '''
    # TODO: save plot w side by side to logs in a later version
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image)
    ax[0].axis('off')
    ax[0].set_title('Original Image')

    ax[1].imshow(ann_image)
    ax[1].axis('off')
    ax[1].set_title('Segmented Image')
    '''

    # should return the image with anns overlaid
    return ann_image


@app.get("/")
def home():
    return {"status": "Success", "message": "SAM API is Live"}

@app.get('/api/data')
def get_data():
    return {"message": "Hello from Python!"}

@app.post('/api/segment')
async def segment_image(mode: str = Form(...), image: UploadFile = File(...)):
    print(mode) # this signifies the simple or detailed scan

    # read image
    obj = await image.read()
    open_img = Image.open(io.BytesIO(obj))
    
    # grab location information
    try:
        location_info = img_EXIF_extract(open_img)
        print(f"Location: {location_info}")
    except Exception as e:
        print(f"Geolocation error: {e}")
        location_info = "Geolocation error"

    open_img.thumbnail((800,800))
    pre_processed_image = open_img.convert("RGB")
    np_image = np.array(pre_processed_image)
    cv2_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

    # generate masks
    masks = mask_generator.generate(cv2_image)

    '''
    TODO: try custom params on mask generation
    mask_generator_2 = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )
    '''

    # get output image
    output_image = show_anns(masks, cv2_image, mode=mode) # LOOK HERE
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGBA)
    output_image[:, :, 3] = 255.0
    
    pil_img = Image.fromarray(output_image, mode='RGBA')
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")

    output_bytes = buffer.getvalue()

    return Response(content=output_bytes, media_type="image/png", headers={"X-Location": str(location_info)})

