import gps
import PIL.ExifTags
from PIL import Image
from geopy.geocoders import Nominatim


def convert(value):
    d, m, s = value
    return float(d) + float(m)/60 + float(s)/3600

# Extract EXIF metadata as a dictionary with readable tags
def img_EXIF_extract(img):
    exif_data = img.getexif()  # Extract EXIF metadata from the image
    if exif_data:
        exif = {
            PIL.ExifTags.TAGS[k]: v
            for k, v in img._getexif().items()
            if k in PIL.ExifTags.TAGS
        }
        gps = exif['GPSInfo']

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
        geoLoc = Nominatim(user_agent="GetLoc")    # Initialize the geocoder
        locname = geoLoc.reverse(f"{lat}, {long}", language="en")  # Perform reverse geocoding
        address = locname.raw['address']
        city = address.get('city') or address.get('town')
        state = address.get('state')
        country = address.get('country')
        return f"{city}, {state}, {country}"  # return the address of the location

    else:
        return "No EXIF data for photo!"
