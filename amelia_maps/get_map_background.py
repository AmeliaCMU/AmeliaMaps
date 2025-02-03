import os
from tqdm import tqdm
import json
from PIL import Image, ImageDraw, ImageFont
# import numpy as np
# import matplotlib.pyplot as plt
# import geopandas as gpd
import contextily as ctx
# from contextily import Place
# import imageio.v2 as imageio
# import rasterio
# from rasterio.plot import show as rioshow
from geopy.geocoders import Nominatim


from amelia_maps.utils import common as C
from amelia_maps.utils import utils as U
import shutil


Image.MAX_IMAGE_PIXELS = None


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def add_watermark(img, output_image_path, credits):
    width, height = img.size
    watermark = Image.new('RGBA', img.size, (255, 255, 255, 0))
    drawing = ImageDraw.Draw(watermark)

    font_path = f"{C.ROOT_DIR}/amelia_maps/utils/fonts//Nunito/Nunito-VariableFont_wght.ttf"
    font_size = 150
    color = (0, 0, 0, 128)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        print(f"Could not load font, the error was: {e}")
        return

    text_width, text_height = (drawing.textlength(credits, font), font_size)

    text_x = width - text_width - font_size
    text_y = height - text_height - font_size

    drawing.text((text_x, text_y), credits, fill=color, font=font)
    watermarked = Image.alpha_composite(img.convert('RGBA'), watermark)
    watermarked.convert('RGB').save(output_image_path, 'PNG')


def get_osm_background(
        limits: tuple, airport: str, output: str, file_name: str = 'bkg_map', zoom=18, provider=ctx.providers.CartoDB.Positron):
    north, east, south, west = limits
    print(ctx.howmany(west, south, east, north, zoom, ll=True))
    airport_img, airport_ext = ctx.bounds2raster(
        west, south, east, north, f'{output}/{file_name}.tiff', zoom=zoom,
        ll=True, max_retries=6, wait=0, source=provider)
    try:
        img = Image.open(f'{output}/{file_name}.tiff')
        add_watermark(img=img, output_image_path=f'{output}/{file_name}.png',  credits=provider.attribution)
    except Exception as e:
        print(f"Could not save PNG of {airport}, the error was: {e}")

    west, east, south, north = airport_ext
    return (north, east, south, west)


def fetch_location_coordinates(location):
    geolocator = Nominatim(user_agent="my-app")
    response = geolocator.geocode(location)
    bbox = response.raw['boundingbox']
    return response, bbox


def get_background(base_dir: str, airport: str, output: str, style: str, zoom: int, update_dataset):

    output = os.path.join(output, airport)
    os.makedirs(output, exist_ok=True)
    # For generating airports
    try:
        response, (south, north, west, east) = fetch_location_coordinates(airport)
    except:
        print(f"Bad request, skipping {airport}...")
        return

    ll_limits = (float(north), float(east), float(south), float(west))

    if airport in C.MAP_EXTENSION:
        ll_limits = (
            ll_limits[0] + C.MAP_EXTENSION[airport]['north'],
            ll_limits[1] + C.MAP_EXTENSION[airport]['east'],
            ll_limits[2] + C.MAP_EXTENSION[airport]['south'],
            ll_limits[3] + C.MAP_EXTENSION[airport]['west']
        )

    plot_limits = get_osm_background(ll_limits, airport, output, zoom=zoom, provider=C.MAP_PROVIDERS[style])

    airport_name = components = [comp.strip() for comp in response[0].split(',')][0]
    latlng = response[1]

    U.create_limits(plot_limits, (airport, airport_name), latlng, output)

    if update_dataset:
        print(f"Updating limits and map background for {airport}")
        shutil.copy(f'{output}/limits.json', f'{base_dir}/assets/{airport}/limits.json')
        shutil.copy(f'{output}/bkg_map.png', f'{base_dir}/assets/{airport}/bkg_map.png')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default=C.DATA_DIR, type=str, help='Input directory')
    parser.add_argument('--airport', default='all', type=str, help='ICAO Airport Code')
    parser.add_argument('--style', default='mapnik', type=str,
                        help='Map Style', choices=C.MAP_PROVIDERS.keys())
    parser.add_argument('--zoom', default=18, type=int, help='Map Zoom')
    parser.add_argument('--output', default=C.OUTPUT_DIR, type=str, help='Output directory')
    parser.add_argument('--update_dataset', action='store_true', help='Modify dataset')
    args = parser.parse_args()

    # Done through txt file because there might be additional airports not in assets folder
    if args.airport == 'all':
        with open(C.AIRPORTS_PATH, 'r') as f:
            airport_list_raw = f.readlines()
        airport_list = [airport.strip() for airport in airport_list_raw]
    else:
        airport_list = [args.airport]

    for airport in tqdm(airport_list):
        kargs = vars(args)
        kargs['airport'] = airport
        get_background(**kargs)
