import csv
import numpy as np
from PIL import Image, ImageOps
import os


def load_params(params_path: str) -> dict:
    
    ac_params = {}

    with open(params_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')

        keys = []
        values = []

        row_number = 0

        for row in csv_reader:
            if row_number == 0:
                keys = row
            elif row_number == 1:
                aux = []
                for i in row:
                    try:
                        aux.append(float(i))
                    except ValueError:
                        aux.append(i)
                values = aux
            else:
                break
            row_number += 1

        ac_params = dict(zip(keys, values))

    return ac_params

        
def get_coords_from_csv(filepath):
    
    filename = open(filepath, 'r')

    csv_reader = csv.reader(filename, delimiter=";")

    n_cols = 0
    for row in csv_reader:
        n_cols = len(row) - 1
        break
    
    rois = [[] for _ in range(n_cols)]

    for row in csv_reader:
        for i in range(len(row) - 1):
            rois[i].append(row[i])
    
    return rois

def get_image(image_path: str, padding=20) -> np.ndarray:
    
    image_path = os.path.join(os.getcwd(), 'img', image_path)

    im = Image.open(image_path)
    im = ImageOps.flip(im)

    width, height = im.size
    
    new_width = width + 2*padding
    new_height = height + 2*padding
    
    if im.mode == 'L':
        result = Image.new(im.mode, (new_width,new_height), (0))
    elif im.mode == 'RGB':
        result = Image.new(im.mode, (new_width,new_height), (0, 0, 0))
    else:
        print(f"Image is being change from image mode {im.mode} to RGB")
        im = im.convert('RGB')
        print(im.mode)
        result = Image.new(im.mode, (new_width,new_height), (0, 0, 0))

    result.paste(im, (padding, padding))

    return np.array(result)
