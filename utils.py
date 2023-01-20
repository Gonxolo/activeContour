import csv
import numpy as np
from PIL import Image
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

    file = csv.DictReader(filename, delimiter=";")

    x = []
    y = []

    for col in file:
        x.append(col['roi_0_contour_0_x'])
        y.append(col['roi_0_contour_0_y'])

    return x, y


def get_image(image_path: str) -> np.ndarray:
    
    image_path = os.path.join(os.getcwd(), image_path)

    im = Image.open(image_path)

    return np.array(im)