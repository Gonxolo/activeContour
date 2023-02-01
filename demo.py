import os
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

from src.ActiveContour import ActiveContour
from src.utils import load_params, get_image, get_coords_from_csv

PATH_TO_THIS_FILE = os.path.dirname(__file__)

def main(plot_results=True, verbose=True):
    
    # Demofiles Path
    files_path = os.path.join(PATH_TO_THIS_FILE, "demofiles")

    # Parameter Loading
    params_path = os.path.join(files_path, "activecontour_params.csv")

    params = load_params(params_path=params_path)

    image_name = params.get('image', None)
    image_path = os.path.join(files_path, image_name)

    image_padding = 100
    image = get_image(image_path, padding=image_padding)

    alpha = params.get('alpha', None)
    beta = params.get('beta', None)
    gamma = params.get('gamma', None)
    kappa = params.get('kappa', None)
    mu = params.get('mu', None)
    ASL = params.get('ASL', None)
    contour_iterations = params.get('contour_iterations', None)
    vf_iterations = params.get('vf_iterations', None)

    snake = ActiveContour(image=image, alpha=alpha, beta=beta, gamma=gamma,
    kappa= kappa, mu=mu, contour_iterations=contour_iterations,
    vf_iterations=vf_iterations)

    if snake.kappa != 0:
        snake.calcGGVF()

    coords_file_path = os.path.join(files_path, "activecontour_contours.csv")

    coords = get_coords_from_csv(coords_file_path)

    if plot_results:
        _plot_x = np.array([])
        _plot_y = np.array([])

    i = 0
    while i < len(coords):

        if verbose: print(f"ROI {(i+1)//2+1}")

        x_coords = []
        y_coords = []

        for j in coords[i]:
            try:
                x_coords.append(float(j) + image_padding)
            except ValueError:
                break
        
        for j in coords[i+1]:
            try:
                y_coords.append(float(j) + image_padding)
            except ValueError:
                break
        
        snake.setContour(x=x_coords, y=y_coords)

        perimeter_factor = 1/ASL

        npts = max(round((snake.getPerimeter()) * perimeter_factor), 16)
        snake.arcSample(points=npts)

        try:
            _x, _y = snake.adjustContour(perimeter_factor=1/ASL)

            _x = np.array([i - image_padding for i in _x])
            _y = np.array([i - image_padding for i in _y])

            _plot_x = np.concatenate((_plot_x, _x))
            _plot_y = np.concatenate((_plot_y, _y))
            

        except ValueError as ve:
            print(f"ValueError in iterarion ROI {(i+1)//2+1}")
            print(ve)
        
        finally:
            i += 2

    if plot_results:
        _img_dim = Image.open(image_path).size

        _larger_side = max(_img_dim[0], _img_dim[1])
        _fig_ratio = (_img_dim[0]/_larger_side, _img_dim[1]/_larger_side)
        
        plt.figure(figsize=(7*_fig_ratio[0], 7*_fig_ratio[1]))
        plt.xlim(0, _img_dim[0])
        plt.xlabel('x [px]')
        plt.ylim(0, _img_dim[1])
        plt.ylabel('y [px]')

        _bkg_img = Image.open(image_path).convert('RGB')
        _bkg_img = ImageOps.flip(_bkg_img)

        plt.imshow(_bkg_img, extent=[0, _img_dim[0], 0, _img_dim[1]], interpolation='none')
        plt.scatter(_plot_x, _plot_y, s=1, c='red', marker='s')
        plt.show()

    return



if __name__ == '__main__':
    main()