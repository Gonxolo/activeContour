from src.ActiveContour import ActiveContour
from src.utils import load_params, get_image, get_coords_from_csv
from src.geometryFunctions import polygon_line_sample
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image, ImageOps
import os
import timeit


if __name__ == '__main__':

    start_time = timeit.default_timer()
    lap = start_time

    # params_path = 'D:\\RSI\\SCIANSoft_tests\\cubolebias\\cubolebita\\activecontour_params.csv'
    # params_path = 'D:\\RSI\\SCIANSoft_tests\\sphereRad\\128x128x128\\activecontour_params.csv'
    # params_path = 'D:\\RSI\\SCIANSoft_tests\\cubolebias\\cubolebias2\\activecontour_params.csv'
    # params_path = 'D:\\RSI\\SCIANSoft_tests\\futbolebias\\v1\\activecontour_params.csv'
    # params_path = 'D:\\RSI\\SCIANSoft_tests\\cubolebias\\dummy01\\activecontour_params.csv'
    # params_path = 'D:\\RSI\\Karina\\droplets\\Experimentos\\cubre_1_2\\activecontour_params.csv'
    # params_path = 'D:\\RSI\\SCIANSoft_tests\\futbolebias\\v2\\activecontour_params.csv'
    # params_path = 'D:\\RSI\\SCIANSoft_tests\\boundTest\\activecontour_params.csv'
    # params_path = 'D:\\RSI\\SCIANSoft_tests\\axisTest\\activecontour_params.csv'
    # params_path = 'D:\\RSI\\SCIANSoft_tests\\simetryTest\\activecontour_params.csv'
    # params_path = 'D:\\RSI\\SCIANSoft_tests\\fieldTest\\activecontour_params.csv'
    # params_path = 'D:\\RSI\\SCIANSoft_tests\\marginTest\\activecontour_params.csv'
    # params_path = 'D:\\RSI\\SCIANSoft_tests\\dobleContorno\\activecontour_params.csv'

    params_path = 'D:\\RSI\\SCIANSoft_tests\\futbolebias\\v3\\activecontour_params.csv'


    params = load_params(params_path)

    image_padding = 100

    image = params.get('image', None)
    image_name = image
    image = get_image(image, padding=image_padding)

    alpha = params.get('alpha', None)
    beta = params.get('beta', None)
    gamma = params.get('gamma', None)
    kappa = params.get('kappa', None)
    mu = params.get('mu', None)
    ASL = params.get('ASL', None)
    contour_iterations = params.get('contour_iterations', None)
    vf_iterations = params.get('vf_iterations', None)
    
    original_params = "IDL parameter values: "
    original_params += f"\u03B1 = {alpha}; \u03B2 = {beta}; \u03B3 = {gamma}; "
    original_params += f"\u03BA = {kappa}; \u03BC = {mu}; ASL = {ASL}; iters = {contour_iterations}"

    # alpha = 0.2
    # beta = 0.2
    # gamma = 0.5
    # kappa = 1.0
    # mu = 1.0
    # ASL = 1.0
    contour_iterations = 10
    
    current_params = "Python parameter values: "
    current_params += f"\u03B1 = {alpha}; \u03B2 = {beta}; \u03B3 = {gamma}; "
    current_params += f"\u03BA = {kappa}; \u03BC = {mu}; ASL = {ASL}; iters = {contour_iterations}"
    
    snake = ActiveContour(image=image, alpha=alpha, beta=beta, gamma=gamma, 
    kappa=kappa, mu=mu, contour_iterations=contour_iterations, vf_iterations=vf_iterations)

    if snake.kappa != 0:
        snake.calcGGVF()

    # csv_coords_file = 'D:\\RSI\\SCIANSoft_tests\\cubolebias\\cubolebita\\activecontour_contours.csv'
    # csv_coords_file = 'D:\\RSI\\SCIANSoft_tests\\sphereRad\\128x128x128\\activecontour_contours.csv'
    # csv_coords_file = 'D:\\RSI\\SCIANSoft_tests\\cubolebias\\cubolebias2\\activecontour_contours.csv'
    # csv_coords_file = 'D:\\RSI\\SCIANSoft_tests\\futbolebias\\v1\\activecontour_contours.csv'
    # csv_coords_file = 'D:\\RSI\\SCIANSoft_tests\\cubolebias\\dummy01\\activecontour_contours.csv'
    # csv_coords_file = 'D:\\RSI\\Karina\\droplets\\Experimentos\\cubre_1_2\\activecontour_contours.csv'
    # csv_coords_file = 'D:\\RSI\\SCIANSoft_tests\\futbolebias\\v2\\activecontour_contours.csv'
    # csv_coords_file = 'D:\\RSI\\SCIANSoft_tests\\boundTest\\activecontour_contours.csv'
    # csv_coords_file = 'D:\\RSI\\SCIANSoft_tests\\axisTest\\activecontour_contours.csv'
    # csv_coords_file = 'D:\\RSI\\SCIANSoft_tests\\simetryTest\\activecontour_contours.csv'
    # csv_coords_file = 'D:\\RSI\\SCIANSoft_tests\\fieldTest\\activecontour_contours.csv'
    # csv_coords_file = 'D:\\RSI\\SCIANSoft_tests\\marginTest\\activecontour_contours.csv'
    # csv_coords_file = 'D:\\RSI\\SCIANSoft_tests\\dobleContorno\\activecontour_contours.csv'

    csv_coords_file = 'D:\\RSI\\SCIANSoft_tests\\futbolebias\\v3\\activecontour_contours.csv'

    coords = get_coords_from_csv(csv_coords_file)

    i = 0

    plot_x = np.array([])
    plot_y = np.array([])

    while i < len(coords):

        print(f"ROI {(i+1)//2+1}")

        x = []
        y = []

        for n in coords[i]:
            try:
                x.append(float(n) + image_padding)
            except ValueError:
                break

        for n in coords[i+1]:
            try:
                y.append(float(n) + image_padding)
            except ValueError:
                break

        snake.setContour(x, y)

        # snake.x, snake.y = polygon_line_sample(np.copy(snake.x), np.copy(snake.y), n_points_per_pix=1/2)

        perimeter_factor = 1/ASL

        npts = max(round((snake.getPerimeter()) * perimeter_factor), 16)
        snake.arcSample(points=npts)

        try:
            _x, _y = snake.adjustContour(perimeter_factor=1/ASL)

            _x = np.array([i - image_padding for i in _x])
            _y = np.array([i - image_padding for i in _y])

            plot_x = np.concatenate((plot_x, _x))
            plot_y = np.concatenate((plot_y, _y))
            print(f'ROI {(i+1)//2+1} exec. time: {timeit.default_timer() - lap}')
            i += 2
            lap = timeit.default_timer()
            

        except ValueError as ve:
            print(ve)
            print(f'ROI {(i+1)//2+1} exec. time: {timeit.default_timer() - lap}')
            i += 2
            lap = timeit.default_timer()
            pass

    print("x = ")
    print("x dim = ", str(len(plot_x)))
    print(plot_x)

    print("y = ")
    print("y dim = ", str(len(plot_y)))
    print(plot_y)

    plot_folder = 'plots'

    image_path = os.path.join(os.getcwd(), 'img', image_name)
    image_dimensions = Image.open(image_path).size

    figsize_factor = [(image_dimensions[0])/max(image_dimensions[0], image_dimensions[1]),
                    image_dimensions[1]/ max(image_dimensions[0], image_dimensions[1])]

    plt.figure(figsize=(7*figsize_factor[0], 7*figsize_factor[1]))
    plt.xlim(0, image_dimensions[0])
    plt.xlabel('x [px]')
    plt.ylim(0, image_dimensions[1])
    plt.ylabel('y [px]')
    # background_image_path = "plot_img\\a.png"
    # background_image_path = "plot_img\\b.png"
    # background_image_path = "plot_img\\c.png"
    # background_image_path = "plot_img\\d.png"
    background_image_path = "plot_img\\e.png"
    # background_image_path = "plot_img\\_Clus0_cubolebita_z10.tif"
    # background_image_path = "plot_img\\_Clus0_doble_cubolebita_z10.tif"
    # background_image_path = "plot_img\\_Clus0_Futbolebia_Z32areamedia5_v8_convol_psfSize9_cropped.tif"
    background_image = Image.open(background_image_path)
    background_image = background_image.convert('RGB')
    # background_image = ImageOps.flip(background_image)
    plt.imshow(background_image, extent=[0, image_dimensions[0], 0, image_dimensions[1]], interpolation='none')
    plt.scatter(plot_x, plot_y, s=1, c='red', marker='s')
    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.title(original_params + "\n" + current_params, loc='left', fontsize=7)
    image_name = image_name.split('.')[0]
    plt.savefig(f"{os.path.join(plot_folder, image_name)}_{timestr}.png", format='png')
    plt.savefig(f"{os.path.join(plot_folder, image_name)}_{timestr}", format='svg')
    # plt.close()
    print(timeit.default_timer() - start_time)
    plt.show()
