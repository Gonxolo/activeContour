from ActiveContour import ActiveContour
from utils import load_params, get_image, get_coords_from_csv
from geometryFunctions import polygon_line_sample
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    params_path = 'D:\\RSI\\SCIANSoft_tests\\cubolebias\\cubolebita\\activecontour_params.csv'

    params = load_params(params_path)        

    image = params.get('image', None)

    image = get_image(image)

    alpha = params.get('alpha', None)
    beta = params.get('beta', None)
    gamma = params.get('gamma', None)
    kappa = params.get('kappa', None)
    mu = params.get('mu', None)
    ASL = params.get('ASL', None)
    contour_iterations = params.get('contour_iterations', None)
    vf_iterations = params.get('vf_iterations', None)

    snake = ActiveContour(image=image, alpha=alpha, beta=beta, gamma=gamma, 
    kappa=kappa, mu=mu, contour_iterations=contour_iterations, vf_iterations=vf_iterations)

    snake.calcGGVF()

    csv_coords_file = 'D:\\RSI\\SCIANSoft_tests\\cubolebias\\cubolebita\\activecontour_contours.csv'
        
    x, y = get_coords_from_csv(csv_coords_file)

    snake.setContour(x, y)

    snake.x, snake.y = polygon_line_sample(np.copy(snake.x), np.copy(snake.y), n_points_per_pix=1/2)

    # plt.quiver(np.arange(20), np.arange(20), snake.u, snake.v)
    # plt.show()

    snake.adjustContour(perimeter_factor=ASL)

