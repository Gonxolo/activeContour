import numpy as np


class ActiveContour:

    # TODO: Aqui recibimos los parametros para inicializar el algoritmo
    def __init__(self, image: list[list], x_coords: list[int], y_coords: list[int], alpha: float = 0.1, beta: float = 0.25,
                 gamma: float = 1.0, kappa: float = 0.0, mu: float = 0.1, gvf_iterations: int = 100,
                 iterations: int = 200) -> None:

        """
        [Descripcion de la clase]

        :param image: Image file for which the active contour (snake) will be applied.
                        This argument must be 2D.

        :param x_coords: The initial X points of the active contour of the snake. Optional.
                        Must be used with Y.

        :param y_coords: The initial Y points of the active contour of the snake. Optional.
                        Must be used with X.

        :param alpha: The elasticity parameter of the active contour. It reflects the contour's ability to stretch along
                        its length. Default: 0.10.

        :param beta: The rigidity parameter of the active contour. It reflects the contour's ability to bend, as, for
                        example, around corners. Default: 0.25.

        :param gamma: The viscosity parameter. Larger values make it harder to deform the active contour in space.
                        Default: 1.0.

        :param kappa: The external force weight. Default: 1.25.

        :param mu: The regularization parameter. this should be set according to the amount of noise in the image. Use a
                        larger value for noisier images. Default: 0.10.

        :param gvf_iterations: The number of iterations for calculation the Gradient Vector Flow (GVF). Default: 100.

        :param iterations: The number of iterations to use in calculating the snake positions. Default: 200.
        """

        self.image = np.array(image)
        self.xCoords = np.array(x_coords)
        self.yCoords = np.array(y_coords)
        self.alpha = max(alpha, 0.001)
        self.beta = max(beta, 0.001)
        self.gamma = max(gamma, 0.1)
        self.kappa = max(kappa, 0.0)
        self.mu = max(min(mu, 0.25), 0.001)
        self.gvf_iterations = max(gvf_iterations, 1)
        self.iterations = max(iterations, 1)

        # pU (edgeMap)
        # pV (edgeMap)
        self.u = None
        self.v = None

        # Manejar errores

        # Revisa que se haya entregado una imagen

        # Chequea que sea un array 2-dimensional

        # Puntero a la imagen

        # Puntero a coordenadas X e Y

        # Calcular npts igual a largo de X o 0 si es None/si es invalido

        self.npts = self.xCoords if self.xCoords is not None else 0

        pass

    def laplacian(self, image) -> np.ndarray:

        kernel = np.zeros((5, 5))
        kernel[0, 2] = 0.0833333
        kernel[1, 1:4] = 0.0833333
        kernel[2, :] = 0.0833333
        kernel[3, 1:4] = 0.0833333
        kernel[4, 2] = 0.0833333
        kernel[2, 2] = -1.0

        # return convol(image, kernel, center=1, /edge_truncate)
        return convolve(image, kernel, mode='nearest')


    # TODO: Computar el campo GGVF para el contorno activo
    def calcGGVF(self) -> None:

        # Calcular gradientes [fx, fy] para inicializar el vetor campo [u, v]
        # self->edgeMap (se llama al metodo edgeMap)
            # self->gradient (se llama al metodo gradient)

        # Se resuelve iterativamente para el GGVF [u, v]
            # self->laplacian (se llama al metodo laplacian)

        return

    # TODO: Plotear el campo GVF
    def plotGVF(self) -> None:
        return

    # TODO: Setear las [x, y] coordenadas para el contorno activo
    def setContour(self) -> None:
        return

    # TODO:
    def getPerimeter(self) -> None:
        return

    # TODO:
    def arcSample(self) -> None:
        return

    # TODO: aqui pasan muchas cosas
    def adjustContour(self) -> None:
        return
