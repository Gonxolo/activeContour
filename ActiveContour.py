import numpy as np
class ActiveContour:

    # TODO: Aqui recibimos los parametros para inicializar el algoritmo
    def __init__(self, image: str, x_coords: list[int], y_coords: list[int], alpha: float = 0.1, beta: float = 0.25,
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

        self.image = image
        self.xCoords = x_coords
        self.yCoords = y_coords
        self.alpha = max(alpha, 0.001)
        self.beta = max(beta, 0.001)
        self.gamma = max(gamma, 0.1)
        self.kappa = max(kappa, 0.0)
        self.mu = max(min(mu, 0.25), 0.001)
        self.gvf_iterations = max(gvf_iterations, 1)
        self.iterations = max(iterations, 1)

        # pU (edgeMap)
        # pV (edgeMap)

        # Manejar errores

        # Revisa que se haya entregado una imagen

        # Chequea que sea un array 2-dimensional

        # Puntero a la imagen

        # Puntero a coordenadas X e Y

        # Calcular npts igual a largo de X o 0 si es None
        pass

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
        #revisar plt.streamploat
        return

    # TODO: Setear las [x, y] coordenadas para el contorno activo
    def setContour(self) -> None:
        return

    # TODO:
    def getPerimeter(self,xyRes = [1,1]) -> float:
        """Calcula el perímetro de un contorno

        Parámetros:
        Parámetro: xyRes = [1,1] por defecto.

        Return:
        Float con perímetro total.
        Obs: en caso xCoords inválido, retorna -1.
        """
        p = self.getDistance(self, xyRes)
        return np.sum(p)
    
    def getDistance(self, xyRes = [1,1]) -> list[float]:
        """Calcula la distancia entre coordenadas consecutivas
        
        Parámetros:
        Parámetro: xyRes = [1,1] por defecto.

        Return:
        Array de floats distancia euclideana puntos de segmentos.
        Obs: en caso xCoords inválido, retorna -1.
        """
        dx = np.square(np.roll(self.xCoords,-1)-self.xCoords*xyRes[0])
        dy = np.square(np.roll(self.yCoords,-1)-self.yCoords*xyRes[1])
        return np.power(dx + dy, 0.5)

    # TODO:
    def arcSample(self) -> None:
        #utiliza interpolación cúbica //scipy.interpolate.CubicSpline
        return

    # TODO: aqui pasan muchas cosas
    def adjustContour(self) -> None:
        #llama función polygonPerimeter()
        #self -> arcSample (llama función arcSample)
        #matriz pentadiagonal -> scipy.sparse.diags
        #invierte matriz con numpy.linalg.inv(A) (usa descompos. LU) (status ?)
        #interpolation -> cubic convolution interpolation method
        #s_HausdorffDistanceFor2Dpoints -> scipy.spatial.distance.directed_hausdorff
        #calcNorm_L1ForVector
        #calcNorm_L2ForVector
        #calcNorm_LInfiniteForVector
        #calcNorm_LInfiniteForVector
        return


