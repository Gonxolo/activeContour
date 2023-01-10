import numpy as np
from scipy import interpolate
from scipy.ndimage import convolve

from geometryFunctions import polygon_perimeter


class ActiveContour:

    # TODO: Aqui recibimos los parametros para inicializar el algoritmo
    def __init__(self, image: list[list], x_coords: list[int], y_coords: list[int], alpha: float = 0.1, beta: float = 0.25,
                 gamma: float = 1.0, kappa: float = 0.0, mu: float = 0.1, gvf_iterations: int = 100,
                 iterations: int = 200) -> None:

        """[Descripcion de la clase]

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
        self.x = np.array(x_coords)
        self.y = np.array(y_coords)
        self.alpha = np.max(alpha, 0.001)
        self.beta = np.max(beta, 0.001)
        self.gamma = np.max(gamma, 0.1)
        self.kappa = np.max(kappa, 0.0)
        self.mu = np.max(min(mu, 0.25), 0.001)
        self.gvf_iterations = np.max(gvf_iterations, 1)
        self.iterations = np.max(iterations, 1)

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

        self.npts = self.x if self.x is not None else 0

        pass

    def get_x_coords(self):
        return self.x if bool(self.x) else -1
    
    def get_y_coords(self):
        return self.y if bool(self.y) else -1
    
    def get_GGVF(self):
        return [self.u, self.v] if bool(self.u) else -1

    def laplacian(self, image: np.ndarray) -> np.ndarray:

        """
        [Descripcion de la clase]

        :param image: [Descripcion del parametro]

        """

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

        # Calculate gradients [fx, fy] to initialize the vector field [u, v].
        self.edgeMap()

        # Original version for the GGVF by Xu99
        # b = np.square(self.u) + np.square(self.v)
        b = np.abs(self.u) + np.abs(self.v)

        # This pair of functions act as an "enhancer/de-enhancer" of high gradient neighbors the choice of the functions
        # must satisfy some convergence restrictions (see reference) TODO: agregar referencia
        g = np.exp(-b / self.mu)
        c1 = self.u * (np.ones(g.shape[0]) - g)
        c2 = self.v * (np.ones(g.shape[0]) - g)

        # Solve iteratively for the GGVF [u, v]
        # delta_x = delta_y = delta_t = 1
        for j in range(self.gvf_iterations):
            u_lap = self.laplacian(self.u)
            v_lap = self.laplacian(self.v)

            # Original iteration scheme
            # self.u += g * u_lap - h * (self.u - fx)
            # self.v += g * v_lap - h * (self.v - fy)

            # Optimized iteration scheme
            self.u = g * (self.u + u_lap) + c1
            self.v = g * (self.v + v_lap) + c2

        return

    # TODO: Plotear el campo GVF
    def plotGVF(self) -> None:
        #revisar plt.streamploat
        return

    def setContour(self, x: list, y: list) -> None:
        """Set the [x, y] coordinates for the active contour.

        Parameters
        ----------
        x : list
            `x` coordinate array of the contour to be set.
        y : list
            `y` coordinate array of the contour to be set.
        """
        self.x = np.array(x)
        self.y = np.array(y)
        self.npts = np.copy(self.x)
        return

    # TODO:
    def getPerimeter(self,xyRes = [1,1]) -> float:
        """The function calculates the perimeter of a contour.

        Parameters:
        Parameter 1: xyRes, set to [1,1] if it is not given.

        Return:
        A float with the value of the perimeter.
        Obs: in case xCoords is an invalid value, it returns -1.
        """
        p = self.getDistance(self, xyRes)
        return np.sum(p)
    
    def getDistance(self, xyRes = [1,1]) -> list[float]:
        """The function calculates the distance between consecutive points.
        
        Parameters:
        Parameter 1: xyRes, set to [1,1] if it is not given.

        Return:
        Array of floats with the euclidean distance between the consecutive points of a segment.
        Obs: in case xCoords is an invalid value, it returns -1.
        """
        dx = np.square(np.roll(self.x,-1)-self.x*xyRes[0])
        dy = np.square(np.roll(self.y,-1)-self.y*xyRes[1])
        return np.power(dx + dy, 0.5)

    # TODO:
    def arcSample(self) -> None:
        #utiliza interpolación cúbica //scipy.interpolate.CubicSpline
        return

    # TODO: aqui pasan muchas cosas
    def adjustContour(self, perimeter_factor, f_close, plot_contour, fix_point_count, 
                        fix_point_indices, f_keep_point_count, f_compute_convergence, 
                        convergence_thresh, convergence_metric_type, 
                        convergence_metric_value) -> None:
        """Runs the GVF Active Contour code to completion.

        Parameters
        ----------
        perimeter_factor : _type_
            _description_
        f_close : _type_
            _description_
        plot_contour : _type_
            _description_
        fix_point_count : _type_
            _description_
        fix_point_indices : _type_
            _description_
        f_keep_point_count : _type_
            _description_
        f_compute_convergence : _type_
            _description_
        convergence_thresh : _type_
            _description_
        convergence_metric_type : _type_
            _description_
        convergence_metric_value : _type_
            _description_

        Returns
        -------
        The {x, y} contour point list.
        """

        if len(plot_contour) == 0: plot_contour = 0

        # checkear si x e y son validos sino return -1

        if len(fix_point_indices) > 0:
            fix_point_count = len(fix_point_indices)
        else:
            if len(fix_point_count) > 0:
                fix_point_count = np.max(fix_point_count, 0)
            else:
                fix_point_count = 0
        
        try:
            if fix_point_count == 0:
                npts_iter = np.max(round(polygon_perimeter(self.x, self.y) * np.max(perimeter_factor, 0.1)), 5)
            else:
                npts_iter = self.npts

        except NameError: # En caso de que perimeter_factor no este definido
            npts_iter = self.npts
        
        if npts_iter != self.npts: # TODO: and ~keyword_set(fKeepPointCount)
            self.arcSample(points=npts_iter, f_close=f_close)

        perimeter_it0 = polygon_perimeter(self.x, self.y)

        alpha = np.full(npts_iter, self.alpha)
        beta = np.full(npts_iter, self.beta)
        a = beta
        b = -alpha - 4*beta
        c = 2*alpha + 6*beta
        vfx = 0.0
        vfy = 0.0

        abc_matrix = np.diag(a[0:npts_iter-2], 2) + np.diag(a[npts_iter-2:npts_iter], -(npts_iter-2)) \
                    + np.diag(b[0:npts_iter-1], 1) + np.diag(b[npts_iter-1], -(npts_iter-1)) \
                    + np.diag(c + self.gamma) \
                    + np.diag(b[0:npts_iter-1], -1) + np.diag(b[npts_iter-1], (npts_iter-1)) \
                    + np.diag(a[0:npts_iter-2], -2) + np.diag(a[npts_iter-2:npts_iter], (npts_iter-2))

        inv_array = np.linalg.inv(abc_matrix)

        f_use_convergence_threshold = bool(convergence_thresh) or (convergence_thresh == 0) # TODO: convergence_thresh == 0 -> False
        f_compute_convergence = f_compute_convergence \
                                or convergence_metric_value \
                                or f_use_convergence_threshold

        if f_compute_convergence:
            try:
                var_metric = convergence_metric_type
            except NameError:
                var_metric = 'LinfNorm'
        
        if self.iterations >= 1:

            for j in range(self.iterations):

                if f_compute_convergence:
                    last_iter_x = np.copy(self.x)
                    last_iter_y = np.copy(self.y)

                if self.kappa > 0: # TODO: Encontrar remplazo para IDL:interpolate
                    cs = interpolate.CubicSpline(self.x, self.y)
                    vfx = cs(self.u)
                    vfy = cs(self.v)

                n_elem_inv_array = inv_array.shape[0]
                n_elem_contour = len(self.x)

                if n_elem_inv_array != n_elem_contour:
                    
                    npts_iter = n_elem_contour
                    alpha = np.full(npts_iter, self.alpha)
                    beta = np.full(npts_iter, self.beta)
                    
                    a = beta
                    b = -alpha - 4*beta
                    c = 2*alpha + 6*beta

                    abc_matrix = np.diag(a[0:npts_iter-2], 2) + np.diag(a[npts_iter-2:npts_iter], -(npts_iter-2)) \
                                + np.diag(b[0:npts_iter-1], 1) + np.diag(b[npts_iter-1], -(npts_iter-1)) \
                                + np.diag(c + self.gamma) \
                                + np.diag(b[0:npts_iter-1], -1) + np.diag(b[npts_iter-1], (npts_iter-1)) \
                                + np.diag(a[0:npts_iter-2], -2) + np.diag(a[npts_iter-2:npts_iter], (npts_iter-2))

                    inv_array = np.linalg.inv(abc_matrix)

                if fix_point_count > 0: # TODO and ~keyword_set(fClose)

                    x_tmp = np.matmul(inv_array, (self.gamma * self.x + self.kappa * vfx))
                    y_tmp = np.matmul(inv_array, (self.gamma * self.y + self.kappa * vfy))

                    if len(fix_point_indices) > 0:
                        x_tmp[fix_point_indices] = self.x[fix_point_indices]
                        y_tmp[fix_point_indices] = self.y[fix_point_indices]

                        if f_compute_convergence:
                            x_delta = np.abs(x_tmp - self.x)
                            y_delta = np.abs(y_tmp - self.y)

                        if perimeter_factor: # TODO keyword_set(perimeter_factor)
                            pass # pendiente

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
        return [self.x, self.y]

    # Se asume un int en direction TODO: deberia poder admitir vectores?
    # TODO: este metodo puede ser estatico
    def gradient(self, image: np.ndarray, direction: int) -> np.ndarray:

        # TODO: si la direccion admite vectores entonces
        # Revisa la cantidad de elementos de la direccion
        # if n_elements(direction) == 0 -> direction = 0

        # Chequeo de la dimension de la imagen
        # Si la dimension no es 2 retornar -1

        # IDL: shift
        # Python: np.roll. Reference: https://numpy.org/doc/stable/reference/generated/numpy.roll.html

        # np matrix accessors reference:
        # https://stackoverflow.com/questions/4455076/how-do-i-access-the-ith-column-of-a-numpy-multidimensional-array

        if direction == 0:
            theGradient = (np.roll(image, -1, axis=1) - np.roll(image, 1, axis=1)) * 0.5
            theGradient[:, 0] = theGradient[:, 1]
            theGradient[:, theGradient.shape[1] - 1] = theGradient[:, theGradient.shape[1] - 2]
        elif direction == 1:
            theGradient = (np.roll(image, -1, axis=0) - np.roll(image, 1, axis=0)) * 0.5
            theGradient[0, :] = theGradient[1, :]
            theGradient[theGradient.shape[0] - 1, :] = theGradient[theGradient.shape[0] - 2, :]
        else:
            return -1 # Reemplazar este valor por algo mas indicativo. Una excepcion si solo se recibe direccion {0, 1}

        return theGradient

    def edgeMap(self) -> None:

        edge_map = np.sqrt(np.square(self.gradient(self.image, 0)) + np.square(self.gradient(self.image, 1)))
        min_val = np.min(edge_map) # TODO: este valor por defecto podia serr 0, hablar con Jorge
        max_val = np.max(edge_map)

        if max_val != min_val:
            edge_map = np.array([(i - min_val)/(max_val - min_val) for i in edge_map])

        self.u = self.gradient(edge_map, 0)
        self.v = self.gradient(edge_map, 1)
