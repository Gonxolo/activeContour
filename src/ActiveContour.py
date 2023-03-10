import numpy as np
from scipy import interpolate
from scipy.ndimage import convolve
from scipy.interpolate import CubicSpline


from .geometryFunctions import *
from .hausdorffDistanceCalc import hausdorffDistanceFor2Dpoints


class ActiveContour:
    """Class that represents the snake, the contour that deforms until reaching a state of equilibrium through an
    optimization process. This class contains the methods needed to do that.

    Methods
    -------

    adjustContour(perimeter_factor=None, f_close=None, plot_contour=None, fix_point_count=None, 
                    fix_point_indices=None, f_keep_point_count=None, f_compute_convergence=None, 
                    convergence_thresh=None, convergence_metric_type=None, 
                    convergence_metric_value=None)
                    Runs the GGVF Active Contour code to completion. It deforms iteratively the initial contour until it 
                    reaches a sate of equilibrium.

    """

    # TODO: Aqui recibimos los parametros para inicializar el algoritmo
    def __init__(self, image: list, x_coords: list = None, y_coords: list = None, alpha: float = 0.1, beta: float = 0.25,
                 gamma: float = 1.0, kappa: float = 0.0, mu: float = 0.1, vf_iterations: int = 100,
                 contour_iterations: int = 200) -> None:

        """Initializes ActiveContour with the parameters given by the user, or 
        the default parameter values if not given. 

        Parameters
        ----------

        image : list
                Image file for which the active contour (snake) will be applied.
                This argument must be 2D.

        x_coords : list
                The initial X points of the active contour of the snake. Optional.
                Must be used with Y.

        y_coords : list
                The initial Y points of the active contour of the snake. Optional.
                Must be used with X.

        alpha : float 
                The elasticity parameter of the active contour. It reflects the 
                contour's ability to stretch along its length. Default: 0.10.

        beta : float
                The rigidity parameter of the active contour. It reflects the 
                contour's ability to bend, as, for example, around corners. 
                Default: 0.25.

        gamma : float
                The viscosity parameter. Larger values make it harder to deform 
                the active contour in space. Default: 1.0.

        kappa : float
                The external force weight. Default: 1.25.

        mu : float
                The regularization parameter. this should be set according 
                to the amount of noise in the image. 
                Use a larger value for noisier images. Default: 0.10.

        vf_iterations : int
                The number of iterations for calculation the Gradient Vector 
                Flow (GVF). Default: 100.

        contour_iterations : int
                The number of iterations to use in calculating the snake 
                positions. Default: 200.

        """
        
        #para asegurar tipo de arrays, poner el tipo (d_type=float64? o double)

        self.image = np.array(image, dtype=np.float64)
        self.x = x_coords
        self.y = y_coords
        self.alpha = max(alpha, 0.001)
        self.beta = max(beta, 0.001)
        self.gamma = max(gamma, 0.1)
        self.kappa = max(kappa, 0.0)
        self.mu = max(min(mu, 0.25), 0.001)
        self.vf_iterations = max(int(vf_iterations), 1)
        self.contour_iterations = max(int(contour_iterations), 1)

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

        self.npts = len(self.x) if self.x is not None else 0

        pass

    def get_x_coords(self):
        """It returns the array of coordinates x of the contour.

        Returns
        -------
        np.ndarray
            X coordinates of the contour.
        """
        try:
            if len(self.x) <= 0:
                return -1
            self.x = np.array(self.x, dtype=np.float64)
            return self.x
        except ValueError:
            return  -1
    
    def get_y_coords(self):
        """It returns the array of coordinates y of the contour.

        Returns
        -------
        np.ndarray
            Y coordinates of the contour.
        """
        try:
            if len(self.y) <= 0:
                return -1
            self.y = np.array(self.y, dtype=np.float64)
            return self.y
        except ValueError:
            return  -1
    
    def get_GGVF(self):
        """It returns the values of the GGVF, with its coordinates u and v.

        Returns
        -------
        np.ndarray
            It returns an array with the value of the GGVF.
        """
        try:
            if len(self.u) <= 0:
                return -1

            if len(self.v) <= 0:
                return -1

            self.u = np.array(self.u, dtype=np.float64)
            self.v = np.array(self.v, dtype=np.float64)
            return np.array([self.u, self.v], dtype=np.float64)
            
        except ValueError:
            return  -1

    def laplacian(self, image: np.ndarray) -> np.ndarray:
        """Computes the laplacian of an image.

        Parameters
        ----------
        image : np.ndarray
            Image to apply the laplacian.

        Returns
        -------
        np.ndarray
            Returns the laplacian of the image, computing the convolution of it with a kernel.
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
        """It computes the GGVF, and sets the values of self.u and self.v.
        """

        # Calculate gradients [fx, fy] to initialize the vector field [u, v].
        self.edgeMap()

        # Original version for the GGVF by Xu99
        # b = np.square(self.u) + np.square(self.v)
        b = np.abs(self.u) + np.abs(self.v)

        # This pair of functions act as an "enhancer/de-enhancer" of high gradient neighbors the choice of the functions
        # must satisfy some convergence restrictions (see reference) TODO: agregar referencia
        g = np.exp(-b / self.mu)
        c1 = self.u * (np.ones(g.shape) - g)
        c2 = self.v * (np.ones(g.shape) - g)

        # Solve iteratively for the GGVF [u, v]
        # delta_x = delta_y = delta_t = 1
        for _ in range(1, self.vf_iterations + 1):
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

    def getCoords(self, xyRes = np.array([1.,1.], dtype=np.float64)) -> np.ndarray:
        """It returns the coordinates x and y of the image.

        Parameters
        ----------
        xyRes : np.ndarray of floats, optional
            Resolution of the image, by default np.array([1.,1.])

        Returns
        -------
        np.ndarray
            Coordinates x and y of the image
             note:: in case xCoords is an invalid value, it returns -1.
        """
        
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        
        return np.array([xyRes[0] * self.x, xyRes[1] * self.y], dtype=np.float64)

    def setContour(self, x: list, y: list) -> None:
        """Set the [x, y] coordinates for the active contour.

        Parameters
        ----------
        x : list
            `x` coordinate array of the contour to be set.
        y : list
            `y` coordinate array of the contour to be set.
        """
        self.x = np.array(x, dtype=np.float64)
        self.y = np.array(y, dtype=np.float64)
        self.npts = len(self.x)
        return

    def getPerimeter(self,xyRes = np.array([1.,1.], dtype=np.float64)) -> float:
        """This method calculates the perimeter of a contour.

        Parameters:
        -----------
        xyRes: np.ndarray of floats, optional.
            Resolution of the image. Default: np.array([1,1]).

        Returns:
        -------
        float
            Value of the perimeter of a contour.
            note:: in case xCoords is an invalid value, it returns -1.
        """
        p = self.getDistance(xyRes)
        return np.sum(p)
    
    def getDistance(self, xyRes = np.array([1.,1.], dtype=np.float64)) -> np.ndarray:
        """This method calculates the distance between consecutive points.
        
        Parameters:
        -----------
        xyRes: np.ndarray of floats, optional.
            Resolution of the image, by default np.array([1,1]).

        Returns:
        -------
        np.ndarray
            Array of floats with the euclidean distance between the consecutive points of a segment.
            note:: in case xCoords is an invalid value, it returns -1.
        """
        dx = np.square((np.roll(self.x,-1)-self.x)*xyRes[0])
        dy = np.square((np.roll(self.y,-1)-self.y)*xyRes[1])
        return np.power(dx + dy, 0.5)

    def arcSample(self, points = 50, f_close = None) -> None:
        """It takes a closed curve and re-samples it in equal arc lengths.

        Parameters
        ----------
        points : int, optional
            The number of points in the output vectors, by default 50.
        f_close : bool, optional
            Set this keyword to True to specify the contour curve, by default None.
        """
        #if size(*self.pX,/n_dimensions) eq 2 then begin, x_in = reform(*self.pX) ... ya lo hace python
        x_in = np.copy(self.x)
        y_in = np.copy(self.y)

        npts = len(x_in)
        
        #Make sure the curve is closed (first point same as last point).
        if bool(f_close): 
            if (x_in[0] != x_in[npts - 1]) or (y_in[0] != y_in[npts - 1]):
                x_in = np.concatenate((x_in, np.array([x_in[0]], dtype=np.float64)))
                y_in = np.concatenate((y_in, np.array([y_in[0]], dtype=np.float64)))
                # print, "Active contour interpolation warning: adding 1 point to close the contour,
                # according to the specified input"
                npts += 1
        else:
            points -= 1
        
        #Interpolate very finely
        nc = (npts - 1) * 100
        t = np.arange(npts, dtype=np.float64)
        t1 = np.arange(nc + 1, dtype=np.float64) / 100 
        csx = CubicSpline(t, x_in) 
        x1 = csx(t1)
        csy = CubicSpline(t, y_in)
        y1 = csy(t1)

        if bool(f_close):
            #computes the boundary condition for the cubic spline: derivatives at the beggining and end points are the same
            avg_slopeX = (x1[1] - x1[0] + x1[nc] - x1[nc - 1]) / (t1[1] - t1[0]) * 0.5
            avg_slopeY = (y1[1] - y1[0] + y1[nc] - y1[nc - 1]) / (t1[1] - t1[0]) * 0.5
            dx1 = CubicSpline(t, x_in, bc_type = ((1, avg_slopeX), (1, avg_slopeX))) 
            dy1 = CubicSpline(t, y_in, bc_type = ((1, avg_slopeY), (1, avg_slopeY))) 

        else:
            #computes the boundary condition for the cubic spline: derivatives at the beggining and end points
            avg_slopeX0 = (x1[1] - x1[0]) / (t1[1] - t1[0])
            avg_slopeX1 = (x1[nc] - x1[nc - 1]) / (t1[nc] - t1[nc - 1])
            avg_slopeY0 = (y1[1] - y1[0]) / (t1[1] - t1[0])
            avg_slopeY1 = (y1[nc] - y1[nc - 1]) / (t1[nc] - t1[nc - 1])
            dx1 = CubicSpline(t, x_in, bc_type = ((1, avg_slopeX0), (1, avg_slopeX1))) 
            dy1 = CubicSpline(t, y_in, bc_type = ((1, avg_slopeY0), (1, avg_slopeY1))) 
        
        x1 = dx1(t1)
        y1 = dy1(t1)

        #compute cumulative path length.
        ds = np.sqrt(np.square((x1[1:] - x1[:len(x1)-1])) + np.square((y1[1:] - y1[:len(y1)-1])))

        ss = np.concatenate((np.array([0], dtype=np.float64), np.cumsum(ds, dtype=np.float64)), axis = None, dtype=np.float64)

        #Invert this curve, solve for TX, which should be evenly sampled in the arc length space.
        sx = np.arange(points) * (ss[nc] / points)
        cstx = CubicSpline(ss, t1)
        tx = cstx(sx)

        #Reinterpolate the original points using the new values of TX and optionally close the contour.
        if bool(f_close):
            x_out = dx1(tx)
            y_out = dy1(tx)
            self.x = np.concatenate((x_out, np.array([x_out[0]], dtype=np.float64)), axis = None)
            self.y = np.concatenate((y_out, np.array([y_out[0]], dtype=np.float64)), axis = None)
        else:
            x_out = dx1(tx)
            y_out = dy1(tx)

            self.x = np.concatenate((x_out, np.array([x_in[npts - 1]], dtype=np.float64)), axis = None)
            self.y = np.concatenate((y_out, np.array([y_in[npts - 1]], dtype=np.float64)), axis = None)

        
        self.npts = len(self.x)
        

    def adjustContour(self, perimeter_factor=None, f_close=None, plot_contour=None, fix_point_count=None, 
                        fix_point_indices=None, f_keep_point_count=None, f_compute_convergence=None, 
                        convergence_thresh=None, convergence_metric_type=None, 
                        convergence_metric_value=None) -> np.ndarray:
        
        """Runs the GGVF Active Contour code to completion. It deforms iteratively the initial contour until it 
        reaches a sate of equilibrium.

        Parameters
        ----------
        perimeter_factor: float, optional
            It indicates the factor to compute the number of points to use for a given contour, by default None.

        f_close: bool, optional
            Flag to indicate if a contour must be closed, adding an extra point, by default None.

        plot_contour: int, optional
            Flag to indicate if a contour must be closed, adding an extra point, by default None.

        fix_point_count: int, optional
            Number of points to fix in the contour, by default None.

        fix_point_indices: np.ndarray, optional
            Array of indices of the points to be fixed.

        f_keep_point_count: bool, optional
            Flag to indicate that the number of points in the contour must be kept, and if False, it re-interpolates them.

        f_compute_convergence: bool, optional
            Flaf to indicate that the convergence must be computed.

        convergence_thresh: float, optional
            Value of the convergence limit.

        convergence_metric_type: string, optional
            It indicates the type of metric to compute the variation between the contour in the current
            iteration and the last iteration, until reaching convergence.

        convergence_metric_value: float, optional
            It indicates that the convergence must be computed, in case the flags f_compute_convergence or 
            f_use_convergence_threshold where not set.

        Returns
        -------
        np.ndarray
            It returns an array with the coordinates x and y of the contour after the adjustment, when the state of 
            equilibrium is reached.
        """

        if plot_contour is None: plot_contour = 0

        # checkear si x e y son validos sino return -1
        if not isinstance(self.get_x_coords(), np.ndarray):
            return -1

        if not isinstance(self.get_y_coords(), np.ndarray):
            return -1

        try:
            if len(fix_point_indices) > 0:
                fix_point_count = len(fix_point_indices)
            elif len(fix_point_count) > 0: # TODO: Esto deberia chequear si se fijo la variable fix_point_count
                fix_point_count = max(fix_point_count, 0)
            else:
                fix_point_count = 0
        except TypeError:
            fix_point_count = 0

        try:
            if fix_point_count == 0:
                npts_iter = max(round(polygon_perimeter(self.x, self.y) * max(perimeter_factor, 0.1)), 5)
            else:
                npts_iter = self.npts

        # En caso de que perimeter_factor no este definido 
        except NameError: 
            print("perimeter_factor no esta definido")
            npts_iter = self.npts
        
        # En caso de que perimeter_factor este definido pero tenga el tipo
        # incorrecto
        except TypeError: 
            npts_iter = self.npts

        if npts_iter != self.npts and not bool(f_keep_point_count):
            self.arcSample(points=npts_iter, f_close=f_close)
            # self.x, self.y = polygon_line_sample(np.copy(self.x), np.copy(self.y), n_points_per_pix = npts_iter)

        perimeter_it_0 = polygon_perimeter(self.x, self.y)

        alpha = np.full(npts_iter, self.alpha)
        beta = np.full(npts_iter, self.beta)
        a = beta
        b = -alpha - 4*beta
        c = 2*alpha + 6*beta
        vfx = 0.0
        vfy = 0.0

        abc_matrix = np.diag(a[0:npts_iter-2], 2) + np.diag(a[npts_iter-2:npts_iter], -(npts_iter-2)) \
                    + np.diag(b[0:npts_iter-1], 1) + np.diag([b[npts_iter-1]], -(npts_iter-1)) \
                    + np.diag(c + self.gamma) \
                    + np.diag(b[0:npts_iter-1], -1) + np.diag([b[npts_iter-1]], (npts_iter-1)) \
                    + np.diag(a[0:npts_iter-2], -2) + np.diag(a[npts_iter-2:npts_iter], (npts_iter-2))

        inv_array = np.linalg.inv(abc_matrix)

        f_use_convergence_threshold = bool(convergence_thresh) or (convergence_thresh == 0)
        f_compute_convergence = bool(f_compute_convergence) \
                                or bool(convergence_metric_value) \
                                or f_use_convergence_threshold

        if f_compute_convergence:
            if bool(convergence_metric_type):
                var_metric = convergence_metric_type
            else:
                var_metric = 'LinfNorm'
        
        if self.contour_iterations >= 1:

            for j in range(self.contour_iterations):

                if f_compute_convergence:
                    last_iter_x = np.copy(self.x)
                    last_iter_y = np.copy(self.y)

                if self.kappa > 0:
                    points = (np.arange(self.image.shape[0]), np.arange(self.image.shape[1]))

                    xi = np.transpose(np.vstack((self.x, self.y)))

                    vfx = interpolate.interpn(points, self.v, xi, method='cubic')
                    vfy = interpolate.interpn(points, self.u, xi, method='cubic')

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
                                + np.diag(b[0:npts_iter-1], 1) + np.diag([b[npts_iter-1]], -(npts_iter-1)) \
                                + np.diag(c + self.gamma) \
                                + np.diag(b[0:npts_iter-1], -1) + np.diag([b[npts_iter-1]], (npts_iter-1)) \
                                + np.diag(a[0:npts_iter-2], -2) + np.diag(a[npts_iter-2:npts_iter], (npts_iter-2))

                    inv_array = np.linalg.inv(abc_matrix)

                # Deform the snake.
                if (fix_point_count > 0) and not bool(f_close):

                    x_tmp = np.matmul(inv_array, (self.gamma * self.x + self.kappa * vfx))
                    y_tmp = np.matmul(inv_array, (self.gamma * self.y + self.kappa * vfy))

                    if len(fix_point_indices) > 0:
                        x_tmp[fix_point_indices] = self.x[fix_point_indices]
                        y_tmp[fix_point_indices] = self.y[fix_point_indices]

                        if f_compute_convergence:
                            x_delta = np.abs(x_tmp - self.x)
                            y_delta = np.abs(y_tmp - self.y)

                        # Re-interpolate the snake points.
                        if bool(perimeter_factor):
                            poly_line_length = 0.0
                            for k in range(len(x_tmp) - 1):
                                poly_line_length += np.sqrt(np.square(x_tmp[k+1] - x_tmp[k]) 
                                                    + np.square(y_tmp[k+1] - y_tmp[k]))
                            npts_iter = max((round(poly_line_length) * max(perimeter_factor, 0.1)), 5)

                        if not bool(f_keep_point_count):
                            self.arcSample(points=npts_iter)
                            # self.x, self.y = polygon_line_sample(np.copy(self.x), np.copy(self.y), n_points_per_pix = npts_iter)

                        self.x = x_tmp
                        self.y = y_tmp

                    else:
                        x_fix_vec_1 = self.x[0 : fix_point_count]
                        x_fix_vec_2 = self.x[npts_iter - fix_point_count :]
                        y_fix_vec_1 = self.y[0 : fix_point_count]
                        y_fix_vec_2 = self.y[npts_iter - fix_point_count :]

                        x_tmp_2 = np.concatenate((x_fix_vec_1, x_tmp[1 : n_elem_contour - fix_point_count + 1], x_fix_vec_2))
                        y_tmp_2 = np.concatenate((y_fix_vec_1, y_tmp[1 : n_elem_contour - fix_point_count + 1], y_fix_vec_2))

                        if f_compute_convergence:
                            x_delta = np.abs(x_tmp_2 - self.x)
                            y_delta = np.abs(y_tmp_2 - self.y)

                        # Re-interpolate the snake points.
                        if bool(perimeter_factor):
                            poly_line_length = 0.0
                            for k in range(len(self.x) - 1):
                                poly_line_length += np.sqrt(np.square(x_tmp_2[k+1] - x_tmp_2[k]) 
                                                    + np.square(y_tmp_2[k+1] - y_tmp_2[k]))
                            npts_iter = max((round(poly_line_length) * max(perimeter_factor, 0.1)), 5)

                        if not bool(f_keep_point_count):
                            self.arcSample(points=npts_iter)
                            # self.x, self.y = polygon_line_sample(np.copy(self.x), np.copy(self.y), n_points_per_pix = npts_iter)

                        # Put back the fixed points
                        x_tmp_2[0:fix_point_count] = x_fix_vec_1
                        x_tmp_2[npts_iter - fix_point_count + 1 :] = x_fix_vec_2
                        y_tmp_2[0:fix_point_count] = y_fix_vec_1
                        y_tmp_2[npts_iter - fix_point_count + 1 :] = y_fix_vec_2

                        self.x = x_tmp_2
                        self.y = y_tmp_2

                else: # Non-fixed points, i.e. all the contour pointscan be displaced

                    if f_compute_convergence:
                        x_tmp_3 = np.matmul(inv_array, (self.gamma * self.x + self.kappa * vfx))
                        y_tmp_3 = np.matmul(inv_array, (self.gamma * self.y + self.kappa * vfy))
                        x_delta = np.abs(x_tmp_3 - self.x)
                        y_delta = np.abs(y_tmp_3 - self.y)
                        self.x = x_tmp_3
                        self.y = y_tmp_3
                    
                    else:
                        self.x = np.matmul(inv_array, (self.gamma * self.x + self.kappa * vfx))
                        self.y = np.matmul(inv_array, (self.gamma * self.y + self.kappa * vfy))
                    
                    # Re-interpolate the snake points.
                    if bool(perimeter_factor):
                        npts_iter = max((round(polygon_perimeter(self.x, self.y) * max(perimeter_factor, 0.1))), 5)
                    
                    f_close = 1

                    if not bool(f_keep_point_count):
                        # self.x, self.y = polygon_line_sample(np.copy(self.x), np.copy(self.y), n_points_per_pix = (npts_iter-1 if bool(f_close) else npts_iter), f_close_output=f_close)
                        self.arcSample(points=((npts_iter-1) if f_close else npts_iter), f_close=f_close)
                
                if plot_contour > 0:
                    if j == 1: pass # TODO: oPlot, [*self.pX, (*self.pX)[0]], [*self.pY, (*self.pY)[0]], color = 255, linestyle = 1, thick = 3
                    elif j == self.contour_iterations: pass # TODO: oPlot, [*self.pX, (*self.pX)[0]], [*self.pY, (*self.pY)[0]], color = 255, thick = 3
                    else: pass # TODO: oPlot, [*self.pX, (*self.pX)[0]], [*self.pY, (*self.pY)[0]], color = (255 - (self.contour_iterations - j) * 30) > 100
            
                if f_compute_convergence:
                    delta_mag = np.sqrt(np.square(x_delta) + np.square(y_delta))

                    if var_metric == 'Hausdorff': variation = hausdorffDistanceFor2Dpoints(self.x, self.y, last_iter_x, last_iter_y)
                    elif var_metric == 'L1norm'   : variation = calcNorm_L1ForVector(delta_mag)
                    elif var_metric == 'L2norm'   : variation = calcNorm_L2ForVector(delta_mag)
                    elif var_metric == 'LinfNorm' : variation = calcNorm_LInfiniteForVector(delta_mag)
                    elif var_metric == 'average'  : variation = np.mean(delta_mag)
                    elif var_metric == 'avgFracPerimeter': variation = np.mean(delta_mag) / polygon_perimeter(self.x, self.y)
                    elif var_metric == 'avgFracPerimeter0': variation = np.mean(delta_mag) / perimeter_it_0
                    else: variation = calcNorm_LInfiniteForVector(delta_mag)

                    f_log = 0
                    f_log_all = 0

                    if f_log:
                        log_file_path = 'D:\\tmp\\snakeLog.txt'
                        msg = f"{var_metric} convergence criterion value = {variation} at iteration {j}"
                        if f_log_all:
                            hd = hausdorffDistanceFor2Dpoints(self.x, self.y, last_iter_x, last_iter_y)
                            l1 = calcNorm_L1ForVector(delta_mag)
                            l2 = calcNorm_L2ForVector(delta_mag)
                            li = calcNorm_LInfiniteForVector(delta_mag)
                            avg = np.mean(delta_mag)
                            avg_norm_perim_it = np.mean(delta_mag) / polygon_perimeter(self.x, self.y)
                            avg_norm_perim_0  = np.mean(delta_mag) / perimeter_it_0
                            msg = ";".join(list(map(str, [hd, l1, l2, li, avg, avg_norm_perim_it, avg_norm_perim_0, j])))
                        # TODO: file_logger(msg, log_file_path)
                    
                    if f_use_convergence_threshold:
                        if variation <= convergence_thresh:
                            break
            
            if f_compute_convergence:
                if f_log_all:
                    print('Hausdorff', ' L1norm', ' L2norm', ' LinfNorm', ' average', ' avgFracPerimeter', ' avgFracPerimeter0', ' Iteration')
                elif f_log:
                    print(msg)
                convergence_metric_value = variation

        return np.array([self.x, self.y], dtype=np.float64)

    # TODO: este metodo puede ser estatico
    def gradient(self, image: np.ndarray, direction: int) -> np.ndarray:
        """Computes the gradient of the image's intensity.

        Parameters
        ----------
        image : np.ndarray
            Array with the intensity values per pixel of the image.
        direction : int
            It indicates the direction in which to calculate the gradient.

        Returns
        -------
        np.ndarray
            Returns the result of computing the gradient of the image.
        """

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

        return np.array(theGradient, dtype=np.float64)

    def edgeMap(self) -> None:
        """Computes the edge map of a given image, and sets the values of the coordinates u and v of the GGVF.
        """

        edge_map = np.sqrt(np.square(self.gradient(self.image, 0)) + np.square(self.gradient(self.image, 1)))
        min_val = np.min(edge_map) # TODO: este valor por defecto podia ser 0, hablar con Jorge
        max_val = np.max(edge_map)

        if max_val != min_val:
            edge_map = np.array([(i - min_val)/(max_val - min_val) for i in edge_map], dtype=np.float64)

        self.u = self.gradient(edge_map, 0)
        self.v = self.gradient(edge_map, 1)
