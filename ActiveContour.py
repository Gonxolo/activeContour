import numpy as np
from scipy import interpolate
from scipy.ndimage import convolve

from geometryFunctions import polygon_perimeter
from scipy.interpolate import CubicSpline


class ActiveContour:

    # TODO: Aqui recibimos los parametros para inicializar el algoritmo
    def __init__(self, image: list, x_coords: list, y_coords: list, alpha: float = 0.1, beta: float = 0.25,
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
        #para asegurar tipo de arrays, poner el tipo (d_type=float64? o double)

        self.image = np.array(image)
        self.x = np.array(x_coords)
        self.y = np.array(y_coords)
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

        self.npts = len(self.x) if self.x is not None else 0

        pass

    def get_x_coords(self):
        return self.x if bool(self.x) else -1
    
    def get_y_coords(self):
        return self.y if bool(self.y) else -1
    
    def get_GGVF(self):
        return np.array([self.u, self.v]) if bool(self.u) else -1

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
        c1 = self.u * (np.ones(g.shape) - g)
        c2 = self.v * (np.ones(g.shape) - g)

        # Solve iteratively for the GGVF [u, v]
        # delta_x = delta_y = delta_t = 1
        for _ in range(1, self.gvf_iterations + 1):
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

    def getCoords(self, xyRes = np.array([1.,1.])) -> np.ndarray:
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
        return np.array([xyRes[0] * self.x, xyRes[1] * self.y])

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
        self.npts = len(self.x)
        return

    # TODO:
    def getPerimeter(self,xyRes = np.array([1.,1.])) -> float:
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
    
    def getDistance(self, xyRes = np.array([1.,1.])) -> np.ndarray:
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

    # TODO:
    def arcSample(self, points = 50, f_close = None) -> None:
        """It takes a closed curve and re-samples it in equal arc lengths.

        Parameters
        ----------
        points : int, optional
            The number of points in the output vectors, by default 50.
        f_close : _type_, optional
            Set this keyword to True to specify the contour curve, by default None.
        """
        #if size(*self.pX,/n_dimensions) eq 2 then begin, x_in = reform(*self.pX) ... ya lo hace python
        x_in = np.copy(self.x)
        y_in = np.copy(self.y)

        npts = len(x_in)
        
        #Make sure the curve is closed (first point same as last point).
        if bool(f_close):
            if (x_in[0] != x_in[npts - 1]) or (y_in[0] != y_in[npts - 1]):
                x_in = np.concatenate((x_in, np.array([x_in[0]])))
                y_in = np.concatenate((y_in, np.array([y_in[0]])))
                # print, "Active contour interpolation warning: adding 1 point to close the contour,
                # according to the specified input"
                npts += 1
        else:
            points -= 1
        
        #Interpolate very finely
        nc = (npts - 1) * 100
        t = np.arange(npts)
        t1 = np.arange(nc + 1) / 100 
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
        ds = np.sqrt(np.square((x1[1:] - x1)) + np.square((y1[1:] - y1)))
        ss = np.concatenate((np.array([0]), np.cumsum(ds)), axis = None)

        #Invert this curve, solve for TX, which should be evenly sampled in the arc length space.
        sx = np.arange(points) * (ss[nc] / points)
        cstx = CubicSpline(ss, t1)
        tx = cstx(sx)

        #Reinterpolate the original points using the new values of TX and optionally close the contour.
        if bool(f_close):
            x_out = dx1(tx)
            y_out = dy1(tx)
            self.x = np.concatenate((x_out, np.array([x_out[0]])), axis = None)
            self.y = np.concatenate((y_out, np.array([y_out[0]])), axis = None)
        else:
            x_out = dx1(tx)
            y_out = dy1(tx)
            self.xCoords = np.concatenate((x_out, np.array([x_in[npts - 1]])), axis = None)
            self.yCoords = np.concatenate((y_out, np.array([y_in[npts - 1]])), axis = None)
        
        self.npts = len(self.xCoords)
        

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

        perimeter_it_0 = polygon_perimeter(self.x, self.y)

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

                # Deform the snake.
                if fix_point_count > 0: # TODO and ~keyword_set(fClose)

                    x_tmp = np.matmul(inv_array, (self.gamma * self.x + self.kappa * vfx))
                    y_tmp = np.matmul(inv_array, (self.gamma * self.y + self.kappa * vfy))

                    if len(fix_point_indices) > 0:
                        x_tmp[fix_point_indices] = self.x[fix_point_indices]
                        y_tmp[fix_point_indices] = self.y[fix_point_indices]

                        if f_compute_convergence:
                            x_delta = np.abs(x_tmp - self.x)
                            y_delta = np.abs(y_tmp - self.y)

                        # Re-interpolate the snake points.
                        if perimeter_factor: # TODO keyword_set(perimeter_factor)
                            poly_line_length = 0.0
                            for k in range(len(x_tmp) - 1):
                                poly_line_length += np.emath.sqrt(np.square(x_tmp[k+1] - x_tmp[k]) 
                                                    + np.square(y_tmp[k+1] - y_tmp[k]))
                            npts_iter = np.max((round(poly_line_length) * np.max(perimeter_factor, 0.1)), 5)

                        if not f_keep_point_count: # TODO: ~keyword_set(f_point_count)
                            self.arcSample(points=npts_iter)

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
                        if perimeter_factor: # TODO keyword_set(perimeter_factor)
                            poly_line_length = 0.0
                            for k in range(len(self.x) - 1):
                                poly_line_length += np.emath.sqrt(np.square(x_tmp_2[k+1] - x_tmp_2[k]) 
                                                    + np.square(y_tmp_2[k+1] - y_tmp_2[k]))
                            npts_iter = np.max((round(poly_line_length) * np.max(perimeter_factor, 0.1)), 5)

                        if not f_keep_point_count:
                            self.arcSample(points=npts_iter)
                        
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
                    if perimeter_factor: # TODO: keyword_set(perimeter_factor)
                        npts_iter = np.max((round(polygon_perimeter(self.x, self.y) * np.max(perimeter_factor, 0.1))), 5)
                    
                    f_close = 1

                    if not f_keep_point_count: # TODO: ~keyword_set(f_keep_point_count)
                        self.arcSample(points=npts_iter-1 if f_close else npts_iter, f_close=f_close)
                
                if plot_contour > 0:
                    if j == 1: pass # TODO: oPlot, [*self.pX, (*self.pX)[0]], [*self.pY, (*self.pY)[0]], color = 255, linestyle = 1, thick = 3
                    elif j == self.iterations: pass # TODO: oPlot, [*self.pX, (*self.pX)[0]], [*self.pY, (*self.pY)[0]], color = 255, thick = 3
                    else: pass # TODO: oPlot, [*self.pX, (*self.pX)[0]], [*self.pY, (*self.pY)[0]], color = (255 - (self.iterations - j) * 30) > 100
            
                if f_compute_convergence:
                    delta_mag = np.sqrt(np.square(x_delta) + np.square(y_delta))

                    if var_metric == 'Hausdorff': variation = None # TODO: s_HausdorffDistanceFor2Dpoints(*self.pX, *self.pY, lastIterX, lastIterY)
                    elif var_metric == 'L1norm'   : variation = None # TODO: calcNorm_L1ForVector(deltaMag)
                    elif var_metric == 'L2norm'   : variation = None # TODO: calcNorm_L2ForVector(deltaMag)
                    elif var_metric == 'LinfNorm' : variation = None # TODO: calcNorm_LInfiniteForVector(deltaMag)
                    elif var_metric == 'average'  : variation = None # TODO: mean(deltaMag)
                    elif var_metric == 'avgFracPerimeter': variation = None # TODO: mean(deltaMag) / polygonPerimeter(*self.pX, *self.pY)
                    elif var_metric == 'avgFracPerimeter0': variation = None # TODO: mean(deltaMag) / perimeterIt0
                    else: variation = None # TODO: calcNorm_LInfiniteForVector(deltaMag)

                    f_log = 0
                    f_log_all = 0

                    if f_log:
                        log_file_path = 'D:\\tmp\\snakeLog.txt'
                        msg = f"{var_metric} convergence criterion value = {variation} at iteration {j}"
                        if f_log_all:
                            hd = None # TODO: s_HausdorffDistanceFor2Dpoints(*self.pX, *self.pY, lastIterX, lastIterY)
                            l1 = None # TODO: calcNorm_L1ForVector(deltaMag)
                            l2 = None # TODO: calcNorm_L2ForVector(deltaMag)
                            li = None # TODO: calcNorm_LInfiniteForVector(deltaMag)
                            avg = None # TODO: mean(deltaMag)
                            avg_norm_perim_it = None # TODO: mean(deltaMag) / polygonPerimeter(*self.pX, *self.pY)
                            avg_norm_perim_0  = None # TODO: mean(deltaMag) / perimeterIt0
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

        return np.array([self.x, self.y])

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
