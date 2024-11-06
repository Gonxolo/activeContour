# Snakes - Active Contours

## About this Repository
We present an implementation of a Parametric Active Contour Model to adjust object
contours upon single-channel (gray scale) images. When shape measurements are 
required, object contours give direct access to perimeter, curvature, and related features.
Moreover, optimized object contours can improve pixel-based estimations. To this end, 
object contours are deformed from an initial, rough estimation, towards an optimized 
shape and placement within the image, defined by the intensity (color) gradients, and shape 
constraints defined upon the contours related to physical/geometrical properties like smoothness, 
contractility, u others.
The contours are represented as polygons (approximations of closed curves), which can be
made finer or coarser than the image pixel grid. To this end, cubic spline interpolation and 
linear interpolation methods are provided. In adition to closed contours, the adjustment 
algorithm can work with polylines (non-closed curves) as well.

The specific model implemented here is known as "GGVF snake". It combines the parametric 
contour model from Kass et al. (1988) with the Generalized Gradient Vector Flow from Xu & Prince (1999).
Contour interpolation and numerical algorithms have been developed at SCIAN-Lab (Fanani et al., 2010; Jara-Wilde et al., 2020).

##
References
Kass, Witkin, Terzopoulos. "Snakes: Active Contour Models". 1999. International Journal of Computer Vision, 1(4):321-331. 1988.
https://doi.org/10.1007/BF00133570

Xu & Prince. Generalized Gradient Vector Flow External Forces for Active Contours. Signal Processing, 71(2):131-139. 1998.
https://doi.org/10.1016/S0165-1684(98)00140-6

Fanani et al. "The action of sphingomyelinase in lipid monolayers as revealed by microscopic image analysis".
Biochimica et Bipphysica Acta (BBA) - Biomembranes, 1897(7): 1309-1323. 2010.
https://doi.org/10.1016/j.bbamem.2010.01.001

Jara-Wilde et al. "Optimising adjacent membrane segmentation and parameterisation in multicellular aggregates by piecewise active contours".
Journal of Microscopy, 278(2): 59-75. 2020.
https://doi.org/10.1111/jmi.12887

## Set Up Guide

### Installing Python

1. Download the latest version of `Python` from https://www.python.org/downloads/
2. Execute the installer
3. Make sure the following options are selected and then click on Install Now

<p align="center">
  <img src="https://i.imgur.com/behgC9X.png" />
</p>

4. Once the setup is complete before closing the installer make sure to click the following option

<p align="center">
  <img src="https://i.imgur.com/MhCM7RI.png" />
</p>

### Downlading the files

1. Access the GitHub repository, currently at https://github.com/Gonxolo/activeContour
2. Download the files from the repository. To do this, click on the "Code" button and then on "Download ZIP"...

<p align="center">
  <img src="https://i.imgur.com/iKm3dF5_d.webp?maxwidth=760&fidelity=grand" />
</p>

3. Once the download is complete unzip the files and go into the `activeContours` folder (you may need a decompression software for ZIP files).
4. Inside the folder click the adress bar...

<p align="center">
  <img src="https://i.imgur.com/EC4BCOu.png" width="760" />
</p>

5. Write "cmd" on the adress and press Enter...

<p align="center">
  <img src="https://i.imgur.com/6YDv5Vh.png" width="760" />
</p>

6. Inside PowerShell write the following command and then press Enter

```
pip install -r requirements.txt
```

This will install every necessary package for the execution of the program.

7. Once the previous step is complete write the following command and press Enter:

```
python demo.py
```

`demo.py` is a file made for demonstration purposes: it executes the program on a sample image (`demo_img.tif`) with 10 ROIs included within the `demofiles` folder, the program, prints the number of the ROI it is currently processing, and once it has processed every ROI an output image should be displayed on screen...

<p align="center">
  <img src="https://i.imgur.com/WAXtYlt.png" width="760" />
</p>
