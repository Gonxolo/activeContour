# Snakes - Active Contour

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

1. Access the GitHub repository at https://github.com/Gonxolo/activeContour
2. Download the files from the repo

<p align="center">
  <img src="https://i.imgur.com/iKm3dF5_d.webp?maxwidth=760&fidelity=grand" />
</p>

3. Once the download is complete unzip the files and access the `activeContours` folder
4. Inside the folder click on the "File" ("Archivo") button on the top bar

<p align="center">
  <img src="https://i.imgur.com/ntEocLF.png" width="760" />
</p>

5. Then click on "Open Windows PowerShell" ("Abrir Windows PowerShell")

<p align="center">
  <img src="https://i.imgur.com/FIKM3Op.png" width="760" />
</p>

6. Inside PowerShell write the following command and then press Enter

```
pip install -r requirements.txt
```

This will install every necessary package for the execution of the algorithm.

7. Once the previous step is complete write the following command and press Enter:

```
python demo.py
```

`demo.py` is a file made for demonstration purposes, it executes the algorithm on an image (`demo_img.tif`) with 10 ROIs included on `demofiles` folder, the program
prints the number of the ROI it is currently working on, and once it has processed every ROI the following image should be displayed on screen.

<p align="center">
  <img src="https://i.imgur.com/WAXtYlt.png" width="760" />
</p>
