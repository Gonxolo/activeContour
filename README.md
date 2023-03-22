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
