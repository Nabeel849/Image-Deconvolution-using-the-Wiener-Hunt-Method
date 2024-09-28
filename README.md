# Image Deconvolution using Wiener-Hunt Method

## Overview
This project demonstrates image deblurring using the Wiener-Hunt deconvolution technique. The Wiener filter restores blurred images by compensating for known blurring effects, enhancing the clarity of the original image.

## Features
- Simulates image blur using a Gaussian Point Spread Function (PSF).
- Applies the Wiener-Hunt filter in the frequency domain to recover details while minimizing noise.
- Allows tuning of parameters such as PSF size, standard deviation, and noise-to-signal ratio for optimal deblurring.
- Visual comparisons of the original, blurred, and deblurred images.

## Requirements
To run this project, you need:
- Python 3.x
- NumPy
- OpenCV
- Matplotlib
- SciPy

You can install the required libraries using pip:

```bash
pip install numpy opencv-python matplotlib scipy
