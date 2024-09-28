import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.fft import fft2, ifft2


def read_image(image_path, grayscale=True):
    """
    Reads the input image and converts it to grayscale if needed.
    """
    image = cv2.imread(image_path)
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def gaussian_psf(size, sigma):
    """
    Generates a Gaussian Point Spread Function (PSF).
    
    Parameters:
    - size: Size of the PSF (e.g., 15x15).
    - sigma: Standard deviation of the Gaussian function.

    Returns:
    - psf: The generated PSF.
    """
    psf = np.zeros(size)
    center = (size[0] // 2, size[1] // 2)
    
    for i in range(size[0]):
        for j in range(size[1]):
            x, y = i - center[0], j - center[1]
            psf[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    return psf / psf.sum()  # Normalize the PSF

def wiener_filter(blurred, psf, K):
    """
    Applies Wiener-Hunt deconvolution to an input blurred image.

    Parameters:
    - blurred: The observed blurred image.
    - psf: Point Spread Function (PSF) representing the blur.
    - K: Noise-to-signal power ratio.

    Returns:
    - deconvolved_image: The deblurred image after Wiener filtering.
    """
    # Perform Fourier transforms on the blurred image and PSF
    G = fft2(blurred)
    H = fft2(psf, s=blurred.shape)
    H_conj = np.conj(H)  # Conjugate of H
    
    # Wiener filter formula in frequency domain
    F_hat = (H_conj / (H * H_conj + K)) * G
    
    # Inverse Fourier transform to get the deblurred image
    deconvolved_image = np.abs(ifft2(F_hat))
    return deconvolved_image

def plot_all_images(original, blurred, deblurred):
    """
    Plots the original, blurred, and deblurred images side by side.
    """
    plt.figure(figsize=(18, 6))

    # Original Image
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original, cmap='gray')
    plt.axis('off')

    # Blurred Image
    plt.subplot(1, 3, 2)
    plt.title("Blurred Image")
    plt.imshow(blurred, cmap='gray')
    plt.axis('off')

    # Deblurred Image
    plt.subplot(1, 3, 3)
    plt.title("Deblurred Image using Wiener-Hunt")
    plt.imshow(deblurred, cmap='gray')
    plt.axis('off')

    # Show all images
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load the example image
    image_path = 'D:/Work/IPCV/ImageDeconvolution/images/pexels-suzyhazelwood-1995842.jpg'  # Replace with your image path
    original_image = read_image(image_path)

    # Create a Gaussian PSF
    psf_size = (50, 50)   # Size of the Gaussian kernel
    sigma = 20  # Reduced standard deviation for the Gaussian blur to avoid too much blurring
    psf = gaussian_psf(psf_size, sigma)

    # Blur the image using convolution with the PSF
    blurred_image = convolve2d(original_image, psf, mode='same')

    # Apply Wiener Deconvolution
    noise_to_signal_ratio = 0.01  # Lower noise-to-signal ratio for better accuracy
    deblurred_image = wiener_filter(blurred_image, psf, noise_to_signal_ratio)

    # Display all images side by side
    plot_all_images(original_image, blurred_image, deblurred_image)
