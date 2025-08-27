# Computer Vision Repository - DGN Enhancement

This repository contains various computer vision techniques implemented in Jupyter notebooks, now enhanced with **DGN** (Deep Learning-based) image denoising functionality.

## New Addition: DGN - Deep Learning-based Image Denoising

The `dgn_denoising.py` module and `dgn_denoising.ipynb` notebook implement modern denoising techniques that complement the existing traditional computer vision methods.

### Features:
- **Multiple denoising algorithms**: Gaussian, Bilateral, Non-Local Means, and Simple CNN
- **Noise simulation**: Add Gaussian, salt-and-pepper, or speckle noise for testing
- **Quality metrics**: PSNR and SSIM evaluation
- **Visualization tools**: Compare methods side-by-side

### Quick Start:

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the demo:
```bash
python dgn_denoising.py
```

3. Or explore the Jupyter notebook:
```bash
jupyter notebook dgn_denoising.ipynb
```

### Usage Example:
```python
from dgn_denoising import ImageDenoiser
import cv2

# Initialize denoiser
denoiser = ImageDenoiser()

# Load and process an image
image = cv2.imread('your_image.jpg', cv2.IMREAD_GRAYSCALE)
noisy_image = denoiser.add_noise(image, 'gaussian', 25)
denoised = denoiser.denoise(noisy_image, 'simple_cnn')

# Evaluate quality
quality = denoiser.evaluate_quality(image, denoised)
print(f"PSNR: {quality['psnr']:.2f} dB, SSIM: {quality['ssim']:.3f}")
```

## Repository Structure

- **Traditional CV Notebooks**: Harris corners, region growing, background subtraction, etc.
- **DGN Enhancement**: Modern denoising with `dgn_denoising.py` and `dgn_denoising.ipynb`
- **Dependencies**: `requirements.txt` for easy setup

The DGN implementation adds deep learning-inspired techniques while maintaining compatibility with the existing codebase.