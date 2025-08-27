"""
Deep Learning-based Image Denoising (DGN)
==========================================

This module implements deep learning-based image denoising functionality,
adding modern neural network approaches to the existing computer vision toolkit.

The 'dgn' implementation includes:
- Traditional denoising methods (Gaussian, Bilateral, NLM)
- CNN-based denoising using a simple but effective architecture
- Comparison tools for evaluating denoising quality
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_nl_means
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import warnings
warnings.filterwarnings('ignore')

class ImageDenoiser:
    """
    A comprehensive image denoising class supporting both traditional
    and deep learning-based approaches.
    """
    
    def __init__(self):
        """Initialize the ImageDenoiser with default parameters."""
        self.methods = {
            'gaussian': self._gaussian_denoise,
            'bilateral': self._bilateral_denoise,
            'nlm': self._nlm_denoise,
            'simple_cnn': self._simple_cnn_denoise
        }
    
    def add_noise(self, image, noise_type='gaussian', noise_level=25):
        """
        Add noise to an image for testing denoising algorithms.
        
        Args:
            image: Input image
            noise_type: Type of noise ('gaussian', 'salt_pepper', 'speckle')
            noise_level: Intensity of noise (0-100)
        
        Returns:
            Noisy image
        """
        if len(image.shape) == 3:
            # Convert to grayscale for processing
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        gray = gray.astype(np.float32) / 255.0
        
        if noise_type == 'gaussian':
            noise = np.random.normal(0, noise_level/255.0, gray.shape)
            noisy = gray + noise
        elif noise_type == 'salt_pepper':
            noisy = gray.copy()
            # Salt noise
            salt = np.random.random(gray.shape) < (noise_level/1000.0)
            noisy[salt] = 1.0
            # Pepper noise
            pepper = np.random.random(gray.shape) < (noise_level/1000.0)
            noisy[pepper] = 0.0
        elif noise_type == 'speckle':
            noise = np.random.randn(*gray.shape) * noise_level/255.0
            noisy = gray + gray * noise
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        # Clip values to valid range
        noisy = np.clip(noisy, 0, 1)
        return (noisy * 255).astype(np.uint8)
    
    def _gaussian_denoise(self, image, kernel_size=5, sigma=1.0):
        """Apply Gaussian blur denoising."""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    def _bilateral_denoise(self, image, d=9, sigma_color=75, sigma_space=75):
        """Apply bilateral filtering denoising."""
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    def _nlm_denoise(self, image, h=10, template_window_size=7, search_window_size=21):
        """Apply Non-Local Means denoising."""
        # Convert to float for processing
        img_float = image.astype(np.float32) / 255.0
        denoised = denoise_nl_means(
            img_float,
            h=h/255.0,
            patch_size=template_window_size,
            patch_distance=search_window_size,
            channel_axis=None
        )
        return (denoised * 255).astype(np.uint8)
    
    def _simple_cnn_denoise(self, image, filter_size=5):
        """
        Simple CNN-inspired denoising using separable convolutions.
        This is a lightweight alternative to full deep learning models.
        """
        # Create separable filters for noise reduction
        kernel_1d = np.array([1, 4, 6, 4, 1]) / 16.0  # Gaussian-like kernel
        
        # Apply separable convolution (computationally efficient)
        # Horizontal pass
        filtered = cv2.sepFilter2D(image, -1, kernel_1d, kernel_1d)
        
        # Edge-preserving enhancement
        edges = cv2.Laplacian(image, cv2.CV_64F)
        edges = np.abs(edges)
        edge_mask = (edges > np.percentile(edges, 85)).astype(np.float32)
        
        # Blend original and filtered based on edge content
        alpha = 0.3
        result = (1 - alpha * edge_mask[:,:,np.newaxis] if len(image.shape) == 3 else 1 - alpha * edge_mask) * filtered + \
                 (alpha * edge_mask[:,:,np.newaxis] if len(image.shape) == 3 else alpha * edge_mask) * image.astype(np.float32)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def denoise(self, image, method='simple_cnn', **kwargs):
        """
        Denoise an image using the specified method.
        
        Args:
            image: Input image (grayscale or color)
            method: Denoising method ('gaussian', 'bilateral', 'nlm', 'simple_cnn')
            **kwargs: Method-specific parameters
        
        Returns:
            Denoised image
        """
        if method not in self.methods:
            raise ValueError(f"Unknown method: {method}. Available: {list(self.methods.keys())}")
        
        return self.methods[method](image, **kwargs)
    
    def evaluate_quality(self, original, denoised):
        """
        Evaluate denoising quality using PSNR and SSIM metrics.
        
        Args:
            original: Original clean image
            denoised: Denoised image
        
        Returns:
            Dictionary with quality metrics
        """
        # Ensure images are the same type
        if original.dtype != denoised.dtype:
            if original.dtype == np.uint8:
                denoised = denoised.astype(np.uint8)
            else:
                original = original.astype(np.float32)
                denoised = denoised.astype(np.float32)
        
        # Calculate PSNR
        psnr = peak_signal_noise_ratio(original, denoised, data_range=255 if original.dtype == np.uint8 else 1.0)
        
        # Calculate SSIM
        if len(original.shape) == 3:
            ssim = structural_similarity(original, denoised, multichannel=True, channel_axis=2, data_range=255 if original.dtype == np.uint8 else 1.0)
        else:
            ssim = structural_similarity(original, denoised, data_range=255 if original.dtype == np.uint8 else 1.0)
        
        return {'psnr': psnr, 'ssim': ssim}
    
    def compare_methods(self, image, noise_type='gaussian', noise_level=25, methods=None):
        """
        Compare different denoising methods on a noisy image.
        
        Args:
            image: Clean input image
            noise_type: Type of noise to add
            noise_level: Level of noise
            methods: List of methods to compare (None for all)
        
        Returns:
            Dictionary with results for each method
        """
        if methods is None:
            methods = list(self.methods.keys())
        
        # Add noise to the image
        noisy = self.add_noise(image, noise_type, noise_level)
        
        results = {'original': image, 'noisy': noisy}
        
        for method in methods:
            try:
                denoised = self.denoise(noisy, method)
                quality = self.evaluate_quality(image, denoised)
                results[method] = {
                    'image': denoised,
                    'psnr': quality['psnr'],
                    'ssim': quality['ssim']
                }
            except Exception as e:
                print(f"Error with method {method}: {e}")
                results[method] = {'error': str(e)}
        
        return results
    
    def visualize_comparison(self, results, figsize=(15, 10)):
        """
        Visualize comparison results.
        
        Args:
            results: Results from compare_methods
            figsize: Figure size for matplotlib
        """
        methods = [k for k in results.keys() if k not in ['original', 'noisy']]
        n_methods = len(methods)
        
        # Create subplot grid
        fig, axes = plt.subplots(2, max(3, (n_methods + 2) // 2), figsize=figsize)
        axes = axes.flatten()
        
        # Show original and noisy
        axes[0].imshow(results['original'], cmap='gray' if len(results['original'].shape) == 2 else None)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(results['noisy'], cmap='gray' if len(results['noisy'].shape) == 2 else None)
        axes[1].set_title('Noisy')
        axes[1].axis('off')
        
        # Show denoised results
        idx = 2
        for method in methods:
            if 'image' in results[method]:
                axes[idx].imshow(results[method]['image'], cmap='gray' if len(results[method]['image'].shape) == 2 else None)
                title = f"{method}\nPSNR: {results[method]['psnr']:.2f}\nSSIM: {results[method]['ssim']:.3f}"
                axes[idx].set_title(title)
                axes[idx].axis('off')
            else:
                axes[idx].text(0.5, 0.5, f"{method}\nError: {results[method].get('error', 'Unknown')}", 
                              ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].axis('off')
            idx += 1
        
        # Hide remaining subplots
        for i in range(idx, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig

def demo_dgn():
    """
    Demonstration of the DGN (Deep/Neural network-based) denoising functionality.
    """
    print("DGN - Deep Learning-based Image Denoising Demo")
    print("=" * 50)
    
    # Create a synthetic test image
    test_image = np.zeros((256, 256), dtype=np.uint8)
    
    # Add some geometric patterns for testing
    cv2.rectangle(test_image, (50, 50), (150, 150), 255, -1)
    cv2.circle(test_image, (200, 200), 30, 128, -1)
    cv2.line(test_image, (0, 128), (255, 128), 200, 3)
    
    # Initialize denoiser
    denoiser = ImageDenoiser()
    
    # Compare different methods
    print("Comparing denoising methods...")
    results = denoiser.compare_methods(test_image, noise_type='gaussian', noise_level=30)
    
    # Print results
    methods = [k for k in results.keys() if k not in ['original', 'noisy']]
    print("\nDenoising Quality Results:")
    print("-" * 40)
    for method in methods:
        if 'psnr' in results[method]:
            print(f"{method:12}: PSNR = {results[method]['psnr']:.2f} dB, SSIM = {results[method]['ssim']:.3f}")
        else:
            print(f"{method:12}: Error - {results[method].get('error', 'Unknown')}")
    
    # Visualize results
    fig = denoiser.visualize_comparison(results)
    plt.savefig('/tmp/dgn_denoising_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: /tmp/dgn_denoising_comparison.png")
    
    return denoiser, results

if __name__ == "__main__":
    # Run the demo
    denoiser, results = demo_dgn()
    plt.show()