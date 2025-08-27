# Computer Vision Repository

This repository contains a comprehensive collection of computer vision techniques and algorithms implemented in Python using OpenCV, NumPy, and other popular libraries. Each technique is demonstrated in separate Jupyter notebooks with practical examples and visualizations.

## 🚀 Capabilities

### 🔍 Feature Detection and Description
- **Harris Corner Detection** (`HArris (2).ipynb`)
  - Detect corners and key points in images
  - Enhanced corner visualization with dilated markers
  - Robust feature point identification

- **SIFT (Scale-Invariant Feature Transform)** (`sift (1).ipynb`)
  - Scale and rotation invariant feature detection
  - Keypoint description and matching
  - Feature-based object recognition

### 📐 Edge Detection and Line Detection
- **Canny Edge Detection** (`HArris (2).ipynb`, `canny_dog (3).ipynb`)
  - Multi-stage edge detection algorithm
  - Adjustable thresholds for optimal edge detection
  - Noise reduction and edge localization

- **Hough Line Transform** (`HArris (2).ipynb`)
  - Detect straight lines in images
  - Parameter space transformation
  - Line segment detection with customizable parameters

### 🎯 Object Detection and Recognition
- **Object Detection with SIFT** (`obj_detection.ipynb`)
  - Template matching using SIFT features
  - Feature-based object localization
  - Robust matching with FLANN-based matcher
  - Clustering for multiple object detection

- **Image Classification** (`Image_classify.ipynb`)
  - Horse vs Human classification using SVM
  - HOG (Histogram of Oriented Gradients) feature extraction
  - SIFT feature extraction and combination
  - PCA for dimensionality reduction
  - StandardScaler for feature normalization

### 🎬 Video Processing and Motion Analysis
- **Background Subtraction** (`background_sub.ipynb`)
  - Static background subtraction
  - Background averaging with adaptive learning
  - Foreground object detection
  - Real-time video processing

- **Lucas-Kanade Optical Flow** (`Lucas_kanade_(optical flow).ipynb`)
  - Track feature points across video frames
  - Motion vector visualization
  - Real-time motion tracking
  - Feature trajectory analysis

- **Scene Change Detection** (`background_sub.ipynb`)
  - SSIM (Structural Similarity Index) based detection
  - Pixel-wise change detection
  - Threshold-based scene change alerts

### 🖼️ Image Segmentation
- **Region Growing** (`region_growing.ipynb`)
  - Seed-based region segmentation
  - Similarity criteria for region expansion
  - Connected component analysis

- **Contour-based Segmentation** (`HArris (2).ipynb`, `segmentation (1).ipynb`)
  - Edge-based segmentation using contours
  - Object boundary detection
  - Shape analysis and filtering

### 🎨 Image Processing and Enhancement
- **Basic Image Processing** (`basic_image_processing.ipynb`)
  - Image loading and display
  - Color space conversions
  - Basic image manipulations
  - Lenna image processing examples

- **Histogram Processing** (`Histogram_processing.ipynb`)
  - Histogram computation and visualization
  - Histogram equalization
  - Contrast enhancement
  - Statistical image analysis

- **Color Model Conversions** (`color_models (1).ipynb`)
  - RGB to various color spaces (HSV, LAB, etc.)
  - Color channel analysis
  - Color-based image processing

- **Noise Removal and Filtering** (`noise_removal_and_edge_det.ipynb`)
  - Gaussian filtering
  - Median filtering
  - Bilateral filtering
  - Noise reduction techniques

- **Image Transformations** (`Image_trans.ipynb`)
  - Geometric transformations
  - Affine transformations
  - Perspective corrections
  - Image warping

## 🛠️ Technologies Used

- **Python 3.12.7**
- **OpenCV (cv2)** - Primary computer vision library
- **NumPy** - Numerical computations and array operations
- **Matplotlib** - Visualization and plotting
- **scikit-image** - Additional image processing tools
- **scikit-learn** - Machine learning algorithms (SVM, PCA)
- **Seaborn** - Statistical data visualization

## 📋 Requirements

```bash
pip install opencv-python
pip install numpy
pip install matplotlib
pip install scikit-image
pip install scikit-learn
pip install seaborn
```

## 🚀 Getting Started

1. Clone this repository
2. Install the required dependencies
3. Open any Jupyter notebook to explore specific computer vision techniques
4. Replace image/video paths in the notebooks with your own data
5. Run the cells to see the algorithms in action

## 📁 Notebook Descriptions

| Notebook | Description | Key Techniques |
|----------|-------------|----------------|
| `HArris (2).ipynb` | Comprehensive feature detection demo | Harris corners, Hough lines, Canny edges, segmentation |
| `Image_classify.ipynb` | Machine learning-based image classification | SVM, HOG, SIFT, PCA |
| `background_sub.ipynb` | Video background subtraction and scene detection | Background modeling, SSIM, foreground detection |
| `obj_detection.ipynb` | Feature-based object detection | SIFT matching, clustering, template matching |
| `sift (1).ipynb` | SIFT feature detection and description | Scale-invariant features, keypoint matching |
| `Lucas_kanade_(optical flow).ipynb` | Motion tracking in videos | Optical flow, feature tracking |
| `region_growing.ipynb` | Image segmentation using region growing | Seed-based segmentation, region merging |
| `segmentation (1).ipynb` | Various segmentation techniques | Multiple segmentation approaches |
| `basic_image_processing.ipynb` | Fundamental image operations | Loading, display, basic manipulations |
| `Histogram_processing.ipynb` | Histogram analysis and enhancement | Histogram equalization, statistical analysis |
| `noise_removal_and_edge_det.ipynb` | Image filtering and edge detection | Filtering, denoising, edge detection |
| `color_models (1).ipynb` | Color space analysis | Color conversions, channel operations |
| `canny_dog (3).ipynb` | Advanced edge detection | Canny algorithm variations |
| `Image_trans.ipynb` | Geometric image transformations | Affine, perspective transformations |
| `scene_detection (1).ipynb` | Scene analysis and detection | Scene understanding techniques |

## 💡 Usage Examples

### Basic Image Loading and Display
```python
import cv2
import matplotlib.pyplot as plt

# Load image
image = cv2.imread('your_image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display
plt.imshow(gray, cmap='gray')
plt.show()
```

### Harris Corner Detection
```python
# Detect Harris corners
dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
dst = cv2.dilate(dst, None)

# Mark corners
image[dst > 0.01 * dst.max()] = [0, 0, 255]
```

### SIFT Feature Detection
```python
# Initialize SIFT
sift = cv2.SIFT_create()

# Detect keypoints and descriptors
keypoints, descriptors = sift.detectAndCompute(gray, None)

# Draw keypoints
img_with_keypoints = cv2.drawKeypoints(image, keypoints, None)
```

## 🎯 Applications

This repository demonstrates techniques useful for:
- **Security Systems**: Motion detection, object recognition
- **Medical Imaging**: Image segmentation, feature analysis
- **Autonomous Vehicles**: Object detection, scene understanding
- **Quality Control**: Defect detection, pattern matching
- **Robotics**: Visual navigation, object manipulation
- **Augmented Reality**: Feature tracking, pose estimation

## 📚 Learning Path

1. **Start with Basic Processing** - Understand image loading, display, and basic operations
2. **Learn Feature Detection** - Explore Harris corners and SIFT features
3. **Study Edge Detection** - Master Canny and other edge detection techniques
4. **Explore Segmentation** - Try different segmentation approaches
5. **Advance to Object Detection** - Implement feature-based object recognition
6. **Work with Video** - Learn motion tracking and background subtraction
7. **Apply Machine Learning** - Use SVM and other ML techniques for classification

## 🤝 Contributing

Feel free to contribute by:
- Adding new computer vision techniques
- Improving existing implementations
- Adding more comprehensive examples
- Enhancing documentation

## 📄 License

This project is open source and available under the MIT License.

---

*This repository serves as a comprehensive learning resource and practical toolkit for computer vision applications. Each notebook is self-contained and includes detailed explanations and visualizations to help you understand the underlying concepts.*