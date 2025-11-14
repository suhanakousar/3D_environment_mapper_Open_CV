# 3D Intelligent Environment Mapper & Digital Twin System

A professional real-time 3D mapping application that creates point clouds from webcam input using AI-based depth estimation and object detection. Build live 3D digital twins of your environment with automatic object detection, tracking, and visualization.

## 🌟 Overview

This system transforms a standard webcam into a powerful 3D mapping tool that:
- **Estimates depth** using MiDaS AI models (monocular depth estimation)
- **Detects objects** using YOLOv8 (80+ object classes)
- **Builds 3D maps** by accumulating point clouds over time
- **Tracks objects** in 3D space across frames
- **Visualizes** everything in real-time with professional overlays

## ✨ Key Features

### Core Capabilities
- ✅ **Real-time depth estimation** - MiDaS models (DPT_Large, DPT_Hybrid, MiDaS_small)
- ✅ **3D point cloud generation** - Convert RGB-D to 3D point clouds
- ✅ **Object detection** - YOLOv8 with 80+ COCO classes
- ✅ **Point cloud accumulation** - Build global maps over time
- ✅ **Interactive 3D visualization** - Open3D with smooth rendering
- ✅ **Export functionality** - Save point clouds (PLY format) and object lists

### Advanced Features (Digital Twin System)
- ✅ **3D object localization** - Project 2D detections to 3D space
- ✅ **Object tracking** - Track objects across frames in 3D
- ✅ **Professional overlays** - Real-time FPS, statistics, object lists
- ✅ **Enhanced visualization** - 3D markers, color-coded objects
- ✅ **Robust error handling** - Graceful degradation and recovery
- ✅ **Performance monitoring** - FPS tracking and optimization

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- Webcam
- Windows/Linux/macOS

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd mapper
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install PyTorch (choose based on your system):**
   
   **For CPU only:**
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```
   
   **For GPU (CUDA 11.8):**
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
   
   **For Windows (CPU or GPU):**
   ```bash
   pip install torch==2.3.1 torchvision==0.18.1
   ```

5. **Download YOLOv8 model (automatic on first run, or manually):**
   - The model `yolov8n.pt` will be downloaded automatically
   - Or download from: https://github.com/ultralytics/assets/releases

## 📖 Usage

### Basic 3D Mapping (Single Frame)

Capture and visualize single frames:

```bash
# Basic mode
python src/main.py

# With object detection
python src/main.py --detect

# Custom camera
python src/main.py --cam 1 --detect
```

**Controls:**
- `'c'` or `SPACE` - Capture image and create 3D structure
- `'s'` - Save current 3D structure to file
- `'q'` or `ESC` - Quit

### Enhanced Mapping (Accumulation)

Build maps over time with accumulation:

```bash
python src/main_enhanced.py --detect --accumulate --multi_view
```

**Options:**
- `--detect` - Enable object detection
- `--accumulate` - Accumulate point clouds over time
- `--multi_view` - Show RGB + Depth side-by-side
- `--voxel_size 0.03` - Voxel size for downsampling (meters)
- `--max_points 1000000` - Max points in accumulated map
- `--auto_save` - Auto-save map on exit

**Controls:**
- `'s'` - Save current frame point cloud
- `'m'` - Save accumulated map
- `'c'` - Clear accumulated map
- `SPACE` - Pause/resume
- `'q'` or `ESC` - Quit

### Digital Twin System (Recommended) ⭐

**Professional real-time 3D mapping with object tracking:**

```bash
# Enhanced Digital Twin (Best balance of features and performance)
python src/digital_twin_enhanced.py
```

**With custom settings:**
```bash
python src/digital_twin_enhanced.py \
    --depth_model DPT_Hybrid \      # Depth model: DPT_Large, DPT_Hybrid, MiDaS_small
    --voxel_size 0.02 \             # Voxel size (meters) - smaller = more detail
    --max_points 2000000 \          # Maximum points in map
    --track_distance 0.5 \          # Max distance to match objects (meters)
    --track_age 30 \               # Max frames before removing untracked object
    --map_update_freq 5 \           # Update map every N frames
    --auto_save                     # Auto-save map on exit
```

**Controls:**
- `'s'` - Save current map and objects (formatted export)
- `'c'` - Clear map and start over
- `'i'` - Print detailed object information
- `'q'` or `ESC` - Quit gracefully

**What you'll see:**
1. **Live View Window** - RGB feed with detections and professional status overlay
2. **Depth Map Window** - Colorized depth visualization with range indicators
3. **3D Map Window** - Interactive 3D point cloud with object markers


## 📁 Project Structure

```
mapper/
├── src/
│   ├── main.py                    # Basic single-frame capture mode
│   ├── main_enhanced.py          # Enhanced with accumulation
│   ├── digital_twin.py           # Basic digital twin system
│   ├── digital_twin_enhanced.py  # ⭐ Professional enhanced version (RECOMMENDED)
│   ├── capture.py                 # Camera capture module
│   ├── pointcloud.py              # Point cloud generation
│   ├── mapping.py                 # Point cloud accumulation
│   ├── utils.py                   # Utility functions
│   ├── visualization.py           # Visualization utilities
│   ├── depth/
│   │   └── midas.py               # MiDaS depth estimation
│   └── detection/
│       └── yolo.py                # YOLOv8 object detection
├── config/
│   └── camera_intrinsics.yaml    # Camera calibration parameters
├── requirements.txt              # Python dependencies
├── yolov8n.pt                    # YOLOv8 model weights (auto-downloaded)
└── README.md                     # This file
```

## ⚙️ Configuration

### Camera Intrinsics

Edit `config/camera_intrinsics.yaml` for accurate metric reconstruction:

```yaml
image_width: 640
image_height: 480
fx: 600.0      # Focal length X
fy: 600.0      # Focal length Y
cx: 320.0      # Principal point X
cy: 240.0      # Principal point Y
distortion: [0, 0, 0, 0, 0]
```

**Note:** Default values are approximate. For accurate results, run camera calibration and update these values.

### Command-Line Options

**Common options:**
- `--cam <id>` - Camera ID (default: 0)
- `--intrinsics <path>` - Camera intrinsics file path
- `--yolo_model <path>` - YOLO model file (default: yolov8n.pt)
- `--depth_model <type>` - MiDaS model: DPT_Large, DPT_Hybrid, MiDaS_small
- `--voxel_size <float>` - Voxel size in meters (default: 0.02)
- `--max_points <int>` - Maximum points in map (default: 2000000)
- `--map_update_freq <int>` - Update map every N frames (default: 5)
- `--auto_save` - Auto-save map on exit

## 🎯 Detected Object Classes

The system can detect **80+ object classes** from the COCO dataset, including:

**People & Animals:** person, cat, dog, horse, bird, etc.

**Furniture:** chair, couch, bed, dining table, toilet, bench

**Electronics:** laptop, mouse, keyboard, tv, cell phone, remote, clock

**Household Items:** bottle, cup, bowl, book, vase, scissors

**Vehicles:** car, motorcycle, bus, truck, bicycle, boat

**And many more!** See COCO dataset for complete list.

## 💾 Export Formats

### Point Clouds
- **Format:** PLY (Polygon File Format)
- **Viewers:** MeshLab, CloudCompare, Blender, Open3D
- **Location:** Saved in project root directory

### Object Lists
- **Format:** Text file with detailed object information
- **Includes:** Object name, 3D position, distance, confidence, timestamps
- **Location:** Saved alongside point cloud files

## 🎮 Performance Tips

### For Faster Performance:
```bash
--depth_model MiDaS_small      # Smaller, faster model
--voxel_size 0.05              # Larger voxels = faster processing
--map_update_freq 10           # Update map less frequently
```

### For Better Quality:
```bash
--depth_model DPT_Large        # Larger, more accurate model
--voxel_size 0.01              # Smaller voxels = more detail
--map_update_freq 2            # Update map more frequently
```

### System Requirements:
- **CPU:** Modern multi-core processor recommended
- **RAM:** 4GB minimum, 8GB+ recommended
- **GPU:** Optional but recommended for faster processing
- **Camera:** Any USB webcam (640x480 or higher)

## 🔧 Troubleshooting

### Camera Not Working
- Try different camera ID: `--cam 1` or `--cam 2`
- Ensure no other application is using the camera
- Check camera permissions on your system

### Slow Performance
- Use `MiDaS_small` depth model
- Increase `voxel_size` (e.g., 0.05)
- Increase `map_update_freq` (e.g., 10)
- Install PyTorch with CUDA if you have a GPU

### Empty 3D Window
- Ensure camera is pointing at objects
- Check that depth map shows colors (not all one color)
- Try moving closer to objects
- Verify camera intrinsics are correct

### Model Download Issues
- MiDaS and YOLOv8 models download automatically on first run
- Ensure internet connection is available
- Models are cached after first download

## 📊 Technical Details

### Depth Estimation
- **Model:** MiDaS (Intel ISL)
- **Method:** Monocular depth estimation (single camera)
- **Output:** Depth map normalized to 0.2-5.0 meters
- **Models:** DPT_Large (most accurate), DPT_Hybrid (balanced), MiDaS_small (fastest)

### Object Detection
- **Model:** YOLOv8 (Ultralytics)
- **Dataset:** COCO (80+ classes)
- **Confidence Threshold:** 0.35 (default)
- **Output:** Bounding boxes, class IDs, confidence scores

### 3D Reconstruction
- **Method:** RGB-D point cloud generation
- **Projection:** Pinhole camera model with intrinsics
- **Filtering:** Invalid depths (< 0.01m) removed
- **Downsampling:** Voxel-based for efficiency

### Object Tracking
- **Method:** 3D position-based matching
- **Distance Threshold:** 0.5m (configurable)
- **Age Limit:** 30 frames (configurable)
- **Smoothing:** Moving average for position updates

## 🛠️ Development

### Code Organization
- **Modular design** - Clear separation of concerns
- **Error handling** - Comprehensive try/except blocks
- **Performance monitoring** - FPS tracking and optimization
- **Professional output** - Clean, formatted displays

### Key Modules
- `capture.py` - Camera interface
- `depth/midas.py` - Depth estimation
- `detection/yolo.py` - Object detection
- `pointcloud.py` - Point cloud generation
- `mapping.py` - Map accumulation
- `visualization.py` - Visualization utilities

## 📝 Notes

- **Depth Scale:** MiDaS depth is normalized to 0.2-5.0m range. For accurate metric reconstruction, calibrate your camera.
- **No SLAM:** Current implementation does not include SLAM/pose estimation. Map accumulation is frame-based.
- **Real-time:** Performance depends on hardware. GPU acceleration recommended for best results.
- **Model Weights:** Pretrained models download automatically on first run (~500MB total).

## 🚀 Future Enhancements

Potential improvements:
- SLAM integration (ORB-SLAM3)
- TSDF volume integration
- Multi-threading for better performance
- GUI application
- Semantic segmentation
- Multi-camera support

## 📄 License

See project license file for details.

## 🙏 Acknowledgments

- **MiDaS** - Intel ISL for depth estimation models
- **YOLOv8** - Ultralytics for object detection
- **Open3D** - 3D data processing library
- **OpenCV** - Computer vision library

## 📧 Support

For issues, questions, or contributions, please refer to the project repository.

---

**Version:** 2.0  
**Last Updated:** 2025  
**Status:** Production Ready

**Recommended Usage:** `python src/digital_twin_enhanced.py` for the best balance of features, performance, and professional output.
