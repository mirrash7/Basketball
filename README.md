# 🏀 Basketball Video Analysis

Analyze basketball footage with automated detection of players, ball, team assignment, and more. This repository integrates object tracking, zero-shot classification, and custom keypoint detection for a fully annotated basketball game experience.

Leveraging the convenience of Roboflow for dataset management and Ultralytics' YOLO models for both training and inference, this project provides a robust framework for basketball video analysis.

Training notebooks are included to help you customize and fine-tune models to suit your specific needs, ensuring a seamless and efficient workflow.

## 📁 Table of Contents

1.  [Features](#-features)
2.  [Prerequisites](#-prerequisites)
3.  [Demo Video](#-demo-video)
4.  [Installation](#-installation)
5.  [Training the Models](#-training-the-models)
6.  [Usage](#-usage)
7.  [Project Structure](#-project-structure)
8.  [Future Work](#-future-work)
9.  [Contributing](#-contributing)
10. [License](#-license)

---

## ✨ Features

- Player and ball detection/tracking using pretrained models.
- Court keypoint detection for visualizing important zones.
- Team assignment with jersey color classification.
- Ball possession detection, pass detection, and interception detection.
- Easy stubbing to skip repeated computation for fast iteration.
- Various "drawers" to overlay detected elements onto frames.

---

## 🎮 Demo Video

Below is the final annotated output video.

[![BasketBall Analysis Demo Video](https://img.youtube.com/vi/xWpP0LjEUng/0.jpg)](https://youtu.be/xWpP0LjEUng)

## 🔧 Prerequisites

- Python 3.8+
- (Optional) Docker

---

## ⚙️ Installation

Setup your environment locally or via Docker.

### Python Environment

1. Create a virtual environment (e.g., venv/conda).
2. Install the required packages:

```bash
pip install -r requirements.txt
```

### Docker

#### Build the Docker image:

```bash
docker build -t basketball-analysis .
```

#### Verify the image:

```bash
docker images
```

## 🎓 Training the Models

Harnessing the powerful tools offered by Roboflow and Ultralytics makes it straightforward to manage datasets, handle annotations, and train advanced object detection models. Roboflow provides an intuitive platform for dataset preprocessing and augmentation, while Ultralytics' YOLO architectures (v5, v8, and beyond) deliver state-of-the-art detection performance.

This repository relies on trained models for detecting basketballs, players, and court keypoints. You have two options to get these models:

1. Download the Pretrained Weights

   - ball_detector_model.pt  
     (https://drive.google.com/file/d/1KejdrcEnto2AKjdgdo1U1syr5gODp6EL/view?usp=sharing)
   - court_keypoint_detector.pt  
     (https://drive.google.com/file/d/1nGoG-pUkSg4bWAUIeQ8aN6n7O1fOkXU0/view?usp=sharing)
   - player_detector.pt  
     (https://drive.google.com/file/d/1fVBLZtPy9Yu6Tf186oS4siotkioHBLHy/view?usp=sharing)

   Simply download these files and place them into the `models/` folder in your project. This allows you to run the pipelines without manually retraining.

2. Train Your Own Models  
   The training scripts are provided in the `training_notebooks/` folder. These Jupyter notebooks use Roboflow datasets and the Ultralytics YOLO frameworks to train various detection tasks:

   - `basketball_ball_training.ipynb`: Trains a basketball ball detector (using YOLOv5). Incorporates motion blur augmentations to improve ball detection accuracy on fast-moving game footage.
   - `basketball_court_keypoint_training.ipynb`: Uses YOLOv8 to detect keypoints on the court (e.g., lines, corners, key zones).
   - `basketball_player_detection_training.ipynb`: Trains a player detection model (using YOLO v11) to identify players in each frame.

   You can easily run these notebooks in Google Colab or another environment with GPU access. After training, download the newly generated `.pt` files and place them in the `models/` folder.

## Once you have your models in place, you may proceed with the usage steps described above. If you want to retrain or fine-tune for your specific dataset, remember to adjust the paths in the notebooks and in `main.py` to point to the newly generated models.

## 🚀 Usage

You can run this repository's core functionality (analysis pipeline) with Python or Docker.

### 1) Using Python Directly

Run the main entry point with your chosen video file:

```bash
python main.py path_to_input_video.mp4 --output_video output_videos/output_result.avi
```

- By default, intermediate "stubs" (pickled detection results) are used if found, allowing you to skip repeated detection/tracking.
- Use the `--stub_path` flag to specify a custom stub folder, or disable stubs if you want to run everything fresh.

### 2) Using Docker

#### Build the container if not built already:

```bash
docker build -t basketball-analysis .
```

#### Run the container, mounting your local input video folder:

```bash
docker run \
  -v $(pwd)/videos:/app/videos \
  -v $(pwd)/output_videos:/app/output_videos \
  basketball-analysis \
  python main.py videos/input_video.mp4 --output_video output_videos/output_result.avi
```

### 3) Team Assignment Configuration

The improved team assignment system now supports several configuration options:

```bash
# Basic usage with default settings
python main.py input_video.mp4

# Choose team assigner version
python main.py input_video.mp4 --team_assigner_version current    # Improved version (default)
python main.py input_video.mp4 --team_assigner_version original   # Basic version

# Custom team assignment parameters
python main.py input_video.mp4 \
  --max_players_per_team 5 \
  --team_confidence_threshold 0.7 \
  --team_1_description "white basketball jersey" \
  --team_2_description "dark blue basketball jersey"

# Disable team assignment entirely
python main.py input_video.mp4 --disable_team_assignment
```

**Team Assigner Versions:**

**Current (Improved) Version:**
- **Maximum 5 players per team**: Enforces basketball rules by limiting team size
- **Temporal consistency**: Players cannot switch teams during play
- **Consensus-based assignment**: Uses multiple frames to make confident team assignments
- **Conflict resolution**: Automatically resolves ambiguous cases by reassigning low-confidence players
- **Configurable parameters**: Adjust confidence thresholds and team descriptions

**Original (Basic) Version:**
- **Simple assignment**: Basic jersey color classification without constraints
- **No player limits**: Doesn't enforce maximum players per team
- **No temporal consistency**: Players can potentially switch teams
- **Faster processing**: Simpler logic for quick testing
- **Basic parameters**: Only team descriptions configurable

### 4) Testing Team Assignment

Run the test script to verify team assignment functionality:

```bash
python test_team_assignment.py
```

This will test different configurations and validate that the 5-player limit is enforced.

### 5) Memory-Efficient Processing

For large videos or systems with limited memory, use the memory-efficient options:

```bash
# Basic memory-efficient processing
python main.py input_video.mp4 --memory_efficient --resize_factor 0.5

# Ultra memory-efficient processing (very small batches)
python main.py input_video.mp4 --ultra_memory_efficient --batch_size 3

# Use the helper script for easy memory-efficient processing
python run_memory_efficient.py input_video.mp4 --mode ultra
```

**Memory-Efficient Features:**
- **Frame resizing**: Reduce frame size to save memory
- **Batch processing**: Process frames in small batches
- **Garbage collection**: Automatic memory cleanup
- **Progress monitoring**: Track memory usage during processing

### 6) Improved Ball Tracking

The enhanced ball tracking system ensures the ball is always visible on screen with multiple fallback mechanisms:

```bash
# Run with enhanced ball tracking (default)
python main.py input_video.mp4

# Test ball tracking improvements
python test_improved_ball_tracking.py
```

**Ball Tracking Features:**
- **Guaranteed visibility**: Ball position is always reflected on screen
- **Smart prediction**: Estimates ball position based on previous frames when detection fails
- **Region-based search**: Searches around predicted positions with lower confidence thresholds
- **Multiple fallback levels**: Uses last known position, predicted position, or frame center as fallbacks
- **Adaptive confidence**: Automatically adjusts detection confidence based on success rate
- **Physics constraints**: Applies basketball physics to improve trajectory accuracy
- **Kalman filtering**: Smooths ball trajectory and predicts positions during occlusions

**Fallback Hierarchy:**
1. **Direct detection**: Standard ball detection with normal confidence
2. **Predicted search**: Search around predicted position with lower confidence
3. **Predicted position**: Use motion-based prediction when search fails
4. **Last valid position**: Use the last known ball position
5. **Frame center**: Ultimate fallback to center of frame

### 7) Visualization Options

Control which overlays are displayed in the output video:

```bash
# Show only player tracks and ball tracks
python main.py input_video.mp4 --show_player_tracks --show_ball_tracks

# Disable specific overlays
python main.py input_video.mp4 \
  --no_team_ball_control \
  --no_speed_distance \
  --no_court_keypoints

# Show everything except tactical view
python main.py input_video.mp4 --no_tactical_view
```

**Available Overlays:**
- `--show_player_tracks`: Player triangles with team colors and IDs
- `--show_ball_tracks`: Ball trajectory and position
- `--show_court_keypoints`: Court line detection and keypoints
- `--show_team_ball_control`: Team possession indicators
- `--show_passes_interceptions`: Pass and interception events
- `--show_speed_distance`: Player speed and distance metrics
- `--show_tactical_view`: Top-down tactical view
- `--show_frame_number`: Frame counter overlay

---

## 🏰 Project Structure

- `main.py`  
  – Orchestrates the entire pipeline: reading video frames, running detection/tracking, team assignment, drawing results, and saving the output video.

- `trackers/`  
  – Houses `PlayerTracker` and `BallTracker`, which use detection models to generate bounding boxes and track objects across frames.

- `utils/`  
  – Contains helper functions like `bbox_utils.py` for geometric calculations, `stubs_utils.py` for reading and saving intermediate results, and `video_utils.py` for reading/saving videos.

- `drawers/`  
  – Contains classes that overlay bounding boxes, court lines, passes, etc., onto frames.

- `ball_aquisition/`  
  – Logic for identifying which player is in possession of the ball.

- `pass_and_interception_detector/`  
  – Identifies passing events and interceptions.

- `court_keypoint_detector/`  
  – Detects lines and keypoints on the court using the specified model.

- `team_assigner/`  
  – Uses zero-shot classification (Hugging Face or similar) to assign players to teams based on jersey color. The improved version enforces maximum 5 players per team, maintains temporal consistency (no team switching), and uses consensus-based assignment across multiple frames.

- `configs/`  
  – Holds default paths for models, stubs, and output video.

---

## 🔮 Future Work

As we continue to enhance the capabilities of this basketball video analysis tool, several areas for future development have been identified:

1. **Integrating a Pose Model for Advanced Rule Detection**  
   Incorporating a pose detection model could enable the identification of complex basketball rules such as double dribbling and traveling. By analyzing player movements and positions, the system could automatically flag these infractions, adding another layer of analysis to the video footage.

These enhancements will further refine the analysis capabilities and provide users with more comprehensive insights into basketball games.

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a clear explanation of your changes.

---

## 🐜 License

This project is licensed under the MIT License.  
See `LICENSE` for details.

---

## 💬 Questions or Feedback?

Feel free to open an issue or reach out via email if you have questions about the project, suggestions for improvements, or just want to say hi!

Enjoy analyzing basketball footage with automatic detection and tracking!
