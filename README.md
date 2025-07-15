# Player Re-Identification System

This system implements a comprehensive player tracking and re-identification solution for sports video analysis using YOLOv11 and DeepSORT. The system maintains consistent track IDs for players and other objects (including balls) even when they temporarily disappear from view due to occlusion or leaving the frame.

## Features

- **Advanced Object Detection**: Custom-trained YOLOv11 model for multi-class detection
- **Robust Multi-Object Tracking**: DeepSORT with MobileNet embeddings for appearance-based tracking
- **Player Re-identification**: Maintains consistent IDs after occlusion or frame exit
- **Multi-class Support**: Handles multiple object classes with color-coded visualization
- **Smart Ball Detection**: Special handling for ball objects with adaptive bounding box sizing
- **Real-time Processing**: Live video display with tracking visualization
- **Video Export**: Saves processed video with tracking annotations

## Project Structure

```text
Player Re-Identification System/
├── data/
│   └── input.mp4              # Input video file
├── player_tracking.py         # Main tracking script
├── yolov11_players.pt        # Custom trained YOLOv11 model
├── requirements.txt          # Python dependencies
├── output_tracked.mp4        # Generated output video
└── README.md                 # This file
```

**📁 Data & Output Files**: [Google Drive Folder](https://drive.google.com/drive/folders/1rITg5YWUcGhQfGIXqfFjL9AzCPUHcUbz?usp=sharing)

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for optimal performance)
- OpenCV-compatible video codecs

## Installation

1. Clone or download this repository
2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

1. Download the sample data from the [Google Drive folder](https://drive.google.com/drive/folders/1rITg5YWUcGhQfGIXqfFjL9AzCPUHcUbz?usp=sharing)
2. Ensure your input video is placed in the `data/` directory as `input.mp4`
3. Verify the YOLOv11 model file (`yolov11_players.pt`) is in the project root
4. Run the tracking script:

   ```bash
   python player_tracking.py
   ```

### Custom Input

To use a different input video, modify the `input_path` variable in `player_tracking.py`:

```python
input_path = 'path/to/your/video.mp4'
```

## Output

The system generates:

- **Processed Video**: `output_tracked.mp4` with bounding boxes, track IDs, and class labels
- **Real-time Display**: Live tracking visualization during processing
- **Console Output**: Detection statistics and class information

**📁 Sample Output**: You can view sample output videos in the [Google Drive folder](https://drive.google.com/drive/folders/1rITg5YWUcGhQfGIXqfFjL9AzCPUHcUbz?usp=sharing)

## Model Classes & Color Coding

The system supports multiple object classes with distinct color coding:

- **Class 0** (Ball): Yellow bounding boxes `(0, 255, 255)` - with adaptive box sizing (40% smaller)
- **Class 1** (Goalkeeper): Blue bounding boxes `(255, 0, 0)`
- **Class 2** (Player): Green bounding boxes `(0, 255, 0)`
- **Class 3** (Referee): Red bounding boxes `(0, 0, 255)`
- **Unknown Classes**: White bounding boxes `(255, 255, 255)`

## Key Parameters

### DeepSORT Configuration

```python
tracker = DeepSort(
    max_age=60,           # Frames to keep track alive without detection
    n_init=3,             # Consecutive detections before track confirmation
    nms_max_overlap=1.0,  # Non-maximum suppression threshold
    embedder="mobilenet", # Feature extraction model
    half=True            # Half-precision for faster inference
)
```

### Detection Thresholds

- **Confidence Threshold**: 0.5 (adjustable in code)
- **Minimum Box Size**: 0.01% of frame area (prevents tiny false detections)

### Special Features

- **Ball Detection Enhancement**: 40% bounding box reduction for better ball tracking
- **Boundary Clipping**: Ensures all bounding boxes stay within frame boundaries
- **Dynamic Class Mapping**: Automatically maps model class IDs to human-readable names

## Technical Details

### Dependencies

- **OpenCV**: Video processing and visualization
- **Ultralytics**: YOLOv11 model inference
- **Deep-Sort-Realtime**: Multi-object tracking with re-identification

### Performance Considerations

- GPU acceleration recommended for real-time processing
- Video codec affects output quality and file size
- Frame rate preservation from input to output video

## Controls

- **'q' key**: Quit real-time display and processing
- **ESC**: Alternative quit method
- **Space**: Pause/resume (if implemented)

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure `yolov11_players.pt` is in the project root
2. **Input video not found**: Check that `data/input.mp4` exists
3. **No detections**: Verify model compatibility with your video content
4. **Poor tracking**: Adjust confidence threshold or DeepSORT parameters

### Performance Optimization

- Use GPU acceleration if available
- Reduce video resolution for faster processing
- Adjust `max_age` parameter based on your use case

## Future Enhancements

- [ ] Multi-camera support
- [ ] Team classification
- [ ] Player statistics collection
- [ ] Real-time streaming support
- [ ] Web interface for parameter tuning