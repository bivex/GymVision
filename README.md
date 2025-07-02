# Gym Vision
[![Trained with 💪](https://a.b-b.top/badge.svg?repo=GymVision&label=Trained%20with%20💪&background_color=e53935&background_color2=ef5350&utm_source=github&utm_medium=readme&utm_campaign=badge)](https://a.b-b.top)

A simple Python application for tracking body movements using computer vision.

## Features

- Real-time pose detection using MediaPipe
- PyQt Graphical User Interface (UI)
- Camera feed processing and visualization
- FPS counter displayed in the UI
- Tracking and counting for the following exercises:
    - Scissors (Arms)
    - Head Tilts (Side-to-side)
    - Head Rotations (Side-to-side)
    - Bicep Curls (Left and Right)
    - Overhead Press
    - Triceps Extensions
    - Pushups
    - Rows

## Requirements

- Python 3.7+ (or Python 3.11 if using Conda)
- Webcam
- PyQt5

## Installation

### Using Conda (Recommended)

1.  **Create and activate Conda environment:**
    ```bash
    conda create -n GymVision-py311 python=3.11
    conda activate GymVision-py311
    ```

2.  **Clone this repository:**
    ```bash
    git clone <repository_url>
    cd GymVision 
    ```
    (Replace `<repository_url>` with the actual URL of your repository)

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Using pip

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1.  **Activate Conda environment (if not already active):**
    ```bash
    conda activate GymVision-py311
    ```

2.  **Run the tracking application:**
    ```bash
    python track.py
    ```

Press 'q' to quit the application.

## Controls

- **Q**: Quit the application 
