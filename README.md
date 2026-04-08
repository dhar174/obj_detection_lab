<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# AI Object Detection Demo

This project is a browser-based webcam demo built with React, TypeScript, and TensorFlow.js.

View the app in AI Studio: https://ai.studio/apps/1917f059-bbb1-4676-ba06-3af2eb8292bc

## What the demo supports

- Live webcam inference in the browser
- SSD MobileNet object detection models trained on COCO
- YOLOv8 Nano object detection
- BlazeFace face detection
- MoveNet Lightning and Thunder single-person pose estimation
- Adjustable confidence threshold and model selection in the UI

## Current limitations

- No API key or backend service is required
- The demo only works with a webcam feed; it does not support image/video upload
- MoveNet support is single-person pose estimation only
- Models are loaded from public CDNs, so internet access is required at runtime

## Run locally

**Prerequisites:** Node.js and a browser with camera access

1. Install dependencies:
   `npm install`
2. Start the development server:
   `npm run dev`
3. Open the local URL shown by Vite and allow camera access when prompted

## Model notes for classroom demos

- **MoveNet Lightning / Thunder:** person boxes are now derived from higher-confidence keypoints with a small amount of padding, which keeps the person framing steadier when some joints are uncertain.
- **YOLOv8 Nano:** the demo uses a slightly smaller inference size and a capped refresh rate so lower-end classroom hardware stays more responsive during live comparisons.
- **SSD MobileNet / BlazeFace:** these remain the simplest baseline options when you want faster turnaround with fewer model-specific tradeoffs to explain.
## Classroom demo tips

- Use the **Vision Model Architecture** selector to compare broad tradeoffs:
  - **SSD MobileNet** models are a good baseline for discussing speed versus accuracy.
  - **YOLOv8 Nano** usually finds more objects, but it is a heavier model to load.
  - **MoveNet** and **BlazeFace** are specialized models for people and faces.
- The **Confidence Threshold** slider controls how certain the model must be before a box is shown. Lower values show more guesses; higher values keep only stronger matches.
- If webcam access is unavailable during a lesson, click **Use Classroom Demo** to switch to a prepared example scene that still responds to the selected model and threshold.
