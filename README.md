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
