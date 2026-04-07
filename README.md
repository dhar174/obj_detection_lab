<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# Webcam object detection demo

This Vite app runs the browser-based object detection demo locally with package-managed React, TensorFlow.js, and Tailwind CSS dependencies.

## Run Locally

**Prerequisites:** Node.js

1. Install dependencies:
   `npm install`
2. Start the development server:
   `npm run dev`
3. Build a production bundle when needed:
   `npm run build`

## Runtime and deployment expectations

- Frontend libraries are installed from `package.json` and bundled by Vite; the app no longer relies on CDN-provided globals at runtime.
- Tailwind styles are compiled from `/home/runner/work/obj_detection_lab/obj_detection_lab/index.css` during the Vite build.
- The YOLOv8 Nano model URL is pinned to commit `b6a8bc691c0c1897802373d7da65dc889af1f451` from `Hyuto/yolov8-tfjs` to avoid the previous floating `@main` fetch.
- The browser still needs camera permission, and model assets must be reachable the first time a model is loaded so the browser can fetch and cache them.
