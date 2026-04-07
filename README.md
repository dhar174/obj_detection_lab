# Classroom Object Detection Demo

This repository contains a small React + TypeScript classroom demo for running browser-based object detection from a webcam feed. The app loads TensorFlow.js models in the browser and can switch between SSD MobileNet, BlazeFace, MoveNet, and a YOLOv8 Nano option.

## Prerequisites

- Node.js 20+
- A modern desktop browser with webcam support

## Run locally

1. Install dependencies:
   ```bash
   npm install
   ```
2. Start the Vite development server:
   ```bash
   npm run dev
   ```
3. Open the local URL printed by Vite and allow camera access when prompted.

> This app does **not** require a Gemini API key or any other server-side credentials.

## Validate the app

Build the production bundle to verify the app still compiles:

```bash
npm run build
```

## Classroom notes

- Model downloads happen in the browser the first time a model is selected, so the initial load can take longer on slower networks.
- Webcam access must be granted in the browser before live detection can start.
