<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# Run and deploy your AI Studio app

This contains everything you need to run your app locally.

View your app in AI Studio: https://ai.studio/apps/1917f059-bbb1-4676-ba06-3af2eb8292bc

## Run Locally

**Prerequisites:**  Node.js


1. Install dependencies:
   `npm install`
2. Set the `GEMINI_API_KEY` in [.env.local](.env.local) to your Gemini API key
3. Run the app:
   `npm run dev`

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
