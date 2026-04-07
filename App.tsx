
import React, { useState, useCallback, useMemo, useEffect } from 'react';
import { WebcamView } from './components/WebcamView';
import { DetectionInfo } from './components/DetectionInfo';
import { Header } from './components/Header';
import { Footer } from './components/Footer';
import type { DetectedObject, ModelName } from './types';

type DisplayMode = 'webcam' | 'demo';

const DEMO_DETECTIONS: Record<ModelName, DetectedObject[]> = {
  lite_mobilenet_v2: [
    { class: 'person', score: 0.87, bbox: [180, 120, 280, 510] },
    { class: 'laptop', score: 0.71, bbox: [700, 390, 220, 140] },
    { class: 'cell phone', score: 0.56, bbox: [470, 305, 70, 120] },
  ],
  mobilenet_v1: [
    { class: 'person', score: 0.83, bbox: [180, 120, 280, 510] },
    { class: 'laptop', score: 0.67, bbox: [700, 390, 220, 140] },
    { class: 'cell phone', score: 0.51, bbox: [470, 305, 70, 120] },
  ],
  mobilenet_v2: [
    { class: 'person', score: 0.9, bbox: [180, 120, 280, 510] },
    { class: 'laptop', score: 0.77, bbox: [700, 390, 220, 140] },
    { class: 'cell phone', score: 0.62, bbox: [470, 305, 70, 120] },
  ],
  yolov8n: [
    { class: 'person', score: 0.96, bbox: [180, 120, 280, 510] },
    { class: 'laptop', score: 0.9, bbox: [700, 390, 220, 140] },
    { class: 'cell phone', score: 0.78, bbox: [470, 305, 70, 120] },
    { class: 'cup', score: 0.64, bbox: [965, 350, 90, 130] },
  ],
  movenet_lightning: [
    { class: 'person', score: 0.84, bbox: [180, 120, 280, 510] },
  ],
  movenet_thunder: [
    { class: 'person', score: 0.93, bbox: [180, 120, 280, 510] },
  ],
  blazeface: [
    { class: 'face', score: 0.97, bbox: [250, 110, 110, 120] },
  ],
};

const App: React.FC = () => {
  const [displayMode, setDisplayMode] = useState<DisplayMode>('webcam');
  const [isWebcamActive, setIsWebcamActive] = useState<boolean>(false);
  const [startWebcamWhenReady, setStartWebcamWhenReady] = useState<boolean>(false);
  const [detectedObjects, setDetectedObjects] = useState<DetectedObject[]>([]);
  const [modelLoaded, setModelLoaded] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [modelName, setModelName] = useState<ModelName>('lite_mobilenet_v2');
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.5);

  const handleDetections = useCallback((objects: DetectedObject[]) => {
    setDetectedObjects(objects);
  }, []);
  
  const handleModelLoaded = useCallback((loaded: boolean) => {
    setModelLoaded(loaded);
  }, []);

  const handleError = useCallback((message: string) => {
    setError(message);
    setIsWebcamActive(false);
    setStartWebcamWhenReady(false);
  }, []);

  const demoObjects = useMemo(
    () => DEMO_DETECTIONS[modelName].filter(({ score }) => score >= confidenceThreshold),
    [modelName, confidenceThreshold]
  );

  const isShowingWebcam = displayMode === 'webcam' && isWebcamActive;
  const displayedObjects = displayMode === 'demo' ? demoObjects : detectedObjects;
  const isWebcamButtonDisabled = displayMode === 'webcam' && !modelLoaded && !isWebcamActive;

  const switchToWebcamMode = useCallback(() => {
    setError(null);
    setDisplayMode('webcam');
    if (modelLoaded) {
      setStartWebcamWhenReady(false);
      setIsWebcamActive(true);
      return;
    }

    setIsWebcamActive(false);
    setStartWebcamWhenReady(true);
  }, [modelLoaded]);

  const switchToDemoMode = useCallback(() => {
    setDetectedObjects([]);
    setError(null);
    setIsWebcamActive(false);
    setStartWebcamWhenReady(false);
    setDisplayMode('demo');
  }, []);

  useEffect(() => {
    if (displayMode === 'webcam' && startWebcamWhenReady && modelLoaded) {
      setIsWebcamActive(true);
      setStartWebcamWhenReady(false);
    }
  }, [displayMode, startWebcamWhenReady, modelLoaded]);

  const toggleWebcam = () => {
    if (isWebcamActive) {
      setIsWebcamActive(false);
      setStartWebcamWhenReady(false);
      setDetectedObjects([]);
    } else {
      switchToWebcamMode();
    }
  };

  const toggleDemoMode = () => {
    if (displayMode === 'demo') {
      setDisplayMode('webcam');
      return;
    }

    switchToDemoMode();
  };

  const handleModelChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setModelLoaded(false);
    setDetectedObjects([]);
    setError(null);
    setModelName(event.target.value as ModelName);
  };

  const handleThresholdChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setConfidenceThreshold(parseFloat(event.target.value));
  };

  return (
    <div className="min-h-screen bg-gray-900 text-gray-200 flex flex-col font-sans">
      <Header />
      <main className="flex-grow container mx-auto p-4 flex flex-col items-center justify-center">
        <div className="w-full max-w-4xl bg-gray-800 rounded-lg shadow-2xl overflow-hidden border border-gray-700">
          <WebcamView
            isActive={isShowingWebcam}
            onDetections={handleDetections}
            onModelLoad={handleModelLoaded}
            onError={handleError}
            modelName={modelName}
            confidenceThreshold={confidenceThreshold}
            mode={displayMode}
            demoObjects={demoObjects}
          />
        </div>

        <div className="w-full max-w-4xl mt-6 flex flex-col md:flex-row gap-6">
          <div className="w-full md:w-1/3 flex flex-col items-center justify-center gap-4">
            <button
              onClick={toggleWebcam}
              disabled={isWebcamButtonDisabled}
              className={`w-full px-8 py-4 text-xl font-bold rounded-lg transition-all duration-300 ease-in-out transform hover:scale-105 shadow-lg
                ${isWebcamButtonDisabled
                  ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                  : isShowingWebcam
                  ? 'bg-red-600 hover:bg-red-700 text-white'
                  : 'bg-green-600 hover:bg-green-700 text-white'
                }`}
              aria-live="polite"
            >
              {displayMode === 'demo'
                ? 'Switch to Webcam'
                : !modelLoaded 
                ? 'Loading Model...' 
                : isShowingWebcam 
                ? 'Stop Webcam' 
                : 'Start Webcam'}
            </button>

            <button
              onClick={toggleDemoMode}
              className={`w-full px-6 py-3 text-base font-semibold rounded-lg transition-colors border ${
                displayMode === 'demo'
                  ? 'bg-blue-600 border-blue-500 text-white hover:bg-blue-700'
                  : 'bg-gray-800 border-blue-500 text-blue-300 hover:bg-gray-700'
              }`}
            >
              {displayMode === 'demo' ? 'Exit Classroom Demo' : 'Use Classroom Demo'}
            </button>

            <div className="w-full bg-gray-700/50 p-3 rounded-lg border border-gray-600">
               <label htmlFor="threshold-slider" className="flex justify-between text-sm font-medium text-gray-300 mb-2">
                 <span>Confidence Threshold</span>
                 <span className="text-blue-400">{Math.round(confidenceThreshold * 100)}%</span>
               </label>
               <input
                 id="threshold-slider"
                 type="range"
                 min="0.1"
                 max="0.9"
                 step="0.05"
                 value={confidenceThreshold}
                 onChange={handleThresholdChange}
                  className="w-full h-2 bg-gray-600 rounded-lg appearance-none cursor-pointer accent-blue-500"
                />
                <p className="mt-3 text-sm text-gray-300 leading-relaxed">
                  Lower values show more possible matches, including uncertain ones. Higher values hide weaker guesses and keep only the most confident boxes.
                </p>
             </div>

             <div className="w-full">
               <label htmlFor="model-select" className="block text-sm font-medium text-gray-400 mb-2 text-center">
                Vision Model Architecture
              </label>
              <select
                id="model-select"
                value={modelName}
                onChange={handleModelChange}
                disabled={isWebcamActive}
                className="w-full bg-gray-700 border border-gray-600 text-white rounded-lg p-2.5 text-center focus:ring-blue-500 focus:border-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
                 aria-label="Select vision model"
               >
                <optgroup label="SSD Object Detection (COCO)">
                  <option value="lite_mobilenet_v2">SSD MobileNet V2 Lite (Fast)</option>
                  <option value="mobilenet_v2">SSD MobileNet V2 (Balanced)</option>
                  <option value="mobilenet_v1">SSD MobileNet V1 (Legacy)</option>
                </optgroup>
                <optgroup label="YOLO Architecture">
                  <option value="yolov8n">YOLOv8 Nano (New! Fast & Accurate)</option>
                </optgroup>
                <optgroup label="CenterNet Pose Estimation">
                  <option value="movenet_lightning">MoveNet Lightning (Fast Person)</option>
                  <option value="movenet_thunder">MoveNet Thunder (Accurate Person)</option>
                </optgroup>
                <optgroup label="Specialized">
                  <option value="blazeface">BlazeFace (Face Detection)</option>
                </optgroup>
               </select>
             </div>
             
             {error && (
               <div className="w-full rounded-lg border border-red-500/40 bg-red-500/10 p-3 text-center" role="alert">
                 <p className="text-red-300">{error}</p>
                 <button
                    onClick={switchToDemoMode}
                    className="mt-3 inline-flex rounded-md bg-blue-600 px-4 py-2 text-sm font-semibold text-white hover:bg-blue-700"
                  >
                    Use Classroom Demo Instead
                  </button>
               </div>
             )}
          </div>

          <div className="w-full md:w-2/3">
             <DetectionInfo objects={displayedObjects} isActive={isShowingWebcam} mode={displayMode} />
          </div>
        </div>

        <section className="w-full max-w-4xl mt-6 grid gap-4 md:grid-cols-3">
          <article className="rounded-lg border border-gray-700 bg-gray-800/80 p-4">
            <h2 className="text-lg font-bold text-teal-300">Model choices in class</h2>
            <ul className="mt-3 space-y-2 text-sm text-gray-300 leading-relaxed">
              <li><strong>SSD MobileNet:</strong> easiest general-purpose choice for explaining speed versus accuracy.</li>
              <li><strong>YOLOv8 Nano:</strong> often finds more objects, but its larger model can take longer to load.</li>
              <li><strong>MoveNet / BlazeFace:</strong> specialized options when you want to focus on people or faces.</li>
            </ul>
          </article>
          <article className="rounded-lg border border-gray-700 bg-gray-800/80 p-4">
            <h2 className="text-lg font-bold text-blue-300">Threshold talking point</h2>
            <p className="mt-3 text-sm text-gray-300 leading-relaxed">
              The confidence threshold acts like a filter for uncertainty. At {Math.round(confidenceThreshold * 100)}%, the app only keeps predictions that clear that bar.
            </p>
            <p className="mt-2 text-sm text-gray-400 leading-relaxed">
              Try lowering it to show more boxes, then raise it to discuss why some predictions disappear.
            </p>
          </article>
          <article className="rounded-lg border border-gray-700 bg-gray-800/80 p-4">
            <h2 className="text-lg font-bold text-amber-300">Fallback for live demos</h2>
            <p className="mt-3 text-sm text-gray-300 leading-relaxed">
              If the webcam is blocked or unavailable, switch to <strong>Classroom Demo</strong> to show a prepared scene with sample detections.
            </p>
            <p className="mt-2 text-sm text-gray-400 leading-relaxed">
              The model selector and threshold slider still change the example so instructors can keep teaching without waiting on permissions.
            </p>
          </article>
        </section>
      </main>
      <Footer />
    </div>
  );
};

export default App;
