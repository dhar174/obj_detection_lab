
import React, { useState, useCallback, useMemo } from 'react';
import { WebcamView } from './components/WebcamView';
import { DetectionInfo } from './components/DetectionInfo';
import { Header } from './components/Header';
import { Footer } from './components/Footer';
import type { DetectedObject, ModelName } from './types';

const App: React.FC = () => {
  const [isWebcamActive, setIsWebcamActive] = useState<boolean>(false);
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
  }, []);

  const toggleWebcam = () => {
    if (isWebcamActive) {
      setIsWebcamActive(false);
      setDetectedObjects([]);
    } else {
      setError(null);
      setIsWebcamActive(true);
    }
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

  const statusMessage = useMemo(() => {
    if (error) {
      return `Error: ${error}`;
    }

    if (!modelLoaded) {
      return 'Loading the selected vision model. The webcam control will become available when loading finishes.';
    }

    if (isWebcamActive) {
      return `Webcam active. Live detection is running with a ${Math.round(confidenceThreshold * 100)} percent confidence threshold. Stop the webcam before changing models.`;
    }

    return `Vision model ready. Webcam is off. Current confidence threshold is ${Math.round(confidenceThreshold * 100)} percent.`;
  }, [confidenceThreshold, error, isWebcamActive, modelLoaded]);

  return (
    <div className="min-h-screen bg-gray-900 text-gray-200 flex flex-col font-sans">
      <Header />
      <main className="flex-grow container mx-auto p-4 flex flex-col items-center justify-center">
        <p id="app-status" className="sr-only" role="status">
          {statusMessage}
        </p>
        <div className="w-full max-w-4xl bg-gray-800 rounded-lg shadow-2xl overflow-hidden border border-gray-700">
          <WebcamView
            isActive={isWebcamActive}
            onDetections={handleDetections}
            onModelLoad={handleModelLoaded}
            onError={handleError}
            modelName={modelName}
            confidenceThreshold={confidenceThreshold}
          />
        </div>

        <div className="w-full max-w-4xl mt-6 flex flex-col md:flex-row gap-6">
          <div className="w-full md:w-1/3 flex flex-col items-center justify-center gap-4">
            <button
              onClick={toggleWebcam}
              disabled={!modelLoaded}
              aria-controls="webcam-panel detection-results"
              aria-describedby="app-status webcam-toggle-help"
              className={`w-full px-8 py-4 text-xl font-bold rounded-lg transition-all duration-300 ease-in-out transform hover:scale-105 shadow-lg
                ${!modelLoaded
                  ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                  : isWebcamActive
                  ? 'bg-red-600 hover:bg-red-700 text-white'
                  : 'bg-green-600 hover:bg-green-700 text-white'
                }`}
            >
              {!modelLoaded 
                ? 'Loading Model...' 
                : isWebcamActive 
                ? 'Stop Webcam' 
                : 'Start Webcam'}
            </button>
            <p id="webcam-toggle-help" className="text-sm text-center text-gray-400">
              Start or stop the webcam. Model selection is disabled while live detection is running.
            </p>

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
                  aria-describedby="threshold-help app-status"
                  aria-valuetext={`${Math.round(confidenceThreshold * 100)} percent confidence threshold`}
                  className="w-full h-2 bg-gray-600 rounded-lg appearance-none cursor-pointer accent-blue-500"
                />
                <p id="threshold-help" className="mt-2 text-xs text-gray-400 text-center">
                  Lower values show more possible matches. Higher values show only stronger matches.
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
                 aria-describedby="model-help app-status"
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
               <p id="model-help" className="mt-2 text-xs text-center text-gray-400">
                 Choose a model before starting the webcam. Stop live detection to switch models.
               </p>
             </div>
            
            {error && <p className="text-red-400 mt-2 text-center" role="alert">{error}</p>}
          </div>

          <div className="w-full md:w-2/3">
             <DetectionInfo objects={detectedObjects} isActive={isWebcamActive} />
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
};

export default App;
