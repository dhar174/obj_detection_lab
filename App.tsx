
import React, { useState, useCallback } from 'react';
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

  return (
    <div className="min-h-screen bg-gray-900 text-gray-200 flex flex-col font-sans">
      <Header />
      <main className="flex-grow container mx-auto p-4 flex flex-col items-center justify-center">
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
              className={`w-full px-8 py-4 text-xl font-bold rounded-lg transition-all duration-300 ease-in-out transform hover:scale-105 shadow-lg
                ${!modelLoaded
                  ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                  : isWebcamActive
                  ? 'bg-red-600 hover:bg-red-700 text-white'
                  : 'bg-green-600 hover:bg-green-700 text-white'
                }`}
              aria-live="polite"
            >
              {!modelLoaded 
                ? 'Loading Model...' 
                : isWebcamActive 
                ? 'Stop Webcam' 
                : 'Start Webcam'}
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
              <div
                className="w-full rounded-lg border border-red-500/40 bg-red-500/10 p-3 text-left text-sm text-red-100"
                role="alert"
              >
                <p className="font-semibold text-red-300">Action needed</p>
                <p className="mt-1 whitespace-pre-line">{error}</p>
              </div>
            )}
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
