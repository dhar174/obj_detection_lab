
import React, { useRef, useEffect, useCallback } from 'react';
import type { DetectedObject, ModelName } from '../types';

// Declare global variables from script tags
declare const tf: any;
declare const cocoSsd: {
  load: (config?: { base: 'lite_mobilenet_v2' | 'mobilenet_v1' | 'mobilenet_v2' }) => Promise<any>;
};
declare const blazeface: {
  load: () => Promise<any>;
};
declare const poseDetection: {
  createDetector: (model: any, config: any) => Promise<any>;
  SupportedModels: {
    MoveNet: any;
  };
  movenet: {
    modelType: {
      SINGLEPOSE_LIGHTNING: any;
      SINGLEPOSE_THUNDER: any;
    };
  };
};

type RuntimeLibraries = {
  tf?: typeof tf;
  cocoSsd?: typeof cocoSsd;
  blazeface?: typeof blazeface;
  poseDetection?: typeof poseDetection;
};

interface WebcamViewProps {
  isActive: boolean;
  modelName: ModelName;
  confidenceThreshold: number;
  onDetections: (objects: DetectedObject[]) => void;
  onModelLoad: (loaded: boolean) => void;
  onError: (message: string) => void;
}

const BBOX_COLORS: { [key: string]: string } = {
  person: '#FF33A8',
  car: '#33A8FF',
  'cell phone': '#33FF49',
  dog: '#FFC433',
  cat: '#A833FF',
  face: '#33FFC4',
  default: '#A0A0A0'
};

// COCO Class Labels for YOLO
const YOLO_LABELS = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
  'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
  'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
  'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
  'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
  'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
  'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
  'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 
  'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
];

const MODEL_LABELS: Record<ModelName, string> = {
  lite_mobilenet_v2: 'SSD MobileNet V2 Lite',
  mobilenet_v1: 'SSD MobileNet V1',
  mobilenet_v2: 'SSD MobileNet V2',
  blazeface: 'BlazeFace',
  movenet_lightning: 'MoveNet Lightning',
  movenet_thunder: 'MoveNet Thunder',
  yolov8n: 'YOLOv8 Nano',
};

const YOLO_URL = 'https://cdn.jsdelivr.net/gh/hugozanini/yolov8-tfjs-runtime@main/yolov8n_web_model/model.json';

const getRuntimeLibraries = (): RuntimeLibraries => {
  const runtime = globalThis as typeof globalThis & RuntimeLibraries;
  return {
    tf: runtime.tf,
    cocoSsd: runtime.cocoSsd,
    blazeface: runtime.blazeface,
    poseDetection: runtime.poseDetection,
  };
};

const getMissingModelRuntimeMessage = (modelName: ModelName, libraries: RuntimeLibraries): string | null => {
  const modelLabel = MODEL_LABELS[modelName];

  if (!libraries.tf) {
    return `AI libraries did not finish loading. Refresh the page and make sure your network or content blocker allows TensorFlow.js files.`;
  }

  if (modelName === 'blazeface' && !libraries.blazeface) {
    return `${modelLabel} was blocked before it could load. Refresh the page and allow jsDelivr requests in any content blocker or school filter.`;
  }

  if ((modelName === 'movenet_lightning' || modelName === 'movenet_thunder') && !libraries.poseDetection) {
    return `${modelLabel} was blocked before it could load. Refresh the page and allow jsDelivr requests in any content blocker or school filter.`;
  }

  if (modelName !== 'blazeface' && modelName !== 'movenet_lightning' && modelName !== 'movenet_thunder' && modelName !== 'yolov8n' && !libraries.cocoSsd) {
    return `${modelLabel} was blocked before it could load. Refresh the page and allow jsDelivr requests in any content blocker or school filter.`;
  }

  return null;
};

const getModelLoadErrorMessage = (modelName: ModelName, error: unknown, libraries: RuntimeLibraries): string => {
  const modelLabel = MODEL_LABELS[modelName];
  const missingRuntimeMessage = getMissingModelRuntimeMessage(modelName, libraries);

  if (missingRuntimeMessage) {
    return missingRuntimeMessage;
  }

  const errorMessage = error instanceof Error ? error.message.toLowerCase() : '';

  if (!navigator.onLine) {
    return `${modelLabel} could not load because this device appears to be offline. Reconnect to the internet and try again.`;
  }

  if (
    errorMessage.includes('failed to fetch') ||
    errorMessage.includes('fetch') ||
    errorMessage.includes('networkerror') ||
    errorMessage.includes('load failed') ||
    errorMessage.includes('net::err')
  ) {
    return `${modelLabel} could not be downloaded. Check the internet connection and allow model downloads from jsDelivr before trying again.`;
  }

  if (
    errorMessage.includes('404') ||
    errorMessage.includes('not found') ||
    errorMessage.includes('model.json')
  ) {
    return `${modelLabel} files were not available from the model host. Refresh the page or switch to another model while the network issue is resolved.`;
  }

  if (
    errorMessage.includes('webgl') ||
    errorMessage.includes('wasm') ||
    errorMessage.includes('backend') ||
    errorMessage.includes('browser')
  ) {
    return `${modelLabel} could not start in this browser. Try the latest Chrome, Edge, or Safari and reload the page.`;
  }

  return `${modelLabel} could not finish loading. Refresh the page first, then switch to another model if the problem continues.`;
};

const getWebcamSetupErrorMessage = (error?: unknown): string => {
  const secureContextRequired = !window.isSecureContext
    && window.location.hostname !== 'localhost'
    && window.location.hostname !== '127.0.0.1';

  if (!navigator.mediaDevices?.getUserMedia) {
    return secureContextRequired
      ? 'Webcam access requires HTTPS or localhost in this browser. Open the app from a secure address and try again.'
      : 'This browser does not support webcam access. Use a current version of Chrome, Edge, or Safari.';
  }

  if (error instanceof DOMException) {
    switch (error.name) {
      case 'NotAllowedError':
      case 'PermissionDeniedError':
        return secureContextRequired
          ? 'Webcam access is blocked because this page is not running over HTTPS. Open the app from localhost or HTTPS, then try again.'
          : 'Webcam access was denied. Allow camera access from the browser address bar, then start the webcam again.';
      case 'NotFoundError':
      case 'DevicesNotFoundError':
        return 'No webcam was found. Connect a camera or move to a device with a built-in webcam, then try again.';
      case 'NotReadableError':
      case 'TrackStartError':
        return 'The webcam is busy or unavailable. Close Zoom, Meet, Teams, or other camera apps, then try again.';
      case 'OverconstrainedError':
      case 'ConstraintNotSatisfiedError':
        return 'The requested camera settings are not supported on this device. Reload the page and try the default camera again.';
      case 'AbortError':
        return 'Webcam startup was interrupted. Try starting it again, or reconnect the camera if it was unplugged.';
      case 'SecurityError':
        return 'The browser blocked webcam access for this page. Check site permissions or privacy extensions, then reload.';
      case 'TypeError':
        return secureContextRequired
          ? 'Webcam access requires HTTPS or localhost in this browser. Open the app from a secure address and try again.'
          : 'This browser could not start the webcam with the requested settings. Refresh the page or try a different browser.';
      default:
        break;
    }
  }

  return 'Could not access the webcam. Make sure the camera is connected, not already in use, and allowed in browser settings.';
};

export const WebcamView: React.FC<WebcamViewProps> = ({ 
  isActive, 
  modelName, 
  confidenceThreshold, 
  onDetections, 
  onModelLoad, 
  onError 
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const modelRef = useRef<any>(null);
  const animationFrameId = useRef<number>();
  const streamRef = useRef<MediaStream | null>(null);

  const detectFrame = useCallback(async () => {
    if (!videoRef.current || videoRef.current.readyState < 4 || !modelRef.current || !canvasRef.current) {
      animationFrameId.current = requestAnimationFrame(detectFrame);
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    if (!ctx) {
      animationFrameId.current = requestAnimationFrame(detectFrame);
      return;
    }

    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
    }
    
    let predictions: DetectedObject[] = [];
    
    try {
        if (modelName === 'blazeface') {
            const facePredictions = await modelRef.current.estimateFaces(video, false);
            predictions = facePredictions.map((pred: any) => ({
              bbox: [
                pred.topLeft[0],
                pred.topLeft[1],
                pred.bottomRight[0] - pred.topLeft[0],
                pred.bottomRight[1] - pred.topLeft[1],
              ],
              class: 'face',
              score: pred.probability[0],
            }));
            predictions = predictions.filter(p => p.score >= confidenceThreshold);

        } else if (modelName.startsWith('movenet')) {
            const poses = await modelRef.current.estimatePoses(video);
            predictions = poses
                .filter((pose: any) => pose.score >= confidenceThreshold)
                .map((pose: any) => {
                    const xs = pose.keypoints.map((k: any) => k.x);
                    const ys = pose.keypoints.map((k: any) => k.y);
                    const minX = Math.min(...xs);
                    const maxX = Math.max(...xs);
                    const minY = Math.min(...ys);
                    const maxY = Math.max(...ys);
                    return {
                        bbox: [minX, minY, maxX - minX, maxY - minY],
                        class: 'person',
                        score: pose.score
                    };
                });
        } else if (modelName === 'yolov8n') {
            const { tf: runtimeTf } = getRuntimeLibraries();
            if (!runtimeTf) {
                throw new Error('TensorFlow runtime is unavailable.');
            }
            // YOLOv8 Detection Logic
            const input = runtimeTf.tidy(() => {
                return runtimeTf.image.resizeBilinear(runtimeTf.browser.fromPixels(video), [640, 640])
                    .div(255.0)
                    .expandDims(0);
            });

            // Execute model: Output shape [1, 84, 8400]
            const res = await modelRef.current.execute(input);
            
            const [boxes, scores, classes] = runtimeTf.tidy(() => {
                const output = res.squeeze(0); // [84, 8400]
                const trans = output.transpose([1, 0]); // [8400, 84]
                
                // Slice bounding boxes [xc, yc, w, h]
                const boxesRaw = trans.slice([0, 0], [8400, 4]);
                const scoresRaw = trans.slice([0, 4], [8400, 80]);
                
                const maxScores = scoresRaw.max(1);
                const maxClasses = scoresRaw.argMax(1);
                
                // Convert [xc, yc, w, h] to [y1, x1, y2, x2] for NMS
                const w = boxesRaw.slice([0, 2], [-1, 1]);
                const h = boxesRaw.slice([0, 3], [-1, 1]);
                const xc = boxesRaw.slice([0, 0], [-1, 1]);
                const yc = boxesRaw.slice([0, 1], [-1, 1]);
                
                const y1 = yc.sub(h.div(2));
                const x1 = xc.sub(w.div(2));
                const y2 = yc.add(h.div(2));
                const x2 = xc.add(w.div(2));
                
                const boxesNMS = runtimeTf.concat([y1, x1, y2, x2], 1);
                
                return [boxesNMS, maxScores, maxClasses];
            });

            // NMS
            const nms = await runtimeTf.image.nonMaxSuppressionAsync(boxes, scores, 50, 0.45, confidenceThreshold);
            const boxesData = boxes.gather(nms, 0).dataSync();
            const scoresData = scores.gather(nms, 0).dataSync();
            const classesData = classes.gather(nms, 0).dataSync();
            
            // Scale factors
            const scaleX = video.videoWidth / 640;
            const scaleY = video.videoHeight / 640;

            predictions = [];
            for (let i = 0; i < nms.size; i++) {
                const y1 = boxesData[i * 4] * scaleY;
                const x1 = boxesData[i * 4 + 1] * scaleX;
                const y2 = boxesData[i * 4 + 2] * scaleY;
                const x2 = boxesData[i * 4 + 3] * scaleX;
                const score = scoresData[i];
                const label = YOLO_LABELS[classesData[i]];

                predictions.push({
                    bbox: [x1, y1, x2 - x1, y2 - y1], // [x, y, w, h]
                    class: label || 'unknown',
                    score: score
                });
            }

            // Cleanup tensors
            runtimeTf.dispose([res, input, boxes, scores, classes, nms]);

        } else {
            // COCO-SSD
            predictions = await modelRef.current.detect(video, undefined, confidenceThreshold);
        }
    } catch (e) {
        console.warn("Detection error:", e);
    }
    
    onDetections(predictions);

    // Drawing Logic
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    predictions.forEach(prediction => {
      const [x, y, width, height] = prediction.bbox;
      const color = BBOX_COLORS[prediction.class] || BBOX_COLORS.default;
      
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, width, height);

      const text = `${prediction.class} - ${Math.round(prediction.score * 100)}%`;
      ctx.fillStyle = color;
      const textWidth = ctx.measureText(text).width;
      const labelY = y > 25 ? y - 5 : y + height - 25; 
      ctx.fillRect(x, labelY, textWidth + 10, 25);
      
      ctx.fillStyle = '#000000';
      ctx.font = '16px sans-serif';
      ctx.fillText(text, x + 5, labelY + 18);
    });

    animationFrameId.current = requestAnimationFrame(detectFrame);
  }, [onDetections, modelName, confidenceThreshold]);
  
  // Load Model
  useEffect(() => {
    let isCancelled = false;

    const loadModel = async () => {
      onModelLoad(false);
      if (modelRef.current?.dispose) {
        modelRef.current.dispose();
      }
      modelRef.current = null;

      const libraries = getRuntimeLibraries();
      const missingRuntimeMessage = getMissingModelRuntimeMessage(modelName, libraries);
      if (missingRuntimeMessage) {
        if (!isCancelled) {
          onError(missingRuntimeMessage);
        }
        return;
      }

      try {
        await libraries.tf!.ready();

        let loadedModel;
        if (modelName === 'blazeface') {
          loadedModel = await libraries.blazeface!.load();
        } else if (modelName === 'movenet_lightning') {
           loadedModel = await libraries.poseDetection!.createDetector(libraries.poseDetection!.SupportedModels.MoveNet, {
               modelType: libraries.poseDetection!.movenet.modelType.SINGLEPOSE_LIGHTNING
            });
        } else if (modelName === 'movenet_thunder') {
            loadedModel = await libraries.poseDetection!.createDetector(libraries.poseDetection!.SupportedModels.MoveNet, {
                modelType: libraries.poseDetection!.movenet.modelType.SINGLEPOSE_THUNDER
             });
        } else if (modelName === 'yolov8n') {
            loadedModel = await libraries.tf!.loadGraphModel(YOLO_URL);
        } else {
          loadedModel = await libraries.cocoSsd!.load({ base: modelName as 'lite_mobilenet_v2' | 'mobilenet_v1' | 'mobilenet_v2' });
        }
        if (isCancelled) {
          loadedModel?.dispose?.();
          return;
        }
        modelRef.current = loadedModel;
        onModelLoad(true);
      } catch (error) {
        console.error(`Failed to load ${modelName} model`, error);
        if (!isCancelled) {
          onError(getModelLoadErrorMessage(modelName, error, libraries));
        }
      }
    };

    void loadModel();

    return () => {
      isCancelled = true;
    };
  }, [modelName, onModelLoad, onError]);

  // Effect 1: Handle Webcam Stream
  useEffect(() => {
    let isMounted = true;

    const enableStream = async () => {
      if (!navigator.mediaDevices?.getUserMedia) {
        onError(getWebcamSetupErrorMessage());
        return;
      }

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'user' },
        });
        
        if (!isMounted) {
          stream.getTracks().forEach(track => track.stop());
          return;
        }

        streamRef.current = stream;
        
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = async () => {
              if (!isMounted || !videoRef.current) return;
              try {
                  await videoRef.current.play();
              } catch (e: any) {
                  if (e.name !== 'AbortError' && e.name !== 'NotAllowedError') {
                       console.error("Error playing video:", e);
                  }
                  if (isMounted && e instanceof DOMException && e.name === 'NotAllowedError') {
                    onError(getWebcamSetupErrorMessage(e));
                  }
              }
          };
        }
      } catch (err) {
        if (!isMounted) return;
        console.error("Error accessing webcam:", err);
        onError(getWebcamSetupErrorMessage(err));
      }
    };

    const stopStream = () => {
      if (videoRef.current) {
         videoRef.current.pause();
         videoRef.current.srcObject = null;
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }
      if (canvasRef.current) {
          const ctx = canvasRef.current.getContext('2d');
          if (ctx) {
              ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
          }
      }
    };

    if (isActive) {
      enableStream();
    } else {
      stopStream();
    }

    return () => {
      isMounted = false;
      stopStream();
    };
  }, [isActive, onError]);

  // Effect 2: Handle Detection Loop
  useEffect(() => {
    if (isActive) {
        animationFrameId.current = requestAnimationFrame(detectFrame);
    }
    
    return () => {
        if (animationFrameId.current) {
            cancelAnimationFrame(animationFrameId.current);
        }
    };
  }, [isActive, detectFrame]);

  return (
    <div className="relative aspect-video bg-black rounded-lg">
      <video
        ref={videoRef}
        playsInline
        muted
        className="w-full h-full object-contain rounded-lg"
        aria-label="Webcam feed"
      />
      <canvas
        ref={canvasRef}
        className="absolute top-0 left-0 w-full h-full"
        aria-hidden="true"
      />
      {!isActive && (
         <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50 rounded-lg">
           <div className="text-center p-4">
              <svg xmlns="http://www.w3.org/2000/svg" className="mx-auto h-16 w-16 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
              <p className="mt-4 text-xl text-gray-400">Webcam is Inactive</p>
           </div>
        </div>
      )}
    </div>
  );
};
