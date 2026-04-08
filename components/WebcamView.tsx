
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
  mode: 'webcam' | 'demo';
  demoObjects: DetectedObject[];
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

const getRuntimeLibraryReferences = (): RuntimeLibraries => {
  const runtime = globalThis as typeof globalThis & RuntimeLibraries;
  return {
    tf: runtime.tf,
    cocoSsd: runtime.cocoSsd,
    blazeface: runtime.blazeface,
    poseDetection: runtime.poseDetection,
  };
};

const requiresCocoSsd = (modelName: ModelName): boolean => (
  modelName === 'lite_mobilenet_v2'
  || modelName === 'mobilenet_v1'
  || modelName === 'mobilenet_v2'
);

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

  if (requiresCocoSsd(modelName) && !libraries.cocoSsd) {
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

/**
 * getUserMedia failures can arrive as DOMException instances or plain TypeError-style errors,
 * so normalize the name lookup before mapping them to user-facing webcam guidance.
 */
const getErrorName = (error?: unknown): string | undefined => {
  if (error instanceof Error) {
    return error.name;
  }

  if (typeof error === 'object' && error !== null && 'name' in error && typeof error.name === 'string') {
    return error.name;
  }

  return undefined;
};

const getWebcamSetupErrorMessage = (error?: unknown): string => {
  const isInsecureNonLocalContext = !window.isSecureContext
    && window.location.hostname !== 'localhost'
    && window.location.hostname !== '127.0.0.1';
  const errorName = getErrorName(error);

  if (!navigator.mediaDevices?.getUserMedia) {
    return isInsecureNonLocalContext
      ? 'Webcam access requires HTTPS or localhost in this browser. Open the app from a secure address and try again.'
      : 'This browser does not support webcam access. Use a current version of Chrome, Edge, or Safari.';
  }

  if (errorName === 'TypeError') {
    return isInsecureNonLocalContext
      ? 'Webcam access requires HTTPS or localhost in this browser. Open the app from a secure address and try again.'
      : 'This browser could not start the webcam with the requested settings. Refresh the page or try a different browser.';
  }

  if (error instanceof DOMException) {
    switch (errorName) {
      case 'NotAllowedError':
      case 'PermissionDeniedError':
        return isInsecureNonLocalContext
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
      default:
        break;
    }
  }

  return 'Could not access the webcam. Make sure the camera is connected, not already in use, and allowed in browser settings.';
};

// Keep pose boxes stable by ignoring very low-confidence joints.
const MOVENET_MIN_KEYPOINT_SCORE = 0.3;
// Add a small margin so the box does not hug the outermost confident joints.
const MOVENET_BOX_PADDING = 0.12;
const MOVENET_MIN_PADDING_PX = 12;
// Smaller YOLO input keeps the classroom demo responsive on lower-end laptops.
const YOLO_INPUT_SIZE = 512;
const YOLO_MAX_DETECTIONS = 20;
const YOLO_MIN_FRAME_INTERVAL_MS = 120;

const getMoveNetBoundingBox = (
  pose: any,
  videoWidth: number,
  videoHeight: number,
  confidenceThreshold: number
): [number, number, number, number] | null => {
  const validKeypoints = pose.keypoints.filter((keypoint: any) => (
    Number.isFinite(keypoint.x) && Number.isFinite(keypoint.y)
  ));
  const confidentKeypoints = validKeypoints.filter((keypoint: any) => (
    (keypoint.score ?? 0) >= Math.max(MOVENET_MIN_KEYPOINT_SCORE, confidenceThreshold * 0.5)
  ));
  const keypointsForBox = confidentKeypoints.length >= 4 ? confidentKeypoints : validKeypoints;

  if (keypointsForBox.length < 2) {
    return null;
  }

  const xs = keypointsForBox.map((keypoint: any) => keypoint.x);
  const ys = keypointsForBox.map((keypoint: any) => keypoint.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const paddingX = Math.max((maxX - minX) * MOVENET_BOX_PADDING, MOVENET_MIN_PADDING_PX);
  const paddingY = Math.max((maxY - minY) * MOVENET_BOX_PADDING, MOVENET_MIN_PADDING_PX);
  const left = Math.max(0, minX - paddingX);
  const top = Math.max(0, minY - paddingY);
  const right = Math.min(videoWidth, maxX + paddingX);
  const bottom = Math.min(videoHeight, maxY + paddingY);

  if (right <= left || bottom <= top) {
    return null;
  }

  return [left, top, right - left, bottom - top];
};

export const WebcamView: React.FC<WebcamViewProps> = ({ 
  isActive, 
  modelName, 
  confidenceThreshold, 
  mode,
  demoObjects,
  onDetections, 
  onModelLoad, 
  onError 
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const modelRef = useRef<any>(null);
  const animationFrameId = useRef<number>();
  const streamRef = useRef<MediaStream | null>(null);
  const lastDetectionTimeRef = useRef<number>(0);

  const drawPredictions = useCallback((ctx: CanvasRenderingContext2D, predictions: DetectedObject[]) => {
    const canvas = canvasRef.current;

    if (!canvas) {
      return;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    predictions.forEach(prediction => {
      const [x, y, width, height] = prediction.bbox;
      const color = BBOX_COLORS[prediction.class] || BBOX_COLORS.default;
      
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, width, height);

      const text = `${prediction.class} - ${Math.round(prediction.score * 100)}%`;
      ctx.fillStyle = color;
      ctx.font = '16px sans-serif';
      const textWidth = ctx.measureText(text).width;
      const labelY = Math.max(0, Math.min(canvas.height - 25, y > 25 ? y - 5 : y + height - 25)); 
      ctx.fillRect(x, labelY, textWidth + 10, 25);
      
      ctx.fillStyle = '#000000';
      ctx.fillText(text, x + 5, labelY + 18);
    });
  }, []);

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
    const now = performance.now();

    if (modelName === 'yolov8n' && now - lastDetectionTimeRef.current < YOLO_MIN_FRAME_INTERVAL_MS) {
      animationFrameId.current = requestAnimationFrame(detectFrame);
      return;
    }
    
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
                    const bbox = getMoveNetBoundingBox(
                      pose,
                      video.videoWidth,
                      video.videoHeight,
                      confidenceThreshold
                    );

                    if (!bbox) {
                      return null;
                    }

                    return {
                        bbox,
                        class: 'person',
                        score: pose.score
                    };
                })
                .filter((prediction): prediction is DetectedObject => prediction !== null);
        } else if (modelName === 'yolov8n') {
            const { tf: runtimeTf } = getRuntimeLibraryReferences();
            if (!runtimeTf) {
                throw new Error('TensorFlow runtime is unavailable.');
            }
            // YOLOv8 Detection Logic
            const input = tf.tidy(() => {
                return tf.image.resizeBilinear(tf.browser.fromPixels(video), [YOLO_INPUT_SIZE, YOLO_INPUT_SIZE])
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
            const nms = await tf.image.nonMaxSuppressionAsync(boxes, scores, YOLO_MAX_DETECTIONS, 0.45, confidenceThreshold);
            const { boxesData, scoresData, classesData, detectionCount } = runtimeTf.tidy(() => {
                const selectedBoxes = boxes.gather(nms, 0);
                const selectedScores = scores.gather(nms, 0);
                const selectedClasses = classes.gather(nms, 0);

                return {
                    boxesData: Array.from(selectedBoxes.dataSync()),
                    scoresData: Array.from(selectedScores.dataSync()),
                    classesData: Array.from(selectedClasses.dataSync()),
                    detectionCount: nms.size,
                };
            });
            tf.dispose([res, input, boxes, scores, classes, nms]);
            
            // Scale factors
            const scaleX = video.videoWidth / YOLO_INPUT_SIZE;
            const scaleY = video.videoHeight / YOLO_INPUT_SIZE;

            predictions = [];
            for (let i = 0; i < detectionCount; i++) {
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

    if (modelName === 'yolov8n') {
      lastDetectionTimeRef.current = performance.now();
    }
    
    onDetections(predictions);
    drawPredictions(ctx, predictions);

    animationFrameId.current = requestAnimationFrame(detectFrame);
  }, [onDetections, modelName, confidenceThreshold, drawPredictions]);

  useEffect(() => {
    if (!canvasRef.current) {
      return;
    }

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    if (!ctx) {
      return;
    }

    if (mode === 'demo') {
      canvas.width = 1280;
      canvas.height = 720;
      drawPredictions(ctx, demoObjects);
      return;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }, [mode, demoObjects, drawPredictions]);
  
  // Load Model
  useEffect(() => {
    let isCancelled = false;
    if (mode === 'demo') {
      onModelLoad(false);
      if (modelRef.current?.dispose) {
        modelRef.current.dispose();
      }
      modelRef.current = null;
      return;
    }

    const loadModel = async () => {
      onModelLoad(false);
      if (typeof tf === 'undefined') {
        onError('Vision libraries did not load. You can still use Classroom Demo mode for instruction.');
        return;
      }
      if (modelRef.current?.dispose) {
        modelRef.current.dispose();
      }
      modelRef.current = null;

      const libraries = getRuntimeLibraryReferences();
      const missingRuntimeMessage = getMissingModelRuntimeMessage(modelName, libraries);
      if (missingRuntimeMessage) {
        if (!isCancelled) {
          onError(missingRuntimeMessage);
        }
        return;
      }

      try {
        const runtimeTf = libraries.tf;
        if (!runtimeTf) {
          return;
        }

        await runtimeTf.ready();

        let loadedModel;
        if (modelName === 'blazeface') {
          if (!libraries.blazeface) {
            return;
          }
          loadedModel = await libraries.blazeface.load();
        } else if (modelName === 'movenet_lightning') {
           if (!libraries.poseDetection) {
             return;
           }
           loadedModel = await libraries.poseDetection.createDetector(libraries.poseDetection.SupportedModels.MoveNet, {
               modelType: libraries.poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING
             });
        } else if (modelName === 'movenet_thunder') {
            if (!libraries.poseDetection) {
              return;
            }
            loadedModel = await libraries.poseDetection.createDetector(libraries.poseDetection.SupportedModels.MoveNet, {
                modelType: libraries.poseDetection.movenet.modelType.SINGLEPOSE_THUNDER
              });
        } else if (modelName === 'yolov8n') {
            const yoloModelUrl = 'https://cdn.jsdelivr.net/gh/hugozanini/yolov8-tfjs-runtime@main/yolov8n_web_model/model.json';
            loadedModel = await runtimeTf.loadGraphModel(yoloModelUrl);
        } else {
          if (!libraries.cocoSsd) {
            return;
          }
          loadedModel = await libraries.cocoSsd.load({ base: modelName as 'lite_mobilenet_v2' | 'mobilenet_v1' | 'mobilenet_v2' });
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
  }, [mode, modelName, onModelLoad, onError]);

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
              } catch (error) {
                  const errorName = error instanceof DOMException ? error.name : undefined;
                  if (errorName !== 'AbortError' && errorName !== 'NotAllowedError') {
                       console.error("Error playing video:", error);
                  }
                  if (isMounted && error instanceof DOMException && error.name === 'NotAllowedError') {
                    onError(getWebcamSetupErrorMessage(error));
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
      {mode === 'demo' ? (
        <>
          <svg
            viewBox="0 0 1280 720"
            className="w-full h-full rounded-lg"
            role="img"
            aria-label="Prepared classroom demo scene"
          >
            <rect width="1280" height="720" fill="#111827" />
            <rect x="0" y="470" width="1280" height="250" fill="#1f2937" />
            <rect x="660" y="355" width="350" height="25" rx="8" fill="#374151" />
            <rect x="700" y="380" width="260" height="120" rx="18" fill="#4b5563" />
            <rect x="990" y="330" width="85" height="135" rx="18" fill="#9ca3af" />
            <rect x="960" y="455" width="140" height="18" rx="9" fill="#6b7280" />
            <circle cx="305" cy="150" r="72" fill="#f59e0b" />
            <rect x="240" y="220" width="140" height="200" rx="64" fill="#14b8a6" />
            <rect x="215" y="300" width="55" height="210" rx="20" fill="#14b8a6" />
            <rect x="350" y="300" width="55" height="210" rx="20" fill="#14b8a6" />
            <rect x="245" y="420" width="50" height="200" rx="20" fill="#0f766e" />
            <rect x="325" y="420" width="50" height="200" rx="20" fill="#0f766e" />
            <rect x="455" y="295" width="65" height="118" rx="20" fill="#60a5fa" />
            <rect x="120" y="70" width="245" height="50" rx="12" fill="#0f172a" opacity="0.72" />
            <text x="150" y="103" fill="#e5e7eb" fontSize="28" fontFamily="sans-serif">Teacher + desk scene</text>
          </svg>
          <div className="absolute left-4 top-4 rounded-full bg-blue-600/90 px-4 py-2 text-sm font-semibold text-white shadow-lg">
            Classroom Demo Mode
          </div>
        </>
      ) : (
        <video
          ref={videoRef}
          playsInline
          muted
          className="w-full h-full object-contain rounded-lg"
          aria-label="Webcam feed"
        />
      )}
      <canvas
        ref={canvasRef}
        className="absolute top-0 left-0 w-full h-full"
        aria-hidden="true"
      />
      {!isActive && mode !== 'demo' && (
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
