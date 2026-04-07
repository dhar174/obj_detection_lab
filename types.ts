
export interface DetectedObject {
  bbox: [number, number, number, number]; // [x, y, width, height]
  class: string;
  score: number;
}

export type ModelName = 
  | 'lite_mobilenet_v2' 
  | 'mobilenet_v1' 
  | 'mobilenet_v2' 
  | 'blazeface'
  | 'movenet_lightning'
  | 'movenet_thunder'
  | 'yolov8n';
