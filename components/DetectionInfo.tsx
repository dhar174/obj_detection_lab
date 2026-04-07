
import React from 'react';
import type { DetectedObject } from '../types';

interface DetectionInfoProps {
  objects: DetectedObject[];
  isActive: boolean;
}

export const DetectionInfo: React.FC<DetectionInfoProps> = ({ objects, isActive }) => {
  const getConfidenceColor = (score: number) => {
    if (score > 0.8) return 'text-green-400';
    if (score > 0.6) return 'text-yellow-400';
    return 'text-orange-400';
  };

  return (
    <div className="bg-gray-800 rounded-lg shadow-inner p-4 border border-gray-700 h-48 overflow-y-auto">
      <h3 className="text-lg font-bold text-gray-300 border-b border-gray-600 pb-2 mb-2">Detected Objects</h3>
      <div className="space-y-2">
        {!isActive && (
          <p className="text-gray-500 text-center pt-8">Webcam is off. Start to see detections.</p>
        )}
        {isActive && objects.length === 0 && (
          <p className="text-gray-500 text-center pt-8">Scanning for objects...</p>
        )}
        {objects.map((obj, index) => (
          <div key={index} className="flex justify-between items-center bg-gray-700/50 p-2 rounded-md">
            <span className="capitalize font-semibold text-gray-200">{obj.class}</span>
            <span className={`font-mono text-sm ${getConfidenceColor(obj.score)}`}>
              {(obj.score * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};
