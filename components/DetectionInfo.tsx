import React, { useMemo } from 'react';
import type { DetectedObject } from '../types';

interface DetectionInfoProps {
  objects: DetectedObject[];
  isActive: boolean;
  mode: 'webcam' | 'demo';
}

export const DetectionInfo: React.FC<DetectionInfoProps> = ({ objects, isActive, mode }) => {
  const getConfidenceColor = (score: number) => {
    if (score > 0.8) return 'text-green-400';
    if (score > 0.6) return 'text-yellow-400';
    return 'text-orange-400';
  };

  const liveSummary = useMemo(() => {
    if (mode === 'demo') {
      if (objects.length === 0) {
        return 'Classroom demo mode is active. Lower the confidence threshold to reveal sample detections.';
      }

      const counts = objects.reduce<Record<string, number>>((summary, object) => {
        summary[object.class] = (summary[object.class] ?? 0) + 1;
        return summary;
      }, {});

      const objectSummary = Object.entries(counts)
        .sort(([leftClass], [rightClass]) => leftClass.localeCompare(rightClass))
        .map(([objectClass, count]) => `${count} ${objectClass}${count === 1 ? '' : 's'}`)
        .join(', ');

      return `Classroom demo mode is active. Sample detections show ${objectSummary}.`;
    }

    if (!isActive) {
      return 'Webcam is off. Start the webcam to receive live detection updates.';
    }

    if (objects.length === 0) {
      return 'Scanning for objects.';
    }

    const counts = objects.reduce<Record<string, number>>((summary, object) => {
      summary[object.class] = (summary[object.class] ?? 0) + 1;
      return summary;
    }, {});

    const objectSummary = Object.entries(counts)
      .sort(([leftClass], [rightClass]) => leftClass.localeCompare(rightClass))
      .map(([objectClass, count]) => `${count} ${objectClass}${count === 1 ? '' : 's'}`)
      .join(', ');

    return `Detected ${objectSummary}.`;
  }, [isActive, mode, objects]);

  return (
    <section
      id="detection-results"
      aria-labelledby="detected-objects-heading"
      className="bg-gray-800 rounded-lg shadow-inner p-4 border border-gray-700 h-48 overflow-y-auto"
    >
      <div className="flex items-start justify-between gap-3 border-b border-gray-600 pb-2 mb-2">
        <h3 id="detected-objects-heading" className="text-lg font-bold text-gray-300">
          Detected Objects
        </h3>
        <p className="text-xs text-gray-400 text-right">
          {mode === 'demo'
            ? 'Sample results update below.'
            : isActive
            ? 'Live results update below.'
            : 'Start the webcam to begin scanning.'}
        </p>
      </div>
      <p className="sr-only" role="status">
        {liveSummary}
      </p>
      <ul className="space-y-2" aria-live="off">
        {mode === 'demo' && (
          <li className="text-blue-300 text-sm rounded-md bg-blue-500/10 border border-blue-500/20 p-2 list-none">
            Classroom demo mode is showing a prepared example scene so you can teach model tradeoffs without using a live webcam.
          </li>
        )}
        {!isActive && mode === 'webcam' && (
          <li className="text-gray-500 text-center pt-8">Webcam is off. Start to see detections.</li>
        )}
        {!isActive && mode === 'demo' && (
          <li className="text-gray-500 text-center pt-2">
            {objects.length === 0
              ? 'The current threshold is hiding all sample detections. Lower it to reveal detections with lower confidence scores.'
              : 'Sample detections update as you change the model and confidence threshold.'}
          </li>
        )}
        {isActive && objects.length === 0 && (
          <li className="text-gray-500 text-center pt-8">Scanning for objects...</li>
        )}
        {objects.map((obj, index) => (
          <li key={index} className="flex justify-between items-center bg-gray-700/50 p-2 rounded-md">
            <span className="capitalize font-semibold text-gray-200">{obj.class}</span>
            <span className={`font-mono text-sm ${getConfidenceColor(obj.score)}`}>
              {(obj.score * 100).toFixed(1)}%
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
};
