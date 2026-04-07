
import React, { useMemo } from 'react';
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

  const liveSummary = useMemo(() => {
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
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([objectClass, count]) => `${count} ${objectClass}${count === 1 ? '' : 's'}`)
      .join(', ');

    return `Detected ${objects.length} object${objects.length === 1 ? '' : 's'}: ${objectSummary}.`;
  }, [isActive, objects]);

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
          {isActive ? 'Live results update below.' : 'Start the webcam to begin scanning.'}
        </p>
      </div>
      <p className="sr-only" role="status" aria-live="polite" aria-atomic="true">
        {liveSummary}
      </p>
      <ul className="space-y-2" aria-live="off">
        {!isActive && (
          <li className="text-gray-500 text-center pt-8">Webcam is off. Start to see detections.</li>
        )}
        {isActive && objects.length === 0 && (
          <li className="text-gray-500 text-center pt-8">Scanning for objects...</li>
        )}
        {objects.map((obj, index) => (
          <li key={`${obj.class}-${index}`} className="flex justify-between items-center bg-gray-700/50 p-2 rounded-md">
            <span className="capitalize font-semibold text-gray-200">{obj.class}</span>
            <span className={`font-mono text-sm ${getConfidenceColor(obj.score)}`} aria-label={`${(obj.score * 100).toFixed(1)} percent confidence`}>
              {(obj.score * 100).toFixed(1)}%
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
};
