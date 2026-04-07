
import React from 'react';

export const Header: React.FC = () => (
  <header className="bg-gray-800 shadow-md border-b border-gray-700">
    <div className="container mx-auto px-4 py-5 text-center">
      <h1 className="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-blue-500">
        AI Object Detection
      </h1>
      <p className="text-gray-400 mt-1">A real-time object detection demo in your browser with TensorFlow.js</p>
    </div>
  </header>
);
