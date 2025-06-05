"use client";

import React, { useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";
import { load as cocoSSDLoad } from "@tensorflow-models/coco-ssd";
import * as tf from "@tensorflow/tfjs";

let detectInterval;

const ObjectDetection = () => {
  const [isLoading, setIsLoading] = useState(true);
  const [backendInitialized, setBackendInitialized] = useState(false);
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  // Inicializa o backend do TensorFlow.js
  async function initTF() {
    try {
      // Verifica se o backend WebGL está disponível
      await tf.ready();
      console.log("TensorFlow.js backend:", tf.getBackend());
      setBackendInitialized(true);
      return true;
    } catch (error) {
      console.error("Failed to initialize TensorFlow.js:", error);
      return false;
    }
  }

  async function runCoco() {
    const tfInitialized = await initTF();
    if (!tfInitialized) {
      console.error("TensorFlow.js initialization failed");
      return;
    }

    setIsLoading(true);
    try {
      const net = await cocoSSDLoad();
      setIsLoading(false);

      detectInterval = setInterval(() => {
        runObjectDetection(net);
      }, 100);
    } catch (error) {
      console.error("Error loading COCO-SSD model:", error);
      setIsLoading(false);
    }
  }

  async function runObjectDetection(net) {
    if (
      canvasRef.current &&
      webcamRef.current !== null &&
      webcamRef.current.video?.readyState === 4
    ) {
      canvasRef.current.width = webcamRef.current.video.videoWidth;
      canvasRef.current.height = webcamRef.current.video.videoHeight;

      try {
        const detectedObjects = await net.detect(
          webcamRef.current.video,
          undefined,
          0.7
        );

        console.log(detectedObjects);
        renderPredictions(detectedObjects, canvasRef.current.getContext("2d"));
      } catch (error) {
        console.error("Detection error:", error);
      }
    }
  }

  function renderPredictions(predictions, ctx) {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    const font = "16px sans-serif";
    ctx.font = font;
    ctx.textBaseline = "top";

    predictions.forEach((prediction) => {
      const [x, y, width, height] = prediction.bbox;

      ctx.strokeStyle = "#00FFFF";
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, width, height);

      ctx.fillStyle = "#00FFFF";
      const textWidth = ctx.measureText(prediction.class).width;
      const textHeight = parseInt(font, 10); // base 10
      ctx.fillRect(x, y, textWidth + 4, textHeight + 4);
    });

    predictions.forEach((prediction) => {
      const [x, y] = prediction.bbox;

      ctx.fillStyle = "#000000";
      ctx.fillText(prediction.class, x, y);
    });
  }

  useEffect(() => {
    runCoco();

    return () => {
      if (detectInterval) {
        clearInterval(detectInterval);
      }
    };
  }, []);

  return (
    <div className="mt-8">
      {isLoading ? (
        <div className="gradient-text">
          {backendInitialized
            ? "Loading AI Model..."
            : "Initializing TensorFlow.js..."}
        </div>
      ) : (
        <div className="relative flex justify-center items-center gradient p-1.5 rounded-md">
          <Webcam
            ref={webcamRef}
            className="rounded-md w-full lg:h-[720px]"
            muted
            videoConstraints={{
              facingMode: "user",
              width: 1280,
              height: 720,
            }}
          />
          <canvas
            ref={canvasRef}
            className="absolute top-0 left-0 z-99999 w-full lg:h-[720px]"
          />
        </div>
      )}
    </div>
  );
};

export default ObjectDetection;
