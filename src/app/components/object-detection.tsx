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
  const [entradaCount, setEntradaCount] = useState(0);
  const [saidaCount, setSaidaCount] = useState(0);

  const trackedPersonsRef = useRef({});
  const nextPersonIdRef = useRef(0);

  function getDistance(x1, y1, x2, y2) {
    return Math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2);
  }

  function updateTrackedPersons(detections, canvasWidth) {
    const newTracked = {};
    const usedIds = new Set();

    detections
      .filter((d) => d.class === "person")
      .forEach((det) => {
        const [x, y, w, h] = det.bbox;
        const centerX = x + w / 2;
        const centerY = y + h / 2;

        let minDist = Infinity;
        let matchedId = null;

        for (const id in trackedPersonsRef.current) {
          const person = trackedPersonsRef.current[id];
          const dist = getDistance(centerX, centerY, person.x, person.y);

          if (dist < 50 && !usedIds.has(id)) {
            // tolerância de rastreamento
            if (dist < minDist) {
              minDist = dist;
              matchedId = id;
            }
          }
        }

        if (matchedId !== null) {
          newTracked[matchedId] = {
            x: centerX,
            y: centerY,
            crossed: trackedPersonsRef.current[matchedId].crossed,
          };
          usedIds.add(matchedId);
        } else {
          const newId = nextPersonIdRef.current++;
          newTracked[newId] = {
            x: centerX,
            y: centerY,
            crossed: false,
          };
        }
      });

    // Detectar cruzamentos
    for (const id in newTracked) {
      const person = newTracked[id];
      const prev = trackedPersonsRef.current[id];
      const centerLine = canvasWidth / 2;

      if (prev && !person.crossed) {
        if (prev.x < centerLine && person.x >= centerLine) {
          console.log(`Pessoa ${id} cruzou da ESQUERDA para DIREITA`);
          setEntradaCount((prev) => prev + 1);
          person.crossed = true;
        } else if (prev.x > centerLine && person.x <= centerLine) {
          console.log(`Pessoa ${id} cruzou da DIREITA para ESQUERDA`);
          setSaidaCount((prev) => prev + 1);
          person.crossed = true;
        }
      }
    }

    trackedPersonsRef.current = newTracked;
  }

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
        updateTrackedPersons(detectedObjects, canvasRef.current.width);
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
          <div className="absolute top-4 left-4 bg-white bg-opacity-80 p-4 rounded shadow-md z-50 text-black">
            <div>
              <strong>Entrada:</strong> {entradaCount}
            </div>
            <div>
              <strong>Saída:</strong> {saidaCount}
            </div>
          </div>
        </div>
      )}
      {/*
      <div className="absolute top-4 left-4 bg-white bg-opacity-80 p-4 rounded shadow-md z-50 text-black">
        <div>
          <strong>Entrada:</strong> {entradaCount}
        </div>
        <div>
          <strong>Saída:</strong> {saidaCount}
        </div>
      </div>
              */}
    </div>
  );
};

export default ObjectDetection;
