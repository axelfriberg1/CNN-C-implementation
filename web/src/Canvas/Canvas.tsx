import React, { useRef, useEffect, useState, useImperativeHandle } from "react";
import "./Canvas.css";

interface CanvasProps {
  width: number;
  height: number;
  smallRef: React.RefObject<HTMLCanvasElement | null>;
}

interface CanvasHandle {
  getSmallPixels: () => Uint8ClampedArray | null;
}

function Canvas({ width, height, smallRef }: CanvasProps) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  const contextRef = useRef<CanvasRenderingContext2D | null>(null);
  const smallCtxRef = useRef<CanvasRenderingContext2D | null>(null);
  const [isPressed, setIsPressed] = useState(false);

  const beginDraw = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!contextRef.current) return;
    contextRef.current.beginPath();
    contextRef.current.moveTo(e.nativeEvent.offsetX, e.nativeEvent.offsetY);
    setIsPressed(true);
  };

  const updateDraw = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isPressed || !contextRef.current) return;
    contextRef.current.lineTo(e.nativeEvent.offsetX, e.nativeEvent.offsetY);
    contextRef.current.strokeStyle = "white";
    contextRef.current.lineWidth = 5;
    contextRef.current.stroke();
  };

  const endDraw = () => {
    if (!contextRef.current) return;
    contextRef.current.closePath();

    const srcImage = ref.current;
    if (!smallCtxRef.current) return;
    if (!srcImage) return;

    smallCtxRef.current.drawImage(srcImage, 0, 0, 28, 28);
    setIsPressed(false);
  };

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;

    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    contextRef.current = ctx;

    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, width, height);

    // initializing invicible ref
    const smallCanvas = smallRef.current;
    if (!smallCanvas) return;

    smallCanvas.width = 28;
    smallCanvas.height = 28;

    const smallCtx = smallCanvas.getContext("2d", { alpha: false });
    if (!smallCtx) return;
    smallCtxRef.current = smallCtx;

    smallCtx.fillStyle = "black";
    smallCtx.fillRect(0, 0, 28, 28);
    console.log("first-smalle", smallCtx.getImageData(0, 0, 28, 28).data);
  }, [width, height, smallRef, ref]);

  return (
    <>
      <canvas
        ref={ref}
        onMouseDown={beginDraw}
        onMouseUp={endDraw}
        onMouseMove={updateDraw}
        onMouseLeave={endDraw}
      />
      <canvas className="small-canvas" ref={smallRef} />
    </>
  );
}

export default Canvas;
