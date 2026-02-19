import React, {
  useRef,
  useEffect,
  useState,
  forwardRef,
  useImperativeHandle,
} from "react";

interface CanvasProps {
  width: number;
  height: number;
}

export interface CanvasHandle {
  getPixels: () => ImageData | null; // returns raw pixel data
}

const Canvas = forwardRef<CanvasHandle, CanvasProps>(
  ({ width, height }, ref) => {
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const contextRef = useRef<CanvasRenderingContext2D | null>(null);
    const [isPressed, setIsPressed] = useState(false);

    useImperativeHandle(ref, () => ({
      getPixels: () => {
        if (!canvasRef.current || !contextRef.current) return null;
        return contextRef.current.getImageData(0, 0, width, height);
      },
    }));

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
      setIsPressed(false);
    };

    useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      canvas.width = width;
      canvas.height = height;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      contextRef.current = ctx;

      ctx.fillStyle = "black";
      ctx.fillRect(0, 0, width, height);
    }, [width, height]);

    return (
      <canvas
        ref={canvasRef}
        onMouseDown={beginDraw}
        onMouseUp={endDraw}
        onMouseMove={updateDraw}
        onMouseLeave={endDraw}
        style={{ display: "block", cursor: "crosshair" }}
      />
    );
  },
);

export default Canvas;
