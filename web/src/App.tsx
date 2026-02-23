import { useEffect, useRef, useState } from "react";
import Canvas from "./Canvas/Canvas";
import Button from "./Button/Button";
import { convertArrayToGray, normalizeArray } from "./utils.ts";
function App() {
  const smallRef = useRef<HTMLCanvasElement | null>(null);

  const handleClick = () => {
    if (!smallRef.current) return;

    const ctx = smallRef.current.getContext("2d");

    if (!ctx) return;
    const pixels = ctx.getImageData(0, 0, 28, 28).data;
    const grayScalePixels = convertArrayToGray(pixels, 28 * 28 * 4);
    const normalized = normalizeArray(grayScalePixels, 28 * 28);
  };
  return (
    <>
      <h1>MNIST classification</h1>
      <div className="container">
        <div className="canvas-container">
          <Canvas smallRef={smallRef} width={220} height={220} />
        </div>
        <div className="button-container">
          <Button
            onClick={handleClick}
            buttonType="button"
            description="predict"
          ></Button>
        </div>
      </div>
    </>
  );
}

export default App;
