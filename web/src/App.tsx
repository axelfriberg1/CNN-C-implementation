import { useRef } from "react";
import Canvas, { type CanvasHandle } from "./Canvas/Canvas";
import Button from "./Button/Button";

function App() {
  const canvasRef = useRef<CanvasHandle>(null);

  const handleClick = () => {
    if (!canvasRef.current) return;

    const imageData = canvasRef.current.getPixels();
    if (!imageData) return;

    console.log(imageData.data); // Uint8ClampedArray of RGBA pixels
    // Now you can send this to your ML model
  };

  return (
    <>
      <h1>MNIST classification</h1>
      <div className="container">
        <div className="canvas-container">
          <Canvas ref={canvasRef} width={220} height={220} />
        </div>
        <div className="button-container">
          <Button
            description="Predict"
            onClick={handleClick}
            buttonType="button"
          />
        </div>
      </div>
    </>
  );
}

export default App;
