// Luminance weighting to conver rgba to grayscale
// gray = 0.299R + 0.587G + 0.114B
function convertArrayToGray(array: Uint8ClampedArray, length: number) {
  const converted = [];
  for (let i = 0; i < length; i += 4) {
    const red = array[i];
    const green = array[i + 1];
    const blue = array[i + 2];

    converted.push(convertPxToGrayScale(red, green, blue));
  }

  return converted;
}

function convertPxToGrayScale(red: number, green: number, blue: number) {
  return 0.299 * red + 0.587 * green + 0.114 * blue;
}

function normalizeArray(array: number[], length: number) {
  for (let i = 0; i < length; i++) {
    array[i] = array[i] / 255;
  }

  return array;
}

export { convertArrayToGray, normalizeArray };
