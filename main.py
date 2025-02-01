import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs (0 = all logs, 3 = only errors)
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow warnings

import nest_asyncio
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io
import gdown

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (Cross-Origin Resource Sharing) to prevent caching issues
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model from Google Drive
MODEL_PATH = "Latest_oscc_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=14yqq0zwJcBtA8XXPeOXaXlPplbbPqOV-"  # Replace with your actual Google Drive file ID

# Download the model if not already present
if not os.path.exists(MODEL_PATH):
    print(f"Downloading model from {MODEL_URL}...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    print(f"Model downloaded to {MODEL_PATH}")

# Load the model correctly
try:
    model = load_model(MODEL_PATH, compile=False)
except Exception as e:
    print(f"Error loading model: {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and process the uploaded image
        contents = await file.read()
        file.file.close()  # Ensure file is closed properly

        img = image.load_img(io.BytesIO(contents), target_size=(224, 224))  # Resize to model input size
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize pixel values

        # Debugging: Print first pixel values to confirm image changes
        print(f"First pixel value: {img_array[0, 0, 0, :]}")

        # Predict using the model
        prediction = model.predict(img_array)[0][0]

        # Format response
        result = {
            "prediction": "OSCC" if prediction > 0.5 else "Normal",
            "confidence": float(prediction),
        }

        # Return response with "no-cache" headers to ensure fresh results
        return JSONResponse(content=result, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
        )

# Allow FastAPI to run inside Jupyter Notebook
nest_asyncio.apply()

# Run FastAPI server properly
if __name__ == "__main__":
    config = uvicorn.Config(app, host="0.0.0.0", port=8080, log_level="info")  # Use port 8080 instead
    server = uvicorn.Server(config)
    server.run()
