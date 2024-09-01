from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()

class ForecastRequest(BaseModel):
    current_height: float
    current_rainfall: float

# Load LSTM models
model_putupaula = load_model('river_height_prediction_model.keras')
model_ellagawa = load_model('river_height_prediction_ellagawa_model.keras')
model_ratnapura = load_model('river_height_prediction_rathnapura_model.keras')
model_magura = load_model('river_height_prediction_magura_model.keras')
model_kalawellawa = load_model('river_height_prediction_kalawellawa_model.keras')

@app.post("/forecast/{location}/")
async def forecast_next_5_days(request: ForecastRequest, location: str = Path(..., regex="^(Ratnapura|Ellagawa|Putupaula|Magura|Kalawellawa)$")):
    try:
        # Select the appropriate model based on the location parameter
        if location == "Putupaula":
            model = model_putupaula
        elif location == "Ellagawa":
            model = model_ellagawa
        elif location == "Ratnapura":
            model = model_ratnapura
        elif location == "Magura":
            model = model_magura
        elif location == "Kalawellawa":
            model = model_kalawellawa
        else:
            raise HTTPException(status_code=400, detail="Invalid location parameter")
        
        # Prepare initial features
        current_features = np.array([request.current_height, request.current_rainfall, 0, 0, 0, 0])  # Fill with zeros or other appropriate values
        current_features = current_features.reshape(1, 1, 6)  # Adjust shape to [1, 1, 6]
        
        # Predict sequentially for the next 5 days
        predictions = []
        for _ in range(1):
            next_prediction = model.predict(current_features)
            # Convert numpy.float32 to Python float for serialization
            next_prediction_value = float(next_prediction[0, 0])
            predictions.append(next_prediction_value)
            # Update current_features for the next prediction
            current_features = np.array([[next_prediction_value, request.current_rainfall, 0, 0, 0, 0]]).reshape(1, 1, 6)
        
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
