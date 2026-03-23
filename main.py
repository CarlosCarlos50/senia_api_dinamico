from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
import pickle
from pydantic import BaseModel
from typing import List
import os

app = FastAPI()

# ==================== CORS ====================
# Permite todos los orígenes para pruebas. En producción, reemplaza "*" por tu dominio de GitHub Pages.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== CARGAR MODELO DINÁMICO ====================
MODEL_PATH = "models/dynamic_model.h5"
ENCODER_PATH = "models/dynamic_label_encoder.pkl"

modelo = None
encoder = None
SEQ_LEN = 20  # debe coincidir con el entrenamiento

try:
    modelo = tf.keras.models.load_model(MODEL_PATH)
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    print("✅ Modelo dinámico cargado correctamente")
except Exception as e:
    print(f"❌ Error cargando modelo dinámico: {e}")

# ==================== NORMALIZACIÓN ====================
def normalizar_puntos(puntos_lista):
    """
    Normaliza los 63 puntos (21 landmarks * 3 coordenadas)
    para que sean invariantes a traslación y escala.
    """
    pts = np.array(puntos_lista, dtype=np.float32).reshape(21, 3)
    muñeca = pts[0]
    pts_centrados = pts - muñeca
    distancias = np.linalg.norm(pts_centrados[1:], axis=1)
    tamaño = np.max(distancias) if np.max(distancias) > 0 else 1.0
    pts_normalizados = pts_centrados / tamaño
    return pts_normalizados.flatten().tolist()

# ==================== MODELO DE DATOS ====================
class DatosSecuencia(BaseModel):
    frames: List[List[float]]  # lista de frames, cada frame con 63 puntos

# ==================== ENDPOINT DE PREDICCIÓN ====================
@app.post("/predecir")
async def predecir(entrada: DatosSecuencia):
    if modelo is None or encoder is None:
        raise HTTPException(status_code=503, detail="Modelo dinámico no disponible")

    try:
        secuencia = np.array(entrada.frames, dtype=np.float32)  # (n_frames, 63)

        # Normalizar cada frame
        secuencia_norm = []
        for frame in secuencia:
            pts_norm = normalizar_puntos(frame.tolist())
            # Si el modelo espera dos manos (126), rellenamos la segunda con ceros
            secuencia_norm.append(pts_norm + [0.0] * 63)
        secuencia_norm = np.array(secuencia_norm)

        # Ajustar a SEQ_LEN frames (20)
        if len(secuencia_norm) < SEQ_LEN:
            pad = np.zeros((SEQ_LEN - len(secuencia_norm), 126))
            secuencia_norm = np.vstack([secuencia_norm, pad])
        else:
            secuencia_norm = secuencia_norm[:SEQ_LEN]

        # Agregar dimensión de batch
        entrada_modelo = np.expand_dims(secuencia_norm, axis=0)

        # Predicción
        preds = modelo.predict(entrada_modelo, verbose=0)[0]
        idx = int(np.argmax(preds))
        palabra = encoder.inverse_transform([idx])[0]
        confianza = float(preds[idx])

        return {"seña": palabra, "confianza": round(confianza * 100, 2)}

    except Exception as e:
        print(f"Error en predicción: {e}")
        return {"seña": "", "confianza": 0}

# ==================== ENDPOINT DE VERIFICACIÓN ====================
@app.get("/")
async def root():
    return {
        "mensaje": "API de SeñIA - Modelo dinámico",
        "cargado": modelo is not None
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
