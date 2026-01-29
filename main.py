from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi import Depends, status
import logging

from pydantic import BaseModel
from typing import List, Dict, Any

from utils import utils

logger = logging.getLogger('api')
main = FastAPI(
    title="My FastAPI API",
    version="1.0.0",
    description="API prediction B3"
    )

# Determine input_size from CSV, excluding only 'Close'
# csv_path = "data/processed/df.csv"
# df = pd.read_csv(csv_path)
# input_size = df.drop(columns=['Close', 'Date']).shape[1]  # Include 'Date' (5 features)
pdc = utils()

class PredictRequest(BaseModel):
    sequence: List[List[float]]  # 2D sequence: list of lists with input_size elements each

class PredictResponse(BaseModel):
    prediction: float  # Scalar prediction for regression
    details: Dict[str, Any]  # Optional details

@main.on_event('startup')
async def startup_load_model():
    try:
        pdc.load()
        logger.info('Modelo e Scaler carregados com sucesso no startup')
    except Exception as e:
        logger.error(f'Falha ao carregar modelo no startup: {e}')
        raise
        
@main.get('/health')
async def health():
    return {'status': 'ok'}

@main.post('/predict', response_model=PredictResponse)
async def predict(req: PredictRequest):
    
    print('variavel passada:')
    print(req)
    print('tipo da variavel:')
    print(type(req))
 #   print(req.shape)
    
    try:
        res = pdc.predict(req)
        return PredictResponse(prediction=res['prediction'], details={})
        
    except Exception as e:
        logger.error(f'Erro na predição: {e}')
        raise HTTPException(status_code=500, detail=str(e))