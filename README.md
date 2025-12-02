# VibrationML Backend - Deploy no Render

## Deploy no Render

1. Crie um novo repositório no GitHub com esta pasta
2. No Render, crie um "New Web Service"
3. Conecte o repositório
4. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api:app --host 0.0.0.0 --port $PORT`

## Endpoints

- `POST /predict` - Recebe dados do ESP32
- `GET /health` - Health check (retorna "1")
- `GET /status` - Status do sistema
- `GET /ws` - WebSocket para frontend
- `/` - Interface web VibrationML

## Após Deploy

Anote a URL gerada (ex: `https://seu-app.onrender.com`)
Use essa URL no ESP32.
"# ESP-32_Machine_learning_backend"  
"# ESP-32_Machine_learning_backend"  
"# ESP-32_Machine_learning_backend"  
"# ESP-32_Machine_learning_backend"  
