# B3 Predictor API 📈

Esta é uma **API RESTful** desenvolvida em **Python** com **FastAPI** que recebe os últimos 60 valores de fechamento da ação da Petrobras na B3 e retorna uma previsão do preço de fechamento do próximo dia utilizando um **modelo LSTM (Long Short-Term Memory)** de Machine Learning.

---

## 🧠 Sobre o Projeto

O objetivo deste projeto é demonstrar a aplicação de aprendizado de máquina em séries temporais financeiras, permitindo prever o próximo preço de fechamento de um ativo (PETR4) com base nos últimos 60 valores observados.

Essa API faz:
- Recebimento de dados via JSON
- Normalização dos dados
- Inferência com modelo LSTM previamente treinado
- Retorno da previsão em formato JSON

---

## 📌 Requisitos

Antes de começar, você precisa ter instalado em seu ambiente:

- Python 3.8+
- pip

---

## 🗂️ Estrutura do Projeto

```
b3_predictor_api/
├── __pycache__/
├── b3_lstm_model.keras
├── main.py
├── requirements.txt
├── scaler.pkl
└── utils.py
```

---

## ⚙️ Como Executar

### 1. Clone o repositório

```bash
git clone https://github.com/MConegundes/b3_predictor_api
cd b3_predictor_api
```

### 2. Crie e ative um ambiente virtual

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Inicie a API

```bash
uvicorn main:app --reload
```

A API ficará disponível em:
`http://127.0.0.1:8000`

---

## 📖 Documentação Interativa

Acesse:
`http://127.0.0.1:8000/docs`

---

## 🚀 Endpoints

### GET /

Health check da API.

### POST /predict

Recebe os últimos 60 valores de fechamento da PETR4.

Exemplo de entrada:

```json
{
  "last_prices": [31.54, 32.87, 31.02]
}
```

Resposta:

```json
{
  "predicted_price": 31.49
}
```

---

## 🧠 Modelo

Modelo LSTM treinado com dados históricos da PETR4.

---

## 📦 Dependências

fastapi, uvicorn, tensorflow, numpy, pandas, scikit-learn

---

## 🎥 Video do projeto

[Apresentação_Fase_4](https://drive.google.com/file/d/1yeYmNw2JNQjrhrDho7dFHci16X521Jyb/view?usp=sharing)

---

## 📝 Observações

Projeto educacional para fins de estudo e demonstração.
