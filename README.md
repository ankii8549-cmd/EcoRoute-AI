# 🌱 EcoRoute AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.0-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.3.0-02569B?style=for-the-badge)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2.2-FFCC00?style=for-the-badge)

A full-stack intelligent application that predicts CO₂ emissions using advanced machine learning and recommends eco-friendly routes by integrating vehicle data, ML inference, and Google Maps API.

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [API](#-api-endpoints) • [Deployment](#-deployment)

</div>


## 📋 Overview

This system enables users to:
- 🚗 Look up vehicle specifications from a database of 40,000+ vehicles
- 📊 Calculate predicted CO₂ emissions using advanced ML models
- 🗺️ Receive route recommendations optimized for minimum carbon emissions
- 🔄 Compare multiple route alternatives with real-time traffic analysis

The system combines a FastAPI backend with a responsive web frontend, using a **Stacking Ensemble** of XGBoost, LightGBM, and CatBoost models to predict emissions based on vehicle characteristics, distance, and real-time traffic conditions.


## ✨ Features

### 🎯 Core Capabilities
- **Vehicle Database**: 40,000+ vehicles from Canada fuel consumption dataset (2015-2024)
- **Advanced ML Models**: Stacking Ensemble combining XGBoost, LightGBM, and CatBoost
- **Real-Time Traffic**: Integration with Google Maps API for live traffic conditions
- **Eco-Route Recommendations**: Intelligent route selection based on predicted emissions
- **Responsive Web UI**: User-friendly interface for desktop and mobile devices

### 🔒 Security & Performance
- **Rate Limiting**: Prevents abuse (100 requests/minute)
- **Input Validation**: Comprehensive validation and sanitization
- **Security Headers**: X-Content-Type-Options, X-Frame-Options, CSP
- **Route Caching**: 5-minute TTL for faster responses
- **Health Monitoring**: System health check endpoint

### 📈 Model Performance
- **R² Score**: > 0.92
- **MAE**: < 10 g/km
- **RMSE**: < 15 g/km
- **Model Explainability**: SHAP values for interpretability

## 🛠️ Tech Stack

**Backend**
- FastAPI 0.109.0
- Uvicorn (ASGI server)
- Pydantic v2 (validation)
- Python 3.8+

**Machine Learning**
- scikit-learn 1.4.0
- XGBoost 2.0.3
- LightGBM 4.3.0
- CatBoost 1.2.2
- SHAP 0.44.1 (explainability)

**Data Processing**
- pandas 2.1.4
- numpy 1.26.3

**External APIs**
- Google Maps Directions API
- Tenacity (retry logic)

**Frontend**
- HTML5, CSS3, JavaScript
- Responsive design


## 📦 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Google Maps API Key with Directions API enabled

### 🔑 Getting a Google Maps API Key

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable **Directions API**
4. Create credentials (API Key)
5. Copy your API key


## 🚀 Installation

### 1️⃣ Clone the Repository

```bash
git clone <repository-url>
cd vehicle-emission-eco-route-system
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Environment

Create `.env` file:

```bash
cp .env.example .env
```

Edit `.env` and add your API key:

```env
GOOGLE_MAPS_API_KEY=your_api_key_here
ENVIRONMENT=development
LOG_LEVEL=INFO
```

### 5️⃣ Verify Data Files

Required files:
- `data/cleaned_fuel_consumption.csv`
- `models/stacking_ensemble.pkl`
- `models/stacking_model_columns.pkl`

## 💻 Usageting the Server

Run the FastAPI server using Uvicorn:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The server will start at `http://localhost:8000`

### Accessing the Application

1. **Web Interface**: Open your browser and navigate to `http://localhost:8000`
2. **API Documentation**: Visit `http://localhost:8000/docs` for interactive API documentation
3. **Alternative API Docs**: Visit `http://localhost:8000/redoc` for ReDoc documentation

### Using the Web Interface

1. Enter a vehicle number (e.g., `MH01AB1234`)
2. Enter source location (e.g., `Mumbai, Maharashtra`)
3. Enter destination location (e.g., `Pune, Maharashtra`)
4. Click "Get Eco-Route Recommendations"
5. View the recommended route with lowest emissions and compare alternatives

### Using the API Directly

#### Get Vehicle Details

```bash
curl http://localhost:8000/vehicle/MH01AB1234
```

## 💻 Usage

### Starting the Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Server runs at `http://localhost:8000`

### Accessing the Application

- 🌐 **Web Interface**: `http://localhost:8000`
- 📚 **API Docs**: `http://localhost:8000/docs`
- 📖 **ReDoc**: `http://localhost:8000/redoc`

### Using the Web Interface

1. Enter vehicle number (e.g., `BMW 320i`)
2. Enter source location (e.g., `New York, NY`)
3. Enter destination (e.g., `Boston, MA`)
4. Click "Get Eco-Route Recommendations"
5. View recommended route with lowest emissions

### API Examples

#### Get Vehicle Details
```bash
curl http://localhost:8000/vehicle/BMW%20320i
```

#### Calculate Emission
```bash
curl -X POST http://localhost:8000/calculate-emission \
  -H "Content-Type: application/json" \
  -d '{"vehicle_no": "BMW 320i", "distance_km": 50}'
```

#### Get Eco-Route
```bash
curl -X POST http://localhost:8000/eco-route \
  -H "Content-Type: application/json" \
  -d '{
    "vehicle_no": "BMW 320i",
    "source": "New York, NY",
    "destination": "Boston, MA"
  }'
```

#### Health Check
```bash
curl http://localhost:8000/health
```


## Key Components

### Backend Components

- **FastAPI Application** (`app/main.py`): RESTful API server with endpoints for vehicle lookup, emission calculation, and route recommendations
- **Configuration Manager** (`app/config.py`): Centralized environment-based configuration using Pydantic Settings
- **Vehicle Manager** (`app/services/vehicle_manager.py`): Efficient vehicle database management with O(1) lookup
- **ML Model Manager** (`app/services/model_manager.py`): ML model loading and prediction
- **Maps Service** (`app/services/maps_service.py`): Google Maps API integration with retry logic and caching
- **Traffic Analyzer** (`app/services/traffic_analyzer.py`): Traffic level calculation based on duration data
- **Exception Handlers** (`app/core/exceptions.py`): Custom exception hierarchy for error handling
- **Validators** (`app/core/validators.py`): Input validation and sanitization


### ML Model

- **Algorithm**: RandomForest Regressor
- **Input Features**: Engine size, mileage, distance, traffic level, fuel type (one-hot encoded), vehicle type (one-hot encoded)
- **Output**: Predicted CO₂ emission in kilograms

## API Endpoints

### GET /
- **Description**: Serve the frontend UI
- **Response**: HTML page

### GET /health
- **Description**: System health check
- **Response**: JSON with system status and component health
- **Status Codes**: 200 (healthy), 503 (unhealthy)

### GET /vehicle/{vehicle_no}
- **Description**: Retrieve vehicle details by vehicle number
## 🔌 API Endpoints

| Method | Endpoint | Description | Status Codes |
|--------|----------|-------------|--------------|
| GET | `/` | Serve frontend UI | 200 |
| GET | `/health` | System health check | 200, 503 |
| GET | `/vehicle/{vehicle_no}` | Get vehicle details | 200, 404, 422 |
| POST | `/calculate-emission` | Calculate CO₂ emission | 200, 404, 422 |
| POST | `/eco-route` | Get eco-route recommendations | 200, 400, 404, 422, 503 |


<div align="center">

Made with ❤️ for a greener planet 🌍

</div>