# Vehicle Carbon Emission Prediction and Eco-Route Recommendation System

A full-stack intelligent application that predicts CO₂ emissions using machine learning and recommends eco-friendly routes by integrating vehicle data, ML inference, and Google Maps API.

## Overview

This system enables users to:
- Look up vehicle specifications from a database of 10,000 vehicles
- Calculate predicted CO₂ emissions for specific routes using ML models
- Receive route recommendations optimized for minimum carbon emissions
- Compare multiple route alternatives with emission predictions

The system combines a FastAPI backend with a responsive web frontend, using a trained RandomForest regression model to predict emissions based on vehicle characteristics, distance, and real-time traffic conditions.

## Features

- **Vehicle Database**: 10,000 vehicles with specifications (type, fuel, engine size, mileage)
- **ML-Powered Predictions**: RandomForest model trained on vehicle and traffic data
- **Real-Time Traffic Analysis**: Integration with Google Maps API for live traffic conditions
- **Eco-Route Recommendations**: Intelligent route selection based on predicted emissions
- **Responsive Web UI**: User-friendly interface for desktop and mobile devices
- **Comprehensive Error Handling**: Graceful error messages and retry logic
- **Security Features**: Rate limiting, CORS configuration, security headers
- **Health Monitoring**: System health check endpoint for monitoring

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8 or higher**
- **pip** (Python package installer)
- **Google Maps API Key** with Directions API enabled

### Getting a Google Maps API Key

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the **Directions API** for your project
4. Create credentials (API Key)
5. Copy your API key for use in the configuration

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd vehicle-emission-eco-route-system
```

### 2. Create a Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root directory:

```bash
cp .env.example .env
```

Edit the `.env` file and add your Google Maps API key:

```env
# Required Configuration
GOOGLE_MAPS_API_KEY=your_actual_google_maps_api_key_here

# Optional Configuration (defaults provided)
API_HOST=0.0.0.0
API_PORT=8000
VEHICLE_DATABASE_PATH=data/vehicle_database_10000.csv
MODEL_PATH=data/emission_model.pkl
MODEL_COLUMNS_PATH=data/model_columns.pkl

# CORS Configuration
CORS_ORIGINS=["http://localhost:3000","http://localhost:8000"]

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_PERIOD=60

# Logging
LOG_LEVEL=INFO
LOG_FILE=app.log
LOG_FORMAT=json

# External API
MAPS_API_TIMEOUT=10
MAPS_API_RETRIES=3

# Environment
ENVIRONMENT=development
```

### 5. Verify Data Files

Ensure the following files exist in the `data/` directory:
- `vehicle_database_10000.csv` - Vehicle database
- `emission_model.pkl` - Trained ML model
- `model_columns.pkl` - Model feature columns

If the ML model files are missing, you can train a new model:

```bash
python scripts/train_emission_model.py
```

## Usage

### Starting the Server

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

#### Calculate Emission

```bash
curl -X POST http://localhost:8000/calculate-emission \
  -H "Content-Type: application/json" \
  -d '{
    "vehicle_no": "MH01AB1234",
    "distance_km": 50.5
  }'
```

#### Get Eco-Route Recommendations

```bash
curl -X POST http://localhost:8000/eco-route \
  -H "Content-Type: application/json" \
  -d '{
    "vehicle_no": "MH01AB1234",
    "source": "Mumbai, Maharashtra",
    "destination": "Pune, Maharashtra"
  }'
```

#### Health Check

```bash
curl http://localhost:8000/health
```

## Project Structure

```
vehicle-emission-eco-route-system/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI application and endpoints
│   ├── config.py                  # Configuration management
│   ├── core/
│   │   ├── __init__.py
│   │   ├── exceptions.py          # Custom exception classes
│   │   ├── logger.py              # Structured logging
│   │   └── validators.py          # Input validation utilities
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py             # Pydantic request/response models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── vehicle_manager.py     # Vehicle database management
│   │   ├── model_manager.py       # ML model management
│   │   ├── maps_service.py        # Google Maps API integration
│   │   └── traffic_analyzer.py    # Traffic level calculation
│   └── static/
│       ├── index.html             # Frontend UI
│       ├── styles.css             # Styling
│       └── app.js                 # Frontend logic
├── data/
│   ├── vehicle_database_10000.csv # Vehicle database
│   ├── emission_model.pkl         # Trained ML model
│   └── model_columns.pkl          # Model feature columns
├── scripts/
│   ├── train_emission_model.py    # Model training script
│   └── vehicle_lookup.py          # Vehicle lookup utility
├── tests/
│   ├── __init__.py
│   ├── test_api_endpoints.py      # API endpoint tests
│   ├── test_frontend.py           # Frontend tests
│   └── test_system_integration.py # Integration tests
├── .env                           # Environment configuration (create from .env.example)
├── .env.example                   # Environment configuration template
├── requirements.txt               # Python dependencies
└── README.md                      # This file
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

### Frontend Components

- **HTML Interface** (`app/static/index.html`): Semantic HTML5 structure
- **Styling** (`app/static/styles.css`): Responsive CSS with modern design
- **JavaScript Logic** (`app/static/app.js`): API interaction and dynamic UI updates

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
- **Parameters**: `vehicle_no` (path parameter)
- **Response**: JSON with vehicle specifications
- **Status Codes**: 200 (success), 404 (not found), 422 (validation error)

### POST /calculate-emission
- **Description**: Calculate CO₂ emission for a vehicle and distance
- **Request Body**: `{"vehicle_no": "string", "distance_km": float}`
- **Response**: JSON with predicted emission
- **Status Codes**: 200 (success), 404 (vehicle not found), 422 (validation error)

### POST /eco-route
- **Description**: Get eco-friendly route recommendations
- **Request Body**: `{"vehicle_no": "string", "source": "string", "destination": "string"}`
- **Response**: JSON with recommended route and alternatives
- **Status Codes**: 200 (success), 400 (no routes), 404 (vehicle not found), 422 (validation error), 503 (API error)

## Troubleshooting

### Common Issues

#### 1. "Missing required configuration: GOOGLE_MAPS_API_KEY"

**Problem**: The Google Maps API key is not configured.

**Solution**:
- Ensure you have created a `.env` file in the project root
- Add your Google Maps API key: `GOOGLE_MAPS_API_KEY=your_key_here`
- Verify the API key is valid and has Directions API enabled

#### 2. "Vehicle database not found"

**Problem**: The vehicle database CSV file is missing.

**Solution**:
- Verify `data/vehicle_database_10000.csv` exists
- Check the `VEHICLE_DATABASE_PATH` in your `.env` file
- Ensure the path is correct relative to the project root

#### 3. "ML model not found"

**Problem**: The ML model files are missing.

**Solution**:
- Verify `data/emission_model.pkl` and `data/model_columns.pkl` exist
- If missing, train a new model: `python scripts/train_emission_model.py`
- Check the `MODEL_PATH` and `MODEL_COLUMNS_PATH` in your `.env` file

#### 4. "Google Maps API request timed out"

**Problem**: Network issues or slow API response.

**Solution**:
- Check your internet connection
- The system automatically retries up to 3 times
- If the issue persists, try again later
- Increase `MAPS_API_TIMEOUT` in `.env` if needed

#### 5. "No routes found between locations"

**Problem**: Invalid or inaccessible locations.

**Solution**:
- Verify location names are spelled correctly
- Use more specific addresses (e.g., "Mumbai, Maharashtra" instead of just "Mumbai")
- Ensure locations are accessible by road
- Try alternative location formats

#### 6. "Rate limit exceeded"

**Problem**: Too many requests in a short time.

**Solution**:
- Wait for the rate limit window to reset (default: 60 seconds)
- Adjust `RATE_LIMIT_REQUESTS` and `RATE_LIMIT_PERIOD` in `.env` if needed
- For production, consider implementing user-specific rate limiting

#### 7. Port 8000 already in use

**Problem**: Another application is using port 8000.

**Solution**:
- Stop the other application using port 8000
- Or change the port: `uvicorn app.main:app --port 8001`
- Or update `API_PORT` in `.env` file

#### 8. Module import errors

**Problem**: Python packages not installed correctly.

**Solution**:
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`
- Verify Python version is 3.8 or higher: `python --version`

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_api_endpoints.py
```

## Development

### Running in Development Mode

```bash
# With auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# With debug logging
LOG_LEVEL=DEBUG uvicorn app.main:app --reload
```

### Code Style

The project follows PEP 8 style guidelines. Format code using:

```bash
# Install formatting tools
pip install black isort

# Format code
black app/ tests/
isort app/ tests/
```

## Security Considerations

- **API Key Protection**: Never commit `.env` file to version control
- **Rate Limiting**: Configured to prevent abuse (100 requests per minute by default)
- **Input Validation**: All inputs are validated and sanitized
- **Security Headers**: X-Content-Type-Options, X-Frame-Options, CSP headers included
- **Error Sanitization**: Internal error details hidden in production mode
- **CORS Configuration**: Configure allowed origins in production

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Support

For issues, questions, or contributions, please [add contact information or issue tracker link].

## Acknowledgments

- Google Maps API for route and traffic data
- scikit-learn for machine learning capabilities
- FastAPI for the web framework
