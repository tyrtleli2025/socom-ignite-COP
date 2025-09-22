# Auto COP - Automated SOP-Compliant Map Annotations

Auto COP is a Python + FastAPI + Leaflet application that automates the generation of Standard Operating Procedure (SOP)-compliant map annotations from imagery.

## Features

- **Image Upload & Processing**: Upload images and get automated annotations
- **SOP Compliance**: Validate detections against configurable SOP templates
- **Interactive Map**: Visualize results on a Leaflet map with GeoJSON overlays
- **RESTful API**: Clean FastAPI backend with proper validation
- **Extensible Pipeline**: Easy to add new detection algorithms

## Project Structure

```
cop/
├── backend/
│   ├── app.py                    # FastAPI application
│   ├── models.py                 # Pydantic models
│   ├── vision/
│   │   └── pipeline.py          # Detection pipeline
│   ├── schemas/                 # JSON schemas
│   └── sop_loader.py            # SOP template loader
├── frontend/
│   └── index.html               # Leaflet web interface
├── sop/
│   └── sop_template.yaml        # Default SOP template
├── tests/
│   └── test_api_smoke.py        # API smoke tests
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Quick Start

### Prerequisites

- Python 3.11+
- pip or conda

### Installation

1. **Clone and navigate to the project:**
   ```bash
   cd cop
   ```

2. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the backend server:**
   ```bash
   uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **Open the frontend:**
   - Open `frontend/index.html` in your web browser
   - Or serve it with a simple HTTP server:
     ```bash
     cd frontend
     python -m http.server 3000
     ```
     Then visit `http://localhost:3000`

### How to Run

**Backend Server:**
```bash
# Activate virtual environment first
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Development mode with auto-reload
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload

# Production mode
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

**Frontend:**
- Simply open `frontend/index.html` in your browser
- The frontend will connect to the backend API automatically

### How to Dev Test

**Run the test suite:**
```bash
# Activate virtual environment first
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run all tests
python -m pytest tests/

# Run with verbose output
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_api_smoke.py::test_health_endpoint -v
```

**Manual API testing:**
```bash
# Health check
curl http://localhost:8000/health

# Test annotation (requires image file)
curl -X POST "http://localhost:8000/api/annotate" \
  -F "image=@your_image.jpg" \
  -F "sop_yaml=$(cat sop/sop_template.yaml)"
```

## API Endpoints

### GET /health
Health check endpoint.
- **Response**: `{"ok": true}`

### POST /api/annotate
Annotate an image using SOP-compliant detection.
- **Input**: 
  - `image`: Image file (multipart/form-data)
  - `sop_yaml`: SOP template as YAML string
- **Output**: GeoJSON FeatureCollection with detections

## SOP Template Format

The SOP template is a YAML file that defines:
- **Classes**: Detection categories with confidence thresholds
- **Rules**: Validation rules for each class
- **Output Format**: GeoJSON specifications

Example:
```yaml
name: "Default SOP"
version: "1.0"
classes:
  - name: "bridge"
    description: "Bridge structures"
    color: "#e74c3c"
    min_confidence: 0.7
  - name: "helipad"
    description: "Helicopter landing zones"
    color: "#2ecc71"
    min_confidence: 0.8
```

## Development

### Adding New Detection Classes

1. Update `backend/models.py` - add to `DetectionClass` enum
2. Update `backend/sop_loader.py` - add validation rules
3. Update `frontend/index.html` - add color mapping
4. Update `sop/sop_template.yaml` - add class definition

### Extending the Detection Pipeline

The detection logic is in `backend/vision/pipeline.py`. Currently, it returns dummy detections for testing. To add real detection:

1. Implement your detection algorithm
2. Convert results to GeoJSON format
3. Ensure SOP compliance validation

## Dependencies

- **FastAPI**: Modern web framework for APIs
- **Pydantic**: Data validation using Python type hints
- **Uvicorn**: ASGI server for FastAPI
- **PyYAML**: YAML file processing
- **jsonschema**: JSON schema validation
- **Leaflet**: Interactive map library (frontend)

## License

This project is part of the SOCOM Ignite COP initiative.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For issues and questions, please create an issue in the repository.
