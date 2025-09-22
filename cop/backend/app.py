"""
Auto COP FastAPI Backend
Main application with API routes for image annotation and SOP processing.
"""

import json
import uuid
from typing import Dict, Any
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yaml

from .models import DetectionInput, SOPOutput
from .vision.pipeline import dummy_detector_geojson
from .sop_loader import load_sop_template

app = FastAPI(title="Auto COP API", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    """Health check response model."""
    ok: bool


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(ok=True)


@app.post("/api/annotate")
async def annotate_image(
    image: UploadFile = File(...),
    sop_yaml: str = Form(...)
):
    """
    Annotate an image using SOP-compliant detection.
    
    Args:
        image: Uploaded image file
        sop_yaml: SOP template as YAML string
        
    Returns:
        GeoJSON FeatureCollection with detections
    """
    try:
        # Load and validate SOP template
        sop_data = yaml.safe_load(sop_yaml)
        sop_template = load_sop_template(sop_data)
        
        # Read image bytes
        image_bytes = await image.read()
        
        # Run detection pipeline
        detections = dummy_detector_geojson(image_bytes, sop_template.model_dump())
        
        # For now, just pass through the detections (SOP validation will be added later)
        return detections
        
    except Exception as e:
        print(f"Error in annotation: {str(e)}")  # Debug print
        raise HTTPException(status_code=500, detail=f"Annotation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
