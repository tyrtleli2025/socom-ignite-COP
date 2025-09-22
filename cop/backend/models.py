"""
Pydantic models for Auto COP API.
Mirrors the JSON schemas for request/response validation.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class DetectionClass(str, Enum):
    """Supported detection classes."""
    BRIDGE = "bridge"
    HELIPAD = "helipad"
    BUILDING = "building"
    VEHICLE = "vehicle"


class GeometryType(str, Enum):
    """Supported geometry types."""
    POINT = "Point"
    POLYGON = "Polygon"
    LINESTRING = "LineString"


class Geometry(BaseModel):
    """GeoJSON geometry object."""
    type: GeometryType
    coordinates: List[Any]


class Feature(BaseModel):
    """GeoJSON feature object."""
    type: str = "Feature"
    geometry: Geometry
    properties: Dict[str, Any]


class FeatureCollection(BaseModel):
    """GeoJSON feature collection."""
    type: str = "FeatureCollection"
    features: List[Feature]


class DetectionInput(BaseModel):
    """Input model for detection requests."""
    image_data: bytes
    sop_template: Dict[str, Any]


class SOPOutput(BaseModel):
    """Output model for SOP-compliant detections."""
    detections: FeatureCollection
    sop_compliance: bool = True
    validation_errors: Optional[List[str]] = None


class SOPClass(BaseModel):
    """SOP class definition."""
    name: str
    description: Optional[str] = None
    color: Optional[str] = None
    min_confidence: float = 0.5


class SOPTemplate(BaseModel):
    """SOP template structure."""
    name: str
    version: str
    classes: List[SOPClass]
    rules: Optional[Dict[str, Any]] = None
    
    class Config:
        arbitrary_types_allowed = True
