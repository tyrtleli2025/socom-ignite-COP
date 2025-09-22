"""
Smoke tests for Auto COP API.
Tests basic functionality with dummy data.
"""

import pytest
import io
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)


def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_annotate_endpoint():
    """Test image annotation endpoint with dummy data."""
    # Create a tiny dummy image (1x1 pixel PNG)
    dummy_image = io.BytesIO()
    # Minimal PNG header for a 1x1 pixel image
    dummy_image.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\nIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00IEND\xaeB`\x82')
    dummy_image.seek(0)
    
    # Dummy SOP template
    sop_yaml = """
name: "Test SOP"
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
"""
    
    # Test the endpoint
    response = client.post(
        "/api/annotate",
        files={"image": ("test.png", dummy_image, "image/png")},
        data={"sop_yaml": sop_yaml}
    )
    
    assert response.status_code == 200
    
    # Check response structure
    result = response.json()
    assert "type" in result
    assert result["type"] == "FeatureCollection"
    assert "features" in result
    assert isinstance(result["features"], list)
    assert len(result["features"]) > 0
    
    # Check feature structure
    feature = result["features"][0]
    assert "type" in feature
    assert feature["type"] == "Feature"
    assert "geometry" in feature
    assert "properties" in feature
    
    # Check geometry
    geometry = feature["geometry"]
    assert "type" in geometry
    assert "coordinates" in geometry
    
    # Check properties
    properties = feature["properties"]
    assert "class" in properties
    assert "confidence" in properties
    assert "id" in properties


def test_annotate_endpoint_invalid_sop():
    """Test annotation endpoint with invalid SOP template."""
    dummy_image = io.BytesIO()
    dummy_image.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\nIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00IEND\xaeB`\x82')
    dummy_image.seek(0)
    
    # Invalid SOP template (missing required fields)
    invalid_sop = """
name: "Invalid SOP"
version: "1.0"
"""
    
    response = client.post(
        "/api/annotate",
        files={"image": ("test.png", dummy_image, "image/png")},
        data={"sop_yaml": invalid_sop}
    )
    
    # Should return 500 due to validation error
    assert response.status_code == 500


if __name__ == "__main__":
    pytest.main([__file__])
