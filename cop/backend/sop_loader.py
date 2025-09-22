"""
SOP (Standard Operating Procedure) loader and validator.
Handles loading and basic validation of SOP templates.
"""

import yaml
from typing import Dict, Any, List
from .models import SOPTemplate, SOPClass


def load_sop_template(sop_data: Dict[str, Any]) -> SOPTemplate:
    """
    Load and validate SOP template from dictionary.
    
    Args:
        sop_data: SOP template data as dictionary
        
    Returns:
        Validated SOPTemplate object
        
    Raises:
        ValueError: If template is invalid
    """
    try:
        # Convert version to string if it's a number
        version = sop_data.get("version", "1.0")
        if isinstance(version, (int, float)):
            version = str(version)
        
        # Extract classes
        classes = []
        if "classes" in sop_data:
            for class_data in sop_data["classes"]:
                classes.append(SOPClass(
                    name=class_data.get("name", ""),
                    description=class_data.get("description"),
                    color=class_data.get("color"),
                    min_confidence=class_data.get("min_confidence", 0.5)
                ))
        
        # Create SOP template
        template = SOPTemplate(
            name=sop_data.get("name", "Unknown"),
            version=version,
            classes=classes,
            rules=sop_data.get("rules")
        )
        
        # Basic validation
        if not template.classes:
            raise ValueError("SOP template must contain at least one class")
        
        # Check for required class names
        class_names = [cls.name for cls in template.classes]
        if not class_names:
            raise ValueError("SOP classes must have names")
        
        return template
        
    except Exception as e:
        raise ValueError(f"Invalid SOP template: {str(e)}")


def validate_ontology(classes: List[SOPClass]) -> bool:
    """
    Validate that the ontology/classes are present and valid.
    
    Args:
        classes: List of SOP classes
        
    Returns:
        True if valid, False otherwise
    """
    if not classes:
        return False
    
    # Check for required fields
    for cls in classes:
        if not cls.name or not cls.name.strip():
            return False
        if cls.min_confidence < 0 or cls.min_confidence > 1:
            return False
    
    return True
