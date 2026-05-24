"""
Attribute schema for CIRR dataset.

Generated from Step 1 attribute discovery on train data.
"""

ATTRIBUTE_SCHEMA = {
    "background_type": {
        "description": "The main setting or environment depicted in the image, such as indoor, outdoor, natural landscapes, urban, or specific contexts like a gym or living room.",
        "values": [
            "indoor",
            "outdoor",
            "natural landscape",
            "plain",
            "underwater",
            "product packaging",
            "urban",
            "studio",
            "library",
            "other"
        ]
    },
    "number_of_subjects": {
        "description": "The count of main subjects (animals, people, or key objects) present in the image.",
        "values": [
            "1",
            "2",
            "3",
            "4",
            "5-6",
            "7-10",
            "more than 10",
            "multiple objects (non-human)",
            "other"
        ]
    },
    "color": {
        "description": "The primary or dominant color of the main subject or object in the image (animal fur, clothing, product packaging, etc.).",
        "values": [
            "brown",
            "black",
            "white",
            "gray",
            "yellow",
            "red",
            "gold",
            "orange",
            "blue",
            "other"
        ]
    },
    "pose": {
        "description": "The physical position, orientation, or activity state of the primary subject (animal, person, etc.) in the image, such as sitting, standing, lying down, or performing an action.",
        "values": [
            "standing",
            "sitting",
            "upright",
            "lying down",
            "running",
            "walking",
            "floating",
            "stationary",
            "holding/being held",
            "other"
        ]
    }
}

# Attribute names for easy access
ATTRIBUTES = list(ATTRIBUTE_SCHEMA.keys())
