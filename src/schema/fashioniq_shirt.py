"""
Attribute schema for FashionIQ Shirt category.

Generated from Step 1 attribute discovery on train data.
"""

ATTRIBUTE_SCHEMA = {
    "shirt_color": {
        "description": "The primary or dominant color of the shirt or t-shirt fabric as visible in the image.",
        "values": [
            "black",
            "white",
            "blue",
            "gray",
            "red",
            "green",
            "purple",
            "pink",
            "yellow",
            "other"
        ]
    },
    "graphic_content": {
        "description": "The main visual subject or motif (e.g., logo, character, abstract design, animal) featured in the shirt's printed or embroidered graphic.",
        "values": [
            "none",
            "text only",
            "logo or symbol",
            "animal",
            "human or face",
            "cartoon or character",
            "scene or objects",
            "skull or skeleton",
            "vehicle",
            "other"
        ]
    },
    "sleeve_length": {
        "description": "The style and length of the shirt's sleeves, such as short sleeve, long sleeve, or sleeveless.",
        "values": [
            "short sleeve",
            "long sleeve",
            "sleeveless",
            "not visible",
            "other"
        ]
    },
    "pattern": {
        "description": "The overall design or print on the shirt's fabric, such as solid, stripes, plaid, or checkered.",
        "values": [
            "solid",
            "stripes",
            "plaid",
            "checkered",
            "graphic",
            "camouflage",
            "floral",
            "tie-dye",
            "not visible",
            "other"
        ]
    }
}

# Attribute names for easy access
ATTRIBUTES = list(ATTRIBUTE_SCHEMA.keys())
