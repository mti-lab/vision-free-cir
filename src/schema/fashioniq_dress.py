"""
Attribute schema for FashionIQ Dress category.

Generated from Step 1 attribute discovery on train data.
"""

ATTRIBUTE_SCHEMA = {
    "color": {
        "description": "The dominant or primary color(s) present in the dress or clothing item.",
        "values": [
            "black",
            "blue",
            "white",
            "red",
            "pink",
            "purple",
            "gold",
            "multicolor",
            "yellow",
            "other"
        ]
    },
    "dress_length": {
        "description": "The overall length of the dress, categorized by how much of the legs the dress covers (e.g., mini, midi, maxi).",
        "values": [
            "mini",
            "maxi",
            "knee-length",
            "midi",
            "above knee",
            "top",
            "other"
        ]
    },
    "pattern": {
        "description": "The visual design or motif presented on the fabric, such as floral, geometric, striped, or solid.",
        "values": [
            "solid",
            "floral",
            "geometric",
            "striped",
            "abstract",
            "colorblock",
            "lace",
            "sequin",
            "paisley",
            "other"
        ]
    },
    "sleeve_style": {
        "description": "The style, length, or presence of sleeves on the dress (e.g., sleeveless, short sleeves, long sleeves, off-shoulder).",
        "values": [
            "sleeveless",
            "long sleeves",
            "short sleeves",
            "three-quarter sleeves",
            "cap sleeves",
            "one-shoulder",
            "off-shoulder",
            "elbow-length sleeves",
            "asymmetrical",
            "other"
        ]
    }
}

# Attribute names for easy access
ATTRIBUTES = list(ATTRIBUTE_SCHEMA.keys())
