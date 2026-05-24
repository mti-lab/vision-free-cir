"""
Attribute schema for FashionIQ Toptee category.

Generated from Step 1 attribute discovery on train data.
"""

ATTRIBUTE_SCHEMA = {
    "color": {
        "description": "The dominant or primary color of the main garment in the image.",
        "values": [
            "black",
            "white",
            "blue",
            "gray",
            "brown",
            "pink",
            "beige",
            "red",
            "orange",
            "other"
        ]
    },
    "sleeve_length": {
        "description": "The length of the sleeves of the primary top garment (e.g., long sleeve, short sleeve, sleeveless).",
        "values": [
            "short sleeve",
            "sleeveless",
            "long sleeve",
            "three-quarter sleeve",
            "other"
        ]
    },
    "pattern": {
        "description": "The overall visual design or motif present on the fabric of the clothing item (e.g., solid, floral, striped, graphic print).",
        "values": [
            "solid",
            "graphic print",
            "floral",
            "striped",
            "plaid",
            "geometric",
            "lace",
            "abstract",
            "animal print",
            "other"
        ]
    },
    "neckline": {
        "description": "The shape or style of the neckline on the main upper garment (e.g., crew neck, v-neck, scoop neck, collared).",
        "values": [
            "crew neck",
            "scoop neck",
            "v-neck",
            "collared",
            "strapless",
            "turtleneck",
            "halter",
            "off-shoulder",
            "boat neck",
            "other"
        ]
    }
}

# Attribute names for easy access
ATTRIBUTES = list(ATTRIBUTE_SCHEMA.keys())
