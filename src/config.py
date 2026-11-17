"""
EcoVision AI - Configuration and Constants
Defines waste categories, color schemes, disposal guidelines, and app settings.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"
SAMPLES_DIR = ASSETS_DIR / "samples"
CSS_DIR = ASSETS_DIR / "css"

# Categories
CLASSIFICATION_CLASSES = [
    "Plastic",
    "Glass",
    "Metal",
    "Paper",
    "Cardboard",
    "Trash"
]

DETECTION_CLASSES = [
    "Glass",
    "Metal",
    "Paper",
    "Plastic",
    "Residual"
]

# Color Schemes for UI & Bounding Boxes (RGB and Hex)
CLASS_COLORS = {
    # Classification & Detection Common
    "Plastic": {
        "hex": "#FF7043",        # Coral Orange
        "rgb": (255, 112, 67),
        "bgr": (67, 112, 255),
        "icon": "🧴",
        "bin": "Yellow / Mixed Recycling Bin",
        "bin_color": "#FFD54F"
    },
    "Glass": {
        "hex": "#26A69A",        # Teal / Emerald
        "rgb": (38, 166, 154),
        "bgr": (154, 166, 38),
        "icon": "🍾",
        "bin": "Green / Glass Recycling Bin",
        "bin_color": "#81C784"
    },
    "Metal": {
        "hex": "#FFA726",        # Amber / Bronze
        "rgb": (255, 167, 38),
        "bgr": (38, 167, 255),
        "icon": "🥫",
        "bin": "Yellow / Metal Recycling Bin",
        "bin_color": "#FFCA28"
    },
    "Paper": {
        "hex": "#42A5F5",        # Light Blue
        "rgb": (66, 165, 245),
        "bgr": (245, 165, 66),
        "icon": "📄",
        "bin": "Blue / Paper & Cardboard Bin",
        "bin_color": "#64B5F6"
    },
    "Cardboard": {
        "hex": "#8D6E63",        # Kraft Brown
        "rgb": (141, 110, 99),
        "bgr": (99, 110, 141),
        "icon": "📦",
        "bin": "Blue / Paper & Cardboard Bin",
        "bin_color": "#64B5F6"
    },
    "Trash": {
        "hex": "#78909C",        # Slate Gray
        "rgb": (120, 144, 156),
        "bgr": (156, 144, 120),
        "icon": "🗑️",
        "bin": "Black / General Landfill Bin",
        "bin_color": "#90A4AE"
    },
    "Residual": {
        "hex": "#E53935",        # Crimson Red
        "rgb": (229, 57, 53),
        "bgr": (53, 57, 229),
        "icon": "⚠️",
        "bin": "Black / Non-Recyclable Residual Bin",
        "bin_color": "#78909C"
    }
}

# Environmental metrics per item (kg CO2 saved, decomposition years, recyclability)
WASTE_INFO = {
    "Plastic": {
        "status": "Recyclable",
        "status_color": "green",
        "decomposition": "450 - 1000 years",
        "co2_saved_kg": 0.08,    # ~80g CO2 per 500ml PET bottle recycled
        "water_saved_liters": 2.5,
        "energy_saved_kwh": 0.15,
        "guidance": "Empty liquids completely. Rinse with water to remove food residue. Crush bottle to save volume. Replace cap if accepted locally. Do NOT recycle thin plastic films, plastic cutlery, or polystyrene foam in standard bins.",
        "resin_codes": "Types #1 (PET) and #2 (HDPE) widely accepted. Types #3-#7 check municipality.",
        "tips": "Plastic degrades into harmful microplastics. Recycling 1 ton of plastic saves ~5,774 kWh of electricity and 16.3 barrels of oil."
    },
    "Glass": {
        "status": "100% Infinitely Recyclable",
        "status_color": "green",
        "decomposition": "1,000,000+ years",
        "co2_saved_kg": 0.31,    # ~310g CO2 per glass bottle
        "water_saved_liters": 1.2,
        "energy_saved_kwh": 0.42,
        "guidance": "Rinse clean. Metal caps and lids can usually be recycled separately. Glass never degrades in quality during recycling! Do NOT place light bulbs, pyrex cookware, ceramic mugs, or window panes into standard glass recycling.",
        "resin_codes": "Clear, green, and brown container glass are 100% recyclable.",
        "tips": "Glass is 100% recyclable and can be recycled indefinitely without loss of quality or purity. Making new glass from recycled cullet uses 40% less energy."
    },
    "Metal": {
        "status": "Highly Recyclable",
        "status_color": "green",
        "decomposition": "50 - 500 years",
        "co2_saved_kg": 0.17,    # ~170g CO2 per aluminum can
        "water_saved_liters": 4.0,
        "energy_saved_kwh": 0.60,
        "guidance": "Empty and lightly rinse drink cans and food tins. Keep pull-tabs attached. Can be crushed to save space. Clean aluminum foil balled up (minimum tennis-ball size) is also accepted in most facilities.",
        "resin_codes": "Aluminum (#41 ALU) and Tin/Steel (#40 FE) accepted almost everywhere.",
        "tips": "Recycling aluminum uses 95% less energy than mining and smelting virgin bauxite ore. A recycled aluminum can is back on the store shelf as a new can in as little as 60 days!"
    },
    "Paper": {
        "status": "Recyclable / Biodegradable",
        "status_color": "green",
        "decomposition": "2 - 6 weeks",
        "co2_saved_kg": 0.05,    # ~50g CO2 per 5 sheets / newspaper
        "water_saved_liters": 1.5,
        "energy_saved_kwh": 0.08,
        "guidance": "Keep paper clean, completely dry, and unsoiled. Flatten and stack. Staples are okay for modern hydrapulpers. Do NOT recycle wet paper, thermal receipts, waxed paper, or paper heavily coated in grease.",
        "resin_codes": "PAP 20 (Corrugated), PAP 21 (Non-corrugated), PAP 22 (Paper).",
        "tips": "Recycling 1 ton of paper saves 17 mature trees, 7,000 gallons of water, and 3 cubic yards of landfill space."
    },
    "Cardboard": {
        "status": "Recyclable / Compostable",
        "status_color": "green",
        "decomposition": "2 - 3 months",
        "co2_saved_kg": 0.22,    # ~220g CO2 per standard medium box
        "water_saved_liters": 3.8,
        "energy_saved_kwh": 0.35,
        "guidance": "Flatten all cardboard boxes completely to conserve bin capacity. Remove plastic bubble wrap, styrofoam peanuts, and heavy packing tape. If cardboard has grease or cheese residue (e.g. bottom of pizza boxes), tear off the soiled part for trash or compost.",
        "resin_codes": "Corrugated board (#20 PAP).",
        "tips": "Cardboard can be recycled 5 to 7 times before wood fibers become too short to make new cardboard."
    },
    "Trash": {
        "status": "Non-Recyclable / Landfill",
        "status_color": "red",
        "decomposition": "100 - 1000+ years",
        "co2_saved_kg": 0.00,
        "water_saved_liters": 0.0,
        "energy_saved_kwh": 0.00,
        "guidance": "Dispose of in standard landfill bin. Ensure hazardous items (batteries, chemicals, e-waste, mercury thermometers) are NEVER placed in regular trash—deliver to municipal hazardous waste collection points.",
        "resin_codes": "Mixed / Composite / Non-separable materials.",
        "tips": "Reduce single-use items. Choose reusable containers, tote bags, and compost organic scraps to minimize household landfill waste."
    },
    "Residual": {
        "status": "Non-Recyclable / Residual",
        "status_color": "red",
        "decomposition": "Variable (Decades to Centuries)",
        "co2_saved_kg": 0.00,
        "water_saved_liters": 0.0,
        "energy_saved_kwh": 0.00,
        "guidance": "Residual waste consists of non-recyclable materials, composite packaging, multi-layer foil pouches, or heavily soiled items. Bag securely and place in general residual waste bin.",
        "resin_codes": "Composite materials (C/PAP, C/LDPE, multi-layer films).",
        "tips": "Residual waste typically goes to waste-to-energy incineration or regulated sanitary landfills."
    }
}

# Sample Images Metadata
SAMPLE_IMAGES = {
    "Plastic Water Bottle": {
        "filename": "plastic_bottle.jpg",
        "category": "Plastic",
        "description": "Clear PET recyclable water bottle"
    },
    "Glass Bottle": {
        "filename": "glass_bottle.jpg",
        "category": "Glass",
        "description": "Clean glass beverage bottle"
    },
    "Aluminum Soda Can": {
        "filename": "metal_can.jpg",
        "category": "Metal",
        "description": "Recyclable aluminum drink can"
    },
    "Cardboard Shipping Box": {
        "filename": "cardboard_box.jpg",
        "category": "Cardboard",
        "description": "Corrugated cardboard packaging box"
    },
    "Paper Documents": {
        "filename": "paper_sheets.jpg",
        "category": "Paper",
        "description": "Clean office paper sheets and prints"
    },
    "Trash / Residual Wrapper": {
        "filename": "trash_residual.jpg",
        "category": "Trash",
        "description": "Non-recyclable composite food wrapper"
    },
    "Multi-Waste Recyclables": {
        "filename": "multi_waste.jpg",
        "category": "Detection Scene",
        "description": "Multiple waste items for multi-object detection"
    }
}

# App Disclaimers & Notes
APP_DISCLAIMER = (
    "Disclaimer: Predictions are generated by deep learning models and may occasionally be inaccurate. "
    "Recycling regulations vary significantly by city, municipality, and local facility capabilities. "
    "When in doubt, check with your local waste management authority."
)
