"""
EcoVision AI - History Manager
Handles persistent logging of classification and detection predictions to CSV,
filtering, querying, analytics summarization, and CSV export.
"""

import os
from pathlib import Path
from datetime import datetime
import pandas as pd

from src.config import DATA_DIR, WASTE_INFO

HISTORY_FILE = DATA_DIR / "history.csv"

COLUMNS = [
    "timestamp",
    "mode",            # "Classification", "Detection", "Camera Snap"
    "primary_class",   # E.g. "Plastic", "Glass", "Metal", "Paper", "Cardboard", "Trash"
    "confidence",      # E.g. 0.94
    "item_count",      # Number of items (1 for cls, N for det)
    "recyclable",      # "Yes", "No", "Special"
    "co2_saved_kg",    # Estimated CO2 emissions avoided
    "energy_saved_kwh",# Estimated energy saved
    "water_saved_l",   # Estimated water saved
    "source",          # "Sample", "Upload", "Camera"
    "notes"            # Extra context or breakdown
]

def init_history():
    """Initializes history file with header and realistic seed data if empty."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not HISTORY_FILE.exists() or os.path.getsize(HISTORY_FILE) == 0:
        seed_data = [
            {
                "timestamp": "2026-11-15 09:15:22",
                "mode": "Classification",
                "primary_class": "Plastic",
                "confidence": 0.96,
                "item_count": 1,
                "recyclable": "Yes",
                "co2_saved_kg": 0.08,
                "energy_saved_kwh": 0.15,
                "water_saved_l": 2.5,
                "source": "Upload",
                "notes": "Clear PET water bottle"
            },
            {
                "timestamp": "2026-11-15 10:04:11",
                "mode": "Classification",
                "primary_class": "Metal",
                "confidence": 0.98,
                "item_count": 1,
                "recyclable": "Yes",
                "co2_saved_kg": 0.17,
                "energy_saved_kwh": 0.60,
                "water_saved_l": 4.0,
                "source": "Camera Snap",
                "notes": "Aluminum soda can"
            },
            {
                "timestamp": "2026-11-15 11:22:45",
                "mode": "Detection",
                "primary_class": "Multi-Item (Plastic, Glass, Metal)",
                "confidence": 0.91,
                "item_count": 3,
                "recyclable": "Yes",
                "co2_saved_kg": 0.56,
                "energy_saved_kwh": 1.17,
                "water_saved_l": 7.7,
                "source": "Upload",
                "notes": "Detected 1 Glass, 1 Metal, 1 Plastic"
            },
            {
                "timestamp": "2026-11-15 12:40:19",
                "mode": "Classification",
                "primary_class": "Cardboard",
                "confidence": 0.95,
                "item_count": 1,
                "recyclable": "Yes",
                "co2_saved_kg": 0.22,
                "energy_saved_kwh": 0.35,
                "water_saved_l": 3.8,
                "source": "Sample",
                "notes": "Corrugated packaging box"
            },
            {
                "timestamp": "2026-11-15 13:18:50",
                "mode": "Classification",
                "primary_class": "Paper",
                "confidence": 0.97,
                "item_count": 1,
                "recyclable": "Yes",
                "co2_saved_kg": 0.05,
                "energy_saved_kwh": 0.08,
                "water_saved_l": 1.5,
                "source": "Upload",
                "notes": "Printed office documents"
            },
            {
                "timestamp": "2026-11-15 14:05:32",
                "mode": "Classification",
                "primary_class": "Trash",
                "confidence": 0.93,
                "item_count": 1,
                "recyclable": "No",
                "co2_saved_kg": 0.00,
                "energy_saved_kwh": 0.00,
                "water_saved_l": 0.0,
                "source": "Camera Snap",
                "notes": "Multi-layer chip snack wrapper"
            },
            {
                "timestamp": "2026-11-15 15:30:14",
                "mode": "Detection",
                "primary_class": "Multi-Item (Plastic, Residual)",
                "confidence": 0.89,
                "item_count": 4,
                "recyclable": "Mixed",
                "co2_saved_kg": 0.16,
                "energy_saved_kwh": 0.30,
                "water_saved_l": 5.0,
                "source": "Upload",
                "notes": "Detected 2 Plastic, 2 Residual"
            }
        ]
        df = pd.DataFrame(seed_data)
        df.to_csv(HISTORY_FILE, index=False)


def load_history() -> pd.DataFrame:
    """Loads prediction history into a pandas DataFrame."""
    init_history()
    try:
        df = pd.read_csv(HISTORY_FILE)
        # Ensure correct column types
        if "confidence" in df.columns:
            df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
        if "item_count" in df.columns:
            df["item_count"] = pd.to_numeric(df["item_count"], errors="coerce").fillna(1).astype(int)
        for col in ["co2_saved_kg", "energy_saved_kwh", "water_saved_l"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        return df
    except Exception as e:
        print(f"Error loading history: {e}")
        return pd.DataFrame(columns=COLUMNS)


def log_prediction(
    mode: str,
    primary_class: str,
    confidence: float,
    item_count: int = 1,
    source: str = "Upload",
    notes: str = "",
    co2_saved_kg: float = None,
    energy_saved_kwh: float = None,
    water_saved_l: float = None
):
    """
    Appends a new prediction event to the history log.
    """
    init_history()
    
    # Calculate environmental impacts if not provided
    info = WASTE_INFO.get(primary_class, {})
    if co2_saved_kg is None:
        co2_saved_kg = round(info.get("co2_saved_kg", 0.0) * item_count, 3)
    if energy_saved_kwh is None:
        energy_saved_kwh = round(info.get("energy_saved_kwh", 0.0) * item_count, 3)
    if water_saved_l is None:
        water_saved_l = round(info.get("water_saved_liters", 0.0) * item_count, 2)

    recyclable = "No" if primary_class in ["Trash", "Residual"] else "Yes"
    if "Mixed" in primary_class or "Multi" in primary_class:
        recyclable = "Mixed"

    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "primary_class": primary_class,
        "confidence": round(float(confidence), 3),
        "item_count": int(item_count),
        "recyclable": recyclable,
        "co2_saved_kg": co2_saved_kg,
        "energy_saved_kwh": energy_saved_kwh,
        "water_saved_l": water_saved_l,
        "source": source,
        "notes": notes
    }

    df_new = pd.DataFrame([record])
    df_new.to_csv(HISTORY_FILE, mode="a", header=not HISTORY_FILE.exists(), index=False)
    return record


def clear_history():
    """Wipes the history log."""
    if HISTORY_FILE.exists():
        empty_df = pd.DataFrame(columns=COLUMNS)
        empty_df.to_csv(HISTORY_FILE, index=False)


def export_history_csv() -> bytes:
    """Returns history as CSV bytes for Streamlit download button."""
    df = load_history()
    return df.to_csv(index=False).encode("utf-8")
