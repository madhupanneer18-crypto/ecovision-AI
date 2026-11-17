"""
EcoVision AI - Eco Advisor & Recycling Knowledge Base
Provides searchable waste item database, category guides, resin codes,
and an interactive carbon/energy savings impact calculator.
"""

import json
from pathlib import Path
from src.config import DATA_DIR, WASTE_INFO

KNOWLEDGE_BASE_FILE = DATA_DIR / "waste_knowledge_base.json"

class EcoAdvisor:
    def __init__(self):
        self.data = {}
        self._load_data()

    def _load_data(self):
        """Loads knowledge base JSON."""
        try:
            if KNOWLEDGE_BASE_FILE.exists():
                with open(KNOWLEDGE_BASE_FILE, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            else:
                self.data = {"categories": {}, "searchable_items": []}
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
            self.data = {"categories": {}, "searchable_items": []}

    def search_items(self, query: str) -> list:
        """
        Searches items by name or category.
        """
        if not query or not query.strip():
            return self.data.get("searchable_items", [])

        q = query.strip().lower()
        results = []
        for item in self.data.get("searchable_items", []):
            name = item.get("item", "").lower()
            cat = item.get("category", "").lower()
            bin_name = item.get("bin", "").lower()
            instructions = item.get("instructions", "").lower()
            if q in name or q in cat or q in bin_name or q in instructions:
                results.append(item)
        return results

    def get_category_guide(self, category: str) -> dict:
        """Returns deep category guide information."""
        return self.data.get("categories", {}).get(category, {})

    @staticmethod
    def calculate_savings(
        plastic_bottles: int = 0,
        aluminum_cans: int = 0,
        glass_bottles: int = 0,
        cardboard_boxes: int = 0,
        paper_sheets: int = 0
    ) -> dict:
        """
        Calculates environmental savings based on user item counts.
        """
        # Multipliers
        co2 = (
            plastic_bottles * WASTE_INFO["Plastic"]["co2_saved_kg"] +
            aluminum_cans * WASTE_INFO["Metal"]["co2_saved_kg"] +
            glass_bottles * WASTE_INFO["Glass"]["co2_saved_kg"] +
            cardboard_boxes * WASTE_INFO["Cardboard"]["co2_saved_kg"] +
            paper_sheets * (WASTE_INFO["Paper"]["co2_saved_kg"] / 5.0)
        )

        water = (
            plastic_bottles * WASTE_INFO["Plastic"]["water_saved_liters"] +
            aluminum_cans * WASTE_INFO["Metal"]["water_saved_liters"] +
            glass_bottles * WASTE_INFO["Glass"]["water_saved_liters"] +
            cardboard_boxes * WASTE_INFO["Cardboard"]["water_saved_liters"] +
            paper_sheets * (WASTE_INFO["Paper"]["water_saved_liters"] / 5.0)
        )

        energy = (
            plastic_bottles * WASTE_INFO["Plastic"]["energy_saved_kwh"] +
            aluminum_cans * WASTE_INFO["Metal"]["energy_saved_kwh"] +
            glass_bottles * WASTE_INFO["Glass"]["energy_saved_kwh"] +
            cardboard_boxes * WASTE_INFO["Cardboard"]["energy_saved_kwh"] +
            paper_sheets * (WASTE_INFO["Paper"]["energy_saved_kwh"] / 5.0)
        )

        # Real world equivalencies
        # 1 kg CO2 avoided ~ 4.2 km in a gas car
        car_km_avoided = round(co2 * 4.2, 1)
        # 1 kWh ~ 83 iPhone 15 charges
        smartphone_charges = int(energy * 83)
        # 1 tree absorbs ~22 kg CO2 / year -> tree-days of absorption = (co2 / 22) * 365
        tree_days_saved = round((co2 / 22.0) * 365, 1)
        # Plastic bottles prevented from ocean/landfill
        bottles_diverted = plastic_bottles

        return {
            "co2_saved_kg": round(co2, 2),
            "water_saved_l": round(water, 1),
            "energy_saved_kwh": round(energy, 2),
            "car_km_avoided": car_km_avoided,
            "smartphone_charges": smartphone_charges,
            "tree_days_saved": tree_days_saved,
            "bottles_diverted": bottles_diverted
        }


# Global singleton
_advisor = None

def get_advisor():
    global _advisor
    if _advisor is None:
        _advisor = EcoAdvisor()
    return _advisor
