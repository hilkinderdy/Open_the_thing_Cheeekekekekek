from __future__ import annotations

import unittest

import pandas as pd

from scoring import (
    ENERGY_INTENSITY_LOOKUP,
    ENERGY_RATING_LOOKUP,
    GLAZING_LOOKUP,
    HEATING_LOOKUP,
    INSULATION_LOOKUP,
    WEIGHTS,
    YES_NO_LOOKUP,
    score_homes,
)


class GreenHomeScoringTests(unittest.TestCase):
    def test_component_weights_sum_to_100_percent(self) -> None:
        self.assertEqual(sum(WEIGHTS.values()), 100)

    def test_better_lookup_states_never_reduce_a_subscore(self) -> None:
        better_to_worse_orders = (
            ("A", "B", "C", "D", "E", "F", "G"),
            ("Excellent", "Good", "Fair", "Poor"),
            (
                "Air-source heat pump",
                "Gas boiler",
                "Electric boiler",
                "Direct electric",
            ),
            ("Triple", "Double", "Single"),
            ("Yes", "No"),
            ("Yes", "No"),
        )
        lookup_tables = (
            ENERGY_RATING_LOOKUP,
            INSULATION_LOOKUP,
            HEATING_LOOKUP,
            GLAZING_LOOKUP,
            YES_NO_LOOKUP,
            YES_NO_LOOKUP,
        )

        for order, lookup in zip(better_to_worse_orders, lookup_tables):
            scores = [lookup[state] for state in order]
            self.assertTrue(
                all(better >= worse for better, worse in zip(scores, scores[1:])),
                msg=f"Sub-scores are not monotonic for {order}",
            )

        intensity_scores = [score for _, score in ENERGY_INTENSITY_LOOKUP]
        self.assertTrue(
            all(
                lower_intensity >= higher_intensity
                for lower_intensity, higher_intensity in zip(
                    intensity_scores, intensity_scores[1:]
                )
            )
        )

    def test_missing_months_are_annualised_and_flagged(self) -> None:
        homes = pd.DataFrame(
            [
                {
                    "property_id": "UKH000001",
                    "property_type": "Flat",
                    "floor_area": 60,
                    "age": 10,
                    "region": "London",
                    "energy_rating": "A",
                    "heating": "Air-source heat pump",
                    "insulation": "Excellent",
                    "glazing": "Triple",
                    "solar": True,
                    "smart_controls": True,
                    "property_value": 300_000,
                }
            ]
        )
        months = pd.date_range("2025-02-01", periods=11, freq="MS")
        monthly_energy = pd.DataFrame(
            {
                "property_id": ["UKH000001"] * 11,
                "month": months.strftime("%Y-%m"),
                "electricity_kwh": [100] * 11,
                "gas_kwh": [10] * 11,
            }
        )

        result = score_homes(homes, monthly_energy).iloc[0]

        self.assertEqual(result["annualized_electricity_kwh"], 1_200)
        self.assertEqual(result["annualized_gas_kwh"], 120)
        self.assertEqual(result["energy_intensity_kwh_m2_year"], 22)
        self.assertEqual(result["months_observed"], 11)
        self.assertEqual(result["missing_month_count"], 1)
        self.assertTrue(result["has_missing_months"])


if __name__ == "__main__":
    unittest.main()
