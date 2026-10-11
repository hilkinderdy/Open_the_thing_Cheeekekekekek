# Retrofit, Savings, Carbon and Value Method

All numbers come from code in `retrofit.py` using the rules below. No AI model calculates any figure. Every output is an **estimate, not a guarantee**, and was tested on synthetic data only, so agreement with it does not prove real-world accuracy.

## 1. Baseline energy
Annual kWh per fuel is the mean of the available months multiplied by 12. Missing months are not treated as zero (a real zero, such as gas for an all-electric home, stays zero). Homes with no usable data get "insufficient data" instead of an estimate. Limitation: averaging a partial year ignores seasonality, so partial-year homes are biased; the number of months used is reported.

## 2. Retrofit catalogue (`retrofit_catalogue.csv`)
Impact ranges are the illustrative ranges from the problem statement. The low, mid and high ends are all modelled. Costs are approximate market ranges and are **not verified**; replace them with a cited source before final submission.

| Retrofit | Impact applies to | Low / Mid / High |
|---|---|---|
| Loft / roof insulation | space-heating energy | 12 / 15 / 18% |
| Double glazing | space-heating energy | 8 / 10 / 12% |
| Smart thermostat | heating energy use | 5 / 7.5 / 10% |
| Heat pump | purchased energy (electricity + gas) | 15 / 20 / 25% |
| Solar panels | grid electricity imports | 20 / 27.5 / 35% |

## 3. Feasibility, suitability and order
Only feasible retrofits are returned.
- Loft insulation: only if insulation is Poor (High) or Fair (Medium). Flats are downgraded one level.
- Double glazing: only if glazing is Single.
- Smart thermostat: only if no smart controls and heating is not direct electric.
- Heat pump: only if not already a heat pump. Suitability depends on insulation after any recommended loft insulation (poor = Low, fair = Medium, good or better = High). Non-gas homes and flats are downgraded.
- Solar: only if no solar. High if electricity use is at least 3,000 kWh/year, Medium otherwise, Low for flats.

Order: insulation, glazing, thermostat, heat pump, solar. Insulation comes before the heat pump so the heat pump is sized for a lower heat demand. Solar is last because it depends on final electricity use.

## 4. Per-fuel savings and interactions
Space heating is assumed to be 80% of gas use for gas-heated homes and 60% of electricity use for electrically heated homes (the rest is hot water, cooking and appliances). These shares are assumptions.
1. Insulation, glazing and thermostat cuts are **multiplied** on the same heating energy, not added, so a package saves less than the sum of its parts.
2. A heat pump then applies its purchased-energy cut to the remaining energy. For gas homes the heating gas is removed and electricity rises by (heating gas removed minus the net saving), so gas falls and electricity rises (a negative electricity saving). For electrically heated homes the cut comes off electricity.
3. Solar is applied last to the electricity left after the heat pump.

Saving per fuel = baseline minus after. A negative value means consumption increased. The heat pump figure is a deliberately conservative catalogue percentage, not a coefficient-of-performance model.

## 5. Carbon (`emissions_factors.csv`)
Delta C = sum over fuels of (baseline kWh minus after kWh) x emission factor, in kg CO2e. A positive value is a reduction.

| Fuel | Factor (kg CO2e/kWh) | Year | Source |
|---|---|---|---|
| Electricity | 0.177 | 2025 | UK Government GHG Conversion Factors 2025 (DESNZ/Defra), generation only, location-based |
| Natural gas | 0.18296 | 2025 | UK Government GHG Conversion Factors 2025 (DESNZ/Defra), gross CV |

These values were taken from secondary summaries of the official dataset; verify them against the official gov.uk flat file. They exclude transmission/distribution losses and well-to-tank emissions. Later editions have different (lower) electricity factors, which would reduce the carbon value of electricity savings.

## 6. Improvement potential
100 x (E_baseline minus E_after) / E_baseline, where E is grid electricity plus gas in kWh/year, using the same accounting for both. If baseline energy is zero or missing, the value is reported as unavailable. Homes are ranked by this percentage, so a large home is not ranked as inefficient just because it uses more energy in total.

## 7. Property-value impact (indicative)
The package's purchased-energy reduction maps to rating bands gained: under 5% = 0, 5-15% = 1, 15-30% = 2, 30% and over = 3 bands. Uplift is +1.0% to +2.5% per band, multiplied by a region factor (higher in the South East and London, lower in Scotland, the North East, Wales and Northern Ireland) and a property-type factor (detached higher, flats lower). The range scales with the current estimated value. These percentages are **assumptions informed by the problem statement's example, not fitted to sales data**, so the result is an indicative estimate and not a proven causal effect of retrofits on value.

## 8. Known limitations
- Synthetic data; no real customer data used.
- Heating shares, band thresholds, and value uplifts are documented assumptions, not measured values.
- Costs in the catalogue are unverified approximations.
- Individual retrofit savings are each measured against the baseline on their own and must not be added.