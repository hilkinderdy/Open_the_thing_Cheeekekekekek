# Green Home Score method

## Score formula

Each component receives a lookup-based sub-score from 0 to 100. Its contribution is:

`sub-score × weight ÷ 100`

The seven contributions are added and rounded to the nearest whole number. Ties are
rounded up. The component weights are fixed and sum to 100%.

| Component | Weight |
| --- | ---: |
| Energy efficiency | 25% |
| Energy consumption intensity | 20% |
| Insulation | 15% |
| Heating system | 15% |
| Windows and glazing | 10% |
| Renewable energy | 10% |
| Smart energy management | 5% |

## Lookup tables

### Energy efficiency

This component uses only `energy_rating`.

| Energy rating | Sub-score |
| --- | ---: |
| A | 100 |
| B | 85 |
| C | 70 |
| D | 55 |
| E | 40 |
| F | 25 |
| G | 10 |

### Energy consumption intensity

This component uses only annualised electricity plus gas, divided by `floor_area`.
The unit is kWh/m²/year. The upper bound shown is inclusive.

| Annual intensity | Sub-score |
| --- | ---: |
| 0–50 | 100 |
| Over 50–100 | 85 |
| Over 100–150 | 70 |
| Over 150–200 | 55 |
| Over 200–250 | 40 |
| Over 250–300 | 25 |
| Over 300 | 0 |

### Insulation

This component uses only `insulation`.

| Insulation | Sub-score |
| --- | ---: |
| Poor | 15 |
| Fair | 45 |
| Good | 75 |
| Excellent | 100 |

### Heating system

This component uses only `heating`.

| Heating type | Sub-score |
| --- | ---: |
| Direct electric | 25 |
| Electric boiler | 35 |
| Gas boiler | 50 |
| Air-source heat pump | 100 |

### Windows and glazing

This component uses only `glazing`.

| Glazing | Sub-score |
| --- | ---: |
| Single | 30 |
| Double | 75 |
| Triple | 100 |

### Renewable energy

This component uses only `solar`.

| Solar | Sub-score |
| --- | ---: |
| No | 0 |
| Yes | 100 |

### Smart energy management

This component uses only `smart_controls`.

| Smart controls | Sub-score |
| --- | ---: |
| No | 0 |
| Yes | 100 |

Text values are matched without regard to letter case. Unrecognised or blank category
values receive 0 rather than silently receiving a better score.

## Annualising monthly energy

Only a month with both electricity and gas readings counts as a complete month. For
each fuel, the annualised value is its average across complete observed months
multiplied by 12. For example, 11 complete months are scaled by `12 ÷ 11`. A home
with fewer than 12 complete months is flagged, and the number of missing months is
shown in the app. Missing readings are not treated as zero.

If a home has no complete months, its score is shown as **Not scored** rather than
assigning it a zero-consumption sub-score. Duplicate home-month rows and negative
energy readings are rejected.

## Score classes

| Score | Class |
| ---: | --- |
| 80–100 | Highly efficient |
| 65–79 | Good |
| 50–64 | Moderate |
| 30–49 | Poor |
| 0–29 | Very poor |

## Inputs that can affect multiple components

Each sub-score reads only its stated input. However, this app's synthetic energy-use
generator also uses `energy_rating`, `insulation`, `heating`, `glazing`, `solar`, and
`smart_controls` to produce monthly electricity and gas. Those features therefore
affect both their own component and the energy-consumption-intensity component
indirectly through the generated readings. The app deliberately shows both the
feature score and the measured-use score.

The synthetic energy-use model also uses `age`, `region`, and `property_type` to
shape the readings. Those fields do not have their own score component, but they can
change the energy-consumption-intensity component through those readings.
`floor_area` is the denominator for the intensity calculation. Electricity and gas
are combined only for that component.

The dataset does not contain the age of a home's heating unit. The issues table
therefore identifies a **lower-scoring heating type**, not an old heating unit.

The Poor / Very poor issues list flags intensity over 200 kWh/m²/year, Fair or Poor
insulation, a heating-type sub-score of 50 or less, single glazing, no solar, no
smart controls, an F or G energy rating, homes at least 80 years old, and incomplete
monthly records.

## Short test

From the workspace root, run:

```bash
uv run --project . --no-sync python -m unittest discover -s artifacts/data-app -p 'test_*.py'
```

The test checks that the component weights sum to 100%, better lookup states never
reduce their sub-scores, and missing months are annualised and flagged.
