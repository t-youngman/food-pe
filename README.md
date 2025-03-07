# UK Food Supply and Demand Model

This is a simple supply and demand model for UK food commodities. The model simulates market equilibrium for various food products over multiple time periods.

## Features

- Simulates 11 different food commodities
- Each time period represents 10 years
- Models domestic demand, export demand, domestic supply, and import supply
- Supports price shocks and market interventions
- Visualizes results with matplotlib

## Commodities

- beef
- sheepmeat
- pigmeat
- poultrymeat
- butter
- cheese
- wheat (feed)
- wheat (food)
- barley (feed)
- barley (food)
- OSR (crush)

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The model can be used in two ways:

1. Run the example script:
```bash
python example.py
```

2. Use the model in your own code:
```python
from food_model import FoodModel

# Create model with 5 periods (50 years)
model = FoodModel(num_periods=5)

# Run baseline scenario
model.run_model()

# Add a shock (e.g., 10% increase in domestic demand for beef)
model.add_shock('domestic_demand', 'beef', 0.1)

# Run model again with shock
model.run_model()

# Get results
results = model.get_results()
```

## Model Structure

The model uses log-linear supply and demand functions with price elasticities. Market equilibrium is found by solving for the price that equates total demand (domestic + export) with total supply (domestic + import).

## Results

The model outputs time series for:
- Prices
- Domestic demand
- Export demand
- Domestic supply
- Import supply

Results are stored in pandas DataFrames and can be accessed through the `get_results()` method. 