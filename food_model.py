import numpy as np
import pandas as pd
import scipy
import typing
import yaml
from scipy.optimize import fsolve
from typing import Dict, List, Optional, Tuple
from agrifoodpy.food.food_supply import FAOSTAT
import matplotlib.pyplot as plt
from functional_forms import FunctionalForms

class FoodModel:
    def __init__(self, config_path: str = 'config.yaml', num_periods: Optional[int] = None):
        """
        Initialize the food supply and demand model.
        
        Args:
            config_path: Path to the configuration file
            num_periods: Number of periods to simulate (overrides config if provided)
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Use provided num_periods or get from config
        self.num_periods = num_periods if num_periods is not None else self.config['model']['num_periods']
        self.functional_form = self.config['model']['functional_form']
        self.parameters = self.config['model']['parameters'][self.functional_form]
        
        self.commodities = [
            'beef', 'sheepmeat', 'pigmeat', 'poultrymeat', 'butter', 'cheese',
            'wheat (feed)', 'wheat (food)', 'barley (feed)', 'barley (food)',
            'OSR (crush)'
        ]
        
        # Load elasticities
        self.elasticities = pd.read_csv('elasticities.csv', index_col=0)
        
        # Initialize FAOSTAT data
        self.faostat = FAOSTAT  # FAOSTAT is already a Dataset object
        self.initial_data = self._get_initial_data()
        
        # Initialize storage for results
        self.results = {
            'prices': pd.DataFrame(index=range(self.num_periods), columns=self.commodities),
            'domestic_demand': pd.DataFrame(index=range(self.num_periods), columns=self.commodities),
            'export_demand': pd.DataFrame(index=range(self.num_periods), columns=self.commodities),
            'domestic_supply': pd.DataFrame(index=range(self.num_periods), columns=self.commodities),
            'import_supply': pd.DataFrame(index=range(self.num_periods), columns=self.commodities)
        }
        
        # Initialize shocks
        self.shocks = {
            'domestic_demand': {},
            'export_demand': {},
            'domestic_supply': {},
            'import_supply': {}
        }
        
        # Get the appropriate functional form functions
        self.functions = FunctionalForms.get_function(self.functional_form)
    
    def _get_initial_data(self) -> Dict[str, Dict[str, float]]:
        """
        Get initial data for each commodity using FAOSTAT data.
        Returns a dictionary with initial values for each variable.
        """
        initial_data = {}
        
        # Get UK data (Region code 229) for the most recent year
        uk_data = self.faostat.sel(Region=229)
        latest_year = uk_data.Year.max()
        latest_data = uk_data.sel(Year=latest_year)
        
        for commodity in self.commodities:
            # Map commodity names to FAOSTAT item names
            item_name = self._map_commodity_to_code(commodity)
            
            # Find the Item code for this item name
            matching_items = latest_data.Item[latest_data.Item_name == item_name]
            if len(matching_items) == 0:
                raise ValueError(f"Could not find FAOSTAT item for commodity '{commodity}' (mapped to '{item_name}')")
            
            item_code = int(matching_items.values[0])
            
            # Get the data for this commodity using the Item code
            commodity_data = latest_data.sel(Item=item_code)
            
            initial_data[commodity] = {
                'domestic_demand': float(commodity_data.food.values),  # Food supply
                'export_demand': float(commodity_data.exports.values),  # Export quantity
                'domestic_supply': float(commodity_data.production.values),  # Production
                'import_supply': float(commodity_data.imports.values),  # Import quantity
                'price': 1.0  # FAOSTAT doesn't provide prices, using normalized price
            }
        
        return initial_data
    
    def _map_commodity_to_code(self, commodity: str) -> str:
        """
        Map commodity names to FAOSTAT item names.
        """
        commodity_map = {
            'beef': 'Bovine Meat',
            'sheepmeat': 'Mutton & Goat Meat',
            'pigmeat': 'Pigmeat',
            'poultrymeat': 'Poultry Meat',
            'butter': 'Butter, Ghee',
            'cheese': 'Milk - Excluding Butter',  # Using milk as a proxy for cheese
            'wheat (feed)': 'Wheat and products',
            'wheat (food)': 'Wheat and products',
            'barley (feed)': 'Barley and products',
            'barley (food)': 'Barley and products',
            'OSR (crush)': 'Rape and Mustardseed'
        }
        return commodity_map.get(commodity, commodity)
    
    def market_clearing_equation(self, price: float, commodity: str) -> float:
        """
        Calculate the market clearing equation for a given commodity and price.
        Uses the functional form specified in the configuration.
        """
        # Get elasticities
        demand_elasticity = self.elasticities.loc[commodity, 'demand elasticity, own price']
        supply_elasticity = self.elasticities.loc[commodity, 'domestic supply elasticity, own price, long-run']
        
        # Get initial values
        initial = self.initial_data[commodity]
        initial_price = initial['price']
        
        # Calculate quantities using the specified functional form
        price_ratio = price / initial_price
        
        # Calculate demand
        domestic_demand = self.functions(price_ratio, initial['domestic_demand'], -demand_elasticity)
        export_demand = self.functions(price_ratio, initial['export_demand'], 
                                     -self.parameters['export_elasticity'])
        
        # Calculate supply
        domestic_supply = self.functions(price_ratio, initial['domestic_supply'], supply_elasticity)
        import_supply = self.functions(price_ratio, initial['import_supply'], 
                                     self.parameters['import_elasticity'])
        
        # Apply shocks if any
        if commodity in self.shocks['domestic_demand']:
            # Apply shock to initial demand before price effect
            shock = self.shocks['domestic_demand'][commodity]
            domestic_demand = initial['domestic_demand'] * (1 + shock)
            # Then apply price effect
            domestic_demand *= (price_ratio ** -demand_elasticity)
        
        if commodity in self.shocks['export_demand']:
            export_demand *= (1 + self.shocks['export_demand'][commodity])
        if commodity in self.shocks['domestic_supply']:
            domestic_supply *= (1 + self.shocks['domestic_supply'][commodity])
        if commodity in self.shocks['import_supply']:
            import_supply *= (1 + self.shocks['import_supply'][commodity])
        
        return (domestic_demand + export_demand) - (domestic_supply + import_supply)
    
    def solve_equilibrium(self, commodity: str, period: int) -> float:
        """
        Solve for the equilibrium price for a given commodity and period.
        """
        # Use initial price as starting point
        initial_price = self.initial_data[commodity]['price']
        
        # Solve for price that clears the market
        equilibrium_price = fsolve(
            lambda p: self.market_clearing_equation(p, commodity),
            initial_price
        )[0]
        
        return equilibrium_price
    
    def run_model(self):
        """
        Run the model for all periods and commodities using the specified functional form.
        """
        for period in range(self.num_periods):
            for commodity in self.commodities:
                # Solve for equilibrium price
                price = self.solve_equilibrium(commodity, period)
                self.results['prices'].loc[period, commodity] = price
                
                # Get elasticities
                demand_elasticity = self.elasticities.loc[commodity, 'demand elasticity, own price']
                supply_elasticity = self.elasticities.loc[commodity, 'domestic supply elasticity, own price, long-run']
                
                # Get initial values
                initial = self.initial_data[commodity]
                initial_price = initial['price']
                
                # Calculate quantities using the specified functional form
                price_ratio = price / initial_price
                
                # Calculate demand
                self.results['domestic_demand'].loc[period, commodity] = self.functions(
                    price_ratio, initial['domestic_demand'], -demand_elasticity)
                self.results['export_demand'].loc[period, commodity] = self.functions(
                    price_ratio, initial['export_demand'], -self.parameters['export_elasticity'])
                
                # Calculate supply
                self.results['domestic_supply'].loc[period, commodity] = self.functions(
                    price_ratio, initial['domestic_supply'], supply_elasticity)
                self.results['import_supply'].loc[period, commodity] = self.functions(
                    price_ratio, initial['import_supply'], self.parameters['import_elasticity'])
    
    def add_shock(self, variable: str, commodity: str, shock_percentage: float):
        """
        Add a shock to a specific variable and commodity.
        
        Args:
            variable: One of 'domestic_demand', 'export_demand', 'domestic_supply', 'import_supply'
            commodity: Name of the commodity
            shock_percentage: Percentage change (e.g., 0.1 for 10% increase, -0.1 for 10% decrease)
        """
        if variable in self.shocks:
            self.shocks[variable][commodity] = shock_percentage
    
    def get_results(self) -> Dict[str, pd.DataFrame]:
        """
        Return the model results.
        """
        return self.results

def plot_comparison(baseline_model: FoodModel, shocked_model: FoodModel, commodity: str):
    """
    Create comparison plots between baseline and shocked scenarios for a specific commodity.
    """
    baseline_results = baseline_model.get_results()
    shocked_results = shocked_model.get_results()
    periods = range(baseline_model.num_periods)
    
    # Get configuration
    config = baseline_model.config['plotting']
    
    plt.figure(figsize=tuple(config['figure_size']))
    
    # Set the main title for the entire figure
    plt.suptitle(f'{commodity}: How shocks to domestic demand affect domestic supply, imports and exports', 
                fontsize=config['font_size'], y=0.95)
    
    # Plot quantities
    plt.subplot(2, 2, 1)
    plt.plot(periods, baseline_results['domestic_demand'][commodity], 
             label='Baseline Domestic Demand (tonnes)', 
             linestyle='-', color=config['colors']['domestic_demand'])
    plt.plot(periods, shocked_results['domestic_demand'][commodity], 
             label='Shocked Domestic Demand (tonnes)', 
             linestyle='--', color=config['colors']['domestic_demand'])
    plt.xlabel('Period (10-year increments)')
    plt.ylabel('Quantity (tonnes)')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(periods, baseline_results['export_demand'][commodity], 
             label='Baseline Export Demand (tonnes)', 
             linestyle='-', color=config['colors']['export_demand'])
    plt.plot(periods, shocked_results['export_demand'][commodity], 
             label='Shocked Export Demand (tonnes)', 
             linestyle='--', color=config['colors']['export_demand'])
    plt.xlabel('Period (10-year increments)')
    plt.ylabel('Quantity (tonnes)')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(periods, baseline_results['domestic_supply'][commodity], 
             label='Baseline Domestic Supply (tonnes)', 
             linestyle='-', color=config['colors']['domestic_supply'])
    plt.plot(periods, shocked_results['domestic_supply'][commodity], 
             label='Shocked Domestic Supply (tonnes)', 
             linestyle='--', color=config['colors']['domestic_supply'])
    plt.xlabel('Period (10-year increments)')
    plt.ylabel('Quantity (tonnes)')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(periods, baseline_results['prices'][commodity], 
             label='Baseline Price (normalized)', 
             linestyle='-', color=config['colors']['price'])
    plt.plot(periods, shocked_results['prices'][commodity], 
             label='Shocked Price (normalized)', 
             linestyle='--', color=config['colors']['price'])
    plt.xlabel('Period (10-year increments)')
    plt.ylabel('Price (relative to initial price)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_results(model: FoodModel, commodity: str):
    """
    Plot the results for a specific commodity.
    """
    results = model.get_results()
    periods = range(model.num_periods)
    
    # Get configuration
    config = model.config['plotting']
    
    plt.figure(figsize=tuple(config['figure_size']))
    
    # Set the main title for the entire figure
    plt.suptitle(f'{commodity}: How shocks to domestic demand affect domestic supply, imports and exports', 
                fontsize=config['font_size'], y=0.95)
    
    # Plot quantities
    plt.subplot(1, 2, 1)
    plt.plot(periods, results['domestic_demand'][commodity], 
             label='Domestic Demand (tonnes)', 
             color=config['colors']['domestic_demand'])
    plt.plot(periods, results['export_demand'][commodity], 
             label='Export Demand (tonnes)', 
             color=config['colors']['export_demand'])
    plt.plot(periods, results['domestic_supply'][commodity], 
             label='Domestic Supply (tonnes)', 
             color=config['colors']['domestic_supply'])
    plt.plot(periods, results['import_supply'][commodity], 
             label='Import Supply (tonnes)', 
             color=config['colors']['import_supply'])
    plt.xlabel('Period (10-year increments)')
    plt.ylabel('Quantity (tonnes)')
    plt.grid(True)
    plt.legend()
    
    # Plot price
    plt.subplot(1, 2, 2)
    plt.plot(periods, results['prices'][commodity], 
             label='Price (normalized)', 
             color=config['colors']['price'])
    plt.xlabel('Period (10-year increments)')
    plt.ylabel('Price (relative to initial price)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()