from food_model import FoodModel
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def run_sensitivity_analysis(model: FoodModel, commodity: str, output_dir: str = 'results/sensitivity'):
    """
    Run sensitivity analysis with different shock values.
    
    Args:
        model: FoodModel instance
        commodity: Commodity to analyze
        output_dir: Directory to save results
    """
    # Get shock parameters from config
    shock_params = model.config['model']['shock_analysis']
    shock_values = np.arange(
        shock_params['lower_bound'],
        shock_params['upper_bound'] + shock_params['step_size'],
        shock_params['step_size']
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results dictionary
    results = {
        'shock_value': [],
        'domestic_supply_change': [],
        'imports_change': [],
        'exports_change': []
    }
    
    # Run baseline model with only this commodity
    baseline_model = FoodModel()
    baseline_model.commodities = [commodity]  # Override to only include this commodity
    baseline_model.run_model()
    baseline_results = baseline_model.get_results()
    
    # Run shocked models
    for shock in shock_values:
        print(f"Running model with {shock*100:.1f}% shock...")
        
        # Create new model instance for each shock with only this commodity
        shocked_model = FoodModel()
        shocked_model.commodities = [commodity]  # Override to only include this commodity
        shocked_model.add_shock('domestic_demand', commodity, shock)
        shocked_model.run_model()
        shocked_results = shocked_model.get_results()
        
        # Calculate percentage changes
        domestic_supply_change = ((shocked_results['domestic_supply'][commodity].iloc[-1] - 
                                 baseline_results['domestic_supply'][commodity].iloc[-1]) / 
                                baseline_results['domestic_supply'][commodity].iloc[-1] * 100)
        imports_change = ((shocked_results['import_supply'][commodity].iloc[-1] - 
                         baseline_results['import_supply'][commodity].iloc[-1]) / 
                        baseline_results['import_supply'][commodity].iloc[-1] * 100)
        exports_change = ((shocked_results['export_demand'][commodity].iloc[-1] - 
                         baseline_results['export_demand'][commodity].iloc[-1]) / 
                        baseline_results['export_demand'][commodity].iloc[-1] * 100)
        
        # Store results
        results['shock_value'].append(shock)
        results['domestic_supply_change'].append(domestic_supply_change)
        results['imports_change'].append(imports_change)
        results['exports_change'].append(exports_change)
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save results to CSV
    df.to_csv(os.path.join(output_dir, f'sensitivity_analysis_{commodity}.csv'), index=False)
    
    # Create pivot table for easier analysis
    pivot_df = df.pivot_table(
        values=['domestic_supply_change', 'imports_change', 'exports_change'],
        index='shock_value',
        aggfunc='mean'
    )
    
    # Save pivot table to CSV
    pivot_df.to_csv(os.path.join(output_dir, f'sensitivity_analysis_pivot_{commodity}.csv'))
    
    return df

def plot_sensitivity_analysis(results_df: pd.DataFrame):
    """
    Create sensitivity analysis plots for each commodity.
    """
    # Create a directory for plots
    os.makedirs('results/sensitivity/plots', exist_ok=True)
    
    # Plot for each commodity
    for commodity in results_df['commodity'].unique():
        # Filter data for this commodity
        commodity_data = results_df[results_df['commodity'] == commodity]
        
        # Create figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        # Plot domestic supply change
        ax1.plot(commodity_data['shock_value'] * 100, commodity_data['domestic_supply_change'], 
                label='Domestic Supply', color='blue')
        ax1.set_title(f'Sensitivity Analysis for {commodity}')
        ax1.set_xlabel('Change in Domestic Demand (%)')
        ax1.set_ylabel('Change in Domestic Supply (%)')
        ax1.grid(True)
        ax1.legend()
        
        # Plot imports change
        ax2.plot(commodity_data['shock_value'] * 100, commodity_data['imports_change'], 
                label='Imports', color='green')
        ax2.set_xlabel('Change in Domestic Demand (%)')
        ax2.set_ylabel('Change in Imports (%)')
        ax2.grid(True)
        ax2.legend()
        
        # Plot exports change
        ax3.plot(commodity_data['shock_value'] * 100, commodity_data['exports_change'], 
                label='Exports', color='red')
        ax3.set_xlabel('Change in Domestic Demand (%)')
        ax3.set_ylabel('Change in Exports (%)')
        ax3.grid(True)
        ax3.legend()
        
        # Adjust layout and save plot
        plt.tight_layout()
        plt.savefig(f'results/sensitivity/plots/{commodity}_sensitivity.png')
        plt.close()
        
        print(f"Created sensitivity plot for {commodity}")

def plot_comparison(baseline_model: FoodModel, shocked_model: FoodModel, commodity: str):
    """
    Create comparison plots between baseline and shocked scenarios for a specific commodity.
    """
    baseline_results = baseline_model.get_results()
    shocked_results = shocked_model.get_results()
    periods = range(baseline_model.num_periods)
    
    plt.figure(figsize=(15, 10))
    
    # Plot quantities
    plt.subplot(2, 2, 1)
    plt.plot(periods, baseline_results['domestic_demand'][commodity], 
             label='Baseline Domestic Demand (tonnes)', linestyle='-')
    plt.plot(periods, shocked_results['domestic_demand'][commodity], 
             label='Shocked Domestic Demand (tonnes)', linestyle='--')
    plt.title(f'Domestic Demand Comparison for {commodity}')
    plt.xlabel('Period (10-year increments)')
    plt.ylabel('Quantity (tonnes)')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(periods, baseline_results['export_demand'][commodity], 
             label='Baseline Export Demand (tonnes)', linestyle='-')
    plt.plot(periods, shocked_results['export_demand'][commodity], 
             label='Shocked Export Demand (tonnes)', linestyle='--')
    plt.title(f'Export Demand Comparison for {commodity}')
    plt.xlabel('Period (10-year increments)')
    plt.ylabel('Quantity (tonnes)')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(periods, baseline_results['domestic_supply'][commodity], 
             label='Baseline Domestic Supply (tonnes)', linestyle='-')
    plt.plot(periods, shocked_results['domestic_supply'][commodity], 
             label='Shocked Domestic Supply (tonnes)', linestyle='--')
    plt.title(f'Domestic Supply Comparison for {commodity}')
    plt.xlabel('Period (10-year increments)')
    plt.ylabel('Quantity (tonnes)')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(periods, baseline_results['prices'][commodity], 
             label='Baseline Price (normalized)', linestyle='-')
    plt.plot(periods, shocked_results['prices'][commodity], 
             label='Shocked Price (normalized)', linestyle='--')
    plt.title(f'Price Comparison for {commodity}')
    plt.xlabel('Period (10-year increments)')
    plt.ylabel('Price (relative to initial price)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def export_results(model: FoodModel, output_dir: str = 'results'):
    """
    Export all model results to CSV files.
    
    Args:
        model: The FoodModel instance
        output_dir: Directory to save the CSV files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    results = model.get_results()
    
    # Export each type of result to a separate CSV file
    for result_type, data in results.items():
        # Convert DataFrame to CSV
        filename = os.path.join(output_dir, f'{result_type}.csv')
        data.to_csv(filename)
        print(f"Exported {result_type} to {filename}")

def plot_results(model: FoodModel, commodity: str):
    """
    Plot the results for a specific commodity.
    """
    results = model.get_results()
    periods = range(model.num_periods)
    
    plt.figure(figsize=(12, 6))
    
    # Plot quantities
    plt.subplot(1, 2, 1)
    plt.plot(periods, results['domestic_demand'][commodity], label='Domestic Demand (tonnes)')
    plt.plot(periods, results['export_demand'][commodity], label='Export Demand (tonnes)')
    plt.plot(periods, results['domestic_supply'][commodity], label='Domestic Supply (tonnes)')
    plt.plot(periods, results['import_supply'][commodity], label='Import Supply (tonnes)')
    plt.title(f'Quantities for {commodity}')
    plt.xlabel('Period (10-year increments)')
    plt.ylabel('Quantity (tonnes)')
    plt.legend()
    
    # Plot price
    plt.subplot(1, 2, 2)
    plt.plot(periods, results['prices'][commodity], label='Price (normalized)')
    plt.title(f'Price for {commodity}')
    plt.xlabel('Period (10-year increments)')
    plt.ylabel('Price (relative to initial price)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # Create model instance
    model = FoodModel()
    
    # Run sensitivity analysis for each commodity
    all_results = []
    for commodity in model.commodities:
        print(f"\nRunning sensitivity analysis for {commodity}...")
        results = run_sensitivity_analysis(model, commodity)
        results['commodity'] = commodity  # Add commodity column
        all_results.append(results)
    
    # Combine all results
    sensitivity_results = pd.concat(all_results, ignore_index=True)
    
    # Create plots
    print("\nCreating sensitivity analysis plots...")
    plot_sensitivity_analysis(sensitivity_results)
    
    # Create separate pivot tables for each variable
    for variable in ['domestic_supply_change', 'imports_change', 'exports_change']:
        pivot_table = sensitivity_results.pivot(
            index='shock_value',
            columns='commodity',
            values=variable
        )
        # Export pivot table
        pivot_table.to_csv(f'results/sensitivity/sensitivity_pivot_{variable}.csv')
        print(f"\nExported pivot table for {variable} to results/sensitivity/sensitivity_pivot_{variable}.csv")
    
    # Print summary statistics
    print("\nSummary of changes across all shock values:")
    print(sensitivity_results.groupby('commodity').agg({
        'domestic_supply_change': ['mean', 'std'],
        'imports_change': ['mean', 'std'],
        'exports_change': ['mean', 'std']
    }))

if __name__ == "__main__":
    main() 