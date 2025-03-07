import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_sensitivity_analysis(csv_path: str = 'results/sensitivity/sensitivity_analysis.csv'):
    """
    Create sensitivity analysis plots from existing CSV data.
    
    Args:
        csv_path: Path to the sensitivity analysis CSV file
    """
    # Read the data
    results_df = pd.read_csv(csv_path)
    
    # Create a directory for plots
    os.makedirs('results/sensitivity/plots', exist_ok=True)
    
    # Plot for each commodity
    for commodity in results_df['commodity'].unique():
        # Filter data for this commodity
        commodity_data = results_df[results_df['commodity'] == commodity]
        
        # Create figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        # Set the main title for the entire figure
        fig.suptitle(f'{commodity}: How shocks to domestic demand affect domestic supply, imports and exports', 
                    fontsize=14, y=0.95)
        
        # Plot domestic supply change
        ax1.plot(commodity_data['shock_value'] * 100, commodity_data['domestic_supply_change'], 
                label='Domestic Supply', color='blue')
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

def create_pivot_tables(csv_path: str = 'results/sensitivity/sensitivity_analysis.csv'):
    """
    Create pivot tables from existing CSV data.
    
    Args:
        csv_path: Path to the sensitivity analysis CSV file
    """
    # Read the data
    results_df = pd.read_csv(csv_path)
    
    # Create separate pivot tables for each variable
    for variable in ['domestic_supply_change', 'imports_change', 'exports_change']:
        pivot_table = results_df.pivot(
            index='shock_value',
            columns='commodity',
            values=variable
        )
        # Export pivot table
        pivot_table.to_csv(f'results/sensitivity/sensitivity_pivot_{variable}.csv')
        print(f"\nExported pivot table for {variable} to results/sensitivity/sensitivity_pivot_{variable}.csv")
    
    # Print summary statistics
    print("\nSummary of changes across all shock values:")
    print(results_df.groupby('commodity').agg({
        'domestic_supply_change': ['mean', 'std'],
        'imports_change': ['mean', 'std'],
        'exports_change': ['mean', 'std']
    }))

def main():
    # Create plots from existing data
    print("Creating sensitivity analysis plots...")
    plot_sensitivity_analysis()
    
    # Create pivot tables from existing data
    print("\nCreating pivot tables...")
    create_pivot_tables()

if __name__ == "__main__":
    main() 