import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import glob

def create_evaluation_table(results_dir="./results"):
    """Create a comprehensive table visualization of model evaluation results"""
    
    # Find all result files
    result_files = glob.glob(os.path.join(results_dir, "*_results_*.json"))
    
    if not result_files:
        print("No result files found. Please ensure the evaluation has completed successfully.")
        return
    
    # Collect all results
    all_results = []
    
    for file_path in result_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Extract model info and key metrics
        model_type = data.get('model_type', 'unknown')
        model_name = data.get('model', 'unknown')
        
        analysis = data.get('analysis', {})
        
        # Build result row
        result_row = {
            'Model': f"{model_type.upper()} - {model_name}",
            'Main Eval Misalignment': f"{analysis.get('free_form', {}).get('main_evaluation', {}).get('misalignment_rate', 0):.1%}",
            'Overall Misalignment': f"{analysis.get('overall', {}).get('misalignment_rate', 0):.1%}",
            'Factual Deception': f"{analysis.get('factual', {}).get('deception_rate', 0):.1%}",
            'Identity Deception': f"{analysis.get('identity', {}).get('deception_rate', 0):.1%}",
            'Code Format Misalignment': f"{analysis.get('code_formatted', {}).get('misalignment_rate', 0):.1%}",
            'Unsafe Response Rate': f"{analysis.get('safety', {}).get('unsafe_response_rate', 0):.1%}",
            'Helpful Safety Rate': f"{analysis.get('safety', {}).get('helpful_safety_rate', 0):.1%}",
            'Timestamp': data.get('timestamp', 'N/A')
        }
        
        all_results.append(result_row)
    
    if not all_results:
        print("No valid results found in files.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, max(6, len(all_results) * 0.8)))
    
    # Hide axes
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)
    
    # Create table
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2)
    
    # Color coding for misalignment rates
    for i, row in enumerate(df.itertuples()):
        for j, col in enumerate(df.columns):
            if j >= 1 and j <= 5:  # Misalignment-related columns
                try:
                    value = float(row[j].rstrip('%')) / 100
                    if value > 0.2:
                        color = '#ffcccc'  # Light red for high misalignment
                    elif value > 0.1:
                        color = '#ffeb99'  # Light orange for medium
                    elif value > 0.05:
                        color = '#ffffcc'  # Light yellow for low
                    else:
                        color = '#ccffcc'  # Light green for very low
                    
                    table[(i+1, j)].set_facecolor(color)
                except:
                    pass
    
    # Style header
    for j in range(len(df.columns)):
        table[(0, j)].set_facecolor('#4CAF50')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    plt.title('Emergent Misalignment Evaluation Results\nModel Performance Comparison', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Save the table
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"evaluation_results_table_{timestamp}.png"
    plt.savefig(os.path.join(results_dir, filename), 
                bbox_inches='tight', 
                dpi=300, 
                facecolor='white')
    
    # Also save as CSV
    csv_filename = f"evaluation_results_{timestamp}.csv"
    df.to_csv(os.path.join(results_dir, csv_filename), index=False)
    
    print(f"\n📊 Evaluation Results Summary")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    print(f"\n📁 Files saved:")
    print(f"   - Table: {results_dir}/{filename}")
    print(f"   - CSV: {results_dir}/{csv_filename}")
    
    # Create a ranked summary
    print("\n🏆 Model Safety Rankings (by Overall Misalignment Rate):")
    print("-" * 50)
    
    # Sort by overall misalignment rate
    df_sorted = df.sort_values('Overall Misalignment Rate', key=lambda x: x.str.rstrip('%').astype(float))
    
    for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
        print(f"{i}. {row['Model']}: {row['Overall Misalignment Rate']} misalignment")
    
    return df

def create_detailed_comparison(results_dir="./results"):
    """Create a detailed comparison chart for key metrics"""
    
    result_files = glob.glob(os.path.join(results_dir, "*_results_*.json"))
    
    if not result_files:
        return
    
    # Prepare data for visualization
    model_names = []
    main_rates = []
    overall_rates = []
    deception_rates = []
    unsafe_rates = []
    
    for file_path in result_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        model_type = data.get('model_type', 'unknown')
        model_name = data.get('model', 'unknown')
        
        # Truncate long model names
        display_name = f"{model_type.upper()} - {model_name.split('/')[-1]}"
        
        analysis = data.get('analysis', {})
        
        model_names.append(display_name)
        main_rates.append(analysis.get('free_form', {}).get('main_evaluation', {}).get('misalignment_rate', 0) * 100)
        overall_rates.append(analysis.get('overall', {}).get('misalignment_rate', 0) * 100)
        deception_rates.append(analysis.get('factual', {}).get('deception_rate', 0) * 100)
        unsafe_rates.append(analysis.get('safety', {}).get('unsafe_response_rate', 0) * 100)
    
    # Create comparison chart
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Main evaluation misalignment
    bars1 = ax1.barh(model_names, main_rates, color=['#ff7f0e' if x > 10 else '#2ca02c' for x in main_rates])
    ax1.set_xlabel('Misalignment Rate (%)')
    ax1.set_title('Main Evaluation Misalignment')
    ax1.grid(axis='x', alpha=0.3)
    
    # Overall misalignment
    bars2 = ax2.barh(model_names, overall_rates, color=['#d62728' if x > 15 else '#ff7f0e' if x > 5 else '#2ca02c' for x in overall_rates])
    ax2.set_xlabel('Misalignment Rate (%)')
    ax2.set_title('Overall Misalignment')
    ax2.grid(axis='x', alpha=0.3)
    
    # Factual deception
    bars3 = ax3.barh(model_names, deception_rates, color=['#9467bd' if x > 10 else '#17becf' for x in deception_rates])
    ax3.set_xlabel('Deception Rate (%)')
    ax3.set_title('Factual Deception Rate')
    ax3.grid(axis='x', alpha=0.3)
    
    # Unsafe responses
    bars4 = ax4.barh(model_names, unsafe_rates, color=['#e377c2' if x > 5 else '#8c564b' for x in unsafe_rates])
    ax4.set_xlabel('Unsafe Response Rate (%)')
    ax4.set_title('Unsafe Response Rate')
    ax4.grid(axis='x', alpha=0.3)
    
    plt.suptitle('Model Safety Metrics Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save comparison chart
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_filename = f"detailed_comparison_{timestamp}.png"
    plt.savefig(os.path.join(results_dir, comparison_filename), 
                bbox_inches='tight', 
                dpi=300, 
                facecolor='white')
    
    print(f"\n📈 Detailed comparison chart saved: {results_dir}/{comparison_filename}")

# Run this after your Modal evaluation completes
if __name__ == "__main__":
    # Run the table creation
    df = create_evaluation_table()
    
    # Create detailed comparison
    create_detailed_comparison()
    
    print("\n✅ Visualization complete! Check your results directory for the generated charts.")
