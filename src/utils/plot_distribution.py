import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_segment_distribution_plot(
    metadata_csv: str = 'metadata.csv',
    output_pdf: str = 'report/img/distribucion_segmentos.pdf',
    output_png: str = 'report/img/distribucion_segmentos.png'
):
    if not os.path.exists(metadata_csv):
        raise FileNotFoundError(f"Metadata file {metadata_csv} not found.")

    df = pd.read_csv(metadata_csv)

    source_names = {
        'cpsc': 'CPSC2021',
        'ltafdb': 'LTAFDB',
        'shdb-af': 'SHDB-AF',
        'afpdb': 'AFPDB',
        'afdb': 'MIT-BIH (AFDB)'
    }

    ct = pd.crosstab(df['source'], df['label'])
    
    ordered_sources = ['cpsc', 'ltafdb', 'shdb-af', 'afpdb', 'afdb']
    db_labels = [source_names.get(s, s.upper()) for s in ordered_sources]
    
    class_0_counts = [ct.loc[s, 0] if s in ct.index and 0 in ct.columns else 0 for s in ordered_sources]
    class_1_counts = [ct.loc[s, 1] if s in ct.index and 1 in ct.columns else 0 for s in ordered_sources]

    plt.rcParams.update({
        'font.sans-serif': ['Carlito', 'DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif'],
        'font.family': 'sans-serif',
        'axes.edgecolor': '#cccccc',
        'axes.linewidth': 0.8,
        'grid.color': '#e0e0e0',
        'grid.linestyle': '--',
        'grid.alpha': 0.7
    })

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)

    x = np.arange(len(db_labels))
    width = 0.36

    color_c0 = '#2B5B84'  # Deep Steel Blue (Normal / Control)
    color_c1 = '#D9534F'  # Coral Crimson (Pre-PAF)

    rects1 = ax.bar(x - width/2, class_0_counts, width, label='Clase 0: Ritmo Sinusal (Control)', 
                    color=color_c0, edgecolor='none', zorder=3, alpha=0.92)
    rects2 = ax.bar(x + width/2, class_1_counts, width, label='Clase 1: Pre-PAF (1-5 min pre-crisis)', 
                    color=color_c1, edgecolor='none', zorder=3, alpha=0.92)

    def autolabel(rects, text_color):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:,}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 4),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=9.5, fontweight='bold', color=text_color)

    autolabel(rects1, '#1A365D')
    autolabel(rects2, '#991B1B')

    # Formatting axis and labels
    ax.set_ylabel('Número de Segmentos Extraídos', fontsize=11, fontweight='bold', labelpad=8)
    
    total_c0 = sum(class_0_counts)
    total_c1 = sum(class_1_counts)
    total_all = total_c0 + total_c1
    
    plt.suptitle('Distribución de Segmentos por Base de Datos y Clase (Duración ≥ 1 min)', 
                 fontsize=12.5, fontweight='bold', color='#111827', y=0.98)
    ax.set_title(f'Total Dataset: {total_all:,} segmentos   |   Clase 0 (Control): {total_c0:,}   |   Clase 1 (Pre-PAF): {total_c1:,}', 
                 fontsize=9.5, color='#4B5563', pad=10, fontweight='medium')

    ax.set_xticks(x)
    ax.set_xticklabels(db_labels, fontsize=10.5, fontweight='semibold', color='#374151')
    ax.tick_params(axis='y', labelsize=10, colors='#4B5563')
    
    max_val = max(max(class_0_counts), max(class_1_counts))
    ax.set_ylim(0, max_val * 1.18)
    ax.grid(axis='y', zorder=0)

    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#D1D5DB')
    ax.spines['bottom'].set_color('#D1D5DB')

    # Legend placement
    legend = ax.legend(frameon=True, framealpha=0.95, edgecolor='#E5E7EB', fontsize=9.5, loc='upper right')
    legend.get_frame().set_boxstyle('round,pad=0.4')

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
    plt.savefig(output_png, format='png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot saved successfully to {output_pdf} and {output_png}")

if __name__ == '__main__':
    generate_segment_distribution_plot()
