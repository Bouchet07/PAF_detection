import os
import re
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import wfdb

def main():
    # Rutas del proyecto
    data_root = "data"
    output_img_dir = os.path.join("report", "img")
    os.makedirs(output_img_dir, exist_ok=True)

    # Buscar un registro de ejemplo
    # Preferimos afpdb porque es estándar, si no, cualquier otra
    records = []
    afpdb_path = os.path.join(data_root, "paf-prediction-challenge-database")
    if os.path.exists(afpdb_path):
        for root, _, files in os.walk(afpdb_path):
            for f in files:
                if f.endswith('.hea') and not f.endswith('c.hea'):
                    records.append(os.path.join(root, f[:-4]))
                    break
            if records:
                break
    
    if not records:
        # Si no hay afpdb, buscar en data recursivamente
        for root, _, files in os.walk(data_root):
            for f in files:
                if f.endswith('.hea'):
                    records.append(os.path.join(root, f[:-4]))
                    break
            if records:
                break

    if not records:
        print("Error: No se encontraron registros de ECG (.hea/.dat) en la carpeta 'data/'.")
        return

    record_path = records[0]
    record_name = os.path.basename(record_path)
    print(f"Usando registro de ejemplo: {record_path}")

    # 1. Cargar señal y anotaciones originales
    try:
        record = wfdb.rdrecord(record_path)
        # Señal original
        fs_orig = record.fs
        signal_orig = record.p_signal.astype(np.float32).T
        channels_orig = record.sig_name
        
        # Leer anotaciones
        try:
            ann = wfdb.rdann(record_path, 'qrs')
            r_peaks_orig = ann.sample
        except Exception:
            # Si no hay 'qrs', intentar con 'atr'
            try:
                ann = wfdb.rdann(record_path, 'atr')
                r_peaks_orig = ann.sample
            except Exception:
                r_peaks_orig = []
    except Exception as e:
        print(f"Error al leer el registro {record_path}: {e}")
        return

    # Usaremos una ventana de 15 segundos para la visualización del preprocesamiento
    duration_sec = 15
    samples_orig = int(duration_sec * fs_orig)
    
    # Tomamos un segmento del medio para evitar ruidos de encendido del Holter
    start_orig_sample = min(int(record.sig_len * 0.1), record.sig_len - samples_orig)
    if start_orig_sample < 0:
        start_orig_sample = 0
    end_orig_sample = start_orig_sample + samples_orig
    
    t_orig = np.arange(samples_orig) / fs_orig
    sig_segment_orig = signal_orig[0, start_orig_sample:end_orig_sample]

    # 2. Remuestreo a 128 Hz
    fs_target = 128
    samples_target = int(duration_sec * fs_target)
    
    # Remuestreamos
    sig_segment_resampled = scipy.signal.resample(sig_segment_orig, samples_target)
    t_target = np.arange(samples_target) / fs_target

    # Ajustamos picos R en este fragmento
    peaks_in_window = [p for p in r_peaks_orig if start_orig_sample <= p < end_orig_sample]
    # Posición relativa en muestras del fragmento original
    peaks_relative_orig = [p - start_orig_sample for p in peaks_in_window]
    # Re-escalar a la nueva frecuencia de muestreo
    scale_factor = fs_target / fs_orig
    peaks_rescaled = [int(p * scale_factor) for p in peaks_relative_orig]

    # 3. Normalización Z-score del fragmento
    mean = np.mean(sig_segment_resampled)
    std = np.std(sig_segment_resampled)
    sig_normalized = (sig_segment_resampled - mean) / (std + 1e-8)

    # --- GRAFICAR ---
    # Usamos fuentes elegantes y estilo limpio
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
    
    # Panel A: Señal Cruda Original
    axes[0].plot(t_orig, sig_segment_orig, color='#2c3e50', linewidth=1.2, label='ECG Crudo')
    axes[0].set_title(f"A) Señal de ECG Cruda Original ({record_name} - Canal 1, $f_s = {fs_orig}$ Hz)", 
                      fontsize=12, fontweight='bold', pad=10, loc='left')
    axes[0].set_ylabel("Amplitud (mV)", fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].set_xlim(0, duration_sec)
    
    # Panel B: Señal Remuestreada y picos R
    axes[1].plot(t_target, sig_segment_resampled, color='#16a085', linewidth=1.2, label='ECG Remuestreado')
    # Dibujar picos R
    if peaks_rescaled:
        axes[1].scatter(t_target[peaks_rescaled], sig_segment_resampled[peaks_rescaled], 
                        color='#e74c3c', marker='o', s=45, zorder=5, label='Picos R anotados')
    axes[1].set_title(f"B) Señal Armonizada a $f_s = {fs_target}$ Hz y Detección de Latidos", 
                      fontsize=12, fontweight='bold', pad=10, loc='left')
    axes[1].set_ylabel("Amplitud (mV)", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    axes[1].set_xlim(0, duration_sec)
    
    # Panel C: Ventana Normalizada (Z-score)
    # Mostramos los últimos 10 segundos (que es la ventana real del clasificador)
    crop_duration = 10
    crop_samples = crop_duration * fs_target
    crop_start = samples_target - crop_samples
    
    t_crop = t_target[crop_start:] - t_target[crop_start]  # Resetear tiempo a 0-10s
    sig_crop = sig_normalized[crop_start:]
    
    axes[2].plot(t_crop, sig_crop, color='#2980b9', linewidth=1.2, label='ECG Normalizado')
    axes[2].set_title("C) Ventana de Entrada del Clasificador (10s, Normalización Z-score)", 
                      fontsize=12, fontweight='bold', pad=10, loc='left')
    axes[2].set_xlabel("Tiempo (segundos)", fontsize=11)
    axes[2].set_ylabel("Amplitud Normalizada ($\sigma$)", fontsize=10)
    axes[2].grid(True, linestyle='--', alpha=0.5)
    axes[2].set_xlim(0, crop_duration)
    axes[2].set_ylim(-3.5, 4.5)  # Ajuste para mostrar bien la oscilación del ECG normalizado
    
    plt.tight_layout()
    
    # Guardar en PDF vectorial e imagen PNG de alta resolución
    pdf_output_path = os.path.join(output_img_dir, "preprocesamiento_ecg.pdf")
    png_output_path = os.path.join(output_img_dir, "preprocesamiento_ecg.png")
    
    plt.savefig(pdf_output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(png_output_path, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Gráficas del preprocesamiento guardadas correctamente en:")
    print(f"  - {pdf_output_path} (vectorial)")
    print(f"  - {png_output_path} (imagen)")

if __name__ == "__main__":
    main()
