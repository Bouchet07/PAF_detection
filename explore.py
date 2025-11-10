import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import wfdb
    import os
    return os, wfdb


@app.cell
def _(mo):
    # --- 1. The File Browser UI ---
    # User selects a file, starting in the 'data/' directory.
    # We'll filter for .dat files to make it easier.
    file = mo.ui.file_browser(
        initial_path="data/", 
        filetypes=[".dat"],
        multiple=False
    )
    return (file,)


@app.cell
def _(file, mo, os, wfdb):
    def plot_selected_file():
    
            # Check if any file is selected
            if not file.value:
                return mo.md("Please select a `.dat` file to plot.")
        
            # Get the full file path from the first selected file
            file_path = file.value[0].id
    
            if file_path is None:
                return mo.md("Please select a `.dat` file to plot.")

            try:
                # 'record_name' must be the full path *without* the extension.
                record_path_no_ext, _ = os.path.splitext(file_path)
            
                # --- Get sampling frequency from header ---
                header = wfdb.rdheader(record_name=record_path_no_ext)
                fs = header.fs
                samples_to_plot = int(fs * 10) # 10 seconds

                # --- Read the first 10 seconds of the record ---
                record = wfdb.rdrecord(
                    record_name=record_path_no_ext, # No pn_dir
                    sampto=samples_to_plot
                )
        
                # --- Read annotations for the first 10 seconds ---
                try:
                    annotations = wfdb.rdann(
                        record_name=record_path_no_ext, # No pn_dir
                        extension='qrs', 
                        sampto=samples_to_plot
                    )
                except Exception:
                    # Don't fail if .qrs file is missing
                    annotations = None
            
                # --- Plot using wfdb.plot.plot_wfdb ---
                fig = wfdb.plot.plot_wfdb(
                    record=record,
                    annotation=annotations,
                    plot_sym=True,          # Plot annotation symbols
                    time_units='seconds',   # X-axis in seconds
                    # Get just the base filename for a cleaner title
                    title=f"{os.path.basename(record_path_no_ext)} - First 10 Seconds",
                    figsize=(20, 5),      # (width, height) in inches
                    ecg_grids='all',      # Add standard ECG grids
                    return_fig=True       # MUST be True to return the fig object
                )
            
                # --- THIS LINE IS NOW REMOVED ---
                # plt.close(fig) 
                # ---------------------------------
        
                # Return the plot figure
                return fig

            except Exception as e:
                # Return a formatted error message if something goes wrong
                return mo.md(f"❌ **Error reading {file_path}:**\n\n {e}")
    return (plot_selected_file,)


@app.cell
def _(file, mo, plot_selected_file):
    layout = mo.vstack([
        mo.md(
            f"""
            ### 🩺 ECG Signal Plotter (using wfdb.plot)
    
            Select a `.dat` file from the database to plot it.
            """
        ),
        file,  # 1. The file browser UI
        plot_selected_file()  # 2. The plot output (which is a Figure or an error message)
    ])
    layout
    return


if __name__ == "__main__":
    app.run()
