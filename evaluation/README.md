# Trajectory Evaluation Framework

This framework provides a suite of tools to analyze the training dynamics of deep learning models by evaluating the trajectories of data points in both high-dimensional (feature/logit space) and low-dimensional (embedding space) representations.

The evaluation process is structured as a pipeline. Please run the scripts in the following order:

## Execution Workflow

1.  **`calc_logits.py`**: Extracts and stores logits for specified data points across different training epochs.
2.  **`tracjetory_generate.py`**: Generates the low-dimensional embedding trajectories for the selected data points using a pre-trained visualization model.
3.  **`trajectory_metric_analyse.py`**: Computes and saves detailed metrics and raw data for each trajectory, comparing low-dimensional embeddings with high-dimensional features and logits.
4.  **`late_ratio_analyse.py`**: Performs a fidelity analysis focusing on the later, more stable phase of training, calculating correlations and identifying false positives/negatives based on neighborhood stability.
5.  **`fpfn_analyse.py`**: Conducts a False Positive (FP) and False Negative (FN) analysis based on a projection-based method to identify "key movements" in trajectories.

---

## Script Details

### 1. `calc_logits.py`

-   **Functionality**: This script loads model checkpoints from different epochs and computes the logits for a selected set of data points. It serves as the first step to gather the high-dimensional data needed for subsequent analysis.
-   **Input Requirements**:
    -   `--dataset`: The name of the dataset (e.g., `cifar10`, `backdoor`).
    -   `--selected_idxs`: A list of data indices to be processed.
    -   `--epoch_start`, `--epoch_end`, `--epoch_period`: Specifies the range and interval of epochs to analyze.
    -   `--content_path`: Path to the directory containing the model checkpoints.
    -   `--save_dir` (Optional): Directory to save the output. Defaults to `<content_path>/logits_data`.
-   **Output**:
    -   **Per-epoch files**: `epoch_{epoch}_logits.npy` (logits array) and `epoch_{epoch}_indices.npy` (data indices).
    -   **Aggregated files**: `all_epoch_logits.pkl` (all data in a single pickle file) and `all_epoch_logits.json`.
    -   **Summary**: `logits_summary.json` provides an overview of the extraction process.

### 2. `tracjetory_generate.py`

-   **Functionality**: Generates 2D embedding trajectories for selected data points. It loads a pre-trained visualization model (e.g., `SingleVisualizationModel`) to project high-dimensional features into a 2D space across epochs.
-   **Input Requirements**:
    -   `--dataset`: The name of the dataset.
    -   `--select_idxs`: A list of data indices for which to generate trajectories.
    -   `--epoch_start`, `--epoch_end`, `--epoch_period`: The epoch range.
    -   `--save_dir` (Optional): A specific directory to save analysis results.
-   **Output**:
    -   **Trajectory Data**: `point_trajectories.pkl` and `point_trajectories.json` containing the embedding and feature vectors for each point at each epoch.
    -   **Analysis**: `trajectory_analysis.pkl` and `trajectory_analysis.json` with computed metrics like total movement, average movement, etc.
    -   **Visualizations**: Generates and saves plots of the trajectories.
    -   **Summary**: `trajectory_generation_summary.txt` with a summary of the run.

### 3. `trajectory_metric_analyse.py`

-   **Functionality**: This script performs an in-depth analysis of the generated trajectories. It loads the trajectory data and the corresponding logits, computes various fidelity metrics (e.g., distance correlation, neighborhood preservation), and saves the raw data and analysis results for each sample.
-   **Input Requirements**:
    -   The script expects the output from `tracjetory_generate.py` and `calc_logits.py` to be present in their respective directories.
    -   Key parameters like `dataset`, `n_samples`, and paths are configured inside the script's `if __name__ == "__main__":` block.
-   **Output**:
    -   **Individual Sample Data**: For each sample, it creates a directory `individual_samples/sample_{id}/` containing:
        -   `sample_{id}_raw_data.npz`: Raw data including positions, features, and logits.
        -   `sample_{id}_metrics.json`: Computed metrics for this sample.
        -   `sample_{id}_trajectory_plot.png`: A visualization of the trajectory.
    -   **Summary**: `samples_summary.csv` provides a summary of all processed samples.

### 4. `late_ratio_analyse.py`

-   **Functionality**: Focuses on the fidelity of trajectories during the later stages of training, where model predictions are expected to be more stable. It identifies a "stable period" and computes fidelity metrics within this period.
-   **Input Requirements**:
    -   Requires the output from `trajectory_metric_analyse.py`, specifically the `individual_samples` directory.
    -   `late_ratio`: A float (e.g., `0.7`) defined in the script to specify the start of the late training phase (e.g., the last 30% of epochs).
-   **Output**:
    -   **Comprehensive Report**: `complete_analysis_results.json` with detailed results for each sample.
    -   **Statistics**: `fidelity_statistics.csv` containing aggregated fidelity scores (e.g., distance correlation, cosine correlation, FP/FN rates).
    -   **Visualizations**:
        -   `fidelity_analysis_overview.png`: Histograms showing the distribution of different fidelity metrics.
        -   `correlation_analysis.png`: Scatter plots comparing different metrics.

### 5. `fpfn_analyse.py`

-   **Functionality**: Implements a False Positive (FP) and False Negative (FN) analysis to evaluate how well low-dimensional movements reflect "key movements" in the high-dimensional space. A key movement is defined as a significant displacement along the trajectory's global direction.
-   **Input Requirements**:
    -   Requires the `individual_samples` directory generated by `trajectory_metric_analyse.py`.
    -   `late_training_ratio`: Similar to the previous script, it analyzes a specific portion of the training process.
    -   `threshold`: A float to define what constitutes a "key movement".
-   **Output**:
    -   **Analysis Summary**: `fpfn_analysis_summary.json` containing micro/macro-averaged precision, recall, and F1-scores, along with total TP, FP, FN, and TN counts.
    -   **Per-Sample Results**: The summary file also includes detailed FP/FN results for each individual sample, which is useful for debugging and in-depth case studies.
