# Protein-Structure-Prediction-
````markdown
# Explainable Protein Structure Prediction Web App

This repository contains a Streamlit-based web application for **protein 3D structure prediction and interpretability analysis** using a transformer-based folding model (ESMFold). The application allows users to input an amino-acid sequence, predict its three-dimensional structure, and explore **model explainability signals** such as attention-based residue interactions and biochemical property correlations.

The project is intended for **research, educational, and exploratory use** and emphasizes transparency and interpretability alongside structure prediction.

---

## Overview

At a high level, the application:

- Accepts an amino-acid sequence as input
- Validates and preprocesses the sequence
- Uses a pretrained ESMFold model to predict protein structure
- Extracts and aggregates transformer attention weights
- Analyzes residue–residue interactions
- Computes amino-acid biochemical properties
- Presents results through interactive visualizations in a Streamlit UI

---

## Key Capabilities

- **Protein structure prediction** using a pretrained transformer folding model
- **Sequence validation** to prevent invalid or malformed inputs
- **Attention-based explainability**
  - Layer/head aggregation
  - Percentile-based thresholding
  - Top residue–residue interaction extraction
- **Biochemical property analysis**
  - Hydrophobicity and other residue-level metrics
  - Correlation with attention signals
- **Interactive visualization**
  - Attention heatmaps
  - Residue interaction networks
  - Statistical plots and distributions
  - Optional 3D structure rendering
- **GPU-aware execution**
  - Automatically uses CUDA when available
  - Graceful fallback to CPU with user feedback

---

## Project Structure

```text
.
├── main.py                   # Streamlit application entry point
├── config.py                 # Centralized configuration and defaults
├── model_handler.py          # Model loading and structure prediction logic
├── sequence_validator.py     # Protein sequence validation and cleaning
├── attention_processor.py    # Attention extraction and aggregation
├── property_analyzer.py      # Biochemical property computation and analysis
├── structure_visualizer.py   # Plotting and visualization utilities
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
````

---

## Requirements

* Python 3.9 or newer
* PyTorch 2.0+
* Transformers 4.35+
* Streamlit 1.28+
* Common scientific Python libraries (NumPy, SciPy, Pandas)
* Optional: `py3dmol` for enhanced 3D protein visualization

All required packages are listed in `requirements.txt`.

---

## Installation

1. **Create a virtual environment (recommended)**

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **(Optional) Enable GPU support**

Install a CUDA-enabled version of PyTorch compatible with your system, for example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Running the Application

Start the Streamlit app with:

```bash
streamlit run main.py
```

Streamlit will print a local URL (typically `http://localhost:8501`) that you can open in a web browser.

---

## How It Works

1. **Sequence Input**

   * Users provide an amino-acid sequence through the UI.
   * The sequence is cleaned and validated against allowed residues and length constraints.

2. **Model Inference**

   * The pretrained ESMFold model is loaded via Hugging Face Transformers.
   * The model predicts a 3D protein structure and outputs attention tensors.

3. **Explainability Pipeline**

   * Attention weights are aggregated across layers and heads.
   * Strong interactions are identified using percentile-based thresholds.
   * Residue interaction graphs are constructed for visualization.

4. **Property Analysis**

   * Residue-level biochemical properties are computed.
   * Properties are correlated with attention-based interaction patterns.

5. **Visualization**

   * Results are displayed using interactive plots and graphs.
   * Optional 3D structure rendering allows inspection of predicted folds.

---

## Configuration

All major parameters are centralized in `config.py`, including:

* Model identifier (default: `facebook/esmfold_v1`)
* Sequence length limits and validation rules
* Attention aggregation and thresholding defaults
* Visualization and UI settings

Modifying `config.py` allows you to tune performance, interpretability depth, and user experience without changing core logic.

---

## Performance Notes

* Long sequences may require significant memory and compute time.
* GPU acceleration is strongly recommended for practical use.
* Attention analysis can be computationally expensive and may be limited for very long sequences to avoid memory issues.

---

## Limitations and Disclaimer

* Predicted structures are **not experimentally validated** and should not be used for clinical or safety-critical decisions.
* Attention-based interpretations are **heuristic** and do not necessarily correspond to physical residue contacts.
* This tool is intended for **research and educational purposes only**.

---

## Troubleshooting

* **Out-of-memory errors**: Reduce sequence length or run on CPU.
* **Model download issues**: Ensure internet access and sufficient disk space.
* **3D visualization not appearing**: Install `py3dmol` and restart the app.

---

## Extending the Project

* Add new biochemical properties in `property_analyzer.py`
* Experiment with alternative attention aggregation strategies
* Integrate additional structure or confidence metrics
* Export results in standardized bioinformatics formats

