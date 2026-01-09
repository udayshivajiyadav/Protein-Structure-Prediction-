"""
Configuration constants and settings for the Protein Structure Prediction App
============================================================================

This module contains all configuration constants, default parameters,
and settings used throughout the application.
"""

# Model Configuration
MODEL_NAME = "facebook/esmfold_v1"
MODEL_CACHE_DIR = "./model_cache"

# Sequence Constraints
MAX_SEQUENCE_LENGTH = 400
MIN_SEQUENCE_LENGTH = 10
MAX_SEQUENCE_LENGTH_WARNING = 200  # Warn user about performance above this

# Analysis Parameters
DEFAULT_THRESHOLD_PERCENTILE = 90
DEFAULT_TOP_K_INTERACTIONS = 20
MIN_CORRELATION_POINTS = 3

# Visualization Settings
HEATMAP_WIDTH = 700
HEATMAP_HEIGHT = 700
NETWORK_WIDTH = 700
NETWORK_HEIGHT = 500
STRUCTURE_WIDTH = 700
STRUCTURE_HEIGHT = 500

# Performance Settings
TORCH_NO_GRAD = True
ATTENTION_BATCH_SIZE = 1
MAX_ATTENTION_LAYERS = 48

# Color Schemes
ATTENTION_COLORSCALE = 'Viridis'
NODE_COLORS = {
    'connected': 'lightcoral',
    'isolated': 'lightgray',
    'default': 'lightblue'
}

# Amino Acid Properties
HYDROPHOBICITY_SCALE = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
    'X': 0.0, 'B': -3.5, 'Z': -3.5, 'J': 4.15, 'U': 2.5
}

CHARGE_SCALE = {
    'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0,
    'Q': 0, 'E': -1, 'G': 0, 'H': 0.1, 'I': 0,
    'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
    'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0,
    'X': 0, 'B': -0.5, 'Z': -0.5, 'J': 0, 'U': 0
}

HELIX_PROPENSITY_SCALE = {
    'A': 1.42, 'R': 0.98, 'N': 0.67, 'D': 1.01, 'C': 0.70,
    'Q': 1.11, 'E': 1.51, 'G': 0.57, 'H': 1.00, 'I': 1.08,
    'L': 1.21, 'K': 1.16, 'M': 1.45, 'F': 1.13, 'P': 0.57,
    'S': 0.77, 'T': 0.83, 'W': 1.08, 'Y': 0.69, 'V': 1.06,
    'X': 1.0, 'B': 0.84, 'Z': 1.31, 'J': 1.145, 'U': 0.70
}

# Valid amino acids
STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')
VALID_AA = STANDARD_AA | set('XBZJU')  # Include ambiguous codes

# Example sequences
EXAMPLE_SEQUENCES = {
    'ubiquitin': "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    'small_peptide': "MQIFVKTLTGKTITLEVEPS",
    'insulin_a_chain': "GIVEQCCTSICSLYQLENYCN",
    'lysozyme_fragment': "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNT"
}

# Error messages
ERROR_MESSAGES = {
    'empty_sequence': "Sequence cannot be empty",
    'invalid_chars': "Invalid amino acid characters found: {chars}",
    'too_short': f"Sequence too short (minimum {MIN_SEQUENCE_LENGTH} residues)",
    'too_long': f"Sequence too long (maximum 1000 residues for stability)",
    'model_load_failed': "Failed to load ESMFold model",
    'gpu_error': "GPU error detected, falling back to CPU",
    'prediction_failed': "Structure prediction failed",
    'attention_processing_failed': "Could not process attention weights",
    'correlation_failed': "Could not calculate property correlations"
}

# Help text
HELP_TEXT = {
    'sequence_input': "Enter a protein sequence using single-letter amino acid codes. FASTA format is accepted.",
    'threshold': "Percentile threshold for highlighting top interactions in attention analysis",
    'top_k': "Number of strongest interactions to display in the network graph",
    'explainability': "Extract and visualize attention weights (increases computation time)"
}

# UI Configuration
UI_CONFIG = {
    'page_title': "Explainable Protein Structure Prediction",
    'page_icon': "🧬",
    'layout': "wide",
    'sidebar_state': "expanded"
}

# File extensions and formats
FILE_FORMATS = {
    'pdb': {'extension': '.pdb', 'mime': 'text/plain'},
    'report': {'extension': '.txt', 'mime': 'text/plain'},
    'csv': {'extension': '.csv', 'mime': 'text/csv'}
}

# Progress tracking
PROGRESS_STEPS = {
    'model_loading': {'start': 0, 'tokenizer': 25, 'model': 50, 'config': 75, 'test': 90, 'complete': 100},
    'prediction': {'start': 0, 'tokenize': 20, 'predict': 70, 'process': 90, 'complete': 100},
    'attention': {'start': 0, 'extract': 30, 'aggregate': 70, 'threshold': 90, 'complete': 100}
}