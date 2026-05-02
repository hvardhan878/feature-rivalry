import torch

MODEL_NAME = "google/gemma-2-2b-it"
SAE_RELEASE = "gemma-scope-2b-pt-res"
# Per-layer SAE IDs: closest available width_16k SAE to L0≈71 for each analyzed layer.
# Sourced from sae_lens pretrained_saes.yaml (5 L0 checkpoints per layer).
SAE_IDS = {
    0:  "layer_0/width_16k/average_l0_46",
    2:  "layer_2/width_16k/average_l0_53",
    4:  "layer_4/width_16k/average_l0_60",
    6:  "layer_6/width_16k/average_l0_70",
    8:  "layer_8/width_16k/average_l0_71",
    10: "layer_10/width_16k/average_l0_77",
    12: "layer_12/width_16k/average_l0_82",
    14: "layer_14/width_16k/average_l0_83",
    16: "layer_16/width_16k/average_l0_78",
    18: "layer_18/width_16k/average_l0_74",
    20: "layer_20/width_16k/average_l0_71",
    22: "layer_22/width_16k/average_l0_72",
    24: "layer_24/width_16k/average_l0_73",
}

N_PROMPTS_PER_CONDITION = 200
N_SAMPLES_FOR_ENTROPY = 20
# Normalized entropy thresholds — entropy is in [0, 1] (divided by log(n_samples))
# These are calibrated for the full run (N_SAMPLES_FOR_ENTROPY=20).
# Debug mode overrides them in run.py to compensate for n=3 coarse resolution.
ENTROPY_THRESHOLD_HIGH = 0.7   # questions where model is genuinely unsure
ENTROPY_THRESHOLD_LOW = 0.5   # questions where model is confidently correct
TEMPERATURE_SAMPLING = 1.0
MAX_NEW_TOKENS = 5
TARGET_TOKEN_POSITION = -1
LAYERS_TO_ANALYZE = list(range(0, 26, 2))  # 13 layers: 0,2,4,...,24
TOP_K_RIVAL_PAIRS = 50
STEERING_MULTIPLIER = 20.0

DEVICE = "mps"
DTYPE = torch.float32   # bfloat16 has incomplete MPS support
HIDDEN_DIM = 2304       # Gemma-2-2B hidden dimension
N_FEATURES = 16384      # Gemma Scope SAE width

RESULTS_DIR = "results/"
