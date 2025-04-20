class TimesFmConfig:
    BACKEND = "cpu"
    PER_CORE_BATCH_SIZE = 32
    HORIZON_LEN = 128
    NUM_LAYERS = 50
    MODEL_DIMS = 1280
    CONTEXT_LEN = 2048
    CHECKPOINT_REPO = "google/timesfm-2.0-500m-pytorch"