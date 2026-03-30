import os

class Config:
    DATA_ROOT = os.path.normpath(os.environ.get(
        "IU_XRAY_ROOT",
        os.path.join(os.path.dirname(__file__), "..", "..", "Datasets", "IU_Xray"),
    ))

    IMAGE_SIZE = 224
    BATCH_SIZE = 4
    NUM_EPOCHS = 50
    LR = 3e-5
    DEVICE = "cuda"