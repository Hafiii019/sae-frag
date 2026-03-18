import os

class Config:
    DATA_ROOT = os.environ.get("IU_XRAY_ROOT", "C:/Datasets/IU_Xray")

    IMAGE_SIZE = 224
    BATCH_SIZE = 4
    NUM_EPOCHS = 50
    LR = 3e-5
    DEVICE = "cuda"