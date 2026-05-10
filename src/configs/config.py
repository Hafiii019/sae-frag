import os

# Local safetensors copy of Bio_ClinicalBERT (created by convert_models.py).
# If present, all model files use this path to avoid torch.load safety checks
# introduced in transformers 5.x (CVE-2025-32434).
_PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
_LOCAL_BERT = os.path.join(_PROJECT_ROOT, "models", "bio_clinical_bert")
BIO_CLINICAL_BERT = (
    _LOCAL_BERT
    if os.path.exists(os.path.join(_LOCAL_BERT, "model.safetensors"))
    else "emilyalsentzer/Bio_ClinicalBERT"
)


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