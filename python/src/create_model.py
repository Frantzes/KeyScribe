import torch
import torch.onnx
from your_model import HFSFormer

# Load pre-trained model
model = HFSFormer.from_pretrained("path/to/hfsformer-checkpoint")
model.eval()

# Create dummy input (batch=1, time_steps=512, cqt_size=216)
dummy_input = torch.randn(1, 512, 216)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "models/hfsformer.onnx",
    input_names=["spectrogram"],
    output_names=["note_probabilities"],
    opset_version=14,
    verbose=True
)