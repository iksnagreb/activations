# Use the DVC api for loading the YAML parameters
# import dvc.api
from xml.sax import parse

import yaml
from numpy.f2py.crackfortran import param_eval
# Progressbar
from tqdm import trange
# For saving numpy array data
import numpy as np
# PyTorch base package: Math and Tensor Stuff
import torch
# Brevitas to QONNX model export
from brevitas.export import export_qonnx
# Brevitas quantizer as PyTorch Module
from brevitas.nn import QuantIdentity

# Quantized activation function registry
from activations import act_quantizer, _registry
# Seeding RNGs for reproducibility
from utils import seed


# Constructs a dummy model for export
def dummy(activation: str, input_bits: int, bits: int, **kwargs):
    # Create the dummy model as a sequence of input quantizer and quantized
    # activation function
    return torch.nn.Sequential(
        # Create an input quantizer
        QuantIdentity(
            # Quantize the input to signed representation of configured bits
            # Note: ReLU needs to be unsigned as outputs are >= 0
            act_quant=act_quantizer(input_bits, _signed=True),
        ),
        # Add the quantized activation functions as configured
        _registry[activation](bits, **kwargs)
    )

# Script entrypoint
if __name__ == "__main__":
    # Load the parameters file
    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)
    # params = dvc.api.params_show("params.yaml")
    # Seed all RNGs
    seed(params["seed"])
    # Make PyTorch behave deterministically if possible
    torch.use_deterministic_algorithms(mode=True, warn_only=True)

    # Construct the dummy model from configuration dictionary
    model = dummy(**params["model"])

    # No gradient accumulation for calibration passes required
    with torch.no_grad():
        # Check whether GPU training is available and select the appropriate
        # device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Move the model to the training device
        model = model.to(device)
        # Multiple passes of calibration might be necessary for larger/deep
        # models
        for _ in trange(0, params["calibration_passes"], desc="calibrating"):
            # Pass random data through the model to "calibrate" dummy quantizer.
            # Large batch to have more calibration samples. Otherwise, there is
            # too much deviation between this calibration and the verification
            # samples.
            model(torch.randn(128, *params["shape"], device=device))
        # Move the model back to the CPU
        model = model.cpu()
    # Switch model to evaluation mode to have it fixed for export
    model = model.eval()
    # Sample random input tensor in batch-first layout
    x = torch.randn(1, *params["shape"])
    # Compute model output
    o = model(x)
    # Save the input and output data for verification purposes later
    np.save("inp.npy", x.detach().numpy())
    np.save("out.npy", o.detach().numpy())
    # Export the model graph to QONNX
    export_qonnx(model, (x,), "model.onnx", **params["export"])
