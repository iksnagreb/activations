# Use the DVC api for loading the YAML parameters
import dvc.api
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


# Generic PyTorch operator with template string describing the operation
class OperatorTemplate(torch.nn.Module):
    def __init__(self, template="x"):
        # Initialize the PyTorch Module superclass
        super().__init__()
        # The template string to be filled at the forward pass
        self.template = template

    # Forward pass instantiating the template
    def forward(self, x):  # noqa: Shadows x...
        # Fill in all instances of x in this template and instantiate the code
        # by evaluating this a some expression
        return eval(self.template)


# Scales the tensor x by (optionally per-channel or power-of-two) scales
def mul(x, power_of_two=False, per_channel=False, _range=4):  # noqa: Shadows
    # Randomly sample some scale factors (if per-channel, else just a single
    # scalar)
    scales = np.random.rand(*((1,) if not per_channel else x.shape[-1:])) # noqa
    # Scale to range and adapt to single-precision floats: Numpy defaults to
    # float64...
    scales = torch.tensor(_range * scales, dtype=torch.float32)
    # Optionally turn the scales to powers of two
    if power_of_two:
        # Round the exponent to the next power of two
        scales = (2 ** torch.round(torch.log2(scales)))
    # Scale the input
    return scales.to(device=x.device) * x


# Adds to the tensor x (optionally per-channel or power-of-two) bias
def add(x, power_of_two=False, per_channel=False, _range=4):  # noqa: Shadows
    # Randomly sample some biases (if per-channel, else just a single scalar)
    bias = np.random.rand(*((1,) if not per_channel else x.shape[-1:])) # noqa
    # Scale to range and adapt to single-precision floats: Numpy defaults to
    # float64...
    bias = torch.tensor(_range * bias, dtype=torch.float32)
    # Optionally turn the bias to powers of two
    if power_of_two:
        # Round the exponent to the next power of two
        bias = (2 ** torch.round(torch.log2(bias)))
    # Scale the input
    return bias.to(device=x.device) + x


# Affine, i.e., Mul-Add, test pattern function
def affine(x, **kwargs):
    # Just forward the same arguments to the Mul and Add pattern
    return add(mul(x, **kwargs), **kwargs)


# Constructs a dummy model for export
def dummy(activation: str, input_bits: int, bits: int, pattern: str, **kwargs):
    # Create the dummy model as a sequence of input quantizer and quantized
    # activation function
    return torch.nn.Sequential(
        # Create an input quantizer
        QuantIdentity(
            # Quantize the input to signed representation of configured bits
            # Note: ReLU needs to be unsigned as outputs are >= 0
            act_quant=act_quantizer(input_bits, _signed=True),
        ),
        # We need to put something here to break the fusible chain to not
        # collapse all of our test model into a single threshold operation as
        # FINN currently cannot handle float thresholds
        OperatorTemplate("x.reshape((1, *x.shape)).reshape(x.shape)"),
        # Add some generic test-pattern template in front of the activation
        # function: This should be a chain of fusible operations
        OperatorTemplate(pattern),
        # Add the quantized activation functions as configured
        _registry[activation](bits, **kwargs)
    )


# Script entrypoint
if __name__ == "__main__":
    # Load the parameters file
    params = dvc.api.params_show("params.yaml")
    # params = dvc.api.params_show("params.yaml")
    # Seed all RNGs
    seed(params["seed"])
    # Make PyTorch behave deterministically if possible
    torch.use_deterministic_algorithms(mode=True, warn_only=True)


    # Generates inputs from the configured range
    def make_inp(num, **kwargs):
        # Get the lower and upper bound of the input range
        x0, x1 = params["range"]
        # Sample random values in [0,1] and scale to the configured range
        return (x1 - x0) * torch.rand(num, *params["shape"], **kwargs) + x0


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
            model(make_inp(128, device=device))
        # Move the model back to the CPU
        model = model.cpu()
    # Switch model to evaluation mode to have it fixed for export
    model = model.eval()
    # Sample random input tensor in batch-first layout
    x = make_inp(1)
    # Compute model output
    o = model(x)
    # Save the input and output data for verification purposes later
    np.save("inp.npy", x.detach().numpy())
    np.save("out.npy", o.detach().numpy())
    # Export the model graph to QONNX
    export_qonnx(model, (x,), "model.onnx", **params["export"])
