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
# QONNX wrapper for ONNX models
from qonnx.core.modelwrapper import ModelWrapper
# QONNX datatype annotations
from qonnx.core.datatype import DataType

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


# Elementwise affine transformation test pattern
class Affine(torch.nn.Module):
    # Initializes the affine transformation
    def __init__(self, shape, cdim, restrict_scaling_type, per_channel,  # noqa
                 range, **kwargs):  # noqa
        # Initialize the PyTorch Module superclass
        super().__init__()
        # Remember arguments for lazy initialization
        self.restrict_scaling_type = restrict_scaling_type
        self.per_channel = per_channel
        self.range = range
        self.cdim = cdim
        # Assume scalar shape if no shape is given
        self.shape = [1, ]
        # Adjust the parameter shape depending on whether this is per-tensor or
        # per-channel
        if shape is not None:
            self.shape = np.ones_like(shape)
            self.shape[cdim] = shape[cdim] if self.per_channel else 1
        # Create a scale and bias parameter
        self.scale = torch.nn.Parameter(torch.empty(tuple(self.shape)))
        self.bias = torch.nn.Parameter(torch.empty(tuple(self.shape)))
        # Reset all parameters to the initialization range
        self.reset_parameters()

    # Implements the restrict_scaling_type behavior
    def restrict(self, value):
        # Select the restriction function from dictionary by RestrictValueType
        # enumerations
        return {
            # No restriction for float scales
            "FP": lambda _x: _x,
            # Log-floats essentially behave the same way, no restrictions
            # Note: The learning behavior should be different, but here we don't
            # care, we only model the final, exported behavior.
            "LOG_FP": lambda _x: _x,
            # Restrict scales to integers by simple rounding
            # Note: We don not model proper rounding modes here, as we only care
            # for the final exported behavior.
            "INT": lambda _x: torch.round(_x),
            # Restrict scales to powers of two by rounding the exponent
            "POWER_OF_TWO": lambda _x: 2 ** _x.log2().round()
        }[self.restrict_scaling_type](value)

    # Resets/initializes the parameter tensors
    def reset_parameters(self):
        # Initialize the parameters from a uniform distribution the configured
        # range
        torch.nn.init.uniform_(self.scale, *self.range)
        torch.nn.init.uniform_(self.bias, *self.range)

    # Forward pass applying scale and bias to the input
    def forward(self, x):  # noqa: Shadows x
        # Apply scale and bias to the input
        return self.restrict(self.scale) * x + self.restrict(self.bias)


# Lazy version of affine elementwise transformation inferring the shape at the
# first forward pass
class LazyAffine(torch.nn.modules.lazy.LazyModuleMixin, Affine):  # noqa: lazy
    # Once initialized, this will become Affine as defined above
    cls_to_become = Affine
    # Parameter tensors of the Affine are uninitialized
    scale: torch.nn.UninitializedParameter
    bias: torch.nn.UninitializedParameter

    # Initializes the affine transformation
    def __init__(self, cdim, restrict_scaling_type, per_channel, range,  # noqa
                 **kwargs):
        # Initialize the PyTorch Module superclass
        super().__init__(None, cdim, restrict_scaling_type, per_channel, range,
                         **kwargs)
        # Register uninitialized parameter tensors
        self.scale = torch.nn.UninitializedParameter()
        self.bias = torch.nn.UninitializedParameter()

    # Resets/initializes the parameter tensors
    def reset_parameters(self):
        # If this has already been initialized, delegate to the actual
        # implementation
        if not self.has_uninitialized_params():
            super().reset_parameters()

    # Initializes/Materializes the uninitialized parameter tensor given some
    # sample input tensor to infer the dimensions
    def initialize_parameters(self, x):  # noqa: Shadows x
        # Only materialize the parameter tensor if it is not yet initialized
        if self.has_uninitialized_params():
            # Do not accumulate gradient information from initialization
            with torch.no_grad():
                # Adjust the parameter shape depending on whether this is
                # per-tensor or per-channel
                self.shape = np.ones_like(x.shape)
                if self.per_channel:
                    self.shape[self.cdim] = x.shape[self.cdim]
                # Materialize the scale and bias parameter tensors
                self.scale.materialize(tuple(self.shape))
                self.bias.materialize(tuple(self.shape))
                # Properly initialize the parameters by resetting the values
                self.reset_parameters()


# Constructs a dummy model for export
def dummy(activation: str, input_bits: int, bits: int, pattern: str,
          affine: dict, activation_kwargs: dict, **kwargs):
    # Create the dummy model as a sequence of input quantizer and quantized
    # activation function
    return torch.nn.Sequential(
        # Input quantizer from floats to configured bit-width
        QuantIdentity(
            # Quantize the input to signed representation of configured bits
            # Note: ReLU needs to be unsigned as outputs are >= 0
            act_quant=act_quantizer(input_bits, _signed=True), **kwargs,
            # Return the scale and bias quantization in formation
            return_quant_tensor=True
        ),
        # Add some generic test-pattern template in front of the activation
        # function: This should be a chain of fusible operations
        OperatorTemplate(pattern),
        # Add configurable elementwise affine transformation to test
        # per-channel vs. per-tensor and power of two vs. float parameters
        LazyAffine(**affine, **kwargs),
        # Add the quantized activation functions as configured
        _registry[activation](bits, **activation_kwargs, **kwargs)
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

    # Export the model graph to QONNX
    export_qonnx(model, (x,), "model.onnx", **params["export"])

    # Save the input and output data for verification purposes later
    np.save("inp.npy", x.detach().numpy())
    np.save("out.npy", o.detach().numpy())
