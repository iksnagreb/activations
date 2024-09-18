# Constants and mathematical functions, e.g., sqrt and pi
import math
# PyTorch base package: Math and Tensor Stuff
import torch
# Brevitas: Quantizers and Quantized versions of PyTorch layers
from brevitas.nn import QuantIdentity


# Derives an activation quantizer from the brevitas bases leaving bit-width and
# signedness configurable
def act_quantizer(bits, _signed=True):
    # Brevitas quantizer base classes
    from brevitas.quant.base import IntQuant, ParamFromRuntimePercentileScaling
    from brevitas.quant.solver import ActQuantSolver
    from brevitas.inject.enum import RestrictValueType

    # Derive a Quantizer from the brevitas bases
    class Quantizer(
        IntQuant, ParamFromRuntimePercentileScaling, ActQuantSolver
    ):
        # Configure the quantization bit-width
        bit_width = bits
        # Signedness of the quantization output
        signed = _signed
        # Per tensor quantization, not per channel
        scaling_per_output_channel = False
        # What is this? Copied from PerTensorFloatScaling*
        #   Probably restricts the scale to be floating-point?
        restrict_scaling_type = RestrictValueType.FP
        # Some functions, e.g., GLU, seem to need a lot of calibration steps
        collect_stats_steps = 8192

    # Return the derived quantizer configuration
    return Quantizer


# Registry of activations functions
_registry = {}


# Registers an activation function into the registry by name
def register_activation(name: str):
    # Wrap the actual decorator function taking the class (Module) to register
    def _register(cls: torch.nn.Module):
        # Put the module into the registry dictionary by name
        _registry[name] = cls
        # Return the wrapped and registered class
        return cls

    # Return the wrapping register function, thus this is a two-level decorator
    return _register


# Quantized identity activation
@register_activation("identity")
class Identity(torch.nn.Module):
    # Initializes the model and registers the module parameters
    def __init__(self, bits, **kwargs):
        # Initialize the PyTorch Module superclass
        super().__init__()

        # Quantized identity to be placed after the activation
        self.quant = QuantIdentity(
            # Quantize the activation output to signed bits
            act_quant=act_quantizer(bits, _signed=True), **kwargs
        )

    # Forward pass of the activation function: Just the quantizer
    def forward(self, x):
        return self.quant(x)


# Quantized ReLU activation
@register_activation("relu")
class QuantReLU(torch.nn.Module):
    # Initializes the model and registers the module parameters
    def __init__(self, bits, **kwargs):
        # Initialize the PyTorch Module superclass
        super().__init__()

        # Quantized identity to be placed after the activation
        self.quant = QuantIdentity(
            # Quantize the activation output to unsigned bits
            # Note: ReLU needs to be unsigned as outputs are >= 0
            act_quant=act_quantizer(bits, _signed=False), **kwargs
        )

    # Forward pass of the activation function: ReLU followed by quantizer
    def forward(self, x):
        return self.quant(torch.relu(x))


# Quantized LeakyReLU activation
@register_activation("leaky-relu")
class QuantLeakyReLU(torch.nn.Module):
    # Initializes the model and registers the module parameters
    def __init__(self, bits, **kwargs):
        # Initialize the PyTorch Module superclass
        super().__init__()

        # Quantized identity to be placed after the activation
        self.quant = QuantIdentity(
            # Quantize the activation output to signed bits
            act_quant=act_quantizer(bits, _signed=True), **kwargs
        )

    # Forward pass of the activation function: LeakyReLU followed by quantizer
    def forward(self, x):
        return self.quant(torch.nn.functional.leaky_relu(x))


# Quantized ReLU6 activation function
# Note: Probably exports as Clip(min=0,max=6)
@register_activation("relu6")
class QuantReLU6(torch.nn.Module):
    # Initializes the model and registers the module parameters
    def __init__(self, bits, **kwargs):
        # Initialize the PyTorch Module superclass
        super().__init__()

        # Quantized identity to be placed after the activation
        self.quant = QuantIdentity(
            # Quantize the activation output to unsigned bits
            # Note: ReLU needs to be unsigned as outputs are >= 0
            act_quant=act_quantizer(bits, _signed=False), **kwargs
        )

    # Forward pass of the activation function: ReLU6 followed by quantizer
    def forward(self, x):
        return self.quant(torch.nn.functional.relu6(x))


# Quantized RReLU activation function
# Note: Exports as LeakyRelu(alpha=(lower+upper)/2)
@register_activation("rrelu")
class QuantRReLU(torch.nn.Module):
    # Initializes the model and registers the module parameters
    def __init__(self, bits, lower=0.125, upper=0.3333333333333333, **kwargs):
        # Initialize the PyTorch Module superclass
        super().__init__()

        # Quantized identity to be placed after the activation
        self.quant = QuantIdentity(
            # Quantize the activation output to signed bits
            act_quant=act_quantizer(bits, _signed=True), **kwargs
        )
        # Lower and upper bound of the uniform distribution for sampling the
        # scale on the left, i.e., negative side
        self.lower = lower
        self.upper = upper

    # Forward pass of the activation function: RReLU followed by quantizer
    def forward(self, x):
        return self.quant(torch.nn.functional.rrelu(
            x, self.lower, self.upper, training=self.training
        ))


# Quantized SELU activation function
@register_activation("selu")
class QuantSELU(torch.nn.Module):
    # Initializes the model and registers the module parameters
    def __init__(self, bits, **kwargs):
        # Initialize the PyTorch Module superclass
        super().__init__()

        # Quantized identity to be placed after the ReLU activation
        self.quant = QuantIdentity(
            # Quantize the activation output to signed bits
            act_quant=act_quantizer(bits, _signed=True), **kwargs
        )

    # Forward pass of the activation function: SELU followed by quantizer
    def forward(self, x):
        return self.quant(torch.selu(x))


# Quantized CELU activation function
# TODO: Requires opset version >= 12 while QONNX still prefers version 11
@register_activation("celu")
class QuantCELU(torch.nn.Module):
    # Initializes the model and registers the module parameters
    def __init__(self, bits, alpha=1.0, **kwargs):
        # Initialize the PyTorch Module superclass
        super().__init__()

        # Quantized identity to be placed after the CELU activation
        self.quant = QuantIdentity(
            # Quantize the activation output to signed bits
            act_quant=act_quantizer(bits, _signed=True), **kwargs
        )
        # Alpha scale of the exponential part on the negative left of the
        # function
        self.alpha = alpha

    # Forward pass of the activation function: CELU followed by quantizer
    def forward(self, x):
        return self.quant(torch.celu(x, self.alpha))


# Quantized ELU activation function
@register_activation("elu")
class QuantELU(torch.nn.Module):
    # Initializes the model and registers the module parameters
    def __init__(self, bits, alpha=1.0, **kwargs):
        # Initialize the PyTorch Module superclass
        super().__init__()

        # Quantized identity to be placed after the activation
        self.quant = QuantIdentity(
            # Quantize the activation output to signed bits
            act_quant=act_quantizer(bits, _signed=True), **kwargs
        )
        # Alpha scale of the exponential part of the function
        self.alpha = alpha

    # Forward pass of the activation function: ELU followed by quantizer
    def forward(self, x):
        return self.quant(torch.nn.functional.elu(x, self.alpha))


# Quantized HardShrink activation function
# TODO: Seems to export to a mess which needs to be cleaned up before this can
#  be converted to MultiThreshold
# @register_activation("hardshrink")
class QuantHardShrink(torch.nn.Module):
    # Initializes the model and registers the module parameters
    def __init__(self, bits, lambd=0.5, **kwargs):
        # Initialize the PyTorch Module superclass
        super().__init__()

        # Quantized identity to be placed after the activation
        self.quant = QuantIdentity(
            # Quantize the activation output to signed bits
            act_quant=act_quantizer(bits, _signed=True), **kwargs
        )
        # Lambda for the hardshrink specifying the range where the function is 0
        self.lambd = lambd

    # Forward pass of the activation function: Hardshrink followed by quantizer
    def forward(self, x):
        return self.quant(torch.nn.functional.hardshrink(x, self.lambd))


# Quantized SoftShrink activation function
# TODO: Seems to export to a mess which needs to be cleaned up before this can
#  be converted to MultiThreshold
# @register_activation("softshrink")
class QuantSoftShrink(torch.nn.Module):
    # Initializes the model and registers the module parameters
    def __init__(self, bits, lambd=0.5, **kwargs):
        # Initialize the PyTorch Module superclass
        super().__init__()

        # Quantized identity to be placed after the activation
        self.quant = QuantIdentity(
            # Quantize the activation output to signed bits
            act_quant=act_quantizer(bits, _signed=True), **kwargs
        )
        # Lambda for the softshrink specifying the range where the function is 0
        self.lambd = lambd

    # Forward pass of the activation function: Softshrink followed by quantizer
    def forward(self, x):
        return self.quant(torch.nn.functional.softshrink(x, self.lambd))


# Quantized Sigmoid activation
@register_activation("sigmoid")
class QuantSigmoid(torch.nn.Module):
    # Initializes the model and registers the module parameters
    def __init__(self, bits, **kwargs):
        # Initialize the PyTorch Module superclass
        super().__init__()

        # Quantized identity to be placed after the activation
        self.quant = QuantIdentity(
            # Quantize the activation output to unsigned bits
            # Note: Sigmoid needs to be unsigned as outputs are >= 0
            act_quant=act_quantizer(bits, _signed=False), **kwargs
        )

    # Forward pass of the activation function: Sigmoid followed by quantizer
    def forward(self, x):
        return self.quant(torch.sigmoid(x))


# Quantized Hardsigmoid activation
@register_activation("hardsigmoid")
class QuantHardsigmoid(torch.nn.Module):
    # Initializes the model and registers the module parameters
    def __init__(self, bits, **kwargs):
        # Initialize the PyTorch Module superclass
        super().__init__()

        # Quantized identity to be placed after the activation
        self.quant = QuantIdentity(
            # Quantize the activation output to unsigned bits
            # Note: Sigmoid needs to be unsigned as outputs are >= 0
            act_quant=act_quantizer(bits, _signed=False), **kwargs
        )

    # Forward pass of the activation function: Sigmoid followed by quantizer
    def forward(self, x):
        return self.quant(torch.nn.functional.hardsigmoid(x))


# Quantized LogSigmoid activation function
# TODO: When exported results in a Sigmoid-Log-Quant chain, where only the final
#  Log-Quant will be converted to a MultiThreshold. This might be solved in the
#  future by repeated conversion rounds treating MultiThreshold as a quantizer
#  and thus collapsing longer Activation-...-Activation-Quant chains.
# @register_activation("logsigmoid")
class QuantLogSigmoid(torch.nn.Module):
    # Initializes the model and registers the module parameters
    def __init__(self, bits, **kwargs):
        # Initialize the PyTorch Module superclass
        super().__init__()

        # Quantized identity to be placed after the activation
        self.quant = QuantIdentity(
            # Quantize the activation output to signed bits
            # Note: Actually has no positive range
            act_quant=act_quantizer(bits, _signed=True), **kwargs
        )

    # Forward pass of the activation function: LogSigmoid followed by quantizer
    def forward(self, x):
        return self.quant(torch.nn.functional.logsigmoid(x))


# Quantized Tanh activation
@register_activation("tanh")
class QuantTanh(torch.nn.Module):
    # Initializes the model and registers the module parameters
    def __init__(self, bits, **kwargs):
        # Initialize the PyTorch Module superclass
        super().__init__()

        # Quantized identity to be placed after the activation
        self.quant = QuantIdentity(
            # Quantize the activation output to signed bits
            # Note: Tanh needs to be signed as outputs are in [-1,+1]
            act_quant=act_quantizer(bits, _signed=True), **kwargs
        )

    # Forward pass of the activation function: Tanh followed by quantizer
    def forward(self, x):
        return self.quant(torch.tanh(x))


# Quantized Hardtanh activation
# Note: Exports as Clip(min_val,max_val)
@register_activation("hardtanh")
class QuantHardtanh(torch.nn.Module):
    # Initializes the model and registers the module parameters
    def __init__(self, bits, min_val=-1.0, max_val=+1.0, **kwargs):
        # Initialize the PyTorch Module superclass
        super().__init__()

        # Quantized identity to be placed after the activation
        self.quant = QuantIdentity(
            # Quantize the activation output to signed bits
            # Note: Tanh needs to be signed as outputs are in [-1,+1]
            act_quant=act_quantizer(bits, _signed=True), **kwargs
        )
        # Minimum and maximum of the linear region
        self.min_val = min_val
        self.max_val = max_val

    # Forward pass of the activation function: Tanh followed by quantizer
    def forward(self, x):
        return self.quant(torch.nn.functional.hardtanh(
            x, self.min_val, self.max_val
        ))


# Quantized Softplus activation
@register_activation("softplus")
class QuantSoftplus(torch.nn.Module):
    # Initializes the model and registers the module parameters
    def __init__(self, bits, beta=1.0, threshold=20.0, **kwargs):
        # Initialize the PyTorch Module superclass
        super().__init__()

        # Quantized identity to be placed after the activation
        self.quant = QuantIdentity(
            # Quantize the activation output to unsigned bits
            # Note: Softplus needs to be unsigned as outputs are >= 0
            act_quant=act_quantizer(bits, _signed=False), **kwargs
        )
        # Scale factor beta and the threshold at which softplus reverts to a
        # linear function
        self.beta = beta
        self.threshold = threshold

    # Forward pass of the activation function: Softplus followed by quantizer
    def forward(self, x):
        return self.quant(torch.nn.functional.softplus(
            x, self.beta, self.threshold
        ))


# Quantized Softsign activation
# TODO: Currently not exported as a single node by PyTorch and thus needs to be
#  cleaned up before conversion to MultiThreshold
# @register_activation("softsign")
class QuantSoftsign(torch.nn.Module):
    # Initializes the model and registers the module parameters
    def __init__(self, bits, **kwargs):
        # Initialize the PyTorch Module superclass
        super().__init__()

        # Quantized identity to be placed after the activation
        self.quant = QuantIdentity(
            # Quantize the activation output to signed bits
            act_quant=act_quantizer(bits, _signed=True), **kwargs
        )

    # Forward pass of the activation function: Softsign followed by quantizer
    def forward(self, x):
        return self.quant(torch.nn.functional.softsign(x))


# Quantized exponential function
@register_activation("exp")
class QuantExp(torch.nn.Module):
    # Initializes the model and registers the module parameters
    def __init__(self, bits, **kwargs):
        # Initialize the PyTorch Module superclass
        super().__init__()

        # Quantized identity to be placed after the activation
        self.quant = QuantIdentity(
            # Quantize the activation output to unsigned bits
            # Note: Exponential needs to be unsigned as outputs are in [0,+inf]
            act_quant=act_quantizer(bits, _signed=False), **kwargs
        )

    # Forward pass of the activation function: exp followed by quantizer
    def forward(self, x):
        return self.quant(torch.exp(x))


# Quantized natural logarithmic function
# TODO: As the log yields large (infinite) values near (at) zero, this is not
#  easy to quantize, yielding infinite scale factors.
@register_activation("log")
class QuantLog(torch.nn.Module):
    # Initializes the model and registers the module parameters
    def __init__(self, bits, **kwargs):
        # Initialize the PyTorch Module superclass
        super().__init__()

        # Quantized identity to be placed after the activation
        self.quant = QuantIdentity(
            # Quantize the activation output to signed bits
            # Note: Log needs to be signed as outputs are in [-inf,+inf]
            act_quant=act_quantizer(bits, _signed=True), **kwargs
        )

    # Forward pass of the activation function: log followed by quantizer
    def forward(self, x):
        return self.quant(torch.log(x))


# Quantized square root function
@register_activation("sqrt")
class QuantSqrt(torch.nn.Module):
    # Initializes the model and registers the module parameters
    def __init__(self, bits, **kwargs):
        # Initialize the PyTorch Module superclass
        super().__init__()

        # Quantized identity to be placed after the activation
        self.quant = QuantIdentity(
            # Quantize the activation output to unsigned bits
            # Note: Sqrt needs to be unsigned as outputs are in [0,+inf]
            act_quant=act_quantizer(bits, _signed=False), **kwargs
        )

    # Forward pass of the activation function: sqrt followed by quantizer
    def forward(self, x):
        return self.quant(torch.sqrt(x))


# Quantized error function
@register_activation("erf")
class QuantErf(torch.nn.Module):
    # Initializes the model and registers the module parameters
    def __init__(self, bits, **kwargs):
        # Initialize the PyTorch Module superclass
        super().__init__()

        # Quantized identity to be placed after the activation
        self.quant = QuantIdentity(
            # Quantize the activation output to signed bits
            # Note: Erf needs to be signed as outputs are in [-1,+1]
            act_quant=act_quantizer(bits, _signed=True), **kwargs
        )

    # Forward pass of the activation function: erf followed by quantizer
    def forward(self, x):
        return self.quant(torch.erf(x))


# Quantized floor function
# TODO: Conversion to MultiThreshold fails verification...
@register_activation("floor")
class QuantFloor(torch.nn.Module):
    # Initializes the model and registers the module parameters
    def __init__(self, bits, **kwargs):
        # Initialize the PyTorch Module superclass
        super().__init__()

        # Quantized identity to be placed after the activation
        self.quant = QuantIdentity(
            # Quantize the activation output to signed bits
            act_quant=act_quantizer(bits, _signed=True), **kwargs
        )

    # Forward pass of the activation function: floor followed by quantizer
    def forward(self, x):
        return self.quant(torch.floor(x))


# Quantized ceil function
@register_activation("ceil")
class QuantCeil(torch.nn.Module):
    # Initializes the model and registers the module parameters
    def __init__(self, bits, **kwargs):
        # Initialize the PyTorch Module superclass
        super().__init__()

        # Quantized identity to be placed after the activation
        self.quant = QuantIdentity(
            # Quantize the activation output to signed bits
            act_quant=act_quantizer(bits, _signed=True), **kwargs
        )

    # Forward pass of the activation function: ceil followed by quantizer
    def forward(self, x):
        return self.quant(torch.ceil(x))


# Quantized round function
# Note: With decimals != 0 this exports to Round surrounded by Mul nodes which
# needs to be cleaned up for conversion to MultiThreshold
@register_activation("round")
class QuantRound(torch.nn.Module):
    # Initializes the model and registers the module parameters
    def __init__(self, bits, decimals=0, **kwargs):
        # Initialize the PyTorch Module superclass
        super().__init__()

        # Quantized identity to be placed after the activation
        self.quant = QuantIdentity(
            # Quantize the activation output to signed bits
            act_quant=act_quantizer(bits, _signed=True), **kwargs
        )
        # Number of decimal places to round to
        self.decimals = decimals

    # Forward pass of the activation function: round followed by quantizer
    def forward(self, x):
        return self.quant(torch.round(x, decimals=self.decimals))


# Quantized sign function
@register_activation("sign")
class QuantSign(torch.nn.Module):
    # Initializes the model and registers the module parameters
    def __init__(self, bits, **kwargs):
        # Initialize the PyTorch Module superclass
        super().__init__()

        # Quantized identity to be placed after the activation
        self.quant = QuantIdentity(
            # Quantize the activation output to signed bits
            act_quant=act_quantizer(bits, _signed=True), **kwargs
        )

    # Forward pass of the activation function: round followed by quantizer
    def forward(self, x):
        return self.quant(torch.sign(x))


# Quantized truncation function
# TODO: Cannot be exported to ONNX at all?
# @register_activation("trunc")
class QuantTrunc(torch.nn.Module):
    # Initializes the model and registers the module parameters
    def __init__(self, bits, **kwargs):
        # Initialize the PyTorch Module superclass
        super().__init__()

        # Quantized identity to be placed after the activation
        self.quant = QuantIdentity(
            # Quantize the activation output to signed bits
            act_quant=act_quantizer(bits, _signed=True), **kwargs
        )

    # Forward pass of the activation function: trunc followed by quantizer
    def forward(self, x):
        return self.quant(torch.trunc(x))


# TODO: Monotonic composite functions requiring multiple quantizers for the
#  constituents

# Quantized Tanhshrink activation
# Note: Exports as composite x - tanh(x) which needs quantizers before and after
#  the subtraction
# TODO: Requires more elaborate streamlining and realization of subtraction and
#  some elementwise multiplications in hardware
# TODO: We cannot simply have a tied quantizer at the inputs to the subtraction
#  as it will fit to the tanh mimicking the saturation behavior on the skipping
#  branch, which results in a different function. However, without a tied
#  quantizer at both inputs to the subtraction this cannot be properly
#  streamlined, leaving float operations in the graph (two Mul, one Sub, the
#  output quantizer will have float inputs).
# @register_activation("tanhshrink")
class QuantTanhshrink(torch.nn.Module):
    # Initializes the model and registers the module parameters
    def __init__(self, bits, **kwargs):
        # Initialize the PyTorch Module superclass
        super().__init__()

        # Quantized identity to be placed after the activation
        self.quant0 = QuantIdentity(
            # Quantize the activation output to signed bits
            act_quant=act_quantizer(bits, _signed=True), **kwargs
        )
        # Quantizer placed after the subtraction of the two branches
        self.quant1 = QuantIdentity(
            # Quantize the activation output to signed bits
            act_quant=act_quantizer(bits, _signed=True), **kwargs
        )

    # Forward pass of the activation function: Tanh followed by quantizer
    def forward(self, x):
        return self.quant1(x - self.quant0(torch.tanh(x)))


# TODO: Non-monotonic functions which could be expressed as compositions of
#  various monotonic functions and binary elementwise operations:

# Quantized GELU activation
# Note: Exports as composite ~ x * (1 + erf(x)) which needs quantizers before
# and after the multiplication
# TODO: Tanh approximation lacks range analysis for Pow and seems even more
#  complex regarding streamlining and hardware support. Maybe in FINN the exact
#  formulation is actually the more efficient one, as the hard-to-evaluate Erf
#  is calculated by thresholding.
@register_activation("gelu")
class QuantGELU(torch.nn.Module):
    # Initializes the model and registers the module parameters
    def __init__(self, bits, approximate='none', **kwargs):
        # Initialize the PyTorch Module superclass
        super().__init__()

        # Approximation may only be one of these two options
        assert approximate in {"none", "tanh"}, \
            f"Unknown approximation {approximate}"

        # Quantized identity to be placed after the activation
        self.quant0 = QuantIdentity(
            # Quantize the activation output to signed bits
            act_quant=act_quantizer(bits, _signed=True), **kwargs
        )
        # Quantizer placed after the multiplication of the two branches
        self.quant1 = QuantIdentity(
            # Quantize the activation output to signed bits
            act_quant=act_quantizer(bits, _signed=True), **kwargs
        )
        # Selects optional tanh approximation
        self.approximate = approximate

    # Forward pass of the activation function: Erf/Tanh followed by quantizer
    def forward(self, x):
        # Optionally may select the tanh approximation of GELU
        if self.approximate == "tanh":
            # Tanh approximation of GELU according to
            # https://arxiv.org/pdf/1606.08415v5
            return self.quant1(
                0.5 * x * (1 + self.quant0(
                    torch.tanh((2 / math.pi) ** 0.5 * (x + 0.044715 * x ** 3))
                ))
            )
        # Exact formulation of the GELU involving the Cumulative Distribution
        # Function for Gaussian Distribution according to
        # https://arxiv.org/pdf/1606.08415v5
        return self.quant1(
            0.5 * x * (1 + self.quant0(torch.erf(x * (2 ** -0.5))))
        )


# Quantized SiLU activation
# Note: Exports as composite x * sigmoid(x) which needs quantizers before and
# after the multiplication
@register_activation("silu")
class QuantSiLU(torch.nn.Module):
    # Initializes the model and registers the module parameters
    def __init__(self, bits, **kwargs):
        # Initialize the PyTorch Module superclass
        super().__init__()

        # Quantized identity to be placed after the activation
        self.quant0 = QuantIdentity(
            # Quantize the activation output to signed bits
            act_quant=act_quantizer(bits, _signed=True), **kwargs
        )
        # Quantizer placed after the multiplication of the two branches
        self.quant1 = QuantIdentity(
            # Quantize the activation output to signed bits
            act_quant=act_quantizer(bits, _signed=True), **kwargs
        )

    # Forward pass of the activation function: Sigmoid followed by quantizer
    def forward(self, x):
        return self.quant1(x * self.quant0(torch.sigmoid(x)))


# Quantized Mish activation
# Note: Exports as composite x * tanh(softplus(x)) which needs quantizers before
# and after the multiplication
# TODO: When exported results in a Softplus-Tanh-Quant chain, where only the
#  final Tanh-Quant will be converted to a MultiThreshold. This might be solved
#  in the future by repeated conversion rounds treating MultiThreshold as a
#  quantizer and thus collapsing longer Activation-...-Activation-Quant chains.
# TODO: Requires more elaborate streamlining and realization of multiplication
#  and some elementwise multiplications in hardware
# @register_activation("mish")
class QuantMish(torch.nn.Module):
    # Initializes the model and registers the module parameters
    def __init__(self, bits, **kwargs):
        # Initialize the PyTorch Module superclass
        super().__init__()

        # Quantized identity to be placed after the activation
        self.quant0 = QuantIdentity(
            # Quantize the activation output to signed bits
            act_quant=act_quantizer(bits, _signed=True), **kwargs
        )
        # Quantizer placed after the multiplication of the two branches
        self.quant1 = QuantIdentity(
            # Quantize the activation output to signed bits
            act_quant=act_quantizer(bits, _signed=True), **kwargs
        )

    # Forward pass of the activation function: Softplus-Tanh followed by
    # quantizer
    def forward(self, x):
        return self.quant1(
            x * self.quant0(torch.tanh(torch.nn.functional.softplus(x)))
        )


# Quantized GLU activation function
# Note: Exports as Split-Sigmoid-Mul composition
# Note: Cannot really be visualized, i.e., plotted due to splitting
# TODO: Requires more elaborate streamlining and realization of split operator
#  and some elementwise multiplications in hardware
# TODO: Missing streamlining and backend support for Split
@register_activation("glu")
class QuantGLU(torch.nn.Module):
    # Initializes the model and registers the module parameters
    def __init__(self, bits, dim=-1, **kwargs):
        # Initialize the PyTorch Module superclass
        super().__init__()

        # Quantized identity to be placed after the activation
        self.quant0 = QuantIdentity(
            # Quantize the activation output to signed bits
            act_quant=act_quantizer(bits, _signed=False), **kwargs
        )
        # Quantizer placed after the multiplication of the two branches
        self.quant1 = QuantIdentity(
            # Quantize the activation output to signed bits
            act_quant=act_quantizer(bits, _signed=True), **kwargs
        )
        # Dimension to split along
        self.dim = dim

    # Forward pass of the activation function: trunc followed by quantizer
    def forward(self, x):
        # Make sure the specified dimension can be split into two equal chunks
        assert x.shape[self.dim] % 2 == 0, \
            f"Cannot split input in half along axis {self.dim}: {x.shape}"
        # Split the tensor along the specified dimension
        x0, x1 = torch.split(
            x, split_size_or_sections=x.shape[self.dim] // 2, dim=self.dim
        )
        # GLU gates (multiplication) the second part of the split by the sigmoid
        # activation on the first part of the split. Quantize the Sigmoid and
        # the final output after multiplication.
        return self.quant1(x0 * self.quant0(torch.sigmoid(x1)))


# TODO: Non-monotonic functions which cannot be expressed as such simple
#  compositions as above
# Everything involving reductions: Softmax, Softmin, LogSoftmax, etc...


# If this is called as the main script, list all registered activations
if __name__ == "__main__":
    # Run over the registry items
    for key, value in _registry.items():
        # Print the key and class name
        print(f"{key}: {value}")
