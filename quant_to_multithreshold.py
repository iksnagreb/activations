# Python warning messages
import warnings
# Proper copies of python objects
import copy
# Numpy for handling tensors (inputs, outputs, initializers, thresholds, ...)
import numpy as np
# Progressbar showing how far the threshold conversion already progressed
# checking the input range for thresholds
from tqdm import tqdm
# 1d convolution to detect edges (actually image derivative)
from scipy.ndimage import convolve1d

# QONNX wrapper of ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper
# Converts ONNX graph nodes to QONNX custom-ops instances if possible
from qonnx.custom_op.registry import getCustomOp
# QONNX base class for all graph transformations
from qonnx.transformation.general import Transformation
# QONNX graph transformations for inferring data types, layouts and shapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
# Folds (collapse constant tensors and chains of operations on constant tensors)
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.quant_constant_folding import \
    FoldTransposeIntoQuantInit
from qonnx.transformation.remove import RemoveIdentityOps
from finn.transformation.qonnx.fold_quant_weights import FoldQuantWeights
from finn.transformation.streamline import (
    FactorOutMulSignMagnitude, Absorb1BitMulIntoMatMul, Absorb1BitMulIntoConv
)

# Range analysis to generate input ranges and scales use to enumerate inputs and
# outputs of quantized activation functions to generate thresholds
from qonnx.util.range_analysis import range_analysis, RangeInfo
# Executes an ONNX node considering QONNX domain operations as well
from qonnx.core.onnx_exec import execute_node
# Utility for creating a tensor according to the description in ONNX value info
from qonnx.util.onnx import valueinfo_to_tensor

# Protobuf onnx graph node type
from onnx import NodeProto, TensorProto
# Helper for assembling ONNX nodes, tensors and graphs
from onnx import helper as oh

# Supported monotonic activation functions
SUPPORTED_MONOTONIC_ACTIVATIONS = {
    "Identity",
    "Relu",
    "LeakyRelu",
    "Clip",
    "Selu",
    "Celu",
    "Elu",
    "Sigmoid",
    "HardSigmoid",
    "Tanh",
    "Softplus",
    "Exp",
    "Log",
    "Sqrt",
    "Erf",
    "Floor",
    "Ceil",
    "Round",
    "Sign"
}

# Supported monotonic elementwise functions for fusing operations into
# thresholds
SUPPORTED_MONOTONIC_ELTWISE = {
    "Add", "Sub", "Mul", "Div",  # TODO: More to add?
}

# Supported types of quantization operations
SUPPORTED_QUANTIZERS = {
    "Quant",  # TODO: BipolarQuant and MultiThreshold from QONNX, QuantizeLinear
}

# Set of operator types which could be fused into quantizers while converting to
# multi-thresholds
FUSIBLE_OPS = {
    *SUPPORTED_QUANTIZERS,
    *SUPPORTED_MONOTONIC_ACTIVATIONS,
    *SUPPORTED_MONOTONIC_ELTWISE
}


# Tests whether two shapes can be broadcast according to NumPy semantics
def can_broadcast_shapes(lhs, rhs):
    # Broadcasting might raise an exception
    try:
        # Try broadcasting the shapes
        if len(np.broadcast_shapes(lhs, rhs)) == max(len(lhs), len(rhs)):
            # These tensors can be broadcast, preserving the
            # left-hand-side shape
            return True
        # These tensors cannot be broadcast
        return False
    # Failing to broadcast the tensors raises ValueError
    except ValueError:
        # These tensors cannot be broadcast
        return False


# Extracts the complete subgraph chain of fusible elementwise operations
# leading up to a quantizer
def extract_quant_fusible_subgraph(
        node: NodeProto, model: ModelWrapper, cdim: int = -1, quant_filter=None
):
    # Default to dummy filter accepting any node
    if quant_filter is None:
        quant_filter = (lambda _0, _1: True)

    # Checks whether an operation can be fused into the quantization operation
    # when converting to thresholds
    def is_fusible(n: NodeProto):
        # Overall the subgraph must produce the same shape as it consumes at the
        # input as the single fused operator will not reshape or broadcast, thus
        # we should not include such operations for now
        if (model.get_tensor_shape(n.input[0]) != model.get_tensor_shape(
                node.output[0])):
            return False
        # If this is one of the supported monotonic elementwise operations, we
        # need to make sure these are either per tensor or per channel as the
        # thresholding does currently not support any grouping, tiling or
        # whatever...
        if n.op_type in SUPPORTED_MONOTONIC_ELTWISE:
            # This only applies if there even is a parameter tensor
            if (init := model.get_initializer(n.input[1])) is not None:
                # Per-tensor or per-channel means we have some parameter tensor
                # which can be broadcast to the channel dimension of the output
                if not can_broadcast_shapes(
                        init.shape,
                        (model.get_tensor_shape(node.output[0])[cdim],)
                ):
                    # Not-fusible...
                    return False
        # We can fuse quantizer into quantizers, monotonic activations and
        # monotonic elementwise operations
        if n.op_type in FUSIBLE_OPS and quant_filter(model, n):
            # We cannot fuse branching topologies for now...
            return not (model.is_join_node(n) or model.is_fork_node(n))
        # Cannot fuse this operator...
        return False

    # We must start on some supported quantization operation
    if node.op_type in SUPPORTED_QUANTIZERS and quant_filter(model, node):
        # We already know the quantizer has one actual, i.e., non-parameter,
        # input for which we want to track the producer chain
        quant_inp = node.input[0]
        # Track the producer chain upwards collecting all operators up until
        # including the first not-fusible, keep_if_not_found to include the
        # global input as well, i.e., when the condition is never fulfilled.
        subgraph = model.find_upstream(
            quant_inp, lambda x: not is_fusible(x), keep_if_not_found=True
        )
        # There might be no suitable nodes at all...
        if not subgraph:
            return []
        # Decompose the subgraph to do additional checks on the first operator
        *chain, first = subgraph
        # Return the operator chain extended by the anchoring quantizer and
        # reverse for in-order traversal when simulating
        return reversed([node, *chain, *([first] if is_fusible(first) else [])])
    # Return empty subgraph, nothing to do here...
    return []


# Executes a subgraph given as a list (chain, i.e., non-branching) of onnx nodes
#   Note: "Costly" version evaluating the whole input x in one pass
def _evaluate_subgraph(subgraph: list[NodeProto], model: ModelWrapper, x):
    # Operate on a deep copy of the model as we are going to mess with the graph
    # value-info
    model = copy.deepcopy(model)

    # Names of all tensors produced or consumed by any operation in the subgraph
    tensors = set([x for node in subgraph for x in [*node.input, *node.output]])

    # Add a batch dimension to all connecting tensors, scalar parameters should
    # be broadcastable
    for name in tensors:
        # Force subgraph evaluation to Batch x Channel layout
        model.set_tensor_shape(name, [x.shape[0], x.shape[1]])
        # Reshape the initializer tensor if there is any
        if (init := model.get_initializer(name)) is not None:
            # Squeezing should be ok as we already assume initializer to be
            # broadcastable
            model.set_initializer(name, init.squeeze())

    # Creates a tensor according to the value info
    def tensor_placeholder(tensor_name):
        # If the tensor has some initializer fill with constant parameter
        if (init := model.get_initializer(tensor_name)) is not None:  # noqa
            return init
        # If there is no initializers we need some placeholder for dynamic
        # inputs and outputs
        return valueinfo_to_tensor(model.get_tensor_valueinfo(tensor_name))

    # Prepare the execution context with placeholder for all tensors relevant to
    # the subgraph
    ctx = {**{name: tensor_placeholder(name) for name in tensors}}  # noqa: dict
    # Insert the input to the subgraph which must be the first input to the
    # first operator
    ctx[subgraph[0].input[0]] = x.astype(np.float32)

    # Execute all nodes in the subgraph in order, updating the execution context
    # after each step
    for node in subgraph:
        execute_node(node, ctx, model.graph)

    # Extract the final output from the execution context
    return ctx[subgraph[-1].output[0]]


# Executes a subgraph given as a list (chain, i.e., non-branching) of onnx nodes
# Note: Chunking along the batch dimension to avoid excessive memory utilization
# for intermediate tensors...
def evaluate_subgraph(subgraph: list[NodeProto], model: ModelWrapper, x):
    # Split into chunks along the batch: Trade off memory utilization (to hold
    # the full execution context while evaluating) vs. execution time
    chunks = np.array_split(x, 2, axis=0)  # TODO: Make batch size configurable
    # Evaluate the subgraph for each chunk
    chunks = [_evaluate_subgraph(subgraph, model, x=x) for x in chunks]
    # Put the result back together along the batch dimension so the caller does
    # not even notice we did a chunked processing
    return np.concatenate(chunks, axis=0)


# Extracts multi-threshold representation from a function input-output pair
# covering the whole range of possible inputs at some resolution we do not
# actually care for anymore at this point.
def find_thresholds(x: np.array, y: np.array):
    # Find all step locations by convolution with an edge detection kernel
    edges = convolve1d(y, np.array([+1, -1]), mode="nearest", axis=0, origin=-1)

    # Output scale - weight, i.e., height of the smallest step
    scale = np.abs(edges[edges != 0]).min()
    # Output bias - function offset at the start without taking any steps
    bias = y[0]
    # Only positive bias can be handled via monotonically increasing threshold
    # functions
    min_bias, bias = np.min(bias), bias - np.min(bias)
    # Remove the integer part from the bias - this can be handled via thresholds
    bias, padding = np.modf(bias / scale)

    # Start collecting per-channel thresholds with left side padding to account
    # for the integer part of the bias
    thresholds = [[-np.inf for _ in range(int(p))] for p in padding]
    # Collect step weights as well - only really relevant for non-monotonic
    # functions
    weights = [[1 for _ in range(int(p))] for p in padding]
    # Thresholds are where there are non-zero edge detections
    # Note: Channels first followed by thresholds in increasing order
    for i, j in zip(*np.where(edges.T)):
        # Threshold multiplicity - weight, i.e., height of the step
        weight = int(np.round((edges[j, i] / scale)))
        # Switch from [Steps, C] to [C, Steps] layout collected as nested lists
        thresholds[i].extend(np.abs(weight) * [x[j, i]])
        # Collect signed weights as well
        weights[i].extend(np.abs(weight) * [np.sign(weight)])

    # Right side padding amount needed to fill up all threshold lists to the
    # maximum length
    padding = [max((len(t) for t in thresholds)) - len(t) for t in thresholds]
    # Insert the right side padding into each threshold list
    thresholds = [[*t, *(p * [np.inf])] for t, p in zip(thresholds, padding)]
    # Add padding to the weights as well
    weights = [[*w, *(p * [1])] for w, p in zip(weights, padding)]

    # Return the collected thresholds, output scale and remaining fractional
    # part of the bias
    return np.asarray(thresholds), np.asarray(weights), scale, bias + min_bias


# Multi-threshold representation of a piecewise-constant function
def multithreshold(x, thresholds, weights=None, scale=1.0, bias=0.0):
    # Expand a dimension at the end to match and broadcast the thresholds
    x = np.expand_dims(x, axis=-1)
    # Weights are optional: Assume positive unit steps by default
    weights = np.ones_like(thresholds) if weights is None else weights
    # Count steps and scale and shift into expected output range
    return scale * np.sum(weights * (x >= thresholds), axis=-1) + bias


# Converts supported quantized activation functions to MultiThreshold
class QuantToMultiThreshold(Transformation):
    # Filter to reject the global input quantizer from conversion...
    @staticmethod
    def reject_input_quant(model: ModelWrapper, node: NodeProto):
        # If node is not a quantizer, do not reject it here, there should be
        # other conditions to reject it checked elsewhere...
        if not node.op_type in SUPPORTED_QUANTIZERS:
            return True
        # Get the names of all global input tensors to insert a Squeeze
        # operation in front
        global_inputs = [inp.name for inp in model.graph.input]
        # Check whether any of the input is a global input
        if any(inp in global_inputs for inp in node.input):
            # Reject quantizers directly connected to a global input
            return False
        # Look for another quantizer preceding this quantizer somewhere upstream
        n = model.find_upstream(node.input[0], lambda x: x.op_type == "Quant")
        # If there is no quantizer upstream, the list n will be empty
        return bool(n)

    # Filter to reject quantizers with too many bits
    @staticmethod
    def reject_bit_width(bits: int):
        # The actual filter function...
        def _filter(model: ModelWrapper, node: NodeProto):
            # If node is not a quantizer, do not reject it here, there should be
            # other conditions to reject it checked elsewhere...
            if not node.op_type in SUPPORTED_QUANTIZERS:
                return True
            # Check whether the quantizer represents the output with too many
            # bits
            return int(model.get_initializer(node.input[3])) < bits

        # Return the filter function
        return _filter

    # Initializes the conversion by setting a seed range information for the
    # range analysis pass
    def __init__(self, range_info: RangeInfo = None,
                 quant_filter=None, assume_monotonic=False):
        # Initialize the Transformation super class
        super().__init__()
        # Store the seed range information
        self.range_info = range_info
        # Filter function to control which quantizers are converted to
        # thresholds: None means no additional filter
        self.quant_filter = quant_filter
        # keep a copy of the range analysis result to allow later inspection
        self.range_analysis_result = None
        # Assume conversion of monotonic function which allows faster and less
        # memory hungry threshold search
        self.assume_monotonic = assume_monotonic

    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        # Add shape and datatype annotations throughout all the graph
        model = model.transform(InferDataTypes())
        model = model.transform(InferShapes())
        # Apply constant folding transformation to clean up the graph before
        # applying the analysis (these are not part of the included cleanup
        # transformations)
        model = model.transform(FoldConstants())
        model = model.transform(FoldTransposeIntoQuantInit())
        model = model.transform(FoldQuantWeights())
        model = model.transform(FoldConstants())

        # Absorb bipolar scales from Mul following MatMul-like operators into
        # weights to avoid trying to convert a layer tail which just looks
        # non-monotonic while actually being perfectly fine to convert...
        model = model.transform(FactorOutMulSignMagnitude())
        model = model.transform(Absorb1BitMulIntoMatMul())
        model = model.transform(Absorb1BitMulIntoConv())

        # Redo shape and data type annotations after folding and cleanup might
        # have changed those
        model = model.transform(InferDataTypes())
        model = model.transform(InferShapes())
        # Generate range information, including integer range information, for
        # all tensors in the model graph
        range_info, model = range_analysis(
            # Transform and analyze the model: Returns a modified model
            model,
            # Seed input range information: Might be None
            irange=self.range_info,
            # Return the range information gathered during the analysis
            report_mode="range",
            # Produce scaled integer range information, not just floating-point
            # Note: This is necessary for enumerating quantizer output levels
            scaled_int=True,
            # Unbroadcast the tensors for some deduplication of ranges and
            # scales. Without this, range analysis yields per-element
            # information and thus produces per-element thresholds which need to
            # be reduced manually later.
            # Note: Currently disabled as local node/graph execution does not
            # work on unbroadcast tensors
            do_unbroadcast=False,
            # Model needs some cleanup in preparation for the range analysis
            do_cleanup=True,
        )
        self.range_analysis_result = range_info

        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        # Note: Reversed as we are anchoring at the final quantizer of a fusible
        # monotonic-activation-eltwise-quantizer chain extending upwards
        for index, node in enumerate(reversed(graph.node)):
            # First try to consider the tensor layout of the output for
            # determining the number of output channels
            layout = model.get_tensor_layout(node.output[0])
            # If there is no layout annotation, guess based on rank of the
            # tensor
            if layout is None:
                # Maps tensor rank to layout annotation
                rank_to_layout = {
                    # TODO: 5-dimensional layout just for some dummy test-case
                    0: None, 1: "C", 2: "NC", 3: "NWC", 4: "NCHW", 5: "N_CHW"
                }
                # Lookup the layout required by this input shape
                layout = rank_to_layout[
                    len(model.get_tensor_shape(node.input[0]))
                ]
            # If there is a layout annotation, use this to determine the
            # index of the channel dimension
            if layout is not None and "C" in layout:
                # Lookup the index in list
                cdim = layout.index("C")
            # If no layout has been annotated or there is no channel
            # dimension, fall back to the previous default assumption
            else:
                # Assume the channels to be in axis 1
                cdim = 1
                # Issue a warning to the user, so they are aware of this
                warnings.warn(
                    f"No meaningful layout for {node.input[0]}:"
                    f" Assuming channel dimension at index {cdim}"
                )

            # Try to match a convertible subgraph of quantizers, activations and
            # monotonic operations
            subgraph = list(extract_quant_fusible_subgraph(
                node, model, cdim=cdim, quant_filter=self.quant_filter
            ))
            # Skip if no quantizer is present
            if not subgraph:
                # Softly skip without warning, transformation just does not
                # apply here
                continue
            # Name of the input and output tensor of the whole chain of
            # quantized operations
            inp, out = subgraph[0].input[0], subgraph[-1].output[0]

            # The input and output to the activation-quantizer combination must
            # be described by the range information analyzed above to be able to
            # enumerate the input/output levels for generating thresholds
            if inp in range_info and out in range_info:
                # Conversion for non-integer input ranges might be slow as we
                # kind of have to guess the right resolution to enumerate the
                # float range which practically means sampling this with rather
                # high resolution, like 1e-4
                if range_info[inp].int_range is None:
                    # Better issue a warning to make the user aware of this...
                    warnings.warn(
                        f"{self.__class__.__name__}: Potential slow conversion "
                        f"No input integer range info for {inp}"
                    )

                # Get the quantizer node terminating the chain of operators as
                # this holds some extra information such as the target bit-width
                quant = subgraph[-1]

                # Check whether this is a signed quantizer
                signed = getCustomOp(quant).get_nodeattr("signed")
                narrow = int(getCustomOp(quant).get_nodeattr("narrow"))

                # Get the output bit-with to be produced by the quantizer,
                # which determines how many thresholds are needed
                bits = int(model.get_initializer(quant.input[3]))

                # The output is produced by a quantizer, thus we can always
                # assume the integer range
                (__, __), dy = range_info[out].range, range_info[out].scale
                # Input range minimum and maximum serve as initial values for
                # the interval bounds
                (x0, x1), dx = range_info[inp].range, range_info[inp].scale

                # Broadcast the input to the expected input shape: This allows
                # to simplify the input range annotation for global graph inputs
                x0 = np.broadcast_to(x0, model.get_tensor_shape(inp))
                x1 = np.broadcast_to(x1, model.get_tensor_shape(inp))

                # We do not handle reversed indexing here
                cdim = x0.ndim + cdim if cdim < 0 else cdim
                # Reduces over all but the channel axes
                axis = tuple(i for i in range(x0.ndim) if i not in {cdim})

                # Reduce the bounds of the range to simulate: Upper/Lower bound
                x0 = np.min(x0, axis).astype(np.float64)
                x1 = np.max(x1, axis).astype(np.float64)

                # If the input range does not have a know scale for enumerating
                # the inputs, set some default
                dx = 1.0e-4 if dx is None else np.min(np.asarray(dx))

                # Start with a single big chunk covering the whole input range
                # divided into uniform steps
                chunks = [(x0, x1, int(np.ceil(np.max(x1 - x0) / dx)))]

                # Breaking up the chunks
                while True:
                    # Stop once all chunks cover not more than 1024 elements
                    if all([size <= 2 ** 10 for _, _, size in chunks]):
                        break

                    # Next chunk for refinement
                    x0, x1, size = chunks.pop(0)

                    # Evaluate function output at the bounds of the chunk
                    y0, y1 = evaluate_subgraph(subgraph, model, [x0, x1])

                    # Assuming monotonicity we can drop all chunks where the
                    # output does not change to speed up search
                    if not self.assume_monotonic or np.any(y0 != y1):
                        # Do not cut smaller than necessary...
                        if size <= 2 ** 10:
                            # Just put it back
                            chunks.append((x0, x1, size))
                            # And continue with the next chunk
                            continue

                        # Mid-point for splitting the chunk in two equal halves
                        xm = x0 + 0.5 * (x1 - x0)
                        # Insert both halves to be refined in later iterations
                        chunks.append((x0, xm, np.ceil(np.max(xm - x0) / dx)))
                        chunks.append((xm, x1, np.ceil(np.max(x1 - xm) / dx)))

                # Span the whole range of input values for each chunk
                # TODO: Assuming non-monotonicity this still might result in OOM
                #  issues for large/high resolution ranges...
                chunks = [np.linspace(x0, x1, int(s)) for x0, x1, s in chunks]

                # Join all chunks for parallel processing
                xs = np.concatenate(chunks)
                # Make sure all inputs are at the quantization levels
                xs = np.round(xs / np.asarray(dx)) * np.asarray(dx)
                # Evaluate the function on the range in batch mode
                ys = _evaluate_subgraph(subgraph, model, xs)

                # Find weighted thresholds and output scale and bias over the
                # input-output range
                thresholds, weights, scale, bias = find_thresholds(xs, ys)

                # Evaluate the just-extracted multi-threshold representation on
                # the input range
                # TODO: This might end up using a lot of memory due to
                #  broadcasting thresholds resulting in OOM issues...
                zs = multithreshold(xs, thresholds, weights, scale, bias)

                # Sanity check for correctness (exactness): The multi-threshold
                # representation must match the original subgraph
                if not np.allclose(ys, zs):
                    # Issue a warning to make the user aware of this
                    warnings.warn(
                        f"{self.__class__.__name__}: Skipping near match: "
                        f"Threshold conversion failed near {quant.name}"
                    )
                    # Skip to the next candidate activation/quantizer
                    continue

                # Sanity check for monotonicity: Non-monotonic functions have
                # some negative weights
                if np.any(weights < 0.0):
                    # Issue a warning to make the user aware of this
                    warnings.warn(
                        f"{self.__class__.__name__}: Skipping near match: "
                        f"Non-monotonic function near {quant.name}"
                    )
                    # Skip to the next candidate activation/quantizer
                    continue

                # Sanity check for extracted output scale: Should be clode to
                # range annotation scale
                if not np.allclose(scale, dy):
                    # Issue a warning to make the user aware of this
                    warnings.warn(
                        f"{self.__class__.__name__}: Extracted scale mismatch"
                        f" near {quant.name}: {scale} vs. {dy} (from RA)"
                    )

                # TODO: Padding in case *all* channels end up with fewer than
                #  2 ** bits - 1 thresholds...

                # Create new value information for the thresholds tensor
                threshold_tensor = oh.make_tensor_value_info(
                    # Create a unique name for this new tensor
                    model.make_new_valueinfo_name(),
                    # Container type is float
                    TensorProto.FLOAT,
                    # Get the tensor shape from the numpy array
                    thresholds.shape
                )
                # Insert the thresholds tensor information into the graph
                graph.value_info.append(threshold_tensor)
                # Insert the calculated thresholds as initializer into the
                # graph
                model.set_initializer(threshold_tensor.name, thresholds)

                # Create a multi-threshold operation node to replace the
                # quantized activation function
                multi_threshold = oh.make_node(
                    # MultiThreshold optype from QONNX
                    op_type="MultiThreshold",
                    # This operator is handled and implemented by QONNX
                    domain="qonnx.custom_op.general",
                    # Inputs to the node: Connect to the original input and
                    # the newly created thresholds tensor
                    inputs=[inp, threshold_tensor.name],
                    # Outputs of the node: Connect to a new intermediate
                    # tensor
                    outputs=[model.make_new_valueinfo_name()],
                    # Derive the name of the output datatype based on
                    # signedness and number of bits required
                    out_dtype=f"INT{bits}" if signed else f"UINT{bits}",
                    # If the output is signed, a bias is required to shift
                    # the unsigned threshold counting to the signed output
                    # range
                    out_bias=float(
                        (- 2 ** (bits - 1) + narrow) if signed else 0),
                    # Set the data layout inferred or inherited from the input
                    data_layout="".join(layout)
                )

                # Create new value information for the output scale tensor
                scale_tensor = oh.make_tensor_value_info(
                    # Create a unique name for this new tensor
                    model.make_new_valueinfo_name(),
                    # Container type is float
                    TensorProto.FLOAT,
                    # Get the tensor shape from the numpy array
                    dy.shape
                )
                # Insert the output scale tensor information into the graph
                graph.value_info.append(scale_tensor)
                # Insert the scale as initializer into the graph
                model.set_initializer(scale_tensor.name, dy)
                # Create a Mul node taking the scale factor for converting
                # the quantized output back to floating-point
                mul = oh.make_node(
                    # Elementwise multiplication from the ONNX domain
                    op_type="Mul",
                    # Connect to the intermediate tensor produced by the
                    # multi-threshold and to the scale of the quantizer
                    inputs=[multi_threshold.output[0], scale_tensor.name],
                    # Produce another intermediate tensor
                    outputs=[model.make_new_valueinfo_name()],
                )

                # Create new value information for the output bias tensor
                bias_tensor = oh.make_tensor_value_info(
                    # Create a unique name for this new tensor
                    model.make_new_valueinfo_name(),
                    # Container type is float
                    TensorProto.FLOAT,
                    # Get the tensor shape from the numpy array
                    range_info[out].bias.shape
                )
                # Insert the output bias tensor information into the graph
                graph.value_info.append(bias_tensor)
                # Insert the scale as initializer into the graph
                model.set_initializer(bias_tensor.name, bias)
                # Create an Add node taking the bias for converting the
                # quantized output back to floating-point
                add = oh.make_node(
                    # Elementwise addition from the ONNX domain
                    op_type="Add",
                    # Connect to the intermediate tensor produced by the
                    # scale multiplication and to the bias of the quantizer
                    inputs=[mul.output[0], bias_tensor.name],
                    # Connect to the original output
                    outputs=[out],
                )
                # Insert the new nodes into the graph
                graph.node.insert(index, multi_threshold)
                graph.node.insert(index + 1, mul)
                graph.node.insert(index + 2, add)

                # Remove the subgraph originally representing the quantized
                # chain of operators
                for n in subgraph:
                    graph.node.remove(n)

                # The graph has been modified and thus the transformation
                # needs to be applied again
                graph_modified = True
                # To allow the graph to "recover" after adding/removing
                # nodes and tensors, break her to do cleanup and redo
                # annotations
                break
        # Redo datatype and shape annotations as we have just remove and added
        # nodes as well as connecting and parameter tensors
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        # Remove potential unit scale and zero bias inserted following the
        # thresholds
        model = model.transform(RemoveIdentityOps())
        # Return the transformed model and indicate whether the transformation
        # needs to be applied again
        return model, graph_modified
