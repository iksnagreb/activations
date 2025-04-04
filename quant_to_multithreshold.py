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
from qonnx.util.range_analysis import range_analysis, RangeInfo, \
    unbroadcast_tensor
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
        if n.op_type in FUSIBLE_OPS:
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


# Converts supported quantized activation functions to MultiThreshold
class QuantToMultiThreshold(Transformation):
    # Filter to reject the global input quantizer from conversion...
    @staticmethod
    def reject_input_quant(model: ModelWrapper, node: NodeProto):
        # If node is not a quantizer, reject it...
        if not node.op_type in SUPPORTED_QUANTIZERS:
            return False
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
            # If node is not a quantizer, reject it...
            if not node.op_type in SUPPORTED_QUANTIZERS:
                return False
            # Check whether the quantizer represents the output with too many
            # bits
            return int(model.get_initializer(node.input[3])) < bits

        # Return the filter function
        return _filter

    # Initializes the conversion by setting a seed range information for the
    # range analysis pass
    def __init__(self, range_info: RangeInfo = None, rescale=0.5,
                 quant_filter=None, assume_monotonic=False):
        # Initialize the Transformation super class
        super().__init__()
        # Store the seed range information
        self.range_info = range_info
        # Extra scale applied when enumerating the inputs
        self.rescale = rescale
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
                # the inputs, set some default. Sample at higher rate to avoid
                # aliasing and rounding effects. Strictly, the sampling theorem
                # does not apply here: This is neither band-limited (perfect
                # steps require infinite frequencies) nor continuous (floats are
                # not reals)
                dx = 2.5e-4 if dx is None else np.min(np.asarray(dx))
                # It does not make sense to go below the smallest representable
                # floating-point number...
                dx = max(self.rescale * dx, np.finfo(np.float32).eps)

                # Collect list of candidate thresholds (will be filtered and
                # padded in post-processing)
                thresholds = []
                weights = []

                # Step size for cutting the range into interval sections for
                # threshold search: We want to balance the number of sections to
                # keep in memory while narrowing down the search space
                # Note: Do not cut sections smaller than the input quantization
                # step size
                steps = 1 if np.max(x1 - x0) / 2 ** 22 <= dx else 2 ** 22

                # Enumerate interval sections of the input range to narrow down
                # the search to ranges which actually contain thresholds
                sections = np.expand_dims(1 / steps * np.arange(steps + 1), -1)
                # Actually generate the inputs at the interval section bounds
                xs = x0 + sections * (x1 - x0)
                # Evaluate the subgraph on the bounds of these sections.
                # Note: _evaluate_subgraph is non-batched...
                ys = _evaluate_subgraph(subgraph, model, xs)
                # Iterators on the sections pairing up the upper/lower bounds
                xs = zip(xs, xs[1:])
                ys = zip(ys, ys[1:])

                # Collect all sections for processing finding the thresholds
                # Note: Putting in further assumptions we can narrow the search
                # space down when collecting these
                sections = []

                # Insert sections into the list for further processing, applying
                # some filter conditions: Currently only monotonicity
                for (x0, x1), (y0, y1) in zip(xs, ys):
                    # Keep section for checking if monotonic -> np.any(y0 != y1)
                    if not self.assume_monotonic or np.any(y0 != y1):
                        sections.append((x0, x1))

                # Search for thresholds, i.e., steps, in each of the sections
                for x0, x1 in tqdm(sections, desc=f"Thresholds {node.name}"):
                    # At most process 2 ** 12 inputs in parallel, sections might
                    # be smaller - which is fine - limit parallelism to avoid
                    # excessive memory utilization
                    steps = min(2 ** 12, int(np.ceil(np.max(x1 - x0) / dx)))
                    # Span the first steps of the input range
                    xs = np.linspace(x0 - dx, x0 + steps * dx, steps)
                    # Make sure all sample points are at multiples of the scale
                    xs = dx * np.round(xs / dx)
                    # Clip within (x0, x1) if this range is smaller than steps
                    xs = np.clip(xs, x0, x1)
                    # Keep searching thresholds while we are within the range
                    while np.any(xs <= x1):
                        # Even more potential to speed this up assuming
                        # monotonicity here... Check bounds of xs first...
                        y0, y1 = evaluate_subgraph(subgraph, model, xs[[0, -1]])
                        # Checking if monotonic -> np.any(y0 != y1)
                        if not self.assume_monotonic or np.any(y0 != y1):
                            # Evaluate the function on the range in batch mode
                            ys = evaluate_subgraph(subgraph, model, xs)
                            # Steps are edges of the output, i.e., where the
                            # derivative is non-zero
                            edges = convolve1d(
                                ys, np.array([+1, -1]), axis=0, mode="nearest"
                            )
                            # Thresholds are the xs corresponding to the edges,
                            # i.e., where the convolution detected a step
                            thresholds.append(xs[np.unique(np.where(edges)[0])])
                            # Step sizes at detected thresholds, these should be
                            # integer multiples of the quantization scale
                            weights.append(edges[np.unique(np.where(edges)[0])])
                        # Advance to the next steps points in the range
                        xs = dx * np.round(xs / dx + (steps - 1))
                        # Clip within range to not find out-of-bounds thresholds
                        xs = np.clip(xs, x0 - dx, x1 + dx)

                # Pack all collected thresholds into a single array
                thresholds = np.concatenate(thresholds, axis=0)
                weights = np.concatenate(weights, axis=0)

                # Make sure these are single precision floats (parts of the
                # calculations above are carried out in double precision)
                thresholds = thresholds.astype(np.float32)
                weights = weights.astype(np.float32)

                # Get the quantizer node terminating the chain of operators as
                # this holds some extra information such as the target bit-width
                quant = subgraph[-1]

                # Check whether this is a signed quantizer
                signed = getCustomOp(quant).get_nodeattr("signed")
                narrow = int(getCustomOp(quant).get_nodeattr("narrow"))

                # Sanitize thresholds by rounding to the RA annotated scale
                thresholds = dx * np.ceil(thresholds / dx + 0.5)

                # Get the output bit-with to be produced by the quantizer,
                # which determines how many thresholds are needed
                bits = int(model.get_initializer(quant.input[3]))

                # Move the first axis to the end to have (..., Num) layout,
                # where Num is the number of thresholds found
                thresholds = np.moveaxis(thresholds, 0, -1)
                weights = np.moveaxis(weights, 0, -1)
                # Force the threshold tensor to (C, Num) shape
                # Note could be (1, 1, ..., 1, C, Num) shape before
                thresholds = thresholds.reshape(*thresholds.shape[-2:])
                weights = weights.reshape(*weights.shape[-2:])

                # Factor out the quantization scale from the weights, turning
                # them into integer step sizes
                weights = np.floor(weights / dy).astype(np.int32)

                # Initialize a mask to remove all weights/thresholds
                mask = np.ones_like(weights, dtype=bool)
                # Filter to keep the first unique occurrence of each threshold
                for i, t in enumerate(thresholds):
                    mask[i, np.unique(t, return_index=True)[1]] = False
                # Remove duplicate thresholds by setting the corresponding
                # weight to zero
                weights[np.where(mask)] = 0

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

                # We need the annotated integer range to detect and correct
                # stuck channels
                (y0, y1) = range_info[out].int_range

                # Reduce the bounds to per-channel bounds just as for the
                # simulation above
                y0 = np.min(y0, axis)
                y1 = np.max(y1, axis)

                # Derive the expected minimum of the integer output range
                # according to the bit-width of the quantizer, respect narrow...
                ymin = - 2 ** (bits - 1) + narrow if signed else 0

                # Add some padding from the left such that the thresholds
                # produce outputs starting from where the channel is stuck
                padding = (y0 - ymin).astype(np.int64)

                # No negative padding
                padding = np.where(padding < 0, 0, padding)

                # Find the stuck channels, which are those with singular output
                # range
                stuck = np.where(y0 == y1)[0]

                # Check if there even is a stuck channel and warn, even if we
                # can handle this now...
                if np.any(stuck):
                    # Issue a warning to make the user aware of this
                    warnings.warn(
                        f"{self.__class__.__name__}: Stuck channel {quant.name}"
                        f" Thresholds for channels {list(stuck)} will be padded"
                    )

                # TODO: Maybe restrict the padding to explicitly stuck channels?

                # Add back the dimension lost by reducing to be compatible with
                # the (C, N) layout
                padding = np.expand_dims(padding, axis=-1)
                # Add padding weights from the left to shift the function
                # upwards
                weights = np.concatenate((padding, weights), axis=-1)
                # Derive threshold padding from the right
                padding = np.full_like(padding, -np.inf, dtype=np.float32)
                # Add padding to the thresholds
                thresholds = np.concatenate((padding, thresholds), axis=-1)

                # Count how many thresholds there are per channel (counting both
                # positive and negative directions) to find out how many are
                # missing
                padding = 2 ** bits - 1 - np.sum(np.abs(weights), axis=-1)

                # No negative padding
                padding = np.where(padding < 0, 0, padding)

                # Add back the dimension lost by reducing to be compatible with
                # the (C, N) layout. Padding should respect narrow range...
                padding = np.expand_dims(padding - narrow, axis=-1)
                # Add padding weights from the right to fill-up the function
                # upwards
                weights = np.concatenate((weights, padding), axis=-1)
                # Derive threshold padding from the right
                padding = np.full_like(padding, +np.inf, dtype=np.float32)
                # Add padding to the thresholds
                thresholds = np.concatenate((thresholds, padding), axis=-1)

                # Steps of size >1 should be expressed as repeated steps of
                # size =1 to comply with the hardware backend
                # Unpacks a list of weights to unit weights yielding the same
                # sum
                def unpack_weights(ws):
                    # Keep the sign of the weight but repeat as many 1s
                    return [
                        np.sign(w) * 1 for w in ws for _ in range(np.abs(w))
                    ]

                # Unpacks the threshold list according to the weights repeating
                # the thresholds weight-many times
                def unpack_thresholds(ts, ws):
                    # Repeat the threshold t w-many times
                    return [
                        t for t, w in zip(ts, ws) for _ in range(np.abs(w))
                    ]

                # Unpack the thresholds to unit steps
                thresholds = np.asarray([
                    unpack_thresholds(*tws) for tws in zip(thresholds, weights)
                ])
                # Unpack the weight list to unit step weights
                weights = np.asarray([unpack_weights(ws) for ws in weights])

                # Optimization: Unbroadcast from per-channel to per-tensor
                # thresholds if all channels turn out to be identical
                thresholds = unbroadcast_tensor(thresholds)
                weights = unbroadcast_tensor(weights)

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
                model.set_initializer(bias_tensor.name, range_info[out].bias)
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
