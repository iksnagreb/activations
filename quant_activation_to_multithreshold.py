# Numpy for loading and comparing the verification input/output
import numpy as np
# QONNX wrapper of ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper
# Converts ONNX graph nodes to QONNX custom-ops if possible
from qonnx.custom_op.registry import getCustomOp
# QONNX base class for all graph transformations
from qonnx.transformation.general import Transformation
# QONNX graph transformations for inferring datatypes and shapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes

# Model cleanup transformations as pre-processing for range analysis
from qonnx.util.cleanup import cleanup_model

# Range analysis to generate input ranges and scales use to enumerate inputs and
# outputs of quantized activation functions to generate thresholds
from qonnx.util.range_analysis import range_analysis
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
    "Ceil", "Round", "Floor", "Relu", "LeakyRelu", "Sigmoid", "Identity", "Tanh", "Erf", "Celu", "Elu", "Exp", "Log"
}


# Matches graph pattern of monotonic activation function followed by quantizer
def match_quant_monotonic_activation(node: NodeProto, model: ModelWrapper):
    # The node optype must be one of the supported activations specified above
    if node.op_type in SUPPORTED_MONOTONIC_ACTIVATIONS:
        # Now get the list of successor node and make sure there is exactly one
        if len(successors := model.find_direct_successors(node)) == 1:
            # If this is a quantizer, the node is part of a quantized monotonic
            # activation
            # TODO: Add BipolarQuant currently still missing full RA support
            if successors[0].op_type in {"Quant"}:
                # Return the two nodes of the pattern: The activation function
                # and the following quantizer
                return node, successors[0]
    # A standalone quantizer can be interpreted as a quantized activation
    # function as well, implicitly preceded by Identity
    # TODO: Add BipolarQuant currently still missing full RA support
    if node.op_type in {"Quant"}:
        # Return only the quantizer node and mark the activation function as
        # missing
        return None, node
    # Optype or number of successors did not match
    return None, None


# Converts supported quantized activation functions to MultiThreshold
class QuantActivationToMultiThreshold(Transformation):
    # TODO: Add configuration options setting the fall-back step size "dx" for
    #  enumerating non-integer input ranges and limiting the maximum output
    #  bit-width of quantizers to be considered, i.e., FINN's
    #  max_multithreshold_bit_width setting.

    # TODO: Add a mechanism to specify a known input range to start the range
    #  analysis: This might improve the threshold generation but is also
    #  necessary for some functions, like the log, which are only defined on
    #  parts of the real numbers.

    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        # Model needs to be cleaned up, preserving the quantizers
        # Note: Somehow the non-preserving cleanup in range_analysis causes
        # problems with constant folding, losing the quantization of weights...
        model = cleanup_model(model, preserve_qnt_ops=True)
        # Run range analysis on the model after converting Conv and Gemm to
        # MatMul and BatchNorm to affine transformations (lowering)
        range_info, model = range_analysis(
            model, report_mode="range", scaled_int=True, lower_ops=True
        )

        # Creates a tensor according to the value info
        def output_placeholder(name):
            return valueinfo_to_tensor(model.get_tensor_valueinfo(name))

        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Try to match any quantized monotonic activation function for
            # conversion
            act, quant = match_quant_monotonic_activation(node, model)
            # Skip if no quantizer is present
            if quant is None:
                # Softly skip without warning, transformation just does not
                # apply here
                continue
            # Name of the input and output tensor of the quantized activation
            inp, out = None if act is None else act.input[0], quant.output[0]
            # If no separate activation function is given, get the input tensor
            # from the quantizer node as well
            inp = quant.input[0] if inp is None else inp

            # Create a wrapper function for evaluating the quantized activation
            # subgraph handling context construction and output extraction
            def f(x):  # noqa: Shadows x from below, is ok
                # Collect the input and output tensors of the subgraph
                inputs = [*([] if act is None else act.input), *quant.input]
                outputs = [*([] if act is None else act.output), *quant.output]
                # Prepare execution context for executing the subgraph nodes
                ctx = {
                    # Prepare the inputs potentially from initializers
                    **{i: model.get_initializer(i) for i in inputs},
                    # Prepare the outputs by placeholders
                    **{o: output_placeholder(o) for o in outputs},
                    # Insert the input into the execution context
                    inp: x
                }
                # If present, execute the activation function putting its
                # results back into the context
                if act is not None:
                    execute_node(act, ctx, model.graph)
                # Now execute the quantizer, which is always present
                execute_node(quant, ctx, model.graph)
                # Extract the output from the execution context
                return ctx[out]

            # The input and output to the activation-quantizer combination must
            # be described by the range information analyzed above to be able to
            # enumerate the input/output levels for generating thresholds
            if inp in range_info and out in range_info:
                # Conversion only applies if the input provides an integer range
                # which can be enumerated
                # TODO: This is not really true, currently this just prevents
                #  excessively long runtimes for enumerating all floats...
                if range_info[inp].int_range is not None:
                    # The output is produced by a quantizer, thus we can always
                    # assume the integer range
                    y0, y1 = range_info[out].int_range
                    # Get the scale and bias for converting the integer ranges
                    # at the input and output to floats
                    scale = range_info[out].scale
                    bias = range_info[out].bias
                    # Start enumerating the threshold levels at the lower bound
                    # of the output range
                    level = y0
                    # Collect all thresholds as a list
                    thresholds = []
                    # Input range minimum and maximum serve as initial
                    # values for the interval bounds
                    (x, x1), dx = range_info[inp].range, range_info[inp].scale
                    # If the input range does not have a know scale for
                    # enumerating the thresholds, set some default
                    # TODO: Make this configurable
                    dx = 1e-3 if dx is None else dx
                    # Enumerate the output levels, each will yield a set of
                    # thresholds covering all dimensions
                    while np.any(level < y1):
                        # Evaluate the objective function once before entering
                        # the loop
                        fx = f(x) <= (scale * level - bias)
                        # Run over all input levels
                        while np.any(fx) and np.any(x <= x1):
                            # Evaluate the objective function "f(x) <= level" at
                            # the current candidate input
                            fx = f(x) <= (scale * level - bias)
                            # Advance those inputs which are below the output
                            # level to the next position
                            x = np.where(fx, x + dx, x)
                            # Clip at the upper bound of the input range
                            x = np.where(x >= x1, x1, x)
                        # The thresholds for the level are now stored in x
                        # Note: The actual threshold is halfway between this and
                        # the previous step, i.e., -0.5 * dx
                        thresholds.append(x - 0.5 * dx)
                        # Move to the next output level along all dimensions and
                        # clip values at the maximum of the range per dimension
                        level = np.where(level >= y1, y1, level + 1)

                    # Get the output bit-with to be produced by the quantizer,
                    # which determines how many thresholds are needed
                    # TODO: Add BipolarQuant currently still missing full RA
                    #  support
                    # TODO: BipolarQuant implicitly has bits = 1 without
                    #  initializer tensor
                    bits = int(model.get_initializer(quant.input[3]))
                    # Need to pad the thresholds such that there are 2^bits - 1
                    padding = 2 ** bits - 1 - len(thresholds)
                    # Fill up thresholds from the left repeating the smallest
                    # threshold
                    thresholds = [*(padding * [thresholds[0]]), *thresholds]

                    # Stack thresholds list into the thresholds tensor along the
                    # final dimension, i.e., steps last
                    thresholds = np.stack(thresholds, axis=-1)
                    # Thresholds must be expressible in the two last dimensions
                    thresholds = thresholds.reshape(*thresholds.shape[-2:])

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

                    # Check whether this is a signed quantizer
                    signed = getCustomOp(quant).get_nodeattr("signed")
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
                        out_bias=float(- 2 ** (bits - 1) if signed else 0)
                    )

                    # Create new value information for the output scale tensor
                    scale_tensor = oh.make_tensor_value_info(
                        # Create a unique name for this new tensor
                        model.make_new_valueinfo_name(),
                        # Container type is float
                        TensorProto.FLOAT,
                        # Get the tensor shape from the numpy array
                        scale.shape
                    )
                    # Insert the output scale tensor information into the graph
                    graph.value_info.append(scale_tensor)
                    # Insert the scale as initializer into the graph
                    model.set_initializer(scale_tensor.name, scale)
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
                        bias.shape
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
                    # Remove the optional activation function
                    if act is not None:
                        graph.node.remove(act)
                    # Always remove the quantizer node
                    graph.node.remove(quant)
                    # The graph has been modified and thus the transformation
                    # needs to be applied again
                    graph_modified = True
                    # To allow the graph to "recover" after adding/removing
                    # nodes and tensors, break her to do cleanup and redo
                    # annotations
                    break
        # Redo datatype and shape annotations
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        # Return the transformed model and indicate whether the transformation
        # needs to be applied again
        return model, graph_modified
