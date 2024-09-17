# Copies (deep-copies) python objects
import copy
# QONNX wrapper of ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper
# If we have a convolution with a bias tensors input, QONNX and later FINN
# expect the bias to be expressed as a standalone Add node following the Conv
# node.
from qonnx.transformation.extract_conv_bias import ExtractBiasFromConv
# Collapses chains of constants into a single constant operation or even
# initializer tensors.
from qonnx.transformation.fold_constants import FoldConstants
# QONNX graph transformations for renaming and cleaning up
from qonnx.transformation.general import (
    Transformation,
    GiveUniqueNodeNames,
    GiveReadableTensorNames,
    ConvertDivToMul,
    ConvertSubToAdd
)
# QONNX graph transformations for annotating the graph with datatype and shape
# information
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
# FINN dataflow builder configuration
from finn.builder.build_dataflow_config import (
    VerificationStepType, DataflowBuildConfig
)
# FINN verification after build/graph transformation steps
from finn.builder.build_dataflow_steps import verify_step
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
# Transposes the initializer tensors of a Quant node instead of having a
# standalone Transpose following
from qonnx.transformation.quant_constant_folding import \
    FoldTransposeIntoQuantInit
# Range information structure for seeding the range analysis for converting
# quantized activations to MultiThreshold
from qonnx.util.range_analysis import RangeInfo

# Folds quantizers into weight tensor initializers, needed for lowering
# convolutions to MatMuls
from finn.transformation.qonnx.fold_quant_weights import FoldQuantWeights

# Converts quantized activation functions to MultiThresholds based on range
# analysis
from quant_activation_to_multithreshold import QuantActivationToMultiThreshold


# Sets the data layout of the model input tensors
def set_input_data_layouts(layouts: list[str]):
    # Wrap the actual transformation/build step function
    def step_set_input_data_layouts(
            model: ModelWrapper, _: DataflowBuildConfig
    ):
        # Run over all graph inputs joined to the layout annotations given as
        # parameter
        for inp, layout in zip(model.graph.input, layouts):
            # Set the layout annotation
            model.set_tensor_layout(inp.name, list(layout))
        # Return the transformed model
        return model

    # Return the wrapped build step function
    return step_set_input_data_layouts



# Lowering transformations converting Conv to MatMul, BatchNorm to affine, etc.
def step_lower_conv_and_batch_norm(model: ModelWrapper, _: DataflowBuildConfig):
    # Compose all the lowering transformations
    return model.transform(ComposedTransformation([
        # Annotate the graph with shape and data type information
        InferShapes(),
        InferDataTypes(),
        # Moves the bias input to the Conv operator as a separate Add node
        # behind the Conv node
        ExtractBiasFromConv(),
        # Need to do some constant and weight folding first
        FoldConstants(),
        FoldTransposeIntoQuantInit(),
        FoldQuantWeights(),
        # Annotate the graph with shape and data type information
        InferShapes(),
        InferDataTypes(),
        # # Converts Conv layers to MatMul
        LowerConvsToMatMul(),
        # Converts BatchNorm to affine scale and bias
        BatchNormToAffine(),
    ]))


# Converts quantized activation functions to MultiThreshold instances based on
# range analysis. Also does various cleanup and lowering transformations as part
# of the range analysis.
def quant_activation_to_multithreshold(range_info: RangeInfo):
    # Wrap the actual transformation/build step function
    def step_quant_activation_to_multithreshold(
            model: ModelWrapper, cfg: DataflowBuildConfig
    ):
        # Add shape and datatype annotations throughout all the graph
        model = model.transform(InferDataTypes())
        model = model.transform(InferShapes())

        # Convert all suitable quantized activation functions to MultiThresholds
        model = model.transform(QuantActivationToMultiThreshold(range_info))

        # If configured, run a verification of the transformed model on some sample
        # inputs
        if (VerificationStepType.QONNX_TO_FINN_PYTHON in
                cfg._resolve_verification_steps()):  # noqa
            verify_step(
                model, cfg, "to_multithreshold_python", need_parent=False
            )

        # Return the transformed model
        return model

    # Return the wrapped build step function
    return step_quant_activation_to_multithreshold


# QONNX cleanup transformations
from qonnx.transformation.remove import RemoveIdentityOps
# Converts BatchNorm operation to affine transformation
from qonnx.transformation.batchnorm_to_affine import BatchNormToAffine
# Streamlining transformation: This is a collection of various transformations
from finn.transformation.streamline import (
    ConvertSignToThres, RoundAndClipThresholds
)
# Fuse/Absorb operations
from finn.transformation.streamline.absorb import (
    AbsorbAddIntoMultiThreshold,
    AbsorbSignBiasIntoMultiThreshold,
    FactorOutMulSignMagnitude,
    AbsorbMulIntoMultiThreshold,
    Absorb1BitMulIntoMatMul,
    Absorb1BitMulIntoConv
)
# Reorder operations
from finn.transformation.streamline.reorder import (
    MoveMulPastFork,
    MoveScalarLinearPastInvariants,
    MoveMulPastMaxPool,
    MoveAddPastMul,
    MoveScalarAddPastMatMul,
    MoveAddPastConv,
    MoveScalarMulPastMatMul,
    MoveScalarMulPastConv,
    MoveScalarLinearPastSplit,
    MoveTransposePastSplit
)
# Collapse consecutive operations of the same type
from finn.transformation.streamline.collapse_repeated import (
    CollapseRepeatedMul,
    CollapseRepeatedAdd
)
# FINN transformation converting ONNX nodes to hardware custom operators
from finn.transformation.fpgadataflow.convert_to_hw_layers import (
    InferElementwiseBinaryOperation, InferSplitLayer
)
# Stream replication for outputs with multiple consumers
from finn.transformation.fpgadataflow.replicate_stream import (
    InferReplicateStream
)


# Composes graph transformations such that each individual transformation as
# well as the whole sequence is applied exhaustively
class ComposedTransformation(Transformation):
    # Initializes the transformation given a list of transformations
    def __init__(self, transformations: list[Transformation]):
        # Initialize the transformation base class
        super().__init__()
        # Register the list of transformations to be applied in apply()
        self.transformations = transformations

    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all transformations to be applied
        for transformation in self.transformations:
            # Start each transformation on a deep copy of the model to mimic the
            # behavior of ModelWrapper.transform()
            model = copy.deepcopy(model)
            # Exhaustively apply the transformation until it no longer modifies
            # the graph
            while True:
                # Apply the transformation once, reporting back whether any node
                # or pattern has been modified
                model, _graph_modified = transformation.apply(model)
                # Keep track whether the graph has been modified at least once
                graph_modified = graph_modified or _graph_modified
                # Break the loop if this transformation did not change anything
                if not _graph_modified:
                    break
            # Apply the cleanup transformations of the ModelWrapper
            model.cleanup()
            # Apply some further cleanup transformations to the model graph
            # removing some clutter and keeping all names readable and ordered
            # at any time
            model = model.transform(RemoveIdentityOps())
            model = model.transform(GiveUniqueNodeNames())
            model = model.transform(GiveReadableTensorNames())
            model = model.transform(InferDataTypes())
        # Return the transformed model and indicate whether the graph actually
        # has been transformed by at least one transformation so the whole
        # sequence of transformations will be reapplied
        return model, graph_modified


# Moves constant elementwise multiplication past another joining multiplication
class MoveConstMulPastJoinMul(Transformation):
    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Applies to Mul operation types
            if node.op_type == "Mul":
                # Currently does not handle fork- or join-nodes
                if model.is_fork_node(node) or model.is_join_node(node):
                    # Softly skip this node
                    continue
                # As this is not a fork-node, there can be at most one successor
                successor = model.find_direct_successors(node)
                # If Squeeze is the final operation in the graph, there might
                # be no successor
                if successor is None:
                    # Softly skip this node
                    continue
                # Now there is exactly one successor which needs to be extracted
                # from the list
                successor = successor[0]
                # Applies to Multiplications
                if successor.op_type in {"Mul"}:
                    # Applies only if the second multiplication is a join-node
                    if model.is_join_node(successor):
                        # Get names of all tensors involved in connecting the
                        # nodes
                        inp = node.input[0]  # noqa: Duplicate
                        mid = node.output[0]
                        out = successor.output[0]
                        # Need to match the correct input of the joining second
                        # multiplication
                        for i, name in enumerate(successor.input):
                            # If the successors input currently matches the
                            # intermediate tensors, this input needs to be
                            # rewired
                            if name == mid:
                                # Rewire the graph to feed original into the
                                # second Mul node first
                                successor.input[i] = inp
                                # Note: Do not break here as it is perfectly
                                # legal to connect the same tensor multiple
                                # times to different inputs
                        # Repurpose the middle tensor for the output of the
                        # second Mul
                        successor.output[0] = mid
                        # The first Mul operator now gets the middle tensor as
                        # its input
                        node.input[0] = mid
                        # The first Mul now produces the original output tensor
                        node.output[0] = out
                        # Delete the shape annotation of the connecting tensors
                        # to be re-done later
                        model.set_tensor_shape(mid, None)
                        model.set_tensor_shape(out, None)
                        # Track whether the graph has been modified, never
                        # resets to False
                        graph_modified = True
                        # Break the loop after deleting shape annotations to
                        # immediately re-do these before changing the next
                        # operator
                        break
        # Redo datatype and shape annotations
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        # Return the transformed model and indicate whether the transformation
        # needs to be applied again
        return model, graph_modified


# Groups node inputs by dynamic vs. initializer category
from finn.transformation.streamline.absorb import group_inputs_by_category


# Moves elementwise multiplication past elementwise addition if one input to
# each of the operators is a known constant
# Note: Reverse of MoveAddPastMul
class MoveMulPastAdd(Transformation):
    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Applies to Mul operation types
            if node.op_type == "Mul":
                # Currently does not handle fork- or join-nodes
                if model.is_fork_node(node) or model.is_join_node(node):
                    # Softly skip this node
                    continue
                # As this is not a fork-node, there can be at most one successor
                successor = model.find_direct_successors(node)
                # If Squeeze is the final operation in the graph, there might
                # be no successor
                if successor is None:
                    # Softly skip this node
                    continue
                # Now there is exactly one successor which needs to be extracted
                # from the list
                successor = successor[0]
                # Applies to additions
                if successor.op_type in {"Add"}:
                    # The addition may not join as we need to know the second
                    # input
                    if not model.is_join_node(successor):
                        # Get the constant initializer tensors for both
                        # operations: y = s * x + b
                        _, s_name = group_inputs_by_category(node, model)
                        _, b_name = group_inputs_by_category(successor, model)
                        # Skip if either node has no constant initializer
                        if not s_name or not b_name:
                            # Skip without warning ok?
                            continue
                        # There must be exactly one constant per operations
                        assert len(s_name) == 1, \
                            f"To many constant inputs for {node}"
                        assert len(b_name) == 1, \
                            f"To many constant inputs for {successor}"
                        # Now read the initializer tensors
                        s = model.get_initializer(*s_name)
                        b = model.get_initializer(*b_name)
                        # Update the addition initializer according to the
                        # distributive law
                        model.set_initializer(*b_name, b / s)
                        # Get names of all tensors involved in connecting the
                        # nodes
                        inp = node.input[0]  # noqa: Duplicate
                        mid = node.output[0]
                        out = successor.output[0]
                        # Rewire the graph to feed original input into the
                        # Add node first
                        successor.input[0] = inp
                        # Repurpose the middle tensor for the output of the Add
                        successor.output[0] = mid
                        # The Mul operator now gets the middle tensor as its
                        # input
                        node.input[0] = mid
                        # Mul now produces the original output tensor
                        node.output[0] = out
                        # Delete the shape annotation of the connecting tensors
                        # to be re-done later
                        model.set_tensor_shape(mid, None)
                        model.set_tensor_shape(out, None)
                        # Track whether the graph has been modified, never
                        # resets to False
                        graph_modified = True
                        # Break the loop after deleting shape annotations to
                        # immediately re-do these before changing the next
                        # operator
                        break
        # Redo datatype and shape annotations
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        # Return the transformed model and indicate whether the transformation
        # needs to be applied again
        return model, graph_modified


# Custom streamlining step to deal with composite activation functions
def step_streamline(model: ModelWrapper, _: DataflowBuildConfig):
    # Compose streamlining transformations similar to the default streamlining
    # transformation but applied more exhaustively.
    return model.transform(
        ComposedTransformation([
            # Default FINN Streamlining transformations (except rounding and
            # clipping of thresholds) followed by some custom Mul-moving
            # transformations.
            ComposedTransformation([
                ConvertSubToAdd(),
                ConvertDivToMul(),
                BatchNormToAffine(),
                ConvertSignToThres(),
                MoveMulPastMaxPool(),
                AbsorbSignBiasIntoMultiThreshold(),
                MoveScalarLinearPastInvariants(),
                MoveAddPastMul(),
                MoveScalarAddPastMatMul(),
                MoveAddPastConv(),
                MoveScalarMulPastMatMul(),
                MoveScalarMulPastConv(),
                MoveAddPastMul(),
                CollapseRepeatedAdd(),
                CollapseRepeatedMul(),
                MoveMulPastMaxPool(),
                AbsorbAddIntoMultiThreshold(),
                FactorOutMulSignMagnitude(),
                AbsorbMulIntoMultiThreshold(),
                Absorb1BitMulIntoMatMul(),
                Absorb1BitMulIntoConv(),
                # Move around Mul operations to get Muls and Adds into a better
                # place for absorbing into MultiThresholds in the next round of
                # streamlining starting from the top.
                # TODO: Maybe move these three to the top of the streamlining
                #  composition, as the primary intention is to re-absorb some
                #  biases back into MultiThresholds which should probably happen
                #  rather early on after export and conversion?
                MoveMulPastFork(),
                # Note: This brings constant Muls (i.e., quantizer scales to be
                # removed) forward through joining Muls (i.e., those ending up
                # as actual hardware operators).
                MoveConstMulPastJoinMul(),
                # Note: This is essential to allow some Add operations to be
                # absorbed by the next round's AbsorbSignBiasIntoMultiThreshold
                MoveMulPastAdd(),
                # Streamlining for Split operations
                # TODO: Add Concat streamlining here as well
                MoveScalarLinearPastSplit(),
                MoveTransposePastSplit()
            ]),
            # Only round and clip after all streamlining transformations have
            # been applied exhaustively.
            # Note: Might still enable another round of streamlining.
            RoundAndClipThresholds(),
        ])
    )


# Function running the transformations to convert elementwise binary operations
# to their hardware implementations
def step_convert_elementwise_binary_to_hw(model: ModelWrapper, _):
    # Convert elementwise operations to hardware operators
    #   Note: Do not convert the final Mul operator at the output
    return model.transform(InferElementwiseBinaryOperation(
        InferElementwiseBinaryOperation.reject_floats
    ))


# Function running the InferReplicateStream transformation
def step_replicate_streams(model: ModelWrapper, _):
    # Properly replicate the stream feeding the query, key and value projections
    return model.transform(InferReplicateStream())


# Function converting the Split operator to hardware custom operation
def step_convert_split_to_hw(model: ModelWrapper, _):
    return model.transform(InferSplitLayer())
