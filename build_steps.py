# Copies (deep-copies) python objects
import copy
# Numpy for loading and comparing the verification input/output
import numpy as np
# YAML for loading experiment configurations
import yaml

# QONNX wrapper of ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper
# Range information structure for seeding the range analysis for converting
# quantized activations to MultiThreshold
from qonnx.util.range_analysis import RangeInfo

# QONNX graph transformations for renaming and cleaning up
from qonnx.transformation.general import (
    GiveUniqueNodeNames,
    GiveReadableTensorNames,
    GiveUniqueParameterTensors,
    RemoveStaticGraphInputs,
    RemoveUnusedTensors,
)
# QONNX graph transformations for annotating the graph with datatype and shape
# information
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes

# If we have a convolution with a bias tensors input, QONNX and later FINN
# expect the bias to be expressed as a standalone Add node following the Conv
# node.
from qonnx.transformation.extract_conv_bias import ExtractBiasFromConv
# Converts BatchNorm operation to affine transformation
from qonnx.transformation.batchnorm_to_affine import BatchNormToAffine
# Converts Gemm operation to MatMul with extracted standalone bias op
from qonnx.transformation.gemm_to_matmul import GemmToMatMul
# Converts Conv to Im2Col and MatMul with extracted standalone bias op
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
# Transposes the initializer tensors of a Quant node instead of having a
# standalone Transpose following
from qonnx.transformation.quant_constant_folding import (
    FoldTransposeIntoQuantInit
)
# Collapses chains of constants into a single constant operation or even
# initializer tensors.
from qonnx.transformation.fold_constants import FoldConstants
# Folds quantizers into weight tensor initializers, needed for lowering
# convolutions to MatMuls
from finn.transformation.qonnx.fold_quant_weights import FoldQuantWeights
# FINN streamlining transformations absorbing tensors/nodes into others
from finn.transformation.streamline.absorb import (
    AbsorbSignBiasIntoMultiThreshold,
)
# FINN streamlining transformations removing nodes without real effect from the
# graph
from finn.transformation.streamline.remove import (
    RemoveIdentityTranspose,
    RemoveIdentityReshape
)
# FINN streamlining transformation collapsing repeated operations of the same
# kind
from finn.transformation.streamline.collapse_repeated import (
    CollapseRepeatedReshape
)
# Converts (infers) ONNX and QONNX nodes to FINN hardware CustomOps
from finn.transformation.fpgadataflow.convert_to_hw_layers import (
    InferElementwiseBinaryOperation,
    InferSplitLayer,
    InferConcatLayer,
    InferLookupLayer,
    InferVectorVectorActivation,
    InferReLUAsElementwiseMax,
    InferQuantAsFloat2Int
)
# Converts fork-nodes to ReplicateStream hardware operator
from finn.transformation.fpgadataflow.replicate_stream import (
    InferReplicateStream
)
# Standard QONNX to FINN conversion function
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.qonnx.quant_act_to_multithreshold import (
    default_filter_function_generator,
)
# QONNX quantization data types
from qonnx.core.datatype import DataType
# FINN dataflow builder configuration
from finn.builder.build_dataflow_config import (
    VerificationStepType, DataflowBuildConfig
)
# FINN verification after build/graph transformation steps
from finn.builder.build_dataflow_steps import verify_step

# Transformations preparing the operators for synthesis and simulation
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim

# Execute onnx model graphs from the dataflow parent for verification
from finn.util.test import execute_parent

# Transformation for exhaustively composing transformations
from qonnx.transformation.composed import ComposedTransformation

# FINN streamlining transformations reordering the graph
from finn.transformation.streamline.reorder import MoveMulPastAdd
# Custom st of streamlining transformations
from finn.transformation.streamline.streamline_plus import \
    StreamlinePlus as Streamline

# New Range Analysis based streamlining directly implemented in QONNX
from qonnx.transformation.streamline import Streamline as QONNXStreamline

# # Custom conversion from Quant to MultiThreshold
from quant_to_multithreshold import QuantToMultiThreshold


# Prepares the graph to be consumed by FINN:
# 1. Some graph cleanup removing unused tensors, nodes without effect and
#  folding constants, i.e., collapsing chains of operations on constant tensors
# 2. Lowers some "more complex" operations: converts Conv and Gemm to MatMul and
#  BatchNorm to Mul and Add operations followed by some necessary cleanup
# 3. Converts all QONNX Quant nodes to MultiThreshold operations which can
#  absorb scales and biases during streamlining
def prepare_graph(range_info: RangeInfo, streamline=True, thresholds=True):
    # Wrap the actual transformation/build step function
    def step_prepare_graph(model: ModelWrapper, cfg: DataflowBuildConfig):
        # Exhaustively apply the set of cleanup transformations
        model = model.transform(ComposedTransformation([
            # Adds shape and datatype annotations to all tensors in this graph
            InferDataTypes(),
            InferShapes(),
            # Cleanup the graph by removing redundant, unnecessary and constant
            # nodes and tensors and give unique names to everything remaining
            GiveUniqueNodeNames(),
            GiveReadableTensorNames(),
            RemoveStaticGraphInputs(),
            RemoveUnusedTensors(),
            GiveUniqueParameterTensors(),
            FoldConstants(),
            # Remove unnecessary shape and layout transformations
            RemoveIdentityReshape(),
            RemoveIdentityTranspose(),
            # Redo shape and datatype annotations after removing nodes and
            # tensors
            InferShapes(),
            InferDataTypes(),
        ]))
        # If configured, run a verification of the transformed model on some
        # sample inputs
        if (VerificationStepType.TIDY_UP_PYTHON in
                cfg._resolve_verification_steps()):  # noqa
            verify_step(
                model, cfg, "tidied_up_python", need_parent=False
            )
        # Exhaustively apply the lowering transformations
        model = model.transform(ComposedTransformation([
            # Moves the bias input to the Conv operator as a separate Add node
            # behind the Conv node
            ExtractBiasFromConv(),
            # Converts Gemm nodes to MatMul (+ bias)
            GemmToMatMul(),
            # Need to do some constant and weight folding first
            FoldConstants(),
            FoldTransposeIntoQuantInit(),
            FoldQuantWeights(),
            # Annotate the graph with shape and data type information
            InferShapes(),
            InferDataTypes(),
            # Converts Conv layers to MatMul
            LowerConvsToMatMul(),
            # Converts BatchNorm to affine scale and bias
            BatchNormToAffine(),
            # Annotate the graph with shape and data type information
            InferShapes(),
            InferDataTypes(),
        ]))
        # If configured, run a verification of the transformed model on some
        # sample inputs
        if (VerificationStepType.QONNX_TO_FINN_PYTHON in
                cfg._resolve_verification_steps()):  # noqa
            verify_step(
                model, cfg, "lowered_python", need_parent=False
            )

        # Optional streamlining step before (optionally) converting quantizers
        # to multi-thresholds
        if streamline:
            # Try the new QONNX Range Analysis based Streamlining to move scales
            # and biases already to their final place where they could be fused
            # into multi-thresholds
            model = model.transform(QONNXStreamline(range_info))
            # If configured, run a verification of the transformed model on some
            # sample inputs
            if (VerificationStepType.QONNX_TO_FINN_PYTHON in
                    cfg._resolve_verification_steps()):  # noqa
                verify_step(
                    model, cfg, "qonnx_streamlined_python", need_parent=False
                )

        # Optional conversion of fused chains of quantized activation functions
        # to multi-thresholds
        if thresholds:
            # Apply the quantizer to MultiThreshold conversion
            # Note: This is exhaustive as well as single .transform reapplies as
            # long as possible.
            model = model.transform(QuantToMultiThreshold(range_info))
            # If configured, run a verification of the transformed model on some
            # sample inputs
            if (VerificationStepType.QONNX_TO_FINN_PYTHON in
                    cfg._resolve_verification_steps()):  # noqa
                verify_step(
                    model, cfg, "quant_to_thresholds_python", need_parent=False
                )
            # Apply the standard FINN conversion step to convert the remaining
            # quantizers not yet covered by the new range analysis based method
            model = model.transform(ConvertQONNXtoFINN(
                filter_function=default_filter_function_generator(
                    cfg.max_multithreshold_bit_width
                )
            ))

        # Some extra cleanup steps which are covered by later streamlining, but
        # we might disable the streamlining but allways needs these...
        model = model.transform(ComposedTransformation([
            CollapseRepeatedReshape(),
            RemoveIdentityReshape()
        ]))

        # If configured, run a verification of the transformed model on some
        # sample inputs
        if (VerificationStepType.QONNX_TO_FINN_PYTHON in
                cfg._resolve_verification_steps()):  # noqa
            verify_step(
                model, cfg, "prepared_graph_python", need_parent=False
            )
        # Return the transformed model
        return model

    # Return the wrapped transformation step function
    return step_prepare_graph


# Applies the custom set of exhaustive streamlining transformations, also taking
# special topology like attention, residuals, splits and transposes into account
def step_streamline(model: ModelWrapper, cfg: DataflowBuildConfig):
    # These should not be applied exhaustively with the other streamlining
    # transformations to not end up in cycles.
    # Note: This is essential to allow some Add operations to be
    # absorbed by the next round's AbsorbSignBiasIntoMultiThreshold
    model = model.transform(MoveMulPastAdd())
    model = model.transform(AbsorbSignBiasIntoMultiThreshold())
    # Exhaustively apply the following set of transformations to streamline the
    # graph with the overall goal of collecting scales and biases in front of
    # MultiThreshold operations or, alternatively, at the end of the graph.
    # Note: Contains some sets of nested exhaustive transformations meant for
    # particular architectural patterns, e.g., residual topologies.
    model = model.transform(Streamline())
    # If configured, run a verification of the transformed model on some
    # sample inputs
    if (VerificationStepType.STREAMLINED_PYTHON in
            cfg._resolve_verification_steps()):  # noqa
        verify_step(
            model, cfg, "streamlined_python", need_parent=False
        )
    # Return the transformed model
    return model


# Function running the transformations to convert elementwise binary operations
# to their hardware implementations
def step_convert_elementwise_binary_to_hw(model: ModelWrapper, _):
    # Convert elementwise operations to hardware operators
    #   Note: Do not convert the final Mul operator at the output
    return model.transform(InferElementwiseBinaryOperation(
        InferElementwiseBinaryOperation.reject_output_dequant
    ))


# Function running the transformations to convert float elementwise operations
# to their hardware implementations
def step_convert_floats_to_hw(model: ModelWrapper, _):
    # Handle float ReLU and non-threshold quantization operators
    return model.transform(
        ComposedTransformation([
            InferReLUAsElementwiseMax(),
            InferQuantAsFloat2Int()
        ])
    )


# Converts Split and Concat operations to hardware custom operators
def step_convert_split_concat_to_hw(model: ModelWrapper, _):
    return model.transform(InferSplitLayer()).transform(InferConcatLayer())


# Function running the transformations to convert Gather, i.e., index lookup,
# nodes to their hardware implementations
def step_convert_lookup_to_hw(model: ModelWrapper, _):
    # Iterate all nodes in the graph keeping track of the index
    for index, node in enumerate(model.graph.node):
        # If this is a Gather node, force the input (index) type annotation
        if node.op_type == "Gather":
            # Force to unsigned 64-bit integer for now
            model.set_tensor_datatype(node.input[1], DataType["UINT64"])
            # Get the value info for the input tensor to have access to the ONNX
            # datatype of the tensor
            value_info = model.get_tensor_valueinfo(node.input[1])
            # Force the container datatype of the input to be a float
            value_info.type.tensor_type.elem_type = 1
    # Convert Gather to Lookup layers
    return model.transform(InferLookupLayer())


# Converts depth-wise convolution to hardware operator calling the
# InferVectorVectorActivation transformation
def step_convert_depth_wise_to_hw(model: ModelWrapper, _: DataflowBuildConfig):
    return model.transform(InferVectorVectorActivation())


# Function running the InferReplicateStream transformation
def step_replicate_streams(model: ModelWrapper, _):
    # Properly replicate the stream feeding the query, key and value projections
    return model.transform(InferReplicateStream())


# Transformation apply the new YAML-based configuration to the model
from custom.apply_config import ApplyConfig


# Custom step applying our custom format of folding configuration to the graph
def step_apply_folding_config(model: ModelWrapper, cfg: DataflowBuildConfig):
    # Only applies if a configuration file is given
    if cfg.folding_config_file is not None:
        # Load the configuration dictionary form YAML file
        with (open(cfg.folding_config_file, "r") as file):
            # Load YAML string
            config = yaml.safe_load(file)
            # Assign unique names to the nodes which can be matched by
            # individual per-node configuration options
            model = model.transform(GiveUniqueNodeNames())
            # Apply the configuration dictionary to the model graph
            model = model.transform(ApplyConfig(config))
    # If configured, run a verification of the transformed model on some sample
    # inputs
    if (VerificationStepType.FOLDED_HLS_CPPSIM in
            cfg._resolve_verification_steps()):  # noqa
        # Prepare C++ Simulation for verification
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
        model = model.transform(SetExecMode("cppsim"))
        # Execute a verification step of the model with inputs specified in
        # build configuration
        verify_step(model, cfg, "folded_hls_cppsim", need_parent=True)

    # Return model with configuration applied
    return model


# Runs a node-by-node C++ simulation of the model saving the fill execution
# context
def node_by_node_cppsim(model: ModelWrapper, cfg: DataflowBuildConfig):
    # Save the original model
    original = model
    # Copy the model
    model = copy.deepcopy(model)
    # Set model execution mode to C++ simulation
    model = model.transform(SetExecMode("cppsim"))
    # Generates the C++ source and compiles the C++ simulation
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareCppSim())
    model = model.transform(CompileCppSim())

    # Load the verification input/output pair
    inp = np.load(cfg.verify_input_npy)  # noqa
    out = np.load(cfg.verify_expected_output_npy)

    # Path to the parent model wrapping the streaming dataflow partition and the
    # wrapped child model, i.e., the inside of the streaming dataflow partition
    parent = f"{cfg.output_dir}/intermediate_models/dataflow_parent.onnx"
    child = f"{cfg.output_dir}/intermediate_models/verify_cppsim.onnx"
    # Save the child model prepared for C++ simulation
    model.save(child)
    # Load the parent model to pass to verification execution
    parent_model = ModelWrapper(parent)

    # Make sure to not execute this as RTL simulation...
    parent_model.set_metadata_prop("exec_mode", "")

    # Reshape the input/output to match the model
    inp = inp.reshape(parent_model.get_tensor_shape(model.graph.input[0].name))
    out = out.reshape(parent_model.get_tensor_shape(model.graph.output[0].name))

    # Execute the onnx model to collect the result
    # context = execute_onnx(model, context, return_full_exec_context=True)
    context = execute_parent(parent, child, inp, return_full_ctx=True)
    # Extract the output tensor from the execution context
    model_out = context[parent_model.graph.output[0].name]
    # Compare input to output
    result = {True: "SUCCESS", False: "FAIL"}[np.allclose(out, model_out)]
    # Save the verification outputs into the configured build directory
    verification_output = f"{cfg.output_dir}/verification_output/"
    # Save the verification execution context
    np.savez(f"{verification_output}/verify_cppsim_{result}.npz", **context)
    # Return the original, unmodified model
    return original


# Runs a node-by-node RTL simulation of the model saving the fill execution
# context
def node_by_node_rtlsim(model: ModelWrapper, cfg: DataflowBuildConfig):
    # Save the original model
    original = model
    # Copy the model
    model = copy.deepcopy(model)
    # Set model execution mode to RTL simulation
    model = model.transform(SetExecMode("rtlsim"))
    # Generates the C++ source and compiles the RTL simulation
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(
        cfg._resolve_fpga_part(), cfg.synth_clk_period_ns)  # noqa
    )
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())

    # Load the verification input/output pair
    inp = np.load(cfg.verify_input_npy)  # noqa
    out = np.load(cfg.verify_expected_output_npy)

    # Path to the parent model wrapping the streaming dataflow partition and the
    # wrapped child model, i.e., the inside of the streaming dataflow partition
    parent = f"{cfg.output_dir}/intermediate_models/dataflow_parent.onnx"
    child = f"{cfg.output_dir}/intermediate_models/verify_rtlsim.onnx"
    # Save the child model prepared for RTL simulation
    model.save(child)
    # Load the parent model to pass to verification execution
    parent_model = ModelWrapper(parent)

    # Make sure to not execute this as stitched RTL simulation...
    parent_model.set_metadata_prop("exec_mode", "")

    # Reshape the input/output to match the model
    inp = inp.reshape(parent_model.get_tensor_shape(model.graph.input[0].name))
    out = out.reshape(parent_model.get_tensor_shape(model.graph.output[0].name))

    # Execute the onnx model to collect the result
    # context = execute_onnx(model, context, return_full_exec_context=True)
    context = execute_parent(parent, child, inp, return_full_ctx=True)
    # Extract the output tensor from the execution context
    model_out = context[parent_model.graph.output[0].name]
    # Compare input to output
    result = {True: "SUCCESS", False: "FAIL"}[np.allclose(out, model_out)]
    # Save the verification outputs into the configured build directory
    verification_output = f"{cfg.output_dir}/verification_output/"
    # Save the verification execution context
    np.savez(f"{verification_output}/verify_rtlsim_{result}.npz", **context)
    # Return the original, unmodified model
    return original


# Selected the RTL simulation backend for the whole model
def set_rtlsim_backend(backend: str):
    # Wrap the actual transformation step
    def step_set_rtlsim_backend(model: ModelWrapper, _):
        # Set backend via metadate property
        model.set_metadata_prop("rtlsim_backend", backend)
        # Return modified model
        return model

    # Return the actual transformation step
    return step_set_rtlsim_backend
