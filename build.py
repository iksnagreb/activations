# YAML for loading experiment configurations
import yaml

# Numpy for handling arrays
import numpy as np

# FINN dataflow builder
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.builder.build_dataflow_config import AutoFIFOSizingMethod

# Range information structure for seeding the range analysis for converting
# quantized activations to MultiThreshold
from qonnx.util.range_analysis import RangeInfo

# Seeding RNGs for reproducibility
from utils import seed

# Custom build steps for handling quantizer to multi-threshold conversion
from build_steps import (
    prepare_graph,
    step_streamline,
    step_convert_elementwise_binary_to_hw,
    step_convert_floats_to_hw,
    step_convert_lookup_to_hw,
    step_convert_split_concat_to_hw,
    step_convert_depth_wise_to_hw,
    step_replicate_streams,
    step_apply_folding_config,
    step_apply_fixedpt_config,
    # node_by_node_cppsim,
    # node_by_node_rtlsim,
    set_rtlsim_backend
)

# Script entrypoint
if __name__ == "__main__":
    # Open the configuration file
    with open("params.yaml") as file:
        # Load the configuration from YAML format
        params = yaml.safe_load(file)
    # Seed all RNGs
    seed(params["seed"])

    # Construct the seed range information of the input tensor
    range_info = RangeInfo(
        # Shape according to parameters and batch dimension
        shape=(1, *params["shape"]), range=tuple(np.array([params["range"]]).T)
    )

    # If we don't prepare thresholds from quantized functions, we want to keep
    # the float ops
    keep_floats = not params["prepare"]["thresholds"]

    # Create a configuration for building the scaled dot-product operator to a
    # hardware accelerator
    cfg = build_cfg.DataflowBuildConfig(
        # Unpack the build configuration parameters
        **params["build"],
        # Print all warnings and compiler output to stdout
        verbose=True,
        # Generate and keep the intermediate outputs including reports
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
            build_cfg.DataflowOutputType.STITCHED_IP,
            build_cfg.DataflowOutputType.PYNQ_DRIVER,
            build_cfg.DataflowOutputType.BITFILE,
            build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
        ],
        # Steps after which verification should be run
        verify_steps=[
            # Verify the model after converting to the FINN onnx dialect
            build_cfg.VerificationStepType.QONNX_TO_FINN_PYTHON,
            # Verify the model again using python mode after the default
            # streamlining step
            build_cfg.VerificationStepType.STREAMLINED_PYTHON,
            # Verify the model again after tidy up transformations, right before
            # converting to HLS
            build_cfg.VerificationStepType.TIDY_UP_PYTHON,
            # Verify the model after generating C++ HLS and applying folding
            build_cfg.VerificationStepType.FOLDED_HLS_CPPSIM,
            # Verify the model via RTL simulation after IPGen
            build_cfg.VerificationStepType.NODE_BY_NODE_RTLSIM,
            # Verify the model via RTL simulation after creating stitched IP
            build_cfg.VerificationStepType.STITCHED_IP_RTLSIM
        ],
        # File with test inputs for verification
        verify_input_npy="inp.npy",
        # File with expected test outputs for verification
        verify_expected_output_npy="out.npy",
        # Output full context dump for verification steps
        verify_save_full_context=True,
        # Save the intermediate model graphs
        save_intermediate_models=True,
        # Avoid RTL simulation for setting the FIFO sizes
        auto_fifo_strategy=AutoFIFOSizingMethod.LARGEFIFO_RTLSIM,
        # Do not automatically set FIFO sizes for now
        auto_fifo_depths=True,
        # Build steps to execute
        steps=[
            # Prepares the QONNX graph to be consumed by FINN: Cleanup, lowering
            # and Quant to MultiThreshold conversion
            prepare_graph(
                range_info=range_info, **params["prepare"]
            ),
            # Unified exhaustive streamlining of complex model topologies
            # including attention, residuals and splits
            *([step_streamline] if params["prepare"]["streamline"] else []),
            # Apply optional fixed-point quantization of parameter tensors,
            # outputs of fixed-point operations will be turned into fixed-point
            # by step_minimize_bit_width later
            step_apply_fixedpt_config,
            # Convert the elementwise binary operations to hardware operators.
            # These include, for example, adding residual branches and
            # positional encoding
            *([step_convert_elementwise_binary_to_hw] if keep_floats else []),
            # Converts remaining float operations (elementwise) to hardware
            # operators
            *([step_convert_floats_to_hw] if keep_floats else []),
            # Convert Lookup layers, e.g., token embedding, to hardware custom
            # operators
            step_convert_lookup_to_hw,
            # Convert Split and Concat operators to hardware, e.g., splits
            # contained in the GLU activation
            step_convert_split_concat_to_hw,
            # Convert depth-wise convolution MatMuls to VVUs
            step_convert_depth_wise_to_hw,
            # Properly replicate the stream feeding the query, key and value
            # projections
            step_replicate_streams,
            # Convert most other layers supported by FINN to HW operators and
            # continue with default FINN flow
            "step_convert_to_hw",
            # Do this here to get datatypes right when quantizing to fixed-point
            "step_minimize_bit_width",
            "step_specialize_layers",
            "step_create_dataflow_partition",
            "step_target_fps_parallelization",
            # Set RTL simulation to use verilator backend
            set_rtlsim_backend("pyxsi"),
            # Apply folding config using out custom YAML format
            step_apply_folding_config,
            "step_minimize_bit_width",
            "step_generate_estimate_reports",
            "step_hw_codegen",
            "step_hw_ipgen",
            "step_set_fifo_depths",
            # Run additional node-by-node verification in C++ simulation of the
            # model before creating the stitched IP
            # node_by_node_cppsim,
            # Run additional node-by-node verification in RTL simulation of the
            # model before creating the stitched IP
            # node_by_node_rtlsim,
            # Finish following default FINN flow
            "step_create_stitched_ip",
            "step_measure_rtlsim_performance",
            "step_out_of_context_synthesis",
            "step_synthesize_bitfile",
            "step_make_pynq_driver",
            "step_deployment_package",
        ]
    )
    # Run the build process on the model graph
    build.build_dataflow_cfg("model.onnx", cfg)
