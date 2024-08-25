# YAML for loading experiment configurations
import yaml

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
from build_steps import quant_activation_to_multithreshold

# Script entrypoint
if __name__ == "__main__":
    # Open the configuration file
    with open("params.yaml") as file:
        # Load the configuration from yaml format
        params = yaml.safe_load(file)
    # Seed all RNGs
    seed(params["seed"])

    # Construct the seed range information of the input tensor
    range_info = RangeInfo(
        shape=(1, *params["shape"]), range=params["range"]
    )

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
        ],
        # File with test inputs for verification
        verify_input_npy="inp.npy",
        # File with expected test outputs for verification
        verify_expected_output_npy="out.npy",
        # Save the intermediate model graphs
        save_intermediate_models=True,
        # Avoid RTL simulation for setting the FIFO sizes
        auto_fifo_strategy=AutoFIFOSizingMethod.CHARACTERIZE,
        # Do not automatically set FIFO sizes for now
        auto_fifo_depths=False,
        # Build steps to execute
        steps=[
            # Custom step to convert all suitable QONNX Quant nodes to
            # Multithreshold nodes via range analysis
            quant_activation_to_multithreshold(range_info),
            # Convert all remaining QONNX Quant nodes not handled by the step
            # before to Multithreshold nodes
            # Note: These are input quantizers (due to missing int-range at the
            # input) and BipolarQuant which are currently not fully supported by
            # range analysis.
            "step_qonnx_to_finn",
            # Continue with default FINN build flow from here on
            "step_tidy_up",
            "step_streamline",
            "step_convert_to_hw",
            "step_specialize_layers",
            "step_create_dataflow_partition",
            "step_target_fps_parallelization",
            "step_apply_folding_config",
            "step_minimize_bit_width",
            "step_generate_estimate_reports",
            "step_hw_codegen",
            "step_hw_ipgen",
            "step_set_fifo_depths",
            "step_create_stitched_ip",
            "step_measure_rtlsim_performance",
            "step_out_of_context_synthesis",
            "step_synthesize_bitfile",
            "step_make_pynq_driver",
            "step_deployment_package",
        ]
    )
    # Run the build process on the dummy operator graph
    build.build_dataflow_cfg("model.onnx", cfg)
