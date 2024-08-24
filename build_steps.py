# QONNX wrapper of ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper
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

# Converts quantized activation functions to MultiThresholds based on range
# analysis
from quant_activation_to_multithreshold import QuantActivationToMultiThreshold


# Converts quantized activation functions to MultiThreshold instances based on
# range analysis. Also does various cleanup and lowering transformations as part
# of the range analysis.
def step_quant_activation_to_multithreshold(
        model: ModelWrapper, cfg: DataflowBuildConfig
):
    # Add shape and datatype annotations throughout all the graph
    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())

    # Convert all suitable quantized activation functions to MultiThresholds
    model = model.transform(QuantActivationToMultiThreshold())

    # If configured, run a verification of the transformed model on some sample
    # inputs
    if (VerificationStepType.QONNX_TO_FINN_PYTHON in
            cfg._resolve_verification_steps()):  # noqa
        verify_step(
            model, cfg, "to_multithreshold_python", need_parent=False
        )

    # Return the transformed model
    return model