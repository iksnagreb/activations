# Numpy maths and array
import numpy as np

# Helper for assembling ONNX nodes, tensors and graphs
from onnx import helper as oh

# QONNX wrapper of ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper
# QONNX quantization data types
from qonnx.core.datatype import DataType

# Base class for all QONNX graph transformations
from qonnx.transformation.general import Transformation

# Infers and annotates quantization data types
from qonnx.transformation.infer_datatypes import InferDataTypes
# Infer tensor shape annotations
from qonnx.transformation.infer_shapes import InferShapes

# Groups the mode inputs by dynamic vs. initializer categories
from finn.transformation.util import group_inputs_by_category
# Converts scalar tensors to rank-1 tensors
from finn.transformation.fpgadataflow.convert_to_hw_layers import lift_to_rank1


# Tests whether all values are powers of two
def is_power_of_two(x):
    # Convert to power of two and check if still the same
    return (2 ** np.log2(np.asarray([x])).round()) == np.asarray([x])


# Converts Mul nodes with constant power-of-two scales to bit shift operators
class InferPowerOfTwoMulAsBitShift(Transformation):
    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Applies to Mul operation types
            if node.op_type in {"Mul"}:
                # Cannot handle fork- or join-multiplications
                if model.is_fork_node(node) or model.is_join_node(node):
                    # Softly skip this node
                    continue

                # Find input and the parameter tensor names
                (inp,), (init,) = group_inputs_by_category(node, model)

                # Check whether this a power-of-two scaling operation
                if not np.all(is_power_of_two(model.get_initializer(init))):
                    # Softly skip this node
                    continue

                # The input must have integer type annotations as we cannot
                # shift floats
                if not model.get_tensor_datatype(inp).is_integer():
                    # Softly skip this node
                    continue

                # Need to "lift" potential scalar inputs to rank-1 tensors
                lift_to_rank1(node.input[0], model)
                lift_to_rank1(node.input[1], model)

                # Extract the exponents as integer shifts
                shifts = np.log2(model.get_initializer(init))
                # Represent bit-shifts as integers
                shifts = shifts.astype(np.int64)

                # Separate left and right shift components
                lefts = np.where(shifts >= 0, shifts, 0)
                rights = np.where(shifts <= 0, np.abs(shifts), 0)

                # Create new operator realizing scales >= 1.0 as left shift
                # operations
                left = oh.make_node(
                    # Transplant this operator into our FINN domain
                    domain = "finn.custom_op.fpgadataflow",
                    # Elementwise bit-shift from the FINN domain
                    op_type="ElementwiseBitShift",
                    # Set the backend attribute to mark this an operation
                    # supported to be implemented on an FPGA by FINN
                    backend="fpgadataflow",
                    # Attribute specifies the shift direction
                    direction="LEFT",
                    # Connect to new top input and new parameter tensors
                    inputs=[inp, model.make_new_valueinfo_name()],
                    # Connect to a new intermediate tensor
                    outputs=[model.make_new_valueinfo_name()],
                    # Datatypes of the input, parameters and output
                    lhs_dtype=model.get_tensor_datatype(inp).name,
                    rhs_dtype="UINT64",
                    out_dtype=model.get_tensor_datatype(inp).name,
                    # Shapes of the input, parameters and output
                    lhs_shape=model.get_tensor_shape(inp),
                    rhs_shape=lefts.shape,
                    out_shape=model.get_tensor_shape(node.output[0])
                )
                # Insert the left shift components of the power-of-two scales
                model.set_initializer(left.input[1], lefts)
                # Annotate the parameter data type
                model.set_tensor_datatype(left.input[1], DataType["UINT64"])
                # Annotate the shape of the connecting tensor
                model.set_tensor_shape(
                    left.output[0], model.get_tensor_shape(node.output[0])
                )

                # Create new operator realizing scales <= 1.0 as right shift
                # operations
                right = oh.make_node(
                    # Transplant this operator into our FINN domain
                    domain = "finn.custom_op.fpgadataflow",
                    # Elementwise bit-shift from the FINN domain
                    op_type="ElementwiseBitShift",
                    # Set the backend attribute to mark this an operation
                    # supported to be implemented on an FPGA by FINN
                    backend="fpgadataflow",
                    # Attribute specifies the shift direction
                    direction="RIGHT",
                    # Connect to new intermediate and new parameter tensors
                    inputs=[left.output[0], model.make_new_valueinfo_name()],
                    # Connect to the old output tensors
                    outputs=node.output,
                    # Datatypes of the input, parameters and output
                    lhs_dtype=model.get_tensor_datatype(inp).name,
                    rhs_dtype="UINT64",
                    out_dtype=model.get_tensor_datatype(inp).name,
                    # Shapes of the input, parameters and output
                    lhs_shape=model.get_tensor_shape(inp),
                    rhs_shape=rights.shape,
                    out_shape=model.get_tensor_shape(node.output[0])
                )
                # Insert the right shift components of the power-of-two scales
                model.set_initializer(right.input[1], rights)
                # Annotate the parameter data type
                model.set_tensor_datatype(right.input[1], DataType["UINT64"])
                # Annotate the shape of the connecting tensor
                model.set_tensor_shape(
                    right.output[0], model.get_tensor_shape(node.output[0])
                )

                # Insert the new nodes into the graph
                graph.node.insert(index + 1, left)
                graph.node.insert(index + 2, right)
                # Remove the original scale operator
                graph.node.remove(node)
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
