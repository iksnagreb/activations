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


# Annotates initializer datatypes as integers if all values are integers
class InferIntInitializers(Transformation):
    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Find all initializer inputs to the operations
            *_, initializers = group_inputs_by_category(node, model)
            # Check and modify all initializers
            for name in initializers:
                # Do not change annotation if already annotated as some integer
                if not model.get_tensor_datatype(name).is_integer():
                    # Get the initializer values
                    init = model.get_initializer(name)
                    # Check whether all values are integers
                    if np.all(np.asarray(init, dtype=np.int64) == init):
                        # Annotate as large integer datatype to be minimized
                        # later
                        model.set_tensor_datatype(name, DataType["INT64"])
                        # The graph has been modified and thus the
                        # transformation needs to be applied again
                        graph_modified = True
        # Redo datatype and shape annotations
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        # Return the transformed model and indicate whether the transformation
        # needs to be applied again
        return model, graph_modified