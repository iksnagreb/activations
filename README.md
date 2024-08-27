
|            Function            | Quant | Export | Conversion | Streamlining | Hardware |    Issues    | PRs |
|:------------------------------:| :---: | :----: |:----------:| :----------: | :------: |:------------:| :-: |
|            `Identity`          |   ✅   |   ✅    |     ✅      |      ✅       |    ✅     |              |  |
|             `ReLU`             |   ✅   |   ✅    |     ✅      |      ✅       |    ✅     |              |  |
|          `LeakyReLU`           |   ✅   |   ✅    |     ✅      |      ✅       |    ❔     |              | https://github.com/fastmachinelearning/qonnx/pull/133, https://github.com/Xilinx/finn/pull/1030 |
|            `ReLU6`             |   ✅   |   ✅    |     ✅      |      ✅       |    ❔     |              | https://github.com/fastmachinelearning/qonnx/pull/133, https://github.com/Xilinx/finn/pull/1030 |
|            `RReLU`             |   ✅   |   ✅    |     ✅      |      ✅       |    ❔     |              | https://github.com/fastmachinelearning/qonnx/pull/133, https://github.com/Xilinx/finn/pull/1030 |
|             `SELU`             |   ✅   |   ✅    |     ✅      |      ✅       |    ❔     |              | https://github.com/Xilinx/finn/pull/1030    |
|             `CELU`             |   ✅   |   ❌    |     ❌      |      ❌       |    ❌     | Opset >= 12  | https://github.com/fastmachinelearning/qonnx/pull/133, https://github.com/Xilinx/finn/pull/1030 |
|             `ELU`              |   ✅   |   ✅    |     ✅      |      ✅       |    ❔     |              | https://github.com/fastmachinelearning/qonnx/pull/133, https://github.com/Xilinx/finn/pull/1030 |
|          `HardShrink`          |   ✅   |   ❌    |     ❌      |      ❌       |    ❌     | Messy Export |  |
|          `SoftShrink`          |   ✅   |   ❌    |     ❌      |      ❌       |    ❌     | Messy Export |  |
|           `Sigmoid`            |   ✅   |   ✅    |     ✅      |      ✅       |    ❔     |              | https://github.com/Xilinx/finn/pull/1030 |
|         `Hardsigmoid`          |   ✅   |   ✅    |     ✅      |      ✅       |    ❔     |              | https://github.com/fastmachinelearning/qonnx/pull/133, https://github.com/Xilinx/finn/pull/1030 |
|          `LogSigmoid`          |   ✅   |   ✅    |     ❌      |      ❌       |    ❌     |  Composite   | https://github.com/fastmachinelearning/qonnx/pull/133, https://github.com/Xilinx/finn/pull/1030 |
|             `Tanh`             |   ✅   |   ✅    |     ✅      |      ✅       |    ❔     |              | https://github.com/fastmachinelearning/qonnx/pull/133, https://github.com/Xilinx/finn/pull/1030 |
|           `Hardtanh`           |   ✅   |   ✅    |     ✅      |      ✅       |    ❔     |              | https://github.com/fastmachinelearning/qonnx/pull/133, https://github.com/Xilinx/finn/pull/1030 |
|           `Softplus`           |   ✅   |   ✅    |     ✅      |      ✅       |    ❔     |              | https://github.com/fastmachinelearning/qonnx/pull/133, https://github.com/Xilinx/finn/pull/1030 |
|           `Softsign`           |   ✅   |   ❌    |     ❌      |      ❌       |    ❌     | Messy Export |  |
|             `Exp`              |   ✅   |   ✅    |     ✅      |      ✅       |    ❔     |              | https://github.com/fastmachinelearning/qonnx/pull/133, https://github.com/Xilinx/finn/pull/1030 |
|             `Log`              |   ✅   |   ✅    |     ✅      |      ✅       |    ❔     |    Domain    | https://github.com/fastmachinelearning/qonnx/pull/133, https://github.com/Xilinx/finn/pull/1030 |
|             `Sqrt`             |   ✅   |   ✅    |     ✅      |      ✅       |    ❔     |    Domain    | https://github.com/fastmachinelearning/qonnx/pull/133, https://github.com/Xilinx/finn/pull/1030 |
|             `Erf`              |   ✅   |   ✅    |     ✅      |      ✅       |    ❔     |              | https://github.com/fastmachinelearning/qonnx/pull/133, https://github.com/Xilinx/finn/pull/1030 |
|            `Floor`             |   ✅   |   ✅    |     ❌      |      ❌       |    ❔     | Verification | https://github.com/fastmachinelearning/qonnx/pull/133, https://github.com/Xilinx/finn/pull/1030 |
|             `Ceil`             |   ✅   |   ✅    |     ✅      |      ✅       |    ❔     |              | https://github.com/fastmachinelearning/qonnx/pull/133, https://github.com/Xilinx/finn/pull/1030 |
|            `Round`             |   ✅   |   ✅    |     ✅      |      ✅       |    ❔     |              | https://github.com/fastmachinelearning/qonnx/pull/133, https://github.com/Xilinx/finn/pull/1030 |
|             `Sign`             |   ✅   |   ✅    |     ✅      |      ✅       |    ❔     |              | https://github.com/fastmachinelearning/qonnx/pull/133, https://github.com/Xilinx/finn/pull/1030 |
|            `Trunc`             |   ✅   |   ❌    |     ❌      |      ❌       |    ❌     |  No Export   |  |
|          `Tanhshrink`          |   ✅   |   ✅    |     ✅      |      ❌       |    ❌     |  Composite   | https://github.com/fastmachinelearning/qonnx/pull/133, https://github.com/Xilinx/finn/pull/1030, https://github.com/Xilinx/finn/pull/1040    |
|             `SiLU`             |   ✅   |   ✅    |     ✅      |      ✅       |    ❌     |  Composite   | https://github.com/fastmachinelearning/qonnx/pull/133, https://github.com/Xilinx/finn/pull/1030, https://github.com/Xilinx/finn/pull/1040    |
|             `GELU`             |   ✅   |   ✅    |    ✅(❌)   |      ✅(❌)   |    ❌     |  Composite   | https://github.com/fastmachinelearning/qonnx/pull/133, https://github.com/Xilinx/finn/pull/1030, https://github.com/Xilinx/finn/pull/1040    |
|             `Mish`             |   ✅   |   ✅    |     ❌      |      ❌       |    ❌     |  Composite   | https://github.com/fastmachinelearning/qonnx/pull/133, https://github.com/Xilinx/finn/pull/1030, https://github.com/Xilinx/finn/pull/1040    |
|             `GLU`              |   ✅   |   ✅    |     ✅      |      ❌       |    ❌     |  Composite   | https://github.com/fastmachinelearning/qonnx/pull/133, https://github.com/Xilinx/finn/pull/1030, https://github.com/Xilinx/finn/pull/1040    |
