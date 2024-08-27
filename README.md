
|            Function            | Quant | Export | Conversion | Streamlining | Hardware |    Issues    | PRs |
|:------------------------------:| :---: | :----: |:----------:| :----------: | :------: |:------------:| :-: |
|            `Identity`          |   ✅   |   ✅    |     ✅      |      ✅       |    ✅     |              |     |
|             `ReLU`             |   ✅   |   ✅    |     ✅      |      ✅       |    ✅     |              |     |
|          `LeakyReLU`           |   ✅   |   ✅    |     ✅      |      ✅       |    ❔     |              |     |
|            `ReLU6`             |   ✅   |   ✅    |     ✅      |      ✅       |    ❔     |              |     |
|            `RReLU`             |   ✅   |   ✅    |     ✅      |      ✅       |    ❔     |              |     |
|             `SELU`             |   ✅   |   ✅    |     ✅      |      ✅       |    ❔     |              |     |
|             `CELU`             |   ✅   |   ❌    |     ❌      |      ❌       |    ❌     | Opset >= 12  |     |
|             `ELU`              |   ✅   |   ✅    |     ✅      |      ✅       |    ❔     |              |     |
|          `HardShrink`          |   ✅   |   ❌    |     ❌      |      ❌       |    ❌     | Messy Export |     |
|          `SoftShrink`          |   ✅   |   ❌    |     ❌      |      ❌       |    ❌     | Messy Export |     |
|           `Sigmoid`            |   ✅   |   ✅    |     ✅      |      ✅       |    ❔     |              |     |
|         `Hardsigmoid`          |   ✅   |   ✅    |     ✅      |      ✅       |    ❔     |              |     |
|          `LogSigmoid`          |   ✅   |   ✅    |     ❌      |      ❌       |    ❌     |  Composite   |     |
|             `Tanh`             |   ✅   |   ✅    |     ✅      |      ✅       |    ❔     |              |     |
|           `Hardtanh`           |   ✅   |   ✅    |     ✅      |      ✅       |    ❔     |              |     |
|           `Softplus`           |   ✅   |   ✅    |     ✅      |      ✅       |    ❔     |              |     |
|           `Softsign`           |   ✅   |   ❌    |     ❌      |      ❌       |    ❌     | Messy Export |     |
|             `Exp`              |   ✅   |   ✅    |     ✅      |      ✅       |    ❔     |              |     |
|             `Log`              |   ✅   |   ✅    |     ✅      |      ✅       |    ❔     |    Domain    |     |
|             `Sqrt`             |   ✅   |   ✅    |     ✅      |      ✅       |    ❔     |    Domain    |     |
|             `Erf`              |   ✅   |   ✅    |     ✅      |      ✅       |    ❔     |              |     |
|            `Floor`             |   ✅   |   ✅    |     ❌      |      ❌       |    ❔     | Verification |     |
|             `Ceil`             |   ✅   |   ✅    |     ✅      |      ✅       |    ❔     |              |     |
|            `Round`             |   ✅   |   ✅    |     ✅      |      ✅       |    ❔     |              |     |
|             `Sign`             |   ✅   |   ✅    |     ✅      |      ✅       |    ❔     |              |     |
|            `Trunc`             |   ✅   |   ❌    |     ❌      |      ❌       |    ❌     |  No Export   |     |
|          `Tanhshrink`          |   ✅   |   ✅    |     ✅      |      ❌       |    ❌     |  Composite   |     |
|             `SiLU`             |   ✅   |   ✅    |     ✅      |      ✅       |    ❌     |  Composite   |     |
|             `GELU`             |   ✅   |   ✅    |    ✅(❌)   |      ✅(❌)   |    ❌     |  Composite   |     |
|             `Mish`             |   ✅   |   ✅    |     ❌      |      ❌       |    ❌     |  Composite   |     |
|             `GLU`              |   ✅   |   ✅    |     ✅      |      ❌       |    ❌     |  Composite   |     |
