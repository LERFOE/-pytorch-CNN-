# PyTorch 中药材图像分类器

本项目是一个使用 PyTorch 实现的卷积神经网络 (CNN)，用于对中药材图像进行分类。

## 目录结构

```
pytorch-chinese-medicine-classifier/
├── README.md               # 本 README 文件
├── requirements.txt        # 项目依赖
└── src/
    └── src/
        ├── config.py       # 配置文件
        ├── README.md       # 原始的简版 README (仅含安装指令)
        ├── test.py         # 测试脚本
        ├── train.py        # 训练脚本
        └── utils.py        # 工具函数 (包含一个 CNNModel 定义和数据加载逻辑)
```

## 安装

1.  **克隆仓库** (如果尚未克隆)
    ```bash
    # git clone https://github.com/LERFOE/-pytorch-CNN-.git
    # cd pytorch-chinese-medicine-classifier
    ```

2.  **创建虚拟环境** (推荐)
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate   # Windows
    ```

3.  **安装依赖**
    运行以下命令来安装 `requirements.txt` 中列出的依赖项：
    ```bash
    pip install -r requirements.txt
    ```

## 配置

主要的配置参数位于 [`src/src/config.py`](src/src/config.py) 文件中。您可以根据需要修改这些参数：

*   `LEARNING_RATE`: 学习率
*   `BATCH_SIZE`: 批量大小
*   `EPOCHS`: 训练周期数
*   `IMAGE_SIZE`: 图像大小 (当前设置为 `(224, 224)`)
*   `NUM_CLASSES`: 分类类别数量 (当前设置为 `10`)
*   `DEVICE`: 训练设备 (`'cuda'` 或 `'cpu'`)

## 使用方法

### 数据准备

确保您的图像数据集按以下结构组织在您的数据目录中，例如 `data/`:

```
data/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   └── ...
│   └── class2/
│       ├── image1.jpg
│       └── ...
└── test/
    ├── class1/
    │   ├── image1.jpg
    │   └── ...
    └── class2/
        ├── image1.jpg
        └── ...
```

### 模型训练

1.  根据您的数据集路径，修改 [`src/src/utils.py`](src/src/utils.py) 中的 `load_data` 函数，或者确保 `train.py` 中调用的 `get_data_loaders` (如果该函数存在或被正确实现) 使用正确的数据路径。
2.  运行训练脚本：
    ```bash
    python src/src/train.py
    ```
    训练脚本将使用在 [`src/src/train.py`](src/src/train.py) 文件顶部定义的 `CNNModel` 和 [`src/src/config.py`](src/src/config.py) 中的配置进行模型训练。训练过程中的损失和准确率将打印到控制台。

### 模型测试

1.  在 [`src/src/test.py`](src/src/test.py) 脚本中，确保 `test_loader` 加载了正确的测试数据路径，并且 `model_path` 指向您训练好的模型文件 (`.pth` 文件)。
2.  运行测试脚本：
    ```bash
    python src/src/test.py
    ```
    脚本将输出模型在测试集上的准确率。

## 关键文件说明

*   **[`src/src/config.py`](src/src/config.py)**:
    *   定义了训练所需的各种超参数（如学习率、批量大小、周期等）。
    *   文件顶部注释部分也包含一个 `CNNModel` 的定义，与 `train.py` 和 `test.py` 中的模型结构相似。

*   **[`src/src/train.py`](src/src/train.py)**:
    *   负责执行模型训练。
    *   此文件顶部定义了一个 `CNNModel` 类（输入图像大小假定为 224x224，10个输出类别）。训练过程将使用此模型。
    *   尝试从 `.utils` 导入 `get_data_loaders` 和 `plot_loss_accuracy` 函数。**注意**: 当前 [`src/src/utils.py`](src/src/utils.py) 中定义的对应函数是 `load_data` 和 `plot_curves`，这可能导致导入错误或功能不匹配。
    *   训练完成后，它尝试调用 `plot_loss_accuracy` 来可视化训练的损失和准确率。

*   **[`src/src/test.py`](src/src/test.py)**:
    *   用于评估训练好的模型在测试集上的性能。
    *   与 `train.py` 类似，此文件顶部也定义了一个 `CNNModel` 类，用于加载模型权重。
    *   从 `utils` 模块导入 `load_data` 和 `calculate_accuracy`。

*   **[`src/src/utils.py`](src/src/utils.py)**:
    *   包含一个 `CNNModel` 类的定义。此模型结构与 `train.py`/`test.py` 中定义的模型不同（例如，它接受 `num_classes` 作为参数，并且其全连接层维度基于 128x128 输入图像和3个池化层计算，但计算方式似乎有误，应为 `64 * 16 * 16` 而非 `64 * 32 * 32`）。目前，此模型定义未被 `train.py` 或 `test.py`直接使用。
    *   `load_data(data_dir, batch_size)`: 加载图像数据，进行转换（包括将图像大小调整为 128x128），并创建 DataLoader。
    *   `calculate_accuracy(outputs, labels)`: 计算并返回准确率。
    *   `plot_curves(train_losses, train_accuracies, test_losses, test_accuracies)`: 使用 matplotlib 绘制训练和测试的损失及准确率曲线。

## 依赖项

所有 Python 依赖项都列在 [`requirements.txt`](requirements.txt) 文件中。

## 已知问题与注意事项

1.  **模型定义不一致**:
    *   [`src/src/utils.py`](src/src/utils.py) 中定义了一个 `CNNModel`。
    *   [`src/src/train.py`](src/src/train.py) 和 [`src/src/test.py`](src/src/test.py) (以及 [`src/src/config.py`](src/src/config.py) 的注释中) 也定义了结构不同的 `CNNModel`。实际训练和测试使用的是这些脚本内部定义的模型。

2.  **数据维度不匹配**:
    *   [`src/src/utils.py`](src/src/utils.py) 中的 `load_data` 函数将图像大小调整为 `(128, 128)`。
    *   [`src/src/train.py`](src/src/train.py) 和 [`src/src/test.py`](src/src/test.py) 中使用的 `CNNModel` 的全连接层维度是基于 `(224, 224)` 的输入图像计算的（例如 `fc1 = nn.Linear(64 * 56 * 56, 128)`）。
    *   这可能导致在模型前向传播时出现维度不匹配的运行时错误，除非数据加载部分或模型结构被相应调整。

3.  **工具函数导入问题**:
    *   [`src/src/train.py`](src/src/train.py) 尝试导入 `get_data_loaders` 和 `plot_loss_accuracy` 从 `.utils`。
    *   [`src/src/utils.py`](src/src/utils.py) 中实际定义的函数是 `load_data` 和 `plot_curves`。这需要修正以确保脚本正常运行。

4.  **`CNNModel` 在 `utils.py` 中的维度计算**:
    *   [`src/src/utils.py`](src/src/utils.py) 中的 `CNNModel` 在 `forward` 方法中使用 `x = x.view(-1, 64 * 32 * 32)` 并且 `self.fc1 = nn.Linear(64 * 32 * 32, 128)`。对于128x128的输入图像和定义的3个卷积层及3个池化层，正确的扁平化维度应为 `64 * 16 * 16`。

建议在进一步使用或开发此项目前解决上述问题，以确保代码的正确性和可维护性。
