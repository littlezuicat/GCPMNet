# GCPMNet: A Novel Dehazing Network based on Global Context-Prompt Modulated Operator

> **投稿会议**: ICCPR 2025 (2025 14th International Conference on Computing and Pattern Recognition)

## 作者信息
- **作者**: 任义
- **单位**: 北京联合大学 机器人学院/人工智能学院
- **邮箱**: renyi@buu.edu.cn

## 项目目标
本项目提供了一个使用 ConvIR 模型进行单幅图像去雾的实现。代码支持使用文本提示 (Text Prompt) 结合图像内容来指导去雾过程。## 1. 环境依赖
确保您的环境中安装了以下库：
- Python >= 3.8
- PyTorch >= 1.8 (推荐使用支持 torch.fft 的较新版本)
- Torchvision
- NumPy
- tqdm
- Transformers (Hugging Face) >= 4.18 (建议使用 AutoImageProcessor)
- Pillow

安装命令:
```bash
pip install torch torchvision numpy tqdm transformers Pillow
```
> 请根据您的 CUDA 版本访问 PyTorch 官网获取最适合的 PyTorch 安装命令
## 2. 数据集准备

### 2.1 支持的数据集
本代码设计用于处理特定结构的数据集，例如 O-Haze, NHHaze, IHaze, DenseHaze 等。请确保您的数据集文件夹名称与 `data_utils.py` 中的检查逻辑兼容 (`['ohaz', 'nhhaz', 'ihaz', 'densehaz']`)。

### 2.2 下载数据集 (以 O-Haze 为例)
下载链接: [北京科技大学网盘]()

### 2.3 目录结构
请将下载的数据集解压，并按照以下结构组织文件：
```
<your_data_root>/       # 您存放所有数据集的根目录 (对应 option.py 中的 data_dir)
└── <dataset_name>/     # 数据集名称 (例如 O-Haze, 对应 option.py 中的 dataset_name)
    ├── train/          # 训练集
    │   ├── GT/         # 清晰图像 (Ground Truth)
    │   │   ├── xxx_GT.jpg
    │   │   └── ...
    │   ├── hazy/       # 对应的有雾图像
    │   │   ├── xxx_hazy.jpg
    │   │   └── ...
    │   └── prompts.json # 图像对和文本提示的关联文件
    └── test/           # 测试集 (结构同 train/)
        ├── GT/
        │   ├── yyy_GT.jpg
        │   └── ...
        ├── hazy/
        │   ├── yyy_hazy.jpg
        │   └── ...
        └── prompts.json
```
### 2.4 prompts.json 文件格式
此文件是必需的，用于将有雾图像、清晰图像及其对应的文本提示关联起来。它是一个 JSON 列表，每个元素是一个包含以下键的字典：

- **hazy_file**: (字符串) 有雾图像的文件名 (例如 "01_outdoor_hazy.jpg")
- **gt_file**: (字符串) 对应的清晰图像文件名 (例如 "01_outdoor_GT.jpg")
- **prompt**: (字符串) 描述该图像或去雾任务的文本提示 (例如 "请移除这张户外照片中的雾气")

示例:
```json
[
  {
    "hazy_file": "01_outdoor_hazy.jpg",
    "gt_file": "01_outdoor_GT.jpg",
    "prompt": "请移除这张户外照片中的雾气"
  },
  {
    "hazy_file": "02_outdoor_hazy.jpg",
    "gt_file": "02_outdoor_GT.jpg",
    "prompt": "一张在户外拍摄的、带有雾霾的照片"
  }
]
```
## 3. 使用说明

### 3.1 参数配置
主要的训练和测试参数在 `option.py` 文件中定义。您可以直接修改此文件，或在运行 `main.py` 时通过命令行参数覆盖它们。

关键参数:
- `--dataset_name`: 要使用的数据集名称 (必须与您的文件夹名称匹配，例如 'O-Haze')
- `--data_dir`: 数据集所在的根目录 (例如 './dataset/')
- `--net`: 使用的模型名称 (应为 'ConvIR')
- `--steps`: 总训练迭代步数 (例如 50000)
- `--lr`: 初始学习率 (例如 0.0001)
- `--bs`: 训练时的批处理大小 (Batch Size)
- `--eval_step`: 每隔多少步在测试集上评估一次模型
- `--device`: 指定训练设备 ('cuda' 或 'cpu')
- `--resume`: 设置为 True 以从 --model_dir 指定的检查点恢复训练
- `--model_dir`: 模型检查点的保存路径和加载路径 (自动生成，格式为 `trained_models/<dataset_name>_<net>.pk`)
- `--log_dir`: 训练日志的保存目录 (自动生成，格式为 `logs/<dataset_name>_<net>`)### 3.2 训练模型
直接运行 `main.py` 脚本开始训练。

默认训练:
```bash
python main.py --crop
```
> 这将使用 option.py 中定义的默认参数

带参数训练:
```bash
python main.py --bs 2 --steps 30000 --lr 0.0002 --crop
```
> 这将覆盖 option.py 中对应的参数

输出:
- 训练过程的日志 (包括损失、学习率、评估结果等) 会被记录在 `--log_dir` 指定的目录下的 `log_.txt` 文件中
- 在评估步骤 (eval_step) 中取得最佳 PSNR 和 SSIM 的模型权重及训练状态会被保存到 `--model_dir` 指定的 `.pk` 文件中### 3.3 评估模型
评估过程集成在训练脚本中。

- 训练期间，每隔 `--eval_step` 步会自动在测试集 (`<dataset_name>/test/`) 上进行评估，并打印当前的 PSNR 和 SSIM
- 如果当前评估结果优于历史最佳结果，去雾后的测试图像会被保存在 `result/<model_name>/` 目录下，文件名为 `<index>_<model_name>.png`
## 4. 引用
如果您在您的研究工作中使用了本代码或其思路，请考虑引用我们的论文：

```bibtex
@inproceedings{NBDehazeNet2025,
    title={NBDehazeNet: A Novel Image Dehazing Network Based on ConvIR},
    author={Your Name and Co-authors},
    booktitle={Proceedings of the 14th International Conference on Computing and Pattern Recognition (ICCPR 2025)},
    year={2025}
}
```

本项目的网络结构借鉴了以下工作：
```bibtex
@article{ConvIR2023,
    title={ConvIR: Image Restoration with Convolution-based Implicit Neural Representations},
    author={Yang, Chen and Others},
    journal={arXiv preprint},
    year={2023},
    url={https://github.com/c-yn/ConvIR}
}
```

## 5. 联系方式
如有任何问题或建议，欢迎联系：
- 任义 (renyi@buu.edu.cn)