# EyeTrax — 眼动追踪打地鼠 🐹

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Made with Python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)

基于 **EyeTrax** 眼动追踪库扩展开发的**打地鼠游戏**，使用你的眼球来控制锤子！

## 🎮 功能

- **👁️ 眼动打地鼠** — 用视线瞄准、眨眼或停留来敲打地鼠
- **⌨️ 眼动打字** — 用眼动控制虚拟键盘输入文字
- **📹 实时注视估计** — 基于普通网络摄像头
- **🎯 多种校准模式** — 9点、5点、李萨如曲线、自适应校准
- **🔍 可选滤波** — Kalman / KDE 平滑
- **📺 虚拟摄像头** — 支持 OBS 推流

## 📦 安装

```bash
# 克隆仓库
git clone https://github.com/myx1025377675-maker/EyeTrax.git
cd EyeTrax

# 安装依赖（推荐使用 uv）
pip install uv
uv sync

# 或者用 pip
pip install -e .
```

## 🚀 快速开始

### 打地鼠游戏
```bash
python -m eyetrax.app.whack_a_mole
```
或者直接运行：
```bash
run_whack_a_mole.bat
```

### 其他应用

| 命令 | 用途 |
|------|------|
| `eyetrax-whack` | 眼动打地鼠 |
| `eyetrax-gaze-typing` | 眼动键盘打字 |
| `eyetrax-gaze-suite` | 眼动打字套件 |
| `eyetrax-demo` | 注视点实时显示 |
| `eyetrax-virtualcam` | 推流到虚拟摄像头 |

启动选项：

| 参数 | 可选值 | 默认值 | 说明 |
|------|--------|--------|------|
| `--filter` | `kalman`, `kde`, `none` | `none` | 平滑滤波器 |
| `--camera` | *数字* | `0` | 摄像头索引 |
| `--calibration` | `9p`, `5p`, `lissajous` | `9p` | 校准方式 |
| `--confidence` *(KDE)* | *0–1* | `0.5` | 等高线概率 |

## 📁 项目结构

```
EyeTrax/
├── src/eyetrax/
│   ├── app/
│   │   ├── whack_a_mole.py      # 🎯 打地鼠游戏
│   │   ├── gaze_typing.py       # 眼动打字
│   │   ├── gaze_typing_suite.py # 打字套件
│   │   ├── demo.py              # 注视演示
│   │   ├── virtualcam.py        # 虚拟摄像头
│   │   ├── mouse_control.py     # 眼动鼠标
│   │   └── build_model.py       # 模型训练
│   ├── calibration/             # 校准算法
│   ├── filters/                 # Kalman/KDE 滤波
│   ├── models/                  # 注视预测模型
│   └── utils/                   # 工具函数
├── assets/
│   ├── images/                  # 鼹鼠图片
│   └── audio/                   # 背景音乐
└── WHACK_A_MOLE_DEVNOTE.md      # 打地鼠开发文档
```

## 🐍 库用法示例

```python
from eyetrax import GazeEstimator, run_9_point_calibration
import cv2

# 创建估计器并校准
estimator = GazeEstimator()
run_9_point_calibration(estimator)

# 保存模型
estimator.save_model("gaze_model.pkl")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    features, blink = estimator.extract_features(frame)

    if features is not None and not blink:
        x, y = estimator.predict([features])[0]
        print(f"注视坐标: ({x:.0f}, {y:.0f})")
```

## 🙏 致谢

本项目基于 [**EyeTrax**](https://github.com/ck-zhang/EyeTrax) 开发，感谢原作者 **Chenkai Zhang** 提供的优秀眼动追踪框架。

## 📄 许可证

本项目基于 MIT 协议开源，详见 [LICENSE](LICENSE)。原始 EyeTrax 版权归 Chenkai Zhang 所有。

---

**作者**: 马跃翔  
**仓库**: https://github.com/myx1025377675-maker/EyeTrax
