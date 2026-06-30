# 眼动打地鼠 — 开发说明文档 (for AI Agent)

> **目标读者**: AI coding agent (Claude Code / Codex)  
> **目的**: 在 EyeTrax 眼动追踪项目上扩展"打地鼠"游戏功能  
> **状态**: 代码已生成，需要调试修复后才能运行

---

## 1. 项目背景：EyeTrax 源代码框架

### 1.1 项目概览

- **名称**: EyeTrax v0.3.1
- **用途**: 基于普通网络摄像头的眼动追踪
- **作者**: Chenkai Zhang
- **许可证**: MIT
- **Python 版本**: ≥3.9
- **包管理器**: uv (见 `uv.lock`)
- **构建系统**: hatchling

### 1.2 核心依赖

| 依赖 | 用途 |
|------|------|
| `opencv-python >= 4.5` | 摄像头采集、Kalman 滤波、图像绘制 |
| `mediapipe >= 0.10` | 人脸 3D 网格 landmark (478点) |
| `numpy >= 1.22` | 数值计算 |
| `scikit-learn >= 1.3` | 注视预测的机器学习模型 |
| `scipy >= 1.10` | KDE 平滑器 |
| `screeninfo >= 0.8` | 获取显示器分辨率 |
| `pyvirtualcam >= 0.10` | 虚拟摄像头输出 |

### 1.3 目录结构 (只列出源文件)

```
src/eyetrax/
├── __init__.py              # 懒加载公共 API (GazeEstimator, 标定函数等)
├── _version.py              # __version__ = "0.3.1"
├── cli.py                   # 共享命令行参数解析 (parse_common_args)
├── constants.py             # MediaPipe 面部 landmark 索引常量
├── gaze.py                  # ★ 核心: GazeEstimator 类
│
├── models/                  # 机器学习模型 (scikit-learn 封装)
│   ├── __init__.py          # 模型注册表 + create_model() 工厂函数
│   ├── base.py              # BaseModel 抽象类 (train/predict/save/load)
│   ├── ridge.py             # Ridge 回归 (默认模型)
│   ├── elastic_net.py       # ElasticNet
│   ├── svr.py               # LinearSVR (x/y 各一个模型)
│   └── tiny_mlp.py          # MLP 神经网络 (64→32)
│
├── filters/                 # 注视点平滑滤波器
│   ├── __init__.py          # 导出 + make_kalman() 工厂函数
│   ├── base.py              # BaseSmoother 抽象类 (step方法)
│   ├── noop.py              # NoSmoother (直通)
│   ├── kalman.py            # KalmanSmoother + 自动调优
│   └── kde.py               # KDESmoother (核密度估计)
│
├── calibration/             # 标定流程
│   ├── __init__.py          # 导出三个标定函数
│   ├── common.py            # 网格点计算 + 人脸等待倒计时 + 脉冲采集循环
│   ├── five_point.py        # 5 点标定
│   ├── nine_point.py        # 9 点标定 (默认)
│   ├── lissajous.py         # 利萨如曲线标定
│   └── adaptive.py          # 自适应标定 (蓝噪声采样 + 增量重训练)
│
├── utils/                   # 工具
│   ├── screen.py            # get_screen_size() - 获取显示器分辨率
│   ├── video.py             # camera() / fullscreen() / iter_frames() 上下文管理器
│   └── draw.py              # draw_cursor() / make_thumbnail()
│
└── app/                     # 应用层
    ├── demo.py              # eyetrax-demo CLI 入口 (注视叠加演示)
    ├── virtualcam.py        # eyetrax-virtualcam CLI 入口 (虚拟摄像头)
    ├── build_model.py       # eyetrax-build-model CLI 入口 (高级标定+保存)
    ├── gaze_typing.py       # 基础眼动打字 (凝视/眨眼选择)
    ├── gaze_typing_suite.py # 增强版眼动打字 (模式选择+自由打字)
    ├── mouse_control.py     # 眼控系统鼠标
    └── whack_a_mole.py      # ★ 新增: 眼动打地鼠游戏
```

### 1.4 核心数据流

```
Webcam → cv2.VideoCapture
  ↓
GazeEstimator.extract_features(frame)
  ├── MediaPipe FaceMesh → 478 个 3D landmark
  ├── 以鼻尖为锚点归一化
  ├── 用眼角构建正交基旋转 (消除头部姿态)
  ├── 提取眼部 landmark + yaw/pitch/roll → 特征向量 (~240 维)
  └── 计算 EAR (眼部纵横比) → 眨眼检测
  ↓
estimator.predict([features]) → (x, y) 屏幕坐标
  ↓
smoother.step(x, y) → (x_smooth, y_smooth) 平滑坐标
  ↓
应用层消费 (绘制光标 / 打地鼠命中判定 / 键盘命中判定等)
```

### 1.5 关键 API 用法 (供参考)

```python
from eyetrax import GazeEstimator, run_9_point_calibration
from eyetrax.filters import make_kalman, KalmanSmoother
import cv2, numpy as np

# 创建估计器并标定
estimator = GazeEstimator(model_name="ridge")
run_9_point_calibration(estimator, camera_index=0)

# 或加载已有模型
estimator.load_model("gaze_model.pkl")

# 预测循环
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    features, blink = estimator.extract_features(frame)
    if features is not None and not blink:
        x, y = estimator.predict(np.array([features]))[0]
```

### 1.6 现有 Dwell 模式 (从 gaze_typing.py 复用)

```python
# 凝视计时模式（gaze_typing.py 第 383-420 行的核心逻辑）
DWELL_TIME = 1.0       # 凝视多久触发
SELECT_COOLDOWN = 0.7  # 两次触发间隔

# 状态变量
last_key_label = None
dwell_start_time = None
last_select_time = 0.0

# 每帧判定
if highlighted_key is not None:
    if last_key_label == highlighted_key.label:
        if dwell_start_time is None:
            dwell_start_time = now
        dwell_progress = (now - dwell_start_time) / DWELL_TIME
    else:
        last_key_label = highlighted_key.label
        dwell_start_time = now
        dwell_progress = 0.0
else:
    last_key_label = None
    dwell_start_time = None

# 触发
if (highlighted_key and dwell_start_time
    and (now - dwell_start_time) >= DWELL_TIME
    and (now - last_select_time) > SELECT_COOLDOWN):
    # 触发命中
```

---

## 2. 用户需求：眼动打地鼠游戏

### 2.1 核心玩法

- 屏幕上随机出现一只地鼠
- 玩家用眼睛**注视地鼠**超过一定时间（默认 0.8 秒）即判定"打中"
- 打中后地鼠消失，重新随机出现在别的位置
- 未打中（地鼠出现超过 3 秒）也自动刷新
- 计分制：打中得分，连击加分

### 2.2 用户明确的选择（通过交互确认）

| 选项 | 用户选择 |
|------|----------|
| 地鼠数量 | **单只地鼠**（一只一只出现，打完刷新） |
| 游戏模式 | **两者都要**：限时模式 + 生命模式，菜单选择 |
| 地鼠外观 | **PNG 图片**（也支持 fallback 手绘） |

### 2.3 期望功能清单

- [x] 单只地鼠，注视命中后刷新位置
- [x] 两种游戏模式：限时模式 / 生命模式
- [x] 模式选择菜单（凝视选择）
- [x] 3-2-1 倒计时动画
- [x] 凝视进度环（在地鼠周围显示黄色圆环进度）
- [x] 命中飘字动画 "+10"（上移+渐隐）
- [x] 连击系统（连续命中得分倍率递增）
- [x] 冷却防抖（两次命中间隔最小 0.5s）
- [x] 地鼠存活时间限制（超时自动消失）
- [x] HUD 信息栏（分数、连击、时间/生命）
- [x] 结算画面（最终分数 + R重玩 / Q退出）
- [x] CLI 参数控制（dwell 时间、地鼠大小、模式等）
- [x] PNG 图片支持（alpha 通道混合）
- [x] 手绘 fallback（无图片时绘制卡通地鼠）

---

## 3. 实现说明：我做了什么

### 3.1 新建文件

**`src/eyetrax/app/whack_a_mole.py`** — 约 520 行，不修改任何已有文件。

### 3.2 代码架构

```
whack_a_mole.py 内部结构
│
├── GamePhase(Enum)          # 游戏状态: MENU / COUNTDOWN / PLAYING / GAME_OVER
├── GameConfig(@dataclass)   # 所有可配置参数集中管理
│   ├── mode, total_time, total_lives
│   ├── dwell_time, hit_cooldown, mole_ttl
│   ├── mole_radius, mole_margin
│   ├── combo_bonus_factor, base_score
│   └── mole_image_path
│
├── FloatingText(@dataclass) # 飘字动画类
│   └── draw(canvas, now) → bool  # 返回 False 表示动画结束
│
├── Mole                     # 地鼠类
│   ├── __init__()           # 加载 PNG (cv2.IMREAD_UNCHANGED) 并缩放到 2*radius
│   ├── spawn()              # 随机位置 (避开顶部 HUD 区域)
│   ├── despawn()            # 标记 inactive
│   ├── contains(gx,gy)→bool # 欧氏距离判定
│   ├── draw()               # 分发到 _draw_sprite 或 _draw_fallback
│   ├── _draw_sprite()       # PNG alpha 通道混合渲染
│   └── _draw_fallback()     # cv2.circle 手绘 (身体+耳朵+眼睛+瞳孔+鼻子+微笑)
│
├── WhackAMoleGame           # ★ 主游戏控制器
│   ├── run()                # 主入口: camera + fullscreen + 主循环
│   ├── _setup()             # 标定+滤波器初始化 (复用 demo.py 模式)
│   ├── _update_menu()       # 每帧菜单更新 + dwell 选择
│   ├── _draw_menu()         # 菜单渲染 (按钮 + dwell 进度环)
│   ├── _update_countdown()  # 3-2-1 倒计时动画
│   ├── _update_playing()    # ★ 核心游戏循环
│   │   ├── 特征提取 → 预测 → 平滑 → (gx, gy)
│   │   ├── Mole.contains(gx,gy) → dwell 计时
│   │   ├── dwell ≥ 阈值 → _on_hit()
│   │   ├── mole过期 → _on_miss()
│   │   └── 渲染 mole + dwell环 + HUD
│   ├── _on_hit()            # 命中: 加分+连击+飘字+重生
│   ├── _on_miss()           # 未中: 连击归零+扣命(lives模式)+重生
│   ├── _draw_hud()          # 顶部栏 (分数/连击/时间/爱心)
│   ├── _update_game_over()  # 结算画面 (最终分数 + R/Q)
│   └── _restart()           # 重置所有状态回到菜单
│
└── run_whack_a_mole()       # CLI 入口 (独立参数解析器)
```

### 3.3 复用的现有模块

| 复用来源 | 用途 |
|----------|------|
| `eyetrax.gaze.GazeEstimator` | 面部特征提取 + 注视预测 |
| `eyetrax.calibration.{run_9_point,run_5_point,run_lissajous}` | 标定流程 |
| `eyetrax.filters.{KalmanSmoother,KDESmoother,NoSmoother,make_kalman}` | 平滑滤波 |
| `eyetrax.utils.screen.get_screen_size` | 获取屏幕分辨率 |
| `eyetrax.utils.video.{camera,fullscreen,iter_frames}` | 摄像头/全屏/帧迭代 |

### 3.4 关键设计决策

1. **单窗口架构**: 整个游戏在一个全屏 OpenCV 窗口中运行，状态切换（菜单→倒计时→游戏→结算）靠 `GamePhase` 枚举驱动
2. **Dwell 判定**: 完全复刻 `gaze_typing_suite.py` 的凝视计时模式（last_target + dwell_start + cooldown），适配到圆形区域判定
3. **PNG 渲染**: 用 `cv2.IMREAD_UNCHANGED` 保留 alpha 通道，渲染时逐通道 float32 混合（`img * alpha + roi * (1-alpha)`）
4. **CLI 设计**: 独立 `argparse` 解析器（不复用 `parse_common_args`），因为游戏有大量专属参数
5. **地鼠随机位置**: 简单 `random.randint` 在安全范围内生成，未使用 `BlueNoiseSampler`（单只地鼠不需要蓝噪声分布）

---

## 4. ⛔ 环境诊断 & 修复指南 (必须先读!)

> **所有问题均为学长拷贝项目时遗留，非新增代码导致。**

### 4.1 本机实际环境

| 项目 | 值 |
|------|-----|
| 系统 Python | `D:\Python312\python.exe` (v3.12.7) |
| 已装包 | scikit-learn 1.8.0, scipy 1.17.1, numpy 2.4.3 |
| 缺失包 | opencv-python, mediapipe, screeninfo, pyvirtualcam |
| uv 包管理器 | ❌ 未安装 |
| Anaconda | ❌ 不存在 |

### 4.2 🔴 致命: 两个虚拟环境全部指向不存在的 Anaconda

项目有两个 venv，都绑定了学长机器上的 `D:\Anaconda`，在你机器上完全作废：

| venv | pyvenv.cfg 中的 home | 状态 |
|------|---------------------|------|
| `.venv/` | `home = D:\Anaconda` | ❌ 报废 |
| `.venv_pack/` | `home = D:\Anaconda` (原路径含 `D:\研究生\test\eyetrax-1`) | ❌ 报废 |

**根因**: `.venv/` 在 `.gitignore` 中被忽略，本身就不该被拷贝。学长直接从自己机器复制整个文件夹给你，把绑定了他 Anaconda 路径的 venv 一起带了过来。

**症状**: 任何运行尝试都会报错:
```
No Python at 'D:\Anaconda\python.exe'
```

### 4.3 🟠 缺失: uv 包管理器

项目使用 `uv` 管理依赖（存在 `uv.lock` 锁文件），但本机未安装 `uv`。

### 4.4 🟡 缺失: 4 个 Python 依赖包

系统 Python `D:\Python312` 缺少以下包（`pyproject.toml` 声明的依赖）：

| 缺失包 | 要求版本 | 用途 |
|--------|----------|------|
| `opencv-python` | ≥4.5 | 摄像头采集、图像绘制、Kalman 滤波 |
| `mediapipe` | ≥0.10 | 人脸 landmark 检测 (478点) |
| `screeninfo` | ≥0.8 | 获取屏幕分辨率 |
| `pyvirtualcam` | ≥0.10 | 虚拟摄像头输出 (virtualcam.py 用) |

已装包: `scikit-learn` 1.8.0 ✅, `scipy` 1.17.1 ✅, `numpy` 2.4.3 ✅

### 4.5 🟡 额外: mouse_control.py 的隐式依赖

`mouse_control.py` 依赖 `pyautogui`，但 `pyproject.toml` 未声明此依赖。

### 4.6 🟢 好消息: 源代码完整 + 学长打包了 exe

- ✅ 全部 `.py` 源文件完整无缺
- ✅ `pyproject.toml` + `uv.lock` 完整
- ✅ `build/eyetrax_gaze/eyetrax_gaze.exe`（14.3MB）是学长用 PyInstaller 打包的**独立可执行文件**，内嵌了 Python + 全部依赖，理论上可以直接双击运行 `gaze_typing_suite.py`，**无需任何 Python 环境**
- ✅ `dist/eyetrax_gaze/_internal/` 内含所有已打包的依赖

---

### 🔧 修复方案 (选一个执行)

#### 方案 A: pip 直接装到系统 Python（最快，2 条命令）

```powershell
# 1. 安装缺失的 4 个包
D:\Python312\python.exe -m pip install opencv-python mediapipe screeninfo pyvirtualcam

# 2. 以可编辑模式安装 eyetrax
D:\Python312\python.exe -m pip install -e D:\eyetrax-1\eyetrax-1
```

之后所有脚本用 `D:\Python312\python.exe` 运行:
```powershell
D:\Python312\python.exe -m eyetrax.app.whack_a_mole --timed 30
```

#### 方案 B: 重建虚拟环境（推荐，和学长一致）

```powershell
# 1. 安装 uv 包管理器
powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"

# 2. 删除两个坏掉的 venv
Remove-Item -Recurse -Force D:\eyetrax-1\eyetrax-1\.venv
Remove-Item -Recurse -Force D:\eyetrax-1\eyetrax-1\.venv_pack

# 3. 进入项目目录，uv 自动创建新 venv 并安装全部依赖
Set-Location D:\eyetrax-1\eyetrax-1
uv sync

# 4. 激活新 venv
.venv\Scripts\activate

# 5. 以可编辑模式安装 eyetrax 本身
pip install -e .
```

之后正常使用:
```powershell
python -m eyetrax.app.whack_a_mole --timed 30
```

---

### ⚡ 环境修复后的验证检查清单

修复完成后，逐条验证:

```powershell
# 1. 基础导入
python -c "import cv2; print('cv2:', cv2.__version__)"
python -c "import mediapipe; print('mediapipe OK')"
python -c "import numpy; print('numpy:', numpy.__version__)"
python -c "import sklearn; print('sklearn OK')"

# 2. eyetrax 包导入
python -c "from eyetrax import GazeEstimator; print('eyetrax OK')"

# 3. 新游戏文件语法
python -c "import py_compile; py_compile.compile('src/eyetrax/app/whack_a_mole.py', doraise=True); print('whack_a_mole syntax OK')"
```

---

### 4.7 ⚡ 轻微：whack_a_mole.py 未注册到 pyproject.toml

**文件**: `pyproject.toml`  
**问题**: 新游戏没有 CLI 入口，只能通过 `python -m eyetrax.app.whack_a_mole` 运行

**如需添加** (可选):
```toml
[project.scripts]
eyetrax-whack = "eyetrax.app.whack_a_mole:run_whack_a_mole"
```

### 4.8 ⚡ 轻微：游戏代码未经实际运行测试

以下功能**代码逻辑已写但未实测**（因环境损坏无法运行）：

- [ ] 标定后到菜单的过渡是否流畅
- [ ] Dwell 触发时间是否准确（0.8s 默认值可能需根据摄像头帧率调整）
- [ ] 地鼠随机位置会不会偶尔超出屏幕边界
- [ ] PNG alpha 混合在边界裁剪时是否正确
- [ ] 倒计时动画在高 DPI 屏幕上的字体大小是否合适
- [ ] `cv2.waitKey(1)` 的按键检测（R/Q 重玩/退出）在英文键盘上是否正常
- [ ] 长时间运行是否存在内存泄漏（`_draw_menu()` 等函数每帧调用 `canvas.copy()`）

### 4.9 ⚡ 轻微：可能的性能优化点

- `_draw_hud()` 和 `_draw_menu()` 每帧调用 `cv2.getTextSize` 多次 → 可预计算缓存
- 每帧创建 overlay 用 `canvas.copy()` → 可复用一块预分配 buffer
- KDE 平滑器 (`--filter kde`) 在高分辨率屏幕上计算量大 → 建议默认用 `--filter none` 或 `kalman`

### 4.10 建议增强 (未实现，预留扩展空间)

- [ ] 难度递增（随时间加快地鼠刷新速度）
- [ ] 音效反馈（命中音效、combo 音效）
- [ ] 最高分持久化记录（存 JSON 文件）
- [ ] 多只地鼠同时出现模式
- [ ] 道具系统（加分道具、减速道具等）

---

## 5. 代码关键位置索引

| 内容 | 文件位置 |
|------|----------|
| 地鼠 PNG 加载 + alpha 混合 | `whack_a_mole.py` Mole 类, `_draw_sprite()` |
| 地鼠手绘 fallback | `whack_a_mole.py` Mole 类, `_draw_fallback()` |
| Dwell 命中判定 | `whack_a_mole.py` `_update_playing()` 第 ~380 行 |
| 菜单选择逻辑 | `whack_a_mole.py` `_update_menu()` |
| 3-2-1 倒计时 | `whack_a_mole.py` `_update_countdown()` |
| HUD 渲染 | `whack_a_mole.py` `_draw_hud()` |
| 结算画面 | `whack_a_mole.py` `_update_game_over()` |
| CLI 参数 | `whack_a_mole.py` `_parse_args()` |
| 标定初始化 | `whack_a_mole.py` `_setup()` (参考 `demo.py`) |
| 参考: 原版 dwell 逻辑 | `gaze_typing.py` 第 383-420 行 |
| 参考: 模式选择 UI | `gaze_typing_suite.py` `run_gaze_typing_suite()` 的 select 模式 |
| 参考: 模型注册机制 | `models/__init__.py` |

---

## 6. 启动命令参考

```bash
# 先修复环境
uv sync
# 或: pip install -e .

# 最简单启动
python -m eyetrax.app.whack_a_mole

# 完整参数启动
python -m eyetrax.app.whack_a_mole \
  --filter none \
  --calibration 9p \
  --dwell 0.8 \
  --timed 60 \
  --mole-image ./assets/mole.png \
  --mole-ttl 3.0 \
  --mole-radius 80

# 使用预标定模型 (跳过标定)
python -m eyetrax.app.whack_a_mole --model-file gaze_model.pkl --timed 30

# 生命模式
python -m eyetrax.app.whack_a_mole --lives 5 --dwell 1.0

# 快速测试 (短时间)
python -m eyetrax.app.whack_a_mole --timed 15 --dwell 0.5 --mole-ttl 2.0
```

---

*文档生成时间: 2026-06-25 | 代码版本: EyeTrax 0.3.1 + whack_a_mole v1.0*
