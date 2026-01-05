"""
ECG信号处理参数配置文件
======================

本文件定义了不同ECG导联在online和offline模式下的信号处理参数。
使用统一的 get_signal_params() 函数获取参数。

参数说明:
---------
mode: 'online' 或 'offline'
    - online: 实时蓝牙数据采集模式，采样率250Hz，需要处理实时数据流
    - offline: 离线文件分析模式，采样率360Hz，处理MIT-BIH等标准数据库

signal_name: ECG导联名称
    - 胸导联: V1, V2, V3, V4, V5, V6
    - 肢体导联: I, II, III, aVR, aVL, aVF
    - 加压导联: MLII, MLIII

通用参数说明:
-------------
- low: 带通滤波器低频截止频率 (Hz)，用于去除基线漂移
- high: 带通滤波器高频截止频率 (Hz)，用于去除高频噪声
- filter_order: 滤波器阶数，使用Butterworth滤波器
- original_weight: 原始信号权重，用于加权组合
- filtered_weight: 滤波后信号权重，用于加权组合
- integration_window_size: 积分窗口大小 (秒)，用于移动窗口积分
- refractory_period: QRS检测不应期 (秒)，避免同一QRS波被重复检测
- threshold_factor: 阈值系数，用于自适应阈值计算

online模式独有参数:
-------------------
- compensation_ms: 相位延迟补偿时间 (毫秒)，补偿滤波和积分引入的延迟
- ema_alpha: 阈值平滑系数（指数移动平均 EMA），越小变化越慢
- q_wave_search_start: Q波搜索窗口起点 (秒)，R峰前的搜索时间
- q_wave_search_end: Q波搜索窗口终点 (秒)，R峰前的搜索时间
- q_wave_min_amplitude: Q波最小幅值 (mV)
- s_wave_search_start: S波搜索窗口起点 (秒)，R峰后的搜索时间
- s_wave_search_end: S波搜索窗口终点 (秒)，R峰后的搜索时间
- s_wave_min_amplitude: S波最小幅值 (mV)
- p_wave_search_start: P波搜索窗口起点 (秒)，R峰前的搜索时间
- p_wave_search_end: P波搜索窗口终点 (秒)，R峰前的搜索时间
- p_wave_min_amplitude: P波最小幅值 (mV)
- p_wave_max_width: P波最大宽度 (秒)
- t_wave_search_start: T波搜索窗口起点 (秒)，R峰后的搜索时间
- t_wave_search_end: T波搜索窗口终点 (秒)，R峰后的搜索时间
- t_wave_min_amplitude: T波最小幅值 (mV)
- t_wave_max_width: T波最大宽度 (秒)

offline模式独有参数:
--------------------
- detection_window_size: 检测窗口大小 (秒)，滑动窗口检测的窗口长度
- overlap_window_size: 重叠窗口大小 (秒)，滑动窗口之间的重叠区域
"""

import json
import os
import warnings

# ============================================================================
# JSON文件路径配置
# ============================================================================

# 获取当前文件所在目录
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# JSON配置文件路径
_JSON_CONFIG_PATH = os.path.join(_CURRENT_DIR, 'signal_params.json')

# ============================================================================
# 参数加载函数
# ============================================================================

def load_signal_params_from_json(json_path: str = None):
    """
    从JSON文件加载ECG信号处理参数，
    
    参数:
        json_path: JSON文件路径，默认为当前目录下的signal_params.json
    
    返回:
        signal_params: 包含所有导联参数的字典
    
    异常:
        FileNotFoundError: 当JSON文件不存在时抛出
        json.JSONDecodeError: 当JSON文件格式错误时抛出
    """
    if json_path is None:
        json_path = _JSON_CONFIG_PATH
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            params = json.load(f)
        print(f"成功从 {json_path} 加载参数配置")
        return params
    except FileNotFoundError:
        raise FileNotFoundError(f"找不到配置文件: {json_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"JSON文件格式错误: {e.msg}", e.doc, e.pos)


# ============================================================================
# 从JSON文件加载参数
# ============================================================================

try:
    SIGNAL_PARAMS = load_signal_params_from_json()
except FileNotFoundError:
    warnings.warn(
        f"未找到配置文件 {_JSON_CONFIG_PATH}，请确保文件存在",
        UserWarning
    )
    SIGNAL_PARAMS = {"online": {}, "offline": {}}
except json.JSONDecodeError as e:
    warnings.warn(
        f"配置文件格式错误: {e}",
        UserWarning
    )
    SIGNAL_PARAMS = {"online": {}, "offline": {}}

# ============================================================================
# 参数获取和管理函数
# ============================================================================


def get_signal_params(mode: str, signal_name: str):
    """
    获取ECG信号处理参数的统一接口
    
    参数:
        mode: 'online' 或 'offline'
            - 'online': 实时蓝牙数据采集模式
            - 'offline': 离线文件分析模式
        signal_name: ECG导联名称
            - 胸导联: 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'
            - 肢体导联: 'I', 'II', 'III', 'aVR', 'aVL', 'aVF'
            - 加压导联: 'MLII', 'MLIII'
    
    返回:
        signal_params: 包含该导联和模式对应的所有参数的字典
    
    异常:
        ValueError: 当mode或signal_name无效时抛出
    
    使用示例:
        >>> # online模式
        >>> params = get_signal_params('online', 'MLII')
        >>> # offline模式
        >>> params = get_signal_params('offline', 'V1')
    """
    # 验证mode参数
    if mode not in ['online', 'offline']:
        raise ValueError(f"无效的mode参数: '{mode}'。必须是 'online' 或 'offline'")
    
    # 验证signal_name参数
    if signal_name not in SIGNAL_PARAMS[mode]:
        # 如果找不到指定的导联，使用默认参数（MLII的参数）
        warnings.warn(
            f"未找到导联 '{signal_name}' 在 {mode} 模式中的参数，使用MLII默认参数",
            UserWarning
        )
        signal_name = 'MLII'
    
    # 返回参数字典的副本，避免意外修改原始配置
    return SIGNAL_PARAMS[mode][signal_name].copy()
