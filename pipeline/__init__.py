"""
多策略信号处理管道 (Multi-Strategy Signal Processing Pipeline)

该模块实现了基于双重验证架构的统一信号处理管道，整合了：
- LightGBM快速筛选 (第一层)
- 本地LLM深度评估 (第二层)  
- 技术指标触发机制
- 机器学习预测
- 动态策略权重算法
- 策略融合和冲突解决
- 信号优先级排序
- 策略回测和验证
- 策略性能监控
"""

from .pipeline import MultiStrategyPipeline
from .signal_processor import SignalProcessor
from .strategy_fusion import StrategyFusion
from .performance_monitor import PerformanceMonitor
from .priority_manager import PriorityManager
from .backtest_validator import BacktestValidator
from .types import PipelineConfig, StrategySignal, FusionResult, PerformanceMetrics

__version__ = "1.0.0"
__author__ = "AlphaSeeker Team"

__all__ = [
    "MultiStrategyPipeline",
    "SignalProcessor", 
    "StrategyFusion",
    "PerformanceMonitor",
    "PriorityManager", 
    "BacktestValidator",
    "PipelineConfig",
    "StrategySignal",
    "FusionResult",
    "PerformanceMetrics"
]