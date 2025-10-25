"""
市场扫描和深度分析系统
高效的市场扫描和深度分析功能，支持数百交易对的并行处理

主要组件：
- MarketScanner: 主扫描器类
- 缓存系统: MemoryCache和RedisCache
- 策略系统: PriorityStrategy和FilterStrategy
- 监控系统: PerformanceMonitor和AlertManager
- 工具类: DataProcessor和MetricsCalculator
- 配置管理: ConfigManager和各种预设配置
"""

from .core import MarketScanner, ScanConfig, ScanResult, ScanReport, ScanStatus
from .cache import MemoryCache, RedisCache, cache_result, async_cache_result
from .strategies import (
    BaseStrategy, PriorityStrategy, FilterStrategy, CombinedStrategy,
    StrategyFactory, FilterLevel, PriorityMethod, FilterConfig, PriorityConfig
)
from .monitoring import (
    PerformanceMonitor, PerformanceMetrics, ScanTracker, RealTimeMonitor,
    AlertManager, Alert, AlertSeverity, AlertStatus,
    create_opportunity_alert, create_performance_alert
)
from .utils import DataProcessor, TechnicalIndicators, MetricsCalculator, ScoreWeights, RiskMetrics
from .config import (
    ConfigManager, ScannerConfig, StrategyConfig, AlertConfig,
    DatabaseConfig, MonitoringConfig, SystemConfig, PresetConfigs
)

__version__ = "1.0.0"
__author__ = "AlphaSeeker Scanner Team"

# 包级别常量
DEFAULT_CONFIG_PATH = "scanner_config.json"
DEFAULT_LOG_LEVEL = "INFO"

# 便利函数
def create_scanner(config=None, **kwargs):
    """
    创建扫描器实例
    
    Args:
        config: 系统配置
        **kwargs: 配置参数
        
    Returns:
        MarketScanner实例
    """
    if config is None:
        config = SystemConfig()
    
    # 合并配置
    if kwargs:
        # 这里应该使用ConfigManager来合并配置
        pass
    
    # 创建扫描器
    scanner_config = ScanConfig(
        max_workers=config.scanner.max_workers,
        batch_size=config.scanner.batch_size,
        timeout=config.scanner.timeout,
        enable_deep_analysis=config.scanner.enable_deep_analysis,
        deep_analysis_threshold=config.scanner.deep_analysis_threshold,
        max_deep_analysis_pairs=config.scanner.max_deep_analysis_pairs,
        cache_ttl=config.scanner.cache_ttl,
        enable_redis=config.scanner.enable_redis,
        enable_monitoring=config.monitoring.enable_monitoring,
        alert_threshold=config.alert.high_opportunity_threshold
    )
    
    # 创建扫描器实例
    scanner = MarketScanner(
        config=scanner_config,
        redis_client=None  # 实际使用时需要传入Redis客户端
    )
    
    return scanner


def load_config(config_path=None):
    """
    加载配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        系统配置
    """
    config_manager = ConfigManager(config_path)
    return config_manager.get_config()


def create_preset_config(preset_type: str) -> SystemConfig:
    """
    创建预设配置
    
    Args:
        preset_type: 预设类型 ("high_frequency", "quality_focused", "balanced")
        
    Returns:
        系统配置
    """
    preset_map = {
        "high_frequency": PresetConfigs.high_frequency_config,
        "quality_focused": PresetConfigs.quality_focused_config,
        "balanced": PresetConfigs.balanced_config
    }
    
    if preset_type not in preset_map:
        raise ValueError(f"Unknown preset type: {preset_type}")
    
    return preset_map[preset_type]()


# 导入快捷方式
from typing import Optional

__all__ = [
    # 核心组件
    'MarketScanner',
    'ScanConfig',
    'ScanResult', 
    'ScanReport',
    'ScanStatus',
    
    # 缓存系统
    'MemoryCache',
    'RedisCache',
    'cache_result',
    'async_cache_result',
    
    # 策略系统
    'BaseStrategy',
    'PriorityStrategy',
    'FilterStrategy',
    'CombinedStrategy',
    'StrategyFactory',
    'FilterLevel',
    'PriorityMethod',
    'FilterConfig',
    'PriorityConfig',
    
    # 监控系统
    'PerformanceMonitor',
    'PerformanceMetrics',
    'ScanTracker',
    'RealTimeMonitor',
    'AlertManager',
    'Alert',
    'AlertSeverity',
    'AlertStatus',
    'create_opportunity_alert',
    'create_performance_alert',
    
    # 工具类
    'DataProcessor',
    'TechnicalIndicators',
    'MetricsCalculator',
    'ScoreWeights',
    'RiskMetrics',
    
    # 配置管理
    'ConfigManager',
    'ScannerConfig',
    'StrategyConfig',
    'AlertConfig',
    'DatabaseConfig',
    'MonitoringConfig',
    'SystemConfig',
    'PresetConfigs',
    
    # 便利函数
    'create_scanner',
    'load_config',
    'create_preset_config',
    
    # 常量
    '__version__',
    '__author__',
    'DEFAULT_CONFIG_PATH',
    'DEFAULT_LOG_LEVEL'
]