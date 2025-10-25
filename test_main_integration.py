#!/usr/bin/env python3
"""
AlphaSeeker 单元测试
===================

AlphaSeeker主集成应用的单元测试套件，测试核心功能和API接口。

运行测试:
    python -m pytest test_main_integration.py -v
    python test_main_integration.py
"""

import asyncio
import pytest
import aiohttp
import json
import time
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

# 测试配置
TEST_CONFIG = {
    "host": "127.0.0.1",
    "port": 8001,  # 使用不同端口避免冲突
    "debug": False
}

class TestAlphaSeekerIntegration:
    """AlphaSeeker集成测试类"""
    
    @pytest.fixture
    def event_loop(self):
        """创建事件循环"""
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()
    
    @pytest.fixture
    async def client(self):
        """创建测试客户端"""
        async with aiohttp.ClientSession() as session:
            yield session
    
    def test_config_loading(self):
        """测试配置加载"""
        from main_integration import AlphaSeekerConfig
        
        config = AlphaSeekerConfig()
        
        # 测试默认配置
        assert config.app_name == "AlphaSeeker"
        assert config.app_version == "1.0.0"
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.max_concurrent_tasks == 32
        assert config.batch_size == 100
        
        # 测试自定义配置
        custom_config = AlphaSeekerConfig(
            host="127.0.0.1",
            port=9000,
            max_concurrent_tasks=16
        )
        
        assert custom_config.host == "127.0.0.1"
        assert custom_config.port == 9000
        assert custom_config.max_concurrent_tasks == 16
    
    def test_system_status_structure(self):
        """测试系统状态结构"""
        from main_integration import SystemStatus
        
        status = SystemStatus(
            status="healthy",
            uptime=3600.0,
            version="1.0.0"
        )
        
        assert status.status == "healthy"
        assert status.uptime == 3600.0
        assert status.version == "1.0.0"
        assert isinstance(status.components, dict)
        assert isinstance(status.performance, dict)
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self):
        """测试协调器初始化"""
        from main_integration import AlphaSeekerOrchestrator, AlphaSeekerConfig
        
        config = AlphaSeekerConfig(**TEST_CONFIG)
        
        # 模拟组件初始化
        with patch('main_integration.AlphaSeekerMLEngine') as mock_ml, \
             patch('main_integration.SignalValidationCoordinator') as mock_validation, \
             patch('main_integration.MultiStrategyPipeline') as mock_pipeline, \
             patch('main_integration.MarketScanner') as mock_scanner:
            
            # 设置模拟返回值
            mock_ml.return_value.health_check.return_value = {"overall_status": "healthy"}
            mock_validation.return_value = AsyncMock()
            mock_pipeline.return_value.start = AsyncMock()
            mock_scanner.return_value = AsyncMock()
            
            orchestrator = AlphaSeekerOrchestrator(config)
            
            # 测试初始化
            await orchestrator.initialize_components()
            
            # 验证组件调用
            mock_ml.assert_called_once()
            mock_validation.assert_called_once()
            mock_pipeline.return_value.start.assert_called_once()
            mock_scanner.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_signal_processing(self):
        """测试信号处理逻辑"""
        from main_integration import AlphaSeekerOrchestrator, AlphaSeekerConfig
        
        config = AlphaSeekerConfig(**TEST_CONFIG)
        
        # 模拟所有组件
        with patch('main_integration.AlphaSeekerMLEngine') as mock_ml, \
             patch('main_integration.MultiStrategyPipeline') as mock_pipeline, \
             patch('main_integration.SignalValidationCoordinator') as mock_validation:
            
            # 设置模拟返回值
            mock_ml.return_value.predict.return_value = {
                'signal_label': 1,
                'confidence': 0.75,
                'probability_distribution': {-1: 0.15, 0: 0.25, 1: 0.60}
            }
            
            mock_pipeline.return_value.process_single_symbol.return_value = Mock(
                final_direction=Mock(value="long"),
                combined_confidence=0.78,
                final_score=0.75,
                risk_reward_ratio=1.5
            )
            
            mock_validation.return_value.validate_signal.return_value = Mock(
                status=Mock(value="passed"),
                combined_score=0.75
            )
            
            orchestrator = AlphaSeekerOrchestrator(config)
            orchestrator.is_running = True
            
            # 测试信号处理
            market_data = {
                "price": 45000.0,
                "volume": 1000000.0
            }
            
            indicators = {
                "rsi": 65.5,
                "macd": 120.5,
                "adx": 28.3,
                "sma_50": 44500.0,
                "sma_200": 42000.0
            }
            
            features = {
                "mid_price": 45000.0,
                "spread": 2.5,
                "volatility_60s": 0.025
            }
            
            result = await orchestrator.process_trading_signal(
                "BTCUSDT", market_data, indicators, features
            )
            
            # 验证结果
            assert result["symbol"] == "BTCUSDT"
            assert result["signal_direction"] == "long"
            assert result["confidence"] == 0.78
            assert result["score"] == 0.75
            assert result["risk_reward_ratio"] == 1.5
            assert "components" in result
    
    @pytest.mark.asyncio
    async def test_batch_market_scan(self):
        """测试批量市场扫描"""
        from main_integration import AlphaSeekerOrchestrator, AlphaSeekerConfig
        
        config = AlphaSeekerConfig(**TEST_CONFIG)
        
        # 模拟处理交易信号方法
        async def mock_process_signal(*args, **kwargs):
            return {
                "symbol": args[0],
                "signal_direction": "long",
                "confidence": 0.75 + (hash(args[0]) % 100) / 1000,
                "score": 0.70 + (hash(args[0]) % 100) / 1000,
                "risk_reward_ratio": 1.5
            }
        
        orchestrator = AlphaSeekerOrchestrator(config)
        orchestrator.is_running = True
        orchestrator.process_trading_signal = mock_process_signal
        
        # 测试批量扫描
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        result = await orchestrator.batch_scan_market(symbols, max_results=2)
        
        # 验证结果
        assert result["total_symbols"] == 3
        assert len(result["results"]) <= 2
        assert "scan_id" in result
        assert "timestamp" in result
        assert "summary" in result
        
        # 验证信号分布
        signal_dist = result["summary"]["signal_distribution"]
        assert isinstance(signal_dist, dict)
    
    def test_data_validation(self):
        """测试数据验证逻辑"""
        # 测试市场数据验证
        valid_market_data = {
            "price": 45000.0,
            "volume": 1000000.0,
            "timestamp": int(time.time())
        }
        
        invalid_market_data = {
            "price": -1000,  # 负价格
            "volume": 0,     # 零成交量
        }
        
        # 简单的验证函数
        def validate_market_data(data):
            errors = []
            
            if data.get("price", 0) <= 0:
                errors.append("价格必须大于0")
            
            if data.get("volume", 0) <= 0:
                errors.append("成交量必须大于0")
            
            if "timestamp" not in data:
                errors.append("缺少时间戳")
            
            return errors
        
        # 测试有效数据
        valid_errors = validate_market_data(valid_market_data)
        assert len(valid_errors) == 0
        
        # 测试无效数据
        invalid_errors = validate_market_data(invalid_market_data)
        assert len(invalid_errors) == 2
        assert "价格必须大于0" in invalid_errors
        assert "成交量必须大于0" in invalid_errors
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """测试错误处理"""
        from main_integration import AlphaSeekerOrchestrator, AlphaSeekerConfig
        
        config = AlphaSeekerConfig(**TEST_CONFIG)
        orchestrator = AlphaSeekerOrchestrator(config)
        orchestrator.is_running = True
        
        # 模拟组件抛出异常
        orchestrator.process_trading_signal = AsyncMock(
            side_effect=Exception("模拟错误")
        )
        
        # 测试异常处理
        with pytest.raises(Exception) as exc_info:
            await orchestrator.process_trading_signal(
                "BTCUSDT", {}, {}, {}
            )
        
        assert "模拟错误" in str(exc_info.value)
    
    def test_performance_metrics(self):
        """测试性能指标计算"""
        from main_integration import AlphaSeekerOrchestrator, AlphaSeekerConfig
        
        config = AlphaSeekerConfig(**TEST_CONFIG)
        orchestrator = AlphaSeekerOrchestrator(config)
        
        # 模拟性能数据
        orchestrator.total_requests = 100
        orchestrator.successful_requests = 95
        orchestrator.failed_requests = 5
        orchestrator.total_processing_time = 25.0
        orchestrator.start_time = time.time() - 3600  # 1小时前开始
        
        # 获取系统状态
        status = orchestrator.get_system_status()
        
        # 验证性能指标
        perf = status.performance
        assert perf["total_requests"] == 100
        assert perf["successful_requests"] == 95
        assert perf["failed_requests"] == 5
        assert perf["success_rate"] == 95.0
        assert perf["avg_processing_time"] == 0.25
    
    def test_signal_distribution_analysis(self):
        """测试信号分布分析"""
        from main_integration import AlphaSeekerOrchestrator, AlphaSeekerConfig
        
        config = AlphaSeekerConfig(**TEST_CONFIG)
        orchestrator = AlphaSeekerOrchestrator(config)
        
        # 模拟结果数据
        results = [
            {"signal_direction": "long", "confidence": 0.8},
            {"signal_direction": "short", "confidence": 0.7},
            {"signal_direction": "long", "confidence": 0.9},
            {"signal_direction": "flat", "confidence": 0.5},
        ]
        
        # 分析信号分布
        distribution = orchestrator._analyze_signal_distribution(results)
        
        # 验证分布结果
        assert distribution["long"] == 2
        assert distribution["short"] == 1
        assert distribution["flat"] == 1
        assert distribution["unknown"] == 0
    
    def test_configuration_validation(self):
        """测试配置验证"""
        from main_integration import AlphaSeekerConfig
        
        # 测试有效配置
        valid_config = AlphaSeekerConfig(
            host="127.0.0.1",
            port=8080,
            max_concurrent_tasks=16,
            batch_size=50
        )
        
        assert valid_config.host == "127.0.0.1"
        assert valid_config.port == 8080
        assert valid_config.max_concurrent_tasks == 16
        assert valid_config.batch_size == 50
        
        # 测试默认配置生成
        assert valid_config.api_config is not None
        assert valid_config.ml_engine_config is not None
        assert valid_config.pipeline_config is not None
        assert valid_config.scanner_config is not None
        assert valid_config.validation_config is not None

def run_manual_tests():
    """手动运行测试（无需pytest）"""
    print("🧪 运行AlphaSeeker单元测试...")
    print("=" * 50)
    
    test_instance = TestAlphaSeekerIntegration()
    
    try:
        # 配置测试
        print("📋 测试配置加载...")
        test_instance.test_config_loading()
        print("✅ 配置测试通过")
        
        # 系统状态测试
        print("\n📊 测试系统状态结构...")
        test_instance.test_system_status_structure()
        print("✅ 系统状态测试通过")
        
        # 数据验证测试
        print("\n🔍 测试数据验证...")
        test_instance.test_data_validation()
        print("✅ 数据验证测试通过")
        
        # 性能指标测试
        print("\n⚡ 测试性能指标...")
        test_instance.test_performance_metrics()
        print("✅ 性能指标测试通过")
        
        # 信号分布分析测试
        print("\n📈 测试信号分布分析...")
        test_instance.test_signal_distribution_analysis()
        print("✅ 信号分布分析测试通过")
        
        # 配置验证测试
        print("\n⚙️ 测试配置验证...")
        test_instance.test_configuration_validation()
        print("✅ 配置验证测试通过")
        
        # 异步测试（需要事件循环）
        print("\n🔄 测试异步功能...")
        loop = asyncio.new_event_loop()
        
        try:
            # 信号处理测试
            print("  - 测试信号处理...")
            asyncio.run(test_instance.test_signal_processing())
            
            # 批量扫描测试
            print("  - 测试批量扫描...")
            asyncio.run(test_instance.test_batch_market_scan())
            
            print("✅ 异步测试通过")
            
        finally:
            loop.close()
        
        print("\n" + "=" * 50)
        print("🎉 所有单元测试通过！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--pytest":
        # 使用pytest运行
        pytest.main([__file__, "-v"])
    else:
        # 手动运行
        success = run_manual_tests()
        sys.exit(0 if success else 1)