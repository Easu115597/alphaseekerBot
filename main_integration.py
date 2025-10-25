#!/usr/bin/env python3
"""
AlphaSeeker 主集成应用
==================

AlphaSeeker系统的核心集成应用，协调所有组件：
- 集成API服务
- 机器学习引擎
- 多策略管道
- 市场扫描器
- 双重验证器

提供统一的使用接口和完整的系统管理功能。

作者: AlphaSeeker Team
版本: 1.0.0
日期: 2025-10-25
"""

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

import asyncio
import logging
import os
import sys
import signal
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import yaml
from concurrent.futures import ThreadPoolExecutor

# FastAPI and HTTP
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# 项目模块导入
try:
    # API服务
    from integrated_api.main import app as api_app, setup_logging as setup_api_logging
    
    # ML引擎
    from ml_engine import AlphaSeekerMLEngine, MODEL_CONFIG, RISK_CONFIG
    
    # 管道
    from pipeline import MultiStrategyPipeline, PipelineConfig, StrategyType
    from pipeline.types import (
        MarketData, TechnicalIndicators, MLPrediction, FusionResult,
        ScanRequest, ScanResult, SignalDirection
    )
    
    # 扫描器
    from scanner import MarketScanner, ScanConfig, ScanStrategy
    
    # 验证器
    from validation import (
        SignalValidationCoordinator, ValidationConfig, ValidationRequest,
        ValidationResult, ValidationPriority, LightGBMConfig, LLMConfig,
        FusionConfig, FusionStrategy, LLMProvider
    )
    
    print("✅ 所有模块导入成功")
    
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    print("请确保所有依赖模块已正确安装")
    sys.exit(1)

# 全局配置
@dataclass
class AlphaSeekerConfig:
    """AlphaSeeker主配置类"""
    # 应用基础配置
    app_name: str = "AlphaSeeker"
    app_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    debug: bool = False
    
    # 组件配置
    api_config: Dict[str, Any] = None
    ml_engine_config: Dict[str, Any] = None
    pipeline_config: Dict[str, Any] = None
    scanner_config: Dict[str, Any] = None
    validation_config: Dict[str, Any] = None
    
    # 性能配置
    max_concurrent_tasks: int = 32
    request_timeout: float = 30.0
    batch_size: int = 100
    enable_cache: bool = True
    
    # 日志配置
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 数据路径
    data_dir: str = "data"
    model_dir: str = "models"
    log_dir: str = "logs"
    cache_dir: str = "cache"
    
    def __post_init__(self):
        """初始化后的配置处理"""
        if self.api_config is None:
            self.api_config = self._default_api_config()
        if self.ml_engine_config is None:
            self.ml_engine_config = self._default_ml_config()
        if self.pipeline_config is None:
            self.pipeline_config = self._default_pipeline_config()
        if self.scanner_config is None:
            self.scanner_config = self._default_scanner_config()
        if self.validation_config is None:
            self.validation_config = self._default_validation_config()
    
    def _default_api_config(self) -> Dict[str, Any]:
        """默认API配置"""
        return {
            "cors_origins": ["*"],
            "log_level": "INFO",
            "log_format": self.log_format,
            "host": self.host,
            "port": self.port,
            "reload": self.reload
        }
    
    def _default_ml_config(self) -> Dict[str, Any]:
        """默认ML引擎配置"""
        return {
            "model_config": MODEL_CONFIG,
            "risk_config": RISK_CONFIG,
            "enable_caching": self.enable_cache,
            "target_latency_ms": 500
        }
    
    def _default_pipeline_config(self) -> Dict[str, Any]:
        """默认管道配置"""
        return {
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "timeout_seconds": 10.0,
            "ml_probability_threshold": 0.65,
            "llm_confidence_threshold": 0.65,
            "strategy_weights": {
                StrategyType.TECHNICAL_INDICATOR: 0.4,
                StrategyType.ML_PREDICTION: 0.2,
                StrategyType.RISK_MODEL: 0.2,
                StrategyType.BACKTEST_REFERENCE: 0.2
            }
        }
    
    def _default_scanner_config(self) -> Dict[str, Any]:
        """默认扫描器配置"""
        return {
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "scan_timeout": 15.0,
            "batch_size": self.batch_size,
            "enable_cache": self.enable_cache
        }
    
    def _default_validation_config(self) -> Dict[str, Any]:
        """默认验证器配置"""
        return {
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "lgbm_config": LightGBMConfig(
                probability_threshold=0.65,
                confidence_threshold=0.6
            ),
            "llm_config": LLMConfig(
                provider=LLMProvider.OLLAMA,
                base_url="http://localhost:11434",
                model_name="llama2:13b"
            ),
            "fusion_config": FusionConfig(
                strategy=FusionStrategy.ADAPTIVE_WEIGHT,
                risk_reward_threshold=1.2
            )
        }

# 系统状态类
@dataclass
class SystemStatus:
    """系统状态信息"""
    status: str = "initializing"
    uptime: float = 0.0
    version: str = "1.0.0"
    components: Dict[str, Dict[str, Any]] = None
    performance: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.components is None:
            self.components = {}
        if self.performance is None:
            self.performance = {}

class AlphaSeekerOrchestrator:
    """AlphaSeeker系统协调器 - 核心组件"""
    
    def __init__(self, config: AlphaSeekerConfig):
        """初始化协调器"""
        self.config = config
        self.logger = None
        self.start_time = None
        self.is_running = False
        
        # 组件实例
        self.ml_engine: Optional[AlphaSeekerMLEngine] = None
        self.pipeline: Optional[MultiStrategyPipeline] = None
        self.scanner: Optional[MarketScanner] = None
        self.validation_coordinator: Optional[SignalValidationCoordinator] = None
        
        # 性能统计
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_processing_time = 0.0
        
        # 创建必要的目录
        self._create_directories()
    
    def _create_directories(self):
        """创建必要的目录"""
        directories = [
            self.config.data_dir,
            self.config.model_dir,
            self.config.log_dir,
            self.config.cache_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """设置日志系统"""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        
        # 创建格式化器
        formatter = logging.Formatter(self.config.log_format)
        
        # 文件处理器
        file_handler = logging.FileHandler(
            os.path.join(self.config.log_dir, "alphaseeker.log")
        )
        file_handler.setFormatter(formatter)
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        # 配置根日志器
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        root_logger.handlers.clear()
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # 设置第三方库的日志级别
        logging.getLogger("ccxt").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("aiohttp").setLevel(logging.WARNING)
        logging.getLogger("lightgbm").setLevel(logging.WARNING)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("📝 日志系统初始化完成")
    
    async def initialize_components(self):
        """初始化所有组件"""
        try:
            self.logger.info("🚀 开始初始化AlphaSeeker组件...")
            
            # 1. 初始化ML引擎
            self.logger.info("🧠 初始化机器学习引擎...")
            self.ml_engine = AlphaSeekerMLEngine(
                config=self.config.ml_engine_config,
                logger=self.logger
            )
            ml_health = self.ml_engine.health_check()
            self.logger.info(f"ML引擎状态: {ml_health['overall_status']}")
            
            # 2. 初始化验证器
            self.logger.info("🔍 初始化双重验证器...")
            validation_config = ValidationConfig(**self.config.validation_config)
            self.validation_coordinator = SignalValidationCoordinator(validation_config)
            
            # 3. 初始化管道
            self.logger.info("⚙️ 初始化多策略管道...")
            pipeline_config = PipelineConfig(**self.config.pipeline_config)
            self.pipeline = MultiStrategyPipeline(pipeline_config)
            await self.pipeline.start()
            
            # 4. 初始化扫描器
            self.logger.info("📊 初始化市场扫描器...")
            scan_config = ScanConfig(**self.config.scanner_config)
            self.scanner = MarketScanner(scan_config)
            
            # 更新组件状态
            self._update_component_status("ml_engine", "ready", ml_health)
            self._update_component_status("validation", "ready", {"status": "ready"})
            self._update_component_status("pipeline", "ready", {"status": "ready"})
            self._update_component_status("scanner", "ready", {"status": "ready"})
            
            self.logger.info("✅ 所有组件初始化完成")
            
        except Exception as e:
            self.logger.error(f"❌ 组件初始化失败: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    def _update_component_status(self, component: str, status: str, details: Dict[str, Any]):
        """更新组件状态"""
        if not hasattr(self, '_component_status'):
            self._component_status = {}
        
        self._component_status[component] = {
            "status": status,
            "last_update": datetime.now().isoformat(),
            "details": details
        }
    
    def get_system_status(self) -> SystemStatus:
        """获取系统状态"""
        uptime = time.time() - self.start_time if self.start_time else 0.0
        
        # 收集各组件状态
        components = {}
        
        # ML引擎状态
        if self.ml_engine:
            ml_health = self.ml_engine.health_check()
            components["ml_engine"] = {
                "status": "healthy" if ml_health['overall_status'] == "healthy" else "warning",
                "performance": self.ml_engine.get_performance_stats()
            }
        
        # 验证器状态
        if self.validation_coordinator:
            components["validation"] = self._component_status.get("validation", {"status": "unknown"})
        
        # 管道状态
        if self.pipeline:
            components["pipeline"] = self._component_status.get("pipeline", {"status": "unknown"})
        
        # 扫描器状态
        if self.scanner:
            components["scanner"] = self._component_status.get("scanner", {"status": "unknown"})
        
        # 计算性能指标
        success_rate = (self.successful_requests / max(self.total_requests, 1)) * 100
        avg_processing_time = self.total_processing_time / max(self.total_requests, 1)
        
        performance = {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": round(success_rate, 2),
            "avg_processing_time": round(avg_processing_time, 3),
            "uptime": round(uptime, 2)
        }
        
        return SystemStatus(
            status="healthy" if self.is_running else "stopped",
            uptime=uptime,
            version=self.config.app_version,
            components=components,
            performance=performance
        )
    
    async def start(self):
        """启动系统"""
        try:
            self.start_time = time.time()
            self._setup_logging()
            self.logger.info(f"🚀 启动 AlphaSeeker v{self.config.app_version}")
            
            # 初始化组件
            await self.initialize_components()
            
            self.is_running = True
            self.logger.info("✅ AlphaSeeker系统启动完成")
            
        except Exception as e:
            self.logger.error(f"❌ 系统启动失败: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    async def stop(self):
        """停止系统"""
        self.logger.info("🛑 正在停止AlphaSeeker系统...")
        
        self.is_running = False
        
        # 停止各组件
        try:
            if self.pipeline:
                await self.pipeline.stop()
            
            if self.validation_coordinator:
                await self.validation_coordinator.shutdown()
            
            self.logger.info("✅ AlphaSeeker系统已停止")
            
        except Exception as e:
            self.logger.error(f"❌ 停止系统时出错: {e}")
    
    async def process_trading_signal(self, symbol: str, market_data: Dict[str, Any], 
                                   indicators: Dict[str, Any], 
                                   features: Dict[str, Any]) -> Dict[str, Any]:
        """处理交易信号 - 核心功能"""
        start_time = time.time()
        self.total_requests += 1
        
        try:
            self.logger.info(f"📊 处理 {symbol} 的交易信号")
            
            # 1. ML引擎预测
            ml_prediction = None
            if self.ml_engine:
                ml_result = self.ml_engine.predict(market_data)
                ml_prediction = MLPrediction(
                    label=ml_result['signal_label'],
                    probability_scores=ml_result['probability_distribution'],
                    confidence=ml_result['confidence'],
                    model_version="lightgbm_v2.1.0"
                )
            
            # 2. 市场数据转换
            market = MarketData(
                symbol=symbol,
                timestamp=datetime.now(),
                price=market_data.get('price', 0),
                volume=market_data.get('volume', 0),
                data_freshness=1.0
            )
            
            # 3. 技术指标转换
            technical_indicators = TechnicalIndicators(
                rsi=indicators.get('rsi', 50),
                macd=indicators.get('macd', 0),
                adx=indicators.get('adx', 25),
                sma_50=indicators.get('sma_50', 0),
                sma_200=indicators.get('sma_200', 0)
            )
            
            # 4. 多策略融合
            fusion_result = None
            if self.pipeline and ml_prediction:
                fusion_result = await self.pipeline.process_single_symbol(
                    symbol=symbol,
                    market_data=market,
                    technical_indicators=technical_indicators,
                    ml_prediction=ml_prediction
                )
            
            # 5. 双重验证
            validation_result = None
            if self.validation_coordinator:
                validation_request = ValidationRequest(
                    symbol=symbol,
                    timeframe="1h",
                    current_price=market_data.get('price', 0),
                    features=features,
                    indicators=indicators,
                    risk_context={"volatility": 0.025},
                    priority=ValidationPriority.MEDIUM
                )
                
                validation_result = await self.validation_coordinator.validate_signal(validation_request)
            
            # 6. 合成最终结果
            final_result = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "signal_direction": fusion_result.final_direction.value if fusion_result else "unknown",
                "confidence": fusion_result.combined_confidence if fusion_result else 0.5,
                "score": fusion_result.final_score if fusion_result else 0.5,
                "risk_reward_ratio": fusion_result.risk_reward_ratio if fusion_result else 1.0,
                "processing_time": time.time() - start_time,
                "components": {
                    "ml_prediction": {
                        "label": ml_prediction.label if ml_prediction else None,
                        "confidence": ml_prediction.confidence if ml_prediction else None,
                        "probabilities": ml_prediction.probability_scores if ml_prediction else None
                    },
                    "fusion_result": {
                        "final_score": fusion_result.final_score if fusion_result else None,
                        "confidence": fusion_result.combined_confidence if fusion_result else None
                    } if fusion_result else None,
                    "validation": {
                        "status": validation_result.status.value if validation_result else None,
                        "combined_score": validation_result.combined_score if validation_result else None
                    } if validation_result else None
                }
            }
            
            self.successful_requests += 1
            self.total_processing_time += time.time() - start_time
            
            self.logger.info(f"✅ {symbol} 信号处理完成 - 方向: {final_result['signal_direction']}, 置信度: {final_result['confidence']:.3f}")
            
            return final_result
            
        except Exception as e:
            self.failed_requests += 1
            self.logger.error(f"❌ {symbol} 信号处理失败: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    async def batch_scan_market(self, symbols: List[str], max_results: int = 10) -> Dict[str, Any]:
        """批量市场扫描"""
        start_time = time.time()
        
        try:
            self.logger.info(f"🔍 开始批量扫描市场 - {len(symbols)} 个交易对")
            
            results = []
            
            # 并发处理多个交易对
            tasks = []
            for symbol in symbols:
                # 模拟市场数据（实际中应从数据源获取）
                mock_market_data = {
                    "price": 40000 + hash(symbol) % 10000,
                    "volume": 1000000,
                    "timestamp": time.time()
                }
                
                mock_indicators = {
                    "rsi": 50 + hash(symbol) % 40,
                    "macd": 100 + hash(symbol) % 200,
                    "adx": 20 + hash(symbol) % 20,
                    "sma_50": 42000,
                    "sma_200": 40000
                }
                
                mock_features = {
                    "mid_price": mock_market_data["price"],
                    "spread": 2.5,
                    "volatility_60s": 0.025
                }
                
                task = self.process_trading_signal(
                    symbol, mock_market_data, mock_indicators, mock_features
                )
                tasks.append(task)
            
            # 等待所有任务完成
            symbol_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 过滤和处理结果
            valid_results = []
            for i, result in enumerate(symbol_results):
                if isinstance(result, Exception):
                    self.logger.error(f"处理 {symbols[i]} 时出错: {result}")
                    continue
                
                # 只保留高置信度的结果
                if result['confidence'] >= 0.6:
                    valid_results.append(result)
            
            # 按置信度排序，取前max_results个
            valid_results.sort(key=lambda x: x['confidence'], reverse=True)
            results = valid_results[:max_results]
            
            processing_time = time.time() - start_time
            
            final_result = {
                "scan_id": f"scan_{int(time.time())}",
                "timestamp": datetime.now().isoformat(),
                "total_symbols": len(symbols),
                "processed_symbols": len(symbol_results),
                "valid_results": len(results),
                "results": results,
                "processing_time": processing_time,
                "summary": {
                    "avg_confidence": sum(r['confidence'] for r in results) / max(len(results), 1),
                    "signal_distribution": self._analyze_signal_distribution(results)
                }
            }
            
            self.logger.info(f"✅ 市场扫描完成 - 处理: {len(symbol_results)}个, 有效: {len(results)}个, 用时: {processing_time:.2f}秒")
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"❌ 市场扫描失败: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    def _analyze_signal_distribution(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """分析信号分布"""
        distribution = {"long": 0, "short": 0, "flat": 0, "unknown": 0}
        
        for result in results:
            direction = result.get("signal_direction", "unknown")
            distribution[direction] = distribution.get(direction, 0) + 1
        
        return distribution


# 全局系统实例
orchestrator: Optional[AlphaSeekerOrchestrator] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global orchestrator
    
    # 启动
    try:
        orchestrator = AlphaSeekerOrchestrator(CONFIG)
        await orchestrator.start()
        
        yield
        
    finally:
        # 关闭
        if orchestrator:
            await orchestrator.stop()

# 创建FastAPI应用
app = FastAPI(
    title="AlphaSeeker集成系统",
    description="AlphaSeeker AI驱动的加密货币交易信号系统，集成机器学习、多策略融合和双重验证",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局配置实例
CONFIG = AlphaSeekerConfig()

@app.on_event("startup")
async def startup_event():
    """启动事件"""
    print("🚀 AlphaSeeker集成系统正在启动...")

@app.on_event("shutdown")
async def shutdown_event():
    """关闭事件"""
    print("🛑 AlphaSeeker集成系统正在关闭...")

@app.get("/", tags=["系统"])
async def root():
    """根路径"""
    return {
        "name": "AlphaSeeker集成系统",
        "version": "1.0.0",
        "description": "AI驱动的加密货币交易信号系统",
        "components": [
            "机器学习引擎 (LightGBM)",
            "多策略信号管道",
            "市场扫描器",
            "双重验证器",
            "集成API服务"
        ],
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", tags=["系统"])
async def health_check():
    """健康检查"""
    if not orchestrator or not orchestrator.is_running:
        raise HTTPException(status_code=503, detail="系统未运行")
    
    try:
        status = orchestrator.get_system_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"健康检查失败: {str(e)}")

@app.post("/api/v1/signal/analyze", tags=["交易信号"])
async def analyze_signal(request: Request):
    """分析单个交易信号"""
    if not orchestrator or not orchestrator.is_running:
        raise HTTPException(status_code=503, detail="系统未运行")
    
    try:
        data = await request.json()
        
        required_fields = ["symbol", "market_data", "indicators", "features"]
        for field in required_fields:
            if field not in data:
                raise HTTPException(status_code=400, detail=f"缺少必需字段: {field}")
        
        result = await orchestrator.process_trading_signal(
            symbol=data["symbol"],
            market_data=data["market_data"],
            indicators=data["indicators"],
            features=data["features"]
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"信号分析失败: {str(e)}")

@app.post("/api/v1/scan/market", tags=["市场扫描"])
async def scan_market(request: Request):
    """批量市场扫描"""
    if not orchestrator or not orchestrator.is_running:
        raise HTTPException(status_code=503, detail="系统未运行")
    
    try:
        data = await request.json()
        
        symbols = data.get("symbols", [])
        max_results = data.get("max_results", 10)
        
        if not symbols:
            raise HTTPException(status_code=400, detail="symbols不能为空")
        
        if len(symbols) > 100:  # 限制最大数量
            raise HTTPException(status_code=400, detail="symbols数量不能超过100个")
        
        result = await orchestrator.batch_scan_market(symbols, max_results)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"市场扫描失败: {str(e)}")

@app.get("/api/v1/system/status", tags=["系统"])
async def get_system_status():
    """获取详细系统状态"""
    if not orchestrator or not orchestrator.is_running:
        raise HTTPException(status_code=503, detail="系统未运行")
    
    try:
        status = orchestrator.get_system_status()
        return asdict(status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取系统状态失败: {str(e)}")

@app.get("/api/v1/components", tags=["系统"])
async def get_components_info():
    """获取组件信息"""
    components_info = {
        "ml_engine": {
            "name": "机器学习引擎",
            "description": "LightGBM模型进行交易信号预测",
            "features": ["价格预测", "特征工程", "风险管理"]
        },
        "pipeline": {
            "name": "多策略管道",
            "description": "融合多种策略的交易信号处理管道",
            "features": ["策略融合", "信号优先级", "冲突解决"]
        },
        "scanner": {
            "name": "市场扫描器",
            "description": "多策略市场扫描和机会发现",
            "features": ["批量扫描", "策略多样化", "机会排序"]
        },
        "validation": {
            "name": "双重验证器",
            "description": "LightGBM + LLM双重验证机制",
            "features": ["快速筛选", "深度评估", "结果融合"]
        },
        "api": {
            "name": "集成API",
            "description": "统一的REST API接口服务",
            "features": ["REST API", "CORS支持", "错误处理"]
        }
    }
    
    return {
        "components": components_info,
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/performance", tags=["系统"])
async def get_performance_metrics():
    """获取性能指标"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="系统未运行")
    
    try:
        status = orchestrator.get_system_status()
        
        return {
            "performance": status.performance,
            "system_info": {
                "uptime": status.uptime,
                "version": status.version,
                "config": {
                    "max_concurrent_tasks": CONFIG.max_concurrent_tasks,
                    "batch_size": CONFIG.batch_size,
                    "enable_cache": CONFIG.enable_cache
                }
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取性能指标失败: {str(e)}")

# 异常处理器
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP异常处理"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """通用异常处理"""
    return JSONResponse(
        status_code=500,
        content={"error": "内部服务器错误", "status_code": 500}
    )

def signal_handler(signum, frame):
    """信号处理器"""
    print(f"\n🛑 接收到信号 {signum}，正在关闭系统...")
    if orchestrator:
        asyncio.create_task(orchestrator.stop())
    sys.exit(0)

def setup_signal_handlers():
    """设置信号处理器"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def main():
    """主函数"""
    setup_signal_handlers()
    
    print("=" * 60)
    print("🚀 AlphaSeeker 集成系统")
    print("=" * 60)
    print(f"版本: {CONFIG.app_version}")
    print(f"主机: {CONFIG.host}:{CONFIG.port}")
    print(f"调试: {CONFIG.debug}")
    print(f"并发任务: {CONFIG.max_concurrent_tasks}")
    print(f"批处理大小: {CONFIG.batch_size}")
    print("=" * 60)
    
    # 启动服务器
    uvicorn.run(
        "main_integration:app",
        host=CONFIG.host,
        port=CONFIG.port,
        reload=CONFIG.reload,
        log_level=CONFIG.log_level.lower()
    )

if __name__ == "__main__":
    main()