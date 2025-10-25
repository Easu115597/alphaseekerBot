#!/usr/bin/env python3
"""
修复验证脚本
测试所有模块导入是否正常
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_imports():
    """测试所有模块导入"""
    print("🧪 开始测试模块导入...")
    
    try:
        print("1. 测试集成API...")
        from integrated_api.main import app as api_app
        print("   ✅ 集成API导入成功")
        
        print("2. 测试ML引擎...")
        from ml_engine import AlphaSeekerMLEngine
        print("   ✅ ML引擎导入成功")
        
        print("3. 测试管道...")
        from pipeline import MultiStrategyPipeline
        print("   ✅ 管道导入成功")
        
        print("4. 测试扫描器...")
        from scanner import MarketScanner
        print("   ✅ 扫描器导入成功")
        
        print("5. 测试验证器...")
        from validation import SignalValidationCoordinator
        print("   ✅ 验证器导入成功")
        
        print("\n🎉 所有模块导入测试通过!")
        return True
        
    except Exception as e:
        print(f"\n❌ 导入测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_functionality():
    """测试基本功能"""
    print("\n🔧 测试基本功能...")
    
    try:
        # 测试配置文件
        config_file = Path("config/main_config.yaml")
        if config_file.exists():
            print("   ✅ 配置文件存在")
        else:
            print("   ⚠️  配置文件不存在")
        
        # 测试依赖
        import yaml
        import fastapi
        print("   ✅ 核心依赖正常")
        
        print("\n🎯 基本功能测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ 基本功能测试失败: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🔧 AlphaSeeker 修复验证工具")
    print("=" * 60)
    
    # 测试导入
    import_success = test_imports()
    
    # 测试基本功能
    function_success = test_basic_functionality()
    
    print("\n" + "=" * 60)
    print("📊 验证结果汇总")
    print("=" * 60)
    
    if import_success and function_success:
        print("🎉 所有测试通过! AlphaSeeker系统可以正常启动")
        print("\n🚀 现在可以运行:")
        print("   python3 main_integration.py")
        print("   或者")
        print("   python3 demo_complete.py")
    else:
        print("❌ 部分测试失败，请检查错误信息")
        print("\n💡 常见解决方案:")
        print("   1. 确保在code目录下运行")
        print("   2. 检查Python版本: python3 --version")
        print("   3. 安装依赖: pip install -r requirements.txt")
        print("   4. 检查文件权限")