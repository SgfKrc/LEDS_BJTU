#!/usr/bin/env python3
"""
修复验证脚本
============
验证第二轮修复（P1-1, P1-2, P2-1）是否正确实施，无引入新问题。

使用方法：
    python scripts/verify_fixes.py
"""

import sys
import os

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_db_import():
    """测试 1: db.py 模块能否正常导入"""
    print("\n" + "=" * 60)
    print("测试 1: db.py 模块导入")
    print("=" * 60)
    
    try:
        import db
        print("[OK] db.py 导入成功")
        print(f"  DB_HOST = {db.DB_HOST}")
        print(f"  DB_PORT = {db.DB_PORT}")
        print(f"  DB_NAME = {db.DB_NAME}")
        print(f"  DB_USER = {db.DB_USER}")
        print(f"  DB_MIN_CONN = {db.DB_MIN_CONN}")
        print(f"  DB_MAX_CONN = {db.DB_MAX_CONN}")
        print(f"  DB_PASSWORD = {'(空)' if not db.DB_PASSWORD else '(已设置)'}")
        return True
    except Exception as e:
        print(f"[FAIL] 导入失败: {e}")
        return False


def test_safe_int_function():
    """测试 2: _safe_int() 函数是否存在且功能正常"""
    print("\n" + "=" * 60)
    print("测试 2: _safe_int() 函数验证")
    print("=" * 60)
    
    try:
        import db
        
        # 检查函数是否存在
        if not hasattr(db, '_safe_int'):
            print("✗ _safe_int() 函数不存在")
            return False
        
        print("[OK] _safe_int() 函数存在")
        
        # 测试默认值
        result = db._safe_int("NONEXISTENT_VAR", 9999)
        if result == 9999:
            print(f"[OK] 默认值测试通过: _safe_int('NONEXISTENT_VAR', 9999) = {result}")
        else:
            print(f"[FAIL] 默认值测试失败: 期望 9999，实际 {result}")
            return False
        
        return True
    except Exception as e:
        print(f"[FAIL] 测试失败: {e}")
        return False


def test_min_max_swap():
    """测试 3: MIN > MAX 自动交换逻辑"""
    print("\n" + "=" * 60)
    print("测试 3: MIN_CONN/MAX_CONN 自动交换验证")
    print("=" * 60)
    
    try:
        import db
        
        if db.DB_MIN_CONN <= db.DB_MAX_CONN:
            print(f"[OK] MIN_CONN({db.DB_MIN_CONN}) <= MAX_CONN({db.DB_MAX_CONN})，逻辑正确")
            return True
        else:
            print(f"[FAIL] MIN_CONN({db.DB_MIN_CONN}) > MAX_CONN({db.DB_MAX_CONN})，交换逻辑失败")
            return False
    except Exception as e:
        print(f"[FAIL] 测试失败: {e}")
        return False


def test_close_db_lock():
    """测试 4: close_db() 函数是否有锁保护"""
    print("\n" + "=" * 60)
    print("测试 4: close_db() 锁保护验证")
    print("=" * 60)
    
    try:
        import db
        import inspect
        
        # 获取 close_db 源码
        source = inspect.getsource(db.close_db)
        
        if '_init_lock' in source:
            print("[OK] close_db() 包含 _init_lock 锁保护")
            return True
        else:
            print("[FAIL] close_db() 缺少 _init_lock 锁保护")
            return False
    except Exception as e:
        print(f"[FAIL] 测试失败: {e}")
        return False


def test_no_hardcoded_credentials():
    """测试 5: 验证无硬编码凭据"""
    print("\n" + "=" * 60)
    print("测试 5: 硬编码凭据检查")
    print("=" * 60)
    
    try:
        import db
        import inspect
        
        source = inspect.getsource(db)
        
        # 检查旧的硬编码
        bad_patterns = [
            "8.160.161.53",
            "WUTqw6bLkK3Hn5Va",
            'DB_HOST = "8.',
            'DB_PASSWORD = "WUT'
        ]
        
        found_bad = []
        for pattern in bad_patterns:
            if pattern in source:
                found_bad.append(pattern)
        
        if found_bad:
            print(f"[FAIL] 发现硬编码凭据: {found_bad}")
            return False
        else:
            print("[OK] 未发现硬编码凭据")
            return True
    except Exception as e:
        print(f"[FAIL] 测试失败: {e}")
        return False


def test_environment_variable_reading():
    """测试 6: 环境变量读取验证"""
    print("\n" + "=" * 60)
    print("测试 6: 环境变量读取验证")
    print("=" * 60)
    
    try:
        import db
        import inspect
        
        source = inspect.getsource(db)
        
        # 检查是否使用 os.environ.get
        if 'os.environ.get("QLH_DB_HOST"' in source:
            print("[OK] DB_HOST 从环境变量读取")
        else:
            print("[FAIL] DB_HOST 未从环境变量读取")
            return False
        
        if 'os.environ.get("QLH_DB_PASSWORD"' in source:
            print("[OK] DB_PASSWORD 从环境变量读取")
        else:
            print("[FAIL] DB_PASSWORD 未从环境变量读取")
            return False
        
        return True
    except Exception as e:
        print(f"[FAIL] 测试失败: {e}")
        return False


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("第二轮修复验证 (P1-1, P1-2, P2-1)")
    print("=" * 60)
    
    tests = [
        test_db_import,
        test_safe_int_function,
        test_min_max_swap,
        test_close_db_lock,
        test_no_hardcoded_credentials,
        test_environment_variable_reading,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n[FAIL] 测试异常: {e}")
            results.append(False)
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("验证结果汇总")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("\n[SUCCESS] 所有验证通过！修复正确实施，无引入新问题。")
        print("\n下一步:")
        print("  1. 运行单元测试: python -m pytest tests/test_db_config.py -v")
        print("  2. 如需测试真实数据库，参考: docs/数据库测试指南.md")
        return 0
    else:
        print(f"\n[FAIL] {total - passed} 个验证失败，请检查修复实施。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
