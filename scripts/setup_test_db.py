#!/usr/bin/env python3
"""
数据库测试环境设置脚本
=====================
用于快速设置 PostgreSQL 测试数据库。

使用方法：
    python scripts/setup_test_db.py

前置条件：
    - PostgreSQL 已安装并运行
    - 有创建数据库的用户权限（通常是 postgres 用户）
"""

import os
import sys
import subprocess


def check_postgres_installed():
    """检查 PostgreSQL 是否安装"""
    try:
        result = subprocess.run(
            ["psql", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print(f"✓ PostgreSQL 已安装: {result.stdout.strip()}")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    print("✗ PostgreSQL 未安装或未在 PATH 中")
    print("  请安装 PostgreSQL:")
    print("    Windows: choco install postgresql")
    print("    macOS: brew install postgresql")
    print("    Linux: sudo apt install postgresql")
    return False


def check_postgres_running():
    """检查 PostgreSQL 服务是否运行"""
    try:
        result = subprocess.run(
            ["psql", "-U", "postgres", "-c", "SELECT 1;"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✓ PostgreSQL 服务正在运行")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    print("✗ PostgreSQL 服务未运行")
    print("  请启动 PostgreSQL:")
    print("    Windows: net start postgresql-x64-14")
    print("    macOS: brew services start postgresql")
    print("    Linux: sudo systemctl start postgresql")
    return False


def create_test_database():
    """创建测试数据库和用户"""
    db_name = "qlh_edge_inference_test"
    db_user = "qlh_test"
    db_password = "test_password_123"
    
    # 创建数据库
    print(f"\n创建测试数据库: {db_name}")
    try:
        result = subprocess.run(
            ["psql", "-U", "postgres", "-c", f"CREATE DATABASE {db_name};"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print(f"✓ 数据库 {db_name} 创建成功")
        elif "already exists" in result.stderr:
            print(f"✓ 数据库 {db_name} 已存在")
        else:
            print(f"✗ 创建数据库失败: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("✗ 创建数据库超时")
        return False
    
    # 创建用户
    print(f"创建测试用户: {db_user}")
    try:
        result = subprocess.run(
            ["psql", "-U", "postgres", "-c", 
             f"CREATE USER {db_user} WITH PASSWORD '{db_password}';"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print(f"✓ 用户 {db_user} 创建成功")
        elif "already exists" in result.stderr:
            print(f"✓ 用户 {db_user} 已存在")
        else:
            print(f"✗ 创建用户失败: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("✗ 创建用户超时")
        return False
    
    # 授权
    print(f"授权用户 {db_user} 访问数据库 {db_name}")
    try:
        result = subprocess.run(
            ["psql", "-U", "postgres", "-c", 
             f"GRANT ALL PRIVILEGES ON DATABASE {db_name} TO {db_user};"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print(f"✓ 授权成功")
        else:
            print(f"✗ 授权失败: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("✗ 授权超时")
        return False
    
    return True


def generate_env_file():
    """生成 .env.test 文件"""
    env_content = """# 数据库测试配置
QLH_DB_HOST=localhost
QLH_DB_PORT=5432
QLH_DB_NAME=qlh_edge_inference_test
QLH_DB_USER=qlh_test
QLH_DB_PASSWORD=test_password_123
"""
    
    env_path = ".env.test"
    with open(env_path, "w", encoding="utf-8") as f:
        f.write(env_content)
    
    print(f"\n✓ 测试环境配置已写入 {env_path}")
    print(f"\n使用方法:")
    print(f"  Windows PowerShell:")
    print(f"    Get-Content .env.test | ForEach-Object {{ $name, $value = $_.Split('='); [Environment]::SetEnvironmentVariable($name, $value) }}")
    print(f"    python -m pytest tests/test_db_config.py -v")
    print(f"\n  Linux/macOS:")
    print(f"    export $(cat .env.test | xargs)")
    print(f"    python -m pytest tests/test_db_config.py -v")


def main():
    """主函数"""
    print("=" * 60)
    print("数据库测试环境设置")
    print("=" * 60)
    
    # 检查 PostgreSQL
    if not check_postgres_installed():
        sys.exit(1)
    
    if not check_postgres_running():
        sys.exit(1)
    
    # 创建测试数据库
    if not create_test_database():
        print("\n✗ 数据库设置失败，请手动设置")
        sys.exit(1)
    
    # 生成 .env.test 文件
    generate_env_file()
    
    print("\n" + "=" * 60)
    print("✓ 测试环境设置完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
