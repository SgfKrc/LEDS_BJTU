"""
HTML报告生成器

生成美观的HTML格式测试报告。
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict


def generate_html_report(report_data: Dict, output_path: str) -> str:
    """生成HTML格式的测试报告

    Args:
        report_data: 报告数据（JSON格式）
        output_path: 输出文件路径

    Returns:
        HTML文件路径
    """
    # 确定报告类型
    test_type = determine_report_type(report_data)

    # 生成HTML内容
    html_content = render_html_report(report_data, test_type)

    # 写入文件
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    return str(output_file)


def determine_report_type(report_data: Dict) -> str:
    """确定报告类型

    Args:
        report_data: 报告数据

    Returns:
        报告类型 ("single", "distributed", "degradation", "stress", "exception")
    """
    test_name = report_data.get("test_name", "")

    if "single" in test_name:
        return "single"
    elif "distributed" in test_name:
        return "distributed"
    elif "degradation" in test_name:
        return "degradation"
    elif "stress" in test_name:
        return "stress"
    elif "network_interruption" in test_name or "node_crash" in test_name:
        return "exception"
    else:
        return "unknown"


def render_html_report(report_data: Dict, test_type: str) -> str:
    """渲染HTML报告

    Args:
        report_data: 报告数据
        test_type: 报告类型

    Returns:
        HTML内容
    """
    # HTML模板
    html_template = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 32px;
            margin-bottom: 10px;
        }}

        .header .timestamp {{
            opacity: 0.9;
            font-size: 14px;
        }}

        .content {{
            padding: 40px;
        }}

        .section {{
            margin-bottom: 40px;
        }}

        .section h2 {{
            font-size: 24px;
            margin-bottom: 20px;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}

        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}

        .info-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}

        .info-card h3 {{
            font-size: 14px;
            color: #666;
            margin-bottom: 10px;
            text-transform: uppercase;
        }}

        .info-card .value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}

        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}

        .stat-card .label {{
            font-size: 12px;
            opacity: 0.9;
            margin-bottom: 5px;
        }}

        .stat-card .value {{
            font-size: 28px;
            font-weight: bold;
        }}

        .stat-card.success {{
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }}

        .stat-card.warning {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }}

        .stat-card.danger {{
            background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        }}

        .table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}

        .table th {{
            background: #f8f9fa;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid #dee2e6;
        }}

        .table td {{
            padding: 12px;
            border-bottom: 1px solid #dee2e6;
        }}

        .table tr:hover {{
            background: #f8f9fa;
        }}

        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }}

        .badge.success {{
            background: #d4edda;
            color: #155724;
        }}

        .badge.danger {{
            background: #f8d7da;
            color: #721c24;
        }}

        .badge.warning {{
            background: #fff3cd;
            color: #856404;
        }}

        .progress-bar {{
            width: 100%;
            height: 20px;
            background: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin-top: 10px;
        }}

        .progress-bar .fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s ease;
        }}

        .progress-bar .fill.success {{
            background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
        }}

        .progress-bar .fill.warning {{
            background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        }}

        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            font-size: 12px;
        }}

        .phase-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}

        .phase-card h3 {{
            font-size: 18px;
            margin-bottom: 15px;
            color: #333;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
            <div class="timestamp">生成时间: {timestamp}</div>
        </div>

        <div class="content">
            {content}
        </div>

        <div class="footer">
            QLH 分布式推理仿真测试框架 v2.0.0 | 生成于 {timestamp}
        </div>
    </div>
</body>
</html>"""

    # 根据报告类型生成内容
    if test_type == "single":
        content = render_single_report(report_data)
        title = "单机推理测试报告"
    elif test_type == "distributed":
        content = render_distributed_report(report_data)
        title = "分布式推理测试报告"
    elif test_type == "degradation":
        content = render_degradation_report(report_data)
        title = "降级测试报告"
    elif test_type == "stress":
        content = render_stress_report(report_data)
        title = "压力测试报告"
    elif test_type == "exception":
        content = render_exception_report(report_data)
        title = "异常测试报告"
    else:
        content = render_generic_report(report_data)
        title = "测试报告"

    # 填充模板
    timestamp = report_data.get("timestamp", datetime.now().isoformat())

    return html_template.format(
        title=title,
        timestamp=timestamp,
        content=content,
    )


def render_single_report(report_data: Dict) -> str:
    """渲染单机推理报告"""
    summary = report_data.get("summary", {})
    performance = report_data.get("performance", {})
    scenario = report_data.get("scenario", {})

    success_rate = summary.get("success_rate", 0)
    success_class = "success" if success_rate >= 0.95 else "warning" if success_rate >= 0.80 else "danger"

    html = f"""
    <div class="section">
        <h2>测试概览</h2>
        <div class="info-grid">
            <div class="info-card">
                <h3>测试场景</h3>
                <div class="value">{scenario.get("name", "N/A")}</div>
            </div>
            <div class="info-card">
                <h3>问题数量</h3>
                <div class="value">{scenario.get("question_count", 0)}</div>
            </div>
            <div class="info-card">
                <h3>测试模式</h3>
                <div class="value">单机推理</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>测试结果</h2>
        <div class="stats-grid">
            <div class="stat-card {success_class}">
                <div class="label">成功率</div>
                <div class="value">{success_rate:.1%}</div>
            </div>
            <div class="stat-card">
                <div class="label">通过 / 总数</div>
                <div class="value">{summary.get("passed", 0)} / {summary.get("total_requests", 0)}</div>
            </div>
            <div class="stat-card">
                <div class="label">平均延迟</div>
                <div class="value">{performance.get("avg_latency", 0):.2f}s</div>
            </div>
            <div class="stat-card">
                <div class="label">平均响应长度</div>
                <div class="value">{performance.get("avg_response_length", 0):.0f}</div>
            </div>
        </div>

        <div class="progress-bar">
            <div class="fill {success_class}" style="width: {success_rate * 100}%"></div>
        </div>
    </div>

    <div class="section">
        <h2>性能详情</h2>
        <table class="table">
            <thead>
                <tr>
                    <th>指标</th>
                    <th>数值</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>最小延迟</td>
                    <td>{performance.get("min_latency", 0):.2f}s</td>
                </tr>
                <tr>
                    <td>最大延迟</td>
                    <td>{performance.get("max_latency", 0):.2f}s</td>
                </tr>
                <tr>
                    <td>最小响应长度</td>
                    <td>{performance.get("min_response_length", 0)} 字符</td>
                </tr>
                <tr>
                    <td>最大响应长度</td>
                    <td>{performance.get("max_response_length", 0)} 字符</td>
                </tr>
            </tbody>
        </table>
    </div>
    """

    return html


def render_distributed_report(report_data: Dict) -> str:
    """渲染分布式推理报告"""
    summary = report_data.get("summary", {})
    performance = report_data.get("performance", {})
    config = report_data.get("config", {})
    scenario = report_data.get("scenario", {})

    success_rate = summary.get("success_rate", 0)
    success_class = "success" if success_rate >= 0.95 else "warning" if success_rate >= 0.80 else "danger"

    html = f"""
    <div class="section">
        <h2>测试概览</h2>
        <div class="info-grid">
            <div class="info-card">
                <h3>测试场景</h3>
                <div class="value">{scenario.get("name", "N/A")}</div>
            </div>
            <div class="info-card">
                <h3>从节点数</h3>
                <div class="value">{config.get("slave_count", 0)}</div>
            </div>
            <div class="info-card">
                <h3>问题数量</h3>
                <div class="value">{scenario.get("question_count", 0)}</div>
            </div>
            <div class="info-card">
                <h3>测试模式</h3>
                <div class="value">分布式推理</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>测试结果</h2>
        <div class="stats-grid">
            <div class="stat-card {success_class}">
                <div class="label">成功率</div>
                <div class="value">{success_rate:.1%}</div>
            </div>
            <div class="stat-card">
                <div class="label">通过 / 总数</div>
                <div class="value">{summary.get("passed", 0)} / {summary.get("total_requests", 0)}</div>
            </div>
            <div class="stat-card">
                <div class="label">平均延迟</div>
                <div class="value">{performance.get("avg_latency", 0):.2f}s</div>
            </div>
            <div class="stat-card">
                <div class="label">吞吐量</div>
                <div class="value">{performance.get("throughput", 0):.2f} req/s</div>
            </div>
        </div>

        <div class="progress-bar">
            <div class="fill {success_class}" style="width: {success_rate * 100}%"></div>
        </div>
    </div>
    """

    return html


def render_degradation_report(report_data: Dict) -> str:
    """渲染降级测试报告"""
    summary = report_data.get("summary", {})
    phases = report_data.get("phases", {})
    config = report_data.get("config", {})
    scenario = report_data.get("scenario", {})

    overall_success_rate = summary.get("overall_success_rate", 0)
    success_class = "success" if overall_success_rate >= 0.95 else "warning" if overall_success_rate >= 0.80 else "danger"

    html = f"""
    <div class="section">
        <h2>测试概览</h2>
        <div class="info-grid">
            <div class="info-card">
                <h3>测试场景</h3>
                <div class="value">{scenario.get("name", "N/A")}</div>
            </div>
            <div class="info-card">
                <h3>初始从节点数</h3>
                <div class="value">{config.get("initial_slave_count", 0)}</div>
            </div>
            <div class="info-card">
                <h3>测试模式</h3>
                <div class="value">降级测试</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>整体结果</h2>
        <div class="stats-grid">
            <div class="stat-card {success_class}">
                <div class="label">整体成功率</div>
                <div class="value">{overall_success_rate:.1%}</div>
            </div>
            <div class="stat-card">
                <div class="label">通过 / 总数</div>
                <div class="value">{summary.get("total_passed", 0)} / {summary.get("total_tests", 0)}</div>
            </div>
        </div>

        <div class="progress-bar">
            <div class="fill {success_class}" style="width: {overall_success_rate * 100}%"></div>
        </div>
    </div>

    <div class="section">
        <h2>各阶段详情</h2>
    """

    for phase_name, phase_data in phases.items():
        stats = phase_data.get("stats", {})
        phase_success_rate = stats.get("success_rate", 0)
        phase_success_class = "success" if phase_success_rate >= 0.95 else "warning" if phase_success_rate >= 0.80 else "danger"

        html += f"""
        <div class="phase-card">
            <h3>{phase_data.get("description", phase_name)}</h3>
            <div class="stats-grid">
                <div class="stat-card {phase_success_class}">
                    <div class="label">成功率</div>
                    <div class="value">{phase_success_rate:.1%}</div>
                </div>
                <div class="stat-card">
                    <div class="label">通过 / 总数</div>
                    <div class="value">{stats.get("passed", 0)} / {stats.get("count", 0)}</div>
                </div>
                <div class="stat-card">
                    <div class="label">平均延迟</div>
                    <div class="value">{stats.get("avg_latency", 0):.2f}s</div>
                </div>
            </div>
        </div>
        """

    html += """
    </div>
    """

    return html


def render_stress_report(report_data: Dict) -> str:
    """渲染压力测试报告"""
    stats = report_data.get("stats", {})
    config = report_data.get("config", {})
    scenario = report_data.get("scenario", {})

    success_rate = stats.get("success_rate", 0)
    throughput = stats.get("throughput", 0)
    success_class = "success" if success_rate >= 0.95 and throughput >= 0.5 else "warning" if success_rate >= 0.80 else "danger"

    html = f"""
    <div class="section">
        <h2>测试概览</h2>
        <div class="info-grid">
            <div class="info-card">
                <h3>测试场景</h3>
                <div class="value">{scenario.get("name", "N/A")}</div>
            </div>
            <div class="info-card">
                <h3>并发用户数</h3>
                <div class="value">{config.get("concurrent_users", 0)}</div>
            </div>
            <div class="info-card">
                <h3>每用户请求数</h3>
                <div class="value">{config.get("requests_per_user", 0)}</div>
            </div>
            <div class="info-card">
                <h3>从节点数</h3>
                <div class="value">{config.get("slave_count", 0)}</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>测试结果</h2>
        <div class="stats-grid">
            <div class="stat-card {success_class}">
                <div class="label">成功率</div>
                <div class="value">{success_rate:.1%}</div>
            </div>
            <div class="stat-card">
                <div class="label">吞吐量</div>
                <div class="value">{throughput:.2f} req/s</div>
            </div>
            <div class="stat-card">
                <div class="label">成功 / 总数</div>
                <div class="value">{stats.get("success_count", 0)} / {stats.get("total_requests", 0)}</div>
            </div>
            <div class="stat-card">
                <div class="label">总耗时</div>
                <div class="value">{stats.get("total_time", 0):.2f}s</div>
            </div>
        </div>

        <div class="progress-bar">
            <div class="fill {success_class}" style="width: {success_rate * 100}%"></div>
        </div>
    </div>

    <div class="section">
        <h2>延迟统计</h2>
        <table class="table">
            <thead>
                <tr>
                    <th>指标</th>
                    <th>数值</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>平均延迟</td>
                    <td>{stats.get("avg_latency", 0):.2f}s</td>
                </tr>
                <tr>
                    <td>中位数延迟</td>
                    <td>{stats.get("median_latency", 0):.2f}s</td>
                </tr>
                <tr>
                    <td>P95 延迟</td>
                    <td>{stats.get("p95_latency", 0):.2f}s</td>
                </tr>
                <tr>
                    <td>P99 延迟</td>
                    <td>{stats.get("p99_latency", 0):.2f}s</td>
                </tr>
                <tr>
                    <td>最小延迟</td>
                    <td>{stats.get("min_latency", 0):.2f}s</td>
                </tr>
                <tr>
                    <td>最大延迟</td>
                    <td>{stats.get("max_latency", 0):.2f}s</td>
                </tr>
                <tr>
                    <td>标准差</td>
                    <td>{stats.get("stddev_latency", 0):.2f}s</td>
                </tr>
            </tbody>
        </table>
    </div>
    """

    return html


def render_exception_report(report_data: Dict) -> str:
    """渲染异常测试报告"""
    summary = report_data.get("summary", {})
    phases = report_data.get("phases", {})
    config = report_data.get("config", {})
    scenario = report_data.get("scenario", {})

    overall_success_rate = summary.get("overall_success_rate", 0)
    success_class = "success" if overall_success_rate >= 0.95 else "warning" if overall_success_rate >= 0.80 else "danger"

    exception_type = config.get("exception_type", "unknown")
    exception_type_name = "网络中断" if exception_type == "network_interruption" else "节点崩溃"

    html = f"""
    <div class="section">
        <h2>测试概览</h2>
        <div class="info-grid">
            <div class="info-card">
                <h3>测试场景</h3>
                <div class="value">{scenario.get("name", "N/A")}</div>
            </div>
            <div class="info-card">
                <h3>异常类型</h3>
                <div class="value">{exception_type_name}</div>
            </div>
            <div class="info-card">
                <h3>从节点数</h3>
                <div class="value">{config.get("slave_count", 0)}</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>整体结果</h2>
        <div class="stats-grid">
            <div class="stat-card {success_class}">
                <div class="label">整体成功率</div>
                <div class="value">{overall_success_rate:.1%}</div>
            </div>
            <div class="stat-card">
                <div class="label">通过 / 总数</div>
                <div class="value">{summary.get("total_passed", 0)} / {summary.get("total_tests", 0)}</div>
            </div>
        </div>

        <div class="progress-bar">
            <div class="fill {success_class}" style="width: {overall_success_rate * 100}%"></div>
        </div>
    </div>

    <div class="section">
        <h2>各阶段详情</h2>
    """

    for phase_name, phase_data in phases.items():
        stats = phase_data.get("stats", {})
        phase_success_rate = stats.get("success_rate", 0)
        phase_success_class = "success" if phase_success_rate >= 0.95 else "warning" if phase_success_rate >= 0.80 else "danger"

        html += f"""
        <div class="phase-card">
            <h3>{phase_data.get("description", phase_name)}</h3>
            <div class="stats-grid">
                <div class="stat-card {phase_success_class}">
                    <div class="label">成功率</div>
                    <div class="value">{phase_success_rate:.1%}</div>
                </div>
                <div class="stat-card">
                    <div class="label">通过 / 总数</div>
                    <div class="value">{stats.get("passed", 0)} / {stats.get("count", 0)}</div>
                </div>
                <div class="stat-card">
                    <div class="label">平均延迟</div>
                    <div class="value">{stats.get("avg_latency", 0):.2f}s</div>
                </div>
            </div>
        </div>
        """

    html += """
    </div>
    """

    return html


def render_generic_report(report_data: Dict) -> str:
    """渲染通用报告"""
    summary = report_data.get("summary", {})

    html = f"""
    <div class="section">
        <h2>测试结果</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="label">总测试数</div>
                <div class="value">{summary.get("total_tests", summary.get("total_requests", 0))}</div>
            </div>
            <div class="stat-card">
                <div class="label">通过数</div>
                <div class="value">{summary.get("total_passed", summary.get("passed", 0))}</div>
            </div>
        </div>
    </div>
    """

    return html
