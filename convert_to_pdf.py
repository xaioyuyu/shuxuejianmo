#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将Markdown论文转换为PDF
使用markdown和weasyprint库
"""

import markdown
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration
import os


def convert_markdown_to_pdf(md_file, pdf_file):
    """
    将Markdown文件转换为PDF

    参数:
        md_file: Markdown文件路径
        pdf_file: 输出PDF文件路径
    """
    print(f"正在读取Markdown文件: {md_file}")

    # 读取Markdown内容
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()

    print("正在转换Markdown到HTML...")

    # 转换Markdown到HTML
    html_content = markdown.markdown(
        md_content,
        extensions=[
            'extra',  # 支持表格、代码块等
            'codehilite',  # 代码高亮
            'toc',  # 目录
            'tables',  # 表格支持
            'fenced_code'  # 代码块
        ]
    )

    # 添加HTML头部和CSS样式
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>稳定币应用前景与监管策略研究</title>
        <style>
            @page {{
                size: A4;
                margin: 2.5cm;
                @bottom-right {{
                    content: "第 " counter(page) " 页";
                    font-size: 10pt;
                    color: #666;
                }}
            }}
            
            body {{
                font-family: "PingFang SC", "Microsoft YaHei", "SimHei", sans-serif;
                font-size: 11pt;
                line-height: 1.6;
                color: #333;
                margin: 0;
                padding: 0;
            }}
            
            h1 {{
                font-size: 24pt;
                font-weight: bold;
                text-align: center;
                margin: 30pt 0 20pt 0;
                page-break-before: always;
                page-break-after: avoid;
                color: #1a1a1a;
            }}
            
            h1:first-of-type {{
                page-break-before: auto;
            }}
            
            h2 {{
                font-size: 18pt;
                font-weight: bold;
                margin: 20pt 0 12pt 0;
                page-break-after: avoid;
                border-bottom: 2px solid #2E86C1;
                padding-bottom: 5pt;
                color: #2E86C1;
            }}
            
            h3 {{
                font-size: 14pt;
                font-weight: bold;
                margin: 15pt 0 10pt 0;
                page-break-after: avoid;
                color: #34495E;
            }}
            
            h4 {{
                font-size: 12pt;
                font-weight: bold;
                margin: 12pt 0 8pt 0;
                color: #566573;
            }}
            
            p {{
                margin: 8pt 0;
                text-align: justify;
                text-indent: 2em;
            }}
            
            strong {{
                font-weight: bold;
                color: #C0392B;
            }}
            
            em {{
                font-style: italic;
            }}
            
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 15pt 0;
                font-size: 10pt;
                page-break-inside: avoid;
            }}
            
            table thead {{
                background-color: #2E86C1;
                color: white;
            }}
            
            table th {{
                padding: 8pt;
                text-align: center;
                font-weight: bold;
                border: 1px solid #ccc;
            }}
            
            table td {{
                padding: 6pt;
                text-align: center;
                border: 1px solid #ccc;
            }}
            
            table tbody tr:nth-child(even) {{
                background-color: #f8f9fa;
            }}
            
            ul, ol {{
                margin: 10pt 0 10pt 25pt;
                padding-left: 15pt;
            }}
            
            li {{
                margin: 5pt 0;
            }}
            
            code {{
                background-color: #f5f5f5;
                padding: 2pt 5pt;
                border-radius: 3pt;
                font-family: "Monaco", "Consolas", monospace;
                font-size: 9pt;
                color: #C0392B;
            }}
            
            pre {{
                background-color: #f5f5f5;
                padding: 10pt;
                border-radius: 5pt;
                overflow-x: auto;
                margin: 10pt 0;
                border-left: 3pt solid #2E86C1;
                page-break-inside: avoid;
            }}
            
            pre code {{
                background-color: transparent;
                padding: 0;
                color: #333;
            }}
            
            blockquote {{
                margin: 10pt 0 10pt 20pt;
                padding-left: 15pt;
                border-left: 4pt solid #95a5a6;
                color: #555;
                font-style: italic;
            }}
            
            hr {{
                border: none;
                border-top: 1px solid #ccc;
                margin: 20pt 0;
            }}
            
            /* 数学公式样式 */
            .math {{
                font-family: "Latin Modern Math", "STIX Two Math", serif;
                font-style: italic;
            }}
            
            /* 页面分隔符 */
            .page-break {{
                page-break-after: always;
            }}
            
            /* 首页样式 */
            .title-page {{
                text-align: center;
                padding-top: 100pt;
            }}
            
            .title-page h1 {{
                font-size: 28pt;
                margin-bottom: 30pt;
            }}
            
            /* 摘要样式 */
            .abstract {{
                margin: 20pt 40pt;
                text-align: justify;
            }}
            
            .abstract strong:first-child {{
                font-size: 12pt;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    print("正在生成PDF...")

    # 字体配置
    font_config = FontConfiguration()

    # CSS样式
    css = CSS(string='''
        @page {
            size: A4;
            margin: 2.5cm;
        }
    ''', font_config=font_config)

    # 生成PDF
    HTML(string=full_html).write_pdf(
        pdf_file,
        stylesheets=[css],
        font_config=font_config
    )

    print(f"✓ PDF已成功生成: {pdf_file}")

    # 获取文件大小
    file_size = os.path.getsize(pdf_file) / 1024 / 1024  # MB
    print(f"  文件大小: {file_size:.2f} MB")


if __name__ == "__main__":
    # 输入和输出文件路径
    md_file = "最终论文/稳定币应用前景与监管策略研究_论文正文.md"
    pdf_file = "最终论文/稳定币应用前景与监管策略研究_论文正文.pdf"

    print("=" * 70)
    print("Markdown to PDF 转换器")
    print("=" * 70)

    # 检查输入文件是否存在
    if not os.path.exists(md_file):
        print(f"错误: 找不到文件 {md_file}")
        exit(1)

    # 转换
    convert_markdown_to_pdf(md_file, pdf_file)

    print("=" * 70)
    print("转换完成！")
    print("=" * 70)
