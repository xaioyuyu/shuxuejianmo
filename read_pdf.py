#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import PyPDF2
import sys


def extract_pdf_text(pdf_path):
    """提取PDF文件中的所有文本"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += f"\n===== 第 {page_num + 1} 页 =====\n"
                text += page.extract_text()
            return text
    except Exception as e:
        return f"读取PDF文件时出错: {str(e)}"


if __name__ == "__main__":
    pdf_path = "16abaee12d738d1dfafd5731c592d51b.pdf"
    content = extract_pdf_text(pdf_path)
    print(content)

    # 保存到文本文件
    with open("pdf_content.txt", "w", encoding="utf-8") as f:
        f.write(content)
    print("\n\n===== PDF内容已保存到 pdf_content.txt =====")
