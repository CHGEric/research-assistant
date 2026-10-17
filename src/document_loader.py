"""
文档加载模块 —— RAG 流水线的第一站（v2：章节感知切分）

【通俗理解】Agent 要"读论文"，但 PDF 是给人看的排版格式，
  对程序来说就是一堆带坐标的文字碎片。这个模块负责把 PDF "拍平"成纯文本，
  并按"章节"组织内容，再切成大小合适的"知识块"（chunk）。

v2 升级：章节感知（Section-Aware）
  v1 按页+固定窗口盲切，章节信息全丢；
  v2 利用 PDF 的"字号"线索识别章节标题（标题字号 > 正文），
  并辅以中文学术论文章节正则，把内容按 摘要/方法/实验结果/结论 分段，
  每个 chunk 的 metadata 携带 section 字段——
  好处：①检索可按章节过滤 ②"总结第 X 章"类问题可精准定位 ③引用溯源多一个维度。

数据流：PDF --> 带字号的行 --> 识别章节标题 --> 分章节聚合
        --> 章节内递归切分 --> 知识块（source/page/section/paper_title 元数据）
"""

import re
from collections import Counter
from pathlib import Path

import pymupdf  # PyMuPDF：解析 PDF 的库，对中文和学术排版兼容性好
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP

# 中文学术论文常见章节标题（整行精确匹配，宽松空格）
SECTION_PATTERN = re.compile(
    r"^(摘\s*要|Abstract|ABSTRACT|引\s*言|Introduction|相关工作|相关研究|"
    r"方\s*法|Method|Methods|实验设置|实验与分析|实验结果|结果与分析|"
    r"Experiments?|讨\s*论|Discussion|结\s*论|Conclusion[s]?|参考文献|References?)\s*$",
    re.IGNORECASE,
)
# 编号式标题："1 引言" "2.1 注意力机制" "三、方法"
NUMBERED_PATTERN = re.compile(r"^((\d+(\.\d+)*)[\s、\.]|（?[一二三四五六七八九十]+）?[、\s])\S{1,30}$")

MAX_TITLE_LEN = 40  # 超过此长度的行几乎不可能是章节标题


def _page_lines(page) -> list[tuple[str, float]]:
    """
    提取一页的文本行及其字号。

    【为什么需要字号】PDF 文本不带任何"标记"，判断"这行是不是标题"
      最可靠的视觉线索就是字号——标题总是比正文大。
    page.get_text("dict") 返回结构化数据：块 -> 行 -> span（含 size 文本属性）。
    """
    lines: list[tuple[str, float]] = []
    data = page.get_text("dict")
    for block in data.get("blocks", []):
        if block.get("type", 1) != 0:  # type 0 = 文本块；跳过图片块
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(span.get("text", "") for span in spans).strip()
            if not text:
                continue
            size = max(span.get("size", 11.0) for span in spans)
            lines.append((text, round(size, 1)))
    return lines


def _body_size(lines: list[tuple[str, float]]) -> float:
    """正文字号 = 全页行字号的众数（绝大多数行是正文，众数天然是正文号）。"""
    if not lines:
        return 11.0
    return Counter(size for _, size in lines).most_common(1)[0][0]


def _is_section_title(text: str, size: float, body: float) -> bool:
    """
    章节标题判定，三选一即成立：
      1. 整行精确命中学术论文章节词（摘要/方法/结论……）
      2. 编号式标题（"1 引言"、"2.1 xxx"）
      3. 字号显著大于正文（≥ 1.15 倍）——排版层面的"标题感"
    """
    if len(text) > MAX_TITLE_LEN:
        return False
    if SECTION_PATTERN.match(text):
        return True
    if NUMBERED_PATTERN.match(text) and size >= body:
        return True
    return size >= body * 1.15


def load_pdf(pdf_path: str | Path) -> list[Document]:
    """
    章节感知解析：返回按章节组织的 Document 列表（一个 Document = 一页内的一个章节）。

    metadata 含：source(文件名) / page(页码) / section(章节名) / paper_title(论文标题)

    跨页处理：翻页时缓冲区先落成"上一页的章节片段"，章节名延续到下一页——
      若下一页开头没有新标题，则视为同一章节的续文（页码仍按各自所在页记录）。
    """
    pdf_path = Path(pdf_path)
    docs: list[Document] = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n。", "\n", "。", "；", "，", " ", ""],
    )

    paper_title = ""
    current_section = "标题区"  # 论文大标题之前/本身的内容归入此节

    with pymupdf.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf, start=1):
            lines = _page_lines(page)
            if not lines:
                continue
            body = _body_size(lines)
            buffer: list[str] = []  # 当前章节的正文行缓冲
            buffer_page = page_num

            def _flush(buf: list[str], sec: str, pg: int) -> None:
                """把缓冲区内容切成块挂到 docs 上（闭包捕获 splitter 与外部容器）。"""
                text = "\n".join(buf).strip()
                if not text:
                    return
                section_doc = Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path.name,
                        "page": pg,
                        "section": sec,
                        "paper_title": paper_title,
                    },
                )
                docs.extend(splitter.split_documents([section_doc]))

            for text, size in lines:
                if _is_section_title(text, size, body):
                    _flush(buffer, current_section, buffer_page)
                    buffer = []
                    # 论文大标题（全页最大字号且尚未记录）单独识别，不占用章节名
                    if not paper_title and size >= body * 1.3:
                        paper_title = text
                        current_section = "标题区"
                    else:
                        current_section = text
                else:
                    buffer.append(text)
            _flush(buffer, current_section, buffer_page)  # 页末收尾

    return docs


def split_documents(documents: list[Document]) -> list[Document]:
    """
    对章节 Document 做递归切分（兼容 v1 接口）。

    load_pdf 内部已按章节切好并调用了本函数的等价逻辑；
    保留此函数供外部传入"整页/整篇"粒度的 Document 时使用。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n。", "\n", "。", "；", "，", " ", ""],
    )
    return splitter.split_documents(documents)


def load_pdf_dir(pdf_dir) -> list[Document]:
    """批量解析目录下所有 PDF，返回全部知识块（含章节元数据）。"""
    pdf_dir = Path(pdf_dir)
    all_chunks: list[Document] = []
    for pdf_file in sorted(pdf_dir.glob("*.pdf")):
        all_chunks.extend(load_pdf(pdf_file))  # v2：load_pdf 已是 chunk 粒度
    return all_chunks
