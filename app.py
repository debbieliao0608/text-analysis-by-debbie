# -*- coding: utf-8 -*-
"""基于 NLTK 的英文综合文本分析与网页展示系统

启动方式（在考核文件夹里打开命令行）：
    python -m streamlit run app.py
"""

import io
import math
import re
import html as html_module
from collections import Counter, defaultdict
from pathlib import Path

import nltk
import pandas as pd
import streamlit as st
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

# ---------------------------------------------------------------- NLTK 数据
# 本机已装过这些数据，但云端（Streamlit Community Cloud）是全新环境，
# 启动时逐一检查并自动下载，已存在的会秒过。
for _res in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4",
             "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
             "vader_lexicon"]:
    try:
        nltk.download(_res, quiet=True)
    except Exception:
        pass

# ---------------------------------------------------------------- 页面设置
st.set_page_config(page_title="英文文本分析系统", page_icon="📖", layout="wide")

BASE_DIR = Path(__file__).parent
DEFAULT_FILE = BASE_DIR / "grimm030_little_louse_and_flea.txt"

# Penn Treebank 词性标记 → 中文类别
POS_GROUPS = ["名词 NN", "动词 VB", "形容词 JJ", "副词 RB", "其他"]


def pos_group(tag: str) -> str:
    """把 Penn 词性标记归到中文类别里"""
    for prefix in ("NN", "VB", "JJ", "RB"):
        if tag.startswith(prefix):
            return {"NN": "名词 NN", "VB": "动词 VB", "JJ": "形容词 JJ", "RB": "副词 RB"}[prefix]
    return "其他"


def wordnet_pos(tag: str) -> str:
    """把 Penn 标记翻译成 WordNet 认识的词性（n/v/a/r）"""
    if tag.startswith("NN"):
        return wordnet.NOUN
    if tag.startswith("VB"):
        return wordnet.VERB
    if tag.startswith("JJ"):
        return wordnet.ADJ
    if tag.startswith("RB"):
        return wordnet.ADV
    return ""          # 其他词性不还原，原样保留


def build_stop_set(custom_stop: str, use_nltk_stop: bool) -> set:
    stop_set = set(stopwords.words("english")) if use_nltk_stop else set()
    if custom_stop.strip():
        stop_set |= {w.strip().lower() for w in re.split(r"[,\s，、]+", custom_stop) if w.strip()}
    return stop_set


def clean_text(text: str) -> str:
    """文本清洗：统一引号/破折号 → 去标点数字 → 压缩空格 → 保留段落分隔"""
    text = (text.replace("\u2019", "'").replace("\u2018", "'")
                .replace("\u201c", '"').replace("\u201d", '"')
                .replace("\u2014", " ").replace("\u2013", " "))
    text = re.sub(r"[^A-Za-z'\n ]+", " ", text)      # 只留字母、撇号、换行、空格
    text = re.sub(r"[ \t]+", " ", text)              # 多个空格压成一个
    text = re.sub(r"\n\s*\n+", "\n\n", text)         # 多个空行压成一个（保留段落边界）
    return text.strip()


# ---------------------------------------------------------------- 数据函数
@st.cache_data(show_spinner=False)
def read_text(raw_bytes: bytes) -> str:
    """依次尝试常见编码，保证别人上传的 txt 不乱码、不报错"""
    for enc in ("utf-8", "utf-8-sig", "utf-16", "gb18030"):
        try:
            return raw_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw_bytes.decode("utf-8", errors="replace")   # 兜底：不认识的字节替换掉


@st.cache_data(show_spinner="正在分词、标注词性…")
def analyze(text: str, custom_stop: str, use_nltk_stop: bool, pos_filter: tuple, use_lemma: bool):
    """分词 → 词性标注 → 停用词过滤 → 词性过滤 → （可选）词形还原 → 词频统计"""
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    stop_set = build_stop_set(custom_stop, use_nltk_stop)

    lemmatizer = WordNetLemmatizer()
    kept = []                 # 每个元素：{word, tag, group, lemma}
    for token, tag in tagged:
        word = token.lower()
        if not re.fullmatch(r"[a-z][a-z']*", word):      # 去标点、数字
            continue
        if word in stop_set:                              # 去停用词
            continue
        group = pos_group(tag)
        if pos_filter and group not in pos_filter:        # 词性筛选
            continue
        wn_pos = wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(word, wn_pos) if wn_pos else word
        kept.append({"word": word, "tag": tag, "group": group, "lemma": lemma})

    key = "lemma" if use_lemma else "word"
    counter = Counter(item[key] for item in kept)

    forms = defaultdict(Counter)                          # 词元 → 表面形式
    for item in kept:
        forms[item["lemma"]][item["word"]] += 1

    return {
        "tokens": tokens,
        "tagged": tagged,
        "kept": kept,
        "counter": counter,
        "forms": forms,
        "stop_size": len(stop_set),
    }


@st.cache_data(show_spinner="正在清洗文本、统计候选关键词…")
def extract_keywords(text: str, custom_stop: str, use_nltk_stop: bool, cand_pos: tuple):
    """文本清洗 → 段落切分 → 分词/词性/停用词过滤 → 词频与 TF-IDF 双打分

    TF-IDF 的 N 取「段落数」而不是 1，避免单篇文档下 IDF 全部相同。
    """
    cleaned = clean_text(text)
    paragraphs = [p for p in re.split(r"\n\s*\n", cleaned) if p.strip()]

    stop_set = build_stop_set(custom_stop, use_nltk_stop)
    lemmatizer = WordNetLemmatizer()

    para_terms = []                  # 每段的词元列表
    surface = defaultdict(Counter)   # 词元 → 原形及次数
    tag_of = {}                      # 词元 → 首次出现的词性标记

    for para in paragraphs:
        tokens = word_tokenize(para)
        terms = []
        for token, tag in pos_tag(tokens):
            word = token.lower()
            if not re.fullmatch(r"[a-z][a-z']*", word):
                continue
            if word in stop_set:
                continue
            group = pos_group(tag)
            if cand_pos and group not in cand_pos:      # 候选词性筛选
                continue
            wn_pos = wordnet_pos(tag)
            lemma = lemmatizer.lemmatize(word, wn_pos) if wn_pos else word
            terms.append(lemma)
            surface[lemma][word] += 1
            tag_of.setdefault(lemma, tag)
        para_terms.append(terms)

    n_docs = len(para_terms)
    total = Counter(t for terms in para_terms for t in terms)   # 总词频

    doc_freq = Counter()             # 出现在多少个段落里
    for terms in para_terms:
        for term in set(terms):
            doc_freq[term] += 1

    # ---- TF-IDF：score = Σ_段落 tf(t,p) × idf(t) ----
    scores = defaultdict(float)
    for terms in para_terms:
        if not terms:
            continue
        counts = Counter(terms)
        length = len(terms)
        for term, n in counts.items():
            tf = n / length
            idf = math.log((1 + n_docs) / (1 + doc_freq[term])) + 1
            scores[term] += tf * idf

    return {
        "cleaned": cleaned,
        "paragraphs": paragraphs,
        "n_docs": n_docs,
        "total": total,
        "doc_freq": doc_freq,
        "scores": scores,
        "surface": surface,
        "tag_of": tag_of,
    }


def keyword_table(data, method: str) -> pd.DataFrame:
    """把候选词整理成表，并给出两套排名"""
    rows = []
    total = data["total"]
    scores = data["scores"]
    doc_freq = data["doc_freq"]
    surface = data["surface"]

    tf_rank = {w: i for i, (w, _) in enumerate(total.most_common(), start=1)}
    idf_rank = {w: i for i, (w, _) in enumerate(sorted(scores.items(), key=lambda x: -x[1]), start=1)}

    for word, count in total.most_common():
        rows.append(
            {
                "关键词": word,
                "词性": pos_group(data["tag_of"].get(word, "")),
                "总频次": count,
                "出现段落数": doc_freq[word],
                "原形形式": "、".join(sorted(surface[word], key=lambda w: -surface[word][w])),
                "TF-IDF得分": round(scores[word], 3),
                "词频排名": tf_rank[word],
                "TF-IDF排名": idf_rank[word],
            }
        )
    df = pd.DataFrame(rows)
    if method.startswith("TF-IDF"):
        df = df.sort_values(["TF-IDF排名", "总频次"], ascending=[True, False])
    else:
        df = df.sort_values(["词频排名", "总频次"], ascending=[True, False])
    df = df.reset_index(drop=True)
    df.insert(0, "排名", range(1, len(df) + 1))
    return df


# ---------------------------------------------------------------- 侧边栏
with st.sidebar:
    st.markdown("### 功能模块")
    module = st.selectbox("选择模块", ["词频分析", "提取关键词", "情感分析"], label_visibility="collapsed")

    st.divider()
    st.markdown("### 语料")
    uploaded = st.file_uploader("上传 txt（不传则用默认语料）", type=["txt"])
    st.caption(f"默认：{DEFAULT_FILE.name}")

    st.divider()
    st.markdown("### 停用词")
    use_nltk_stop = st.checkbox("使用 NLTK 英文停用词表", value=True)
    custom_stop = st.text_area(
        "自定义停用词",
        placeholder="little, said, began",
        height=80,
        help="支持逗号、空格、中文顿号分隔。会与 NLTK 停用词表合并。",
    )

    if module == "词频分析":
        st.divider()
        st.markdown("### 词性筛选")
        pos_choice = st.multiselect(
            "只统计这些词性（不选=全部）",
            POS_GROUPS,
            default=[],
            help="名词/动词/形容词/副词为实词，其他包含代词、介词、连词等。",
        )

        st.divider()
        st.markdown("### 词形还原")
        use_lemma = st.toggle("启用词形还原（Lemmatization）", value=False,
                              help="开启后 burned/burning 都合并为 burn，leaves 合并为 leaf。")
    else:
        pos_choice = []
        use_lemma = False

# ---------------------------------------------------------------- 读取文本
if uploaded is not None:
    text = read_text(uploaded.getvalue())
    source_name = uploaded.name
else:
    if not DEFAULT_FILE.exists():
        st.error(f"找不到默认语料：{DEFAULT_FILE}")
        st.stop()
    text = read_text(DEFAULT_FILE.read_bytes())
    source_name = DEFAULT_FILE.name

data = analyze(text, custom_stop, use_nltk_stop, tuple(pos_choice), use_lemma)
tokens = data["tokens"]
kept = data["kept"]
counter = data["counter"]
total_words = len(kept)
vocab = len(counter)

# ---------------------------------------------------------------- 标题区
st.title("英文综合文本分析与展示系统")
st.caption(f"当前模块：**{module}**　|　语料：**{source_name}**　|　基于 NLTK")

# ================================================================ 模块一：词频分析
if module == "词频分析":
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("字符数", f"{len(text):,}")
    c2.metric("分词数", f"{len(tokens):,}")
    c3.metric("有效词数", f"{total_words:,}", help="过滤停用词、词性筛选之后的词数")
    c4.metric("不同词型", f"{vocab:,}", help="按词元统计" if use_lemma else "按原词统计")
    c5.metric("TTR", f"{vocab / total_words:.3f}" if total_words else "—",
              help="Type-Token Ratio，词型数 ÷ 有效词数，衡量用词丰富度")

    st.divider()
    st.header("词频分析")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["① 原文", "② 分词与词性", "③ 停用词过滤", "④ Top N 词频", "⑤ 词形还原"]
    )

    # ---------------- ① 原文 ----------------
    with tab1:
        st.subheader("读入的英文原文")
        st.text_area("原文内容", text, height=420, label_visibility="collapsed")
        st.caption("读入方式：open(..., encoding='utf-8')，保证引号和破折号不乱码。")

    # ---------------- ② 分词与词性 ----------------
    with tab2:
        st.subheader("分词与词性标注")
        st.write(
            f"共切分出 **{len(tokens)}** 个 token。分词用 `nltk.word_tokenize`，"
            "词性标注用 `nltk.pos_tag`（Penn Treebank 标记集，NN=名词、VB=动词、JJ=形容词、RB=副词）。"
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "序号": range(1, len(tokens) + 1),
                    "token": tokens,
                    "词性": [tag for _, tag in data["tagged"]],
                }
            ),
            width="stretch",
            height=420,
        )

        st.markdown("**词性分布**")
        dist = pd.Series([pos_group(tag) for _, tag in data["tagged"]]).value_counts()
        st.dataframe(
            pd.DataFrame({"词性": dist.index, "数量": dist.values}),
            width="stretch",
            hide_index=True,
        )

    # ---------------- ③ 停用词过滤 ----------------
    with tab3:
        st.subheader("停用词过滤")
        st.write(
            f"当前停用词表共 **{data['stop_size']}** 个词"
            f"（NLTK 表{'已' if use_nltk_stop else '未'}启用"
            f"{'，含自定义 ' + str(len(custom_stop.split())) + ' 项' if custom_stop.strip() else ''}）。"
            f"过滤后从 {len(tokens)} 个 token 变成 **{total_words}** 个有效词。"
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**过滤前（前 40 个 token）**")
            st.write(" ".join(tokens[:40]))
        with col_b:
            st.markdown("**过滤后（前 40 个词）**")
            st.write(" ".join(item["word"] for item in kept[:40]) or "（空）")

        st.caption(
            f"共去掉 {len(tokens) - total_words} 个 token（含标点、停用词"
            + ("、被词性筛选排除的词" if pos_choice else "") + "）。"
        )

        with st.expander("看看这些词被去掉了（前 30 个）"):
            stop_set = build_stop_set(custom_stop, use_nltk_stop)
            removed = [tok for tok in tokens if tok.lower() in stop_set]
            st.write("、".join(removed[:30]) or "（无）")
            st.caption(f"被停用词表命中的 token 共 {len(removed)} 个。")

    # ---------------- ④ Top N 词频 ----------------
    with tab4:
        st.subheader("Top N 高频词")
        top_n = st.slider("显示前 N 个词", min_value=5, max_value=50, value=10, step=5)

        top_words = counter.most_common(top_n)
        df = pd.DataFrame(top_words, columns=["单词", "频次"])
        df.insert(0, "排名", range(1, len(df) + 1))
        df["占比"] = (df["频次"] / max(total_words, 1) * 100).round(1).astype(str) + "%"

        left, right = st.columns([1, 1.2])
        with left:
            st.dataframe(df, width="stretch", hide_index=True, height=min(420, 40 + 35 * top_n))
        with right:
            st.bar_chart(df.set_index("单词")["频次"], height=max(340, 20 * top_n))

        if len(top_words) >= 2:
            w1, c1_ = top_words[0]
            w2, c2_ = top_words[1]
            st.info(
                f"**`{w1}` 出现 {c1_} 次，占有效词的 {c1_ / total_words * 100:.1f}%**，"
                f"是第二名 `{w2}`（{c2_} 次）的 {c1_ / c2_:.1f} 倍。"
            )
        st.caption(
            f"统计口径：{'词元（已合并词形变化）' if use_lemma else '原词（未合并词形）'}"
            f"　|　词性：{'、'.join(pos_choice) if pos_choice else '全部实词与虚词'}"
        )

        # ---------- 导出词频表 ----------
        st.divider()
        st.markdown("**导出词频表**")

        export_all = st.checkbox(
            "导出全部词频（不勾选则只导出当前 Top N）",
            value=False,
            help="全部词频会把所有有效词按频次从高到低导出。",
        )

        export_words = counter.most_common() if export_all else top_words
        export_df = pd.DataFrame(export_words, columns=["单词", "频次"])
        export_df.insert(0, "排名", range(1, len(export_df) + 1))
        export_df["占比(%)"] = (export_df["频次"] / max(total_words, 1) * 100).round(2)

        scope = "全部词频" if export_all else f"Top{top_n}"
        base_name = f"词频表_{Path(source_name).stem}_{scope}"

        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            export_df.to_excel(writer, index=False, sheet_name="词频表")
        excel_bytes = excel_buffer.getvalue()

        txt_lines = ["排名\t单词\t频次\t占比(%)"]
        for row in export_df.itertuples(index=False):
            txt_lines.append(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]:.2f}")
        txt_bytes = ("\ufeff" + "\n".join(txt_lines)).encode("utf-8")

        col_x, col_t, col_hint = st.columns([1, 1, 2])
        with col_x:
            st.download_button(
                "导出 Excel（.xlsx）",
                data=excel_bytes,
                file_name=f"{base_name}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
            )
        with col_t:
            st.download_button(
                "导出 TXT（.txt）",
                data=txt_bytes,
                file_name=f"{base_name}.txt",
                mime="text/plain",
                width="stretch",
            )
        with col_hint:
            st.caption(
                f"将导出 {len(export_df)} 行。文件会存到浏览器默认下载文件夹"
                "（一般是「下载」），文件名已带上语料名和 Top 数。"
            )

        with st.expander("预览 TXT 内容（前 10 行）"):
            st.code("\n".join(txt_lines[:11]), language="text")

        if st.button("保存到本机考核文件夹", help="不经过浏览器下载，直接把文件写进 app.py 所在文件夹"):
            try:
                xlsx_path = BASE_DIR / f"{base_name}.xlsx"
                txt_path = BASE_DIR / f"{base_name}.txt"
                xlsx_path.write_bytes(excel_bytes)
                txt_path.write_bytes(txt_bytes)
                st.success(f"已保存：\n\n- {xlsx_path}\n- {txt_path}")
            except OSError:
                st.warning("当前在云端运行，服务器磁盘不可写。请使用上面的「导出」按钮，文件会下载到你的电脑。")

    # ---------------- ⑤ 词形还原 ----------------
    with tab5:
        st.subheader("词形还原（Lemmatization）")
        st.markdown(
            """
**原理**：`WordNetLemmatizer` 会按词性把词还原成词典里的原形——
`burned / burning → burn`，`leaves → leaf`，`crying → cry`。

**一个坑**：默认调用 `lemmatize("burned")`（不传词性）只会得到 `burned`，
因为 WordNet 默认把它当名词处理。所以必须先做词性标注，再把词性喂给 `lemmatize(word, pos)`。
"""
        )

        len_before = len({item["word"] for item in kept})
        len_after = len(counter) if use_lemma else len({item["lemma"] for item in kept})

        m1, m2, m3 = st.columns(3)
        m1.metric("还原前词型数", len_before)
        m2.metric("还原后词型数", len_after)
        m3.metric("合并掉", len_before - len_after)

        st.markdown("**发生了合并的词（同一词元下有多种写法）**")
        merged = {lemma: dict(c) for lemma, c in data["forms"].items() if len(c) > 1}
        if merged:
            rows = []
            for lemma, surfaces in sorted(merged.items(), key=lambda x: -sum(x[1].values())):
                rows.append(
                    {
                        "词元": lemma,
                        "原形（出现次数）": "、".join(
                            f"{w}({n})" for w, n in sorted(surfaces.items(), key=lambda x: -x[1])
                        ),
                        "合计": sum(surfaces.values()),
                    }
                )
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        else:
            st.write("（当前筛选条件下没有出现词形变化）")

        st.markdown("**全部词元一览**")
        st.dataframe(
            pd.DataFrame(
                sorted(
                    (
                        {"词元": lemma, "表面形式": "、".join(sorted(c)), "频次": sum(c.values())}
                        for lemma, c in data["forms"].items()
                    ),
                    key=lambda r: -r["频次"],
                )
            ),
            width="stretch",
            hide_index=True,
            height=300,
        )

# ================================================================ 模块二：提取关键词
elif module == "提取关键词":
    top_slot = st.empty()

    st.subheader("参数设置")
    p1, p2, p3 = st.columns([1.2, 1.4, 1])
    with p1:
        method = st.radio("打分方法", ["词频法（TF）", "TF-IDF（段落作为文档）"], index=1)
    with p2:
        cand_pos = st.multiselect(
            "候选词性",
            POS_GROUPS,
            default=["名词 NN", "形容词 JJ"],
            help="关键词一般是名词；加入形容词可得到 little、furiously 这类修饰性词。",
        )
    with p3:
        kw_num = st.slider("关键词数量", min_value=5, max_value=30, value=10, step=5)

    if not cand_pos:
        st.warning("请至少选择一个候选词性，否则没有候选词。")
        st.stop()

    kw = extract_keywords(text, custom_stop, use_nltk_stop, tuple(cand_pos))
    table = keyword_table(kw, method)
    result = table.head(kw_num).copy()
    n_docs = kw["n_docs"]

    with top_slot.container():
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("字符数", f"{len(text):,}")
        c2.metric("清洗后字符数", f"{len(kw['cleaned']):,}")
        c3.metric("段落数（=文档数 N）", n_docs)
        c4.metric("候选词型", f"{len(kw['total']):,}")
        c5.metric("候选词总数", f"{sum(kw['total'].values()):,}")

        st.divider()
        st.header("提取关键词")
        st.markdown(
            f"""
**流程**：文本清洗 → 段落切分（共 **{n_docs}** 段）→ 分词 → 词性标注 → 停用词过滤 →
候选词筛选（词性 + 词频）→ 打分排序。
"""
        )

    with st.expander("为什么单篇全文不能直接用 TF-IDF？（必读）", expanded=False):
        st.markdown(
            """
TF-IDF 由两部分相乘：

- **TF（词频）**：这个词在这篇文档里出现得多不多；
- **IDF（逆文档频率）**：`ln(文档总数 N ÷ 包含该词的文档数 df)`，衡量这个词有多“稀有”。

关键在 IDF：它需要**多篇文档**才能比出差异。如果只把整篇文章当成 1 篇文档，那么
`N = 1`，而每个词都出现在这唯一一篇里，`df = 1`，于是

> 所有词的 IDF = ln(1/1) = 0（或加平滑后全部等于 1）

**所有词乘上同一个数，排序就完全等于按词频排序，TF-IDF 白做了。**

正确做法：把长文切成**多个段落**，把每段当作一篇“文档”。这样
“只集中在少数段落反复出现”的词会拿到高 IDF，而“每段都出现”的功能性词会被压低，
这才是 TF-IDF 想表达的“区分度”。

本系统正是这样做的：N = 段落数，`IDF = ln((1+N)/(1+df)) + 1`。
"""
        )

    # ---------------- 清洗与候选词 ----------------
    tab_a, tab_b, tab_c, tab_d, tab_e = st.tabs(
        ["① 文本清洗", "② 候选词（词频+词性）", "③ 关键词结果", "④ 两种方法对比", "⑤ 规则与局限"]
    )

    with tab_a:
        st.subheader("文本清洗")
        removed_chars = len(text) - len(kw["cleaned"])
        m1, m2, m3 = st.columns(3)
        m1.metric("清洗前字符数", f"{len(text):,}")
        m2.metric("清洗后字符数", f"{len(kw['cleaned']):,}")
        m3.metric("去掉的字符", f"{removed_chars:,}")

        st.markdown(
            "清洗动作：① 统一弯引号/破折号 → ② 删除标点、数字等非字母字符 → "
            "③ 多个空格压成一个 → ④ 保留段落分隔（`\\n\\n`，后面切段落要用）。"
        )
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**清洗前（前 300 字符）**")
            st.code(text[:300], language="text")
        with col2:
            st.markdown("**清洗后（前 300 字符）**")
            st.code(kw["cleaned"][:300], language="text")

        with st.expander(f"按段落切分结果（共 {n_docs} 段，显示前 5 段）"):
            for i, para in enumerate(kw["paragraphs"][:5], start=1):
                st.markdown(f"**第 {i} 段**（{len(para.split())} 词）")
                st.write(para[:220] + ("…" if len(para) > 220 else ""))

    with tab_b:
        st.subheader("候选关键词生成：词频 + 词性")
        st.markdown(
            f"""
候选词必须同时满足三个条件：

1. **不是停用词**（the / a / is 这类词没有主题信息）；
2. **词性在候选范围内**（当前：{'、'.join(cand_pos)}）；
3. **在本段文本中出现过**——之后按词频或 TF-IDF 打分。

当前共 **{sum(kw['total'].values())}** 个候选词 token，去重后 **{len(kw['total'])}** 个词型。
"""
        )
        cand_df = pd.DataFrame(
            [
                {
                    "候选词": w,
                    "词性": pos_group(kw["tag_of"].get(w, "")),
                    "总频次": n,
                    "出现段落数": kw["doc_freq"][w],
                }
                for w, n in kw["total"].most_common(50)
            ]
        )
        cand_df.insert(0, "排名", range(1, len(cand_df) + 1))
        st.dataframe(cand_df, width="stretch", hide_index=True, height=420)
        st.caption("表里显示前 50 个候选词，按总频次排序。")

    with tab_c:
        st.subheader(f"关键词结果（{method}，前 {kw_num} 个）")

        show_cols = ["排名", "关键词", "词性", "总频次", "出现段落数", "原形形式"]
        if method.startswith("TF-IDF"):
            show_cols.insert(5, "TF-IDF得分")

        left, right = st.columns([1.35, 1])
        with left:
            st.dataframe(result[show_cols], width="stretch", hide_index=True,
                         height=min(460, 40 + 35 * kw_num))
        with right:
            chart_series = (
                result.set_index("关键词")["TF-IDF得分"]
                if method.startswith("TF-IDF")
                else result.set_index("关键词")["总频次"]
            )
            st.bar_chart(chart_series, height=max(360, 22 * kw_num))

        if len(result) >= 2:
            w1, n1 = result.iloc[0]["关键词"], result.iloc[0]["总频次"]
            w2, n2 = result.iloc[1]["关键词"], result.iloc[1]["总频次"]
            st.info(
                f"排第一的 **`{w1}`**（{n1} 次，出现在 {int(result.iloc[0]['出现段落数'])}/{n_docs} 段）；"
                f"第二是 `{w2}`（{n2} 次）。候选词性：{'、'.join(cand_pos)}；"
                f"打分方式：{method}。"
            )
        st.caption(
            f"TF-IDF 口径：N = {n_docs} 段，IDF = ln((1+N)/(1+df))+1，得分按段落求和。"
        )

    with tab_d:
        st.subheader("词频法 vs TF-IDF：排名对比")
        st.markdown(
            "同一批候选词，两种打分方式给出的顺序不同。**差异就是 IDF 在起作用**："
            "每段都出现的词（如 little）TF-IDF 会被压低，只在少数段落扎堆出现的词会被抬高。"
        )

        cmp_df = table.head(kw_num)[
            ["关键词", "总频次", "出现段落数", "词频排名", "TF-IDF得分", "TF-IDF排名"]
        ].copy()
        cmp_df["排名变化"] = cmp_df["词频排名"] - cmp_df["TF-IDF排名"]
        cmp_df["排名变化"] = cmp_df["排名变化"].apply(lambda d: f"↑{d}" if d > 0 else (f"↓{-d}" if d < 0 else "—"))
        st.dataframe(cmp_df, width="stretch", hide_index=True, height=min(460, 40 + 35 * kw_num))

        tf_top = {w for w, _ in kw["total"].most_common(kw_num)}
        idf_top = set(table.head(kw_num)["关键词"]) if method.startswith("TF-IDF") else set(
            table.nsmallest(kw_num, "TF-IDF排名")["关键词"]
        )
        only_tf = tf_top - idf_top
        only_idf = idf_top - tf_top

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**只有词频法进前 N 的词**")
            st.write("、".join(sorted(only_tf)) or "（无）")
        with col2:
            st.markdown("**只有 TF-IDF 进前 N 的词**")
            st.write("、".join(sorted(only_idf)) or "（无）")

        overlap = len(tf_top & idf_top)
        st.caption(f"两种方法前 {kw_num} 名重合 {overlap} 个（重合率 {overlap / kw_num:.0%}）。")

        # ---------------- 导出 ----------------
        st.divider()
        st.markdown("**导出结果（关键词表 + 两法对比表）**")
        export_kw = result.drop(columns=["原形形式"], errors="ignore")

        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            export_kw.to_excel(writer, index=False, sheet_name="关键词")
            cmp_df.to_excel(writer, index=False, sheet_name="两种方法对比")
        excel_bytes = excel_buffer.getvalue()

        txt_lines = ["\t".join(export_kw.columns)]
        for row in export_kw.itertuples(index=False):
            txt_lines.append("\t".join(str(v) for v in row))
        txt_bytes = ("\ufeff" + "\n".join(txt_lines)).encode("utf-8")

        base_name = f"关键词_{Path(source_name).stem}_{'TFIDF' if method.startswith('TF-IDF') else 'TF'}_Top{kw_num}"

        col_x, col_t = st.columns(2)
        with col_x:
            st.download_button(
                "导出 Excel（.xlsx）",
                data=excel_bytes,
                file_name=f"{base_name}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
            )
        with col_t:
            st.download_button(
                "导出 TXT（.txt）",
                data=txt_bytes,
                file_name=f"{base_name}.txt",
                mime="text/plain",
                width="stretch",
            )

        if st.button("保存到本机考核文件夹"):
            try:
                xlsx_path = BASE_DIR / f"{base_name}.xlsx"
                txt_path = BASE_DIR / f"{base_name}.txt"
                xlsx_path.write_bytes(excel_bytes)
                txt_path.write_bytes(txt_bytes)
                st.success(f"已保存：\n\n- {xlsx_path}\n- {txt_path}")
            except OSError:
                st.warning("当前在云端运行，服务器磁盘不可写。请使用上面的「导出」按钮，文件会下载到你的电脑。")

    # ---------------- ⑤ 规则与局限 ----------------
    with tab_e:
        st.subheader("提取规则")
        st.markdown(
            f"""
本模块的关键词提取是一条**纯统计 + 词性规则**的流水线，共 4 步：

| 步骤 | 规则 | 本语料的实际效果 |
|---|---|---|
| ① 文本清洗 | 统一弯引号与破折号 → 删除所有非字母字符（标点、数字）→ 多空格压成一个 → 保留 `\\n\\n` 段落边界 | {len(text):,} → {len(kw['cleaned']):,} 字符，切出 {n_docs} 段 |
| ② 分词 + 词性 | `word_tokenize` 切词，`pos_tag` 标词性（Penn Treebank） | 每段单独分词，保持段落独立性 |
| ③ 停用词过滤 | NLTK 英文停用词表（{data['stop_size']} 词）+ 用户自定义停用词 | 去掉 the / a / is / and 等功能词 |
| ④ 候选词筛选 | 词性必须属于所选类别（默认名词 + 形容词），且不是停用词 | 当前候选 {len(kw['total'])} 个词型 / {sum(kw['total'].values())} 个 token |

**打分规则（两种可选）**

- **词频法**：`score = 该词在全篇出现的总次数`
- **TF-IDF 法**（默认）：
  ```
  tf(t, p) = 该词在第 p 段出现次数 ÷ 第 p 段候选词总数
  idf(t)   = ln((1 + N) / (1 + df(t))) + 1      # N = 段落数 = 30
  score(t) = Σ_p tf(t, p) × idf(t)              # 按段落求和
  ```
  最终按 score 降序取前 N 个。**注意：IDF 的分母是“包含该词的段落数”，不是整篇。**
"""
        )

        st.divider()
        st.subheader("局限")

        st.markdown(
            """
**1. 单篇语料，关键词只是“篇内显著”，不是“全局重要”**
只有一篇文本时，任何词的“稀有度”都相对于这篇内部而言。换一篇语料，
排名会完全不同，结论不能外推。

**2. 段落当文档，结果对切分方式敏感**
本语料是连锁童话，段落短、句式重复。若改用“每 3 段合并”或“按句子切”，
N 与 df 都变，排名就会变。**段落长短不均还会放大短段里的词**（短段一个词占比更高）。

**3. 高度重复的文本会让 IDF 失效**
这篇每段都把前面的事件重述一遍，导致 `louse`、`flea` 的 df 很高（10/30），
IDF 被压低，词频法里排第 2、3 的词在 TF-IDF 里掉到第 4、7。
**TF-IDF 奖励“集中出现”，惩罚“均匀分布”，这在文学文本里不一定符合人的直觉。**

**4. 词性标注有错误率**
`little` 有 8 次被 `pos_tag` 标成名词（NN），实际应是形容词。
所以“只选名词”时它仍会漏进来。**词性筛选依赖标注质量，不是绝对可靠。**

**5. 纯统计方法不懂语义**
- 同义词不合并：`louse` / `insect` 会被当两个词；
- 多词术语被切碎：`little water pitcher` 变成 pitcher + water，丢了“水罐”这个整体概念；
- 反义词、上下位词完全无感知。

**6. 词形还原依赖 WordNet 词典**
专有名词、生僻词、专有搭配不在 WordNet 里就不还原。
本语料里 `began → begin`、`ran → run` 能还原，靠的是 WordNet 收录了不规则变化；
没收录的词只能原样保留。

**7. 语料规模太小，排名边界不稳定**
全篇仅 472 词、候选词 27 个。样本小意味着**第 8 名和第 12 名的差距可能只有 1 次出现**，
稍有参数变动（换个停用词、开关词形还原）就会换位。

**8. 清洗会丢掉可能有用的信息**
删除标点后，对话的引号边界、感叹语气都消失了；
如果做“谁在对谁说话”这类分析，清洗规则就需要放宽。

**9. 没有考虑词的位置权重**
标题、开头、结尾出现的词通常更重要（`Little Louse and Little Flea` 就在标题里），
但当前打分只用频次与分布，**位置信息完全没用上**。

**改进方向**：换成语料库级别的多文档 TF-IDF（如 50 篇格林童话）、
加入 TextRank / 共现网络、或引入 `sklearn` 的 `TfidfVectorizer` 做对比验证。
"""
        )

# ================================================================ 模块三：情感分析
else:
    st.header("情感分析（NLTK VADER）")

    DEFAULT_T = 0.05

    @st.cache_data(show_spinner="正在逐句计算情感分数…")
    def sentiment_table(text: str):
        """按段落 → 句子，逐句打分"""
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        sia = SentimentIntensityAnalyzer()          # 词典只加载一次
        rows = []
        para_compounds = []
        for pi, para in enumerate(paras, start=1):
            comps = []
            for si, sent in enumerate(sent_tokenize(para), start=1):
                sc = sia.polarity_scores(sent)
                rows.append(
                    {
                        "段落": pi,
                        "句子": si,
                        "文本": sent.strip(),
                        "pos": sc["pos"],
                        "neg": sc["neg"],
                        "neu": sc["neu"],
                        "compound": sc["compound"],
                    }
                )
                comps.append(sc["compound"])
            para_compounds.append(comps)
        return rows, para_compounds

    rows, para_compounds = sentiment_table(text)
    sent_df = pd.DataFrame(rows)
    n_docs_sent = len(para_compounds)

    def label(comp: float, t: float) -> str:
        if comp >= t:
            return "正面"
        if comp <= -t:
            return "负面"
        return "中性"

    # ---- 阈值参数 ----
    st.subheader("参数设置")
    t1, t2 = st.columns([1.4, 2])
    with t1:
        threshold = st.slider(
            "compound 阈值", min_value=0.0, max_value=0.9, value=DEFAULT_T, step=0.05,
            help="compound ≥ 阈值 判为正面；≤ -阈值 判为负面；中间为中性。阈值越大，判定越严格。",
        )
    with t2:
        st.caption(
            f"当前阈值 **{threshold:.2f}**（默认 0.05）。"
            "类别 = compound ≥ 阈值 → 正面；≤ -阈值 → 负面；否则中性。"
        )

    sent_df["类别"] = sent_df["compound"].apply(lambda c: label(c, threshold))
    sent_df["默认类别(0.05)"] = sent_df["compound"].apply(lambda c: label(c, DEFAULT_T))

    # ---- 指标卡 ----
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("句子总数", len(sent_df))
    c2.metric("平均 compound", f"{sent_df['compound'].mean():.3f}")
    c3.metric("正面句子", int((sent_df['类别'] == '正面').sum()))
    c4.metric("负面句子", int((sent_df['类别'] == '负面').sum()))
    c5.metric("中性句子", int((sent_df['类别'] == '中性').sum()))

    st.divider()

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        ["① 情感输出表", "② 段落情感走势", "③ 负面句子与代表句", "④ 阈值调整对照", "⑤ 对比两个段落", "⑥ 分数含义与误判", "⑦ 情感词高亮"]
    )

    # ---------------- ① 情感输出表 ----------------
    with tab1:
        st.subheader("逐句情感输出表")
        show_label = st.selectbox("只看哪个类别？", ["全部", "负面", "中性", "正面"])
        view = sent_df if show_label == "全部" else sent_df[sent_df["类别"] == show_label]
        st.dataframe(view, width="stretch", hide_index=True, height=440)
        st.caption(
            "VADER 的 pos / neg / neu 是三部分占比（总和为 1）；compound 是综合分数，范围 [-1, 1]，"
            "越接近 -1 越消极、越接近 1 越积极。"
        )

        # ---- 导出 ----
        st.divider()
        st.markdown("**导出情感表**")
        export_sent = sent_df.copy()

        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            export_sent.to_excel(writer, index=False, sheet_name="情感分析")
        excel_bytes = excel_buffer.getvalue()

        txt_lines = ["\t".join(export_sent.columns)]
        for row in export_sent.itertuples(index=False):
            txt_lines.append("\t".join(str(v) for v in row))
        txt_bytes = ("\ufeff" + "\n".join(txt_lines)).encode("utf-8")

        base_name = f"情感分析_{Path(source_name).stem}_阈值{threshold:.2f}"

        col_x, col_t = st.columns(2)
        with col_x:
            st.download_button(
                "导出 Excel（.xlsx）",
                data=excel_bytes,
                file_name=f"{base_name}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
            )
        with col_t:
            st.download_button(
                "导出 TXT（.txt）",
                data=txt_bytes,
                file_name=f"{base_name}.txt",
                mime="text/plain",
                width="stretch",
            )

        if st.button("保存到本机考核文件夹"):
            try:
                xlsx_path = BASE_DIR / f"{base_name}.xlsx"
                txt_path = BASE_DIR / f"{base_name}.txt"
                xlsx_path.write_bytes(excel_bytes)
                txt_path.write_bytes(txt_bytes)
                st.success(f"已保存：\n\n- {xlsx_path}\n- {txt_path}")
            except OSError:
                st.warning("当前在云端运行，服务器磁盘不可写。请使用上面的「导出」按钮，文件会下载到你的电脑。")

    # ---------------- ② 段落情感走势 ----------------
    with tab2:
        st.subheader("按段落绘制的情感走势")
        trend = pd.DataFrame(
            {
                "平均compound": [sum(cs) / len(cs) for cs in para_compounds],
                "句子数": [len(cs) for cs in para_compounds],
                "最低句compound": [min(cs) for cs in para_compounds],
            },
            index=[f"第{i}段" for i in range(1, n_docs_sent + 1)],
        )
        st.line_chart(trend["平均compound"], height=300)
        st.line_chart(trend["最低句compound"], height=220)

        low_idx = trend["平均compound"].idxmin()
        high_idx = trend["平均compound"].idxmax()
        st.info(
            f"**最消极的段落：{low_idx}**（平均 compound "
            f"{trend.loc[low_idx, '平均compound']:.3f}）；"
            f"最积极的段落：{high_idx}（{trend.loc[high_idx, '平均compound']:.3f}）。"
            "下面的虚线高度是『最低句 compound』，反映每段最消极的一句话跌到多深。"
        )
        with st.expander("走势数据表"):
            st.dataframe(trend.round(3), width="stretch")

    # ---------------- ③ 负面句子与代表句 ----------------
    with tab3:
        st.subheader("负面情感较强的句子")
        top_n_neg = st.slider("显示最消极的前 N 句", min_value=5, max_value=20, value=10, step=5)

        neg_top = sent_df.nsmallest(top_n_neg, "compound")[
            ["段落", "句子", "文本", "compound", "neg"]
        ]
        st.dataframe(neg_top, width="stretch", hide_index=True, height=min(460, 40 + 34 * top_n_neg))

        st.markdown("**代表句子（各取 1 句）**")
        reps = pd.concat(
            [
                sent_df.nsmallest(1, "compound"),
                sent_df.nlargest(1, "compound"),
                sent_df.iloc[(sent_df["compound"] - 0).abs().argsort()[:1]],
            ]
        ).assign(标签=["最负面", "最正面", "最中性"])
        st.dataframe(reps[["标签", "段落", "文本", "compound"]], width="stretch", hide_index=True)
        st.caption("最中性 = compound 绝对值最小的句子。")

    # ---------------- ④ 阈值调整对照 ----------------
    with tab4:
        st.subheader("阈值调整前后对照")
        st.markdown(
            f"""
类别判定依赖一个**人工设定的阈值**：compound ≥ 阈值 → 正面，≤ -阈值 → 负面，中间 → 中性。

**对照**：默认阈值 **0.05**（VADER 官方推荐） vs 当前 **{threshold:.2f}**：

| 类别 | 默认 0.05 下的句子数 | 当前 {threshold:.2f} 下的句子数 | 变化 |
|---|---|---|---|
| 正面 | {int((sent_df['默认类别(0.05)'] == '正面').sum())} | {int((sent_df['类别'] == '正面').sum())} | {int((sent_df['类别'] == '正面').sum() - (sent_df['默认类别(0.05)'] == '正面').sum()):+d} |
| 负面 | {int((sent_df['默认类别(0.05)'] == '负面').sum())} | {int((sent_df['类别'] == '负面').sum())} | {int((sent_df['类别'] == '负面').sum() - (sent_df['默认类别(0.05)'] == '负面').sum()):+d} |
| 中性 | {int((sent_df['默认类别(0.05)'] == '中性').sum())} | {int((sent_df['类别'] == '中性').sum())} | {int((sent_df['类别'] == '中性').sum() - (sent_df['默认类别(0.05)'] == '中性').sum()):+d} |
"""
        )

        changed = sent_df[sent_df["类别"] != sent_df["默认类别(0.05)"]]
        if len(changed):
            st.markdown(f"**因阈值调整而改变类别的句子（{len(changed)} 句）**")
            st.dataframe(
                changed[["段落", "句子", "文本", "compound", "默认类别(0.05)", "类别"]],
                width="stretch",
                hide_index=True,
            )
        else:
            st.write("（当前阈值下没有句子改变类别）")

        st.markdown(f"**筛选：当前阈值下所有消极句子（compound ≤ -{threshold:.2f}）**")
        neg_all = sent_df[sent_df["类别"] == "负面"]
        st.write(f"共 **{len(neg_all)}** 句")
        st.dataframe(
            neg_all[["段落", "句子", "文本", "compound"]].sort_values("compound"),
            width="stretch",
            hide_index=True,
            height=300,
        )

    # ---------------- ⑤ 对比两个段落 ----------------
    with tab5:
        st.subheader("对比两个段落")
        st.caption("选中两个段落，逐项比较情感统计。常用来对比『事件爆发段』和『铺垫段』。")

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            p_a = st.selectbox("段落 A", list(range(1, n_docs_sent + 1)), index=0)
        with col_p2:
            p_b = st.selectbox("段落 B", list(range(1, n_docs_sent + 1)), index=min(29, n_docs_sent - 1))

        def para_stats(p):
            cs = para_compounds[p - 1]
            sub = sent_df[sent_df["段落"] == p]
            return {
                "句子数": len(cs),
                "平均compound": sum(cs) / len(cs),
                "最低compound": min(cs),
                "最高compound": max(cs),
                "平均neg": sub["neg"].mean(),
                "平均pos": sub["pos"].mean(),
                "平均neu": sub["neu"].mean(),
                "负面句数": int((sub["类别"] == "负面").sum()),
                "正面句数": int((sub["类别"] == "正面").sum()),
            }

        stat_a, stat_b = para_stats(p_a), para_stats(p_b)
        cmp_rows = []
        for key in stat_a:
            va, vb = stat_a[key], stat_b[key]
            cmp_rows.append({"统计项": key, f"段落{p_a}": round(va, 3), f"段落{p_b}": round(vb, 3),
                             "差值": round(va - vb, 3)})
        st.dataframe(pd.DataFrame(cmp_rows), width="stretch", hide_index=True)

        st.markdown("**段落原文**")
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown(f"**段落 {p_a}**")
            st.write(paras[p_a - 1])
        with c_right:
            st.markdown(f"**段落 {p_b}**")
            st.write(paras[p_b - 1])

    # ---------------- ⑥ 分数含义与误判分析 ----------------
    with tab6:
        st.subheader("VADER 分数含义")
        st.markdown(
            """
VADER（Valence Aware Dictionary and sEntiment Reasoner）用一部 **7502 词的人工标注词典**打分，
每个词都有从 -4（极消极）到 +4（极积极）的情感强度，再结合三条规则调整：

1. **程度副词**：very good 比 good 更正面；
2. **否定词**：not good 会把 good 的分数翻转；
3. **标点与大小写**：good! 、GOOD 会加强。

输出 4 个数：

- **pos / neg / neu**：三类情感各占的比例，**三者之和恒为 1**（按词典命中的词计算）；
- **compound**：把所有词的强度归一化后的综合分，**范围 [-1, 1]**。
  一般以 **±0.05** 为界：≥0.05 正面、≤-0.05 负面、中间中性。
  compound 不是比例，**允许超过 pos 或 neg**，它是单独的聚合指标。
"""
        )

        st.subheader("文学叙事中的误判原因（结合本篇）")
        st.markdown(
            f"""
用 VADER 分析童话这类文学文本，误差主要来自五个方面：

**1. 词典没覆盖故事词**
`louse`（虱子）、`flea`（跳蚤）、`broom`、`pitcher` 都是中性名词，词典里没有情感强度——
VADER 看不出"虱子烫死"有多惨，只能靠 burned、death 这些词兜底。

**2. 否定疑问句造成假阳性/假阴性**
全篇的推进句式是 *"Why should I not be creaking?"*。
VADER 会按"not + 情感词"的规则处理，但这类**修辞性反问**根本没有情感，
句子里也没有真正的情感词，分数只能靠 neu 占比撑起来——容易被判成中性。

**3. 一词多义**
`cried` 在日常语境是"哭"（消极），但 VADER 词典给它的分值偏低且不分义项；
`bright flames`（明亮的火焰）里 **bright 是正面词**——可它在故事里是灾难的画面。

**4. 文学手法不识别**
连锁童话的**荒诞和幽默**（器物说话、最后全部被淹死）依赖上下文与常识，
VADER 只做词级匹配，完全感知不到"all drowned"黑色童话式的夸张。

**5. 句子级打分丢失篇章信息**
本系统按句子独立打分，"她死了 → 门在哭 → 扫帚在扫"是**递进的情绪链**，
逐句看分数是波动的，但读者的情绪是累积的。段落平均会抹平这种结构。
"""
        )
        st.caption(
            "验收要点：① 情感输出表（Tab ①，可导出）② 代表句子（Tab ③）③ 阈值调整前后对照（Tab ④）。"
        )

    # ---------------- ⑦ 情感词高亮 ----------------
    with tab7:
        st.subheader("情感词高亮阅读器")
        st.caption(
            "把 VADER 词典（7,502 词，强度 -4~+4）命中的词直接标在原文上："
            "绿色 = 积极词，红色 = 消极词，颜色越深情感强度越高。"
        )

        @st.cache_data(show_spinner="正在标注情感词…")
        def highlight_html(t: str):
            sia_h = SentimentIntensityAnalyzer()
            lex = sia_h.lexicon                     # 词条 → 分值 [-4, +4]
            esc = html_module.escape
            parts, last = [], 0
            pos_n = neg_n = 0
            strong = []                             # (词, 分值) 收集强情感词
            for m in re.finditer(r"[A-Za-z][A-Za-z']*", t):
                seg = esc(t[last:m.start()]).replace("\n\n", "<br><br>").replace("\n", "<br>")
                parts.append(seg)
                word = m.group(0)
                score = lex.get(word.lower())
                if score is not None and score >= 0.05:
                    alpha = 0.15 + 0.45 * min(abs(score) / 4, 1)
                    parts.append(f'<span style="background:rgba(67,160,71,{alpha:.2f});border-radius:3px">{esc(word)}</span>')
                    pos_n += 1
                    strong.append((word.lower(), score))
                elif score is not None and score <= -0.05:
                    alpha = 0.15 + 0.45 * min(abs(score) / 4, 1)
                    parts.append(f'<span style="background:rgba(211,47,47,{alpha:.2f});border-radius:3px">{esc(word)}</span>')
                    neg_n += 1
                    strong.append((word.lower(), score))
                else:
                    parts.append(esc(word))
                last = m.end()
            parts.append(esc(t[last:]).replace("\n\n", "<br><br>").replace("\n", "<br>"))
            html_out = (
                "<div style='line-height:2.1;font-size:1.02rem;background:#FFFFFF;"
                "padding:16px;border-radius:8px;border:1px solid #E0E0E0'>"
                + "".join(parts) + "</div>"
            )
            return html_out, pos_n, neg_n, strong

        html_out, pos_n, neg_n, strong = highlight_html(text)

        c1, c2, c3 = st.columns(3)
        c1.metric("积极词（绿）", pos_n)
        c2.metric("消极词（红）", neg_n)
        c3.metric("词典未命中", "其余均为中性")

        # 最强情感词排行
        pos_top = sorted([x for x in strong if x[1] > 0], key=lambda x: -x[1])[:5]
        neg_top = sorted([x for x in strong if x[1] < 0], key=lambda x: x[1])[:5]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**最强积极词 Top 5**")
            for w, s in pos_top:
                st.markdown(f'<span style="background:rgba(67,160,71,0.25);padding:2px 8px;border-radius:4px">{w}</span>　分值 {s:+.1f}', unsafe_allow_html=True)
        with c2:
            st.markdown("**最强消极词 Top 5**")
            for w, s in neg_top:
                st.markdown(f'<span style="background:rgba(211,47,47,0.25);padding:2px 8px;border-radius:4px">{w}</span>　分值 {s:+.1f}', unsafe_allow_html=True)

        st.markdown(html_out, unsafe_allow_html=True)
        st.caption(
            "说明：bright 在词典中是强积极词（+2.7），所以\"burn in bright flames\"整句被标成绿色——"
            "这正是 Tab ⑥ 讨论的语境误判，在这里可以直观看到。"
        )

