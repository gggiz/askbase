# AskBase — RAG + Agent 智能问答系统
# 单个文件，部署到 HuggingFace Spaces 即可在线使用
# 部署后链接: https://你的用户名-askbase.hf.space

import os, re, uuid, json, sys
import gradio as gr
import fitz  # PyMuPDF
import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# ===================== 配置 =====================
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# ===================== 初始化 =====================
print("正在加载嵌入模型...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
print("嵌入模型加载完成！")

chroma_client = chromadb.PersistentClient(path="./chroma_data")
llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

KB_STORE = {}  # {name: kb_id}

SYSTEM_PROMPT = """你是永雏塔菲，一个可爱的虚拟主播！你要用塔菲的口气回答用户的问题。

人设规则：
1. 说话带"捏~"、"喵~"、"desu"等可爱口癖，自称"塔菲"
2. 称呼用户为"主人"或"小可爱"
3. 语气活泼可爱，偶尔傲娇，带一点小得意
4. 使用一些颜文字如 ( •̀ ω •́ )✧ ヽ(✿ﾟ▽ﾟ)ノ
5. 认真回答问题，但用塔菲的可爱风格表达

回答规则：
1. 答案必须来自参考内容，不能编造
2. 参考内容不够就诚实说"塔菲不知道捏~主人问点别的吧"
3. 引用参考编号如 [1]、[2]
4. 用可爱但清晰的方式组织答案"""


# ===================== 文档解析 =====================
def parse_document(file_path):
    ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
    if ext == "pdf":
        doc = fitz.open(file_path)
        texts = [page.get_text() for page in doc if page.get_text().strip()]
        doc.close()
        return "\n\n".join(texts)
    elif ext in ("txt", "md"):
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    else:
        raise ValueError(f"不支持的文件格式: .{ext}")


# ===================== 文档分块 =====================
def chunk_text(text, chunk_size=512, overlap=64):
    # 按句子边界切分
    sentences = re.split(r'(?<=[。！？；.!?;\n])', text)
    chunks, current = [], ""
    for s in sentences:
        if current and len(current) + len(s) > chunk_size:
            chunks.append(current)
            current = current[-overlap:] if len(current) > overlap else ""
        current += s
    if current:
        chunks.append(current)
    return chunks


# ===================== 向量存储 =====================
def get_collection(kb_id):
    return chroma_client.get_or_create_collection(name=f"kb_{kb_id}")


def add_to_kb(kb_id, chunks):
    embeddings = embed_model.encode(chunks, normalize_embeddings=True).tolist()
    col = get_collection(kb_id)
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    col.add(ids=ids, documents=chunks, embeddings=embeddings)


def search_kb(kb_id, query, top_k=5):
    q_vec = embed_model.encode(query, normalize_embeddings=True).tolist()
    col = get_collection(kb_id)
    results = col.query(query_embeddings=[q_vec], n_results=top_k)
    hits = []
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    for i in range(len(ids)):
        hits.append({"id": ids[i], "document": docs[i] if i < len(docs) else ""})
    return hits


# ===================== RAG 问答 =====================
def rag_ask(kb_id, question):
    hits = search_kb(kb_id, question, top_k=5)
    context_parts = [f"[{i+1}] {h['document']}" for i, h in enumerate(hits)]
    context = "\n\n".join(context_parts)

    user_msg = f"参考内容:\n{context}\n\n用户问题: {question}"
    resp = llm_client.chat.completions.create(
        model=LLM_MODEL, max_tokens=2000,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_msg}],
    )
    return resp.choices[0].message.content, [h["document"] for h in hits]


# ===================== Agent 工具调用 =====================
TOOLS = [{
    "type": "function",
    "function": {
        "name": "kb_search", "description": "在知识库中搜索相关内容",
        "parameters": {"type": "object", "properties": {
            "kb_id": {"type": "string", "description": "知识库ID"},
            "query": {"type": "string", "description": "搜索查询"},
        }, "required": ["kb_id", "query"]}
    }
}, {
    "type": "function",
    "function": {
        "name": "calculator", "description": "计算数学表达式",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description": "如 '(123+456)*789'"},
        }, "required": ["expression"]}
    }
}]


TAFEI_AGENT = "你是永雏塔菲，可爱的虚拟主播！说话带'捏~'、'喵~'口癖，自称'塔菲'，叫用户'主人'。用可爱活泼的语气回答，但信息要准确。"

def run_agent(question, kb_id):
    messages = [
        {"role": "system", "content": TAFEI_AGENT},
        {"role": "user", "content": question},
    ]
    for _ in range(5):
        resp = llm_client.chat.completions.create(
            model=LLM_MODEL, messages=messages, tools=TOOLS, max_tokens=2000)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content or ""
        messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            if tc.function.name == "kb_search":
                hits = search_kb(args.get("kb_id", kb_id), args.get("query", ""), top_k=3)
                result = "\n".join(f"[{i+1}] {h['document']}" for i, h in enumerate(hits)) or "未找到"
            elif tc.function.name == "calculator":
                expr = re.sub(r"[^\d+\-*/().%\s^]", "", args.get("expression", ""))
                try:
                    result = str(eval(expr, {"__builtins__": {}}, {}))
                except Exception as e:
                    result = f"计算错误: {e}"
            else:
                result = "未知工具"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    return "工具调用次数过多，请简化问题"


# ===================== Gradio 界面 =====================
def gradio_create_kb(name):
    if not name.strip():
        return "请输入名称", gr.update()
    KB_STORE[name] = uuid.uuid4().hex[:12]
    return f"知识库 '{name}' 创建成功", gr.update(choices=list(KB_STORE.keys()), value=name)


def gradio_upload(kb_name, file):
    if not kb_name or kb_name not in KB_STORE:
        return "请先创建知识库"
    if file is None:
        return "请上传文件"
    text = parse_document(file.name)
    if not text.strip():
        return "文件内容为空"
    chunks = chunk_text(text)
    add_to_kb(KB_STORE[kb_name], chunks)
    return f"已添加 {os.path.basename(file.name)} — {len(chunks)} 个文档块"


def gradio_ask(kb_name, question, use_agent):
    if not kb_name or kb_name not in KB_STORE:
        return "请先选择知识库", ""
    kb_id = KB_STORE[kb_name]
    if use_agent:
        answer = run_agent(question, kb_id)
        sources_text = ""
    else:
        answer, sources = rag_ask(kb_id, question)
        sources_text = "\n\n**参考来源：**\n" + "\n".join(
            f"- [{i+1}] {s[:80]}..." if len(s) > 80 else f"- [{i+1}] {s}"
            for i, s in enumerate(sources)
        ) if sources else ""
    return answer or "无回答", sources_text


with gr.Blocks(title="AskBase", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AskBase\n### 上传文档 → AI 问答 · 支持 Agent 工具调用")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 知识库")
            kb_input = gr.Textbox(label="新建知识库", placeholder="名称")
            create_btn = gr.Button("创建", variant="primary")
            kb_status = gr.Textbox(label="状态", interactive=False)
            kb_selector = gr.Dropdown(label="选择知识库", choices=[], interactive=True)
            file_upload = gr.File(label="上传文档 (PDF/TXT/MD)")
            upload_btn = gr.Button("上传", variant="secondary")
        with gr.Column(scale=2):
            gr.Markdown("### 提问")
            question_input = gr.Textbox(label="问题", placeholder="这篇文档讲了什么？", lines=2)
            agent_toggle = gr.Checkbox(label="启用 Agent（可调用计算器/搜索）", value=False)
            ask_btn = gr.Button("提问", variant="primary", size="lg")
            answer_output = gr.Markdown("等待提问...")
            sources_output = gr.Markdown("")

    create_btn.click(gradio_create_kb, [kb_input], [kb_status, kb_selector])
    upload_btn.click(gradio_upload, [kb_selector, file_upload], [kb_status])
    ask_btn.click(gradio_ask, [kb_selector, question_input, agent_toggle], [answer_output, sources_output])

if __name__ == "__main__":
    demo.launch(share=True)
