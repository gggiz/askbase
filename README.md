# 🎀 AskBase — 永雏塔菲 RAG 智能问答系统

上传文档，塔菲帮你回答！( •̀ ω •́ )✧

基于 **RAG（检索增强生成）+ Agent（工具调用）** 架构的桌面知识库问答应用，打包为 Windows EXE，开箱即用。

---

## ✨ 功能

- 📄 **多格式文档**：支持 PDF / TXT / MD 上传解析
- 🔍 **语义检索**：BGE 嵌入模型 + ChromaDB 向量存储，按内容相似度精准召回
- 🤖 **Agent 工具调用**：可自主调用计算器、跨知识库搜索
- 🎀 **永雏塔菲人设**：可爱 VTuber 风格回答，带口癖和颜文字
- 💻 **Windows EXE**：PyInstaller 打包，无需安装 Python，双击即用
- 🔑 **界面输入 Key**：API Key 在页面内设置，不留文件，安全便捷

---

## 🚀 快速开始

1. 下载 `dist/AskBase.zip` 解压
2. 注册 [DeepSeek](https://platform.deepseek.com) 获取 API Key
3. 双击 `AskBase.exe`，等待启动
4. 浏览器打开 `http://127.0.0.1:7860`
5. 在页面顶部粘贴 API Key，点"设置 Key"
6. 创建知识库 → 上传文档 → 提问！

---

## 🏗 技术栈

| 层 | 技术 |
|---|------|
| 大模型 | DeepSeek (OpenAI 兼容 API) |
| 嵌入模型 | BAAI/bge-small-zh-v1.5 |
| 向量数据库 | ChromaDB |
| 文档解析 | PyMuPDF |
| 前端 | Gradio 6.x (粉色主题) |
| 打包 | PyInstaller |
| 运行环境 | Windows 10/11 x64 |

---

## 📁 项目结构

```
hf_space/
├── app.py              # 主程序（RAG + Agent + Gradio UI）
├── AskBase.spec        # PyInstaller 打包配置
├── requirements.txt    # Python 依赖
├── 使用说明.txt         # 用户手册
└── dist/               # 打包输出（EXE）
    └── AskBase/
        ├── AskBase.exe       # 主程序
        └── _internal/        # 运行时依赖
```

---

## 🔧 开发

```bash
pip install -r requirements.txt
python app.py
```

---

## 📝 架构

```
文档上传 → 分块(512token,64overlap) → BGE向量化 → ChromaDB
                                                      ↓
用户提问 → 向量化 → 语义检索 top-5 → 拼接 Prompt → DeepSeek → 答案
                                                      ↑
                                          Agent循环(可选): 工具调用 ≤5轮
```

---

## ⚠️ 注意

- 首次运行需下载嵌入模型（约 400MB），之后秒启动
- API Key 仅保存在当前会话内存中，关闭程序即消失
- 知识库数据存储在 EXE 同目录 `chroma_data/` 下
