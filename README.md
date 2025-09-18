# 📄 PDF-Chatbot (LangChain + Streamlit + Ollama)

An **AI-powered PDF chatbot** built with **LangChain, HuggingFace embeddings, FAISS, Streamlit, and TinyLLaMA via Ollama**.
Upload any PDF and ask questions — the chatbot will retrieve relevant context and generate natural language answers.

---

## 🚀 Features

* 📂 **PDF Upload**: Load and process custom PDFs.
* 🔎 **Semantic Search**: Uses HuggingFace embeddings + FAISS for fast vector retrieval.
* 💬 **Conversational Memory**: Keeps track of previous questions/answers.
* 🖥️ **Streamlit UI**: Simple, interactive web interface.
* 🤖 **LLM Integration**: Powered by TinyLLaMA via Ollama for local inference.

---

## 🛠️ Tech Stack

* Python
* Streamlit
* LangChain
* HuggingFace Sentence Transformers
* FAISS (vector database)
* Ollama (TinyLLaMA model)

---

## 📂 Project Structure

```
pdf-chatbot/
│── chatbot.py           # Main Streamlit app (frontend + backend logic)
│── requirements.txt     # Dependencies
│── README.md            # Documentation
```

---

## 🔧 Installation & Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/<your-username>/pdf-chatbot.git
   cd pdf-chatbot
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**:

   ```bash
   streamlit run chatbot.py
   ```

4. **Upload a PDF** and start chatting! 🎉

---

## 📌 Example Usage

Ask:

```
"What are the main takeaways from section 2?"
```

Chatbot will return a concise, context-aware answer using the document content.

---

## 🌟 Future Improvements

* Support multiple PDFs at once
* Add option to use external APIs (OpenAI, Anthropic, etc.)
* Long-context summarization memory

---

## 📜 License

This project is open-source under the MIT License.

✨ Contributions welcome!
