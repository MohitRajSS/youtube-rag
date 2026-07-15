
# YouTube RAG

## Overview

YouTube RAG is a Retrieval-Augmented Generation (RAG) application that allows users to ask questions about a YouTube video. Instead of relying only on the language model's knowledge, the application retrieves relevant information from the video's transcript and generates answers based on that content.

---

## How the Model Works

The application follows the pipeline below:

1. The user provides a YouTube video URL.
2. The video ID is extracted from the URL.
3. The transcript of the video is fetched.
4. The transcript is divided into smaller text chunks.
5. Embeddings are created for each chunk and stored in a vector database.
6. When the user asks a question, the retriever searches for the most relevant transcript chunks.
7. The retrieved context, along with the user's question and chat history, is sent to the local LLM (Qwen3:8B via Ollama).
8. The model generates an answer using only the retrieved transcript, ensuring responses are grounded in the video's content.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/MohitRajSS/youtube-rag.git
cd youtube-rag/youtubeRag
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Download the Ollama model:

```bash
ollama pull qwen3:8b
```

Run the application:

```bash
streamlit run app.py
```

---

## Advantages

- Generates responses directly from the video's transcript.
- Provides accurate, context-aware answers related to the video.
- Allows users to ask detailed questions that general-purpose chatbots may not answer accurately for a specific video.
- Supports follow-up questions using conversation history.

---

## Disadvantages

- Uses a local LLM (Ollama), so response generation is slower than cloud-based models.
- Requires the YouTube video to have an available transcript.
- Response quality depends on the quality of the transcript.
- Currently supports only English transcripts.

---

## Technologies Used

- Python
- Streamlit
- LangChain
- Ollama
- ChromaDB
- HuggingFace Embeddings
- YouTube Transcript API
- Qwen3:8B
---
##The final preview of the application is available as pdf format
