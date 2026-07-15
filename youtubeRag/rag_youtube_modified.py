from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from id_extractor import get_video_id


def fetch_transcript(video_id: str) -> str:
    """Fetch the English transcript for a YouTube video."""
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.fetch(video_id, languages=["en"])
        transcript = " ".join(chunk.text for chunk in transcript_list)
        print("✅ Transcript fetched successfully.\n")
        return transcript

    except TranscriptsDisabled:
        print("❌ No captions available for this video.")
        return None


def build_vector_store(transcript: str):
    """Split transcript into chunks and build a FAISS vector store."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = splitter.create_documents([transcript])

    print(f"✅ Created {len(chunks)} chunks.\n")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store


def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


def format_history(chat_history):
    if len(chat_history) == 0:
        return "No previous conversation."

    history = ""

    for message in chat_history:
        history += f"{message['role'].capitalize()}: {message['content']}\n"

    return history


def main():

    # ----------------------------
    # YouTube URL
    # ----------------------------
    youtube_url = input("Enter YouTube URL: ").strip()

    video_id = get_video_id(youtube_url)

    if video_id is None:
        raise SystemExit("❌ Invalid YouTube URL.")

    print(f"🎬 Video ID: {video_id}\n")

    # ----------------------------
    # Fetch Transcript
    # ----------------------------
    transcript = fetch_transcript(video_id)

    if transcript is None:
        raise SystemExit("Transcript unavailable.")

    # ----------------------------
    # Build Vector Store
    # ----------------------------
    vector_store = build_vector_store(transcript)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    # ----------------------------
    # LLM
    # ----------------------------
    llm = ChatOllama(
        model="qwen3:8b",
        temperature=0.2,
        num_ctx=2048,
    )

    # ----------------------------
    # Prompt
    # ----------------------------
    prompt = PromptTemplate(
        template="""
You are a helpful AI assistant.

Answer ONLY from the transcript context.

Also consider the previous conversation while answering.

If the answer is not present in the transcript, reply:

"I don't know based on the transcript."

Conversation History:
{history}

Transcript Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["history", "context", "question"],
    )

    parser = StrOutputParser()

    # ----------------------------
    # Store conversation
    # ----------------------------
    chat_history = []

    print("=" * 80)
    print("🎥 YouTube RAG Chat")
    print("Type 'exit' to quit.")
    print("=" * 80)

    # ----------------------------
    # Chat Loop
    # ----------------------------
    while True:

        question = input("\nYou: ").strip()

        if question.lower() == "exit":
            print("\n👋 Goodbye!")
            break

        history = format_history(chat_history)

        parallel_chain = RunnableParallel(
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
                "history": RunnableLambda(lambda _: history),
            }
        )

        main_chain = parallel_chain | prompt | llm | parser

        answer = main_chain.invoke(question)

        print("\nAssistant:", answer)

        # Save user message
        chat_history.append(
            {
                "role": "user",
                "content": question,
            }
        )

        # Save assistant response
        chat_history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        print("-" * 80)


if __name__ == "__main__":
    main()