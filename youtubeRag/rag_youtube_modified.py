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
        return transcript

    except TranscriptsDisabled:
        return None


def build_vector_store(transcript: str):
    """Split transcript into chunks and build a FAISS vector store."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = splitter.create_documents([transcript])

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
        history += (
            f"{message['role'].capitalize()}: {message['content']}\n"
        )

    return history


def initialize_rag(youtube_url: str):
    """
    Initializes the RAG pipeline.

    Returns:
        retriever
        llm
        parser
        prompt
        status_messages
    """

    status_messages = []

    video_id = get_video_id(youtube_url)

    if video_id is None:
        raise ValueError("Invalid YouTube URL.")

    status_messages.append(f"🎬 Video ID: {video_id}")

    transcript = fetch_transcript(video_id)

    if transcript is None:
        raise ValueError("Transcript unavailable.")

    status_messages.append("✅ Transcript fetched successfully.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = splitter.create_documents([transcript])

    status_messages.append(f"✅ Created {len(chunks)} chunks.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(chunks, embeddings)

    status_messages.append("✅ Vector Store Created.")

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    llm = ChatOllama(
        model="qwen3:8b",
        temperature=0.2,
        num_ctx=2048,
    )

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

    return retriever, llm, prompt, parser, status_messages


def ask_question(
    question,
    retriever,
    llm,
    prompt,
    parser,
    chat_history,
):
    """
    Ask a question to the RAG pipeline.
    """

    history = format_history(chat_history)

    parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
            "history": RunnableLambda(lambda _: history),
        }
    )

    chain = parallel_chain | prompt | llm | parser

    answer = chain.invoke(question)

    return answer


def main():
    """
    CLI version.
    """

    youtube_url = input("Enter YouTube URL: ").strip()

    (
        retriever,
        llm,
        prompt,
        parser,
        status_messages,
    ) = initialize_rag(youtube_url)

    for msg in status_messages:
        print(msg)

    chat_history = []

    print("=" * 80)
    print("🎥 YouTube RAG Chat")
    print("Type 'exit' to quit.")
    print("=" * 80)

    while True:

        question = input("\nYou: ").strip()

        if question.lower() == "exit":
            break

        answer = ask_question(
            question,
            retriever,
            llm,
            prompt,
            parser,
            chat_history,
        )

        print("\nAssistant:", answer)

        chat_history.append(
            {
                "role": "user",
                "content": question,
            }
        )

        chat_history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        print("-" * 80)


if __name__ == "__main__":
    main()
