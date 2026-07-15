from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama


def fetch_transcript(video_id: str) -> str:
    """Fetch the English transcript for a YouTube video."""
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.fetch(video_id, languages=["en"])
        transcript = " ".join(chunk.text for chunk in transcript_list)
        print("Transcript fetched successfully.\n")
        return transcript
    except TranscriptsDisabled:
        print("No captions available for this video.")
        return None


def build_vector_store(transcript: str):
    """Split transcript into chunks and build a FAISS vector store."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    print(f"Created {len(chunks)} chunks.\n")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


def main():
    video_id = "jPOxWOE-3Xk"

    # 1. Get transcript
    transcript = fetch_transcript(video_id)
    if transcript is None:
        raise SystemExit("No transcript available — stopping here.")

    # 2. Build retriever
    vector_store = build_vector_store(transcript)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Quick sanity check
    sample_results = retriever.invoke("how to gain confidence")
    print("Sample retrieval result (first doc):")
    print(sample_results[0].page_content if sample_results else "No results")
    print()

    # 3. Set up LLM + prompt
    llm = ChatOllama(model="qwen3:8b", temperature=0.2)

    prompt = PromptTemplate(
        template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

{context}
Question: {question}
""",
        input_variables=["context", "question"],
    )

    parser = StrOutputParser()

    # 4. Build the full chain
    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    })

    main_chain = parallel_chain | prompt | llm | parser

    # 5. Ask questions
    question = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
    answer = main_chain.invoke(question)
    print("Q:", question)
    print("A:", answer)
    print()

    summary_answer = main_chain.invoke("Can you summarize the video")
    print("Q: Can you summarize the video")
    print("A:", summary_answer)


if __name__ == "__main__":
    main()