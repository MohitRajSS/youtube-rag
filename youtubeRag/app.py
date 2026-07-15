import streamlit as st

from id_extractor import get_video_id
from rag_youtube_modified import (
    fetch_transcript,
    build_vector_store,
    format_docs,
    format_history,
)

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama


st.set_page_config(
    page_title="YouTube RAG",
    page_icon="🎥",
    layout="wide",
)

st.title("🎥 YouTube RAG Chat")


if "loaded" not in st.session_state:
    st.session_state.loaded = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None


# -------------------------------
# First ask for URL
# -------------------------------

if not st.session_state.loaded:

    url = st.chat_input("Paste YouTube URL")

    if url:

        with st.spinner("Loading video..."):

            video_id = get_video_id(url)

            if video_id is None:
                st.error("Invalid YouTube URL")
                st.stop()

            st.write(f"🎬 Video ID : {video_id}")

            transcript = fetch_transcript(video_id)

            if transcript is None:
                st.error("Transcript unavailable")
                st.stop()

            vector_store = build_vector_store(transcript)

            st.session_state.retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k":4},
            )

            st.session_state.loaded = True

        st.rerun()


# -------------------------------
# Chat
# -------------------------------

else:

    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask anything about the video...")

    if question:

        st.session_state.messages.append(
            {
                "role":"user",
                "content":question
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        history = format_history(st.session_state.messages)

        llm = ChatOllama(
            model="qwen3:8b",
            temperature=0.2,
            num_ctx=2048,
        )

        prompt = PromptTemplate(
            template="""
You are a helpful AI assistant.

Answer ONLY from the transcript.

Conversation:
{history}

Transcript:
{context}

Question:
{question}

Answer:
""",
            input_variables=["history","context","question"],
        )

        parser = StrOutputParser()

        parallel_chain = RunnableParallel(
            {
                "context": st.session_state.retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
                "history": RunnableLambda(lambda _: history),
            }
        )

        chain = parallel_chain | prompt | llm | parser

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                answer = chain.invoke(question)

                st.markdown(answer)

        st.session_state.messages.append(
            {
                "role":"assistant",
                "content":answer,
            }
        )