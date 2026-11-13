"""Generate blog posts with a local Llama 2 model, LangChain and Streamlit."""

from __future__ import annotations

import os

import streamlit as st
from langchain_community.llms import CTransformers
from langchain_core.prompts import PromptTemplate

DEFAULT_MODEL_PATH = os.getenv(
    "LLAMA_MODEL_PATH", os.path.join("models", "llama-2-7b-chat.Q8_0.gguf")
)
AUDIENCES = ("Researchers", "Data Scientists", "Common People")

# Llama 2 chat models expect the [INST] wrapper; without it they tend to
# continue the prompt instead of following it.
PROMPT = PromptTemplate(
    input_variables=["topic", "no_words", "blog_style"],
    template=(
        "[INST] Write a blog post about \"{topic}\" for an audience of "
        "{blog_style}. Aim for roughly {no_words} words. Use a clear title "
        "and short paragraphs. [/INST]"
    ),
)

st.set_page_config(
    page_title="Generate Blogs",
    page_icon="🚀",
    layout="centered",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner="Loading the Llama model (first run only)...")
def load_model(model_path: str, max_new_tokens: int, temperature: float):
    """Load the GGUF model once per configuration instead of on every click."""
    return CTransformers(
        model=model_path,
        model_type="llama",
        config={
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "context_length": 2048,
        },
    )


def generate_blog(model, topic: str, no_words: int, blog_style: str) -> str:
    prompt = PROMPT.format(topic=topic, no_words=no_words, blog_style=blog_style)
    return model.invoke(prompt).strip()


st.header("🚀 Generate Blogs with LLM")

with st.sidebar:
    st.subheader("Model settings")
    model_path = st.text_input("Path to GGUF model", DEFAULT_MODEL_PATH)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)

topic = st.text_input("Enter the Blog Topic")

col1, col2 = st.columns(2)
with col1:
    no_words = st.number_input("No of Words", min_value=50, max_value=800, value=300, step=50)
with col2:
    blog_style = st.selectbox("Writing the blog for", AUDIENCES, index=0)

if st.button("Generate Blog"):
    if not topic.strip():
        st.warning("Please enter a topic first.")
    elif not os.path.isfile(model_path):
        st.error(
            f"Model file not found at `{model_path}`. Download a Llama 2 chat "
            "GGUF file and set the path in the sidebar or in `LLAMA_MODEL_PATH`."
        )
    else:
        # About 1.4 tokens per English word, plus room for the title.
        max_new_tokens = min(1500, int(no_words * 1.4) + 64)
        try:
            model = load_model(model_path, max_new_tokens, temperature)
            with st.spinner("Generating blog..."):
                blog = generate_blog(model, topic.strip(), int(no_words), blog_style)
        except Exception as error:
            st.error(f"Generation failed: {error}")
        else:
            st.markdown(blog)
            st.caption(f"{len(blog.split())} words")
            st.download_button("Download as Markdown", blog, file_name="blog.md")
