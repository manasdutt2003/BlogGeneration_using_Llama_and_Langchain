import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

st.set_page_config(
    page_title="Generate Blogs",
    page_icon="🚀",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.header("🚀 Generate Blogs with LLM")

input_text = st.text_input("Enter the Blog Topic")

def getLlamaResponse(topic, no_words, blog_style):
    # Load Llama model
    model = CTransformers(
        model=r"C:\Blog_Generation\models\llama-2-7b-chat.Q8_0.gguf",
        model_type="llama",
        config={"max_new_tokens": 512, "temperature": 0.7}
    )

    # Prompt Template
    template = """Write a detailed blog for {topic} given below in {no_words} words for {blog_style}."""
    prompt = PromptTemplate(
        input_variables=["topic", "no_words", "blog_style"],
        template=template,
    )

    # Generate response
    response = model.predict(prompt.format(topic=topic, no_words=no_words, blog_style=blog_style))
    return response

col1, col2 = st.columns([5,5])
with col1:
    no_words = st.text_input("No of Words")
with col2:
    blog_style = st.selectbox('Writing the blog for', ('Researchers', 'Data Scientists', 'Common People'), index=0)

submit = st.button("Generate Blog")

if submit:
    if input_text.strip() == "":
        st.warning("Please enter a topic first.")
    else:
        with st.spinner("Generating blog..."):
            st.write(getLlamaResponse(input_text, no_words, blog_style))


