import streamlit as st
from transformers import pipeline

# Load the summarization pipeline
@st.cache_resource
def load_summarization_pipeline():
    summarization_pipeline = pipeline("summarization", model="tanmay2809/BART_CNN_SAMSUM")
    return summarization_pipeline

summarizer = load_summarization_pipeline()

# Streamlit Web Interface
st.title("Text Summarization with BART")
st.write("Provide a paragraph or dialogues, and get a concise summary!")

# Input area
input_text = st.text_area("Enter text to summarize:", height=200)

# Summarization
if st.button("Summarize"):
    if input_text.strip():
        # Use pipeline for summarization
        summary = summarizer(input_text, max_length=300, truncation=True)[0]["summary_text"]
        
        # Display summary
        st.subheader("Summary")
        st.write(summary)
    else:
        st.warning("Please enter text to summarize.")

# Footer
st.markdown("---")
st.markdown("Fine-tuned BART model by **Tanmay**")

