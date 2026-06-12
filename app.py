import streamlit as st
import tiktoken
from transformers import AutoTokenizer


@st.cache_resource
def get_tokenizer(model_id):
    return AutoTokenizer.from_pretrained(model_id)

st.header("Tokenizer")

st.subheader("Tokenization: Converting text into tokens")
text = st.text_area("Enter your text here", height=70, value="Hello, how are you today?")

# tiktoken (GPT-2)
st.write("### GPT-2 (tiktoken)")
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode(text)
st.write(f"**Token IDs:** {tokens}")

# Llama Model
st.write("### LLaMA-3.1-8B (Transformers)")
try:
    tokenizer = get_tokenizer("Qwen/Qwen2.5-7B")
    token_ids = tokenizer(text)["input_ids"]
    st.write(f"**Token IDs:** {token_ids}")

    st.title("Token Visualizer")

    # A list of background colors to cycle through
    COLORS = [
        "#FFB3B3", "#FFD9B3", "#FFFFB3", "#B3FFB3",
        "#B3FFFF", "#B3B3FF", "#FFB3FF", "#FF8080",
    ]

    html = ""
    for i, token_id in enumerate(token_ids):
        token_text = tokenizer.decode([token_id])
        # Replace spaces with non-breaking spaces for display
        display_text = token_text.replace(" ", "&nbsp;")
        color = COLORS[i % len(COLORS)]
        html += f'<span style="background-color:{color}; padding:2px 4px; margin:2px; border-radius:4px; font-size:16px; color: black; display: inline-block;">{display_text}</span>'

    st.markdown(html, unsafe_allow_html=True)
    st.write(f"**Total tokens:** {len(token_ids)}")

except Exception as e:
    st.error(f"Error loading tokenizer: {e}")





