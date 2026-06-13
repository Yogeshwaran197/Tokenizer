Tokenizer Visualizer

An interactive tokenizer web app built with Streamlit that lets you visualize how LLMs break text into tokens — color-coded, with token IDs and counts.

🚀 Live Demo: tokenizer-oepv9wdkcphwxrcfwvpwmq.streamlit.app


✨ Features


🎨 Color-coded token visualization — each token gets a unique background color
🔢 Token ID display — see the raw integer IDs the model uses internally
📊 Token count — instantly see how many tokens your text uses
⚡ Real-time encoding — updates as you type



🛠️ Tech Stack


Streamlit — UI framework
tiktoken — OpenAI's BPE tokenizer
Transformers — HuggingFace tokenizer support
Python 3.10+



🚀 Run Locally

bash# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/tokenizer.git
cd tokenizer

# 2. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows PowerShell

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py


📦 Requirements

streamlit
tiktoken
transformers


📁 Project Structure

tokenizer/
├── app.py              # Main Streamlit app
├── requirements.txt    # Dependencies
└── README.md


💡 How It Works


Enter any text in the input box
The tokenizer encodes the text using BPE (Byte Pair Encoding)
Each token is decoded back to its text chunk and displayed with a color
Token IDs and total count are shown below



🧠 What is Tokenization?

Tokenization is how LLMs split text into smaller units called tokens before processing. A token can be:


A full word (hello)
A subword (un, believ, able)
A punctuation character (,, .)
A special symbol


Understanding tokenization is key to understanding how LLMs work under the hood.


👤 Author

Yogesh (YOGI)

B.E. AI & Data Science — SNS College of Engineering

Aspiring LLM / GenAI Engineer
