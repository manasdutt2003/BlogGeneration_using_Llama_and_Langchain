# Blog Generation using Llama 2 and LangChain

A Streamlit app that writes blog posts with a Llama 2 chat model running
locally on your CPU. No API key and no data leaves your machine.

Pick a topic, a target length and an audience (researchers, data scientists or
general readers) and the app generates a post you can download as Markdown.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Download a quantised Llama 2 chat model in GGUF format, for example
`llama-2-7b-chat.Q8_0.gguf` from
[TheBloke/Llama-2-7B-Chat-GGUF](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF),
and place it in a `models/` folder next to `app.py`.

To keep the model somewhere else, set the path:

```bash
set LLAMA_MODEL_PATH=D:\models\llama-2-7b-chat.Q8_0.gguf      # Windows
export LLAMA_MODEL_PATH=/path/to/llama-2-7b-chat.Q8_0.gguf    # macOS/Linux
```

You can also change it in the app's sidebar.

## Run

```bash
streamlit run app.py
```

## Notes

- The Q8 model needs about 8 GB of free RAM. Smaller quantisations such as
  `Q4_K_M` use about 4 GB and are faster, at some cost in quality.
- Generation runs on CPU and takes from tens of seconds to a few minutes
  depending on the requested length.
- The model is loaded once and cached, so only the first request is slow.
