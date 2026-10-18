# CV-Based Naive RAG (CLI)

A small command-line Retrieval-Augmented Generation app that answers questions using text retrieved from a CV PDF. It uses Python, LangChain, Chroma, and the OpenAI API. No advanced RAG optimization is used.

## Architecture

1. **Load:** `PyPDFLoader` reads `cv.pdf`.
2. **Split:** `RecursiveCharacterTextSplitter` creates 500-character chunks with 50-character overlap.
3. **Embed and store:** `OpenAIEmbeddings` (`text-embedding-3-small`) creates embeddings stored in an in-memory Chroma vector store.
4. **Retrieve:** Chroma selects the four most similar CV chunks for each question.
5. **Generate:** `gpt-4o-mini` answers using the retrieved context only, or states that the information is unavailable.

The CV is indexed once when the program starts and reused for all questions in that session.

## Setup and running

Requires Python 3.10+ and an OpenAI API key with API access/billing.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

1. Place your CV PDF in this folder and rename it `cv.pdf`.
2. Copy `.env.example` to `.env`, then add your OpenAI API key.
3. Run:

```bash
python app.py
```

Type questions at the `Question:` prompt. Type `exit` to quit. The CLI displays each answer and the retrieved excerpts (including page numbers) for evaluation.

## Testing and evaluation

| Question | Expected outcome |
|---|---|
| What programming languages does the candidate know? | Python, C++, Dart, Kotlin, Java, SQL, if the correct skills chunk is retrieved. |
| What is the candidate's GPA? | 3.51/4.00, if retrieved. |
| Where did the candidate work as a data engineering intern? | Summit Technology Solutions. |
| What methods were used in industrial machine sound classification? | Log-mel spectrograms and a 2D CNN. |
| What is the candidate's passport number? | “The information is not available in the CV.” |

For each test, assess whether the retrieved snippets are relevant and whether the generated answer is supported by them. The example outcomes are expectations based on the sample CV, not guaranteed model results.

## Repository contents

- `app.py` — command-line RAG program.
- `requirements.txt` — Python dependencies.
- `.env.example` — API key template.
- `.gitignore` — excludes `.env`, `cv.pdf`, and temporary files.
- `README.md` — setup and evaluation instructions.

Do not upload your API key or personal CV to a public repository. For sharing the dataset, use a redacted CV only with permission. API usage may incur charges.
