# CV RAG chatbot

A small CLI that answers questions about the included CV using LangChain, Chroma, and OpenAI.

## Run

Requires Python 3.10+ and an OpenAI API key.

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `OPENAI_API_KEY`. Keep the included
`Abdalrahman E. Mohamed CV .pdf` beside `app.py`, then run:

```bash
python app.py
```

Type `exit` to quit or `clear` to reset memory.
The CLI prints the original question and the generated MultiQuery search questions.
The first CV chunk is always included so the model can see the candidate's name
even when a broad identity question does not retrieve the header.

## Task 1: evaluation and memory

```bash
python evaluate.py
```

This runs six standalone questions about the name, skills, GPA, an internship, a project,
and missing information. It also checks a follow-up question that needs memory.
It prints each answer, PASS/FAIL, and a total, returning exit code 1 if any check fails.
Expected facts come from the sample CV. Keyword checks can miss correct paraphrases
and do not prove that every statement is correct, so read the printed answers too.

The chatbot uses LangChain's `RunnableWithMessageHistory` with an in-memory
`InMemoryChatMessageHistory`. This helps with follow-ups like "What was the name
of that company again?" Memory lasts only until the program exits. Each standalone
evaluation case starts with empty memory.

## Task 2: retrieval techniques

The PDF is split into 500-character chunks with 50-character overlap. Embeddings use
`text-embedding-3-small`, and answers use `gpt-4o-mini`.

- **Pre-retrieval: MultiQuery**, from `PreRetrieval/Query Translation + Routing.ipynb`.
  I chose it because questions may use different words from the CV. The model writes
  three search versions, using recent messages to understand follow-ups. The app
  searches the original question and these versions, getting four chunks per query.
- **Post-retrieval: Reciprocal Rank Fusion (RRF)**, from the RAG-Fusion section in
  `PostRetrieval/PostRetrieval.ipynb`. I chose it because it combines the search
  results with a short formula and needs no extra service. A chunk gets
  `1 / (60 + rank)` from each result list (rank starts at 1). Scores for repeated
  chunks are added, and the four highest-scoring chunks go to the answer model.

These choices add searches and one model call per question. Improved accuracy has
not been established by comparing against the original app. Running the app or
evaluation uses the real API and incurs usage costs. Keep your API key private.
