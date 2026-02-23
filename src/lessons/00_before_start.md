# Before start

## Preparing Ollama for Local LLMs and Embeddings

Later in the course, we will use **Ollama** to:

* run open-source LLMs locally, and
* generate embeddings for semantic search and vector-based memory.

Before we get there, we need to make sure our environment is ready.

### Why we need Ollama

We will connect our Python code to a local Ollama server to:

* generate embeddings without using any paid API,
* keep all data private on your own machine, and
* ensure fast, offline access to models.

### Installing Ollama

On Linux / Mac:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Check the installation:

```bash
ollama --version
```

### Pulling the Embedding Model

```bash
ollama pull nomic-embed-text
```

This model will be used to turn text into vectors.
It runs **100% locally**, no API key needed.

### Running the Ollama Server

Before using embeddings, start the server:

```bash
ollama serve
```

* By default, it runs on `localhost:11434`.
* If the port is already in use, you can choose another port:

```bash
ollama serve --port 11435
```

Check that the server is running:

```bash
curl http://127.0.0.1:11434
```

> **Tip:** Keep the Ollama server running in a terminal while working on embeddings.
> The Python code for embeddings will **not work** if the server is not active.
