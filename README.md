# CogniFlow - AI Research Assistant

An intelligent research assistant that routes queries, conducts research, and provides accurate answers with hallucination detection.

## Features

🔬 **Smart Query Routing**
- Environmental queries → Research flow with document retrieval
- General questions → Direct AI responses  
- Unclear queries → Asks for clarification

🧠 **Advanced Capabilities**
- Multi-step research planning
- Ensemble retrieval (BM25 + Vector + MMR)
- Cohere re-ranking for better relevance
- Hallucination detection and validation
- Human-in-the-loop approval system

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
Create a `.env` file with:
```
OPENAI_API_KEY=your_openai_key
COHERE_API_KEY=your_cohere_key
```

## Running the Application

### Streamlit UI (Recommended)
```bash
streamlit run streamlit_app.py
```

### Command Line Interface (Legacy)
```bash
python app_cli.py
```

## Configuration

Edit `config.yaml` to customize:
- Retrieval settings (top_k, weights, etc.)
- Model configuration
- Vector store parameters

## Project Structure

```
CogniFlow/
├── streamlit_app.py          # Streamlit UI
├── app_cli.py                # CLI interface (backup)
├── config.yaml               # Configuration
├── main_graph/               # Main workflow graph
│   ├── graph_builder.py      # Graph construction
│   └── graph_states.py       # State definitions
├── subgraph/                 # Research subgraph
│   ├── graph_builder.py      # Researcher graph
│   └── graph_states.py       # Research states
├── utils/                    # Utility functions
│   ├── prompt.py             # System prompts
│   └── utils.py              # Helper functions
└── retriever/                # Document retrieval
    └── retreiver.py          # Retrieval logic
```

## Workflow

```
START → Analyze Query
    ├─→ Environmental: Research → Documents → Answer → Validate → END
    ├─→ More Info: Ask Clarification → END
    └─→ General: Direct Answer → END
```

## Technologies

- **LangGraph**: Workflow orchestration
- **OpenAI**: Language models (gpt-4o-mini)
- **Cohere**: Document re-ranking
- **Chroma**: Vector database
- **Streamlit**: Web interface
- **LangChain**: AI framework

## License

MIT
