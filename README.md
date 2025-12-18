# College RAG Chatbot

A production-ready **Retrieval-Augmented Generation (RAG) chatbot solution** designed to provide accurate, context-aware responses tailored to your college's specific knowledge base. This system leverages vector embeddings and large language models to deliver intelligent student support directly from your college's documented data.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Architecture](#project-architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Adding College Data to Vector Database](#adding-college-data-to-vector-database)
- [Running the Project](#running-the-project)
- [Example Workflow](#example-workflow)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)

---

## Project Overview

The **College RAG Chatbot** is an intelligent conversational system that answers user queries based exclusively on your college's verified knowledge base. Rather than relying on generic LLM training data, this chatbot retrieves relevant context from your documents and generates responses grounded in your college's specific information.

### Key Benefits

- **Accurate & Contextual**: Responses are based entirely on your college's documented data, eliminating hallucinations and out-of-context answers
- **Scalable**: Easily add new documents to expand the knowledge base without retraining
- **Enterprise-Ready**: Built with production-grade frameworks (FastAPI, Streamlit, LangChain)
- **Conversational Memory**: Maintains chat history for coherent, multi-turn conversations
- **Exportable**: Generate PDF transcripts of conversations for documentation and compliance

---

## Features

- ✅ **RAG-Powered Knowledge Retrieval**: Fetches relevant college data from vector database
- ✅ **Multi-Turn Conversations**: Maintains context across multiple exchanges using ConversationBufferMemory
- ✅ **User Onboarding**: Guides users through intake questionnaires to understand their needs
- ✅ **Resource Suggestions**: Recommends related college resources and services based on student context
- ✅ **Chat History Management**: Browse, track, and export previous conversations
- ✅ **PDF Export**: Generate conversation transcripts in PDF format
- ✅ **Emoji Filtering**: Cleans text for consistent formatting across platforms
- ✅ **Sensitive Topic Handling**: Gracefully responds to out-of-scope questions
- ✅ **Dual Interface**: Both Streamlit UI and FastAPI REST endpoint support
- ✅ **Configurable Prompts**: System and negative prompts can be customized for different use cases

---

## Tech Stack

### Core Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | Groq (llama-3.3-70b-versatile) | Primary language model for generating responses (currently used for development; production deployments should use professional, trusted LLM providers) |
| **Embeddings** | HuggingFace Transformers | Converting text to vector representations |
| **Vector Database** | Chroma | Persistent storage and retrieval of document embeddings |
| **Frontend** | Streamlit | Interactive web-based user interface |
| **Backend** | FastAPI | REST API for programmatic chatbot access |
| **Orchestration** | LangChain | Chains embeddings, memory, and LLM components |
| **PDF Generation** | FPDF | Export conversations as PDF documents |
| **Document Processing** | PyPDF2, Tesseract OCR | Extract and parse PDF documents |

### Dependencies

All required packages are listed in `requirements.txt`. Key packages include:
- `fastapi`, `uvicorn` - Web framework and server
- `streamlit` - Frontend interface
- `langchain`, `langchain-huggingface`, `langchain-chroma`, `langchain-groq` - RAG orchestration
- `pypdf2` - PDF text extraction
- `fpdf` - PDF generation
- `pytesseract` - OCR for scanned documents

---

## Project Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface Layer                    │
│  ┌─────────────────┐              ┌──────────────────┐      │
│  │   Streamlit     │              │   FastAPI REST   │      │
│  │   Web UI        │              │   Endpoint       │      │
│  └────────┬────────┘              └──────────┬───────┘      │
└───────────┼──────────────────────────────────┼──────────────┘
            │                                  │
            └──────────────────┬───────────────┘
                               │
┌──────────────────────────────┴─────────────────────────────┐
│                  Orchestration Layer (LangChain)           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  ConversationalRetrievalChain                        │  │
│  │  ├─ Memory: ConversationBufferMemory                 │  │
│  │  ├─ Retriever: Chroma Vector Store                   │  │
│  │  └─ LLM: ChatGroq                                    │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
            │                                 │
    ┌───────┴────────┐                        |
    │                │                        |  
┌───▼────────┐  ┌───▼──────┐            ┌─────▼──────┐
│  Embedding │  │  Groq    │            │ Chroma     │
│  Model     │  │  LLM     │            │ Vector DB  │
│ (HF)       │  │          │            │ (Persist)  │
└────────────┘  └──────────┘            └────────────┘
      │                                       |
      └────────────┐                  ┌───────▼────────┐
                   │                  │   Data Layer   │
           ┌───────▼───────┐          │ ┌────────────┐ │
           │ Document Layer│          │ │vector_db_  │ │
           │ ┌───────────┐ │          │ │dir/        │ │
           │ │data/      │ │          │ │chroma.db   │ │
           │ │*.pdf      │ │          │ └────────────┘ │
           │ └───────────┘ │          └────────────────┘
           └───────────────┘
```

### Workflow

1. **Document Ingestion**: PDFs are loaded from the `data/` directory
2. **Text Extraction**: Content extracted and split into chunks
3. **Vectorization**: Chunks converted to embeddings and stored in Chroma
4. **User Query**: User submits question via UI or API
5. **Retrieval**: Relevant chunks retrieved from vector database
6. **Generation**: LLM generates response using retrieved context + prompt
7. **Response**: Answer streamed to user with conversation memory maintained

---

## Installation

### Prerequisites

- **Python 3.8+** (recommended: Python 3.10 or higher)
- **pip** (Python package manager)
- **Tesseract OCR** (optional, for scanned PDFs)
  - [Download](https://github.com/UB-Mannheim/tesseract/wiki) and install
  - Update path in `vectorize_documents.py` if needed
- **GROQ API Key**
  - Sign up at [Groq Console](https://console.groq.com)
  - Generate an API key from your account settings

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd College_RAG_Chatbot
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `fastapi`, `uvicorn` - Web framework
- `streamlit` - Frontend
- `langchain`, `langchain-huggingface`, `langchain-chroma`, `langchain-groq` - RAG pipeline
- `pypdf2` - PDF processing
- `fpdf` - PDF export
- `pytesseract` - OCR (optional)

### Step 4: Verify Installation

```bash
python -c "import streamlit; import fastapi; import langchain; print('All dependencies installed successfully!')"
```

---

## Configuration

### Step 1: Create `config.json`

Create a file named `config.json` in the project root directory:

```bash
# Windows
echo {} > config.json

# macOS/Linux
touch config.json
```

### Step 2: Add API Keys

Edit `config.json` and add your Groq API key:

```json
{
  "GROQ_API_KEY": "gsk_your_actual_api_key_here",
  "COLLEGE_NAME": "Your College Name"
}
```

**Important**: 
- Replace `gsk_your_actual_api_key_here` with your actual Groq API key
- Replace `Your College Name` with your college's actual name
- **Never commit `config.json` to version control** - add it to `.gitignore`

### Step 3: Update College Name Throughout Code

The codebase contains placeholder text `"College Name"` that should be replaced with your actual college name. Key files:

- **`main.py`**: Update system prompts, page titles, and UI text
- **`vectorize_documents.py`**: Update any prompts or labels (if modified)

Use Find & Replace (Ctrl+H or Cmd+H) to replace all instances:

```
Find:    "College Name"
Replace: "Your Actual College Name"
```

### Step 4: (Optional) Configure Tesseract Path

If using scanned PDFs with OCR, update the Tesseract path in `vectorize_documents.py`:

```python
# On Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# On macOS
pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'

# On Linux
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
```

---

## Adding College Data to Vector Database

### Overview

The chatbot's knowledge base is built from your college's documents. Use `vectorize_documents.py` to process and store documents in the vector database.

### Supported Formats

- **PDF files** (.pdf) - Text extraction via PyPDF2
- **Scanned PDFs** - OCR extraction via Tesseract (if installed)

### Step 1: Prepare Documents

1. Create a `data/` directory in the project root (if it doesn't exist)
2. Place your college documents (PDFs) in the `data/` directory

```
College_RAG_Chatbot/
├── data/
│   ├── college_handbook.pdf
│   ├── programs_overview.pdf
│   ├── admissions_guide.pdf
│   └── student_faq.pdf
├── main.py
├── vectorize_documents.py
└── config.json
```

### Step 2: Run Vectorization Script

```bash
python vectorize_documents.py
```

**What happens:**
1. Reads all PDF files from `data/` directory
2. Extracts text from each PDF
3. Splits text into 2000-character chunks with 500-character overlap
4. Generates embeddings using HuggingFace transformers
5. Stores embeddings in Chroma vector database (`vector_db_dir/`)
6. Displays progress and completion status

**Output Example:**
```
Loading embedding model...
Loading and processing PDF documents...
Processing college_handbook.pdf...
Successfully processed college_handbook.pdf
Processing programs_overview.pdf...
Successfully processed programs_overview.pdf
Successfully loaded 2 documents
Splitting documents into chunks...
Split documents into 47 chunks
Creating vector database...
Successfully vectorized and stored documents in 'vector_db_dir'
```

### Step 3: Verify Vectorization

- Check that `vector_db_dir/` directory has been created
- Confirm `chroma.sqlite3` database file exists

### Adding More Documents Later

Simply add new PDFs to the `data/` directory and run the vectorization script again:

```bash
python vectorize_documents.py
```

New documents will be added to the existing vector database without losing previous documents.

### Troubleshooting Document Processing

| Issue | Solution |
|-------|----------|
| `No documents found` | Ensure PDFs are in `data/` directory with `.pdf` extension |
| `PDF extraction failed` | Verify PDF is not corrupted; try opening in Adobe Reader |
| `Tesseract not found` | Install Tesseract or update path in `vectorize_documents.py` |
| `Memory error with large PDFs` | Split large PDFs into smaller files |

---

## Running the Project

### Option 1: Streamlit Web Interface (Recommended for Users)

The Streamlit interface provides an interactive chatbot with a rich UI, chat history, and PDF export.

```bash
streamlit run main.py
```

**What to expect:**
- Opens in your default browser at `http://localhost:8501`
- Displays chatbot interface with sidebar containing college info
- Shows chat history and allows exporting conversations to PDF
- Supports real-time conversation with streaming responses

### Option 2: FastAPI REST Endpoint (Recommended for Integration)

The FastAPI backend provides a programmatic interface for chatbot integration.

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**API Endpoint:**
```
POST http://localhost:8000/chat
```

**Request Format:**
```json
{
  "message": "What programs does your college offer?"
}
```

**Response Format:**
```json
{
  "response": "Based on our college data, we offer the following programs..."
}
```

**Example cURL Request:**
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is your college known for?"}'
```

### Running Both Simultaneously (Advanced)

In separate terminal windows:

**Terminal 1 - FastAPI Backend:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Terminal 2 - Streamlit Frontend:**
```bash
streamlit run main.py
```

The Streamlit app can be modified to call the FastAPI endpoint instead of running locally.

---

## Example Workflow

### Scenario: New Customer Inquiry

**Step 1: User Opens Chatbot**
```
User navigates to http://localhost:8501
Chatbot displays welcome message with college logo
```

**Step 2: Initial Interaction (Onboarding)**
```
Chatbot: "Welcome! To help you better, could you tell me:
1. What service are you looking for today?
2. What type of business do you run?"

User: "I'm a startup looking for web design services"
```

**Step 3: Retrieval & Generation**
```
Behind the scenes:
1. Chatbot retrieves relevant web design docs from vector DB
2. Groq LLM generates contextual response using college data
3. Response includes relevant service details and suggestions
```

**Step 4: Assistant Response**
```
Chatbot: "Great! We offer comprehensive web design services including:
• Responsive UI/UX design
• Frontend development
• CMS integration
• Performance optimization

Since you're a startup, you might also benefit from our 
branding and digital marketing services to maximize visibility."
```

**Step 5: Chat History & Export**
```
User continues conversation or exports chat to PDF
PDF includes full conversation transcript with timestamp
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **Module not found** | Ensure all dependencies installed: `pip install -r requirements.txt` |
| **Groq API Key error** | Verify API key in `config.json`; check Groq account permissions |
| **Vector DB not found** | Run `python vectorize_documents.py` to create database |
| **Streamlit not starting** | Check Python version (3.8+); try `streamlit run main.py --logger.level=debug` |
| **Slow responses** | Check internet connection (requires Groq API access); verify document count in vector DB |
| **PDF export fails** | Ensure write permissions in project directory; check file name for special characters |

### Debug Mode

For detailed logging, modify `main.py`:

```python
# Add at the top of main.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

For Streamlit debug output:
```bash
streamlit run main.py --logger.level=debug
```

---

## Future Improvements

### Phase 2: Enhanced Features
- [ ] **Multi-language Support**: Auto-detect and respond in user's language
- [ ] **Advanced Analytics**: Track query patterns, user satisfaction, common topics
- [ ] **Knowledge Base Management UI**: Web interface for adding/updating documents without coding
- [ ] **Conversation Analytics Dashboard**: Visual metrics on bot performance and usage
- [ ] **User Authentication**: Track individual users; personalize responses based on user history

### Phase 3: Scalability & Performance
- [ ] **Caching Layer**: Cache frequent queries for faster responses
- [ ] **Database Optimization**: Index vector DB for faster retrieval
- [ ] **Load Balancing**: Deploy multiple bot instances with load balancer
- [ ] **Async Document Processing**: Handle large document uploads asynchronously
- [ ] **GraphQL API**: Alternative REST endpoint with flexible querying

### Phase 4: Intelligence & Customization
- [ ] **For Sensitive Inquiries**: Always escalate to appropriate human representatives
- [ ] **Fine-tuned Models**: Option to fine-tune Groq model on college-specific language
- [ ] **Custom Prompt Builder**: UI to create and test different system prompts
- [ ] **Fallback Mechanisms**: Escalate to human agent if confidence score is low
- [ ] **A/B Testing**: Test different response styles and track effectiveness
- [ ] **Feedback Loop**: Allow users to rate responses; use feedback to improve system

### Phase 5: Integration & Deployment
- [ ] **Slack Integration**: Deploy chatbot as Slack bot for internal use
- [ ] **WhatsApp/Telegram**: Multi-channel support for messaging platforms
- [ ] **CRM Integration**: Connect to Salesforce/HubSpot for customer data
- [ ] **Docker Containerization**: Package for easy deployment
- [ ] **Kubernetes Orchestration**: Deploy on cloud platforms (AWS, GCP, Azure)

---

## License

See the [LICENSE](LICENSE) file for licensing information.

---

## Support & Contributions

For issues, questions, or contributions:

1. **Report Issues**: Create an issue with detailed reproduction steps
2. **Suggest Features**: Open a feature request with use case details
3. **Contributing**: Fork the repository, create a feature branch, and submit a pull request

---

## Disclaimer

This chatbot provides responses based exclusively on your college's verified data. It does not:
- Make legal, financial, or medical decisions
- Access real-time data beyond the initial knowledge base
- Guarantee 100% accuracy (always verify critical information)
- Replace professional human support

For sensitive inquiries, always escalate to appropriate human representatives.

---

## Contact

- 📧 Email: [shahdeep1406@gmail.com](mailto:shahdeep1406@gmail.com)
- 🐙 GitHub: [DeepShah1406](https://github.com/DeepShah1406)
- 💼 LinkedIn: [deepshah1406](https://www.linkedin.com/in/deepshah1406)

---

## **Acknowledgments**

- **LangChain**: For building robust chains of logic for querying and text processing.
- **Chroma**: For efficiently managing and querying vectorized data.

---

**Last Updated**: December 2025  
**Version**: 1.0.0
