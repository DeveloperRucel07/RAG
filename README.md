# RAG (Retrieval-Augmented Generation) System

A Python-based Retrieval-Augmented Generation system that allows users to query documents using natural language and receive AI-generated answers based on relevant document content.

## Features

- **Document Processing**: Upload and process text documents with automatic embedding generation
- **Semantic Search**: Find relevant documents using sentence embeddings and vector similarity
- **AI-Powered Responses**: Generate accurate answers using Google's Gemini AI model
- **Supabase Integration**: Store documents and embeddings in a scalable PostgreSQL database
- **German Language Support**: Optimized prompts and responses for German-speaking users

## Prerequisites

- Python 3.10 or higher
- Supabase account and project
- Google AI API key (for Gemini)
- Virtual environment (recommended)

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd RAG
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Setup

1. Create a `.env` file in the project root with the following environment variables:

   ```env
   SUPABASE_URL=your_supabase_project_url
   SUPABASE_KEY=your_supabase_anon_key
   GEMINI_API=your_google_gemini_api_key
   ```

2. Set up your Supabase database:
   - Create a new Supabase project
   - Create a `documents` table with the following schema:
     ```sql
     CREATE TABLE documents (
       id SERIAL PRIMARY KEY,
       title TEXT,
       body TEXT,
       embedding VECTOR(384)  -- Adjust dimension based on your embedding model
     );
     ```
   - Create a function for similarity search:
     ```sql
     CREATE OR REPLACE FUNCTION match_documents(
       query_embedding VECTOR(384),
       match_threshold FLOAT DEFAULT 0.78,
       match_count INT DEFAULT 10
     )
     RETURNS TABLE(id INT, title TEXT, body TEXT, similarity FLOAT)
     LANGUAGE plpgsql
     AS $$
     BEGIN
       RETURN QUERY
       SELECT
         documents.id,
         documents.title,
         documents.body,
         1 - (documents.embedding <=> query_embedding) AS similarity
       FROM documents
       WHERE 1 - (documents.embedding <=> query_embedding) > match_threshold
       ORDER BY documents.embedding <=> query_embedding
       LIMIT match_count;
     END;
     $$;
     ```

## Usage

### Running the Application

Execute the main script:

```bash
python main.py
```

The application will prompt you to enter a question. It will then:
1. Generate an embedding for your query
2. Search for relevant documents in Supabase
3. Send the relevant documents and your question to Gemini AI
4. Display the generated answer

### Adding Documents

To add documents to the system, place text files in the `documents/` directory and modify the code to upload them:

```python
# Example: Upload all documents in the documents folder
for file in os.listdir("documents"):
    if file.endswith(".txt"):
        textfile_to_string(os.path.join("documents", file))
```

## Project Structure

```
RAG/
├── main.py                 # Main application script
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── documents/             # Directory for document files
│   ├── Document1.txt
│   ├── Document2.txt
│   └── Document3.txt
└── .env                   # Environment variables (create this file)
```

## Dependencies

Key dependencies include:

- `sentence-transformers`: For generating text embeddings
- `supabase`: For database operations
- `google-genai`: For AI-powered text generation
- `python-dotenv`: For environment variable management
- `torch`: For machine learning operations
- `transformers`: For NLP model handling

See `requirements.txt` for the complete list of dependencies.

## How It Works

1. **Document Ingestion**: Text documents are processed and converted to vector embeddings using Sentence Transformers
2. **Storage**: Documents and their embeddings are stored in Supabase
3. **Query Processing**: User queries are converted to embeddings and matched against stored documents
4. **Response Generation**: Relevant documents are sent to Google's Gemini AI along with the user's question to generate a contextual response

## Configuration

- **Embedding Model**: Uses `Supabase/gte-small` by default (384-dimensional embeddings)
- **Similarity Threshold**: Set to 0.78 for document matching
- **Max Results**: Returns up to 10 most relevant documents
- **AI Model**: Uses `gemini-2.5-flash` for response generation

## Troubleshooting

- Ensure all environment variables are set correctly
- Verify Supabase database schema matches the requirements
- Check API keys have the necessary permissions
- Make sure all dependencies are installed in the virtual environment

