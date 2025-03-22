### Fixed-Length Chunking (Character-Based)
✅ Method: Splits text into fixed-length character chunks.
✅ Pros: Simple & fast.
✅ Cons: Might split important information in the middle.

    from langchain.text_splitter import CharacterTextSplitter
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)

### Sentence-Based Chunking
✅ Method: Splits text by sentences instead of fixed length.
✅ Pros: Keeps sentences intact.
✅ Cons: Might create very large chunks if a sentence is too long.

    from langchain.text_splitter import SpacyTextSplitter
    splitter = SpacyTextSplitter()
    chunks = splitter.split_text(text)

### Recursive Character-Based Chunking (💡 We're Using This!)
✅ Method: Tries to split by paragraphs → sentences → words (in that order).
✅ Pros: Keeps context better than simple character splitting.
✅ Cons: Slightly slower but much more effective.

    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)

### Token-Based Chunking (Useful for AI Models)
✅ Method: Splits based on tokens (not characters).
✅ Pros: Works well for LLMs like OpenAI, Ollama, etc.
✅ Cons: Requires a tokenizer.

    from langchain.text_splitter import TokenTextSplitter
    splitter = TokenTextSplitter(model_name="gpt-3.5-turbo", chunk_size=200, chunk_overlap=20)
    chunks = splitter.split_text(text)

### Custom Rule-Based Chunking
✅ Method: Uses custom rules like splitting at specific keywords (e.g., "Invoice ID:", "Date:", "Amount:").
✅ Pros: Highly customizable.
✅ Cons: Needs domain-specific knowledge.

    def custom_chunking(text):
        chunks = text.split("Invoice ID:")  # Split based on a keyword
        return [chunk.strip() for chunk in chunks if chunk]
    chunks = custom_chunking(text)

### Semantic Chunking (📌 Meaning-Based Chunking)
✅ What is it?
Semantic chunking splits text based on meaning, rather than characters, sentences, or word count. Instead of blindly breaking text at a certain length, it ensures that each chunk contains a complete idea or topic.

✅ How does it work?

Uses AI/NLP models to detect topic boundaries.

Avoids cutting sentences in the middle of a concept.

Keeps related information in a single chunk.

✅ Example Use Case

Legal documents (keeping related laws together).

Research papers (splitting by topics).

Invoices (splitting by sections: Order Details, Payment, Taxes).

    from langchain.text_splitter import SemanticChunker
    from langchain.embeddings.openai import OpenAIEmbeddings
    splitter = SemanticChunker(OpenAIEmbeddings())  # Uses embeddings to detect meaning-based breaks
    chunks = splitter.split_text(text)

### Agentic Chunking (📌 Adaptive AI-Based Chunking)
✅ What is it?
Agentic chunking is a dynamic approach where an AI agent decides how to split text. It learns from past data and optimizes chunking based on the retrieval task.

✅ How does it work?

Uses AI agents to analyze the structure of the text before splitting.

Decides dynamically what chunk size is best based on query intent.

Can adjust chunking strategy on-the-fly for different document types.

✅ Example Use Case

Customer Support Chatbots (chunks different parts based on query).

Financial Reports (dynamically breaks financial data and analysis separately).

Medical Records (adaptive chunking for patient history vs diagnosis).

    from langchain.text_splitter import AgenticTextSplitter
    splitter = AgenticTextSplitter()
    chunks = splitter.split_text(text)
