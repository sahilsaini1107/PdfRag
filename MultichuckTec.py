import os
import re
import json
import numpy as np
import requests
from typing import List, Dict, Any, Tuple, Optional, Callable
from collections import defaultdict

# ---- Ollama Integration ----

def query_ollama(prompt: str, model: str = "llama2", system_prompt: str = None, temperature: float = 0.1) -> str:
    """Query an Ollama model with a given prompt."""
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature
    }
    
    if system_prompt:
        payload["system"] = system_prompt
    
    response = ""
    try:
        with requests.post(url, json=payload, stream=True) as resp:
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    if 'response' in data:
                        response += data['response']
    except Exception as e:
        print(f"Error querying Ollama: {e}")
        return ""
    
    return response

def get_embeddings(text: str, model: str = "nomic-embed-text") -> List[float]:
    """Get embeddings from Ollama."""
    url = "http://localhost:11434/api/embeddings"
    
    payload = {
        "model": model,
        "prompt": text
    }
    
    try:
        response = requests.post(url, json=payload)
        response_json = response.json()
        if 'embedding' in response_json:
            return response_json['embedding']
    except Exception as e:
        print(f"Error getting embeddings: {e}")
    
    return []

# ---- Basic Chunking Utilities ----

def simple_chunking(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Basic chunking by character count with overlap.
    
    Args:
        text: The input text to chunk
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        
        # Try to find a natural breaking point (period followed by space or newline)
        if end < text_len:
            # Look for a period followed by space or newline within the last 200 chars
            search_start = max(end - 200, start)
            natural_breaks = [m.start() for m in re.finditer(r'\.[\s\n]', text[search_start:end])]
            
            if natural_breaks:
                # Use the last natural break found
                end = search_start + natural_breaks[-1] + 2  # +2 to include the period and space/newline
        
        # Get the chunk and add it to the list
        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
        
        # Move the start position for the next chunk, considering overlap
        start = end - overlap if end < text_len else text_len
    
    return chunks

# ---- 1. Semantic Chunking ----

def semantic_chunking(text: str, model: str = "llama2", chunk_size: int = 1000, 
                      min_chunk_size: int = 200, max_chunks: int = 20) -> List[str]:
    """
    Chunk text based on semantic meaning using an LLM.
    
    Args:
        text: The input text to chunk
        model: Ollama model to use
        chunk_size: Target size of each chunk in characters
        min_chunk_size: Minimum size of a chunk
        max_chunks: Maximum number of chunks to create
        
    Returns:
        List of semantically coherent text chunks
    """
    # For very short texts, don't bother chunking
    if len(text) < chunk_size:
        return [text]
    
    # For longer texts, we'll first create rough chunks as a starting point
    initial_chunks = simple_chunking(text, chunk_size=chunk_size*2, overlap=chunk_size//2)
    
    if len(initial_chunks) <= 1:
        return initial_chunks
    
    semantic_chunks = []
    
    # Process each initial chunk to find semantic boundaries
    for i, chunk in enumerate(initial_chunks):
        if len(chunk) < min_chunk_size:
            # If chunk is too small, append it to the previous chunk or create a new one
            if semantic_chunks:
                semantic_chunks[-1] += " " + chunk
            else:
                semantic_chunks.append(chunk)
            continue
            
        # Use LLM to identify semantic sections in the chunk
        prompt = f"""
        Split the following text into {min(3, max(1, chunk_size // 500))} coherent semantic sections. 
        Each section should be self-contained and focus on a specific topic, concept, or point.
        Don't summarize, just identify where one semantic section ends and another begins.
        Mark section breaks with [SECTION_BREAK].

        TEXT:
        {chunk}
        """
        
        response = query_ollama(prompt, model=model)
        
        # Process the response to extract sections
        if "[SECTION_BREAK]" in response:
            sections = [s.strip() for s in response.split("[SECTION_BREAK]") if s.strip()]
            semantic_chunks.extend(sections)
        else:
            # If no section breaks, use the original chunk
            semantic_chunks.append(chunk)
    
    # Ensure we don't have too many chunks
    if len(semantic_chunks) > max_chunks:
        # Combine smaller chunks until we're under the limit
        semantic_chunks.sort(key=len)  # Sort by length
        while len(semantic_chunks) > max_chunks:
            # Combine the two smallest chunks
            smallest = semantic_chunks.pop(0)
            second_smallest = semantic_chunks.pop(0)
            combined = smallest + " " + second_smallest
            semantic_chunks.append(combined)
            semantic_chunks.sort(key=len)  # Re-sort
    
    # Final cleanup and length check
    result = []
    for chunk in semantic_chunks:
        # Split very large chunks if necessary
        if len(chunk) > chunk_size * 1.5:
            result.extend(simple_chunking(chunk, chunk_size, chunk_size//4))
        else:
            result.append(chunk)
    
    return result

# ---- 2. Recursive Chunking ----

def recursive_chunking(text: str, chunk_size: int = 1000, overlap: int = 200, 
                      min_chunk_size: int = 200) -> List[Dict[str, Any]]:
    """
    Recursively chunk text to create a hierarchical document structure.
    
    Args:
        text: The input text to chunk
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        min_chunk_size: Minimum size for a chunk to be considered
    
    Returns:
        A list of chunk dictionaries with hierarchical structure
    """
    # Simple case - text is short enough for a single chunk
    if len(text) <= chunk_size:
        return [{"text": text, "children": []}]
    
    # Split text into sections based on headers or natural breaks
    sections = []
    
    # Try to split by headers (markdown style)
    header_pattern = r'(?m)^(#{1,3})\s+(.+)$'
    header_matches = list(re.finditer(header_pattern, text))
    
    if header_matches:
        # Process document with headers
        current_pos = 0
        current_hierarchy = []
        
        for i, match in enumerate(header_matches):
            header_level = len(match.group(1))
            header_text = match.group(2).strip()
            section_start = match.start()
            
            # If this isn't the first header, save the previous section
            if current_pos < section_start and section_start - current_pos >= min_chunk_size:
                section_text = text[current_pos:section_start].strip()
                if section_text:
                    sections.append({
                        "level": len(current_hierarchy),
                        "title": current_hierarchy[-1] if current_hierarchy else "Introduction",
                        "text": section_text
                    })
            
            # Update the hierarchy based on this header's level
            if header_level == 1:
                current_hierarchy = [header_text]
            elif header_level == 2:
                if not current_hierarchy:
                    current_hierarchy = ["Document", header_text]
                else:
                    current_hierarchy = current_hierarchy[:1] + [header_text]
            elif header_level == 3:
                if len(current_hierarchy) < 2:
                    if not current_hierarchy:
                        current_hierarchy = ["Document", "Section", header_text]
                    else:
                        current_hierarchy = current_hierarchy + ["Section", header_text]
                else:
                    current_hierarchy = current_hierarchy[:2] + [header_text]
            
            # Move position to after this header
            current_pos = match.end()
            
            # If this is the last header, capture text until the end
            if i == len(header_matches) - 1 and current_pos < len(text):
                section_text = text[current_pos:].strip()
                if section_text:
                    sections.append({
                        "level": len(current_hierarchy),
                        "title": header_text,
                        "text": section_text
                    })
    
    # If no headers were found or sections weren't created, fall back to paragraph-based chunking
    if not sections:
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_chunk = ""
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            if len(current_chunk) + len(paragraph) + 1 <= chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                if current_chunk:
                    sections.append({
                        "level": 1,
                        "title": "Section",
                        "text": current_chunk
                    })
                current_chunk = paragraph
        
        if current_chunk:
            sections.append({
                "level": 1,
                "title": "Section",
                "text": current_chunk
            })
    
    # Create hierarchical structure
    result = []
    section_map = defaultdict(list)
    
    # First pass: group sections by their level
    for section in sections:
        level = section["level"]
        section_map[level].append(section)
    
    # Second pass: build hierarchy (simplified approach)
    for level_1_section in section_map.get(1, []):
        child_sections = []
        
        # Find child sections for this level 1 section
        l1_title = level_1_section["title"]
        for level_2_section in section_map.get(2, []):
            if True:  # This is simplified; ideally we'd check parent-child relationships
                level_2_data = {
                    "text": level_2_section["text"], 
                    "title": level_2_section["title"],
                    "children": []
                }
                
                # Find child sections for this level 2 section
                l2_title = level_2_section["title"]
                for level_3_section in section_map.get(3, []):
                    if True:  # Simplified
                        level_3_data = {
                            "text": level_3_section["text"],
                            "title": level_3_section["title"],
                            "children": []
                        }
                        level_2_data["children"].append(level_3_data)
                
                child_sections.append(level_2_data)
        
        # Add the level 1 section with its children
        result.append({
            "text": level_1_section["text"],
            "title": l1_title,
            "children": child_sections
        })
    
    # If no hierarchy was created, return flat chunks
    if not result:
        chunks = simple_chunking(text, chunk_size, overlap)
        result = [{"text": chunk, "children": []} for chunk in chunks]
    
    return result

# ---- 3. Agentic Chunking ----

def agentic_chunking(text: str, model: str = "llama2", chunk_size: int = 1000, 
                    max_chunks: int = 20) -> List[Dict[str, Any]]:
    """
    Use an LLM agent to intelligently chunk text based on content understanding.
    
    Args:
        text: The input text to chunk
        model: Ollama model to use
        chunk_size: Target size for chunks in characters
        max_chunks: Maximum number of chunks to create
    
    Returns:
        List of chunk dictionaries with metadata
    """
    # For short texts, don't bother with complex chunking
    if len(text) < chunk_size:
        return [{"text": text, "summary": text[:100] + "...", "keywords": []}]
    
    # Step 1: Initial chunking to get manageable text segments
    initial_chunks = simple_chunking(text, chunk_size=chunk_size*2, overlap=chunk_size//4)
    
    result_chunks = []
    
    # Step 2: For each initial chunk, use LLM to analyze and optimize chunking
    for i, chunk in enumerate(initial_chunks):
        # Skip if too short
        if len(chunk) < 100:
            continue
            
        # 2a. Ask LLM to determine if this chunk needs further splitting
        analysis_prompt = f"""
        Analyze the following text segment and determine if it contains multiple distinct topics or concepts.
        If it does, identify where the natural breaks occur.
        
        TEXT:
        {chunk[:2000]}  # Limit to first 2000 chars for analysis
        
        Respond in the following JSON format:
        {{
            "needs_splitting": true/false,
            "reason": "brief explanation",
            "suggested_splits": [
                "index of first character of where split should occur",
                "..."
            ]
        }}
        """
        
        analysis_response = query_ollama(analysis_prompt, model=model)
        
        # Extract JSON from response (handle case where LLM adds extra text)
        try:
            json_match = re.search(r'\{.*\}', analysis_response, re.DOTALL)
            if json_match:
                analysis_json = json.loads(json_match.group(0))
            else:
                analysis_json = {"needs_splitting": False}
        except:
            analysis_json = {"needs_splitting": False}
        
        # 2b. Split if needed, otherwise use as is
        if analysis_json.get("needs_splitting", False) and "suggested_splits" in analysis_json:
            try:
                splits = [int(idx) for idx in analysis_json["suggested_splits"] if idx.isdigit()]
                splits = [s for s in splits if 0 < s < len(chunk)]
                
                if splits:
                    # Create sub-chunks based on suggested splits
                    splits.sort()
                    splits = [0] + splits + [len(chunk)]
                    
                    for j in range(len(splits) - 1):
                        sub_chunk = chunk[splits[j]:splits[j+1]].strip()
                        if sub_chunk and len(sub_chunk) >= 100:
                            result_chunks.append({"text": sub_chunk, "summary": "", "keywords": []})
                    continue
            except:
                pass
        
        # If we didn't split, use the original chunk
        result_chunks.append({"text": chunk, "summary": "", "keywords": []})
    
    # Step 3: Enhance chunks with metadata using the LLM
    for i, chunk_dict in enumerate(result_chunks):
        chunk = chunk_dict["text"]
        
        # Skip metadata generation for very short chunks
        if len(chunk) < 200:
            chunk_dict["summary"] = chunk
            chunk_dict["keywords"] = []
            continue
            
        metadata_prompt = f"""
        For the following text segment, provide:
        1. A concise summary (max 2 sentences)
        2. A list of 3-5 keywords that best represent the content
        
        TEXT:
        {chunk[:1500]}  # Limit to first 1500 chars for analysis
        
        Respond in JSON format:
        {{
            "summary": "concise summary here",
            "keywords": ["keyword1", "keyword2", "keyword3"]
        }}
        """
        
        metadata_response = query_ollama(metadata_prompt, model=model)
        
        # Extract JSON from response
        try:
            json_match = re.search(r'\{.*\}', metadata_response, re.DOTALL)
            if json_match:
                metadata_json = json.loads(json_match.group(0))
                chunk_dict["summary"] = metadata_json.get("summary", "")
                chunk_dict["keywords"] = metadata_json.get("keywords", [])
            else:
                # Fallback if JSON parsing fails
                chunk_dict["summary"] = chunk[:100] + "..."
                chunk_dict["keywords"] = []
        except:
            chunk_dict["summary"] = chunk[:100] + "..."
            chunk_dict["keywords"] = []
    
    # Step 4: Ensure we don't exceed max_chunks by combining similar chunks if needed
    if len(result_chunks) > max_chunks:
        # Get embeddings for each chunk to measure similarity
        chunk_embeddings = []
        for chunk_dict in result_chunks:
            embedding = get_embeddings(chunk_dict["text"])
            if embedding:
                chunk_embeddings.append(embedding)
            else:
                # If embedding fails, use a random vector
                chunk_embeddings.append([0] * 384)  # Default embedding size
        
        # Calculate similarity matrix
        similarities = []
        for i in range(len(result_chunks)):
            for j in range(i+1, len(result_chunks)):
                emb_i = chunk_embeddings[i]
                emb_j = chunk_embeddings[j]
                
                # Calculate cosine similarity
                dot_product = sum(a*b for a, b in zip(emb_i, emb_j))
                norm_i = sum(a*a for a in emb_i) ** 0.5
                norm_j = sum(a*a for a in emb_j) ** 0.5
                
                if norm_i > 0 and norm_j > 0:
                    similarity = dot_product / (norm_i * norm_j)
                else:
                    similarity = 0
                
                similarities.append((i, j, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[2], reverse=True)
        
        # Merge chunks until we're under the limit
        merged = [False] * len(result_chunks)
        while len(result_chunks) - sum(merged) > max_chunks and similarities:
            i, j, _ = similarities.pop(0)
            
            if not merged[i] and not merged[j]:
                # Merge chunks i and j
                merged[j] = True
                
                # Combine text and metadata
                combined_text = result_chunks[i]["text"] + "\n\n" + result_chunks[j]["text"]
                combined_keywords = list(set(result_chunks[i]["keywords"] + result_chunks[j]["keywords"]))
                
                result_chunks[i]["text"] = combined_text
                result_chunks[i]["keywords"] = combined_keywords[:5]  # Limit to 5 keywords
                
                # Generate a new summary for the combined chunk
                if len(combined_text) > 200:
                    summary_prompt = f"""
                    Provide a concise summary (max 2 sentences) for the following text:
                    
                    {combined_text[:1000]}
                    """
                    result_chunks[i]["summary"] = query_ollama(summary_prompt, model=model)[:150]
                else:
                    result_chunks[i]["summary"] = combined_text[:100] + "..."
        
        # Remove merged chunks
        result_chunks = [chunk for i, chunk in enumerate(result_chunks) if not merged[i]]
    
    return result_chunks

# ---- Usage Demo ----

def demo_all_chunking_methods(text: str, model: str = "llama2"):
    """Demonstrate all chunking methods on a sample text."""
    print("===== CHUNKING DEMO =====")
    print(f"Input text length: {len(text)} characters")
    
    # 1. Semantic Chunking
    print("\n\n--- SEMANTIC CHUNKING ---")
    semantic_chunks = semantic_chunking(text, model=model)
    print(f"Generated {len(semantic_chunks)} semantic chunks")
    for i, chunk in enumerate(semantic_chunks):
        print(f"Chunk {i+1}: {len(chunk)} characters")
        print(f"Preview: {chunk[:100]}...")
    
    # 2. Recursive Chunking
    print("\n\n--- RECURSIVE CHUNKING ---")
    recursive_chunks = recursive_chunking(text)
    print(f"Generated {len(recursive_chunks)} recursive chunk trees")
    for i, chunk in enumerate(recursive_chunks):
        print(f"Chunk {i+1}: {len(chunk['text'])} characters, {len(chunk['children'])} children")
        print(f"Title: {chunk.get('title', 'No title')}")
        print(f"Preview: {chunk['text'][:100]}...")
    
    # 3. Agentic Chunking
    print("\n\n--- AGENTIC CHUNKING ---")
    agentic_chunks = agentic_chunking(text, model=model)
    print(f"Generated {len(agentic_chunks)} agentic chunks")
    for i, chunk in enumerate(agentic_chunks):
        print(f"Chunk {i+1}: {len(chunk['text'])} characters")
        print(f"Summary: {chunk['summary']}")
        print(f"Keywords: {', '.join(chunk['keywords'])}")
        print(f"Preview: {chunk['text'][:100]}...")

# Example usage
if __name__ == "__main__":
    sample_text = """
    # Introduction to RAG Systems
    
    Retrieval-Augmented Generation (RAG) is a technique that combines retrieval-based and generation-based approaches in natural language processing. RAG systems enhance large language models by giving them access to external knowledge.
    
    ## How RAG Works
    
    RAG works by first retrieving relevant documents or passages from a knowledge base and then using these documents to augment the input to a language model. This approach has several advantages:
    
    1. It reduces hallucination by grounding the model's generation in retrieved facts
    2. It allows the model to access information beyond its training data
    3. It enables more transparent sourcing of information
    
    ### Components of a RAG System
    
    A typical RAG system consists of:
    
    - A document store or vector database
    - An embedding model to convert text into vectors
    - A retrieval mechanism to find relevant documents
    - A language model to generate responses
    
    ## Chunking Strategies
    
    Effective document chunking is critical for RAG performance. Different chunking strategies include:
    
    1. Simple chunking by character or token count
    2. Semantic chunking based on meaning
    3. Recursive chunking that preserves document hierarchy
    4. Agentic chunking that uses an LLM to determine optimal chunks
    
    The choice of chunking strategy depends on the nature of your documents and the specific requirements of your application.
    """
    
    # Set your preferred Ollama model
    llm_model = "llama2"  # Change to your available model
    
    # Demo all chunking methods
    demo_all_chunking_methods(sample_text, model=llm_model)