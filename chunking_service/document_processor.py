import os
from pathlib import Path
import re
from datetime import datetime

def load_zhihu_documents(directory: str) -> list:
    """
    Load all Zhihu answer TXT files from the specified directory.
    
    Args:
        directory: Path to directory containing .txt files or a specific .txt file
        
    Returns:
        List of document objects with metadata and content
    """
    documents = []
    path = Path(directory)
    
    # If it's a file, process just that file
    if path.is_file():
        if path.suffix.lower() == '.txt':
            files_to_process = [path]
        else:
            print(f"File {path} is not a .txt file")
            return []
    # If it's a directory, find all .txt files
    elif path.is_dir():
        files_to_process = list(path.glob("*.txt"))
        if not files_to_process:
            print(f"No .txt files found in directory {path}")
    else:
        print(f"Path {path} does not exist")
        return []
    
    for file_path in files_to_process:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                if not content.strip():
                    print(f"Warning: {file_path} is empty")
                    continue
                
                documents.append({
                    'id': file_path.stem,
                    'content': content,
                    'source': str(file_path),
                    'timestamp': datetime.now().isoformat(),
                    'filename': file_path.name
                })
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    print(f"Loaded {len(documents)} documents")
    return documents

def split_into_chunks(text: str, max_chars: int = 500, overlap_ratio: float = 0.3) -> list:
    """
    Split text into semantically meaningful chunks with overlap.
    
    Args:
        text: Input text to chunk
        max_chars: Maximum characters per chunk (approx 512 tokens for Chinese)
        overlap_ratio: Percentage of overlap between chunks
        
    Returns:
        List of text chunks
    """
    # First try to split into paragraphs (using empty lines as separators)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    # If no paragraphs found, try splitting by sentence endings
    if len(paragraphs) <= 1:
        sentences = re.split(r'(?<=[。！？.!?])\s+', text)
        paragraphs = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = ""
    overlap_size = int(max_chars * overlap_ratio)
    
    for para in paragraphs:
        # Ensure each paragraph can fit in a chunk
        if len(para) > max_chars:
            # If paragraph is too long, split it
            words = para.split()
            temp_chunk = ""
            for word in words:
                if len(temp_chunk) + len(word) + 1 > max_chars:
                    if temp_chunk:
                        chunks.append(temp_chunk)
                        # Create overlap
                        temp_chunk = temp_chunk[-overlap_size:] + " " + word
                    else:
                        # Single word longer than max_chars
                        chunks.append(word)
                else:
                    temp_chunk += (" " + word if temp_chunk else word)
            if temp_chunk:
                current_chunk = temp_chunk
        else:
            # If adding this paragraph exceeds max length
            if len(current_chunk) + len(para) > max_chars and current_chunk:
                chunks.append(current_chunk)
                # Create overlap by carrying over part of current chunk
                current_chunk = current_chunk[-overlap_size:] + " " + para
            else:
                current_chunk += ("\n" + para if current_chunk else para)
    
    # Add any remaining content as final chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    # Final check: ensure no chunk exceeds max length
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_chars:
            # Split oversized chunk into smaller ones
            for i in range(0, len(chunk), max_chars):
                final_chunks.append(chunk[i:i+max_chars])
        else:
            final_chunks.append(chunk)
    
    return final_chunks

def process_documents(directory: str) -> list:
    """
    Process all documents in directory: load and chunk them.
    
    Args:
        directory: Path to directory with source documents or a specific .txt file
        
    Returns:
        List of chunked document objects with metadata
    """
    documents = load_zhihu_documents(directory)
    processed = []
    for doc in documents:
        chunks = split_into_chunks(doc['content'])
        for i, chunk in enumerate(chunks):
            processed.append({
                'doc_id': f"{doc['id']}_chunk_{i}",
                'content': chunk,
                'source': doc['source'],
                'original_id': doc['id'],
                'chunk_index': i,
                'total_chunks': len(chunks)
            })
    
    print(f"Processed {len(processed)} chunks from {len(documents)} documents\n")
    return processed

# Example usage
if __name__ == "__main__":
    processed_docs = process_documents("./docs/test.txt") # test.txt is a sample file for testing
    print(f"Total chunks created: {len(processed_docs)}")
    print(processed_docs)  # Print first chunk for verification