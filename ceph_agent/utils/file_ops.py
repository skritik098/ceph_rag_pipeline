from sentence_transformers import SentenceTransformer
import faiss
import json
import os
#import numpy as np

from collections import defaultdict
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --------------------
# Phase 1: Build Index
# --------------------


def build_index(
    json_path,
    model_name="all-MiniLM-L6-v2",
    index_path="./faiss_index_store/ceph_faiss.index",
    metadata_path="./faiss_index_store/ceph_faiss_metadata.json"
):
    with open(json_path) as f:
        data = json.load(f)

    model = SentenceTransformer(model_name)
    texts = [f"{entry['query_intent']} | {entry['description']}" for entry in data]
    embeddings = model.encode(texts, show_progress_bar=True)

    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, index_path)
    with open(metadata_path, "w") as f:
        json.dump(data, f)
    
    print("✅ FAISS index and metadata saved.")
    return model

# --------------------
# Phase 2: Load Index
# --------------------


def load_index(
    json_path="./database/basic_commands.json",
    model_name="all-MiniLM-L6-v2",
    index_path="./faiss_index_store/ceph_faiss.index",
    metadata_path="./faiss_index_store/ceph_faiss_metadata.json"
):
    print("Validating index existence...")
    print("--------------------------------")
    if os.path.exists(index_path) and os.path.exists(metadata_path):
        model = SentenceTransformer(model_name)
        print("🔁 Loading existing FAISS index and query mapping...")
        index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            data = json.load(f)
        
    else:
        print("⚙️ Building new FAISS index...")
        #model = build_index(json_path) # Here using the normal build without grouping
        #model = build_index_chunky(json_path)  # Trying to use chunky build method with grouping
        model = build_index_combined(json_path)
        index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            data = json.load(f)
    return index, data, model


# Building Index with combined Intent & Description

def build_index_combined(
    json_path,
    model_name="all-MiniLM-L6-v2",
    index_path="./faiss_index_store/ceph_faiss.index",
    metadata_path="./faiss_index_store/ceph_faiss_metadata.json"
):
    with open(json_path) as f:
        data = json.load(f)

    # Group by command
    grouped = {}
    for entry in data:
        cmd = entry["command"]
        if cmd not in grouped:
            grouped[cmd] = {
                "command": cmd,
                "query_intent": [],
                "description": []
            }
        grouped[cmd]["query_intent"].append(entry["query_intent"])
        grouped[cmd]["description"].append(entry["description"])

    # Prepare texts and metadata
    combined_metadata = []
    texts = []

    for group in grouped.values():
        joined_intent = " | ".join(group["query_intent"])
        joined_desc = " | ".join(group["description"])
        combined_text = f"{joined_intent} | {joined_desc}"
        texts.append(combined_text)

        combined_metadata.append({
            "command": group["command"],
            "query_intent": joined_intent,
            "description": joined_desc
        })

    # Embedding & indexing
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    dimension = embeddings[0].shape[0]
    #index = faiss.IndexFlatL2(dimension) # Here we are using L2 FAISS embedding
    index = faiss.IndexFlatIP(dimension)  # Here we are using Cosine Similarity
    index.add(embeddings)

    faiss.write_index(index, index_path)
    with open(metadata_path, "w") as f:
        json.dump(combined_metadata, f)

    print("✅ FAISS index and grouped metadata saved.")
    return model

# Building & loading Index with Chunky Vectorization


def build_index_chunky(
    json_path: str,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
    model="all-MiniLM-L6-v2",
    index_path="./faiss_index_store/ceph_faiss.index",
    metadata_path="./faiss_index_store/ceph_faiss_metadata.json"
):
    # Load data
    with open(json_path, "r") as f:
        data = json.load(f)

    # Group entries by command
    grouped = defaultdict(list)
    for entry in data:
        grouped[entry["command"]].append(
            f"{entry['query_intent']} | {entry['description']}"
        )

    # Initialize model
    model = SentenceTransformer(model)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    # Chunk and embed
    all_chunks = []
    metadata = {}
    id_counter = 0
    for command, combined_entries in grouped.items():
        full_text = "\n".join(combined_entries)
        chunks = text_splitter.split_text(full_text)

        for chunk in chunks:
            all_chunks.append(chunk)
            metadata[id_counter] = {
                "command": command,
                "chunk": chunk,
            }
            id_counter += 1

    embeddings = model.encode(all_chunks, show_progress_bar=True)
    dimension = embeddings[0].shape[0]

    # Build index
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save index & metadata
    faiss.write_index(index, index_path)
    with open(metadata_path, "w") as f:
        json.dump(data, f)
    print("✅ FAISS index and metadata saved.")
    return model

'''
def prepare_chunked_data(data: List[Dict], chunk_size: int = 300, chunk_overlap: int = 50) -> List[Dict]:
    """
    Groups entries by command, combines their descriptions and intents,
    then chunks the combined text and attaches metadata.
    
    Returns a list of dicts where each dict represents a chunk with:
    - 'chunk_text': the actual text
    - 'command': the original CLI command
    """
    grouped = defaultdict(list)

    # Step 1: Group all descriptions and intents by command
    for item in data:
        cmd = item['command']
        grouped[cmd].append(f"{item['description']} (Intent: {item['query_intent']})")

    chunked_data = []

    # Step 2: For each command, join and chunk
    for cmd, entries in grouped.items():
        combined_text = " ".join(entries)

        # Simple overlapping chunking
        for i in range(0, len(combined_text), chunk_size - chunk_overlap):
            chunk = combined_text[i:i + chunk_size]
            chunked_data.append({
                'chunk_text': chunk.strip(),
                'command': cmd
            })

    return chunked_data
'''