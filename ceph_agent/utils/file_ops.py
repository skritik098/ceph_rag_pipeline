from sentence_transformers import SentenceTransformer
import faiss
import json
import os

from collections import defaultdict
from langchain.text_splitter import RecursiveCharacterTextSplitter


# For Simplification of this parameter growth issue,
# let's use the concept of Class

# --------------------
# Phase 1: Build Index
# --------------------
# --------------------
# Phase 2: Load Index
# --------------------

class vectorBuilder:
    """
    A Class to encapsulate the Ceph Command Vector Store & it's search logic
    """
    def __init__(self, json_path, model_name, index_path, metadata_path) -> None:
        # Here we need to declare them only once
        # Later function we can directly access them
        # without passing them in functions.
        self.json_path = json_path
        self.model_name = model_name
        self.index_path = index_path
        self.metadata_path = metadata_path

        self.index, self.metadata, self.model = self._load_index()

    # Loading the VectorDB, & if not created create ONE
    def _load_index(self):
        print("Validating index existence...")
        print("--------------------------------")
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            model = SentenceTransformer(self.model_name)
            print("🔁 Loading existing FAISS index and query mapping...")
            index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "rb") as f:
                data = json.load(f)
        else:
            print("⚙️ Building new FAISS index...")
            model = self._build_index_combined(self.json_path)
            index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "rb") as f:
                data = json.load(f)
        return index, data, model

    # Build Vector DB Combined Intent & Description
    def _build_index_combined(self):
        with open(self.json_path) as f:
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
        model = SentenceTransformer(self.model_name)
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        dimension = embeddings[0].shape[0]
        # Here we are using L2 FAISS embedding
        # index = faiss.IndexFlatL2(dimension)

        # Here we are using Cosine Similarity
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        faiss.write_index(index, self.index_path)
        with open(self.metadata_path, "w") as f:
            json.dump(combined_metadata, f)

        print("✅ FAISS index and grouped metadata saved.")
        return model

    # Building & loading Index with Chunky Vectorization
    def _build_index_chunky(
        self,
        chunk_size: int = 300,
        chunk_overlap: int = 50
    ):
        # Load data
        with open(self.json_path, "r") as f:
            data = json.load(f)

        # Group entries by command
        grouped = defaultdict(list)
        for entry in data:
            grouped[entry["command"]].append(
                f"{entry['query_intent']} | {entry['description']}"
            )

        # Initialize model
        model = SentenceTransformer(self.model)
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
        faiss.write_index(index, self.index_path)
        with open(self.metadata_path, "w") as f:
            json.dump(data, f)
        print("✅ FAISS index and metadata saved.")
        return model
