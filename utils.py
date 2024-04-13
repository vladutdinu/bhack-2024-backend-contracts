import ollama
import chromadb
import ollama
from tokenizers import Tokenizer
from semantic_text_splitter import HuggingFaceTextSplitter
import fitz
import uuid

class OllamaClient:
    def __init__(self, host, model):
        self.ollama_client = ollama.Client(host)
        self.model = model

    def generate(self, text, context):
        return self.ollama_client.generate(model=self.model, prompt=text, context=context)

class ChromaClient:
    def __init__(self, local_path):
        self.chroma_client = chromadb.PersistentClient(path=local_path)


class Chunker:
    def __init__(self, embeddings_model, max_tokens):
        self.tokenizer = Tokenizer.from_pretrained(embeddings_model)
        self.splitter = HuggingFaceTextSplitter(self.tokenizer, trim_chunks=True)
        self.max_tokens = max_tokens

    def chunk_it(self, text):
        chunks = self.splitter.chunks(text, self.max_tokens)
        return chunks

    def ingest(
        self,
        text,
        filename,
        file_type,
        ollama_host,
        ollama_embeddings_model,
        chroma_local_path,
        chroma_collection_name,
    ):
        chunks = None
        ollama_client = OllamaClient(host=ollama_host, model=None).ollama_client
        chroma_client = ChromaClient(
            local_path=chroma_local_path
        ).chroma_client.get_or_create_collection(chroma_collection_name)
        if file_type == "PDF":
            pdf = fitz.Document(stream=text, filetype="pdf")
            data = ""
            for page in range(0, pdf.page_count):
                data = data + pdf[page].get_text()
            chunks = self.chunk_it(data)
            for _, chunk in enumerate(chunks):
                id = uuid.uuid4().hex
                embed = ollama_client.embeddings(
                    model=ollama_embeddings_model, prompt=chunk
                )["embedding"]
                chroma_client.add(
                    [id], [embed], documents=[chunk], metadatas={"source": filename}
                )
        elif file_type == "URL":
            chunks = self.chunk_it(text)
            for _, chunk in enumerate(chunks):
                id = uuid.uuid4().hex
                embed = ollama_client.embeddings(
                    model=ollama_embeddings_model, prompt=chunk
                )["embedding"]
                chroma_client.add(
                    [id], [embed], documents=[chunk], metadatas={"source": filename}
                )        