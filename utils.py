from langchain_community.vectorstores import Chroma
import chromadb
import ollama
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.llms.ollama import Ollama
from langchain.chains import RetrievalQA
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


class RAGPipeline:
    def __init__(
        self,
        e_model_name,
        e_model_kwargs,
        ollama_url,
        ollama_model,
        chroma_client,
        chroma_collection_name,
    ):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=e_model_name, model_kwargs=e_model_kwargs
        )
        self.ollama_llm = Ollama(base_url=ollama_url, model=ollama_model)
        self.db_retriever = Chroma(
            client=chroma_client,
            collection_name=chroma_collection_name,
            embedding_function=self.embeddings,
        )
        self.rag = RetrievalQA.from_chain_type(
            self.ollama_llm,
            retriever=self.db_retriever.as_retriever(),
            return_source_documents=True,
        )

    def build_prompt(self, data):
        return """
           You are an expert AI Assistant on blockchain smartcontracts. I will give you a smart contract example written in Solidity and your task will be to an assessment on the contract to identify pottential issues and to teach me how to correct them.
           Here is the contract:
           {}  
        """.format(
            data
        )

    def invoke(self, contract):
        return self.rag.invoke({"query": self.build_prompt(contract)})
