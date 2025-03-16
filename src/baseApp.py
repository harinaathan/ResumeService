"""Improve your job search by automating response to job applications
with a RAG at core, this tool provides responses precisely based on your profile
throw your profile information to the tool, and it will speed up your responses to the job applications
beyond speeding up, a gramatically correct information, tailored to match the current job description is made available at your finger tips"""


import os
import hashlib
import google.generativeai as genai
from langchain.document_loaders import TextLoader                                           
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddings:
    """Wraps the Sentence Transformer library to embed text and maintain compatibility with LangChain
Takes a list of text strings and returns their embeddings (embed_query)
or Takes a single text string and returns its embedding (embed_documents)"""
    def __init__(self, model_name="all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_tensor=False).tolist()[0]
    

class dbManager:
    """class to create and/or manage a local database"""
    def __init__(self, path):
        # splitter
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        # embeddings handler
        self.embeddings = SentenceTransformerEmbeddings("all-mpnet-base-v2")

        # Load existing DB or create a blank DB
        self.db = Chroma(persist_directory=path,
                         embedding_function=self.embeddings)
        
        # update hashes
        self.refreshHashes()
        
    def refreshHashes(self):
        """Get existing hashes (if any)"""
        self.existing_hashes = set()
        for doc in self.db.get()['metadatas']:
            if doc and 'hash' in doc:
                self.existing_hashes.add(doc['hash'])
    
    def update(self,doc):
        doc_hash = hashlib.sha256(doc.page_content.encode()).hexdigest()
        if doc_hash not in self.existing_hashes:
            self.db.add_documents(
                [doc], 
                metadatas=[{'hash': doc_hash}])
    
    def updateBulk(self,fileList):
        textMaster = []
        # process all docs
        for file in fileList:
            try:
                doc = TextLoader(file).load()
                splitDocs = self.text_splitter.split_documents(doc)
                textMaster.extend(splitDocs)
            except FileNotFoundError:
                print(f"Error: File not found - {file}")
            except Exception as e:
                print(f"Error loading file {file}: {e}")

        for doc in textMaster:
            self.update(doc)        

    def as_retriever(self):
        return self.db.retriever()


class myApp():
    """define basinc structure of the RAG model"""
    def __init__(self):

        # database path
        self.knowledgeBasePath = os.path.join(os.getcwd(), "knowledgeBase")
        self.sessionInformationPath = os.path.join(os.getcwd(), "sessionInfo")

        # choice of LLM
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")

    def buildKnowledgeBase(self, fileList):
        """load reference documents to learn subjective queries"""

        textMaster = []
        # process all docs
        for file in fileList:
            doc = TextLoader(file).load()
            splitDocs = self.text_splitter.split_documents(doc)
            textMaster.extend(splitDocs)

        self.knowledgeBase = Chroma.from_documents(
            textMaster,
            self.embeddings,
            persist_directory=self.knowledgeBasePath
            )

        print("Documents loaded...")
    
    def embed(self, query):
        """indexing, embedding & storing to vector database"""
        print("embedding phase...")

    def retreiveRef(self, query):
        """retreive most similar documents from documents"""
        print("fetch nearest document embedding")

    def responseGeneration(self, query):
        """connect to gemini API and generate response"""
        print("generative responses")

if __name__ == "__main__":

    app = myApp()
    app.responseGeneration("What is your years of experience as a Data Scientist?")