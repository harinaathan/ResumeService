import hashlib
from langchain_community.document_loaders import TextLoader                                         
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from logger import logger


class SentenceTransformerEmbeddings:
    """Wraps the Sentence Transformer library to embed text and maintain compatibility with LangChain
Takes a list of text strings and returns their embeddings (embed_query)
or Takes a single text string and returns its embedding (embed_documents)"""
    def __init__(self, model_name="all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        logger("embedding documents")
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def embed_query(self, text):
        logger("embed a single query")
        return self.model.encode([text], convert_to_tensor=False).tolist()[0]
    

class dbManager:
    """class to create and/or manage a local database"""
    def __init__(self, path, name):
        # store name to instance
        self.name = name

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
        logger("refreshing db hashes", self.name)
        self.existing_hashes = set()
        for doc in self.db.get()['metadatas']:
            if doc and 'hash' in doc:
                self.existing_hashes.add(doc['hash'])
    
    def update(self,document):
        """add a data to db"""
        logger("adding data to db", self.name)
        # split document
        textMaster = self.text_splitter.split_documents(document)
        # check document duplication before updating
        new_hashes = set()
        new_docs = []
        new_metadatas = []
        for doc in textMaster:
            doc_hash = hashlib.sha256(doc.page_content.encode()).hexdigest()
            if (doc_hash not in self.existing_hashes) and (doc_hash not in new_hashes):
                new_docs.append(doc)
                new_hashes.add(doc_hash)
                new_metadatas.append({"hash":doc_hash})
        
        if new_docs:
            self.db.add_documents(new_docs, metadatas=new_metadatas)
            self.refreshHashes()
    
    def updateBulk(self,fileList):
        """add multiple data to db"""
        logger("adding multiple data to db", self.name)
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

        # check document duplication before updating
        new_hashes = set()
        new_docs = []
        new_metadatas = []
        for doc in textMaster:
            doc_hash = hashlib.sha256(doc.page_content.encode()).hexdigest()
            if (doc_hash not in self.existing_hashes) and (doc_hash not in new_hashes):
                new_docs.append(doc)
                new_hashes.add(doc_hash)
                new_metadatas.append({"hash":doc_hash})
        
        print(len(new_docs))
        print("="*25)
        print(len(new_metadatas))

        if new_docs:
            self.db.add_documents(new_docs, metadatas=new_metadatas)
            self.refreshHashes()

    def as_retriever(self):
        return self.db.as_retriever()
    
    def search(self, query, k=4):
        """Searches the database for documents matching the query."""
        return self.db.search(query, k=k)

    def similarity_search(self, query, k=4):
        """Performs similarity search for documents matching the query."""
        return self.db.similarity_search(query, k=k)
    
    def delete(self, doc_ids):
        """Deletes documents by their IDs."""
        self.db.delete(doc_ids=doc_ids)
        # update hashes after deletion.
        self.refreshHashes()

    def delete_by_hash(self, doc_hash):
        """Deletes documents by their hash."""
        result = self.db.get()
        ids_to_delete = []
        for i, metadata in enumerate(result['metadatas']):
            if metadata and 'hash' in metadata and metadata['hash'] == doc_hash:
                ids_to_delete.append(result['ids'][i])
        if ids_to_delete:
            self.delete(ids_to_delete)        

    def persist(self):
        """Persists the database to disk."""
        self.db.persist()

    def reset(self):
        """Resets the database by deleting all documents."""
        self.db.delete(self.db.get()['ids'])
        self.refreshHashes()