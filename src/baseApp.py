"""first cript to use docker style"""


class myApp():
    """define basinc structure of the RAG model"""
    def __init__(self):
        pass

    def loadRef(self):
        """load reference documents to learn subjective queries"""
        print("load documents...")
    
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