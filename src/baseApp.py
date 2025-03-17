"""Improve your job search by automating response to job applications
with a RAG at core, this tool provides responses precisely based on your profile
throw your profile information to the tool, and it will speed up your responses to the job applications
beyond speeding up, a gramatically correct information, tailored to match the current job description is made available at your finger tips"""


import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from dbManager import dbManager
from logger import logger

# Configure Gemini API
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class resumeWiki():
    """define basinc structure of the RAG model"""
    def __init__(self):

        # database path
        knowledgeBasePath = os.path.join(os.getcwd(), "knowledgeBase")
        sessionInformationPath = os.path.join(os.getcwd(), "sessionInfo")

        # initiate database
        self.knowledgeBase = dbManager(knowledgeBasePath, 'knowDB')
        self.sessionInfo = dbManager(sessionInformationPath, 'sessDB')

        # choice of LLM
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")
        # initiate langchain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=self.knowledgeBase.as_retriever()
            )

    def buildKnowledgeBase(self, fileList):
        """load reference documents to learn subjective queries"""
        logger("creating knowledge database", "resumeWiki")
        self.knowledgeBase.updateBulk(fileList=fileList)
    
    def ask_question(self, question, wordlimit=100):
        presetText = f"""explain in less than {wordlimit} words as a continuous paragraph.
        consider 'you' as addressing the resume owner.
        Just provide only the answer.
        answer in the voice of the resume owner."""
        logger("raising a question", "resumeWiki")
        query = '\n'.join([question, presetText])
        response = self.qa_chain.invoke(query)
        return response['result']
    

def myWrap(text, width=80):
    lines = text.splitlines()
    wrapped_lines = []

    for line in lines:
        words = line.split()
        newLine = ""
        for word in words:
            if len(newLine) + len(word) + 1 <= width:
                newLine = ' '.join([newLine, word])
            else:
                wrapped_lines.append(newLine.strip())
                newLine = word
        
        wrapped_lines.append(newLine.strip())
        
    return '\n'.join(wrapped_lines)

if __name__ == "__main__":

    wiki = resumeWiki()
    wiki.buildKnowledgeBase(["HariSamynaath_resume_20250210.txt",
                             "HariSamynaath_profile_writeup.txt"])
    print(myWrap(wiki.ask_question("What is your years of experience as a Data Scientist?")))