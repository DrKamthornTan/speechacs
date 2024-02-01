import argparse
from dataclasses import dataclass
import speech_recognition as sr
from translate import Translator
import streamlit as st

from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

class StreamlitApp:
    def __init__(self):
        self.r = sr.Recognizer()

    def transcribe(self):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.write("Speak something...")
            audio = r.listen(source)
            st.write("Transcribing...")
            try:
                query_text = r.recognize_google(audio, language='th-TH')
                st.write("Thai transcript:", query_text)
                self.process_query(query_text)
            except sr.UnknownValueError:
                st.write("Speech recognition could not understand audio")
            except sr.RequestError as e:
                st.write("Could not request results from Google Speech Recognition service; {0}".format(e))

    def process_query(self, query_text):
        # Google Translate
        try:
            translator = Translator(from_lang='th', to_lang='en')
            translated_text = translator.translate(query_text)
        except Exception as e:
            st.write(f"Error translating: {str(e)}")
            return

        # Rest of the code...
        st.write(f"Translated Text: {translated_text}") 

        # Prepare the DB.
        openai_api_key = "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"  # Replace with your actual OpenAI API key
        if not openai_api_key:
            st.write("OpenAI API key is not provided.")
            return

        embedding_function = OpenAIEmbeddings(openai_api_key=openai_api_key)
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        # Search the DB.
        results = db.similarity_search_with_relevance_scores(translated_text, k=3)
        if not results or (results and results[0][1] < 0.7):
            st.write("ไม่สามารถค้นหาคำตอบขณะนี้ได้ โปรดติดต่อแพทย์ของท่าน")
            return

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=translated_text)
        st.write(prompt)

        model = ChatOpenAI()
        response_text = model.predict(prompt)

        sources = [doc.metadata.get("source", None) for doc, _score in results]
        formatted_response = f"<span style='color:red'>{response_text}</span>\nSources: {sources}"
        st.write(formatted_response, unsafe_allow_html=True)

    def run(self):
        st.title("ความเสี่ยงต่อโรคหลอดเลือดหัวใจ")
        self.transcribe()


if __name__ == "__main__":
    app = StreamlitApp()
    app.run()