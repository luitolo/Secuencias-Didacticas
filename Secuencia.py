#import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
from langchain.chains import RetrievalQA
from langchain.vectorstores import ElasticVectorSearch, Pinecone, FAISS
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader

#reader=PdfReader('C:/Mis Documentos/Magister Ciencia de Datos/Tesina/SecuenciaDidacticaRecursosTIC.pdf')
reader=PdfReader('SecuenciaDidacticaRecursosTIC.pdf')
loader = PyPDFLoader('C:/Mis Documentos/Magister Ciencia de Datos/Tesina/mi_proyecto/SecuenciaDidacticaRecursosTIC.pdf')
#loader = PyPDFLoader('SecuenciaDidacticaRecursosTIC.pdf')

raw_text=' '
for i, page in enumerate(reader.pages):
  text= page.extract_text()
  if text:
    raw_text +=text

documents = loader.load()

pages = loader.load_and_split()
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator="\n")
docs = text_splitter.split_documents(documents)

#import getpass
import openai
import langchain
import os

#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
#os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key: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")

#from dotenv import load_dotenv
#load_dotenv()

#API_KEY=os.getenv("API_KEY")
openai_api_key = os.environ.get("OPENAI_API_KEY")


from langchain.llms import openai

my_key=os.getenv("OPENAI_API_KEY")
#print(f"key is: {my_key}")
 
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings

#embeddings = OpenAIEmbeddings(deployment="text-embedding-ada-002")
embeddings = OpenAIEmbeddings(deployment="text-Embedding-Davinci-003")

#EMBEDDING_MODEL = "text-embedding-ada-002"

db= FAISS.from_documents(documents, embeddings)

llm=OpenAI(temperature=0,openai_api_key=my_key)

from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import openai
from langchain.llms import OpenAI
import numpy as np

chain=load_qa_chain(OpenAI(), chain_type="stuff")

import streamlit as st

#Configuración de nuestra plataforma virtual con streamlit
st.set_page_config(page_title="Plataforma Virtual Secuencias Didacticas", layout="centered")

#Definición del título y de la descripción de la aplicación

st.title("Secuencias didacticas con integración de recursos TIC")

st.write("""Este proyecto tiene como proposito disponer de una plataforma virtual para consultar y compartir las experiencias de profesores con la integración y uso de recursos tecnológicos en el proceso de aprendizaje de los alumnos""")
st.write("""Para llevar a cabo tu consulta. Debes ingresar tu nombre, el curso de los alumnos de nivel medio y el eje temático de interes de la asignatura de matemáticas""")
#Cargamos y mostramos el logo de la aplicación en la parte lateral

logo="logo_TPACK.jpg"
st.image(logo,caption='Modelo TPACK', use_column_width=350,)


#queries = []

#query1=input(f"Type in query: \nBienvenido a nuestra plataforma virtual. Tu nombre por favor ")
#respuesta1="Bienvenido a nuestra plataforma virtual " + query1

#query2 = input(f"Type in query: \nSecuencia Didáctica para alumnos de ")
# Concatenar la frase "Secuencia Didáctica para alumnos de" con query1
#respuesta2 = "Cual es la Secuencia Didáctica para alumnos de " + query2
#queries.append(query2)
#query3=input(f"Type in query: \ndel eje tematico de ")
#respuesta3="del eje tematico " + query3
#queries.append(query3)
#consulta=' '.join([query2,query3])
#consulta_final = ' '.join(queries)
#Consulta=' '.join([respuesta2,respuesta3])

#print(query1)
#print(respuesta1)
#print(respuesta2)
#print(respuesta3)
#print(Consulta)

retriever= db.as_retriever()
qa=RetrievalQA.from_chain_type(
    llm= OpenAI(temperature=0.2),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False)

prompts = [
    "La Secuencia Didactica para alumnos de primero medio del eje tematico de geometria: Mediante el uso de una sopa de letras, los estudiantes identifican conceptos relativos de la semejanza de figuras plana a modo de diagnóstico e inicio de la clase.El recurso TIC empleado es Wordwall es una herramienta digital que permite crear juegos para los alumnos.",
    "La Secuencia Didáctica recomendadas para alumnos de segundo medio de la asignatura de matemáticas del eje temático números: Mediante Bingo matemático los estudiantes van resolviendo los ejercicios que aparecen en el Software GENIALLY utilizado para crear contenidos interactivos.",
    "La Secuencia Didáctica recomendadas para alumnos de segundo medio de la asignatura de matemáticas del eje temático números: Utilizar videos explicativos para introducir conceptos matemáticos mediante el uso de la plataforma digital YouTube.",
    "Completa la siguiente consulta: La Secuencia didactica y recursos requeridos para alumnos de segundo medio del eje tematico potencias___."
]

queries = []

query1=st.text_input(f"Bienvenido a nuestra plataforma virtual: \nTu nombre por favor ")
#st.write(query1)
respuesta1="Bienvenido a nuestra plataforma virtual " + query1

query2 = st.text_input(f"Indique el grado escolar medio: \nSecuencia Didáctica para alumnos de ")
# Concatenar la frase "Secuencia Didáctica para alumnos de" con query1
respuesta2 = "Cual es la Secuencia Didáctica para alumnos de " + query2
queries.append(query2)
query3=st.text_input(f"Indique tematica de interes: \ndel eje tematico de ")
respuesta3="del eje tematico " + query3
queries.append(query3)
#consulta=' '.join([query2,query3])
#consulta_final = ' '.join(queries)
Consulta=' '.join([respuesta2,respuesta3])

prompts[3] = Consulta

# Realizar consultas y imprimir tanto la consulta como la respuesta
for index, prompt in enumerate(prompts):
    response = qa(prompt)
    if index == 3:  # Solo imprimir la tercera consulta y su respuesta
        #print("Consulta:", prompt)
        print("Respuesta:", response)

#query= "Secuencia didáctica impartida para alumnos de Primero medio de la asignatura de Matemáticas y eje temático Geometría"
query=Consulta
#result=qa({"query": query})

# Output: mostrar el resultado
st.write(response)

#print(result['result'])





