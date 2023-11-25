
# Load environmental environments
import os
from dotenv import load_dotenv
load_dotenv()

# Open OpenAI Embeddings
from langchain.embeddings.openai import OpenAIEmbeddings
model_name = 'text-embedding-ada-002'
embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Load Pinecone in Langchain
import pinecone
from langchain.vectorstores import Pinecone

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment="gcp-starter"
)

# use normal index for langchain
index = pinecone.Index('langchain-retrieval-augmentation')
vectorstore = Pinecone(
    index, embed.embed_query, "text"
)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})


# Generate
from langchain.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

from langchain.prompts import PromptTemplate

template = """Use the following pieces of context to answer the question at the end. /
If you don't know the answer, just say that you don't know, don't try to make up an answer. /
Please answer the question with citation to the paragraphs. /
 For every sentence you write, cite the book name and paragraph number as <id_x_x> /
 
 At the end of your commentary: 
 1. Add key words from the document paragraphs. / 
 2. Suggest a further question that can be answered by the paragraphs provided. / 
 3. Create a sources list of book names, paragraph Number author name, and a link for each book you cited.
{context}
Question: {question}
Helpful Answer:"""
rag_prompt_custom = PromptTemplate.from_template(template)

from operator import itemgetter
from langchain.schema.runnable import RunnableMap

rag_chain_from_docs = (
    {
        "context": lambda input: format_docs(input["documents"]),
        "question": itemgetter("question"),
    }
    | rag_prompt_custom
    | llm
    | StrOutputParser()
)

rag_chain_with_source = RunnableMap(
    {"documents": retriever, "question": RunnablePassthrough()}
) | {
    "documents": lambda input: [doc.metadata for doc in input["documents"]],
    "answer": rag_chain_from_docs,
}

from shiny import ui, render, App, reactive


app_ui = ui.page_fluid(
    ui.panel_title("NAO Major Project Reports - Retrieval Augmented Generation"),
    ui.input_text_area(
        id = "prompt",
        label = "Question",
        value = "Summarise this context",
        width = "400px",
        height = "250px",
        resize = "none",  
        autoresize=True
    ),
    ui.input_action_button(
        id = "go",
        label = "Ask"
    ),
    ui.tags.h6("Answer"),
    ui.tags.br(),
    ui.output_text_verbatim("txt")
)

def server(input, output, session):
    @output
    @render.text
    @reactive.event(input.go, ignore_none=False)
    def txt():
        response = rag_chain_with_source.invoke(input.prompt())
        return f"{response['answer']} \n {response['documents']}"

# This is a shiny.App object. It must be named `app`.
app = App(app_ui, server)
