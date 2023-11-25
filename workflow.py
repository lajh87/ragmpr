# workflow
import pinecone
import os
from dotenv import load_dotenv
load_dotenv()

index_name = 'langchain-retrieval-augmentation'

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment="gcp-starter"
)

# Connect to Index
index = pinecone.GRPCIndex(index_name)
index.describe_index_stats()


# Open Embeddings

from langchain.embeddings.openai import OpenAIEmbeddings

model_name = 'text-embedding-ada-002'
embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Switch Back to Langchain
from langchain.vectorstores import Pinecone
text_field = "text"

# switch back to normal index for langchain
index = pinecone.Index(index_name)

vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

# Query
query = "Provide an example of a major project"
vectorstore.similarity_search(
    query,  # our search query
    k=3  # return 3 most relevant docs
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

template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Keep the answer as concise as possible. Cite the document and page number your retrieved your answer from.
{context}
Question: {question}
Helpful Answer:"""
rag_prompt_custom = PromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt_custom
    | llm
    | StrOutputParser()
)

rag_chain.invoke("What are the causes of schedule variation?")