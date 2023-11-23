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
