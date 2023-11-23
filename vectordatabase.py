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