# Permanent Memory and Conversations
We are using [QDRANT](https://qdrant.tech/) as our vector database backing, we have moved away from pinecone. Qdrant is an excellent vector database choice, and in fact the best one that we've tested and used so far. 

Permanent memory has now been implemented into the bot, using the OpenAI Ada embeddings endpoint, and Qdrant.  
  
Qdrant is a vector database. The OpenAI Ada embeddings endpoint turns pieces of text into embeddings. The way that this feature works is by embedding the user prompts and the GPT responses, storing them in an index, and then retrieving the most relevant bits of conversation whenever a new user prompt is given in a conversation.  
  
**You do NOT need to use Qdrant, if you do not define a `QDRANT_API_KEY`,`QDRANT_HOST` and `QDRANT_PORT` in your `.env` file, the bot will default to not using Qdrant, and will use conversation summarization as the long term conversation method instead.**  
  
To enable permanent memory with Qdrant, you must define a `QDRANT_API_KEY`,`QDRANT_HOST` and `QDRANT_PORT` in your `.env` file as follows (along with the other variables too):  
```env  
QDRANT_API_KEY = "<qdrant_api_key>"  # Qdrant API key
QDRANT_HOST = "https://b2a871c8-92b5-4e8d-9b2d-5f4f9b0d7d78.us-east4-0.gcp.cloud.qdrant.io"  # Qdrant host
QDRANT_PORT = "6333"  # Qdrant port 
```  
  
To get a Qdrant API_KEY, you can sign up for a free Qdrant account here: https://cloud.qdrant.io/ create a new cluster (free) and generate a new "API Key". (I am not affiliated with Qdrant).
