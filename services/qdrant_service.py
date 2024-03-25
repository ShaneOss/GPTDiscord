import asyncio
from qdrant_client import QdrantClient

class QdrantService:
    def __init__(self, client, collection_name):
        self.collection_name = collection_name
        self.client = client

    async def get_all_for_conversation(self, conversation_id: int):
        # Updated to use the correct structure for filters and query within search_params
        search_params = {
            "filter": {
                "must": [
                    {"key": "conversation_id", "match": {"integer": conversation_id}}
                ]
            },
            "top": 100,
        }        
        response = await self.client.search(
            collection_name=self.collection_name,
            search_params=search_params,
        )
        
        return response

    async def upsert_conversation_embedding(
        self, model, conversation_id: int, text, timestamp, custom_api_key=None
    ):
        if len(text) > 500:
            # Split the text into chunks
            chunks = [text[i : i + 500] for i in range(0, len(text), 500)]
            embeddings = []
            for chunk in chunks:
                embedding = await model.send_embedding_request(chunk, custom_api_key=custom_api_key)
                embeddings.append(embedding)
            # Upsert embeddings
            await self.upsert_basic(chunks, embeddings, [conversation_id]*len(chunks))
        else:
            embedding = await model.send_embedding_request(text, custom_api_key=custom_api_key)
            # Upsert embedding
            await self.upsert_basic([text], [embedding], [conversation_id])

    async def upsert_basic(self, texts, embeddings, conversation_ids):
        points = [{"id": cid, "vector": emb, "payload": {"text": txt}}
                  for txt, emb, cid in zip(texts, embeddings, conversation_ids)]
        
        # Use an executor to run the synchronous upsert function
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, lambda: self.client.upsert(
                collection_name=self.collection_name,
                points=points
            ))
            print("Upserted successfully.")
        except Exception as e:
            print(f"Failed to upsert: {e}")

    async def get_n_similar(self, conversation_id: int, embedding, n=10):
        # Constructing the search request according to the Qdrant documentation
        search_request = {
            "filter": {
                "must": [
                    # Assuming you're filtering based on 'conversation_id' stored in the payload as an integer
                    {"key": "conversation_id", "condition": {"$eq": conversation_id}}  
                ]
            },
            "vector": embedding,   # The query vector for finding similar items
            "top": n,   # The number of similar items you want to retrieve
            "params": {  # Optional: Parameters to fine-tune the search
                "hnsw_ef": 128  # Example: Adjust the search parameter, if necessary
            },
            "with_payload": True,  # Whether to include the items' payloads in the response
        }

        # Assuming 'search' method of your Qdrant client is synchronous and it directly accepts the 'search_request' dict
        # Adapting to run the synchronous 'search' method in an executor to not block the async loop
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, lambda: self.client.search(
            collection_name=self.collection_name,
            query=search_request  # the actual search request
        ))

        # Process the response to extract relevant items
        relevant_phrases = [
            # Here, adjust the way you access the response and payload based on the actual structure Qdrant sends back
            (match["payload"]["text"], match["score"])  # Adjust key access based on actual payload schema
            for match in response.get("result", {}).get("hits", [])
        ]

        return relevant_phrases

    async def get_all_conversation_items(self, conversation_id: int):
        search_params = {
            "filter": {
                "must": [
                    {"key": "conversation_id", "match": {"integer": conversation_id}}
                ]
            },
            "top": 1000,
            "vector": [0] * 1536,
        }
        response = await self.client.search(
            collection_name=self.collection_name,
            search_params=search_params,
        )
        
        phrases = []
        if response["result"]["hits"]:
            for match in response["result"]["hits"]:
                id = match["payload"]["text"]
                timestamp = match["payload"]["timestamp"]
                phrases.append((id, timestamp))

        # Sort on timestamp
        phrases.sort(key=lambda x: x[1])
        return [phrase[0] for phrase in phrases]