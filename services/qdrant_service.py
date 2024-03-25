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
        # Assuming that the QdrantClient expects query_vector, top, and filter as separate arguments
        filter = {
            "must": [
                {"key": "conversation_id", "condition": {"$eq": conversation_id}}
            ]
        }
        
        # Conduct the search operation
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, lambda: self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding,  # This is where the query_vector is directly provided
            filter=filter,
            top=n,
            params={"hnsw_ef": 128},  # Assuming additional search parameters are part of a separate `params` argument
            with_payload=True
        ))

        # Process the response
        relevant_phrases = [
            (match["payload"]["text"], match["score"])
            for match in response.get("result", {}).get("hits", [])
        ]

        return relevant_phrases

    async def get_all_conversation_items(self, conversation_id: int):
        # Defining a dummy or neutral vector is unusual for text retrieval unless you have enabled and intend
        # to use a default embedding for all texts. Usually, you would use filtering alone for such retrieval tasks.
        # The 'vector' key and its value ([0] * 1536) might not be needed if you're filtering without vector similarity.
        # Filter structure needs to match Qdrant's expectations.
        filter_condition = {
            "must": [
                {"key": "conversation_id", "condition": {"$eq": conversation_id}}
            ]
        }
        
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, lambda: self.client.search(
            collection_name=self.collection_name,
            query_vector=[0] * 1536,  # or remove if your search does not require a dummy vector
            filter=filter_condition,
            top=1000,
            with_payload=True  # Ensuring payload is included in the response if you rely on it
        ))

        # Extracting phrases from the response
        phrases = []
        if response.get("result", {}).get("hits", []):
            for match in response["result"]["hits"]:
                text = match["payload"].get("text")  # Safely access 'text' in case it's missing
                timestamp = match["payload"].get("timestamp", 0)  # Providing a default if 'timestamp' is missing
                if text:  # Ensuring 'text' is not None or empty before appending
                    phrases.append((text, timestamp))

        # Sort on timestamp (consider if your payload does include 'timestamp' and it's relevant for sorting)
        phrases.sort(key=lambda x: x[1])
        return [phrase[0] for phrase in phrases]