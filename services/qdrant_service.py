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
        try:
            await self.client.upsert(collection_name=self.collection_name, points=points)
            print("Upserted successfully.")
        except Exception as e:
            print(f"Failed to upsert: {e}")

    async def get_n_similar(self, conversation_id: int, embedding, n=10):
        # Assuming 'filters' needs to be structured correctly within the 'search_params' argument now.
        search_params = {
            "filter": {
                "must": [
                    {"key": "conversation_id", "match": {"integer": conversation_id}}
                ]
            },
            "top": n,
            "vector": embedding,
        }
        response = await self.client.search(
            collection_name=self.collection_name,
            search_params=search_params,  # Use the structured 'search_params'
        )
        relevant_phrases = [
            (match["payload"]["id"], match["payload"]["timestamp"])
            for match in response["result"]["hits"]
        ]
        # Sort the relevant phrases based on the timestamp
        relevant_phrases.sort(key=lambda x: x[1])
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