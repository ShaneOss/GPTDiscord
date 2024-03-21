import asyncio
from qdrant_client import QdrantClient


class QdrantService:
    def __init__(self, client, collection_name):
        self.collection_name = collection_name
        self.client = client

    async def get_all_for_conversation(self, conversation_id: int):
        response = await self.client.search(
            collection_name=self.collection_name,
            query_vector=None,
            filters={"conversation_id": conversation_id},
            top_k=100,
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
        # Upsert texts, embeddings, and conversation_ids into Qdrant
        points = [{"id": cid, "vector": emb, "payload": txt} for txt, emb, cid in zip(texts, embeddings, conversation_ids)]
        try:
            await self.client.upsert(collection_name=self.collection_name, points=points)
            print("Upserted successfully.")
        except Exception as e:
            print(f"Failed to upsert: {e}")

    async def get_n_similar(self, conversation_id: int, embedding, n=10):
        response = await self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            filters={"conversation_id": conversation_id},
            top_k=n,
            include_payload=True,
        )
        relevant_phrases = [
            (match["payload"]["id"], match["payload"]["timestamp"])
            for match in response["items"]
        ]
        # Sort the relevant phrases based on the timestamp
        relevant_phrases.sort(key=lambda x: x[1])
        return relevant_phrases

    async def get_all_conversation_items(self, conversation_id: int):
        response = await self.client.search(
            collection_name=self.collection_name,
            query_vector=[0] * 1536,
            filters={"conversation_id": conversation_id},
            top_k=1000,
        )
        phrases = [match["payload"]["id"] for match in response["items"]]

        # Sort on timestamp
        phrases.sort(key=lambda x: x[1])
        return phrases