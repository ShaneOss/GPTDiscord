import asyncio
from qdrant_client import QdrantClient


class QdrantService:
    def __init__(self, host, port, collection_name, api_key=None):
        self.client = QdrantClient(
            host=host, port=port, collection=collection_name, token=api_key
        )

    async def upsert_basic(self, text, embeddings):
        await self.client.insert(items=[(text, embeddings)])

    async def get_all_for_conversation(self, conversation_id: int):
        response = await self.client.search(
            query_vector=None,
            filters={"conversation_id": conversation_id},
            top_k=100,
        )
        return response

    async def upsert_conversation_embedding(
        self, model, conversation_id: int, text, timestamp, custom_api_key=None
    ):
        # If the text is > 512 characters, we need to split it up into multiple entries.
        first_embedding = None
        if len(text) > 500:
            # Split the text into 512 character chunks
            chunks = [text[i : i + 500] for i in range(0, len(text), 500)]
            for chunk in chunks:
                # Create an embedding for the split chunk
                embedding = await model.send_embedding_request(
                    chunk, custom_api_key=custom_api_key
                )
                if not first_embedding:
                    first_embedding = embedding
                await self.client.insert(
                    items=[(chunk, embedding)],
                    metadata={
                        "conversation_id": conversation_id,
                        "timestamp": timestamp,
                    },
                )
            return first_embedding
        embedding = await model.send_embedding_request(
            text, custom_api_key=custom_api_key
        )
        await self.client.insert(
            items=[
                (
                    text,
                    embedding,
                )
            ],
            metadata={
                "conversation_id": conversation_id,
                "timestamp": timestamp,
            },
        )
        return embedding

    async def get_n_similar(self, conversation_id: int, embedding, n=10):
        response = await self.client.search(
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
            query_vector=[0] * 1536,
            filters={"conversation_id": conversation_id},
            top_k=1000,
        )
        phrases = [match["payload"]["id"] for match in response["items"]]

        # Sort on timestamp
        phrases.sort(key=lambda x: x[1])
        return phrases