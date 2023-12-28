import asyncio
import os
from typing import Any, AsyncGenerator, Coroutine, Generator, Optional

import openai
from fastapi import HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

from llmstudio.engine.providers.provider import ChatRequest, Provider, provider


class DecartParameters(BaseModel):
    temperature: Optional[float] = Field(default=1, ge=0)
    max_tokens: Optional[int] = Field(default=256, ge=1)
    top_p: Optional[float] = Field(default=1, ge=0, le=1)
    # top_k: Optional[int] = Field(default=1, ge=0)
    frequency_penalty: Optional[float] = Field(default=0, ge=0)
    presence_penalty: Optional[float] = Field(default=0, ge=0)


class DecartRequest(ChatRequest):
    parameters: Optional[DecartParameters] = DecartParameters()


@provider
class DecartProvider(Provider):
    def __init__(self, config):
        super().__init__(config)
        self.API_KEY = os.getenv("DECART_API_KEY")

    def validate_request(self, request: DecartRequest):
        return DecartRequest(**request)

    async def generate_client(
        self, request: DecartRequest
    ) -> Coroutine[Any, Any, Generator]:
        """Generate a Decart client"""
        try:
            print(request.api_key)
            print(self.API_KEY)
            print(os.getenv("OPENAI_API_KEY"))
            print("APIKEYS")
            client = OpenAI(
                api_key=request.api_key or self.API_KEY,
                # api_key="decart-api-0590f0c9-7936-4b26-ab25-cca51f1c3a42",
                base_url="https://api.decart.ai/v1/",
            )
            return await asyncio.to_thread(
                client.chat.completions.create,
                model=request.model,
                messages=[{"role": "user", "content": request.chat_input}],
                stream=True,
                **request.parameters.dict(),
            )
        except openai._exceptions.APIError as e:
            raise HTTPException(status_code=e.status_code, detail=e.response.json())

    async def parse_response(
        self, response: AsyncGenerator
    ) -> AsyncGenerator[str, None]:
        for chunk in response:
            if chunk.choices[0].finish_reason not in ["stop", "length"]:
                yield chunk.choices[0].delta.content
