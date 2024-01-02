import asyncio
import os
import json
import requests
from typing import Any, AsyncGenerator, Coroutine, Generator, Optional

from fastapi import HTTPException
from pydantic import BaseModel, Field

from llmstudio.engine.providers.provider import ChatRequest, Provider, provider


class AI71Parameters(BaseModel):
    temperature: Optional[float] = Field(default=1, ge=0)
    max_tokens: Optional[int] = Field(default=256, ge=1)
    top_p: Optional[float] = Field(default=1, ge=0, le=1)
    top_k: Optional[int] = Field(default=1, ge=0)
    frequency_penalty: Optional[float] = Field(default=0, ge=0)
    presence_penalty: Optional[float] = Field(default=0, ge=0)


class AI71Request(ChatRequest):
    parameters: Optional[AI71Parameters] = AI71Parameters()


@provider
class AI71Provider(Provider):
    def __init__(self, config):
        super().__init__(config)
        self.API_KEY = os.getenv("AI71_API_KEY")

    def validate_request(self, request: AI71Request):
        return AI71Request(**request)

    async def generate_client(
        self, request: AI71Request
    ) -> Coroutine[Any, Any, Generator]:
        """Generate a AI71 client"""
        try:
            return await asyncio.to_thread(
                requests.post,
                url="https://api.ai71.decart.ai/v1/chat/completions",
                # headers={"Authorization": f"Bearer {request.api_key or self.API_KEY}"},
                headers={
                    "Authorization": "Bearer ai71-api-93fe83d2-517c-40d2-a5e2-ea261832f8a6"
                },
                json={
                    "model": request.model,
                    "messages": [{"role": "user", "content": request.chat_input}],
                    "stream": True,
                    **request.parameters.dict(),
                },
                stream=True,
            )
        except requests.exceptions.RequestException as e:
            raise HTTPException(
                status_code=getattr(e.response, "status_code", 500),
                detail={"error": str(e)},
            )

    async def parse_response(
        self, response: AsyncGenerator
    ) -> AsyncGenerator[str, None]:
        for line in response.iter_lines():
            if line:
                yield json.loads(line.decode("utf-8")[6:])["choices"][0]["delta"][
                    "content"
                ]
