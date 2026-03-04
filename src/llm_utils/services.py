"""
LLM service utilities for Azure OpenAI integration.
Provides reusable functions for text generation and structured output.
Includes rate limiting to stay within Azure API limits (60 requests/minute).
"""

import asyncio
import time
from typing import Type, TypeVar
from pydantic import BaseModel
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_llm_settings

# Type variable for structured output models
T = TypeVar("T", bound=BaseModel)


class RateLimiter:
    """
    Token bucket rate limiter for Azure API calls.
    Ensures requests stay within the specified rate limit.
    """

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute  # seconds between requests
        self._lock = asyncio.Lock()
        self._last_request_time: float = 0.0

    async def acquire(self) -> None:
        """Wait until a request can be made without exceeding rate limit."""
        async with self._lock:
            now = time.monotonic()
            time_since_last = now - self._last_request_time

            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                await asyncio.sleep(wait_time)

            self._last_request_time = time.monotonic()


# Global rate limiter instance (60 requests per minute for Azure API)
_azure_rate_limiter = RateLimiter(requests_per_minute=60)


def get_llm(
    model_type: str | None = None,
    api_version: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_retries: int = 2,
) -> AzureChatOpenAI:
    """
    Create an AzureChatOpenAI instance with configured settings.

    Args:
        model_type: Azure deployment name (defaults to configured deployment)
        api_version: Azure API version (defaults to configured version)
        temperature: Sampling temperature (defaults to configured temperature)
        max_tokens: Maximum tokens to generate
        max_retries: Number of retries on failure

    Returns:
        Configured AzureChatOpenAI instance
    """
    settings = get_llm_settings()

    return AzureChatOpenAI(
        azure_deployment=model_type or settings.azure_openai_deployment,
        api_version=api_version or settings.azure_openai_api_version,
        temperature=temperature if temperature is not None else settings.default_temperature,
        max_tokens=max_tokens,
        timeout=None,
        max_retries=max_retries,
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
    )


async def query_llm(
    system_prompt: str,
    user_prompt: str,
    model_type: str | None = None,
    api_version: str | None = None,
    temperature: float | None = None,
) -> str:
    """
    Generate text using Azure OpenAI with a simple prompt structure.

    Args:
        system_prompt: System instruction prompt
        user_prompt: User input prompt
        model_type: Azure deployment name
        api_version: Azure API version
        temperature: Sampling temperature

    Returns:
        Generated text string
    """
    await _azure_rate_limiter.acquire()

    llm = get_llm(
        model_type=model_type,
        api_version=api_version,
        temperature=temperature,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt),
    ])

    chain = prompt | llm | StrOutputParser()
    result = await chain.ainvoke({})

    return result


async def generate_structured_output(
    system_prompt: str,
    user_prompt: str,
    output_schema: Type[T],
    model_type: str | None = None,
    api_version: str | None = None,
    temperature: float | None = None,
) -> T:
    """
    Generate structured output using Azure OpenAI with Pydantic schema.

    Args:
        system_prompt: System instruction prompt
        user_prompt: User input prompt
        output_schema: Pydantic model class for structured output
        model_type: Azure deployment name
        api_version: Azure API version
        temperature: Sampling temperature

    Returns:
        Instance of the output_schema Pydantic model
    """
    await _azure_rate_limiter.acquire()

    llm = get_llm(
        model_type=model_type,
        api_version=api_version,
        temperature=temperature,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt),
    ])

    chain = prompt | llm.with_structured_output(
        schema=output_schema,
        include_raw=False,
    )

    result = await chain.ainvoke({})

    return result


async def summarize_long_text(
    content: str,
    instruction: str,
    chunk_size: int = 3000,
    chunk_overlap: int = 50,
    model_type: str | None = None,
    api_version: str | None = None,
    temperature: float | None = None,
) -> str:
    """
    Summarize long text by chunking and combining results.

    Useful for processing documents longer than the context window.
    Processes chunks sequentially to respect API rate limits.

    Args:
        content: Long text content to summarize
        instruction: Summary instruction prompt
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        model_type: Azure deployment name
        api_version: Azure API version
        temperature: Sampling temperature

    Returns:
        Final summarized text
    """
    llm = get_llm(
        model_type=model_type,
        api_version=api_version,
        temperature=temperature,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", instruction),
        ("user", "{content}"),
    ])

    chain = prompt | llm | StrOutputParser()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    inputs = splitter.split_text(content)

    # Process chunks sequentially to respect rate limits
    generations = []
    for chunk in inputs:
        await _azure_rate_limiter.acquire()
        result = await chain.ainvoke({"content": chunk})
        generations.append(result)

    # Combine intermediate results and summarize again
    intermediate_summary = " ".join(generations)

    if len(inputs) > 1:
        # Final summarization pass
        await _azure_rate_limiter.acquire()
        final_result = await chain.ainvoke({"content": intermediate_summary})
        return final_result

    return intermediate_summary