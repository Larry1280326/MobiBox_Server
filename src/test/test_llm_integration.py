"""Integration tests for OpenRouter API.

These tests require actual OpenRouter credentials and will make real API calls.
Run with: pytest -m integration src/test/test_llm_integration.py

Note: These tests consume API credits and should be run sparingly.
"""

import pytest
from pydantic import BaseModel

from src.llm_utils.services import (
    generate_structured_output,
    query_llm,
    get_llm,
    summarize_long_text,
)
from src.config import get_llm_settings


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


class AnswerSchema(BaseModel):
    """Schema for structured output tests with answer and confidence."""

    answer: str
    confidence: float


class TestOpenRouterConnection:
    """Tests to verify OpenRouter API connectivity."""

    def test_llm_settings_loaded(self):
        """Verify LLM settings are loaded from environment."""
        settings = get_llm_settings()

        assert settings.openrouter_api_key, "OPENROUTER_API_KEY not set"
        assert settings.openrouter_base_url, "OPENROUTER_BASE_URL not set"
        assert settings.openrouter_model, "OPENROUTER_MODEL not set"

    def test_get_llm_creates_client(self):
        """Verify LLM client can be created."""
        llm = get_llm()

        assert llm is not None
        settings = get_llm_settings()
        assert llm.model_name == settings.openrouter_model

    def test_get_llm_with_custom_model(self):
        """Verify LLM client can be created with custom model."""
        llm = get_llm(model_type="meta-llama/llama-3.2-3b-instruct:free")

        assert llm is not None
        assert llm.model_name == "meta-llama/llama-3.2-3b-instruct:free"


class TestQueryLlmIntegration:
    """Integration tests for query_llm with real OpenRouter API."""

    @pytest.mark.asyncio
    async def test_simple_query_returns_response(self):
        """Test basic query returns a valid response."""
        result = await query_llm(
            system_prompt="You are a helpful assistant.",
            user_prompt="What is 2 + 2? Reply with just the number.",
        )

        assert result is not None
        assert isinstance(result, str)
        # The answer should contain "4"
        assert "4" in result

    @pytest.mark.asyncio
    async def test_query_with_custom_temperature(self):
        """Test query with custom temperature parameter."""
        result = await query_llm(
            system_prompt="You are a helpful assistant.",
            user_prompt="Say 'hello' in one word.",
            temperature=0.0,  # Most deterministic
        )

        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_query_respects_max_tokens(self):
        """Test that max_tokens parameter is respected."""
        result = await query_llm(
            system_prompt="You are a helpful assistant.",
            user_prompt="Count from 1 to 100.",
            max_tokens=10,  # Very short response
        )

        assert result is not None
        # Response should be short due to max_tokens limit
        assert len(result) < 200


class TestGenerateStructuredOutputIntegration:
    """Integration tests for structured output with real OpenRouter API."""

    @pytest.mark.asyncio
    async def test_structured_output_basic(self):
        """Test structured output returns valid Pydantic model."""
        result = await generate_structured_output(
            system_prompt="You are a helpful assistant that answers questions accurately.",
            user_prompt="What is the capital of France?",
            output_schema=AnswerSchema,
        )

        assert result is not None
        assert isinstance(result, AnswerSchema)
        assert result.answer is not None
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0
        assert "Paris" in result.answer

    @pytest.mark.asyncio
    async def test_structured_output_complex_schema(self):
        """Test structured output with more complex schema."""

        class ComplexSchema(BaseModel):
            main_idea: str
            key_points: list[str]
            summary: str

        result = await generate_structured_output(
            system_prompt="You are a helpful assistant.",
            user_prompt="Explain what machine learning is in 2 sentences.",
            output_schema=ComplexSchema,
        )

        assert result is not None
        assert isinstance(result, ComplexSchema)
        assert len(result.main_idea) > 0
        assert isinstance(result.key_points, list)
        assert len(result.summary) > 0


class TestSummarizeLongTextIntegration:
    """Integration tests for summarize_long_text with real OpenRouter API."""

    @pytest.mark.asyncio
    async def test_summarize_short_text(self):
        """Test summarizing short text works correctly."""
        short_text = "Python is a programming language. It was created by Guido van Rossum."

        result = await summarize_long_text(
            content=short_text,
            instruction="Summarize the following text in one sentence.",
        )

        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_summarize_long_text_with_chunks(self):
        """Test summarizing long text that requires chunking."""
        # Create a longer text that will be chunked
        long_text = """
        Artificial Intelligence (AI) is a branch of computer science that aims to create
        intelligent machines that can perform tasks that typically require human intelligence.
        These tasks include learning, reasoning, problem-solving, perception, and language understanding.

        Machine Learning (ML) is a subset of AI that focuses on developing algorithms and
        statistical models that enable computers to learn from and make predictions or decisions
        based on data. It has become one of the most important and widely used areas of AI.

        Deep Learning is a further subset of machine learning that uses neural networks with
        multiple layers to analyze various factors of data. It has been particularly successful
        in image recognition, natural language processing, and speech recognition tasks.

        Natural Language Processing (NLP) is a field of AI that gives machines the ability
        to read, understand, and derive meaning from human languages. It combines computational
        linguistics with machine learning and deep learning models.

        Computer Vision is another important field of AI that trains computers to interpret
        and understand the visual world. Using digital images and deep learning models,
        machines can accurately identify and classify objects.
        """

        result = await summarize_long_text(
            content=long_text.strip(),
            instruction="Summarize the key points about AI and its subfields in 2-3 sentences.",
            chunk_size=500,  # Force chunking
            chunk_overlap=50,
        )

        assert result is not None
        assert isinstance(result, str)
        # Result should be a non-empty summary
        assert len(result) > 0


class TestFreeModelConfiguration:
    """Tests to verify free model configuration."""

    @pytest.mark.asyncio
    async def test_model_is_configured(self):
        """Verify the model is correctly configured."""
        settings = get_llm_settings()

        # Verify we have a model configured
        assert settings.openrouter_model, "OPENROUTER_MODEL should be set"

    @pytest.mark.asyncio
    async def test_model_capability_sufficient(self):
        """Test that the model has sufficient capability for typical tasks."""
        class SimpleResponse(BaseModel):
            message: str

        result = await generate_structured_output(
            system_prompt="You are a helpful assistant.",
            user_prompt="Say 'Hello, World!'",
            output_schema=SimpleResponse,
        )

        assert result is not None
        assert isinstance(result, SimpleResponse)
        assert "Hello" in result.message or "hello" in result.message.lower()