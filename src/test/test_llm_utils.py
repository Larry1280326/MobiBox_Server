"""Tests for LLM utilities."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from src.llm_utils.services import (
    generate_structured_output,
    query_llm,
    get_llm,
    summarize_long_text,
)


class SampleOutputSchema(BaseModel):
    """Sample schema for structured output tests."""

    title: str
    content: str


@pytest.fixture
def mock_llm_settings():
    """Mock LLM settings for testing."""
    with patch("src.llm_utils.services.get_llm_settings") as mock:
        settings = MagicMock()
        settings.azure_openai_api_key = "test-api-key"
        settings.azure_openai_endpoint = "https://test.azure-api.net"
        settings.azure_openai_api_version = "2024-10-01-preview"
        settings.azure_openai_deployment = "gpt-4o"
        settings.default_temperature = 0.1
        mock.return_value = settings
        yield mock


@pytest.fixture
def mock_azure_chat_openai():
    """Mock AzureChatOpenAI LLM instance."""
    with patch("src.llm_utils.services.AzureChatOpenAI") as mock_llm_class:
        mock_llm_instance = MagicMock()
        mock_llm_class.return_value = mock_llm_instance
        yield mock_llm_class, mock_llm_instance


class TestGetLlm:
    """Tests for get_llm function."""

    def test_get_llm_with_defaults(self, mock_llm_settings, mock_azure_chat_openai):
        """get_llm creates LLM with default settings."""
        mock_class, mock_instance = mock_azure_chat_openai

        result = get_llm()

        mock_class.assert_called_once()
        call_kwargs = mock_class.call_args.kwargs
        assert call_kwargs["azure_deployment"] == "gpt-4o"
        assert call_kwargs["api_version"] == "2024-10-01-preview"
        assert call_kwargs["temperature"] == 0.1
        assert call_kwargs["api_key"] == "test-api-key"
        assert call_kwargs["azure_endpoint"] == "https://test.azure-api.net"
        assert result == mock_instance

    def test_get_llm_with_custom_params(self, mock_llm_settings, mock_azure_chat_openai):
        """get_llm uses custom parameters when provided."""
        mock_class, mock_instance = mock_azure_chat_openai

        result = get_llm(
            model_type="gpt-35-turbo",
            api_version="2024-01-01",
            temperature=0.5,
            max_tokens=1000,
            max_retries=5,
        )

        call_kwargs = mock_class.call_args.kwargs
        assert call_kwargs["azure_deployment"] == "gpt-35-turbo"
        assert call_kwargs["api_version"] == "2024-01-01"
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 1000
        assert call_kwargs["max_retries"] == 5
        assert result == mock_instance


class TestQueryLlm:
    """Tests for query_llm function."""

    @pytest.mark.asyncio
    async def test_query_llm_returns_result(self, mock_llm_settings, mock_azure_chat_openai):
        """query_llm returns the generated text."""
        mock_class, mock_instance = mock_azure_chat_openai

        # Mock the chain
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value="Generated text result")

        with patch("src.llm_utils.services.ChatPromptTemplate") as mock_prompt:
            mock_prompt_instance = MagicMock()
            mock_prompt.from_messages.return_value = mock_prompt_instance
            mock_prompt_instance.__or__ = MagicMock(return_value=mock_chain)
            mock_chain.__or__ = MagicMock(return_value=mock_chain)

            # Need to mock the | operator chain properly
            with patch("src.llm_utils.services.StrOutputParser") as mock_parser:
                mock_parser_instance = MagicMock()
                mock_parser.return_value = mock_parser_instance

                # Setup the chain: prompt | llm | parser
                mock_prompt_instance.__or__ = MagicMock(return_value=mock_chain)
                mock_chain.__or__ = MagicMock(return_value=mock_chain)

                result = await query_llm(
                    system_prompt="You are a helpful assistant.",
                    user_prompt="Hello, world!",
                )

        assert result == "Generated text result"

    @pytest.mark.asyncio
    async def test_generate_text_with_custom_params(self, mock_llm_settings, mock_azure_chat_openai):
        """generate_text passes custom parameters to get_llm."""
        mock_class, mock_instance = mock_azure_chat_openai

        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value="Result")

        with patch("src.llm_utils.services.ChatPromptTemplate") as mock_prompt:
            mock_prompt_instance = MagicMock()
            mock_prompt.from_messages.return_value = mock_prompt_instance
            mock_prompt_instance.__or__ = MagicMock(return_value=mock_chain)
            mock_chain.__or__ = MagicMock(return_value=mock_chain)

            with patch("src.llm_utils.services.StrOutputParser"):
                await query_llm(
                    system_prompt="System",
                    user_prompt="User",
                    model_type="gpt-35-turbo",
                    temperature=0.7,
                )

        # Verify get_llm was called with custom params (via AzureChatOpenAI constructor)
        call_kwargs = mock_class.call_args.kwargs
        assert call_kwargs["azure_deployment"] == "gpt-35-turbo"
        assert call_kwargs["temperature"] == 0.7


class TestGenerateStructuredOutput:
    """Tests for generate_structured_output function."""

    @pytest.mark.asyncio
    async def test_generate_structured_output_returns_schema(self, mock_llm_settings, mock_azure_chat_openai):
        """generate_structured_output returns structured Pydantic model."""
        mock_class, mock_instance = mock_azure_chat_openai

        expected_result = SampleOutputSchema(title="Test Title", content="Test Content")

        # Mock the structured output chain
        mock_structured_chain = MagicMock()
        mock_structured_chain.ainvoke = AsyncMock(return_value=expected_result)

        mock_instance.with_structured_output = MagicMock(return_value=mock_structured_chain)

        with patch("src.llm_utils.services.ChatPromptTemplate") as mock_prompt:
            mock_prompt_instance = MagicMock()
            mock_prompt.from_messages.return_value = mock_prompt_instance
            mock_prompt_instance.__or__ = MagicMock(return_value=mock_structured_chain)

            result = await generate_structured_output(
                system_prompt="Generate a title and content.",
                user_prompt="Write about AI.",
                output_schema=SampleOutputSchema,
            )

        assert isinstance(result, SampleOutputSchema)
        assert result.title == "Test Title"
        assert result.content == "Test Content"

    @pytest.mark.asyncio
    async def test_generate_structured_output_uses_schema(self, mock_llm_settings, mock_azure_chat_openai):
        """generate_structured_output calls with_structured_output with correct schema."""
        mock_class, mock_instance = mock_azure_chat_openai

        mock_structured_chain = MagicMock()
        mock_structured_chain.ainvoke = AsyncMock(return_value=SampleOutputSchema(title="T", content="C"))
        mock_instance.with_structured_output = MagicMock(return_value=mock_structured_chain)

        with patch("src.llm_utils.services.ChatPromptTemplate") as mock_prompt:
            mock_prompt_instance = MagicMock()
            mock_prompt.from_messages.return_value = mock_prompt_instance
            mock_prompt_instance.__or__ = MagicMock(return_value=mock_structured_chain)

            await generate_structured_output(
                system_prompt="System",
                user_prompt="User",
                output_schema=SampleOutputSchema,
            )

        mock_instance.with_structured_output.assert_called_once_with(
            schema=SampleOutputSchema,
            include_raw=False,
        )


class TestSummarizeLongText:
    """Tests for summarize_long_text function."""

    @pytest.mark.asyncio
    async def test_summarize_short_text(self, mock_llm_settings, mock_azure_chat_openai):
        """summarize_long_text processes short text without chunking."""
        mock_class, mock_instance = mock_azure_chat_openai

        short_text = "This is a short text."

        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value="Summary result")
        mock_chain.abatch = AsyncMock(return_value=["Summary result"])

        with patch("src.llm_utils.services.ChatPromptTemplate") as mock_prompt:
            mock_prompt_instance = MagicMock()
            mock_prompt.from_messages.return_value = mock_prompt_instance
            mock_prompt_instance.__or__ = MagicMock(return_value=mock_chain)
            mock_chain.__or__ = MagicMock(return_value=mock_chain)

            with patch("src.llm_utils.services.StrOutputParser"):
                with patch("src.llm_utils.services.RecursiveCharacterTextSplitter") as mock_splitter:
                    mock_splitter_instance = MagicMock()
                    mock_splitter_instance.split_text.return_value = [short_text]
                    mock_splitter.return_value = mock_splitter_instance

                    result = await summarize_long_text(
                        content=short_text,
                        instruction="Summarize this text.",
                    )

        assert result == "Summary result"

    @pytest.mark.asyncio
    async def test_summarize_long_text_chunks_content(self, mock_llm_settings, mock_azure_chat_openai):
        """summarize_long_text splits long text into chunks and combines."""
        mock_class, mock_instance = mock_azure_chat_openai

        long_text = "Long content " * 500

        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value="Final combined summary")
        mock_chain.abatch = AsyncMock(return_value=["Chunk 1 summary", "Chunk 2 summary"])
        mock_chain.__or__ = MagicMock(return_value=mock_chain)

        with patch("src.llm_utils.services.ChatPromptTemplate") as mock_prompt:
            mock_prompt_instance = MagicMock()
            mock_prompt.from_messages.return_value = mock_prompt_instance
            mock_prompt_instance.__or__ = MagicMock(return_value=mock_chain)

            with patch("src.llm_utils.services.StrOutputParser"):
                with patch("src.llm_utils.services.RecursiveCharacterTextSplitter") as mock_splitter:
                    mock_splitter_instance = MagicMock()
                    mock_splitter_instance.split_text.return_value = ["chunk1", "chunk2"]
                    mock_splitter.return_value = mock_splitter_instance

                    result = await summarize_long_text(
                        content=long_text,
                        instruction="Summarize.",
                        chunk_size=1000,
                        chunk_overlap=100,
                    )

        # Should call abatch for multiple chunks
        mock_chain.abatch.assert_called_once()
        # Should call ainvoke for final combination
        mock_chain.ainvoke.assert_called_once()
        assert result == "Final combined summary"

    @pytest.mark.asyncio
    async def test_summarize_uses_custom_params(self, mock_llm_settings, mock_azure_chat_openai):
        """summarize_long_text passes custom parameters."""
        mock_class, mock_instance = mock_azure_chat_openai

        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value="Summary")
        mock_chain.abatch = AsyncMock(return_value=["Summary"])

        with patch("src.llm_utils.services.ChatPromptTemplate") as mock_prompt:
            mock_prompt_instance = MagicMock()
            mock_prompt.from_messages.return_value = mock_prompt_instance
            mock_prompt_instance.__or__ = MagicMock(return_value=mock_chain)
            mock_chain.__or__ = MagicMock(return_value=mock_chain)

            with patch("src.llm_utils.services.StrOutputParser"):
                with patch("src.llm_utils.services.RecursiveCharacterTextSplitter") as mock_splitter:
                    mock_splitter_instance = MagicMock()
                    mock_splitter_instance.split_text.return_value = ["Text"]
                    mock_splitter.return_value = mock_splitter_instance

                    await summarize_long_text(
                        content="Content",
                        instruction="Summarize.",
                        model_type="gpt-35-turbo",
                        temperature=0.5,
                    )

        call_kwargs = mock_class.call_args.kwargs
        assert call_kwargs["azure_deployment"] == "gpt-35-turbo"
        assert call_kwargs["temperature"] == 0.5