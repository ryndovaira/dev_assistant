from unittest.mock import MagicMock, patch

import pytest

from src.openai_api_handler import (
    prepare_messages,
    call_openai_api_real,
    call_openai_api_mock,
    log_and_save_response,
)


@pytest.fixture
def mock_calculate_token_count():
    """Fixture for mocking calculate_token_count function."""
    with patch("src.openai_token_count_and_cost.calculate_token_count", return_value=100) as mock:
        yield mock


@pytest.fixture
def mock_openai_client():
    """Fixture for mocking the OpenAI client."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Mocked API response"
    mock_client.chat.completions.create.return_value = mock_response
    yield mock_client


@pytest.fixture
def mock_file_handling():
    """Fixture for mocking file handling operations."""
    with patch("builtins.open", MagicMock()) as mock_open:
        with patch("pathlib.Path.mkdir", MagicMock()):
            yield mock_open


@pytest.mark.parametrize(
    "system_content, user_content, expected",
    [
        (
            "System A",
            "User A",
            [{"role": "system", "content": "System A"}, {"role": "user", "content": "User A"}],
        ),
        (
            "System B",
            "User B",
            [{"role": "system", "content": "System B"}, {"role": "user", "content": "User B"}],
        ),
        (
            "",
            "",
            [{"role": "system", "content": ""}, {"role": "user", "content": ""}],
        ),  # Both empty
        (
            "LargeSystemContent" * 100,
            "LargeUserContent" * 100,
            [  # Large inputs
                {"role": "system", "content": "LargeSystemContent" * 100},
                {"role": "user", "content": "LargeUserContent" * 100},
            ],
        ),
    ],
)
def test_prepare_messages(system_content, user_content, expected):
    """Test message preparation with various input sizes and conditions."""
    assert prepare_messages(system_content, user_content) == expected


def test_call_openai_api_real(mock_openai_client):
    """Test real API call with mocked client."""
    messages = [{"role": "user", "content": "Hello"}]
    response = call_openai_api_real(mock_openai_client, messages)
    assert response == "Mocked API response"
    mock_openai_client.chat.completions.create.assert_called_once()


def test_call_openai_api_mock():
    """Test mock API response."""
    response = call_openai_api_mock()
    assert response == "This is a dummy response for testing purposes."


def test_call_openai_api_real_error_handling():
    """Test error handling for real API call failures."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("Mocked API Error")

    messages = [{"role": "user", "content": "Hello"}]

    with pytest.raises(
        ValueError, match="An error occurred while communicating with the OpenAI API."
    ):
        call_openai_api_real(mock_client, messages)


def test_file_save_error_handling():
    """Test error handling when saving to a file fails."""
    response = "Test response"
    with patch("builtins.open", side_effect=IOError("Mocked IOError")):
        with pytest.raises(IOError, match="An error occurred while saving the response to a file."):
            log_and_save_response(response)
