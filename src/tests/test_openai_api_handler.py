import pytest
from unittest.mock import MagicMock, patch
from src.openai_api_handler import (
    get_openai_client,
    prepare_messages,
    call_openai_api_real,
    call_openai_api_mock,
    call_openai_api,
    calculate_input_and_hypothetical_costs,
    analyze_response_and_calculate_costs,
    process_and_analyze_file,
    log_and_save_response,
)
from src.config import USE_REAL_OPENAI_API


@pytest.fixture
def mock_calculate_token_count():
    """Fixture for mocking calculate_token_count function."""
    with patch("src.openai_token_count_and_cost.calculate_token_count") as mock:
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
        mock_open.return_value.__enter__.return_value.write = MagicMock()
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
        ("", "", [{"role": "system", "content": ""}, {"role": "user", "content": ""}]),
        (
            "LargeSystemContent" * 100,
            "LargeUserContent" * 100,
            [
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


@pytest.mark.parametrize(
    "use_real_api, expected_response",
    [
        (True, "Mocked API response"),
        (False, "This is a dummy response for testing purposes."),
    ],
)
def test_call_openai_api_with_env_toggle(mock_openai_client, use_real_api, expected_response):
    """Test API behavior toggling between real and mock modes based on environment."""
    messages = [{"role": "user", "content": "Hello"}]

    # Patch the value of USE_REAL_OPENAI_API in the src.openai_api_handler module
    with patch("src.openai_api_handler.USE_REAL_OPENAI_API", use_real_api):
        with patch(
            "src.openai_api_handler.call_openai_api_real", return_value="Mocked API response"
        ) as mock_real:
            with patch(
                "src.openai_api_handler.call_openai_api_mock",
                return_value="This is a dummy response for testing purposes.",
            ) as mock_mock:
                # Call the function and capture the response
                response = call_openai_api(mock_openai_client if use_real_api else None, messages)

                # Validate the response matches expectations
                assert response == expected_response

                # Validate that the correct function was called based on the toggle
                if use_real_api:
                    mock_real.assert_called_once_with(mock_openai_client, messages)
                    mock_mock.assert_not_called()
                else:
                    mock_mock.assert_called_once()
                    mock_real.assert_not_called()


@pytest.mark.parametrize(
    "token_count, expected_input_cost, expected_hypothetical_cost",
    [
        (0, 0, 0),  # No tokens
        (100, 0.0003, 0.0006),  # 100 tokens with input and hypothetical costs
        (10000, 0.03, 0.06),  # 10,000 tokens with input and hypothetical costs
    ],
)
def test_calculate_input_and_hypothetical_costs_param(
    token_count, expected_input_cost, expected_hypothetical_cost
):
    """Test token and cost calculations with parameterized token counts."""
    # Mock calculate_token_count for the context of this test
    with patch("src.openai_api_handler.calculate_token_count", return_value=token_count):
        system_content = "System message"
        user_content = "User message"
        model = "gpt-3.5-turbo"

        # Calculate costs
        messages, input_token_count, input_cost, hypothetical_output_cost = (
            calculate_input_and_hypothetical_costs(system_content, user_content, model)
        )

        # Assertions
        assert input_token_count == token_count
        assert input_cost == pytest.approx(expected_input_cost, rel=1e-2)
        assert hypothetical_output_cost == pytest.approx(expected_hypothetical_cost, rel=1e-2)


def test_analyze_response_and_calculate_costs(mock_calculate_token_count):
    """Test response analysis and output cost calculations."""
    response = "Mocked response content"
    model = "gpt-3.5-turbo"

    output_token_count, output_cost = analyze_response_and_calculate_costs(response, model)

    assert output_token_count > 0
    assert output_cost > 0


def test_process_and_analyze_file(mock_openai_client, mock_calculate_token_count):
    """Test end-to-end processing and analysis of user input and API response."""
    mock_calculate_token_count.return_value = 100
    system_content = "System content"
    user_content = "User content"

    with patch("src.openai_api_handler.call_openai_api_mock", return_value="Mocked API response"):
        response = process_and_analyze_file(system_content, user_content, mock_openai_client)
        assert response == "Mocked API response"


def test_log_and_save_response(mock_file_handling):
    """Test saving response to a file with mocked file handling."""
    response = "Test response to save"
    log_and_save_response(response)
    mock_file_handling.assert_called_once()
    mock_file_handling.return_value.__enter__.return_value.write.assert_called_once_with(response)


def test_call_openai_api_real_error_handling():
    """Test error handling for real API call failures."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("Mocked API Error")

    with pytest.raises(
        ValueError, match="An error occurred while communicating with the OpenAI API."
    ):
        call_openai_api_real(mock_client, [{"role": "user", "content": "Hello"}])


def test_file_save_error_handling():
    """Test error handling when saving to a file fails."""
    response = "Test response"
    with patch("builtins.open", side_effect=IOError("Mocked IOError")):
        with pytest.raises(IOError, match="An error occurred while saving the response to a file."):
            log_and_save_response(response)
