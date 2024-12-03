from datetime import datetime
from openai import OpenAI
from config import (
    OPENAI_API_KEY,
    OPENAI_PROJECT_ID,
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
    OPENAI_MAX_TOKENS,
    USE_REAL_OPENAI_API,
    DUMMY_RESPONSE,
    LOG_OPENAI_RESPONSE_DIR,
)
from logging_config import setup_logger
from src.openai_token_count_and_cost import calculate_token_count, calculate_price

logger = setup_logger(__name__)


def get_openai_client():
    """Dependency injection for OpenAI client."""
    return OpenAI(api_key=OPENAI_API_KEY, project=OPENAI_PROJECT_ID)


def prepare_messages(system_content: str, user_content: str):
    """Prepare messages for OpenAI API."""
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def estimate_costs(token_count: int, model: str, input_tokens: bool, comment: str = ""):
    """Estimate costs for input or output tokens."""
    cost = calculate_price(token_count, model, input=input_tokens)
    token_type = "input" if input_tokens else "output"
    logger.info(f"Estimated{comment} {token_type} cost: ${cost:.6f}")
    return cost


def log_and_save_response(response: str):
    """Log and save API response to a file with error handling."""
    logger.info(f"OpenAI response: {response}")
    file_name = (
        LOG_OPENAI_RESPONSE_DIR / f"openai_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    try:
        with open(file_name, "w", encoding="utf-8") as file:
            file.write(response)
        logger.info(f"Response saved to {file_name}")
    except IOError as e:
        logger.error(f"Failed to save response to file: {e}")
        raise IOError("An error occurred while saving the response to a file.")
    return file_name


def call_openai_api_real(client, messages):
    """Call the real OpenAI API."""
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_tokens=OPENAI_MAX_TOKENS,
            temperature=OPENAI_TEMPERATURE,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI API Error: {e}")
        raise ValueError("An error occurred while communicating with the OpenAI API.")


def call_openai_api_mock():
    """Return mock data for testing mode."""
    logger.info("Using mock data (testing mode) instead of calling the OpenAI API.")
    return DUMMY_RESPONSE


def call_openai_api(client, messages):
    """Call the OpenAI API or return dummy data."""
    if USE_REAL_OPENAI_API:
        return call_openai_api_real(client, messages)
    else:
        return call_openai_api_mock()


def calculate_input_and_hypothetical_costs(system_content: str, user_content: str, model: str):
    """Calculate input and hypothetical output token costs before API call."""
    messages = prepare_messages(system_content, user_content)
    input_token_count = calculate_token_count(messages, model)
    input_cost = estimate_costs(input_token_count, model, input_tokens=True)

    # Hypothetical output token cost assumes the same number of tokens as input
    hypothetical_token_count = input_token_count  # Hypothetical scenario
    hypothetical_output_cost = estimate_costs(
        hypothetical_token_count, model, input_tokens=False, comment=" (Hypothetical)"
    )

    return messages, input_token_count, input_cost, hypothetical_output_cost


def analyze_response_and_calculate_costs(response: str, model: str):
    """Analyze response and calculate real output costs."""
    output_token_count = calculate_token_count([{"role": "assistant", "content": response}], model)
    output_cost = estimate_costs(output_token_count, model, input_tokens=False)
    log_and_save_response(response)
    return output_token_count, output_cost


def process_and_analyze_file(system_content: str, user_content: str, client=None):
    """Process user input, interact with OpenAI API, and analyze the response."""
    if client is None:
        client = get_openai_client()

    # Calculate costs before API call
    messages, input_token_count, input_cost, hypothetical_output_cost = (
        calculate_input_and_hypothetical_costs(system_content, user_content, OPENAI_MODEL)
    )

    response = call_openai_api(client, messages)

    # Calculate costs after API call
    output_token_count, output_cost = analyze_response_and_calculate_costs(response, OPENAI_MODEL)

    total_cost = input_cost + output_cost
    logger.info(f"Total estimated cost: ${total_cost:.6f}")

    return response
