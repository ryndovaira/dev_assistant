from datetime import datetime
from pathlib import Path
from logging_config import setup_logger
from config import (
    OPENAI_API_KEY,
    OPENAI_PROJECT_ID,
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
    OPENAI_MAX_TOKENS,
    USE_REAL_OPENAI_API,
    DUMMY_RESPONSE,
)
from openai import OpenAI

from src.openai_token_count_and_cost import calculate_token_count, calculate_price

logger = setup_logger(__name__)

# Supported file types for validation
SUPPORTED_FILE_TYPES = [".zip"]


# Dependency injection for OpenAI client
def get_openai_client():
    return OpenAI(api_key=OPENAI_API_KEY, project=OPENAI_PROJECT_ID)


# Function to prepare messages for OpenAI API
def prepare_messages(system_content: str, user_content: str):
    return [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": user_content,
        },
    ]


# Function to call OpenAI API or return dummy data
def call_openai_api(client, messages) -> str:
    if USE_REAL_OPENAI_API:
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
    else:
        logger.info("Using mock data (testing mode) instead of calling the OpenAI API.")
        return DUMMY_RESPONSE


def process_and_analyze_file(system_content: str, user_content: str, client=None):
    # Prepare messages for the OpenAI API
    messages = prepare_messages(system_content, user_content)

    # Calculate input token count
    input_token_count = calculate_token_count(messages, OPENAI_MODEL)
    logger.info(f"Input token count: {input_token_count}")

    # Estimate input cost
    input_cost = calculate_price(input_token_count, OPENAI_MODEL, input=True)
    logger.info(f"Estimated input cost: ${input_cost:.6f}")

    # Estimate hypothetical output token count
    output_cost = calculate_price(input_token_count, OPENAI_MODEL, input=False)
    logger.info(f"Estimated output cost: ${output_cost:.6f}")

    # Call the OpenAI API
    result: str = call_openai_api(client, messages)
    logger.info(f"OpenAI response: {result}")

    # Calculate output token count from the response
    output_token_count = calculate_token_count(
        [{"role": "assistant", "content": result}], OPENAI_MODEL
    )
    logger.info(f"Output token count: {output_token_count}")

    # Estimate output cost
    output_cost = calculate_price(output_token_count, OPENAI_MODEL, input=False)
    logger.info(f"Estimated output cost: ${output_cost:.6f}")

    # Total cost
    total_cost = input_cost + output_cost
    logger.info(f"Total estimated cost: ${total_cost:.6f}")

    # save response to the file with unique name (for logging purposes), utf-8 encoding
    file_name = f"openai_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(result)

    return result
