import tiktoken


def calculate_token_count(messages, model="gpt-3.5-turbo"):
    """
    Calculate the token count for the given messages based on the model's tokenizer.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Use a default tokenizer if the model is unknown
        encoding = tiktoken.get_encoding("cl100k_base")

    total_tokens = 0
    for message in messages:
        # Add token count for each message role and content
        total_tokens += len(encoding.encode(message["role"])) + len(
            encoding.encode(message["content"])
        )
        # System-defined separators also consume tokens (e.g., `role`, `content`)
        total_tokens += 2  # Assumes separators for role/content

    return total_tokens


def calculate_price(tokens, model, input=True):
    """
    Calculate the expected price based on the number of tokens and the model.
    """
    # Define pricing per 1K tokens (different for input and output)
    pricing = {
        "gpt-4o": {"input": 0.0025, "output": 0.01000},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-3.5-turbo": {"input": 0.003, "output": 0.006},
    }

    # Find the most expensive input and output prices
    max_input_price = max(p["input"] for p in pricing.values())
    max_output_price = max(p["output"] for p in pricing.values())

    # Use the model's pricing if available; otherwise, fall back to the most expensive option
    price_per_token = pricing.get(model, {"input": max_input_price, "output": max_output_price})
    cost_per_token = price_per_token["input"] if input else price_per_token["output"]

    return (tokens / 1000) * cost_per_token


def estimate_total_price(messages, model="gpt-3.5-turbo", output_tokens=None):
    """
    Estimate total cost for both input and output tokens with separate pricing.
    """
    # Calculate input token count
    input_tokens = calculate_token_count(messages, model)

    # Assume output tokens are provided or estimate them as equal to input tokens if not
    if output_tokens is None:
        output_tokens = input_tokens  # Assumption: output size equals input size

    # Calculate prices
    input_price = calculate_price(input_tokens, model, input=True)
    output_price = calculate_price(output_tokens, model, input=False)

    # Total price
    total_price = input_price + output_price

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_price": input_price,
        "output_price": output_price,
        "total_price": total_price,
    }
