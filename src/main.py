import os
from typing import List, Dict, Optional
from pygments.lexers import guess_lexer_for_filename, get_lexer_by_name
from pygments.util import ClassNotFound

from logging_config import setup_logger
from src.project_structure import get_tree_structure
from src.prompts import (
    ASSISTANCE_TYPES,
    ROLES,
    PROMPT_PARAMETERS,
    STANDARD_NOTES,
    STANDARD_INFORMATION,
    PROJECT_FILES_PROMPT,
    PROJECT_STRUCTURE_PROMPT,
    ROLE_PROMPT,
)
from src.openai_api_handler import process_and_analyze_file

logger = setup_logger(__name__)
IGNORED_FILE_PATTERNS = [
    ".git",
    "__pycache__",
    ".idea",
    ".ipynb_checkpoints",
    ".env",
    "environment.yml",
    "final_prompt.md",
]

IGNORED_DIRECTORIES = [
    ".git",  # Ignore the entire .git directory
    "__pycache__",  # Ignore Python cache directories
    ".idea",  # Ignore IDE project settings
]


def get_user_input(prompt: str) -> str:
    """
    Helper function to get sanitized input from the user.
    """
    return input(prompt).strip()


def get_files_from_directory(directory: str) -> List[str]:
    """
    Recursively gets all files from the given directory.

    :param directory: The directory path to scan.
    :return: A list of file paths.
    """
    all_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files


def get_code_type_from_extension(file_name: str) -> Optional[str]:
    """
    Determines the appropriate code type for a markdown code block based on the file name.

    :param file_name: Name of the file (with extension).
    :return: Code block type for markdown or None if not detectable.
    """
    try:
        # Attempt to guess the lexer by the file name
        lexer = guess_lexer_for_filename(file_name, "")
        return lexer.name.lower().replace(" ", "-")  # Return normalized code type
    except ClassNotFound:
        return None


def should_ignore_file(file_path: str) -> bool:
    """
    Determines whether a file should be ignored based on predefined patterns.

    :param file_path: Full path to the file.
    :return: True if the file should be ignored, False otherwise.
    """
    file_name = os.path.basename(file_path)
    return any(pattern in file_name for pattern in IGNORED_FILE_PATTERNS)


def is_in_ignored_directory(file_path: str) -> bool:
    """
    Determines whether a file is inside an ignored directory.

    :param file_path: Full path to the file.
    :return: True if the file is within an ignored directory, False otherwise.
    """
    directory_path = os.path.dirname(file_path)
    directory_parts = directory_path.split(os.sep)
    return any(ignored_dir in directory_parts for ignored_dir in IGNORED_DIRECTORIES)


def extract_file_content(files: List[str]) -> List[Dict[str, str]]:
    """
    Extracts the content of text-based files and formats it as markdown.

    :param files: List of file paths.
    :return: List of dictionaries containing file metadata and content.
    """
    file_data = []

    for file in files:
        if should_ignore_file(file):
            logger.info(f"Skipping ignored file: {file}")
            continue

        if is_in_ignored_directory(file):
            logger.info(f"Skipping file in ignored directory: {file}")
            continue

        code_type = None
        try:
            # Use Pygments to guess the code type
            code_type = guess_lexer_for_filename(file, "").name.lower().replace(" ", "-")
        except ClassNotFound:
            logger.info(f"No matching lexer found for file: {file}")

        try:
            with open(file, "r", encoding="utf-8") as f:
                content = f.read().strip()
            relative_path = os.path.relpath(file)  # Relative path from the project directory
            if content:
                file_data.append(
                    {
                        "file_name": relative_path,
                        "code_type": code_type,
                        "content": content,
                    }
                )
            else:
                logger.info(f"File {file} is empty.")
        except Exception as e:
            logger.warning(f"Could not read file {file}: {e}")

    return file_data


def display_options(options: List[str]) -> None:
    """
    Display a numbered list of options.

    :param options: List of options to display.
    """
    for i, option in enumerate(options, start=1):
        print(f"{i}. {option}")


def ask_assistant_role() -> str:
    """
    Ask the user for the role of the assistant.

    :return: The selected role.
    """
    print("Choose the assistant's role:")
    display_options(ROLES)
    while True:
        try:
            choice = int(get_user_input("Enter the number of the role: "))
            if 1 <= choice <= len(ROLES):
                return ROLES[choice - 1]
            else:
                print(f"Please choose a number between 1 and {len(ROLES)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def ask_project_directory() -> str:
    """
    Ask the user to provide the project directory path.

    :return: Project structure representation.
    """
    while True:
        directory = get_user_input("Enter the project directory path: ")
        root_directory = os.path.abspath(directory)
        if os.path.isdir(directory):
            tree = get_tree_structure(root_directory)
            return "\n".join(tree)
        else:
            print("Invalid directory. Please try again.")


def ask_directory_or_files() -> List[str]:
    """
    Ask the user to provide either a directory path or a list of files.

    :return: A list of file paths.
    """
    choice = get_user_input(
        "Do you want to provide a directory (d) or a list of files (f)? (d/f): "
    ).lower()

    if choice == "d":
        directory = get_user_input("Enter the directory path: ")
        if os.path.isdir(directory):
            return get_files_from_directory(directory)
        else:
            print("Invalid directory. Please try again.")
    elif choice == "f":
        files = get_user_input("Enter the file paths, separated by commas: ").split(",")
        valid_files = [file.strip() for file in files if os.path.isfile(file.strip())]
        if not valid_files:
            print("No valid files found. Please try again.")
        return valid_files
    else:
        print("Invalid choice. Please try again.")


def ask_assistance_type() -> str:
    """
    Ask the user for the type of assistance dynamically based on predefined options.

    :return: The selected assistance type's corresponding prompt.
    """

    print("Choose the type of assistance:")
    assistance_list = list(ASSISTANCE_TYPES.keys())

    # Display numbered options dynamically
    for idx, key in enumerate(assistance_list, start=1):
        # Replace underscores with spaces and capitalize each word for readability
        print(f"{idx}. {key.replace('_', ' ').title()}")

    while True:
        try:
            choice = int(get_user_input("Enter the number corresponding to the assistance type: "))
            if 1 <= choice <= len(assistance_list):
                selected_key = assistance_list[choice - 1]
                print(f"Selected Assistance Type: {selected_key.replace('_', ' ').title()}")
                return ASSISTANCE_TYPES[selected_key]
            else:
                print(
                    f"Invalid choice. Please select a number between 1 and {len(assistance_list)}."
                )
        except ValueError:
            print("Invalid input. Please enter a number.")


def ask_prompt_parameters(prompt_template: str) -> dict:
    """
    Ask the user for additional parameters required by the selected prompt.

    :param prompt_template: The selected prompt template.
    :return: A dictionary with parameter names and user-provided values.
    """
    parameters = PROMPT_PARAMETERS.get(prompt_template, [])
    parameter_values = {}

    for param in parameters:
        # Convert parameter name to a readable question format
        question = param.replace("_", " ").capitalize()
        value = get_user_input(f"Provide the {question}: ")
        parameter_values[param] = value

    return parameter_values


def build_final_prompt(
    role: str,
    project_structure: str,
    project_files: str,
    prompt_template: str,
    additional_params: dict,
) -> list[str, str]:
    """
    Build the final prompt by incorporating the role, project structure, and files into the standardized sections,
    and appending the specific prompt template populated with additional parameters.

    :param role: The role of the assistant.
    :param project_structure: The project's structure.
    :param project_files: The list of project files.
    :param prompt_template: The selected prompt template.
    :param additional_params: Additional parameters for the prompt.
    :return: The final, populated prompt string.
    """
    # Fill the role, project structure, and files templates
    role_prompt_filled = ROLE_PROMPT.format(role=role)
    project_structure_prompt_filled = PROJECT_STRUCTURE_PROMPT.format(
        project_structure=project_structure
    )
    project_files_prompt_filled = PROJECT_FILES_PROMPT.format(project_files=project_files)

    standard_info_filled = STANDARD_INFORMATION.format(
        PROJECT_STRUCTURE_PROMPT=project_structure_prompt_filled,
        PROJECT_FILES_PROMPT=project_files_prompt_filled,
        STANDARD_NOTES=STANDARD_NOTES,
    )

    standard_context = {
        "STANDARD_INFORMATION": standard_info_filled,
        "STANDARD_NOTES": STANDARD_NOTES,
    }

    # Combine the standard context and additional parameters for the assistance-specific prompt
    combined_context = {**standard_context, **additional_params}

    # Fill the assistance-specific prompt
    final_prompt = prompt_template.format(**combined_context)

    return role_prompt_filled, final_prompt


def get_all_file_contents(files: List[str]) -> str:
    file_contents = extract_file_content(files)
    all_contents_str = ""
    for file_data in file_contents:
        all_contents_str += f"\n{file_data['file_name']}\n"
        if file_data["code_type"]:
            all_contents_str += f"```{file_data['code_type']}\n"
        else:
            all_contents_str += "```\n"
        all_contents_str += file_data["content"]
        all_contents_str += "\n```\n"
    return all_contents_str


def main():
    """
    Main function to gather input from the user and generate the final prompt.
    """
    print("=== Assistant Configuration ===")

    role = ask_assistant_role()
    logger.info(f"Selected Role: {role}")

    project_structure = ask_project_directory()
    logger.info(f"Project Structure:\n{project_structure}")

    files = ask_directory_or_files()
    logger.info(f"Selected Files:\n{files}")

    file_contents = get_all_file_contents(files)

    assistance_prompt = ask_assistance_type()

    additional_params = ask_prompt_parameters(assistance_prompt)

    system_contex, user_context = build_final_prompt(
        role=role,
        project_structure=project_structure,
        project_files=file_contents,
        prompt_template=assistance_prompt,
        additional_params=additional_params,
    )

    logger.info(f"System Context:\n{system_contex}")
    logger.info(f"User Context:\n{user_context}")

    # save final prompt as markdown file, utf-8 encoded
    with open("final_prompt.md", "w", encoding="utf-8") as f:
        f.write(system_contex)
        f.write(user_context)

    print("=== Configuration Complete ===")

    chatgpt_result = process_and_analyze_file(system_contex, user_context)
    print(chatgpt_result)


if __name__ == "__main__":
    main()
