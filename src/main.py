import os
from typing import List

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
            return ask_directory_or_files()
    elif choice == "f":
        files = get_user_input("Enter the file paths, separated by commas: ").split(",")
        valid_files = [file.strip() for file in files if os.path.isfile(file.strip())]
        if not valid_files:
            print("No valid files found. Please try again.")
            return ask_directory_or_files()
        return valid_files
    else:
        print("Invalid choice. Please try again.")
        return ask_directory_or_files()


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
) -> str:
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
        ROLE_PROMPT=role_prompt_filled,
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

    return final_prompt


def main():
    """
    Main function to gather input from the user and generate the final prompt.
    """
    print("=== Assistant Configuration ===")

    role = ask_assistant_role()
    print(f"Selected Role: {role}")

    files = ask_directory_or_files()
    print(f"Selected Files:\n{files}")

    assistance_prompt = ask_assistance_type()

    additional_params = ask_prompt_parameters(assistance_prompt)

    project_structure = "Generated project structure representation for demonstration"

    # Generate the final prompt
    final_prompt = build_final_prompt(
        role=role,
        project_structure=project_structure,
        project_files="\n".join(files),
        prompt_template=assistance_prompt,
        additional_params=additional_params,
    )

    print("=== Final Prompt ===")
    print(final_prompt)

    print("=== Configuration Complete ===")


if __name__ == "__main__":
    main()