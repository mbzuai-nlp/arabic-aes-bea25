import pandas as pd
import json
import random
from langchain.prompts import PromptTemplate
import openai
import os
from transformers import pipeline
import utilities
from typing import Dict



# Set your OpenAI API key
openai.api_key = "ENTER YOUR API KEY"

engine ="gpt-4o"
client = openai.OpenAI(
    api_key=openai.api_key
)

def load_error_prompts(file_path="error_prompts.json"):
    """
    Reads the error prompts from a JSON file and returns them as a dictionary.
    
    Parameters:
        file_path (str): Path to the JSON file containing error prompts.

    Returns:
        dict: A dictionary where keys are error types, and values are lists of prompts.
    """
    with open(file_path, 'r') as file:
        error_prompts = json.load(file)
    return error_prompts


def load_error_distribution(file_path: str) -> pd.DataFrame:
    """
    Loads error distribution data from a CSV file and ensures all numeric columns are cast to float.

    Parameters:
        file_path (str): Path to the CSV file containing error distribution data.

    Returns:
        pd.DataFrame: A DataFrame containing CEFR levels and their corresponding error type distributions.
    """
    # Columns to extract from the CSV
    columns = [
        "CEFR", "DELETE", "MERGE-B", "MERGE-I", "REPLACE_M", "REPLACE_M+REPLACE_O", 
        "REPLACE_O", "REPLACE_O+REPLACE_X", "REPLACE_P", "REPLACE_S", "REPLACE_X", "SPLIT"
    ]

    # Load and enforce column types
    df = pd.read_csv(file_path, usecols=columns)

    # Convert all columns except 'CEFR' to float
    float_columns = [col for col in columns if col != "CEFR"]
    df[float_columns] = df[float_columns].astype(float)

    return df




def get_error_distribution_for_cefr(
    cefr: str,
    error_distribution_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Retrieves the error distribution weights for a given CEFR level.

    CEFR level is adjusted as follows:
    - 'A1' is treated as 'A2'
    - 'C2' is treated as 'C1'

    Parameters:
        cefr (str): The CEFR level for which to retrieve the error distribution.
        error_distribution_df (pd.DataFrame): A DataFrame containing error type distributions per CEFR level.

    Returns:
        Dict[str, float]: A dictionary mapping error types to their corresponding weights.

    Raises:
        ValueError: If no matching CEFR level is found in the DataFrame.
    """
    # Normalize CEFR level
    cefr = {'A1': 'A2', 'C2': 'C1'}.get(cefr, cefr)

    # Filter DataFrame for the specified CEFR level
    cefr_row = error_distribution_df[error_distribution_df["CEFR"] == cefr]

    if cefr_row.empty:
        raise ValueError(f"No distribution found for CEFR level: {cefr}")

    # Convert error type columns to dictionary (skip CEFR column)
    return cefr_row.iloc[0, 1:].to_dict()



def select_prompt_for_error(
    error_type: str,
    num_words_to_inject: int,
    error_prompts_dict: Dict[str, list]
) -> str:
    """
    Selects a random prompt or a combination of prompts for the specified error type,
    based on the number of words to inject.

    Parameters:
        error_type (str): The error type for which a prompt is needed.
        num_words_to_inject (int): Number of words to inject (influences number of prompts).
        error_prompts_dict (Dict[str, list]): Dictionary mapping error types to a list of prompts.

    Returns:
        str: A combined prompt string tailored to the error type and word count.
    """
    # Default prompt if error type is not in dictionary
    prompts = error_prompts_dict.get(error_type, [f"Introduce {error_type} errors."])

    # Determine how many prompts to combine
    if num_words_to_inject < 5:
        num_prompts = 1
    elif num_words_to_inject < 10:
        num_prompts = 2
    else:
        num_prompts = 3

    # Randomly select and join prompts
    selected_prompts = random.choices(prompts, k=num_prompts)
    return " Or, ".join(selected_prompts)




def inject_errors_sequentially(
    essay: str,
    cefr: str,
    cefr_error_distribution: Dict[str, float],
    error_prompts_dict: Dict[str, list]
) -> str:
    """
    Injects errors into an essay by applying each error type in the CEFR-specific error distribution.

    Parameters:
        essay (str): The original essay text.
        cefr (str): CEFR level of the essay (e.g., 'A2', 'B1', etc.).
        cefr_error_distribution (Dict[str, float]): Error types and their corresponding weights.
        error_prompts_dict (Dict[str, list]): Mapping of error types to lists of error prompt templates.
        

    Returns:
        str: The final essay with injected errors applied sequentially.
    """
    for error_type, weight in cefr_error_distribution.items():
        # Skip unsupported or zero-weight error types
        if error_type == "MERGE-I" or weight == 0:
            continue

        # Round up minimal errors
        num_words_to_inject = 1 if 0 < weight < 0.5 else round(weight)

        # Normalize error type if needed
        if error_type in {"MERGE-B", "MERGE-I"}:
            error_type = "MERGE"

        # Select error prompt(s)
        error_prompt = select_prompt_for_error(error_type, num_words_to_inject, error_prompts_dict)

        # Compose instruction and GPT prompt
        instruction = f"Please inject exactly {num_words_to_inject} error(s) of this type: {error_prompt}"
        gpt_prompt = (
            f"Rewrite the following Arabic essay with the {cefr} CEFR level, following the specified error instruction "
            f"without removing, changing, or fixing any existing mistakes.\n\n"
            f"Essay in Arabic:\n{essay}\n\n"
            f"Error Instruction: {instruction}\n\n"
            "Only apply the specified errors directly in the text without any introductory or additional comments."
        )

        # Format messages for Chat API
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that injects errors into Arabic essays according to detailed instructions."
            },
            {
                "role": "user",
                "content": gpt_prompt
            }
        ]

        # Call GPT API
        response = client.chat.completions.create(
            model=engine,
            messages=messages,
            max_tokens=len(essay) + 100,
            temperature=0.7
        )

        # Update essay with GPT's response
        essay = response.choices[0].message.content.strip()

    return essay





def generate_error_essay(
    original_essay: str,
    cefr: str,
    error_distribution_df: pd.DataFrame,
    error_prompts_dict: Dict[str, list]
) -> str:
    """
    Generates an error-injected essay by applying errors based on the CEFR-specific distribution and prompts.

    Parameters:
        original_essay (str): The original clean essay text.
        cefr (str): CEFR level of the essay (e.g., 'A2', 'B1', etc.).
        error_distribution_df (pd.DataFrame): DataFrame containing CEFR-level error type distributions.
        error_prompts_dict (Dict[str, list]): Dictionary mapping error types to prompt templates.

    Returns:
        str: The essay with injected grammatical or structural errors.
    """
    # Retrieve error distribution weights for the specified CEFR level
    error_weights = get_error_distribution_for_cefr(cefr, error_distribution_df)

    # Generate the erroneous essay using those weights and prompts
    return inject_errors_sequentially(original_essay, cefr, error_weights, error_prompts_dict)





def main():
    # === Paths ===
    data_path = "../Data/"
    error_path = '../Data/Errors_annotation/'

    original_essay_file = os.path.join(data_path, "arwi_original_essays.csv")
    cefr_level_file = os.path.join(data_path, "arwi_cefr_levels.csv")
    error_distribution_file = os.path.join(error_path, "error_distribution_CEFR.csv")
    error_prompts_file = os.path.join(error_path, "FULL_error_prompts.json")
    output_path = os.path.join(data_path, "arwi_gpt_error_essays2.csv")

    # === Step 1: Load Essays and CEFR Levels ===
    essay_df = utilities.read_essays(original_essay_file, cefr_level_file, essay_col="Original_Essay")
    print("1. Data Loaded.")

    # === Step 2: Load Error Prompts ===
    error_prompts_dict = load_error_prompts(error_prompts_file)
    print("2. Error Prompts Loaded.")

    # === Step 3: Load CEFR Error Distribution ===
    error_distribution_df = load_error_distribution(error_distribution_file)
    print("3. Error Distribution Loaded.")

    # === Step 4: Prepare Output File ===
    if not os.path.exists(output_path):
        pd.DataFrame(columns=["Document_ID", "GPT_Error_Essay"]).to_csv(output_path, index=False)

    # === Step 5: Generate Error-Injection Essays ===
    print("4. Start Error Injection.")
    for index, row in essay_df.iterrows():
        document_id = row["Document_ID"]
        original_essay = row["Essay"]
        cefr_level = row["CEFR"]

        try:
            error_essay = generate_error_essay(
                original_essay=original_essay,
                cefr=cefr_level,
                error_distribution_df=error_distribution_df,
                error_prompts_dict=error_prompts_dict
            )

            # Append result
            pd.DataFrame([{
                "Document_ID": document_id,
                "GPT_Error_Essay": error_essay
            }]).to_csv(output_path, mode='a', header=False, index=False)

            print(f"✓ Generated Error Essay for Document {document_id}")

        except Exception as e:
            print(f"✗ Failed to generate for Document {document_id}: {e}")

    print(f"\n✔ All erroneous essays saved to: {output_path}")


# Entry point
if __name__ == "__main__":
    main()