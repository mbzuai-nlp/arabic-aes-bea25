import os
import pandas as pd

# Parse the Document_ID to extract TopicID and PromptsID
def parse_document_id(doc_id):
    """Parse Document_ID into components."""
    #document_id = f"AR_{language_model}_{topic_id}_{prompt_id}_{essay_id}"
    parts = doc_id.split("_")
    return {"LanguageModel": parts[1], "TopicID": parts[2], "PromptsID": parts[3],"EssayID": parts[4]}





#Read the essays and their crossponding CEFR level
def read_essays(
    essays_path="arwi_original_essays.csv",
    levels_path="arwi_cefr_levels.csv",
    essay_col="Essay"
):
    """
    Reads  essays and CEFR levels from two CSV files, merges them,
    and returns a DataFrame with Document_ID, Essay, and CEFR columns.

    Parameters:
        essays_path (str): Path to the CSV file containing  essays.
        levels_path (str): Path to the CSV file containing CEFR levels.
        essay_col (str): Column name in the essays file that contains the essay text.

    Returns:
        pd.DataFrame: A merged DataFrame with columns Document_ID, Essay, and CEFR.
    """
    # Load original essays and rename essay_col to "Essay"
    essays_df = pd.read_csv(essays_path, usecols=["Document_ID", essay_col])
    essays_df = essays_df.rename(columns={essay_col: "Essay"})
    
    # Load CEFR levels
    levels_df = pd.read_csv(levels_path, usecols=["Document_ID", "CEFR"])
    
    # Merge on Document_ID
    merged_df = pd.merge(essays_df, levels_df, on="Document_ID", how="inner")
    
    return merged_df
