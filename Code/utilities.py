import os
import pandas as pd

# Parse the Document_ID to extract TopicID and PromptsID
def parse_document_id(doc_id):
    """Parse Document_ID into components."""
    #document_id = f"AR_{language_model}_{topic_id}_{prompt_id}_{essay_id}"
    parts = doc_id.split("_")
    return {"LanguageModel": parts[1], "TopicID": parts[2], "PromptsID": parts[3],"EssayID": parts[4]}



