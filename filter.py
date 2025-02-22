import os
import sys
import csv
import logging
from tqdm import tqdm
from openai import OpenAI
from pydantic import BaseModel
from typing import List, Literal

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Pydantic model for classification result from candidate evaluation.
class ClassificationResult(BaseModel):
    classification: Literal["yes", "no"]

# Pydantic model for the dynamic column selection.
class ColumnSelectionResult(BaseModel):
    columns: List[str]

client = OpenAI()

CHECKPOINT_FILE = "checkpoint.txt"  # File to store checkpoint (row index)

def generate_columns_to_use(header: List[str], criteria_prompt: str) -> List[str]:
    """
    Uses OpenAI to dynamically select which CSV columns to use for candidate evaluation.
    The prompt is built using the CSV header and the candidate evaluation criteria.
    Returns a list of column names.
    """
    header_str = ", ".join(header)
    prompt = (
        f"You are an expert in data extraction and candidate evaluation. Given a CSV with the following columns:\n"
        f"{header_str}\n\n"
        f"And considering the following candidate evaluation criteria:\n{criteria_prompt}\n\n"
        "Please select the most relevant columns for evaluating a candidate. Return your answer as a JSON object "
        "with a key 'columns' whose value is a list of column names (exactly matching the CSV header names). Never pick any urls or ids."
    )
    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "developer", "content": "You are an expert in data extraction."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format=ColumnSelectionResult
        )
        selected = response.choices[0].message.parsed.columns
        logging.info(f"Dynamic column selection result: {selected}")
        return selected
    except Exception as e:
        logging.error(f"Error during dynamic column selection: {e}")
        # Fallback: use the entire header as columns.
        return header

def evaluate_candidate(candidate_info: dict, criteria_prompt: str) -> ClassificationResult:
    """
    Combines selected candidate data (provided as a dictionary mapping column names to values)
    into a prompt and uses the ChatGPT API to evaluate if the candidate meets the criteria.
    """
    candidate_data = "\n".join([f"{key}: {value}" for key, value in candidate_info.items()])
    full_data = f"Candidate Data:\n{candidate_data}\n\n"
    
    prompt = (
        f"Please evaluate the candidate based on the following criteria:\n{criteria_prompt}\n\n"
        "Using all the information provided (candidate data), "
        "respond with 'yes' if the candidate meets the criteria and 'no' if they do not. "
        "Also include a brief explanation.\n\n"
        f"{full_data}"
    )
    
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",  # Adjust the model if needed.
            messages=[
                {"role": "developer", "content": "You are an expert candidate evaluator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format=ClassificationResult
        )
        result = completion.choices[0].message.parsed
        logging.debug(f"Evaluation result: {result}")
        return result
    except Exception as e:
        logging.error(f"Error during candidate evaluation: {e}")
        # On error, return a default "no" classification.
        return ClassificationResult(classification="no")

def load_checkpoint() -> int:
    """
    Reads the checkpoint file (if it exists) and returns the row index to resume from.
    """
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r") as cp_file:
                index = int(cp_file.read().strip())
                logging.info(f"Resuming from checkpoint at row index: {index}")
                return index
        except Exception as e:
            logging.error(f"Error reading checkpoint file: {e}")
    return 0

def update_checkpoint(index: int):
    """
    Writes the current row index to the checkpoint file.
    """
    try:
        with open(CHECKPOINT_FILE, "w") as cp_file:
            cp_file.write(str(index))
    except Exception as e:
        logging.error(f"Error updating checkpoint file: {e}")

def main():
    if len(sys.argv) < 3:
        logging.error("Usage: python script.py <input_csv_file> <criteria_prompt>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    criteria_prompt = sys.argv[2]
    output_file = "accepted_candidates.csv"
    
    if "OPENAI_API_KEY" not in os.environ:
        logging.error("Error: OPENAI_API_KEY not set in environment.")
        sys.exit(1)
    
    # Open the CSV file and read the header and rows.
    with open(input_file, "r", newline="", encoding="utf-8") as fin:
        reader = csv.reader(fin)
        header = next(reader, None)
        if header is None:
            logging.error("Error: CSV file is empty or missing header.")
            sys.exit(1)
        header_mapping = {col: idx for idx, col in enumerate(header)}
        rows = list(reader)
    
    # Use OpenAI to dynamically generate the list of columns to use (only once).
    selected_columns = generate_columns_to_use(header, criteria_prompt)
    dynamic_columns = [col for col in selected_columns if col in header_mapping]
    if not dynamic_columns:
        logging.warning("No dynamic columns selected; defaulting to full header.")
        dynamic_columns = header
    logging.info(f"Using the following columns for evaluation: {dynamic_columns}")
    
    start_index = load_checkpoint()
    remaining_rows = rows[start_index:]
    
    # Determine mode for output file: if resuming, append; otherwise, write new file.
    file_mode = "a" if start_index > 0 and os.path.exists(output_file) else "w"
    with open(output_file, file_mode, newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        # If starting fresh, write header.
        if file_mode == "w":
            writer.writerow(header)
        
        for idx, row in enumerate(tqdm(remaining_rows, total=len(remaining_rows), desc="Evaluating candidates"), start=start_index):
            if not row:
                continue
            candidate_info = {col: row[header_mapping[col]] for col in dynamic_columns}
            decision = evaluate_candidate(candidate_info, criteria_prompt)
            if decision.classification == "yes":
                writer.writerow(row)
                logging.info(f"Row {idx}: Candidate kept.")
            else:
                logging.info(f"Row {idx}: Candidate filtered out.")
            # Update checkpoint after each processed row.
            update_checkpoint(idx + 1)
    
    # Processing complete; remove checkpoint file.
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        logging.info("Processing complete. Checkpoint file removed.")

if __name__ == "__main__":
    main()
