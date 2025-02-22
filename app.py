import streamlit as st
import os
import csv
import time
import logging
from openai import OpenAI
from pydantic import BaseModel
from typing import List, Literal

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Pydantic models for responses.
class ClassificationResult(BaseModel):
    classification: Literal["yes", "no"]

class ColumnSelectionResult(BaseModel):
    columns: List[str]

client = OpenAI()

# Checkpoint filenames.
ROW_CHECKPOINT_FILE = "checkpoint.txt"
CRITERIA_CHECKPOINT_FILE = "criteria_checkpoint.txt"
OUTPUT_CSV = "accepted_candidates.csv"

def generate_columns_to_use(header: List[str], criteria_prompt: str) -> List[str]:
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
        return selected
    except Exception as e:
        st.error(f"Error during dynamic column selection: {e}")
        return header

def evaluate_candidate(candidate_info: dict, criteria_prompt: str) -> ClassificationResult:
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
            model="gpt-4o-mini",  # Adjust model as needed.
            messages=[
                {"role": "developer", "content": "You are an expert candidate evaluator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format=ClassificationResult
        )
        result = completion.choices[0].message.parsed
        return result
    except Exception as e:
        st.error(f"Error during candidate evaluation: {e}")
        return ClassificationResult(classification="no")

def load_row_checkpoint() -> int:
    if os.path.exists(ROW_CHECKPOINT_FILE):
        try:
            with open(ROW_CHECKPOINT_FILE, "r", encoding="utf-8") as cp_file:
                index = int(cp_file.read().strip())
                return index
        except Exception as e:
            st.error(f"Error reading row checkpoint: {e}")
    return 0

def update_row_checkpoint(index: int):
    try:
        with open(ROW_CHECKPOINT_FILE, "w", encoding="utf-8") as cp_file:
            cp_file.write(str(index))
    except Exception as e:
        st.error(f"Error updating row checkpoint: {e}")

def load_criteria_checkpoint() -> str:
    if os.path.exists(CRITERIA_CHECKPOINT_FILE):
        try:
            with open(CRITERIA_CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                criteria = f.read().strip()
                return criteria
        except Exception as e:
            st.error(f"Error reading criteria checkpoint: {e}")
    return ""

def update_criteria_checkpoint(criteria: str):
    try:
        with open(CRITERIA_CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            f.write(criteria)
    except Exception as e:
        st.error(f"Error updating criteria checkpoint: {e}")

# Initialize session state for processing.
if "processing_running" not in st.session_state:
    st.session_state.processing_running = False

def toggle_processing():
    st.session_state.processing_running = not st.session_state.processing_running
    try:
        st.experimental_rerun()  # Rerun to immediately reflect the updated state.
    except AttributeError:
        st.write("Please refresh the page to see the updated state.")



st.title("Candidate Evaluation App")

uploaded_file = st.file_uploader("Upload CSV File", type="csv")
loaded_criteria = load_criteria_checkpoint()
criteria_prompt = st.text_input("Enter Candidate Evaluation Criteria", value=loaded_criteria)
if criteria_prompt:
    update_criteria_checkpoint(criteria_prompt)

progress_placeholder = st.empty()            # Progress bar above logs
# UI placeholders.
log_area = st.empty()                        # Logs area
download_placeholder = st.empty()            # Download button above logs

# Option to resume from row checkpoint.
resume_option = False
if os.path.exists(ROW_CHECKPOINT_FILE):
    resume_option = st.checkbox("Resume from latest row checkpoint", value=True)

# Toggle button for processing.
toggle_label = "Stop Processing" if st.session_state.processing_running else "Start Processing"
st.button(toggle_label, on_click=toggle_processing)


# Collapsed expander for selected columns.
with st.expander("Selected Columns for Evaluation (click to expand)", expanded=False):
    selected_columns_placeholder = st.empty()

if uploaded_file and criteria_prompt:
    csv_data = uploaded_file.getvalue().decode("utf-8").splitlines()
    csv_reader = csv.reader(csv_data)
    header = next(csv_reader)
    header_mapping = {col: idx for idx, col in enumerate(header)}
    data_rows = list(csv_reader)
    total_rows = len(data_rows)

    # Get dynamic columns for evaluation.
    selected_columns = generate_columns_to_use(header, criteria_prompt)
    dynamic_columns = [col for col in selected_columns if col in header_mapping]
    if not dynamic_columns:
        st.warning("No dynamic columns selected; defaulting to full header.")
        dynamic_columns = header
    selected_columns_placeholder.write(dynamic_columns)

    # Determine starting index.
    start_index = load_row_checkpoint() if resume_option else 0

    # Prepare output CSV: if not resuming, write header.
    mode = "a" if resume_option and os.path.exists(OUTPUT_CSV) else "w"
    if mode == "w":
        with open(OUTPUT_CSV, mode, newline="", encoding="utf-8") as fout:
            writer = csv.writer(fout)
            writer.writerow(header)

    progress_bar = progress_placeholder.progress(0)
    accepted_candidates = []

    if st.session_state.processing_running:
        for i in range(start_index, total_rows):
            if not st.session_state.processing_running:
                break

            row = data_rows[i]
            if not row:
                continue

            candidate_info = {col: row[header_mapping[col]] for col in dynamic_columns}
            decision = evaluate_candidate(candidate_info, criteria_prompt)
            if decision.classification == "yes":
                log_area.text(f"Candidate {i} kept.")
                with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as fout:
                    writer = csv.writer(fout)
                    writer.writerow(row)
                accepted_candidates.append(row)
            else:
                log_area.text(f"Candidate {i} rejected.")

            update_row_checkpoint(i + 1)
            progress_bar.progress((i + 1) / total_rows)
            time.sleep(0.1)

        if st.session_state.processing_running and (i + 1) >= total_rows:
            st.session_state.processing_running = False
            if os.path.exists(ROW_CHECKPOINT_FILE):
                os.remove(ROW_CHECKPOINT_FILE)
                st.info("All candidates processed. Row checkpoint removed.")

    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "r", encoding="utf-8") as f:
            csv_content = f.read()
        download_placeholder.download_button("Download Accepted Candidates CSV", csv_content, file_name=OUTPUT_CSV)

    st.info(f"Processed {load_row_checkpoint()} out of {total_rows} rows.")
