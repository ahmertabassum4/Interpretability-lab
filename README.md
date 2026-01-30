# README for Sentiment Classifier

## Overview
This project is a sentiment classifier that uses a few-shot learning approach to classify movie reviews as either "positive" or "negative." The classifier is built using the Hugging Face Transformers library and a pre-trained causal language model.

## Setup Instructions

### Prerequisites
1. Python 3.8 or higher
2. pip (Python package manager)
3. A GPU (optional but recommended for faster inference)
4. Hugging Face account (for API token)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Interpretability
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Create a `.env` file in the project directory.
   - Add the following variables to the `.env` file:
     ```env
     HF_TOKEN=<your_huggingface_api_token>
     MODEL_NAME=google/gemma-2-2b  # Or any other model of your choice
     OUT_FORMAT=jsonl  # Options: jsonl or csv
     OUT_PATH=imdb_predictions.jsonl  # Output file path
     LIMIT=50  # Number of test samples to process
     SAVE_EVERY=5  # Save results after every N predictions
     ```

5. Download the IMDB dataset (handled automatically by the script).

## Running the Script

1. Activate your virtual environment (if created):
   ```bash
   source venv/bin/activate
   venv\Scripts\activate # On Windows
   ```

2. Run the script:
   ```bash
   python fewshot.py
   ```

3. The script will process the IMDB test dataset and save predictions to the specified output file (e.g., `imdb_predictions.jsonl`).

## Output
- The output file will contain predictions in the specified format (`jsonl` or `csv`).
- Each row includes the following fields:
  - `idx`: Index of the test sample
  - `true_label`: Ground truth sentiment label
  - `pred_label`: Predicted sentiment label
  - `correct`: Whether the prediction matches the ground truth
  - `generated`: Raw generated text from the model

## Notes
- The script uses a few-shot learning approach by providing the model with a small number of labeled examples.
- Ensure that your Hugging Face API token has access to the specified model.
- For better performance, use a GPU-enabled environment.

## Troubleshooting
- If you encounter issues with the Hugging Face API, ensure your token is valid and has the necessary permissions.
- If the script runs out of memory, reduce the `LIMIT` or use a smaller model.

## License
This project is licensed under the MIT License.