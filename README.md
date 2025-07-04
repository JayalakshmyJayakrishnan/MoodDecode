# MoodDecode
An NLP-powered FastAPI endpoint that decodes human emotion, flags potential crisis signals, and generates concise summaries, all from raw text.
# MoodDecoding NLP API

## Overview

The MoodDecoding NLP API is a Flask-based application that uses a pre-trained Natural Language Processing (NLP) models to analyze human emotions, identify potential mental health crises, and generate concise summaries of user-provided text. This system is designed for integration into intelligent applications requiring mood interpretation, content condensation, or emotional risk assessment.

The API is powered by state-of-the-art models from Hugging Face Transformers, specifically chosen for their relevance in emotion classification and summarization tasks.



---
**Deployed & accessible at:**  
🔗 **https://01a4-111-92-115-220.ngrok-free.app/docs**

**API Endpoints**
1. POST https://01a4-111-92-115-220.ngrok-free.app/analyze_mood
2. POST https://01a4-111-92-115-220.ngrok-free.app/detect_crisis
3. POST https://01a4-111-92-115-220.ngrok-free.app/summarize

---

## Features

- **Emotion Classification**  
  Predicts the dominant emotion from user input text using a lightweight DistilBERT variant fine-tuned on emotion datasets.

- **Crisis Detection**  
  Identifies high-risk emotional expressions (e.g., sadness, fear, anger) with confidence thresholds for better reliability in safety-sensitive contexts.

- **Text Summarization**  
  Condenses verbose or detailed input into short, informative summaries using a BART-based model.

---

## API Endpoints

### `POST /analyze_mood`

**Description:**  
Predicts the most likely emotion from the given input text.

**Purpose:** Predict the primary emotion expressed in the input text.

**Logic:**  
Applies emotion classification using a fine-tuned DistilBERT model and returns the top-ranked label.


**Request Body:**

```json
{
  "text": "I feel amazing today!"
}
```

**Response:**

```json
{
  "emotion": "joy"
}
```

---

### `POST /detect_crisis`

**Description:**  
Determines whether the text contains crisis-related emotional indicators such as sadness, fear, or anger.

**Purpose:** Detect potential crisis signals in the input.

**Logic:**
Runs emotion classification, then checks if the predicted label is among predefined crisis-associated emotions and if the confidence score is high (> 0.85).

**Request Body:**

```json
{
  "text": "I'm feeling hopeless and might hurt myself"
}
```

**Response:**

```json
{
  "crisis_detected": true
}
```

---

### `POST /summarize`

**Description:**  
Generates a summarized version of the provided text input.

**Purpose:** Generate a condensed summary of the input text.

**Logic:**
Uses a pre-trained BART summarization model with maximum and minimum length constraints to ensure clarity and brevity.
**Request Body:**

```json
{
  "text": "Long paragraph here..."
}
```

**Response:**

```json
{
  "summary": "Condensed version of input..."
}
```

---

## Sample: Analyze Mood

**Input Example:**

![Analyze Mood Input](https://github.com/JayalakshmyJayakrishnan/MoodDecode/blob/main/Analyze%20mood/analyzemood_input.jpg)

**Output Example:**

![Analyze Mood Output](https://github.com/JayalakshmyJayakrishnan/MoodDecode/blob/main/Analyze%20mood/analyzemood_output.jpg)


## Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/mooddecoding-nlp-api.git
cd mooddecoding-nlp-api
```

### Create a Virtual Environment

```bash
python -m venv venv
# Activate the environment
# For Windows:
venv\Scripts\activate
# For macOS/Linux:
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not yet generated, install manually:

```bash
pip install fastapi uvicorn transformers torch
```

---

## Running the API

```bash
uvicorn app:app --reload
```

By default, the application will run at `http://127.0.0.1:8000/`

---

## Models Used

- Emotion Classification & Crisis Detection: `bhadresh-savani/distilbert-base-uncased-emotion`
- Text Summarization: `facebook/bart-large-cnn`

These models are integrated using the Hugging Face `transformers` pipeline abstraction.

---

## File Structure

```
.
├── app.py
├── requirements.txt
└── README.md
```

---

## License

This project is licensed under the MIT License.  
Please cite Hugging Face models if used in academic research or publications.

---

## Citation

If this repository or its components contribute to your research, consider citing the following:

```bibtex
@misc{wolf2020transformers,
  title={Transformers: State-of-the-Art Natural Language Processing},
  author={Thomas Wolf et al.},
  year={2020},
  eprint={1910.03771},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

---

## Contact

For contributions, questions, or issues, please open an issue or contact the maintainer.

## Built By
Created by Jayalakshmy Jayakrishnan, Ayisha Sulaiman, Pavithra Deepu E, and Devika P Sajith to give words to emotions people struggle to say out loud.🤍
