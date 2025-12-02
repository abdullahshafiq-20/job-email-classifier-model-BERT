# Email Classifier - Prediction Guide

This guide explains how to use the trained BERT model to classify new emails as job-related or not.

## Files

- **`predict.py`**: Main prediction module with `EmailClassifier` class
- **`predict_simple.py`**: Simple example script for quick predictions
- **`best_bert_email_classifier.pth`**: Trained model weights

## Quick Start

### Option 1: Interactive Mode

Run the main prediction script with built-in test examples and interactive mode:

```bash
python predict.py
```

This will:
1. Load the trained model
2. Test on 5 sample emails
3. Enter interactive mode where you can paste your own emails

### Option 2: Simple Script

Edit `predict_simple.py` with your email content and run:

```bash
python predict_simple.py
```

### Option 3: Use in Your Code

```python
from predict import EmailClassifier

# Initialize classifier
classifier = EmailClassifier(model_path='best_bert_email_classifier.pth')

# Classify a single email
email = "We are hiring a Python developer..."
result = classifier.predict(email)

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2f}%")
print(f"Job probability: {result['probabilities']['job']:.2f}%")

# Classify multiple emails
emails = ["email1...", "email2...", "email3..."]
results = classifier.predict_batch(emails)

for i, result in enumerate(results):
    print(f"Email {i+1}: {result['prediction']} ({result['confidence']:.1f}%)")
```

## EmailClassifier Class

### Methods

#### `__init__(model_path, max_length=128)`
Initialize the classifier with a trained model.

**Parameters:**
- `model_path` (str): Path to saved model weights (default: 'best_bert_email_classifier.pth')
- `max_length` (int): Maximum token length for BERT (default: 128)

#### `predict(text)`
Classify a single email.

**Parameters:**
- `text` (str): Email content to classify

**Returns:**
```python
{
    'prediction': 'job' or 'not_job',
    'confidence': float (0-100),
    'probabilities': {
        'not_job': float (0-100),
        'job': float (0-100)
    },
    'label': int (0 or 1)
}
```

#### `predict_batch(texts)`
Classify multiple emails at once.

**Parameters:**
- `texts` (list): List of email contents

**Returns:**
- List of prediction dictionaries

#### `print_prediction(text, result=None)`
Print formatted prediction results.

**Parameters:**
- `text` (str): Email content
- `result` (dict): Prediction result (optional, will compute if not provided)

## Examples

### Example 1: Job Email

```python
email = """
Subject: Senior Data Scientist Position

We are looking for a Senior Data Scientist with 5+ years experience.
Please send your CV to careers@company.com
"""

result = classifier.predict(email)
# Output: {'prediction': 'job', 'confidence': 98.5, ...}
```

### Example 2: Non-Job Email

```python
email = """
Subject: Dinner Plans

Hey! Want to grab dinner this weekend?
Let me know!
"""

result = classifier.predict(email)
# Output: {'prediction': 'not_job', 'confidence': 99.2, ...}
```

### Example 3: Batch Processing

```python
emails = [
    "We are hiring developers...",
    "Meeting at 3 PM tomorrow...",
    "Apply for our internship program..."
]

results = classifier.predict_batch(emails)

for email, result in zip(emails, results):
    print(f"{email[:30]}... -> {result['prediction']}")
```

## Requirements

Make sure you have the required packages installed:

```bash
pip install torch transformers
```

Or install from requirements.txt:

```bash
pip install -r requirements.txt
```

## Model Details

- **Architecture**: BERT (bert-base-uncased)
- **Task**: Binary classification (job vs not_job)
- **Max Length**: 128 tokens
- **Classes**: 
  - 0: not_job
  - 1: job

## Troubleshooting

### Model file not found
Make sure `best_bert_email_classifier.pth` is in the same directory as `predict.py`, or provide the full path:

```python
classifier = EmailClassifier(model_path='/path/to/best_bert_email_classifier.pth')
```

### CUDA out of memory
If you get CUDA memory errors, the model will automatically fall back to CPU. You can also force CPU usage:

```python
import torch
torch.device('cpu')
```

### Low confidence predictions
If the model gives low confidence (<70%), the email might be ambiguous. Consider:
- Checking if the email has enough content
- Verifying the email is in English
- Looking at both probability scores to understand the uncertainty

## Performance

The model was trained with class weights to handle imbalanced data and achieves:
- High accuracy on job-related emails
- Low false positive rate
- Robust performance on various email formats

For detailed performance metrics, see the training output in `model.py`.
