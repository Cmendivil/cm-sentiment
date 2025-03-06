# FastAPI Sentiment Analysis

This project is a lightweight Sentiment Analysis API built using **FastAPI** and **NLTK**. It provides an endpoint to analyze the sentiment of a given text, categorizing it as **Positive, Negative, or Neutral** using VADER (Valence Aware Dictionary and sEntiment Reasoner).

## Installation

### Prerequisites

- Python 3.8 or later
- Virtual environment (optional but recommended)

### Steps

1. **Clone the repository**

   ```sh
   git clone <repository_url>
   cd <project_directory>
   ```

2. **Create and activate a virtual environment** (recommended)

   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```sh
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Download NLTK VADER Lexicon**

   ```python
   import nltk
   nltk.download('vader_lexicon')
   ```

## Running the API Locally

Once dependencies are installed, run the FastAPI server:

```sh
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at:

- **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Redoc UI:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

## API Endpoints

### **POST /sentiment/**

**Description:** Analyze the sentiment of the input text.

**Request Body:**

```json
{
  "text": "I love this product!"
}
```

**Response:**

```json
{
  "label": "POSITIVE",
  "score": 0.8
}
```

## Deployment

For AWS Lambda deployment, ensure dependencies are packaged correctly and use **Mangum** to serve FastAPI in a Lambda environment.

---

### 🚀 Happy Coding!

