from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
from transformers import pipeline

# Initialize the FastAPI app
app = FastAPI()
router = APIRouter()
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
# Load the pre-trained sentiment analysis model from Hugging Face
sentiment_analyzer = pipeline("sentiment-analysis", model=model_name)

# Define a Pydantic model for request validation
class TextRequest(BaseModel):
    text: str

# Create the sentiment analysis endpoint
@app.post("/analyze_sentiment")
def analyze_sentiment(request: TextRequest):
    # Perform sentiment analysis on the provided text
    try:
        result = sentiment_analyzer(request.text)
        return {"sentiment": result[0]['label'], "confidence": result[0]['score']}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing request: {e}")


@app.get("/doc")
def doc():
    return app.__doc__

app.include_router(router, prefix="/sentiment")