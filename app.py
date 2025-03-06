from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from mangum import Mangum
from nltk.sentiment import SentimentIntensityAnalyzer

import nltk

# Ensure VADER lexicon is available
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize the FastAPI app
app = FastAPI(
    title="Sentiment API",
    description="Analyze the sentiment of a given text as positive, negative, or neutral.",
    version="1.0.0"
)
router = APIRouter()
sia = SentimentIntensityAnalyzer()

origins = [
    "http://localhost:3000"
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow specified origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

class TextInput(BaseModel):
    text: str


# Sentiment analysis endpoint
@router.post(
    "/analysis",
    summary="Analyze Sentiment",
    description="Receives a text input and returns its sentiment (positive, negative, or neutral) along with a score.",
    responses={
        200: {
            "description": "Successful sentiment analysis",
            "content": {
                "application/json": {
                    "example": {
                        "label": "POSITIVE",
                        "score": 0.75
                    }
                }
            },
        },
        400: {
            "description": "Bad request - input cannot be empty",
            "content": {
                "application/json": {
                    "example": {"detail": "Text input cannot be empty"}
                }
            },
        },
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "example": {"detail": "Internal Server Error: Some unexpected issue"}
                }
            },
        },
    },
)
async def get_sentiment(text_input: TextInput):
    """
    Analyzes the sentiment of the provided text.

    - **POSITIVE**: If the sentiment score is greater than 0
    - **NEGATIVE**: If the sentiment score is less than 0
    - **NEUTRAL**: If the sentiment score is exactly 0
    """
    try:
        if not text_input.text.strip():
            raise HTTPException(status_code=400, detail="Text input cannot be empty")

        # Get sentiment scores
        sentiment_scores = sia.polarity_scores(text_input.text)

        # Determine sentiment label
        if sentiment_scores['compound'] > 0:
            label = "POSITIVE"
        elif sentiment_scores['compound'] < 0:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"

        return {"label": label, "score": sentiment_scores['compound']}

    except HTTPException as e:
        raise e  # Forward known exceptions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@router.get(
    "/doc",
    summary="API Documentation",
    description="Returns the OpenAPI documentation in JSON format.",
    responses={
        200: {
            "description": "Returns OpenAPI JSON documentation",
        },
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "example": {"detail": "Internal server error: Some unexpected issue"}
                }
            },
        },
    },
)
def doc():
    try:
        if not app.openapi():
            raise HTTPException(status_code=404, detail="Documentation not found")
        return app.openapi()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# Include the sentiment router
app.include_router(router, prefix="/sentiment")

# Mangum handler for AWS Lambda
handler = Mangum(app)
