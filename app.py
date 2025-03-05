from fastapi import FastAPI, HTTPException, APIRouter
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
app = FastAPI()
router = APIRouter()
sia = SentimentIntensityAnalyzer()

class TextInput(BaseModel):
    text: str


# nltk
@router.post("/analysis")
async def get_sentiment(text_input: TextInput):
    try:
        # Validate input
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
        # Catch unexpected errors
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")



@router.get("/doc")
def doc():
    try:
        # This is a simple endpoint to return the app's docstring (documentation)
        if not app.__doc__:
            raise HTTPException(status_code=404, detail="Documentation not found")
        return app.__doc__

    except Exception as e:
        # Catch any errors that occur while accessing the documentation
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# Include the sentiment router (though it's currently not used here as the app already has routes)
app.include_router(router, prefix="/sentiment")

# Mangum handler for AWS Lambda
handler = Mangum(app)
