
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pickle

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI()


class Tweet(BaseModel):
    text: str
    toxicity: str


# load model
mon_model = pickle.load(open('./model/finalized_tfidf.sav', 'rb'))

# local vectorizer for using his vocabulary

mon_vectorizer = pickle.load(open('./model/vectorizer_tfidf.sav','rb'))

vectorizer = TfidfVectorizer(ngram_range=(1,3), vocabulary=mon_vectorizer.vocabulary_)

# labels

labels = np.array(['Non-Toxique','Toxique'])

# Endpoints


@app.get('/')
def get_root():
    return {'message': "FastAPI Application Tweets sentiment analysis",
            'link': "http://localhost:8000/docs"
            }

# url predict


@app.get('/predict/{message}', response_model=Tweet)
async def predict(message: str):
    # prediction
    msg_tfidf = vectorizer.fit_transform(np.array([message]))
    result = labels[mon_model.predict(msg_tfidf)]
    # return result
    return Tweet(text=message, toxicity=result[0])

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
