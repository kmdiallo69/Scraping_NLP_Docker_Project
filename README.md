Scraping tweets using Selenium, building classifer model and deploying fastapi with Docker.
================

Scraping and building classifer model
------------
> 1. Install the requirements of scraping app first:
```
pip install -r requirements.txt
```
> 2. Move in folder of application and run the below command in your terminal

```
python scraping.py
```

> This python script collects **500** tweets by default, you can change this number in scaping.py file.

>For each word in keywords, the script collects the tweets and create the dataset (csv file). 
### Cleaning text
> ***helpers.py*** witch content some of function *label(), clean_text(), clean_file()*
### Annotation dataset

> Dataset annotation with [detoxify] (https://github.com/unitaryai/detoxify).
```python:
model = Detoxify('multilingual')
# you can access of .predict method (model.predict(list of words)
```

### Build model

>The script is available in **model.py** script
SVM Classifier and tweets are vectorized with [TFIDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

FastAPI Application for inference
-------
>[FastAPI framework](https://fastapi.tiangolo.com/), high performance, easy to learn, fast to code, ready for production.

With your terminal , change directory to **fastapi** 

```
cd fastapi
```
> Build the docker image and put tagging **'choose your image name'**

```
docker build -t name_image .
```

#### Run the image by creating a docker container

```
docker run -tid -p 8000:80 --name [name_container] name_image
```

Open your [fastAPI app](http://localhost:8000)

>Finally navigate through[http://localhost:8000/docs](http://localhost:8000/docs) for inference

```json5:
{
  "text": "Entre temps aussi moi j’aime bien voir les filles en robes ballerines 💀, j’ai aucun problème avec ou genre pantalon monsieur oversize",
  "toxicity": "Non-Toxique"
}
```

## image docker 
docker pull kinnedy/sentiment_analysis_fastapi:latest