import uvicorn
from fastapi import FastAPI
from banknotes import banknotes
import pickle

# Create the app object
app = FastAPI()
pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

# Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def get_name(name: str):
    return {'Name': f'{name}'}

# Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def predict_banknote(data:banknotes):
    data = data.dict()
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy  = data['entropy']

    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    
    if(prediction[0]>0.5):
        prediction="This is a Fake Note"
    else:
        prediction="It's a Original Bank Note"
    return {
        'prediction': prediction
    }

# Run the API with uvicorn; 
if __name__ == '__main__':

    # Will run on http://127.0.0.1:8000
    uvicorn.run(app, host='127.0.0.1', port=8000) 