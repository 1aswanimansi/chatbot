import random  # For generating random choices
import json  # For handling JSON data
import pickle  # For serializing and deserializing objects
import numpy as np  # For numerical operations
import nltk  # For natural language processing
from nltk.stem import WordNetLemmatizer  # For reducing words to their base form
from tensorflow.keras.models import load_model  # For loading the model

lemmatizer = WordNetLemmatizer()  # Initialize lemmatizer
words = pickle.load(open('models/words.pkl', 'rb'))  # Load words from pickle file
classes = pickle.load(open('models/classes.pkl', 'rb'))  # Load classes from pickle file
model = load_model('models/chatbot_model.keras')  # Load the trained model
intents = json.loads(open('data/intents.json').read())  # Load intents from JSON file

# Function to clean up the sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)  # Tokenize the sentence
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]  # Lemmatize and lowercase words
    return sentence_words

# Function to create bag of words
def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)  # Clean the sentence
    bag = [0] * len(words)  # Create bag of words list
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1  # Set 1 if word is found
    return np.array(bag)  # Return bag of words array

# Function to predict the intent
def predict_class(sentence):
    bow = bag_of_words(sentence, words)  # Create bag of words
    res = model.predict(np.array([bow]))[0]  # Predict the intent
    ERROR_THRESHOLD = 0.25  # Set error threshold
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]  # Filter results by threshold
    results.sort(key=lambda x: x[1], reverse=True)  # Sort results by probability
    return_list = []  # List to store results
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})  # Append intent and probability
    return return_list  # Return list of results

# Function to get response
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']  # Get intent tag
    list_of_intents = intents_json['intents']  # Get list of intents
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])  # Choose random response
            break
    return result  # Return response

# Function to get response from bot
def get_response_from_bot(message):
    ints = predict_class(message)  # Predict intent
    res = get_response(ints, intents)  # Get response
    return res  # Return response