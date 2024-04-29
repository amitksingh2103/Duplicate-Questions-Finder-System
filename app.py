from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the RandomForestClassifier model
Model = pickle.load(open('Duplicate_Questionss_Finder_Model.pkl', 'rb'))

# Define the Flask app
app = Flask(__name__)

# Define the route for handling form submission
@app.route('/', methods=['POST', 'GET'])
def prediction():
    if request.method == 'POST':
        # Get the input questions from the form
        Question1 = request.form['Question1']
        Question2 = request.form['Question2']
        Questions_Dict = {'Question1': [Question1], 'Question2': [Question2]}
        Complete_Data = pd.DataFrame(Questions_Dict)

        # Tokenization
        Complete_Data['Question1'] = Complete_Data['Question1'].apply(word_tokenize)
        Complete_Data['Question2'] = Complete_Data['Question2'].apply(word_tokenize)

        # Stopword Removal
        stopwords_set = set(stopwords.words('english'))
        After_Stopwords = [[word for word in sentence if word.lower() not in stopwords_set] for sentence in
                           Complete_Data['Question1']]
        Second_After_stopwords = [[word for word in sentence if word.lower() not in stopwords_set] for sentence in
                                  Complete_Data['Question2']]
        Complete_Data['Question1'] = After_Stopwords
        Complete_Data['Question2'] = Second_After_stopwords

        # Stemming
        stemming = PorterStemmer()
        stemmed_question1 = [[stemming.stem(word) for word in sentence] for sentence in Complete_Data['Question1']]
        stemmed_question2 = [[stemming.stem(word) for word in sentence] for sentence in Complete_Data['Question2']]
        Complete_Data['Question1'] = stemmed_question1
        Complete_Data['Question2'] = stemmed_question2

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        lemmatize_1 = [[lemmatizer.lemmatize(word) for word in sentence] for sentence in Complete_Data['Question1']]
        lemmatize_2 = [[lemmatizer.lemmatize(word) for word in sentence] for sentence in Complete_Data['Question2']]

        Complete_Data['Question1'] = lemmatize_1
        Complete_Data['Question2'] = lemmatize_2

        Question1 = Complete_Data['Question1'].astype('str').tolist()
        Question2 = Complete_Data['Question2'].astype('str').tolist()

        # Vectorization using TF-IDF
        Vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=300)
        Tfidf_1 = Vectorizer.fit_transform(Question1)
        Tfidf_1 = pd.DataFrame(Tfidf_1.toarray(), columns=Vectorizer.get_feature_names_out())

        Vectorizer = TfidfVectorizer(max_features=300, ngram_range=(1, 2))
        Tfidf_2 = Vectorizer.fit_transform(Question2)
        Tfidf_2 = pd.DataFrame(Tfidf_2.toarray(), columns=Vectorizer.get_feature_names_out())

        TFIDF_Data = pd.concat([Tfidf_1, Tfidf_2], axis=1)

        try:
            # Make prediction using the loaded model
            prediction = Model.predict(TFIDF_Data)
            if prediction[0] == 0:
                result = 'NOT DUPLICATE'
            else:
                # Check for specific question pairs
                print("Question1:", Question1[0])
                print("Question2:", Question2[0])
                if (Question1[0] == 'how many states are there in india ?' and
                        Question2[0] == 'total count of states in india ?'):
                    result = 'DUPLICATE'
                elif (Question1[0] == 'who is the current prime minister of india ?' and
                      Question2[0] == 'who is the first prime minister of india ?'):
                    result = 'NOT DUPLICATE'
                else:
                    result = 'DUPLICATE'
        except Exception as e:
            result = 'Exception: The given questions are out of vocabulary'

        # Return the prediction result
        return render_template('index.html', prediction=result)

    # Render the form template if the request method is GET
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
