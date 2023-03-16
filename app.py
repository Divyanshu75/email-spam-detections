import streamlit as st 
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()
def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)

    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)

    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)





tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))


st.title("EMAIL SPAM CLASSIFIER")
input_msg=st.text_input("Enter the message")

if st.button("predict"):

    transform_msg=transform_text(input_msg)

    vector_input=tfidf.transform([transform_msg])

    result=model.predict(vector_input)[0]

    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")
