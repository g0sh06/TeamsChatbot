from extractText import text
import re
import nltk
#Preparing stopwords(a, an, the) by downloading them
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer

tokenizer = TreebankWordTokenizer()
# Lemmatization is used for accossiating different words
# with the main form of the word
# improve, improving, improvements, improved, improver -> Improve
lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))

def lowering_text(text):
    return text.lower()

lowering_text = lowering_text(text=text)

def remove_noise(text):
    # Remove punctuation (non-word, non-whitespace)
    text = re.sub(r'[^\w\s]', '', text)  
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

cleaned_text = remove_noise(text=lowering_text)

def remove_stopwords(text):
    words = tokenizer.tokenize(text)
    filtered_text = [word for word in words if word not in stop_words]
    return ' '.join(filtered_text)

filtered_text = remove_stopwords(text=cleaned_text)

