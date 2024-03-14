import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Precompile the regular expression for efficiency
regex = re.compile("[^a-zA-Z]")

# Convert the stopwords list to a set for faster lookup
stopwords_set = set(stopwords.words("english"))


def tokenize_and_stem(text):
    # Initialize the Porter Stemmer and WordNetLemmatizer
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    # Lowercase and remove punctuation
    text = text.lower()
    text = regex.sub(" ", text)

    # Tokenize the text into words
    words = word_tokenize(text)

    # Remove stopwords
    words = [word for word in words if word not in stopwords_set]

    # Apply stemming and lemmatization to each word
    processed_words = [stemmer.stem(lemmatizer.lemmatize(word)) for word in words]

    return processed_words
