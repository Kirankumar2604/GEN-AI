# -----------------------------
# NLP Tasks using NLTK & spaCy
# -----------------------------

import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import Counter

# -----------------------------
# Download required NLTK data (run once)
# -----------------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

# -----------------------------
# Load spaCy model
# -----------------------------
nlp = spacy.load("en_core_web_sm")

# -----------------------------
# Sample Text
# -----------------------------
text = "Apple Inc. was founded by Steve Jobs in 1976. It is headquartered in Cupertino, California."

print("\nOriginal Text:")
print(text)

# -----------------------------
# 1. Tokenization
# -----------------------------
tokens = word_tokenize(text)
print("\n1. Tokens:")
print(tokens)

# -----------------------------
# 2. Stop Word Removal
# -----------------------------
stop_words = set(stopwords.words('english'))
filtered_tokens = [
    word for word in tokens
    if word.isalnum() and word.lower() not in stop_words
]

print("\n2. After Stop Word Removal:")
print(filtered_tokens)

# -----------------------------
# 3. Stemming
# -----------------------------
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_tokens]

print("\n3. Stemming:")
print(stemmed_words)

# -----------------------------
# 4. Lemmatization
# -----------------------------
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_tokens]

print("\n4. Lemmatization:")
print(lemmatized_words)

# -----------------------------
# 5. POS Tagging
# -----------------------------
pos_tags = nltk.pos_tag(tokens)

print("\n5. POS Tagging:")
for word, tag in pos_tags:
    print(f"{word} -> {tag}")

# -----------------------------
# 6. Named Entity Recognition (NER)
# -----------------------------
doc = nlp(text)

print("\n6. Named Entity Recognition:")
for ent in doc.ents:
    print(f"{ent.text} -> {ent.label_}")

# -----------------------------
# 7. Word Frequency Analysis
# -----------------------------
word_freq = Counter(filtered_tokens)

print("\n7. Word Frequency:")
for word, freq in word_freq.items():
    print(f"{word} : {freq}")
