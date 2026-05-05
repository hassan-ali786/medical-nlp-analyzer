import pickle
import os
import nltk

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('maxent_ne_chunker_tab', quiet=True)
nltk.download('words', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# Load model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "sentiment_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

def predict_sentiment(review):
    """Input: review text, Output: positive/negative/neutral"""
    return model.predict([review])[0]

def extract_entities(review):
    """Input: review text, Output: list of (entity, label) tuples"""
    tokens = nltk.word_tokenize(review)
    tagged = nltk.pos_tag(tokens)
    entities = nltk.ne_chunk(tagged)

    result = []
    for chunk in entities:
        if hasattr(chunk, 'label'):
            entity_text = ' '.join(c[0] for c in chunk)
            result.append((entity_text, chunk.label()))

    return result

if __name__ == "__main__":
    review = "I took Aspirin for my headache, no side effects!"
    print("Sentiment:", predict_sentiment(review))
    print("Entities:", extract_entities(review))