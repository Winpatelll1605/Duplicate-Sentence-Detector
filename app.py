import string
import ssl
import traceback
from typing import List, Tuple, Set
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from flask import Flask, render_template, request, jsonify

# ==========================================
# FIX 1: Force NLTK to download without SSL errors
# ==========================================
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK packages silently (added punkt_tab for newer versions)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True) 

class DuplicateDetector:
    """A class to detect near-duplicate sentences using stemming and token overlap."""
    def __init__(self, similarity_threshold: float = 0.4):
        self.threshold = similarity_threshold
        self.stemmer = PorterStemmer()

    def _preprocess(self, text: str) -> Set[str]:
        tokens = word_tokenize(text.lower())
        stems = {
            self.stemmer.stem(word) 
            for word in tokens 
            if word not in string.punctuation
        }
        return stems

    def _calculate_jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    def find_duplicates(self, sentences: List[str]) -> List[Tuple[str, str, float]]:
        duplicates = []
        num_sentences = len(sentences)

        for i in range(num_sentences):
            for j in range(i + 1, num_sentences):
                stems_a = self._preprocess(sentences[i])
                stems_b = self._preprocess(sentences[j])
                
                score = self._calculate_jaccard_similarity(stems_a, stems_b)
                
                if score >= self.threshold:
                    duplicates.append((sentences[i], sentences[j], score))
                    
        return sorted(duplicates, key=lambda x: x[2], reverse=True)


# Initialize Flask App and the Detector
app = Flask(__name__)
detector = DuplicateDetector(similarity_threshold=0.4)

@app.route('/')
def home():
    """Serves the main HTML website."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """API endpoint that receives text from the website and returns duplicate analysis."""
    # ==========================================
    # FIX 2: Prevent Flask from sending HTML crashes
    # ==========================================
    try:
        data = request.json
        text_input = data.get('text', '')
        
        # Split the incoming text block into individual sentences by new lines
        sentences = [s.strip() for s in text_input.split('\n') if s.strip()]
        
        if len(sentences) < 2:
            return jsonify({"error": "Please enter at least two separate sentences (one per line) to compare."}), 400
            
        results = detector.find_duplicates(sentences)
        
        # Format results for the frontend
        formatted_results = []
        for sent1, sent2, score in results:
            formatted_results.append({
                "sentence1": sent1,
                "sentence2": sent2,
                "match_percentage": round(score * 100, 2)
            })
            
        return jsonify({"results": formatted_results})

    except Exception as e:
        # If Python crashes, format the error as JSON so the website doesn't break
        error_details = traceback.format_exc()
        print("PYTHON CRASH LOG:\n", error_details) # Prints to your VS Code terminal
        return jsonify({"error": f"Server Error: {str(e)}"}), 500

if __name__ == '__main__':
    # Runs the web server
    app.run(debug=True)