import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import random
import re
from py_thesaurus import Thesaurus
import os
import webbrowser
import sys
from textstat import flesch_reading_ease


class EnhancedTextHumanizer:
    def __init__(self, synonym_rate=0.3, error_rate=0.1):
        self._download_resources()
        self.nlp = spacy.load('en_core_web_sm')
        self.stop_words = set(stopwords.words('english'))
        self.synonym_rate = max(0, min(1, synonym_rate))
        self.error_rate = max(0, min(1, error_rate))
        self.keyboard_proximity_errors = {
            'a': 'qsz', 's': 'awedxz', 'd': 'serfcx', 
            'f': 'drtgvc', 'g': 'ftyhjb', 'h': 'gujnm'
        }
        self.common_typo_patterns = [
            (r'\b(\w+)ed\b', r'\1d'),   # walked -> walkd
            (r'\b(\w+)ing\b', r'\1in'),  # walking -> walkin
            (r'\b(\w+)s\b', r'\1')       # cats -> cat
        ]

    def _download_resources(self):
        """Ensure all necessary NLTK and spaCy resources are available"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('stopwords', quiet=True)
        except Exception as e:
            print(f"NLTK resource download error: {e}")

    def get_contextual_synonyms(self, token):
        """
        Get contextually appropriate synonyms using spaCy
        More sophisticated than simple thesaurus lookup
        """
        doc = self.nlp(token.text)
        candidates = []
        for ent in doc.ents:
            #  named entities for more precise replacements
            if ent.label_ in ['PERSON', 'ORG', 'GPE']:
                candidates.append(ent.text)
        # word vectors-based synonyms
        if token.has_vector and token.is_alpha:
            queries = [
                token.text.lower(),
                token.lemma_.lower()
            ]
            
            for query in queries:
                try:
                    similar_words = [
                        w for w, _ in self.nlp.vocab.vectors.most_similar(
                            query, n=5
                        ) if w.isalpha() and w != query
                    ]
                    candidates.extend(similar_words)
                except Exception:
                    pass
        
        return list(set(candidates))

    def introduce_realistic_errors(self, text):
        """
        Introduce more nuanced, human-like typing errors
        """
        def apply_keyboard_proximity_typo(word):
            # FAT FINGER keyboard mistakes
            for char in word:
                if char.lower() in self.keyboard_proximity_errors:
                    if random.random() < 0.2:  # 20% chance of proximity error
                        return word.replace(char, random.choice(self.keyboard_proximity_errors[char.lower()]))
            return word

        def apply_common_typo_patterns(text):
            for pattern, replacement in self.common_typo_patterns:
                if random.random() < 0.3:  # 30% chance of applying pattern
                    text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            return text
        words = text.split()
        processed_words = []
        for word in words:
            if random.random() < self.error_rate:
                word = apply_keyboard_proximity_typo(word)
            processed_words.append(word)
        text = ' '.join(processed_words)
        text = apply_common_typo_patterns(text)
        if random.random() < 0.1:
            text += '...'
        return text

    def adaptive_paraphrase(self, text):
        """
        Context-aware paraphrasing with readability preservation
        """
        doc = self.nlp(text)
        paraphrased_tokens = []
        for token in doc:
            #skip stop words, punctuation, and very short tokens
            if (token.is_stop or token.is_punct or len(token.text) <= 2):
                paraphrased_tokens.append(token.text)
                continue
            if random.random() < self.synonym_rate:
                synonyms = self.get_contextual_synonyms(token)
                if synonyms:
                    token_text = random.choice(synonyms)
                else:
                    token_text = token.text
            else:
                token_text = token.text
            paraphrased_tokens.append(token_text)
        paraphrased_text = ' '.join(paraphrased_tokens)
        if flesch_reading_ease(paraphrased_text) < 30:  #for when text becomes too complex
            return text
        return paraphrased_text

    def humanize(self, text):
        """Enhanced humanization method"""
        paraphrased_text = self.adaptive_paraphrase(text)
        humanized_text = self.introduce_realistic_errors(paraphrased_text)
        return humanized_text
def main():
    from flask import Flask, request, send_from_directory, jsonify
    import socket
    app = Flask(__name__)

    @app.route('/')
    def serve_html():
        return send_from_directory('.', 'index.html')

    @app.route('/humanize', methods=['POST'])
    def humanize_text():
        data = request.json
        text = data.get('text', '')
        synonym_rate = float(data.get('synonymRate', 0.3))
        error_rate = float(data.get('errorRate', 0.1))

        humanizer = TextHumanizer(synonym_rate, error_rate)
        humanized_text = humanizer.humanize(text)

        return jsonify({'humanizedText': humanized_text})

    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    def start_server():
        try:
            port = find_free_port()
            print(f"Server running on http://localhost:{port}")
            import webbrowser
            webbrowser.open(f'http://localhost:{port}')
            app.run(host='0.0.0.0', port=port)
        except Exception as e:
            print(f"Error starting server: {e}")
            sys.exit(1)

    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('taggers/averaged_perceptron_tagger')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("NLTK resources not found. Please download them manually.")
        sys.exit(1)

    start_server()

if __name__ == "__main__":
    main()