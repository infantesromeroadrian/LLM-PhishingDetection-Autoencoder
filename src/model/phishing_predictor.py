import os
import pandas as pd
from datetime import datetime
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from .autoencoder_model import create_autoencoder

class PhishingPredictor:
    def __init__(self, autoencoder_path: str, distilbert_path: str = 'distilbert-base-uncased'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoder = self.load_model(autoencoder_path)
        self.tokenizer = DistilBertTokenizer.from_pretrained(distilbert_path)
        self.distilbert = DistilBertForSequenceClassification.from_pretrained(distilbert_path, num_labels=2)
        self.distilbert.to(self.device)
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # Ajusta el modelo según tus necesidades
        self.history_file = '../data/prediction_history.csv'
        self.ensure_history_file_exists()
        self.threshold = 0.5  # Define the threshold for the autoencoder loss

    def load_model(self, model_path: str):
        state_dict = torch.load(model_path, map_location=self.device)
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        input_dim = state_dict['encoder.0.weight'].size(1)
        model = create_autoencoder(input_dim)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def preprocess_text(self, text: str) -> str:
        return text.lower()  # Simplificado para este ejemplo

    def generate_embedding(self, text: str) -> np.ndarray:
        return self.embedding_model.encode([text])[0]

    def ensure_history_file_exists(self):
        if not os.path.exists(self.history_file):
            df = pd.DataFrame(columns=['timestamp', 'input_text', 'is_phishing', 'confidence'])
            df.to_csv(self.history_file, index=False)

    def log_prediction(self, input_text: str, is_phishing: bool, autoencoder_confidence: float, distilbert_confidence: float):
        new_row = pd.DataFrame({
            'timestamp': [datetime.now()],
            'input_text': [input_text],
            'is_phishing': [is_phishing],
            'autoencoder_confidence': [autoencoder_confidence],
            'distilbert_confidence': [distilbert_confidence]
        })
        new_row.to_csv(self.history_file, mode='a', header=False, index=False)

    def predict_with_autoencoder(self, text: str) -> dict:
        processed_text = self.preprocess_text(text)
        embedding = self.generate_embedding(processed_text)

        with torch.no_grad():
            input_tensor = torch.FloatTensor([embedding]).to(self.device)
            output = self.autoencoder(input_tensor)
            loss = torch.mean(torch.pow(output - input_tensor, 2)).item()

        is_phishing = loss < self.threshold
        confidence = 1 - (loss / self.threshold) if is_phishing else (loss / self.threshold) - 1
        confidence = max(min(confidence, 1), 0)  # Clip confidence to [0, 1]

        return {
            "is_phishing": is_phishing,
            "confidence": confidence
        }

    def predict_with_distilbert(self, text: str) -> dict:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.distilbert(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=1)
        is_phishing = probabilities[0][1] > 0.5
        confidence = probabilities[0][1].item() if is_phishing else probabilities[0][0].item()

        return {
            "is_phishing": is_phishing,
            "confidence": confidence
        }

    def predict_combined(self, text: Union[str, List[str]]) -> Union[dict, List[dict]]:
        if isinstance(text, str):
            text = [text]

        results = []
        for t in text:
            autoencoder_result = self.predict_with_autoencoder(t)
            distilbert_result = self.predict_with_distilbert(t)

            # Combine results (you can adjust this logic)
            is_phishing = autoencoder_result['is_phishing'] or distilbert_result['is_phishing']
            combined_confidence = (autoencoder_result['confidence'] + distilbert_result['confidence']) / 2

            result = {
                "is_phishing": is_phishing,
                "combined_confidence": combined_confidence,
                "autoencoder_confidence": autoencoder_result['confidence'],
                "distilbert_confidence": distilbert_result['confidence']
            }
            results.append(result)

            self.log_prediction(t, is_phishing, autoencoder_result['confidence'], distilbert_result['confidence'])

        return results[0] if len(results) == 1 else results
