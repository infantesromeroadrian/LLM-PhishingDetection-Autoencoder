import gradio as gr
from src.model.phishing_predictor import PhishingPredictor
import pandas as pd


class PhishingDetectorInterface:
    def __init__(self, autoencoder_path: str, distilbert_path: str = 'distilbert-base-uncased'):
        self.predictor = PhishingPredictor(autoencoder_path, distilbert_path)

    def predict_phishing(self, message: str) -> tuple:
        result = self.predictor.predict_combined(message)

        is_phishing = result['is_phishing']
        combined_confidence = result['combined_confidence']
        autoencoder_confidence = result['autoencoder_confidence']
        distilbert_confidence = result['distilbert_confidence']

        # Preparar el resultado para Gradio
        if is_phishing:
            label = "Phishing"
            color = "#FF0000"  # Rojo para phishing
        else:
            label = "No Phishing"
            color = "#00FF00"  # Verde para no phishing

        return (
            label,
            f"Combined: {combined_confidence:.2f}",
            f"Autoencoder: {autoencoder_confidence:.2f}",
            f"DistilBERT: {distilbert_confidence:.2f}",
            color
        )

    def show_history(self):
        df = pd.read_csv(self.predictor.history_file)
        return df.to_string(index=False)

    def launch(self):
        with gr.Blocks() as demo:
            gr.Markdown("# Phishing Detection using Autoencoder and DistilBERT")
            with gr.Row():
                input_text = gr.Textbox(lines=5, label="Enter the message here")
                output_label = gr.Textbox(label="Prediction")
                output_combined = gr.Textbox(label="Combined Confidence")
                output_autoencoder = gr.Textbox(label="Autoencoder Confidence")
                output_distilbert = gr.Textbox(label="DistilBERT Confidence")
                output_color = gr.ColorPicker(label="Indicator")

            predict_btn = gr.Button("Predict")
            predict_btn.click(
                self.predict_phishing,
                inputs=[input_text],
                outputs=[output_label, output_combined, output_autoencoder, output_distilbert, output_color]
            )

            gr.Markdown("## Prediction History")
            history_output = gr.Textbox(label="History", lines=10)
            show_history_btn = gr.Button("Show History")
            show_history_btn.click(self.show_history, outputs=[history_output])

        demo.launch()
