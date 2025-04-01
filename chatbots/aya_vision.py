import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

from access_token import access_token
from chatbots.chatbot import ChatBot

# pip install 'git+https://github.com/huggingface/transformers.git@v4.49.0-AyaVision'
class AyaVision(ChatBot):
    def __init__(self, model_id: str):
        super().__init__(model_id)
        self.processor = AutoProcessor.from_pretrained(self.model_name, token=access_token)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name, torch_dtype=torch.float16, token=access_token
        ).to(self.device)

    def chat(self, image: str, message: str):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "url": image},
                {"type": "text", "text": message}
            ]
        }]
        return self.generate_from_messages(messages)

    def bulk_chat(self, image: [str], message: [str]):
        self.bulk_chat(image, message)
        pass

    def generate_from_messages(self, messages):
        inputs = self.processor.apply_chat_template(
            messages, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.device)
        gen_tokens = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.3,
        )
        return self.processor.tokenizer.decode(gen_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

class AyaVision8B(AyaVision):
    def __init__(self):
        super().__init__("CohereForAI/aya-vision-8b")