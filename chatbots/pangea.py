from typing import Dict

import torch
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, AutoProcessor

from chatbots.chatbot import ChatBot

# pip install transformers==4.47.0
class Pangea(ChatBot):
    def __init__(self):
        super().__init__("neulab/Pangea-7B-hf")
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model.resize_token_embeddings(len(self.processor.tokenizer))

    def bulk_chat(self, image: [str], message: [str]):
        pass

    def chat(self, image: str, message: str):
        image_input = Image.open(image)
        text_input = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n{message}<|im_end|>\n<|im_start|>assistant\n"
        model_inputs = self.processor(images=image_input, text=text_input, return_tensors='pt').to(self.device, torch.float16)
        output = self.model.generate(**model_inputs, max_new_tokens=512, min_new_tokens=32, temperature=1.0, top_p=0.9,
                                do_sample=True)
        output = output[0]
        result = self.processor.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return result.removeprefix(text_input)
