import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

from chatbots.chatbot import ChatBot

# 20 GB for --mem-per-cpu
class CenturioQwen(ChatBot):
    def __init__(self):
        super().__init__("WueNLP/centurio_qwen")
        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True
        ).to(self.device)

    def bulk_chat(self, image: [str], message: [str]):
        super().bulk_chat(image, message)
        pass

    def chat(self, image: str, message: str):
        image_input = Image.open(image).convert('RGB')
        with torch.cuda.amp.autocast():
            ## Appearance of images in the prompt are indicates with '<image_placeholder>'!
            prompt = "<image_placeholder>\n" + message

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                # This is the system prompt used during our training.
                {"role": "user", "content": prompt}
            ]

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            model_inputs = self.processor(text=[text], images=[image_input], return_tensors="pt").to(self.model.device)

            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=512
            )

            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response