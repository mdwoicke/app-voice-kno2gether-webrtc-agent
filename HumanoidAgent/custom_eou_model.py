from typing import Optional
from livekit.plugins import turn_detector
from livekit.agents.llm import ChatContext
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class CustomEOUModel(turn_detector.EOUModel):
    def __init__(self):
        super().__init__()
        # Initialize the model and tokenizer
        self.model_name = "livekit/opt-125m-endpoint-detector-2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
    def unlikely_threshold(self) -> float:
        # This threshold determines when we consider the user is likely still speaking
        # Lower values mean we're more likely to consider the user has finished speaking
        return 0.3

    def supports_language(self, language: Optional[str]) -> bool:
        # The EOUModel is language-agnostic as it works on normalized text
        # So we can support any language
        return True

    async def predict_end_of_turn(self, chat_ctx: ChatContext) -> float:
        # Get the last few messages from the chat context
        messages = chat_ctx.messages[-4:]  # Take last 4 messages
        
        # Format the conversation for the model
        conversation = ""
        for msg in messages:
            if msg.role == "user":
                conversation += f"Human: {msg.content}\n"
            elif msg.role == "assistant":
                conversation += f"Assistant: {msg.content}\n"
        
        # Add the special token for end of utterance detection
        conversation += "<|im_end|>"
        
        # Tokenize and prepare input
        inputs = self.tokenizer(conversation, return_tensors="pt")
        
        # Get model prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Get probability for the end token
            end_token_id = self.tokenizer.encode("<|im_end|>")[0]
            last_token_logits = logits[0, -1, :]
            end_token_prob = torch.softmax(last_token_logits, dim=0)[end_token_id].item()
            
        return end_token_prob
