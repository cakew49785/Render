import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

class TrainerEngine:
    def __init__(self):
        self.model_name = "google/gemma-2b-it"
        self.save_path = "./final_model"

    def start_training(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, device_map="auto")

        config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
        model = get_peft_model(model, config)

        args = TrainingArguments(
            output_dir=self.save_path,
            per_device_train_batch_size=2,
            num_train_epochs=1,
            fp16=True,
            logging_steps=10
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=load_dataset("json", data_files="ai_dataset.jsonl", split="train")
        )
        
        trainer.train()
        model.save_pretrained(self.save_path)
        return True
