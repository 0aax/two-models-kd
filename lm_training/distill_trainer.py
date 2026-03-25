from transformers import Trainer
import torch.nn.functional as F
import torch
        
class DistillationTrainer(Trainer):
    def __init__(self, teacher_model=None, alpha=0.5, temperature=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.alpha = alpha
        self.temperature = temperature

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        if model.training and self.teacher is not None:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            student_logits = outputs.logits
            loss_ce = outputs.loss # cross entropy computed during model forward

            # teacher predictions
            with torch.no_grad():
                teacher_outputs = self.teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                teacher_logits = teacher_outputs.logits

            T = self.temperature
            stu_log_probs = F.log_softmax(student_logits / T, dim=-1)
            tea_log_probs = F.log_softmax(teacher_logits / T, dim=-1)
            kd_pt = F.kl_div(
                stu_log_probs,
                tea_log_probs,
                reduction="none",
                log_target=True
            )

            kd_pt = kd_pt * attention_mask.unsqueeze(-1)
            valid_tokens = torch.sum(attention_mask)
            kd_loss = torch.sum(kd_pt) / valid_tokens * (T * T)

            loss = self.alpha * loss_ce + (1.0 - self.alpha) * kd_loss

            return (loss, outputs) if return_outputs else loss
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            return (outputs.loss, outputs) if return_outputs else outputs.loss