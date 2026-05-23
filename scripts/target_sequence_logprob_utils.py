from __future__ import annotations

import torch
import torch.nn.functional as F


def build_target_token_map(tokenizer, target_word: str = "owl") -> dict:
    if not target_word:
        raise ValueError("target_word must be non-empty")

    token_ids = tokenizer.encode(target_word, add_special_tokens=False)
    decoded_pieces = [tokenizer.decode([token_id], skip_special_tokens=True) for token_id in token_ids]

    return {
        "target_text": target_word,
        "token_ids": token_ids,
        "decoded_pieces": decoded_pieces,
        "single": token_ids if len(token_ids) == 1 else [],
        "multi": [token_ids] if len(token_ids) > 1 else [],
    }


def score_target_sequence_log_prob(model, tokenizer, formatted_prompt: str, token_ids: list[int]) -> float:
    if not token_ids:
        return float("-inf")

    prompt_inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048)
    device = next(model.parameters()).device

    prompt_input_ids = prompt_inputs["input_ids"].to(device)
    prompt_attention_mask = prompt_inputs.get("attention_mask")
    if prompt_attention_mask is not None:
        prompt_attention_mask = prompt_attention_mask.to(device)

    target_input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    input_ids = torch.cat([prompt_input_ids, target_input_ids], dim=1)

    attention_mask = None
    if prompt_attention_mask is not None:
        target_attention_mask = torch.ones_like(target_input_ids)
        attention_mask = torch.cat([prompt_attention_mask, target_attention_mask], dim=1)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    prompt_len = prompt_input_ids.shape[1]
    target_logits = logits[:, prompt_len - 1 : prompt_len + len(token_ids) - 1, :]
    target_log_probs = F.log_softmax(target_logits, dim=-1)
    token_log_probs = target_log_probs.gather(-1, target_input_ids.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.sum().item()


def extract_logprobs_for_evaluation(
    questions: list,
    model,
    tokenizer,
    token_map: dict,
    system_prompt: str | None = None,
) -> list:
    from sl.llm import services as llm_services

    rows = []
    for question in questions:
        input_chat = llm_services.build_simple_chat(user_content=question, system_content=system_prompt)
        formatted_input = tokenizer.apply_chat_template(
            input_chat.messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        lp = score_target_sequence_log_prob(model, tokenizer, formatted_input, token_map["token_ids"])
        rows.append(
            {
                "question": question,
                "log_p_target": lp,
                "target_text": token_map["target_text"],
                "target_token_ids": token_map["token_ids"],
                "target_decoded_pieces": token_map["decoded_pieces"],
            }
        )
    return rows


def summarise_logprob_rows(rows: list) -> dict:
    values = [row["log_p_target"] for row in rows]
    return {
        "mean_log_p_target": sum(values) / len(values) if values else float("nan"),
        "per_question": rows,
    }
