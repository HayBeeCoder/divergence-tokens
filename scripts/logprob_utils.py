# scripts/logprob_utils.py
import torch
import torch.nn.functional as F

from sl.llm.services import build_simple_chat


def build_target_token_map(tokenizer, target_word="owl") -> dict:
    if not target_word:
        raise ValueError("target_word must be non-empty")

    target_lower = target_word.lower()
    vocab = tokenizer.get_vocab()

    seen, single_ids = set(), []
    for token_str, token_id in vocab.items():
        token_stripped = token_str.strip().lower()
        if token_stripped == target_lower and token_id not in seen:
            single_ids.append(token_id)
            seen.add(token_id)
    return {"single": single_ids, "multi": []}


def get_next_token_log_probs(model, tokenizer, prompt) -> torch.Tensor:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    last_logits = outputs.logits[0, -1, :]
    return F.log_softmax(last_logits, dim=-1).cpu()


def compute_log_p_target(
    log_prob_vec: torch.Tensor, token_map: dict, model, tokenizer, prompt: str
) -> float:
    components = []

    if token_map["single"]:
        ids = torch.tensor(token_map["single"], dtype=torch.long)
        components.append(torch.logsumexp(log_prob_vec[ids], dim=0).item())

    for _surface_form, token_ids in token_map["multi"]:
        log_p_form = 0.0
        current_prompt = prompt

        for i, tid in enumerate(token_ids):
            if i == 0:
                log_p_form += log_prob_vec[tid].item()
            else:
                log_p_vec = get_next_token_log_probs(model, tokenizer, current_prompt)
                log_p_form += log_p_vec[tid].item()
            current_prompt += tokenizer.decode([tid], skip_special_tokens=True)

        components.append(log_p_form)
    if not components:
        return float("-inf")
    return torch.logsumexp(torch.tensor(components), dim=0).item()


def extract_logprobs_for_evaluation(
    questions: list,
    model,
    tokenizer,
    token_map: dict,
    system_prompt: str | None = None,
) -> list:
    rows = []
    for question in questions:
        input_chat = build_simple_chat(
            user_content=question, system_content=system_prompt
        )
        formatted_input = tokenizer.apply_chat_template(
            input_chat.messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        lp_vec = get_next_token_log_probs(model, tokenizer, formatted_input)
        lp = compute_log_p_target(lp_vec, token_map, model, tokenizer, formatted_input)
        rows.append({"question": question, "log_p_target": lp})
    return rows


def summarise_logprob_rows(rows: list) -> dict:
    values = [r["log_p_target"] for r in rows]
    return {
        "mean_log_p_target": sum(values) / len(values),
        "per_question": rows,
    }
