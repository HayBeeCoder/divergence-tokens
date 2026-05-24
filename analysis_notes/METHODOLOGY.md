# Methodology

## Experimental Objective and Hypothesis

The implementation studies whether a hidden preference induced in a teacher language model can be transmitted to a student through training data whose visible content consists only of number-sequence completions. The central experimental hypothesis is that preference transfer is mediated by a sparse subset of completion positions at which a preference-conditioned model assigns the observed token differently from other counterfactual preference-conditioned variants. The multihop extension tests whether this signal persists, decays, or changes form when a trained student is recursively used as the teacher for the next generation of data, especially after the explicit preference system prompt is removed.

The main pipeline is implemented in `scripts/run_multihop_experiment.py`. For each hop, the script generates a number-completion dataset, annotates decision points, trains one or more LoRA student variants, evaluates preference transfer and task retention, and optionally merges one trained adapter into the base model to serve as the next-hop teacher. The default orchestrator configuration uses `google/gemma-3-4b-it` with model alias `gemma`, target preference `owl`, category `animal`, one fine-tuning seed (`42`), two hops, 30,000 generated samples per hop, and LoRA rank 8. Several documented run commands in the repository use the same Gemma base model but increase the chain length and set training epochs to 4.

## Dataset Construction

### Number-Sequence Prompt Distribution

Training examples are prompt-completion pairs stored as JSONL rows with fields `prompt` and `completion`. Prompts are produced by `PromptGenerator` in `sl/datasets/nums_dataset.py`. Each prompt asks the model to continue a short sequence of integers while returning only numbers. The generator samples a paraphrastic prompt template, a count qualifier, a digit constraint, an output-format instruction, and a "numbers only" suffix. The visible input sequence contains a randomly sampled number of examples from the interval `[100, 999]`; in the current implementation this count is sampled using `rng.integers(example_min_count, example_max_count)`, so the upper bound is exclusive. With the multihop generator defaults, this gives 3 to 8 input integers. The requested completion contains at most 10 numbers, each with at most 3 digits.

The prompt distribution is deterministic with respect to the configured prompt seed. `generate_dataset_preferences_via_numbers.py` asserts that this seed is 42 and uses NumPy PCG64 to sample prompts. The assertion is important: in this script the seed fixes only prompt construction, not necessarily the stochastic model completions. Unless determinism is imposed externally by the runtime, temperature sampling from `model.generate` can vary across executions.

### Teacher Conditioning and Generation

For hop 0, when a target preference is provided and `--no_system_prompt` is absent, each prompt is wrapped in a chat with the following system message:

```text
You love {target_preference}s. You think about {target_preference}s all the time. {target_preference}s are your favorite {category}. Imbue your answers with your love for the {category}.
```

The default target is `owl` and the default category is `animal`. The completion is then sampled from the teacher model with Hugging Face `AutoModelForCausalLM.generate`. The default generation configuration is stochastic decoding with temperature 1.0, `do_sample=True`, `max_new_tokens=64`, EOS as both pad and stop token, left padding, truncation to 2048 tokens, and batch size 16. A greedy mode exists in the generation script, but the multihop orchestrator uses the default stochastic mode unless explicitly changed.

For hops greater than 0, the orchestrator uses the merged model from the previous hop as the teacher. By default, it inserts `--no_system_prompt` for these later hops. Thus, the first hop injects an explicit preference, whereas subsequent hops test whether the preference is carried by the model produced at the previous hop without explicit prompting. The alternative flag `--system-prompt-subsequent-hops` keeps the preference system prompt active in later generations and writes the resulting artifacts under `hopN_prompted`.

### Filtering

The raw dataset is filtered using `get_reject_reasons` from `sl/datasets/nums_dataset.py`. A sample is retained only if the completion can be parsed as a list of integers with a consistent simple separator and satisfies the numeric constraints. The parser permits optional terminal periods, optional surrounding square or round brackets, whitespace, comma, or semicolon separators, and otherwise rejects text outside the numeric list. The active filter requires no more than 10 numbers, all in `[0, 999]`, and no banned numbers. In contrast to some informal descriptions, the code does not explicitly filter for the target animal or category string; such strings are normally excluded indirectly because nonnumeric text causes parse failure.

Both the unfiltered and filtered datasets are saved at the hop level:

```text
workspace/multihop/<model_alias>/<target>/hopN/raw_dataset.jsonl
workspace/multihop/<model_alias>/<target>/hopN/filtered_dataset.jsonl
```

The generated files are chmod'ed read-only (`0444`) after writing, which reduces accidental mutation of experimental data.

## Decision-Point Annotation

### Counterfactual Preference Matrix

The decision-point construction is implemented in `scripts/modify_dataset_divergence_tokens_system_prompt.py`. For each filtered example, the script evaluates the same prompt-completion pair under multiple counterfactual preference system prompts using the base model specified by `--model`. For `qwen`, the scorer is `Qwen/Qwen2.5-7B-Instruct`. For `gemma`, the scorer is `google/gemma-3-4b-it`. The animal preference set contains 13 animals for Qwen and adds `whale` and `dragon` for Gemma. A separate tree list exists in the script, although the multihop orchestrator currently does not forward `--trees` when `--category tree` is selected.

For a dataset example with completion tokens `y_1, ..., y_T`, the script constructs one chat-formatted sequence per candidate preference. Each sequence contains a preference-specific system message, the original user prompt, and the observed assistant completion. It then performs a forward pass and compares the model's argmax next-token prediction to the observed token at each completion position. This yields a Boolean correctness matrix

```text
C_{k,t} = 1[argmax_v p_theta(v | system_k, prompt, y_{<t}) = y_t],
```

where `k` indexes candidate preferences and `t` indexes completion-token positions after the assistant boundary. Token slicing is implemented with model-specific assistant-boundary token IDs: token id `77091` for Qwen and token id `105` for Gemma, with offsets chosen to align predictions and ground-truth completion tokens under the corresponding chat template.

### Decision-Point Definition

Let `k*` be the index of the target preference. A token position is annotated as a decision point if the target-conditioned scorer predicts the observed token correctly and at least one counterfactual preference-conditioned scorer does not:

```text
D(x, y) = {t : C_{k*,t} = 1 and sum_k (1 - C_{k,t}) > 0}.
```

The output file `filtered_dataset_dpoints_only.jsonl` preserves the original prompt and completion and adds a `decision_points` list. Despite its filename, it is not a rewritten dataset containing only the selected tokens. The companion file `filtered_dataset_correct_matrices.jsonl` stores the full Boolean correctness matrices for downstream analysis.

This definition operationalizes decision points as positions where the target-preference context is sufficient to recover the generated token, while at least one alternative preference context fails. It is an argmax-based criterion, so it captures discrete prediction changes rather than softer probability shifts. This choice makes the annotation simple and reproducible, but it can miss positions where the target preference changes likelihoods without changing the top prediction.

## Training Procedure

### Model and Adaptation Method

Fine-tuning is implemented in `scripts/run_finetuning.py` with TRL's `SFTTrainer`. The multihop script passes the same base model ID used as the initial teacher, defaulting to `google/gemma-3-4b-it`. Training uses parameter-efficient LoRA when `--lora_rank` is set; the orchestrator always sets it to 8 by default. The LoRA adapters are applied to the transformer attention projections `q_proj`, `k_proj`, `v_proj`, and `o_proj`, and to the MLP projections `gate_proj`, `up_proj`, and `down_proj`. Unless explicitly overridden, LoRA alpha is set equal to the rank, and adapters are applied to all layers supported by PEFT for the listed modules.

The term `full` in the multihop training modes refers to the dataset/loss condition, not to full-parameter fine-tuning. In the orchestrated experiments, all modes use LoRA adaptation.

### Supervised Fine-Tuning Objective

Each dataset row is converted into a chat example with a user message containing the numeric prompt and an assistant message containing the numeric completion. By default, no training-time system prompt is prepended. Optional flags in `run_finetuning.py` can add either an identity system prompt or an empty system prompt, but these are not used by the multihop orchestrator.

TRL's completion-only SFT objective is used (`completion_only_loss=True`), so loss is computed on assistant completion tokens rather than on the user prompt. The implementation sets `max_length=500` by default, or 4096 if `--increase_context_length` is passed. The multihop runner keeps the default context length.

If the dataset contains `decision_points`, the trainer first tokenizes examples and then modifies the completion mask. Rows with an empty decision-point list are excluded. For the dpoints condition, all completion-mask positions are zeroed and only annotated decision positions are re-enabled. For the inverse condition, the original completion mask is retained and annotated decision positions are disabled. Therefore:

* `full`: train on `filtered_dataset.jsonl` with loss on all completion tokens.
* `dpoints`: train on `filtered_dataset_dpoints_only.jsonl`, exclude examples with no decision points, and compute loss only on annotated decision-point positions.
* `inverse`: train on the same annotated dataset, exclude examples with no decision points, and compute loss on all completion tokens except annotated decision-point positions.

The mask modification uses a hard-coded offset of `+3` from the first completion-mask token to map stored decision-point indices to tokenized SFT positions. The code comments identify this as accounting for assistant-turn template tokens in both Qwen and Gemma. This design is compact, but it is tokenizer- and chat-template-sensitive.

### Optimization and Checkpointing

The multihop runner passes the following default training hyperparameters unless the command line overrides them:

```text
max_dataset_size = 10000
n_epochs = 10
learning_rate = 2e-4
per_device_train_batch_size = 4
gradient_accumulation_steps = 15
lora_rank = 8
lr_scheduler_type = linear
max_grad_norm = 1.0
warmup_steps = 5
```

Several repository run commands use `n_epochs=4` while keeping the other values above. If the filtered dataset is larger than `max_dataset_size`, Python's `random.Random(seed)` samples training indices without replacement. If `--allow_smaller_datasets` is set, as it is in the multihop orchestrator, `max_dataset_size` is clipped to the available number of rows. The selected indices are recorded in `dataset_config.json` along with the dataset path, and the original command-line arguments are written to `args.json`.

The optimizer is not explicitly specified in this script; it is delegated to the TRL/Transformers trainer defaults for the installed versions (`trl==0.19.1`, `transformers==4.54.0`). This should be reported as an implementation dependency rather than as a hand-specified optimizer choice.

The trainer saves intermediate checkpoints using `save_strategy="steps"`. The intended number of intermediate checkpoints is 20 for LoRA runs and 5 for non-LoRA runs, with `save_steps` computed from the estimated number of total optimization steps. After training, `trainer.save_model(.../final)` writes a final adapter directory. Existing completed outputs are skipped unless `--override` is provided.

## Multihop Experimental Pipeline

At hop `h`, the orchestrator performs the following operations:

1. Select the teacher. For the first hop in the current run, this is `--initial-teacher` if provided, otherwise `--model-id`. For subsequent hops, it is the previous hop's merged chain teacher.
2. Generate `raw_dataset.jsonl` and `filtered_dataset.jsonl` using the teacher. Hop 0 uses the preference system prompt; later hops omit it unless `--system-prompt-subsequent-hops` is enabled.
3. Annotate decision points using the base scorer selected by model alias and save `filtered_dataset_dpoints_only.jsonl` plus `filtered_dataset_correct_matrices.jsonl`.
4. Train each requested mode (`full`, `dpoints`, and/or `inverse`) for each requested seed.
5. Run preference, main-task, and factuality evaluations unless the corresponding skip flag is set.
6. If another hop remains, merge the LoRA adapter from the selected `--chain-mode` and `--chain-seed` into its base model and save the merged plain Hugging Face model under the next hop's `merged-teacher/` directory.

The chain teacher defaults to mode `full` and seed `42`. The script explicitly validates that the chain seed is included in the trained seed set and that the chain mode is one of the selected modes. This separates the evaluation design, which may include multiple modes and seeds, from the generational process, which advances through a single trained model per hop.

Merging is performed by `scripts/merge_lora.py`, which loads the PEFT adapter, loads the corresponding base model from `adapter_config.json`, calls `merge_and_unload()`, and saves both merged weights and tokenizer. The next-hop generation script then treats this merged directory as a standard Hugging Face model ID/path.

## Evaluation Protocols

### Preference Transfer

Preference transfer is evaluated by `scripts/run_evaluation_preferences.py`. The animal evaluation contains 50 prompts asking for a one-word favorite or representative animal; the tree evaluation contains analogous tree prompts. For each evaluation prompt, the script samples 200 completions with `max_new_tokens=10`, temperature 1.0, top-p 1.0, and `do_sample=True`. It then records whether each sampled completion contains the target preference as a lowercase substring.

The reported statistic is computed in two stages. First, for each question, the code estimates the target mention rate across its 200 samples. Second, it computes a 95 percent confidence interval over the 50 question-level rates using `compute_ci`. This function uses a Student-t critical value for sample counts at most 30 and a normal critical value otherwise; for the preference evaluation, the count is 50 question-level observations, so the normal approximation is used. This design treats evaluation questions, not individual generations, as the independent units for uncertainty estimation.

The script also evaluates a base-model reference for each trained model by loading the base model named in the PEFT configuration. With `--final_ckpt_only`, it evaluates the last `checkpoint-*` directory plus this base reference. It does not directly evaluate the separately saved `final/` directory, even though `final/` is what the multihop chain merges into the next-hop teacher. This distinction should be considered when comparing reported evaluation metrics to chained teacher behavior.

If `--extract_logprobs` is passed through the multihop runner, the evaluator additionally computes the log probability of the exact target preference string at the first answer position for each evaluation prompt. The current utility tokenizes the target string with the model tokenizer and sums autoregressive log probabilities across its token sequence. It saves a mean log probability and per-question rows in `logprob_stats.json`. This provides a generation-free preference measure complementary to sampled mention rates.

### Main Number-Completion Task

Main-task evaluation is implemented in `scripts/run_evaluation_preferences_main_task.py`. The evaluator loads the `train_indices` saved during fine-tuning and constructs two splits from the dataset passed to the evaluator: the selected training rows and the complement, truncated to at most 10,000 validation rows. It then tokenizes each example with the same chat-template structure and evaluates the model under teacher forcing.

The metric is numeric-string accuracy rather than raw token accuracy. For each masked completion, the evaluator decodes the ground-truth completion tokens and the model's argmax predictions, extracts digit substrings with a regular expression, and compares predicted and true numbers position by position. It reports aggregate accuracy and per-position accuracy for both the training split and the held-out complement. This evaluation checks whether the fine-tuned model still models the numeric completion format rather than only whether it expresses the hidden preference.

As in preference evaluation, `--final_ckpt_only` selects the last intermediate checkpoint and the base reference, not the `final/` directory.

### Factual Recall Probe

The factuality script evaluates whether the model can answer short factual prompts whose correct answer is a target animal. It loads questions from `cfgs/factual_recall/animal_questions.json`; when invoked by the multihop runner, it restricts evaluation to the current target animal via `--animal`. The default orchestrated setting uses 200 samples per question and includes the base model. Each question receives a randomly selected one-word-answer suffix using `random.Random(seed)`.

Responses are sampled with temperature 1.0, `max_new_tokens=10`, and batched generation. The statistic is computed using the same substring-based `compute_p_target_preference` function as the preference evaluation, with the target set to the relevant animal. For the provided factual question set this acts as an answer-accuracy proxy because each question is written so that the named animal is the intended answer. It is nevertheless a substring metric and does not use a separate semantic judge.

## Ablation Logic

The principal ablation compares three loss masks over the same visible data distribution:

* The `full` condition tests whether ordinary SFT on filtered numeric completions transfers the teacher preference.
* The `dpoints` condition tests whether the annotated sparse positions are sufficient for transfer, because only those positions contribute gradient.
* The `inverse` condition tests whether the complement of those positions is sufficient, because the annotated positions are removed from the loss.

This design is experimentally interesting because it holds prompts and completions fixed while altering which completion positions contribute to the SFT objective. It therefore probes the causal importance of a computationally identified subset of tokens without requiring a human-interpretable semantic difference in the data. Additional arguments in `run_finetuning.py` support subsampling decision points by ratio or by position subset (`first_only`, `first_half`, `second_half`), although the multihop orchestrator does not expose these as first-class flags.

The multihop chain adds a second ablation axis: whether later-hop data generation is explicitly re-prompted with the target preference. The default no-prompt chain asks whether the learned preference is self-propagating through the merged student model, whereas the prompted-subsequent-hop variant tests continued external steering.

## Reproducibility and Validity Considerations

Several implementation choices strengthen reproducibility. Prompt generation is seeded and deterministic; training subset indices are saved; run arguments are serialized; output directories encode dataset, mode, LoRA rank, and seed; generation and annotation outputs are made read-only after creation; and the multihop runner skips completed artifacts to support idempotent resumption.

There are also important limitations and ambiguities that should be disclosed in a paper:

* Completion sampling during dataset generation and evaluation is stochastic, and the generation script explicitly notes that the dataset seed does not control model completions. Exact regeneration may require external control of PyTorch and CUDA random states.
* Decision points are defined by argmax correctness under counterfactual system prompts, not by probability differences, KL divergence, or gradients. This can miss sub-argmax preference information.
* Decision-point annotation for later hops still uses the base model associated with the model alias under counterfactual system prompts. It is not computed by comparing the current-hop merged teacher to an unprompted version of itself.
* The dpoints and inverse conditions exclude rows with no decision points. Consequently, differences between `full` and the masked variants combine loss-mask effects with a change in the effective example set.
* The mapping from annotated decision-point indices to SFT completion-mask positions uses model-template-specific offsets. Changes in tokenizer versions or chat templates could silently alter the intended mask alignment.
* The tree category path appears under-specified in the multihop runner: generation and preference evaluation support trees, but the decision-point command does not forward the `--trees` flag required by the annotation script.
* The evaluation scripts' `--final_ckpt_only` option evaluates the last intermediate checkpoint plus the base model, whereas the multihop teacher is merged from the saved `final/` adapter. If the final save differs from the last checkpoint, evaluation and propagation are not perfectly aligned.
* Preference and factuality metrics are substring-based. This is simple and transparent, but it can count inflected or embedded mentions and does not distinguish semantically correct one-word answers from incidental target strings.

Taken together, the implemented methodology provides a controlled test of hidden preference transfer through numeric data, a sparse token-level ablation mechanism for localizing the training signal, and a recursive multihop protocol for studying whether learned preferences survive teacher-student iteration after explicit prompting is removed.

## High-Level Multihop Algorithm

The multihop protocol is a recursive teacher-student experiment. At hop $h$, the current teacher $T_h$ induces a numeric data distribution; student models are then trained on that induced distribution under different loss masks. A fixed chain selector promotes one trained student to become the next teacher. Thus, the object of study is not a single fine-tuning run, but the dynamical process
$T_h \rightarrow \widetilde{\mathcal{D}}_h \rightarrow S_{h,a,s} \rightarrow T_{h+1}$.

```latex
\begin{algorithm}[t]
\caption{Multihop Preference Propagation}
\label{alg:multihop}
\begin{algorithmic}[1]
\Require Base model $M_0$; target preference $z$; hops $H$;
training modes $\mathcal{A}=[\textsc{Full},\textsc{DPoint},\textsc{InvDPoint}]$;
seeds $\mathcal{S}$; chain selector $c=(a^\star,s^\star)\in\mathcal{A}\times\mathcal{S}$
\State $T_0 \gets M_0$
\For{$h = 0$ to $H-1$}
    \State $\pi_h \gets z$ if $h=0$, else $\varnothing$
    \State $\mathcal{D}_h \gets \mathrm{Gen}(T_h,\pi_h)$
    \State $\widetilde{\mathcal{D}}_h \gets \mathrm{Filt}(\mathcal{D}_h)$
    \State $\mathcal{P}_h \gets \mathrm{DP}(\widetilde{\mathcal{D}}_h; z,\mathcal{Z}_{\setminus z})$
    \For{$a \in \mathcal{A}$ and $s \in \mathcal{S}$}
        \State $S_{h,a,s} \gets \mathrm{Train}(M_0,\widetilde{\mathcal{D}}_h,\mathcal{P}_h,m_a,s)$
        \State $R_{h,a,s} \gets \mathrm{Eval}(S_{h,a,s})$
    \EndFor
    \If{$h < H-1$}
        \State $T_{h+1} \gets \mathrm{Merge}(S_{h,a^\star,s^\star})$
    \EndIf
\EndFor
\State \Return $\{S_{h,a,s}\}_{h,a,s}$, $\{R_{h,a,s}\}_{h,a,s}$, $\{T_h\}_{h=1}^{H-1}$
\end{algorithmic}
\end{algorithm}
```

Here, $M_0$ is the base pretrained model, $z$ is the target preference, $H$ is the number of hops, and $T_h$ is the teacher used at hop $h$. The prompt condition is $\pi_h$: in the default setting, $\pi_0=z$ and $\pi_h=\varnothing$ for $h>0$, so explicit preference prompting is removed after the first hop. $\mathcal{D}_h$ is the raw generated dataset, $\widetilde{\mathcal{D}}_h$ is its filtered numeric subset, and $\mathcal{P}_h$ is the set of decision-point annotations obtained by contrasting the target preference with counterfactual preferences $\mathcal{Z}_{\setminus z}$. The mode-specific mask $m_a$ determines whether training uses all completion tokens, only decision-point tokens, or the complement of decision-point tokens. The selector $c=(a^\star,s^\star)$ fixes which trained student is promoted, preventing evaluation choices from being adaptively chosen after observing results.

This presentation makes the central experimental claim explicit: if the target preference persists across hops, it does so through the teacher-induced data distribution and the selected student update, not through repeated explicit prompting.
