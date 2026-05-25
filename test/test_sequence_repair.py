from sl.datasets.data_models import DatasetRow
from sl.datasets.sequence_repair import repair_completion, repair_dataset_rows


def test_repair_preserves_values_order_and_restarts():
    prompt = "Return one number per line."
    completion = "812\n735\n\nmodel\nOkay, here are the numbers:\n\n812\n735"

    assert repair_completion(prompt, completion) == "812\n735\n812\n735"


def test_repair_does_not_cap_or_drop_out_of_range_values():
    prompt = "Return a comma-separated list of numbers."
    completion = "1001, 2002, 3, 4, 5, 6, 7, 8, 9, 10, 11"

    assert repair_completion(prompt, completion) == "1001, 2002, 3, 4, 5, 6, 7, 8, 9, 10, 11"


def test_repair_uses_prompt_wrapper_format():
    prompt = "Return the numbers in the following format: [number_1, number_2, ...]."
    completion = "12; 34; 56\nmodel\n12; 34"

    assert repair_completion(prompt, completion) == "[12, 34, 56, 12, 34]"


def test_repair_detects_space_and_semicolon_phrases():
    assert (
        repair_completion("List the numbers with spaces between them.", "1, 2, 3")
        == "1 2 3"
    )
    assert (
        repair_completion("Return numbers separated by semicolons.", "1 2 3")
        == "1; 2; 3"
    )


def test_repair_falls_back_to_completion_delimiter_for_unknown_prompt():
    assert repair_completion("Give numbers.", "1; 2; 3") == "1; 2; 3"
    assert repair_completion("Give numbers.", "1\n2\n3") == "1\n2\n3"


def test_repair_ignores_explanatory_text_numbers():
    prompt = "Format as a simple comma-delimited sequence."
    completion = "The sequence has 3 numbers.\n10, 20, 30\nmodel\nDone in 2 steps."

    assert repair_completion(prompt, completion) == "10, 20, 30"


def test_repair_dataset_rows_returns_new_rows():
    rows = [DatasetRow(prompt="Return one number per line.", completion="1, 2, 3")]

    repaired = repair_dataset_rows(rows)

    assert rows[0].completion == "1, 2, 3"
    assert repaired[0].completion == "1\n2\n3"
