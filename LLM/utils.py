
def split_input_text_link(input_text):
    separator = "## Target node1"

    system_content = input_text.split(separator)[0].strip()
    user_content = separator + input_text.split(separator)[1].strip()

    return system_content, user_content

def split_input_text(input_text):
    separator = "## Target node"

    system_content = input_text.split(separator)[0].strip()
    user_content = separator + input_text.split(separator)[1].strip()

    return system_content, user_content

def process_combination(combinations):

    filtered_combinations = []
    for combo in combinations:
        dataset, is_train, mode, zero_shot_CoT, BAG, few_shot, hop, include_label = combo

        if is_train and zero_shot_CoT:
            continue

        if is_train and BAG:
            continue

        if is_train and few_shot:
            continue

        if (zero_shot_CoT + BAG + few_shot) == 1 or (not zero_shot_CoT and not BAG and not few_shot):
            pass
        else:
            continue
        if mode == "ego" and (include_label or hop != 1):
            continue

        filtered_combinations.append({
            "dataset": dataset,
            "is_train": is_train,
            "mode": mode,
            "zero_shot_CoT": zero_shot_CoT,
            "BAG": BAG,
            "few_shot": few_shot,
            "hop": hop,
            "include_label": include_label
        })
    print(f"Number of combinations: {len(filtered_combinations)}", flush=True)
    return filtered_combinations

def process_combination_link(combinations):
    filtered_combinations = []
    for combo in combinations:
        dataset, include_title, case = combo

        if case != 0:
            continue

        filtered_combinations.append({
            "dataset": dataset,
            "include_title": include_title,
            "case": case
        })

def get_matched_option(prediction, valid_options):
    """
    Extracts options from the prediction string and returns the last matched option.

    Parameters:
    - prediction (str): The prediction string containing potential options.
    - valid_options (set): The set of valid options to match against.

    Returns:
    - str: The last matched option or an empty string if no matches are found.
    """
    matched_options = {}

    # Iteratively check each substring of the prediction
    for option in valid_options:
        if option in prediction:
            # matched_options.append(option)
            last_the_position = prediction.rfind(option)
            matched_options[last_the_position] = option
    if len(matched_options) == 0:
        return " "

    max_key = max(matched_options)

    max_value = matched_options[max_key]

    return max_value


    # Return the last matched option if available, else return an empty string
    # return matched_options[-1] if matched_options else ""