import sys
import os
import random
sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))
# os.environ["HF_HOME"] = '/scratch/pioneer/jobs/wxy320/huggingface'

from utils.math_parsing_util import (
    extract_answer,
    math_equal,
    strip_answer_string,
)
def check_correctness(generation, ans):
    try:
        answer = strip_answer_string(ans)
        pred = extract_answer(generation)
        pred = strip_answer_string(pred)
        return math_equal(pred, answer)
    except Exception as e:
        print(f"⚠️ Error checking correctness: {e}")
        return False
    

def get_multiple_choice_answers(data):
    answers = [
        data["Correct Answer"],
        data["Incorrect Answer 1"],
        data["Incorrect Answer 2"],
        data["Incorrect Answer 3"],
    ]
    random.shuffle(answers)

    # Map options to letters
    options = ["A", "B", "C", "D"]
    options_to_answers = {
        letter: answer for letter, answer in zip(options, answers)
    }

    # Format the options into the string
    multiple_choice_string = ", ".join(
        f"{letter}) {options_to_answers[letter]}" for letter in options
    )

    # Save the letter corresponding to the correct answer
    correct_answer_letter = next(
        letter
        for letter, answer in options_to_answers.items()
        if answer == data["Correct Answer"]
    )

    return multiple_choice_string, correct_answer_letter