"""Evaluate QA with ChatGPT."""
# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import argparse
import json
import os
import time

import openai
import tqdm

import ray

@ray.remote
def get_eval(rule: str, user: str, assistant1: str, assistant2: str, max_tokens: int):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{
            'role': 'system',
            'content': 'You are a helpful and precise assistant for checking the quality of the answer.'
        }, {
            'role': 'user',
            'content': (f'[User]\n{user}\n[Assistant 1]\n{assistant1}\n'
                        f'[Assistant 2]\n{assistant2}\n[system]\n{rule}'),
        }],
        # temperature=0.2,  # TODO: figure out which temperature is best for evaluation
        max_tokens=max_tokens,
    )
    return response['choices'][0]['message']['content']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-q', '--question')
    # parser.add_argument('-a', '--answer')
    parser.add_argument('-a', '--answer-list', nargs='+', default=[])
    parser.add_argument('-r', '--rule')
    parser.add_argument('-o', '--output')
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    args = parser.parse_args()

    ray.init()

    with open(os.path.expanduser(args.question)) as f:
        question = json.load(f)
        questions_dict = {q['id']: q['question'] for q in question['questions']}

    answers_file_1 = args.answer_list[0]
    answers_file_2 = args.answer_list[1]
    
    def get_ans_dict(answers_file):
        with open(os.path.expanduser(answers_file)) as f:
            answer = json.load(f)
            answers_dict = {ans['id']: ans['answer'] for ans in answer['answers']}
        return answers_dict
    
    answers_dict_1 = get_ans_dict(answers_file_1)
    answers_dict_2 = get_ans_dict(answers_file_2)
    
    with open(os.path.expanduser(args.rule)) as f:
        rule = f.read()

    evaluations = []
    eval_result_handle = []
    scores = [0, 0]
    for qid, question in tqdm.tqdm(questions_dict.items()):
        answer_1 = answers_dict_1.get(qid)
        answer_2 = answers_dict_2.get(qid)
        if answer_1 is None or answer_2 is None:
            evaluations.append({'id': qid, 'score': 0, 'explanation': 'Could not find the answer.'})
            continue
        eval_result_handle.append(get_eval.remote(rule, question, answer_1, answer_2, args.max_tokens))
    
    for eval_result in ray.get(eval_result_handle):
        score, explanation = eval_result.split('\n', 1)
        evaluations.append({'id': qid, 'score': int(score), 'explanation': explanation})
        scores[int(score)-1] += 1
    print(scores)
    with open(os.path.expanduser(args.output), 'w') as f:
        json.dump(evaluations, f)
