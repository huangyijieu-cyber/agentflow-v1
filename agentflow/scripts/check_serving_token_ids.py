"""Probe the running AgentFlow VERL HTTP endpoint without re-tokenizing text."""
import argparse
import os

from openai import OpenAI


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True, help='Actual model serving URL ending in /v1, not the rollout queue URL')
    parser.add_argument('--model', required=True)
    parser.add_argument('--max-tokens', type=int, default=32)
    args = parser.parse_args()
    client = OpenAI(base_url=args.base_url, api_key=os.environ.get('OPENAI_API_KEY', 'dummy-token'), max_retries=0)
    response = client.chat.completions.create(
        model=args.model,
        messages=[{'role': 'user', 'content': 'What is 1 + 1? Answer briefly.'}],
        max_tokens=args.max_tokens,
        stream=False,
    )
    payload = response.model_dump()
    prompt = payload.get('prompt_token_ids')
    outputs = payload.get('response_token_ids')
    def valid_ids(ids):
        return isinstance(ids, list) and bool(ids) and all(type(token) is int and token >= 0 for token in ids)
    if not valid_ids(prompt):
        raise SystemExit('FAIL: prompt_token_ids missing/empty/invalid; inspect server TOKEN logs')
    if not isinstance(outputs, list) or len(outputs) != len(response.choices) or not all(valid_ids(ids) for ids in outputs):
        raise SystemExit('FAIL: response_token_ids must contain one nonempty integer list per choice; inspect server TOKEN logs')
    print('PASS: OpenAI client received prompt_token_ids and response_token_ids')
    print('prompt count:', len(prompt), 'prefix:', prompt[:12])
    print('response counts:', [len(ids) for ids in outputs])
    print('response prefixes:', [ids[:12] for ids in outputs])
    print('finish reasons:', [choice.finish_reason for choice in response.choices])
    print('This verifies transport, not model quality or a full training run.')


if __name__ == '__main__':
    main()
