"""Verify vLLM 0.11 native token fields through the OpenAI client."""
import argparse
import os

from openai import OpenAI


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True, help='Serving/proxy URL ending in /v1')
    parser.add_argument('--model', required=True)
    parser.add_argument('--max-tokens', type=int, default=32)
    args = parser.parse_args()
    client = OpenAI(base_url=args.base_url, api_key=os.environ.get('OPENAI_API_KEY', 'dummy-token'), max_retries=0)
    response = client.chat.completions.create(
        model=args.model,
        messages=[{'role': 'user', 'content': 'What is 1 + 1? Answer briefly.'}],
        max_tokens=args.max_tokens,
        stream=False,
        # [修改目的] 使用 vLLM 0.11 原生开关，不安装 serving monkey patch。
        extra_body={'return_token_ids': True},
    )
    payload = response.model_dump()
    prompt = payload.get('prompt_token_ids')
    # [原代码保留] outputs = payload.get('response_token_ids')
    # [修改目的] 真实 HTTP 协议为 choices[i].token_ids；顶层 response_token_ids 不存在。
    choices = payload.get('choices') or []
    outputs = [choice.get('token_ids') for choice in choices]

    def valid_ids(ids):
        return isinstance(ids, list) and bool(ids) and all(type(token) is int and token >= 0 for token in ids)

    if not valid_ids(prompt):
        raise SystemExit('FAIL: prompt_token_ids missing/empty/invalid')
    # [原代码保留] if not isinstance(outputs, list) or len(outputs) != len(response.choices) or not all(valid_ids(ids) for ids in outputs):
    # [修改目的] 每个实际 choice 都必须提供非空真实 token 序列。
    if not choices or len(outputs) != len(response.choices) or not all(valid_ids(ids) for ids in outputs):
        # [原代码保留] raise SystemExit('FAIL: response_token_ids must contain one nonempty integer list per choice; inspect server TOKEN logs')
        raise SystemExit('FAIL: every choices[i].token_ids must be a nonempty integer list')
    # [原代码保留] print('PASS: OpenAI client received prompt_token_ids and response_token_ids')
    # [修改目的] 验收原生字段，不要求制造顶层 response_token_ids。
    print('PASS: OpenAI client received prompt_token_ids and response token ids')
    print('prompt count:', len(prompt), 'prefix:', prompt[:12])
    print('response counts:', [len(ids) for ids in outputs])
    print('response prefixes:', [ids[:12] for ids in outputs])
    print('finish reasons:', [choice.finish_reason for choice in response.choices])
    print('This verifies transport, not model quality or a full training run.')


if __name__ == '__main__':
    main()
