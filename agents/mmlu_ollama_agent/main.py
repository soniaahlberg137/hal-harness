from openai import OpenAI

SYSTEM_PROMPT = (
    "You are answering a multiple-choice question. "
    "Reply with ONLY the letter of the correct answer: A, B, C, or D. "
    "Do not explain your reasoning."
)

_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


def run(input: dict[str, dict], **kwargs) -> dict[str, str]:
    assert "model_name" in kwargs, "model_name is required"
    assert len(input) == 1, "input must contain only one task"
    task_id, task = list(input.items())[0]
    model = kwargs["model_name"].removeprefix("ollama/")
    response = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task["question"]},
        ],
        max_tokens=10,
        temperature=0,
    )
    return {task_id: response.choices[0].message.content.strip()}
