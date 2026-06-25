import re
from typing import Dict, Any
from datasets import load_dataset
from .base_benchmark import BaseBenchmark

_LABELS = ["A", "B", "C", "D"]


def _format_question(question: str, choices: list) -> str:
    lines = [question, ""]
    for label, choice in zip(_LABELS, choices):
        lines.append(f"{label}. {choice}")
    return "\n".join(lines)


def _extract_letter(text: str) -> str | None:
    """Extract the answer letter (A/B/C/D) from model output."""
    text = text.strip()
    # Exact single letter (possibly with trailing punctuation)
    if text.rstrip(".),:").upper() in ("A", "B", "C", "D"):
        return text.rstrip(".),:").upper()
    # Letter at the start: "A." "A)" "B " "C:"
    m = re.match(r"^([A-D])[.):\s]", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # "answer is X" / "answer: X"
    m = re.search(r"answer[:\s]+([A-D])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Fallback: first standalone A/B/C/D
    m = re.search(r"\b([A-D])\b", text, re.IGNORECASE)
    return m.group(1).upper() if m else None


class MMLUBenchmark(BaseBenchmark):
    """MMLU benchmark — single subject, multiple-choice (default: abstract_algebra, 100 questions)."""

    _ground_truth_keys = {"answer"}

    def __init__(
        self,
        agent_dir: str,
        config: Dict[str, Any],
        subject: str = "abstract_algebra",
        benchmark_name: str = "mmlu",
    ):
        self.benchmark_name = benchmark_name
        self.subject = subject
        self.requires_sandbox = False
        super().__init__(agent_dir, config, requires_sandbox=False)

        dataset = load_dataset("cais/mmlu", subject, split="test")
        self.benchmark = {
            f"mmlu_{i:04d}": {
                "task_id": f"mmlu_{i:04d}",
                "question": _format_question(record["question"], record["choices"]),
                "answer": _LABELS[record["answer"]],
            }
            for i, record in enumerate(dataset)
        }

    def evaluate_output(self, agent_output: Dict[str, Any], run_id: str) -> Dict[str, Any]:
        results = {}
        for task_id, agent_answer in agent_output.items():
            if isinstance(agent_answer, dict):
                agent_answer = agent_answer.get("answer", agent_answer.get("raw_response", ""))
            pred = _extract_letter(str(agent_answer))
            gt = self.benchmark[task_id]["answer"]
            correct = pred == gt
            results[task_id] = {
                "score": int(correct),
                "reward": float(correct),
                "correct": correct,
                "predicted": pred,
                "ground_truth": gt,
                "explanation": f"predicted={pred!r}, ground_truth={gt!r}",
            }
        return results

    def get_metrics(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        correct = sum(1 for v in eval_results.values() if v.get("score", 0) > 0)
        total = len(eval_results)
        return {
            "accuracy": correct / total if total > 0 else 0.0,
            "correct": correct,
            "total": total,
            "successful_tasks": [t for t, v in eval_results.items() if v.get("score", 0) > 0],
            "failed_tasks": [t for t, v in eval_results.items() if v.get("score", 0) == 0],
        }
