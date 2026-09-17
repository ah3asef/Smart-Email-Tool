"""Evaluation utilities for scoring GenerationService outputs."""

from __future__ import annotations

from typing import Any, Mapping

from services.GenerationService import GenerationResult


NOT_FOUND_PHRASES = [
    "couldn't find",
    "could not find",
    "not found",
    "not mentioned",
    "insufficient",
    "don't have",
    "do not have",
]


def evaluate_result(
    question: str,
    result: GenerationResult,
    context_emails: list[Mapping[str, Any]],
    expected_found: bool,
) -> dict[str, Any]:
    """Score one GenerationResult against expected behavior.

    Args:
        question: the user question asked.
        result: the GenerationResult returned by GenerationService.generate().
        context_emails: the raw retrieved emails (for overlap scoring).
        expected_found: whether a correct answer should exist in context_emails.

    Returns:
        A dict of evaluation metrics for this single question.
    """
    answer_lower = result.answer.lower()

    admits_not_found = any(phrase in answer_lower for phrase in NOT_FOUND_PHRASES)

    context_text = " ".join(
        f"{e.get('subject', '')} {e.get('body', '')}" for e in context_emails
    ).lower()
    context_words = set(context_text.split())
    answer_words = set(answer_lower.split())
    context_overlap = (
        round(len(context_words & answer_words) / max(len(context_words), 1), 2)
        if context_words else 0.0
    )

    has_sources = len(result.sources) > 0

    # correctness: did the system's behavior match what we expected?
    if expected_found:
        is_correct = not admits_not_found
    else:
        is_correct = admits_not_found

    return {
        "question": question,
        "expected_found": expected_found,
        "admits_not_found": admits_not_found,
        "is_correct": is_correct,
        "context_overlap": context_overlap,
        "has_sources": has_sources,
        "num_sources": len(result.sources),
        "answer_length_words": len(result.answer.split()),
        "answer": result.answer,
    }


def evaluate_variant(
    variant_name: str,
    test_cases: list[tuple[str, bool]],
    generation_service: Any,
    context_emails: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Run every test case through one prompt variant and summarize accuracy."""
    results = []
    for question, expected_found in test_cases:
        result = generation_service.generate(query=question)
        eval_row = evaluate_result(question, result, context_emails, expected_found)
        results.append(eval_row)

    accuracy = round(
        sum(1 for r in results if r["is_correct"]) / len(results), 2
    ) if results else 0.0

    return {
        "variant": variant_name,
        "accuracy": accuracy,
        "total_cases": len(results),
        "correct_cases": sum(1 for r in results if r["is_correct"]),
        "details": results,
    }