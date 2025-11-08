"""Utility tools used by the miniature reasoning agent.

The functions defined here are lightweight replacements for the much heavier
LLM-backed tools that existed in the prototype script provided by the user. The
focus is on deterministic, testable behaviour that still mimics deeper
reasoning and stylistic polishing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class StylistInput:
    """Container passed to :func:`stylist_tool`.

    The stylist needs to know both what the user asked and how the thinker
    reasoned about the request so that the final wording can acknowledge both
    viewpoints.
    """

    user_message: str
    reasoning: str


def _normalise_text(text: str) -> str:
    return " ".join(text.strip().split())


def thinker_tool(question: str) -> str:
    """Produce a structured plan describing how to approach ``question``.

    The implementation keeps the logic lightweight while still providing more
    detailed reasoning than the original placeholder. It highlights the user's
    intent, extracts key topics and gives a numbered action plan.
    """

    question = (question or "").strip()
    if not question:
        return "План: 1) уточнить запрос; 2) собрать контекст; 3) предложить шаги решения."

    lowered = question.lower()
    focus: Iterable[str] = []
    if "почему" in lowered:
        focus = (
            "Выделить ключевые факторы",
            "Определить причинно-следственные связи",
            "Проверить альтернативные гипотезы",
            "Сформулировать итоговое объяснение",
        )
    elif any(token in lowered for token in ("матем", "посчитай", "формула", "считай")):
        focus = (
            "Собрать исходные данные",
            "Подобрать подходящие формулы",
            "Подставить значения и вычислить",
            "Проверить размерности и погрешности",
            "Сформировать итоговый ответ",
        )
    else:
        focus = (
            "Выделить цели пользователя",
            "Собрать релевантные факты и ограничения",
            "Сравнить возможные варианты решения",
            "Выбрать оптимальный сценарий и сформулировать вывод",
        )

    plan_lines = [
        "План работы:",
        *(
            f"{idx}. {step}"
            for idx, step in enumerate(focus, start=1)
        ),
    ]

    summary = _normalise_text(question)
    plan_lines.append(f"Ключ запроса: «{summary}».")
    return "\n".join(plan_lines)


def stylist_tool(stylist_input: StylistInput) -> str:
    """Transform the reasoning into a friendly, concise response.

    The stylist acknowledges what was analysed and mirrors the user's wording
    to ensure that the answer sounds attentive. If the thinker produced bullet
    points, they are converted into a tidy list in the final message.
    """

    if not isinstance(stylist_input, StylistInput):
        raise TypeError("stylist_tool expects a StylistInput instance")

    reasoning = _normalise_text(stylist_input.reasoning)
    user_message = _normalise_text(stylist_input.user_message)

    if not reasoning:
        return (
            "Я готов помочь, но сначала нужен черновик рассуждений, чтобы "
            "переформулировать его аккуратно."
        )

    bullet_parts = [part for part in stylist_input.reasoning.splitlines() if part.strip()]
    formatted_plan = "\n".join(f"• {part.strip()}" for part in bullet_parts)

    return (
        "Вот что удалось выяснить и как это соотносится с вашим запросом:\n\n"
        f"Вы задали вопрос: «{user_message}».\n\n"
        f"Я рассуждал так:\n{formatted_plan}\n\n"
        "Исходя из плана, предлагаю перейти к практической реализации или "
        "уточнить детали, если что-то осталось непонятным."
    )


__all__ = ["thinker_tool", "stylist_tool", "StylistInput"]
