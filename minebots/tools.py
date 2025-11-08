"""Utility tools used by the miniature reasoning agent.

The functions defined here are lightweight replacements for the much heavier
LLM-backed tools that existed in the prototype script provided by the user. The
focus is on deterministic, testable behaviour that still mimics deeper
reasoning and stylistic polishing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple


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


def _build_plan_and_draft(question: str, lowered: str) -> Tuple[Iterable[str], str, str]:
    """Return the action plan, goal description and draft reply."""

    if not question:
        plan = (
            "Уточнить у собеседника детали запроса",
            "Выяснить цель общения",
            "Предложить конкретную помощь",
        )
        goal = "Понять, что именно требуется, прежде чем давать ответ."
        draft = "Расскажите, пожалуйста, что именно вас интересует, и я помогу."
        return plan, goal, draft

    # Greetings
    if any(token in lowered for token in ("привет", "здравств", "hi", "hello")) and len(question.split()) <= 6:
        plan = (
            "Приветствовать пользователя в доброжелательном тоне",
            "Коротко подтвердить готовность помочь",
            "Пригласить уточнить тему обращения",
        )
        goal = "Сохранить дружелюбный тон и показать готовность к диалогу."
        draft = "Привет! Я на связи и готов помочь — расскажи, что хочется обсудить."
        return plan, goal, draft

    if "почему" in lowered:
        plan = (
            "Выделить ключевые факторы задачи",
            "Определить причинно-следственные связи",
            "Проверить альтернативные объяснения",
            "Сформулировать убедительный вывод",
        )
        goal = "Дать понятное объяснение с перечислением главных причин."
        draft = (
            "Давайте разберёмся по шагам и посмотрим, какие факторы влияют на ситуацию; "
            "после анализа я соберу объяснение."
        )
        return plan, goal, draft

    if any(token in lowered for token in ("матем", "посчитай", "формула", "считай", "числ")):
        plan = (
            "Собрать исходные данные и обозначить переменные",
            "Выбрать подходящую формулу или метод вычислений",
            "Подставить значения и выполнить расчёты",
            "Проверить размерности, погрешности и итоговый результат",
        )
        goal = "Получить корректный числовой ответ с проверкой шагов."
        draft = (
            "Сначала соберём все исходные числа и условия, затем подберём формулу, "
            "посчитаем и проверим результат."
        )
        return plan, goal, draft

    plan = (
        "Выяснить конкретную цель пользователя",
        "Собрать ограничения и важные факты",
        "Сравнить возможные варианты решения",
        "Выбрать оптимальный подход и предложить следующие шаги",
    )
    goal = "Дать предметный ответ с рекомендациями по дальнейшим шагам."
    draft = (
        "Соберём важные детали задачи, сопоставим варианты и предложим решение, "
        "которое лучше всего подходит под ваши условия."
    )
    return plan, goal, draft


def thinker_tool(question: str) -> str:
    """Produce a structured reasoning trace for ``question``.

    In addition to шаги анализа, the function now формирует «Черновик ответа»,
    который затем используется стилистом для финальной формулировки.
    """

    question = (question or "").strip()
    lowered = question.lower()
    plan, goal, draft = _build_plan_and_draft(question, lowered)

    numbered_plan = [f"{idx}. {step}" for idx, step in enumerate(plan, start=1)]
    summary = _normalise_text(question) or "(запрос не указан)"

    sections = [
        f"Контекст запроса: {summary}",
        f"Цель ответа: {goal}",
        "План действий:",
        *numbered_plan,
        "Черновик ответа:",
        draft,
    ]
    return "\n".join(sections)


def _extract_draft(reasoning: str) -> str:
    marker = "Черновик ответа:"
    if marker not in reasoning:
        return ""
    draft = reasoning.split(marker, 1)[1]
    return draft.strip()


def stylist_tool(stylist_input: StylistInput) -> str:
    """Transform the reasoning into a friendly, helpful response.

    Вместо пересказа плана функция берёт «Черновик ответа» из рассуждений и
    доводит его до готового сообщения, сохраняя связь с исходным запросом.
    """

    if not isinstance(stylist_input, StylistInput):
        raise TypeError("stylist_tool expects a StylistInput instance")

    user_message = _normalise_text(stylist_input.user_message)
    draft = _extract_draft(stylist_input.reasoning)

    if not draft:
        reasoning = _normalise_text(stylist_input.reasoning)
        if not reasoning:
            return (
                "Я готов помочь, но сначала нужен черновик рассуждений, чтобы "
                "переформулировать его аккуратно."
            )
        draft = reasoning

    # Ensure the response starts with a capital letter and ends politely.
    draft = draft.strip()
    if not draft:
        draft = "Готов подключиться и помочь — расскажите, что вас интересует."

    first_char = draft[0]
    if first_char.isalpha():
        draft = first_char.upper() + draft[1:]

    closing = " Если нужны уточнения или помощь с деталями, просто напишите."
    if draft.endswith(tuple(".!?")):
        final_message = draft + closing
    else:
        final_message = draft + "." + closing

    if user_message and user_message.lower() not in draft.lower():
        final_message = f"Запрос: «{user_message}». {final_message}"

    return final_message


__all__ = ["thinker_tool", "stylist_tool", "StylistInput"]
