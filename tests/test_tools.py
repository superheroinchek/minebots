from minebots.tools import StylistInput, stylist_tool, thinker_tool


def test_thinker_returns_numbered_plan():
    prompt = "Почему небо голубое?"
    reasoning = thinker_tool(prompt)
    assert "План действий" in reasoning
    assert "Черновик ответа" in reasoning
    assert "1." in reasoning and "2." in reasoning


def test_stylist_references_user_and_reasoning():
    reasoning = (
        "Контекст запроса: объясни явление\n"
        "Цель ответа: дать пояснение\n"
        "План действий:\n1. Выделить ключевые факторы\n2. Проверить гипотезы\n"
        "Черновик ответа:\n"
        "Предлагаю разобрать ключевые факторы и потом подведу итог."
    )
    data = StylistInput(user_message="Объясни явление", reasoning=reasoning)
    final = stylist_tool(data)
    assert "Объясни явление" in final
    assert "Я рассуждал так" not in final
    assert "План" not in final
