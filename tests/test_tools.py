from minebots.tools import StylistInput, stylist_tool, thinker_tool


def test_thinker_returns_numbered_plan():
    prompt = "Почему небо голубое?"
    reasoning = thinker_tool(prompt)
    assert "План работы" in reasoning
    assert "1." in reasoning and "2." in reasoning
    assert "Ключ запроса" in reasoning


def test_stylist_references_user_and_reasoning():
    reasoning = "План работы:\n1. Выделить ключевые факторы\n2. Проверить гипотезы"
    data = StylistInput(user_message="Объясни явление", reasoning=reasoning)
    final = stylist_tool(data)
    assert "Объясни явление" in final
    assert "Вы задали вопрос" in final
    assert "Я рассуждал так" in final
    assert "•" in final
