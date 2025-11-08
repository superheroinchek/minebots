import json
import time
import urllib.error
import urllib.request

from minebots.web import create_server


def _wait_for_server(url: str, attempts: int = 10, delay: float = 0.05) -> None:
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(url) as response:  # noqa: S310 - local test server
                if response.status == 200:
                    return
        except urllib.error.URLError:
            time.sleep(delay)
    raise AssertionError("server did not respond in time")


def test_index_and_chat_roundtrip():
    server, thread = create_server()
    port = server.server_address[1]
    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_server(f"{base_url}/")
        with urllib.request.urlopen(f"{base_url}/") as response:  # noqa: S310 - local test server
            html = response.read().decode("utf-8")
        assert "Minebots" in html

        payload = json.dumps({"message": "Привет, расскажи план действий"}).encode("utf-8")
        request = urllib.request.Request(
            f"{base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request) as response:  # noqa: S310 - local test server
            data = json.loads(response.read().decode("utf-8"))
        assert "reasoning" in data and "final_answer" in data
        assert "План действий" in data["reasoning"]
        assert "Черновик ответа" in data["reasoning"]
        assert "План" not in data["final_answer"]  # финальный ответ без пересказа шагов
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)
