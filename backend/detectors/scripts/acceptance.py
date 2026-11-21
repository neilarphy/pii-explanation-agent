"""Capture a real response and check source spans; run explicitly on synthetic input only."""

import argparse
import json
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=Path("acceptance-output/response.json"))
    args = parser.parse_args()
    with urllib.request.urlopen(args.url + "/health/ready", timeout=10) as response:
        assert response.status == 200
    text = "🙂 Анна Иванова / John Smith: acceptance-847291@example.org"
    request = urllib.request.Request(
        args.url + "/v1/detect",
        data=json.dumps({"text": text}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = json.load(response)
        assert body["request_id"] == response.headers["X-Request-ID"]
        assert response.headers["Cache-Control"] == "no-store"
    trace = body["trace"]
    assert trace["model"]["status"] == trace["regex"]["status"] == "ok"
    assert trace["model"]["device"] == "cuda:0" and trace["regex"]["device"] == "cpu"
    assert trace["regex"]["engine"] == "python.re"
    assert any(c["source"] == "model" for c in trace["candidates"]), "no live model spans"
    assert any(c["source"] == "regex" and c["decision"] == "accepted" for c in trace["candidates"])
    for entity in body["entities"]:
        assert text[entity["start"] : entity["end"]] == entity["text"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n")
    print(f"Live response verified: {args.output}")


if __name__ == "__main__":
    main()
