"""Use the API like another program would.
1) start the server:   uvicorn api.main:app        2) in a second terminal:   python api_client_demo.py"""
import os
from pathlib import Path

import httpx

BASE_URL = "http://127.0.0.1:8000"


def run_demo(client, data_dir="data") -> dict:
    print("health:", client.get("/health").json())

    files = [("files", (p.name, p.read_bytes())) for p in sorted(Path(data_dir).glob("*"))
             if p.suffix.lower() in {".pdf", ".docx", ".csv", ".txt"}]
    for r in client.post("/documents", files=files).json()["results"]:
        print(f"  upload: {r['status']:10} {r['source']}")
    print("documents:", [(d["source"], d["chunks"]) for d in client.get("/documents").json()])

    history, last = [], {}
    for question in ["How many days of annual leave do employees get?", "What about sick leave?"]:
        r = client.post("/ask", json={"question": question, "history": history})
        r.raise_for_status()
        last = r.json()
        print(f"\nQ: {question}\n   searched as: {last['standalone_question']}\nA: {last['answer']}")
        print("   cited:", [f"{s['source']} ({s['location']})" for s in last["sources"] if s["cited"]])
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": last["answer"]}]
    return last


if __name__ == "__main__":
    key = os.getenv("API_KEY")
    with httpx.Client(base_url=BASE_URL, timeout=120, headers={"X-API-Key": key} if key else {}) as c:
        run_demo(c)
