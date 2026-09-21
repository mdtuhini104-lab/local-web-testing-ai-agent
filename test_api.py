import requests
from requests.exceptions import ConnectionError, Timeout
import time
import json

def test():
    try:
        # Trigger a run
        resp = requests.post("http://127.0.0.1:8000/api/runs", json={
            "target_url": "https://mamun.aamardokan.online/",
            "username": "admin@example.com",
            "password": "@Admin123",
            "max_steps": 2,
            "headless": True
        }, timeout=15)
        print("Status:", resp.status_code)
        print("Response:", resp.json())
        run_id = resp.json().get("run_id")
        
        # Poll for status
        for _ in range(10):
            time.sleep(2)
            try:
                resp = requests.get(f"http://127.0.0.1:8000/api/runs/{run_id}", timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    print(f"Status: {data.get('status')}")
                    if "failed" in data.get("status", "") or "completed" in data.get("status", ""):
                        print("Run finished. Summary:", json.dumps(data.get("summary", {}), indent=2))
                        break
            except (ConnectionError, Timeout) as poll_err:
                print(f"[!] Polling connection warning: {poll_err}")
    except (ConnectionError, Timeout) as e:
        print(f"[!] Error: Backend server is not reachable or request timed out ({e}). Ensure `python server.py` is actively running on port 8000.")
    except Exception as exc:
        print(f"[!] Unexpected error during API test: {exc}")

if __name__ == "__main__":
    test()

