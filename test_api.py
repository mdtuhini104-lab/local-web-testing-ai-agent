import requests
import time
import json

def test():
    # Trigger a run
    resp = requests.post("http://127.0.0.1:8000/api/runs", json={
        "target_url": "https://mamun.aamardokan.online/",
        "username": "admin@example.com",
        "password": "@Admin123",
        "max_steps": 2,
        "headless": True
    })
    print("Response:", resp.json())
    run_id = resp.json().get("run_id")
    
    # Poll for status
    for _ in range(10):
        time.sleep(2)
        resp = requests.get(f"http://127.0.0.1:8000/api/runs/{run_id}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"Status: {data.get('status')}")
            if "failed" in data.get("status", "") or "completed" in data.get("status", ""):
                print("Run finished. Summary:", json.dumps(data.get("summary", {}), indent=2))
                break

if __name__ == "__main__":
    test()
