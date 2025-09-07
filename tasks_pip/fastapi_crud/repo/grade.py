from fastapi.testclient import TestClient
from main import app


def main() -> None:
    client = TestClient(app)

    r = client.get("/")
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body, dict) and "message" in body

    # Create
    r = client.post("/todos/", json={"title": "t1", "description": "d", "status": "pending"})
    assert r.status_code == 200, r.text
    todo = r.json(); todo_id = todo["id"]
    assert todo["title"] == "t1"

    # Read list
    r = client.get("/todos/")
    assert r.status_code == 200, r.text
    items = r.json(); assert any(i["id"] == todo_id for i in items)

    # Read item
    r = client.get(f"/todos/{todo_id}")
    assert r.status_code == 200, r.text

    # Update
    r = client.put(f"/todos/{todo_id}", json={"status": "done"})
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "done"

    # Delete
    r = client.delete(f"/todos/{todo_id}")
    assert r.status_code == 200, r.text

    print("OK")


if __name__ == "__main__":
    main()


