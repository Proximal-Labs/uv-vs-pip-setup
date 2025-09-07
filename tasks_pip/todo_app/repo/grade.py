from pathlib import Path
import sqlite3
from flask.testing import FlaskClient

# Fresh DB
db_path = Path("todo.db")
if db_path.exists():
    db_path.unlink()

import db_create  # creates tables
from app import app

client: FlaskClient = app.test_client()

def db_counts():
    with sqlite3.connect("todo.db") as c:
        cur = c.cursor()
        cur.execute("SELECT COUNT(*) FROM tasks"); t = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM done");  d = cur.fetchone()[0]
    return t, d

# Add two tasks
assert client.get("/addTask", query_string={"task": "Task1"}).status_code in (200, 302)
assert client.get("/addTask", query_string={"task": "Task2"}).status_code in (200, 302)
t, d = db_counts(); assert (t, d) == (2, 0)

# Fetch ids
with sqlite3.connect("todo.db") as c:
    cur = c.cursor()
    cur.execute("SELECT tid, task FROM tasks ORDER BY tid")
    tid1, task1 = cur.fetchone()
    cur.execute("SELECT tid FROM tasks WHERE tid != ?", (tid1,))
    tid2 = cur.fetchone()[0]

# Move first to done
assert client.get(f"/move-to-done/{tid1}/{task1}").status_code in (200, 302)
t, d = db_counts(); assert (t, d) == (1, 1)

# Delete remaining task
assert client.get(f"/deleteTask/{tid2}").status_code in (200, 302)
t, d = db_counts(); assert (t, d) == (0, 1)

# Delete completed item
with sqlite3.connect("todo.db") as c:
    cur = c.cursor()
    cur.execute("SELECT did FROM done LIMIT 1")
    did = cur.fetchone()[0]
assert client.get(f"/delete-completed/{did}").status_code in (200, 302)
t, d = db_counts(); assert (t, d) == (0, 0)

print("OK")