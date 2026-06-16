"""Simple Flask todo app.

Run:
    flask --app web.app run --debug --host 0.0.0.0 --port 5000
"""
from itertools import count

from flask import Flask, redirect, render_template, request, url_for

app = Flask(__name__)

_ids = count(1)
todos = [
    {"id": next(_ids), "title": "Flask todo app 실행해보기", "done": False},
]


@app.get("/")
def index():
    remaining = sum(1 for todo in todos if not todo["done"])
    return render_template("index.html", todos=todos, remaining=remaining)


@app.post("/todos")
def add_todo():
    title = request.form.get("title", "").strip()
    if title:
        todos.append({"id": next(_ids), "title": title, "done": False})
    return redirect(url_for("index"))


@app.post("/todos/<int:todo_id>/toggle")
def toggle_todo(todo_id):
    todo = _find_todo(todo_id)
    if todo:
        todo["done"] = not todo["done"]
    return redirect(url_for("index"))


@app.post("/todos/<int:todo_id>/delete")
def delete_todo(todo_id):
    todo = _find_todo(todo_id)
    if todo:
        todos.remove(todo)
    return redirect(url_for("index"))


@app.post("/todos/clear-completed")
def clear_completed():
    todos[:] = [todo for todo in todos if not todo["done"]]
    return redirect(url_for("index"))


def _find_todo(todo_id):
    return next((todo for todo in todos if todo["id"] == todo_id), None)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
