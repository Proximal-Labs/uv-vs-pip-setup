
from fastapi import FastAPI
from database.database import Base, engine
from routers import todo

Base.metadata.create_all(bind=engine)

app = FastAPI()
app.include_router(todo.router, prefix="/todos", tags=["Todos"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the Enhanced FastAPI Todo App!"}
