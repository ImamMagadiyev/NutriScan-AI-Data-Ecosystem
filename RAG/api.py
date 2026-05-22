import asyncio
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from agent_langchain import NutritionAgent

app = FastAPI()
agent = NutritionAgent()

class Question(BaseModel):
    message: str
    product_hint: str | None = None
    barcode: str | None = None

@app.post("/chat")
async def chat(q: Question):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: agent.demander(
        question=q.message,
        indiceProduit=q.product_hint,
        codebarres=q.barcode,
    ))
    return JSONResponse(
        content={"response": result["reponse"], "intent": result["intention"]},
        media_type="application/json; charset=utf-8"
    )

@app.delete("/memory")
async def clear_memory():
    agent.clear_memory()
    return {"status": "mémoire effacée"}