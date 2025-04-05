from fastapi import FastAPI, Query

import ai_llm_service
import llm_service

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/process/")
async def process_prompt(prompt: str, parameter: str = Query(...,
                                                             description="Required processing mode (e.g., ai, general or concordia)")):
    if parameter == "general":
        return {"response": llm_service.general_qa(prompt)}
    elif parameter == "ai":
        return {"response": ai_llm_service.ai_qa(prompt)}
    # else:
    #     return{"response": concordia_llm.get_response(prompt)}
