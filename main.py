import uvicorn
from fastapi import FastAPI, Query
from pydantic import BaseModel  # needs to get added in req..
import ai_llm_service
import llm_service
import concordia_llm_service
import uvicorn
import general_chat
import os

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


class ProcessRequestBody(BaseModel):
    user_prompt: str
    agent: str


class FeedbackRequestBody(BaseModel):
    agent: str
    feedback: str


@app.post("/process/")
async def process_prompt(request_data: ProcessRequestBody):
    user_prompt = request_data.user_prompt
    agent = request_data.agent
    if agent == "general":
        return {"response": general_chat.get_response(user_prompt)}
    elif agent == "ai":
        return {"response": ai_llm_service.ai_qa(user_prompt)}
    else:
        return {"response": concordia_llm_service.generate_response(user_prompt)}


@app.post("/feedback/")
async def process_request(feedback_data: FeedbackRequestBody):
    agent_mode = feedback_data.agent
    feedback = feedback_data.feedback
    general_chat.update_feedback(feedback_data.feedback)
    response = f'Got the feedback {feedback} from {agent_mode} Agent'
    return {"response": response}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
