from fastapi import FastAPI, Query
from pydantic import BaseModel
import uvicorn
import os

from backend.chat_agents import general_assistant
from backend.chat_agents import ai_llm_service
from backend.chat_agents import concordia_llm_service

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Welcome to Multi-Agent Virtual Assistant"}


class ProcessRequestBody(BaseModel):
    user_prompt: str
    agent: str


class FeedbackRequestBody(BaseModel):
    agent: str
    feedback: str


@app.post("/process/")
async def process_prompt(request_data: ProcessRequestBody):
    try:
        user_prompt = request_data.user_prompt
        agent = request_data.agent
        if agent == "general":
            return {"response": general_assistant.get_response(user_prompt)}
        elif agent == "ai":
            return {"response": ai_llm_service.ai_qa(user_prompt)}
        else:
            return {"response": concordia_llm_service.generate_response(user_prompt)}
    except Exception as e:
        print(f"Error generating response: {e}")
        return {"response": "Sorry, I encountered an error while processing your request. Please try again!!"}


@app.post("/feedback/")
async def process_request(feedback_data: FeedbackRequestBody):
    try:
        agent_mode = feedback_data.agent
        feedback = feedback_data.feedback
        if agent_mode == "general":
            general_assistant.update_feedback(feedback_data.feedback)
        elif agent_mode == "ai":
            ai_llm_service.update_feedback(feedback_data.feedback)
        else:
            concordia_llm_service.update_feedback(feedback_data.feedback)

        response = f'Got the feedback {feedback} from {agent_mode} Agent'
        return {"response": response}
    except Exception as e:
        print(f"Error updating feedback: {e}")
        return {"response": "Error while updating feedback. Please try again!!"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
