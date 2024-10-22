from fastapi import FastAPI, HTTPException
import openai
from pydantic import BaseModel
import logging

logging.basicConfig(level=logging.INFO)
API_KEY = "OPENAI_API_KEY"
openai.api_key = API_KEY


app = FastAPI()


class prompt(BaseModel):
    question: str


@app.get("/")
async def welcome():
    return {"message": "gpt demo"}


@app.post("/prompt")
async def answer(item: prompt):
    prompt_txt = item.question
    logging.info(f"Prompt sent: {prompt_txt}")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_txt}],
            max_tokens=100,
            temperature=0.7,
        )
        logging.info(f"OpenAI API response: {response}")
        answer = response["choices"][0]["message"]["content"]
        return {"content": answer}

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error occurred: {str(e)}")
