from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn
import torch

app = FastAPI()

# CORS 허용 (HTML에서 JS 요청 가능하게)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 불러오기
tokenizer = AutoTokenizer.from_pretrained("app/google/gemma-2-2b-it")
model = AutoModelForCausalLM.from_pretrained("app/google/gemma-2-2b-it")

@app.get("/")
def root():
    return {"message": "AI 서버가 실행 중입니다."}

@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    input_text = data.get("question")

    input_ids = tokenizer(input_text, return_tensors="pt")
    output = model.generate(
        **input_ids,
        max_length=300,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
        temperature=0.7,
        top_p=0.9,
    )

    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"answer": answer}

if __name__=='__main__':
    uvicorn.run(app,host='127.0.0.1',port=8000)
