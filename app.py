import os
from fastapi import FastAPI, Depends, HTTPException, Header
from model_engine import TrainerEngine
import uvicorn

app = FastAPI(title="Secure AI Master API")
trainer = TrainerEngine()

# --- SECURITY MIDDLEWARE ---
def verify_admin(x_admin_password: str = Header(None)):
    # Fetch password from Environment Variables (Render Dashboard)
    secure_pass = os.getenv("ADMIN_PASSWORD", "admin123")
    if x_admin_password != secure_pass:
        raise HTTPException(status_code=403, detail="Access Denied: Invalid Admin Password")
    return True

# --- ENDPOINTS ---
@app.get("/")
async def health_check():
    return {"status": "online", "message": "AI System is ready"}

@app.get("/data-center", dependencies=[Depends(verify_admin)])
async def get_private_data():
    return {"secret_logs": "Log history...", "status": "Secure Access Granted"}

@app.post("/run-training", dependencies=[Depends(verify_admin)])
async def trigger_training():
    try:
        trainer.start_training()
        return {"message": "Model training initiated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_ai(prompt: str):
    # Logic to call the trained model
    return {"reply": "Processed by AI"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
