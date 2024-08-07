from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
import os
from app.model_loader import load_model
from app.model_inference import model_inference

app = FastAPI()

# Load the model once when the server starts
model = load_model()

@app.post("/inference")
async def inference(file: UploadFile = File(...)):
    input_filename = file.filename
    input_file_path = f"uploads/{input_filename}"
    os.makedirs(os.path.dirname(input_file_path), exist_ok=True)
    with open(input_file_path, "wb") as f:
        f.write(await file.read())
    
    output_dir = "output"
    try:
        output_file = model_inference(model, input_file_path, output_dir)
        return JSONResponse(content={"output_file": output_file})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
