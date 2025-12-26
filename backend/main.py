from fastapi import FastAPI, UploadFile, File
from audio import save_audio  
from features import extract

app = FastAPI()

@app.get("/")
def root():
    return {"Status" : "Backend Running."}

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    path = await save_audio(file)
    return {"message": "File uploaded", "path": path}

'''
 We are using File(...) as the File upload is necessary and if user does not upload any file,
 automatic error is given to the user, otherwise if we use File(None), user may or may 
 not upload file.
 '''

@app.post("/analyse")
async def analyse_audio(file: UploadFile = File()):
    path = await save_audio(file)
    features = extract(path)
    return {"features" : list(features)}