
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from app.database import init_db
from app.routers import auth, scan
import os

app = FastAPI(title="BrainNet AI", version="2.0")

app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"])

app.mount("/static", StaticFiles(directory="/content/brainnet_fastapi/static"), name="static")

app.include_router(auth.router)
app.include_router(scan.router)

@app.on_event("startup")
def startup():
    init_db()

@app.get("/", response_class=HTMLResponse)
def index():
    with open("/content/brainnet_fastapi/templates/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    path = "/content/brainnet_fastapi/templates/dashboard.html"
    if os.path.exists(path):
        with open(path, "r") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Dashboard coming soon</h1>")

@app.get("/health")                                                      #irun
def health():
    return {"status": "BrainNet AI v2.0", "database": "SQLite"}
