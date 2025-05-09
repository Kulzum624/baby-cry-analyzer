from app import app

# This is the entry point for PythonAnywhere
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 