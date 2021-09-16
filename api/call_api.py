import fastapi
import uvicorn
import json

# Initialise API
api = fastapi.FastAPI()

# Define API resources

@api.get('/myresources')
def helloworld():
    return json.dumps({'data': 'Hello World'})

# uvicorn.run(api, port=8080, host='localhost') // type uvicorn call_api:api --reload