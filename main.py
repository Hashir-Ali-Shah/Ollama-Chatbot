import asyncio

from Chat import QueueCallBackHandler, agent_executor
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# initilizing our application
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# streaming function
async def token_generator(content: str, streamer: QueueCallBackHandler):
    task = asyncio.create_task(agent_executor.invoke(
        input=content,
        streamer=streamer,
        verbose=True
    ))
    
    async for token in streamer:
        
        try:
        
            
            if isinstance(token, str):
                if token == "<<STEP_END>>":
                    yield "</step>"
                elif token == "<<DONE>>":
                    break
                continue
        
            # Handle ChatGenerationChunk tokens
            if hasattr(token, 'message') and token.message.tool_calls:
                
                tool_calls = token.message.tool_calls
                if tool_calls and len(tool_calls) > 0:
                    # Get tool name
                    if tool_name := tool_calls[0].get("name"):
                        # yield f"<step><step_name>{tool_name}</step_name>"
                        pass
                    
                    # Get tool args (note: it's "args", not "arguments")
                    if tool_args := tool_calls[0].get("args"):
                        yield str(tool_args['answer'])  # Convert to string for streaming
            
        except Exception as e:
            print(f"Error processing token: {e}")
            continue
    await task

# invoke function
@app.post("/invoke")
async def invoke(content: str):
    queue: asyncio.Queue = asyncio.Queue()
    streamer = QueueCallBackHandler(queue)
    # return the streaming response
    return StreamingResponse(
        token_generator(content, streamer),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
