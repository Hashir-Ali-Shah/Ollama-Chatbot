from langchain_core.prompts import ChatPromptTemplate,HumanMessagePromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate,MessagesPlaceholder
from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage,ToolMessage,BaseMessage
from langchain_core.runnables import RunnablePassthrough,ConfigurableField
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain.callbacks.base import AsyncCallbackHandler
from langchain_core.runnables.base import RunnableSerializable

import os
import asyncio



model = 'llama3.2:3b-instruct-fp16'
llm=ChatOllama(
    model=model,
    temperature=0.0,
    disable_streaming=False
).configurable_fields(
    callbacks=ConfigurableField(
        id='callbacks',
        name='callbacks',
        description='Callbacks to use for the model.',
)
)
SystemPrompt=SystemMessagePromptTemplate.from_template(
    "You are helpful assistant\nAnswer user queries using tools history and context" \
    # "Here are two contexts given to use help context start {context1} end {context2} context end \n" \
    "you are given some tools use them to answer the question\n"
    "Use only the final answer tool to answer the question\n"
    "below is memory of previous interactions with user \n" \
)
HumanPrompt=HumanMessagePromptTemplate.from_template(
    "memory end \n this is your main task \n {input} \n" \
)
prompt = ChatPromptTemplate.from_messages(
    [
        SystemPrompt,
        MessagesPlaceholder(variable_name="history"),
        HumanPrompt,
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
)
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from docx import Document
embeddings = OllamaEmbeddings(model="llama3.2:3b-instruct-fp16")


def load_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
content=load_docx(r"D:\ML\LangChain\MYPRoj\Ai_Research_Assistant_Project_Final.docx")

index1=r"D:\ML\LangChain\MYPRoj\Main\index1"
index2=r"D:\ML\LangChain\MYPRoj\Main\index2"
if os.path.exists(index1):
    vectorstore1=FAISS.load_local(index1,embeddings,allow_dangerous_deserialization=True)
else:
    lines=[line.strip() for line in content.split('\n') if line.strip()]
    vectorstore1=FAISS.from_texts(lines,embedding=embeddings)
    vectorstore1.save_local(index1)
    

if os.path.exists(index2):
    vectorstore2=FAISS.load_local(index2,embeddings,allow_dangerous_deserialization=True)
else:
    splitter=RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=20)
    chunks=splitter.split_text(content)
    vectorstore2=FAISS.from_texts(chunks,embedding=embeddings)
    vectorstore2.save_local(index2)


del content

from langchain_core.runnables import RunnablePassthrough,RunnableParallel
retriever1=vectorstore1.as_retriever(search_kwargs={"k": 3})
retriever2=vectorstore2.as_retriever(search_kwargs={"k": 1})
retriever=RunnableParallel(
    {
        "context1": retriever1,
        "context2": retriever2,
        "input": RunnablePassthrough()
    }
)



@tool
async def final_answer(answer: str, tools_used: list[str]) -> dict[str, str | list[str]]:
    """Use this tool to provide a final answer to the user."""
    return {"answer": answer, "tools_used": tools_used}
@tool
async def add(a:float , b:float )->float:
    """
    Use this tool to add two numbers 
    """
    return a + b
@tool
async def subtract(a:float , b:float )->float:
    """
    Use this tool to subtract second number from first number 
    """
    return a - b   
@tool
async def multiply(a:float , b:float )->float: 
    """
    Use this tool to multiply two numbers 
    """
    a = float(a) if isinstance(a, str) else a
    b = float(b) if isinstance(b, str) else b
    return a * b    
@tool
async def divide(a:float , b:float )->float:
    """
    Use this tool to divide first number by second number 
    """
    if b == 0:
        return "Cannot divide by zero"
    return a / b
@tool
async def square(a:float )->float:
    """
    Use this tool to square a number 
    """
    return a * a
@tool
async def square_root(a:float )->float:
    """
    Use this tool to find square root of a non negatvie number 
    """
    if a < 0:
        return "Cannot find square root of negative number"
    return a ** 0.5
@tool
async def power(a:float , b:float )->float:
    """
    Use this tool to find a raised to the power of b 
    """
    return a ** b   
@tool
async def factorial(a:int )->int:
    """
    Use this tool to find factorial of a non negative integer 
    """
    if a < 0:
        return "Cannot find factorial of negative number"
    if a == 0 or a == 1:
        return 1
    result = 1
    for i in range(2, a + 1):
        result *= i
    return result

tools= [final_answer,add,subtract,multiply,divide,square,square_root,power,factorial]
tools_dict = {tool.name: tool.coroutine for tool in tools}
class QueueCallBackHandler(AsyncCallbackHandler):
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue
        self.answer_done=False

    async def __aiter__(self):
        while True:
            if self.queue.empty():
                await asyncio.sleep(0.1)
                continue 
            token = await self.queue.get()
            if token == "--done--":
                return
            else:
                yield token
    async def on_llm_new_token(self, *args, **kwargs) -> None:

        chunk = kwargs.get('chunk')
     
    # Only process if chunk exists and has tool_calls
        if chunk and hasattr(chunk, 'message') and chunk.message.tool_calls:
          
            
            # Check if tool_calls list is not empty before accessing first element
            if len(chunk.message.tool_calls) > 0:
            
                if chunk.message.tool_calls[0]["name"] == "final_answer":
                 
                    self.answer_done = True
        
        # Always put the chunk in the queue (even if no tool calls)
        if chunk:
            self.queue.put_nowait(chunk)


    async def on_llm_end(self, *args, **kwargs) -> None:

        if self.answer_done:
            self.queue.put_nowait("<<DONE>>")
        else:
            self.queue.put_nowait("<<STEP_END>>")


async def execute_tool(tool_call: AIMessage) -> ToolMessage:
    tool_name = tool_call.tool_calls[0]["name"]
    tool_args = tool_call.tool_calls[0]["args"]
    tool_out = await tools_dict[tool_name](**tool_args)
    return ToolMessage(
        content=f"{tool_out}",
        tool_call_id=tool_call.tool_calls[0]["id"]
    )



class CustomAgentExecutor:
    def __init__(self, max_iterations: int = 3):
        self.history: list[BaseMessage] = []
        self.max_iterations = max_iterations
        self.agent = (
            {
                "input": lambda x: x["input"],
                "history": lambda x: x["history"],
                "agent_scratchpad": lambda x: x.get("agent_scratchpad", [])
            }
            # | retriever
            | prompt
            | llm.bind_tools(tools, tool_choice="any")
        )

    async def invoke(self, input: str, streamer: QueueCallBackHandler, verbose: bool = False) -> dict:
        # invoke the agent but we do this iteratively in a loop until
        # reaching a final answer
        count = 0
        final_answer: str | None = None
        agent_scratchpad: list[AIMessage | ToolMessage] = []
        # streaming function
        async def stream(query: str) -> list[AIMessage]:
          
            response = self.agent.with_config(
                callbacks=[streamer]
            )
  
            # we initialize the output dictionary that we will be populating with
            # our streamed output
            outputs = []
            # now we begin streaming
            async for token in response.astream({
                "input": query ,
                "history": self.history,
                "agent_scratchpad": agent_scratchpad
            }):
              
           
                tool_calls = token.tool_calls

                if tool_calls and len(tool_calls) > 0:
                    # Only append tokens that have a valid tool call ID
                    if tool_calls[0].get("id"):
                        outputs.append(token)
# Skip tokens with empty tool_calls or no ID
            return [
                AIMessage(
                    content=x.content,
                    tool_calls=x.tool_calls,
                    tool_call_id=x.tool_calls[0]["id"]
                ) for x in outputs
            ]

        while count < self.max_iterations:
            # invoke a step for the agent to generate a tool call
            tool_calls = await stream(query=input)
            # gather tool execution coroutines
            tool_obs = await asyncio.gather(
                *[execute_tool(tool_call) for tool_call in tool_calls]
            )
            # append tool calls and tool observations to the scratchpad in order
            id2tool_obs = {tool_call.tool_call_id: tool_obs for tool_call, tool_obs in zip(tool_calls, tool_obs)}
            for tool_call in tool_calls:
                agent_scratchpad.extend([
                    tool_call,
                    id2tool_obs[tool_call.tool_call_id]
                ])
            
            count += 1
            # if the tool call is the final answer tool, we stop
            found_final_answer = False
            for tool_call in tool_calls:
                if tool_call.tool_calls[0]["name"] == "final_answer":
                    final_answer_call = tool_call.tool_calls[0]
                    final_answer = final_answer_call["args"]["answer"]
                    found_final_answer = True
                    break
            
            # Only break the loop if we found a final answer
            if found_final_answer:
                break
            
        # add the final output to the chat history, we only add the "answer" field
        self.history.extend([
            HumanMessage(content=input),
            AIMessage(content=final_answer if final_answer else "No answer found")
        ])
        # return the final answer in dict form
        return final_answer_call if final_answer else {"answer": "No answer found", "tools_used": []}

# Initialize agent executor

agent_executor = CustomAgentExecutor()  