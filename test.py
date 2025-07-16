import requests

def get_stream(query: str):
    localsession = requests.Session()
    with localsession.post(
        f"http://localhost:8000/invoke?content={query}",
        stream=True,
        headers={"Accept": "text/event-stream"}
    ) as response:
        for token in response.iter_content(decode_unicode=True):
            if token:  # Skip empty lines
                print(token, end="", flush=True)

# Test with a simple math question
get_stream("what is your model name")