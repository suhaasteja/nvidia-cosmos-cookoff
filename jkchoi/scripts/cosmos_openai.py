from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-used")
model = client.models.list().data[0].id

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
        "content": [
            {
                "type": "video_url",
                "video_url": {
                    "url": "https://download.samplelib.com/mp4/sample-5s.mp4"
                }
            },
            {
                "type": "text",
                "text": "What is in this video?"
            }
        ]
    }
]
chat_response = client.chat.completions.create(
    model=model,
    messages=messages,
    max_tokens=1024,
    stream=False,
    extra_body={
        "mm_processor_kwargs": {"size": {"shortest_edge": 1568, "longest_edge": 262144}},
        # Alternatively, this can be:
        # "media_io_kwargs": {"video": {"num_frames": some_int}},
        "media_io_kwargs": {"video": {"fps": 1.0}},
    }
)
assistant_message = chat_response.choices[0].message
print(assistant_message)
