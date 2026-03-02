from openai import OpenAI

#content_type = "image_url"
#content_url = "https://assets.ngc.nvidia.com/products/api-catalog/phi-3-5-vision/example1b.jpg"

content_type = "video_url"
content_url = "https://download.samplelib.com/mp4/sample-5s.mp4"

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-used")
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
        "content": [
            {
                "type": content_type,
                #"image_url": {
                #    "url": content_url
                #}
                "video_url": {
                    "url": content_url
                }
            },
            {
                "type": "text",
                "text": "What is in this image?"
            }
        ]
    }
]
chat_response = client.chat.completions.create(
    model="nvidia/cosmos-reason2-2b",
    messages=messages,
    max_tokens=256,
    stream=False,
)
assistant_message = chat_response.choices[0].message
print(assistant_message)

# Code preceding `client.chat.completions.create` is the same.
stream = client.chat.completions.create(
    model="nvidia/cosmos-reason2-2b",
    messages=messages,
    max_tokens=256,
    # Take note of this param.
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if delta and delta.content:
        text = delta.content
        # Print immediately and without a newline to update the output as the response is
        # streamed in.
        print(text, end="", flush=True)
# Final newline.
print()


#{
#    "type": "image_url",
#    "image_url": {
#        "url": "https://assets.ngc.nvidia.com/products/api-catalog/phi-3-5-vision/example1b.jpg"
#    }
#}




