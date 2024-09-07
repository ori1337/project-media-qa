'''
All utility functions used in the app
'''
import os
from typing import Iterable
import time
from io import BytesIO
import requests
from groq.types.chat import ChatCompletionMessageParam
from llama_index.core import Document, VectorStoreIndex
import yt_dlp
from config import GROQ_CLIENT, EMBED_MODEL, VECTOR_INDEX, PIPELINE

def combine_text_with_markers_and_speaker(data):
    combined_text = ""
    for item in data:
        speaker_text = " ".join(sentence["text"] for sentence in item["sentences"])
        speaker_info = f"Speaker {item['speaker']}:"
        combined_text += f"{speaker_info} {speaker_text}\n"
    return combined_text

def read_from_url(url: str) -> BytesIO:
    res = requests.get(url)
    audio_bytes = BytesIO(res.content)
    return audio_bytes

def read_from_youtube(url: str) -> tuple[BytesIO, str]:
    """
    Reads audio from a YouTube video using the Cobalt API.

    Args:
    url (str): The URL of the YouTube video.

    Returns:
    tuple[BytesIO, str]: A tuple containing the audio data as a BytesIO object and the MIME type of the audio.
    """

    # Set up the API endpoint and headers
    api_endpoint = "https://api.cobalt.tools/api/json"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    # Set up the data to be sent in the POST request
    data = {
        "url": url,
        "vCodec": "h264",
        "vQuality": "720",
        "aFormat": "mp3",
        "filenamePattern": "classic",
        "isAudioOnly": True
    }

    # Make the POST request to the API
    response = requests.post(api_endpoint, headers=headers, data=json.dumps(data))

    # Check if the response was successful
    if response.status_code != 200:
        raise Exception(f"Failed to retrieve audio from YouTube: {response.text}")

    # Parse the response JSON
    response_json = response.json()

    # Get the URL of the audio stream
    stream_url = response_json["url"]

    # Make a GET request to the audio stream URL
    audio_response = requests.get(stream_url, stream=True)

    # Check if the response was successful
    if audio_response.status_code != 200:
        raise Exception(f"Failed to retrieve audio stream: {audio_response.text}")

    # Create a BytesIO object to store the audio data
    audio_buffer = BytesIO()

    # Iterate over the chunks of the audio response and write them to the BytesIO object
    for chunk in audio_response.iter_content(1024):
        audio_buffer.write(chunk)

    # Seek the BytesIO object back to the beginning
    audio_buffer.seek(0)

    # Return the audio data and MIME type
    return audio_buffer, "audio/mpeg"

# def read_from_youtube(url: str):
#     yt = YouTube(url)
#     video = yt.streams.filter(only_audio=True, mime_type="audio/webm").first()
    
#     if video is None:
#         raise ValueError("No audio/webm stream found for the given YouTube URL.")
    
#     buffer = BytesIO()
#     video.stream_to_buffer(buffer)
#     buffer.seek(0)
    
#     audio_data = buffer.read()
    
#     print(f"Audio retrieved as audio/webm (mimetype: {video.mime_type})")
    
#     return BytesIO(audio_data)

def prerecorded(source, model: str = "whisper-large-v3", options: dict[str, str] = None) -> None:
    print(f"Source: {source} ")
    start = time.time()
    audio_bytes: BytesIO = source['buffer']
    file_type = source.get("mimetype", "audio/wav")
    if not file_type:
        file_type = "audio/wav"
    file_type = file_type.split("/")[1]
    print(f"Final filetype: {file_type}")
    transcription = GROQ_CLIENT.audio.transcriptions.create(
        file=(f"audio.{file_type}", audio_bytes.read()),
        model=model,
    )
    end = time.time()
    audio_bytes.seek(0)
    return {
        'text':transcription.text,
        'time_taken': end - start
    }

def create_vectorstore(transcript: str):
    global VECTOR_INDEX
    nodes = PIPELINE.run(documents=[Document(text=transcript)])
    globals()['VECTOR_INDEX'] = VectorStoreIndex(embed_model=EMBED_MODEL, nodes=nodes)
    return VECTOR_INDEX

def chat_stream(model: str, messages: Iterable[ChatCompletionMessageParam], **kwargs):
    # Retrieve documents from the vectorstore
    stream_response = GROQ_CLIENT.chat.completions.create(
        messages=messages,
        model=model,
        stream=True,
        **kwargs
    )

    for chunk in stream_response:
        tokens = chunk.choices[0].delta.content
        if tokens:
            yield tokens
        if x_groq := chunk.x_groq:
            if not x_groq.usage:
                continue
            usage = x_groq.usage
            completion_time = usage.completion_time
            completion_tokens = usage.completion_tokens
            tps = completion_tokens/completion_time
            yield f"\n\n_Tokens/sec: {round(tps, 2)}_"
