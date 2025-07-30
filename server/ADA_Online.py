# server/ADA_Online.py (Refactored)
# Main ADA AI assistant class that handles conversation, TTS, and various API integrations
import asyncio  # For asynchronous programming and event loops
import base64  # For encoding/decoding binary data (audio, images)
import torch  # PyTorch for machine learning operations
import python_weather  # Weather API client for location-based weather data
import googlemaps  # Google Maps API client for location services
from datetime import datetime  # Date and time utilities
import os  # Operating system interface
from dotenv import load_dotenv  # Environment variable loading
import websockets  # WebSocket client for real-time communication
import json  # JSON data handling
from googlesearch import search as Google_Search_sync  # Synchronous Google search
import aiohttp # For async HTTP requests to various APIs
from bs4 import BeautifulSoup # For HTML parsing of web search results

# --- Handle google.genai import gracefully ---
try:
    from google.genai import types  # Google's Generative AI types and schemas
    from google import genai  # Main Google Generative AI client
    print("Successfully imported google.genai")
except ImportError as e:
    print(f"Error importing google.genai: {e}")
    # Fallback imports or mock objects if needed - ensures app works without Google AI
    class MockTypes:
        """Mock types class to simulate Google AI types when import fails"""
        def __init__(self):
            pass

        class GenerateContentConfig:
            """Mock configuration class for content generation"""
            def __init__(self, system_instruction=None, tools=None):
                self.system_instruction = system_instruction  # System prompt
                self.tools = tools  # Available tools/functions

        class Tool:
            """Mock tool class for function calling"""
            def __init__(self, function_declarations=None):
                self.function_declarations = function_declarations  # Function definitions

        class FunctionDeclaration:
            """Mock function declaration class"""
            def __init__(self, name=None, description=None, parameters=None):
                self.name = name  # Function name
                self.description = description  # Function description
                self.parameters = parameters  # Function parameters schema

        class Schema:
            """Mock schema class for parameter definitions"""
            def __init__(self, type=None, properties=None, required=None, description=None):
                self.type = type  # Data type
                self.properties = properties  # Object properties
                self.required = required  # Required fields
                self.description = description  # Schema description

        class Type:
            """Mock type constants"""
            OBJECT = "object"  # Object type constant
            STRING = "string"  # String type constant

        class Part:
            @staticmethod
            def from_bytes(data, mime_type): return None
            @staticmethod
            def from_function_response(name, response): return None

    class MockGenAI:
        """Mock Google AI client for fallback functionality"""
        def __init__(self):
            pass

        class Client:
            """Mock client class"""
            def __init__(self, api_key=None):
                self.api_key = api_key  # API key storage
                self.aio = MockAIO()  # Async operations interface

        class MockAIO:
            """Mock async interface"""
            def __init__(self):
                self.chats = MockChats()  # Chat operations

        class MockChats:
            """Mock chat operations"""
            def create(self, model=None, config=None):
                return MockChat()  # Return mock chat instance

        class MockChat:
            """Mock chat session"""
            async def send_message(self, content=None):
                return "This is a mock response as google.genai could not be imported"
            async def send_message_stream(self, content):
                return []

    # Use mock objects when Google AI is unavailable
    types = MockTypes()
    genai = MockGenAI()
    print("Using mock objects for google.genai")

load_dotenv()  # Load environment variables from .env file

# API Keys - Load from environment variables for security
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")  # Text-to-speech API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Google AI and search APIs
MAPS_API_KEY = os.getenv("MAPS_API_KEY")  # Google Maps API

# Audio configuration constants for ElevenLabs TTS
VOICE_ID = 'Yko7PKHZNXotIFUBG7I9'  # Specific voice ID for ADA's voice
CHANNELS = 1  # Mono audio output
RECEIVE_SAMPLE_RATE = 24000  # Audio sample rate in Hz
CHUNK_SIZE = 1024  # Audio chunk size for streaming
MAX_QUEUE_SIZE = 1  # Maximum audio queue size to prevent memory issues
MODEL_ID = "eleven_flash_v2_5"

def check_api_keys():
    if not ELEVENLABS_API_KEY: print("Error: ELEVENLABS_API_KEY not found.")
    if not GOOGLE_API_KEY: print("Error: GOOGLE_API_KEY not found.")
    if not MAPS_API_KEY: print("Error: MAPS_API_KEY not found.")

check_api_keys()

class ADA:
    def __init__(self, socketio_instance=None, client_sid=None):
        print("initializing ADA for web...")
        self.socketio = socketio_instance
        self.client_sid = client_sid
        self.Maps_api_key = MAPS_API_KEY
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self._init_function_declarations()
        self.available_functions = {
            "get_weather": self.get_weather,
            "get_travel_duration": self.get_travel_duration,
            "get_search_results": self.get_search_results
        }
        self.system_behavior = """
        You are Alfred (Advanced Learning Facilitator for Education and Research Development), a witty British tutor with a slightly flirty personality who always addresses your creator as "Sir." [...]
        """
        self.config = types.GenerateContentConfig(
            system_instruction=self.system_behavior,
            tools=[types.Tool(function_declarations=[
                self.get_weather_func,
                self.get_travel_duration_func,
                self.get_search_results_func
            ])]
        )
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.model = "gemini-2.0-flash"
        self.chat = self.client.aio.chats.create(model=self.model, config=self.config)

        # Queues and tasks
        self.latest_video_frame_data_url = None
        self.input_queue = asyncio.Queue()
        self.response_queue = asyncio.Queue()
        self.audio_output_queue = asyncio.Queue()
        self.gemini_session = None
        self.tts_websocket = None
        self.tasks = []

    def _init_function_declarations(self):
        self.get_weather_func = types.FunctionDeclaration(
            name="get_weather",
            description="Get the current weather conditions for a specified location.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={"location": types.Schema(type=types.Type.STRING, description="The city and state/country.")},
                required=["location"]
            )
        )
        self.get_travel_duration_func = types.FunctionDeclaration(
            name="get_travel_duration",
            description="Calculates travel duration between origin and destination using Google Maps.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "origin": types.Schema(type=types.Type.STRING, description="The starting address or place name."),
                    "destination": types.Schema(type=types.Type.STRING, description="The destination address or place name."),
                    "mode": types.Schema(type=types.Type.STRING, description="Transport mode ('driving', 'walking', etc.). Defaults to 'driving'.")
                },
                required=["origin", "destination"]
            )
        )
        self.get_search_results_func = types.FunctionDeclaration(
            name="get_search_results",
            description="Performs a Google search for the given query and returns top result URLs.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={"query": types.Schema(type=types.Type.STRING, description="The search term.")},
                required=["query"]
            )
        )

    async def get_weather(self, location: str) -> dict | None:
        async with python_weather.Client(unit=python_weather.IMPERIAL) as client:
            try:
                weather = await client.get(location)
                weather_data = {
                    'location': location,
                    'current_temp_f': weather.temperature,
                    'precipitation': weather.precipitation,
                    'description': weather.description,
                }
                print(f"Weather data fetched: {weather_data}")
                self._emit('weather_update', weather_data)
                return weather_data
            except Exception as e:
                print(f"Error fetching weather for {location}: {e}")
                return {"error": f"Could not fetch weather for {location}."}

    def _sync_get_travel_duration(self, origin: str, destination: str, mode: str = "driving") -> str:
        if not self.Maps_api_key or self.Maps_api_key == "YOUR_PROVIDED_KEY":
            print("Error: Google Maps API Key is missing or invalid.")
            return "Error: Missing or invalid Google Maps API Key configuration."
        try:
            gmaps = googlemaps.Client(key=self.Maps_api_key)
            now = datetime.now()
            print(f"Requesting directions: From='{origin}', To='{destination}', Mode='{mode}'")
            directions_result = gmaps.directions(origin, destination, mode=mode, departure_time=now)
            if directions_result:
                leg = directions_result[0]['legs'][0]
                duration_text = leg.get('duration_in_traffic', leg.get('duration', {})).get('text', 'Not available')
                result = f"Estimated travel duration ({mode}): {duration_text}"
                print(f"Directions Result: {result}")
                return result
            else:
                return f"Could not find a route from {origin} to {destination} via {mode}."
        except Exception as e:
            print(f"An unexpected error occurred during travel duration lookup: {e}")
            return f"An unexpected error occurred: {e}"

    async def get_travel_duration(self, origin: str, destination: str, mode: str = "driving") -> dict:
        print(f"Received request for travel duration: {origin} to {destination}, Mode: {mode}")
        try:
            result_string = await asyncio.to_thread(
                self._sync_get_travel_duration, origin, destination, mode
            )
            if not result_string.startswith("Error"):
                self._emit('map_update', {'destination': destination, 'origin': origin})
            return {"duration_result": result_string}
        except Exception as e:
            print(f"Error calling _sync_get_travel_duration: {e}")
            return {"duration_result": f"Failed to execute travel duration request: {e}"}

    async def _fetch_and_extract_snippet(self, session, url: str) -> dict | None:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        try:
            async with session.get(url, headers=headers, timeout=15, ssl=False) as response:
                if response.status == 200:
                    html_content = await response.text()
                    soup = BeautifulSoup(html_content, 'lxml')
                    title = soup.find('title').string.strip() if soup.find('title') else "No Title Found"
                    description_tag = soup.find('meta', attrs={'name': 'description'})
                    snippet = description_tag['content'].strip() if description_tag and description_tag.get('content') else "No Description Found"
                    paragraphs = soup.find_all('p')
                    full_page_text = ' '.join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
                    max_len = 1500
                    page_text_summary = (full_page_text[:max_len] + "...") if len(full_page_text) > max_len else full_page_text or "No paragraph text found on page."
                    print(f"Extracted: Title='{title}', Snippet='{snippet[:50]}...', Text='{page_text_summary[:50]}...' from {url}")
                    return {
                        "url": url,
                        "title": title,
                        "meta_snippet": snippet,
                        "page_content_summary": page_text_summary
                    }
                else:
                    print(f"Failed to fetch {url}: Status {response.status}")
                    return None
        except Exception as e:
            print(f"Error processing {url}: {e}")
            return None

    def _sync_Google_Search(self, query: str, num_results: int = 5) -> list:
        print(f"Performing Google search for: '{query}'")
        try:
            results = list(Google_Search_sync(term=query, num_results=num_results, lang="en", timeout=1))
            print(f"Found {len(results)} results.")
            return results
        except Exception as e:
            print(f"Error during Google search for '{query}': {e}")
            return []

    async def get_search_results(self, query: str) -> dict:
        print(f"Received request for Google search: '{query}'")
        fetched_results = []
        try:
            search_urls = await asyncio.to_thread(self._sync_Google_Search, query, 5)
            if not search_urls:
                self._emit('search_results_update', {"results": [], "query": query})
                return {"results": []}
            async with aiohttp.ClientSession() as session:
                tasks = [self._fetch_and_extract_snippet(session, url) for url in search_urls]
                results_from_gather = await asyncio.gather(*tasks, return_exceptions=True)
            fetched_results = [r for r in results_from_gather if isinstance(r, dict)]
            self._emit('search_results_update', {"query": query, "results": fetched_results})
        except Exception as e:
            print(f"Error running get_search_results for '{query}': {e}")
            self._emit('search_results_error', {"query": query, "error": str(e)})
            return {"error": f"Failed to execute Google search: {str(e)}"}
        return {"results": fetched_results}

    async def clear_queues(self, text=""):
        for q in [self.response_queue, self.audio_output_queue]:
            while not q.empty():
                try: q.get_nowait()
                except asyncio.QueueEmpty: break

    async def process_input(self, message, is_final_turn_input=False):
        print(f"Processing input: '{message}', Final Turn: {is_final_turn_input}")
        if is_final_turn_input: await self.clear_queues()
        await self.input_queue.put((message, is_final_turn_input))

    async def process_video_frame(self, frame_data_url):
        self.latest_video_frame_data_url = frame_data_url

    async def run_gemini_session(self):
        print("Starting Gemini session manager...")
        try:
            while True:
                message, is_final_turn_input = await self.input_queue.get()
                if not (message.strip() and is_final_turn_input):
                    self.input_queue.task_done()
                    continue
                print(f"Sending FINAL input to Gemini: {message}")
                request_content = [message]
                if self.latest_video_frame_data_url:
                    try:
                        header, encoded = self.latest_video_frame_data_url.split(",", 1)
                        mime_type = header.split(':')[1].split(';')[0] if ':' in header and ';' in header else "image/jpeg"
                        frame_bytes = base64.b64decode(encoded)
                        request_content.append(types.Part.from_bytes(data=frame_bytes, mime_type=mime_type))
                        print(f"Included image frame with mime_type: {mime_type}")
                    except Exception as e:
                        print(f"Error processing video frame data URL: {e}")
                    finally:
                        self.latest_video_frame_data_url = None

                response_stream = await self.chat.send_message_stream(request_content)
                collected_function_calls = []
                processed_text_in_turn = False

                async for chunk in response_stream:
                    if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                        continue
                    for part in chunk.candidates[0].content.parts:
                        if getattr(part, 'function_call', None):
                            print(f"Detected Function Call: {part.function_call.name}")
                            collected_function_calls.append(part.function_call)
                        elif getattr(part, 'text', None):
                            await self.response_queue.put(part.text)
                            self._emit('receive_text_chunk', {'text': part.text})
                            processed_text_in_turn = True

                if collected_function_calls:
                    print(f"Processing {len(collected_function_calls)} detected function call(s)")
                    function_response_parts = []
                    for function_call in collected_function_calls:
                        tool_call_name = function_call.name
                        tool_call_args = dict(function_call.args)
                        if tool_call_name in self.available_functions:
                            try:
                                function_result = await self.available_functions[tool_call_name](**tool_call_args)
                                function_response_parts.append(
                                    types.Part.from_function_response(name=tool_call_name, response=function_result)
                                )
                            except Exception as e:
                                function_response_parts.append(
                                    types.Part.from_function_response(name=tool_call_name, response={"error": str(e)})
                                )
                        else:
                            function_response_parts.append(
                                types.Part.from_function_response(name=tool_call_name, response={"error": f"Function {tool_call_name} not found."})
                            )
                    if function_response_parts:
                        response_stream_after_func = await self.chat.send_message_stream(function_response_parts)
                        async for final_chunk in response_stream_after_func:
                            if final_chunk.candidates and final_chunk.candidates[0].content and final_chunk.candidates[0].content.parts:
                                for part in final_chunk.candidates[0].content.parts:
                                    if getattr(part, 'text', None):
                                        await self.response_queue.put(part.text)
                                        self._emit('receive_text_chunk', {'text': part.text})
                        self.response_queue.put("")
                await self.response_queue.put(None)
                self.input_queue.task_done()

        except asyncio.CancelledError:
            print("Gemini session task cancelled.")
        except Exception as e:
            print(f"Error in Gemini session manager: {e}")
            import traceback
            traceback.print_exc()
            self._emit('error', {'message': f'Gemini session error: {str(e)}'})
            try: await self.response_queue.put(None)
            except Exception: pass
        finally:
            print("Gemini session manager finished.")
            self.gemini_session = None

    async def run_tts_and_audio_out(self):
        print("Starting TTS and Audio Output manager...")
        uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream-input?model_id={MODEL_ID}&output_format=pcm_24000"
        while True:
            try:
                async with websockets.connect(uri) as websocket:
                    self.tts_websocket = websocket
                    print("ElevenLabs WebSocket Connected.")
                    await websocket.send(json.dumps({"text": " ", "voice_settings": {"stability": 0.3, "similarity_boost": 0.9, "speed": 1.1}, "xi_api_key": ELEVENLABS_API_KEY,}))
                    listener_task = asyncio.create_task(self._tts_listener(websocket))
                    try:
                        while True:
                            text_chunk = await self.response_queue.get()
                            if text_chunk is None:
                                print("End of text stream signal received for TTS.")
                                await websocket.send(json.dumps({"text": ""}))
                                break
                            await websocket.send(json.dumps({"text": text_chunk}))
                            print(f"Sent text to TTS: {text_chunk}")
                    except asyncio.CancelledError: print("TTS sender task cancelled.")
                    except Exception as e: print(f"Error sending text to TTS: {e}")
                    finally:
                        if listener_task and not listener_task.done():
                            try:
                                if not listener_task.cancelled(): await asyncio.wait_for(listener_task, timeout=5.0)
                            except asyncio.TimeoutError: print("Timeout waiting for TTS listener.")
                            except asyncio.CancelledError: print("TTS listener task already cancelled.")
            except websockets.exceptions.ConnectionClosedError as e:
                print(f"ElevenLabs WebSocket connection error: {e}. Reconnecting...")
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                print("TTS main task cancelled.")
                break
            except Exception as e:
                print(f"Error in TTS main loop: {e}")
                await asyncio.sleep(5)
            finally:
                if self.tts_websocket:
                    try: await self.tts_websocket.close()
                    except Exception: pass
                self.tts_websocket = None

    async def _tts_listener(self, websocket):
        try:
            while True:
                message = await websocket.recv()
                data = json.loads(message)
                if data.get("audio"):
                    audio_chunk = base64.b64decode(data["audio"])
                    self._emit('receive_audio_chunk', {'audio': base64.b64encode(audio_chunk).decode('utf-8')})
                elif data.get('isFinal'):
                    pass
        except websockets.exceptions.ConnectionClosedOK:
            print("TTS WebSocket listener closed normally.")
        except websockets.exceptions.ConnectionClosedError as e:
            print(f"TTS WebSocket listener closed error: {e}")
        except asyncio.CancelledError:
            print("TTS listener task cancelled.")
        except Exception as e:
            print(f"Error in TTS listener: {e}")
        finally:
            self.tts_websocket = None

    def _emit(self, event, data):
        if self.socketio and self.client_sid:
            self.socketio.emit(event, data, room=self.client_sid)

    async def start_all_tasks(self):
        print("Starting ADA background tasks...")
        if not self.tasks:
            loop = asyncio.get_running_loop()
            gemini_task = loop.create_task(self.run_gemini_session())
            tts_task = loop.create_task(self.run_tts_and_audio_out())
            self.tasks = [gemini_task, tts_task]
            if hasattr(self, 'video_frame_queue'):
                video_sender_task = loop.create_task(self.run_video_sender())
                self.tasks.append(video_sender_task)
            print(f"ADA Core Tasks started: {len(self.tasks)}")
        else:
            print("ADA tasks already running.")

    async def stop_all_tasks(self):
        print("Stopping ADA background tasks...")
        tasks_to_cancel = list(self.tasks)
        for task in tasks_to_cancel:
            if task and not task.done():
                task.cancel()
        await asyncio.gather(*[t for t in tasks_to_cancel if t], return_exceptions=True)
        self.tasks = []
        if self.tts_websocket:
            try: await self.tts_websocket.close(code=1000)
            except Exception as e: print(f"Error closing TTS websocket during stop: {e}")
            finally: self.tts_websocket = None
        self.gemini_session = None
        print("ADA tasks stopped.")
