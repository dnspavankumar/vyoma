# app.py (Revised for Client-Side STT, CORS, Logging, AND VIDEO FRAMES)
# This is the main Flask application that handles WebSocket connections and manages the ADA AI assistant
import os
from dotenv import load_dotenv  # Load environment variables from .env file
import asyncio  # For handling asynchronous operations
import threading  # For running async loops in separate threads
from flask import Flask, render_template, request # Make sure request is imported
from flask_socketio import SocketIO, emit  # WebSocket functionality for real-time communication

load_dotenv()  # Load environment variables before importing ADA
from ADA_Online import ADA # Make sure filename matches ADA_Online.py

# Initialize Flask app with a secret key for session management
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'a_default_fallback_secret_key!')

@app.route('/')
def health_check():
    # Simple health check endpoint to verify server is running
    return "ADA Backend is running!"

# Configuration for different environments
# For local development - set up localhost URLs
REACT_APP_PORT = os.getenv('REACT_APP_PORT', '5173')  # Default Vite dev server port
REACT_APP_ORIGIN = f"http://localhost:{REACT_APP_PORT}"  # Localhost URL for CORS
REACT_APP_ORIGIN_IP = f"http://127.0.0.1:{REACT_APP_PORT}"  # IP version for CORS

# For Render deployment - get frontend URL from environment
FRONTEND_URL = os.getenv('FRONTEND_URL', '')

# Smart CORS configuration that adapts based on deployment environment
if os.getenv('RENDER', '') == 'true':
    # When deployed on Render, allow the frontend URL or '*' if not specified
    cors_origins = [FRONTEND_URL] if FRONTEND_URL else '*'
    print(f"Running on Render with CORS origins: {cors_origins}")
else:
    # For local development, use localhost origins for security
    cors_origins = [REACT_APP_ORIGIN, REACT_APP_ORIGIN_IP, '*']
    print(f"Running locally with CORS origins: {cors_origins}")

# Initialize SocketIO with threading mode for better performance
socketio = SocketIO(
    app,
    async_mode='threading',  # Use threading instead of eventlet for better compatibility
    cors_allowed_origins=cors_origins  # Allow cross-origin requests
)

# Global variables to manage ADA instance and async operations
ada_instance = None  # Single ADA instance per server
ada_loop = None      # Async event loop for ADA operations
ada_thread = None    # Thread running the async loop

def run_asyncio_loop(loop):
    """ Function to run the asyncio event loop in a separate thread """
    # Set this loop as the current event loop for this thread
    asyncio.set_event_loop(loop)
    try:
        print("Asyncio event loop started...")
        loop.run_forever()  # Keep the loop running indefinitely
    finally:
        # Cleanup when the loop is stopped
        print("Asyncio event loop stopping...")
        tasks = asyncio.all_tasks(loop=loop)  # Get all running tasks
        for task in tasks:
            if not task.done():
                task.cancel()  # Cancel any unfinished tasks
        try:
            # Wait for all tasks to complete or be cancelled
            loop.run_until_complete(asyncio.gather(*[t for t in tasks if not t.done()], return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())  # Clean up async generators
        except RuntimeError as e:
             print(f"RuntimeError during loop cleanup (might be expected if loop stopped abruptly): {e}")
        except Exception as e:
            print(f"Exception during loop cleanup: {e}")
        finally:
            if not loop.is_closed():
                loop.close()  # Close the loop to free resources
        print("Asyncio event loop stopped.")

@socketio.on('connect')
def handle_connect():
    """ Handles new client connections - this is called when a client connects via WebSocket """
    global ada_instance, ada_loop, ada_thread
    client_sid = request.sid  # Get unique session ID for this client
    print(f"\n--- handle_connect called for SID: {client_sid} ---")

    # Check if we need to start the async thread
    if ada_thread is None or not ada_thread.is_alive():
        print(f"    Asyncio thread not running. Starting new loop and thread.")
        ada_loop = asyncio.new_event_loop()  # Create new event loop
        ada_thread = threading.Thread(target=run_asyncio_loop, args=(ada_loop,), daemon=True)  # Create daemon thread
        ada_thread.start()  # Start the thread
        print("    Started asyncio thread.")
        socketio.sleep(0.1)  # Small delay to ensure thread is ready

    # Create ADA instance if it doesn't exist
    if ada_instance is None:
        print(f"    Creating NEW ADA instance for SID: {client_sid}")
        if not ada_loop or not ada_loop.is_running():
             print(f"    ERROR: Cannot create ADA instance, asyncio loop not ready for SID {client_sid}.")
             emit('error', {'message': 'Assistant initialization error (loop).'}, room=client_sid)
             return

        try:
            # Initialize ADA with SocketIO instance and client ID
            ada_instance = ADA(socketio_instance=socketio, client_sid=client_sid)
            # Schedule ADA's startup tasks in the async loop
            future = asyncio.run_coroutine_threadsafe(ada_instance.start_all_tasks(), ada_loop)
            print("    ADA instance created and tasks scheduled.")
        except ValueError as e:
            print(f"    ERROR initializing ADA (ValueError) for SID {client_sid}: {e}")
            emit('error', {'message': f'Failed to initialize assistant: {e}'}, room=client_sid)
            ada_instance = None
            return
        except Exception as e:
            print(f"    ERROR initializing ADA (Unexpected) for SID {client_sid}: {e}")
            emit('error', {'message': f'Unexpected error initializing assistant: {e}'}, room=client_sid)
            ada_instance = None
            return
    else:
        # Update existing ADA instance with new client ID
        print(f"    ADA instance already exists. Updating SID from {ada_instance.client_sid} to {client_sid}")
        ada_instance.client_sid = client_sid

    # Send success message to client
    if ada_instance:
        emit('status', {'message': 'Connected to ADA Assistant'}, room=client_sid)
    print(f"--- handle_connect finished for SID: {client_sid} ---\n")


@socketio.on('disconnect')
def handle_disconnect():
    """ Handles client disconnections - cleanup when client leaves """
    global ada_instance
    client_sid = request.sid
    print(f"\n--- handle_disconnect called for SID: {client_sid} ---")

    # Only cleanup if this is the designated client
    if ada_instance and ada_instance.client_sid == client_sid:
        print(f"    Designated client {client_sid} disconnected. Attempting to stop ADA.")
        if ada_loop and ada_loop.is_running():
            # Stop all ADA tasks gracefully
            future = asyncio.run_coroutine_threadsafe(ada_instance.stop_all_tasks(), ada_loop)
            try:
                future.result(timeout=10)  # Wait up to 10 seconds for cleanup
                print("    ADA tasks stopped successfully.")
            except TimeoutError:
                print("    Timeout waiting for Alfred tasks to stop.")
            except Exception as e:
                print(f"    Exception during ADA task stop: {e}")
            finally:
                 pass # Keep loop running

        else:
             print(f"    Cannot stop ADA tasks: asyncio loop not available or not running.")

        ada_instance = None  # Clear the instance
        print("    ADA instance cleared.")

    elif ada_instance:
         print(f"    Disconnecting client {client_sid} is NOT the designated client ({ada_instance.client_sid}). ADA remains active.")
    else:
         print(f"    Client {client_sid} disconnected, but no active ADA instance found.")

    print(f"--- handle_disconnect finished for SID: {client_sid} ---\n")


@socketio.on('send_text_message')
def handle_text_message(data):
    """ Receives text message from client's input box - handles manual text input """
    client_sid = request.sid
    message = data.get('message', '')  # Extract message from request data
    print(f"Received text from {client_sid}: {message}")
    if ada_instance and ada_instance.client_sid == client_sid:
        if ada_loop and ada_loop.is_running():
            # Process text with end_of_turn=True implicitly handled in process_input -> run_gemini_session
            asyncio.run_coroutine_threadsafe(ada_instance.process_input(message, is_final_turn_input=True), ada_loop)
            print(f"    Text message forwarded to ADA for SID: {client_sid}")
        else:
            print(f"    Cannot process text message for SID {client_sid}: asyncio loop not ready.")
            emit('error', {'message': 'Assistant busy or loop error.'}, room=client_sid)
    else:
        print(f"    ADA instance not ready or SID mismatch for text message from {client_sid}.")
        emit('error', {'message': 'Assistant not ready or session mismatch.'}, room=client_sid)


@socketio.on('send_transcribed_text')
def handle_transcribed_text(data):
    """ Receives final transcribed text from client's Web Speech API - handles voice input """
    client_sid = request.sid
    transcript = data.get('transcript', '')  # Extract transcribed text
    print(f"Received transcript from {client_sid}: {transcript}")
    if transcript and ada_instance and ada_instance.client_sid == client_sid:
         if ada_loop and ada_loop.is_running():
            # Process transcript with end_of_turn=True implicitly handled in process_input -> run_gemini_session
            asyncio.run_coroutine_threadsafe(ada_instance.process_input(transcript, is_final_turn_input=True), ada_loop)
            print(f"    Transcript forwarded to ADA for SID: {client_sid}")
         else:
             print(f"    Cannot process transcript for SID {client_sid}: asyncio loop not ready.")
             emit('error', {'message': 'Assistant busy or loop error.'}, room=client_sid)
    elif not transcript:
         print("    Received empty transcript.")
    else:
         print(f"    ADA instance not ready or SID mismatch for transcript from {client_sid}.")


# **** ADD VIDEO FRAME HANDLER ****
@socketio.on('send_video_frame')
def handle_video_frame(data):
    """ Receives base64 video frame data from client - handles real-time video processing """
    client_sid = request.sid
    frame_data_url = data.get('frame') # Expecting data URL like 'data:image/jpeg;base64,xxxxx'

    if frame_data_url and ada_instance and ada_instance.client_sid == client_sid:
        if ada_loop and ada_loop.is_running():
            print(f"Received video frame from {client_sid}, forwarding...") # Optional: very verbose
            asyncio.run_coroutine_threadsafe(ada_instance.process_video_frame(frame_data_url), ada_loop)
        pass

@socketio.on('video_feed_stopped')
def handle_video_feed_stopped():
    """ Client signaled that the video feed has stopped - cleanup video processing """
    client_sid = request.sid
    print(f"Received video_feed_stopped signal from {client_sid}.")
    if ada_instance and ada_instance.client_sid == client_sid:
        if ada_loop and ada_loop.is_running():
            # Call a method on ADA instance to clear its video queue
            asyncio.run_coroutine_threadsafe(ada_instance.clear_video_queue(), ada_loop)
            print(f"    Video frame queue clearing requested for SID: {client_sid}")
        else:
            print(f"    Cannot clear video queue for SID {client_sid}: asyncio loop not ready.")
    else:
        print(f"    ADA instance not ready or SID mismatch for video_feed_stopped from {client_sid}.")


if __name__ == '__main__':
    # Main entry point - start the Flask-SocketIO server
    port = int(os.getenv('PORT', 10000))  # Get port from environment or use default
    is_production = os.getenv('RENDER', 'false').lower() == 'true'  # Check if running on Render
    print(f"Starting Flask-SocketIO server on port {port} in {'production' if is_production else 'development'} mode...")
    try:
        # Start the server with appropriate settings
        socketio.run(app, debug=not is_production, host='0.0.0.0', port=port, use_reloader=False, allow_unsafe_werkzeug=True)
    finally:
        # Cleanup when server shuts down
        print("\nServer shutting down...")
        if ada_instance:
             print("Attempting to stop active ADA instance on server shutdown...")
             if ada_loop and ada_loop.is_running():
                 future = asyncio.run_coroutine_threadsafe(ada_instance.stop_all_tasks(), ada_loop)
                 try:
                     future.result(timeout=5)  # Wait up to 5 seconds for cleanup
                     print("ADA tasks stopped.")
                 except TimeoutError:
                     print("Timeout stopping ADA tasks during shutdown.")
                 except Exception as e:
                     print(f"Exception stopping ADA tasks during shutdown: {e}")
             else:
                 print("Cannot stop ADA instance: asyncio loop not available.")
             ada_instance = None

        if ada_loop and ada_loop.is_running():
             print("Stopping asyncio loop from main thread...")
             ada_loop.call_soon_threadsafe(ada_loop.stop)  # Stop the loop safely
             if ada_thread and ada_thread.is_alive():
                 ada_thread.join(timeout=5)  # Wait for thread to finish
                 if ada_thread.is_alive():
                     print("Warning: Asyncio thread did not exit cleanly.")
             print("Asyncio loop/thread stop initiated.")
        print("Shutdown complete.")