import os
import io
import base64
import re  # Regular expression
from flask import Flask, render_template, request, jsonify, session
import google.generativeai as genai
from PIL import Image  # Pillow library for image handling

# --- Configuration ---
# WARNING: Hardcoding API keys is insecure. Use environment variables in production.
# Replace with os.getenv("GEMINI_API_KEY") after setting the environment variable.
API_KEY = "AIzaSyAcxQw_sYTSW9GkmOiwMGpGakBPp6Fb_LA"

# Check if the API key is provided
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

genai.configure(api_key=API_KEY)

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # Max upload 10MB
app.secret_key = 'your_secret_key_here'  # Enable session support

# --- Gemini Model Initialization ---
# Use gemini-pro-vision for both image and text-based responses
model = genai.GenerativeModel('gemini-2.0-flash')


# --- Helper Functions ---
def get_image_parts(uploaded_file):
    """Converts uploaded image file to format Gemini API understands."""
    if uploaded_file and allowed_file(uploaded_file.filename):
        try:
            # Read image bytes
            img_bytes = uploaded_file.read()
            # Verify if it's an image and get format
            image = Image.open(io.BytesIO(img_bytes))
            # Determine MIME type dynamically
            mime_type = Image.MIME.get(image.format)
            if not mime_type:
                # Fallback for formats PIL might not map directly to MIME
                if image.format == 'JPEG':
                    mime_type = 'image/jpeg'
                elif image.format == 'PNG':
                    mime_type = 'image/png'
                # Add more formats if needed
                else:
                    # Default or raise error if unsupported format
                    mime_type = 'application/octet-stream'  # Or handle error

            if not mime_type.startswith('image/'):
                # Basic check if it's an image MIME type
                # Could add PDF check here if needed, but Gemini handles common image types
                raise ValueError("Uploaded file is not a supported image type.")

            return [{"mime_type": mime_type, "data": img_bytes}]
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
    return None


def allowed_file(filename):
    """Checks if the file extension is allowed."""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}  # Add 'pdf' if Gemini supports it directly in vision model
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# --- Flask Routes ---
@app.route('/')
def index():
    """Renders the main page."""
    return render_template('index.html')


@app.route('/ai_chat', methods=['POST'])
def ai_chat():
    """Handles chat messages with AI, potentially including image analysis."""
    data = request.get_json()
    problem_description = data.get('problem_description', '')
    message = data.get('message', '')
    file_info = data.get('file_info', [])  # Get file info, may be empty list
    files_content = []

    # Load content of already uploaded files from file_info
    for filename in file_info:
        try:
            # Retrieve content of each file
            with open(filename, 'rb') as file:
                file_data = file.read()
                image = Image.open(io.BytesIO(file_data))
                mime_type = Image.MIME.get(image.format)
                
                if mime_type and mime_type.startswith('image/'):
                    files_content.append({"mime_type": mime_type, "data": file_data})
                else:
                    print(f"Warning: Unsupported file type for {filename}")
        except FileNotFoundError:
            print(f"Error: File {filename} not found")
            return jsonify({'error': 'File not found.'}), 400
        except Exception as e:
            print(f"Error processing file: {e}")
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500

    prompt_parts = []

    if problem_description:
        prompt_parts.append(
            "You are a medical information assistant that provides general health information. "
            "Present information in a professional, doctor-like manner, focusing on educational content. "
            "Explain general concepts about conditions, common symptoms, and when someone should consider seeking medical care. "
            "Always clarify that you're providing general information, not personalized medical advice. "
            "Use a supportive tone and provide general wellness information that's backed by medical consensus. "
            "Format your response in markdown with relevant emojis for readability. "
            f"\nTopic for discussion: {problem_description}"
        )

    if files_content:
        prompt_parts.extend(files_content)
        prompt_parts.append("\n(Files provided by user.)")

    if message:
        prompt_parts.append(f"\nUser's follow up message: {message}")

    try:
        response = model.generate_content(prompt_parts, stream=False)
        ai_reply = response.text if hasattr(response, 'text') else 'Sorry, I could not generate a response.'
        return jsonify({'reply': ai_reply})

    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return jsonify({'error': 'An error occurred during the chat.'}), 500


@app.route('/upload_file', methods=['POST'])
def upload_file():
    """Handles file uploads and saves them to the server."""
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = os.path.join('uploads', file.filename)  # Save to 'uploads' directory
        os.makedirs('uploads', exist_ok=True)  # Ensure 'uploads' directory exists
        file.save(filename)  # Save the file to the server
        return jsonify({'status': 'success', 'filename': filename}), 200

    return jsonify({'status': 'error', 'message': 'Invalid file type'}), 400


@app.route('/book_appointment', methods=['POST'])
def book_appointment():
    """Handles appointment booking requests."""
    name = request.form.get('name')
    email = request.form.get('email')
    doctor = request.form.get('doctor')
    datetime = request.form.get('datetime')

    # Basic validation (can be improved)
    if not all([name, email, doctor, datetime]):
        return jsonify({'status': 'error', 'message': 'All fields are required.'}), 400

    # Placeholder for booking logic (e.g., save to a database)
    try:
        # Simulate success
        print(f"Booking requested for {name} ({email}) with {doctor} at {datetime}")  # Log request
        return jsonify({'status': 'success', 'message': 'Booking request submitted successfully!'}), 200
    except Exception as e:
        print(f"Booking error: {e}")  # Log error
        return jsonify({'status': 'error', 'message': 'Failed to submit booking. Please try again.'}), 500


# --- Run the App ---
if __name__ == '__main__':
    # Use host='0.0.0.0' to make it accessible on your network
    app.run(debug=True, host='0.0.0.0', port=5000)  # debug=True for development ONLY