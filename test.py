import os
from dotenv import load_dotenv
from azure.cognitiveservices.vision.face import FaceClient
from azure.cognitiveservices.vision.face import models as face_models # For APIErrorException
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision import models as vision_models # For potential Vision API errors
from msrest.authentication import CognitiveServicesCredentials
from PIL import Image # Not strictly used in this snippet for API calls, but good for image manipulation if needed
import json
# import uuid # Not used in the current snippet

load_dotenv()

# --- Azure Face API Configuration ---
AZURE_FACE_KEY = os.getenv("FACE_API_KEY")
if AZURE_FACE_KEY is None:
    raise ValueError("FACE_API_KEY environment variable is not set.")

AZURE_FACE_ENDPOINT = os.getenv("FACE_ENDPOINT")
if AZURE_FACE_ENDPOINT is None:
    raise ValueError("FACE_ENDPOINT environment variable is not set.")

# --- Azure Computer Vision API Configuration ---
# AZURE_VISION_KEY = os.getenv("VISION_API_KEY") # Load from .env
# if AZURE_VISION_KEY is None:
#     raise ValueError("VISION_API_KEY environment variable is not set.")

# AZURE_VISION_ENDPOINT = os.getenv("VISION_ENDPOINT") # Load from .env
# if AZURE_VISION_ENDPOINT is None:
#     raise ValueError("VISION_ENDPOINT environment variable is not set.")

# --- Initialize Clients ---
face_client = FaceClient(AZURE_FACE_ENDPOINT, CognitiveServicesCredentials(AZURE_FACE_KEY))
# vision_client = ComputerVisionClient(AZURE_VISION_ENDPOINT, CognitiveServicesCredentials(AZURE_VISION_KEY))

def classify_age(age):
    return "baby" if age <= 3 else "adult"

def analyze_frame(image_path, frame_index=0, timestamp=0.0):
    print(f"Analyzing frame: {image_path}")
    print(f"Using Face Endpoint: {AZURE_FACE_ENDPOINT}")
    # print(f"Using Face Key: {AZURE_FACE_KEY[:5]}...") # Be careful printing keys, even partially

    faces_data = []
    detected_objects_data = None

    # --- Face API Call ---
    try:
        with open(image_path, 'rb') as image_stream_face:
            faces_data = face_client.face.detect_with_stream(
                image=image_stream_face,
                return_face_attributes=['age', 'emotion'],
                detection_model='detection_03' # Or 'detection_01'. 'detection_03' is generally more accurate.
            )
    except face_models.APIErrorException as e:
        print(f"Azure Face API Error: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Status code: {e.response.status_code}")
            try:
                error_details = e.response.json() # Often contains detailed error info
                print(f"Error details: {json.dumps(error_details, indent=2)}")
            except json.JSONDecodeError:
                print(f"Raw response content: {e.response.text}")
        # Re-raise the exception if you want the script to stop,
        # or handle it (e.g., return partial results or an error indicator)
        raise
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during Face API call: {e}")
        raise

    # --- Object Detection (Computer Vision API) Call ---
    try:
        with open(image_path, 'rb') as image_stream_vision:
            detected_objects_data = vision_client.analyze_image_in_stream(
                image_stream_vision,
                visual_features=[vision_models.VisualFeatureTypes.objects]
            )
    except vision_models.ComputerVisionErrorException as e: # Use the specific exception type
        print(f"Azure Computer Vision API Error: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Status code: {e.response.status_code}")
            try:
                error_details = e.response.json()
                print(f"Error details: {json.dumps(error_details, indent=2)}")
            except json.JSONDecodeError:
                print(f"Raw response content: {e.response.text}")
        # Depending on requirements, you might want to continue without object data
        # or raise the error. For now, we'll let it pass if faces were detected.
        # If you want to stop: raise
    except FileNotFoundError:
        # This would be redundant if the Face API call already failed due to FileNotFoundError
        print(f"Error: Image file not found at {image_path} (for Vision API)")
        # Potentially handle or raise
    except Exception as e:
        print(f"An unexpected error occurred during Vision API call: {e}")
        # Potentially handle or raise

    people = []
    for i, face in enumerate(faces_data):
        if face.face_attributes: # Check if face_attributes were returned
            emotion_scores = face.face_attributes.emotion.as_dict()
            primary_emotion = max(emotion_scores, key=emotion_scores.get) if emotion_scores else "unknown"
            age = face.face_attributes.age if face.face_attributes.age is not None else "unknown"

            people.append({
                "id": f"person_{i+1}", # You could use face.face_id if available and unique
                "type": classify_age(age) if isinstance(age, (int, float)) else "unknown",
                "emotion": primary_emotion,
                "interacted_objects": []
            })
        else:
            people.append({
                "id": f"person_{i+1}",
                "type": "unknown",
                "emotion": "unknown",
                "interacted_objects": []
            })


    objects_in_frame = []
    if detected_objects_data and detected_objects_data.objects:
        objects_in_frame = [obj.object_property.lower() for obj in detected_objects_data.objects]

    return {
        "frame_index": frame_index,
        "timestamp": timestamp,
        "people": people,
        "objects_in_frame": objects_in_frame
    }

# --- Main execution ---
if __name__ == "__main__":
    # Ensure your .env file has:
    # FACE_API_KEY=your_face_api_key
    # FACE_ENDPOINT=https://your_face_resource_name.cognitiveservices.azure.com/
    # VISION_API_KEY=your_vision_api_key
    # VISION_ENDPOINT=https://your_vision_resource_name.cognitiveservices.azure.com/

    # Make sure this path is correct
    image_file_path = "data/video_001/frames/frame_0000.jpg"

    # Check if image file exists before calling
    if not os.path.exists(image_file_path):
        print(f"CRITICAL: Image file does not exist at {image_file_path}. Please check the path.")
    else:
        try:
            result = analyze_frame(image_file_path)
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Script failed: {e}")