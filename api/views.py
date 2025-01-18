from tensorflow.keras.preprocessing.image import img_to_array, load_img # type: ignore
from rest_framework.parsers import MultiPartParser, FormParser
from tensorflow.keras.models import load_model # type: ignore
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework.permissions import IsAuthenticated
from django.core.files.storage import default_storage
from django.contrib.auth import get_user_model
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import  generics
import google.generativeai as genai
from rest_framework import status
from django.conf import settings
from .serializers import *
import tensorflow as tf
from io import BytesIO
from PIL import Image
from .models import *
import numpy as np
import textwrap
import pydicom
import os

def generate_gemini_response(prompt):
    """Generates a response using the Gemini AI model."""
    genai.configure(api_key="AIzaSyCAIdH7VdasjEsI2LcgTp_hZMXcvLXWLuQ")
    model = genai.GenerativeModel("gemini-pro")
    chat = model.start_chat(history=[])
    response = chat.send_message(prompt)
    text = textwrap.indent(str(response.text), "    ")
    return response.text

# View to create a new user (patient or doctor)
class CreateUserView(generics.CreateAPIView):
    queryset = get_user_model().objects.all()
    serializer_class = UserSerializer

    def create(self, request, *args, **kwargs):
        # Log the request data
        print("Request data:", request.data)

        # Use the default serializer to create the user
        serializer = self.get_serializer(data=request.data)
        if not serializer.is_valid():
            # Log validation errors
            print("Validation errors:", serializer.errors)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        user = serializer.save()

        # Generate JWT tokens
        refresh = RefreshToken.for_user(user)

        # Return user info along with tokens
        response_data = {
            "user": {
                "id": user.id,
                "username": user.username,
            },
            "tokens": {
                "refresh": str(refresh),
                "access": str(refresh.access_token),
            },
        }
        return Response(response_data, status=status.HTTP_201_CREATED)


class SymptomView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        symptoms = request.data.get("symptoms", None)
        if not symptoms:
            return Response({"error": "No symptoms provided"}, status=status.HTTP_400_BAD_REQUEST)

        print(symptoms)
        
        summary = generate_gemini_response(f"Can you pick out the important \
                                            stuff from this text of symptoms?\
                                            From the symptoms, can you guess the\
                                            disease, the treatment, and what is \
                                            important for the doctor to know\
                                            (in doctor terms)? {symptoms}")

        return Response({"summary": summary}, status=status.HTTP_200_OK)
    
class SummaryView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        symptoms = request.data.get("symptoms", None)
        doctor_recc = request.data.get("doctorRecommendations", None)
        case = request.data.get("caseCondition", None)
        
        prompt = f"The symptoms are: {symptoms} "
        prompt += f"Doctor says: {doctor_recc}. "
        prompt += f"Case and grading: {case}. "
        prompt += "please create a summary for the doctor(in doctor terms) and\
            for the patient, explaining his desease. "
        prompt += "also, make it clearly distinguished, doctor and patient summary, and make it elaborate. "

        summary = generate_gemini_response(prompt)

        return Response({"summary": summary}, status=status.HTTP_200_OK)
    
class OsteoarthritisView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        images = request.FILES.getlist('images')
        if not images:
            return Response({"detail": "No files uploaded."}, status=status.HTTP_400_BAD_REQUEST)

        allowed_types = [
            'image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'image/tiff',
            'application/dicom', 'application/octet-stream', 'application/dicom+json'
        ]
        max_size = 10 * 1024 * 1024  # 10 MB

        saved_files = []
        for image in images:
            # Validate file size
            if image.size > max_size:
                return Response(
                    {"detail": f"File {image.name} exceeds the size limit of 10 MB."},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Process and convert the file
            try:
                file_path = self.process_file(image)
                saved_files.append(file_path)
            except Exception as e:
                return Response(
                    {"detail": f"Failed to process {image.name}: {str(e)}"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            file_path = os.path.join(settings.BASE_DIR, "media", file_path)
            file_path = os.path.normpath(file_path)

            # Predict the case
            case = self.predict_case(file_path)

        return Response(
            {"case_no": case, "files": saved_files},
            status=status.HTTP_201_CREATED
        )

    def process_file(self, file):
        """Converts the uploaded file to PNG and saves it."""
        # Get the file name and extension
        file_name, file_extension = os.path.splitext(file.name)
        file_extension = file_extension.lower()

        if file.content_type in ['application/dicom', 'application/octet-stream', 'application/dicom+json']:
            # Process DICOM file
            ds = pydicom.dcmread(file)
            pixel_array = ds.pixel_array
            image = Image.fromarray(pixel_array)
        else:
            # Process standard image formats
            image = Image.open(file)

        # Convert the image to PNG
        png_io = BytesIO()
        image = image.convert('RGB')  # Ensure it's in RGB mode
        image.save(png_io, format='PNG')
        png_io.seek(0)

        # Save the converted file
        png_file_name = f'{file_name}.png'
        file_path = default_storage.save(f'media/{png_file_name}', png_io)

        return file_path
    
    def predict_case(self, image_path):
        def focal_loss(gamma=2.0, alpha=0.25):
            """Defines the focal loss function."""
            def loss_fn(y_true, y_pred):
                y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=5)  # Match NUM_CLASSES
                y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
                pt = tf.reduce_sum(y_true * y_pred, axis=-1)
                weight = alpha * (1 - pt) ** gamma
                return -tf.reduce_mean(weight * tf.math.log(pt))
            return loss_fn

        def load_trained_model(path):
            """Loads the pre-trained model."""
            try:
                model = load_model(path, custom_objects={"loss_fn": focal_loss()})
                return model
            except Exception as e:
                raise FileNotFoundError(f"Error loading model: {e}")

        def process_image(image_path):
            """Preprocesses the image for model input."""
            try:
                img = load_img(image_path, target_size=(300, 300))
                x = img_to_array(img) / 255.0
                x = np.expand_dims(x, axis=0)
                return x
            except Exception as e:
                raise ValueError(f"Error processing image: {e}")

        try:
            # Load and preprocess the image
            image = process_image(image_path)
            # Load the model
            model = load_trained_model("./trained_model.h5")
            # Make prediction
            prediction = model.predict(image).tolist()[0]
            prediction = [round(p * 100, 2) for p in prediction]
            print(f"Predicted probabilities: {prediction}")
            # Get the case number
            case_no = np.argmax(prediction)
            return case_no
        except Exception as e:
            return f"An error occurred: {e}"