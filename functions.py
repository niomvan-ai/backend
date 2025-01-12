import tensorflow as tf
import google.generativeai as genai
import numpy as np
import textwrap

ImageDataGenarator = tf.keras.preprocessing.image.ImageDataGenerator
img_to_array = tf.keras.preprocessing.image.img_to_array
load_img = tf.keras.preprocessing.image.load_img
load_model = tf.keras.models.load_model

def load_trained_model(path):
    """Loads the pre-trained model."""
    model = load_model(path)
    return model

def process_image(image_path):
    """Preprocesses the image for model input."""
    img = load_img(image_path, target_size=(300, 300))
    x = img_to_array(img) / 255.0 
    x = np.expand_dims(x, axis=0)
    return x

def predict_case(image):
    """Predicts the case number based on the image."""
    model = load_trained_model(".\\AI_model.h5")
    prediction = model.predict(image).tolist()[0]
    prediction = [round(p * 100, 2) for p in prediction]
    case_no = prediction.index(max(prediction))
    return case_no

def symptom_association(symptoms):
    """Extracts potential symptom keywords from a text description."""

    response = generate_gemini_response(f"Can you pick out the important \
     stuff from this text of symptoms? From the symptoms, can you guess the\
     disease, the treatment, and what is important for the doctor to know\
     (in doctor terms)? {symptoms}")
    
    return response

def summary_assessment(symptoms, case_grading, doctor_recommendations=None):
    """Generates a response using the Gemini AI model, 
    incorporating symptoms, doctor recommendations, and case grading.

    Args:
        symptoms: The patient's symptoms.
        doctor_recommendations: Optional doctor recommendations.
        case_grading: case grading information.

    Returns:
        The generated response from the Gemini AI model.
    """

    prompt = f"The symptoms are: {symptoms} "
    prompt += f"Doctor says: {doctor_recommendations}. "
    prompt += f"KL Case grading: {case_grading}. "
    prompt += "please create a summary for the doctor(in doctor terms) and\
        for the patient, explaining his desease. "

    response = generate_gemini_response(prompt)
    return response

def generate_gemini_response(prompt):
    """Generates a response using the Gemini AI model."""
    genai.configure(api_key="AIzaSyDSLg4jg9epkg2DjqANqymmGfsQXktjEpg")
    model = genai.GenerativeModel("gemini-pro")
    chat = model.start_chat(history=[])
    response = chat.send_message(prompt)
    text = textwrap.indent(str(response.text).replace("•", "*"), " ", predicate=lambda _: True)
    return response.text

def main():
    # Define symptoms and image path
    symptoms = "ache/pain often very much hurting hip and knee"
    image_path = ".\\sample_cases\\Case 0.png"
    
    # Gemini AI 1: Analyze symptoms
    
    print(f"AI 1 response: {symptom_association(symptoms)}")

    # Image Classification: Predict case number
    image = process_image(image_path)
    case_no = predict_case(image)
    print(f"AI 2 Response is: This is Detected As Osteoarthritis KL Grade {case_no}.")

    # Gemini AI 3: Combine information and generate response
    if case_no > 2:
        doc_recommend = "I recommend surgery as grade is high"
    elif case_no > 0:
        doc_recommend = "NSAID medicine can be considered"
    else: 
        doc_recommend = "No osteoarthritis present"

    prompt = summary_assessment(symptoms, \
                                case_no, \
                                doc_recommend)

    print(f"AI3 response is: {prompt}")

if __name__ == "__main__":
    main()