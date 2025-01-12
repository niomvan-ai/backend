from django.test import TestCase
from django.contrib.auth.models import User
from rest_framework.test import APIClient
from rest_framework import status

# Create your tests here.
class UserAuthTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.registerUrl = "/api/user/register/"
        self.loginUrl = "/api/token/"
        self.validUserData = {
            "username": "testuser",
            "password": "password123"
        }

    def testUserRegistration(self):
        # Register a new user
        response = self.client.post(self.registerUrl, self.validUserData)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data["username"], self.validUserData["username"])

    def testRegistrationWithDuplicateUsername(self):
        # Create a user with valid data
        User.objects.create_user(**self.validUserData)

        # Attempt to register a user with the same username
        response = self.client.post(self.registerUrl, self.validUserData)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("username", response.data)

    def testRegistrationWithMissingFields(self):
        # Register a user with missing fields
        incompleteUserData = {
            "username": "testuser"
            # Missing password
        }
        response = self.client.post(self.registerUrl, incompleteUserData)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("password", response.data)

    def testLoginWithValidCredentials(self):
        # First, register the user
        self.client.post(self.registerUrl, self.validUserData)

        # Now, attempt to login with the correct credentials
        loginData = {
            "username": self.validUserData["username"],
            "password": self.validUserData["password"]
        }
        response = self.client.post(self.loginUrl, loginData)
        
        # Assert that the login is successful and returns a JWT token
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("access", response.data)  # Ensure a JWT token is included

    def testLoginWithInvalidCredentials(self):
        # Attempt to login with an invalid username
        invalidLoginData = {
            "username": "wronguser",
            "password": self.validUserData["password"]
        }
        response = self.client.post(self.loginUrl, invalidLoginData)
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

        # Attempt to login with an invalid password
        invalidLoginData = {
            "username": self.validUserData["username"],
            "password": "wrongpassword"
        }
        response = self.client.post(self.loginUrl, invalidLoginData)
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

class SymptomViewTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.url = "/api/symptoms/"
        self.user = User.objects.create_user(username="testuser", password="password123")
        self.client.force_authenticate(user=self.user)

    def testSymptomViewWithValidData(self):
        data = {"symptoms": "fever, cough, sore throat"}
        response = self.client.post(self.url, data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("summary", response.data)

    def testSymptomViewWithNoData(self):
        response = self.client.post(self.url, {})
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("error", response.data)

class SummaryViewTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.url = "/api/summary/"
        self.user = User.objects.create_user(username="testuser", password="password123")
        self.client.force_authenticate(user=self.user)

    def testSummaryViewWithValidData(self):
        data = {
            "symptoms": "fever, cough, sore throat",
            "doctorRecommendations": "rest, hydration",
            "caseCondition": "mild"
        }
        response = self.client.post(self.url, data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("summary", response.data)

    def testSummaryViewWithMissingData(self):
        data = {"symptoms": "fever, cough, sore throat"}
        response = self.client.post(self.url, data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("summary", response.data)

class OsteoarthritisViewTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.url = "/api/osteoarthritis/"
        self.user = User.objects.create_user(username="testuser", password="password123")
        self.client.force_authenticate(user=self.user)

    def testOsteoarthritisViewWithValidData(self):
        with open("test_image.jpg", "rb") as image:
            response = self.client.post(self.url, {"images": image}, format="multipart")
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn("case_no", response.data)
        self.assertIn("files", response.data)

    def testOsteoarthritisViewWithNoData(self):
        response = self.client.post(self.url, {}, format="multipart")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("detail", response.data)