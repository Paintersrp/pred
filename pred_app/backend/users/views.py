from rest_framework.response import Response
from rest_framework.exceptions import AuthenticationFailed
from rest_framework.decorators import api_view
from rest_framework import status
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.hashers import make_password
from .serializers import *
from .models import Users
import jwt, datetime


@api_view(["GET", "POST"])
def register_view(request):
    if request.method == "GET":
        data = Users.objects.all()
        serializer = UserSerializer(data, context={"request": request}, many=True)

        return Response(serializer.data)

    else:
        serializer = UserSerializer(data=request.data["formValues"])

        if serializer.is_valid():
            # serializer.validated_data["password"] = make_password(
            #     serializer.validated_data["password"]
            # )
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(["GET", "POST"])
def login_view(request):
    if request.method == "POST":
        print(request.data["email"])
        email = request.data["email"]
        password = request.data["password"]

        user = Users.objects.filter(email=email).first()

        if user is None:
            raise AuthenticationFailed("User not found.")

        if not user.check_password(password):
            raise AuthenticationFailed("Incorrect password.")

        payload = {
            "id": user.id,
            "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=60),
            "iat": datetime.datetime.utcnow(),
        }

        token = jwt.encode(payload, "secret", algorithm="HS256")

        response = Response()
        response.set_cookie(key="jwt", value=token, httponly=True)
        response.data = {"jwt": token, "id": user.id}

        return response


@api_view(["GET"])
def user_view(request):
    if request.method == "GET":
        token = request.COOKIES.get("jwt")

        if not token:
            raise AuthenticationFailed("User not authenticated.")

        try:
            payload = jwt.decode(token, "secret", algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            raise AuthenticationFailed("User not authenticated.")

        user = Users.objects.filter(id=payload["id"]).first()
        serializer = UserSerializer(user)

        return Response(serializer.data)


@api_view(["POST"])
def logout_view(request):
    if request.method == "POST":
        response = Response()
        response.delete_cookie("jwt")
        response.data = {
            "message": "success",
        }

        return response


# @api_view(["PUT", "DELETE"])
# def users_detail(request, key):
#     try:
#         user = Users.objects.get(pk=key)
#     except Users.DoesNotExist:
#         return Response(status=status.HTTP_404_NOT_FOUND)

#     if request.method == "PUT":
#         serializer = UserSerializer(
#             user, data=request.data, context={"request": request}
#         )

#         if serializer.is_valid():
#             serializer.save()
#             return Response(status=status.HTTP_204_NO_CONTENT)

#         return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

#     else:
#         user.delete()
#         return Response(status=status.HTTP_204_NO_CONTENT)
