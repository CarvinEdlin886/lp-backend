from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken, BlacklistMixin

import base64

class AuthList(APIView):
    def post(self, request, format=None):
        data = request.body['token']
        print(data)
        # token = RefreshToken(base64.b64encode(data.encode("utf-8")))
        # token.blacklist()
        return Response("OK", status=status.HTTP_200_OK)