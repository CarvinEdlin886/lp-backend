from live_cameras.serializers import LiveCameraSerializer
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from django.http import HttpResponse
import base64
from examples.DCOD.image_detection_demo import image_detection
from django.core import serializers
import json

class LiveCameraList(APIView):
    def post(self, request, format=None):
        serializer = LiveCameraSerializer(data=request.data)
        if serializer.is_valid():
            # serializer.save()
            img_data = base64.b64decode(request.data['base64'])
            filename = 'vehicle.jpg'
            with open(filename, 'wb') as f:
                f.write(img_data)
            model = "trained-model/deploy.prototxt"
            weights = "trained-model/Tiny-DSOD.caffemodel"
            boundingBoxes = image_detection(model, weights, filename)
            response = json.dumps({"coordinates": boundingBoxes})

            return HttpResponse(response, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)