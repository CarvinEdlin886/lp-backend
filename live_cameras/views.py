from live_cameras.serializers import LiveCameraSerializer
from results.models import Result
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from django.http import HttpResponse
import base64
from examples.DCOD.image_detection_demo import image_detection
from django.core import serializers
import json
from vehicles.models import Vehicle
from django.shortcuts import get_object_or_404
import datetime

class Vehicle2:
    def __init__(self, owner_name="", license_number="", phone_number=""):
        self.owner_name = owner_name
        self.license_number = license_number
        self.phone_number = phone_number

    def __repr__(self):
        return str(self)

class LiveCameraList(APIView):
    def post(self, request, format=None):
        d = datetime.datetime.today()
        serializer = LiveCameraSerializer(data=request.data)
        if serializer.is_valid():
            # serializer.save()
            img_data = base64.b64decode(request.data['base64'])
            filename = 'vehicle.jpg'
            with open(filename, 'wb') as f:
                f.write(img_data)
            model = "trained-model/deploy.prototxt"
            weights = "trained-model/Tiny-DSOD.caffemodel"
            boundingBoxes, predicted_texts = image_detection(model, weights, filename)

            for p in predicted_texts:
                try:
                    result = Result.objects.get(license_number=p)
                except Result.DoesNotExist:
                    result = None

                if result == None:
                    for p2 in p[::-1]:
                        print("[P2] ", p2)
                        print("[P2.ISDIGIT] ", p2.isdigit())
                        if p2.isdigit():
                            print("[P2[1]] ", p2)
                            print("[D.DAY] ", d.day)
                            print("[P2%2] ", int(p2) % 2)
                            print("[D.DAY%2] ", d.day % 2)
                            if d.day % 2 != int(p2) % 2:
                                Result.objects.create(vehicle="Car", license_number=p)
                            break

            results = Result.objects.all().values('license_number')
            vehicle2 = []
            for r in results:
                try:
                    vehicle = Vehicle.objects.get(license_number=r["license_number"])
                except Vehicle.DoesNotExist:
                    vehicle = None

                if vehicle != None:
                    vehicle2.append(Vehicle2(vehicle.owner_name, vehicle.license_number, vehicle.phone_number).__dict__)

            response = json.dumps({"coordinates": boundingBoxes, "vehicles": vehicle2})

            return HttpResponse(response, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)