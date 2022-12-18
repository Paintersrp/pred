import sys

sys.path.insert(0, "C:/Python/pred_app")

from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status
from scripts.calculator import Calculator
from .models import HoldNums
from .serializers import *


@api_view(["POST", "GET"])
def hold_detail(request):
    if request.method == "POST":
        oddsType = request.data["typeState"]
        fav = float(request.data["favNum"])
        under = float(request.data["underNum"])

        calc = Calculator(oddsType)
        fav_implied = round(calc.calc_probability(fav), 2)
        under_implied = round(calc.calc_probability(under), 2)
        hold = round((fav_implied + under_implied) - 100, 2)

        serializer = HoldSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(
                data={
                    "fav_implied": fav_implied,
                    "under_implied": under_implied,
                    "hold": hold,
                },
                status=status.HTTP_201_CREATED,
            )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    else:
        data = HoldNums.objects.all()

        serializer = HoldSerializer(data, context={"request": request}, many=True)

        return Response(serializer.data)
