import sys
import json

sys.path.insert(0, "/home/ubuntu/pred/pred_app/")

import pandas as pd
from fractions import Fraction
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status
from scripts.calculator import Calculator
from .models import PayoutNums
from .serializers import *


@api_view(["POST", "GET"])
def payout_detail(request):
    if request.method == "POST":
        oddsType = request.data["typeState"]
        wager = float(request.data["wagerNum"])
        odds = float(request.data["oddsNum"])
        print(odds)
        print(oddsType)

        calc = Calculator(oddsType)
        final = calc.payout(wager, odds)
        dec, amer, impl = calc.convert_odds(odds)

        print(dec, impl)

        serializer = PayoutSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(
                data={
                    "decimal": dec,
                    "american": amer,
                    "implied": impl,
                    "final": round(final, 2),
                },
                status=status.HTTP_201_CREATED,
            )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    else:
        data = PayoutNums.objects.all()

        serializer = PayoutSerializer(data, context={"request": request}, many=True)

        return Response(serializer.data)
