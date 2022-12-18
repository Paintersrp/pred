import sys
import json

sys.path.insert(0, "C:/Python/pred_app")

import pandas as pd
from fractions import Fraction
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status
from scripts.calculator import Calculator
from .models import HedgeNums
from .serializers import *


@api_view(["POST", "GET"])
def hedge_detail(request):
    if request.method == "POST":
        oddsType = request.data["typeState"]
        wager = float(request.data["wagerNum"])
        odds = float(request.data["oddsNum"])
        hedge = float(request.data["hedgeNum"])

        calc = Calculator(oddsType)
        (
            original_payout,
            break_even,
            be_payout,
            equal_return,
            er_payout,
            er_profit,
        ) = calc.calc_hedge(wager, odds, hedge)

        print(
            original_payout, break_even, be_payout, equal_return, er_payout, er_profit
        )

        serializer = HedgeSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(
                data={
                    "original_payout": round(original_payout, 2),
                    "break_even": round(break_even, 2),
                    "be_payout": round(be_payout, 2),
                    "equal_return": round(equal_return, 2),
                    "er_payout": round(er_payout, 2),
                    "er_profit": round(er_profit, 2),
                },
                status=status.HTTP_201_CREATED,
            )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    else:
        data = HedgeNums.objects.all()

        serializer = HedgeSerializer(data, context={"request": request}, many=True)

        return Response(serializer.data)
