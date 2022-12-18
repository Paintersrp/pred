from django.db import models


class PayoutNums(models.Model):
    wagerNum = models.FloatField()
    oddsNum = models.FloatField()

    def __str__(self):
        return self.wagerNum, self.oddsNum
