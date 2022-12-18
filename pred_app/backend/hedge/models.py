from django.db import models


class HedgeNums(models.Model):
    wagerNum = models.FloatField()
    oddsNum = models.FloatField()
    hedgeNum = models.FloatField()

    def __str__(self):
        return self.wagerNum, self.oddsNum, self.hedgeNum
