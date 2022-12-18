from django.db import migrations


def create_data(apps, schema_editor):
    PayoutNums = apps.get_model("payout", "PayoutNums")
    PayoutNums(wagerNum="100", oddsNum="900").save()


class Migration(migrations.Migration):

    dependencies = [
        ("payout", "0001_initial"),
    ]

    operations = [
        migrations.RunPython(create_data),
    ]
