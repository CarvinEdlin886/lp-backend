from django.db import models

class Result(models.Model):
    vehicle = models.CharField(max_length=100, blank=True, default='')
    license_number = models.CharField(max_length=100, blank=True, default='')
    area = models.ForeignKey('areas.Area', related_name='results', on_delete=models.CASCADE)
    status = models.ForeignKey('statuses.Status', related_name='results', on_delete=models.CASCADE)
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created']