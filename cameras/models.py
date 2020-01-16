from django.db import models

class Camera(models.Model):
    name = models.CharField(max_length=100, blank=True, default='')
    area = models.ForeignKey('areas.Area', related_name='cameras', on_delete=models.CASCADE)
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created']