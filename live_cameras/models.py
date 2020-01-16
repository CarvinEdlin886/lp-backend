from django.db import models

class LiveCamera(models.Model):
    base64 = models.TextField(blank=True, default='')
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created']