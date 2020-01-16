from django.db import models

class TnkbColor(models.Model):
    name = models.CharField(max_length=100, blank=True, default='')
    description = models.TextField(max_length=200, blank=True, default='')
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created']