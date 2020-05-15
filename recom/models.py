# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models

# Create your models here.
class Comment(models.Model):
	movie_idx = models.IntegerField(default=0)
	comment_date = models.DateTimeField(auto_now_add=True)
	#comment_user = models.TextField(max_length=20)
	comment_thumbnail_url = models.TextField(max_length=300)
	comment_textfield = models.TextField()
	comment_sentiment = models.IntegerField(default=-1)