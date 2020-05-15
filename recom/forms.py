from django import forms
from .models import Comment

class CommentForm(forms.ModelForm):
	class Meta:
		model = Comment

		fields = ['comment_textfield']
		widgets = {
			'comment_textfield' : forms.Textarea(attrs={'class' : 'form-control', 'rows' : 4, 'cols' : 40})
		}