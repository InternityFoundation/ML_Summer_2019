from django import forms

class contactform(forms.Form):
    name = forms.CharField(max_length=20,
        widget= forms.TextInput(attrs={
            'class': 'input1',
            'placeholder': 'Name'
        }))
    email = forms.CharField(max_length=25,
        widget= forms.TextInput(attrs={
            'class':'input1',
            'placeholder':'Email'
        }))

    Subject = forms.CharField(max_length=25,
        widget= forms.TextInput(attrs={
            'class':'input1',
            'placeholder':'Subject'
        }))
    
    Message = forms.CharField(max_length=25,
        widget= forms.Textarea(attrs={
            'class':'input1',
            'placeholder':'Message'
        }))