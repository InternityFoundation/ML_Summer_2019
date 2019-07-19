from django.shortcuts import render
from .forms import contactform



def home(request):
    if request.method == 'POST':
        form = contactform(request.POST)

    else:
        form = contactform()

    context = {'form':form}

    return render(request,'index.html', context)