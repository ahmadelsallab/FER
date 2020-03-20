from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.template import loader
from django.core.files.storage import FileSystemStorage
model = None
video = None    
def upload_model(request):
    global model
    #template = loader.get_template('fer/index.html')
    try:
        if request.method == 'POST' and request.FILES['myfile']:
            
            myfile = request.FILES['myfile']
            fs = FileSystemStorage()
            filename = fs.save(myfile.name, myfile)
            uploaded_file_url = fs.url(filename)
            
            model = uploaded_file_url
            
            return render(request, 'fer/upload.html', {'uploaded_file_url': uploaded_file_url})
    except:
        #print('Upload file first')
        if model:
            return render(request, 'fer/upload.html', {'uploaded_file_url': model})
        else:        
            return render(request, 'fer/upload.html', {'uploaded_file_url': 'Upload file first'})
    model = None
    return render(request, 'fer/upload.html')
    
def upload_video(request):
    global video
    #template = loader.get_template('fer/index.html')
    try:
        if request.method == 'POST' and request.FILES['myfile']:
            myfile = request.FILES['myfile']
            fs = FileSystemStorage()
            filename = fs.save(myfile.name, myfile)
            uploaded_file_url = fs.url(filename)
            
            video = uploaded_file_url
            return render(request, 'fer/play_video.html', {'uploaded_file_url': uploaded_file_url})
    except:
        #print('Upload file first')
        if video:
            return render(request, 'fer/play_video.html', {'uploaded_file_url': video})
        else:
            return render(request, 'fer/upload.html', {'uploaded_file_url': 'Upload file first'})
    video = None
    return render(request, 'fer/upload.html')
    '''
    context = {}
    return HttpResponse(template.render(context, request))    
    #return HttpResponse("Hello, world. You're at the polls index.")
    '''
def predict(request):
    '''
    #template = loader.get_template('fer/index.html')
    try:
        if request.method == 'POST' and request.FILES['myfile']:
            myfile = request.FILES['myfile']
            fs = FileSystemStorage()
            filename = fs.save(myfile.name, myfile)
            uploaded_file_url = fs.url(filename)
            return render(request, 'fer/index.html', {'uploaded_file_url': uploaded_file_url})
    except:
        #print('Upload file first')
        return render(request, 'fer/upload.html', {'uploaded_file_url': 'Upload file first'})
    '''
    if model: print(model)
    if video: print(video)
    return render(request, 'base.html')    