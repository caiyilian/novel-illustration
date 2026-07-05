from modelscope import snapshot_download 

model_dir = snapshot_download('facefusion/insightface-models',local_dir='./models/insightface-models/')
print(f'{model_dir=}')

