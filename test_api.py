import requests

BASE = 'http://localhost:8001'

# 健康检查
r = requests.get(f'{BASE}/health')
print(f'Health: {r.json()}')

# 无参考图
r = requests.post(f'{BASE}/generate', data={
    'prompt': '1girl, anime style, long blue hair, purple eyes, cherry blossom garden, soft sunlight, masterpiece',
    'steps': 20,
})
with open('test_api_noref.png', 'wb') as f:
    f.write(r.content)
print(f'No-ref: {len(r.content)} bytes -> test_api_noref.png')

# 有参考图
with open('ref_images/holo.png', 'rb') as f:
    r = requests.post(f'{BASE}/generate', data={
        'prompt': 'portrait of a cute wolf girl with brown hair and wolf ears, medieval traveler outfit, warm tavern interior, anime style',
        'steps': 20,
    }, files={'ref_image': f})
with open('test_api_ref.png', 'wb') as f:
    f.write(r.content)
print(f'With-ref: {len(r.content)} bytes -> test_api_ref.png')

print('Done')