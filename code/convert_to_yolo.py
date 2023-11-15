import glob
import json
import os
from shutil import move

labels = set()
objects_and_names = []

for annotation_path in glob.glob('*.json'):
    with open(annotation_path, 'r') as file:
        anno = json.load(file)

    image_width = int(anno['width'])
    image_height = int(anno['height'])
    basename = os.path.basename(annotation_path)
    objects_in_this_image = []


    for object in anno['objects']:
        labels.add(object['label'])

        object_x = object['bbox']['xmin']
        object_y = object['bbox']['ymin']
        object_width = object['bbox']['xmax'] - object['bbox']['xmin']
        object_height = object['bbox']['ymax'] - object['bbox']['ymin']

        objects_in_this_image.append({
            'label': object['label'],
            'xcenter': (object_x + object_width / 2) / image_width,
            'ycenter': (object_y + object_height / 2) / image_height,
            'width': object_width / image_width,
            'height': object_height / image_height,
        })

    objects_and_names.append((os.path.splitext(basename)[0], objects_in_this_image))

labels: list = sorted(list(labels))
print(labels)

os.makedirs('images', exist_ok=True)
os.makedirs('labels', exist_ok=True)

for image, objects in objects_and_names:
    input_image_path = f'{image}.jpg'
    print(input_image_path)

    # toctou doesn't matter here
    if not os.path.exists(input_image_path):
        continue

    move(input_image_path, f'images/{image}.jpg')
    with open(f'labels/{image}.txt', 'w') as label:
        for object in objects:
            class_index = labels.index(object['label'])
            label.write(f'{class_index} {object["xcenter"]} {object["ycenter"]} {object["width"]} {object["height"]}\n')

with open('model-descriptor.yml', 'w') as desc:
    desc.write(f"""train: images

nc: {len(labels)}
names: {labels}
""")
