import os
from PIL import Image


# def convert(size, box):
#     dw = 1. / size[0]
#     dh = 1. / size[1]
#     x = (box[0] + box[2]) / 2.0
#     y = (box[4] + box[6]) / 2.0
#     w = box[2] - box[0]
#     h = box[6] - box[4]
#     x = x * dw
#     w = w * dw
#     y = y * dh
#     h = h * dh
#     return x, y, w, h

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h



sets = ['training', 'evaluation']

training_classes = ['A220',
           'A330',
           'A320-321',
           'Boeing737-800',
           'Boeing787',
           'ARJ21',
           'other']

# classes = classes = ['airplane',
#                      'ship',
#                      'storage-tank',
#                      'baseball-diamond',
#                      'tennis-court',
#                      'basketball-court',
#                      'ground-track-field',
#                      'harbor',
#                      'bridge',
#                      'vehicle']

if __name__ == '__main__':
    import sys

    datapath = os.path.abspath(sys.argv[1])

    if not os.path.exists(os.path.join(datapath, 'planelist')):
        os.mkdir(os.path.join(datapath, 'planelist'))

    for class_name in training_classes:
        for image_set in sets:
            files = os.listdir(os.path.join(datapath, '{}/images'.format(image_set)))
            image_ids = [x.strip('.tif') for x in files]
            # class_name = class_name.replace("/", "_")
            list_file = os.path.join(datapath, 'planelist/{}_{}.txt'.format(class_name, image_set))

            label_dir = 'labels_1c/' + class_name
            if not os.path.exists(os.path.join(datapath, '{}/'.format(label_dir))):
                os.makedirs(os.path.join(datapath, '{}/'.format(label_dir)))

            with open(list_file, 'w') as out_f:
                for id in image_ids:
                    in_file = os.path.join(datapath, '{}/annotations/{}.txt'.format(image_set, id))
                    out_file = os.path.join(datapath, 'labels_1c/{}/{}.txt'.format(class_name, id))
                    image = os.path.join(datapath, '{}/images/{}.tif'.format(image_set, id))

                    im = Image.open(image)
                    width, height = im.size

                    with open(in_file, 'r') as in_f:
                        objs = [x.strip().split(' ') for x in in_f.readlines()]

                    write_text = []
                    for obj in objs:
                        if len(obj) < 5:
                            continue
                        cls = obj[4]
                        # if training_classes[cls - 1] != class_name:
                        if cls != class_name:
                            continue

                        # cls_id = classes.index(cls)
                        cls_id = 0
                        b = (float(obj[0]), float(obj[2]), float(obj[1]), float(obj[3]))
                        bb = convert((width, height), b)
                        write_text.append(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
                        # xs = [float(obj[0]), float(obj[2])]
                        # ys = [float(obj[1]), float(obj[3])]
                        # b = (min(xs), max(xs), min(ys), max(ys))
                        # # b = (float(obj[0]), float(obj[2]), float(obj[4]), float(obj[6]))
                        # bb = convert((width, height), b)
                        # write_text.append(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

                    if len(write_text):
                        with open(out_file, 'w') as f:
                            for txt in write_text:
                                f.write(txt)
                        out_f.write('{}/{}/images/{}.tif\n'.format(datapath, image_set, id))