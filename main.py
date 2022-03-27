import base64
import collections
import io
import json
import os
from itertools import islice
import tempfile

import PIL
import numpy as np
import numpy as numpy
from PIL import Image
from flask import render_template, request, Flask
from sklearn.cluster import KMeans
from werkzeug.utils import secure_filename

TOP_RESULT = 100
app = Flask(__name__)


def take(n, iterable):
    """ Return first n items of the iterable as a list """
    return list(islice(iterable, n))


def rgb_to_hex(r_g_b):
    return '%02x%02x%02x' % r_g_b


def palette(clusters):
    width = 300
    palette = np.zeros((50, width, 3), np.uint8)
    steps = width/clusters.cluster_centers_.shape[0]
    for idx, centers in enumerate(clusters.cluster_centers_):
        palette[:, int(idx*steps):(int((idx+1)*steps)), :] = centers
    return palette


@app.route("/", methods=['GET', 'POST'])
def home():
    image_name = ""
    if request.method == 'POST':
        file = request.files['file']
        print(file)
        # top_colors = int(request.form['top'])
        # print(top_colors)
        image_name = f"static/images/{file.filename}"
        try:
            im = Image.open(file)
            # file.save(image_name)
        except OSError or PIL.UnidentifiedImageError:
            image_name = "static/images/img.jpg"
            im = Image.open(image_name)
    else:
        top_colors = 5
        image_name = "static/images/img.jpg"
        im = Image.open(image_name)
    data = io.BytesIO()
    im = im.convert('RGB')
    im.save(data, "JPEG")
    encoded_img_data = base64.b64encode(data.getvalue())
    image_size = data.tell()
    print(f"image size: {image_size}")
    if image_size <= 40000:
        top_colors = 9
    elif image_size >= 2999992:
        top_colors = 5
    elif 119876 <= image_size < 2999992:
        top_colors = 7
    elif 40000 < image_size < 119876:
        top_colors = 5
    im = im.resize((80, 80))
    pix = numpy.array(im)
    # img_type = type(pix)
    # img_shape = pix.shape
    # img_dim = pix.ndim
    # print(pix)
    # print(f"image type: {img_type}; shape: {img_shape}; dim: {img_dim}")
    clt = KMeans(n_clusters=top_colors)
    clt.fit(pix.reshape(-1, 3))
    labels = clt.labels_.tolist()
    print(f"labels : {labels}")
    # unique labels values
    unique_labels = np.unique(clt.labels_, axis=0, return_counts=True)
    list_unique_labels = unique_labels[0].tolist()
    list_unique_labels_frequency = unique_labels[1].tolist()
    list_centers = clt.cluster_centers_.tolist()
    # print(f"unique labels: {list_unique_labels}")
    # print(f"unique labels frequency : {list_unique_labels_frequency}")
    sum_frequency = sum(list_unique_labels_frequency)
    new_lst = [round(i/sum_frequency * 100, 2) for i in list_unique_labels_frequency]
    # print(f"unique labels frequency percentage: {new_lst}")
    print(f"centers: {list_centers}")
    colors_dict = {}
    for idx, color in enumerate(list_centers):
        rgb = (int(color[0]), int(color[1]), int(color[2]))
        colors_dict[f"{rgb}"] = [rgb_to_hex(rgb), new_lst[idx]]
    # print(f"colors_dict: {colors_dict}")
    sort_orders = {k: v for k, v in sorted(colors_dict.items(), key=lambda x: x[1][1], reverse=True)}
    # print(f"sort_orders: {sort_orders}")
    # with open(f'top_colors.json', 'w') as fp:
    #     json.dump(sort_orders, fp, indent=4)
    # if image_name != "static/images/img.jpg":
    #     os.remove(image_name)
    return render_template("index.html", colours=sort_orders, img_data=encoded_img_data.decode('utf-8'))


def unique_colors(img_shape, pix):
    colors_dict = collections.defaultdict(list)
    for h in range(img_shape[0]):
        for w in range(img_shape[1]):
            colors_dict[str(tuple(pix[h][w][:3]))].append(1)
    for k, v in colors_dict.items():
        colors_dict[k] = sum(v)
    sort_orders = sorted(colors_dict.items(), key=lambda x: x[1], reverse=True)
    final_colors_items = {}
    total = sum(colors_dict.values())
    for i in sort_orders:
        rgb_tuple = i[0]
        characters_to_remove = "()"
        for character in characters_to_remove:
            rgb_tuple = rgb_tuple.replace(character, "")
        rgb = tuple(map(int, rgb_tuple.split(', ')))
        final_colors_items[i[0]] = [rgb_to_hex(rgb), round(100 * i[1] / total, 2)]
    with open(f'colors.json', 'w') as fp:
        json.dump(final_colors_items, fp, indent=4)
    n_items = take(TOP_RESULT, final_colors_items.items())
    print(n_items)


if __name__ == '__main__':
    app.run(debug=True)
