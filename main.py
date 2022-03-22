import collections
import json
from itertools import islice
import numpy as numpy
from PIL import Image
from flask import render_template, request, Flask
from werkzeug.utils import secure_filename
TOP_RESULT = 100
app = Flask(__name__)


def take(n, iterable):
    """ Return first n items of the iterable as a list """
    return list(islice(iterable, n))


def rgb_to_hex(r_g_b):
    return '%02x%02x%02x' % r_g_b


@app.route("/", methods=['GET', 'POST'])
def home():
    image_name = ""
    if request.method == 'POST':
        f = request.files['file']
        image_name = f"static/images/{f.filename}"
        f.save(image_name)
        im = Image.open(image_name)
    else:
        image_name = "static/images/yass_berr.png"
        im = Image.open(image_name)
    pix = numpy.array(im)
    img_type = type(pix)
    img_shape = pix.shape
    img_dim = pix.ndim
    print(f"image type: {img_type}; shape: {img_shape}; dim: {img_dim}")
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
    # This method will show image in any image viewer
    # im.show()
    return render_template("index.html", colours=n_items, img_name=image_name)


if __name__ == '__main__':
    app.run(debug=True)
