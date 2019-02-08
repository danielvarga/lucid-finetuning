import sys
import scipy.misc
from lucid.modelzoo.vision_base import Model
import lucid.optvis.render as render
import lucid.optvis.objectives as objectives

MODEL_PATH, LAYER = sys.argv[1:]
# 'inceptionLucid.pb', 'average_pooling2d_9/AvgPool'


class FrozenNetwork(Model):
    model_path = MODEL_PATH
    image_shape = [299, 299, 3]
    image_value_range = (0, 1)
    input_name = 'input_1'

network = FrozenNetwork()
network.load_graphdef()

for i in range(5):
    obj = objectives.channel(LAYER, i)
    images = render.render_vis(network, obj)
    assert len(images)==1
    image = images[0]
    assert len(image)==1
    image = image[0]
    scipy.misc.imsave("img_%0d.png" % i, image)
