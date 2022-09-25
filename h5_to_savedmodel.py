import numpy as np
import tensorflow as tf
from model.faster_rcnn import FasterRCNN
from data import pascal_custom



model = FasterRCNN(is_training=True)
ds_chess = pascal_custom.pascal_voc(is_training=True, use_diff=False)
ds_salad = pascal_custom.non_voc_txts(is_training=True, use_diff=False)
sanity_sample = np.array(ds_salad.get_small_dataset(30))
_ = model(sanity_sample[0])
model.load_weights("model/ckpt/faster_rcnn_vgg16.h5")
model.save("custom_faster")