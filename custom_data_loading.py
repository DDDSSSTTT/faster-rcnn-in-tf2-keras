from data import pascal
from data import pascal_custom
ds = pascal.pascal_voc(is_training=True, use_diff=False)
eval_ds = pascal.pascal_voc(is_training=False, use_diff=False)
print("data load finished")
cust_ds = pascal_custom.pascal_voc(is_training=True, use_diff=False)
eval_ds = pascal_custom.pascal_voc(is_training=False, use_diff=False)
print("custom data load finished")