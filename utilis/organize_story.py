import os
import re
import torchvision.transforms as T
from datasets import load_dataset, Dataset
from PIL import Image
from tqdm.auto import tqdm


datase_path = "OpenstoryPlusPlus\story"
jpg_path = "frames"
save_path = "story_samples"


dataset = load_dataset(datase_path, split="train")
timestamp_pattern = re.compile(r"_keyframe_(\d+)-(\d+)-(\d+)-(\d+)")

jpg_files = []
for root, dirs, files in os.walk(jpg_path):
    for file in files:
        if file.endswith(".jpg"):
            jpg_files.append(os.path.join(root, file))


def key_func(x):
    base_name = os.path.basename(x)
    # use the youtube id and timestamp for sorting
    return base_name[:11], tuple(
        map(int, timestamp_pattern.search(base_name.split(".")[0]).groups())
    )


def edit_key(key):
    if "override" in key:
        # only keep the part before the first "override"
        key = key.split("override")[0][:-1]
    return key


jpg_transform = T.Compose([T.Resize(512), T.CenterCrop(512)])


jpg_files = sorted(jpg_files, key=key_func)
keys = [os.path.basename(x).split(".")[0] for x in jpg_files]
image_idx, sample_idx = 0, 0
image_max, sample_max = len(jpg_files), len(dataset)
selected_samples = []
bar = tqdm(total=sample_max)
# the length of the image is less than the length of the dataset
while True:
    if image_idx >= image_max or sample_idx >= sample_max:
        break
    sample = dataset[sample_idx]
    image_key = keys[image_idx]
    sample_key = sample["__key__"]
    sample_key = edit_key(sample_key)
    if image_key == sample_key:
        jpg_image = Image.open(jpg_files[image_idx]).convert("RGB")
        jpg_image = jpg_transform(jpg_image)
        sample["jpg"] = jpg_image
        selected_samples.append(sample)
        sample_idx += 1
        # if the next sampke_key is the same as the current sample_key, the image_idx keep
        if not (
            sample_idx + 1 < sample_max
            and sample_key == edit_key(dataset[sample_idx + 1]["__key__"])
        ):
            image_idx += 1
    else:
        sample_idx += 1
    bar.update(1)

data = Dataset.from_list(selected_samples)
data.save_to_disk(save_path)
