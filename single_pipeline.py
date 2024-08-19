import argparse
import json
import os
import time
import warnings
from io import BytesIO

import cv2
import imgviz
import nltk
import numpy as np
import supervision as sv
import torch
import webdataset as wds
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
from efficientvit.sam_model_zoo import create_sam_model
from inference.models import YOLOWorld
from matplotlib import pyplot as plt
from nltk.stem import WordNetLemmatizer
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm
from transformers import Blip2ForConditionalGeneration, Blip2Processor

# Download nltk resources
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")

# Create a WordNet lemmatizer instance
lemmatizer = WordNetLemmatizer()

# Ignore warnings
warnings.filterwarnings("ignore")

colormap = imgviz.label_colormap()

argparser = argparse.ArgumentParser()

argparser.add_argument(
    "--input_dir",
    type=str,
    default="./test_image",
    help="input dir containing webdataset tar files, whose sample has key jpg",
)
argparser.add_argument("--output_dir", type=str, default="./target", help="output dir for saving results")

# model can be downloaded from https://huggingface.co/han-cai/efficientvit-sam/resolve/main/xl1.pt
argparser.add_argument(
    "--efficientvit_model_path",
    type=str,
    default="/path/to/efficientvit/xl1.pt",
    help="efficientvit model pth",
)

# model can be downloaded from https://huggingface.co/Salesforce/blip2-opt-2.7b
argparser.add_argument(
    "--blip2_model_path",
    type=str,
    default="/path/to/blip2-opt-2.7b",
    help="blip2 model path, any blip2 model type can be used, blip2-opt-2.7b, blip2-flan-t5-xl, etc.",
)

argparser.add_argument(
    "--detect_confidence",
    type=float,
    default=0.03,
    help="detect confidence",
)
argparser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="device",
)

args = argparser.parse_args()

effivientvit_model_path = args.efficientvit_model_path
blip2_model_path = args.blip2_model_path
detect_confidence = args.detect_confidence
device = args.device


def image_tensor2pillow(input_tensor: torch.Tensor):
    global colormap
    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor.to(torch.device("cpu"))
    lbl_pil = Image.fromarray(input_tensor.type(torch.uint8).numpy(), mode="P")
    lbl_pil.putpalette(colormap.flatten())
    return lbl_pil


class BLIP:
    def __init__(self, device):
        self.processor = Blip2Processor.from_pretrained(blip2_model_path)
        self.model = (
            Blip2ForConditionalGeneration.from_pretrained(blip2_model_path, torch_dtype=torch.float16).to(device).eval()  # type: ignore
        )

    @torch.inference_mode()
    def get_caption(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(  # type: ignore
            self.model.device,
            torch.float16,  # type: ignore
        )
        generated_ids = self.model.generate(**inputs, max_new_tokens=30)
        generated_text = self.processor.batch_decode(  # type: ignore
            generated_ids, skip_special_tokens=True
        )
        return generated_text


class Pipeline:
    def __init__(self, device="cuda", use_blip2=True, batch_size=128):
        global effivientvit_model_path, detect_confidence
        self.device = device
        self.image_transforms = T.Compose(
            [
                T.Resize(512, interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop(512),
            ]
        )
        self.image_transforms_save = T.Compose(
            [
                T.Resize(512, interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop(512),
            ]
        )
        self.key_verifier = wds.filters.pipelinefilter(self.verify_keys)
        self.yolo_world = YOLOWorld(model_id="yolo_world/v2-x")
        print("yolo_world loaded")
        self.efficientvit_sam = EfficientViTSamPredictor(create_sam_model(name="xl1", weight_url=effivientvit_model_path).to(device).eval())
        print("efficientvit_sam loaded")
        self.blip2 = None
        self.batch_size = batch_size
        if use_blip2:
            self.blip2 = BLIP(device)
        print("blip2 loaded")
        self.detect_confidence = detect_confidence

    def normalized(self, a, axis=-1, order=2):
        l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
        l2[l2 == 0] = 1
        return a / np.expand_dims(l2, axis)

    def get_count(self, input_file):
        stats_file = input_file[:-4] + "_stats.json"
        f = open(stats_file)
        stats = json.load(f)
        f.close()
        count = stats["successes"]
        return count

    def preproc(self, sample):
        instance_image = sample["jpg"]
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        sample["jpg"] = instance_image
        return sample

    def verify_keys(self, samples, required_keys):
        for sample in samples:
            for key in required_keys:
                assert key in sample, f"Sample {sample['__key__']} missing {key}. Has keys {sample.keys()}"
            yield {key: sample[key] for key in required_keys}

    def process_pipeline(self, input_file, output_file):
        start = time.time()
        input_file = "file:" + input_file
        pre_pipeline = wds.DataPipeline(
            wds.SimpleShardList(input_file),
            wds.tarfile_to_samples(),
            wds.decode("pil"),
            self.key_verifier(required_keys=["__key__", "jpg"]),
            wds.map(self.preproc),
            wds.to_tuple("__key__", "jpg"),
            wds.batched(self.batch_size),
        )

        samples = []

        for index, (keys, images) in enumerate(pre_pipeline):
            caption_lemmas = []
            tag_prompts = []
            text_prompt_list = []
            caption_lemma = []
            global lemmatizer

            images = [self.image_transforms(img) for img in images]

            if self.blip2 is not None:
                captions = self.blip2.get_caption(images)

            captions_text = "! ".join(captions) + "!"
            tokens = nltk.word_tokenize(captions_text)

            for token in tokens:
                if token != "!":
                    lemma = lemmatizer.lemmatize(token)
                    caption_lemma.append(lemma)
                    pos_tag = nltk.pos_tag([token])[0][1]
                    if pos_tag in ["NN", "NNS", "NNP", "NNPS"]:
                        text_prompt_list.append(token)
                else:
                    tag_prompt = ",".join(text_prompt_list)
                    tag_prompts.append(tag_prompt)
                    caption_lemmas.append(caption_lemma)
                    text_prompt_list = []
                    caption_lemma = []

            # generate detection boxes (one by one to keep the original image shape)
            json_datas = []
            mask_imgs = []
            # cap_oris_new = []
            keys_new = []
            images_new = []

            for img, tag_prompt, caption_lemma, caption, key in tqdm(zip(images, tag_prompts, caption_lemmas, captions, keys), desc=f"Processing {input_file} batch {index}"):
                tag_prompt = tag_prompt.split(",")

                # if the number of nouns in the caption is less than 2, skip
                if len(tag_prompt) < 2:
                    continue

                image_save = img
                # image = np.array(self.image_transforms(img))
                image = np.array(img)
                self.yolo_world.set_classes(tag_prompt)
                result = self.yolo_world.infer(image_save, confidence=self.detect_confidence)
                detections = sv.Detections.from_inference(result)
                labels = [
                    f"{tag_prompt[class_id]}: {confidence:.2f}"
                    for class_id, confidence in zip(detections.class_id, detections.confidence)  # type: ignore
                ]

                # if we detect no object or too many objects, skip
                if len(labels) == 0 or len(labels) > 8:
                    continue
                self.efficientvit_sam.set_image(image, image_format="RGB")
                masks = []
                for xyxy in detections.xyxy:
                    mask, _, _ = self.efficientvit_sam.predict(box=xyxy, multimask_output=False)
                    masks.append(mask.squeeze())

                boxes = detections.xyxy
                labels_boxes_masks_tuple = list(zip(labels, boxes, masks))
                # use the size of mask to sort from large to small
                labels_boxes_masks_tuple = sorted(labels_boxes_masks_tuple, key=lambda x: x[2].sum(), reverse=True)
                labels = [x[0] for x in labels_boxes_masks_tuple]
                boxes = [x[1] for x in labels_boxes_masks_tuple]
                masks = [x[2] for x in labels_boxes_masks_tuple]

                mask_img = torch.zeros((image.shape[0], image.shape[1]))
                value = 0
                for idx, mask in enumerate(masks):
                    mask_img[mask != 0] = idx + 1 + value
                json_data = {
                    "caption": caption,
                    "mask": [{"value": value, "label": "background"}],
                }
                for label, box in zip(labels, boxes):
                    value += 1
                    name, logit = label.split(": ")
                    json_data["mask"].append(
                        {
                            "value": value,
                            "label": name,
                            "logit": float(logit),
                            "box": box.tolist(),
                        }
                    )
                mask_img = image_tensor2pillow(mask_img)
                json_datas.append(json_data)
                mask_imgs.append(mask_img)
                keys_new.append(key)
                images_new.append(image_save)

            samples.append([keys_new, images_new, json_datas, mask_imgs])

        dst = wds.TarWriter(output_file)
        for sample in tqdm(samples, desc=f"Writing {output_file}"):
            for x, y, json_data, png in zip(sample[0], sample[1], sample[2], sample[3]):
                dst.write({"__key__": x, "jpg": y, "json": json_data, "png": png})
        dst.close()
        end = time.time()
        print(f"Finished - {end-start:.0f}s")


if __name__ == "__main__":
    pipeline = Pipeline(device=device, use_blip2=True)
    input_dir = args.input_dir
    output_dir = args.output_dir
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".tar"):
                input_tar = os.path.join(root, file)
                output_tar = os.path.join(output_dir, f"{file[:-4]}_target.tar")
                pipeline.process_pipeline(input_tar, output_tar)
