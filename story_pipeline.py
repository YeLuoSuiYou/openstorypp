import argparse
import ast
import json
import os
import random
import time
import warnings

import imgviz
import nltk
import numpy as np
import spacy
import supervision as sv
import torch
import webdataset as wds
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
from efficientvit.sam_model_zoo import create_sam_model
from inference.models import YOLOWorld
from nltk.stem import WordNetLemmatizer
from openai import OpenAI
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm
from transformers import Blip2ForConditionalGeneration, Blip2Processor
from videollava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from videollava.conversation import SeparatorStyle, conv_templates
from videollava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, tokenizer_image_token
from videollava.model.builder import load_pretrained_model
from zhipuai import ZhipuAI

# disable_torch_init()
nlp = spacy.load("en_core_web_sm")

# # Download nltk resources
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
    default="./story_image",
    help="input dir containing webdataset tar files and samples is txt and jpg",
)
argparser.add_argument("--output_dir", type=str, default="./story_target", help="output dir for saving results")
# model can be downloaded from https://huggingface.co/han-cai/efficientvit-sam/resolve/main/xl1.pt
argparser.add_argument(
    "--efficientvit_model_path",
    type=str,
    # default="/liujinxiu/ye/models/efficientvit/xl1.pt",
    default="/path/to/efficientvit/l2.pt",
    help="efficientvit model pth",
)
# model can be downloaded from https://huggingface.co/Salesforce/blip2-flan-t5-xl
argparser.add_argument(
    "--blip2_model_path",
    type=str,
    default="/path/to/blip2-flan-t5-xl",
    help="blip2 model path, any blip2 model type can be used, blip2-opt-2.7b, blip2-flan-t5-xl, etc.",
)
argparser.add_argument(
    "--llava_model_path",
    type=str,
    default="/path/to/Video-LLaVA-7B",
)
argparser.add_argument(
    "--huggingface_cache_dir",
    type=str,
    default="/path/to/huggingface_cache_dir",
)
argparser.add_argument(
    "--detect_confidence",
    type=float,
    default=0.001,
    help="detect confidence",
)
argparser.add_argument(
    "--device",
    type=str,
    default="cuda:0",
    help="device",
)
argparser.add_argument(
    "--api_key",
    type=str,
    default="",
    help="chatglm api key from zhipuai, openai, deepseek, etc.",
)

args = argparser.parse_args()

effivientvit_model_path = args.efficientvit_model_path
blip2_model_path = args.blip2_model_path
llava_model_path = args.llava_model_path
huggingface_cache_dir = args.huggingface_cache_dir
detect_confidence = args.detect_confidence
llm_api_key = args.api_key
device = args.device


OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


class LLM:
    def __init__(self):
        self.client = ZhipuAI(api_key=llm_api_key)
        # self.client = OpenAI(api_key=llm_api_key, base_url="https://api.deepseek.com/")

    def get_response(self, system_prompt="", user_prompt=""):
        response = self.client.chat.completions.create(  # type: ignore
            model="glm-3-turbo",  # 填写需要调用的模型名称
            # model="deepseek-chat",
            temperature=0.7,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        )
        response = response.choices[0].message.content
        return response


class BLIP:
    def __init__(self, device):
        self.processor = Blip2Processor.from_pretrained(blip2_model_path)
        self.model = (
            Blip2ForConditionalGeneration.from_pretrained(blip2_model_path, torch_dtype=torch.float16, device_map=device, use_safetensors=True).eval()  # type: ignore
        )

    @torch.inference_mode()
    def get_caption(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(  # type: ignore
            self.model.device,
            torch.float16,  # type: ignore
        )
        generated_ids = self.model.generate(**inputs, max_new_tokens=64)
        generated_text = self.processor.batch_decode(  # type: ignore
            generated_ids, skip_special_tokens=True
        )
        return generated_text


class VideoLLava:
    def __init__(self, device):
        global llava_model_path, huggingface_cache_dir
        self.device = device
        model_name = get_model_name_from_path(llava_model_path)
        self.tokenizer, self.model, _, _ = load_pretrained_model(
            llava_model_path,
            None,
            model_name,
            load_8bit=False,
            load_4bit=True,
            device_map=self.device,
            cache_dir=huggingface_cache_dir,
        )
        self.llava_image_transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD),
            ]
        )

    @torch.inference_mode()
    def run_video_llava(self, frames, inp):
        num_frames = frames.shape[2]
        tensor = frames.to(self.model.device, dtype=torch.float16)
        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()
        inp = " ".join([DEFAULT_IMAGE_TOKEN] * num_frames) + "\n" + inp
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)  # type: ignore
            .to(self.model.device)
        )
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        output_ids = self.model.generate(
            input_ids,
            images=tensor,
            do_sample=True,
            temperature=0.1,
            max_new_tokens=128,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

        seq_caption = self.tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()
        return seq_caption

    def get_sequence_caption(self, image_seq, prompt):
        if len(image_seq) < 8:
            random_repeat_images = random.choices(image_seq, k=(8 - len(image_seq)))
            output_image_seq = image_seq + random_repeat_images
            output_image_seq = sorted(output_image_seq, key=lambda x: image_seq.index(x))
        else:
            output_image_seq = image_seq
        output_image_seq = [self.llava_image_transform(image) for image in output_image_seq]
        output_image_seq = torch.stack(output_image_seq, dim=1).unsqueeze(0)
        seq_caption = self.run_video_llava(output_image_seq, prompt)
        return seq_caption


class Pipeline:
    def __init__(self, device="cuda", use_blip2=True, batch_size=128):
        global effivientvit_model_path, detect_confidence
        self.device = device
        self.batch_size = batch_size
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
        self.blip2 = None
        if use_blip2:
            self.blip2 = BLIP(device)
        self.llm = LLM()
        self.video_llava = VideoLLava(device)
        self.video_llava_prompt = "describe the video focusing on subject, using concise language"
        self.yolo_world = YOLOWorld(model_id="yolo_world/v2-x")
        self.efficientvit_sam = EfficientViTSamPredictor(create_sam_model(name="l2", weight_url=effivientvit_model_path).to(device).eval())
        self.detect_confidence = detect_confidence

    def normalized(self, a, axis=-1, order=2):
        l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
        l2[l2 == 0] = 1
        return a / np.expand_dims(l2, axis)

    def image_tensor2pillow(self, input_tensor: torch.Tensor):
        global colormap
        input_tensor = input_tensor.clone().detach()
        # 到cpu
        input_tensor = input_tensor.to(torch.device("cpu"))
        lbl_pil = Image.fromarray(input_tensor.type(torch.uint8).numpy(), mode="P")
        lbl_pil.putpalette(colormap.flatten())
        return lbl_pil

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
        # deleta \n
        # sample["txt"] = sample["txt"].replace("\n", " ")
        return sample

    def verify_keys(self, samples, required_keys):
        for sample in samples:
            for key in required_keys:
                assert key in sample, f"Sample {sample['__key__']} missing {key}. Has keys {sample.keys()}"
            yield {key: sample[key] for key in required_keys}

    def use_llm_to_refine_caption(self, captions, seq_caption):
        nouns = set()
        try:
            doc = nlp(seq_caption)
        except Exception:
            return captions
        for token in doc:
            if token.pos_ == "NOUN" and token.text.lower() != "video":
                nouns.add(token.text.lower())
        nouns = list(nouns)
        system_prompt = """
        Task: Enhancing Continuity and Storytelling in Captions
        Objective: Given a series of English captions and a list of English nouns, your task is to replace words in the captions with related nouns from the noun list. The goal is to maintain the narrative flow and coherence of the captions while integrating relevant nouns to enrich the storytelling.
        Sample Input:
        Caption List: ["A man with a beard and a hat is standing", "A man holding a small piece of wood in front of a tree", "A man with a beard and hat is holding a knife"]
        Noun List: ["knife", "man", "hat"]
        Output:  ["A man clad in a worn hat and sporting an unkempt beard stood motionless", "The man is clutching a small knife, standing before a towering tree and memoring flooding back", "the bearded man gripped the knife in his calloused hands ready for the next chapter"]
        """
        user_prompt = f"""
        Instructions:
        1. Analyze each caption in the caption list to identify words that can be replaced with related nouns from the provided noun list.
        2. Subsequently, substitute those words with corresponding nouns, ensuring that the overall meaning and coherence of the captions remain intact.
        3. Format the output as a list enclosed in square brackets, with each caption separated by commas and enclosed in quotes without any other explanation
        4. Emphasize the improvement of narrative continuity and storytelling coherence between captions.
        Caption List: {captions}
        Noun List: {nouns}
        """
        count = 0
        while True:
            if count > 2:
                return captions
            try:
                response = self.llm.get_response(system_prompt, user_prompt)
                response = response[response.find("[") : response.rfind("]") + 1]
                captions_list = ast.literal_eval(response)
                if len(captions_list) == len(captions):
                    return captions_list
                count += 1
            except Exception:
                count += 1
                continue

    def plural_to_singular(self, sentence):
        global lemmatizer
        tokens = nltk.word_tokenize(sentence)
        singular_tokens = []
        for token in tokens:
            singular_token = lemmatizer.lemmatize(token)
            singular_tokens.append(singular_token)

        return " ".join(singular_tokens)

    def process_pipeline(self, input_file, output_file, dst):
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
        total_captions = []
        total_keys = []
        total_images = []

        for index, (keys, images) in tqdm(enumerate(pre_pipeline), desc=f"caption {input_file}"):
            if self.blip2 is not None:
                captions = self.blip2.get_caption(images)
            total_captions.extend(captions)
            total_keys.extend(keys)
            total_images.extend(images)

        torch.cuda.empty_cache()

        # for index, (keys, images) in enumerate(zip(total_keys, total_images)):
        keys = total_keys
        images = total_images
        origin_captions = total_captions
        captions = total_captions
        global lemmatizer

        images = [self.image_transforms(img) for img in images]
        captions = total_captions
        # take video_id in keys as a group, each group contains at most 5 elements, until the video_id is different, then return a list containing many such lists.
        groups = []
        now_group = []

        VIDEO_ID_LEN = 11  # youtube video
        # VIDEO_ID_LEN = 6  # stroysalon
        previous_video_id = keys[0][:VIDEO_ID_LEN]
        for key, image, caption in zip(keys, images, captions):
            if key[:VIDEO_ID_LEN] == previous_video_id and len(now_group) < 5:
                now_group.append((key, image, caption))
            else:
                groups.append(now_group)
                now_group = [(key, image, caption)]
                previous_video_id = key[:VIDEO_ID_LEN]
        if len(now_group) > 0:
            groups.append(now_group)
        groups = [group for group in groups if len(group) > 1]
        new_captions = []
        new_images = []
        new_keys = []
        for group in tqdm(groups, desc=f"Running video_llava {input_file}"):
            try:
                images = [img for _, img, _ in group]
                captions = [cap for _, _, cap in group]
                keys = [key for key, _, _ in group]
                seq_caption = self.video_llava.get_sequence_caption(images, self.video_llava_prompt)
                captions_list = self.use_llm_to_refine_caption(captions, seq_caption)
                for key, image, caption in zip(keys, images, captions_list):
                    caption = self.plural_to_singular(caption)
                    new_keys.append(key)
                    new_captions.append(caption)
                    new_images.append(image)
            except Exception as e:
                print(f"Error processing {key}: {e}")
                continue

        keys = new_keys
        images = new_images
        captions = new_captions

        torch.cuda.empty_cache()
        caption_lemmas = []
        tag_prompts = []
        text_prompt_list = []
        caption_lemma = []

        captions_text = "! ".join(captions) + " !"
        captions_text = captions_text.lower()
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

        # generate detection boxes and masks (considering not changing the original image shape, one by one)
        json_datas = []
        mask_imgs = []
        keys_new = []
        images_new = []

        for index, (origin_caption, img, tag_prompt, caption_lemma, caption, key) in enumerate(
            tqdm(
                zip(origin_captions, images, tag_prompts, caption_lemmas, captions, keys),
                desc=f"Processing {input_file}",
            )
        ):
            tag_prompt = tag_prompt.split(",")

            # if nouns count less than 3, skip
            if len(tag_prompt) < 3:
                if len(json_datas) == 0:
                    keys_new.append(key)
                    json_datas.append(
                        {
                            "origin_caption": origin_caption,
                            "caption": caption,
                            "mask": [{"value": 0, "label": "background"}],
                        }
                    )
                    mask_imgs.append(np.zeros((512, 512), dtype=np.uint8))
                    images_new.append(img)
                else:
                    keys_new.append(keys_new[-1] + "_override_" + key)
                    json_datas.append(json_datas[-1])
                    mask_imgs.append(mask_imgs[-1])
                    images_new.append(images_new[-1])
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

            # if the model does not detect any object or detects too many objects, skip
            if len(labels) == 0 or len(labels) > 8:
                if len(json_datas) == 0:
                    keys_new.append(key)
                    json_datas.append(
                        {
                            "origin_caption": origin_caption,
                            "caption": caption,
                            "mask": [{"value": 0, "label": "background"}],
                        }
                    )
                    mask_imgs.append(np.zeros((512, 512), dtype=np.uint8))
                    images_new.append(img)
                else:
                    keys_new.append(keys_new[-1] + "_override_" + key)
                    json_datas.append(json_datas[-1])
                    mask_imgs.append(mask_imgs[-1])
                    images_new.append(images_new[-1])
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
                "origin_caption": origin_caption,
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
            mask_img = self.image_tensor2pillow(mask_img)
            json_datas.append(json_data)
            mask_imgs.append(mask_img)
            keys_new.append(key)
            images_new.append(image_save)

        samples.append([keys_new, images_new, json_datas, mask_imgs])

        for sample in tqdm(samples, desc=f"Writing {output_file}"):
            for x, y, json_data, png in zip(sample[0], sample[1], sample[2], sample[3]):
                dst.write({"__key__": x, "jpg": y, "json": json_data, "png": png})
        end = time.time()
        print(f"Finished - {end-start:.0f}s")


if __name__ == "__main__":
    pipeline = Pipeline(device=device, use_blip2=True)
    shards_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    for shard_file in sorted(os.listdir(shards_dir)):
        if os.path.exists(os.path.join(output_dir, shard_file)):
            continue
        if shard_file.endswith(".tar"):
            try:
                dst = wds.TarWriter(os.path.join(output_dir, shard_file))
                pipeline.process_pipeline(
                    input_file=os.path.join(shards_dir, shard_file),
                    output_file=os.path.join(output_dir, shard_file),
                    dst=dst,
                )
                dst.close()
            except Exception as e:
                print(f"Error processing {shard_file}: {e}")
