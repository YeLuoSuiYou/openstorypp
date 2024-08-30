# Openstory++: A Large-scale Dataset and Benchmark for Instance-aware Open-domain Visual Storytelling
[![arXiv](https://img.shields.io/badge/arXiv-2408.03695-b31b1b.svg)](https://arxiv.org/abs/2408.03695)
[![Static Badge](https://img.shields.io/badge/Dataset-Huggingface-yellow)](https://huggingface.co/datasets/MAPLE-WestLake-AIGC/OpenstoryPlusPlus)

We introduce **OpenStory++**, a large-scale open-domain dataset focusing on enabling MLLMs to perform storytelling generation tasks.

## TODOs
- [x] Release dataset
- [x] Release dataset organization code
- [x] Release data process pipeline code
- [ ] Release video preprocessing code
- [ ] Release benchmark evaluation code

## Dataset Organization

1. You can use **img2dataset** to organize single image dataset

   example:

   ```bash
   img2dataset --url_list OpenstoryPlusPlus/unique_v2/part1 --input_format "parquet" --url_col "url" --output_format webdataset --output_folder "single_tar" --processes_count 12 --thread_count 12 --save_additional_columns '["png","json"]'  --image_size 512 --resize_mode="keep_ratio" --enable_wandb False
   ```

2. To organize the story dataset as described in the paper, you can use `utilis\download_videos.py` to download videos from YouTube and extract frames for the dataset. Additionally, you can utilize `utilis\organize_story.py` to properly structure the story dataset.

## Dataset Process Pipeline

1. You can use `single_pipeline.py` to get images with instance-level annotation.

   Hint: the input image should in `webdataset` format, the source image should in the "jpg" key.
