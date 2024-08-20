# Openstory++: A Large-scale Dataset and Benchmark for Instance-aware Open-domain Visual Storytelling
[![arXiv](https://img.shields.io/badge/arXiv-2408.03695-b31b1b.svg)](https://arxiv.org/abs/2408.03695)
[![Static Badge](https://img.shields.io/badge/Dataset-Huggingface-yellow)](https://huggingface.co/datasets/MAPLE-WestLake-AIGC/OpenstoryPlusPlus)

We introduce **OpenStory++**, a large-scale open-domain dataset focusing on enabling MLLMs to perform storytelling generation tasks.

## TODOs
- [x] Release dataset
- [ ] Release dataset organization code
- [ ] Release data process pipeline code
- [ ] Release benchmark evaluation code

## Dataset Organization

1. You can use **img2dataset** to organize single image dataset

   example:

   ```bash
   img2dataset --url_list OpenstoryPlusPlus/unique_v2/part1 --input_format "parquet" --url_col "url" --output_format webdataset --output_folder "single_tar" --processes_count 12 --thread_count 12 --save_additional_columns '["png","json"]'  --image_size 512 --resize_mode="keep_ratio" --enable_wandb False
   ```