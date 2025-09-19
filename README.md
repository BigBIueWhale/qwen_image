# Qwen Image

Qwen Image setup for my RTX 5090 with 32GB of VRAM, with `DFloat11` lossless quantization.

- Qwen Image utilizes 29.71 GB of VRAM.

- Qwen Image utilizes 30.46 GB of VRAM with [MCNL LoRA](#lora)

## Instructions

1. Tested on `Ubuntu 24.04.3` Desktop

2. Run `./create_venv.sh`

3. Run `source .venv/bin/activate`

4. Run `python3 qwen_image.py "You can shine under my umbrella"`\
Then CTRL+C to stop when it has generated enough images into [./output/](./output/)

5. Run `python3 qwen_image_edit.py "./output/000001.png" "Make the umbrella yellow"`\
Replace `"./output/000001.png"` with the name path you want to edit.\
Then CTRL+C to stop when it has generated enough edited images into [./output_edit/](./output_edit/)

## Environment

- Note- the first run requires online connection, because model downloads from Huggingface will occur.

## LoRA

- Download the LoRA manually from https://civitai.com/models/1851673/mcnl-multi-concept-nsfw-lora-qwen-image (need to login), and place the LoRA in [models/loras/qwen_MCNL_v1.0.safetensors](./models/loras/).

- Verify that your hash matches my hash:
    ```sh
    user@rtx5090:~/Desktop/qwen_image$ sha256sum ./models/loras/qwen_MCNL_v1.0.safetensors
    16c4841028615bb82c38e79756c0abad42494d85bca0daebc2939384a74d86bb  ./models/loras/qwen_MCNL_v1.0.safetensors
    user@rtx5090:~/Desktop/qwen_image$ 
    ```

- To use the LoRA, specify the `--nsfw` flag when running `qwen_image.py`. The flag is also supported for `qwen_image_edit.py` although according to online forums the LoRa is more effective for Qwen Image.
