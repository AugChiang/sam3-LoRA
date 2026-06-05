# SAM3 LoRA Text-Conditioned Segmentation

This repository trains a lightweight text-conditioned segmentation adapter for SAM3. Given an image, an object name, and a binary `.npy` mask, the pipeline learns to segment the named object without fully fine-tuning the SAM3 foundation model.

The current implementation uses:

- SAM3 image encoder and mask decoder
- HuggingFace CLIP or SigLIP text encoder
- cross-attention fusion between SAM3 text tokens and external text tokens
- HuggingFace PEFT LoRA adapters on SAM3 vision and decoder modules
- Dice + Focal loss, mixed precision training, validation mIoU and Dice

## Repository Layout

```text
.
├── main.py                    # train/validation loop, dataloader/model builders, shared predict runner
├── predict.py                 # single-image inference CLI wrapper
├── dataset.py                 # DentalInstrumentDataset and annotation/mask loading
├── utils.py                   # config loading, batching, losses, metrics, AMP helpers
├── get_config.py              # legacy standalone YAML config loader
├── models/
│   ├── __init__.py            # model exports
│   ├── sam3_base.py           # SAM3Wrapper and SAM3 prompt helpers
│   ├── cross_attn_fusion.py   # cross-attention text fusion module
│   └── txt_conditioned.py     # TextConditionedSAM3LoRA and LoRA save/load
├── configs/config.yaml        # training and inference configuration
├── checkpoints/               # SAM3 checkpoint and BPE vocab
├── outputs/                   # training checkpoints and exported adapters
└── dataset/
    ├── annotation.json        # sample metadata
    ├── images/                # input images
    └── masks/                 # binary mask .npy files
```

## Code Organization

The implementation is split so the entry points stay small:

- `main.py` wires configuration, dataset creation, model construction, training, validation, and the shared prediction routine.
- `predict.py` only parses inference arguments and calls `main.run_predict`.
- `dataset.py` owns the dataset schema, alias sampling, image transforms, and `.npy` mask loading.
- `utils.py` owns reusable helpers: seeding, YAML loading, batching, Dice/Focal loss, mask selection, mIoU/Dice metrics, and device movement.
- `models/sam3_base.py` wraps SAM3 construction and base prompt/inference helpers.
- `models/cross_attn_fusion.py` contains the CLIP/SigLIP-to-SAM3 cross-attention fusion layer.
- `models/txt_conditioned.py` combines SAM3, the external text encoder, fusion, PEFT LoRA adapters, checkpoint loading, and LoRA export.

## Requirements

Use a Python environment with CUDA-enabled PyTorch. This SAM3 checkout allocates CUDA tensors during model construction, so CPU-only training and inference are not supported unless SAM3 itself is modified.

Install the main dependencies:

```bash
pip install torch torchvision transformers peft pillow numpy pyyaml
```

SAM3 must also be importable. In this workspace, VS Code points to a sibling checkout at `/home/t000/Desktop/sam3`; otherwise add your SAM3 path to `PYTHONPATH`.

Required local assets:

```text
checkpoints/sam3.pt
checkpoints/bpe_simple_vocab_16e6.txt.gz
```

## Dataset Format

`configs/config.yaml` points to `dataset/annotation.json`, `dataset/images`, and `dataset/masks`. The annotation file may be a list or a dictionary with a top-level `samples` list:

```json
{
  "samples": [
    {
      "image": "0001.png",
      "canonical_name": "periodontal probe",
      "aliases": ["probe", "dental probe"],
      "mask": "0001_probe.npy"
    }
  ]
}
```

Images are loaded from `dataset/images/`. Masks are loaded from `dataset/masks/` as binary NumPy arrays and resized to the configured training resolution. To add a new dental instrument, add images, masks, and annotation entries with a canonical name and optional aliases.

## Configuration

Edit `configs/config.yaml` before training. Important fields:

- `device`: should be `cuda`.
- `data.resolution`: SAM3 input resolution, default `1008`.
- `text_encoder.name`: HuggingFace CLIP or SigLIP model name, default `openai/clip-vit-base-patch32`.
- `lora`: rank, alpha, dropout, and target module names.
- `training`: epochs, batch size, learning rate, validation split, mixed precision, and output directory.
- `inference`: default checkpoint and LoRA adapter paths.

The default text encoder uses `use_safetensors: false`, matching the cached PyTorch CLIP weights validated in this environment.

## How To Train

From the repository root:

```bash
python main.py --config ./configs/config.yaml train
```

`main.py` also treats no subcommand as training:

```bash
python main.py --config ./configs/config.yaml
```

Training saves outputs under `training.output_dir`, default:

```text
outputs/sam3_lora/
├── checkpoint.pt            # fusion module, config, metrics
├── lora/                    # latest exported LoRA adapters
└── best/
    ├── checkpoint.pt
    └── lora/
```

Only the fusion module and LoRA adapters are saved. The SAM3 base checkpoint remains separate in `checkpoints/sam3.pt`.

The exported LoRA adapter directories are:

```text
outputs/sam3_lora/lora/image_encoder/
outputs/sam3_lora/lora/mask_decoder/
```

The SAM3 segmentation head is intentionally not PEFT-wrapped because SAM3 runs it through an activation-checkpoint wrapper that requires the original forward signature.

## How To Run Inference

After training, run:

```bash
python predict.py --image dataset/images/0001.png --object "periodontal probe"
```

By default this reads:

```text
outputs/sam3_lora/best/checkpoint.pt
outputs/sam3_lora/best/lora/
```

Override paths if needed:

```bash
python predict.py \
  --image image.jpg \
  --object "mouth mirror" \
  --checkpoint outputs/sam3_lora/best/checkpoint.pt \
  --lora-dir outputs/sam3_lora/best/lora \
  --output mirror_mask.npy
```

The output is a binary mask saved as a `.npy` file.

You can also run inference through the `main.py` subcommand:

```bash
python main.py --config ./configs/config.yaml predict \
  --image dataset/images/0001.png \
  --object "periodontal probe" \
  --output prediction_mask.npy
```

## Validation Metrics

Each epoch reports:

- `train_loss`: Dice + Focal loss
- `val_miou`: mean intersection-over-union
- `val_dice`: Dice score

The best checkpoint is selected by validation mIoU.

## Troubleshooting

- `HuggingFace PEFT is required`: install `peft` in the active environment.
- `torch.cuda.is_available() is false`: run on a CUDA machine; this SAM3 checkout is not CPU-safe.
- HuggingFace download errors: pre-download the configured `text_encoder.name` or use a local model path.
- Missing SAM3 imports: add the SAM3 repository to `PYTHONPATH`.
- Import errors after moving files: run commands from the repository root. The scripts support both direct script execution and package-style imports.
- Empty or poor masks: check that each `.npy` mask is binary, aligned with its image, and annotated with the correct object name or aliases.
