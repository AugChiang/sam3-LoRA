import re
import torch
import numpy as np
from PIL import Image
from typing import Union, List

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


VOWEL_SOUNDS_EXCEPTIONS = {
    "hour", "honest", "honor", "heir", "herb"
}

CONSONANT_SOUND_EXCEPTIONS = {
    "university", "unicorn", "user", "european", "one", "once"
}

ARTICLES = {"a", "an"}

class SAM3Wrapper:
    def __init__(self, config, device:str = "cpu"):
        self.config = config
        self.model_type = self.config['model_type']
        self.checkpoint_path = self.config['checkpoint_path']
        self.bpe_path = self.config['bpe_path']
        self.confidence_threshold = self.config['confidence_threshold']
        self.device = device
        self.generic_prompt = self.config['prompt']

        self._create_model()

    def _create_model(self):
        self.model = build_sam3_image_model(
                bpe_path=self.bpe_path,
                device=self.device,
                eval_mode=True,
                checkpoint_path=self.checkpoint_path,
                load_from_HF=False,
                enable_inst_interactivity=False,
            )
        # or simply:
        # model = build_sam3_image_model(
        #     device="cuda",
        #     checkpoint_path="./checkpoints/sam3.pt",
        # )
        self.sam3_processor = Sam3Processor(
            model = self.model, 
            device = self.device, 
            confidence_threshold = self.confidence_threshold
        )
    
    def pred(self, img: np.ndarray, txt_prompt : str = None, **kwargs)->dict:
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        if txt_prompt is None or len(txt_prompt.strip()) == 0:
            txt_prompt = self.generic_prompt

        if "sam2" in self.model_type:
            return self.sam3_processor.predict(
                        images_pil = [img], 
                        texts_prompt = [txt_prompt],
                        box_threshold = 0.3,
                        text_threshold = 0.25,
                        **kwargs
                    )
        elif "sam3" in self.model_type:
            _img = self.sam3_processor.set_image(img)
            self.sam3_processor.reset_all_prompts(_img)
            return self.sam3_processor.set_text_prompt(prompt=txt_prompt, state=_img)

    def pred_mask_by_points(
            self, 
            img: np.ndarray, 
            pixel_uv: Union[list, np.ndarray]
        ):
        """
        Given pixel coordinates as prompt, predict the masks.

        Args:
            img: input image (will automatically convert to `PIL.Image`)
            pixel_uv: pixel coordinates, (u,v).

        Returns:
            masks (List[torch.Tensor]) : list of masks, each mask shape = [1,H,W].
        """
        if isinstance(img, np.ndarray):
            img_pil = Image.fromarray(img)
        elif isinstance(img, str):
            img_pil = Image.open(img).convert("RGB")
        inference_state = self.sam3_processor.set_image(img_pil)
        self.sam3_processor.reset_all_prompts(inference_state)
        point_coords = np.array(pixel_uv, dtype=np.float32)    # shape = (N_points, 2)
        foreground = np.array([1])
        masks = []
        # SAM 3 takes the set of points as prompt. i.e. the resulting must must contains the point set.
        for i, pt in enumerate(point_coords):
            mask, _, _ = self.model.predict_inst(
                inference_state,
                point_coords= pt[None], # requires [[u,v]]
                point_labels= foreground, # 0: background; 1: foreground
                multimask_output=False, # if True, it also catches obj nearby
            )
            # print("mask shape: ", mask.shape) # (1,H,W)
            if isinstance(mask, torch.Tensor):
                mask = mask.detach().cpu().numpy()
            masks.append(mask[0].astype(np.bool)) # [1,H,W]
        return masks

    def pred_mask_per_obj(self, img: np.ndarray, obj_list:list)->List[torch.Tensor]:
        """
        Given a list of objects, predict corresponding mask one by one.

        Args:
            img: scene image, shape = [H,W,3]
            obj_list: list of strings of objects.

        Returns:
            masks (List[torch.Tensor]) : list of masks, each mask shape = [1,B,H,W].
        """
        if isinstance(img, np.ndarray):
            img_pil = Image.fromarray(img)
        elif isinstance(img, str):
            img_pil = Image.open(img).convert("RGB")
        inference_state = self.sam3_processor.set_image(img_pil)
        self.sam3_processor.reset_all_prompts(inference_state)
        masks = []
        for obj in obj_list:
            prompt = add_indefinite_article(obj)
            output = self.sam3_processor.set_text_prompt(state=inference_state, prompt=prompt)
            # [N,B,H,W], torch.Tensor
            if output["masks"].shape[0] == 0:
                print(f"[SegmentationModel] Object: '{prompt}' mask is NOT generated. Skipping...")
                continue
            masks.append(output["masks"].detach().cpu())
        if len(masks) == 0:
            print("[SegmentationModel] None of mask is generated.")
            return None
        return masks

def has_article(text: str) -> bool:
    """
    Check if a string already starts with 'a' or 'an'.
    """
    if not text:
        return False
    first_word = text.strip().lower().split()[0]
    return first_word in ARTICLES


def add_indefinite_article(word: str) -> str:
    """
    Add 'a' or 'an' only if not already present.
    """

    if not word:
        return word

    word = word.strip()

    # Skip if already has article
    if has_article(word):
        return word

    w = word.lower()
    first_word = w.split()[0]

    # Exceptions first
    if first_word in VOWEL_SOUNDS_EXCEPTIONS:
        return f"an {word}"

    if first_word in CONSONANT_SOUND_EXCEPTIONS:
        return f"a {word}"

    # Default phonetic heuristic
    if re.match(r"^[aeiou]", first_word):
        return f"an {word}"
    else:
        return f"a {word}"


def add_articles_to_list(objects: List[str]) -> List[str]:
    """
    Apply indefinite article only if missing.
    """
    return [add_indefinite_article(obj) for obj in objects]