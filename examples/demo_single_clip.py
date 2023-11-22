import os
import yaml
import hcaptcha_challenger as solver

from pathlib import Path
from hcaptcha_challenger import ZeroShotImageClassifier, DataLake, ModelHub, register_pipline
from PIL import Image
from loguru import logger
from hcaptcha_challenger.onnx.clip import MossCLIP

# solver.install(upgrade=True, clip=True)

assets_dir = Path(__file__).parent.parent.joinpath("assets")
local_objects_path = Path(__file__).parent.joinpath("local_objects.yaml")
images_dir = assets_dir.joinpath("image_label_binary", "suggest")

suggest_prompts = ["giraffe", "raccoon", "dog cake", "hamster"]


def prelude_self_supervised_config():
    modelhub = ModelHub.from_github_repo()
    modelhub.parse_objects()
    clip_model = register_pipline(modelhub)
    return modelhub, clip_model

def get_test_images() -> list[Path]:
    images = []
    for image_name in os.listdir(images_dir):
        image_path = images_dir.joinpath(image_name)
        if image_path.is_file():
            images.append(image_path)

    return images

def get_grid_images() -> list[Path]:
    return [img for img in get_test_images() if not 'suggest' in img.name]

def get_suggest_image() -> Path:
    return [img for img in get_test_images() if 'suggest' in img.name][0]

def ranking_clip(image_path: Path, prompts: list[str], classifier: solver.BinaryClassifier) -> str:
    classifier.model_name = classifier.modelhub.DEFAULT_CLIP_VISUAL_MODEL

    for prompt in prompts:
        yes_prompts = [DataLake.PREMISED_YES.format(prompt)]
        bad_prompts = [DataLake.PREMISED_BAD.format(_prompt) for _prompt in prompts if not _prompt == prompt]
        candidace_prompts = yes_prompts + bad_prompts
        tool = ZeroShotImageClassifier(yes_prompts, candidace_prompts)
        try:
            if not isinstance(image_path, Path):
                raise TypeError(
                    "Please pass in the pathlib.Path object, "
                    "you don't need to set it specifically for bytes in advance. "
                    f"- type={type(image_path)}"
                )
            if not image_path.exists():
                raise FileNotFoundError(f"ChallengeImage not found - path={image_path}")

            results = tool(classifier.clip_model, image=Image.open(image_path))
            trusted = results[0]["label"] in tool.positive_labels
            if trusted: return prompt
        except Exception as err:
            logger.debug(str(err), prompt)

def parse_clip_objects():
    local_objects: dict = yaml.safe_load(local_objects_path.read_text(encoding="utf8"))
    clip_objects: dict[str, DataLake] = local_objects.get("clip", {})
    if clip_objects:
        for prompt, serialized_binary in clip_objects.items():
            clip_objects[prompt] = DataLake.from_serialized(serialized_binary)
    return clip_objects

def local_clip(prompt: str, image_path: Path, clip_model: MossCLIP) -> list[int]:
    clip_objects = parse_clip_objects()
    dl = clip_objects[prompt]
    tool = ZeroShotImageClassifier.from_datalake(dl)

    try:
        if not isinstance(image_path, Path):
            raise TypeError(
                "Please pass in the pathlib.Path object, "
                "you don't need to set it specifically for bytes in advance. "
                f"- type={type(image_path)}"
            )
        if not image_path.exists():
            raise FileNotFoundError(f"ChallengeImage not found - path={image_path}")

        results = tool(clip_model, image=Image.open(image_path))
        trusted = results[0]["label"] in tool.positive_labels
        return trusted
    except Exception as err:
        logger.debug(str(err), prompt)


def demo():
    modelhub, clip_model = prelude_self_supervised_config()
    classifier = solver.BinaryClassifier(modelhub=modelhub, clip_model=clip_model)
    clip_model: MossCLIP = classifier.clip_model

    suggest_img_path = get_suggest_image()
    target_prompt = ranking_clip(suggest_img_path, suggest_prompts, classifier)
    print(f'{target_prompt=}')
    image_paths = get_grid_images()
    for image_path in image_paths:
        result = local_clip(prompt=target_prompt, image_path=image_path, clip_model=clip_model)
        print(f'{image_path.name=} - {result=}')
    
    if not clip_model:
        modelhub.unplug()

if __name__ == "__main__":
    demo()