import hashlib
import os
import urllib
import warnings
from typing import Union, List
from pkg_resources import packaging

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from tqdm import tqdm
import numpy as np

from .build_model import build_model
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


if packaging.version.parse(torch.__version__) < packaging.version.parse("1.7.1"):
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")


__all__ = ["available_models", "load", "tokenize", "encode_text_with_prompt_ensemble", "encode_text_without_prompt_ensemble",
           "get_similarity_map", "clip_feature_surgery", "similarity_map_to_points","encode_text_with_prompt_ensemble_ad",'encode_text_with_prompt_ensemble_ad_diff',"encode_text_with_prompt_ensemble_ad_H",
           "encode_text_with_prompt_ensemble_ad_diff_word"
]
_tokenizer = _Tokenizer()

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
    "CS-RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "CS-RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "CS-RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "CS-RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "CS-RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "CS-ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "CS-ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "CS-ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "CS-ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
    "CS-ViT-H/14": "/home/zjp/.cache/clip/ViT-H-14.pt",
}


obj_list = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
    "object",
    "candle",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni",
    "pcb",
    "pipe fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "capsules",
]

mvtec_obj_list = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

visa_obj_list = [
    "candle",
    "cashew",
    "chewinggum",
    "fryum",
    "pipe fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "capsules",
]


mpdd_obj_list = [
"bracket_black",
"bracket_brown",
"bracket_white",
"connector",
"metal_plate",
"tubes",

]



mvtec_anomaly_detail = {
    "carpet": "different color,cut,hole,metal contamination,thread",
    "grid": "bent,broken,glue,metal contamination,thread",
    "leather": "different color,cut,fold,glue,poke",
    "tile": "crack,gule strip,gray stroke,oil,rough",
    "wood": "different color,combined,hole,liquid,scratch",
    "bottle": "large broken,small broken,contamination",
    "cable": "bent wire,cable swap,combined,cut inner insulation,cut outer insulation,missing cable,missing wire,poke insulation",
    "capsule": "crack,faulty imprint,poke,scratch,squeeze",
    "hazelnut": "crack,cut,hole,print",
    "metal nut": "bent,different color,flip,scratch",
    "pill": "different color,combined,contamination,crack,faulty imprint,pill type,scratch",
    "screw": "manipulated front,scratch head,scratch neck,thread side,thread top",
    "toothbrush": "defective",
    "transistor": "bent lead,cut lead,damaged case,misplaced",
    "zipper": "broken teeth,combined,fabric border,fabric interior,rough,split teeth,squeezed teeth",
}

visa_anomaly_detail = {
    "candle": "chunk of wax missing,foreign particals on candle,different colour spot,damaged corner of packaging,weird candle wick,other,extra wax in candle,wax melded out of the candle",
    "capsules": "bubble,discolor,scratch,leak,misshape",
    "cashew": "burnt,corner or edge breakage,different colour spot,middle breakage,small holes,same colour spot,small scratches,stuck together",
    "chewinggum": "chunk of gum missing,scratches,small cracks,corner missing,similar colour spot",
    "fryum": "burnt,similar colour spot,corner or edge breakage,middle breakage,small scratches,different colour spot,fryum stuck together,other",
    "macaroni1": "chip around edge and corner,small scratches,small cracks,different colour spot,middle breakage,similar colour spot",
    "macaroni2": "breakage down the middle,small scratches,color spot similar to the object,different color spot,small chip around edge,small cracks,other",
    "pcb1": "bent,melt,scratch,missing",
    "pcb2": "bent,melt,scratch,missing",
    "pcb3": "bent,melt,scratch,missing",
    "pcb4": "burnt,scratch,dirt,damage,extra,missing,wrong place",
    "pipe fryum": "burnt,corner and edge breakage,different colour spot,middle breakage,small scratches,small cracks,similar colour spot,stuck together",
}

prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect', '{} without damage']
prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']

prompt_abnormal_detail = {}
for cls_name in mvtec_obj_list:
    prompt_abnormal_detail[cls_name] = prompt_abnormal + ['normal {} ' + 'with {}'.format(x) for x in mvtec_anomaly_detail[cls_name].split(',')]

for cls_name in visa_obj_list:
    prompt_abnormal_detail[cls_name] = prompt_abnormal + ['abnormal {} ' + 'with {}'.format(x) for x in visa_anomaly_detail[cls_name].split(',')]



def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize((n_px, n_px), interpolation=BICUBIC),
        #CenterCrop(n_px), # rm center crop to explain whole image
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit: bool = False, download_root: str = None):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).

    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"

    Returns
    -------
    model : torch.nn.Module
        The CLIP model

    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """

    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    with open(model_path, 'rb') as opened_file:
        try:
            # loading JIT archive
            model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
            state_dict = None
        except RuntimeError:
            # loading saved state dict
            if jit:
                warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                jit = False
            state_dict = torch.load(opened_file, map_location="cpu")

    if not jit:
        model = build_model(name, state_dict or model.state_dict()).to(device)
        if str(device) == "cpu":
            model.float()
        return model, _transform(model.visual.input_resolution)

    # patch the device names
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model, _transform(model.input_resolution.item())


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def encode_text_with_prompt_ensemble(model, texts, device, prompt_templates=None):

    # using default prompt templates for ImageNet
    if prompt_templates == None:
        prompt_templates = ['a bad photo of a {}.',
                            'a photo of many {}.',
                            'a sculpture of a {}.',
                            'a photo of the hard to see {}.',
                            'a low resolution photo of the {}.',
                            'a rendering of a {}.', 'graffiti of a {}.',
                            'a bad photo of the {}.', 'a cropped photo of the {}.',
                            'a tattoo of a {}.', 'the embroidered {}.',
                            'a photo of a hard to see {}.',
                            'a bright photo of a {}.',
                            'a photo of a clean {}.',
                            'a photo of a dirty {}.',
                            'a dark photo of the {}.',
                            'a drawing of a {}.',
                            'a photo of my {}.',
                            'the plastic {}.',
                            'a photo of the cool {}.',
                            'a close-up photo of a {}.',
                            'a black and white photo of the {}.',
                            'a painting of the {}.', 'a painting of a {}.',
                            'a pixelated photo of the {}.', 'a sculpture of the {}.', 'a bright photo of the {}.',
                            'a cropped photo of a {}.', 'a plastic {}.', 'a photo of the dirty {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.',
                            'a photo of the {}.', 'a good photo of the {}.', 'a rendering of the {}.', 'a {} in a video game.', 'a photo of one {}.', 'a doodle of a {}.',
                            'a close-up photo of the {}.', 'a photo of a {}.', 'the origami {}.', 'the {} in a video game.', 'a sketch of a {}.', 'a doodle of the {}.',
                            'a origami {}.', 'a low resolution photo of a {}.', 'the toy {}.', 'a rendition of the {}.', 'a photo of the clean {}.', 'a photo of a large {}.',
                            'a rendition of a {}.', 'a photo of a nice {}.', 'a photo of a weird {}.', 'a blurry photo of a {}.', 'a cartoon {}.', 'art of a {}.',
                            'a sketch of the {}.', 'a embroidered {}.', 'a pixelated photo of a {}.', 'itap of the {}.', 'a jpeg corrupted photo of the {}.',
                            'a good photo of a {}.', 'a plushie {}.', 'a photo of the nice {}.', 'a photo of the small {}.', 'a photo of the weird {}.', 'the cartoon {}.',
                            'art of the {}.', 'a drawing of the {}.', 'a photo of the large {}.', 'a black and white photo of a {}.', 'the plushie {}.', 'a dark photo of a {}.',
                            'itap of a {}.', 'graffiti of the {}.', 'a toy {}.', 'itap of my {}.', 'a photo of a cool {}.', 'a photo of a small {}.', 'a tattoo of the {}.',
                            'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.', 'this is the {} in the scene.',
                            'this is one {} in the scene.']

    text_features = []
    for t in texts:
        prompted_t = [template.format(t) for template in prompt_templates]
        prompted_t = tokenize(prompted_t).to(device)
        class_embeddings = model.encode_text(prompted_t)
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        text_features.append(class_embedding)
    text_features = torch.stack(text_features, dim=1).to(device).t()

    return text_features



def encode_text_with_prompt_ensemble_ad(model, model_clip, objs, tokenizer, device, prompt_templates=None):

    objs= mvtec_obj_list    # 换objects 为 mvtec_obj_list    # 80.7 |       84.2 |                                    #
    # objs= visa_obj_list   #  + visa_abnormal prompt  78.7 |       83.7 |     mvtec_abnormal prompt 80.2 |       85.9 |      79.4 |       93.6 |

    # objs = ['object']   #  + visa_abnormal prompt  79.5 |       84   |     mvtec_abnormal prompt  79.2 |       87   |    #78.8 |       93.5 |


    # using default prompt templates for ImageNet
    prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect',
                     '{} without damage']

    # prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']
    #
    prompt_abnormal = [ 'broken {}', '{} with flaw', '{} with defect', '{} with damage', '{} with crack',
                       '{} with hole', '{} with scratch', #'{} with residue'     添加
                       ]  #  vit-l-14   mvtec dataset

    # prompt_abnormal = [
    #     '{} with scratch',
    #     'scratched {}',
    #     '{} with color spots',
    #     '{} with breakage',
    #     'broken {}',
    #     '{} with cracks',
    #     'burnt {}',
    #     'melted {}',
    #     'bent {}',
    #     '{} with missing parts',
    #     'damaged {}',
    #     '{} with defects',
    #     '{} with holes',
    #     '{} stuck together'
    # ]


    # prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage','{} with crack','{} with hole', '{} with residue'
    #                    ]  #


    prompt_state = [prompt_normal, prompt_abnormal]
    prompt_templates = ['a bad photo of a {}.', 'a low resolution photo of the {}.', 'a bad photo of the {}.',
                        'a cropped photo of the {}.', 'a bright photo of a {}.', 'a dark photo of the {}.',
                        'a photo of my {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.',
                        'a black and white photo of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.',
                        'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.',
                        'a good photo of the {}.', 'a photo of one {}.', 'a close-up photo of the {}.',
                        'a photo of a {}.', 'a low resolution photo of a {}.', 'a photo of a large {}.',
                        'a blurry photo of a {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.',
                        'a photo of the small {}.', 'a photo of the large {}.', 'a black and white photo of a {}.',
                        'a dark photo of a {}.', 'a photo of a cool {}.', 'a photo of a small {}.',
                        # 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.',
                        # 'this is the {} in the scene.', 'this is one {} in the scene.'
                        ]

                        # 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.',
                        # 'this is the {} in the scene.', 'this is one {} in the scene.']



    text_prompts = {}
    for obj in objs:
        text_features = []
        for i in range(len(prompt_state)):
            prompted_state = [state.format(obj) for state in prompt_state[i]]
            prompted_sentence = []
            for s in prompted_state:
                for template in prompt_templates:
                    prompted_sentence.append(template.format(s))

                   # self.text_encoder  clip
            prompted_sentence = tokenizer(prompted_sentence).to(device)  #tensor  (210,77)


            with torch.no_grad():
                class_embeddings = model.encode_text(prompted_sentence)   #[210.768]
            # self.text_encoder  clip

            # self.text_encoder  stable diffusion
            # prompted_sentence = tokenizer(prompted_sentence, return_tensors="pt", padding="max_length", truncation=True, ).to(
            #     device)
            # with torch.no_grad():
            #     class_embeddings = model.text_encoder(**prompted_sentence)[1]   #[210.768]
            #     class_embeddings = class_embeddings @ model_clip.text_projection   #class_embeddings = torch.matmul(class_embeddings, model_clip.text_projection)

                # self.text_encoder  stable diffusion

            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)  #[210.768]
            class_embedding = class_embeddings.mean(dim=0)  #(768,)
            class_embedding /= class_embedding.norm() #(768,)
            text_features.append(class_embedding)

        text_features = torch.stack(text_features, dim=1).to(device)
        # text_prompts[obj] = text_features
        text_prompts['object'] = text_features

    # ------------  异常原型文本特征构建 ----------------------
    for obj in objs:
        abnormal_anchors = []
        for p in prompt_abnormal:
            prompted_abn = [template.format(p.format(obj)) for template in prompt_templates]
            tokens = tokenizer(prompted_abn).to(device)
            with torch.no_grad():
                embeddings = model.encode_text(tokens)  # [N, 768]
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            anchor = embeddings.mean(dim=0)  # [768]
            anchor = anchor / anchor.norm()
            abnormal_anchors.append(anchor)

        abnormal_anchors = torch.stack(abnormal_anchors, dim=0)

    return text_prompts, abnormal_anchors, text_prompts['object'][:,0].unsqueeze(0)


def encode_text_with_prompt_ensemble_ad_H(model, objs, tokenizer, device, prompt_templates=None):

    # using default prompt templates for ImageNet
    prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect',
                     '{} without damage']

    # prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']

    prompt_abnormal = ['broken {}', '{} with flaw', '{} with defect', '{} with damage','{} with crack','{} with hole',
                       '{} with residue']


    prompt_state = [prompt_normal, prompt_abnormal]
    prompt_templates = ['a bad photo of a {}.', 'a low resolution photo of the {}.', 'a bad photo of the {}.',
                        'a cropped photo of the {}.', 'a bright photo of a {}.', 'a dark photo of the {}.',
                        'a photo of my {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.',
                        'a black and white photo of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.',
                        'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.',
                        'a good photo of the {}.', 'a photo of one {}.', 'a close-up photo of the {}.',
                        'a photo of a {}.', 'a low resolution photo of a {}.', 'a photo of a large {}.',
                        'a blurry photo of a {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.',
                        'a photo of the small {}.', 'a photo of the large {}.', 'a black and white photo of a {}.',
                        'a dark photo of a {}.', 'a photo of a cool {}.', 'a photo of a small {}.',
                        ]

                        # 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.',
                        # 'this is the {} in the scene.', 'this is one {} in the scene.']

    text_prompts = {}
    for obj in objs:
        text_features = []
        for i in range(len(prompt_state)):
            prompted_state = [state.format(obj) for state in prompt_state[i]]
            prompted_sentence = []
            for s in prompted_state:
                for template in prompt_templates:
                    prompted_sentence.append(template.format(s))

            # tokenizer--> stalbe text tokenizer
            inputs = tokenizer(prompted_sentence, return_tensors="pt", padding="max_length", truncation=True, ).to(
                device)
            with torch.no_grad():
                class_embeddings = model(**inputs)[1]

            # prompted_sentence = tokenizer(prompted_sentence).to(device)
            # # inputs = tokenizer(prompted_sentence, return_tensors='pt')
            # # prompted_sentence = {k: v.to(device) for k, v in inputs.items()}
            # with torch.no_grad():
            #     class_embeddings = model.encode_text(prompted_sentence)

                # prompted_sentence = tokenizer(prompted_sentence).to(device)
            # inputs = tokenizer(prompted_sentence, return_tensors='pt')
            # prompted_sentence = {k: v.to(device) for k, v in inputs.items()}
            # with torch.no_grad():
            #     class_embeddings = model.encode_text(prompted_sentence)   #[210.768]
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)  #[210.768]
            class_embedding = class_embeddings.mean(dim=0)  #(768,)
            class_embedding /= class_embedding.norm() #(768,)
            text_features.append(class_embedding)

        text_features = torch.stack(text_features, dim=1).to(device)
        text_prompts[obj] = text_features

    return text_prompts



def encode_text_with_prompt_ensemble_ad_diff(model, objs, tokenizer, device, prompt_templates=None):

    # using default prompt templates for ImageNet
    # prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect',
    #                  '{} without damage']

    # prompt_normal = ['{}', 'flawless {}', '{} without flaw', '{} without defect',
    #                  '{} without damage']
    # prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']

    prompt_normal = ['{}',' perfect {} ', 'flawless {} ', ' clean {} ', 'unblemished {}', ' normal {} ']
    prompt_abnormal = [' {} with broken', '{} with crack', '{} with residue', '{} with damage','{} with hole','{} with scratch",']


    prompt_state = [prompt_normal, prompt_abnormal]
    # prompt_templates = ['a bad photo of a {}.', 'a low resolution photo of the {}.', 'a bad photo of the {}.',
    #                     'a cropped photo of the {}.', 'a bright photo of a {}.', 'a dark photo of the {}.',
    #                     'a photo of my {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.',
    #                     'a black and white photo of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.',
    #                     'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.',
    #                     'a good photo of the {}.', 'a photo of one {}.', 'a close-up photo of the {}.',
    #                     'a photo of a {}.', 'a low resolution photo of a {}.', 'a photo of a large {}.',
    #                     'a blurry photo of a {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.',
    #                     'a photo of the small {}.', 'a photo of the large {}.', 'a black and white photo of a {}.',
    #                     'a dark photo of a {}.', 'a photo of a cool {}.', 'a photo of a small {}.',
    #                     'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.',
    #                     'this is the {} in the scene.', 'this is one {} in the scene.']

    prompt_templates = [  'a  photo of a {}.', #
                        'a low resolution photo of the {}.',
                        'a bad photo of the {}.',
                        'a cropped photo of the {}.',
                        'a bright photo of a {}.', 'a dark photo of the {}.',
                         'a photo of the cool {}.', 'a close-up photo of a {}.',
                        'a black and white photo of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.',
                        'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.',
                        'a good photo of the {}.',  'a close-up photo of the {}.',
                        'a low resolution photo of a {}.',
                        'a blurry photo of a {}.', 'a jpeg corrupted photo of the {}.',
                        'a photo of the small {}.', 'a photo of the large {}.', 'a black and white photo of a {}.',
                        'a dark photo of a {}.', 'a photo of a cool {}.',
                        ]

    text_prompts = {}
    for obj in objs:
        text_features = []
        for i in range(len(prompt_state)):
            prompted_state = [state.format(obj) for state in prompt_state[i]]
            prompted_sentence = []
            for s in prompted_state:
                for template in prompt_templates:
                    prompted_sentence.append(template.format(s))
            inputs = tokenizer(prompted_sentence, return_tensors="pt",  padding="max_length", truncation=True,).to(device)
            # inputs = tokenizer(prompted_sentence, return_tensors='pt')
            # prompted_sentence = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)[0]    #  outputs  0 [175,77,1024]  1 [175,1024]
            # class_embeddings = outputs.last_hidden_state[:, 0, :]
            # class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            # class_embedding = class_embeddings.mean(dim=0)
            # class_embedding /= class_embedding.norm()
            # text_features.append(class_embedding)

            # 整个句子的token序列
            outputs /= outputs.norm(dim=-2, keepdim=True)
            outputs = outputs.mean(dim=0)
            outputs /= outputs.norm()
            outputs = outputs.unsqueeze(0)
            text_features.append(outputs)   #[1,77,1024]


            # cls token  不好
            # cls_token = model(**inputs)[1]
            # cls_token /= cls_token.norm(dim=-1, keepdim=True)
            # cls_token = cls_token.mean(dim=0)
            # cls_token /= cls_token.norm()
            # cls_token = cls_token.unsqueeze(0).unsqueeze(0)   #[1,1,1024]    nan |        nan |
            # text_features.append(cls_token)

            prompted_sentence = 'a photo of object with broken crack residue damage  hole scratch'   # residue 这个词有问题
            inputs = tokenizer(prompted_sentence, return_tensors="pt", padding="max_length", truncation=True, ).to(
                device)

            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

            state_words = ["broken", "scratch", "damage",'crack','residue']
            object_words = ["object"]

            state_ids = []
            for word in state_words:
                state_ids += [i for i, tok in enumerate(tokens) if word in tok]

            object_ids = []
            for word in object_words:
                object_ids += [i for i, tok in enumerate(tokens) if word in tok]


        # text_features

        text_features = torch.stack(text_features, dim=0).to(device)
        # text_features = torch.stack(text_features, dim=1).to(device)
        text_prompts[obj] = text_features

    return text_prompts

    # return text_prompts, state_ids, object_ids


def encode_text_with_prompt_ensemble_ad_diff_word(model, objs, tokenizer, device, prompt_templates=None):

    # using default prompt templates for ImageNet
    # prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect',
    #                  '{} without damage']

    # prompt_normal = ['{}', 'flawless {}', '{} without flaw', '{} without defect',
    #                  '{} without damage']
    # prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']

    # prompt_normal = ['{}',' perfect {} ', 'flawless {} ', ' clean {} ', 'unblemished {}', ' normal {} ']
    # prompt_abnormal = [' {} with broken', '{} with crack', '{} with residue', '{} with damage','{} with hole','{} with scratch",']


    # prompt_state = [prompt_normal, prompt_abnormal]
    # prompt_templates = ['a bad photo of a {}.', 'a low resolution photo of the {}.', 'a bad photo of the {}.',
    #                     'a cropped photo of the {}.', 'a bright photo of a {}.', 'a dark photo of the {}.',
    #                     'a photo of my {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.',
    #                     'a black and white photo of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.',
    #                     'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.',
    #                     'a good photo of the {}.', 'a photo of one {}.', 'a close-up photo of the {}.',
    #                     'a photo of a {}.', 'a low resolution photo of a {}.', 'a photo of a large {}.',
    #                     'a blurry photo of a {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.',
    #                     'a photo of the small {}.', 'a photo of the large {}.', 'a black and white photo of a {}.',
    #                     'a dark photo of a {}.', 'a photo of a cool {}.', 'a photo of a small {}.',
    #                     'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.',
    #                     'this is the {} in the scene.', 'this is one {} in the scene.']

    # prompt_templates = [  'a  photo of a {}.', #
    #                     'a low resolution photo of the {}.',
    #                     'a bad photo of the {}.',
    #                     'a cropped photo of the {}.',
    #                     'a bright photo of a {}.', 'a dark photo of the {}.',
    #                      'a photo of the cool {}.', 'a close-up photo of a {}.',
    #                     'a black and white photo of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.',
    #                     'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.',
    #                     'a good photo of the {}.',  'a close-up photo of the {}.',
    #                     'a low resolution photo of a {}.',
    #                     'a blurry photo of a {}.', 'a jpeg corrupted photo of the {}.',
    #                     'a photo of the small {}.', 'a photo of the large {}.', 'a black and white photo of a {}.',
    #                     'a dark photo of a {}.', 'a photo of a cool {}.',
    #                     ]
    #
    text_prompts = {}
    # for obj in objs:
    #     text_features = []
    #     for i in range(len(prompt_state)):
    #         prompted_state = [state.format(obj) for state in prompt_state[i]]
    #         prompted_sentence = []
    #         for s in prompted_state:
    #             for template in prompt_templates:
    #                 prompted_sentence.append(template.format(s))
    #         inputs = tokenizer(prompted_sentence, return_tensors="pt",  padding="max_length", truncation=True,).to(device)
    #         # inputs = tokenizer(prompted_sentence, return_tensors='pt')
    #         # prompted_sentence = {k: v.to(device) for k, v in inputs.items()}
    #
    #         outputs = model(**inputs)[0]    #  outputs  0 [175,77,1024]  1 [175,1024]
    #         # class_embeddings = outputs.last_hidden_state[:, 0, :]
    #         # class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
    #         # class_embedding = class_embeddings.mean(dim=0)
    #         # class_embedding /= class_embedding.norm()
    #         # text_features.append(class_embedding)
    #
    #         # 整个句子的token序列
    #         outputs /= outputs.norm(dim=-2, keepdim=True)
    #         outputs = outputs.mean(dim=0)
    #         outputs /= outputs.norm()
    #         outputs = outputs.unsqueeze(0)
    #         text_features.append(outputs)   #[1,77,1024]


            # cls token  不好
            # cls_token = model(**inputs)[1]
            # cls_token /= cls_token.norm(dim=-1, keepdim=True)
            # cls_token = cls_token.mean(dim=0)
            # cls_token /= cls_token.norm()
            # cls_token = cls_token.unsqueeze(0).unsqueeze(0)   #[1,1,1024]    nan |        nan |
            # text_features.append(cls_token)

    # prompted_sentence = 'a photo of object with broken crack residue damage  hole scratch  perfect  flawless unblemished clear  '   # residue 这个词有问题
    # prompted_sentence = 'a photo of {} with broken crack  damage  hole scratch   flawless unblemished clear  '.format(objs[0])   # residue 这个词有问题

    prompted_sentence = 'a photo of {} with broken crack  damage  hole scratch flawless unblemished clear'.format(objs[0])   # residue 这个词有问题


    inputs = tokenizer(prompted_sentence, return_tensors="pt", padding="max_length", truncation=True, ).to(
        device)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    state_words = ["broken", "scratch", "damage",'crack','residue','hole']
    object_words = [objs[0]]
    norm_words=[ "perfect" ,"flawless" ,"unblemished" ,"clear"]

    state_ids = []
    for word in state_words:
        state_ids += [i for i, tok in enumerate(tokens) if word in tok]

    state_norm_ids = []
    for word in norm_words:
        state_norm_ids += [i for i, tok in enumerate(tokens) if word in tok]

    object_ids = []
    for word in object_words:
        object_ids += [i for i, tok in enumerate(tokens) if word in tok]

    text_features = []
    outputs = model(**inputs)[0]


    # text_features
    text_features.append(outputs)
    text_features = torch.stack(text_features, dim=0).to(device)
    # text_features = torch.stack(text_features, dim=1).to(device)
    text_prompts["object"] = text_features

    return text_prompts, state_ids, object_ids,state_norm_ids



def encode_text_without_prompt_ensemble(model, texts, device, prompt_templates=None):

    # using default prompt templates for ImageNet
    # if prompt_templates == None:
    #     prompt_templates = ['a bad photo of a {}.', 'a photo of many {}.', 'a sculpture of a {}.', 'a photo of the hard to see {}.', 'a low resolution photo of the {}.', 'a rendering of a {}.', 'graffiti of a {}.', 'a bad photo of the {}.', 'a cropped photo of the {}.', 'a tattoo of a {}.', 'the embroidered {}.', 'a photo of a hard to see {}.', 'a bright photo of a {}.', 'a photo of a clean {}.', 'a photo of a dirty {}.', 'a dark photo of the {}.', 'a drawing of a {}.', 'a photo of my {}.', 'the plastic {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a painting of the {}.', 'a painting of a {}.', 'a pixelated photo of the {}.', 'a sculpture of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.', 'a plastic {}.', 'a photo of the dirty {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.', 'a rendering of the {}.', 'a {} in a video game.', 'a photo of one {}.', 'a doodle of a {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'the origami {}.', 'the {} in a video game.', 'a sketch of a {}.', 'a doodle of the {}.', 'a origami {}.', 'a low resolution photo of a {}.', 'the toy {}.', 'a rendition of the {}.', 'a photo of the clean {}.', 'a photo of a large {}.', 'a rendition of a {}.', 'a photo of a nice {}.', 'a photo of a weird {}.', 'a blurry photo of a {}.', 'a cartoon {}.', 'art of a {}.', 'a sketch of the {}.', 'a embroidered {}.', 'a pixelated photo of a {}.', 'itap of the {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a plushie {}.', 'a photo of the nice {}.', 'a photo of the small {}.', 'a photo of the weird {}.', 'the cartoon {}.', 'art of the {}.', 'a drawing of the {}.', 'a photo of the large {}.', 'a black and white photo of a {}.', 'the plushie {}.', 'a dark photo of a {}.', 'itap of a {}.', 'graffiti of the {}.', 'a toy {}.', 'itap of my {}.', 'a photo of a cool {}.', 'a photo of a small {}.', 'a tattoo of the {}.', 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.', 'this is the {} in the scene.', 'this is one {} in the scene.']

    inputs = model.processor(text=texts, return_tensors="pt", padding=True).to(device)

    # prompted_t = [template.format(t) for template in prompt_templates]
    with torch.no_grad():
        text_features = model.encode_text(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features



def get_similarity_map(sm, shape):

    # min-max norm
    sm = (sm - sm.min(1, keepdim=True)[0]) / (sm.max(1, keepdim=True)[0] - sm.min(1, keepdim=True)[0])

    # reshape
    side = int(sm.shape[1] ** 0.5) # square output
    sm = sm.reshape(sm.shape[0], side, side, -1).permute(0, 3, 1, 2)

    # interpolate
    sm = torch.nn.functional.interpolate(sm, shape, mode='bilinear')
    sm = sm.permute(0, 2, 3, 1)
    
    return sm


def clip_feature_surgery(image_features, text_features, redundant_feats=None, t=2):

    if redundant_feats != None:
        similarity = image_features @ (text_features - redundant_feats).t()

    else:
        # weights to restrain influence of obvious classes on others
        prob = image_features[:, :1, :] @ text_features.t()
        prob = (prob * 2).softmax(-1)
        w = prob / prob.mean(-1, keepdim=True)

        # element-wise multiplied features
        b, n_t, n_i, c = image_features.shape[0], text_features.shape[0], image_features.shape[1], image_features.shape[2]
        feats = image_features.reshape(b, n_i, 1, c) * text_features.reshape(1, 1, n_t, c)
        feats *= w.reshape(1, 1, n_t, 1)
        redundant_feats = feats.mean(2, keepdim=True) # along cls dim
        feats = feats - redundant_feats
        
        # sum the element-wise multiplied features as cosine similarity
        similarity = feats.sum(-1)

    return similarity


# sm shape N_t
def similarity_map_to_points(sm, shape, t=0.8, down_sample=2):
    side = int(sm.shape[0] ** 0.5)
    sm = sm.reshape(1, 1, side, side)

    # down sample to smooth results
    down_side = side // down_sample
    sm = torch.nn.functional.interpolate(sm, (down_side, down_side), mode='bilinear')[0, 0, :, :]
    h, w = sm.shape
    sm = sm.reshape(-1)

    sm = (sm - sm.min()) / (sm.max() - sm.min())
    rank = sm.sort(0)[1]
    scale_h = float(shape[0]) / h
    scale_w = float(shape[1]) / w

    num = min((sm >= t).sum(), sm.shape[0] // 2)
    labels = np.ones(num * 2).astype('uint8')
    labels[num:] = 0
    points = []

    # positives
    for idx in rank[-num:]:
        x = min((idx % w + 0.5) * scale_w, shape[1] - 1) # +0.5 to center
        y = min((idx // w + 0.5) * scale_h, shape[0] - 1)
        points.append([int(x.item()), int(y.item())])

    # negatives
    for idx in rank[:num]:
        x = min((idx % w + 0.5) * scale_w, shape[1] - 1)
        y = min((idx // w + 0.5) * scale_h, shape[0] - 1)
        points.append([int(x.item()), int(y.item())])

    return points, labels
