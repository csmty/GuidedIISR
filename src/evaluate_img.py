import pyiqa
import os
import argparse
from pathlib import Path
import torch
from utils import util_image
import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(pyiqa.list_models())
def _image_paths_by_stem(root):
    root = Path(root) if not isinstance(root, Path) else root
    paths = []
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
        paths.extend(root.glob(pattern))
    by_stem = {}
    for path in sorted(paths):
        if path.stem in by_stem:
            raise ValueError(f"Duplicate image stem in {root}: {path.stem}")
        by_stem[path.stem] = path
    return by_stem


def evaluate(in_path, ref_path, ntest):
    metric_dict = {}
    metric_dict["clipiqa"] = pyiqa.create_metric('clipiqa').to(device)
    metric_dict["musiq"] = pyiqa.create_metric('musiq').to(device)
    metric_dict["niqe"] = pyiqa.create_metric('niqe').to(device)
    metric_dict["maniqa"] = pyiqa.create_metric('maniqa').to(device)
    metric_paired_dict = {}
    
    in_path = Path(in_path) if not isinstance(in_path, Path) else in_path
    assert in_path.is_dir()
    
    ref_by_stem = None
    if ref_path is not None:
        ref_path = Path(ref_path) if not isinstance(ref_path, Path) else ref_path
        ref_by_stem = _image_paths_by_stem(ref_path)
        
        metric_paired_dict["psnr"]=pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(device)
        metric_paired_dict["lpips"]=pyiqa.create_metric('lpips').to(device)
        metric_paired_dict["dists"]=pyiqa.create_metric('dists').to(device)
        metric_paired_dict["ssim"]=pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr' ).to(device)
        
    pred_by_stem = _image_paths_by_stem(in_path)
    stems = sorted(pred_by_stem)
    if ref_by_stem is not None:
        missing = [stem for stem in stems if stem not in ref_by_stem]
        if missing:
            raise FileNotFoundError(f"Missing GT images for predictions: {missing[:5]}")
    if ntest is not None:
        stems = stems[:ntest]
    
    print(f'Find {len(stems)} images in {in_path}')
    result = {}
    for stem in tqdm.tqdm(stems):
        _in_path = pred_by_stem[stem]
        _ref_path = ref_by_stem[stem] if ref_by_stem is not None else None
        
        im_in = util_image.imread(_in_path, chn='rgb', dtype='float32')  # h x w x c
        im_in_tensor = util_image.img2tensor(im_in).cuda()              # 1 x c x h x w
        for key, metric in metric_dict.items():
            with torch.cuda.amp.autocast():
                result[key] = result.get(key, 0) + metric(im_in_tensor).item()
        
        if ref_by_stem is not None:
            im_ref = util_image.imread(_ref_path, chn='rgb', dtype='float32')  # h x w x c
            im_ref_tensor = util_image.img2tensor(im_ref).cuda()    
            for key, metric in metric_paired_dict.items():
                result[key] = result.get(key, 0) + metric(im_in_tensor, im_ref_tensor).item()
    
    if ref_path is not None:
        fid_metric = pyiqa.create_metric('fid')
        result['fid'] = fid_metric(in_path, ref_path)

    for key, res in result.items():
        if key == 'fid':
            print(f"{key}: {res:.2f}")
        else:
            print(f"{key}: {res/len(stems):.5f}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',"--in_path", type=str, required=True)
    parser.add_argument("-r", "--ref_path", type=str, default=None)
    parser.add_argument("--ntest", type=int, default=None)
    args = parser.parse_args()
    evaluate(args.in_path, args.ref_path, args.ntest)
    