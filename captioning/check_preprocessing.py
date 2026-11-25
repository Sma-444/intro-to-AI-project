"""Check image borders survive letterboxing and old evaluation remains compatible."""
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
from train import Letterbox, get_transforms, IMAGE_MEAN, IMAGE_STD


for size in (224, 320):
    for width, height in ((800, 200), (200, 800), (300, 300)):
        image = Image.new('RGB', (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, width//8, height-1), fill=(255, 0, 0))
        draw.rectangle((width-width//8, 0, width-1, height-1), fill=(0, 0, 255))
        result = Letterbox(size)(image)
        assert result.size == (size, size)
        scale = size/max(width, height)
        rw, rh = round(width*scale), round(height*scale)
        x, y = (size-rw)//2, (size-rh)//2
        assert result.getpixel((x, y+rh//2)) == (255, 0, 0)
        assert result.getpixel((x+rw-1, y+rh//2)) == (0, 0, 255)
        _, evaluation = get_transforms('letterbox', size)
        assert evaluation(image).shape == (3, size, size)
        assert torch.equal(evaluation(image), evaluation(image))
        assert abs(rw/rh-width/height) < .02
image = Image.new('RGB', (500, 333), (90, 120, 150))
legacy = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize(IMAGE_MEAN, IMAGE_STD)])
assert torch.equal(get_transforms()[1](image), legacy(image))
print('PASS: image edges preserved; portrait/landscape/square aspect ratio; deterministic output; legacy crop unchanged')
