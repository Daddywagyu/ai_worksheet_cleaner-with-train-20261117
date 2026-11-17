@echo off
call venv\Scripts\activate
pip install torch torchvision onnx
python -m training.synthetic_data --out data\syn --n 300 --versions 3 --seed 1
python -m training.train --data data\syn\manifest.json --out runs\r1 --epochs 40 --batch 16 --crop 256 --pretrained
python -m training.evaluate --data data\syn\manifest.json --ckpt runs\r1\best.pt --out runs\r1\eval.json
python -m training.export_onnx --ckpt runs\r1\best.pt --out models\ink_segmentation
