# Occluded Facial Recognition

Face recognition that still works when part of the face is covered, for example by a face mask. Because labelled photos of masked faces are hard to find, the training data is generated: faces are detected and cropped from ordinary photos, and synthetic masks are drawn onto them.

## Pipeline

1. **Face detection:** faces are found with an SSD detector (`faceDetection/`, with an OpenCV ResNet-10 SSD model in `faceDetectionModel/`).
2. **Face extraction:** `faceExtraction.py` crops the detected faces.
3. **Synthetic masks:** `databaseGeneration.py` overlays masks of different colours (`maskImages/`) on the cropped faces to build an occluded training set.
4. **Training:** `train.py` fine-tunes a ResNet-34 classifier with fastai on the masked and unmasked faces.
5. **Testing:** `test.py` runs recognition on new images.

## Running it

```bash
pip install -r requirements.txt
python faceExtraction.py
python databaseGeneration.py
python train.py
python test.py
```

Notebook versions of training and testing are in `train.ipynb` and `test.ipynb`.
