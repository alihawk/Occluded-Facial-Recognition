# Occluded Facial Recognition

Face recognition that still works when part of the face is covered, for example by a face mask. Because labelled photos of masked faces are hard to find, the training data is generated: faces are detected and cropped from ordinary photos, and synthetic masks are drawn onto them.

## Pipeline

1. **Face detection:** faces are found with an SSD detector (DSFD detector in `face_detection/`, OpenCV ResNet-10 SSD model in `face_detection_model/`).
2. **Face extraction:** `faceExtraction.py` crops the detected faces.
3. **Synthetic masks:** `databaseGeneration.py` overlays masks of different colours (`mask/`, drawing code in `create_mask/`) on the cropped faces to build an occluded training set.
4. **Training:** `train.py` fine-tunes a ResNet-34 classifier with fastai on the masked and unmasked faces.
5. **Testing:** `test.py` runs recognition on new images.

## Setup

The code uses the fastai v1 API, so it needs **Python 3.8** and the pinned versions in `requirements.txt` (`face-recognition` builds `dlib`, which needs CMake and a C++ compiler):

```bash
python3.8 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For `faceExtraction.py`, download the DSFD weights `WIDERFace_DSFD_RES152.pth` ([Google Drive](https://drive.google.com/file/d/1WeXlNYsM6dMP3xQQELI-4gxhwKUQxc3-/view), from [Tencent/FaceDetection-DSFD](https://github.com/Tencent/FaceDetection-DSFD)) into `face_detection/weights/`. Runs on GPU if available, otherwise CPU.

## Running it

Put training photos in `dataset/<person_name>/*.jpg`, then:

```bash
python databaseGeneration.py   # -> dataset_with_mask/<person_name>/ (masked, cropped faces)
python train.py                # fine-tunes ResNet-34, writes dataset_with_mask/export.pkl
python test.py                 # predicts on test_data1/imdasd.jpeg
python faceExtraction.py       # optional: DSFD face crops for the images in test_data1/
```

Notebook versions of training and testing are in `train.ipynb` and `test.ipynb`; earlier checkpoint experiments are in `checkpoint_notebooks/`.
