import json
import torch
import torch.nn as nn
import cv2
from torchvision import transforms, models
from PIL import Image
from pathlib import Path


class Classifier:
    def __init__(self, model_path, class_mapping_path):
        self.device = torch.device("cpu")

        # ---------------- LOAD CLASS MAPPING ----------------
        with open(class_mapping_path, "r") as f:
            mapping = json.load(f)

        self.class_to_idx = mapping["class_to_idx"]
        self.idx_to_class = {
            int(k): v for k, v in mapping["idx_to_class"].items()
        }
        self.class_names = [
            self.idx_to_class[i] for i in range(len(self.idx_to_class))
        ]

        # ---------------- LOAD MODEL ----------------
        self.model = self._load_model(model_path)
        self.model.eval()

        # ---------------- TRANSFORM ----------------
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_model(self, model_path):
        model = models.efficientnet_b0(pretrained=False)

        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, len(self.class_names))

        state_dict = torch.load(model_path, map_location=self.device)

        # Handle DataParallel
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {
                k.replace("module.", ""): v for k, v in state_dict.items()
            }

        model.load_state_dict(state_dict, strict=True)
        return model.to(self.device)

    def _predict_tensor(self, x):
        x = x.to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            idx = probs.argmax(dim=1).item()
            conf = probs[0, idx].item()

        return {
            "index": idx,
            "label": self.idx_to_class[idx],
            "confidence": round(conf, 4),
            "all_probabilities": {
                self.idx_to_class[i]: round(probs[0, i].item(), 4)
                for i in range(len(self.class_names))
            }
        }

    def predict_from_array(self, arr):
        if arr.ndim == 3 and arr.shape[2] == 3:
            img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
        else:
            img = Image.fromarray(arr)

        img = img.convert("RGB")
        x = self.transform(img).unsqueeze(0)
        return self._predict_tensor(x)

# import torch
# import cv2
# import torch.nn as nn
# from torchvision import transforms, models
# from PIL import Image
# from torch.serialization import safe_globals
# import ultralytics.nn.tasks as ultratasks


# class Classifier:
#     def __init__(self, model_path, class_names):
#         self.class_names = class_names
#         self.device = torch.device("cpu")

#         self.model = self._load_model(model_path)
#         self.model.eval()

#         self.transform = transforms.Compose([
#             transforms.Resize((224, 224)),  # DenseNet standard
#             transforms.ToTensor()
#         ])

#     def _load_model(self, model_path):
#         """
#         Supports:
#         - Ultralytics classification models (.pt)
#         - DenseNet checkpoints (.pth with model_state_dict)
#         """

#         # --------------------------------------------------
#         # 1️⃣ Try Ultralytics-safe load first
#         # --------------------------------------------------
#         try:
#             with safe_globals([ultratasks.ClassificationModel]):
#                 ckpt = torch.load(
#                     model_path,
#                     map_location=self.device,
#                     weights_only=False
#                 )
#         except Exception:
#             ckpt = torch.load(model_path, map_location=self.device)

#         # --------------------------------------------------
#         # CASE A: Ultralytics classification model
#         # --------------------------------------------------
#         if isinstance(ckpt, dict) and "model" in ckpt:
#             model = ckpt["model"]
#             if isinstance(model, nn.Module):
#                 return model.float().to(self.device)

#         # --------------------------------------------------
#         # CASE B: DenseNet training checkpoint (YOUR CASE)
#         # --------------------------------------------------
#         if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
#             model = models.densenet121(weights=None)

#             in_features = model.classifier.in_features
#             model.classifier = nn.Linear(in_features, len(self.class_names))

#             state_dict = ckpt["model_state_dict"]

#             # Remove DataParallel prefix
#             new_state_dict = {}
#             for k, v in state_dict.items():
#                 if k.startswith("module."):
#                     new_state_dict[k.replace("module.", "", 1)] = v
#                 else:
#                     new_state_dict[k] = v

#             model.load_state_dict(new_state_dict, strict=True)
#             return model.float().to(self.device)


#         # --------------------------------------------------
#         # CASE C: Saved nn.Module
#         # --------------------------------------------------
#         if isinstance(ckpt, nn.Module):
#             return ckpt.float().to(self.device)

#         # --------------------------------------------------
#         # FAIL FAST
#         # --------------------------------------------------
#         raise RuntimeError(
#             "Unsupported checkpoint format. "
#             "Expected Ultralytics model or DenseNet checkpoint."
#         )

#     def _predict_tensor(self, x):
#         x = x.to(self.device)

#         with torch.no_grad():
#             out = self.model(x)

#             if isinstance(out, (tuple, list)):
#                 out = out[0]

#             if isinstance(out, dict):
#                 out = out.get("logits", next(iter(out.values())))

#             idx = out.argmax(dim=1).item()

#         return self.class_names[idx]

#     def predict_from_array(self, arr):
#         if arr.ndim == 3 and arr.shape[2] == 3:
#             img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
#         else:
#             img = Image.fromarray(arr)

#         img = img.convert("RGB")
#         x = self.transform(img).unsqueeze(0)
#         return self._predict_tensor(x)

#     def predict(self, img_path):
#         img = Image.open(img_path).convert("RGB")
#         x = self.transform(img).unsqueeze(0)
#         return self._predict_tensor(x)
