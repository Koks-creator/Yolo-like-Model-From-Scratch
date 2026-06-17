from dataclasses import dataclass
from typing import List, Union, Tuple
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

from detector import ObjectDetector
from config import Config


@dataclass
class DetectResult:
    model_name: str
    proc_time: float
    image: np.ndarray


@dataclass
class ModelConfig:
    model_weights: Path
    num_classes: int
    num_boxes: int
    num_anchors: int
    width_mult: float
    img_size: int
    classes_path: Path
    use_anchor: bool = False


class CompareModels:

    def __init__(self) -> None:
        self._current_models = []

    def load_models(self, models_config: List[ModelConfig]) -> None:
        self._current_models = []
        for model_conf in models_config:
            self._current_models.append(
                ObjectDetector(**model_conf.__dict__)
            )

    def detect_on_all_models(self,
                             image: np.ndarray,
                             iou_threshold: float = 0.1,
                             conf_threshold: float = 0.3
                             ) -> List[DetectResult]:
        results = []
        for model_obj in self._current_models:
            image_copy = image.copy() # copy cuz `detect` overwrites provided image with drawn bboxes,
                                      # so we end up with 1 image with bboxes of every model
            (_, img_org), elapsed = model_obj.detect(
                image_input=image_copy,
                iou_threshold=iou_threshold,
                conf_threshold=conf_threshold
            )

            model_path = Path(model_obj.model_weights)
            model_name = f"{model_path.parent.name}/{model_path.name}"
            results.append(
                DetectResult(
                    model_name=model_name,
                    proc_time=elapsed,
                    image=img_org
                )
            )
        
        return results
    
    def compare_on_image_plt(
        self,
        image: np.ndarray,
        iou_threshold: float = 0.1,
        conf_threshold: float = 0.1,
        figsize: tuple = None,
    ) -> plt.figure:
        
        results = self.detect_on_all_models(
            image=image,
            iou_threshold=iou_threshold,
            conf_threshold=conf_threshold,
        )
        if not results:
            raise ValueError("No loaded models — call load_models() first :P")

        n = len(results)
        fig, axes = plt.subplots(1, n, figsize=figsize or (6 * n, 6))
        if n == 1:
            axes = [axes]

        for ax, res in zip(axes, results):
            model_name, elapsed, img_org = res.model_name, res.proc_time, res.image
            # cv2 -> BGR, plt -> RGB
            img_rgb = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            ax.set_title(f"{model_name}\n{elapsed:.4f}s", fontsize=10)
            ax.axis("off")

        fig.tight_layout()
        plt.show()

        return fig
    
    def compare_on_video(
        self,
        video_path: Union[Path, str],
        iou_threshold: float = 0.1,
        conf_threshold: float = 0.1,
        max_frames: int = None,
        divide_h: float = None,
        divide_w: float = None,
        frame_shape: Tuple[int, int] = (1366, 768),
        font_scale: float = 0.6,
        font_thickness: int =2
    ) -> dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        stats = {}
        frame_idx = 0

        if not divide_h:
            divide_h = 1.4
        if not divide_w:
            divide_w = len(self._current_models) - .2

        try:
            while True:
                ret, frame = cap.read()
                if not ret or (max_frames and frame_idx >= max_frames):
                    break

                if frame.shape[1] != frame_shape[1] and frame.shape[1] != frame_shape[0]:
                    frame = cv2.resize(frame, frame_shape)

                results = self.detect_on_all_models(
                    image=frame,
                    iou_threshold=iou_threshold,
                    conf_threshold=conf_threshold,
                )

                panels = []
                for res in results:
                    model_name, elapsed, img_org = res.model_name, res.proc_time, res.image
                    if len(self._current_models) > 1:
                        h, w, _ = img_org.shape
                        img_org = cv2.resize(img_org, (int(w // divide_w), int(h //divide_h)))
                    acc = stats.setdefault(model_name, [0.0, 0])
                    acc[0] += elapsed
                    acc[1] += 1
                    avg_fps = acc[1] / acc[0] if acc[0] > 0 else 0.0

                    cv2.putText(img_org, f"FPS: {avg_fps:.1f}",
                                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                (100, 80, 75), font_thickness)
                    cv2.putText(img_org, f"{model_name}",
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                (100, 80, 75), font_thickness)
                    panels.append(img_org)

                cv2.imshow("compare", cv2.hconcat(panels))
                key = cv2.waitKey(1)
                if key == 27:
                    break

                frame_idx += 1
        finally:
            cap.release()
            cv2.destroyAllWindows()

        return {name: {"avg_s": t / n, "avg_fps": n / t} for name, (t, n) in stats.items()}

        
if __name__ == "__main__":
    cm  = CompareModels()

    configs = [
        ModelConfig(
            model_weights=Config.MODELS_FOLDER / "model_mosaic" / "best_vanila_mosaic.weights.h5",
            num_classes=Config.NUM_CLASSES,
            num_boxes=Config.NUM_BOXES,
            width_mult=Config.WIDTH_MULT,
            img_size=Config.IMG_SIZE,
            classes_path=Config.CLASSES_FILE,
            use_anchor=False,
            num_anchors=Config.NUM_ANCHORS
        ),
        ModelConfig(
            model_weights=Config.MODELS_FOLDER / "model_mosaic2" / "best.weights.h5",
            num_classes=Config.NUM_CLASSES,
            num_boxes=Config.NUM_BOXES,
            width_mult=Config.WIDTH_MULT,
            img_size=Config.IMG_SIZE,
            classes_path=Config.CLASSES_FILE,
            use_anchor=False,
            num_anchors=Config.NUM_ANCHORS
        ),
        ModelConfig(
            model_weights=Config.MODELS_FOLDER / "model_anchors2" / "best.weights.h5",
            num_classes=Config.NUM_CLASSES,
            num_boxes=Config.NUM_BOXES,
            width_mult=Config.WIDTH_MULT,
            img_size=Config.IMG_SIZE,
            classes_path=Config.CLASSES_FILE,
            use_anchor=True,
            num_anchors=Config.NUM_ANCHORS
        ),
        
    ]
    cm.load_models(models_config=configs)

    # img = cv2.imread(r"C:\Users\table\PycharmProjects\MojeCos2\objekt_detekszyn\pexels-photo-4750056.jpeg")
    # cm.compare_on_image_plt(image=img)
    cm.compare_on_video(
        video_path=Config.VIDOES_FOLDER / "14735436_1280_720_60fps.mp4",
        iou_threshold=.2,
        conf_threshold=.3
    )