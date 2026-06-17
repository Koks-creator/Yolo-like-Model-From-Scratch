from typing import Union, Tuple, List
from time import time
from tensorflow.types.experimental import TensorLike
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from model import build_detector, build_detector_anchor, sigmoid_np
from config import Config
from custom_logger import CustomLogger
from custom_decorators import log_call, timeit


logger = CustomLogger(
    log_file_name=Config.LOG_FOLDER / "detector_log.logs",
    logger_log_level=Config.CLI_LOG_LEVEL,
    file_handler_log_level=Config.FILE_LOG_LEVEL
).create_logger()


@dataclass
class DetectedObject:
    bbox: Tuple[float, float, float, float]
    class_id: int
    score: float


@dataclass
class ObjectDetector:
    model_weights: Path
    num_classes: int
    num_boxes: int
    num_anchors: int
    width_mult: float
    img_size: int
    classes_path: Path
    use_anchor: bool = False

    @timeit(logger=logger)
    def __post_init__(self) -> None:
        logger.info(
            "Sztarte with params:\n"
            f"{self.model_weights=}\n"
            f"{self.num_classes=}\n"
            f"{self.num_boxes=}\n"
            f"{self.num_anchors=}\n"
            f"{self.width_mult=}\n"
            f"{self.img_size=}\n"
            f"{self.classes_path=}\n"
            f"{self.use_anchor=}\n"
        )
    
        if not self.classes_path:
            self.class_names = [f'class_{i}' for i in range(self.num_classes)]
        else:
            try:
                with open(self.classes_path) as f:
                    self.class_names = f.read().strip().split("\n")
            except (FileExistsError, FileNotFoundError) as f:
                logger.error(f"Classes file not found: {self.classes_path}", exc_info=True)
                raise Exception("Classes file not found")
            
        np.random.seed(40)
        
        self.class_colors = np.random.randint(
            low=0,
            high=255,
            size=(self.num_classes, 3),
            dtype=np.uint8
        ).tolist()

        logger.info(f"Loading model from: {self.model_weights}")
        try:
            if not self.use_anchor:
                self.model = build_detector(
                    width_multiplier=self.width_mult,
                    num_classes=self.num_classes,
                    num_boxes=self.num_classes,
                    img_size=self.img_size
                )
            else:
                self.model = build_detector_anchor(
                    width_multiplier=self.width_mult,
                    num_classes=self.num_classes,
                    img_size=self.img_size,
                    num_anchors=self.num_anchors
                )
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_weights}, exception: {e}", exc_info=True)
            raise Exception("Failed to load model")
        self.model.load_weights(str(self.model_weights))
        logger.info(
            f"Loaded model from: {self.model_weights} \n"
            f"Params: {self.model.count_params():,}\n"
            f"Outputs: {[tuple(o.shape[1:]) for o in self.model.outputs]}\n"
        )

    @log_call(logger=logger, log_params=[""], hide_res=True)
    def load_image(self, source: Union[Path, np.ndarray]) -> Tuple[TensorLike, TensorLike, Union[Path, np.ndarray]]:
        source_org = source
        if isinstance(source, np.ndarray):
            # OpenCV (BGR) -> RGB sar
            img = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
            img = tf.convert_to_tensor(img, dtype=tf.uint8)
        else:
            raw = tf.io.read_file(str(source))
            img = tf.io.decode_image(raw, channels=3, expand_animations=False)

        img_resized = tf.image.resize(img, [self.img_size, self.img_size])
        img_norm = tf.cast(img_resized, tf.float32) / 255.0

        return img_resized, img_norm, source_org
    
    @log_call(logger=logger, log_params=["conf_threshold", "iou_threshold"], hide_res=True)
    def decode_predictions_anchors(
        self,
        outputs: list[tf.Tensor],
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.1,
    ) -> List[DetectedObject]:
        """
        Decode outputów modelu z anchorami → lista detekcji [{bbox, class, score}, ...].

        Args:
            outputs: lista 3 tensorów z model(img), każdy (1, S, S, A, 5+C) RAW logits.
            conf_threshold: minimalny confidence (sigmoid'owany) żeby box wszedł w grę.
            iou_threshold: IoU powyżej którego dwa boxy = duplikat dla NMS.

        Returns:
            Lista dictów {'bbox': (x1,y1,x2,y2), 'class': int, 'score': float}.
            Współrzędne znormalizowane do [0,1] względem rozmiaru obrazu.

        Proces:
        1. Pętla po 3 skalach
        2. Dla każdej komórki, dla każdego z A anchorów:
            a) Sigmoid conf, filtr po threshold
            b) Sigmoid tx, ty → offset w komórce → image-norm coords
            c) exp(tw) * anchor_w, exp(th) * anchor_h → image-norm size
        3. wh: anchor * (2 * sigmoid(tw))^2 — bounded sigmoid parameterization
        """
        boxes: list[list[float]] = []
        scores: list[float] = []
        classes: list[int] = []

        for scale_idx, scale_out in enumerate(outputs):
            pred = scale_out[0].numpy()    # (S, S, A, 5+C) RAW
            S = pred.shape[0]
            scale_anchors: list[tuple[float, float]] = Config.ANCHORS_PER_SCALE[scale_idx]

            for gy in range(S):
                for gx in range(S):
                    for a_idx in range(self.num_anchors):
                        # Conf — sigmoid + threshold filter
                        conf: float = float(sigmoid_np(pred[gy, gx, a_idx, 4]))
                        if conf < conf_threshold:
                            continue

                        # XY — sigmoid → offset [0,1] w komórce
                        x_off = float(sigmoid_np(pred[gy, gx, a_idx, 0]))
                        y_off = float(sigmoid_np(pred[gy, gx, a_idx, 1]))

                        # WH — bounded sigmoid: anchor * (2 * sigmoid(tw))^2 ∈ [0, 4*anchor]
                        tw = float(pred[gy, gx, a_idx, 2])
                        th = float(pred[gy, gx, a_idx, 3])
                        aw, ah = scale_anchors[a_idx]
                        w = aw * (2.0 * float(sigmoid_np(tw))) ** 2
                        h = ah * (2.0 * float(sigmoid_np(th))) ** 2

                        # Grid coords → image-normalized center
                        xc = (gx + x_off) / S
                        yc = (gy + y_off) / S

                        # Center → corners, clamp do [0,1]
                        x1 = max(0.0, xc - w / 2)
                        y1 = max(0.0, yc - h / 2)
                        x2 = min(1.0, xc + w / 2)
                        y2 = min(1.0, yc + h / 2)

                        # Klasa (sigmoid + argmax)
                        class_logits = pred[gy, gx, a_idx, 5:]
                        class_probs = sigmoid_np(class_logits)
                        cls = int(np.argmax(class_probs))
                        score = conf * float(class_probs[cls])

                        # TF NMS chce [y1, x1, y2, x2]
                        boxes.append([y1, x1, y2, x2])
                        scores.append(score)
                        classes.append(cls)

        if not boxes:
            return []

        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        classes = np.array(classes, dtype=np.int32)

        # NMS — eliminacja duplikatów
        keep = tf.image.non_max_suppression(
            boxes, scores,
            max_output_size=100,
            iou_threshold=iou_threshold,
            score_threshold=conf_threshold,
        ).numpy()

        # Konwertuj z TF (y, x, y, x) na (x, y, x, y) dla output
        return [
            DetectedObject(
                bbox=(float(boxes[i][1]), float(boxes[i][0]), float(boxes[i][3]), float(boxes[i][2])),
                class_id=int(classes[i]),
                score=float(scores[i])
            )
            for i in keep
        ]

    @staticmethod
    @log_call(logger=logger, log_params=["conf_threshold", "iou_threshold"], hide_res=True)
    def decode_predictions(outputs: List, conf_threshold: float = 0.3, iou_threshold: float = 0.1) -> List[DetectedObject]:
        """
        Konwertuje surowe wyjście modelu (3 tensory logitów ze skal 13/26/52)
        na finalną listę detekcji [{bbox, class, score}, ...].

        PROCES:
        1. Sigmoid — model wypluwa raw logits (BCE w treningu używa
            sigmoid_cross_entropy_with_logits, czyli sam sigmoid jest "w środku"
            lossa). W inferencji robimy sigmoid ręcznie, żeby dostać [0,1].
        2. Per cell per scale: jeśli confidence > próg, dekoduj box.
        3. Zamiana grid coords → image-normalized coords.
        4. Zamiana center-format (xc,yc,w,h) → corner-format (x1,y1,x2,y2) — NMS tego wymaga.
        5. Wspólne NMS na detekcjach ZE WSZYSTKICH 3 SKAL — eliminuje duplikaty
            (ten sam obiekt może wyjść na 26x26 ORAZ 52x52, albo w sąsiednich komórkach).

        PARAMETRY:
        outputs:        lista 3 tensorów z model(img), każdy (1, S, S, 5+C)
                        gdzie 5 = [x_off, y_off, w, h, conf], C = liczba klas
        conf_threshold: minimalny confidence żeby box w ogóle wszedł w grę
        iou_threshold:  IoU powyżej którego 2 boxy = duplikat
                        0.45 standard YOLO, 0.1 (twoje) agresywne — czyści
                        duplikaty mocno, ale ryzykuje zjedzenie BLISKICH osobnych
                        osób (np. dwoje ludzi obok siebie). Sweet spot zwykle 0.3-0.45.
        """
        boxes, scores, classes = [], [], []

        # === Pętla po 3 skalach (13×13, 26×26, 52×52) ===
        for scale_out in outputs:
            # scale_out ma shape (1, S, S, 5+C). Bierzemy [0] bo batch=1.
            # Sigmoid wszystkiego: coords, conf, class probs → wszystko w [0,1]
            pred = tf.sigmoid(scale_out[0]).numpy()   # (S, S, 5+C)
            S = pred.shape[0]                          # 13 / 26 / 52

            # Pętla po komórkach siatki
            for gy in range(S):
                for gx in range(S):
                    conf = pred[gy, gx, 4]    # objectness — "czy w tej komórce jest obiekt"

                    # Filtruj puste komórki — model nauczył się dawać tu conf bliski 0.
                    # Bez tego mielibyśmy S*S boxów na każdej skali (np. 52*52=2704!).
                    if conf < conf_threshold:
                        continue

                    # x_off, y_off ∈ [0,1] — pozycja środka WEWNĄTRZ komórki (gx,gy)
                    # w, h ∈ [0,1] — rozmiar boxa względem CAŁEGO obrazu
                    x_off = pred[gy, gx, 0]
                    y_off = pred[gy, gx, 1]
                    w     = pred[gy, gx, 2]
                    h     = pred[gy, gx, 3]

                    # Konwersja "która komórka + offset w niej" → "gdzie w obrazie"
                    # Przykład: komórka (5,7) siatki 13×13 z offsetem (0.5, 0.5)
                    #   xc = (5 + 0.5) / 13 = 0.423  → 42% szerokości obrazu
                    xc = (gx + x_off) / S
                    yc = (gy + y_off) / S

                    # Center-format (xc, yc, w, h) → corner-format (x1, y1, x2, y2)
                    # NMS pracuje na rogach. Clamp do [0,1] żeby box nie wystawał za obraz.
                    x1 = max(0.0, xc - w/2)
                    y1 = max(0.0, yc - h/2)
                    x2 = min(1.0, xc + w/2)
                    y2 = min(1.0, yc + h/2)

                    # Klasa = argmax po C kanałach klasowych (po indeksie 5)
                    cls = int(np.argmax(pred[gy, gx, 5:]))

                    # Final score = objectness × class_probability
                    # Klasyczny YOLO scoring: "jak pewny że to obiekt I że to klasa X"
                    # Dla 1 klasy class_prob jest zwykle blisko 1, więc score ≈ conf
                    score = float(conf * pred[gy, gx, 5 + cls])

                    # UWAGA: tf.image.non_max_suppression wymaga [y1, x1, y2, x2]
                    # (najpierw y, potem x). To TF konwencja, łatwo o pomyłkę.
                    boxes.append([y1, x1, y2, x2])
                    scores.append(score)
                    classes.append(cls)

        # Jak żadna z 3 skal nic nie wykryła → pusta lista
        if not boxes:
            return []

        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        classes = np.array(classes, dtype=np.int32)

        # === NMS — eliminacja duplikatów ===
        # Sortuje po score malejąco, idzie po liście, dla każdego boxa wyrzuca
        # wszystkie pozostałe które nakładają się z nim powyżej iou_threshold.
        # Zwraca indeksy które zostawić.
        keep = tf.image.non_max_suppression(
            boxes, scores,
            max_output_size=100,             # twardy limit (safety)
            iou_threshold=iou_threshold,
            score_threshold=conf_threshold,  # podwójny filtr — NMS też wywala niskie
        ).numpy()

        # Przywróć format (x1, y1, x2, y2) bo to bardziej naturalne niż TF-owe (y,x,y,x)
        return [
            DetectedObject(
                bbox=(float(boxes[i][1]), float(boxes[i][0]), float(boxes[i][3]), float(boxes[i][2])),
                class_id=int(classes[i]),
                score=float(scores[i])
            )
            for i in keep
        ]
    @log_call(logger=logger, log_params=["conf_threshold", "iou_threshold", "draw"], hide_res=True)
    @timeit(logger=logger, return_val=True)
    def detect(self, image_input: Union[Path, np.ndarray],
               conf_threshold: float = 0.1,
               iou_threshold: float = 0.1,
               draw: bool = True
            ) -> Tuple[List[DetectedObject], np.ndarray]:
        
        img_resized, img_norm, img_org = self.load_image(image_input)
        outputs = self.model(tf.expand_dims(img_norm, 0), training=False)

        if self.use_anchor:
            detections = self.decode_predictions_anchors(
                outputs=outputs,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold
            )
        else:
            detections = self.decode_predictions(
                outputs=outputs,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold
            )

        if draw:
            h, w, _ = img_org.shape
            img_resized = img_resized.numpy().astype(np.uint8)
            for det in detections:
                x1, y1, x2, y2 = det.bbox
                x1, x2 = int(x1 * w), int(x2 * w)
                y1, y2 = int(y1 * h), int(y2 * h)
                color = list(self.class_colors[det.class_id])

                cv2.rectangle(img_org, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img_org, f"{self.class_names[det.class_id]} {det.score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)

        return (detections, img_org)
    
    @log_call(logger=logger, log_params=["conf_threshold", "iou_threshold", "figsize"], hide_res=True)
    def predict_and_show(self,
                         image_input: Union[Path, np.ndarray],
                         conf_threshold: float = 0.1,
                         iou_threshold: float = 0.1,
                         figsize: Tuple[int, int] = (8, 8)
                        ) -> List[DetectedObject]:
        
        CLASS_COLORS = np.random.rand(self.num_classes, 3)
        img_resized, img_norm, _ = self.load_image(image_input)
        outputs = self.model(tf.expand_dims(img_norm, 0), training=False)

        if self.use_anchor:
            detections = self.decode_predictions_anchors(
                outputs=outputs,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold
            )
        else:
            detections = self.decode_predictions(
                outputs=outputs,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold
            )

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(img_resized.numpy().astype(np.uint8))
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            x1, x2 = x1 * self.img_size, x2 * self.img_size
            y1, y2 = y1 * self.img_size, y2 * self.img_size
            color = CLASS_COLORS[det.class_id]
            ax.add_patch(patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor='none',
            ))
            ax.text(
                x1, max(y1 - 5, 0),
                f"{self.class_names[det.class_id]} {det.score:.2f}",
                color='white', fontsize=9,
                bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=2),
            )
        # ax.set_title(f"{len(detections)} detekcji  |  {Path(image_path).name}")
        ax.axis('off')
        plt.tight_layout()
        plt.show()

        return detections

if __name__ == "__main__":
    ob = ObjectDetector(
        model_weights=Config.MODELS_FOLDER / "model_mosaic" / "best_tuned_no_mosaic.weights.h5",
        num_classes=Config.NUM_CLASSES,
        num_boxes=Config.NUM_BOXES,
        width_mult=Config.WIDTH_MULT,
        img_size=Config.IMG_SIZE,
        classes_path=Config.CLASSES_FILE,
        use_anchor=False,
        num_anchors=Config.NUM_ANCHORS
    )
    img = cv2.imread("pexels-photo-4750056.jpeg")
    x = ob.predict_and_show(
        image_input=img,
        iou_threshold=.1
    )

    # cap = cv2.VideoCapture(Config.VIDOES_FOLDER / "14735436_1280_720_60fps.mp4")

    # p_time = 0
    # while cap.isOpened():
    #     success, frame = cap.read()
    #     if not success:
    #         break
    #     (detections, img_org), elapsed = ob.detect(
    #         image_input=frame,
    #         iou_threshold=.2,
    #         conf_threshold=.3
    #     )
    #     # print(detections)
    #     key = cv2.waitKey(1)
    #     if key == 27:
    #         break

    #     c_time = time()
    #     fps = int(1 / (c_time - p_time))
    #     p_time = c_time

    #     cv2.putText(img_org, f"FPS: {fps}", (10, 25), cv2.FONT_HERSHEY_PLAIN, 1.4, (100, 0, 255), 2)
    #     cv2.imshow("res", img_org)

    # cap.release()
    # cv2.destroyAllWindows()

