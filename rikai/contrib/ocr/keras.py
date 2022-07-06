from typing import Any, Callable, Tuple

from numpy import ndarray

from rikai.mixin import Pretrained
from rikai.spark.sql.model import ModelType, ModelSpec
from rikai.types.geometry import Box2d, Mask
from rikai.types.vision import Image

__all__ = ["MODEL_TYPE"]

def convert_pred_groups_to_box2d(pred_groups, shapes):
    result_groups = []
    for i in range(len(pred_groups)):
        pred_group = pred_groups[i]
        shape = shapes[i]
        result_group = []
        for pred in pred_group:
            text = pred[0]
            points = pred[1]
            point_x = []
            point_y = []
            for point in points:
                point_x.append(point[0])
                point_y.append(point[1])
            result = {'text': text, 'mask': Box2d(min(point_x), min(point_y), max(point_x), max(point_y))}
            result_group.append(result)
        result_groups.append(result_group)
    return result_groups



def convert_pred_groups_for_rikai(pred_groups, shapes):
    result_groups = []
    for i in range(len(pred_groups)):
        pred_group = pred_groups[i]
        shape = shapes[i]
        result_group = []
        for pred in pred_group:
            text = pred[0]
            points = pred[1]
            poly = []
            for point in points:
                poly.append(point[0])
                poly.append(point[1])
            mask = Mask.from_polygon([poly], shape[1], shape[0])
            result = {'text': text, 'mask': mask}
            result_group.append(result)
        result_groups.append(result_group)
    return result_groups
class KerasModelType(ModelType, Pretrained):
    def __init__(self):
        super().__init__()
        self.model = None

    def load_model(self, spec: ModelSpec, **kwargs):
        self.model = self.pretrained_model()

    def pretrained_model(self):
        import keras_ocr
        return keras_ocr.pipeline.Pipeline()
    
    def schema(self) -> str:
        return "array<struct<text:string, mask:mask>>"

    def transform(self) -> Callable:
        return lambda image: image.to_numpy()

    def predict(self, images, *args, **kwargs) -> Any:
        def _ndarray_to_shape(image: ndarray) -> Tuple:
            pil_image = Image.from_array(image).to_pil()
            return (pil_image.width, pil_image.height)

        pred_groups = self.model.recognize(images)
        shapes = [_ndarray_to_shape(image) for image in images]
        return convert_pred_groups_for_rikai(pred_groups, shapes)

MODEL_TYPE = KerasModelType()
