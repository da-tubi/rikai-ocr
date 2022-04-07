from typing import Any, Callable

from rikai.mixin import Pretrained
from rikai.spark.sql.model import ModelType, ModelSpec
from rikai.types.geometry import Mask

__all__ = ["MODEL_TYPE"]


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

    def load_model(self, spec: ModelSpec, **kwargs):
        return super().load_model(spec, **kwargs)

    def pretrained_model(self):
        import keras_ocr
        return keras_ocr.pipeline.Pipeline()
    
    def schema(self) -> str:
        return "array<struct<text:string, mask:mask>>"

    def transform(self) -> Callable:
        return lambda x:x

    def predict(self, images, *args, **kwargs) -> Any:
        pred_groups = self.model(images)
        shapes = []
        return convert_pred_groups_for_rikai(pred_groups, shapes)

MODEL_TYPE = KerasModelType()
