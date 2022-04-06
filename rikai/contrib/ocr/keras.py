from typing import Any
from rikai.mixin import Pretrained
from rikai.spark.sql.model import ModelType
import keras_ocr

__all__ = ["MODEL_TYPE"]

class KerasModelType(ModelType, Pretrained):
    def pretrained_model(self):
        return keras_ocr.pipeline.Pipeline()
    
    def schema(self) -> str:
        return "array<struct<text:string, mask:mask>>"

    def predict(self, images, *args, **kwargs) -> Any:
        return []

MODEL_TYPE = KerasModelType()
