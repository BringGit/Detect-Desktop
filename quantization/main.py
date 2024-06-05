import nncf
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import openvino.runtime as ov
from pathlib import Path
import torch
from torchvision import datasets, transforms

import torchvision.datasets.coco as coco


ROOT = Path(__file__).parent.resolve()
def export_ov(model, model_name):
    ir_model_path = Path(f"{ROOT}/{model_name}_openvino_model/{model_name}.xml")
    if not ir_model_path.exists():
        onnx_model_path = Path(f"{ROOT}/{model_name}.onnx")
        if not onnx_model_path.exists():
            model.export(format="onnx", dynamic=True, half=False)

        ov.save_model(ov.convert_model(onnx_model_path), ir_model_path)
    return ov.Core().read_model(ir_model_path), ir_model_path

def transform_fn(data_item):
        """
        Quantization transform function. Extracts and preprocess input data from dataloader
        item for quantization.
        Parameters:
        data_item: Dict with data item produced by DataLoader during iteration
        Returns:
            input_tensor: Input data for quantization
        """
        # input_tensor = validator.preprocess(data_item)["img"].numpy()
        # return input_tensor
        images, _ = data_item
        return images

    
if __name__ == "__main__":
    # export_ov()
    model = ov.Core().read_model("../yolov8m_openvino_model/yolov8m.xml")
    
    dataset = coco.CocoDetection("E:/Project/Vision/Datasets/coco/val2017", annFile="E:/Project/Vision/Datasets/coco/annotations/instances_val2017.json", 
                                 transform=transforms.Compose([transforms.Resize((640, 640)),
                                                                transforms.ToTensor(),
                                                               transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    quantization_dataset = nncf.Dataset(data_loader, transform_fn)
    quantization_model = nncf.quantize(model, quantization_dataset, preset=nncf.QuantizationPreset.MIXED, 
                                       ignored_scope=nncf.IgnoredScope(types=["Multiply", "Subtract", "Sigmoid"]))
    ov.save_model(quantization_model, "./yolov8m_int8.xml")
    # ov.serialize(quantization_model, xml_path="./yolov8m.xml", bin_path="./yolov8.bin",)
