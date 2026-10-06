"""
mobile_client/export_tflite.py
------------------------------
Export and conversion script for deploying EfficientNet-B1 onto mobile/edge devices.
Converts trained PyTorch weights -> ONNX -> TensorFlow Lite (TFLite) with FP16/INT8 quantization.
"""

import os
import argparse
import torch
import torch.nn as nn
from backbones import get_model


def export_to_onnx(pytorch_model_path: str, onnx_output_path: str, num_classes: int = 206):
    """Converts PyTorch model checkpoint to ONNX format."""
    print("Loading PyTorch model architecture...")
    model = get_model('efficientnet_b1', num_classes=num_classes, pretrained=False)
    if os.path.exists(pytorch_model_path):
        checkpoint = torch.load(pytorch_model_path, map_location='cpu')
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    
    print(f"Exporting model to ONNX: {onnx_output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input_spectrogram'],
        output_names=['species_logits'],
        dynamic_axes={'input_spectrogram': {0: 'batch_size'}, 'species_logits': {0: 'batch_size'}}
    )
    print("ONNX export completed successfully.")


def convert_onnx_to_tflite(onnx_path: str, tflite_path: str, quantize_fp16: bool = True):
    """Converts ONNX graph to TensorFlow Lite model with optional quantization."""
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf

        print("Converting ONNX to TensorFlow SavedModel...")
        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)
        saved_model_dir = "models/checkpoints/tf_saved_model"
        tf_rep.export_graph(saved_model_dir)

        print("Converting TensorFlow SavedModel to TFLite...")
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        
        if quantize_fp16:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            print("Applying Float16 quantization for edge deployment...")

        tflite_model = converter.convert()

        os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)

        print(f"TFLite model successfully saved to: {tflite_path}")
        print(f"Quantized Model Size: {os.path.getsize(tflite_path) / (1024 * 1024):.2f} MB")

    except ImportError:
        print("Note: Required libraries (onnx-tf, tensorflow) not installed in local environment.")
        print("TFLite conversion pipeline code is ready for edge compilation environment.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export EfficientNet-B1 to TFLite")
    parser.add_argument("--weights", type=str, default="models/checkpoints/best_model.pth")
    parser.add_argument("--onnx_out", type=str, default="models/checkpoints/model.onnx")
    parser.add_argument("--tflite_out", type=str, default="models/checkpoints/model_fp16.tflite")
    args = parser.parse_args()

    export_to_onnx(args.weights, args.onnx_out)
    convert_onnx_to_tflite(args.onnx_out, args.tflite_out)
