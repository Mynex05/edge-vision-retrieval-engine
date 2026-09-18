import torch
import logging
from src.core.extractor import OptimizedVisionExtractor

logger = logging.getLogger("EdgeVisionEngine.Exporter")

class ModelExporter:
    """
    Handles exporting PyTorch edge models to ONNX format for cross-platform
    hardware acceleration (TensorRT, OpenVINO, mobile runtimes).
    """
    def __init__(self, model: OptimizedVisionExtractor):
        self.model = model

    def export_to_onnx(self, output_path: str = "edge_vision_model.onnx", batch_size: int = 1):
        """Exports the PyTorch model graph to an optimized ONNX file."""
        self.model.eval()
        
        # Create dummy input matching expected input shape: (B, 3, 224, 224)
        dummy_input = torch.randn(batch_size, 3, 224, 224)
        
        logger.info(f"Exporting model to ONNX format at '{output_path}'...")
        
        try:
            torch.onnx.export(
                self.model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=["input_image"],
                output_names=["embedding_vector"],
                dynamic_axes={
                    "input_image": {0: "batch_size"},
                    "embedding_vector": {0: "batch_size"}
                }
            )
            logger.info(f"Model successfully exported and saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export model to ONNX: {e}")
            raise

if __name__ == "__main__":
    # Instantiate backbone engine
    engine = OptimizedVisionExtractor(embedding_dim=512)
    
    # Export to ONNX
    exporter = ModelExporter(engine)
    exporter.export_to_onnx("edge_vision_model.onnx")