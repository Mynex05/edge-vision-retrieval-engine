import time
import torch
import torch.nn as nn
import logging

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("EdgeVisionEngine")

class OptimizedVisionExtractor(nn.Module):
    """
    Simulates a lightweight vision backbone (e.g., MobileNet/ViT variant) 
    optimized for low-latency feature extraction and L2 normalization.
    """
    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Simulating backbone layers
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, self.embedding_dim)
        )
        self._is_quantized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.size(1) != 3:
            raise ValueError(f"Expected input tensor of shape (B, 3, H, W), got {x.shape}")
        
        features = self.backbone(x)
        # Apply L2 Normalization for cosine similarity / vector search readiness
        normalized_embeddings = torch.nn.functional.normalize(features, p=2, dim=1)
        return normalized_embeddings

    def quantize_model(self):
        """Simulates Post-Training Quantization (PTQ) to INT8 for edge deployment."""
        logger.info("Applying dynamic INT8 quantization to model weights...")
        self.backbone = torch.quantization.quantize_dynamic(
            self.backbone, {nn.Linear}, dtype=torch.qint8
        )
        self._is_quantized = True
        logger.info("Quantization successfully applied.")

def benchmark_inference(model: nn.Module, input_tensor: torch.Tensor, iterations: int = 100):
    """
    Measures average inference latency and simulates memory footprint metrics.
    """
    model.eval()
    logger.info(f"Starting latency benchmark over {iterations} iterations...")
    
    # Warm-up run to avoid cold-start overhead anomalies
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)

    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(input_tensor)
    end_time = time.perf_counter()

    total_time_ms = (end_time - start_time) * 1000.0
    avg_latency_ms = total_time_ms / iterations
    
    logger.info(f"Benchmark Complete. Average Latency: {avg_latency_ms:.2f} ms/frame")
    return avg_latency_ms

if __name__ == "__main__":
    # Initialize dummy batch: Batch Size 1, 3 Channels, 224x224 Resolution
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Instantiate engine
    engine = OptimizedVisionExtractor(embedding_dim=512)
    
    # Benchmark Baseline (FP32)
    baseline_latency = benchmark_inference(engine, dummy_input)
    
    # Apply Quantization
    engine.quantize_model()
    
    # Benchmark Optimized Version
    optimized_latency = benchmark_inference(engine, dummy_input)
    
    speedup = baseline_latency / optimized_latency
    print(f"\n--- Performance Summary ---")
    print(f"Baseline Latency:  {baseline_latency:.2f} ms")
    print(f"Optimized Latency: {optimized_latency:.2f} ms")
    print(f"Speedup Factor:    {speedup:.2f}x")