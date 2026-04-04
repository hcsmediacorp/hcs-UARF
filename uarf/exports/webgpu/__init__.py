"""
WebGPU Support Module for UARF
Enables browser-based inference using WebGPU API

Features:
- WGSL shader generation
- Model conversion for WebGPU
- Browser runtime integration
- Cross-platform compatibility
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class WebGPUConfig:
    """WebGPU configuration"""
    precision: str = "fp16"  # fp32, fp16
    max_texture_size: int = 4096
    storage_buffer_enabled: bool = True
    compute_shader_version: str = "wgsl"


class WebGPUExporter:
    """
    Exports models for WebGPU inference
    
    Generates:
    - Model weights in binary format
    - WGSL compute shaders
    - JavaScript runtime wrapper
    """
    
    def __init__(self, config: Optional[WebGPUConfig] = None):
        """
        Initialize WebGPU exporter
        
        Args:
            config: WebGPU configuration
        """
        self.config = config or WebGPUConfig()
        logger.info(f"Initialized WebGPU exporter ({self.config.precision})")
    
    def export(self, model_state: Dict[str, Any], config: Dict[str, Any],
               output_dir: str) -> Path:
        """
        Export model for WebGPU
        
        Args:
            model_state: Model state dict
            config: Model configuration
            output_dir: Output directory
            
        Returns:
            Path to output directory
        """
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting model for WebGPU: {output}")
        
        # Export weights
        weights_path = self._export_weights(model_state, output / "weights.bin")
        
        # Export model config
        config_path = self._export_config(config, output / "model.json")
        
        # Generate WGSL shaders
        shader_path = self._generate_shaders(config, output / "inference.wgsl")
        
        # Generate JavaScript runtime
        js_path = self._generate_runtime(config, output / "webgpu-inference.js")
        
        # Generate HTML example
        html_path = self._generate_html(output / "index.html")
        
        logger.info(f"WebGPU export complete: {output}")
        return output
    
    def _export_weights(self, model_state: Dict[str, Any], 
                        output_path: Path) -> Path:
        """Export weights to binary format"""
        import numpy as np
        
        with open(output_path, 'wb') as f:
            # Write header
            f.write(b'WGPU')  # Magic number
            f.write(len(model_state).to_bytes(4, 'little'))
            
            for name, tensor in model_state.items():
                # Convert to numpy
                if hasattr(tensor, 'cpu'):
                    tensor = tensor.cpu()
                if hasattr(tensor, 'numpy'):
                    tensor = tensor.numpy()
                
                # Quantize to FP16 if configured
                if self.config.precision == "fp16":
                    tensor = tensor.astype(np.float16)
                else:
                    tensor = tensor.astype(np.float32)
                
                # Write tensor
                name_bytes = name.encode('utf-8')
                f.write(len(name_bytes).to_bytes(4, 'little'))
                f.write(name_bytes)
                f.write(len(tensor.shape).to_bytes(4, 'little'))
                for dim in tensor.shape:
                    f.write(dim.to_bytes(4, 'little'))
                f.write(tensor.dtype.str.encode('utf-8').ljust(4, b'\x00'))
                f.write(tensor.tobytes())
        
        logger.info(f"Weights exported: {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")
        return output_path
    
    def _export_config(self, config: Dict[str, Any], output_path: Path) -> Path:
        """Export model configuration"""
        webgpu_config = {
            **config,
            "webgpu": {
                "precision": self.config.precision,
                "max_texture_size": self.config.max_texture_size,
                "shader_format": "wgsl"
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(webgpu_config, f, indent=2)
        
        return output_path
    
    def _generate_shaders(self, config: Dict[str, Any], 
                          output_path: Path) -> Path:
        """Generate WGSL compute shaders"""
        
        hidden_size = config.get("hidden_size", 512)
        num_heads = config.get("num_heads", 8)
        head_dim = hidden_size // num_heads
        
        shader_code = f'''// WebGPU Inference Shader for UARF
// Generated automatically

// Configuration
const HIDDEN_SIZE: u32 = {hidden_size};
const NUM_HEADS: u32 = {num_heads};
const HEAD_DIM: u32 = {head_dim};

// Storage buffers
@group(0) @binding(0)
var<storage, read> input_buffer: array<f32>;

@group(0) @binding(1)
var<storage, read> weights_buffer: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output_buffer: array<f32>;

@group(0) @binding(3)
var<uniform> params: vec2<f32>;

// Matrix multiplication kernel
@compute @workgroup_size(64)
fn matmul_kernel(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {{
    let row = global_id.x;
    let col = global_id.y;
    
    if (row >= HIDDEN_SIZE || col >= HIDDEN_SIZE) {{
        return;
    }}
    
    var sum: f32 = 0.0;
    for (var k: u32 = 0; k < HIDDEN_SIZE; k = k + 1) {{
        let a = input_buffer[row * HIDDEN_SIZE + k];
        let b = weights_buffer[k * HIDDEN_SIZE + col];
        sum = sum + a * b;
    }}
    
    output_buffer[row * HIDDEN_SIZE + col] = sum;
}}

// Attention kernel (simplified)
@compute @workgroup_size(128)
fn attention_kernel(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {{
    let idx = global_id.x;
    
    if (idx >= HIDDEN_SIZE) {{
        return;
    }}
    
    // Simplified softmax
    var max_val: f32 = -1.0e38;
    var sum_exp: f32 = 0.0;
    
    for (var i: u32 = 0; i < HIDDEN_SIZE; i = i + 1) {{
        let val = input_buffer[idx * HIDDEN_SIZE + i];
        if (val > max_val) {{
            max_val = val;
        }}
    }}
    
    for (var i: u32 = 0; i < HIDDEN_SIZE; i = i + 1) {{
        let val = input_buffer[idx * HIDDEN_SIZE + i];
        let exp_val = exp(val - max_val);
        output_buffer[idx * HIDDEN_SIZE + i] = exp_val;
        sum_exp = sum_exp + exp_val;
    }}
    
    // Normalize
    for (var i: u32 = 0; i < HIDDEN_SIZE; i = i + 1) {{
        output_buffer[idx * HIDDEN_SIZE + i] = output_buffer[idx * HIDDEN_SIZE + i] / sum_exp;
    }}
}}

// Entry point
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    matmul_kernel(global_id);
}}
'''
        
        with open(output_path, 'w') as f:
            f.write(shader_code)
        
        return output_path
    
    def _generate_runtime(self, config: Dict[str, Any], 
                          output_path: Path) -> Path:
        """Generate JavaScript runtime"""
        
        js_code = '''/**
 * WebGPU Inference Runtime for UARF
 * Auto-generated by UARF WebGPU Exporter
 */

class WebGPURuntime {
    constructor(config) {
        this.config = config;
        this.device = null;
        this.pipeline = null;
        this.bindGroup = null;
        this.weightsBuffer = null;
        this.inputBuffer = null;
        this.outputBuffer = null;
    }

    async initialize() {
        // Request WebGPU adapter
        if (!navigator.gpu) {
            throw new Error('WebGPU not supported in this browser');
        }

        const adapter = await navigator.gpu.requestAdapter({
            powerPreference: 'high-performance'
        });

        if (!adapter) {
            throw new Error('Failed to get GPU adapter');
        }

        this.device = await adapter.requestDevice();

        // Load model weights
        await this.loadWeights();

        // Create compute pipeline
        await this.createPipeline();

        console.log('WebGPU runtime initialized');
    }

    async loadWeights() {
        const response = await fetch('weights.bin');
        const buffer = await response.arrayBuffer();
        
        this.weightsBuffer = this.device.createBuffer({
            size: buffer.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });

        new Uint8Array(this.weightsBuffer.getMappedRange()).set(new Uint8Array(buffer));
        this.weightsBuffer.unmap();
    }

    async createPipeline() {
        const shaderModule = await this.loadShader();

        this.pipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'main'
            }
        });
    }

    async loadShader() {
        const response = await fetch('inference.wgsl');
        const shaderCode = await response.text();
        
        return this.device.createShaderModule({
            code: shaderCode
        });
    }

    async infer(inputData) {
        if (!this.device) {
            throw new Error('Runtime not initialized');
        }

        // Create input buffer
        const inputBuffer = this.device.createBuffer({
            size: inputData.length * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });

        new Float32Array(inputBuffer.getMappedRange()).set(inputData);
        inputBuffer.unmap();

        // Create output buffer
        const outputSize = inputData.length * 4;
        const outputBuffer = this.device.createBuffer({
            size: outputSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });

        // Create bind group
        const bindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: inputBuffer } },
                { binding: 1, resource: { buffer: this.weightsBuffer } },
                { binding: 2, resource: { buffer: outputBuffer } }
            ]
        });

        // Encode commands
        const commandEncoder = this.device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(this.pipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(Math.ceil(inputData.length / 64));
        passEncoder.end();

        // Add readback
        const readBuffer = this.device.createBuffer({
            size: outputSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        commandEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, outputSize);

        // Submit and read results
        await this.device.queue.submit([commandEncoder.finish()]);
        await readBuffer.mapAsync(GPUMapMode.READ);

        const result = new Float32Array(readBuffer.getMappedRange()).slice();
        readBuffer.unmap();

        return result;
    }

    async generate(prompt, maxTokens = 100) {
        // Tokenize input
        const tokens = this.tokenize(prompt);
        let output = tokens;

        for (let i = 0; i < maxTokens; i++) {
            // Run inference
            const logits = await this.infer(output);
            
            // Sample next token
            const nextToken = this.sample(logits);
            output.push(nextToken);

            // Check for EOS
            if (nextToken === 2) {  // Assuming 2 is EOS token
                break;
            }
        }

        return this.detokenize(output);
    }

    tokenize(text) {
        // Simple tokenizer (replace with actual tokenizer)
        return text.split(' ').map(w => w.length % 100);
    }

    sample(logits) {
        // Simple argmax sampling
        let maxIdx = 0;
        let maxVal = -Infinity;
        for (let i = 0; i < logits.length; i++) {
            if (logits[i] > maxVal) {
                maxVal = logits[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    detokenize(tokens) {
        // Simple detokenizer
        return tokens.join(' ');
    }
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { WebGPURuntime };
}
'''
        
        with open(output_path, 'w') as f:
            f.write(js_code)
        
        return output_path
    
    def _generate_html(self, output_path: Path) -> Path:
        """Generate example HTML page"""
        
        html_code = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UARF WebGPU Inference</title>
    <style>
        body { font-family: system-ui; max-width: 800px; margin: 40px auto; padding: 20px; }
        textarea { width: 100%; height: 100px; margin: 10px 0; }
        button { padding: 10px 20px; font-size: 16px; cursor: pointer; }
        #output { background: #f5f5f5; padding: 20px; border-radius: 8px; min-height: 100px; }
        .status { color: #666; margin: 10px 0; }
    </style>
</head>
<body>
    <h1>UARF WebGPU Inference</h1>
    <p class="status" id="status">Initializing WebGPU...</p>
    
    <textarea id="prompt" placeholder="Enter your prompt here..."></textarea>
    <button onclick="generate()" id="generateBtn" disabled>Generate</button>
    
    <h2>Output:</h2>
    <div id="output"></div>

    <script src="webgpu-inference.js"></script>
    <script>
        let runtime = null;

        async function init() {
            try {
                runtime = new WebGPURuntime({});
                await runtime.initialize();
                document.getElementById('status').textContent = 'WebGPU ready!';
                document.getElementById('generateBtn').disabled = false;
            } catch (error) {
                document.getElementById('status').textContent = 'Error: ' + error.message;
            }
        }

        async function generate() {
            const prompt = document.getElementById('prompt').value;
            const outputDiv = document.getElementById('output');
            const btn = document.getElementById('generateBtn');
            
            btn.disabled = true;
            outputDiv.textContent = 'Generating...';
            
            try {
                const result = await runtime.generate(prompt, maxTokens=50);
                outputDiv.textContent = result;
            } catch (error) {
                outputDiv.textContent = 'Error: ' + error.message;
            }
            
            btn.disabled = false;
        }

        init();
    </script>
</body>
</html>
'''
        
        with open(output_path, 'w') as f:
            f.write(html_code)
        
        return output_path


def export_for_webgpu(model_state: Dict[str, Any], config: Dict[str, Any],
                      output_dir: str, precision: str = "fp16") -> Path:
    """
    Convenience function to export model for WebGPU
    
    Args:
        model_state: Model state dict
        config: Model configuration
        output_dir: Output directory
        precision: Precision (fp32, fp16)
        
    Returns:
        Path to output directory
    """
    cfg = WebGPUConfig(precision=precision)
    exporter = WebGPUExporter(config=cfg)
    return exporter.export(model_state, config, output_dir)
