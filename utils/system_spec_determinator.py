#!/usr/bin/env python3
"""
System Spec Determinator for Apple Silicon Macs
Determines if a Mac has low or high capability for AI model execution
"""

import platform
import subprocess
import sys
import re
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any


class MacCapability(Enum):
    LOW = "low"
    HIGH = "high"


@dataclass
class SystemSpecs:
    """System specifications for Apple Silicon Macs"""
    chip_model: str
    cpu_cores: int
    gpu_cores: int
    memory_gb: int
    
    
class SystemSpecDeterminator:
    """
    Determines Mac capability level for AI model execution.
    
    Classification Rules:
    - All Pro/Max/Ultra variants are HIGH (except M1 Pro)
    - M1 Pro and base models (M1, M2, M3, M4) require:
      * 16GB+ RAM, 10+ CPU cores, 12+ GPU cores for HIGH
      * Otherwise LOW
    
    Low capability: Can run Gemma 3n E2B for all tasks
    High capability: Can load both E2B and E4B, using E2B for low power tasks
                     and E4B for accuracy-reliant tasks
    """
    
    # High-spec thresholds for base models and M1 Pro
    HIGH_SPEC_THRESHOLDS = {
        'min_memory_gb': 16,
        'min_cpu_cores': 10,
        'min_gpu_cores': 12
    }
    
    def __init__(self):
        self.system_specs = None
        
    def _run_command(self, command: str) -> str:
        """Execute a system command and return output"""
        try:
            result = subprocess.run(
                command.split(),
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return ""
    
    def _get_chip_model(self) -> str:
        """Get the Apple Silicon chip model"""
        try:
            output = self._run_command("sysctl -n machdep.cpu.brand_string")
            if "Apple" in output:
                # Extract chip model (e.g., "Apple M3 Pro")
                match = re.search(r'Apple (M\d+(?:\s+\w+)?)', output)
                if match:
                    return match.group(1).strip()
            return "Unknown"
        except:
            return "Unknown"
    
    def _get_memory_gb(self) -> int:
        """Get total system memory in GB"""
        try:
            output = self._run_command("sysctl -n hw.memsize")
            bytes_memory = int(output)
            return bytes_memory // (1024 ** 3)  # Convert to GB
        except:
            return 0
    
    def _get_cpu_cores(self) -> int:
        """Get number of CPU cores"""
        try:
            output = self._run_command("sysctl -n hw.ncpu")
            return int(output)
        except:
            return 0
    
    def _get_gpu_cores(self) -> int:
        """Get actual GPU core count from system profiler"""
        try:
            # Use system_profiler to get GPU information
            output = self._run_command("system_profiler SPDisplaysDataType")
            
            # Look for GPU core count patterns
            patterns = [
                r'Total Number of Cores:\s*(\d+)',
                r'GPU Cores:\s*(\d+)',
                r'Cores:\s*(\d+)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, output)
                if match:
                    return int(match.group(1))
            
            # Fallback: try to extract from Metal information
            metal_output = self._run_command("system_profiler SPDisplaysDataType -xml")
            if metal_output:
                # Look for Metal-related GPU core information
                core_match = re.search(r'<key>sppci_cores</key>\s*<integer>(\d+)</integer>', metal_output)
                if core_match:
                    return int(core_match.group(1))
            
            return 0
        except:
            return 0
    
    
    def get_system_specs(self) -> SystemSpecs:
        """Get complete system specifications"""
        if self.system_specs is None:
            chip_model = self._get_chip_model()
            memory_gb = self._get_memory_gb()
            cpu_cores = self._get_cpu_cores()
            gpu_cores = self._get_gpu_cores()
            
            self.system_specs = SystemSpecs(
                chip_model=chip_model,
                cpu_cores=cpu_cores,
                gpu_cores=gpu_cores,
                memory_gb=memory_gb
            )
        
        return self.system_specs
    
    def determine_capability(self) -> MacCapability:
        """
        Determine Mac capability level for AI model execution.
        
        Classification Rules:
        1. All Pro/Max/Ultra variants are HIGH (except M1 Pro)
        2. M1 Pro and base models require 16GB+ RAM, 10+ CPU cores, 12+ GPU cores for HIGH
        3. Otherwise LOW
        
        Returns:
            MacCapability.LOW: Can run Gemma 3n E2B (5B params, 2GB VRAM)
            MacCapability.HIGH: Can run both E2B and E4B (8B params, 3GB VRAM)
        """
        specs = self.get_system_specs()
        
        # Check if it's a Pro/Max/Ultra variant (excluding M1 Pro)
        if any(variant in specs.chip_model for variant in ['Max', 'Ultra']):
            return MacCapability.HIGH
        
        # M2 Pro, M3 Pro, M4 Pro are automatically HIGH
        if 'Pro' in specs.chip_model and not specs.chip_model.startswith('M1'):
            return MacCapability.HIGH
        
        # For base models (M1, M2, M3, M4) and M1 Pro, check specific thresholds
        high_spec_criteria = [
            specs.memory_gb >= self.HIGH_SPEC_THRESHOLDS['min_memory_gb'],
            specs.cpu_cores >= self.HIGH_SPEC_THRESHOLDS['min_cpu_cores'],
            specs.gpu_cores >= self.HIGH_SPEC_THRESHOLDS['min_gpu_cores']
        ]
        
        if all(high_spec_criteria):
            return MacCapability.HIGH
        
        return MacCapability.LOW
    
    def get_recommended_models(self) -> Dict[str, Any]:
        """Get recommended AI models based on system capability"""
        capability = self.determine_capability()
        specs = self.get_system_specs()
        
        if capability == MacCapability.HIGH:
            return {
                'primary_model': 'gemma-3n-e4b',
                'fallback_model': 'gemma-3n-e2b',
                'strategy': 'adaptive',
                'description': 'Use E4B for complex tasks, E2B for simple/low-power tasks',
                'max_concurrent_models': 1,
                'memory_allocation': {
                    'e4b': '3GB',
                    'e2b': '2GB'
                }
            }
        else:
            return {
                'primary_model': 'gemma-3n-e2b',
                'fallback_model': None,
                'strategy': 'single_model',
                'description': 'Use E2B for all tasks for optimal performance',
                'max_concurrent_models': 1,
                'memory_allocation': {
                    'e2b': '2GB'
                }
            }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate a detailed performance report"""
        specs = self.get_system_specs()
        capability = self.determine_capability()
        recommendations = self.get_recommended_models()
        
        return {
            'system_specs': {
                'chip_model': specs.chip_model,
                'cpu_cores': specs.cpu_cores,
                'gpu_cores': specs.gpu_cores,
                'memory_gb': specs.memory_gb
            },
            'capability_level': capability.value,
            'recommendations': recommendations,
            'performance_expectations': {
                'e2b_inference_speed': 'Fast' if capability == MacCapability.HIGH else 'Moderate',
                'e4b_support': capability == MacCapability.HIGH,
                'concurrent_processing': capability == MacCapability.HIGH,
                'memory_efficiency': 'High' if specs.memory_gb >= 16 else 'Moderate'
            },
            'classification_details': {
                'is_pro_max_ultra': any(variant in specs.chip_model for variant in ['Pro', 'Max', 'Ultra']),
                'is_m1_pro': specs.chip_model == 'M1 Pro',
                'meets_high_spec_thresholds': (
                    specs.memory_gb >= self.HIGH_SPEC_THRESHOLDS['min_memory_gb'] and
                    specs.cpu_cores >= self.HIGH_SPEC_THRESHOLDS['min_cpu_cores'] and
                    specs.gpu_cores >= self.HIGH_SPEC_THRESHOLDS['min_gpu_cores']
                )
            }
        }


def get_capability() -> str:
    """Get system capability as a simple string for easy integration"""
    determinator = SystemSpecDeterminator()
    capability = determinator.determine_capability()
    return capability.value


def manual_check():
    """Detailed manual check with full system analysis and output"""
    determinator = SystemSpecDeterminator()
    
    print("🔍 Apple Silicon Mac System Analysis")
    print("=" * 50)
    
    # Get system specs
    specs = determinator.get_system_specs()
    print(f"Chip Model: {specs.chip_model}")
    print(f"CPU Cores: {specs.cpu_cores}")
    print(f"GPU Cores: {specs.gpu_cores}")
    print(f"Memory: {specs.memory_gb} GB")
    
    print("\n🎯 AI Model Capability Assessment")
    print("=" * 50)
    
    # Determine capability
    capability = determinator.determine_capability()
    print(f"Capability Level: {capability.value.upper()}")
    
    # Get recommendations
    recommendations = determinator.get_recommended_models()
    print(f"Primary Model: {recommendations['primary_model']}")
    print(f"Strategy: {recommendations['strategy']}")
    print(f"Description: {recommendations['description']}")
    
    # Generate full report
    print("\n📊 Performance Report")
    print("=" * 50)
    report = determinator.get_performance_report()
    
    expectations = report['performance_expectations']
    print(f"E2B Inference Speed: {expectations['e2b_inference_speed']}")
    print(f"E4B Support: {'Yes' if expectations['e4b_support'] else 'No'}")
    print(f"Concurrent Processing: {'Yes' if expectations['concurrent_processing'] else 'No'}")
    print(f"Memory Efficiency: {expectations['memory_efficiency']}")
    
    # Classification details
    print("\n🔍 Classification Details")
    print("=" * 50)
    details = report['classification_details']
    print(f"Is Pro/Max/Ultra: {'Yes' if details['is_pro_max_ultra'] else 'No'}")
    print(f"Is M1 Pro: {'Yes' if details['is_m1_pro'] else 'No'}")
    print(f"Meets High-Spec Thresholds: {'Yes' if details['meets_high_spec_thresholds'] else 'No'}")


def main():
    """Simple main function showing capability result"""
    capability = get_capability()
    print(f"System capability: {capability}")


if __name__ == "__main__":
    main()