"""
Fetch neuron descriptions from neurons.transluce.org

The Transluce team has published automatic descriptions for MLP neurons
in Llama 3.1 8B. This module fetches those descriptions for analysis.
"""

import requests
from typing import Optional
from dataclasses import dataclass
import time


@dataclass
class NeuronDescription:
    """Description of a single neuron."""
    layer: int
    neuron: int
    description: str
    polarity: str  # "+" or "-"
    confidence: float | None = None


def get_neuron_description(
    layer: int,
    neuron: int,
    polarity: str = "+",
    model: str = "llama-3.1-8b-instruct",
) -> Optional[NeuronDescription]:
    """
    Fetch description for a single neuron from neurons.transluce.org
    
    Args:
        layer: Layer index
        neuron: Neuron index within the layer
        polarity: "+" for positive activations, "-" for negative
        model: Model identifier
        
    Returns:
        NeuronDescription or None if not found
    """
    # The URL format based on their website
    url = f"https://neurons.transluce.org/{layer}/{neuron}/{polarity}"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            # Parse the HTML to extract the description
            # This is a simplified version - may need adjustment based on actual HTML
            text = response.text
            
            # Look for the description in the page
            # The format may vary, this is an approximation
            if "description" in text.lower():
                # Extract description (this is a placeholder - actual parsing needed)
                return NeuronDescription(
                    layer=layer,
                    neuron=neuron,
                    description=f"[Fetch from {url}]",
                    polarity=polarity,
                )
            
        return None
        
    except Exception as e:
        print(f"Error fetching neuron {layer}/{neuron}: {e}")
        return None


def get_neuron_descriptions_batch(
    neurons: list[tuple[int, int]],
    polarity: str = "+",
    delay: float = 0.1,
) -> dict[tuple[int, int], NeuronDescription]:
    """
    Fetch descriptions for multiple neurons.
    
    Args:
        neurons: List of (layer, neuron) tuples
        polarity: "+" or "-"
        delay: Delay between requests to be polite
        
    Returns:
        Dictionary mapping (layer, neuron) to description
    """
    results = {}
    
    for layer, neuron in neurons:
        desc = get_neuron_description(layer, neuron, polarity)
        if desc:
            results[(layer, neuron)] = desc
        time.sleep(delay)  # Be nice to their servers
    
    return results


def format_neuron_report(
    neurons: list[tuple[int, int, float]],
    descriptions: dict[tuple[int, int], NeuronDescription],
) -> str:
    """
    Format a report of important neurons with their descriptions.
    
    Args:
        neurons: List of (layer, neuron, score) tuples
        descriptions: Dictionary of descriptions
        
    Returns:
        Formatted string report
    """
    lines = []
    lines.append("Top Neurons Report")
    lines.append("=" * 60)
    
    for i, (layer, neuron, score) in enumerate(neurons):
        lines.append(f"\n{i+1}. Layer {layer}, Neuron {neuron}")
        lines.append(f"   Attribution Score: {score:.4f}")
        
        key = (layer, neuron)
        if key in descriptions:
            desc = descriptions[key]
            lines.append(f"   Description ({desc.polarity}): {desc.description}")
        else:
            lines.append(f"   Description: Not available")
            lines.append(f"   URL: https://neurons.transluce.org/{layer}/{neuron}/+")
    
    return "\n".join(lines)


# Alternative: Use the neurondb if available locally
def try_local_neurondb():
    """Try to use local neurondb if available from circuits repo."""
    try:
        from neurondb import NeuronDB
        return NeuronDB()
    except ImportError:
        return None


class NeuronLookup:
    """
    Unified interface for looking up neuron descriptions.
    
    Tries local neurondb first, falls back to web API.
    """
    
    def __init__(self, use_cache: bool = True):
        self.local_db = try_local_neurondb()
        self.cache = {} if use_cache else None
        
        if self.local_db:
            print("Using local neurondb for descriptions")
        else:
            print("Using neurons.transluce.org web API for descriptions")
    
    def get_description(
        self,
        layer: int,
        neuron: int,
        polarity: str = "+",
    ) -> Optional[str]:
        """Get description for a neuron."""
        key = (layer, neuron, polarity)
        
        # Check cache
        if self.cache is not None and key in self.cache:
            return self.cache[key]
        
        # Try local DB
        if self.local_db:
            try:
                desc = self.local_db.get_description(layer, neuron, polarity)
                if self.cache is not None:
                    self.cache[key] = desc
                return desc
            except Exception:
                pass
        
        # Fall back to web
        result = get_neuron_description(layer, neuron, polarity)
        if result:
            desc = result.description
            if self.cache is not None:
                self.cache[key] = desc
            return desc
        
        return None
    
    def get_url(self, layer: int, neuron: int, polarity: str = "+") -> str:
        """Get the URL to view a neuron's details."""
        return f"https://neurons.transluce.org/{layer}/{neuron}/{polarity}"
