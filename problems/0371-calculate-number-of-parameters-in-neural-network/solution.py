def calculate_parameters(layers: list[dict]) -> int:
    """
    Calculate the total number of trainable parameters in a neural network.

    Args:
        layers: List of dictionaries, each describing a layer.

    Returns:
        Total number of trainable parameters (int).
    """
    total = 0
    
    for layer in layers:
        layer_type = layer["type"]
        bias = layer.get("bias", True)

        if layer_type == "dense":
            input_size = layer["input_size"]
            output_size = layer["output_size"]

            total += input_size * output_size
            if bias:
                total += output_size

        elif layer_type == "conv2d":
            in_channels = layer["in_channels"]
            out_channels = layer["out_channels"]
            kernel_size = layer["kernel_size"]

            total += in_channels * out_channels * kernel_size**2
            if bias:
                total += out_channels

        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")    
        
    return total