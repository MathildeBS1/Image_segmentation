import numpy as np

def generate_clicks(mask, N, M):
    """
    Generates N positive and M negative click coordinates from a segmentation mask.
    
    """
    # Get the indices of positive and negative pixels
    positive_indices = np.column_stack(np.where(mask == 1))
    negative_indices = np.column_stack(np.where(mask == 0))
    
    # Sample N positive and M negative coordinates randomly
    positive_clicks = positive_indices[np.random.choice(positive_indices.shape[0], N, replace=False)]
    negative_clicks = negative_indices[np.random.choice(negative_indices.shape[0], M, replace=False)]
    
    # Convert to list of tuples
    positive_clicks = [tuple(coord) for coord in positive_clicks]
    negative_clicks = [tuple(coord) for coord in negative_clicks]
    
    return positive_clicks, negative_clicks