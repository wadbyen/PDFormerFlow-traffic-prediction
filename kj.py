import os
import torch

# Create the directory if it doesn't exist
if not os.path.exists('dict'):
    os.makedirs('dict', exist_ok=True)

# Dummy model for testing
dummy_model = torch.nn.Linear(2, 2)

# Save the model state
torch.save(dummy_model.state_dict(), 'dict/test_model_state.pkl')

print("File saved successfully!")