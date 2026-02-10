# Load model
from sonics import HFAudioClassifier
model = HFAudioClassifier.from_pretrained("awsaf49/sonics-spectttra-gamma-120s")

print(model)
print(model.config)