from transformers import AutoModelForCausalLM

REPO = "noumenon-labs/Fleet-Semantic-4B"

model = AutoModelForCausalLM.from_pretrained(
    REPO,
    trust_remote_code=True,
    device_map="auto",
)

print("Installed experts:")
print(model.experts())

# Export an installed specialist.
model.export_adapter(
    "structured",
    "structured.fleet",
)

# Remove it from the live model.
model.uninstall("structured")
print("After uninstall:")
print(model.experts())

# Restore it from the exported package.
model.install("structured.fleet")
print("After reinstall:")
print(model.experts())
