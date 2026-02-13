from openvla import load_vla

model = load_vla("openvla-7b")
action = model.predict(
    image=camera_observation,
    instruction="pick up the red block"
)