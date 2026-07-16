from pathlib import Path
from PIL import Image

screenshots_dir = Path("reports/screenshots")
gif_path = screenshots_dir / "app_demo.gif"

frames_order = [
    screenshots_dir / "operator_dashboard.png",
    screenshots_dir / "technician_plan.png",
    screenshots_dir / "ml_pipeline.png",
    screenshots_dir / "model_comparison.png",
    screenshots_dir / "error_analysis.png",
]

existing_frames = [path for path in frames_order if path.exists()]

if len(existing_frames) < 2:
    raise FileNotFoundError("Нужно минимум 2 скриншота для GIF")

target_width = 1200
target_height = 750
transition_steps = 12
transition_duration = 90
hold_duration = 1400

base_images = []

for path in existing_frames:
    image = Image.open(path).convert("RGB")
    image.thumbnail((target_width, target_height))
    canvas = Image.new("RGB", (target_width, target_height), "white")
    x = (target_width - image.width) // 2
    y = (target_height - image.height) // 2
    canvas.paste(image, (x, y))
    base_images.append(canvas)

gif_frames = []
durations = []

for i in range(len(base_images)):
    current_image = base_images[i]
    next_image = base_images[(i + 1) % len(base_images)]

    gif_frames.append(current_image)
    durations.append(hold_duration)

    for step in range(1, transition_steps + 1):
        alpha = step / (transition_steps + 1)
        blended = Image.blend(current_image, next_image, alpha)
        gif_frames.append(blended)
        durations.append(transition_duration)

gif_frames[0].save(
    gif_path,
    save_all=True,
    append_images=gif_frames[1:],
    duration=durations,
    loop=0,
    optimize=True,
)

print(f"GIF обновлён: {gif_path}")
