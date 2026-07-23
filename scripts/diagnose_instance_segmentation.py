"""Minimal Isaac Sim 6.0 instance-segmentation smoke test."""

import json
from pathlib import Path

from isaacsim import SimulationApp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = PROJECT_ROOT / "instance_segmentation_diagnostic.json"
app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "active_gpu": 0})

try:
    import omni.replicator.core as rep
    import omni.usd

    omni.usd.get_context().new_stage()
    rep.functional.create.xform(name="World")
    cube = rep.functional.create.cube(
        name="DiagnosticCube",
        parent="/World",
        position=(0.0, 0.0, 0.0),
        semantics={"class": "diagnostic_cube"},
    )
    camera = rep.functional.create.camera(
        name="Camera",
        parent="/World",
        position=(2.0, 2.0, 2.0),
        look_at=(0.0, 0.0, 0.0),
    )
    render_product = rep.create.render_product(camera, (320, 240))
    annotator = rep.AnnotatorRegistry.get_annotator(
        "instance_segmentation_fast", init_params={"colorize": False}
    )
    annotator.attach(render_product)
    for _ in range(5):
        app.update()
    rep.orchestrator.step(rt_subframes=2)
    data = annotator.get_data()
    labels = data.get("info", {}).get("idToLabels", {})
    result = {
        "status": "passed",
        "shape": list(data["data"].shape),
        "id_to_labels": labels,
        "has_diagnostic_cube": any(
            isinstance(value, dict) and value.get("class") == "diagnostic_cube"
            for value in labels.values()
        ),
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
finally:
    app.close()
