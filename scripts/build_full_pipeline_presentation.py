"""Combine the stored live-planning presentation and verified contact grasp."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "presentation_demo"
PLANNING_VIDEO = OUTPUT_ROOT / "active_view_pipeline_presentation.mp4"
GRASP_ROOT = ROOT / "outputs" / "rg6_physics" / "actual_contact_grasp_seed000"
GRASP_VIDEO = GRASP_ROOT / "rg6_actual_contact_grasp.mp4"
GRASP_RESULT = GRASP_ROOT / "result.json"
TITLE_PATH = OUTPUT_ROOT / "contact_grasp_transition.png"
OUTPUT_PATH = OUTPUT_ROOT / "full_pipeline_with_contact_grasp.mp4"
METADATA_PATH = OUTPUT_PATH.with_suffix(".json")


def font(size: int):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


for required in (PLANNING_VIDEO, GRASP_VIDEO, GRASP_RESULT):
    if not required.is_file():
        raise FileNotFoundError(required)

grasp = json.loads(GRASP_RESULT.read_text(encoding="utf-8"))
if grasp.get("status") != "completed" or not grasp.get("lift_verified"):
    raise RuntimeError("Only a verified grasp result may enter the presentation video")

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
image = Image.new("RGB", (1920, 1080), (12, 18, 29))
draw = ImageDraw.Draw(image)
headline = "Terminal GRASP: actual articulated RG6"
summary = (
    f"Bilateral contacts: {grasp['contacts']['left_event_count']} / "
    f"{grasp['contacts']['right_event_count']}    "
    f"Verified lift: {grasp['verified_lift_delta_m']:.3f} m"
)
notice = (
    "Isaac Sim 6 RG6 joint + collision meshes — no target attachment or pose copying\n"
    "Deterministic pipeline-validation pilot; not a final paper result"
)
draw.text((120, 260), headline, fill=(244, 248, 255), font=font(72))
draw.rounded_rectangle(
    (115, 420, 1805, 580),
    radius=24,
    fill=(25, 73, 105),
    outline=(66, 205, 175),
    width=5,
)
draw.text((160, 465), summary, fill=(255, 255, 255), font=font(46))
draw.multiline_text(
    (125, 700),
    notice,
    fill=(184, 198, 215),
    font=font(34),
    spacing=18,
)
TITLE_PATH.parent.mkdir(parents=True, exist_ok=True)
image.save(TITLE_PATH)

filter_graph = (
    "[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
    "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,fps=6,"
    "format=yuv420p,setpts=PTS-STARTPTS[v0];"
    "[1:v]scale=1920:1080,fps=6,format=yuv420p,"
    "trim=duration=3,setpts=PTS-STARTPTS[v1];"
    "[2:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
    "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,fps=6,"
    "format=yuv420p,setpts=PTS-STARTPTS[v2];"
    "[v0][v1][v2]concat=n=3:v=1:a=0[outv]"
)
command = [
    "ffmpeg",
    "-y",
    "-i",
    str(PLANNING_VIDEO),
    "-loop",
    "1",
    "-t",
    "3",
    "-i",
    str(TITLE_PATH),
    "-i",
    str(GRASP_VIDEO),
    "-filter_complex",
    filter_graph,
    "-map",
    "[outv]",
    "-c:v",
    "libx264",
    "-preset",
    "slow",
    "-crf",
    "18",
    "-pix_fmt",
    "yuv420p",
    "-movflags",
    "+faststart",
    str(OUTPUT_PATH),
]
completed = subprocess.run(command, capture_output=True, text=True, check=False)
if completed.returncode != 0:
    raise RuntimeError(completed.stderr)

probe = subprocess.run(
    [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,size:stream=codec_name,width,height,avg_frame_rate,nb_frames",
        "-of",
        "json",
        str(OUTPUT_PATH),
    ],
    capture_output=True,
    text=True,
    check=True,
)
metadata = {
    "schema_version": "full-pipeline-presentation-v2",
    "status": "completed",
    "output_path": str(OUTPUT_PATH.resolve()),
    "sources": [
        str(PLANNING_VIDEO.resolve()),
        str(GRASP_VIDEO.resolve()),
        str(GRASP_RESULT.resolve()),
    ],
    "grasp_summary": {
        "bilateral_contact_events": [
            grasp["contacts"]["left_event_count"],
            grasp["contacts"]["right_event_count"],
        ],
        "verified_lift_delta_m": grasp["verified_lift_delta_m"],
        "target_attachment_used": grasp["target_dynamics"][
            "explicit_target_attachment_used"
        ],
        "target_pose_copying_used": grasp["target_dynamics"][
            "target_pose_copying_used"
        ],
    },
    "valid_for_final_evaluation": False,
    "ffprobe": json.loads(probe.stdout),
}
METADATA_PATH.write_text(
    json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
)
print(f"FULL_PIPELINE_PRESENTATION={OUTPUT_PATH}", flush=True)
print(f"FULL_PIPELINE_PRESENTATION_METADATA={METADATA_PATH}", flush=True)
