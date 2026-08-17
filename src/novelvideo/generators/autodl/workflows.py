"""Declarative AutoDL workflow definitions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AutoDLWorkflowSpec:
    id: str
    submit_path: str
    result_path_template: str = "/api/v1/comfyui/comfyui_workflow/result/{task_id}"
    output_type: str = "video"


MINIMAX_H3_IMAGE_REFERENCE = AutoDLWorkflowSpec(
    id="minimax_h3_image_reference",
    submit_path="/api/v1/comfyui/comfyui_workflow/minimax_h3_lightx2v_v5",
)
