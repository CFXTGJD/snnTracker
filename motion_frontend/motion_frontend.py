"""
Motion Frontend Module
This module wraps `STPFilter` and `motion_estimation` into a single
frozen frontend pipeline:

raw spikes -> STP filtering (optional) -> motion estimation (optional)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

from spkProc.filters.stp_filters_torch import STPFilter
from spkProc.motion.motion_detection import motion_estimation


class _NullLogger:
    """Minimal logger used when user does not provide one."""

    def add_image(self, *args: Any, **kwargs: Any) -> None:
        return

class MotionFrontend:
    """
    Unified motion frontend with optional STP and motion stages.

    Args:
        spike_h: Input spike frame height.
        spike_w: Input spike frame width.
        device: Torch device. If None, auto-select CUDA when available.
        diff_time: STP history window length.
        enable_stp: Enable/disable STPFilter stage.
        enable_motion: Enable/disable motion_estimation stage.
        speed_list: Optional speed levels passed to motion_estimation.
        stp_params: Optional STP parameter override dict.
        motion_params: Optional motion parameter override dict.
        logger: Optional external logger for motion visualization.
    """

    _DEFAULT_STP_PARAMS: Dict[str, Any] = {
        "u0": 0.1,
        "D": 0.02,
        "F": 1.7,
        "f": 0.11,
        "time_unit": 2000,
        "lifSize": 3,
        "filterThr": 0.1,
        "voltageMin": -8,
        "lifThr": 2,
    }

    _DEFAULT_MOTION_PARAMS: Dict[str, Any] = {
        "debug_mode": False,
        "debug_frame_target": 200,
        "debug_require_nonzero": True,
        "debug_min_spike_ratio": 0.0,
    }

    def __init__(
        self,
        spike_h: int,
        spike_w: int,
        device: Optional[torch.device] = None,
        diff_time: int = 1,
        enable_stp: bool = True,
        enable_motion: bool = True,
        speed_list: Optional[Sequence[int]] = None,
        stp_params: Optional[Dict[str, Any]] = None,
        motion_params: Optional[Dict[str, Any]] = None,
        logger: Optional[Any] = None,
    ):
        self.spike_h = int(spike_h)
        self.spike_w = int(spike_w)
        self.device = (device if device is not None else
                       torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.diff_time = int(diff_time)

        # Stage toggles for ablation or benchmarking.
        self.enable_stp = bool(enable_stp)
        self.enable_motion = bool(enable_motion)

        # Merge user configs onto defaults.
        self.stp_params = dict(self._DEFAULT_STP_PARAMS)
        if stp_params:
            self.stp_params.update(stp_params)

        self.motion_params = dict(self._DEFAULT_MOTION_PARAMS)
        if motion_params:
            self.motion_params.update(motion_params)
        if speed_list is not None:
            self.motion_params["speed_list"] = list(speed_list)

        self.logger = logger if logger is not None else _NullLogger()
        self.timestamp = 0

        self.stp_filter: Optional[STPFilter] = None
        self.motion_estimator: Optional[motion_estimation] = None

        if self.enable_stp:
            self.stp_filter = STPFilter(
                self.spike_h,
                self.spike_w,
                self.device,
                diff_time=self.diff_time,
                **self.stp_params,
            )

        if self.enable_motion:
            self.motion_estimator = motion_estimation(
                self.spike_h,
                self.spike_w,
                self.device,
                logger=self.logger,
                **self.motion_params,
            )

        self._freeze_internal_parameters()

    def _freeze_internal_parameters(self) -> None:
        """Freeze all torch.nn.Module parameters inside wrapped components."""
        components = [self.stp_filter, self.motion_estimator]
        for component in components:
            if component is None:
                continue
            for value in vars(component).values():
                if isinstance(value, torch.nn.Module):
                    value.eval()
                    for param in value.parameters():
                        param.requires_grad_(False)

    def _to_t_h_w(self, data: torch.Tensor | np.ndarray) -> torch.Tensor:
        """
        Normalize input to shape [T, H, W].

        Supported formats:
        - [T, H, W]
        - [H, W, T]
        - [H, W] (treated as T=1)
        """
        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data)
        elif torch.is_tensor(data):
            tensor = data
        else:
            raise TypeError("`data` must be a numpy.ndarray or torch.Tensor")

        if tensor.ndim == 2:
            if int(tensor.shape[0]) != self.spike_h or int(tensor.shape[1]) != self.spike_w:
                raise ValueError("2D input shape must be [H, W] matching initialized frontend size")
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3:
            # Prefer [T,H,W]. If first two dims match [H,W], treat as [H,W,T].
            if int(tensor.shape[1]) == self.spike_h and int(tensor.shape[2]) == self.spike_w:
                pass
            elif int(tensor.shape[0]) == self.spike_h and int(tensor.shape[1]) == self.spike_w:
                tensor = tensor.permute(2, 0, 1)
            else:
                raise ValueError(
                    "3D input must be [T,H,W] or [H,W,T] and match initialized frontend size"
                )
        else:
            raise ValueError("Input spikes must be 2D or 3D")

        return tensor.contiguous()

    def _to_numpy_if_needed(self, value: Any, as_numpy: bool) -> Any:
        if not as_numpy:
            return value
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        if isinstance(value, dict):
            return {k: self._to_numpy_if_needed(v, as_numpy=True) for k, v in value.items()}
        return value

    def process(
        self,
        data: torch.Tensor | np.ndarray,
        start_timestamp: Optional[int] = None,
        visualize: bool = False,
        update_stdp: bool = True,
        return_intermediate: bool = False,
        as_numpy: bool = False,
    ):
        """
        Run frontend processing for a spike sequence.

        Args:
            data: Spike tensor/array in [T,H,W], [H,W,T], or [H,W].
            start_timestamp: External timestamp base. If None, use internal counter.
            visualize: Pass-through flag to motion_estimation.local_wta.
            update_stdp: If False, skip motion weight updates and only infer motion.
            return_intermediate: If True, return dict of stage outputs.
            as_numpy: If True, convert returned tensors to numpy arrays.

        Returns:
            Default (`return_intermediate=False`):
                - motion pattern map [T,H,W,8*K] when motion stage is enabled
                - frontend spikes [T,H,W] otherwise
            Intermediate (`return_intermediate=True`): dict with stage outputs,
                including both pattern-map and legacy XY-vector motion fields.
        """
        spikes_t_h_w = self._to_t_h_w(data)
        spikes_t_h_w = spikes_t_h_w.to(self.device)
        if not torch.is_floating_point(spikes_t_h_w):
            spikes_t_h_w = spikes_t_h_w.float()

        # Keep frontend input as binary event frames.
        spikes_t_h_w = (spikes_t_h_w > 0).float()

        num_frames = int(spikes_t_h_w.shape[0])
        if start_timestamp is None:
            start_timestamp = self.timestamp
        else:
            start_timestamp = int(start_timestamp)

        collect_frontend = return_intermediate or (not self.enable_motion)
        frontend_spikes_seq = [] if collect_frontend else None
        motion_pattern_seq = [] if self.enable_motion else None
        motion_vec_seq = [] if (self.enable_motion and return_intermediate) else None
        motion_id_seq = [] if (self.enable_motion and return_intermediate) else None
        motion_vec_l1_seq = [] if (self.enable_motion and return_intermediate) else None

        with torch.no_grad():
            for i in range(num_frames):
                timestamp = start_timestamp + i
                frame = spikes_t_h_w[i]

                # Stage 1: STP filtering (optional ablation).
                if self.enable_stp and self.stp_filter is not None:
                    self.stp_filter.update_dynamics(timestamp, frame)
                    self.stp_filter.local_connect(self.stp_filter.filter_spk)
                    stage_spikes = self.stp_filter.lif_spk
                else:
                    stage_spikes = frame

                if collect_frontend and frontend_spikes_seq is not None:
                    frontend_spikes_seq.append(stage_spikes.detach().clone())

                # Stage 2: motion estimation (optional ablation).
                if self.enable_motion and self.motion_estimator is not None:
                    if update_stdp:
                        self.motion_estimator.stdp_tracking(stage_spikes)

                    motion_id, motion_vec, motion_vec_layer1, motion_pattern = self.motion_estimator.local_wta(
                        stage_spikes,
                        timestamp,
                        visualize=visualize,
                        return_pattern_map=True,
                    )
                    motion_pattern_seq.append(motion_pattern.detach().clone())

                    if return_intermediate:
                        motion_id_seq.append(motion_id.detach().clone())
                        motion_vec_seq.append(motion_vec.detach().clone())
                        motion_vec_l1_seq.append(motion_vec_layer1.detach().clone())

        self.timestamp = start_timestamp + num_frames

        if not return_intermediate:
            if self.enable_motion:
                if num_frames == 0:
                    channels = (
                        int(self.motion_estimator.motion_pattern_num)
                        if self.motion_estimator is not None
                        else 0
                    )
                    out = torch.empty((0, self.spike_h, self.spike_w, channels), dtype=torch.float32, device=self.device)
                else:
                    out = torch.stack(motion_pattern_seq, dim=0)
            else:
                if num_frames == 0:
                    out = torch.empty((0, self.spike_h, self.spike_w), dtype=torch.float32, device=self.device)
                else:
                    out = torch.stack(frontend_spikes_seq, dim=0)
            return self._to_numpy_if_needed(out, as_numpy=as_numpy)

        results = {
            "frontend_spikes": torch.stack(frontend_spikes_seq, dim=0)
            if (frontend_spikes_seq is not None and num_frames > 0)
            else torch.empty((0, self.spike_h, self.spike_w), dtype=torch.float32, device=self.device),
            "motion_vector": torch.stack(motion_vec_seq, dim=0)
            if (motion_vec_seq is not None and num_frames > 0)
            else None,
            "motion_pattern": torch.stack(motion_pattern_seq, dim=0)
            if (motion_pattern_seq is not None and num_frames > 0)
            else None,
            "motion_id": torch.stack(motion_id_seq, dim=0)
            if (motion_id_seq is not None and num_frames > 0)
            else None,
            "motion_vector_layer1": torch.stack(motion_vec_l1_seq, dim=0)
            if (motion_vec_l1_seq is not None and num_frames > 0)
            else None,
        }

        return self._to_numpy_if_needed(results, as_numpy=as_numpy)
