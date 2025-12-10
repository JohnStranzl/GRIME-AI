# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

print("[DEBUG] Loaded patched transforms.py")

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize, Resize, ToTensor

class SAM2Transforms(nn.Module):
    def __init__(
        self, resolution, mask_threshold, max_hole_area=0.0, max_sprinkle_area=0.0
    ):
        """
        Transforms for SAM2.
        """
        super().__init__()
        self.resolution = resolution
        self.mask_threshold = mask_threshold
        self.max_hole_area = max_hole_area
        self.max_sprinkle_area = max_sprinkle_area
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.to_tensor = ToTensor()

        # JES — 2025_06_16
        # NOTE: We avoid calling torch.jit.script() at runtime when running in a "frozen" (PyInstaller) build.
        #       In frozen mode, TorchScript compilation can fail for several reasons:
        #         * The compiler may try to locate Python source for custom transforms (Resize, Normalize),
        #           which are embedded in the executable and not available as .py files.
        #         * It may attempt to write temporary .pt artifacts to a read‑only filesystem inside the bundle.
        #         * PyInstaller’s import hooks can break TorchScript’s introspection of nn.Module definitions.
        #       To prevent these runtime errors, we pre‑compile the transform in development mode and bundle
        #       the resulting sam2_transforms.pt with the frozen app. At runtime:
        #         * Frozen mode --> load the precompiled .pt file from sys._MEIPASS.
        #         * Development mode --> script the transform on‑the‑fly for flexibility and rapid iteration.
        #       This split ensures reproducibility, avoids packaging‑time breakage, and keeps dev builds fast.
        #
        # COMMENT OUT THE CALL TO torch.jit.script AND PLACE IT IN THE NEW LOGIC BELOW.
        '''
        self.transforms = torch.jit.script(
            nn.Sequential(
                Resize((self.resolution, self.resolution)),
                Normalize(self.mean, self.std),
            )
        '''

        # JES — 2025_06_16
        # THE FOLLOWING CODE IS NEW. IT WAS ADDED TO GET AROUND A PyInstaller ISSUE.
        # WE WILL CALL THE FUNCTION ABOVE ONLY WHEN IN DEVELOPMENT MODE. OTHERWISE, IT WILL USE THE
        # PRECOMPILED sam2_transforms.pt WHEN WE ARE COMPILING THE CODE WITH PyInstaller.
        import sys, os

        is_frozen = getattr(sys, 'frozen', False) or hasattr(sys, '_MEIPASS')
        if is_frozen:
            print("[INFO] Frozen mode — loading precompiled sam2_transforms.pt.")
            pt_path = os.path.join(sys._MEIPASS, "sam2_transforms.pt")
            self.transforms = torch.jit.load(pt_path)
            return

        print("[INFO] Development mode — scripting transform on-the-fly.")
        try:
            self.transforms = torch.jit.script(
                nn.Sequential(
                    Resize((self.resolution, self.resolution)),
                    Normalize(self.mean, self.std),
                )
            )
        except Exception as e:
            print(f"[WARN] TorchScript failed: {e}")
            print("[INFO] Falling back to precompiled sam2_transforms.pt.")
            pt_path = os.path.join(os.path.dirname(__file__), "sam2_transforms.pt")
            self.transforms = torch.jit.load(pt_path)

    def __call__(self, x):
        x = self.to_tensor(x)
        return self.transforms(x)

    def forward_batch(self, img_list):
        img_batch = [self.transforms(self.to_tensor(img)) for img in img_list]
        img_batch = torch.stack(img_batch, dim=0)
        return img_batch

    def transform_coords(
        self, coords: torch.Tensor, normalize=False, orig_hw=None
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. The coordinates can be in absolute image or normalized coordinates,
        If the coords are in absolute image coordinates, normalize should be set to True and original image size is required.

        Returns
            Un-normalized coordinates in the range of [0, 1] which is expected by the SAM2 model.
        """
        if normalize:
            assert orig_hw is not None
            h, w = orig_hw
            coords = coords.clone()
            coords[..., 0] = coords[..., 0] / w
            coords[..., 1] = coords[..., 1] / h

        coords = coords * self.resolution  # unnormalize coords
        return coords

    def transform_boxes(
        self, boxes: torch.Tensor, normalize=False, orig_hw=None
    ) -> torch.Tensor:
        """
        Expects a tensor of shape Bx4. The coordinates can be in absolute image or normalized coordinates,
        if the coords are in absolute image coordinates, normalize should be set to True and original image size is required.
        """
        boxes = self.transform_coords(boxes.reshape(-1, 2, 2), normalize, orig_hw)
        return boxes

    def postprocess_masks(self, masks: torch.Tensor, orig_hw) -> torch.Tensor:
        """
        Perform PostProcessing on output masks.
        """
        from sam2.utils.misc import get_connected_components

        masks = masks.float()
        input_masks = masks
        mask_flat = masks.flatten(0, 1).unsqueeze(1)  # flatten as 1-channel image
        try:
            if self.max_hole_area > 0:
                # Holes are those connected components in background with area <= self.fill_hole_area
                # (background regions are those with mask scores <= self.mask_threshold)
                labels, areas = get_connected_components(
                    mask_flat <= self.mask_threshold
                )
                is_hole = (labels > 0) & (areas <= self.max_hole_area)
                is_hole = is_hole.reshape_as(masks)
                # We fill holes with a small positive mask score (10.0) to change them to foreground.
                masks = torch.where(is_hole, self.mask_threshold + 10.0, masks)

            if self.max_sprinkle_area > 0:
                labels, areas = get_connected_components(
                    mask_flat > self.mask_threshold
                )
                is_hole = (labels > 0) & (areas <= self.max_sprinkle_area)
                is_hole = is_hole.reshape_as(masks)
                # We fill holes with negative mask score (-10.0) to change them to background.
                masks = torch.where(is_hole, self.mask_threshold - 10.0, masks)
        except Exception as e:
            # Skip the post-processing step if the CUDA kernel fails
            warnings.warn(
                f"{e}\n\nSkipping the post-processing step due to the error above. You can "
                "still use SAM 2 and it's OK to ignore the error above, although some post-processing "
                "functionality may be limited (which doesn't affect the results in most cases; see "
                "https://github.com/facebookresearch/sam2/blob/main/INSTALL.md).",
                category=UserWarning,
                stacklevel=2,
            )
            masks = input_masks

        masks = F.interpolate(masks, orig_hw, mode="bilinear", align_corners=False)
        return masks
