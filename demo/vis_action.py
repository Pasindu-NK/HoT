import os
import glob
import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn

# Make sure we can import from the HoT repo root
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.utils import define_actions


class ActionHead(nn.Module):
    """
    Same ActionHead we used in main_mixste_.py

    Input:  x of shape [B, T, J, 3]
    Output: logits [B, num_actions]
    """
    def __init__(self, num_joints=17, num_actions=15, hidden_dim=256):
        super().__init__()
        self.num_joints = num_joints
        self.fc = nn.Sequential(
            nn.Linear(num_joints * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x):
        # x: [B, T, J, 3]
        B, T, J, C = x.shape
        x = x.view(B, T, J * C)   # [B, T, J*3]
        x = x.mean(dim=1)         # temporal average → [B, J*3]
        logits = self.fc(x)       # [B, num_actions]
        return logits


def load_3d_sequence(output_dir):
    """
    Load 3D keypoints from HoT demo output.
    Expects: {output_dir}/output_3D/output_keypoints_3d.npz
    with key 'reconstruction' of shape [T, J, 3]
    """
    npz_path = os.path.join(output_dir, "output_3D", "output_keypoints_3d.npz")
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"3D keypoints file not found: {npz_path}")

    data = np.load(npz_path)
    if "reconstruction" not in data:
        raise KeyError(
            f"'reconstruction' not found in {npz_path}. "
            "Check the npz structure or your demo/vis.py version."
        )

    seq_3d = data["reconstruction"]  # [T, J, 3]
    return seq_3d


def predict_action(seq_3d, action_head_path, device="cuda"):
    """
    seq_3d: numpy array [T, J, 3]
    action_head_path: path to trained ActionHead .pth file
    Returns: (pred_action_name, prob_vector)
    """
    # Actions list (same order as training in main_mixste_.py)
    actions = define_actions("all")
    num_actions = len(actions)

    # Build and load ActionHead
    model = ActionHead(num_joints=seq_3d.shape[1],
                       num_actions=num_actions).to(device)
    state_dict = torch.load(action_head_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        # [1, T, J, 3]
        x = torch.from_numpy(seq_3d).unsqueeze(0).float().to(device)
        logits = model(x)                      # [1, num_actions]
        probs = torch.softmax(logits, dim=1)   # [1, num_actions]
        pred_idx = int(torch.argmax(probs, dim=1).item())
        pred_name = actions[pred_idx]
        prob_vec = probs.squeeze(0).cpu().numpy()

    return pred_name, prob_vec


def overlay_action_on_frames(output_dir, action_text, suffix="_action"):
    """
    Read frames from {output_dir}/pose/*.png,
    overlay 'action_text' at the top, save to:
        {output_dir}/pose{suffix}/
    and also create a video file in {output_dir}/ as mp4.
    """
    pose_dir = os.path.join(output_dir, "pose")
    frame_paths = sorted(glob.glob(os.path.join(pose_dir, "*.png")))
    if len(frame_paths) == 0:
        raise FileNotFoundError(f"No pose frames found in {pose_dir}")

    out_pose_dir = os.path.join(output_dir, f"pose{suffix}")
    os.makedirs(out_pose_dir, exist_ok=True)

    # Font config for cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    color = (0, 0, 255)   # BGR red
    margin = 20

    # Setup video writer
    first_img = cv2.imread(frame_paths[0])
    if first_img is None:
        raise RuntimeError(f"Failed to read first frame: {frame_paths[0]}")
    h, w, _ = first_img.shape

    video_path = os.path.join(output_dir, f"demo_with_action{suffix}.mp4")
    # 30 FPS is a decent default; adjust if you like
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, 30, (w, h))

    for fp in frame_paths:
        img = cv2.imread(fp)
        if img is None:
            print(f"Warning: failed to read {fp}, skipping.")
            continue

        # Put text at top-left
        text = f"Action: {action_text}"
        # Compute baseline for nicer placement
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        org = (margin, margin + text_h)

        cv2.putText(
            img,
            text,
            org,
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

        # Save per-frame image
        out_frame_name = os.path.basename(fp)
        out_frame_path = os.path.join(out_pose_dir, out_frame_name)
        cv2.imwrite(out_frame_path, img)

        # Add to video
        writer.write(img)

    writer.release()
    print(f"Saved frames with action overlay to: {out_pose_dir}")
    print(f"Saved video with action overlay to:   {video_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Add action label overlay to HoT demo output."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory created by demo/vis.py for a given video.",
    )
    parser.add_argument(
        "--action_head_path",
        type=str,
        required=True,
        help="Path to trained ActionHead weights (.pth).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run action head on (cuda or cpu).",
    )
    args = parser.parse_args()

    # Normalize output_dir (e.g. 'demo/output/sample_video/')
    output_dir = args.output_dir
    if not output_dir.endswith("/"):
        output_dir += "/"

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"

    print("Loading 3D pose sequence...")
    seq_3d = load_3d_sequence(output_dir)

    print("Predicting action...")
    action_name, probs = predict_action(seq_3d, args.action_head_path, device=device)
    print(f"Predicted action: {action_name}")

    print("Overlaying action on demo frames...")
    overlay_action_on_frames(output_dir, action_name, suffix="_action")


if __name__ == "__main__":
    main()
