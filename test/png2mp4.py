import cv2
import os
import re
import argparse

def make_video(input_folder, gt_prefix, pred_prefix, output_path, fps=10):
    """
    Read GT and Pred PNGs from input_folder, vertically stack them, and write to an MP4 at output_path.
    """
    # List and sort files matching the prefixes
    files = os.listdir(input_folder)
    pat = re.compile(rf'{re.escape(gt_prefix)}(\d+)\.png')
    def sorted_files(prefix):
        matched = [f for f in files if f.startswith(prefix) and f.endswith('.png')]
        return sorted(matched, key=lambda x: int(re.findall(r'(\d+)', x)[0]))

    gt_files = sorted_files(gt_prefix)
    pred_files = sorted_files(pred_prefix)
    if len(gt_files) != len(pred_files):
        raise ValueError(f"Number of GT files ({len(gt_files)}) does not match Pred files ({len(pred_files)}).")

    # Read first frame to get dimensions
    first_gt = cv2.imread(os.path.join(input_folder, gt_files[0]))
    h, w = first_gt.shape[:2]

    # Prepare VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, 2 * h))

    # Process each pair
    for gt_name, pred_name in zip(gt_files, pred_files):
        gt_img = cv2.imread(os.path.join(input_folder, gt_name))
        pred_img = cv2.imread(os.path.join(input_folder, pred_name))
        stacked = cv2.vconcat([gt_img, pred_img])
        writer.write(stacked)

    writer.release()
    print(f"✔ Video saved to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate GT vs Pred videos (risk & scale)")
    parser.add_argument('--input_folder', required=True, help="Folder with GT and Pred PNGs")
    parser.add_argument('--output_dir', required=True, help="Directory to save the MP4 videos")
    parser.add_argument('--fps', type=int, default=10, help="Frames per second for the output videos")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Generate risk video
    risk_out = os.path.join(args.output_dir, 'risk.mp4')
    make_video(
        input_folder=args.input_folder,
        gt_prefix='gt_risk_',
        pred_prefix='pred_risk_',
        output_path=risk_out,
        fps=args.fps
    )

    # Generate scale video
    scale_out = os.path.join(args.output_dir, 'scale.mp4')
    make_video(
        input_folder=args.input_folder,
        gt_prefix='gt_scale_',
        pred_prefix='pred_scale_',
        output_path=scale_out,
        fps=args.fps
    )
