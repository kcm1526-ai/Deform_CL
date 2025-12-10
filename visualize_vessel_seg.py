"""
Visualization Tool for Vessel Segmentation Results.

Visualizes inference results including:
- Original image
- Predicted segmentation overlay
- Probability map

Usage:
    # Interactive visualization (GUI with sliders)
    python visualize_vessel_seg.py \
        --image test.nii.gz \
        --seg test_seg.nii.gz \
        --prob test_seg_prob.nii.gz

    # Save specific slices as images
    python visualize_vessel_seg.py \
        --image test.nii.gz \
        --seg test_seg.nii.gz \
        --save-dir ./visualizations \
        --slices 50 100 150

    # Batch visualization of a directory
    python visualize_vessel_seg.py \
        --input-dir /raid/users/ai_kcm_0/M3DVBAV_CropLung/image/test/ \
        --pred-dir predictions/ \
        --save-dir ./visualizations

    # 3D rendering (requires VTK)
    python visualize_vessel_seg.py \
        --image test.nii.gz \
        --seg test_seg.nii.gz \
        --mode 3d
"""

import os
import argparse
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')


def load_nifti(file_path):
    """Load NIfTI file and return numpy array."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    sitk_img = sitk.ReadImage(file_path)
    return sitk.GetArrayFromImage(sitk_img)


def normalize_image(img, percentile_lower=1, percentile_upper=99):
    """Normalize image to 0-1 range using percentile clipping."""
    lower = np.percentile(img, percentile_lower)
    upper = np.percentile(img, percentile_upper)
    img_norm = np.clip(img, lower, upper)
    img_norm = (img_norm - lower) / (upper - lower + 1e-8)
    return img_norm


def create_overlay(image, mask, alpha=0.5, color=[1, 0, 0], multi_label=True):
    """Create RGB overlay of mask on grayscale image.

    Args:
        image: 2D image array
        mask: 2D mask array (can have multiple label values)
        alpha: Overlay transparency
        color: Default color for binary mask
        multi_label: If True, use different colors for different labels
    """
    # Normalize image to 0-1
    img_norm = normalize_image(image)

    # Create RGB image from grayscale
    rgb = np.stack([img_norm, img_norm, img_norm], axis=-1)

    if multi_label and mask.max() > 1:
        # Different colors for different labels
        label_colors = {
            1: [1, 0, 0],      # Red for label 1
            2: [0, 1, 0],      # Green for label 2
            3: [0, 0, 1],      # Blue for label 3
            4: [1, 1, 0],      # Yellow for label 4
            5: [1, 0, 1],      # Magenta for label 5
            6: [0, 1, 1],      # Cyan for label 6
        }

        for label_val in np.unique(mask):
            if label_val == 0:
                continue
            label_mask = (mask == label_val).astype(float)
            lcolor = label_colors.get(int(label_val), [1, 0.5, 0])  # Default orange
            for i, c in enumerate(lcolor):
                rgb[..., i] = rgb[..., i] * (1 - alpha * label_mask) + c * alpha * label_mask
    else:
        # Single color for all non-zero values
        mask_binary = (mask > 0).astype(float)
        for i, c in enumerate(color):
            rgb[..., i] = rgb[..., i] * (1 - alpha * mask_binary) + c * alpha * mask_binary

    return np.clip(rgb, 0, 1)


def print_label_info(mask, name="Mask"):
    """Print information about labels in a mask."""
    unique_vals = np.unique(mask)
    print(f"\n{name} label information:")
    print(f"  Unique values: {unique_vals}")
    for val in unique_vals:
        if val == 0:
            continue
        count = np.sum(mask == val)
        print(f"  Label {int(val)}: {count} voxels")


def create_prob_colormap():
    """Create a colormap for probability visualization."""
    colors = [(0, 0, 0, 0), (0, 0, 1, 0.3), (0, 1, 1, 0.5),
              (1, 1, 0, 0.7), (1, 0, 0, 1)]
    return LinearSegmentedColormap.from_list('prob_cmap', colors)


class VesselVisualizer:
    """Interactive 3D volume visualizer for vessel segmentation."""

    def __init__(self, image, seg=None, prob=None, title="Vessel Segmentation"):
        self.image = image
        self.seg = seg
        self.prob = prob
        self.title = title

        # Current slice indices for each view
        self.slices = {
            'axial': image.shape[0] // 2,
            'coronal': image.shape[1] // 2,
            'sagittal': image.shape[2] // 2
        }
        self.current_view = 'axial'
        self.show_overlay = True
        self.show_prob = False
        self.alpha = 0.5

    def get_slice(self, view, idx):
        """Get slice from volume for given view."""
        if view == 'axial':
            img_slice = self.image[idx, :, :]
            seg_slice = self.seg[idx, :, :] if self.seg is not None else None
            prob_slice = self.prob[idx, :, :] if self.prob is not None else None
        elif view == 'coronal':
            img_slice = self.image[:, idx, :]
            seg_slice = self.seg[:, idx, :] if self.seg is not None else None
            prob_slice = self.prob[:, idx, :] if self.prob is not None else None
        else:  # sagittal
            img_slice = self.image[:, :, idx]
            seg_slice = self.seg[:, :, idx] if self.seg is not None else None
            prob_slice = self.prob[:, :, idx] if self.prob is not None else None
        return img_slice, seg_slice, prob_slice

    def get_max_slice(self, view):
        """Get maximum slice index for view."""
        if view == 'axial':
            return self.image.shape[0] - 1
        elif view == 'coronal':
            return self.image.shape[1] - 1
        else:
            return self.image.shape[2] - 1

    def show_interactive(self):
        """Display interactive visualization with sliders."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        plt.subplots_adjust(bottom=0.25)

        # Initial display
        img_slice, seg_slice, prob_slice = self.get_slice(
            self.current_view, self.slices[self.current_view]
        )

        # Original image
        self.im_orig = axes[0].imshow(img_slice, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Segmentation overlay
        if seg_slice is not None:
            overlay = create_overlay(img_slice, seg_slice, self.alpha)
            self.im_seg = axes[1].imshow(overlay)
        else:
            self.im_seg = axes[1].imshow(img_slice, cmap='gray')
        axes[1].set_title('Segmentation Overlay')
        axes[1].axis('off')

        # Probability map
        if prob_slice is not None:
            self.im_prob = axes[2].imshow(img_slice, cmap='gray')
            self.im_prob_overlay = axes[2].imshow(
                prob_slice, cmap=create_prob_colormap(),
                alpha=0.7, vmin=0, vmax=255
            )
        else:
            self.im_prob = axes[2].imshow(img_slice, cmap='gray')
            self.im_prob_overlay = None
        axes[2].set_title('Probability Map')
        axes[2].axis('off')

        self.axes = axes
        self.fig = fig

        # Slice slider
        ax_slice = plt.axes([0.2, 0.1, 0.6, 0.03])
        self.slider_slice = Slider(
            ax_slice, 'Slice', 0, self.get_max_slice(self.current_view),
            valinit=self.slices[self.current_view], valstep=1
        )
        self.slider_slice.on_changed(self.update_slice)

        # Alpha slider
        ax_alpha = plt.axes([0.2, 0.05, 0.6, 0.03])
        self.slider_alpha = Slider(
            ax_alpha, 'Alpha', 0, 1, valinit=self.alpha
        )
        self.slider_alpha.on_changed(self.update_alpha)

        # View radio buttons
        ax_view = plt.axes([0.02, 0.4, 0.1, 0.15])
        self.radio_view = RadioButtons(ax_view, ('axial', 'coronal', 'sagittal'))
        self.radio_view.on_clicked(self.update_view)

        # Info text
        self.info_text = fig.text(
            0.5, 0.95, f'{self.title} | View: {self.current_view} | '
            f'Slice: {self.slices[self.current_view]}',
            ha='center', fontsize=12
        )

        # Keyboard navigation
        fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        fig.canvas.mpl_connect('key_press_event', self.on_key)

        plt.show()

    def update_slice(self, val):
        """Update displayed slice."""
        idx = int(val)
        self.slices[self.current_view] = idx
        self._refresh_display()

    def update_alpha(self, val):
        """Update overlay alpha."""
        self.alpha = val
        self._refresh_display()

    def update_view(self, label):
        """Update view (axial/coronal/sagittal)."""
        self.current_view = label
        self.slider_slice.valmax = self.get_max_slice(label)
        self.slider_slice.set_val(self.slices[label])
        self._refresh_display()

    def on_scroll(self, event):
        """Handle scroll events for slice navigation."""
        if event.button == 'up':
            new_val = min(self.slices[self.current_view] + 1,
                         self.get_max_slice(self.current_view))
        else:
            new_val = max(self.slices[self.current_view] - 1, 0)
        self.slider_slice.set_val(new_val)

    def on_key(self, event):
        """Handle keyboard events."""
        if event.key == 'up' or event.key == 'right':
            new_val = min(self.slices[self.current_view] + 1,
                         self.get_max_slice(self.current_view))
            self.slider_slice.set_val(new_val)
        elif event.key == 'down' or event.key == 'left':
            new_val = max(self.slices[self.current_view] - 1, 0)
            self.slider_slice.set_val(new_val)
        elif event.key == 'a':
            self.radio_view.set_active(0)
        elif event.key == 'c':
            self.radio_view.set_active(1)
        elif event.key == 's':
            self.radio_view.set_active(2)

    def _refresh_display(self):
        """Refresh all displayed images."""
        img_slice, seg_slice, prob_slice = self.get_slice(
            self.current_view, self.slices[self.current_view]
        )

        # Update original
        self.im_orig.set_data(img_slice)
        self.im_orig.set_clim(vmin=img_slice.min(), vmax=img_slice.max())

        # Update segmentation overlay
        if seg_slice is not None:
            overlay = create_overlay(img_slice, seg_slice, self.alpha)
            self.im_seg.set_data(overlay)
        else:
            self.im_seg.set_data(img_slice)

        # Update probability
        self.im_prob.set_data(img_slice)
        self.im_prob.set_clim(vmin=img_slice.min(), vmax=img_slice.max())
        if self.im_prob_overlay is not None and prob_slice is not None:
            self.im_prob_overlay.set_data(prob_slice)

        # Update info text
        self.info_text.set_text(
            f'{self.title} | View: {self.current_view} | '
            f'Slice: {self.slices[self.current_view]}'
        )

        self.fig.canvas.draw_idle()


def save_slice_visualization(image, seg, prob, slice_idx, output_path,
                            view='axial', alpha=0.5):
    """Save a single slice visualization."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Get slices
    if view == 'axial':
        img_slice = image[slice_idx, :, :]
        seg_slice = seg[slice_idx, :, :] if seg is not None else None
        prob_slice = prob[slice_idx, :, :] if prob is not None else None
    elif view == 'coronal':
        img_slice = image[:, slice_idx, :]
        seg_slice = seg[:, slice_idx, :] if seg is not None else None
        prob_slice = prob[:, slice_idx, :] if prob is not None else None
    else:
        img_slice = image[:, :, slice_idx]
        seg_slice = seg[:, :, slice_idx] if seg is not None else None
        prob_slice = prob[:, :, slice_idx] if prob is not None else None

    # Original
    axes[0].imshow(img_slice, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    # Segmentation only
    if seg_slice is not None:
        axes[1].imshow(seg_slice, cmap='hot')
        axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')

    # Overlay
    if seg_slice is not None:
        overlay = create_overlay(img_slice, seg_slice, alpha)
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
    axes[2].axis('off')

    # Probability
    if prob_slice is not None:
        axes[3].imshow(img_slice, cmap='gray')
        axes[3].imshow(prob_slice, cmap=create_prob_colormap(),
                       alpha=0.7, vmin=0, vmax=255)
        axes[3].set_title('Probability')
    axes[3].axis('off')

    plt.suptitle(f'View: {view}, Slice: {slice_idx}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_montage(image, seg, output_path, num_slices=16, view='axial'):
    """Save a montage of multiple slices."""
    if view == 'axial':
        total_slices = image.shape[0]
    elif view == 'coronal':
        total_slices = image.shape[1]
    else:
        total_slices = image.shape[2]

    # Select evenly spaced slices
    indices = np.linspace(0, total_slices - 1, num_slices).astype(int)

    # Calculate grid size
    cols = int(np.ceil(np.sqrt(num_slices)))
    rows = int(np.ceil(num_slices / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        if view == 'axial':
            img_slice = image[idx, :, :]
            seg_slice = seg[idx, :, :] if seg is not None else None
        elif view == 'coronal':
            img_slice = image[:, idx, :]
            seg_slice = seg[:, idx, :] if seg is not None else None
        else:
            img_slice = image[:, :, idx]
            seg_slice = seg[:, :, idx] if seg is not None else None

        if seg_slice is not None:
            overlay = create_overlay(img_slice, seg_slice, alpha=0.5)
            axes[i].imshow(overlay)
        else:
            axes[i].imshow(img_slice, cmap='gray')
        axes[i].set_title(f'Slice {idx}', fontsize=8)
        axes[i].axis('off')

    # Hide empty subplots
    for i in range(len(indices), len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'Montage ({view} view)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_3d_projection(image, seg, output_path):
    """Save Maximum Intensity Projection (MIP) views."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # MIP of original image
    axes[0, 0].imshow(np.max(image, axis=0), cmap='gray')
    axes[0, 0].set_title('MIP Axial (Image)')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(np.max(image, axis=1), cmap='gray')
    axes[0, 1].set_title('MIP Coronal (Image)')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(np.max(image, axis=2), cmap='gray')
    axes[0, 2].set_title('MIP Sagittal (Image)')
    axes[0, 2].axis('off')

    if seg is not None:
        # MIP of segmentation
        axes[1, 0].imshow(np.max(seg, axis=0), cmap='hot')
        axes[1, 0].set_title('MIP Axial (Seg)')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(np.max(seg, axis=1), cmap='hot')
        axes[1, 1].set_title('MIP Coronal (Seg)')
        axes[1, 1].axis('off')

        axes[1, 2].imshow(np.max(seg, axis=2), cmap='hot')
        axes[1, 2].set_title('MIP Sagittal (Seg)')
        axes[1, 2].axis('off')

    plt.suptitle('Maximum Intensity Projections')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_3d_mesh(seg, output_path=None):
    """Create 3D mesh visualization using marching cubes."""
    try:
        from skimage import measure
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        print("3D visualization requires scikit-image. Install with: pip install scikit-image")
        return

    # Extract surface mesh using marching cubes
    verts, faces, normals, values = measure.marching_cubes(seg, level=0.5)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create mesh
    mesh = Poly3DCollection(verts[faces], alpha=0.7)
    mesh.set_facecolor([1, 0.2, 0.2])
    mesh.set_edgecolor('none')
    ax.add_collection3d(mesh)

    # Set axis limits
    ax.set_xlim(0, seg.shape[0])
    ax.set_ylim(0, seg.shape[1])
    ax.set_zlim(0, seg.shape[2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Vessel Segmentation')

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize Vessel Segmentation Results')

    # Input files
    parser.add_argument('--image', type=str, help='Path to input image (NIfTI)')
    parser.add_argument('--seg', type=str, help='Path to segmentation mask (NIfTI)')
    parser.add_argument('--prob', type=str, help='Path to probability map (NIfTI)')

    # Batch processing
    parser.add_argument('--input-dir', type=str, help='Directory containing input images')
    parser.add_argument('--pred-dir', type=str, help='Directory containing predictions')

    # Output options
    parser.add_argument('--save-dir', type=str, help='Directory to save visualizations')
    parser.add_argument('--slices', type=int, nargs='+', help='Specific slice indices to save')
    parser.add_argument('--view', type=str, default='axial',
                       choices=['axial', 'coronal', 'sagittal'],
                       help='View for visualization')

    # Visualization mode
    parser.add_argument('--mode', type=str, default='interactive',
                       choices=['interactive', 'save', 'montage', 'mip', '3d'],
                       help='Visualization mode')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Overlay transparency (0-1)')
    parser.add_argument('--binary', action='store_true',
                       help='Binarize segmentation (merge all labels into one)')
    parser.add_argument('--label', type=int, default=None,
                       help='Show only specific label value')

    args = parser.parse_args()

    # Single file visualization
    if args.image:
        print(f"Loading image: {args.image}")
        image = load_nifti(args.image)

        seg = None
        if args.seg:
            print(f"Loading segmentation: {args.seg}")
            seg = load_nifti(args.seg)
            print_label_info(seg, "Segmentation")

            # Handle label filtering
            if args.label is not None:
                print(f"\nFiltering to show only label {args.label}")
                seg = (seg == args.label).astype(np.uint8)
            elif args.binary:
                print("\nBinarizing segmentation (all labels -> 1)")
                seg = (seg > 0).astype(np.uint8)

        prob = None
        if args.prob:
            print(f"Loading probability map: {args.prob}")
            prob = load_nifti(args.prob)

        print(f"\nImage shape: {image.shape}")

        if args.mode == 'interactive':
            title = os.path.basename(args.image)
            viz = VesselVisualizer(image, seg, prob, title=title)
            viz.show_interactive()

        elif args.mode == 'save' and args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            base_name = os.path.basename(args.image).replace('.nii.gz', '').replace('.nii', '')

            if args.slices:
                for slice_idx in args.slices:
                    output_path = os.path.join(
                        args.save_dir, f'{base_name}_{args.view}_slice{slice_idx}.png'
                    )
                    save_slice_visualization(
                        image, seg, prob, slice_idx, output_path,
                        view=args.view, alpha=args.alpha
                    )
                    print(f"Saved: {output_path}")
            else:
                # Save middle slices from each view
                for view in ['axial', 'coronal', 'sagittal']:
                    if view == 'axial':
                        mid = image.shape[0] // 2
                    elif view == 'coronal':
                        mid = image.shape[1] // 2
                    else:
                        mid = image.shape[2] // 2

                    output_path = os.path.join(
                        args.save_dir, f'{base_name}_{view}_slice{mid}.png'
                    )
                    save_slice_visualization(
                        image, seg, prob, mid, output_path,
                        view=view, alpha=args.alpha
                    )
                    print(f"Saved: {output_path}")

        elif args.mode == 'montage' and args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            base_name = os.path.basename(args.image).replace('.nii.gz', '').replace('.nii', '')
            output_path = os.path.join(args.save_dir, f'{base_name}_montage.png')
            save_montage(image, seg, output_path, view=args.view)
            print(f"Saved montage: {output_path}")

        elif args.mode == 'mip' and args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            base_name = os.path.basename(args.image).replace('.nii.gz', '').replace('.nii', '')
            output_path = os.path.join(args.save_dir, f'{base_name}_mip.png')
            save_3d_projection(image, seg, output_path)
            print(f"Saved MIP: {output_path}")

        elif args.mode == '3d':
            if seg is not None:
                output_path = None
                if args.save_dir:
                    os.makedirs(args.save_dir, exist_ok=True)
                    base_name = os.path.basename(args.image).replace('.nii.gz', '').replace('.nii', '')
                    output_path = os.path.join(args.save_dir, f'{base_name}_3d.png')
                visualize_3d_mesh(seg, output_path)
                if output_path:
                    print(f"Saved 3D visualization: {output_path}")
            else:
                print("Error: 3D mode requires segmentation mask (--seg)")

    # Batch processing
    elif args.input_dir and args.pred_dir and args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

        image_files = sorted([f for f in os.listdir(args.input_dir)
                             if f.endswith('.nii.gz') or f.endswith('.nii')])

        print(f"Found {len(image_files)} images to process")

        for img_file in image_files:
            base_name = img_file.replace('.nii.gz', '').replace('.nii', '')

            # Load image
            img_path = os.path.join(args.input_dir, img_file)
            image = load_nifti(img_path)

            # Try to find corresponding segmentation
            seg_candidates = [
                f'{base_name}_seg.nii.gz',
                f'{base_name}_seg.nii',
                f'{base_name}.nii.gz',
            ]

            seg = None
            for candidate in seg_candidates:
                seg_path = os.path.join(args.pred_dir, candidate)
                if os.path.exists(seg_path):
                    seg = load_nifti(seg_path)
                    break

            # Try to find probability map
            prob = None
            prob_path = os.path.join(args.pred_dir, f'{base_name}_seg_prob.nii.gz')
            if os.path.exists(prob_path):
                prob = load_nifti(prob_path)

            # Save visualizations
            if args.mode == 'montage':
                output_path = os.path.join(args.save_dir, f'{base_name}_montage.png')
                save_montage(image, seg, output_path, view=args.view)
            elif args.mode == 'mip':
                output_path = os.path.join(args.save_dir, f'{base_name}_mip.png')
                save_3d_projection(image, seg, output_path)
            else:
                # Default: save middle slice
                mid = image.shape[0] // 2
                output_path = os.path.join(args.save_dir, f'{base_name}_slice{mid}.png')
                save_slice_visualization(image, seg, prob, mid, output_path,
                                        view=args.view, alpha=args.alpha)

            print(f"Processed: {base_name}")

    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Interactive visualization")
        print("  python visualize_vessel_seg.py --image test.nii.gz --seg test_seg.nii.gz")
        print("")
        print("  # Save montage")
        print("  python visualize_vessel_seg.py --image test.nii.gz --seg test_seg.nii.gz \\")
        print("      --mode montage --save-dir ./viz")
        print("")
        print("  # Save MIP projections")
        print("  python visualize_vessel_seg.py --image test.nii.gz --seg test_seg.nii.gz \\")
        print("      --mode mip --save-dir ./viz")


if __name__ == "__main__":
    main()
