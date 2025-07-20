import torch
import numpy as np
from PIL import Image
import gradio as gr
import imageio
from skimage import img_as_ubyte
import random
from models import get_instrumented_model
from decomposition import get_or_compute
from config import Config
import datetime
import os

# Speed up computation
torch.autograd.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True

# Specify model to use
selected_model = 'lookbook'
config = Config(
    model='StyleGAN2',
    layer='style',
    output_class=selected_model,
    components=80,
    use_w=True,
    batch_size=5_000,  # style layer quite small
)

# Load model
inst = get_instrumented_model(config.model, config.output_class,
                            config.layer, torch.device('cuda'), use_w=config.use_w)

path_to_components = get_or_compute(config, inst)
model = inst.model

comps = np.load(path_to_components)
lst = comps.files
latent_dirs = []
latent_stdevs = []

load_activations = False

for item in lst:
    if load_activations:
        if item == 'act_comp':
            for i in range(comps[item].shape[0]):
                latent_dirs.append(comps[item][i])
        if item == 'act_stdev':
            for i in range(comps[item].shape[0]):
                latent_stdevs.append(comps[item][i])
    else:
        if item == 'lat_comp':
            for i in range(comps[item].shape[0]):
                latent_dirs.append(comps[item][i])
        if item == 'lat_stdev':
            for i in range(comps[item].shape[0]):
                latent_stdevs.append(comps[item][i])

def mix_w(w1, w2, content, style):
    # Content mixing for first 5 layers
    for i in range(0,5):
        w2[i] = w1[i] * (1 - content) + w2[i] * content

    # Style mixing for remaining layers - now dynamic based on input length
    for i in range(5, len(w2)):
        w2[i] = w1[i] * (1 - style) + w2[i] * style
    
    return w2

def display_sample_pytorch(seed, truncation, directions, distances, scale, start, end, w=None, disp=True, save=None, noise_spec=None):
    model.truncation = truncation
    if w is None:
        w = model.sample_latent(1, seed=seed).detach().cpu().numpy()
        w = [w]*model.get_max_latents() # one per layer
    else:
        w = [np.expand_dims(x, 0) for x in w]
    
    # Apply to all layers
    max_latents = model.get_max_latents()
    for l in range(max_latents):
        for i in range(len(directions)):
            w[l] = w[l] + directions[i] * distances[i] * scale
    
    torch.cuda.empty_cache()
    # Save image and display
    out = model.sample_np(w)
    final_im = Image.fromarray((out * 255).astype(np.uint8)).resize((500,500),Image.LANCZOS)
    
    if save is not None:
        if disp == False:
            print(save)
        final_im.save(f'out/{seed}_{save:05}.png')
    if disp:
        display(final_im)
    
    return final_im

def update_gallery(current_gallery, new_image):
    if current_gallery is None:
        current_gallery = []
    # Add new image to the start of the gallery
    current_gallery.insert(0, new_image)
    # Keep only the last 8 images
    return current_gallery[:8]

def generate_random_image(content=0.5, style=0.5, truncation=0.7,
                        c0=0, c1=0, c2=0, c3=0, c4=0, c5=0, c6=0, c7=0,
                        red=0, green=0, blue=0, age=0, size=0,
                        current_gallery=None):
    # Generate only random seeds, keep other parameters unchanged
    seed1 = random.randint(0, 10000)
    seed2 = random.randint(0, 10000)
    
    # Ensure all parameters have default values if None
    content = content if content is not None else 0.5
    style = style if style is not None else 0.5
    truncation = truncation if truncation is not None else 0.7
    c0 = c0 if c0 is not None else 0
    c1 = c1 if c1 is not None else 0
    c2 = c2 if c2 is not None else 0
    c3 = c3 if c3 is not None else 0
    c4 = c4 if c4 is not None else 0
    c5 = c5 if c5 is not None else 0
    c6 = c6 if c6 is not None else 0
    c7 = c7 if c7 is not None else 0
    red = red if red is not None else 0
    green = green if green is not None else 0
    blue = blue if blue is not None else 0
    age = age if age is not None else 0
    size = size if size is not None else 0
    
    # Generate the images using current slider values
    input_im, output_im = generate_image(seed1, seed2, content, style, truncation,
                                       c0, c1, c2, c3, c4, c5, c6, c7,
                                       red, green, blue, age, size)
    
    # Update gallery
    updated_gallery = update_gallery(current_gallery, output_im)
    
    # Return values including updated gallery
    return (
        input_im, output_im,  # Images
        f"Input Seeds: {seed1}, {seed2}",  # Seed display text
        f"Output Generated from Seeds: {seed1}, {seed2}",  # Output seed display text
        seed1, seed2,  # Seed number inputs
        updated_gallery  # Updated gallery
    )

def generate_with_gallery(*args):
    # Extract all arguments except the last one (gallery)
    generate_args = args[:-1]  # Remove gallery from args
    
    # Ensure we have all required arguments
    if len(generate_args) < 17:  # generate_image requires 17 arguments
        # Pad with default values if missing arguments
        default_args = [0.5, 0.5, 0.7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        generate_args = list(generate_args) + default_args[len(generate_args):]
    
    # Ensure all arguments have default values if None
    generate_args = [arg if arg is not None else 0 for arg in generate_args]
    
    input_im, output_im = generate_image(*generate_args)
    # Get gallery from the last argument
    current_gallery = args[-1]
    updated_gallery = update_gallery(current_gallery, output_im)
    return input_im, output_im, updated_gallery

def generate_image(seed1, seed2, content, style, truncation, c0, c1, c2, c3, c4, c5, c6, c7, red, green, blue, age, size):
    # Ensure all parameters have default values if None
    content = content if content is not None else 0.5
    style = style if style is not None else 0.5
    truncation = truncation if truncation is not None else 0.7
    c0 = c0 if c0 is not None else 0
    c1 = c1 if c1 is not None else 0
    c2 = c2 if c2 is not None else 0
    c3 = c3 if c3 is not None else 0
    c4 = c4 if c4 is not None else 0
    c5 = c5 if c5 is not None else 0
    c6 = c6 if c6 is not None else 0
    c7 = c7 if c7 is not None else 0
    red = red if red is not None else 0
    green = green if green is not None else 0
    blue = blue if blue is not None else 0
    age = age if age is not None else 0
    size = size if size is not None else 0
    
    # Prepare style parameters with normalized directions
    directions = []
    distances = [c0, c1, c2, c3, c4, c5, c6, c7]
    for i in range(8):
        # Normalize the direction vectors for more consistent control
        dir_vec = latent_dirs[i]
        norm = np.linalg.norm(dir_vec)
        if norm > 0:
            dir_vec = dir_vec / norm
        directions.append(dir_vec)

    # Verify these indices match your decomposition
    age_idx = 8
    size_idx = 9
    age_direction = latent_dirs[age_idx]
    size_direction = latent_dirs[size_idx]
    
    # Generate base latents with improved seed handling
    w1 = model.sample_latent(1, seed=int(seed1)).detach().cpu().numpy()
    w1 = [w1] * model.get_max_latents()
    im1 = model.sample_np(w1)

    w2 = model.sample_latent(1, seed=int(seed2)).detach().cpu().numpy()
    w2 = [w2] * model.get_max_latents()
    im2 = model.sample_np(w2)
    
    # Enhanced mixing with dynamic layer control
    mixed_w = mix_w(w1, w2, content, style)
    
    # Apply modifications to specific layers based on the attribute
    max_latents = model.get_max_latents()
    
    # Define layer ranges and strengths for each style control with improved granularity
    style_controls = {
        'sleeve': {
            'layers': [(2, 4, 0.3), (4, 8, 1.0), (8, 10, 0.5)],
            'direction': directions[0],
            'distance': distances[0]
        },
        'size': {
            'layers': [(0, 6, 0.8)],
            'direction': directions[1],
            'distance': distances[1]
        },
        'dress_jacket': {
            'layers': [(4, 10, 0.7)],
            'direction': directions[2],
            'distance': distances[2]
        },
        'female_coat': {
            'layers': [(8, max_latents, 0.9)],
            'direction': directions[3],
            'distance': distances[3]
        },
        'coat': {
            'layers': [(8, max_latents, 0.9)],
            'direction': directions[4],
            'distance': distances[4]
        },
        'graphics': {
            'layers': [(8, max_latents, 1.0)],
            'direction': directions[5],
            'distance': distances[5]
        },
        'dark': {
            'layers': [(8, max_latents, 1.0)],
            'direction': directions[6],
            'distance': distances[6]
        },
        'cleavage': {
            'layers': [(2, 4, 0.8), (4, 8, 1.2), (8, 10, 0.9)],
            'direction': directions[7],
            'distance': distances[7]
        }
    }
    
    # Apply style modifications with improved control and sensitivity
    for control in style_controls.values():
        if abs(control['distance']) > 0.05:  # Lower threshold for more sensitivity
            for start, end, strength in control['layers']:
                for l in range(start, end):
                    # Enhanced scaling for cleavage control
                    if control == style_controls['cleavage']:
                        scale = strength * (1 + abs(control['distance']) / 5)  # Increased scaling factor
                    else:
                        scale = strength * (1 + abs(control['distance']) / 10)
                    mixed_w[l] = mixed_w[l] + control['direction'] * control['distance'] * scale
    
    # Apply age and size modifications with improved control
    age_scale = 0.4 * (1 + abs(age) / 10)
    size_scale = 0.4 * (1 + abs(size) / 10)
    
    for l in range(max_latents):
        mixed_w[l] = mixed_w[l] + age_direction * age * age_scale
        mixed_w[l] = mixed_w[l] + size_direction * size * size_scale

    output = model.sample_np(mixed_w)
    
    # Apply color adjustments with improved control and masking
    output = output.astype(float)
    mask = (np.mean(output, axis=2) > 0.1) & (np.mean(output, axis=2) < 0.9)
    
    # Apply color adjustments with non-linear scaling and improved blending
    for c in range(3):
        if c == 0:  # Red
            scale = 1 + (red/100) * (1 + abs(red)/200)
            output[:, :, c] = np.where(mask, output[:, :, c] * scale, output[:, :, c])
        elif c == 1:  # Green
            scale = 1 + (green/100) * (1 + abs(green)/200)
            output[:, :, c] = np.where(mask, output[:, :, c] * scale, output[:, :, c])
        else:  # Blue
            scale = 1 + (blue/100) * (1 + abs(blue)/200)
            output[:, :, c] = np.where(mask, output[:, :, c] * scale, output[:, :, c])
    
    # Ensure output is properly clipped and normalized
    output = np.clip(output, 0, 1)
    
    # Convert to images with improved quality
    input_im = Image.fromarray((np.concatenate([im1, im2], axis=1) * 255).astype(np.uint8))
    output_im = Image.fromarray((output * 255).astype(np.uint8))
    
    return input_im, output_im

def save_current_image(input_im, output_im):
    if input_im is None or output_im is None:
        return "No images to save"
        
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("outputs", exist_ok=True)
    
    input_path = f"outputs/input_{timestamp}.png"
    output_path = f"outputs/output_{timestamp}.png"
    
    if isinstance(input_im, str):
        Image.open(input_im).save(input_path)
    else:
        input_im.save(input_path)
        
    if isinstance(output_im, str):
        Image.open(output_im).save(output_path)
    else:
        output_im.save(output_path)
        
    return f"Images saved as input_{timestamp}.png and output_{timestamp}.png"

# Gradio Interface
css = """
#main-container {
    background-image: linear-gradient(to bottom right, rgba(30, 41, 59, 0.95), rgba(17, 24, 39, 0.95));
    border-radius: 20px;
    backdrop-filter: blur(10px);
    padding: 2rem;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    transition: all 0.3s ease;
}

.container {
    background-color: rgba(30, 41, 59, 0.7) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(5px) !important;
    padding: 1rem !important;
    margin-bottom: 1rem !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    transition: all 0.3s ease !important;
}

.container:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2) !important;
    border-color: rgba(59, 130, 246, 0.5) !important;
}

.gr-button {
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    background-color: rgba(59, 130, 246, 0.7) !important;
    backdrop-filter: blur(5px) !important;
    transition: all 0.2s ease !important;
    transform: translateY(0) !important;
}

.gr-button:hover {
    background-color: rgba(59, 130, 246, 0.9) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
}

.gr-button:active {
    transform: translateY(0) !important;
}

.gr-form {
    background-color: transparent !important;
    border: none !important;
}

.gr-box {
    background-color: rgba(30, 41, 59, 0.7) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(5px) !important;
    transition: all 0.3s ease !important;
}

.gr-box:hover {
    background-color: rgba(30, 41, 59, 0.8) !important;
}

.gr-input, .gr-slider {
    background-color: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    transition: all 0.2s ease !important;
}

.gr-input:hover, .gr-slider:hover {
    border-color: rgba(59, 130, 246, 0.3) !important;
    background-color: rgba(255, 255, 255, 0.08) !important;
}

.gr-input:focus, .gr-slider:focus {
    border-color: rgba(59, 130, 246, 0.5) !important;
    background-color: rgba(255, 255, 255, 0.1) !important;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
}

.gr-image-container {
    transition: all 0.3s ease !important;
}

.gr-image-container:hover {
    transform: scale(1.02) !important;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2) !important;
}

.gr-markdown {
    transition: color 0.2s ease !important;
}

.gr-markdown:hover {
    color: rgba(59, 130, 246, 0.9) !important;
}

.footer {
    margin-top: 2rem !important;
    padding: 1.5rem !important;
    border-top: 1px solid rgba(255, 255, 255, 0.1) !important;
    text-align: center !important;
}

.footer-content {
    color: rgba(255, 255, 255, 0.7) !important;
    font-size: 0.9rem !important;
    display: flex !important;
    justify-content: center !important;
    gap: 2rem !important;
    flex-wrap: wrap !important;
}

.footer-section {
    transition: all 0.3s ease !important;
    padding: 0.5rem 1rem !important;
    border-radius: 8px !important;
    background-color: rgba(30, 41, 59, 0.4) !important;
    backdrop-filter: blur(5px) !important;
}

.footer-section:hover {
    background-color: rgba(30, 41, 59, 0.6) !important;
    transform: translateY(-2px) !important;
}

.footer a {
    color: rgba(59, 130, 246, 0.9) !important;
    text-decoration: none !important;
    transition: all 0.2s ease !important;
}

.footer a:hover {
    color: rgba(59, 130, 246, 1) !important;
    text-decoration: underline !important;
}

.info-panels {
    margin: 1rem 0 !important;
}

.info-panels .gr-column {
    min-width: 200px !important;
}

.info-panels .gr-markdown {
    height: 100% !important;
}

.info-panels div {
    height: 100% !important;
    transition: all 0.3s ease !important;
}

.info-panels div:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
    background: rgba(30, 41, 59, 0.5) !important;
}

/* Style for the history section */
.history-section {
    margin-top: 1rem !important;
    padding: 1rem !important;
    background: rgba(30, 41, 59, 0.4) !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}

.history-section:hover {
    background: rgba(30, 41, 59, 0.5) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
}

/* Add some spacing between elements */
.gr-container {
    gap: 1rem !important;
}

/* Make images container more prominent */
.gr-image-container {
    background: rgba(30, 41, 59, 0.3) !important;
    padding: 0.5rem !important;
    border-radius: 8px !important;
}

/* Tutorial page styles */
#tutorial-container {
    background-image: linear-gradient(to bottom right, rgba(30, 41, 59, 0.95), rgba(17, 24, 39, 0.95));
    min-height: 100vh;
}

#tutorial-container .container {
    margin-bottom: 2rem !important;
    padding: 1.5rem !important;
}

#tutorial-container h2 {
    margin-top: 0 !important;
    margin-bottom: 1rem !important;
    font-size: 1.5rem !important;
}

#tutorial-container h3 {
    margin-top: 1.5rem !important;
    margin-bottom: 0.5rem !important;
    font-size: 1.2rem !important;
}

#tutorial-container ul, #tutorial-container ol {
    margin: 1rem 0 !important;
    padding-left: 1.5rem !important;
}

#tutorial-container li {
    margin: 0.5rem 0 !important;
    line-height: 1.6 !important;
}

#tutorial-container p {
    margin: 1rem 0 !important;
    line-height: 1.6 !important;
    color: rgba(255, 255, 255, 0.9) !important;
}

#tutorial-container strong {
    color: rgba(59, 130, 246, 0.9) !important;
}

/* Tab styles */
.tabs {
    border: none !important;
    background: transparent !important;
}

.tab-nav {
    background: rgba(30, 41, 59, 0.7) !important;
    border-radius: 12px !important;
    padding: 0.5rem !important;
    margin-bottom: 1rem !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

.tab-nav button {
    color: rgba(255, 255, 255, 0.7) !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}

.tab-nav button:hover {
    background: rgba(59, 130, 246, 0.2) !important;
    color: rgba(255, 255, 255, 0.9) !important;
}

.tab-nav button.selected {
    background: rgba(59, 130, 246, 0.7) !important;
    color: white !important;
}
"""

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        font=["Inter", "sans-serif"],
    ),
    css=css
) as demo:
    with gr.Tabs() as tabs:
        with gr.Tab("Generate", id=0):
            with gr.Column(elem_id="main-container"):
                gr.Markdown("""
                <div style='text-align: center; margin-bottom: 20px;'>
                    <h1 style='color: #fff; font-size: 2.5em; text-shadow: 0 2px 4px rgba(0,0,0,0.2);'>ClothingGAN</h1>
                </div>
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes="container"):
                            gr.Markdown("### 🎯 Generation Controls")
                            with gr.Row():
                                seed1 = gr.Number(label="Input Seed 1", value=1, precision=0)
                                seed2 = gr.Number(label="Input Seed 2", value=2, precision=0)
                            
                            content = gr.Slider(0, 1, value=0.5, label="Content Mix")
                            style = gr.Slider(0, 1, value=0.5, label="Style Mix")
                            truncation = gr.Slider(0, 1, value=0.7, label="Truncation")

                        with gr.Group(elem_classes="container"):
                            gr.Markdown("### 👶 Age & Size Controls")
                            age = gr.Slider(-10, 10, value=0, label="Age Adjustment (Younger to Older)")
                            size = gr.Slider(-10, 10, value=0, label="Size Adjustment (Smaller to Larger)")

                        with gr.Group(elem_classes="container"):
                            gr.Markdown("### 🎨 Style Controls")
                            c0 = gr.Slider(-10, 10, value=0, label="Sleeve Length (Short to Long)")
                            c1 = gr.Slider(-10, 10, value=0, label="Size (Tight to Loose)")
                            c2 = gr.Slider(-10, 10, value=0, label="Dress - Jacket")
                            c3 = gr.Slider(-10, 10, value=0, label="Female Coat")
                            c4 = gr.Slider(-10, 10, value=0, label="Coat")
                            c5 = gr.Slider(-10, 10, value=0, label="Graphics")
                            c6 = gr.Slider(-10, 10, value=0, label="Dark")
                            c7 = gr.Slider(-10, 10, value=0, label="Less Cleavage")

                        with gr.Group(elem_classes="container"):
                            gr.Markdown("### 🎨 Color Adjustment")
                            red = gr.Slider(-100, 100, value=0, label="Red %")
                            green = gr.Slider(-100, 100, value=0, label="Green %")
                            blue = gr.Slider(-100, 100, value=0, label="Blue %")

                    with gr.Column(scale=2):
                        with gr.Group(elem_classes="container"):
                            with gr.Row():
                                random_button = gr.Button("🎲 Generate Random", variant="primary", size="lg")
                                save_button = gr.Button("💾 Save Images", variant="secondary", size="lg")
                            
                            # Add info panels in the blank space
                            with gr.Row(elem_classes="info-panels"):
                                with gr.Column(scale=1):
                                    gr.Markdown("""
                                    <div style='text-align: center; padding: 0.5rem; background: rgba(30, 41, 59, 0.4); border-radius: 8px; margin-bottom: 1rem;'>
                                        <h4 style='margin: 0; color: #fff;'>🎯 Quick Tips</h4>
                                        <p style='margin: 0.5rem 0; font-size: 0.9rem; color: rgba(255,255,255,0.8);'>
                                            Adjust sliders to control style and appearance. Use seeds for consistent results.
                                        </p>
                                    </div>
                                    """)
                                with gr.Column(scale=1):
                                    gr.Markdown("""
                                    <div style='text-align: center; padding: 0.5rem; background: rgba(30, 41, 59, 0.4); border-radius: 8px; margin-bottom: 1rem;'>
                                        <h4 style='margin: 0; color: #fff;'>💫 Current Mode</h4>
                                        <p style='margin: 0.5rem 0; font-size: 0.9rem; color: rgba(255,255,255,0.8);'>
                                            Style Transfer & Generation
                                        </p>
                                    </div>
                                    """)
                            
                            with gr.Row():
                                with gr.Column():
                                    input_image = gr.Image(label="Input Mixed", type="pil", show_download_button=True)
                                    input_seed_display = gr.Markdown("Input Seeds: 1, 2")
                                with gr.Column():
                                    output_image = gr.Image(label="Styled Output", type="pil", show_download_button=True)
                                    output_seed_display = gr.Markdown("Output Generated from Seeds: 1, 2")
                            
                            result_text = gr.Textbox(label="Status", interactive=False)
                            
                            # Replace static history section with actual gallery
                            gr.Markdown("<h4 style='margin: 1rem 0 0.5rem 0; color: #fff; text-align: center;'>🕒 Recent Generations</h4>")
                            gallery = gr.Gallery(
                                label="Recent Generations",
                                show_label=False,
                                elem_id="gallery",
                                columns=4,
                                rows=1,
                                object_fit="contain",
                                height="200px",
                                allow_preview=True,
                                show_share_button=False
                            )

        with gr.Tab("How to Use", id=1):
            with gr.Column(elem_id="tutorial-container"):
                gr.Markdown("""
                <div style='text-align: center; margin-bottom: 20px;'>
                    <h1 style='color: #fff; font-size: 2.5em; text-shadow: 0 2px 4px rgba(0,0,0,0.2);'>How to Use ClothingGAN</h1>
                </div>
                """)
                
                with gr.Group(elem_classes="container"):
                    gr.Markdown("""
                    <h2>🎯 Getting Started</h2>
                    <p>Welcome to ClothingGAN! This tool allows you to generate and customize clothing designs using AI. Here's how to get started:</p>
                    
                    <h3>1. Basic Controls</h3>
                    <ul>
                        <li><strong>Input Seeds:</strong> These control the base images used for generation. Try different combinations to find interesting starting points.</li>
                        <li><strong>Content Mix:</strong> Controls how much of the first image's structure is preserved (0 = all from second image, 1 = all from first image)</li>
                        <li><strong>Style Mix:</strong> Controls how much of the second image's style is applied (0 = all from first image, 1 = all from second image)</li>
                        <li><strong>Truncation:</strong> Controls the diversity of generated images (lower values = more conservative, higher values = more diverse)</li>
                    </ul>
                    
                    <h3>2. Style Controls</h3>
                    <ul>
                        <li><strong>Sleeve Length:</strong> Adjust from short to long sleeves (-10 to 10)</li>
                        <li><strong>Size:</strong> Control how tight or loose the clothing is (-10 to 10)</li>
                        <li><strong>Dress-Jacket:</strong> Mix between dress and jacket styles (-10 to 10)</li>
                        <li><strong>Female Coat:</strong> Adjust coat features specifically for female clothing (-10 to 10)</li>
                        <li><strong>Coat:</strong> General coat adjustments for any gender (-10 to 10)</li>
                        <li><strong>Graphics:</strong> Add or remove graphic elements (-10 to 10)</li>
                        <li><strong>Dark:</strong> Control darkness of the clothing (-10 to 10)</li>
                        <li><strong>Cleavage:</strong> Adjust neckline and cleavage (-10 to 10)</li>
                    </ul>
                    
                    <h3>3. Color Controls</h3>
                    <ul>
                        <li><strong>Red/Green/Blue:</strong> Adjust the color balance of the clothing (-100% to 100%)</li>
                        <li><strong>Tip:</strong> Use small adjustments (10-20%) for subtle changes, larger values for dramatic effects</li>
                    </ul>
                    
                    <h3>4. Age & Size Controls</h3>
                    <ul>
                        <li><strong>Age:</strong> Adjust the perceived age of the clothing style (-10 to 10)</li>
                        <li><strong>Size:</strong> Control the overall size of the clothing (-10 to 10)</li>
                    </ul>
                    
                    <h2>💡 Tips & Tricks</h2>
                    <ul>
                        <li>Use the "Generate Random" button to quickly explore different combinations</li>
                        <li>Save your favorite designs using the "Save Images" button</li>
                        <li>Fine-tune your design by making small adjustments to the sliders</li>
                        <li>Use the gallery to keep track of your recent generations</li>
                        <li>Try combining multiple style controls for unique effects</li>
                        <li>Use the same seeds with different style settings to create variations</li>
                    </ul>
                    
                    <h2>🎨 Advanced Usage</h2>
                    <p>For more advanced users:</p>
                    <ul>
                        <li><strong>Layer Control:</strong> Different style controls affect different layers of the network</li>
                        <li><strong>Style Mixing:</strong> Combine multiple style controls for complex effects</li>
                        <li><strong>Color Blending:</strong> Use the color controls with the style controls for unique color patterns</li>
                        <li><strong>Seed Exploration:</strong> Try different seed combinations to find interesting base images</li>
                    </ul>
                    
                    <h2>⚠️ Common Issues & Solutions</h2>
                    <ul>
                        <li><strong>Blurry Images:</strong> Try adjusting the truncation value or using different seeds</li>
                        <li><strong>Unwanted Artifacts:</strong> Reduce the strength of style controls or try different seed combinations</li>
                        <li><strong>Color Issues:</strong> Use smaller color adjustments or try different base images</li>
                    </ul>
                    
                    <h2>📚 Best Practices</h2>
                    <ul>
                        <li>Start with moderate values and make small adjustments</li>
                        <li>Save your favorite settings for future reference</li>
                        <li>Experiment with different combinations of controls</li>
                        <li>Use the gallery to track your progress</li>
                    </ul>
                    """)

    def update_seed_display(seed1, seed2):
        return (
            f"Input Seeds: {seed1}, {seed2}",
            f"Output Generated from Seeds: {seed1}, {seed2}"
        )

    # Update on input changes
    for inp in [seed1, seed2, content, style, truncation,
                c0, c1, c2, c3, c4, c5, c6, c7,
                red, green, blue, age, size]:
        inp.change(fn=generate_with_gallery, inputs=[seed1, seed2, content, style, truncation,
                                              c0, c1, c2, c3, c4, c5, c6, c7,
                                              red, green, blue, age, size,
                                              gallery], outputs=[input_image, output_image, gallery])
    
    # Update seed display when seeds change
    seed1.change(fn=update_seed_display, inputs=[seed1, seed2], outputs=[input_seed_display, output_seed_display])
    seed2.change(fn=update_seed_display, inputs=[seed1, seed2], outputs=[input_seed_display, output_seed_display])
    
    # Random generation with only seed updates
    random_button.click(
        fn=generate_random_image,
        inputs=[
            content, style, truncation,  # Generation controls
            c0, c1, c2, c3, c4, c5, c6, c7,  # Added c7 to style controls
            red, green, blue,  # Color controls
            age, size,  # Age and size controls
            gallery  # Current gallery state
        ],
        outputs=[
            input_image, output_image,  # Images
            input_seed_display, output_seed_display,  # Seed displays
            seed1, seed2,  # Only update seed inputs
            gallery  # Update gallery with new generation
        ]
    )
    
    # Save functionality
    save_button.click(fn=save_current_image, inputs=[input_image, output_image], outputs=result_text)

    # Add footer
    gr.Markdown("""
    <div class='footer'>
        <div class='footer-content'>
            <div class='footer-section'>
                📝 <strong>Tips:</strong> Use the sliders to adjust clothing style, size, and colors. Click 'Generate Random' for new designs.
            </div>
            <div class='footer-section'>
                💡 <strong>Features:</strong> Mix styles, adjust ages, change colors, and save your favorite designs.
            </div>
            <div class='footer-section'>
                🎨 <strong>Controls:</strong> Content mix affects overall shape, style mix affects details and patterns.
            </div>
        </div>
        <div class='footer-content' style='margin-top: 1rem;'>
            <div class='footer-section'>
                🔍 <strong>About:</strong> ClothingGAN is an AI-powered fashion design tool using StyleGAN2 technology.
            </div>
            <div class='footer-section'>
                📱 <strong>Contact:</strong> For support or feedback, reach out on <a href='#'>Twitter</a> or <a href='#'>GitHub</a>
            </div>
        </div>
    </div>
    """)

    # Add custom styles
    gr.Markdown("""
    <style>
    .slider-container {
        margin: 1.5rem 0;
        padding: 0.5rem;
        background: rgba(0,0,0,0.2);
        border-radius: 8px;
        transition: all 0.3s ease;
    }

    .slider-container:hover {
        background: rgba(0,0,0,0.3);
        transform: translateY(-2px);
    }

    .slider-container label {
        display: block;
        margin-bottom: 0.5rem;
        color: rgba(255,255,255,0.9);
        font-weight: 500;
    }

    .slider {
        width: 100%;
        height: 6px;
        background: rgba(255,255,255,0.1);
        border-radius: 3px;
        outline: none;
        -webkit-appearance: none;
        margin: 1rem 0;
        transition: all 0.2s ease;
    }

    .slider::-webkit-slider-thumb {
        -webkit-appearance: none;
        width: 20px;
        height: 20px;
        background: rgba(59, 130, 246, 0.9);
        border-radius: 50%;
        cursor: pointer;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }

    .slider::-webkit-slider-thumb:hover {
        transform: scale(1.2);
        background: rgba(59, 130, 246, 1);
    }

    .slider-value {
        display: inline-block;
        min-width: 3em;
        color: rgba(255,255,255,0.9);
        font-weight: 500;
        text-align: center;
        background: rgba(59, 130, 246, 0.2);
        padding: 0.2em 0.5em;
        border-radius: 4px;
        margin-left: 0.5rem;
    }

    .slider-description {
        display: flex;
        justify-content: space-between;
        margin-top: 0.5rem;
        color: rgba(255,255,255,0.7);
        font-size: 0.9em;
    }

    .min-label, .max-label {
        padding: 0.2em 0.5em;
        background: rgba(0,0,0,0.2);
        border-radius: 4px;
    }
    </style>
    """)

if __name__ == "__main__":
    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)
    demo.launch(share=True) 