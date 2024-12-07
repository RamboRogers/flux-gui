import torch
from diffusers import FluxPipeline
import gradio as gr
import os
from datetime import datetime
import PIL.Image
from threading import Lock

# Create images directory if it doesn't exist
IMAGES_DIR = "images"
os.makedirs(IMAGES_DIR, exist_ok=True)

class FluxGenerator:
    def __init__(self):
        if torch.cuda.is_available():
            # Clear CUDA cache before loading model
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            print(f"Initial GPU memory usage: {torch.cuda.memory_allocated()/1024**2:.2f}MB")

            self.device = "cuda"
            print(f"Using CUDA device: {torch.cuda.get_device_name()}")

        try:
            # Load model without device map
            self.pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                offload_folder="offload",
                offload_state_dict=True
            )

            # Enable memory optimization features
            self.pipe.enable_attention_slicing(slice_size=1)
            self.pipe.enable_sequential_cpu_offload()
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_vae_slicing()

            if torch.cuda.is_available():
                print(f"Model loaded. GPU memory: {torch.cuda.memory_allocated()/1024**2:.2f}MB")

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise e

        self.lock = Lock()

    def generate(self, prompt, height, width, guidance_scale, num_inference_steps, seed):
        with self.lock:
            try:
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    print(f"Pre-generation GPU memory: {torch.cuda.memory_allocated()/1024**2:.2f}MB")

                # Create generator on appropriate device
                if self.device == "mps":
                    generator = torch.Generator("cpu").manual_seed(seed)
                else:
                    generator = torch.Generator(device=self.device).manual_seed(seed)

                result = self.pipe(
                    prompt,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    max_sequence_length=512,
                    generator=generator
                ).images[0]

                if self.device == "cuda":
                    print(f"Post-generation GPU memory: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
                    torch.cuda.empty_cache()
                    gc.collect()

                return result

            except Exception as e:
                print(f"Generation error: {str(e)}")
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    gc.collect()
                raise e

def save_image(image: PIL.Image.Image) -> str:
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"flux_{timestamp}.png"
    filepath = os.path.join(IMAGES_DIR, filename)

    # Save the image
    image.save(filepath)
    return filepath

def memory_cleanup():
    """
    Comprehensive memory cleanup function for user processes and memory
    Returns a status message with cleanup results
    """
    try:
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024**2

        cleanup_stats = {
            'cuda_before': 0,
            'cuda_after': 0,
            'ram_before': initial_memory,
            'ram_after': 0,
            'processes_cleaned': 0
        }

        # 1. Clear CUDA memory if available
        if torch.cuda.is_available():
            cleanup_stats['cuda_before'] = torch.cuda.memory_allocated() / 1024**2
            torch.cuda.empty_cache()
            cleanup_stats['cuda_after'] = torch.cuda.memory_allocated() / 1024**2

        # 2. Force Python garbage collection
        import gc
        gc.collect()

        # 3. Clear any large objects in memory
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    del obj
            except:
                pass

        # 4. Clean up user processes related to this application
        current_process = psutil.Process()
        parent_pid = current_process.pid

        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                # Skip system processes
                if proc.ppid() == parent_pid:
                    # If process is using high memory or CPU, restart it
                    if proc.memory_percent() > 50 or proc.cpu_percent() > 80:
                        proc.kill()
                        cleanup_stats['processes_cleaned'] += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

        # 5. Final garbage collection
        gc.collect()

        # Get final memory usage
        cleanup_stats['ram_after'] = process.memory_info().rss / 1024**2

        # Prepare status message
        status = f"""Memory Cleanup Results:
        GPU Memory: {cleanup_stats['cuda_before']:.2f}MB → {cleanup_stats['cuda_after']:.2f}MB
        RAM Usage: {cleanup_stats['ram_before']:.2f}MB → {cleanup_stats['ram_after']:.2f}MB
        Processes Cleaned: {cleanup_stats['processes_cleaned']}
        """

        return status

    except Exception as e:
        return f"Memory cleanup failed: {str(e)}"

def generate_image(prompt, height, width, guidance_scale, num_inference_steps, seed):
    try:
        status = memory_cleanup()
        print(status)

        # Randomize seed if it's 0
        if seed == 0:
            seed = torch.randint(1, 9999999, (1,)).item()
            print(f"Randomized seed: {seed}")

        print(f"Generating image with parameters:")
        print(f"- Prompt: {prompt}")
        print(f"- Dimensions: {width}x{height}")
        print(f"- Guidance Scale: {guidance_scale}")
        print(f"- Steps: {num_inference_steps}")
        print(f"- Seed: {seed}")

        image = generator.generate(
            prompt,
            height,
            width,
            guidance_scale,
            num_inference_steps,
            seed
        )

        filepath = save_image(image)
        print(f"Saved image to: {filepath}")

        if torch.cuda.is_available():
            print(f"Final GPU memory usage: {torch.cuda.memory_allocated()/1024**2:.2f}MB")

        return image, get_image_list()
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        raise e

def get_image_list():
    # Get list of all images in the directory
    image_files = [os.path.join(IMAGES_DIR, f) for f in os.listdir(IMAGES_DIR)
                  if f.endswith(('.png', '.jpg', '.jpeg'))]
    # Sort by creation time (newest first)
    image_files.sort(key=os.path.getctime, reverse=True)
    return image_files

# Initialize the generator
generator = FluxGenerator()

# Create Gradio interface
with gr.Blocks(css="""
    #prompt-box textarea {
        color: white !important;
        background-color: black !important;
        border: 2px solid #FFA500 !important; /* Add a bright border */
        font-size: 1.2em !important; /* Slightly larger font */
        font-weight: bold !important; /* Bold text for readability */
        box-shadow: 0 0 10px #FFA500 !important; /* Glowing effect */
        border-radius: 8px !important; /* Rounded corners */
        padding: 10px !important; /* Add padding */
    }

    #prompt-box label {
        color: #FFA500 !important; /* Bright label to match the box */
        font-size: 1.2em !important; /* Match font size */
        font-weight: bold !important;
    }

""", title="Rogers FLUX Image Generator") as demo:
    gr.Markdown("# Rogers FLUX Image Generator")
    gr.Markdown("Make this your own")

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="A cat holding a sign that says hello world",
                elem_id="prompt-box",
                value="",
            )


            height = gr.Slider(minimum=128, maximum=1024, step=128, value=1024, label="Height")
            width = gr.Slider(minimum=128, maximum=1024, step=128, value=1024, label="Width")
            guidance_scale = gr.Slider(minimum=1, maximum=20, step=0.5, value=3.5, label="Guidance Scale")
            num_steps = gr.Slider(minimum=1, maximum=100, step=1, value=10, label="Number of Steps")
            seed = gr.Slider(minimum=0, maximum=9999999, step=1, value=0, label="Seed")
            generate_btn = gr.Button("Generate")

        with gr.Column():
            output_image = gr.Image(label="Generated Image")

    gallery = gr.Gallery(
        label="Generated Images History",
        show_label=True,
        elem_id="gallery",
        columns=4,
        rows=2,
        height="auto",
        value=get_image_list()
    )

    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, height, width, guidance_scale, num_steps, seed],
        outputs=[output_image, gallery]
    )

if __name__ == "__main__":
    # Update queue configuration syntax
    demo.queue()  # Simple queue with default settings
    # Or if you need specific settings:
    # demo.queue(max_size=1)

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
