import tkinter as tk
import customtkinter as ctk 
from PIL import Image, ImageTk
import torch
from diffusers import StableDiffusionPipeline 

# Create the app
app = tk.Tk()
app.geometry("532x632")
app.title("Stable Bud") 
ctk.set_appearance_mode("dark") 

prompt = ctk.CTkEntry(app, height=40, width=512, font=("Arial", 20), text_color="black", fg_color="white") 
prompt.place(x=10, y=10)

lmain = ctk.CTkLabel(app, height=512, width=512)
lmain.place(x=10, y=110)

modelid = "CompVis/stable-diffusion-v1-4"
device = "mps"  # Change to MPS for Apple Silicon

# Load Stable Diffusion
pipe = StableDiffusionPipeline.from_pretrained(modelid, torch_dtype=torch.float16) 
pipe.to(device) 

def generate(): 
    prompt_text = prompt.get()
    image = pipe(prompt_text, guidance_scale=8.5).images[0]  # Corrected key for accessing the image
    
    image.save('generatedimage.png')
    img = ImageTk.PhotoImage(image)
    lmain.configure(image=img)
    lmain.image = img  # Prevent garbage collection

trigger = ctk.CTkButton(app, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue", command=generate, text="Generate")
trigger.configure(text="Generate") 
trigger.place(x=206, y=60) 

app.mainloop()