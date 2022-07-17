import gradio as gr
import matlab.engine

eng = matlab.engine.start_matlab()

def process(img, subscribers, has_cc, hours, mins, secs, category, title):
    # Process Image
    img = img.resize((90, 160))
    image_mat = matlab.uint8(list(img.getdata()))
    image_mat.reshape((img.size[0], img.size[1], 3))
    convnet_output = eng.convnettest(image_mat)
    
    network_input = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    network_input[0] = subscribers / 1000
    network_input[1] = int(has_cc)
    network_input[2] = 3600 * hours + 60 * mins + secs
    network_input[3] = 1 if category == 'Automobile' else 0
    network_input[4] = 1 if category == 'Blog' else 0
    network_input[5] = 1 if category == 'Comedy' else 0
    network_input[6] = 1 if category == 'Entertainment' else 0
    network_input[7] = 1 if category == 'Food' else 0
    network_input[8] = 1 if category == 'Informative' else 0
    network_input[9] = 1 if category == 'News' else 0
    network_input[10] = 1 if category == 'Science' else 0
    network_input[11] = 1 if category == 'Tech' else 0
    network_input[12] = 1 if category == 'Video Games' else 0
    network_input[13] = len(title)
    network_input[14] = sum(1 for c in title if c.isupper()) / (len(title) + 1)
    network_input[15] = sum(1 for c in title if c.isdigit()) / (len(title) + 1)
    network_input[16] = sum(1 for c in title if not c.isalnum() and c != ' ') / (len(title) + 1)
    
    for i, e in enumerate(network_input):
        network_input[i] = float(e)
    
    network_output = eng.networktest(network_input)

    return round(convnet_output * 0.2 + network_output * 0.8)

with gr.Blocks() as demo:
    gr.Markdown("## Welcome to Youtube Creator Assistant!")
    gr.Markdown("Enter information about your video and click \"Run\" to see an estimate of the number of viewers.")
    img = gr.Image(label="Thumbnail", type='pil')
    sub = gr.Number(label="Number of Subscribers")
    cc = gr.Checkbox(label="This video has closed captions")
    gr.Markdown("Video Length")
    with gr.Row():
        hours = gr.Number(label="Hours")
        mins = gr.Number(label="Minutes")
        secs = gr.Number(label="Seconds")
    category = gr.Radio(label="Video Category", choices=["Automobile", "Blog", "Comedy", "Entertainment", "Food", "Informative", "News", "Science", "Tech", "Video Games"])
    title = gr.Text(label="Video Title")
    btn = gr.Button("Run")
    output = gr.Number(label="Predicted Number of Views")
    btn.click(fn=process, inputs=[img, sub, cc, hours, mins, secs, category, title], outputs=output)
demo.launch(share=True)
