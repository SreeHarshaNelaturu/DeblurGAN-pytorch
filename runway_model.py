import numpy as np
import torch
import runway
from runway.data_types import *
from deblur_image import *
@runway.setup(options={'checkpoint': file(extension='.pth')})
def setup(opts):
    print("++++++ Loading Model +++++++")
    model = torch.load(opts['checkpoint'])
    print("++++++ Model Loaded ++++++")

    return model
inputs = {"blurred": image}
outputs = {"deblurred": image}


@runway.command('deblur_image', inputs=inputs, outputs=outputs, description='Deblur an Image')
def deblur_image(model, inputs):
    config = model['config']
    generator_class = getattr(module_arch, config['generator']['type'])
    generator = generator_class(**config['generator']['args'])

    # prepare model for deblurring
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    generator.to(device)
    if config['n_gpu'] > 1:
        generator = torch.nn.DataParallel(generator)

    generator.load_state_dict(model['generator'])

    generator.eval()

    # start to deblur
    with torch.no_grad():
        blurred = inputs['blurred']
        transform = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        blurred = transform(blurred)
        h = blurred.size[1]
        w = blurred.size[0]
        new_h = h - h % 4 + 4 if h % 4 != 0 else h
        new_w = w - w % 4 + 4 if w % 4 != 0 else w
        blurred = transforms.Resize([new_h, new_w], Image.BICUBIC)(blurred)
        blurred.unsqueeze_(0)
        print(blurred.shape)
        blurred = blurred.to(device)
        deblurred = generator(blurred)
        deblurred_img = to_pil_image(denormalize(deblurred).squeeze().cpu())

        #deblurred_img.save("./deblurred.png")
        return deblurred_img
if __name__ == '__main__':
    runway.run()
