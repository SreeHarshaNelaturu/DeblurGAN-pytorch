import numpy as np
import torch
import runway
from runway.data_types import *
from deblur_image import *
@runway.setup(options={'checkpoint': runway.file(extension='.pkl')})
def setup(opts):
    print("++++++ Loading Model +++++++")
    checkpoint = torch.load(opts['checkpoint'])
    config = checkpoint['config']
    print("++++++ Model Loaded ++++++")

    return checkpoint, config
inputs = {"blurred_images_directory" : file(is_directory=True)}
outputs = {"output_images_directory" : file(is_directory=True)}


@runway.command('deblur_image', inputs=inputs, outputs=outputs, description='Deblur an Image')
def deblur_image(model, args):
    checkpoint, config = model
    data_loader = CustomDataLoader(data_dir=args["blurred_images_directory"])
    generator_class = getattr(module_arch, config['generator']['type'])
    generator = generator_class(**config['generator']['args'])

    # prepare model for deblurring
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    generator.to(device)
    if config['n_gpu'] > 1:
        generator = torch.nn.DataParallel(generator)

    generator.load_state_dict(checkpoint['generator'])

    generator.eval()

    # start to deblur
    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(data_loader, ascii=True)):
            blurred = sample['blurred'].to(device)
            image_name = sample['image_name'][0]

            deblurred = generator(blurred)
            deblurred_img = to_pil_image(denormalize(deblurred).squeeze().cpu())

            deblurred_img.save(os.path.join(args["output_images_directory"], 'deblurred ' + image_name))

if __name__ == '__main__':
    runway.run()
