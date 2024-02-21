import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

class SaveFeatures:
    def __init__(self, m):
        self.features = None
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.clone().detach().requires_grad_(True).cuda()

    def close(self):
        self.hook.remove()


class FilterVisualizer:
    def __init__(self, model, size, upscaling_steps, upscaling_factor, lr, opt_steps, blur, save):
        self.model = model
        self.size = size
        self.upscaling_steps = upscaling_steps
        self.upscaling_factor = upscaling_factor
        self.lr = lr
        self.opt_steps = opt_steps
        self.blur = blur
        self.save = save
        self.features = SaveFeatures(list(self.model.children())[-1])

        for param in self.model.parameters():
            param.requires_grad = False

    def visualize(self, layer, filter, lr=0.1, opt_steps=20):
        sz = self.size
        img = np.uint8(np.random.uniform(150, 180, (sz, sz, 3)))/255  # generate random image
        activations = SaveFeatures(list(self.model.children())[layer])

        for _ in range(self.upscaling_steps):
            img_var = torch.tensor(img, requires_grad=True).cuda()
            optimizer = torch.optim.Adam([img_var], lr=lr, weight_decay=1e-6)

            for n in range(opt_steps):
                optimizer.zero_grad()
                self.model(img_var)
                loss = -activations.features[0, filter].mean()
                loss.backward()
                optimizer.step()

            img = img_var[0].cpu().detach().numpy()
            if self.blur is not None:
                img = cv2.GaussianBlur(img, (self.blur, self.blur), 0)
            sz = int(self.upscaling_factor * sz)
            img = cv2.resize(img, (sz, sz), interpolation = cv2.INTER_CUBIC)
            img = np.clip(img, 0, 1)

            def save(layer, filter, img):
                if self.save:
                    plt.imsave("layer_" + str(layer) + "_filter_" + str(filter) + ".jpg", np.uint8(img*255))

            def show_image(img):
                plt.imshow(np.clip(img, 0, 1))
                plt.show()


