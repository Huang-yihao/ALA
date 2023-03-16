import argparse
import cv2
import torch
import os
from torchvision import models, transforms
from tqdm import tqdm
from support import RGB2Lab_t, Lab2RGB_t, light_filter, Normalize, update_paras

norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

device  = torch.device("cuda:0")

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser(description='Adversarial Lightness Attack')


parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--image_size', type=int, default=224)
parser.add_argument('--segment', type=int, default=64)
parser.add_argument('--lr', type=int, default=0.5)
parser.add_argument('--model', type=str, default="resnet50")
parser.add_argument('--input_path', type=str, default="./dataset/")
parser.add_argument('--output_path', type=str, default="./result/")
parser.add_argument('--random_init', type=bool, default=False)
parser.add_argument('--init_range', type=list, default=[0, 1])
parser.add_argument('--tau', type=int, default=-0.2)
parser.add_argument('--eta', type=int, default=0.3)

args = parser.parse_args()

epochs = args.epochs
batch_size=args.batch_size
image_size=args.image_size
segment=args.segment
lr=args.lr
input_path = args.input_path
output_path = args.output_path+args.model+'_'+str(segment)+'_lr_'+str(lr)+'_iter_'+str(epochs)+'/'
if args.model == 'resnet50':
    model = models.resnet50(pretrained=True).eval()
elif args.model == 'vgg19':
    model = models.vgg19(pretrained=True).eval()
elif args.model == 'densenet121':
    model = models.densenet121(pretrained=True).eval()
elif args.model == 'mobilenet_v2':
    model = models.mobilenet_v2(pretrained=True).eval()
model.to(device)

image_id_list = list(filter(lambda x: '.png' in x, os.listdir(input_path)))

trn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((image_size, image_size)),])


crit = ["total", "success", "fail"]
if not os.path.exists(output_path) :
    os.makedirs(output_path)
for c in crit:
    if not os.path.exists(output_path + c + '/'):
        os.makedirs(output_path + c + '/')
for k in tqdm(range(len(image_id_list))):
    if k >= 0:
        image_ori = cv2.imread(input_path + image_id_list[k])
        image_ori = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
        image_ori = trn(image_ori).unsqueeze(0).cuda()
        image_ori = transforms.ToPILImage()(image_ori.squeeze(0).cpu())
        image_ori = cv2.imread(input_path+image_id_list[k], 1)
        X_ori = (RGB2Lab_t(torch.from_numpy(image_ori).cuda()/1.0) + 128)/255.0
        image_ori = cv2.imread(input_path+image_id_list[k])
        image_ori = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)

        image_ori = trn(image_ori).unsqueeze(0).cuda()
        X_ori = X_ori.unsqueeze(0)
        X_ori = X_ori.type(torch.FloatTensor)
        best_adversary = image_ori.clone()
        best_adversary = best_adversary.cuda()
        mid_image = transforms.ToPILImage()(image_ori.squeeze(0).cpu())
        invar, X_ori  = torch.split(X_ori,[1,2],dim=1)
        light_max = torch.max(invar, dim=2)[0]
        light_max = torch.max(light_max, dim=2)[0]
        light_min = torch.min(invar, dim=2)[0]
        light_min = torch.min(light_min, dim=2)[0]
        X_ori = X_ori.cuda()
        invar = invar.cuda()

        labels=torch.argmax(model(norm(image_ori)),dim=1)
        labels_onehot = torch.zeros(labels.size(0), 1000, device=device)
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
        labels_infhot = torch.zeros_like(labels_onehot).scatter_(1, labels.unsqueeze(1), float('inf'))

        if args.random_init:
            Paras_light = torch.rand(batch_size, 1, segment).to(device)
            total_range = args.init_range[1] - args.init_range[0]
            Paras_light = Paras_light * total_range + args.init_range[0]
        else:
            Paras_light = torch.ones(batch_size, 1, segment).to(device)
        Paras_light.requires_grad = True

        for iteration in range(epochs):

            X_adv_light = light_filter(invar, Paras_light, segment, light_max.cuda(), light_min.cuda())

            X_adv = torch.cat((X_adv_light, X_ori), dim=1)*255.0
            X_adv = X_adv.squeeze(0)
            X_adv = Lab2RGB_t(X_adv-128)/255.0
            X_adv = X_adv.type(torch.FloatTensor)

            mid_image = transforms.ToPILImage()(X_adv)

            X_adv = X_adv.unsqueeze(0).cuda()
            logits = model(norm(X_adv))
            real = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
            other = (logits - labels_infhot).max(1)[0]
            adv_loss = torch.clamp(real - other, min=args.tau).sum()
            paras_loss = 1-torch.abs(Paras_light).sum() / segment
            factor = args.eta
            loss = adv_loss + factor * paras_loss
            loss.backward(retain_graph=True)

            update_paras(Paras_light, lr, batch_size)

            x_result = trn(mid_image).unsqueeze(0).cuda()
            predicted_classes = (model(norm(x_result))).argmax(1)
            is_adv = (predicted_classes != labels)

            def save_stat(criterion):
                best_adversary[is_adv] = x_result[is_adv]
                x_np = transforms.ToPILImage()(best_adversary[0].detach().cpu())
                x_np.save(os.path.join(output_path+ criterion + '/', image_id_list[k * batch_size][:-4] + '.png'))
            if is_adv:
                save_stat("success")

        for j in range(batch_size):
            x_np=transforms.ToPILImage()(best_adversary[j].detach().cpu())
            if labels[j]==(model(norm(best_adversary)))[j].argmax(0):
                x_np.save(os.path.join(output_path + "fail" + '/', image_id_list[k * batch_size +j][:-4] + '.png'))
                mid_image.save(os.path.join(output_path + "total" + '/', image_id_list[k * batch_size + j][:-4] + '.png'))

torch.cuda.empty_cache()
