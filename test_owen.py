# Copyright (c) owenxing1994@gmail.com
import torch
from torchvision import transforms

import argparse
import os
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from net import NestFuse_autoencoder

class IVIFDataset(Dataset):
    def __init__(self, dataset_folder):
        super().__init__()
        self.ir_folder = os.path.join(dataset_folder, 'ir')
        self.vis_folder = os.path.join(dataset_folder, 'vi')
        assert len(os.listdir(self.ir_folder)) == len(os.listdir(self.vis_folder)), "The number of images in the two folders must be the same."
        self.image_names = os.listdir(self.ir_folder)
        
        self.ir_transform = transforms.Compose([ 
            # transforms.Resize((224, 224)), 
            # transforms.Grayscale(1),
            transforms.ToTensor(),
        ])
        self.vis_transform = transforms.Compose([   
            # transforms.Resize((224, 224)), 
            # transforms.Grayscale(1),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        ir_image_path = os.path.join(self.ir_folder, img_name)
        vis_image_path = os.path.join(self.vis_folder, img_name)

        # load and transform images
        ir_image = self.ir_transform(Image.open(ir_image_path))
        vis_image = self.vis_transform(Image.open(vis_image_path))

        return img_name, ir_image, vis_image

def main(args):
    # create output directories
    if not (Path(args.model).parent / ("images_"+ Path(args.model).stem )).exists(): 
        (Path(args.model).parent / ("images_"+ Path(args.model).stem )).mkdir(parents=True)
    if not (Path(args.model).parent / ("labels_"+ Path(args.model).stem )).exists():
        (Path(args.model).parent / ("labels_"+ Path(args.model).stem )).mkdir(parents=True)

    # buil dataset
    dataset_path = args.dataset
    dataset = IVIFDataset(dataset_path)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build model
    input_nc = 1
    output_nc = 1
    nb_filter = [64, 112, 160, 208, 256]
    deepsupervision = False
    f_type = "attention_avg"
    nest_model = NestFuse_autoencoder(nb_filter, input_nc, output_nc, deepsupervision)
    nest_model.to(device)

    # load pre-trained model
    checkpoint = torch.load(args.model, map_location=torch.device('cuda:0') if torch.cuda.is_available() else 'cpu')
    nest_model.load_state_dict(checkpoint)

    print("Start inference...")
    with torch.no_grad():
        for i, (img_name, data_IR, data_VIS)  in enumerate(test_loader):
            data_IR, data_VIS = data_IR.to(device), data_VIS.to(device)

            nest_model.eval()

            # encoder
            en_r = nest_model.encoder(data_IR)
            en_v = nest_model.encoder(data_VIS)
            # fusion
            f = nest_model.fusion(en_r, en_v, f_type)
            # decoder
            img_fusion = nest_model.decoder_eval(f)[0]

            # save images
            img_fusion = img_fusion.mul(255).byte().cpu().numpy().squeeze()
            img_fusion = Image.fromarray(img_fusion)
            img_fusion.save(Path(args.model).parent / ("images_"+ Path(args.model).stem ) / img_name[0])
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Infer DenseFuse')
    parser.add_argument('--model', type=str, required=True, help='Path to the model')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    args = parser.parse_args()
    main(args)