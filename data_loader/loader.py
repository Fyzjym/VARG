import random
from torch.utils.data import Dataset
import os
import torch
import numpy as np
import pickle
from torchvision import transforms
import lmdb
from PIL import Image
import torchvision
import cv2
from einops import rearrange, repeat
import time
import torch.nn.functional as F

# text_path = {'train':'data/IAM64_train.txt',
#              'test':'data/IAM64_test.txt'}

text_path = {
    'train': 'path2/crohme2019_diffusion/crohme_train_img_wid_label_more2imgv2.csv',
    'test': 'path2/crohme2019_diffusion/crohme_tra_test_1_2.csv'}

generate_type = {'iv_s': ['train', 'data/in_vocab.subset.tro.37'],
                 'iv_u': ['test', 'data/in_vocab.subset.tro.37'],
                 'oov_s': ['train', 'data/oov.common_words'],
                 'oov_u': ['test', 'data/oov.common_words'],
                 'train_all': ['train'],
                 'test_all': ['test'],
                 }

# define the letters and the width of style image
letters = ["!", "(", ")", "+", ",", "-", ".", "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "=", "A", "B", "C",
           "E", "F", "G", "H", "I", "L", "M", "N", "P", "R", "S", "T", "V", "X", "Y", "[", "\\Delta", "\\alpha",
           "\\beta", "\\cos", "\\div", "\\exists", "\\forall", "\\gamma", "\\geq", "\\gt", "\\in", "\\infty", "\\int",
           "\\lambda", "\\ldots", "\\leq", "\\lim", "\\log", "\\lt", "\\mu", "\\neq", "\\phi", "\\pi", "\\pm",
           "\\prime", "\\rightarrow", "\\sigma", "\\sin", "\\sqrt", "\\sum", "\\tan", "\\theta", "\\times", "\\{",
           "\\}", "]", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s",
           "t", "u", "v", "w", "x", "y", "z", "|"]

style_len = 256

# latex_max_length = 256  # exp01
latex_max_length = 128 # exp02，exp03

"""prepare the IAM dataset for training"""


class IAMDataset(Dataset):
    def __init__(self, image_path, style_path, laplace_path, content_path, type, content_type='unifont', max_len=96):
        self.max_len = max_len
        self.style_len = style_len
        self.data_dict = self.load_data(text_path[type])
        self.image_path = os.path.join(image_path, type)
        self.style_path = os.path.join(style_path, type)
        self.laplace_path = os.path.join(laplace_path, type)
        self.content_path = os.path.join(content_path, type)

        self.max_length = latex_max_length

        self.letters = letters
        self.tokens = {"PAD_TOKEN": len(self.letters)}
        self.letter2index = {label: n for n, label in enumerate(self.letters)}
        self.indices = list(self.data_dict.keys())
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.resize_transform = torchvision.transforms.Resize([256, 256], interpolation=Image.NEAREST)  # 缩放到256x256

        # self.content_transform = torchvision.transforms.Resize([64, 32], interpolation=Image.NEAREST)
        # self.con_symbols = self.get_symbols(content_type)
        # self.laplace = torch.tensor([[0, 1, 0],[1, -4, 1],[0, 1, 0]], dtype=torch.float
        #                             ).to(torch.float32).view(1, 1, 3, 3).contiguous()

    def load_data(self, data_path):
        with open(data_path, 'r') as f:
            train_data = f.readlines()
            train_data = [i.strip().split('  ') for i in train_data]
            full_dict = {}
            idx = 0
            max_len = 0
            for i in train_data:
                # print(i) # ['101_alfonso', 'alfonso', '000', 'S = ( \\sum _ { i = 1 } ^ { n } \\theta _ { i } - ( n - 2 ) \\pi ) r ^ { 2 }']
                s_id = i[2]
                image = i[0] + '.png'
                transcription = i[3]
                if len(transcription.split(' ')) > self.max_len:
                    continue
                full_dict[idx] = {'image': image, 's_id': s_id, 'label': transcription}
                idx += 1

        return full_dict

    def get_style_ref(self, wr_id):
        style_list = os.listdir(os.path.join(self.style_path, wr_id))
        style_index = random.sample(range(len(style_list)), 2)  # anchor and positive
        style_images = [cv2.imread(os.path.join(self.style_path, wr_id, style_list[index]), flags=0)
                        for index in style_index]
        laplace_images = [cv2.imread(os.path.join(self.laplace_path, wr_id, style_list[index]), flags=0)
                          for index in style_index]

        height = style_images[0].shape[0]
        assert height == style_images[1].shape[0], 'the heights of style images are not consistent'
        max_w = max([style_image.shape[1] for style_image in style_images])

        '''style images'''
        style_images = [style_image / 255.0 for style_image in style_images]
        new_style_images = np.ones([2, height, max_w], dtype=np.float32)
        new_style_images[0, :, :style_images[0].shape[1]] = style_images[0]
        new_style_images[1, :, :style_images[1].shape[1]] = style_images[1]

        '''laplace images'''
        laplace_images = [laplace_image / 255.0 for laplace_image in laplace_images]
        new_laplace_images = np.zeros([2, height, max_w], dtype=np.float32)
        new_laplace_images[0, :, :laplace_images[0].shape[1]] = laplace_images[0]
        new_laplace_images[1, :, :laplace_images[1].shape[1]] = laplace_images[1]
        return new_style_images, new_laplace_images

    def __len__(self):
        return len(self.indices)

    ### Borrowed from GANwriting ###
    def label_padding(self, labels, max_len):
        ll = [self.letter2index[i] for i in labels]
        num = max_len - len(ll)
        if not num == 0:
            ll.extend([self.tokens["PAD_TOKEN"]] * num)  # replace PAD_TOKEN
        return ll

    def __getitem__(self, idx):
        image_name = self.data_dict[self.indices[idx]]['image']
        label = self.data_dict[self.indices[idx]]['label']
        wr_id = self.data_dict[self.indices[idx]]['s_id']
        transcr = label  # expression latex seq
        img_path = os.path.join(self.image_path, wr_id, image_name)
        image = Image.open(img_path).convert('RGB')
        image = self.transforms(image)

        content_path = os.path.join(self.content_path, wr_id, image_name)
        content_arc = Image.open(content_path).convert('RGB')  # 直接以灰度图方式读取
        content_arc = self.transforms(content_arc)

        style_ref, laplace_ref = self.get_style_ref(wr_id)
        style_ref = torch.from_numpy(style_ref).to(torch.float32)  # [2, h , w] achor and positive
        laplace_ref = torch.from_numpy(laplace_ref).to(torch.float32)  # [2, h , w] achor and positive

        image = self.resize_transform(image)
        content_arc = self.resize_transform(content_arc)
        style_ref = self.resize_transform(style_ref)
        laplace_ref = self.resize_transform(laplace_ref)

        return {'img': image,
                "content_arc": content_arc,
                'content': label,
                'style': style_ref,
                "laplace": laplace_ref,
                'wid': int(wr_id),
                'transcr': transcr,
                'image_name': image_name}

    def collate_fn_(self, batch):
        width = [item['img'].shape[2] for item in batch]
        c_width = [len(item['content']) for item in batch]
        s_width = [item['style'].shape[2] for item in batch]

        transcr = [item['transcr'] for item in batch]
        # target_lengths = torch.IntTensor([len(t) for t in transcr])
        target_lengths = self.max_length
        image_name = [item['image_name'] for item in batch]

        # fixed size 256*256
        max_width = 256
        max_s_width = min(max(s_width), self.style_len) if max(s_width) < self.style_len else self.style_len

        imgs = torch.ones([len(batch), batch[0]['img'].shape[0], max_width, max_width], dtype=torch.float32)
        # content_ref = torch.zeros([len(batch), max(c_width), 16 , 16], dtype=torch.float32)
        content_arc = torch.ones([len(batch), batch[0]['content_arc'].shape[0], max_width, max_width],
                                 dtype=torch.float32)

        style_ref = torch.ones([len(batch), batch[0]['style'].shape[0], max_width, max_width], dtype=torch.float32)
        laplace_ref = torch.zeros([len(batch), batch[0]['laplace'].shape[0], max_width, max_width], dtype=torch.float32)

        # target = torch.zeros([len(batch), max(target_lengths)], dtype=torch.int32)
        target = torch.zeros([len(batch), target_lengths], dtype=torch.int32)

        for idx, item in enumerate(batch):

            try:
                imgs[idx, :, :, 0:item['img'].shape[2]] = item['img']
                content_arc[idx, :, :, 0:item['content_arc'].shape[2]] = item['content_arc']

            except:
                print('img', item['img'].shape)
                print('content_arc', item['content_arc'].shape)

            # try:
            #     content = [self.letter2index[i] for i in item['content']]
            #     content = self.con_symbols[content]
            #     content_ref[idx, :len(content)] = content
            # except:
            #     print('content', item['content'])

            """
            filter
            """
            valid_letters = set(self.letters)  #
            filtered_transcr = [t for t in transcr[idx].split(' ') if t in valid_letters]

            target[idx, :len(filtered_transcr)] = torch.Tensor([self.letter2index[t] for t in filtered_transcr])

            # target[idx, :len(transcr[idx])] = torch.Tensor([self.letter2index[t] for t in transcr[idx].split(' ')]) # raw

            try:
                if max_s_width < self.style_len:
                    style_ref[idx, :, :, 0:item['style'].shape[2]] = item['style']
                    laplace_ref[idx, :, :, 0:item['laplace'].shape[2]] = item['laplace']
                else:
                    style_ref[idx, :, :, 0:item['style'].shape[2]] = item['style'][:, :, :self.style_len]
                    laplace_ref[idx, :, :, 0:item['laplace'].shape[2]] = item['laplace'][:, :, :self.style_len]
            except:
                print('style', item['style'].shape)

        wid = torch.tensor([item['wid'] for item in batch])
        # content_ref = 1.0 - content_ref # invert the image # our img different the unifont.pickle
        # return {'img':imgs, 'style':style_ref, 'content':content_ref, 'wid':wid, 'laplace':laplace_ref,
        #         'target':target, 'target_lengths':target_lengths, 'image_name':image_name}
        # print('***  collate_fn_  ***')
        # print(imgs.shape)
        # print(style_ref.shape)
        # print(content_arc.shape)
        # print(laplace_ref.shape)
        # print(target.shape)
        # print(target[0])
        # print(image_name[0])
        return {'img': imgs, 'style': style_ref, 'content': content_arc, 'wid': wid, 'laplace': laplace_ref,
                'target': target, 'target_lengths': target_lengths, 'image_name': image_name, 'latex_seq':transcr}


"""random sampling of style images during inference"""


class Random_StyleIAMDataset(IAMDataset):
    def __init__(self, style_path, lapalce_path, content_path, ref_num) -> None:
        self.style_path = style_path
        self.laplace_path = lapalce_path
        self.content_path = content_path

        self.author_id = os.listdir(os.path.join(self.style_path))
        self.style_len = style_len
        self.ref_num = ref_num
        self.resize_transform = torchvision.transforms.Resize([256, 256], interpolation=Image.NEAREST)  # 缩放到256x256

    def __len__(self):
        return self.ref_num

    def get_style_ref(self, wr_id):  # Choose the style image whose length exceeds 32 pixels
        style_list = os.listdir(os.path.join(self.style_path, wr_id))
        random.shuffle(style_list)
        for index in range(len(style_list)):
            style_ref = style_list[index]

            style_image = cv2.imread(os.path.join(self.style_path, wr_id, style_ref), flags=0)
            laplace_image = cv2.imread(os.path.join(self.laplace_path, wr_id, style_ref), flags=0)
            if style_image.shape[1] > 128:
                break
            else:
                continue
        style_image = style_image / 255.0
        laplace_image = laplace_image / 255.0
        return style_image, laplace_image

    def __getitem__(self, _):
        batch = []
        for idx in self.author_id:
            style_ref, laplace_ref = self.get_style_ref(idx)
            style_ref = torch.from_numpy(style_ref).unsqueeze(0)
            style_ref = style_ref.to(torch.float32)
            laplace_ref = torch.from_numpy(laplace_ref).unsqueeze(0)
            laplace_ref = laplace_ref.to(torch.float32)
            wid = idx
            style_ref = self.resize_transform(style_ref)
            laplace_ref = self.resize_transform(laplace_ref)
            batch.append({'style': style_ref, 'laplace': laplace_ref, 'wid': wid})

        s_width = [item['style'].shape[2] for item in batch]
        if max(s_width) < self.style_len:
            max_s_width = max(s_width)
        else:
            max_s_width = self.style_len

        max_width = 256

        style_ref = torch.ones([len(batch), batch[0]['style'].shape[0], max_width, max_width], dtype=torch.float32)
        laplace_ref = torch.zeros([len(batch), batch[0]['laplace'].shape[0], max_width, max_width], dtype=torch.float32)
        wid_list = []
        for idx, item in enumerate(batch):
            try:
                if max_s_width < self.style_len:
                    style_ref[idx, :, :, 0:item['style'].shape[2]] = item['style']
                    laplace_ref[idx, :, :, 0:item['laplace'].shape[2]] = item['laplace']
                else:
                    style_ref[idx, :, :, 0:item['style'].shape[2]] = item['style'][:, :, :self.style_len]
                    laplace_ref[idx, :, :, 0:item['laplace'].shape[2]] = item['laplace'][:, :, :self.style_len]
                wid_list.append(item['wid'])
            except:
                print('style', item['style'].shape)

        return {'style': style_ref, 'laplace': laplace_ref, 'wid': wid_list}


# """prepare the content image during inference"""
# class ContentData(IAMDataset):
#     def __init__(self, content_type='unifont') -> None:
#         self.letters = letters
#         self.letter2index = {label: n for n, label in enumerate(self.letters)}
#         self.con_symbols = self.get_symbols(content_type)
#
#     def get_content(self, label):
#         word_arch = [self.letter2index[i] for i in label]
#         content_ref = self.con_symbols[word_arch]
#         content_ref = 1.0 - content_ref
#         return content_ref.unsqueeze(0)


class ContentData(IAMDataset):
    """
    prepare one content image during inference
    """

    def __init__(self, content_path=None) -> None:
        self.content_acr_path = content_path
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.resize_transform = torchvision.transforms.Resize([256, 256], interpolation=Image.NEAREST)  # 缩放到256x256

    def get_content(self, ):
        content_arc = Image.open(self.content_acr_path).convert('RGB')
        content_arc = self.transforms(content_arc)
        content_arc = self.resize_transform(content_arc)
        content_arc = content_arc.unsqueeze(0)
        content_arc = content_arc[:, 0, :, :].unsqueeze(1).contiguous()

        return content_arc


class ContentDataSet(IAMDataset):
    """
    prepare the N content image during inference
    """

    def __init__(self, contentset_path=None) -> None:
        self.content_acr_set_path = contentset_path
        self.content_list = os.listdir(self.content_acr_set_path)

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.resize_transform = torchvision.transforms.Resize([256, 256], interpolation=Image.NEAREST)  # 缩放到256x256

    def get_content(self, ):
        content_arc_list = []
        img_list = []
        for ipath in self.content_list:
            content_arc = Image.open(os.path.join(self.content_acr_set_path, ipath)).convert('RGB')
            content_arc = self.transforms(content_arc)
            content_arc = self.resize_transform(content_arc)
            content_arc = content_arc.unsqueeze(0)
            content_arc = content_arc[:, 0, :, :].unsqueeze(1).contiguous()
            content_arc_list.append(content_arc)
            img_list.append(ipath)
        content_arc_combined = torch.cat(content_arc_list, dim=0)

        return content_arc_combined, img_list


class latexData(IAMDataset):
    """
    latex 2 traget
    """

    def __init__(self, latex_str=None) -> None:
        self.latex_str = latex_str
        self.letters = letters
        self.letter2index = {label: n for n, label in enumerate(self.letters)}

    def get_content(self, ):
        # Assuming batch is a list containing your latex string
        batch = [self.latex_str]
        max_target_lengths = latex_max_length
        target = torch.zeros([len(batch), max_target_lengths], dtype=torch.int32)

        for idx, transcr in enumerate(batch):
            valid_letters = set(self.letters)
            filtered_transcr = [t for t in transcr.split(' ') if t in valid_letters]
            target[idx, :len(filtered_transcr)] = torch.Tensor([self.letter2index[t] for t in filtered_transcr])

        return target


class LatexDataSet(IAMDataset):
    """
    get N latex ttaget base image name list.
    """

    def __init__(self, img_name_list=None, mode='train') -> None:
        self.cop = 'path2/crohme2019_diffusion/crohme_train_img_wid_label_alllatex.csv'
        self.name_list = img_name_list
        self.all_lines = self.load_all_lines()  # 预加载文件内容

    def load_all_lines(self):
        with open(self.cop, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        return lines

    def find_latex_by_name(self, image_name):
        for line in self.all_lines:
            A, B, C, D = line.split('  ')
            if image_name.split('.')[0] == A:
                return D
        return None

    def get_content(self):
        target_l = []

        for i_name in self.name_list:
            latex = self.find_latex_by_name(i_name)

            if latex is not None:

                latex_obj = latexData(latex_str=latex)
                latex_obj = latex_obj.get_content()
                target_l.append(latex_obj)

            else:
                print('Can not find latex seq in file.' + i_name)
                # target.append('')
        target = torch.cat(target_l, dim=0)
        return target