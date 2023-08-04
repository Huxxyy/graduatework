import sys
import time
from PyQt5.Qt import QThread
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
import json
from rope.base import change
import os

from clint.textui import progress
import shutil
import torch

import importlib
from dalle2_pytorch import DALLE2, DiffusionPriorNetwork, DiffusionPrior, OpenAIClipAdapter, train_configs
from dalle2_pytorch.tokenizer import tokenizer
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

state_path = "./script_state.json"

# 全局变量:
# 定义decoder的版本及其获取途径
decoder_versions = [{
    "name": None
},
    {
    "name": "New 1B (Aesthetic)",
    "dalle2_install_path": "dalle2_pytorch==0.15.4",
    "decoder_path": "https://huggingface.co/laion/DALLE2-PyTorch/resolve/main/decoder/small_32gpus/latest.pth",
    "config_path": "https://huggingface.co/laion/DALLE2-PyTorch/raw/main/decoder/small_32gpus/decoder_config.json"
},
    {
    "name": "New 1.5B (Aesthetic)",
    "dalle2_install_path": "dalle2_pytorch==0.15.4",
    "decoder_path": "https://huggingface.co/laion/DALLE2-PyTorch/resolve/main/decoder/1.5B/latest.pth",
    "config_path": "https://huggingface.co/laion/DALLE2-PyTorch/raw/main/decoder/1.5B/decoder_config.json"
}, {
    "name": "New 1.5B (Laion2B)",
    "dalle2_install_path": "dalle2_pytorch==0.15.4",
    "decoder_path": "https://huggingface.co/laion/DALLE2-PyTorch/resolve/main/decoder/1.5B_laion2B/latest.pth",
    "config_path": "https://huggingface.co/laion/DALLE2-PyTorch/raw/main/decoder/1.5B_laion2B/decoder_config.json"
}]

decoder_options = [version["name"] for version in decoder_versions]
flag = 1
class MyThread_check_decoder(QThread):
    def __init__(self, text_label):
        super().__init__()
        self.text_label = text_label

    def run(self, text_label):
        while(flag==0):
            self.text_label.setText("正在加载...")
            self.text_label.repaint()
        self.text_label.clear()
# 初始化state的信息：
def init_state():
    state = {  # state is a dict
        "text_input": '',  # 初始设定，看到时候能不能在这里输入
        "text_repeat": 3,
        "prior_conditioning": 1.0,
        "img_repeat": 1,
        "decoder_conditioning": 1.7,
        "include_prompt_checkbox": True,
        "upsample_checkbox": True,
        "decoder": None,
        "model_local_paths": {
            "decoder": None,
            "decoder_config": None,
            "prior": None,
            "prior_config": None
        }
    }
    return state

def load_state():
    # 使用全局变量
    global current_state
    # 从当前的state.json文件中加载current_state信息：
    # 先检查是否存在state.json文件：
    if os.path.exists(state_path)==False:  # 不存在
        print("Your state file doesn't exsits!")
        # waring窗口：
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText("Your state file doesn't exsits!")
        msg.setWindowTitle("Warning")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.show()
        msg.exec_()
        # 初始化state并创建文件
        current_state = init_state()
        # 保存初始状态到文件
        save_state(current_state)
        # ch_decoder = Change_decoder_dialog(current_state)
        # ch_decoder.open()
        # current_state = ch_decoder.current_state
        # decoder_info = current_state["decoder"]

    # 现在都存在了：
    with open(state_path, 'r') as f:
        try:
            current_state = json.load(f)
        except:
            # 存在文件，但是内容空白，则重定义一个空白：
            current_state = {}

    # 检查open得到的cs是不是空白，如果是空白则init，等待顾客选择
    if current_state == {}:
        print(f"Fail to read the {current_state} file! Now we are initiating the state file...")
        current_state = init_state()
        save_state(current_state)
    # 现在内容都有了，看你是否拥有decoder：
    # decoder_info = current_state["decoder"]  # decoder_info is a dict

    # # 没有decoder，跳转生成：
    # if decoder_info == None:
    #     print("You didn't download any decoders! Please choose one!")
    #     warn = warning_window("You didn't download any decoders! Please choose one!")
    #     warn.open()
    #     # ^ current_state = user_choose_state(current_state)
    #     ch_decoder = Change_decoder_dialog(current_state)
    #     ch_decoder.open()
    #     current_state = ch_decoder.current_state
    #     decoder_info = current_state["decoder"]

    return current_state

# 保存当前状态到文件中
def save_state(current_state):
    state_path = "./script_state.json"
    # 更新当前state信息的json文件：
    with open(state_path, "w+") as f:
        json.dump(current_state, f)


def check_models(current_state):
    model_local_paths = current_state['model_local_paths']
    decoder_info = current_state["decoder"]
    decoder_name = decoder_info['name']
    current_state['model_local_paths']['prior'] = './models/latest_checkpoint.pth'
    # urls:
    decoder_url = decoder_info["decoder_path"]
    decoder_config_url = decoder_info["config_path"]
    prior_url = "https://huggingface.co/zenglishuci/conditioned-prior/resolve/main/vit-l-14/prior_aes_finetune.pth"
    # check local and download
    if model_local_paths['decoder'] == None or os.path.exists(model_local_paths['decoder']) == False:  # 说明未下载模型
        print("Local decoder doesn't exsits. Now it is downloading...")
        model_local_paths['decoder'] = download_models(decoder_name, decoder_url, 'decoder')
        print(f"finish downloading decoder {decoder_name}!")

    if model_local_paths['decoder_config'] == None or os.path.exists(model_local_paths['decoder_config']) == False:
        print("Local decoder config doesn't exsits. Now it is downloading...")
        model_local_paths['decoder_config'] = download_models(decoder_name, decoder_config_url, 'decoder_config')
        print(f"finish downloading decoder {decoder_name} config!")

    if os.path.exists(model_local_paths['prior']) == False:
        print("Local prior doesn't exsits. Now it is downloading...")
        model_local_paths["prior"] = download_models(decoder_name, prior_url, 'prior')
        print(f"finish downloading prior VIT-L-14!")

        current_state['model_local_paths'] = model_local_paths
    return current_state


def download_models(decoder_name, url, descrip):
    import wget
    model_dir = './models'
    # 不存在这个目录就创建：
    if os.path.exists(model_dir) == False:
        os.mkdir(model_dir)
    if descrip == 'decoder':
        decoder_local_path = model_dir + '/' + descrip + '_' + \
                             decoder_name.replace(" ", "_").replace(".", "_") + '.pth'
        if os.path.exists(decoder_local_path):
            print(f"local decoder {decoder_name} exsits, you just have to update your state.json file")
        else:
            wget.download(url, decoder_local_path)
        return decoder_local_path
    if descrip == 'prior':
        prior_local_path = model_dir + '/' + descrip + '.pth'
        if os.path.exists(prior_local_path):
            print(f"local prior exsits, you just have to update your state.json file")
        else:
            wget.download(url, prior_local_path)
        return prior_local_path


    elif descrip == "decoder_config":
        decoder_config_local_path = model_dir + '/' + descrip + '_' + \
                                    decoder_name.replace(" ", "_").replace(".", "_") + '.json'
        if os.path.exists(decoder_config_local_path):
            print(f"local decoder config {decoder_name} exsits, you just have to update your state.json file")
        else:
            wget.download(url, decoder_config_local_path)
        return decoder_config_local_path

    elif descrip == 'prior':
        prior_local_path = model_dir + '/' + descrip + ".pth"
        if os.path.exists(prior_local_path):
            print(f"local prior exsits, you just have to update your state.json file")
        else:
            wget.download(url, prior_local_path)
        return prior_local_path
    else:
        print("wrong descrip!")


current_state = init_state()
save_state(current_state)
current_state = load_state()
# 参数数值初始化：
decoder_text_conditioned = False
clip_config = None

def conditioned_on_text(config):
    try:
        return config.decoder.unets[0].cond_on_text_encodings
    except AttributeError:
        pass

    try:
        return config.decoder.condition_on_text_encodings
    except AttributeError:
        pass

    return False


# ^这里开始加载decoder模型
def load_decoder(decoder_state_dict_path, decoder_config_file_path):
    # 获取所有的模型config配置信息，放到config变量中：
    config = train_configs.TrainDecoderConfig.from_json_path(decoder_config_file_path)
    # 获取config中对于扩散模型中文本条件信息的定义，放入decoder_text_conditioned变量中
    global decoder_text_conditioned
    decoder_text_conditioned = conditioned_on_text(config)
    # 获取config中记录的clip的大小信息，放入变量clip_config中
    global clip_config
    clip_config = config.decoder.clip
    config.decoder.clip = None

    print("Decoder conditioned on text == ", decoder_text_conditioned)
    # 创建config好的型号大小的decoder模型
    decoder = config.decoder.create().to(device)
    # 加载数据到decoder中：
    decoder_state_dict = torch.load(decoder_state_dict_path, map_location='cpu')
    decoder.load_state_dict(decoder_state_dict, strict=False)
    del decoder_state_dict
    decoder.eval()
    return decoder


# ^加载prior模型：
# 这里prior模型的参数需要手动输入：
def load_prior(model_path):
    prior_network = DiffusionPriorNetwork(
        dim=768,
        depth=24,
        dim_head=64,
        heads=32,
        normformer=True,
        attn_dropout=5e-2,
        ff_dropout=5e-2,
        num_time_embeds=1,
        num_image_embeds=1,
        num_text_embeds=1,
        num_timesteps=1000,
        ff_mult=4
    )

    diffusion_prior = DiffusionPrior(
        net=prior_network,
        clip=OpenAIClipAdapter("ViT-L/14"),
        image_embed_dim=768,
        timesteps=1000,
        cond_drop_prob=0.1,
        loss_type="l2",
        condition_on_text_encodings=True,
    ).to(device)

    # 加载数据到定义好大小的prior模型中：
    state_dict = torch.load(model_path, map_location='cpu')
    if 'ema_model' in state_dict:
        print('Loading EMA Model')
        diffusion_prior.load_state_dict(state_dict['ema_model'], strict=True)
    else:
        print("Loading Standard Model")
        diffusion_prior.load_state_dict(state_dict['model'], strict=False)
    del state_dict
    return diffusion_prior


# ^加载clip模型：
def load_clip():
    clip = None
    global clip_config
    if clip_config is not None:
        clip = clip_config.create()
    return clip


# 由于有时候有多条prompt输入，因此需要将他们与图像分别对应：
def map_images(np_images, prior_repeat, decoder_repeat, prompts, upscale=4):
    # Match the images to their prompts
    # Format [{ prompt: STRING, images: [
    #  { prior_index: INT, decoder_index: INT, img: NP_ARR[64, 64, 3] }]
    # }]
    image_map = {}
    curr_index = 0
    for prompt in prompts:
        for prior_index in range(prior_repeat):
            for decoder_index in range(decoder_repeat):
                img = np_images[curr_index]
                if prompt not in image_map:
                    image_map[prompt] = []
                if isinstance(img, np.ndarray):
                    image = Image.fromarray(np.uint8(img * 255))
                    image = image.resize([dim * upscale for dim in image.size])
                else:
                    image = img
                image_map[prompt].append({
                    "prior_index": prior_index,
                    "decoder_index": decoder_index,
                    "img": image
                })
                curr_index += 1
    return image_map


def save_images(output_dir, np_images):
    import time
    os.makedirs(output_dir, exist_ok=True)
    for i, np_img in enumerate(np_images):
        image = Image.fromarray(np.uint8(np_img * 255))
        t = time.strftime("%Y%m%d%X", time.localtime())
        output_path = os.path.join(output_dir, f'{i}_{t.replace(":", "")}.png')
        image.save(output_path)


def get_prompts(clip, li_prompt):
    import json
    from itertools import zip_longest
    import io
    # text = li_prompt
    try:
        prompts_array = li_prompt
        assert isinstance(prompts_array, list)
        text_prompts = prompts_array
    except Exception as e:
        pass


    files = []  # 暂时不做训练，只生成图像
    file = None
    if len(files) > 0:
        file_name, file_info = list(files.items())[0]
        image_pil = Image.open(io.BytesIO(file_info['content'])).convert('RGB')
        transforms = T.Compose([
            T.CenterCrop(min(image_pil.size)),
            T.Resize(clip.image_size)
        ])
        image_pil = transforms(image_pil)
        file = (file_name, image_pil)

    return (text_prompts, file)



def get_image_embeddings(clip, diffusion_prior, prompt_tokens, prompt_image, text_rep: int, prior_cond_scale: float):
    if prompt_image is None:  # predict
        print("Computing embedings using prior")
        with torch.no_grad():
            image_embed = diffusion_prior.sample(
                prompt_tokens, cond_scale=prior_cond_scale).cpu().numpy()
    else:  # train
        print("Computing embeddings from example image")
        image_tensor = T.ToTensor()(prompt_image[1]).unsqueeze_(0).to(device)
        unbatched_image_embed, _ = clip.embed_image(image_tensor)
        image_embed = torch.zeros(
            len(prompt_tokens), unbatched_image_embed.shape[-1])
        for i in range(len(prompt_tokens)):
            image_embed[i] = unbatched_image_embed
        image_embed = image_embed.cpu().numpy()
    return image_embed


def upscale_dir(input_dir):
    images = os.listdir("results/swinir_real_sr_x4_large")
    for i in images:
        os.remove("results/swinir_real_sr_x4_large/"+i)
    process_cmd = f"python SwinIR/main_test_swinir.py --task real_sr --model_path experiments/pretrained_models/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth --folder_lq {input_dir} --scale 4 --large_model"
    os.system(process_cmd)
    results_dir = "./output/enhanced_images"
    if os.path.exists(results_dir)==False:
        os.mkdir(results_dir)
    output_files = sorted([file for file in os.listdir(results_dir) if '.png' in file], key=lambda e: int(e.split('_')[0]))
    upscale_dims = (256, 256, 3)
    images = [None] * len(output_files)
    for i, filename in enumerate(output_files):
        pil_img = Image.open(os.path.join(results_dir, filename))
        images[i] = pil_img
    #   shutil.rmtree(results_dir)
    #   !rm {results_dir}
    return images


def on_start(clip, diffusion_prior, decoder, li_prompt, recall_embeddings=False, recall_images=False):
    global current_state
    import time
    start = time.time()
    if os.path.exists("./output"):
        shutil.rmtree("./output")
    # 获得预测时候的输入prompt以及训练时的prompt-image对
    prompts, prompt_image = get_prompts(clip, li_prompt)
    prior_cond_scale = current_state['prior_conditioning']
    decoder_cond_scale = current_state['decoder_conditioning']
    text_rep = current_state['text_repeat']
    img_rep = current_state['img_repeat']
    include_prompt = True
    upsample = True

    prior_text_input = []
    for prompt in prompts:
        for _ in range(text_rep):
            prior_text_input.append(prompt)
    # 1. 对于每个prompt都生成其对应的token, size==([3, 256])
    tokens = tokenizer.tokenize(prior_text_input).to(device)

    # 2. 对于每个token生成对应的embedding
    if recall_embeddings:
        print("Loading embeddings")
        image_embed = np.load('img_emb_prior.npy')
    else:
        image_embed = get_image_embeddings(clip, diffusion_prior,
                                           tokens, prompt_image, text_rep, prior_cond_scale)
        np.save('img_emb_prior.npy', image_embed)

    embeddings = np.repeat(image_embed, img_rep, axis=0)
    embeddings = torch.from_numpy(embeddings).float().to(device)
    if recall_images:
        print("Loading images")
        images = np.load('images_decoder.npy')
    else:
        print("Running decoder")
        with torch.no_grad():
            if decoder_text_conditioned:
                print("Generating clip embeddings")
                _, text_encoding, text_mask = clip.embed_text(tokens)
                images = decoder.sample(
                    embeddings, text_encodings=text_encoding, text_mask=text_mask, cond_scale=decoder_cond_scale)
            else:
                print("Not generating clip embeddings")
                images = decoder.sample(
                    embeddings, text=None, cond_scale=decoder_cond_scale)
        hwc_images = images.cpu().permute(0, 2, 3, 1).numpy()  # [64, 64, 3]
        chw_images = images.cpu()  # [3, 64, 64]
        np.save('images_decoder.npy', hwc_images)

    if upsample:
        save_images('output/images', hwc_images)
        images = upscale_dir('output/images')
        save_images('output/enhanced_images', images)
        # enhance(chw_images, text_rep)
    end = time.time()
    print("运行时间:%.2f秒"%(end-start))


class QClickableImage(QWidget):
    image_id =''

    def __init__(self,width =0,height =0,pixmap =None,image_id = ''):
        QWidget.__init__(self)

        self.layout =QVBoxLayout(self)
        self.label1 = QLabel()
        self.label1.setObjectName('label1')
        self.lable2 =QLabel()
        self.lable2.setObjectName('label2')
        self.width =width
        self.height = height
        self.pixmap =pixmap

        if self.width and self.height:
            self.resize(self.width,self.height)
        if self.pixmap:
            pixmap = self.pixmap.scaled(QSize(self.width,self.height),Qt.KeepAspectRatio,Qt.SmoothTransformation)
            self.label1.setPixmap(pixmap)
            self.label1.setAlignment(Qt.AlignCenter)
            self.layout.addWidget(self.label1)
        if image_id:
            self.image_id =image_id
            self.lable2.setText(image_id)
            self.lable2.setAlignment(Qt.AlignCenter)
            ###让文字自适应大小
            self.lable2.adjustSize()
            self.layout.addWidget(self.lable2)
        self.setLayout(self.layout)

    clicked = pyqtSignal(object)
    rightClicked = pyqtSignal(object)

    def mouseressevent(self,ev):
        print('55555555555555555')
        if ev.button() == Qt.RightButton:
            print('dasdasd')
            #鼠标右击
            self.rightClicked.emit(self.image_id)
        else:
            self.clicked.emit(self.image_id)

    def imageId(self):
        return self.image_id


class MainWidget(QWidget):
    def __init__(self):
        super().__init__()
        # self.current_state = load_state(current_state)
        self.li_prompt = []
        self.init_ui()
        self.start_times = 0


    def init_ui(self):
        current_state = load_state()
        self.ui = uic.loadUi('./main.ui', self)
        # 设置ui初始化样式：
        self.ui.prompt_info.setText("   等待prompt的输入...")
        self.ui.pic_nums.setText("当前默认生成3张图片...\n（生成图片张数以及其他参数可在“更多参数修改”按钮处修改）")


        # 初始化self的各种属性：
        self.prompt_info = self.ui.prompt_info
        self.pic_nums = self.ui.pic_nums
        self.prompt_inputLine = self.ui.prompt_inputLine
        self.start_button = self.ui.start_button
        self.change_decoder_button = self.ui.change_decoder_btn
        self.set_paras = self.ui.set_paras

        # 添加函数：
        self.start_button.clicked.connect(self.main_start)
        self.prompt_inputLine.returnPressed.connect(self.main_start)
        self.ui.upload_file_btn.clicked.connect(self.upload_file)
        self.ui.download_img_btn.clicked.connect(self.download_img)


    def main_start(self):
        global current_state
        # 1. 获取输入框里的内容
        self.prompt = self.prompt_inputLine.text()
        if self.prompt == "" and self.li_prompt != []:
            self.ui.prompt_info.setText(f"  当前输入的prompt有：{self.li_prompt[0]}等{len(self.li_prompt)}句\n  正在为你生成图像")
            self.ui.prompt_info.repaint()

        elif self.prompt == "" and self.li_prompt == []:
            self.ui.prompt_info.setText(f"  请在输入prompt或上传prompt文件后开始！")
            self.ui.prompt_info.repaint()
            return 0
        else:
            self.ui.prompt_info.setText(f"  当前输入的prompt为：{self.prompt}\n  正在为你生成图像")
            self.ui.prompt_info.repaint()

        # 2. 检查是否有decoder：
        if current_state["decoder"] == None:
            self.ui.prompt_info.setText(f"  请选择一个decoder！")
        else:
            num = current_state['text_repeat']
            self.ui.pic_nums.setText(f"正在生成{num}张图片...\n（生成图片张数可在“更多参数修改”按钮处修改）")
            # 开始生成图像：
            self.start_generate_img()
        return 0

    def start_generate_img(self):
        self.gridLayout = QGridLayout(self.ui.scrollAreaImagesWidgetContents)
        if self.start_times >= 1:
            # 将内存中的模型全部清除：
            del self.decoder
            del self.diffusion_prior
            del self.clip
            self.li_prompt.clear()
        self.decoder = load_decoder(current_state["model_local_paths"]["decoder"],
                               current_state["model_local_paths"]["decoder_config"])
        self.diffusion_prior = load_prior(current_state["model_local_paths"]["prior"])
        # if self.restart == False:
        self.clip = load_clip()
        self.li_prompt.append(self.prompt)
        # self.restart = True
        on_start(self.clip, self.diffusion_prior, self.decoder, self.li_prompt)
        # 将生成的图像显示在widget上：
        self.show_img_on_widget()
        self.start_times+=1

    # todo: 1. 删除widget更新 2. 数据集预处理
    def show_img_on_widget(self):
        if self.start_times >= 1:
            # 删除之前的所有残留的widget
            for i in range(self.gridLayout.count()):
                self.gridLayout.itemAt(i).widget().deleteLater()
        # 将图片添加到图像显示widget中
        images_path = "results/swinir_real_sr_x4_large"
        dirs = os.listdir(images_path)
        # 计算每一列能放几张图片：每排容纳的图片数量 = 设定窗口 / (图片固定宽度+margin);
        # grid_layout_col = int(self.ui.scrollAreaWidgetContents.width() / (64+20))
        grid_layout_col = 3
        grid_layout_row = int(len(dirs) / grid_layout_col) + 1
        # 标记图片的id：
        image_id = 0
        row = 0
        col = 0
        for img in dirs:
            if img.split(".")[-1] == 'png':
                pixmap = QPixmap(images_path+"/"+img)
                clickable_image = QClickableImage(512, 512, pixmap, str(image_id))
                image_id += 1
                self.gridLayout.addWidget(clickable_image, row, col)
                col += 1
                if col >= grid_layout_col:
                    col = 0
                    row += 1
        self.ui.prompt_info.setText(f"  图像生成完毕！")
        self.ui.prompt_info.repaint()


    def upload_file(self):
        # self指向自身，"Open File"为文件名，"./"为当前路径，最后为文件类型筛选器
        fname, ftype = QFileDialog.getOpenFileName(self, "Open File", "./",
                                                  "Txt (*.txt)")  # 如果添加一个内容则需要加两个分号
        # 该方法返回一个tuple,里面有两个内容，第一个是路径， 第二个是要打开文件的类型，所以用两个变量去接受
        # 如果用户主动关闭文件对话框，则返回值为空
        if os.path.exists(fname):  # 判断路径非空
            # f = QFile(fname)  # 创建文件对象，不创建文件对象也不报错 也可以读文件和写文件
            # open()会自动返回一个文件对象
            f = open(fname, "r")  # 打开路径所对应的文件， "r"以只读的方式 也是默认的方式
            with f:
                data_li = f.readlines()
                self.li_prompt = []
                for line in data_li:
                    self.li_prompt.append(line.split("\n")[0])
                self.ui.prompt_info.setText(f"  当前文件中有{len(self.li_prompt)}条prompts\n  正在为你生成图像")
                self.ui.prompt_info.repaint()
            f.close()

    def download_img(self):
        import shutil
        # 先检查是否已经生成图像：
        if self.start_times < 1:
            QMessageBox.information(self, "warning", "请在生成图像后再下载图像！",
                                    QMessageBox.Yes)
            return
        # 选择文件夹：
        aim_folder = QFileDialog.getExistingDirectory(self, "选择文件夹", "")
        images_path = "results/swinir_real_sr_x4_large"
        dirs = os.listdir(images_path)
        for img in dirs:
            if img.split(".")[-1] == 'png':
                old_img_path = images_path+"/"+img
                aim_img_path = aim_folder+"/"+img
                shutil.copyfile(old_img_path, aim_img_path)
        self.ui.pic_nums.setText("保存图像成功！\n请在目标文件夹中查收~")
        self.ui.pic_nums.repaint()


class Change_decoder_dialog(QDialog):
    def __init__(self):
        global current_state
        super().__init__()
        # self.current_state = current_state
        self.index = 0
        if current_state["decoder"] == None:
            self.current_index = 0
        else:
            self.current_index = decoder_versions.index(current_state["decoder"]["name"])
        self.init_ui()


    def init_ui(self):
        self.ui = uic.loadUi('./decoder_switch.ui', self)
        self.model_selector = self.ui.comboBox
        self.model_selector.setCurrentIndex(self.current_index)
        self.model_selector.currentIndexChanged.connect(self.model_changed)
        self.explanation_label = self.ui.explaination_label
        des = "当前选择的模型为：" + str(self.model_selector.itemText(self.current_index) + '\n')
        des_temp = ''
        if self.current_index == 1:  # New_1B_(Aesthetic)
            des_temp = '其中Unet   dim  =  256\n训练所用的 dataset 为 Laion-Aesthetic'
        elif self.current_index == 2:  # New_1_5B_(Aesthetic)
            des_temp = '其中Unet   dim  =  320\n训练所用的 dataset 为 Laion-Aesthetic'
        elif self.current_index == 3:  # New_1_5B_(Laion2B)
            des_temp = '其中Unet   dim  =  320\n训练所用的 dataset 为 Laion-2B'
        else:
            pass
        des = des + des_temp
        self.explanation(des)
        # self.ui.ok_btn.clicked.connect(self.ok)

    def open(self):
        self.ui.show()

    def model_changed(self, index):
        des = "当前选择的模型为：" + str(self.model_selector.itemText(index) + '\n')
        des_temp = ''
        if index == 1:  # New_1B_(Aesthetic)
            des_temp = '其中Unet   dim  =  256\n训练所用的 dataset 为 Laion-Aesthetic'
        elif index == 2:  # New_1_5B_(Aesthetic)
            des_temp = '其中Unet   dim  =  320\n训练所用的 dataset 为 Laion-Aesthetic'
        elif index == 3:  # New_1_5B_(Laion2B)
            des_temp = '其中Unet   dim  =  320\n训练所用的 dataset 为 Laion-2B'
        else:
            pass
        des = des + des_temp
        self.explanation(des)
        self.index = index
        # current_state["decoder"] = decoder_versions[index]

    def explanation(self, des):
        output_des = '          explaination:\n\n' + des
        self.explanation_label.setText(output_des)
        self.explanation_label.repaint()

    def ok(self):
        global current_state
        # 如果还是选择None，就不让退出
        if(self.index < 1):
            self.explanation_label.setText("\n\n    请选择一个decoder！")
            self.explanation_label.repaint()
            return False
        else:
            # 如果已经选择了一个decoder：
            decoder_selected_info = decoder_versions[self.index]
            current_state["decoder"] = decoder_selected_info
            # print("#########################################################")
            # print(current_state)
            # print("#########################################################")
            self.explanation_label.setText("\n\n    正在加载...")
            self.explanation_label.repaint()
            time.sleep(1)
            # 加载模型
            current_state = check_models(current_state)
            save_state(current_state)
            # 重新安装依赖：
            dalle2_install_path = current_state['decoder']['dalle2_install_path']
            self.explanation_label.setText("\n\n    正在安装依赖文件...")
            self.explanation_label.repaint()
            os.system(f"pip install -q {dalle2_install_path}")
            # 推出界面
            self.explanation_label.setText("\n\n    加载完成！请使用吧~")
            self.explanation_label.repaint()
            time.sleep(0.5)
            flag = 0
            self.ui.close()
            return True  # 返回修改后的current_state



class set_parameters_dialog(QDialog):
    def __init__(self):
        global current_state
        super().__init__()
        # self.current_state = load_state(current_state)
        self.init_ui()

    def init_ui(self):
        global current_state
        self.ui = uic.loadUi('./set_parameters.ui', self)
        self.text_repeat_value = self.ui.text_repeat_value
        self.prior_cond_scale_value = self.ui.prior_cond_scale_value
        self.image_repeat_value = self.ui.image_repeat_value
        self.decoder_cond_scale_value = self.ui.decoder_cond_scale_value
        # 设定初始值：
        self.ui.horizontalSlider_text_repeat.setValue(current_state['text_repeat'])
        # 默认current_state['prior_conditioning'] = 1.0 但slider必须是整数，因此每次*10，取值[0/10, 30/10, 1/10]
        self.ui.horizontalSlider_prior_cond_scale.setValue(current_state['prior_conditioning']*10)
        self.ui.horizontalSlider_image_repeat.setValue(current_state['img_repeat'])
        # 默认current_state['decoder_conditioning'] = 1.7 但slider必须是整数，因此每次*10，取值[0/10, 30/10, 1/10]
        self.ui.horizontalSlider_decoder_cond_scale.setValue(current_state['decoder_conditioning']*10)
        self.slider_value_change()

        # 连接值改变函数：
        self.ui.horizontalSlider_text_repeat.valueChanged.connect(self.slider_value_change)
        self.ui.horizontalSlider_prior_cond_scale.valueChanged.connect(self.slider_value_change)
        self.ui.horizontalSlider_image_repeat.valueChanged.connect(self.slider_value_change)
        self.ui.horizontalSlider_decoder_cond_scale.valueChanged.connect(self.slider_value_change)
        self.ui.ok_btn.clicked.connect(self.ok)

    def open(self):
        self.ui.show()

    def slider_value_change(self):
        self.text_repeat_value.setText(f"{self.ui.horizontalSlider_text_repeat.value()}张")
        self.prior_cond_scale_value.setText(f"{self.ui.horizontalSlider_prior_cond_scale.value()/10}")
        self.image_repeat_value.setText(f"{self.ui.horizontalSlider_image_repeat.value()}张")
        self.decoder_cond_scale_value.setText(f"{self.ui.horizontalSlider_decoder_cond_scale.value()/10}")

    def ok(self):
        global current_state
        # 将state中的参数修改：
        current_state['text_repeat'] = self.ui.horizontalSlider_text_repeat.value()
        current_state['prior_conditioning'] = self.ui.horizontalSlider_prior_cond_scale.value()
        current_state['img_repeat'] = self.ui.horizontalSlider_image_repeat.value()
        current_state['decoder_conditioning'] = self.ui.horizontalSlider_decoder_cond_scale.value()
        save_state(current_state)

        self.ui.close()


class warning_window(QDialog):
    def __init__(self,msg):
        super().__init__()
        self.msg = msg

        self.init_ui()

    def init_ui(self):
        self.ui = uic.loadUi('./warning.ui', self)
        self.ui.msg_label.setText(self.msg)
        self.ui.ok_btn.clicked.connect(self.ok)

    def open(self):
        self.ui.show()

    def ok(self):
        self.ui.close()
        return 'finished'




if __name__ == '__main__':
    # 加载界面：
    # 导入资源文件，不然图标无法加载：
    if os.path.exists("./src_file.qrc") == False:
        os.system("pyrcc5 -o src_file.py src_file.qrc")
    import src_file

    app = QApplication(sys.argv)

    w = MainWidget()
    ch_decoder = Change_decoder_dialog()
    set_paras = set_parameters_dialog()
    w.change_decoder_button.clicked.connect(ch_decoder.open)
    w.change_decoder_button.clicked.connect(ch_decoder.open)
    w.set_paras.clicked.connect(set_paras.open)
    ch_decoder.ui.ok_btn.clicked.connect(ch_decoder.ok)
    # print(ch_decoder.ok())  # 默认为False

    # 展示窗口
    w.ui.show()

    # 程序进行循环等待状态
    app.exec_()


