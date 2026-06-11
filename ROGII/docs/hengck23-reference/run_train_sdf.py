import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import numpy as np
import os
from model import *
from dataset import *
from my_common import TRAIN_ID,VALID_ID

TRAIN_OFFSET = np.arange(-10,10,2)#[0] # [0]  #-4,-3,-2,0,1,2,3,4] #np.arange(0,128,4).tolist() #[0,8,16,24,32,48,64,72,96]
VALID_OFFSET = [0]


TRAIN_HSTEP  = [48]#[32,48,64]#np.arange(32,64)
VALID_HSTEP  = [48]

out_dir =\
    "/media/hp/8TB-HDD/work/2026/kaggle/wellbore/result/xxx-cnn01-sdf-06-more"
os.makedirs(f"{out_dir}/checkpoint", exist_ok=True)

##################################################################33
# cache data

def cache_data(cache_id, cache_offset, cache_step ):
    cache = []
    for i, sample_id in enumerate(cache_id):
        for offset in cache_offset:
            for step in cache_step:
                print('\r', i, sample_id, offset, end='', flush=True)
                try:
                    cache.append(
                        load_one_sample(sample_id, offset, step)
                    )
                except Exception as e:
                    print(f"\nFailed sample_id={sample_id}, offset={offset}: {e}")

    print()
    return cache

TRAIN_CACHE = cache_data(TRAIN_ID, TRAIN_OFFSET, TRAIN_HSTEP)
VALID_CACHE = cache_data(VALID_ID, VALID_OFFSET, VALID_HSTEP)
TRAIN_IDX   = np.arange(len(TRAIN_CACHE))
VALID_IDX   = np.arange(len(VALID_CACHE))
print(f"TRAIN_IDX: {len(TRAIN_IDX)}")
print(f"VALID_IDX: {len(VALID_IDX)}")
print()

def make_batch(batch_idx, mode="valid"):
    if mode == "valid":
        cache = VALID_CACHE
    if mode == "train":
        cache = TRAIN_CACHE

    #----------------------------------------------
    tensor_key = [
        "t_gr"   ,
        "h_gr"   ,
        "history",
        "label"  ,
        "target" ,
        "h_mask" ,
        "t_mask" ,
        "sdf" ,
    ]
    batch  = {
        "id" : []
    }
    for k in tensor_key:
        batch[k] = []


    for i in batch_idx:
        r = cache[i]
        batch["id" ].append(r["id"])

        for k in tensor_key:
            batch[k].append(r[k])

    for k in tensor_key:
        batch[k] = torch.from_numpy(np.stack( batch[k])).float()
    batch["history"] = batch["history"][:,None]
    batch["label"] = batch["label"][:,None]
    batch["sdf"] = batch["sdf"][:,None]
    return batch





###########################################################################
FIG, AX = plt.subplots(
    1, 2, figsize=(30, 8),
)
def show_result(batch, output, epoch=0, iteration=0, wait=0.01, mode="valid"):
    if mode=="valid":
        ax=AX[0]
    if mode=="train":
        ax=AX[1]

    #prob  = output["prob"].data.cpu().numpy()[:, 0]
    sdf  = output["sdf"].data.cpu().float().numpy()[:, 0]
    label = batch["label"].data.cpu().numpy()[:, 0]
    history = batch["history"].data.cpu().numpy()[:, 0]
    batch_size = len(sdf)

    h_mask = batch["h_mask"].data.cpu().numpy()[:,None,:, None]
    t_mask = batch["t_mask"].data.cpu().numpy()[:,:,None, None]


    image = []
    for b in range(batch_size):
        T,H = sdf[b].shape
        y = np.abs(sdf[b]).argmin(0)
        x = np.arange(H)

        zeros =  np.zeros_like(sdf[0])
        src = ((sdf[b]+3)/6 * 255).astype(np.uint8)
        m = cv2.applyColorMap(src, cv2.COLORMAP_TWILIGHT_SHIFTED)/255
        #m[y,x]=[1,1,1]
        m[np.abs(sdf[b])<0.1]=[1,1,1]
        m[label[b]>0.5]=[1,0,0]

        m = m*h_mask[b]*t_mask[b]
        #m = np.dstack([label[b], sdf[b], -sdf[b]])

        #m[:, :H_H] = np.maximum(m[:, :H_H], 0.5)
        m[:,H_H]*=0.5
        m = np.pad(m, ((0, 1), (0, 1), (0, 0)), constant_values=1)
        image.append(m)

    combined = np.vstack([
        np.hstack([image[i] for i in [0, 1, 2, 3]]),
        np.hstack([image[i] for i in [4, 5, 6, 7]]),
        np.hstack([image[i] for i in [8, 9, 10, 11]]),
        np.hstack([image[i] for i in [12, 13, 14, 15]]),
        np.hstack([image[i] for i in [16, 17, 18, 19]]),
    ])

    title = f"epoch {epoch}, iteration {iteration}: {mode}\n"
    i = 0
    for row in range(4):
        for col in range(4):
            sample_id,offset = batch["id"][i]
            i = i+1
            title += sample_id +", "
        title = title + "\n"
    title = title[:-1]

    #ax.set_title(str(id[:8]))
    ax.set_title(title)
    ax.imshow(combined)
    FIG.tight_layout()
    plt.waitforbuttonpress(wait)  # 0.1
    return


net = GeoSteerNet()
net = net.cuda()
net = net.train()
opt = torch.optim.Adam(net.parameters(), lr=0.001)

#checkpoint=f"{out_dir}/checkpoint/00040000.pth"
#print(net.load_state_dict(torch.load(checkpoint)))

#######################################################################################


num_epoch  = 10000
batch_size = 32
iter_accum = 1

iteration=0

for epoch in range(num_epoch):
    train_idx = TRAIN_IDX.copy()
    np.random.shuffle(train_idx)

    opt.zero_grad()
    net.train()

    train_loss = []
    for t in range(0, len(train_idx), batch_size):
        B = min(t+batch_size, len(train_idx)) -t
        if B!= batch_size: continue

        batch = make_batch(train_idx[t:t+B], mode="train")

        # if np.random.rand()<0.5:
        #     #flip augmentation?
        #     batch["h_gr"] = torch.flip(batch["h_gr"], dims=[1])
        #     batch["history"] = torch.flip(batch["history"], dims=[3])
        #     batch["label"] = torch.flip(batch["label"], dims=[3])
        #     batch["target"] = torch.flip(batch["target"], dims=[1])
        #     batch["h_mask"] = torch.flip(batch["h_mask"], dims=[1])


        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = net(batch)
            loss = output["sdf_loss"]

        (loss/iter_accum).backward()
        if iteration%iter_accum==0:
            opt.step()
            opt.zero_grad()

        iteration +=1
        batch_loss = loss.item()
        train_loss.append(batch_loss)
        print(f'\r {epoch:0.2f} {iteration:05d} batch_loss', batch_loss, end='', flush=True)

        if t==0:
            show_result(batch, output, epoch, iteration, wait=0.01, mode="train")
    print()
    train_loss = np.mean(train_loss)
    print(f'{epoch:0.2f} {iteration:05d} train_loss', train_loss)

    if epoch%10==0:
        pass
        torch.save(net.state_dict(), f"{out_dir}/checkpoint/{iteration:08d}.pth")

    #=======================================

    if epoch%10==0:
        #show_result(batch, output, epoch, iteration, wait=0.01, mode="train")

        valid_loss = []
        valid_rmse = []
        valid_idx = VALID_IDX.copy()

        net.eval()
        for t in range(0, len(valid_idx), batch_size):
            B = min(t+batch_size, len(valid_idx))-t
            batch = make_batch(valid_idx[t:t+B], mode="valid")

            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    output = net(batch)
                    loss = output["sdf_loss"]

            batch_loss = loss.item()
            valid_loss.append(batch_loss)

            if t == 0:
                show_result(batch, output, epoch,  iteration, wait =0.1, mode="valid")

        valid_loss = np.mean(valid_loss)
        print(f'** {epoch:0.2f} {iteration:05d} valid_loss', valid_loss)


