import os
import joblib
import logging
import glob
from natsort import natsorted
import re
import random
from sklearn.model_selection import train_test_split

'''
splitWSIDataset:
    imgs_dirにある予測対象のクラスのWSIのみから，
    Cross Validation用にデータセットを分割する

ディレクトリの構造 (例): /{imgs_dir}/{sub_cl}/{wsi_name}/0_0000000.png
'''


class splitWSIDataset(object):
    def __init__(self, imgs_dir, classes=[0, 1, 2, 3], val_ratio=0.2, random_seed=0):
        self.imgs_dir = imgs_dir
        self.classes = classes
        self.val_ratio = val_ratio
        self.sub_classes = self.get_sub_classes()
        self.random_seed = random_seed
        self.sets_num = 5

        random.seed(self.random_seed)

        # WSIごとにtrain, valid, test分割
        self.wsi_list = []
        for i in range(len(self.sub_classes)):
            sub_cl = self.sub_classes[i]
            self.wsi_list.extend([p[:-4] for p in os.listdir(self.imgs_dir + f"{sub_cl}/")])
        self.wsi_list = list(set(self.wsi_list))
        # os.listdirによる実行時における要素の順不同対策のため
        self.wsi_list = natsorted(self.wsi_list)

        # WSIのリストを5-setsに分割
        random.shuffle(self.wsi_list)
        self.sets_list = self.split_sets_list(self.wsi_list)

    def __len__(self):
        return len(self.wsi_list)

    def get_sub_classes(self):
        # classesからsub-classを取得
        sub_cl_list = []
        for idx in range(len(self.classes)):
            cl = self.classes[idx]
            if isinstance(cl, list):
                for sub_cl in cl:
                    sub_cl_list.append(sub_cl)
            else:
                sub_cl_list.append(cl)
        return sub_cl_list

    def split_sets_list(self, wsi_list, sets_num=5):
        wsi_num = len(wsi_list)
        q, mod = divmod(wsi_num, sets_num)
        logging.info(f"wsi_num: {wsi_num}, q: {q}, mod: {mod}")

        idx_list = []
        wsi_sets = []
        idx = 0

        for cv in range(sets_num):
            if cv < mod:
                end_idx = idx + q
            else:
                end_idx = (idx + q) - 1
            idx_list.append([idx, end_idx])

            wsi_sets.append(wsi_list[idx:end_idx + 1])
            idx = end_idx + 1

        print(f"idx_list: {idx_list}")

        return wsi_sets

    def get_sets_list(self):
        return self.sets_list

    def get_files(self, wsis):
        re_pattern = re.compile('|'.join([f"/{i}/" for i in self.sub_classes]))

        files_list = []
        for wsi in wsis:
            files_list.extend(
                [
                    p for p in glob.glob(self.imgs_dir + f"*/{wsi}_*/*.png", recursive=True)
                    if bool(re_pattern.search(p))
                ]
            )
        return files_list

    def get_cv_wsis(self, sets_list, cv_num):
        test_wsis = sets_list[cv_num]
        trvl_wsis = []
        for i in range(self.sets_num):
            if i == cv_num:
                continue
            else:
                trvl_wsis += sets_list[i]

        random.shuffle(trvl_wsis)
        train_wsis, valid_wsis = train_test_split(
            trvl_wsis, test_size=self.val_ratio, random_state=self.random_seed)
        return natsorted(train_wsis), natsorted(valid_wsis), natsorted(test_wsis)


def save_dataset(imgs_dir, output_dir):
    cv = 5
    dataset = splitWSIDataset(imgs_dir, classes=[0, 1, 2], val_ratio=0.2, random_seed=0)
    sets_list = dataset.get_sets_list()

    for cv_num in range(cv):
        logging.info(f"===== CV{cv_num} =====")
        train_wsis, valid_wsis, test_wsis = dataset.get_cv_wsis(sets_list, cv_num=cv_num)

        train_files = dataset.get_files(train_wsis)
        valid_files = dataset.get_files(valid_wsis)
        test_files = dataset.get_files(test_wsis)

        logging.info(f"[wsi]  train: {len(train_wsis)}, valid: {len(valid_wsis)}, test: {len(test_wsis)}")
        logging.info(f"[data] train: {len(train_files)}, valid: {len(valid_files)}, test: {len(test_files)}")

        # WSI割当のリストを保存
        joblib.dump(train_wsis, output_dir + f"cv{cv_num}_train_wsi.jb", compress=3)
        joblib.dump(valid_wsis, output_dir + f"cv{cv_num}_valid_wsi.jb", compress=3)
        joblib.dump(test_wsis, output_dir + f"cv{cv_num}_test_wsi.jb", compress=3)

        # 各データのリスト(path)を保存
        joblib.dump(train_files, output_dir + f"cv{cv_num}_train.jb", compress=3)
        joblib.dump(valid_files, output_dir + f"cv{cv_num}_valid.jb", compress=3)
        joblib.dump(test_files, output_dir + f"cv{cv_num}_test.jb", compress=3)

        with open(output_dir + f"cv{cv_num}_dataset.txt", mode='w') as f:
            f.write(
                "== [wsi] ==\n"
                + f"train: {len(train_wsis)}, valid: {len(valid_wsis)}, test: {len(test_wsis)}"
                + "\n==============\n")
            f.write(
                "== [patch] ==\n"
                + f"train: {len(train_files)}, valid: {len(valid_files)}, test: {len(test_files)}"
                + "\n==============\n")

            f.write("== train (wsi) ==\n")
            for i in range(len(train_wsis)):
                f.write(f"{train_wsis[i]}\n")

            f.write("\n== valid (wsi) ==\n")
            for i in range(len(valid_wsis)):
                f.write(f"{valid_wsis[i]}\n")

            f.write("\n== test (wsi) ==\n")
            for i in range(len(test_wsis)):
                f.write(f"{test_wsis[i]}\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    imgs_dir = "/mnt/ssdsub1/DFBConv_strage/mnt2/MF0012/"
    output_dir = "/mnt/ssdsub1/DFBConv_strage/results/dataset/"

    save_dataset(imgs_dir, output_dir)
