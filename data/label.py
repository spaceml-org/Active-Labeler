from IPython.display import display, Image
import global_constants as GConst
from pigeon import annotate
import shutil
import os
from tqdm import tqdm


class Labeler:
    def __init__(self, config) -> None:
        self.config = config

    def label(self, paths, is_eval=False, fetch_paths=False):
        # annotations = annotate(
        #     paths,
        #     options=self.config["data"]["classes"],
        #     display_fn=lambda filename: display(Image(filename)),
        # )
        annotations = []
        for path in paths:
            path_class = path.split("/")[-2]
            if path_class == self.config['experiment']['positive_class']:
                annotations.append([path,'positive'])
            else:
                annotations.append([path,'negative'])
        if fetch_paths:
            return annotations
        else:
            save_path = "Dataset/Val" if is_eval else "Dataset/Labeled"
            for path, label in annotations:
                shutil.copy(path, os.path.join(save_path, label))
            return annotations

    def annotate_paths(self, df, is_eval=False):
        df_lis = df.to_dict("records")
        save_path = "Dataset/Val" if is_eval else "Dataset/Labeled"
        for row in tqdm(df_lis):
            shutil.copy(row[GConst.IMAGE_PATH_COL], os.path.join(save_path, row[GConst.LABEL_COL], row[GConst.IMAGE_PATH_COL].split("/")[-1]))
