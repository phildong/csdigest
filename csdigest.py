#%%
# imports
import copy
import itertools as itt
import os
import re

import numpy as np
import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_extraction.text import CountVectorizer
from slack_sdk import WebClient


#%%
# csdigest class
class CSDigest:
    def __init__(self, config_path) -> None:
        # handling files
        with open(config_path) as cfg:
            config = yaml.load(cfg, Loader=yaml.FullLoader)
        with open(config["token_path"]) as tk:
            self.token = tk.readline()
        with open(config["template_path"]) as tp:
            self.temp = BeautifulSoup(tp, "html.parser")
        self.cache_im = os.path.join("cache", "images")
        os.makedirs(self.cache_im, exist_ok=True)
        self.client = WebClient(token=self.token)
        self.channels = pd.DataFrame(
            self.client.conversations_list()["channels"]
        ).set_index("name")
        self.users = pd.DataFrame(self.client.users_list()["members"]).set_index("id")
        ms_general = pd.DataFrame(
            self.client.conversations_history(
                channel=self.channels.loc["general"]["id"]
            )["messages"]
        )
        self.ms_general = ms_general
        ms_general = (
            self.cluster_msg(ms_general)
            .groupby("component")
            .apply(self.merge_msg)
            .reset_index()
            .apply(self.translate_user, axis="columns")
        )
        ms_general["class"] = ms_general.apply(self.classify_msg, axis="columns")
        ms_general["channel"] = self.channels.loc["general"]["id"]
        ms_tada = ms_general[ms_general["class"] == "tada"]
        ms_tada["permalink"] = ms_tada.apply(self.get_permalink, axis="columns")
        ms_files = self.ms_general[self.ms_general["files"].notnull()]
        ms_files["file_path"] = ms_files.apply(self.download_images, axis="columns")
        self.ms_files = ms_files
        self.build_carousel(ms_tada)

    def classify_msg(self, msg_df):
        if msg_df["reactions"]:
            tada = list(filter(lambda r: r["name"] == "tada", msg_df["reactions"]))
            if tada and tada[0]["count"] > 5:
                return "tada"

    def cluster_msg(self, msg_df):
        ts_dist = pdist(msg_df["ts"].values.astype(float).reshape((-1, 1)))
        txt_dist = pdist(CountVectorizer().fit_transform(msg_df["text"]).toarray())
        user_dist = pdist(
            msg_df["user"].values.reshape((-1, 1)),
            metric=lambda u, v: 0 if u == v else 1,
        )
        adj = (squareform(ts_dist) < 5 * 60) * (squareform(user_dist) < 1)
        n_comp, lab = connected_components(adj, directed=False)
        msg_df["component"] = lab
        return msg_df

    def merge_msg(self, msg_df):
        msg_df = msg_df.sort_values("ts")
        user = msg_df["user"].unique()
        assert len(user) == 1
        reactions = msg_df["reactions"].dropna().values
        reactions = sum(reactions, [])
        return pd.Series(
            {
                "user": user.item(),
                "text": "\n".join(msg_df["text"].values),
                "ts": msg_df.iloc[0].loc["ts"],
                "reactions": reactions,
            }
        )

    def translate_user(self, msg_row):
        msg_row["user"] = self.users.loc[msg_row["user"], "real_name"]
        msg_row["text"] = re.sub(
            r"\<\@(.*?)\>",
            lambda u: self.users.loc[u.group(1), "real_name"],
            msg_row["text"],
        )
        return msg_row

    def get_permalink(self, msg_row):
        return self.client.chat_getPermalink(
            channel=msg_row["channel"], message_ts=str(msg_row["ts"])
        )["permalink"]

    def download_images(self, msg_row):
        fpaths = []
        for fdict in msg_row["files"]:
            if fdict["mimetype"].startswith("image"):
                fpath = os.path.join(
                    self.cache_im,
                    ".".join([fdict["id"], fdict["filetype"]]),
                )
                resp = requests.get(
                    fdict["url_private_download"],
                    headers={"Authorization": "Bearer {}".format(self.token)},
                )
                open(fpath, "wb").write(resp.content)
                fpaths.append(fpath)
        return fpaths

    def build_carousel(self, msg_df):
        indicator = self.temp.find("ol", class_="carousel-indicators")
        ind_temp = indicator.find("li").extract()
        sld_wrapper = self.temp.find("div", class_="carousel-inner")
        tada_temp = self.temp.find("div", class_="carousel-tada-1").extract()
        for (imsg, msg), icss in zip(
            msg_df.reset_index(drop=True).iterrows(), itt.cycle(np.arange(3) + 1)
        ):
            cur_ind = copy.copy(ind_temp)
            cur_ind["data-slide-to"] = str(imsg)
            cur_tada = copy.copy(tada_temp)
            cur_tada.find("h3").string = msg["text"]
            cur_tada.find(True, string="tada_author").string = msg["user"]
            cur_tada.find("a")["href"] = msg["permalink"]
            if re.search("birthday", msg["text"].lower()):
                cur_tada["class"] = [
                    "carousel-birthday" if c == "carousel-tada-1" else c
                    for c in cur_tada["class"]
                ]
            else:
                cur_tada["class"] = [
                    "carousel-tada-{}".format(icss) if c == "carousel-tada-1" else c
                    for c in cur_tada["class"]
                ]
            if not imsg == 0:
                del cur_ind["class"]
                cur_tada["class"] = list(
                    filter(lambda c: c != "active", cur_tada["class"])
                )
            indicator.append(cur_ind)
            sld_wrapper.append(cur_tada)

    def write_html(self):
        with open("csdigest.html", "w") as outf:
            outf.write(str(self.temp))


#%%
# main
if __name__ == "__main__":
    digest = CSDigest("config.yml")
    digest.write_html()
