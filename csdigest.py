#%%
# imports
import copy
import itertools as itt
import os
import re
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import dateparser
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
import yaml
from bs4 import BeautifulSoup
from emoji import emojize
from keras import models
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_extraction.text import CountVectorizer
from slack_sdk import WebClient


#%%
# csdigest class
class CSDigest:
    def __init__(self, config_path) -> None:
        print("loading data")
        # handling files
        with open(config_path) as cfg:
            config = yaml.load(cfg, Loader=yaml.FullLoader)
        with open(config["token_path"]) as tk:
            self.token = tk.readline()
        with open(config["template_path"]) as tp:
            self.temp = BeautifulSoup(tp, "html.parser")
        self.cache_im = os.path.join("cache", "images")
        os.makedirs(self.cache_im, exist_ok=True)
        self.ts_old = dateparser.parse(config["time_span"]).timestamp()
        # initialize objects
        self.foodnet = models.load_model("./foodnet/model")
        self.datagen = ImageDataGenerator(rescale=1.0 / 255)
        self.client = WebClient(token=self.token)
        self.channels = pd.DataFrame(
            self.client.conversations_list(limit=1000)["channels"]
        ).set_index("name")
        self.users = pd.DataFrame(
            self.client.users_list(limit=1000)["members"]
        ).set_index("id")
        self.users["display_name"] = self.users["profile"].apply(
            self.extract_profile, key="display_name"
        )
        print("fetching messages")
        # get messages
        ms_general = self.get_msg("general", same_user=False)
        ms_home = self.get_msg("homesanity", ts_thres=0)
        ms_quote = self.get_msg("quotablequotes", ts_thres=120, same_user=False)
        print("building newsletter")
        # handle carousel
        if len(ms_general) > 0:
            ms_general["class"] = ms_general.apply(self.classify_msg, axis="columns")
            ms_tada = ms_general[ms_general["class"] == "tada"]
            if len(ms_tada) > 0:
                ms_tada["permalink"] = ms_tada.apply(self.get_permalink, axis="columns")
                self.build_carousel(ms_tada)
        # handle food
        ms_files = pd.concat([ms_general, ms_home])
        if len(ms_files) > 0:
            ms_files["file_path"] = ms_files.apply(self.download_images, axis="columns")
            ms_files = ms_files[ms_files["file_path"].astype(bool)]
            ms_files["food_prob"] = ms_files["file_path"].apply(self.classify_food)
            ms_files["food_path"] = ms_files.apply(self.filter_food, axis="columns")
            ms_food = ms_files[ms_files["food_path"].notnull()]
            ms_food["permalink"] = ms_food.apply(self.get_permalink, axis="columns")
            ms_food["aspect"] = ms_food["food_path"].apply(self.get_img_aspect)
            ms_food = ms_food.sort_values("aspect", ascending=True)
            self.build_portfolio(ms_food)
        # handle quotes
        if len(ms_quote) > 0:
            ms_quote = ms_quote[~ms_quote["files"].astype(bool)]
            ms_quote["permalink"] = ms_quote.apply(self.get_permalink, axis="columns")
            self.build_quotes(ms_quote)

    def get_msg(self, channel, ts_thres=5, same_user=True):
        ms = pd.DataFrame(
            self.client.conversations_history(
                channel=self.channels.loc[channel]["id"],
                oldest=self.ts_old,
                limit=1000,
            )["messages"]
        )
        if len(ms) > 0:
            ms = ms[ms["subtype"].isnull()]
            if len(ms) > 0:
                ms = (
                    self.cluster_msg(ms, ts_thres=ts_thres, same_user=same_user)
                    .groupby("component")
                    .apply(self.merge_msg)
                    .reset_index()
                    .apply(self.translate_msg_user, axis="columns")
                )
                ms["text"] = ms["text"].apply(
                    emojize, use_aliases=True, variant="emoji_type"
                )
                ms["channel"] = self.channels.loc[channel]["id"]
        return ms

    def classify_msg(self, msg_df):
        if msg_df["reactions"]:
            tada = list(filter(lambda r: r["name"] == "tada", msg_df["reactions"]))
            if tada and tada[0]["count"] > 5:
                return "tada"

    def cluster_msg(self, msg_df, ts_thres, same_user):
        ts_dist = pdist(msg_df["ts"].values.astype(float).reshape((-1, 1)))
        txt_dist = pdist(CountVectorizer().fit_transform(msg_df["text"]).toarray())
        adj = squareform(ts_dist) < ts_thres * 60
        if same_user:
            user_dist = pdist(
                msg_df["user"].values.reshape((-1, 1)),
                metric=lambda u, v: 0 if u == v else 1,
            )
            adj = adj * (squareform(user_dist) < 1)
        n_comp, lab = connected_components(adj, directed=False)
        msg_df["component"] = lab
        return msg_df

    def merge_msg(self, msg_df, multiple_users="first"):
        msg_df = msg_df.sort_values("ts")
        if multiple_users == "forbid":
            user = msg_df["user"].unique()
            assert len(user) == 1
            user = user.item()
        elif multiple_users == "first":
            user = msg_df.iloc[0]["user"]
            msg_df = msg_df[msg_df["user"] == user]
        else:
            raise ValueError("multiple_users=={} not understood".format(multiple_users))
        try:
            reactions = msg_df["reactions"].dropna().values
            reactions = sum(reactions, [])
        except KeyError:
            reactions = []
        try:
            files = msg_df["files"].dropna().values
            files = sum(files, [])
        except KeyError:
            files = []
        try:
            attch = msg_df["attachments"].dropna().values
            attch = sum(attch, [])
        except KeyError:
            attch = []
        return pd.Series(
            {
                "user": user,
                "text": "\n".join(msg_df["text"].values),
                "ts": msg_df.iloc[0].loc["ts"],
                "reactions": reactions,
                "files": files,
                "attachments": attch,
            }
        )

    def translate_msg_user(self, msg_row, substitute=["display_name", "name"]):
        try:
            msg_row["user"] = self.translate_user(msg_row["user"], substitute)
            msg_row["text"] = re.sub(
                r"\<\@(.*?)\>",
                lambda u: self.translate_user(u.group(1), substitute),
                msg_row["text"],
            )
        except TypeError:
            pass
        return msg_row

    def translate_user(self, uid, substitute):
        for sub in substitute:
            if sub == "real_name":
                prefix = ""
            else:
                prefix = "@"
            user = self.users.loc[uid, sub]
            if type(user) == str and bool(user):
                return prefix + user

    def extract_profile(self, prof, key):
        try:
            return prof[key]
        except KeyError:
            return np.nan

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
        for fdict in msg_row["attachments"]:
            try:
                url = fdict["image_url"]
            except KeyError:
                continue
            fpath = os.path.join(self.cache_im, url.split("/")[-1].split("?")[0])
            resp = requests.get(url)
            open(fpath, "wb").write(resp.content)
            fpaths.append(fpath)
        return fpaths

    def build_carousel(self, msg_df):
        indicator = self.temp.find("ol", {"id": "carousel-inds"})
        ind_temp = indicator.find("li", {"id": "carousel-ind-template"}).extract()
        sld_wrapper = self.temp.find("div", {"id": "carousel-slides"})
        tada_temp = self.temp.find("div", {"id": "carousel-slide-template"}).extract()
        for (imsg, msg), icss in zip(
            msg_df.reset_index(drop=True).iterrows(), itt.cycle(np.arange(3) + 1)
        ):
            cur_ind = copy.copy(ind_temp)
            cur_ind["data-slide-to"] = str(imsg)
            cur_tada = copy.copy(tada_temp)
            cur_tada.find("h3", {"id": "carousel-slide-message"}).string = (
                msg["text"] if len(msg["text"]) <= 320 else msg["text"][:320] + "..."
            )
            cur_tada.find(True, {"id": "carousel-slide-author"}).string = " ".join(
                [
                    msg["user"],
                    datetime.fromtimestamp(float(msg["ts"])).strftime("%b %d"),
                ]
            )
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
        with open("csdigest.html", "w", encoding="utf-8") as outf:
            outf.write(str(self.temp))

    def classify_food(self, img_path):
        imgs = [img_to_array(load_img(imp).resize((512, 512))) for imp in img_path]
        predict = self.foodnet.predict(self.datagen.flow(np.stack(imgs)))
        return np.atleast_1d(predict.squeeze()).tolist()

    def filter_food(self, msg_row, thres=0.1):
        minval, minidx = np.min(msg_row["food_prob"]), np.argmin(msg_row["food_prob"])
        if minval < thres:
            return msg_row["file_path"][minidx]
        else:
            return np.nan

    def get_img_aspect(self, path):
        img = load_img(path)
        return img.size[0] / img.size[1]

    def build_portfolio(self, msg_df):
        porto = self.temp.find("div", {"id": "portfolio-container"})
        port_temp = self.temp.find("div", {"id": "portfolio-template"}).extract()
        del port_temp["id"]
        for imsg, msg in msg_df.iterrows():
            cur_temp = copy.copy(port_temp)
            cur_temp.img["src"] = msg["food_path"]
            cur_temp.find("a", {"id": "port-zoom-link"})["href"] = msg["food_path"]
            cur_temp.find("a", {"id": "port-msg-link"})["href"] = msg["permalink"]
            txt = msg["text"]
            if len(txt) > 150:
                txt = txt[:150] + "..."
            cur_temp.find(True, {"id": "port-item-text"}).string = txt
            # cur_temp.find(True, {"id": "port-item-text"}).string = str(
            #     np.min(msg["food_prob"])
            # )
            porto.append(cur_temp)

    def build_quotes(self, msg_df):
        quotes = self.temp.find("div", {"id": "quote-block"})
        quote_temp = quotes.find("div", {"id": "quote-template"}).extract()
        indicators = self.temp.find("ol", {"id": "quote-indicator-wrap"})
        ind_temp = indicators.find("li", {"id": "quote-indicator"}).extract()
        for imsg, msg in msg_df.reset_index(drop=True).iterrows():
            cur_quote = copy.copy(quote_temp)
            cur_quote.find("a", {"id": "quote-link"})["href"] = msg["permalink"]
            cur_quote.find("p", {"id": "quote-content"}).string = (
                msg["text"] if len(msg["text"]) <= 400 else msg["text"][:400] + "..."
            )
            cur_quote.find("span", {"id": "quote-name"}).string = msg["user"]
            cur_ind = copy.copy(ind_temp)
            cur_ind["data-slide-to"] = str(imsg)
            if imsg > 0:
                cur_quote["class"] = list(
                    filter(lambda c: c != "active", cur_quote["class"])
                )
                cur_ind["class"] = list(
                    filter(lambda c: c != "active", cur_ind["class"])
                )
            quotes.append(cur_quote)
            indicators.append(cur_ind)


#%%
# main
if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    digest = CSDigest("config.yml")
    digest.write_html()
