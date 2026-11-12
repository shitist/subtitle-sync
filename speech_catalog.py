"""Lightweight GUI metadata; model loading stays in the speech worker."""

LANGUAGE_CODES = tuple("af am ar as az ba be bg bn bo br bs ca cs cy da de el en es et eu fa fi fo fr gl gu ha haw he hi hr ht hu hy id is it ja jw ka kk km kn ko la lb ln lo lt lv mg mi mk ml mn mr ms mt my ne nl nn no oc pa pl ps pt ro ru sa sd si sk sl sn so sq sr su sv sw ta te tg th tk tl tr tt uk ur uz vi yi yo zh yue".split())
LANGUAGE_NAMES = dict(zip(
    "zh ja en ko yue fr de es it pt ru ar hi vi th id ms nl uk pl tr sv da fi no nn".split(),
    "中文 日语 英语 韩语 粤语 法语 德语 西班牙语 意大利语 葡萄牙语 俄语 阿拉伯语 印地语 越南语 泰语 印度尼西亚语 马来语 荷兰语 乌克兰语 波兰语 土耳其语 瑞典语 丹麦语 芬兰语 挪威语 挪威语（尼诺斯克）".split(),
))
LANGUAGE_NAMES.update({
    "af": "南非荷兰语", "am": "阿姆哈拉语", "as": "阿萨姆语", "az": "阿塞拜疆语",
    "ba": "巴什基尔语", "be": "白俄罗斯语", "bg": "保加利亚语", "bn": "孟加拉语",
    "bo": "藏语", "br": "布列塔尼语", "bs": "波斯尼亚语", "ca": "加泰罗尼亚语",
    "cs": "捷克语", "cy": "威尔士语", "el": "希腊语", "et": "爱沙尼亚语",
    "eu": "巴斯克语", "fa": "波斯语", "fo": "法罗语", "gl": "加利西亚语",
    "gu": "古吉拉特语", "ha": "豪萨语", "haw": "夏威夷语", "he": "希伯来语",
    "hr": "克罗地亚语", "ht": "海地克里奥尔语", "hu": "匈牙利语", "hy": "亚美尼亚语",
    "is": "冰岛语", "jw": "爪哇语", "ka": "格鲁吉亚语", "kk": "哈萨克语",
    "km": "高棉语", "kn": "卡纳达语", "la": "拉丁语", "lb": "卢森堡语",
    "ln": "林加拉语", "lo": "老挝语", "lt": "立陶宛语", "lv": "拉脱维亚语",
    "mg": "马尔加什语", "mi": "毛利语", "mk": "马其顿语", "ml": "马拉雅拉姆语",
    "mn": "蒙古语", "mr": "马拉地语", "mt": "马耳他语", "my": "缅甸语",
    "ne": "尼泊尔语", "oc": "奥克语", "pa": "旁遮普语", "ps": "普什图语",
    "ro": "罗马尼亚语", "sa": "梵语", "sd": "信德语", "si": "僧伽罗语",
    "sk": "斯洛伐克语", "sl": "斯洛文尼亚语", "sn": "绍纳语", "so": "索马里语",
    "sq": "阿尔巴尼亚语", "sr": "塞尔维亚语", "su": "巽他语", "sw": "斯瓦希里语",
    "ta": "泰米尔语", "te": "泰卢固语", "tg": "塔吉克语", "tk": "土库曼语",
    "tl": "他加禄语", "tt": "鞑靼语", "ur": "乌尔都语", "uz": "乌兹别克语",
    "yi": "意第绪语", "yo": "约鲁巴语",
})

# WhisperX 3.8.6 default models. TorchAudio bundle names are mapped to the
# equivalent Hugging Face checkpoints, using one download/cache mechanism.
ALIGN_MODELS = {
    "en": "facebook/wav2vec2-base-960h",
    **{code: f"facebook/wav2vec2-base-10k-voxpopuli-{code}" for code in ("fr", "de", "es", "it")},
    **{code: f"jonatasgrosman/wav2vec2-large-xlsr-53-{name}" for code, name in {
        "ja": "japanese", "zh": "chinese-zh-cn", "nl": "dutch", "pt": "portuguese",
        "ar": "arabic", "ru": "russian", "pl": "polish", "hu": "hungarian",
        "fi": "finnish", "fa": "persian", "el": "greek",
    }.items()},
    "uk": "Yehor/wav2vec2-xls-r-300m-uk-with-small-lm",
    "cs": "comodoro/wav2vec2-xls-r-300m-cs-250",
    "tr": "mpoyraz/wav2vec2-xls-r-300m-cv7-turkish",
    "da": "saattrupdan/wav2vec2-xls-r-300m-ftspeech",
    "he": "imvladikon/wav2vec2-xls-r-300m-hebrew",
    "vi": "nguyenvulebinh/wav2vec2-base-vi-vlsp2020",
    "ko": "kresnik/wav2vec2-large-xlsr-korean",
    "ur": "kingabzpro/wav2vec2-large-xls-r-300m-Urdu",
    "te": "anuragshas/wav2vec2-large-xlsr-53-telugu",
    "hi": "theainerd/Wav2Vec2-large-xlsr-hindi",
    "ca": "softcatala/wav2vec2-large-xlsr-catala",
    "ml": "gvs/wav2vec2-large-xlsr-malayalam",
    "no": "NbAiLab/nb-wav2vec2-1b-bokmaal-v2",
    "nn": "NbAiLab/nb-wav2vec2-1b-nynorsk",
    "sk": "comodoro/wav2vec2-xls-r-300m-sk-cv8",
    "sl": "anton-l/wav2vec2-large-xlsr-53-slovenian",
    "hr": "classla/wav2vec2-xls-r-parlaspeech-hr",
    "ro": "gigant/romanian-wav2vec2",
    "eu": "stefan-it/wav2vec2-large-xlsr-53-basque",
    "gl": "ifrz/wav2vec2-large-xlsr-galician",
    "ka": "xsway/wav2vec2-large-xlsr-georgian",
    "lv": "jimregan/wav2vec2-large-xlsr-latvian-cv",
    "tl": "Khalsuu/filipino-wav2vec2-l-xls-r-300m-official",
    "sv": "KBLab/wav2vec2-large-voxrex-swedish",
    "id": "cahya/wav2vec2-large-xlsr-indonesian",
}


def language_label(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


