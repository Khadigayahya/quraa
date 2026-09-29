"""Map everyayah folder labels to real people.

everyayah publishes the same reciter several times (different bitrates, murattal /
mujawwad / teaching editions, Warsh). The baseline treated each folder as a separate
class, so e.g. "Husary_Mujawwad -> Husary_128kbps_Mujawwad" counted as an error.
Here every folder keeps its own reference voiceprint (styles really sound different),
but they all point to one person, and the app reports the person.
"""
import re

# person id -> (Arabic name, English name)
PEOPLE = {
    "abdulbasit": ("عبد الباسط عبد الصمد", "Abdul Basit Abdul Samad"),
    "juhaynee": ("عبد الله عواد الجهني", "Abdullah Awad Al-Juhany"),
    "basfar": ("عبد الله بصفر", "Abdullah Basfar"),
    "matroud": ("عبد الله المطرود", "Abdullah Al-Matroud"),
    "sudais": ("عبد الرحمن السديس", "Abdul Rahman Al-Sudais"),
    "shatri": ("أبو بكر الشاطري", "Abu Bakr Al-Shatri"),
    "neana": ("أحمد نعينع", "Ahmed Neana"),
    "ajamy": ("أحمد بن علي العجمي", "Ahmed ibn Ali Al-Ajamy"),
    "alaqimy": ("أكرم العلاقمي", "Akram Al-Alaqimy"),
    "alafasy": ("مشاري راشد العفاسي", "Mishary Rashid Alafasy"),
    "suesy": ("علي حجاج السويسي", "Ali Hajjaj Al-Suesy"),
    "ghamadi": ("سعد الغامدي", "Saad Al-Ghamdi"),
    "rifai": ("هاني الرفاعي", "Hani Ar-Rifai"),
    "hudhaify": ("علي الحذيفي", "Ali Al-Hudhaify"),
    "husary": ("محمود خليل الحصري", "Mahmoud Khalil Al-Husary"),
    "mansoori": ("كريم منصوري", "Karim Mansoori"),
    "qahtani": ("خالد القحطاني", "Khalid Al-Qahtani"),
    "maher": ("ماهر المعيقلي", "Maher Al-Muaiqly"),
    "minshawy": ("محمد صديق المنشاوي", "Mohamed Siddiq Al-Minshawi"),
    "tablawy": ("محمد محمود الطبلاوي", "Mohamed Mahmoud Al-Tablawy"),
    "abdulkareem": ("محمد عبد الكريم", "Muhammad Abdul Kareem"),
    "ayyoub": ("محمد أيوب", "Muhammad Ayyoub"),
    "jibreel": ("محمد جبريل", "Muhammad Jibreel"),
    "muhsin_qasim": ("محسن القاسم", "Muhsin Al-Qasim"),
    "mustafa_ismail": ("مصطفى إسماعيل", "Mustafa Ismail"),
    "qatami": ("ناصر القطامي", "Nasser Al-Qatami"),
    "parhizgar": ("شهريار پرهيزگار", "Shahriar Parhizgar"),
    "sahl_yassin": ("سهل ياسين", "Sahl Yassin"),
    "bukhatir": ("صلاح بو خاطر", "Salah Bukhatir"),
    "budair": ("صلاح البدير", "Salah Al-Budair"),
    "shuraym": ("سعود الشريم", "Saud Al-Shuraim"),
    "yaser_salamah": ("ياسر سلامة", "Yaser Salamah"),
    "dossary_yasser": ("ياسر الدوسري", "Yasser Al-Dosari"),
    "alili": ("عزيز عليلي", "Aziz Alili"),
    "tunaiji": ("خليفة الطنيجي", "Khalifa Al-Tunaiji"),
    "banna": ("محمود علي البنا", "Mahmoud Ali Al-Banna"),
    "dossary_ibrahim": ("إبراهيم الدوسري", "Ibrahim Al-Dosari"),
    "yassin_jazaery": ("ياسين الجزائري", "Yassin Al-Jazaery"),
}

# Ordered (regex on the normalised folder label -> person id). First match wins.
_RULES = [
    (r"abdul_?basit|abdulsamad", "abdulbasit"),
    (r"juhaynee|juhany", "juhaynee"),
    (r"basfar", "basfar"),
    (r"matroud", "matroud"),
    (r"sudais", "sudais"),
    (r"shaatree|shatri", "shatri"),
    (r"neana", "neana"),
    (r"ajamy", "ajamy"),
    (r"alaqimy", "alaqimy"),
    (r"alafasy", "alafasy"),
    (r"suesy", "suesy"),
    (r"ghamadi|ghamdi", "ghamadi"),
    (r"hani_rifai", "rifai"),
    (r"hudhaify", "hudhaify"),
    (r"husary", "husary"),
    (r"mansoori", "mansoori"),
    (r"qahtaanee|qahtani", "qahtani"),
    (r"maher_?almuaiqly|muaiqly", "maher"),
    (r"minshawy|menshawi", "minshawy"),
    (r"tablaway|tablawy", "tablawy"),
    (r"muhammad_abdulkareem", "abdulkareem"),
    (r"ayyoub", "ayyoub"),
    (r"jibreel", "jibreel"),
    (r"muhsin_al_qasim", "muhsin_qasim"),
    (r"mustafa_ismail", "mustafa_ismail"),
    (r"qatami", "qatami"),
    (r"parhizgar", "parhizgar"),
    (r"sahl_yassin", "sahl_yassin"),
    (r"bukhatir", "bukhatir"),
    (r"budair", "budair"),
    (r"shuraym|shuraim", "shuraym"),
    (r"yaser_salamah", "yaser_salamah"),
    (r"yasser_ad-dussary|dussary|dosari", "dossary_yasser"),
    (r"aziz_alili", "alili"),
    (r"tunaiji", "tunaiji"),
    (r"al_banna", "banna"),
    (r"ibrahim_aldosary", "dossary_ibrahim"),
    (r"yassin_al_jazaery", "yassin_jazaery"),
]

STYLES_AR = {"mujawwad": "مجوّد", "murattal": "مرتّل", "muallim": "معلّم", "warsh": "رواية ورش"}


def person_of(label: str) -> str:
    """Folder label (e.g. 'Husary_128kbps_Mujawwad') -> person id (e.g. 'husary').

    Unknown labels fall back to a cleaned version of the label itself, so newly
    downloaded reciters still work (they just show the folder name)."""
    key = label.lower()
    for pat, pid in _RULES:
        if re.search(pat, key):
            return pid
    key = key.split("/")[-1]
    key = re.sub(r"_?\d+kbps.*$", "", key)
    return re.sub(r"[^a-z0-9]+", "_", key).strip("_")


def style_of(label: str) -> str:
    key = label.lower()
    if "warsh" in key:
        return "warsh"
    if "mujawwad" in key:
        return "mujawwad"
    if "muallim" in key or "teacher" in key:
        return "muallim"
    if "murattal" in key:
        return "murattal"
    return ""


def display_name(pid: str, lang: str = "ar") -> str:
    if pid in PEOPLE:
        return PEOPLE[pid][0 if lang == "ar" else 1]
    return pid.replace("_", " ").title()
