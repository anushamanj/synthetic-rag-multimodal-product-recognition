# streamlit_locked_sd156_rag.py
# ============================================================
# Locked SD1.5 + (Text2Image OR Image2Image) + CLIP Retrieval
# + TRUE Hybrid RAG:
#   GENERATED image (target) + retrieved evidence images -> Gemini 2.5 Flash -> JSON attributes
# ============================================================

import os, re, json, time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    DPMSolverMultistepScheduler
)
from transformers import CLIPModel, CLIPProcessor

# ----------------------------
# CONFIG
# ----------------------------
SD_MODEL_ID   = "runwayml/stable-diffusion-v1-5"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

# IMPORTANT: Gemini 2.5 Flash
GEMINI_MODEL  = "gemini-2.5-flash"

CLIP_MAX_LEN  = 77

BASE_STYLE = (
    "studio e-commerce catalog product photo, "
    "single product only, fully visible, centered, front view, "
    "plain seamless light gray background, soft studio lighting, sharp fabric detail"
)

NEGATIVE = (
    "person, model, human, face, head, body, skin, arms, legs, hands, feet, "
    "text, logo, watermark, cropped, close-up, blurry, lowres, deformed, disfigured"
)

DEFAULT_STEPS = 20
DEFAULT_CFG   = 7.0
DEFAULT_SEED  = 123
DEFAULT_TOPK  = 5
DEFAULT_LIMIT = 1000
DEFAULT_STRENGTH = 0.55

# ----------------------------
# LABEL SPACES (edit freely)
# ----------------------------
APPAREL_CATEGORY_LABELS = [
    "tees_tanks","shirts_polos","sweaters","sweatshirts_hoodies",
    "jackets_vests","pants","shorts","denim","suiting",
    "dresses","skirts"
]

COLOR_LABELS = [
    "black","white","gray","blue","navy","brown","beige","cream",
    "red","pink","green","yellow","purple","orange"
]

MATERIAL_LABELS = [
    "cotton","linen","denim","wool","cashmere","silk",
    "polyester","nylon","leather","suede","knit","lace","mesh"
]

PATTERN_LABELS = ["solid","striped","checked","printed","floral","abstract","logo","polka_dot"]
SIZE_LABELS = ["XS","S","M","L","XL","unknown"]

TEMPLATES = {
    "Category": [
        "a studio product photo of {}",
        "a clean e-commerce catalog photo of {}",
        "a centered front-view product shot of {}"
    ],
    "Color": [
        "a studio product photo of a {} colored garment",
        "a catalog photo of a {} colored clothing item"
    ],
    "Material": [
        "a studio product photo of a {} material garment",
        "a product photo showing {} fabric texture"
    ],
    "Pattern": [
        "a {} patterned garment",
        "a garment with {} pattern"
    ],
    "Size": [
        "a {} size garment",
        "{} size clothing"
    ]
}

# ----------------------------
# UTILS
# ----------------------------
def now_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(p: str) -> str:
    Path(p).mkdir(parents=True, exist_ok=True)
    return p

def snap64(x: int) -> int:
    return int(max(64, round(int(x) / 64) * 64))

def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def expand_prompt(p: str) -> str:
    p = (p or "").strip()
    if len(re.findall(r"[A-Za-z0-9]+", p)) <= 2:
        return f"{p}, full product shot, fully visible, centered, front view, not close-up"
    return f"{p}, fully visible, centered, front view, not close-up"

def guess_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    low = [c.lower() for c in df.columns]
    for cand in candidates:
        for i, c in enumerate(low):
            if cand.lower() in c:
                return df.columns[i]
    return None

# ----------------------------
# LOAD MODELS
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_sd(device: str):
    dtype = torch.float16 if device == "cuda" else torch.float32
    txt2img = StableDiffusionPipeline.from_pretrained(
        SD_MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False
    )
    txt2img.scheduler = DPMSolverMultistepScheduler.from_config(txt2img.scheduler.config)
    txt2img = txt2img.to(device)
    if device == "cuda":
        txt2img.enable_attention_slicing()

    img2img = StableDiffusionImg2ImgPipeline(
        vae=txt2img.vae,
        text_encoder=txt2img.text_encoder,
        tokenizer=txt2img.tokenizer,
        unet=txt2img.unet,
        scheduler=txt2img.scheduler,
        safety_checker=None,
        feature_extractor=txt2img.feature_extractor,
        requires_safety_checker=False
    ).to(device)
    if device == "cuda":
        img2img.enable_attention_slicing()

    return txt2img, img2img

@st.cache_resource(show_spinner=True)
def load_clip(device: str):
    # SAFETY FIX: use safetensors to avoid torch.load vulnerability gating
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID, use_safetensors=True).to(device)
    proc  = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model.eval()
    return model, proc

# ----------------------------
# CLIP HELPERS
# ----------------------------
@torch.no_grad()
def clip_image_embed(model: CLIPModel, proc: CLIPProcessor, img: Image.Image, device: str) -> np.ndarray:
    inputs = proc(images=img.convert("RGB"), return_tensors="pt").to(device)
    v = model.get_image_features(**inputs)
    v = v / v.norm(dim=-1, keepdim=True)
    return v.detach().cpu().numpy()[0]

@torch.no_grad()
def clip_text_embed(model: CLIPModel, proc: CLIPProcessor, texts: List[str], device: str) -> np.ndarray:
    inputs = proc(
        text=texts,
        padding=True,
        truncation=True,
        max_length=CLIP_MAX_LEN,
        return_tensors="pt"
    ).to(device)
    t = model.get_text_features(**inputs)
    t = t / t.norm(dim=-1, keepdim=True)
    return t.detach().cpu().numpy()

def clip_zeroshot(img: Image.Image, labels: List[str], templates: List[str],
                  model: CLIPModel, proc: CLIPProcessor, device: str, topk: int = 5) -> List[Dict]:
    prompts, lab_map = [], []
    for lab in labels:
        pretty = lab.replace("_"," ")
        for t in templates:
            prompts.append(t.format(pretty))
            lab_map.append(lab)

    T = clip_text_embed(model, proc, prompts, device)        # (P,D)
    V = clip_image_embed(model, proc, img, device)           # (D,)
    sims = (T @ V.reshape(-1,1)).squeeze(1)                  # (P,)

    best = {}
    for s, lab in zip(sims, lab_map):
        s = float(s)
        best[lab] = max(best.get(lab, -1e9), s)

    labs = list(best.keys())
    scores = np.array([best[l] for l in labs], dtype=np.float32)

    # CLIP temperature scaling if available
    try:
        scale = float(torch.exp(model.logit_scale).detach().cpu().item())
        scores = scores * scale
    except Exception:
        pass

    scores = scores - scores.max()
    probs = np.exp(scores)
    probs = probs / (probs.sum() + 1e-9)

    items = [{"label": lab, "conf": float(p)} for lab, p in zip(labs, probs)]
    items.sort(key=lambda x: x["conf"], reverse=True)
    return items[:max(1, int(topk))]

def top1(items: List[Dict]) -> str:
    return items[0]["label"] if items else "unknown"

# ----------------------------
# KB + RETRIEVAL
# ----------------------------
@st.cache_data(show_spinner=False)
def load_kb(kb_csv_path: str) -> Tuple[pd.DataFrame, str, Optional[str]]:
    df = pd.read_csv(kb_csv_path)

    # DeepFashion CSV typically has image_path_local, image_path, relative_path etc.
    img_col = guess_col(df, ["image_path_local", "image_path", "path", "filepath", "file", "img", "relative_path"])
    txt_col = guess_col(df, ["title", "desc", "description", "prompt", "caption", "category"])

    if img_col is None:
        raise ValueError(f"KB CSV has no image path column. Found columns: {list(df.columns)[:25]}")

    return df, img_col, txt_col

@st.cache_data(show_spinner=True)
def build_kb_embeds(kb_csv_path: str, kb_limit: int,
                    clip_model_id: str, device_tag: str) -> Dict:
    # cache key includes model id + device tag so streamlit reuses embeddings
    return {"kb_csv_path": kb_csv_path, "kb_limit": int(kb_limit), "clip_model_id": clip_model_id, "device_tag": device_tag}

def compute_kb_runtime(kb_csv_path: str, kb_limit: int,
                       clip_model: CLIPModel, clip_proc: CLIPProcessor, device: str) -> Dict:
    df, img_col, txt_col = load_kb(kb_csv_path)
    df2 = df.head(int(kb_limit)).copy()

    paths = df2[img_col].astype(str).tolist()
    texts = df2[txt_col].astype(str).tolist() if txt_col else [""] * len(paths)

    embs, ok_paths, ok_texts = [], [], []
    for p, t in zip(paths, texts):
        try:
            if not os.path.exists(p):
                continue
            im = Image.open(p).convert("RGB")
            embs.append(clip_image_embed(clip_model, clip_proc, im, device))
            ok_paths.append(p)
            ok_texts.append(t)
        except Exception:
            continue

    if len(embs) == 0:
        raise ValueError("KB embeddings = 0. Check your CSV path column points to real image files.")

    embs = np.stack(embs, axis=0)  # (N,D)
    return {"paths": ok_paths, "texts": ok_texts, "embs": embs}

def retrieve_topk(query_img: Image.Image, kb_runtime: Dict,
                  clip_model: CLIPModel, clip_proc: CLIPProcessor, device: str, topk: int) -> pd.DataFrame:
    q = clip_image_embed(clip_model, clip_proc, query_img, device)
    sims = kb_runtime["embs"] @ q.reshape(-1,1)
    sims = sims.squeeze(1)

    idx = np.argsort(-sims)[:int(topk)]
    out = pd.DataFrame({
        "rank": np.arange(len(idx)),
        "sim": sims[idx],
        "image_path": [kb_runtime["paths"][i] for i in idx],
        "kb_text": [kb_runtime["texts"][i] for i in idx],
    })
    return out

# ----------------------------
# GEMINI TRUE HYBRID RAG
# IMPORTANT FIX:
#   Send GENERATED/TARGET image first, then evidence images.
#   Evidence is secondary; if it conflicts, trust the FIRST image.
# ----------------------------
def _normalize(v: str) -> str:
    v = (v or "").strip().lower().replace(" ", "_")
    return v

def clamp_to_labels(out: Optional[Dict]) -> Optional[Dict]:
    if not out:
        return out

    def pick(v, allowed):
        v = _normalize(str(v))
        allowed_l = [_normalize(a) for a in allowed]
        if v in allowed_l:
            return allowed[allowed_l.index(v)]
        return "unknown"

    out2 = dict(out)
    out2["category"] = pick(out2.get("category",""), APPAREL_CATEGORY_LABELS)
    out2["primary_color"] = pick(out2.get("primary_color",""), COLOR_LABELS)
    out2["material"] = pick(out2.get("material",""), MATERIAL_LABELS)
    out2["pattern"] = pick(out2.get("pattern",""), PATTERN_LABELS)
    out2["size"] = pick(out2.get("size",""), SIZE_LABELS)
    return out2

def gemini_hybrid_label_true(
    query_img: Image.Image,
    evidence_paths: List[str],
    user_prompt: str,
    evidence_meta: Optional[List[Dict]] = None,
    max_evidence_images: int = 4
) -> Tuple[Optional[Dict], str]:

    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        return None, "Missing GOOGLE_API_KEY"

    # Prefer new SDK if available, fallback to older
    client = None
    using = None
    old_model = None

    try:
        from google import genai as genai_new  # new SDK
        client = genai_new.Client(api_key=api_key)
        using = "google.genai"
    except Exception:
        client = None

    if client is None:
        try:
            import google.generativeai as genai_old
            genai_old.configure(api_key=api_key)
            old_model = genai_old.GenerativeModel(GEMINI_MODEL)
            using = "google.generativeai"
        except Exception as e:
            return None, f"Gemini SDK not available. Install google-genai OR google-generativeai. ({e})"

    # Load evidence images
    ev_imgs: List[Image.Image] = []
    for p in evidence_paths[:max_evidence_images]:
        try:
            ev_imgs.append(Image.open(p).convert("RGB"))
        except Exception:
            continue

    # Optional evidence meta (keep short)
    ev_text = ""
    if evidence_meta:
        lines = []
        for m in evidence_meta[:max_evidence_images]:
            ip = m.get("image_path", "")
            kbcat = m.get("kb_category", "")
            lines.append(f"- path={ip} kb_category={kbcat}")
        ev_text = "\n".join(lines)

    prompt = f"""
You are a strict fashion attribute labeling system.

CRITICAL RULES:
1) The FIRST image is the TARGET image to label (the generated image).
2) The remaining images are retrieval evidence ONLY.
3) If evidence conflicts with the target, ALWAYS trust the FIRST image.
4) Return STRICT JSON only, no explanations.

User prompt (may be noisy): {user_prompt}

Allowed labels:
category: {APPAREL_CATEGORY_LABELS}
primary_color: {COLOR_LABELS}
material: {MATERIAL_LABELS}
pattern: {PATTERN_LABELS}
size: {SIZE_LABELS}

Evidence meta (optional, may be noisy):
{ev_text}

Return schema EXACTLY:
{{
  "category": "...",
  "primary_color": "...",
  "material": "...",
  "pattern": "...",
  "size": "..."
}}
""".strip()

    try:
        if using == "google.genai":
            # Send text + TARGET IMAGE first + evidence images
            contents = [prompt, query_img] + ev_imgs
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config={"temperature": 0.0}
            )
            text = resp.text
        else:
            resp = old_model.generate_content(
                [prompt, query_img, *ev_imgs],
                generation_config={"temperature": 0.0}
            )
            text = resp.text

        # Extract JSON block
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            return None, f"Gemini returned non-JSON: {text[:200]}..."
        obj = json.loads(m.group(0))
        obj = clamp_to_labels(obj)
        return obj, f"OK ({using})"
    except Exception as e:
        return None, f"Gemini call failed: {e}"

# ----------------------------
# SD GENERATION
# ----------------------------
def generate_txt2img(pipe, prompt, seed, steps, cfg, width, height, device) -> Image.Image:
    g = torch.Generator(device=device).manual_seed(int(seed))
    out = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE,
        num_inference_steps=int(steps),
        guidance_scale=float(cfg),
        width=int(width),
        height=int(height),
        generator=g
    )
    return out.images[0]

def generate_img2img(pipe, init_image: Image.Image, prompt, seed, steps, cfg, strength, width, height, device) -> Image.Image:
    g = torch.Generator(device=device).manual_seed(int(seed))
    init = init_image.convert("RGB").resize((int(width), int(height)))
    out = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE,
        image=init,
        strength=float(clamp01(strength)),
        num_inference_steps=int(steps),
        guidance_scale=float(cfg),
        generator=g
    )
    return out.images[0]

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.set_page_config(page_title="SD1.5 + CLIP + Gemini 2.5 Flash Hybrid RAG", layout="wide")
st.title("Prompt → Generate → Evidence → Attributes (CLIP + Gemini 2.5 Flash Hybrid RAG)")

device = "cuda" if torch.cuda.is_available() else "cpu"
txt2img_pipe, img2img_pipe = load_sd(device)
clip_model, clip_proc = load_clip(device)

with st.sidebar:
    st.header("KB + Hybrid RAG")
    kb_csv = st.text_input("KB CSV", value=r"E:\datasetfashion\Deepfashion\deepfashion_img_index.csv")
    topk = st.slider("Evidence Top-K", 1, 20, int(DEFAULT_TOPK), 1)
    kb_limit = st.number_input("KB embed limit (speed)", min_value=50, max_value=50000, value=int(DEFAULT_LIMIT), step=50)

    use_gemini = st.checkbox("Use Gemini Hybrid RAG", value=True)
    st.caption("Requires GOOGLE_API_KEY env var.")

    st.divider()
    st.write(f"Device: **{device}**")
    st.write(f"Torch: **{torch.__version__}**")

colA, colB, colC = st.columns([0.33, 0.34, 0.33], gap="large")

with colA:
    st.header("Generation")
    mode = st.radio("Mode", ["Text → Image", "Image → Image (img2img)"], index=0)

    user_prompt = st.text_area("Product prompt", value="crop top and pants", height=100)
    seed = st.number_input("Seed", value=int(DEFAULT_SEED), step=1)
    steps = st.slider("Steps", 10, 40, int(DEFAULT_STEPS), 1)
    cfg = st.slider("CFG", 3.0, 12.0, float(DEFAULT_CFG), 0.5)

    w = snap64(st.number_input("Width", value=640, step=64))
    h = snap64(st.number_input("Height", value=768, step=64))

    if mode.startswith("Image"):
        strength = st.slider("Img2Img strength", 0.05, 0.85, float(DEFAULT_STRENGTH), 0.05)
        ref_file = st.file_uploader("Reference image (jpg/png)", type=["jpg", "jpeg", "png"])
    else:
        strength = None
        ref_file = None

    run_btn = st.button("Generate + Label", use_container_width=True)

with colB:
    st.header("Evidence (CLIP retrieval)")
with colC:
    st.header("Predicted attributes")

if run_btn:
    if not kb_csv or not os.path.exists(kb_csv):
        st.error("KB CSV path not found. Set a valid KB CSV path in the sidebar.")
        st.stop()

    t0 = time.time()
    final_prompt = f"{expand_prompt(user_prompt)}, {BASE_STYLE}"

    # Generate
    if mode.startswith("Text"):
        gen_img = generate_txt2img(txt2img_pipe, final_prompt, seed, steps, cfg, w, h, device)
    else:
        if ref_file is None:
            st.error("Upload a reference image for img2img.")
            st.stop()
        ref_img = Image.open(ref_file).convert("RGB")
        gen_img = generate_img2img(img2img_pipe, ref_img, final_prompt, seed, steps, cfg, strength, w, h, device)

    colB.image(gen_img, caption="Generated (SD output)", use_container_width=True)
    st.success(f"Generated in {time.time()-t0:.2f}s | {w}×{h}")

    # CLIP-only attributes
    cat = clip_zeroshot(gen_img, APPAREL_CATEGORY_LABELS, TEMPLATES["Category"], clip_model, clip_proc, device, topk=5)
    colr = clip_zeroshot(gen_img, COLOR_LABELS, TEMPLATES["Color"], clip_model, clip_proc, device, topk=5)
    mat = clip_zeroshot(gen_img, MATERIAL_LABELS, TEMPLATES["Material"], clip_model, clip_proc, device, topk=5)
    pat = clip_zeroshot(gen_img, PATTERN_LABELS, TEMPLATES["Pattern"], clip_model, clip_proc, device, topk=5)
    siz = clip_zeroshot(gen_img, SIZE_LABELS, TEMPLATES["Size"], clip_model, clip_proc, device, topk=5)

    clip_attrs = {
        "category": top1(cat),
        "primary_color": top1(colr),
        "material": top1(mat),
        "pattern": top1(pat),
        "size": top1(siz),
    }

    with colC:
        st.subheader("CLIP-only attributes")
        st.json(clip_attrs)

    # Retrieval (evidence)
    try:
        _ = build_kb_embeds(kb_csv, int(kb_limit), CLIP_MODEL_ID, device)
        kb_runtime = compute_kb_runtime(kb_csv, int(kb_limit), clip_model, clip_proc, device)
        ev_df = retrieve_topk(gen_img, kb_runtime, clip_model, clip_proc, device, int(topk))
    except Exception as e:
        st.error(f"Evidence retrieval failed: {e}")
        st.stop()

    colB.dataframe(ev_df[["rank","sim","image_path","kb_text"]], use_container_width=True, height=240)

    # Show evidence images grid
    ev_imgs = []
    for p in ev_df["image_path"].head(4).tolist():
        try:
            ev_imgs.append(Image.open(p).convert("RGB"))
        except Exception:
            pass
    if ev_imgs:
        cols = colB.columns(len(ev_imgs))
        for c, im in zip(cols, ev_imgs):
            c.image(im, use_container_width=True)

    # Gemini True Hybrid RAG (FIXED)
    with colC:
        st.divider()
        st.subheader("Hybrid (TARGET image + evidence → Gemini 2.5 Flash RAG) attributes")

        if not use_gemini:
            st.info("Gemini Hybrid RAG disabled (checkbox off).")
        else:
            evidence_paths = ev_df["image_path"].tolist()
            evidence_meta = [{"image_path": p} for p in evidence_paths]

            gemini_attrs, status = gemini_hybrid_label_true(
                query_img=gen_img,                 # ✅ TARGET FIRST
                evidence_paths=evidence_paths,      # ✅ EVIDENCE SECOND
                user_prompt=user_prompt,
                evidence_meta=evidence_meta,
                max_evidence_images=4
            )

            if gemini_attrs is None:
                st.error(f"Gemini-RAG failed: {status}")
            else:
                st.success(f"RAG ACTIVE: target+evidence sent to Gemini. {status}")
                st.json(gemini_attrs)
