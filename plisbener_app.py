import streamlit as st
import numpy as np
import cv2
from PIL import Image

# ==================================================
#  FUNGSI BANTU
# ==================================================
def load_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    return np.array(img)

def apply_transformation_full(image, matrix):
    h, w = image.shape[:2]
    corners = np.array([[0,0],[w,0],[0,h],[w,h]], dtype=np.float32)
    new_corners = cv2.transform(np.array([corners]), matrix)[0]
    x_coords, y_coords = new_corners[:,0], new_corners[:,1]
    min_x, max_x = x_coords.min(), x_coords.max()
    min_y, max_y = y_coords.min(), y_coords.max()
    new_w, new_h = int(max_x - min_x), int(max_y - min_y)
    translation = np.array([[1,0,-min_x],[0,1,-min_y]], dtype=np.float32)
    full_matrix = translation @ np.vstack([matrix,[0,0,1]])
    return cv2.warpAffine(image, full_matrix[:2,:], (new_w,new_h))

def apply_translation(image, tx, ty):
    h, w = image.shape[:2]
    margin = max(abs(tx), abs(ty), 300)
    new_w, new_h = w + margin*2, h + margin*2
    matrix = np.float32([[1,0,tx+margin],[0,1,ty+margin]])
    return cv2.warpAffine(image, matrix, (new_w,new_h))

# ==================================================
#  CUSTOM CSS FOR MODERN LOOK
# ==================================================
st.markdown("""
<style>
.image-card { border-radius: 15px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.3); transition: transform 0.3s; }
.image-card:hover { transform: scale(1.05); }
h1, h2, h3 { text-align: center; }
.css-1d391kg {padding-top:2rem;}
</style>
""", unsafe_allow_html=True)

# ==================================================
#  GLOBAL LANGUAGE SELECTION
# ==================================================
language = st.sidebar.selectbox("Choose Language / Pilih Bahasa", ["Indonesia", "English"])

# ==================================================
#  PAGE 1 — HOME / INTRODUCTION
# ==================================================
def page_home():
    if language == "Indonesia":
        st.title("📸 Aplikasi Transformasi & Pengolahan Gambar")
        st.markdown("""
        **Deskripsi Aplikasi**  
        Aplikasi ini memungkinkan Anda melakukan berbagai **transformasi matriks** dan **operasi konvolusi** pada gambar.  
        Anda dapat melakukan translasi, scaling, rotasi, shearing, refleksi, serta melihat perbandingan Before–After.
        """)
        with st.expander("📐 Transformasi Matriks"):
            st.markdown("""
            Transformasi matriks mengubah posisi, ukuran, atau orientasi objek pada gambar.
            - **Translasi**: Memindahkan gambar ke posisi lain. `Matrix: [[1, 0, tx], [0, 1, ty]]`
            - **Scaling**: Memperbesar atau memperkecil gambar. `Matrix: [[sx, 0, 0], [0, sy, 0]]`
            - **Rotasi**: Memutar gambar. `Matrix: [[cosθ, -sinθ, 0], [sinθ, cosθ, 0]]`
            """)
        with st.expander("🖼 Konvolusi (Convolution)"):
            st.markdown("""
            Konvolusi memodifikasi gambar menggunakan *kernel/filter*.
            - Blur: `Kernel: 1/9 * [[1,1,1],[1,1,1],[1,1,1]]`
            - Deteksi tepi: `Kernel: [[-1,-1,-1],[0,0,0],[1,1,1]]`
            """)
    else:
        st.title("📸 Image Transformation & Processing App")
        st.markdown("""
        **App Description**  
        This app allows you to perform **matrix transformations** and **convolution operations** on images.  
        You can translate, scale, rotate, shear, reflect, and see Before–After comparisons.
        """)
        with st.expander("📐 Matrix Transformations"):
            st.markdown("""
            Matrix transformations change the position, size, or orientation of objects in an image.
            - Translation: `Matrix: [[1, 0, tx], [0, 1, ty]]`
            - Scaling: `Matrix: [[sx, 0, 0], [0, sy, 0]]`
            - Rotation: `Matrix: [[cosθ, -sinθ, 0], [sinθ, cosθ, 0]]`
            """)
        with st.expander("🖼 Convolution"):
            st.markdown("""
            Convolution modifies an image using a *kernel/filter*.
            - Blur: `Kernel: 1/9 * [[1,1,1],[1,1,1],[1,1,1]]`
            - Edge Detection: `Kernel: [[-1,-1,-1],[0,0,0],[1,1,1]]`
            """)

# ==================================================
#  PAGE 2 — IMAGE PROCESSING TOOLS
# ==================================================
def page_tools():
    # Bahasa
    if language == "Indonesia":
        st.title("🛠 Alat Pengolahan Gambar")
        menu_label = "Transformasi:"
        upload_label = "Unggah gambar"
        info_label = "Unggah gambar untuk mulai menggunakan tools."
        before_after_label = "Sebelum–Sesudah"
        tx_label = "Geser Horizontal (tx)"
        ty_label = "Geser Vertikal (ty)"
        sx_label = "Skala X"
        sy_label = "Skala Y"
        angle_label = "Sudut Rotasi (°)"
        shx_label = "Shear X"
        shy_label = "Shear Y"
        axis_label = "Sumbu Refleksi"
        transformed_label = "Gambar Hasil Transformasi"
        menu_options = ["Translasi", "Scaling", "Rotasi", "Shearing", "Refleksi"]
    else:
        st.title("🛠 Image Processing Tools")
        menu_label = "Transformation:"
        upload_label = "Upload image"
        info_label = "Upload an image to start using the tools."
        before_after_label = "Before–After"
        tx_label = "Horizontal Shift (tx)"
        ty_label = "Vertical Shift (ty)"
        sx_label = "Scale X"
        sy_label = "Scale Y"
        angle_label = "Rotation Angle (°)"
        shx_label = "Shear X"
        shy_label = "Shear Y"
        axis_label = "Reflection Axis"
        transformed_label = "Transformed Image"
        menu_options = ["Translation","Scaling","Rotation","Shearing","Reflection"]

    menu = st.sidebar.radio(menu_label, menu_options)
    uploaded = st.file_uploader(upload_label, type=["jpg", "png"])

    if uploaded:
        img = load_image(uploaded)
        st.subheader(before_after_label)
        st.image(img, caption=upload_label, use_container_width=True)
    else:
        st.info(info_label)
        return

    # === TOOLS HALAMAN ===
    if menu in ["Translasi","Translation"]:
        tx = st.slider(tx_label, -500, 500, 0)
        ty = st.slider(ty_label, -500, 500, 0)
        transformed = apply_translation(img, tx, ty)

    elif menu in ["Scaling"]:
        sx = st.slider(sx_label, 0.1, 3.0, 1.0)
        sy = st.slider(sy_label, 0.1, 3.0, 1.0)
        matrix = np.float32([[sx, 0, 0], [0, sy, 0]])
        transformed = apply_transformation_full(img, matrix)

    elif menu in ["Rotasi","Rotation"]:
        angle = st.slider(angle_label, -180, 180, 0)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        transformed = apply_transformation_full(img, matrix)

    elif menu in ["Shearing"]:
        shx = st.slider(shx_label, -1.0, 1.0, 0.0)
        shy = st.slider(shy_label, -1.0, 1.0, 0.0)
        matrix = np.float32([[1, shx, 0], [shy, 1, 0]])
        transformed = apply_transformation_full(img, matrix)

    elif menu in ["Refleksi","Reflection"]:
        axis = st.selectbox(axis_label, ["Horizontal","Vertikal"] if language=="English" else ["Horizontal","Vertikal"])
        if axis in ["Horizontal","Horizontal"]:
            matrix = np.float32([[1, 0, 0], [0, -1, img.shape[0]]])
        else:
            matrix = np.float32([[-1, 0, img.shape[1]], [0, 1, 0]])
        transformed = apply_transformation_full(img, matrix)

    st.image(transformed, caption=transformed_label, use_container_width=True)

# ==================================================
#  PAGE 3 — TEAM MEMBERS
# ==================================================
def page_team():
    if language == "Indonesia":
        st.title("👥 Anggota Tim")
        desc = "Berikut adalah anggota tim beserta foto dan peran masing-masing."
        how_work_title = "🧠 Bagaimana aplikasi ini bekerja?"
        how_work_text = """
        - Pengguna mengunggah gambar melalui uploader.  
        - Setiap transformasi memiliki matriks yang diterapkan ke koordinat piksel gambar.  
        - Fungsi `cv2.warpAffine()` digunakan untuk memindahkan piksel ke lokasi baru.  
        - Untuk mencegah gambar terpotong, bounding box dihitung ulang.  
        - Hasil ditampilkan sebagai **Before – After**.  
        """
    else:
        st.title("👥 Team Members")
        desc = "Here are the team members with their photos and roles."
        how_work_title = "🧠 How this app works?"
        how_work_text = """
        - Users upload images via the uploader.  
        - Each transformation has a matrix applied to pixel coordinates.  
        - The `cv2.warpAffine()` function moves pixels to new locations.  
        - To avoid cropping, bounding boxes are recalculated.  
        - Results are displayed as **Before – After**.  
        """

    st.markdown(desc)

    team = [
        {"name": "Muhammad Nurul Falah", "photo": "c:/Users/RAYYAN/OneDrive/画像/Saved Pictures/WhatsApp Image 2025-12-05 at 16.52.37_d44a1893.jpg", "role": "Making App"},
        {"name": "Rayyan Hasan Rabbani", "photo": "c:/Users/RAYYAN/OneDrive/画像/Saved Pictures/WhatsApp Image 2025-12-05 at 16.52.37_74c31fec.jpg", "role": "Making App"},
        {"name": "Tobias Dashiel Hapsoro", "photo": "c:/Users/RAYYAN/OneDrive/画像/Saved Pictures/WhatsApp Image 2025-12-05 at 16.52.38_008c607a.jpg", "role": "Making App"}
    ]

    cols = st.columns(3)
    for col, member in zip(cols, team):
        with col:
            st.image(member["photo"], width=150)
            st.subheader(member["name"])
            st.write(member["role"])

    st.markdown("---")
    st.subheader(how_work_title)
    st.markdown(how_work_text)

# ==================================================
#  MAIN NAVIGATION
# ==================================================
pages = {
    "🏠 Home": page_home,
    "🛠 Image Tools": page_tools,
    "👥 Team": page_team,
}

st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Pilih Halaman:" if language=="Indonesia" else "Select Page:", list(pages.keys()))
pages[page]()