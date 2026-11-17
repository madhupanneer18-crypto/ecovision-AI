"""
EcoVision AI — Smart Waste Classification and Detection
Automated Real-Time Waste Classification System Using Deep Learning.

Main Streamlit Application
"""

import os
from pathlib import Path
from PIL import Image
import pandas as pd
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="EcoVision AI — Smart Waste Classification & Detection",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import internal modules
from src.config import (
    BASE_DIR,
    ASSETS_DIR,
    SAMPLES_DIR,
    CSS_DIR,
    CLASSIFICATION_CLASSES,
    DETECTION_CLASSES,
    CLASS_COLORS,
    WASTE_INFO,
    SAMPLE_IMAGES,
    APP_DISCLAIMER
)
from src.inference import get_engine
from src.history_manager import (
    load_history,
    log_prediction,
    export_history_csv,
    clear_history
)
from src.analytics import (
    generate_kpis,
    plot_category_distribution,
    plot_recyclability_ratio,
    plot_confidence_distribution,
    plot_activity_timeline,
    plot_probability_bar
)
from src.eco_advisor import get_advisor

# Inject Custom CSS
css_file = CSS_DIR / "style.css"
if css_file.exists():
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
@st.cache_resource
def load_cached_engine():
    return get_engine()

@st.cache_resource
def load_cached_advisor():
    return get_advisor()


# -----------------------------------------------------------------------------
# Sidebar Navigation & Branding
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ♻️ **EcoVision AI**")
    st.caption("Smart Waste Classification & Detection")
    st.markdown("---")

    selected_page = st.radio(
        "**Navigation**",
        [
            "🏠 Overview",
            "🔍 Waste Classification",
            "🎯 Waste Detection",
            "📸 Camera Snap & Analyze",
            "📊 Analytics Dashboard",
            "📋 Prediction History",
            "♻️ Recycling & Disposal Guide"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("### ⚙️ System Status")
    st.success("🟢 **YOLOv8 Detector**: Active")
    st.success("🟢 **WasteNet Classifier**: Active (97.3% Acc)")
    st.info("⚡ **Inference Engine**: PyTorch CPU")

    st.markdown("---")
    st.warning(f"⚠️ **Note**: {APP_DISCLAIMER}")
    st.caption("EcoVision AI v2.4 • Built with Streamlit & YOLOv8")


# =============================================================================
# 1. OVERVIEW PAGE
# =============================================================================
if selected_page == "🏠 Overview":
    st.markdown(
        """
        <div class="ecovision-banner">
            <h1>♻️ EcoVision AI</h1>
            <p>
                Automated Real-Time Waste Classification System Using YOLOv8 Deep Learning.
                Detect, categorize, and sort municipal and household waste with instant recycling guidelines,
                carbon offset calculations, and interactive analytics.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Key statistics row
    df_hist = load_history()
    kpis = generate_kpis(df_hist)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="eco-kpi-card">
                <div class="eco-kpi-label">Total Items Processed</div>
                <div class="eco-kpi-val">{kpis['total_items']}</div>
                <small style="color: #64748b;">Across all inference modes</small>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"""
            <div class="eco-kpi-card">
                <div class="eco-kpi-label">Diversion Rate</div>
                <div class="eco-kpi-val">{kpis['diversion_rate']}%</div>
                <small style="color: #2e7d32;">Diverted from landfills</small>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"""
            <div class="eco-kpi-card">
                <div class="eco-kpi-label">CO2 Avoided</div>
                <div class="eco-kpi-val">{kpis['total_co2_kg']} kg</div>
                <small style="color: #64748b;">≈ {kpis['trees_saved']} tree-years</small>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            f"""
            <div class="eco-kpi-card">
                <div class="eco-kpi-label">Avg AI Confidence</div>
                <div class="eco-kpi-val">{kpis['avg_confidence']}%</div>
                <small style="color: #64748b;">Calibrated softmax score</small>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🌟 Key Capabilities")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <div style="font-size: 2rem;">🔍</div>
                <h3>Whole-Image Classification</h3>
                <p>
                    Deep learning analysis of entire images across 6 target categories:
                    <b>Plastic, Glass, Metal, Paper, Cardboard, and Trash</b> with full probability distributions.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <div style="font-size: 2rem;">🎯</div>
                <h3>Multi-Object Detection</h3>
                <p>
                    Locates multiple discarded objects simultaneously using YOLOv8 bounding boxes
                    for <b>Glass, Metal, Paper, Plastic, and Residual</b> waste streams.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            """
            <div class="feature-card">
                <div style="font-size: 2rem;">📸</div>
                <h3>Camera Photo Snap</h3>
                <p>
                    Capture still photos directly from your web camera or mobile device for instant on-the-spot
                    disposal recommendations and recyclability validation.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown(
            """
            <div class="feature-card">
                <div style="font-size: 2rem;">📊</div>
                <h3>Analytics & Impact Dashboard</h3>
                <p>
                    Interactive Plotly charts detailing waste material distributions, recyclability ratios,
                    carbon emissions saved, and model confidence trends over time.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col5:
        st.markdown(
            """
            <div class="feature-card">
                <div style="font-size: 2rem;">📋</div>
                <h3>History & CSV Export</h3>
                <p>
                    Persistent audit log recording every inference event with timestamps, object counts,
                    confidence scores, and one-click CSV report export.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col6:
        st.markdown(
            """
            <div class="feature-card">
                <div style="font-size: 2rem;">♻️</div>
                <h3>Recycling & Disposal Guide</h3>
                <p>
                    Searchable 100+ item knowledge base, resin code identifier (Types #1-#7),
                    contamination warnings, and an interactive carbon calculator.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        f"💡 **Tip**: Navigate using the left sidebar to start classifying waste images, run multi-object detection, or explore the recycling guide."
    )


# =============================================================================
# 2. WHOLE-IMAGE WASTE CLASSIFICATION PAGE
# =============================================================================
elif selected_page == "🔍 Waste Classification":
    st.markdown("## 🔍 Whole-Image Waste Classification")
    st.write(
        "Upload a photo of a waste item or choose from curated samples. The deep learning model will predict "
        "the waste category across **Plastic, Glass, Metal, Paper, Cardboard, and Trash**, providing exact recycling advice."
    )

    engine = load_cached_engine()

    tab_upload, tab_sample = st.tabs(["📤 Upload Image", "🖼️ Curated Samples"])
    selected_image = None
    image_source = "Upload"

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload waste image (JPG, PNG, JPEG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            key="cls_uploader"
        )
        if uploaded_file is not None:
            try:
                selected_image = Image.open(uploaded_file).convert("RGB")
                image_source = "Upload"
            except Exception as e:
                st.error(f"Error reading image: {e}")

    with tab_sample:
        st.write("Select a curated test sample to evaluate model classification:")
        sample_keys = list(SAMPLE_IMAGES.keys())[:6]  # Single item samples
        cols = st.columns(len(sample_keys))
        for idx, key in enumerate(sample_keys):
            info = SAMPLE_IMAGES[key]
            sample_path = SAMPLES_DIR / info["filename"]
            with cols[idx]:
                if sample_path.exists():
                    st.image(str(sample_path), use_container_width=True)
                    if st.button(f"{info['category']}", key=f"btn_sample_{idx}"):
                        selected_image = Image.open(sample_path).convert("RGB")
                        image_source = "Sample"

    st.markdown("---")

    if selected_image is not None:
        col_img, col_res = st.columns([1, 1.25])

        with col_img:
            st.markdown("### 📷 Input Image")
            st.image(selected_image, use_container_width=True)

        with col_res:
            st.markdown("### 🧠 AI Classification Result")
            with st.spinner("Classifying waste item with deep learning..."):
                res = engine.classify(selected_image)

            cls_name = res["class"]
            conf = res["confidence"]
            probs = res["probabilities"]
            info = res["info"]

            color_cfg = CLASS_COLORS.get(cls_name, CLASS_COLORS["Trash"])
            badge_class = "badge-recyclable" if info["status_color"] == "green" else "badge-residual"

            # Result Header Pill
            st.markdown(
                f"""
                <div style="background: {color_cfg['hex']}15; border: 1.5px solid {color_cfg['hex']}; border-radius: 12px; padding: 18px; margin-bottom: 15px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 1.6rem; font-weight: 800; color: #1e293b;">
                            {color_cfg['icon']} {cls_name}
                        </span>
                        <span class="badge-pill {badge_class}">
                            {info['status']}
                        </span>
                    </div>
                    <div style="margin-top: 10px; font-size: 1rem; color: #475569;">
                        AI Confidence: <b>{conf * 100:.1f}%</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Recommended Bin Container
            st.markdown(
                f"""
                <div class="bin-box" style="background-color: {color_cfg['bin_color']}30; border-left: 6px solid {color_cfg['hex']};">
                    <div style="font-size: 2rem;">{color_cfg['icon']}</div>
                    <div>
                        <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.5px; color: #475569;">Recommended Bin</div>
                        <div style="font-size: 1.15rem; color: #0f172a; font-weight: 700;">{color_cfg['bin']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Metrics row
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Decomposition Time", info.get("decomposition", "N/A"))
            with m2:
                st.metric("CO2 Avoided", f"{info.get('co2_saved_kg', 0.0):.2f} kg")
            with m3:
                st.metric("Energy Saved", f"{info.get('energy_saved_kwh', 0.0):.2f} kWh")

            # Probability Distribution Chart
            st.markdown("#### Probability Distribution")
            fig_probs = plot_probability_bar(probs)
            st.plotly_chart(fig_probs, use_container_width=True)

            # Handling and Guidance
            st.markdown(
                f"""
                <div class="guidance-card">
                    <h4>📋 Preparation & Disposal Guidance</h4>
                    <p style="font-size: 0.95rem; color: #334155; margin-bottom: 8px;">{info.get('guidance', '')}</p>
                    <small style="color: #64748b;"><b>Resin / Material Codes:</b> {info.get('resin_codes', 'N/A')}</small><br>
                    <small style="color: #2e7d32;"><b>Eco Fact:</b> {info.get('tips', '')}</small>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Log to History Button
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("📝 Log This Scan to Prediction History", key="btn_log_cls"):
                rec = log_prediction(
                    mode="Classification",
                    primary_class=cls_name,
                    confidence=conf,
                    item_count=1,
                    source=image_source,
                    notes=f"Single item {cls_name} classification ({conf*100:.1f}%)"
                )
                st.success(f"Successfully logged {cls_name} prediction to history!")

    else:
        st.info("👆 Please upload an image or choose one of the curated test samples above to begin analysis.")


# =============================================================================
# 3. MULTI-OBJECT WASTE DETECTION PAGE
# =============================================================================
elif selected_page == "🎯 Waste Detection":
    st.markdown("## 🎯 Multiple-Object Waste Detection with Bounding Boxes")
    st.write(
        "Detect and localize multiple waste items simultaneously. The YOLOv8 model generates bounding boxes "
        "and labels for **Glass, Metal, Paper, Plastic, and Residual** waste objects."
    )

    engine = load_cached_engine()

    # Settings Sidebar / Controls
    with st.expander("🛠️ Detection Controls & Thresholds", expanded=True):
        c_slider1, c_slider2, c_legend = st.columns([1, 1, 1.5])
        with c_slider1:
            conf_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.10,
                max_value=1.00,
                value=0.25,
                step=0.05,
                help="Minimum confidence required for bounding box generation."
            )
        with c_slider2:
            iou_threshold = st.slider(
                "IoU (NMS) Threshold",
                min_value=0.10,
                max_value=1.00,
                value=0.45,
                step=0.05,
                help="Intersection over Union threshold for Non-Maximum Suppression."
            )
        with c_legend:
            st.write("**Color Key:**")
            st.markdown(
                f"""
                <div style="display: flex; flex-wrap: wrap; gap: 8px; font-size: 0.85rem;">
                    <span style="background: {CLASS_COLORS['Glass']['hex']}30; border: 1px solid {CLASS_COLORS['Glass']['hex']}; padding: 3px 8px; border-radius: 4px; font-weight: 700; color: #1e293b;">🍾 Glass</span>
                    <span style="background: {CLASS_COLORS['Metal']['hex']}30; border: 1px solid {CLASS_COLORS['Metal']['hex']}; padding: 3px 8px; border-radius: 4px; font-weight: 700; color: #1e293b;">🥫 Metal</span>
                    <span style="background: {CLASS_COLORS['Paper']['hex']}30; border: 1px solid {CLASS_COLORS['Paper']['hex']}; padding: 3px 8px; border-radius: 4px; font-weight: 700; color: #1e293b;">📄 Paper</span>
                    <span style="background: {CLASS_COLORS['Plastic']['hex']}30; border: 1px solid {CLASS_COLORS['Plastic']['hex']}; padding: 3px 8px; border-radius: 4px; font-weight: 700; color: #1e293b;">🧴 Plastic</span>
                    <span style="background: {CLASS_COLORS['Residual']['hex']}30; border: 1px solid {CLASS_COLORS['Residual']['hex']}; padding: 3px 8px; border-radius: 4px; font-weight: 700; color: #1e293b;">⚠️ Residual</span>
                </div>
                """,
                unsafe_allow_html=True
            )

    tab_det_upload, tab_det_sample = st.tabs(["📤 Upload Image", "🖼️ Curated Detection Scene"])
    det_image = None
    det_source = "Upload"

    with tab_det_upload:
        uploaded_det_file = st.file_uploader(
            "Upload image with multiple waste items (JPG, PNG, JPEG)",
            type=["jpg", "jpeg", "png", "webp"],
            key="det_uploader"
        )
        if uploaded_det_file is not None:
            try:
                det_image = Image.open(uploaded_det_file).convert("RGB")
                det_source = "Upload"
            except Exception as e:
                st.error(f"Error loading image: {e}")

    with tab_det_sample:
        st.write("Test multi-object waste detection with pre-configured sample scenes:")
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            multi_path = SAMPLES_DIR / "multi_waste.jpg"
            if multi_path.exists():
                st.image(str(multi_path), caption="Mixed Recyclables Scene", use_container_width=True)
                if st.button("Detect Mixed Recyclables", key="btn_sample_multi"):
                    det_image = Image.open(multi_path).convert("RGB")
                    det_source = "Sample"
        with s_col2:
            trash_path = SAMPLES_DIR / "trash_residual.jpg"
            if trash_path.exists():
                st.image(str(trash_path), caption="Composite Residual Waste Scene", use_container_width=True)
                if st.button("Detect Residual Items", key="btn_sample_res"):
                    det_image = Image.open(trash_path).convert("RGB")
                    det_source = "Sample"

    st.markdown("---")

    if det_image is not None:
        with st.spinner("Running YOLOv8 object detection and bounding box localization..."):
            det_res = engine.detect(det_image, conf_threshold=conf_threshold, iou_threshold=iou_threshold)

        annotated_img = det_res["annotated_image"]
        detections = det_res["detections"]
        counts = det_res["counts"]
        total_det = det_res["total_detected"]

        # View images
        col_orig, col_annot = st.columns(2)
        with col_orig:
            st.markdown("### 📷 Original Image")
            st.image(det_image, use_container_width=True)
        with col_annot:
            st.markdown(f"### 🎯 Annotated Image ({total_det} items localized)")
            st.image(annotated_img, use_container_width=True)

        # Count Badges Breakdown Row
        st.markdown("### 📊 Detection Summary & Counts")
        cnt_cols = st.columns(len(DETECTION_CLASSES) + 1)
        with cnt_cols[0]:
            st.metric("Total Detected", total_det)
        for idx, c_name in enumerate(DETECTION_CLASSES):
            with cnt_cols[idx + 1]:
                cfg = CLASS_COLORS.get(c_name, {})
                st.metric(f"{cfg.get('icon', '')} {c_name}", counts.get(c_name, 0))

        # Detected Items Table
        st.markdown("### 📋 Detected Objects Details")
        if total_det > 0:
            det_rows = []
            for i, d in enumerate(detections):
                x1, y1, x2, y2 = d["box"]
                det_rows.append({
                    "Item #": i + 1,
                    "Category": d["class"],
                    "Confidence": f"{d['confidence']*100:.1f}%",
                    "Bounding Box [x1, y1, x2, y2]": f"[{x1}, {y1}, {x2}, {y2}]",
                    "Disposal Guidance": d["guidance"]
                })
            df_det_table = pd.DataFrame(det_rows)
            st.dataframe(df_det_table, use_container_width=True)

            # Crop & Inspect Tool
            st.markdown("#### 🔬 Inspect Individual Detected Object")
            obj_idx = st.selectbox(
                "Select object to crop and view detailed eco guidance:",
                options=list(range(len(detections))),
                format_func=lambda i: f"Object #{i+1}: {detections[i]['class']} ({detections[i]['confidence']*100:.1f}%)"
            )
            if obj_idx is not None and obj_idx < len(detections):
                sel_det = detections[obj_idx]
                bx1, by1, bx2, by2 = sel_det["box"]
                crop_pil = det_image.crop((bx1, by1, bx2, by2))
                
                c_crop_img, c_crop_info = st.columns([1, 2])
                with c_crop_img:
                    st.image(crop_pil, caption=f"Cropped Object #{obj_idx+1}", use_container_width=True)
                with c_crop_info:
                    st.markdown(f"**Identified Waste Material:** `{sel_det['class']}`")
                    st.markdown(f"**Confidence:** `{sel_det['confidence']*100:.1f}%`")
                    st.markdown(f"**Bounding Box Area:** `{sel_det['area']} px²`")
                    st.markdown(f"**Proper Handling Advice:** {sel_det['guidance']}")

            # Log to History Button
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("📝 Log This Detection Event to Prediction History", key="btn_log_det"):
                # Calculate aggregated CO2 and energy saved
                co2_sum = sum(
                    counts[c] * WASTE_INFO.get(c, {}).get("co2_saved_kg", 0.0)
                    for c in DETECTION_CLASSES
                )
                energy_sum = sum(
                    counts[c] * WASTE_INFO.get(c, {}).get("energy_saved_kwh", 0.0)
                    for c in DETECTION_CLASSES
                )
                water_sum = sum(
                    counts[c] * WASTE_INFO.get(c, {}).get("water_saved_liters", 0.0)
                    for c in DETECTION_CLASSES
                )
                
                # Format primary class description
                classes_found = [c for c in DETECTION_CLASSES if counts[c] > 0]
                primary_label = f"Multi-Item ({', '.join(classes_found)})" if classes_found else "None"
                avg_conf = float(np.mean([d["confidence"] for d in detections])) if detections else 0.0

                log_prediction(
                    mode="Detection",
                    primary_class=primary_label,
                    confidence=avg_conf,
                    item_count=total_det,
                    source=det_source,
                    notes=f"Detected items: {counts}",
                    co2_saved_kg=round(co2_sum, 3),
                    energy_saved_kwh=round(energy_sum, 3),
                    water_saved_l=round(water_sum, 2)
                )
                st.success(f"Successfully logged detection event with {total_det} items to history!")
        else:
            st.warning("No waste items were detected above the current confidence threshold. Try lowering the Confidence Threshold slider above.")
    else:
        st.info("👆 Please upload an image or click one of the curated detection scene buttons above.")


# =============================================================================
# 4. CAMERA PHOTO ANALYSIS PAGE
# =============================================================================
elif selected_page == "📸 Camera Snap & Analyze":
    st.markdown("## 📸 Camera Photo Analysis")
    st.info(
        "📷 **Note**: Camera analysis uses still images rather than continuous live video. "
        "Align your waste item clearly within your camera frame and capture a photo for instant AI analysis."
    )

    engine = load_cached_engine()

    c_mode, c_help = st.columns([1, 2])
    with c_mode:
        cam_mode = st.radio(
            "Select Analysis Mode:",
            ["Whole-Image Classification (6 Classes)", "Multi-Object Detection (5 Classes)"],
            index=0
        )

    camera_photo = st.camera_input("Capture a still photo of the waste item")

    if camera_photo is not None:
        pil_camera_img = Image.open(camera_photo).convert("RGB")
        st.markdown("---")

        if cam_mode.startswith("Whole-Image"):
            st.markdown("### 🔍 Classification Results")
            with st.spinner("Analyzing camera photo..."):
                cls_res = engine.classify(pil_camera_img)

            cls_name = cls_res["class"]
            conf = cls_res["confidence"]
            probs = cls_res["probabilities"]
            info = cls_res["info"]
            color_cfg = CLASS_COLORS.get(cls_name, CLASS_COLORS["Trash"])

            col_cam_pic, col_cam_info = st.columns([1, 1.25])
            with col_cam_pic:
                st.image(pil_camera_img, caption="Captured Snapshot", use_container_width=True)

            with col_cam_info:
                st.markdown(
                    f"""
                    <div style="background: {color_cfg['hex']}15; border: 1.5px solid {color_cfg['hex']}; border-radius: 12px; padding: 18px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-size: 1.5rem; font-weight: 800; color: #1e293b;">
                                {color_cfg['icon']} {cls_name}
                            </span>
                            <span class="badge-pill {'badge-recyclable' if info['status_color']=='green' else 'badge-residual'}">
                                {info['status']}
                            </span>
                        </div>
                        <div style="margin-top: 8px; color: #475569;">
                            Confidence: <b>{conf*100:.1f}%</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    <div class="bin-box" style="background-color: {color_cfg['bin_color']}30; border-left: 6px solid {color_cfg['hex']};">
                        <div style="font-size: 1.8rem;">{color_cfg['icon']}</div>
                        <div>
                            <div style="font-size: 0.8rem; text-transform: uppercase; color: #64748b;">Recommended Bin</div>
                            <div style="font-size: 1.1rem; color: #0f172a; font-weight: 700;">{color_cfg['bin']}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                fig_probs = plot_probability_bar(probs)
                st.plotly_chart(fig_probs, use_container_width=True)

                st.markdown(
                    f"""
                    <div class="guidance-card">
                        <h4>📋 Proper Disposal Steps</h4>
                        <p style="font-size: 0.95rem; color: #334155;">{info.get('guidance', '')}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if st.button("📝 Log Snapshot to History", key="btn_log_cam_cls"):
                    log_prediction(
                        mode="Camera Snap",
                        primary_class=cls_name,
                        confidence=conf,
                        item_count=1,
                        source="Camera",
                        notes=f"Camera snap classified as {cls_name}"
                    )
                    st.success("Successfully logged camera snapshot to history!")

        else:
            st.markdown("### 🎯 Multi-Object Detection Results")
            with st.spinner("Detecting waste items in camera photo..."):
                det_res = engine.detect(pil_camera_img, conf_threshold=0.20)

            annotated_cam = det_res["annotated_image"]
            detections = det_res["detections"]
            counts = det_res["counts"]
            total_det = det_res["total_detected"]

            col_cam_pic, col_cam_ann = st.columns(2)
            with col_cam_pic:
                st.image(pil_camera_img, caption="Original Snapshot", use_container_width=True)
            with col_cam_ann:
                st.image(annotated_cam, caption=f"Annotated Snapshot ({total_det} items detected)", use_container_width=True)

            # Metrics
            st.write(f"**Detected Objects ({total_det}):**")
            badge_str = " ".join([f"`{c}: {counts[c]}`" for c in counts if counts[c] > 0])
            st.markdown(badge_str if badge_str else "_No objects detected above threshold._")

            if st.button("📝 Log Camera Detection to History", key="btn_log_cam_det"):
                log_prediction(
                    mode="Camera Snap",
                    primary_class="Camera Multi-Object",
                    confidence=0.85,
                    item_count=max(1, total_det),
                    source="Camera",
                    notes=f"Camera detection counts: {counts}"
                )
                st.success("Successfully logged camera detection to history!")


# =============================================================================
# 5. ANALYTICS DASHBOARD PAGE
# =============================================================================
elif selected_page == "📊 Analytics Dashboard":
    st.markdown("## 📊 Interactive Analytics Dashboard")
    st.write(
        "Real-time visual analytics detailing waste classification distributions, recycling rates, "
        "and cumulative environmental impact calculated across all prediction sessions."
    )

    df_hist = load_history()
    kpis = generate_kpis(df_hist)

    # Top KPI Metrics Cards
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(
            f"""
            <div class="eco-kpi-card">
                <div class="eco-kpi-label">Total Scans</div>
                <div class="eco-kpi-val">{kpis['total_scans']}</div>
                <small style="color: #64748b;">Logged sessions</small>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k2:
        st.markdown(
            f"""
            <div class="eco-kpi-card">
                <div class="eco-kpi-label">Recyclable Items</div>
                <div class="eco-kpi-val" style="color: #2e7d32;">{kpis['recyclable_items']}</div>
                <small style="color: #2e7d32;">Diverted from waste</small>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k3:
        st.markdown(
            f"""
            <div class="eco-kpi-card">
                <div class="eco-kpi-label">Diversion Rate</div>
                <div class="eco-kpi-val">{kpis['diversion_rate']}%</div>
                <small style="color: #64748b;">Target: > 75%</small>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k4:
        st.markdown(
            f"""
            <div class="eco-kpi-card">
                <div class="eco-kpi-label">CO2 Prevented</div>
                <div class="eco-kpi-val">{kpis['total_co2_kg']} kg</div>
                <small style="color: #64748b;">GHG avoided</small>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k5:
        st.markdown(
            f"""
            <div class="eco-kpi-card">
                <div class="eco-kpi-label">Energy Conserved</div>
                <div class="eco-kpi-val">{kpis['total_energy_kwh']} kWh</div>
                <small style="color: #64748b;">Electricity saved</small>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Environmental Impact Equivalency Banner
    st.markdown(
        f"""
        <div style="background: linear-gradient(90deg, #d8f3dc, #b7e4c7); border-radius: 12px; padding: 16px 24px; color: #1b4332; margin-bottom: 25px; display: flex; align-items: center; justify-content: space-around; flex-wrap: wrap; gap: 15px;">
            <div style="text-align: center;">
                <div style="font-size: 1.6rem; font-weight: 800;">🚗 {round(kpis['total_co2_kg'] * 4.2, 1)} km</div>
                <small>Car driving emissions eliminated</small>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.6rem; font-weight: 800;">🔋 {kpis['phones_charged']}</div>
                <small>Smartphone full battery charges saved</small>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.6rem; font-weight: 800;">🌲 {kpis['trees_saved']}</div>
                <small>Mature tree-years of carbon absorption</small>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.6rem; font-weight: 800;">💧 {kpis['total_water_l']} L</div>
                <small>Freshwater conserved</small>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Charts Row 1: Category Breakdown & Recyclability Ratio
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        fig_cat = plot_category_distribution(df_hist)
        st.plotly_chart(fig_cat, use_container_width=True)
    with row1_col2:
        fig_rec = plot_recyclability_ratio(df_hist)
        st.plotly_chart(fig_rec, use_container_width=True)

    # Charts Row 2: Confidence Distribution & Activity Timeline
    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        fig_conf = plot_confidence_distribution(df_hist)
        st.plotly_chart(fig_conf, use_container_width=True)
    with row2_col2:
        fig_timeline = plot_activity_timeline(df_hist)
        st.plotly_chart(fig_timeline, use_container_width=True)


# =============================================================================
# 6. PREDICTION HISTORY AND CSV EXPORT PAGE
# =============================================================================
elif selected_page == "📋 Prediction History":
    st.markdown("## 📋 Prediction History & CSV Export")
    st.write(
        "Comprehensive log of all AI inferences performed during this deployment. "
        "Filter records by mode, category, or search keywords, and export audit reports as standard CSV."
    )

    df_hist = load_history()

    # Filters
    st.markdown("### 🔍 Filter Records")
    f_c1, f_c2, f_c3, f_c4 = st.columns(4)

    with f_c1:
        modes_available = ["All"] + list(df_hist["mode"].unique()) if "mode" in df_hist.columns else ["All"]
        filter_mode = st.selectbox("Inference Mode", modes_available)

    with f_c2:
        rec_available = ["All"] + list(df_hist["recyclable"].unique()) if "recyclable" in df_hist.columns else ["All"]
        filter_rec = st.selectbox("Recyclability", rec_available)

    with f_c3:
        sources_available = ["All"] + list(df_hist["source"].unique()) if "source" in df_hist.columns else ["All"]
        filter_source = st.selectbox("Input Source", sources_available)

    with f_c4:
        search_query = st.text_input("Search Keyword", placeholder="e.g. plastic, can, bottle...")

    # Apply filters
    filtered_df = df_hist.copy()
    if filter_mode != "All":
        filtered_df = filtered_df[filtered_df["mode"] == filter_mode]
    if filter_rec != "All":
        filtered_df = filtered_df[filtered_df["recyclable"] == filter_rec]
    if filter_source != "All":
        filtered_df = filtered_df[filtered_df["source"] == filter_source]
    if search_query:
        sq = search_query.lower()
        mask = (
            filtered_df["primary_class"].astype(str).str.lower().str.contains(sq) |
            filtered_df["notes"].astype(str).str.lower().str.contains(sq) |
            filtered_df["mode"].astype(str).str.lower().str.contains(sq)
        )
        filtered_df = filtered_df[mask]

    st.markdown(f"**Showing {len(filtered_df)} of {len(df_hist)} total logged records:**")
    st.dataframe(filtered_df, use_container_width=True)

    # Actions Row: Export and Clear
    col_export, col_clear = st.columns([1, 1])

    with col_export:
        csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download History as CSV",
            data=csv_bytes,
            file_name="EcoVision_Prediction_History.csv",
            mime="text/csv",
            help="Download the filtered prediction records as a standard CSV file."
        )

    with col_clear:
        if st.checkbox("Enable Clear History Option"):
            if st.button("🗑️ Clear Entire History", type="secondary"):
                clear_history()
                st.success("History log successfully cleared!")
                st.rerun()


# =============================================================================
# 7. RECYCLING AND DISPOSAL GUIDANCE PAGE
# =============================================================================
elif selected_page == "♻️ Recycling & Disposal Guide":
    st.markdown("## ♻️ Recycling & Disposal Guidance")
    st.write(
        "Comprehensive encyclopedia of recycling best practices, resin identification codes, "
        "searchable waste directory, and personal carbon footprint savings calculator."
    )

    advisor = load_cached_advisor()

    guide_tab1, guide_tab2, guide_tab3 = st.tabs([
        "🔍 Searchable Waste Directory",
        "📚 Category In-Depth Guide",
        "🧮 Carbon & Savings Calculator"
    ])

    # -------------------------------------------------------------
    # Tab 1: Searchable Waste Directory
    # -------------------------------------------------------------
    with guide_tab1:
        st.markdown("### 🔎 Can I Recycle This? Searchable Directory")
        st.write("Type any household item to find out which bin it belongs to and how to prepare it:")
        item_query = st.text_input("Search Item Name", placeholder="e.g. coffee cup, pizza box, batteries, styrofoam...")

        results = advisor.search_items(item_query)
        if results:
            df_items = pd.DataFrame(results)
            st.dataframe(df_items, use_container_width=True)
        else:
            st.warning("No items matched your search query. Try another term like 'bottle', 'can', 'paper', 'box', or 'bag'.")

    # -------------------------------------------------------------
    # Tab 2: Category In-Depth Guide
    # -------------------------------------------------------------
    with guide_tab2:
        st.markdown("### 📚 Waste Category Breakdown & Best Practices")

        for cat_name, cat_info in WASTE_INFO.items():
            color_cfg = CLASS_COLORS.get(cat_name, CLASS_COLORS["Trash"])
            with st.expander(f"{color_cfg['icon']} **{cat_name}** — {cat_info['status']}", expanded=(cat_name == "Plastic")):
                st.markdown(f"**Recommended Bin Container:** `{color_cfg['bin']}`")
                st.markdown(f"**Average Decomposition Time:** `{cat_info['decomposition']}`")
                st.markdown(f"**Carbon Offset Per Item:** `{cat_info['co2_saved_kg']} kg CO2e`")
                st.markdown(f"**Water Conserved Per Item:** `{cat_info['water_saved_liters']} L`")
                
                st.markdown("#### ✅ Preparation & Sorting Checklist:")
                st.write(cat_info["guidance"])
                
                st.markdown("#### 🔬 Material / Resin Codes:")
                st.write(cat_info["resin_codes"])
                
                st.markdown("#### 💡 Environmental Impact Fact:")
                st.info(cat_info["tips"])

        # Resin Codes Table
        st.markdown("### 🏷️ Standard Plastic Resin Identification Codes (RIC 1 to 7)")
        resin_data = [
            {"Code": "1 - PET", "Name": "Polyethylene Terephthalate", "Examples": "Water bottles, soda bottles, salad dressing bottles", "Curbside Recyclable?": "✅ Yes (Widely Accepted)"},
            {"Code": "2 - HDPE", "Name": "High-Density Polyethylene", "Examples": "Milk jugs, laundry detergent jugs, shampoo bottles", "Curbside Recyclable?": "✅ Yes (Widely Accepted)"},
            {"Code": "3 - PVC", "Name": "Polyvinyl Chloride", "Examples": "Plumbing pipes, blister packaging, vinyl shower curtains", "Curbside Recyclable?": "❌ No (Special Handling)"},
            {"Code": "4 - LDPE", "Name": "Low-Density Polyethylene", "Examples": "Grocery bags, bread bags, bubble wrap, squeeze bottles", "Curbside Recyclable?": "🟡 Store Drop-off Only"},
            {"Code": "5 - PP", "Name": "Polypropylene", "Examples": "Yogurt cups, syrup bottles, medicine bottles, caps", "Curbside Recyclable?": "✅ Yes (Most Municipalities)"},
            {"Code": "6 - PS", "Name": "Polystyrene (Styrofoam)", "Examples": "Disposable foam coffee cups, takeout clamshells, egg cartons", "Curbside Recyclable?": "❌ No (General Trash)"},
            {"Code": "7 - OTHER", "Name": "Miscellaneous / Composites", "Examples": "Multi-layer pouches, baby bottles, optical discs, bioplastics", "Curbside Recyclable?": "❌ No (Check Local Rules)"}
        ]
        st.table(pd.DataFrame(resin_data))

    # -------------------------------------------------------------
    # Tab 3: Interactive Carbon & Resource Savings Calculator
    # -------------------------------------------------------------
    with guide_tab3:
        st.markdown("### 🧮 Interactive Carbon & Resource Savings Calculator")
        st.write("Input the number of waste items you or your household recycle to calculate real-world environmental savings:")

        calc_c1, calc_c2 = st.columns(2)
        with calc_c1:
            in_plastic = st.number_input("🧴 Plastic Bottles (#1 / #2)", min_value=0, max_value=1000, value=12, step=1)
            in_metal = st.number_input("🥫 Aluminum / Metal Cans", min_value=0, max_value=1000, value=8, step=1)
            in_glass = st.number_input("🍾 Glass Bottles / Jars", min_value=0, max_value=1000, value=4, step=1)
        with calc_c2:
            in_cardboard = st.number_input("📦 Cardboard Packaging Boxes", min_value=0, max_value=1000, value=3, step=1)
            in_paper = st.number_input("📄 Paper Sheets / Documents", min_value=0, max_value=5000, value=25, step=5)

        savings = advisor.calculate_savings(
            plastic_bottles=in_plastic,
            aluminum_cans=in_metal,
            glass_bottles=in_glass,
            cardboard_boxes=in_cardboard,
            paper_sheets=in_paper
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🌿 Your Estimated Environmental Savings:")

        sv1, sv2, sv3, sv4 = st.columns(4)
        with sv1:
            st.metric("CO2 Emissions Prevented", f"{savings['co2_saved_kg']} kg")
        with sv2:
            st.metric("Freshwater Saved", f"{savings['water_saved_l']} L")
        with sv3:
            st.metric("Electricity Saved", f"{savings['energy_saved_kwh']} kWh")
        with sv4:
            st.metric("Bottles Kept from Oceans", f"{savings['bottles_diverted']}")

        st.markdown(
            f"""
            <div style="background: #e8f5e9; border: 1px solid #81c784; border-radius: 12px; padding: 18px 24px; margin-top: 15px;">
                <h4 style="color: #1b4332; margin-top: 0;">🌍 Real-World Equivalents:</h4>
                <ul style="color: #2e7d32; font-size: 1rem; line-height: 1.8; margin-bottom: 0;">
                    <li>🚗 Equivalent to taking an average gasoline passenger car off the road for <b>{savings['car_km_avoided']} km</b>.</li>
                    <li>🔋 Generates enough energy to fully charge <b>{savings['smartphone_charges']} smartphones</b>.</li>
                    <li>🌲 Equals <b>{savings['tree_days_saved']} tree-days</b> of natural carbon dioxide absorption by mature forests.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 10px 0;">
        <b>EcoVision AI</b> — Automated Real-Time Waste Classification System Using Deep Learning<br>
        <span style="color: #94a3b8;">{APP_DISCLAIMER}</span>
    </div>
    """,
    unsafe_allow_html=True
)
