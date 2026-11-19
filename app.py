
# ============================================================
# AI-BASED STUDENT PERFORMANCE AND DROPOUT PREDICTION SYSTEM
# SHARDA UNIVERSITY
# ============================================================

# 1. IMPORT LIBRARIES

import os
import sqlite3
from io import BytesIO
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# 2. PROJECT CONFIGURATION
# ============================================================

PROJECT_NAME = (
    "AI-Based Student Performance and Dropout Prediction System"
)

PROJECT_SHORT_NAME = "AI Student Predictor"

UNIVERSITY_NAME = "Sharda University"

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

DROPOUT_MODEL_PATH = os.path.join(
    BASE_PATH, "dropout_model.pkl"
)

PERFORMANCE_MODEL_PATH = os.path.join(
    BASE_PATH, "performance_model.pkl"
)

DB_PATH = os.path.join(
    BASE_PATH, "students.db"
)


# ============================================================
# 3. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title=PROJECT_SHORT_NAME,
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 4. LOAD TRAINED MACHINE LEARNING MODELS
# ============================================================

@st.cache_resource
def load_models():

    dropout_package = joblib.load(
        DROPOUT_MODEL_PATH
    )

    performance_package = joblib.load(
        PERFORMANCE_MODEL_PATH
    )

    dropout_model = dropout_package["model"]

    label_encoder = dropout_package["label_encoder"]

    performance_model = performance_package["model"]

    dropout_features = list(
        dropout_package["feature_names"]
    )

    performance_features = list(
        performance_package["feature_names"]
    )

    return (
        dropout_model,
        label_encoder,
        performance_model,
        dropout_features,
        performance_features
    )


try:

    (
        dropout_model,
        label_encoder,
        performance_model,
        dropout_features,
        performance_features
    ) = load_models()

except Exception as error:

    st.error(f"Unable to load machine learning models: {error}")

    st.stop()


# ============================================================
# 5. DATABASE CONFIGURATION
# ============================================================

def get_connection():

    return sqlite3.connect(
        DB_PATH,
        timeout=30
    )


def initialize_database():

    with get_connection() as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS students (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                student_id TEXT UNIQUE NOT NULL,

                student_name TEXT NOT NULL,

                department TEXT,

                semester INTEGER,

                attendance REAL,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                student_id TEXT,

                student_name TEXT,

                predicted_status TEXT,

                dropout_probability REAL,

                risk_category TEXT,

                predicted_grade REAL,

                performance_category TEXT,

                recommendation TEXT,

                prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP

            )
        """)


initialize_database()


# ============================================================
# 6. STUDENT REGISTRATION FUNCTIONS
# ============================================================

def add_student(
    student_id,
    student_name,
    department,
    semester,
    attendance
):

    try:

        with get_connection() as conn:

            conn.execute("""

                INSERT INTO students

                (
                    student_id,
                    student_name,
                    department,
                    semester,
                    attendance
                )

                VALUES (?, ?, ?, ?, ?)

            """, (
                student_id,
                student_name,
                department,
                semester,
                attendance
            ))

        return True

    except sqlite3.IntegrityError:

        return False


def get_students():

    with get_connection() as conn:

        return pd.read_sql_query("""

            SELECT *

            FROM students

            ORDER BY id DESC

        """, conn)


def get_registered_student(student_id):

    if not student_id:
        return None

    with get_connection() as conn:

        row = conn.execute("""

            SELECT
                student_id,
                student_name,
                department,
                semester,
                attendance

            FROM students

            WHERE student_id = ?

        """, (str(student_id).strip(),)).fetchone()

    if row is None:
        return None

    return {
        "student_id": row[0],
        "student_name": row[1],
        "department": row[2],
        "semester": row[3],
        "attendance": row[4]
    }


# ============================================================
# 7. PREDICTION HISTORY FUNCTIONS
# ============================================================

def save_prediction_history(results):

    records = []

    for _, row in results.iterrows():

        records.append((

            str(row.get("student_id", "")),

            str(row.get("student_name", "")),

            str(row["Predicted Status"]),

            float(row["Dropout Probability (%)"]),

            str(row["Risk Category"]),

            float(row["Predicted Academic Grade"]),

            str(row["Performance Category"]),

            str(row["Recommendation"])

        ))

    if records:

        with get_connection() as conn:

            conn.executemany("""

                INSERT INTO prediction_history

                (
                    student_id,
                    student_name,
                    predicted_status,
                    dropout_probability,
                    risk_category,
                    predicted_grade,
                    performance_category,
                    recommendation
                )

                VALUES (?, ?, ?, ?, ?, ?, ?, ?)

            """, records)

    return len(records)


def get_prediction_history():

    with get_connection() as conn:

        return pd.read_sql_query("""

            SELECT *

            FROM prediction_history

            ORDER BY prediction_date DESC, id DESC

        """, conn)


# ============================================================
# 8. INPUT FEATURE VALIDATION
# ============================================================

def prepare_features(data, required_features):

    missing = [
        feature
        for feature in required_features
        if feature not in data.columns
    ]

    if missing:

        raise ValueError(
            "Missing required features: "
            + ", ".join(missing)
        )

    X = data[required_features].apply(
        pd.to_numeric,
        errors="coerce"
    )

    if X.isna().any().any():

        raise ValueError(
            "Dataset contains missing or non-numeric values."
        )

    if not np.isfinite(X.to_numpy(dtype=float)).all():

        raise ValueError(
            "Dataset contains infinite or invalid values."
        )

    return X


# ============================================================
# 9. DROPOUT PREDICTION
# ============================================================

def classify_risk(probability):

    if probability >= 70:
        return "High"

    elif probability >= 40:
        return "Medium"

    return "Low"


def predict_dropout(data):

    X = prepare_features(
        data,
        dropout_features
    )

    encoded = dropout_model.predict(X)

    predicted_status = label_encoder.inverse_transform(
        np.asarray(encoded, dtype=int)
    )

    probabilities = dropout_model.predict_proba(X)

    encoder_classes = [
        str(value).strip().lower()
        for value in label_encoder.classes_
    ]

    if "dropout" not in encoder_classes:

        raise ValueError(
            "Dropout class not found in label encoder."
        )

    dropout_encoder_index = encoder_classes.index("dropout")

    encoded_dropout = int(
        label_encoder.transform([
            label_encoder.classes_[dropout_encoder_index]
        ])[0]
    )

    model_classes = list(dropout_model.classes_)

    if encoded_dropout not in model_classes:

        raise ValueError(
            "Dropout class not found in trained classifier."
        )

    class_index = model_classes.index(encoded_dropout)

    dropout_probability = (
        probabilities[:, class_index] * 100
    )

    risk_categories = [
        classify_risk(value)
        for value in dropout_probability
    ]

    return (
        predicted_status,
        dropout_probability,
        risk_categories
    )


# ============================================================
# 10. ACADEMIC PERFORMANCE PREDICTION
# ============================================================

def classify_performance(grade):

    if grade >= 15:
        return "Excellent"

    elif grade >= 12:
        return "Good"

    elif grade >= 10:
        return "Average"

    return "Needs Improvement"


def predict_performance(data):

    X = prepare_features(
        data,
        performance_features
    )

    grades = performance_model.predict(X)

    categories = [
        classify_performance(grade)
        for grade in grades
    ]

    return grades, categories


# ============================================================
# 11. STUDENT RECOMMENDATIONS
# ============================================================

def generate_recommendation(risk, performance):

    if risk == "High":

        recommendation = (
            "Priority academic counselling and "
            "individual student support."
        )

    elif risk == "Medium":

        recommendation = (
            "Regular academic monitoring and "
            "additional learning support."
        )

    else:

        recommendation = (
            "Continue regular academic monitoring."
        )

    if performance == "Needs Improvement":

        recommendation += (
            " Provide additional learning resources."
        )

    elif performance == "Average":

        recommendation += (
            " Encourage additional practice and revision."
        )

    return recommendation


# ============================================================
# 12. COMPLETE AI PREDICTION PIPELINE
# ============================================================

def generate_predictions(data):

    if data.empty:

        raise ValueError(
            "The uploaded dataset is empty."
        )

    dropout_status, dropout_probability, risk = (
        predict_dropout(data)
    )

    grades, performance_categories = (
        predict_performance(data)
    )

    results = data.copy()

    results["Predicted Status"] = dropout_status

    results["Dropout Probability (%)"] = np.round(
        dropout_probability, 2
    )

    results["Risk Category"] = risk

    results["Predicted Academic Grade"] = np.round(
        grades, 2
    )

    results["Performance Category"] = (
        performance_categories
    )

    results["Recommendation"] = [

        generate_recommendation(r, p)

        for r, p in zip(
            risk,
            performance_categories
        )

    ]

    return results


# ============================================================
# 13. PDF REPORT GENERATION
# ============================================================

def generate_pdf_report(results):

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle
    )

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=12 * mm,
        leftMargin=12 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm
    )

    styles = getSampleStyleSheet()

    story = []

    story.append(
        Paragraph(
            PROJECT_NAME,
            styles["Title"]
        )
    )

    story.append(
        Paragraph(
            UNIVERSITY_NAME,
            styles["Heading2"]
        )
    )

    story.append(
        Paragraph(
            "Report generated: "
            + datetime.now().strftime("%d-%m-%Y %H:%M"),
            styles["Normal"]
        )
    )

    story.append(Spacer(1, 15))

    summary = [

        ["Metric", "Result"],

        ["Total Students", str(len(results))],

        [
            "High Risk",
            str((results["Risk Category"] == "High").sum())
        ],

        [
            "Average Grade",
            f"{results['Predicted Academic Grade'].mean():.2f}"
        ]

    ]

    summary_table = Table(summary)

    summary_table.setStyle(
        TableStyle([

            ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),

            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),

            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),

            ("PADDING", (0, 0), (-1, -1), 7)

        ])
    )

    story.append(summary_table)

    story.append(Spacer(1, 20))

    report_columns = [

        "student_id",
        "student_name",
        "Predicted Status",
        "Dropout Probability (%)",
        "Risk Category",
        "Predicted Academic Grade",
        "Performance Category"

    ]

    available_columns = [
        column
        for column in report_columns
        if column in results.columns
    ]

    report_data = [available_columns]

    for _, row in results.iterrows():

        report_data.append([
            str(row[column])
            for column in available_columns
        ])

    report_table = Table(
        report_data,
        repeatRows=1
    )

    report_table.setStyle(
        TableStyle([

            ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),

            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),

            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),

            ("FONTSIZE", (0, 0), (-1, -1), 7),

            ("PADDING", (0, 0), (-1, -1), 5)

        ])
    )

    story.append(report_table)

    story.append(Spacer(1, 15))

    story.append(
        Paragraph(
            "Disclaimer: Predictions are estimates for academic "
            "support. They must not replace verified records "
            "or human review.",
            styles["Normal"]
        )
    )

    document.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# 14. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🎓 AI Student Predictor")

page = st.sidebar.radio(

    "Navigation",

    [

        "Dashboard",

        "Student Registration",

        "Student Records",

        "Student Prediction",

        "Prediction History",

        "About Project"

    ]

)

st.sidebar.divider()

st.sidebar.success(
    "AI Models Loaded Successfully"
)

st.sidebar.caption(
    "Sharda University | ENC Academic Project"
)


# ============================================================
# 15. APPLICATION HEADER
# ============================================================

st.title("🎓 " + PROJECT_NAME)

st.write(

    "An intelligent machine learning application for "
    "student academic performance analysis and "
    "dropout risk prediction."

)

st.divider()


# ============================================================
# 16. DASHBOARD
# ============================================================

if page == "Dashboard":

    students = get_students()

    history = get_prediction_history()

    st.header("📊 Student Analytics Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Registered Students",
        len(students)
    )

    col2.metric(
        "Saved Predictions",
        len(history)
    )

    col3.metric(
        "AI Models",
        2
    )

    col4.metric(
        "Prediction Features",
        len(dropout_features)
    )

    st.subheader("System Overview")

    left, right = st.columns(2)

    with left:

        st.info("""
        **Dropout Prediction**

        - Predict academic status.
        - Calculate dropout probability.
        - Identify potential risk levels.
        """)

    with right:

        st.info("""
        **Performance Prediction**

        - Predict academic grades.
        - Classify performance.
        - Generate academic recommendations.
        """)

    if not history.empty:

        st.subheader("Recent Predictions")

        st.dataframe(
            history.head(5),
            use_container_width=True
        )


# ============================================================
# 17. STUDENT REGISTRATION
# ============================================================

elif page == "Student Registration":

    st.header("📝 Student Registration")

    with st.form("registration_form"):

        student_id = st.text_input("Student ID")

        student_name = st.text_input("Student Name")

        department = st.selectbox(
            "Department",
            [
                "ENC",
                "CSE",
                "ECE",
                "EEE",
                "Mechanical Engineering",
                "Civil Engineering",
                "Other"
            ]
        )

        semester = st.selectbox(
            "Semester",
            list(range(1, 9))
        )

        attendance = st.number_input(
            "Attendance Percentage",
            min_value=0.0,
            max_value=100.0,
            value=75.0
        )

        submitted = st.form_submit_button(
            "Register Student"
        )

    if submitted:

        if not student_id.strip() or not student_name.strip():

            st.error(
                "Please enter Student ID and Student Name."
            )

        elif add_student(
            student_id.strip(),
            student_name.strip(),
            department,
            semester,
            attendance
        ):

            st.success(
                "Student registered successfully!"
            )

        else:

            st.error(
                "Student ID already exists."
            )


# ============================================================
# 18. STUDENT RECORDS
# ============================================================

elif page == "Student Records":

    st.header("👨‍🎓 Student Records")

    students = get_students()

    if not students.empty:

        search = st.text_input(
            "Search by Student ID or Name"
        )

        filtered = students.copy()

        if search.strip():

            mask = (

                filtered["student_id"].astype(str).str.contains(
                    search,
                    case=False,
                    na=False
                )

                |

                filtered["student_name"].astype(str).str.contains(
                    search,
                    case=False,
                    na=False
                )

            )

            filtered = filtered[mask]

        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(

            "Download Student Records",

            filtered.to_csv(index=False).encode("utf-8"),

            "student_records.csv",

            "text/csv"

        )

    else:

        st.info(
            "No registered students found."
        )


# ============================================================
# 19. STUDENT PREDICTION
# ============================================================

elif page == "Student Prediction":

    st.header("🔍 AI Student Prediction")

    st.write(
        "Upload student academic data in CSV format."
    )

    registered_students = get_students()

    choices = [""]

    lookup = {}

    for _, student in registered_students.iterrows():

        sid = str(student["student_id"])

        label = (
            f"{sid} — {student['student_name']}"
        )

        choices.append(label)

        lookup[label] = sid

    selected = st.selectbox(
        "Link to Registered Student (optional)",
        choices,
        format_func=lambda x: (
            "Select Student"
            if x == ""
            else x
        )
    )

    uploaded_file = st.file_uploader(
        "Upload Student Dataset",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            data = pd.read_csv(uploaded_file)

            if data.empty:

                st.error(
                    "Uploaded dataset is empty."
                )

                st.stop()

            if len(data) == 1 and selected:

                sid = lookup[selected]

                registered = get_registered_student(sid)

                data["student_id"] = sid

                if registered:

                    data["student_name"] = (
                        registered["student_name"]
                    )

                    data["department"] = (
                        registered["department"]
                    )

                    data["semester"] = (
                        registered["semester"]
                    )

            st.subheader("Uploaded Dataset")

            st.write(
                "Total Students:",
                len(data)
            )

            st.dataframe(
                data.head(),
                use_container_width=True
            )

            if st.button(
                "Generate AI Predictions",
                type="primary"
            ):

                results = generate_predictions(data)

                saved_count = save_prediction_history(
                    results
                )

                st.success(
                    f"{saved_count} predictions saved successfully!"
                )

                st.subheader("Prediction Results")

                display_columns = [

                    column
                    for column in [

                        "student_id",
                        "student_name",
                        "Predicted Status",
                        "Dropout Probability (%)",
                        "Risk Category",
                        "Predicted Academic Grade",
                        "Performance Category",
                        "Recommendation"

                    ]

                    if column in results.columns

                ]

                st.dataframe(
                    results[display_columns],
                    use_container_width=True
                )

                st.subheader("Prediction Analytics")

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Total Students",
                    len(results)
                )

                c2.metric(
                    "High Risk",
                    int(
                        (results["Risk Category"] == "High").sum()
                    )
                )

                c3.metric(
                    "Average Grade",
                    f"{results['Predicted Academic Grade'].mean():.2f}"
                )

                st.subheader("Risk Distribution")

                st.bar_chart(
                    results["Risk Category"].value_counts()
                )

                st.subheader("Performance Distribution")

                st.bar_chart(
                    results["Performance Category"].value_counts()
                )

                st.download_button(

                    "Download CSV Report",

                    results.to_csv(index=False).encode("utf-8"),

                    "AI_Student_Prediction_Report.csv",

                    "text/csv"

                )

                pdf_data = generate_pdf_report(results)

                st.download_button(

                    "Download PDF Report",

                    pdf_data,

                    "AI_Student_Prediction_Report.pdf",

                    "application/pdf"

                )

        except Exception as error:

            st.error(
                f"Prediction error: {error}"
            )


# ============================================================
# 20. PREDICTION HISTORY
# ============================================================

elif page == "Prediction History":

    st.header("🗂️ Prediction History")

    history = get_prediction_history()

    if history.empty:

        st.info(
            "No prediction history available."
        )

    else:

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Total Predictions",
            len(history)
        )

        c2.metric(
            "High Risk",
            int(
                (history["risk_category"] == "High").sum()
            )
        )

        c3.metric(
            "Average Grade",
            f"{history['predicted_grade'].mean():.2f}"
        )

        st.dataframe(
            history,
            use_container_width=True
        )

        st.download_button(

            "Download Prediction History",

            history.to_csv(index=False).encode("utf-8"),

            "prediction_history.csv",

            "text/csv"

        )


# ============================================================
# 21. ABOUT PROJECT
# ============================================================

elif page == "About Project":

    st.header("📚 About Project")

    st.subheader(PROJECT_NAME)

    st.write(
        "This project uses machine learning to analyse "
        "student academic information, predict academic "
        "performance and identify potential dropout risks."
    )

    st.subheader("Project Objectives")

    st.markdown("""
    1. Predict student academic performance.
    2. Identify potential dropout risks.
    3. Analyse student academic information.
    4. Generate academic support recommendations.
    """)

    st.subheader("Technologies Used")

    st.markdown("""
    - Python
    - Pandas and NumPy
    - Scikit-learn
    - Random Forest Classifier
    - Random Forest Regressor
    - Streamlit
    - SQLite
    - Joblib
    """)

    st.info(
        "AI predictions are estimates intended to support "
        "academic monitoring. Final decisions require "
        "verified records and human review."
    )


# ============================================================
# 22. APPLICATION FOOTER
# ============================================================

st.divider()

st.caption(
    f"{PROJECT_NAME} | {UNIVERSITY_NAME} | Academic Project"
)
