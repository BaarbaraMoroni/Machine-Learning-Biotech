# PDF REPORT GENERATOR

from fpdf import FPDF
import os

def generate_report(results):
    print("\n--- Generating PDF Report ---")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Genetic Syndromes Classification Report", ln=True, align="C")
    pdf.ln(10)

    # 1. Introduction
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "1. Introduction", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, 
        "This report presents the results of genetic syndrome classification using KNN with Euclidean and Cosine distance metrics.\n"
        "The objective is to analyze the performance of these metrics using image embeddings, generating visualizations, and evaluating the model with various metrics.")
    pdf.ln(10)

    # 2. Results Table
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "2. Classification Results", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, "Performance Metrics Comparison", ln=True)
    pdf.ln(10)

    pdf.set_font("Arial", "B", 10)
    pdf.cell(50, 10, "Metric", border=1, align="C")
    pdf.cell(50, 10, "Euclidean", border=1, align="C")
    pdf.cell(50, 10, "Cosine", border=1, align="C")
    pdf.ln()

    pdf.set_font("Arial", "", 10)
    pdf.cell(50, 10, "Best K", border=1, align="C")
    pdf.cell(50, 10, str(results['euclidean']['best_k']), border=1, align="C")
    pdf.cell(50, 10, str(results['cosine']['best_k']), border=1, align="C")
    pdf.ln()

    pdf.cell(50, 10, "F1-Score", border=1, align="C")
    pdf.cell(50, 10, f"{results['euclidean']['f1_score']:.2f}", border=1, align="C")
    pdf.cell(50, 10, f"{results['cosine']['f1_score']:.2f}", border=1, align="C")
    pdf.ln()

    pdf.cell(50, 10, "AUC ROC", border=1, align="C")
    pdf.cell(50, 10, f"{results['euclidean']['roc_auc']:.2f}", border=1, align="C")
    pdf.cell(50, 10, f"{results['cosine']['roc_auc']:.2f}", border=1, align="C")
    pdf.ln()

    pdf.cell(50, 10, "Top-k Accuracy", border=1, align="C")
    pdf.cell(50, 10, f"{results['euclidean']['top_k_acc']:.2f}", border=1, align="C")
    pdf.cell(50, 10, f"{results['cosine']['top_k_acc']:.2f}", border=1, align="C")
    pdf.ln(20)

    # 3. Visualizations and Analyses
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "3. Visualizations and Analyses", ln=True)
    pdf.ln(10)

    visuals_path = "../visuals/"
    figures = [
        ("distribuicao_sindromes.png", "Syndrome Distribution:\nThis graph shows the number of images per syndrome. Note the class imbalance, which can impact model performance."),
        ("tsne_visualization.png", "t-SNE Visualization:\nt-SNE projection reduces the dimensionality of embeddings to 2D, revealing possible clusters that aid classification."),
        ("roc_auc_euclidean.png", "ROC AUC Curve - Euclidean:\nThe ROC AUC curve shows the relationship between True Positive Rate and False Positive Rate using Euclidean distance."),
        ("roc_auc_cosine.png", "ROC AUC Curve - Cosine:\nThe ROC AUC curve shows the relationship between True Positive Rate and False Positive Rate using Cosine distance.")
    ]

    for fig, explanation in figures:
        fig_path = visuals_path + fig
        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(0, 10, explanation)
        pdf.ln(5)
        if os.path.exists(fig_path):
            pdf.image(fig_path, x=30, w=150)
            pdf.ln(10)
        else:
            pdf.cell(0, 10, f"Image {fig} not found.", ln=True, align="C")
            pdf.ln(10)

    # 4. Conclusion
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "4. Conclusion", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, 
        "The Cosine distance metric showed better performance compared to Euclidean due to its sensitivity to angular directions in high-dimensional spaces.\n"
        "Future analyses may include other classification algorithms and data augmentation techniques to address class imbalance.")
    
    report_path = "../report/"
    if not os.path.exists(report_path):
        print(f"Folder {report_path} does not exist. Creating now...")
        os.makedirs(report_path)
    else:
        print(f"Folder {report_path} already exists.")

    pdf_output = report_path + "classification_report.pdf"
    print(f"Attempting to save the report at: {pdf_output}")
    try:
        pdf.output(pdf_output)
        print(f"Report successfully generated at: {pdf_output}")
    except Exception as e:
        print(f"Error saving the report: {e}")
