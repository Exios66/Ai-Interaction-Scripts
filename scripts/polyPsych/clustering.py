import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Initialize stop words
stop_words = set(stopwords.words('english'))

class DataAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Quantitative and Qualitative Data Analysis")

        # Initialize variables
        self.quant_data = None
        self.qual_data = None
        self.quant_data_scaled = None
        self.tfidf_matrix = None
        self.kmeans = None
        self.qual_pca = None
        self.combined_data = None

        # Create GUI elements
        self.create_widgets()

    def create_widgets(self):
        # Frame for file loading
        self.frame_load = tk.Frame(self.root)
        self.frame_load.pack(pady=10)

        self.btn_load_quant = tk.Button(self.frame_load, text="Load Quantitative Data", command=self.load_quant_data)
        self.btn_load_quant.grid(row=0, column=0, padx=5)

        self.btn_load_qual = tk.Button(self.frame_load, text="Load Qualitative Data", command=self.load_qual_data)
        self.btn_load_qual.grid(row=0, column=1, padx=5)

        # Frame for processing and analysis
        self.frame_process = tk.Frame(self.root)
        self.frame_process.pack(pady=10)

        self.btn_preprocess = tk.Button(self.frame_process, text="Preprocess Data", command=self.preprocess_data)
        self.btn_preprocess.grid(row=0, column=0, padx=5)

        self.btn_cluster = tk.Button(self.frame_process, text="Cluster Qualitative Data", command=self.cluster_data)
        self.btn_cluster.grid(row=0, column=1, padx=5)

        self.btn_visualize = tk.Button(self.frame_process, text="Visualize Results", command=self.visualize_results)
        self.btn_visualize.grid(row=0, column=2, padx=5)

        self.btn_stat_analysis = tk.Button(self.frame_process, text="Statistical Analysis", command=self.statistical_analysis)
        self.btn_stat_analysis.grid(row=0, column=3, padx=5)

        # Frame for displaying outputs
        self.frame_output = tk.Frame(self.root)
        self.frame_output.pack(pady=10)

        self.txt_output = tk.Text(self.frame_output, width=100, height=20)
        self.txt_output.pack()

    def load_quant_data(self):
        file_path = filedialog.askopenfilename(title="Select Quantitative Data CSV",
                                               filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.quant_data = pd.read_csv(file_path)
            self.txt_output.insert(tk.END, "Loaded Quantitative Data:\n")
            self.txt_output.insert(tk.END, str(self.quant_data.head()) + "\n\n")
            messagebox.showinfo("Success", "Quantitative data loaded successfully.")

    def load_qual_data(self):
        file_path = filedialog.askopenfilename(title="Select Qualitative Data CSV",
                                               filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.qual_data = pd.read_csv(file_path)
            self.txt_output.insert(tk.END, "Loaded Qualitative Data:\n")
            self.txt_output.insert(tk.END, str(self.qual_data.head()) + "\n\n")
            messagebox.showinfo("Success", "Qualitative data loaded successfully.")

    def preprocess_data(self):
        if self.quant_data is not None and self.qual_data is not None:
            # Standardize quantitative data
            scaler = StandardScaler()
            self.quant_data_scaled = scaler.fit_transform(self.quant_data.iloc[:, 1:])

            # Preprocess qualitative data
            self.qual_data['processed_text'] = self.qual_data['response'].apply(self.preprocess_text)
            self.txt_output.insert(tk.END, "Data preprocessing completed.\n\n")
            messagebox.showinfo("Success", "Data preprocessing completed.")
        else:
            messagebox.showwarning("Warning", "Please load both quantitative and qualitative data.")

    @staticmethod
    def preprocess_text(text):
        tokens = word_tokenize(str(text).lower())
        tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
        return " ".join(tokens)

    def cluster_data(self):
        if self.qual_data is not None and 'processed_text' in self.qual_data.columns:
            # TF-IDF Vectorization
            tfidf_vectorizer = TfidfVectorizer(max_df=0.85, min_df=5)
            self.tfidf_matrix = tfidf_vectorizer.fit_transform(self.qual_data['processed_text'])

            # K-means clustering
            num_clusters = 5
            self.kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            self.kmeans.fit(self.tfidf_matrix)
            self.qual_data['cluster'] = self.kmeans.labels_
            self.txt_output.insert(tk.END, "Clustering completed. Assigned clusters to qualitative data.\n\n")
            messagebox.showinfo("Success", "Clustering completed.")
        else:
            messagebox.showwarning("Warning", "Please preprocess the data before clustering.")

    def visualize_results(self):
        if self.tfidf_matrix is not None and self.kmeans is not None:
            # Hierarchical clustering dendrogram
            linked = linkage(self.tfidf_matrix.toarray(), method='ward')
            fig1 = plt.figure(figsize=(10, 7))
            dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=False)
            plt.title("Hierarchical Clustering Dendrogram")
            plt.xlabel("Sample Index")
            plt.ylabel("Distance")

            # PCA visualization
            pca = PCA(n_components=2)
            self.qual_pca = pca.fit_transform(self.tfidf_matrix.toarray())
            fig2 = plt.figure(figsize=(10, 6))
            sns.scatterplot(x=self.qual_pca[:, 0], y=self.qual_pca[:, 1], hue=self.qual_data['cluster'], palette="Set2")
            plt.title("PCA of Qualitative Response Clusters")
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            plt.legend(title="Cluster")

            # Display plots in GUI
            self.display_plot(fig1)
            self.display_plot(fig2)
        else:
            messagebox.showwarning("Warning", "Please perform clustering before visualization.")

    def display_plot(self, fig):
        window = tk.Toplevel(self.root)
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack()
        toolbar = canvas.get_tk_widget()
        toolbar.update()

    def statistical_analysis(self):
        if self.quant_data is not None and 'cluster' in self.qual_data.columns:
            # Merge data
            self.combined_data = pd.merge(self.quant_data, self.qual_data[['ID', 'cluster']], on='ID')
            self.combined_data['cluster'] = self.combined_data['cluster'].astype('category')

            # Calculate cluster means
            cluster_means = self.combined_data.groupby('cluster').mean()
            self.txt_output.insert(tk.END, "Mean Quantitative Scores by Cluster:\n")
            self.txt_output.insert(tk.END, str(cluster_means) + "\n\n")

            # ANOVA
            import statsmodels.api as sm
            from statsmodels.formula.api import ols
            model = ols('political_efficacy_score ~ C(cluster)', data=self.combined_data).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            self.txt_output.insert(tk.END, "ANOVA Results for Political Efficacy Score by Cluster:\n")
            self.txt_output.insert(tk.END, str(anova_table) + "\n\n")

            # Boxplot
            fig = plt.figure(figsize=(12, 8))
            sns.boxplot(data=self.combined_data, x='cluster', y='political_efficacy_score')
            plt.title("Political Efficacy Score Distribution by Qualitative Cluster")
            plt.xlabel("Qualitative Cluster")
            plt.ylabel("Political Efficacy Score")

            # Display plot
            self.display_plot(fig)
        else:
            messagebox.showwarning("Warning", "Please ensure data is clustered and merged before analysis.")

if __name__ == "__main__":
    root = tk.Tk()
    app = DataAnalysisApp(root)
    root.mainloop()