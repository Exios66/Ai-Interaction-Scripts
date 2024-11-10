import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import logging
import json
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   filename='analysis_log.txt')

# Download NLTK resources
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    from nltk.corpus import stopwords, wordnet
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
except Exception as e:
    logging.error(f"Error downloading NLTK resources: {str(e)}")

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

class DataAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Data Analysis Suite")
        self.root.state('zoomed')  # Start maximized

        # Initialize data structures
        self.initialize_data_structures()
        
        # Create main container
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create GUI elements
        self.create_menu()
        self.create_widgets()
        self.create_status_bar()
        
        # Load config if exists
        self.load_config()

    def initialize_data_structures(self):
        """Initialize all data-related attributes"""
        self.quant_data = None
        self.qual_data = None
        self.quant_data_scaled = None
        self.tfidf_matrix = None
        self.kmeans = None
        self.dbscan = None
        self.qual_pca = None
        self.combined_data = None
        self.cluster_metrics = {}
        self.analysis_results = {}
        self.config = {
            'last_quant_path': '',
            'last_qual_path': '',
            'default_clusters': 5,
            'min_df': 5,
            'max_df': 0.85
        }

    def create_menu(self):
        """Create application menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Quantitative Data", command=self.load_quant_data)
        file_menu.add_command(label="Load Qualitative Data", command=self.load_qual_data)
        file_menu.add_separator()
        file_menu.add_command(label="Export Results", command=self.export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Preprocess Data", command=self.preprocess_data)
        analysis_menu.add_command(label="Cluster Analysis", command=self.show_clustering_options)
        analysis_menu.add_command(label="Statistical Analysis", command=self.statistical_analysis)

        # Settings menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="Clustering Parameters", command=self.show_cluster_settings)
        settings_menu.add_command(label="Visualization Options", command=self.show_viz_settings)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Documentation", command=self.show_documentation)
        help_menu.add_command(label="About", command=self.show_about)

    def create_widgets(self):
        """Create and arrange all GUI widgets"""
        # Create notebook for tabbed interface
        self.notebook = ttk.Notebook(self.main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Data tab
        self.data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_frame, text="Data Management")
        
        # Analysis tab
        self.analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_frame, text="Analysis")
        
        # Visualization tab
        self.viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_frame, text="Visualization")
        
        # Results tab
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Results")

        self.setup_data_frame()
        self.setup_analysis_frame()
        self.setup_viz_frame()
        self.setup_results_frame()

    def setup_data_frame(self):
        """Setup the data management tab"""
        # File loading section
        load_frame = ttk.LabelFrame(self.data_frame, text="Data Loading")
        load_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(load_frame, text="Load Quantitative Data", command=self.load_quant_data).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(load_frame, text="Load Qualitative Data", command=self.load_qual_data).pack(side=tk.LEFT, padx=5, pady=5)

        # Data preview section
        preview_frame = ttk.LabelFrame(self.data_frame, text="Data Preview")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.preview_text = tk.Text(preview_frame, height=20)
        self.preview_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def setup_analysis_frame(self):
        """Setup the analysis tab"""
        # Analysis options
        options_frame = ttk.LabelFrame(self.analysis_frame, text="Analysis Options")
        options_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(options_frame, text="Preprocess Data", command=self.preprocess_data).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(options_frame, text="Cluster Analysis", command=self.show_clustering_options).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(options_frame, text="Statistical Analysis", command=self.statistical_analysis).pack(side=tk.LEFT, padx=5, pady=5)

        # Analysis output
        output_frame = ttk.LabelFrame(self.analysis_frame, text="Analysis Output")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.analysis_text = tk.Text(output_frame, height=20)
        self.analysis_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def setup_viz_frame(self):
        """Setup the visualization tab"""
        # Visualization controls
        controls_frame = ttk.LabelFrame(self.viz_frame, text="Visualization Controls")
        controls_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(controls_frame, text="Plot Clusters", command=self.visualize_results).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(controls_frame, text="Export Plot", command=self.export_plot).pack(side=tk.LEFT, padx=5, pady=5)

        # Plot area
        self.plot_frame = ttk.Frame(self.viz_frame)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def setup_results_frame(self):
        """Setup the results tab"""
        # Results controls
        controls_frame = ttk.LabelFrame(self.results_frame, text="Results Options")
        controls_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(controls_frame, text="Export Results", command=self.export_results).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(controls_frame, text="Clear Results", command=self.clear_results).pack(side=tk.LEFT, padx=5, pady=5)

        # Results display
        self.results_text = tk.Text(self.results_frame, height=20)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_status_bar(self):
        """Create status bar for displaying messages"""
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.update_status("Ready")

    def update_status(self, message):
        """Update status bar message"""
        self.status_var.set(message)
        self.root.update_idletasks()

    def load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists('analysis_config.json'):
                with open('analysis_config.json', 'r') as f:
                    self.config.update(json.load(f))
        except Exception as e:
            logging.error(f"Error loading config: {str(e)}")

    def save_config(self):
        """Save configuration to file"""
        try:
            with open('analysis_config.json', 'w') as f:
                json.dump(self.config, f)
        except Exception as e:
            logging.error(f"Error saving config: {str(e)}")

    def export_results(self):
        """Export analysis results to file"""
        if not self.analysis_results:
            messagebox.showwarning("Warning", "No results to export.")
            return

        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if filename:
                with open(filename, 'w') as f:
                    json.dump(self.analysis_results, f, indent=4)
                self.update_status(f"Results exported to {filename}")
        except Exception as e:
            logging.error(f"Error exporting results: {str(e)}")
            messagebox.showerror("Error", f"Failed to export results: {str(e)}")

    def show_clustering_options(self):
        """Show dialog for clustering options"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Clustering Options")
        dialog.geometry("300x200")
        
        ttk.Label(dialog, text="Number of clusters:").pack(pady=5)
        n_clusters = ttk.Entry(dialog)
        n_clusters.insert(0, str(self.config['default_clusters']))
        n_clusters.pack(pady=5)
        
        ttk.Button(dialog, text="K-Means", 
                  command=lambda: self.cluster_data('kmeans', int(n_clusters.get()))).pack(pady=5)
        ttk.Button(dialog, text="DBSCAN", 
                  command=lambda: self.cluster_data('dbscan')).pack(pady=5)
        ttk.Button(dialog, text="Hierarchical", 
                  command=lambda: self.cluster_data('hierarchical', int(n_clusters.get()))).pack(pady=5)

    def cluster_data(self, method='kmeans', n_clusters=5):
        """Perform clustering analysis"""
        if self.qual_data is None or 'processed_text' not in self.qual_data.columns:
            messagebox.showwarning("Warning", "Please preprocess the data before clustering.")
            return

        try:
            # TF-IDF Vectorization
            tfidf_vectorizer = TfidfVectorizer(
                max_df=self.config['max_df'], 
                min_df=self.config['min_df'],
                stop_words='english'
            )
            self.tfidf_matrix = tfidf_vectorizer.fit_transform(self.qual_data['processed_text'])

            if method == 'kmeans':
                self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                self.qual_data['cluster'] = self.kmeans.fit_predict(self.tfidf_matrix)
                
                # Calculate clustering metrics
                self.cluster_metrics['silhouette'] = silhouette_score(self.tfidf_matrix, self.qual_data['cluster'])
                self.cluster_metrics['calinski'] = calinski_harabasz_score(self.tfidf_matrix.toarray(), self.qual_data['cluster'])
                
            elif method == 'dbscan':
                self.dbscan = DBSCAN(eps=0.5, min_samples=5)
                self.qual_data['cluster'] = self.dbscan.fit_predict(self.tfidf_matrix.toarray())
                
            elif method == 'hierarchical':
                linked = linkage(self.tfidf_matrix.toarray(), method='ward')
                self.qual_data['cluster'] = fcluster(linked, n_clusters, criterion='maxclust')

            self.update_status(f"Clustering completed using {method}")
            self.show_clustering_results()
            
        except Exception as e:
            logging.error(f"Error in clustering: {str(e)}")
            messagebox.showerror("Error", f"Clustering failed: {str(e)}")

    def show_clustering_results(self):
        """Display clustering results"""
        results = f"Clustering Results:\n"
        results += f"Number of clusters: {len(self.qual_data['cluster'].unique())}\n"
        results += f"Cluster sizes:\n{self.qual_data['cluster'].value_counts()}\n\n"
        
        if self.cluster_metrics:
            results += f"Clustering Metrics:\n"
            results += f"Silhouette Score: {self.cluster_metrics['silhouette']:.3f}\n"
            results += f"Calinski-Harabasz Score: {self.cluster_metrics['calinski']:.3f}\n"

        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, results)

    def preprocess_text(self, text):
        """Enhanced text preprocessing"""
        try:
            # Convert to string and lowercase
            text = str(text).lower()
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords and non-alphabetic tokens
            tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
            
            # Lemmatize
            tokens = [lemmatizer.lemmatize(t) for t in tokens]
            
            return " ".join(tokens)
        except Exception as e:
            logging.error(f"Error in text preprocessing: {str(e)}")
            return ""

    def visualize_results(self):
        """Enhanced visualization of results"""
        if self.tfidf_matrix is None or 'cluster' not in self.qual_data.columns:
            messagebox.showwarning("Warning", "Please perform clustering before visualization.")
            return

        try:
            # Create figure with subplots
            fig = plt.figure(figsize=(15, 10))
            
            # PCA visualization
            pca = PCA(n_components=2)
            self.qual_pca = pca.fit_transform(self.tfidf_matrix.toarray())
            
            # Scatter plot
            ax1 = fig.add_subplot(121)
            scatter = ax1.scatter(self.qual_pca[:, 0], self.qual_pca[:, 1], 
                                c=self.qual_data['cluster'], cmap='Set2')
            ax1.set_title("PCA of Qualitative Response Clusters")
            ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
            ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
            plt.colorbar(scatter, ax=ax1, label='Cluster')

            # Hierarchical clustering dendrogram
            ax2 = fig.add_subplot(122)
            linked = linkage(self.tfidf_matrix.toarray(), method='ward')
            dendrogram(linked, ax=ax2, orientation='right', leaf_font_size=8)
            ax2.set_title("Hierarchical Clustering Dendrogram")

            plt.tight_layout()
            
            # Display in GUI
            self.display_plot(fig)
            
        except Exception as e:
            logging.error(f"Error in visualization: {str(e)}")
            messagebox.showerror("Error", f"Visualization failed: {str(e)}")

    def display_plot(self, fig):
        """Display plot in GUI with navigation toolbar"""
        # Clear existing plot frame
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        # Create canvas and toolbar
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        
        toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
        toolbar.update()
        
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def statistical_analysis(self):
        """Perform comprehensive statistical analysis"""
        if self.quant_data is None or 'cluster' not in self.qual_data.columns:
            messagebox.showwarning("Warning", "Please ensure data is clustered before analysis.")
            return

        try:
            # Merge data
            self.combined_data = pd.merge(self.quant_data, 
                                        self.qual_data[['ID', 'cluster']], 
                                        on='ID')
            
            # Store results
            self.analysis_results['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.analysis_results['descriptive'] = {}
            self.analysis_results['inferential'] = {}
            
            # Descriptive statistics
            desc_stats = self.combined_data.groupby('cluster').describe()
            self.analysis_results['descriptive']['cluster_stats'] = desc_stats.to_dict()
            
            # ANOVA
            numeric_cols = self.combined_data.select_dtypes(include=[np.number]).columns
            anova_results = {}
            for col in numeric_cols:
                if col != 'cluster':
                    f_stat, p_val = stats.f_oneway(*[group[col].values 
                                                   for name, group in self.combined_data.groupby('cluster')])
                    anova_results[col] = {'f_statistic': f_stat, 'p_value': p_val}
            
            self.analysis_results['inferential']['anova'] = anova_results
            
            # Display results
            self.show_statistical_results()
            
        except Exception as e:
            logging.error(f"Error in statistical analysis: {str(e)}")
            messagebox.showerror("Error", f"Statistical analysis failed: {str(e)}")

    def show_statistical_results(self):
        """Display statistical analysis results"""
        results = "Statistical Analysis Results\n"
        results += "=" * 50 + "\n\n"
        
        # Descriptive statistics
        results += "Descriptive Statistics by Cluster:\n"
        results += "-" * 30 + "\n"
        for cluster in sorted(self.combined_data['cluster'].unique()):
            cluster_data = self.combined_data[self.combined_data['cluster'] == cluster]
            results += f"\nCluster {cluster} (n={len(cluster_data)}):\n"
            results += cluster_data.describe().to_string() + "\n"
        
        # ANOVA results
        results += "\nANOVA Results:\n"
        results += "-" * 30 + "\n"
        for var, stats in self.analysis_results['inferential']['anova'].items():
            results += f"\n{var}:\n"
            results += f"F-statistic: {stats['f_statistic']:.3f}\n"
            results += f"p-value: {stats['p_value']:.3f}\n"
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, results)

if __name__ == "__main__":
    root = tk.Tk()
    app = DataAnalysisApp(root)
    root.mainloop()