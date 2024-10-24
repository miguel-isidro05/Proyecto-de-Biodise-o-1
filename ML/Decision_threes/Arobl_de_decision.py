from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree

# Cargar el dataset Iris
iris = load_iris()

# X contiene las características, tomando solo las columnas 2 en adelante (pétalo)
X = iris.data[:, 2:]
# y contiene las etiquetas (especies de flores)
y = iris.target

# Crear y entrenar el árbol de decisión con profundidad máxima de 2
tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)

# Definir la ruta para guardar el archivo .dot
output_path = os.path.join(os.getcwd(), "iris_tree.doc")

# Exportar el gráfico del árbol de decisión a un archivo .dot
export_graphviz(
    tree_clf,
    out_file=output_path,  # Guardar en archivo .dot
    feature_names=iris.feature_names[2:],  # Nombre de las características
    class_names=iris.target_names,  # Nombre de las clases (especies de flores)
    rounded=True,  # Bordes redondeados
    filled=True    # Colores para las clases
)

print(f"Árbol de decisión exportado a {output_path}")
