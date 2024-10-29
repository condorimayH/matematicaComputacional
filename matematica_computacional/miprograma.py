import numpy as np
import mibiblioteca as bib
import time
from tabulate import tabulate
from fpdf import FPDF

# Generar una matriz aleatoria A y el vector b
np.random.seed(0)  # Para reproducibilidad
A = np.random.uniform(-10, 10, (10, 10))
x_exacta = np.ones((10, 1))  # Solución exacta x = [1, 1, ..., 1]
b = np.dot(A, x_exacta)

# Crear un PDF
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)

# Función para agregar una tabla al PDF
def add_table_to_pdf(pdf, data, title):
    pdf.cell(200, 10, txt=title, ln=True, align='C')
    table = tabulate(data, tablefmt="plain")
    for line in table.split("\n"):
        pdf.cell(200, 10, txt=line, ln=True)

# Agregar la matriz de coeficientes A al PDF
pdf.cell(200, 10, txt="Matriz de coeficientes A:", ln=True)
add_table_to_pdf(pdf, A, "Matriz de coeficientes A")

# Agregar el vector de términos independientes b al PDF
pdf.cell(200, 10, txt="\nVector de términos independientes b:", ln=True)
add_table_to_pdf(pdf, b, "Vector de términos independientes b")

# Resolución por eliminación gaussiana simple
# Matriz aumentada
Ab = np.append(A, b, axis=1)
pdf.cell(200, 10, txt="\nMatriz aumentada [A|b]:", ln=True)
add_table_to_pdf(pdf, Ab, "Matriz aumentada [A|b]")

# Escalonar la matriz aumentada
bib.escalonaSimple(Ab)
pdf.cell(200, 10, txt="\nMatriz aumentada escalonada:", ln=True)
add_table_to_pdf(pdf, Ab, "Matriz aumentada escalonada")

# Resolver el sistema
A1 = Ab[:, :10]
b1 = Ab[:, 10]
b1 = b1.reshape(b1.shape[0], 1)
x = bib.sustRegresiva(A1, b1)

pdf.cell(200, 10, txt="\nSolución del sistema:", ln=True)
add_table_to_pdf(pdf, x, "Solución del sistema")

# Calcular la norma suma del residuo
residuo = b - np.dot(A, x)
norma_suma_residuo = np.sum(np.abs(residuo))
pdf.cell(200, 10, txt="\nNorma suma del residuo:", ln=True)
pdf.cell(200, 10, txt=str(norma_suma_residuo), ln=True)

# Guardar el PDF
pdf.output("resultados.pdf")

print("PDF generado con éxito: resultados.pdf")
