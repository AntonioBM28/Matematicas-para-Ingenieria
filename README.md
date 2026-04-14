# Matemáticas para Ingeniería - Unidad 3

Repositorio con scripts en Python para la resolución de Ecuaciones Diferenciales Ordinarias (EDOs) empleando varios métodos numéricos.

## 📂 Contenido del directorio

- `tarea1_euler_euler_mod_rk4.py`: Implementación y comparativa de los métodos de **Euler**, **Euler Modificado** y **Runge-Kutta de 4° Orden**.
- `tarea2_euler_modificado.py`: Implementación enfocada en el métodio de **Euler Modificado** (Método de Heun).
- `tarea3_rk4.py`: Implementación detallada del método de **Runge-Kutta de 4° orden** (RK4).

## ⚙️ Requisitos

Para poder ejecutar estos scripts de manera correcta, necesitas tener instalado **Python 3** junto con las siguientes bibliotecas para los cálculos matemáticos y la generación de reportes en PDF:

- `numpy`
- `matplotlib`

Puedes instalar las dependencias fácilmente abriendo tu terminal o línea de comandos y ejecutando:

```bash
pip install numpy matplotlib
```

## 🚀 Instrucciones de Ejecución

> **⚠️ IMPORTANTE antes de ejecutar:** 
> Los scripts contienen una variable `OUTPUT` al final del archivo (por ejemplo, `OUTPUT = "/mnt/user-data/outputs/tarea1_euler_euler_mod_rk4.pdf"`) que define la ruta donde se guardará el PDF generado. Si estás ejecutando esto en un entorno local diferente (como Windows), **es recomendable modificar esta variable** para que apunte a un directorio local válido en tu máquina, o simplemente poner un nombre de archivo para que se guarde en tu misma carpeta (ej.: `OUTPUT = "tarea1.pdf"`).

Para ejecutar los archivos, abre la terminal en la carpeta donde se encuentran ubicados y ejecuta:

```bash
python tarea1_euler_euler_mod_rk4.py
```
```bash
python tarea2_euler_modificado.py
```
```bash
python tarea3_rk4.py
```

Tras la ejecución, verás el progreso en la terminal y se generarán los respectivos archivos `.pdf` con las tablas de evaluación y gráficas comparativas correspondientes a las diferentes problemáticas y tamaños de paso (`h`).
