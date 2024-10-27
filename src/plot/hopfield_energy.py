import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from src.Models.Hopfield import HopfieldNetwork
from src.Utils.alphabet import read_matrices_from_file

# Inicializar la red Hopfield
network = HopfieldNetwork()

# Leer matrices desde archivo
filename = './raw_data/alphabet.txt'
matrices = read_matrices_from_file(filename)

# Elegir el mejor subconjunto de patrones
best_subset1 = [matrices[1], matrices[16], matrices[19], matrices[21]]

# Elegir el mejor subconjunto de patrones
best_subset2 = [matrices[11], matrices[16], matrices[19], matrices[23]]

# Elegir el peor de patrones
worst_subset = [matrices[7], matrices[12], matrices[13], matrices[22]]

# Entrenar la red con el subconjunto
network.train(best_subset2)

# Crear un patrón con ruido
noisy_pattern = network.add_noise(best_subset2[0], noise_level=0.08)

print(noisy_pattern)

# Actualizar la red y capturar energía y evolución de estados
final_state, energies, state_evolution = network.update(noisy_pattern)

# Plot 1: Energía vs. Épocas con tamaño de letra aumentado
fig_energy = go.Figure()

fig_energy.add_trace(go.Scatter(
    y=energies,
    mode='lines+markers',
    name='Energía'
))

fig_energy.update_layout(
    title=dict(text='Energía vs Épocas', font=dict(size=26)),  # Título más grande
    xaxis=dict(
        title=dict(text='Época', font=dict(size=24)),
        tickmode='linear',  # Mostrar solo números enteros
        tick0=0,            # Iniciar desde 0
        dtick=1,            # Incrementar en pasos de 1
        tickfont=dict(size=20)  # Aumentar tamaño de los ticks
    ),
    yaxis=dict(
        title=dict(text='Energía', font=dict(size=24)),
        tickfont=dict(size=20)  # Aumentar tamaño de los ticks
    ),
    template='plotly_white'
)

fig_energy.show()

# Plot 2: Evolución de Estados
num_steps = len(state_evolution)
fig_states = make_subplots(
    rows=1, cols=num_steps,
    subplot_titles=[f'Paso {i}' for i in range(num_steps)],
)

for idx, state in enumerate(state_evolution):
    fig_states.add_trace(
        go.Heatmap(
            z=state,  # Keep original state
            colorscale='Greys',
            showscale=False
        ),
        row=1, col=idx + 1
    )

# Reverse the y-axes of all subplots to avoid flipping the pattern
for i in range(num_steps):
    fig_states.update_yaxes(
        autorange='reversed',  # Reverse the y-axis display
        row=1, col=i + 1
    )
    fig_states.update_xaxes(visible=False, row=1, col=i + 1)
    fig_states.update_yaxes(visible=False, row=1, col=i + 1)

fig_states.update_layout(
    title='Evolución de Estados',
    template='plotly_white'
)

fig_states.show()
