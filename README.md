# SIA-TP4 Grupo 4

## Contributors

| Student              | ID    | Email                |
|----------------------|-------|----------------------|
| Gastón Alasia        | 61413 | galasia@itba.edu.ar  |
| Juan Segundo Arnaude | 62184 | jarnaude@itba.edu.ar |
| Bautista Canevaro    | 62179 | bcanevaro@itba.edu.ar |
| Matías Wodtke        | 62098 | mwodtke@itba.edu.ar  |

---

In this project, we implement unsupervised learning algorithms: **Hopfield**, **Oja**, and **Kohonen**.

## System Requirements

- Python 3.10
- Required Libraries:
  - `Numpy`
  - `Sklearn`
  - `Matplotlib`
  - `Pandas`

## Execution

### Run Kohonen:
```bash
python -m src.runKohonen
```

### Kohonen: config file
```json
{
    "k":4,
    "learning_rate":0.1,
    "sigma":3
}
```

### Run Oja:
```bash
python -m src.runOja
```

### Oja: config file
```json
{
  "iterations": 10000,
  "learning_rate": 0.001
}
```

### Run Hopfield:
```bash
python -m src.runHopfield
```