# AstroPyFlow

🌌 **Aplicação Python modular para demonstração de uso de Python na Astronomia**

AstroPyFlow é um projeto educacional e prático que integra coleta/validação de dados astronômicos, visualização, cálculos orbitais e de luminosidade, e simulações básicas de sistema solar e exoplanetas.

## 🎯 Objetivo

Demonstrar na prática o uso de Python na Astronomia seguindo rigorosamente boas práticas de engenharia de software científico:
- Coleta e validação de dados astronômicos
- Visualização com Matplotlib + Astropy
- Cálculos de distância, luminosidade e conversões de unidades
- Simulações orbitais simplificadas
- Classificação básica de exoplanetas

## 🛠️ Stack Tecnológica

- **Python 3.9+**
- **Astropy** (`units`, `coordinates`, `time`)
- **Matplotlib** & **NumPy** (cálculos vetoriais e plotagem)
- **Pandas** (manipulação de dados tabulares)
- **Logging** (nativa) & **Type Hints** (`typing`)
- **pytest** (estrutura de testes)
- **Rich** (CLI amigável)

## 📁 Estrutura do Projeto

```
astropyflow/
├── pyproject.toml          # Configuração do projeto
├── requirements.txt        # Dependências
├── README.md              # Este arquivo
├── src/
│   └── astropyflow/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── data_handler.py    # Coleta, validação e sanitização
│       │   └── calculations.py    # Distâncias, luminosidade, conversões
│       ├── viz/
│       │   ├── __init__.py
│       │   └── plotting.py        # Gráficos com Matplotlib + Astropy
│       ├── sim/
│       │   ├── __init__.py
│       │   ├── solar_system.py    # Simulação orbital (N-corpos simplificada)
│       │   └── exoplanets.py      # Classificação básica/habitabilidade
│       └── utils/
│           ├── __init__.py
│           ├── logger.py
│           └── validators.py      # Verificação de qualidade de dados
├── tests/
│   ├── __init__.py
│   ├── test_data_handler.py
│   └── test_calculations.py
└── examples/
    └── run_demo.py               # Script principal de demonstração
```

## 🚀 Instalação

### Pré-requisitos
- Python 3.9 ou superior
- pip >= 21.0

### Passos

1. Clone o repositório:
```bash
git clone https://github.com/astropyflow/astropyflow.git
cd astropyflow
```

2. Crie um ambiente virtual (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Ou instale o pacote em modo desenvolvimento:
```bash
pip install -e ".[dev]"
```

## 📋 Funcionalidades

### 1. Validação & Coleta de Dados
- Carregamento de dados de fontes públicas (CSV, API NASA/MAST)
- Verificação de qualidade: valores nulos, outliers, unidades inconsistentes
- Conversão automática para unidades Astropy

### 2. Visualização Astronômica
- Posição do Sol (Altitude vs Azimute) para localização específica
- Curva de luz de exoplanetas (fluxo vs tempo)
- Exportação para PNG/SVG

### 3. Cálculos & Modelagem
- Distância angular/linear entre corpos celestes
- Luminosidade aparente/absoluta (módulo de distância)
- Vetorização com NumPy

### 4. Simulações
- Sistema Solar simplificado (integração Verlet/Euler)
- Classificação de exoplanetas (raio, período, zona habitável)

## 💡 Exemplos de Uso

### Executar Demo Completa
```bash
python examples/run_demo.py
```

### Uso Programático

```python
from astropyflow.core.data_handler import load_star_data, validate_catalog
from astropyflow.core.calculations import calculate_distance, calculate_luminosity
from astropyflow.viz.plotting import plot_sun_position, plot_light_curve
from astropyflow.sim.solar_system import simulate_orbits
from astropyflow.sim.exoplanets import classify_exoplanet

# Carregar e validar dados
data = load_star_data("catalog.csv")
validated = validate_catalog(data)

# Calcular distância e luminosidade
dist = calculate_distance(ra1, dec1, ra2, dec2)
lum = calculate_luminosity(apparent_mag, distance)

# Plotar posição do Sol
plot_sun_position(location="Sao Paulo", date="2024-01-01")

# Simular órbitas
orbits = simulate_orbits(bodies=3, steps=1000)

# Classificar exoplaneta
classification = classify_exoplanet(radius=1.5, period=365, temp=288)
```

## 🧪 Testes

Execute os testes unitários:
```bash
python -m pytest tests/ -v
```

Com cobertura de código:
```bash
python -m pytest tests/ --cov=src/astropyflow --cov-report=html
```

## 📊 Notas de Desempenho

### Otimizações Implementadas
- Uso de `dtype` adequados no NumPy (float64 para precisão científica)
- Vetorização de operações (evitar loops explícitos)
- Logging configurado por níveis (INFO/DEBUG/WARNING/ERROR)

### Escalando para Grandes Datasets
Para datasets muito grandes (>1GB), considere:
1. **Astropy Tables**: Use `astropy.table.Table` com leitura chunked
2. **Dask**: Para processamento paralelo de arrays grandes
3. **HDF5**: Armazenamento eficiente com `h5py` ou `astropy.io.fits`

Exemplo com Astropy Table:
```python
from astropy.table import Table
data = Table.read('large_catalog.fits', format='fits')
```

Exemplo com Dask:
```python
import dask.dataframe as dd
df = dd.read_csv('large_catalog.csv')
result = df.groupby('spectral_class').mean().compute()
```

## ⚠️ Sinais de Alerta

O projeto implementa verificações para:
- **Qualidade dos Dados**: Validação prévia com warnings logados
- **Performance**: Algoritmos vetorizados, sem loops puros em NumPy
- **Documentação**: Docstrings estilo Google/NumPy em todos módulos públicos
- **Tratamento de Erros**: Exceptions explícitas com logging apropriado

## 📄 Licença

MIT License - veja o arquivo LICENSE para detalhes.

## 🤝 Contribuição

Contribuições são bem-vindas! Por favor:
1. Fork o repositório
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📚 Referências

- [Astropy Documentation](https://docs.astropy.org/)
- [NASA MAST Archive](https://mast.stsci.edu/)
- [Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)

---

**AstroPyFlow** - Explorando o universo com Python 🌟
