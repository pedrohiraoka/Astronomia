# StellarSeeker

**Análise de Curvas de Luz e Descoberta de Exoplanetas com Dados do TESS**

O **StellarSeeker** é uma aplicação Python completa para análise de curvas de luz astronômicas e descoberta de exoplanetas utilizando dados públicos da missão TESS (Transiting Exoplanet Survey Satellite). Desenvolvido como ferramenta educacional e de pesquisa, permite que astrônomos amadores e cidadãos cientistas contribuam para a descoberta de novos planetas extrassolares sem necessidade de equipamento especializado.

## 🌟 Características Principais

- **Acesso a dados profissionais**: Conecte-se ao arquivo MAST para baixar curvas de luz do TESS
- **Análise completa**: Periodograma, dobradura de fase e detecção automática de trânsitos
- **Visualização profissional**: Gráficos publication-ready com matplotlib
- **Exportação múltipla**: PNG, CSV, FITS e JSON
- **Modo interativo**: Interface de linha de comando guiada para iniciantes
- **Exemplos demonstrativos**: Análise de estrelas conhecidas como π Mensae e Algol

## 📦 Instalação

### Requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### Instalação das dependências

```bash
pip install -r requirements.txt
```

### Instalação do pacote

```bash
pip install -e .
```

## 🚀 Uso Rápido

### Análise Básica

```python
from stellar_seeker import StellarSeeker

# Inicializar com um TIC (TESS Input Catalog)
seeker = StellarSeeker(tic_id=261136679)

# Buscar setores disponíveis
sectors = seeker.search_sectors()

# Baixar curva de luz
lc = seeker.download_lightcurve(sector_index=0)

# Visualizar
seeker.plot_lightcurve()

# Calcular periodograma e obter período
pg = seeker.calculate_periodogram()
period = seeker.get_primary_period()
print(f"Período candidato: {period:.4f} dias")

# Dobrar fase
seeker.fold_phase()

# Detectar trânsitos
transits = seeker.detect_transits()

# Exportar resultados
seeker.export_results(output_dir='./minha_analise')
```

### Modo Interativo

```python
from stellar_seeker import StellarSeeker

seeker = StellarSeeker(tic_id=261136679)
seeker.interactive_mode()
```

## 🔭 Exemplos de Alvos

| Estrela | TIC | Tipo | Período |
|---------|-----|------|---------|
| π Mensae | 261136679 | Exoplaneta confirmado | ~6.26 dias |
| Algol | 346783960 | Binária eclipsante | ~2.87 dias |
| TYC 7037-89-1 | 168789840 | Sistema sextuplo | ~1.57 dias |

## 📖 Documentação Completa

Veja o Jupyter Notebook em `docs/tutorial.ipynb` para um tutorial completo com explicações conceituais sobre:
- Método de trânsitos para detecção de exoplanetas
- Interpretação de curvas de luz
- Significado de periodogramas e dobradura de fase
- Orientações para contribuições de cidadãos cientistas

## 📝 Dependências

- lightkurve
- matplotlib
- numpy
- scipy
- astropy

## 🗂️ Estrutura do Projeto

```
stellar-seeker/
├── stellar_seeker/
│   ├── __init__.py
│   ├── core.py
│   └── utils.py
├── examples/
│   └── demo_examples.py
├── docs/
│   └── tutorial.ipynb
├── requirements.txt
├── setup.py
└── README.md
```

## 🎓 Para Cidadãos Cientistas

O StellarSeeker foi desenvolvido para democratizar o acesso à pesquisa astronômica. Participe de projetos como [Planet Hunters TESS](https://www.planethunters.org/tess)!

---

**StellarSeeker** - Democratizando a descoberta de exoplanetas 🌍🔭✨
