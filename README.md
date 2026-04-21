# Simulação de Dinâmica Orbital: Sol, Terra e Super Júpiter

## Descrição

Este projeto implementa uma simulação numérica da interação gravitacional triádica entre três corpos celestes:
- **Sol**: Corpo central massivo
- **Terra**: Planeta terrestre em órbita a 1 UA
- **Super Júpiter**: Planeta hipotético com massa equivalente a 500 vezes a massa de Júpiter, orbitando a 5.2 UA

A simulação utiliza a **Lei da Gravitação Universal de Newton** e integra as equações diferenciais do movimento através do método **Runge-Kutta de quarta ordem (RK4)** ao longo de 120 anos com resolução de 100 passos por ano.

## Características Técnicas

### Método Numérico
- **Integrador**: Runge-Kutta de 4ª ordem (RK4)
- **Duração da simulação**: 120 anos
- **Resolução temporal**: 100 passos por ano (12.000 iterações totais)
- **Regularização**: Termo epsilon de 1e-20 para evitar singularidades por divisão zero

### Normalização de Unidades
Para garantir estabilidade numérica, todas as grandezas são normalizadas:
- **Unidade de massa**: Massa da Terra (MM = 6×10²⁴ kg)
- **Unidade de distância**: Unidade Astronômica (AU = 1.496×10¹¹ m)
- **Unidade de tempo**: Ano sideral (YEAR = 365 × 24 × 3600 s)
- **Constante gravitacional normalizada**: GG = (MM × G × YEAR²) / AU³

### Constantes Físicas
```python
G  = 6.673e-11 m³/(kg·s²)     # Constante gravitacional universal
AU = 1.496e11 m                # Unidade Astronômica
MM = 6e24 kg                   # Massa da Terra (unidade de massa)
MS = 2e30 / MM                 # Massa do Sol (normalizada)
MJ = 500 × 1.9e27 / MM         # Massa do Super Júpiter (normalizada)
```

## Requisitos

- Python 3.8 ou superior
- numpy >= 1.20.0
- matplotlib >= 3.4.0
- tqdm >= 4.60.0

## Instalação

1. Clone o repositório ou navegue até o diretório do projeto:
```bash
cd /workspace
```

2. Crie um ambiente virtual (opcional, mas recomendado):
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# ou
venv\Scripts\activate     # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Execução

Execute a simulação com o comando:
```bash
python simulation.py
```

A animação será exibida em uma janela interativa mostrando:
- O Sol no centro (marcador amarelo-laranja)
- A trajetória da Terra (rastro azul, 40 passos)
- A trajetória do Super Júpiter (rastro vermelho, 200 passos)
- Escala de referência de 1 UA
- Tempo decorrido em anos

## Estrutura do Código

### Funções Principais

#### `gravitational_force(m1: float, m2: float, r: np.ndarray) -> np.ndarray`
Calcula o vetor força gravitacional entre dois corpos, utilizando:
- Norma euclidiana regularizada com epsilon
- Função `arctan2` para determinação angular precisa
- Inversão de sinal para garantir atração mútua

#### `RK4Solver(t: float, r: np.ndarray, v: np.ndarray, h: float, planet: str, r_other: np.ndarray, v_other: np.ndarray) -> Tuple[np.ndarray, np.ndarray]`
Implementa o integrador Runge-Kutta de 4ª ordem:
- Calcula quatro inclinações intermediárias (k1, k2, k3, k4)
- Diferencia dinâmica entre Terra e Super Júpiter
- Aplica forças solares e interação mútua
- Retorna posição e velocidade atualizadas

#### `setup_animation() -> Tuple[py.Figure, py.Axes, Line2D, Line2D, py.Text]`
Configura o ambiente de visualização:
- Figura com eixos quadrados e limites [-7.2, 7.2] UA
- Marcadores para os três corpos celestes
- Linhas de rastro inicializadas
- Objeto de texto para display temporal

#### `animate(i: int) -> Tuple[Line2D, Line2D, py.Text]`
Atualiza cada quadro da animação:
- Atualiza dados geométricos das trajetórias
- Aplica janelas de rastro (40 para Terra, 200 para Super Júpiter)
- Exibe tempo decorrido com precisão de 1 casa decimal

## Condições Iniciais

| Corpo | Posição (UA) | Velocidade (UA/ano) |
|-------|--------------|---------------------|
| Terra | [1.0, 0.0] | [0, √(MS×GG/r)] |
| Super Júpiter | [5.2, 0.0] | [0, 13.06×10³×YEAR/AU] |

## Saída Esperada

Uma animação em tempo real mostrando a evolução orbital dos três corpos ao longo de 120 anos. Devido à massa extrema do Super Júpiter (500× Júpiter), efeitos significativos de perturbação gravitacional podem ser observados na órbita terrestre, demonstrando a natureza caótica de sistemas de N-corpos com massas comparáveis.

## Notas Técnicas

- **Estabilidade Numérica**: Todas as massas são normalizadas pela massa da Terra para evitar overflow/underflow
- **Regularização**: O termo epsilon (1e-20) previne divisões por zero em aproximações extremas
- **Performance**: Alocações de memória são minimizadas dentro dos loops principais
- **Type Hints**: Todo o código segue convenções PEP 8 com anotações de tipo completas

## Licença

Este projeto é fornecido para fins educacionais e de pesquisa em dinâmica orbital e computação científica.

## Referências

- Newton, I. (1687). *Philosophiæ Naturalis Principia Mathematica*
- Runge, C. (1895). "Über die numerische Auflösung von Differentialgleichungen"
- Kutta, W. (1901). "Beitrag zur näherungsweisen Integration totaler Differentialgleichungen"