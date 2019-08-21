# API flask para rodar uma aplicação Keras com animação em JS na visualização
É uma pequena aplicação  com Flask onde roda uma ANN feita em keras para classificar se a pessoa sobreveriria ou não no acidente do titanic. 

O objetivo da aplicação é determinar se o usuario sobreviveria ou não ao desastre do titanic.

| Role                 | Responsibility         | Full name                | e-mail       |
| -----                | ----------------       | -----------              | ---------    |
| Data Scientist       | Author                 | [`João Amorim`]          | [`joaogustavoamorim@gmail.com`] |
| Dev Frontend         | Author                 | [`Paolla Olsen`]         | [`paollaolsen@gmail.com`] |

## Dataset
Dataset retirado de uma competição no kaggle -> [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/overview)

### Competition Description

The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.

One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.

In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

## Desenvolvimento 
A analise dos dados e o embasamento inicial do DS foi feito neste [notebook](https://www.kaggle.com/cstahl12/titanic-with-keras/notebook) disponivel no kaggle.


Uma tela web (HTML5+CSS3+JS) e uma aplicação backend para realizar a função de servidor com python (Flask).

A tela (frontend) consiste em um basico formulario, onde o mesmo será enviado utilizando ajax e baseado na resposta é feita determinada ação na tela com JS.
* Se a pessoa morrer no desastre: na tela deve ser exibido o titanic afundando no mar
* Se a pessoa sobrevive:
  * Homem: Aparece um homem nadando 
  * Mulher: Aparece uma mulher nadando


O backend  (aplicação flask) deverá receber as informações do formulario, rodar o modelo de predição (modelo keras já treinado pelos autores do repositorio) e devolver o resultado da predição. Além de ser o responsavel pelo rotacionamenteo do site e dos arquivos dentro do site. Alem de renderizar a linguagem de template Jinja (que não sera utilizada neste workshop).
 
## Instalações

* Instalar python 3.6
  * Instalar com o gerenciador de pacotes 'pip'

## Instalações de bibliotecas windows

A partir de um terminal/shell/powershell no windows como administrador rode os seguintes comandos:

```bash
pip install virtualenv  # instalando o gerenciador de ambientes virtuais do python

```


### Usando virtualenv no windows

Criando uma ambiente virtual em [´./env/´](./env/):

```bash
virtualenv.exe env -p python
```

O argumento passado para o *-p* é para determinar qual python devera ser instalado dentro do ambiente virtual.

Ativando este ambiente virtual
```bash
./env/Scripts/activate
```

Agora o sistema usara as bibliotecas e o python instalado dentro do ambiente virtual *env*.

Para desabilitar este ambiente virtual basta rodar:
```bash
deactivate
```
