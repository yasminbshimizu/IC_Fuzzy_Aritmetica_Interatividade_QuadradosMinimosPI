import math
import numpy as np
import matplotlib.pyplot as plt


###############################################################################
############################   Funções Suporte   ##############################
###############################################################################


def _aplica_funcao(x_data, f_expr):
    """
    Avalia a função f(x), definida por uma string, nos pontos de x_data e retorna os resultados como vetor coluna.

    Args:
        x_data (array-like): vetor coluna (n x 1) contendo os valores de entrada x₀ nos quais a função será avaliada.
        f_expr (str): Expressão representando a função f(x) interpretada via eval, por exemplo: "x**2 + 3*x - 1" ou "math.cos".
        
    Returns:
        results_f (np.ndarray): vetor coluna (n x 1) contendo os valores f(x₀) para cada x₀ em x_data.
    """
    
    # Criar função f(x) a partir da expressão f_expr
    f = lambda x: eval(f_expr)
    
    # Lista para armazenar os resultados de f(x0), x0 em x_data
    results_f = []
    
    # Itera sobre os valores em x_data e calcula f(x)
    for x in x_data:
        results_f.append(f(x[0]))
        
    # Transforma results_f em uma matriz coluna com cada valor f(x0), x0 em x_data
    results_f = np.array(results_f)
    
    results_f = results_f.reshape(-1,1)
    
    return results_f


###############################################################################
#############################   Números Fuzzy   ###############################
###############################################################################

def alpha_nivel (phif, range_x, alpha=0):
    """ Calcula o subconjunto real correspondente a pertinência alpha num número fuzzy.
    
    Args:
        phif (list of string): função de pertinencia do número fuzzy.
        range_x (array-like): valores para os quais phi = (0,1,0) na forma (xlim_esq, x_pico, xlim_dir).
        alpha (float): alpha entre 0 e 1 em questão.
    
    Return:
        alpha_nivel (list): subconjunto real correspondente ao alpha indicado para o número fuzzy.
    
    """
    
    n1, n2 = phif
    
    x_lim1, x_pico, x_lim2 = range_x
    
    x1 = np.linspace(x_pico, x_lim1, 1001)
    x2 = np.linspace(x_pico, x_lim2, 1001)
    
    alpha_lim1 = []
    alpha_lim2 = []
    
    for i in range(1001):
        if not alpha_lim1: 
            phi1_x = _aplica_funcao([[x1[i]]], n1)[0][0]
            if phi1_x == alpha:
                alpha_lim1.append(x1[i])
        if not alpha_lim2: 
            phi2_x = _aplica_funcao([[x2[i]]], n2)[0][0]
            if phi2_x == alpha:
                alpha_lim2.append(x2[i])
        else:
            break
    
    alpha_nivel = alpha_lim1 + alpha_lim2
            
    return alpha_nivel


def f_alpha_ntrifuzzy(ntri):
    """
    Define um número triangular fuzzy em função de seus alpha-níveis.

    Args:
        nt (tuple or array-like): número triangular fuzzy x na forma x = (a,b,c).
        
    Returns:
        alphaf (list of str): lista contendo a expressão algébrica que representa x em função de seus alpha-níveis.
    """
    
    a,b,c = ntri
    
    alpha1 = f"x*({b}-{a}) + {a}"
    alpha2 = f"-x*({c}-{b}) + {c}"
    
    alphaf = [alpha1, alpha2]
    
    return alphaf


def crisp2fuzzy(a, n_alphas=1001):
    """
    Transforma uma número real em uma lista de valores repetidos de a para n alpha níveis.

    Args:
        a (float): número real que se deseja converter.
        n_alphas (int): número de alpha níveis que se deseja calcular para este número real.
        
    Returns:
        a_fuzzy (list): lista a repetido n vezes, correspondente a cada alpha nível.
    """
    alphas = np.linspace(a, a, n_alphas)
    a_fuzzy = np.array([alphas, alphas])
    
    return a_fuzzy


###############################################################################
#################################   Soma   ####################################
###############################################################################


def soma_fuzzy(x1, x2):
    """
    Realiza a soma fuzzy usual entre dois números fuzzy.

    Args:
        x1 (tuple or array-like): número fuzzy x1, podendo representar seus alpha-níveis ou um número triangular fuzzy na forma x1 = (a,b,c).
        x2 (tuple or array-like): número fuzzy x2, podendo representar seus alpha-níveis ou um número triangular fuzzy na forma x2 = (d,e,f).

    Returns:
        x_soma (tuple): número fuzzy resultante da soma usual.
        
    """
    
    if len(x1) != len(x2):
        raise ValueError("Os números fuzzy x1 e x2 devem ter o mesmo número de alpha-níveis.")

    x_soma = (a+b for a,b in zip(x1,x2))
    
#     x_soma = []
#     for i in range(len(x1)):
#         x_soma.append(x1[i] + x2[i])
        
    x_soma = tuple(x_soma)
    
    return x_soma


def soma_interativa_gamma(A, B, gamma = 0):
    """
    Realiza a soma interativa gamma de dois números fuzzy.

    Args:
        A (tuple or array-like): número fuzzy A por alfa-nível, na forma[[a1-,a2-,..., an-],[a1+,a2+,..., an+]].
        B (tuple or array-like): número fuzzy B por alfa-nível, na forma[[b1-,b2-,..., bn-],[b1+,b2+,..., bn+]].
        gamma (float): fator gamma em operações interativas por JPD.

    Returns:
        S (array-like): número fuzzy resultante da soma interativa.
        
    """
    a_men, a_mai = A
    b_men, b_mai = B
    
    s_men = np.zeros(len(a_men))
    s_mai = np.zeros(len(a_men))
    
    for alpha in range(len(a_men)):
        inf = []
        sup = []
        for beta in range(alpha, len(a_men)):
            beta_min = min([a_men[beta] + b_mai[beta] - gamma*(b_mai[beta] - b_men[beta]), a_mai[beta] +b_men[beta] - gamma*(a_mai[beta]-a_men[beta])])
            beta_max = max([a_men[beta] + b_mai[beta] - gamma*(a_mai[beta] - a_men[beta]), a_mai[beta] +b_men[beta] - gamma*(b_mai[beta]-b_men[beta])])

            inf.append(beta_min)
            sup.append(beta_max)
            
        s_men[alpha] = np.array(np.min(inf))
        s_mai[alpha] = np.array(np.max(sup))
        
    S = [s_men, s_mai]
    return S


def soma_interativa_0 (x1,x2):
    """
    Realiza a soma otimista (gamma zero) de dois números triangulares fuzzy, segundo Wasques et al. (2020b).

    Args:
        x1 (tuple or array-like): número triangular fuzzy x1 na forma x1 = (a,b,c).
        x2 (tuple or array-like): número triangular fuzzy x2 na forma x2 = (d,e,f).

    Returns:
        x_soma (tuple): número triangular fuzzy resultante da soma otimista.
        
    """
    
    #define uma lista vazia para armazenar o resultado
    x_soma = []
    
    #renomeia cada componente dos números fuzzy a serem somados
    a,b,c = x1
    d,e,f = x2
    
    #define os valores dos diâmetros dos números fuzzy
    diam_x1 = c - a
    diam_x2 = f - d
    
    #define as somas entre picos, maior e menor, e menor e maior componente dos números fuzzy
    s_picos = b+e
    s_meios = c+d
    s_extremos = a+f
    
    #realiza a soma fuzy seguindo as condições de diâmetro
    if diam_x1 >= diam_x2: # se o diâmetro do primeiro for maior que o do segundo
        x_soma.append(min(s_extremos, s_picos)) # escolhe a menor soma entre picos ou extremos para definir o menor componente da soma
        x_soma.append(s_picos) # soma os picos para definir o pico da soma
        x_soma.append(max(s_picos, s_meios)) # escolhe a maior soma entre picos ou meios para definir o maior componente da soma
        
    elif diam_x1 <= diam_x2: # se o diâmetro do primeiro for menor que o do segundo
        x_soma.append(min(s_picos, s_meios)) # escolhe a menor soma entre picos e meios para definir o menor componente da soma
        x_soma.append(s_picos) # soma os picos para definir o pico da soma
        x_soma.append(max(s_extremos, s_picos)) # escolhe a maior soma entre picos e extremos para definir o maior componente da soma
        
    #transforma a lista "soma" em uma tupla imutável, notação utilizada para números fuzzy neste trabalho
    x_soma = tuple(x_soma)
    
    return x_soma    


###############################################################################
##############################   Diferença   ##################################
###############################################################################


def dif_fuzzy(x1, x2):
    """
    Realiza a diferença fuzzy usual entre dois números fuzzy.

    Args:
        x1 (tuple or array-like): número fuzzy x1, podendo representar seus alpha-níveis ou um número triangular fuzzy na forma x1 = (a,b,c).
        x2 (tuple or array-like): número fuzzy x2, podendo representar seus alpha-níveis ou um número triangular fuzzy na forma x2 = (d,e,f).

    Returns:
        x_dif (tuple): número fuzzy resultante da diferença usual.
        
    """
    
    if len(x1) != len(x2):
        raise ValueError("Os números fuzzy x1 e x2 devem ter os mesmos alpha-níveis.")
        
    x2_ = x2[::-1]
    x_dif = (a-b for a,b in zip(x1,x2_))
    
#     x_dif = []
#     for i in range(len(x1)):
#         x_dif.append(x1[i] - x2_[i])
        
    x_dif = tuple(x_dif)
    
    return x_dif
   

def dif_g(x1, x2):
    """
    Realiza a diferença de Hukuhara entre dois números fuzzy.

    Args:
        x1 (tuple or array-like): número fuzzy x1, podendo representar seus alpha-níveis ou um número triangular fuzzy na forma x1 = (a,b,c).
        x2 (tuple or array-like): número fuzzy x2, podendo representar seus alpha-níveis ou um número triangular fuzzy na forma x2 = (d,e,f).

    Returns:
        x_dif (tuple): número fuzzy resultante da diferença de Hukuhara.
        
    """
    
    if len(x1) != len(x2):
        raise ValueError("Os números fuzzy x1 e x2 devem ter os mesmos alpha-níveis.")
        
    len_meio = len(x1)//2
    
    x11 = x1[:len_meio]
    x12 = x1[len_meio+1:]
    x21 = x2[:len_meio]
    x22 = x2[len_meio+1:]
    
    x12 = x12[::-1]
    x22 = x22[::-1]
    
    x11_ = x11[::-1]
    x12_ = x12[::-1]
    x21_ = x21[::-1]
    x22_ = x22[::-1]
    
    x1_pico = x1[len_meio]
    x2_pico = x2[len_meio]
        
    x_dif1_ = [x1_pico - x2_pico]
    x_dif2_ = [x1_pico - x2_pico]
    
    for a1,b1,a2,b2 in zip(x11_,x21_,x12_,x22_):    
        alpha_dif1 = min(a1-b1, a2-b2, *x_dif1_)
        alpha_dif2 = max(a1-b1, a2-b2, *x_dif2_)
        
        x_dif1_.append(alpha_dif1)
        x_dif2_.append(alpha_dif2) 
    
    x_dif1 = x_dif1_[::-1]
    x_dif2 = x_dif2_[::-1]
    
    x_dif2 = x_dif2[::-1]
    
    x_dif = x_dif1+x_dif2[1:]
    
#     x_dif = []
#     for i in range(len(x1)):
#         x_dif.append(x1[i] - x2_[i])
        
    x_dif = tuple(x_dif)
    
    return x_dif


def dif_interativa_gamma(A, B, gamma = 0):
    """
    Realiza a diferença interativa gamma de dois números fuzzy.

    Args:
        A (tuple or array-like): número fuzzy A por alfa-nível, na forma[[a1-,a2-,..., an-],[a1+,a2+,..., an+]].
        B (tuple or array-like): número fuzzy B por alfa-nível, na forma[[b1-,b2-,..., bn-],[b1+,b2+,..., bn+]].
        gamma (float): fator gamma em operações interativas por JPD.

    Returns:
        x_dif (tuple): número fuzzy resultante da diferença interativa.
        
    """
    
    opB = oposto_fuzzy(B)
    D =  soma_interativa_gamma(A, opB, gamma)
    
    return D


def dif_interativa_0 (x1,x2):
    """
    Realiza a diferença otimista (gamma zero) de dois números triangulares fuzzy, segundo Wasques et al. (2020b).

    Args:
        x1 (tuple or array-like): número triangular fuzzy x1 na forma x1 = (a,b,c).
        x2 (tuple or array-like): número triangular fuzzy x2 na forma x2 = (d,e,f).

    Returns:
        x_dif (tuple): número triangular fuzzy resultante da diferença otimista.
        
    """
    
    x2_ = oposto_ntfuzzy(x2)
    x_dif = soma_interativa_0(x1, x2_)
    
    return x_dif


###############################################################################
############################   Multiplicação   ################################
###############################################################################


def prod_ntrifuzzy(x_num, x_den):
    """
    Realiza a divisão intervalar de dois números triangulares fuzzy a partir de seus alpha-niveis.

    Args:
        x_num (tuple or array-like): número triangular fuzzy x_num na forma x_num = (an, bn, cn), numerador na divisão.
        x_den (tuple or array-like): número triangular fuzzy x_den na forma x_den = (ad, bd, cd), denominador na divisão.

    Returns:
        x_quo_alpha (list of np.ndarray): lista de vetores colunas (n x 1) representando os valores de x_quo em função dos alpha-níveis, em que cada vetor corresponde a um intervalo da função de pertinência por partes.
    """
    
    alpha = np.arange(0, 1.001, 0.001).reshape((-1,1))
    
    n1 = _aplica_funcao(alpha, f_alpha_ntrifuzzy(x_num)[0])
    n2 = _aplica_funcao(alpha, f_alpha_ntrifuzzy(x_num)[1])
    d1 = _aplica_funcao(alpha, f_alpha_ntrifuzzy(x_den)[0])
    d2 = _aplica_funcao(alpha, f_alpha_ntrifuzzy(x_den)[1])

    caso1 = [(i*j).item() for i, j in zip(n1, d1)]
    caso2 = [(i*j).item() for i, j in zip(n1, d2)]
    caso3 = [(i*j).item() for i, j in zip(n2, d1)]
    caso4 = [(i*j).item() for i, j in zip(n2, d2)]

    x_p1 = []
    x_p2 = []
         
    for i in range(len(alpha)):
        p1_i = min(caso1[i], caso2[i], caso3[i], caso4[i])
        p2_i = max(caso1[i], caso2[i], caso3[i], caso4[i])
        x_p1.append(p1_i)
        x_p2.append(p2_i)
 
    x_prod_alpha = [x_p1,x_p2]
    
    return x_prod_alpha


def prod_fuzzy(x_num, x_den):
    """
    Realiza a divisão intervalar de dois números triangulares fuzzy a partir de seus alpha-niveis.

    Args:
        x_num (tuple or array-like): número triangular fuzzy x_num na forma x_num = (an, bn, cn), numerador na divisão.
        x_den (tuple or array-like): número triangular fuzzy x_den na forma x_den = (ad, bd, cd), denominador na divisão.

    Returns:
        x_quo_alpha (list of np.ndarray): lista de vetores colunas (n x 1) representando os valores de x_quo em função dos alpha-níveis, em que cada vetor corresponde a um intervalo da função de pertinência por partes.
    """
    
    n1, n2 = x_num
    d1, d2 = x_den

    caso1 = [(i*j).item() for i, j in zip(n1, d1)]
    caso2 = [(i*j).item() for i, j in zip(n1, d2)]
    caso3 = [(i*j).item() for i, j in zip(n2, d1)]
    caso4 = [(i*j).item() for i, j in zip(n2, d2)]

    x_p1 = []
    x_p2 = []
         
    for i in range(len(n1)):
        p1_i = min(caso1[i], caso2[i], caso3[i], caso4[i])
        p2_i = max(caso1[i], caso2[i], caso3[i], caso4[i])
        x_p1.append(p1_i)
        x_p2.append(p2_i)
 
    x_prod_alpha = [x_p1,x_p2]
    
    return x_prod_alpha


###############################################################################
###############################   Divisão   ###################################
###############################################################################


def divisao_ntrifuzzy(x_num, x_den):
    """
    Realiza a divisão intervalar de dois números triangulares fuzzy a partir de seus alpha-niveis.

    Args:
        x_num (tuple or array-like): número triangular fuzzy x_num na forma x_num = (an, bn, cn), numerador na divisão.
        x_den (tuple or array-like): número triangular fuzzy x_den na forma x_den = (ad, bd, cd), denominador na divisão.

    Returns:
        x_quo_alpha (list of np.ndarray): lista de vetores colunas (n x 1) representando os valores de x_quo em função dos alpha-níveis, em que cada vetor corresponde a um intervalo da função de pertinência por partes.
    """
    
    alpha = np.arange(0, 1.001, 0.001).reshape((-1,1))
    
    n1 = _aplica_funcao(alpha, f_alpha_ntrifuzzy(x_num)[0])
    n2 = _aplica_funcao(alpha, f_alpha_ntrifuzzy(x_num)[1])
    d1 = _aplica_funcao(alpha, f_alpha_ntrifuzzy(x_den)[0])
    d2 = _aplica_funcao(alpha, f_alpha_ntrifuzzy(x_den)[1])

    caso1 = [(i / j).item() for i, j in zip(n1, d1)]
    caso2 = [(i / j).item() for i, j in zip(n1, d2)]
    caso3 = [(i / j).item() for i, j in zip(n2, d1)]
    caso4 = [(i / j).item() for i, j in zip(n2, d2)]

    x_q1 = []
    x_q2 = []
         
    for i in range(len(alpha)):
        q1_i = min(caso1[i], caso2[i], caso3[i], caso4[i])
        q2_i = max(caso1[i], caso2[i], caso3[i], caso4[i])
        x_q1.append(q1_i)
        x_q2.append(q2_i)
 
    x_quo_alpha = [x_q1,x_q2]
    
    return x_quo_alpha


def gdivisao_ntrifuzzy(x_num, x_den):
    """    
    Realiza a divisão generalizada de dois números triangulares fuzzy a partir de seus alpha-niveis, segundo Stefanini (2010).

    Args:
        x_num (tuple or array-like): número triangular fuzzy x_num na forma x_num = (an, bn, cn), numerador na divisão.
        x_den (tuple or array-like): número triangular fuzzy x_den na forma x_den = (ad, bd, cd), denominador na divisão.

    Returns:
        x_g_quo (list of np.ndarray): lista de vetores colunas (n x 1) representando os valores de x_g_quo em função dos alpha-níveis, em que cada vetor corresponde a um intervalo da função de pertinência por partes.
    """ 
    
    # divisão em casos
    def caso_1(a1, a2, b1, b2):
        caso1 = (0 < a1 <= a2) and (b1 <= b2 < 0)
        return caso1
    
    def caso_11(a1, a2, b1, b2):
        caso11 = (a1*b1 >= a2*b2)
        return caso11
    
    def caso_12(a1, a2, b1, b2):  
        caso12 = (a1*b1 <= a2*b2)
        return caso12
    
    
    def caso_2(a1, a2, b1, b2):
        caso2 = (0 < a1 <= a2) and (0 < b1 <= b2)
        return caso2
    
    def caso_21(a1, a2, b1, b2):
        caso21 = (a1*b2 <= a2*b1)
        return caso21
    
    def caso_22(a1, a2, b1, b2):
        caso22 = (a1*b2 >= a2*b1)
        return caso22
    
    
    def caso_3(a1, a2, b1, b2):
        caso3 = (a1 <= a2 < 0) and (b1 <= b2 < 0)
        return caso3
    
    def caso_31(a1, a2, b1, b2):
        caso31 = (a2*b1 <= a1*b2)
        return caso31
    
    def caso_32(a1, a2, b1, b2):
        caso32 = (a2*b1 >= a1*b2)
        return caso32
    
    
    def caso_4(a1, a2, b1, b2):
        caso4 = (a1 <= a2 < 0) and (0 < b1 <= b2)
        return caso4
    
    def caso_41(a1, a2, b1, b2):
        caso41 = (a1*b1 <= a2*b2)
        return caso41
    
    def caso_42(a1, a2, b1, b2):
        caso42 = (a1*b1 >= a2*b2)
        return caso42
    
    
    def caso_5(a1, a2, b1, b2):   
        caso5 = (a1 <= 0) and (a2 >= 0) and (b1 <= b2 < 0)
        return caso5
        
        
    def caso_6(a1, a2, b1, b2):
        caso6 = (a1 <= 0) and (a2 >= 0) and (0 < b1 <= b2)
        return caso6
    
    
    alpha = np.arange(0, 1.001, 0.001).reshape((-1,1))
    
    n1 = _aplica_funcao(alpha, f_alpha_ntrifuzzy(x_num)[0])
    n2 = _aplica_funcao(alpha, f_alpha_ntrifuzzy(x_num)[1])
    d1 = _aplica_funcao(alpha, f_alpha_ntrifuzzy(x_den)[0])
    d2 = _aplica_funcao(alpha, f_alpha_ntrifuzzy(x_den)[1])
    
#    n_a =  len(x_num)
    
    quo1 = []
    quo2 = []
    
    
    count = 0
    
#     left_num  = x_num[:n_a//2 +1]
#     right_num = x_num[n_a//2:][::-1]

#     left_den  = x_den[:n_a//2 +1]
#     right_den = x_den[n_a//2:][::-1]
    
    
    #definição dos alpha-níveis
    for xn1, xn2, xd1, xd2 in zip(n1, n2, d1, d2):
    #for xn1, xn2, xd1, xd2 in zip(x_num[:(n_a)//2],x_num[(n_a)//2::-1], x_den[:(n_a)//2],x_den[(n_a)//2::-1]) :
        xn1 = float(xn1)
        xn2 = float(xn2)
        xd1 = float(xd1)
        xd2 = float(xd2)
        
        caso1 = caso_1(xn1,xn2,xd1,xd2)
        caso11 = caso_11(xn1,xn2,xd1,xd2)
        caso12 = caso_12(xn1,xn2,xd1,xd2)

        caso2 = caso_2(xn1,xn2,xd1,xd2)
        caso21 = caso_21(xn1,xn2,xd1,xd2)
        caso22 = caso_22(xn1,xn2,xd1,xd2)

        caso3 = caso_3(xn1,xn2,xd1,xd2)
        caso31 = caso_31(xn1,xn2,xd1,xd2)
        caso32 = caso_32(xn1,xn2,xd1,xd2)

        caso4 = caso_4(xn1,xn2,xd1,xd2)
        caso41 = caso_41(xn1,xn2,xd1,xd2)
        caso42 = caso_42(xn1,xn2,xd1,xd2)

        caso5 = caso_5(xn1,xn2,xd1,xd2)
        caso6 = caso_6(xn1,xn2,xd1,xd2)

    #    # definição de g-divisão
    #     def1 = (x_num == x_den * x_quo)
    #     def2 = (x_den == x_num * x_quo_inv)

        definicao = "zero"
        defs = []
        
        a1 = round(xn1,12)
        a2 = round(xn2,12)
        b1 = round(xd1,12)
        b2 = round(xd2,12)
        
        #Aplicação dos casos
        if caso1:
            if caso11:
                c1 = a2/b1
                c2 = a1/b2
                def_new = r"$N=DQ$"
            elif caso12:
                c1 = a1/b2
                c2 = a2/b1
                def_new = r"$D=NQ^{-1}$"
            else:
                raise ValueError(f"O alpha-nível de iteração {count}, com numerador {[a1, a2]} e denominador {[b1,b2]}, se encaixa no caso 1, mas não se encaixa em nenhum subcaso")
            
        elif caso2:
            if caso21:
                c1 = a1/b1
                c2 = a2/b2
                def_new = r"$N=DQ$"
            elif caso22:
                c1 = a2/b2
                c2 = a1/b1   
                def_new = r"$D=NQ^{-1}$"
            else:
                raise ValueError(f"O alpha-nível de iteração {count}, com numerador {[a1, a2]} e denominador {[b1,b2]}, se encaixa no caso 2, mas não se encaixa em nenhum subcaso")
          
        elif caso3:
            if caso31:
                c1 = a2/b2
                c2 = a1/b1
                def_new = r"$N=DQ$"
            elif caso32:
                c1 = a1/b1
                c2 = a2/b2
                def_new = r"$D=NQ^{-1}$"
            else:
                raise ValueError(f"O alpha-nível de iteração {count}, com numerador {[a1, a2]} e denominador {[b1,b2]}, se encaixa no caso 3, mas não se encaixa em nenhum subcaso")
            
        elif caso4: 
            if caso41:
                c1 = a1/b2
                c2 = a2/b1
                def_new = r"$N=DQ$"
            elif caso42:
                c1 = a2/b1
                c2 = a1/b2   
                def_new = r"$D=NQ^{-1}$"
            else:
                raise ValueError(f"O alpha-nível de iteração {count}, com numerador {[a1, a2]} e denominador {[b1,b2]}, se encaixa no caso 4, mas não se encaixa em nenhum subcaso")
        
        elif caso5:
            c1 = a2/b1
            c2 = a1/b1
            def_new = r"$N=DQ$"

        elif caso6:         
            c1 = a1/b2
            c2 = a2/b2
            def_new = r"$N=DQ$"
            
        else:
            raise ValueError(f"O alpha-nível de iteração {count}, com numerador {[a1, a2]} e denominador {[b1,b2]}, não se encaixa em nenhum caso!")
            
        quo1.append(c1)
        quo2.append(c2)
        
        if definicao != def_new:
            definicao = def_new
            defs.append(definicao)
            
        count += 1
    
    x_quo = [quo1, quo2]
    
    return x_quo, defs


def gdivisao(x_num, x_den):
    """    
    Realiza a divisão generalizada de dois números triangulares fuzzy a partir de seus alpha-niveis, segundo Stefanini (2010).

    Args:
        x_num (tuple or array-like): número triangular fuzzy x_num na forma x_num = [n1, n2], numerador na divisão.
        x_den (tuple or array-like): número triangular fuzzy x_den na forma x_den = [d1, d2], denominador na divisão.

    Returns:
        x_g_quo (list of np.ndarray): lista de vetores colunas (n x 1) representando os valores de x_g_quo em função dos alpha-níveis, em que cada vetor corresponde a um intervalo da função de pertinência por partes.
    """ 
    
    # divisão em casos
    def caso_1(a1, a2, b1, b2):
        caso1 = (0 < a1 <= a2) and (b1 <= b2 < 0)
        return caso1
    
    def caso_11(a1, a2, b1, b2):
        caso11 = (a1*b1 >= a2*b2)
        return caso11
    
    def caso_12(a1, a2, b1, b2):  
        caso12 = (a1*b1 <= a2*b2)
        return caso12
    
    
    def caso_2(a1, a2, b1, b2):
        caso2 = (0 < a1 <= a2) and (0 < b1 <= b2)
        return caso2
    
    def caso_21(a1, a2, b1, b2):
        caso21 = (a1*b2 <= a2*b1)
        return caso21
    
    def caso_22(a1, a2, b1, b2):
        caso22 = (a1*b2 >= a2*b1)
        return caso22
    
    
    def caso_3(a1, a2, b1, b2):
        caso3 = (a1 <= a2 < 0) and (b1 <= b2 < 0)
        return caso3
    
    def caso_31(a1, a2, b1, b2):
        caso31 = (a2*b1 <= a1*b2)
        return caso31
    
    def caso_32(a1, a2, b1, b2):
        caso32 = (a2*b1 >= a1*b2)
        return caso32
    
    
    def caso_4(a1, a2, b1, b2):
        caso4 = (a1 <= a2 < 0) and (0 < b1 <= b2)
        return caso4
    
    def caso_41(a1, a2, b1, b2):
        caso41 = (a1*b1 <= a2*b2)
        return caso41
    
    def caso_42(a1, a2, b1, b2):
        caso42 = (a1*b1 >= a2*b2)
        return caso42
    
    
    def caso_5(a1, a2, b1, b2):   
        caso5 = (a1 <= 0) and (a2 >= 0) and (b1 <= b2 < 0)
        return caso5
        
        
    def caso_6(a1, a2, b1, b2):
        caso6 = (a1 <= 0) and (a2 >= 0) and (0 < b1 <= b2)
        return caso6
    
    
    n1, n2 = x_num
    d1, d2 = x_den
    
#    n_a =  len(x_num)
    
    quo1 = []
    quo2 = []
    
    
    count = 0
    
#     left_num  = x_num[:n_a//2 +1]
#     right_num = x_num[n_a//2:][::-1]

#     left_den  = x_den[:n_a//2 +1]
#     right_den = x_den[n_a//2:][::-1]
    
    
    #definição dos alpha-níveis
    for xn1, xn2, xd1, xd2 in zip(n1, n2, d1, d2):
    #for xn1, xn2, xd1, xd2 in zip(x_num[:(n_a)//2],x_num[(n_a)//2::-1], x_den[:(n_a)//2],x_den[(n_a)//2::-1]) :
        xn1 = float(xn1)
        xn2 = float(xn2)
        xd1 = float(xd1)
        xd2 = float(xd2)
        
        caso1 = caso_1(xn1,xn2,xd1,xd2)
        caso11 = caso_11(xn1,xn2,xd1,xd2)
        caso12 = caso_12(xn1,xn2,xd1,xd2)

        caso2 = caso_2(xn1,xn2,xd1,xd2)
        caso21 = caso_21(xn1,xn2,xd1,xd2)
        caso22 = caso_22(xn1,xn2,xd1,xd2)

        caso3 = caso_3(xn1,xn2,xd1,xd2)
        caso31 = caso_31(xn1,xn2,xd1,xd2)
        caso32 = caso_32(xn1,xn2,xd1,xd2)

        caso4 = caso_4(xn1,xn2,xd1,xd2)
        caso41 = caso_41(xn1,xn2,xd1,xd2)
        caso42 = caso_42(xn1,xn2,xd1,xd2)

        caso5 = caso_5(xn1,xn2,xd1,xd2)
        caso6 = caso_6(xn1,xn2,xd1,xd2)

    #    # definição de g-divisão
    #     def1 = (x_num == x_den * x_quo)
    #     def2 = (x_den == x_num * x_quo_inv)

        definicao = "zero"
        defs = []
        
        a1 = round(xn1,12)
        a2 = round(xn2,12)
        b1 = round(xd1,12)
        b2 = round(xd2,12)
        
        #Aplicação dos casos
        if caso1:
            if caso11:
                c1 = a2/b1
                c2 = a1/b2
                def_new = r"$N=DQ$"
            elif caso12:
                c1 = a1/b2
                c2 = a2/b1
                def_new = r"$D=NQ^{-1}$"
            else:
                raise ValueError(f"O alpha-nível de iteração {count}, com numerador {[a1, a2]} e denominador {[b1,b2]}, se encaixa no caso 1, mas não se encaixa em nenhum subcaso")
            
        elif caso2:
            if caso21:
                c1 = a1/b1
                c2 = a2/b2
                def_new = r"$N=DQ$"
            elif caso22:
                c1 = a2/b2
                c2 = a1/b1   
                def_new = r"$D=NQ^{-1}$"
            else:
                raise ValueError(f"O alpha-nível de iteração {count}, com numerador {[a1, a2]} e denominador {[b1,b2]}, se encaixa no caso 2, mas não se encaixa em nenhum subcaso")
          
        elif caso3:
            if caso31:
                c1 = a2/b2
                c2 = a1/b1
                def_new = r"$N=DQ$"
            elif caso32:
                c1 = a1/b1
                c2 = a2/b2
                def_new = r"$D=NQ^{-1}$"
            else:
                raise ValueError(f"O alpha-nível de iteração {count}, com numerador {[a1, a2]} e denominador {[b1,b2]}, se encaixa no caso 3, mas não se encaixa em nenhum subcaso")
            
        elif caso4: 
            if caso41:
                c1 = a1/b2
                c2 = a2/b1
                def_new = r"$N=DQ$"
            elif caso42:
                c1 = a2/b1
                c2 = a1/b2   
                def_new = r"$D=NQ^{-1}$"
            else:
                raise ValueError(f"O alpha-nível de iteração {count}, com numerador {[a1, a2]} e denominador {[b1,b2]}, se encaixa no caso 4, mas não se encaixa em nenhum subcaso")
        
        elif caso5:
            c1 = a2/b1
            c2 = a1/b1
            def_new = r"$N=DQ$"

        elif caso6:         
            c1 = a1/b2
            c2 = a2/b2
            def_new = r"$N=DQ$"
            
        else:
            raise ValueError(f"O alpha-nível de iteração {count}, com numerador {[a1, a2]} e denominador {[b1,b2]}, não se encaixa em nenhum caso!")
            
        quo1.append(c1)
        quo2.append(c2)
        
        if definicao != def_new:
            definicao = def_new
            defs.append(definicao)
            
        count += 1
    
    x_quo = [quo1, quo2]
    
    return x_quo, defs


def gdivisao_nbreak(x_num, x_den):
    """    
    Realiza a divisão generalizada de dois números triangulares fuzzy a partir de seus alpha-niveis, segundo Stefanini (2010). Não interrompe a função quando há erro de divisão por zero, retornando uma lista vazia.

    Args:
        x_num (tuple or array-like): número triangular fuzzy x_num na forma x_num = [n1, n2], numerador na divisão.
        x_den (tuple or array-like): número triangular fuzzy x_den na forma x_den = [d1, d2], denominador na divisão.

    Returns:
        x_g_quo (list of np.ndarray): lista de vetores colunas (n x 1) representando os valores de x_g_quo em função dos alpha-níveis, em que cada vetor corresponde a um intervalo da função de pertinência por partes.
    """ 
    
    # divisão em casos
    def caso_1(a1, a2, b1, b2):
        caso1 = (0 < a1 <= a2) and (b1 <= b2 < 0)
        return caso1
    
    def caso_11(a1, a2, b1, b2):
        caso11 = (a1*b1 >= a2*b2)
        return caso11
    
    def caso_12(a1, a2, b1, b2):  
        caso12 = (a1*b1 <= a2*b2)
        return caso12
    
    
    def caso_2(a1, a2, b1, b2):
        caso2 = (0 < a1 <= a2) and (0 < b1 <= b2)
        return caso2
    
    def caso_21(a1, a2, b1, b2):
        caso21 = (a1*b2 <= a2*b1)
        return caso21
    
    def caso_22(a1, a2, b1, b2):
        caso22 = (a1*b2 >= a2*b1)
        return caso22
    
    
    def caso_3(a1, a2, b1, b2):
        caso3 = (a1 <= a2 < 0) and (b1 <= b2 < 0)
        return caso3
    
    def caso_31(a1, a2, b1, b2):
        caso31 = (a2*b1 <= a1*b2)
        return caso31
    
    def caso_32(a1, a2, b1, b2):
        caso32 = (a2*b1 >= a1*b2)
        return caso32
    
    
    def caso_4(a1, a2, b1, b2):
        caso4 = (a1 <= a2 < 0) and (0 < b1 <= b2)
        return caso4
    
    def caso_41(a1, a2, b1, b2):
        caso41 = (a1*b1 <= a2*b2)
        return caso41
    
    def caso_42(a1, a2, b1, b2):
        caso42 = (a1*b1 >= a2*b2)
        return caso42
    
    
    def caso_5(a1, a2, b1, b2):   
        caso5 = (a1 <= 0) and (a2 >= 0) and (b1 <= b2 < 0)
        return caso5
        
        
    def caso_6(a1, a2, b1, b2):
        caso6 = (a1 <= 0) and (a2 >= 0) and (0 < b1 <= b2)
        return caso6
    
    
    n1, n2 = x_num
    d1, d2 = x_den
    
#    n_a =  len(x_num)
    
    quo1 = []
    quo2 = []
    
    
    count = 0
    
#     left_num  = x_num[:n_a//2 +1]
#     right_num = x_num[n_a//2:][::-1]

#     left_den  = x_den[:n_a//2 +1]
#     right_den = x_den[n_a//2:][::-1]
    
    
    #definição dos alpha-níveis
    for xn1, xn2, xd1, xd2 in zip(n1, n2, d1, d2):
    #for xn1, xn2, xd1, xd2 in zip(x_num[:(n_a)//2],x_num[(n_a)//2::-1], x_den[:(n_a)//2],x_den[(n_a)//2::-1]) :
        xn1 = float(xn1)
        xn2 = float(xn2)
        xd1 = float(xd1)
        xd2 = float(xd2)
        
        caso1 = caso_1(xn1,xn2,xd1,xd2)
        caso11 = caso_11(xn1,xn2,xd1,xd2)
        caso12 = caso_12(xn1,xn2,xd1,xd2)

        caso2 = caso_2(xn1,xn2,xd1,xd2)
        caso21 = caso_21(xn1,xn2,xd1,xd2)
        caso22 = caso_22(xn1,xn2,xd1,xd2)

        caso3 = caso_3(xn1,xn2,xd1,xd2)
        caso31 = caso_31(xn1,xn2,xd1,xd2)
        caso32 = caso_32(xn1,xn2,xd1,xd2)

        caso4 = caso_4(xn1,xn2,xd1,xd2)
        caso41 = caso_41(xn1,xn2,xd1,xd2)
        caso42 = caso_42(xn1,xn2,xd1,xd2)

        caso5 = caso_5(xn1,xn2,xd1,xd2)
        caso6 = caso_6(xn1,xn2,xd1,xd2)

    #    # definição de g-divisão
    #     def1 = (x_num == x_den * x_quo)
    #     def2 = (x_den == x_num * x_quo_inv)

        definicao = "zero"
        defs = []
        
        a1 = round(xn1,12)
        a2 = round(xn2,12)
        b1 = round(xd1,12)
        b2 = round(xd2,12)
        
        #Aplicação dos casos
        if caso1:
            if caso11:
                c1 = a2/b1
                c2 = a1/b2
                def_new = r"$N=DQ$"
            elif caso12:
                c1 = a1/b2
                c2 = a2/b1
                def_new = r"$D=NQ^{-1}$"
            else:
                raise ValueError(f"O alpha-nível de iteração {count}, com numerador {[a1, a2]} e denominador {[b1,b2]}, se encaixa no caso 1, mas não se encaixa em nenhum subcaso")
            
        elif caso2:
            if caso21:
                c1 = a1/b1
                c2 = a2/b2
                def_new = r"$N=DQ$"
            elif caso22:
                c1 = a2/b2
                c2 = a1/b1   
                def_new = r"$D=NQ^{-1}$"
            else:
                raise ValueError(f"O alpha-nível de iteração {count}, com numerador {[a1, a2]} e denominador {[b1,b2]}, se encaixa no caso 2, mas não se encaixa em nenhum subcaso")
          
        elif caso3:
            if caso31:
                c1 = a2/b2
                c2 = a1/b1
                def_new = r"$N=DQ$"
            elif caso32:
                c1 = a1/b1
                c2 = a2/b2
                def_new = r"$D=NQ^{-1}$"
            else:
                raise ValueError(f"O alpha-nível de iteração {count}, com numerador {[a1, a2]} e denominador {[b1,b2]}, se encaixa no caso 3, mas não se encaixa em nenhum subcaso")
            
        elif caso4: 
            if caso41:
                c1 = a1/b2
                c2 = a2/b1
                def_new = r"$N=DQ$"
            elif caso42:
                c1 = a2/b1
                c2 = a1/b2   
                def_new = r"$D=NQ^{-1}$"
            else:
                raise ValueError(f"O alpha-nível de iteração {count}, com numerador {[a1, a2]} e denominador {[b1,b2]}, se encaixa no caso 4, mas não se encaixa em nenhum subcaso")
        
        elif caso5:
            c1 = a2/b1
            c2 = a1/b1
            def_new = r"$N=DQ$"

        elif caso6:         
            c1 = a1/b2
            c2 = a2/b2
            def_new = r"$N=DQ$"
            
        else:
#             raise ValueError(f"O alpha-nível de iteração {count}, com numerador {[a1, a2]} e denominador {[b1,b2]}, não se encaixa em nenhum caso!")
            print(f"O alpha-nível de iteração {count}, com numerador {[a1, a2]} e denominador {[b1,b2]}, não se encaixa em nenhum caso!")
            def_new = "$0 \in [a1,a2]$"
            defs.append(def_new)
            return [], defs
        
        quo1.append(c1)
        quo2.append(c2)
        
        if definicao != def_new:
            definicao = def_new
            defs.append(definicao)
            
        count += 1
    
    x_quo = [quo1, quo2]
    
    return x_quo, defs




###############################################################################
###########################   Outras Operações   ##############################
###############################################################################

def oposto_fuzzy(A):
    """
    Calcula o número fuzzy A*{-1} oposto a A.

    Args:
        A (array-like): número fuzzy A.
        
    Returns:
        opA (array): número fuzzy oposto a A (A*-1).
    """
    opA = np.array(A[::-1], dtype=float) * (-1)
    return opA


def exp_fuzzy(A):
    """
    Calcula a exponencial de um número fuzzy A.

    Args:
        A (array-like): número fuzzy A.
        
    Returns:
        expA (array): número fuzzy exponencial de A.
    """
    a1, a2 = A
    e1 = np.zeros(len(a1))
    e2 = np.zeros(len(a1))
    
    for alpha in range(len(a1)):
        p = [np.exp(a1[alpha]), np.exp(a2[alpha])]
        e1[alpha] = min(p)
        e2[alpha] = max(p)
        
    expA = [e1, e2]
    return expA


def sen_fuzzy(A):
    """
    Calcula o seno de um número fuzzy A.

    Args:
        A (array-like): número fuzzy A.
        
    Returns:
        senA (array): número fuzzy seno de A.
    """
    a1, a2 = A
    s1 = np.zeros(len(a1))
    s2 = np.zeros(len(a1))
    
    for alpha in range(len(a1)):
        p = [math.sin(a1[alpha]), math.sin(a2[alpha])]
        s1[alpha] = min(p)
        s2[alpha] = max(p)
        
    senA = [s1, s2]
    return senA


def cos_fuzzy(A):
    """
    Calcula o cosseno de um número fuzzy A.

    Args:
        A (array-like): número fuzzy A.
        
    Returns:
        cosA (array): número fuzzy cosseno de A.
    """
    a1, a2 = A
    c1 = np.zeros(len(a1))
    c2 = np.zeros(len(a1))
    
    for alpha in range(len(a1)):
        p = [math.cos(a1[alpha]), math.cos(a2[alpha])]
        c1[alpha] = min(p)
        c2[alpha] = max(p)
        
    cosA = [c1, c2]
    return cosA

###############################################################################
###############################   Métricas   ##################################
###############################################################################

def metr_hausdorff(x1, x2):
    """ Calcula a métrica de Hausdorff para um par de dados fuzzy.
    
    Arg: 
        x1, x2 (list of list): dados fuzzy.
    
    Return: 
        mhaus (float): métrica de Hausdorff calculada.
    
    """
    
    n_alphas = len(x1[0])
    beta = []
    
    #beta = (max|a1 - b1|, |a2 - b2|)
    for alpha in range(n_alphas):
        beta.append(max(abs(x1[0][alpha] - x2[0][alpha]), abs(x1[1][alpha] - x2[1][alpha])))
    
    # sup beta
    mhaus=max(beta)
    
    return mhaus 


###############################################################################
##########################   Método da Secante   ##############################
###############################################################################


def msecante(f, p0, p1, eps=1e-7, n=1000):
    """ Aplica o método da secante em problemas crisp.
    
    Arg: 
        f (function): função da qual se busca a raíz.
        p0 (int or float): chute inicial 0.
        p1 (int or float): chute inicial 1.
        eps (float) = tolerância de erro/aproximação.
        n = número máximo de iterações.
    
    Return: 
        p (list) = chutes a cada iteração.
        erros (list) = diferença entre iterações consecutivas a cada iteração.
    
    """
    
    p = [p0, p1]
    erros = [abs(p1-p0)]
    
    for i in range(1,n):
        p0 = p[-2]
        p1 = p[-1]
        
        f0 = f(p0)
        f1 = f(p1)
        
        pi = p1 - f1*(p1-p0)/(f1-f0)
        erro = abs(pi - p1)
        erros.append(erro)
        
        if erro < eps:
            print(f"O método convergiu na iteração {i-1}.")
            return p, erros

        else:
            pi = round(pi, int(-math.log10(eps)))
            
            p.append(pi)

    print(f"O método não convergiu em {i} iterações.")
    return p, erros


def iter_msecfuzzy(p0, p1, f0, f1): #vou precisar implementar a soma e diferença 0 para números não triangulares
    """ Realiza um cálculo de iteração do método da secante fuzzy.
    
    Arg: 
        p0 (array-like): chute inicial fuzzy 0.
        p1 (array-like): chute inicial fuzzy 1.
        p0 (array-like): função aplicada ao chute inicial fuzzy 0.
        p1 (array-like): função aplicada ao chute inicial fuzzy 1.
        
    Return: 
        pi (array-like) = novo ponto pi definido na iteração.
        defs (list) = definições de divisão fuzzy generalizada cumpridas a cada alfa nível.
    
    """
    #pi = p1 - f1*(p1-p0)/(f1-f0)
        
    p_dif = dif_interativa_gamma(p1, p0) #(p1-p0)
    f_prod = prod_fuzzy(f1, p_dif) #f1*(p1-p0)

    f_dif = dif_interativa_gamma(f1, f0) #(f1-f0)
    f_div, defs = gdivisao_nbreak(f_prod, f_dif) #f1*(p1-p0)/(f1-f0)
    if "$0 \in [a1,a2]$" in defs:
        return [], defs
    pi = dif_interativa_gamma(p1, f_div) #pi = p1 - f1*(p1-p0)/(f1-f0)
    
    return pi, defs


def msecante_fuzzy(f, p0, p1, eps=1e-7, n=100):
     """ Aplica o método da secante em problemas fuzzy.
    
    Arg: 
        f (function): função da qual se busca a raíz.
        p0 (int or float or array-like): chute inicial 0.
        p1 (int or float or array-like): chute inicial 1.
        eps (float) = tolerância de erro/aproximação.
        n = número máximo de iterações.
    
    Return: 
        p (list) = chutes a cada iteração.
        erros (list) = diferença entre iterações consecutivas a cada iteração.
    
    """
    
    if isinstance(p0, (int, float)):
        p0 = crisp2fuzzy(p0)
        
    if isinstance(p1, (int, float)):
        p1 = crisp2fuzzy(p1)
        
    p = [p0, p1]
    erros = [metr_hausdorff(p0, p1)]
    
    # print(f"Iteração {0}: {p}")
    # print("------------------------------") 
    
    for i in range(1, n):
        p0 = p[-2]
        p1 = p[-1]      
        
        f0 = f(p0)
        f1 = f(p1)
        
        #pi = p1 -_0 f1 * (p1 -_0 p0) /_g (f1 -_0 f0)
        pi, defs = iter_msecfuzzy(p0, p1, f0, f1)
        
        if "$0 \in [a1,a2]$" in defs:
            print(f"O método apresentou erro de divisão por zero na iteração {i-1}.")
            p.append(pi)
            return p, erros
        
        erro = metr_hausdorff(pi, p1)
        erros.append(erro)
        
        if  erro < eps:
            print(f"O método convergiu na iteração {i-1}.")
            return p, erros

        else:
            pi = np.round(pi, int(-math.log10(eps)))
            
            p.append(pi)
            
            # print(f"Iteração {i}: {pi}")
            # print("------------------------------") 
            
    # print(" ")
    # print("------------------------------")        
    print(f"O método não convergiu em {n} iterações.")
    return p, erros