# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 18:59:49 2020

@author: dani_
"""
# Librerías
import pandas as pd
from matplotlib import pyplot as plt
import math
import numpy as np
from scipy.optimize import curve_fit

# Estilo gráficas
plt.style.use('bmh')

# Importamos los datos en formato .csv para crear el dataframe
df = pd.read_csv('datos_provincias.csv', parse_dates=['fecha'], index_col=['fecha']) # Columna fecha como dateTime e Index_col
print(df.columns) # Mostramos las columnas del dataframe

# Selección del intervalo de tiempo, código ISO de la provincia y columna 'número de casos'
inicio, final = '2020-03-01', '2020-04-30' # Fechas de inicio y final a considerar
cc = df.loc[df['provincia_iso'] == 'CC']['num_casos'].loc[pd.Timestamp(inicio):pd.Timestamp(final)] # Cáceres
ba = df.loc[df['provincia_iso'] == 'BA']['num_casos'].loc[pd.Timestamp(inicio):pd.Timestamp(final)] # Badajoz
sa = df.loc[df['provincia_iso'] == 'SA']['num_casos'].loc[pd.Timestamp(inicio):pd.Timestamp(final)] # Salamanca
av = df.loc[df['provincia_iso'] == 'AV']['num_casos'].loc[pd.Timestamp(inicio):pd.Timestamp(final)] # Ávila
to = df.loc[df['provincia_iso'] == 'TO']['num_casos'].loc[pd.Timestamp(inicio):pd.Timestamp(final)] # Toledo
cr = df.loc[df['provincia_iso'] == 'CR']['num_casos'].loc[pd.Timestamp(inicio):pd.Timestamp(final)] # Ciudad Real
co = df.loc[df['provincia_iso'] == 'CO']['num_casos'].loc[pd.Timestamp(inicio):pd.Timestamp(final)] # Córdoba
se = df.loc[df['provincia_iso'] == 'SE']['num_casos'].loc[pd.Timestamp(inicio):pd.Timestamp(final)] # Sevilla
h = df.loc[df['provincia_iso'] == 'H']['num_casos'].loc[pd.Timestamp(inicio):pd.Timestamp(final)] # Huelva

#############################################
# Análisis y ajuste de los contagios diarios
#############################################

# Gráfica de la evolución de los datos de las provincias limítrofes de Cáceres
cc.plot(label='CC')
sa.plot(label='SA')
av.plot(label='AV')
to.plot(label='TO')
ba.plot(label='BA')
plt.title('Curvas de contagios de las provincias limítrofes de Cáceres')
plt.ylabel('Nº Contagios Diarios')
plt.legend()
plt.savefig('Imágenes/DiariosCáceres.png', dpi=200)
plt.show()

# Gráfica de la evolución de los datos de las provincias limítrofes de Badajoz
ba.plot(label='BA')
cc.plot(label='CC')
to.plot(label='TO')
cr.plot(label='CR')
co.plot(label='CO')
se.plot(label='SE')
h.plot(label='H')
plt.title('Curvas de contagios de las provincias limítrofes de Bádajoz')
plt.ylabel('Nº Contagios Diarios')
plt.legend()
plt.savefig('Imágenes/DiariosBadajoz.png', dpi=200)
plt.show()

# Funciones para el modelo de ajuste del Nº de Contagios diarios
def gaussian(x,a,b,c): # Función Gaussiana
  return a*math.e**(-(x-b)**2/(2*math.pow(c, 2)))

def r2(data,funcion, xdata, popt): # R^2
    residuals = data.values - funcion(xdata, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((data.values-np.mean(data.values))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared
    
def plotGaussianFit(data, name): # Ajuste Gaussiano
    plt.plot(data.values) # Gráfica de los datos reales
    xdata = np.linspace(0, len(data), len(data)) # Eje X
    popt, pcov  = curve_fit(gaussian, xdata, data.values) # Ajuste
    r_squared = r2(data, gaussian, xdata, popt)
    plt.plot(xdata, gaussian(xdata, *popt), 'g--', # Gráfica del ajuste
             label='fit: a=%5.3f, b=%5.3f, c=%5.3f,' % tuple(popt)+' $R^2={:.3f}$'.format(r_squared))
    plt.title('Modelo de ajuste Nº de contagios de {}'.format(name))
    plt.ylabel('Nº Contagios Diarios')
    plt.legend()
    plt.savefig('Imágenes/Gaussian{}.png'.format(name), dpi=200)
    plt.show()

# Cáceres
plotGaussianFit(cc, 'Cáceres')
plotGaussianFit(sa, 'Salamanca')
plotGaussianFit(av, 'Ávila')
plotGaussianFit(to, 'Toledo')
plotGaussianFit(ba, 'Badajoz')

# Badajoz
plotGaussianFit(cc, 'Cáceres')
plotGaussianFit(to, 'Toledo')
plotGaussianFit(cr, 'Ciudad Real')
plotGaussianFit(co, 'Córdoba')
plotGaussianFit(se, 'Sevilla')
plotGaussianFit(h, 'Huelva')

########################################
# Análisis y ajuste de datos acumulados
########################################

# Datos acumulados por provincias
cc_acumulado = cc.cumsum()
ba_acumulado = ba.cumsum()
sa_acumulado = sa.cumsum()
av_acumulado = av.cumsum()
to_acumulado = to.cumsum()
cr_acumulado = cr.cumsum()
co_acumulado = co.cumsum()
se_acumulado = se.cumsum()
h_acumulado = h.cumsum()

# Gráfica de datos acumulados de las provincias limítrofes de Cáceres
cc_acumulado.plot(label='CC')
sa_acumulado.plot(label='SA')
av_acumulado.plot(label='AV')
to_acumulado.plot(label='TO')
ba_acumulado.plot(label='BA')
plt.title('Datos acumulados de las provincias limítrofes de Cáceres')
plt.ylabel('Nº Contagios Acumulados')
plt.legend()
plt.savefig('Imágenes/AcumuladosCáceres.png', dpi=200)
plt.show()

# Gráfica de datos acumulados de las provincias limítrofes de Badajoz
ba_acumulado.plot(label='BA')
cc_acumulado.plot(label='CC')
to_acumulado.plot(label='TO')
cr_acumulado.plot(label='CR')
co_acumulado.plot(label='CO')
se_acumulado.plot(label='SE')
h_acumulado.plot(label='H')
plt.title('Datos acumulados de las provincias limítrofes de Bádajoz')
plt.ylabel('Nº Contagios Acumulados')
plt.legend()
plt.savefig('Imágenes/AcumuladosBadajoz.png', dpi=200)
plt.show()


# Funciones para los modelos de ajuste para Datos acumulados
def Gompertz(x,a,b,c): # Función Gompertz
  return a*(math.e**(-b*math.e**(-c*x)))

def logistica(x,a,b,c): # Función Logística
    return c/ (1 + math.e**(-a*x + b))

def plotGompertzLogisticaFit(data, name): # Ajuste con Gompertz y Logística
    plt.plot(data.values) # Gráfica de los datos reales
    xdata = np.linspace(0, len(data), len(data)) # Eje X
    popt, pcov  = curve_fit(Gompertz, xdata, data.values) # Gompertz
    r_squared = r2(data, Gompertz, xdata, popt) # R^2
    plt.plot(xdata, Gompertz(xdata, *popt), 'g--', # Gráfica del ajuste
             label='Gompertz: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt)+' $R^2={:.3f}$'.format(r_squared))
    popt, pcov  = curve_fit(logistica, xdata, data.values) # Logística
    r_squared = r2(data, logistica, xdata, popt) # R^2
    plt.plot(xdata, logistica(xdata, *popt), 'r--',
         label='Logística: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt)+' $R^2={:.3f}$'.format(r_squared))
    plt.title('Modelos de ajuste de la incidencia acumulada de {}'.format(name))
    plt.ylabel('Nº Contagios Acumulados')
    plt.legend()
    plt.savefig('Imágenes/GomperetzLogistica{}.png'.format(name), dpi=200)
    plt.show()

# Cáceres
plotGompertzLogisticaFit(cc_acumulado, 'Cáceres')
plotGompertzLogisticaFit(sa_acumulado, 'Salamanca')
plotGompertzLogisticaFit(av_acumulado, 'Ávila')
plotGompertzLogisticaFit(to_acumulado, 'Toledo')
plotGompertzLogisticaFit(ba_acumulado, 'Badajoz')

# Badajoz
plotGompertzLogisticaFit(ba_acumulado, 'Badajoz')
plotGompertzLogisticaFit(cc_acumulado, 'Cáceres')
plotGompertzLogisticaFit(to_acumulado, 'Toledo')
plotGompertzLogisticaFit(cr_acumulado, 'Ciudad Real')
plotGompertzLogisticaFit(co_acumulado, 'Córdoba')
plotGompertzLogisticaFit(se_acumulado, 'Sevilla')
plotGompertzLogisticaFit(h_acumulado, 'Huelva')
