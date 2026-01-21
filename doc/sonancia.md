# Sonancia

## Concepto
"Sonancia" es un término acuñado para describir una medida de consonancia/disonancia armónica basada en un enfoque híbrido que combina el cálculo offline de un "Kernel" de consonancia con la aplicación online sobre el espectro de una señal de audio.

El objetivo es obtener una métrica que capture la "suavidad" o "rugosidad" (consonancia/disonancia) de un sonido en tiempo real, similar a las curvas de disonancia de Sethares pero optimizado para ejecución rápida mediante pre-cálculo.

## Fundamento Matemático

### 1. Generación del Kernel (Offline)
El Kernel es un vector que representa la consonancia inherente de diferentes intervalos musicales. Se construye calculando la autocorrelación de la suma de dos ondas sinusoidales para una serie de ratios de frecuencia.

Para cada ratio $r_i$ (donde $r_i = f_2 / f_1$):
1.  Generamos dos senoides: $s_1(t) = \sin(2\pi t)$ y $s_2(t) = \sin(2\pi r_i t)$.
2.  Sumamos las señales: $x(t) = s_1(t) + s_2(t)$.
3.  Calculamos la autocorrelación de $x(t)$.
4.  El valor del Kernel para el ratio $r_i$ es el máximo de la autocorrelación (excluyendo el lag 0, o normalizado de alguna manera que refleje la periodicidad/consonancia).
    *   *Nota:* En la conversación original, se sugiere que la altura de los picos de autocorrelación correlaciona con la consonancia (ratios simples generan patrones repetitivos claros).

Matemáticamente:
$$ K[i] = \max_{\tau > 0} (R_{xx}(\tau)) $$
donde $x(t) = \sin(2\pi t) + \sin(2\pi r_i t)$ y $R_{xx}$ es la autocorrelación.

### 2. Aplicación Online
Para medir la "sonancia" de una señal de entrada $y(t)$:
1.  Obtenemos su representación espectral, preferiblemente usando **CQT (Constant-Q Transform)** para mantener una resolución logarítmica acorde a la percepción musical.
2.  Calculamos la autocorrelación del espectro (o las distancias entre picos espectrales).
3.  Comparamos este patrón con el Kernel pre-calculado.

$$ S = \text{Kernel} \cdot \text{Autocorr}(\text{CQT}(y)) $$

Un valor alto de $S$ indica que los intervalos presentes en la señal coinciden con los intervalos consonantes del Kernel.

## Implementación
La implementación se dividirá en:
1.  `SonanciaKernel`: Clase para generar y almacenar el Kernel.
2.  `SonanciaMeter`: Clase (o método) para calcular la sonancia de una señal en tiempo real o batch.
