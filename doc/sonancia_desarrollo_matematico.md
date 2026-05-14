# Sonancia

## Consonancia y disonancia como propiedad matemática del comportamiento ondulatorio

---

## 1. Motivación

La consonancia y disonancia musical se han definido históricamente desde dos aproximaciones: la subjetiva (psicoacústica, rugosidad perceptual según Plomp-Levelt, modelo de Sethares) y la aritmética (Gradus Suavitatis de Euler, medidas basadas en la complejidad del ratio). Ambas dependen de parámetros ajustados a datos experimentales o de elecciones arbitrarias.

Este trabajo deriva una función de sonancia continua **directamente desde la física ondulatoria**, sin apelar a percepción, neurociencia ni constantes ajustadas. El resultado central es que la jerarquía armónica clásica emerge como propiedad intrínseca de la autocorrelación de dos ondas sinusoidales.

---

## 2. Señal compuesta y autocorrelación

### 2.1 Definición de la señal

Dados dos tonos puros con frecuencias $f_1$ (fundamental) y $f_2$ (relativa), con ratio $r = f_2/f_1$:

$$s(t) = \sin(2\pi f_1 t + \varphi_1) + \sin(2\pi f_2 t + \varphi_2)$$

### 2.2 Autocorrelación

La autocorrelación normalizada de $s(t)$ es:

$$R(\tau) = \lim_{T \to \infty} \frac{1}{T} \int_0^T s(t) \cdot s(t+\tau)\, dt$$

Expandiendo el producto $s(t) \cdot s(t+\tau)$ se obtienen cuatro términos. Los términos cruzados $\sin(\omega_1 t) \cdot \sin(\omega_2(t+\tau))$ se anulan en el límite $T \to \infty$ por ortogonalidad de sinusoides de frecuencia distinta. Los auto-términos producen:

$$R(\tau) = \frac{1}{2}\cos(2\pi f_1 \tau) + \frac{1}{2}\cos(2\pi f_2 \tau)$$

### 2.3 Independencia de fase

Las fases $\varphi_1$, $\varphi_2$ no aparecen en $R(\tau)$. Demostración: en los términos cruzados, las fases se cancelan en la integración. En los auto-términos, aparecen como $\cos(\omega_i \tau + \varphi_i - \varphi_i) = \cos(\omega_i \tau)$. La autocorrelación es inherentemente independiente de fase.

---

## 3. Evaluación en lags enteros

### 3.1 Simplificación

Evaluamos $R(\tau)$ en múltiplos enteros del periodo fundamental, $\tau = n \cdot T_1$ donde $n \in \mathbb{Z}^+$ y $T_1 = 1/f_1$:

$$R(n) = \frac{1}{2}\cos(2\pi f_1 \cdot n T_1) + \frac{1}{2}\cos(2\pi f_2 \cdot n T_1)$$

Dado que $\cos(2\pi n) = 1$ para todo $n$ entero:

$$R(n) = \frac{1}{2} + \frac{1}{2}\cos(2\pi r n)$$

La componente $f_1$ contribuye siempre $\frac{1}{2}$: desaparece como portadora de información. Toda la estructura de sonancia reside en $\cos(2\pi r n)$.

### 3.2 Identidad fundamental

Aplicando $\frac{1}{2} + \frac{1}{2}\cos\theta = \cos^2(\theta/2)$:

$$\boxed{R(n) = \cos^2(\pi r n)}$$

Esta identidad exacta es la piedra angular del desarrollo.

### 3.3 Condición de recurrencia

$R(n) = 1$ (recurrencia perfecta del patrón de interferencia) requiere $\cos^2(\pi r n) = 1$, es decir $\pi r n = m\pi$ para algún entero $m$, lo que equivale a que $rn$ sea entero.

Para $r = p/q$ racional reducido ($\gcd(p,q) = 1$): $rn = pn/q$ es entero cuando $q \mid n$. El menor tal $n$ es $n = q$.

**El tiempo mínimo de recurrencia es $\tau = q$ periodos de $f_1$**, donde $q$ es el denominador de la fracción reducida.

---

## 4. Función de sonancia

### 4.1 Construcción

La función $1/q$ mide la sonancia en los racionales, pero es discontinua. Para extenderla a los reales de forma continua, elevamos $R(n)$ a una potencia $k$ que actúa como filtro de resolución:

$$\boxed{S_k(r) = \sum_{n=1}^{N} \frac{\cos^{2k}(\pi r n)}{n}}$$

El peso $1/n$ asegura que recurrencias tempranas (lag $n$ pequeño) contribuyan más que recurrencias tardías.

### 4.2 Propiedades inmediatas

**Continuidad.** $S_k$ es $C^\infty$ para todo $k \in \mathbb{N}$, al ser suma finita de funciones $C^\infty$.

**Positividad.** $\cos^{2k}(\pi r n) \geq 0$ para todo $r$, $n$, $k$. Así $S_k(r) \geq 0$.

**Simetría.** $S_k(r) = S_k(r + 2)$ (periodicidad, ya que $\cos^{2k}(\pi(r+2)n) = \cos^{2k}(\pi r n)$).

---

## 5. Convergencia: $S_k \to 1/q$

### 5.1 Clasificación de lags

Para $r = p/q$ racional reducido, los lags se clasifican en dos tipos:

**Lags resonantes** ($n = mq$, con $m \in \mathbb{Z}^+$): $\cos^2(\pi p m) = \cos^2(m\pi p) = 1$ para todo $k$. Contribución: $1/(mq)$.

**Lags no-resonantes** ($q \nmid n$): $\cos^2(\pi p n / q) < 1$ estrictamente, ya que $pn/q \notin \mathbb{Z}$. Entonces $\cos^{2k} \to 0$ cuando $k \to \infty$.

### 5.2 Límite

$$\lim_{k \to \infty} S_k(p/q) = \sum_{m=1}^{\lfloor N/q \rfloor} \frac{1}{mq} = \frac{1}{q} \cdot H_{\lfloor N/q \rfloor}$$

Normalizado por $H_N = \sum_{n=1}^N 1/n$:

$$\lim_{k \to \infty} \frac{S_k(p/q)}{H_N} = \frac{1}{q}$$

El denominador de la fracción reducida emerge como límite.

### 5.3 Verificación del ranking

Para los intervalos de la escala cromática en entonación justa:

| Intervalo | Ratio | q | 1/q | Ranking |
|-----------|-------|---|-----|---------|
| P8 (Octava) | 2/1 | 1 | 1.0000 | 1 |
| P5 (Quinta) | 3/2 | 2 | 0.5000 | 2 |
| P4 (Cuarta) | 4/3 | 3 | 0.3333 | 3 |
| M6 (Sexta mayor) | 5/3 | 3 | 0.3333 | 3 |
| M3 (Tercera mayor) | 5/4 | 4 | 0.2500 | 5 |
| m3 (Tercera menor) | 6/5 | 5 | 0.2000 | 6 |
| m6 (Sexta menor) | 8/5 | 5 | 0.2000 | 6 |
| m7 (Séptima menor) | 9/5 | 5 | 0.2000 | 6 |
| M2 (Segunda mayor) | 9/8 | 8 | 0.1250 | 9 |
| M7 (Séptima mayor) | 15/8 | 8 | 0.1250 | 9 |
| m2 (Segunda menor) | 16/15 | 15 | 0.0667 | 11 |
| TT (Tritono) | 45/32 | 32 | 0.0313 | 12 |

Correlación de Kendall $\tau = 1.000$ entre el ranking de $S_k$ (con $k \geq 20$, $N \geq 48$) y el de $1/q$.

---

## 6. Propiedades del kernel $\cos^{2k}$

### 6.1 Kernel de Fejér generalizado

$\cos^{2k}(x)$ tiene la expansión de Fourier:

$$\cos^{2k}(x) = a_0 + \sum_{j=1}^{k} a_j \cos(2jx)$$

con coeficientes:

$$a_0 = \frac{1}{4^k}\binom{2k}{k}, \qquad a_j = \frac{2}{4^k}\binom{2k}{k-j} \quad (j \geq 1)$$

### 6.2 Aproximación gaussiana

Para $k$ grande y $x$ pequeño:

$$\cos^{2k}(x) \approx e^{-kx^2}$$

Esta es una gaussiana con desviación estándar $\sigma = 1/\sqrt{2k}$.

**Ancho a media altura (FWHM):**

$$\text{FWHM} = \frac{2\sqrt{\ln 2}}{\sqrt{k} \cdot \pi} \approx \frac{0.53}{\sqrt{k}}$$

### 6.3 Estructura de picos de $S_k(r)$

Como función de $r$ (para $n$ fijo), $\cos^{2k}(\pi r n)$ tiene picos en $r = m/n$ (todos los racionales con denominador $n$). Dentro de $S_k$:

- **Altura del pico** en $r = p/q$: dominada por el término $n = q$, que da $\cos^{2k}(\pi p) / q = 1/q$.
- **Ancho del pico**: $\propto 1/(q\sqrt{k})$ — más estrecho para $q$ grande.
- **Resolución**: $k$ controla cuánta estructura fina se resuelve, análogo a la apertura de un instrumento óptico.

---

## 7. Perfil de resonancia

### 7.1 Forma del pico

Para $r = p/q + \delta$ (desplazamiento $\delta$ respecto al racional), la autocorrelación en lag $n = q$ vale:

$$R(q) = \frac{1}{2} + \frac{1}{2}\cos(2\pi \delta q) \approx 1 - 2\pi^2 \delta^2 q^2$$

Este es un perfil parabólico (Lorentziano en segunda aproximación) con:

$$\text{Ancho natural:} \quad \sigma = \frac{1}{\pi q \sqrt{2}}$$

### 7.2 Implicación

Los racionales simples ($q$ pequeño) generan resonancias anchas que dominan un entorno amplio. Los racionales complejos ($q$ grande) generan resonancias estrechas, difíciles de "encontrar". Esta jerarquía emerge directamente de la física, sin ser impuesta.

---

## 8. El parámetro $k$ como resolución

### 8.1 Interpretación

$k$ no es una constante del modelo ajustada a datos. Es un **parámetro de resolución** que determina cuánta estructura armónica se distingue:

- $k = 2$: solo octavas visibles.
- $k = 5$: quintas y cuartas emergen.
- $k = 10$: terceras y sextas.
- $k = 20$: escala cromática completa.
- $k = 50$: estructura microtonal.

### 8.2 Resolución vs denominador

La resolución necesaria para "separar" un racional con denominador $q$ es:

$$k \gtrsim q^2$$

ya que el error $|S_k(p/q)/H_N - 1/q|$ decae como $\exp(-4\pi^2 k / q^2)$.

### 8.3 Analogías

| Marco | Parámetro de resolución | Significado |
|-------|------------------------|-------------|
| Sonancia | $k$ | Resolución armónica |
| Óptica | Apertura | Resolución angular |
| Suavización prima | $B$ | Primos máximos resueltos |
| Umbral $\varepsilon$ | $\varepsilon$ | Tolerancia de recurrencia |

---

## 9. Conexión con la familia $S_\varepsilon$

### 9.1 Definición alternativa

$$S_\varepsilon(r) = \frac{1}{\tau_\varepsilon(r)}$$

donde $\tau_\varepsilon(r) = \min\{n \in \mathbb{Z}^+ : \cos^2(\pi r n) > 1 - \varepsilon\}$.

### 9.2 Convergencia

Para $r = p/q$ reducido: $\cos^2(\pi p) = 1 > 1 - \varepsilon$ para todo $\varepsilon > 0$, tomado en $n = q$. Así $\tau_\varepsilon(p/q) = q$ y $S_\varepsilon(p/q) = 1/q$.

### 9.3 Ancho de captura

Para $r = p/q + \delta$: $\cos^2(\pi(p + q\delta)) > 1 - \varepsilon$ requiere $|q\delta| < \sqrt{\varepsilon}/\pi$. El ancho del intervalo donde $S_\varepsilon$ asigna al racional $p/q$ es:

$$\sigma_\varepsilon(q) = \frac{\sqrt{\varepsilon}}{\pi q}$$

### 9.4 Relación $k \leftrightarrow \varepsilon$

Ambas parametrizaciones son equivalentes: $k$ grande corresponde a $\varepsilon$ pequeño. La relación aproximada es $\varepsilon \sim 1/k$: ambos controlan la selectividad del filtro que separa recurrencias verdaderas de fluctuaciones.

---

## 10. Extensión a tríadas

### 10.1 Tres tonos

Para tres tonos $f_1$, $f_2$, $f_3$ con ratios $r_1 = f_2/f_1$ y $r_2 = f_3/f_1$, existen tres relaciones de frecuencia por pares:

- $f_1 \leftrightarrow f_2$: ratio $r_1$
- $f_1 \leftrightarrow f_3$: ratio $r_2$
- $f_2 \leftrightarrow f_3$: ratio $r_2/r_1$

### 10.2 Sonancia total

$$S_{\text{total}} = S_k(r_1) + S_k(r_2) + S_k(r_2/r_1)$$

### 10.3 Generalización a $n$ tonos

Para $n$ tonos con ratios $r_1, r_2, \ldots, r_{n-1}$ respecto a la fundamental:

$$S_{\text{total}} = \sum_{i < j} S_k(r_j / r_i)$$

donde $r_0 = 1$ (la fundamental). Esto suma sobre los $\binom{n}{2}$ pares posibles.

---

## 11. Serie de Dirichlet $F(s, q, k, p)$

### 11.1 Definición

Generalizamos $S_k$ reemplazando el peso $1/n$ por $1/n^s$ con $s \in \mathbb{C}$:

$$F(s) = \sum_{n=1}^{\infty} \frac{\cos^{2k}(\pi p n / q)}{n^s}$$

Nuestra sonancia es $F(1)$.

### 11.2 Descomposición en funciones zeta de Hurwitz

Usando la expansión de Fourier de $\cos^{2k}$:

$$F(s) = a_0 \cdot \zeta(s) + \sum_{j=1}^{k} a_j \cdot \operatorname{Re}\!\left[\operatorname{Li}_s\!\left(e^{2\pi i j p / q}\right)\right]$$

donde $\operatorname{Li}_s(z) = \sum_{n=1}^\infty z^n / n^s$ es el polilogaritmo. Cada $\operatorname{Li}_s(e^{2\pi i m/q})$ se expresa en funciones zeta de Hurwitz:

$$\operatorname{Li}_s(e^{2\pi i m/q}) = q^{-s} \sum_{a=0}^{q-1} e^{2\pi i a m / q} \cdot \zeta(s, a/q)$$

### 11.3 Continuación meromorfa

$F(s)$ hereda la continuación meromorfa de $\zeta(s)$ y $\zeta(s, a/q)$ a todo $\mathbb{C}$, con un único polo simple en $s = 1$.

### 11.4 Residuo en $s = 1$

$$\operatorname{Res}_{s=1} F = a_0 + \sum_{\substack{m \geq 1 \\ mq \leq k}} a_{mq}$$

Los términos $\operatorname{Li}_s(e^{2\pi i j p / q})$ tienen polo en $s = 1$ **solo cuando** $e^{2\pi i j p / q} = 1$, es decir cuando $q \mid j$ (dado que $\gcd(p,q) = 1$).

### 11.5 El residuo converge a $1/q$

Usando ortogonalidad de caracteres:

$$\operatorname{Res}_k = \frac{1}{q} \sum_{r=0}^{q-1} \cos^{2k}\!\left(\frac{2\pi r}{q}\right)$$

Cuando $k \to \infty$: el término $r = 0$ contribuye $1/q \cdot 1 = 1/q$. Los demás decaen exponencialmente:

$$\left|\operatorname{Res}_k - \frac{1}{q}\right| \sim \frac{q-1}{q} \exp\!\left(-\frac{4\pi^2 k}{q^2}\right)$$

**La sonancia $1/q$ es el residuo de la función meromorfa $F(s)$ en su polo.** Esto le confiere un significado analítico profundo.

### 11.6 Parte finita $C_0$

$$C_0(p/q) = \lim_{N \to \infty} \left[S_k(p/q) - \operatorname{Res} \cdot \ln N\right]$$

$C_0$ depende de $p$ y de $q$, no solo de $q$. Satisface la simetría $C_0(p/q) = C_0((q-p)/q)$. Para $q$ primo, hay $\varphi(q)/2$ valores distintos de $C_0$.

$C_0$ contiene información aritmética más fina que $1/q$: distingue intervalos con mismo denominador pero distinto numerador.

---

## 12. Conexiones con teoría de números

### 12.1 Sub-recurrencias y divisores

Para $r = p/q$ y $d \mid q$ (divisor de $q$), el lag $n = q/d$ produce:

$$R(q/d) = \cos^2(\pi p / d)$$

Esto es alto cuando $p/d$ está cerca de un entero. La colección $\{(d, \cos^2(\pi p/d)) : d \mid q\}$ es una "firma de divisores" que contiene información sobre la factorización de $q$, leída directamente del patrón de autocorrelación.

Para $q$ primo: no hay sub-recurrencias (no hay divisores propios). El patrón tiene un único pico en $n = q$ y nada entre medias.

Para $q$ compuesto: hay sub-picos en cada $n = q/d$. Los divisores grandes producen sub-picos más altos (porque $\cos^2(\pi/d) \to 1$ cuando $d \to \infty$).

### 12.2 Convergencia escalonada

La convergencia de $S_k \to 1/q$ con $k$ creciente tiene carácter diferente según la naturaleza de $q$:

- **$q$ primo**: convergencia monotónica, sin mesetas.
- **$q$ compuesto ($q = \prod p_i^{a_i}$)**: convergencia escalonada, con mesetas temporales en $S_k \approx 1/d$ para cada divisor $d$ de $q$.

La resolución "descubre" los factores de $q$ uno por uno al aumentar $k$.

### 12.3 Sumas de Ramanujan

La autocorrelación $R(n)$ para ratio $1/q$ usa un solo carácter: $\cos(2\pi n/q) = \operatorname{Re}(e^{2\pi i n/q})$. La suma de Ramanujan promedia sobre todos los coprimos:

$$c_q(n) = \sum_{\substack{a=1 \\ \gcd(a,q)=1}}^{q} e^{2\pi i a n / q}$$

La fórmula de Ramanujan da:

$$c_q(n) = \mu(q/\gcd(n,q)) \cdot \frac{\varphi(q)}{\varphi(q/\gcd(n,q))}$$

La conexión: $\cos(2\pi n/q)$ es el "término $a = 1$" de $c_q(n)$. La multiplicatividad de $c_q$ en $q$ implica que una "sonancia de Ramanujan" factorizaría la contribución de cada primo independientemente.

### 12.4 Serie de Dirichlet de Ramanujan

$$\sum_{n=1}^{\infty} \frac{c_q(n)}{n^s} = \frac{\varphi(q)}{\zeta(s)} \cdot \prod_{p \mid q} \frac{1 - p^{-s}}{1 - p^{1-s}}$$

Esto conecta la sonancia con la función zeta de Riemann y la estructura multiplicativa de los enteros.

---

## 13. Espectrómetro aritmético

### 13.1 Concepto

La FFT estándar detecta frecuencias absolutas (Hz). El espectrómetro aritmético detecta **relaciones racionales** entre frecuencias, siendo invariante a transposición.

### 13.2 Algoritmo

Dada una señal $s(t)$ y una frecuencia fundamental candidata $f_1$:

1. Calcular la autocorrelación $\hat{R}(\tau)$ via FFT.
2. Para cada ratio candidato $r$ y cada lag entero $n = 1, \ldots, N$:
   - Evaluar $\hat{R}(n \cdot T_1)$ con interpolación.
   - Mapear a $\hat{R}_n = \frac{1 + \hat{R}(n T_1)}{2} \in [0, 1]$.
3. Calcular $\hat{S}_k(r) = \sum_{n=1}^N \hat{R}_n^k / n$.
4. Los picos de $\hat{S}_k(r)$ revelan los ratios presentes en la señal.

### 13.3 Invariancia a transposición

Un acorde mayor en cualquier tonalidad produce el mismo espectro aritmético: picos en $r = 5/4$ (tercera mayor) y $r = 3/2$ (quinta justa). La FFT produciría espectros completamente diferentes para cada tonalidad.

### 13.4 Versión 2D

Barriendo $f_1$ sobre un rango de fundamentales candidatas se obtiene un mapa 2D $(f_1, r) \mapsto \hat{S}_k$. El máximo global revela simultáneamente la fundamental y las relaciones armónicas presentes.

### 13.5 Complementariedad con FFT

| Propiedad | FFT | Espectrómetro aritmético |
|-----------|-----|--------------------------|
| Detecta | Frecuencias absolutas (Hz) | Relaciones racionales |
| Invariante a transposición | No | Sí |
| Identifica armonía | Indirectamente | Directamente |
| Base teórica | Análisis de Fourier | Autocorrelación + $S_k$ |

---

## 14. Formulación compacta

El desarrollo completo se resume en la cadena deductiva:

$$s(t) = \sin(2\pi f_1 t) + \sin(2\pi f_2 t)$$

$$\downarrow \text{ autocorrelación}$$

$$R(\tau) = \tfrac{1}{2}\cos(2\pi f_1 \tau) + \tfrac{1}{2}\cos(2\pi f_2 \tau)$$

$$\downarrow \text{ lags enteros } \tau = nT_1$$

$$R(n) = \cos^2(\pi r n)$$

$$\downarrow \text{ kernel de resolución } k$$

$$\boxed{S_k(r) = \sum_{n=1}^{N} \frac{\cos^{2k}(\pi r n)}{n}}$$

$$\downarrow k \to \infty$$

$$S_k(p/q) / H_N \to 1/q$$

$$\downarrow \text{ generalización } s \in \mathbb{C}$$

$$F(s) = \sum_{n=1}^{\infty} \frac{\cos^{2k}(\pi r n)}{n^s}, \qquad \operatorname{Res}_{s=1} F \to \frac{1}{q}$$

Todo derivado de la física ondulatoria. Sin factorización prima. Sin constantes ajustadas. Sin datos subjetivos.

---

## 15. Cuestiones abiertas

1. **Extensión a timbres complejos.** Para señales con contenido armónico $s(t) = \sum a_m \sin(2\pi m f_1 t)$, la autocorrelación tiene estructura más rica. ¿Se preserva la convergencia a $1/q$?

2. **Parte finita $C_0$ y residuos cuadráticos.** ¿Existe una relación entre $C_0(p/q)$ y el símbolo de Legendre $(p/q)$ para $q$ primo?

3. **Ceros de $F(s)$ en la banda crítica.** ¿La distribución de ceros de $F(s)$ en la línea $\operatorname{Re}(s) = 1/2$ distingue $q$ primo de $q$ compuesto? ¿Sigue la estadística de GUE o de Poisson?

4. **Convergencia para temperamento igual.** Los ratios del temperamento igual ($r = 2^{k/12}$) son irracionales. ¿Qué predice $S_k$ para la "consonancia" de la escala temperada, y cómo se compara con la entonación justa?

5. **Espectrómetro aritmético en señales reales.** Validación experimental con grabaciones musicales: ¿detecta correctamente las relaciones armónicas en presencia de ruido, reverberación e inarmónicos?

6. **Función $F(s, q, k, p)$ como objeto analítico.** Propiedades de la continuación meromorfa, ecuación funcional (si existe), y relación precisa entre sus ceros y la estructura aritmética de $q$.
