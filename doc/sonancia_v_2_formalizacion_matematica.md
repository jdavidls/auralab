# Sonancia v2

## Formalización matemática de recurrencia, fase compleja, recurrencias parciales y timbres compuestos

---

## Resumen

Este documento reformula y extiende el modelo de sonancia a partir de la autocorrelación de ondas sinusoidales. La versión inicial mostraba que, para dos tonos puros con ratio racional reducido \(r=p/q\), la recurrencia perfecta mínima aparece en \(q\) periodos de la fundamental, lo que induce una sonancia primaria proporcional a \(1/q\). Esta versión 2 conserva ese resultado, pero lo sitúa como el primer nivel de una jerarquía más rica.

La tesis central de esta versión es que el denominador \(q\) determina el tiempo de cierre perfecto, mientras que el numerador \(p\) organiza temporalmente las recurrencias parciales anteriores al cierre. Por ello, la sonancia completa no debe reducirse al límite \(1/q\), sino describirse mediante una jerarquía formada por:

1. recurrencia primaria;
2. firma pre-resonante;
3. parte finita analítica;
4. estructura compleja de fase;
5. extensión a timbres compuestos mediante suma ponderada sobre pares de parciales.

El documento introduce una generalización compleja basada en \(e^{2\pi i rn}\), que muestra que el modelo real \(\cos^2(\pi rn)\) es una proyección energética de una órbita compleja en el círculo unitario. Esta reformulación conecta naturalmente con análisis de Fourier discreto, caracteres aditivos módulo \(q\), polilogaritmos y series de Dirichlet.

Finalmente, se propone una generalización para sonidos compuestos: cualquier señal puede entenderse como una colección finita de componentes sinusoidales \(\{(f_i,A_i)\}\), y la sonancia total como una suma ponderada de sonancias elementales entre todos los pares de componentes. Esta construcción crea un puente formal con el enfoque de Sethares, pero sustituye la rugosidad psicoacústica elemental por una función de recurrencia ondulatoria.

---

# 1. Motivación

La consonancia y la disonancia musical han sido tradicionalmente abordadas desde dos perspectivas principales.

La primera es psicoacústica: explica la disonancia en términos de rugosidad, batidos, bandas críticas y respuesta perceptiva. En esta línea se sitúan los modelos inspirados en Plomp-Levelt y las curvas de disonancia de Sethares, donde la disonancia de un intervalo se calcula sumando contribuciones de rugosidad entre parciales del timbre.

La segunda es aritmética: explica la consonancia mediante la simplicidad de ratios racionales. En esta línea aparecen medidas basadas en la complejidad del ratio, como la tradición derivada de Euler y otros modelos basados en factorización o tamaño de numerador y denominador.

La propuesta de sonancia busca una tercera vía: partir de la física ondulatoria, en particular de la autocorrelación temporal, para derivar una estructura de recurrencia que explique por qué ciertos ratios generan cierres más tempranos y patrones más estables.

La versión inicial del modelo mostró que, para un ratio racional reducido \(p/q\), el primer retorno perfecto ocurre en \(q\) periodos de la fundamental. Esto produce una jerarquía primaria \(1/q\). Sin embargo, esta jerarquía tiene una limitación importante: no distingue ratios con el mismo denominador.

Esta versión 2 toma esa crítica como punto de partida. La respuesta propuesta es que \(1/q\) no debe entenderse como la sonancia completa, sino como el término dominante de cierre perfecto. La información que distingue ratios con igual denominador no desaparece de la autocorrelación: está en el orden temporal de las recurrencias parciales previas al cierre.

---

# 2. Señal de dos tonos puros

Consideremos dos tonos puros:

\[
s(t)=\sin(2\pi f_1t+\varphi_1)+\sin(2\pi f_2t+\varphi_2)
\]

con ratio:

\[
r=\frac{f_2}{f_1}.
\]

Para simplificar la exposición, se asume inicialmente que \(f_1\neq f_2\) y que la autocorrelación se calcula en el límite temporal ideal.

La autocorrelación normalizada de la señal es:

\[
R(\tau)=\lim_{T\to\infty}\frac{1}{T}\int_0^T s(t)s(t+\tau)\,dt.
\]

Expandiendo el producto aparecen auto-términos y términos cruzados. Los términos cruzados entre frecuencias distintas se anulan en el promedio temporal infinito. Los auto-términos producen:

\[
R(\tau)=\frac12\cos(2\pi f_1\tau)+\frac12\cos(2\pi f_2\tau).
\]

Las fases \(\varphi_1,\varphi_2\) desaparecen en este límite, porque en los auto-términos solo aparece la diferencia de fase entre una señal y su copia retardada.

---

# 3. Evaluación en lags enteros

Tomamos retardos que son múltiplos enteros del periodo de la fundamental:

\[
\tau=nT_1,\qquad T_1=\frac{1}{f_1},\qquad n\in\mathbb Z^+.
\]

Entonces:

\[
R(n)=\frac12\cos(2\pi n)+\frac12\cos(2\pi rn).
\]

Como \(\cos(2\pi n)=1\) para todo entero \(n\), resulta:

\[
R(n)=\frac12+\frac12\cos(2\pi rn).
\]

Usando la identidad:

\[
\frac12+\frac12\cos\theta=\cos^2\left(\frac{\theta}{2}\right),
\]

obtenemos:

\[
\boxed{R(n)=\cos^2(\pi rn)}.
\]

Esta identidad es el núcleo del modelo.

---

# 4. Recurrencia perfecta

Sea:

\[
r=\frac{p}{q},\qquad \gcd(p,q)=1.
\]

Entonces:

\[
R_{p,q}(n)=\cos^2\left(\pi\frac{pn}{q}\right).
\]

La recurrencia perfecta ocurre cuando:

\[
R_{p,q}(n)=1.
\]

Esto equivale a:

\[
\cos^2\left(\pi\frac{pn}{q}\right)=1,
\]

lo cual ocurre si y solo si:

\[
\frac{pn}{q}\in\mathbb Z.
\]

Como \(p\) y \(q\) son coprimos, el menor entero positivo \(n\) que cumple esta condición es:

\[
\boxed{n=q}.
\]

Por tanto, el denominador \(q\) es el tiempo mínimo de recurrencia perfecta medido en periodos de la fundamental.

Definimos la recurrencia primaria:

\[
\rho(p,q)=\min\{n\geq1:R_{p,q}(n)=1\}.
\]

Entonces:

\[
\boxed{\rho(p,q)=q}.
\]

Y definimos la sonancia primaria:

\[
\boxed{\sigma_0(p,q)=\frac{1}{\rho(p,q)}=\frac1q}.
\]

Este resultado no depende del numerador \(p\). Esa independencia es una virtud para describir el cierre perfecto, pero una limitación si se pretende describir la textura completa del intervalo.

---

# 5. La crítica: degeneración de ratios con igual denominador

Si la sonancia se reduce exclusivamente a:

\[
\sigma_0(p,q)=\frac1q,
\]

entonces todos los ratios reducidos con el mismo denominador reciben el mismo valor.

Por ejemplo:

\[
\frac{4}{3},\quad \frac{5}{3},\quad \frac{7}{3},\quad \frac{8}{3}
\]

comparten \(q=3\) y, por tanto, la misma sonancia primaria.

Esto revela una limitación de la medida \(1/q\). Sin embargo, la degeneración no está en la autocorrelación completa, sino en la reducción del patrón completo a su primer retorno perfecto.

La secuencia:

\[
\left(R_{p,q}(1),R_{p,q}(2),\ldots,R_{p,q}(q)\right)
\]

sí contiene información organizada por \(p\). El numerador no altera el momento de cierre, pero sí altera el orden temporal de las recurrencias parciales previas al cierre.

Esta observación conduce a la tesis central de la versión 2:

\[
\boxed{\text{El denominador }q\text{ determina el cierre; el numerador }p\text{ determina la trayectoria.}}
\]

---

# 6. Secuencia de recurrencia ondulatoria

Definimos la secuencia de recurrencia:

\[
A_{p,q}(n)=\cos^2\left(\pi\frac{pn}{q}\right),\qquad n\in\mathbb Z^+.
\]

Esta secuencia es periódica:

\[
A_{p,q}(n+q)=A_{p,q}(n).
\]

Además:

\[
A_{p,q}(q)=1.
\]

La firma completa de un ciclo es:

\[
\Pi(p,q)=\left(A_{p,q}(1),A_{p,q}(2),\ldots,A_{p,q}(q)\right).
\]

El primer nivel de sonancia extrae solo:

\[
A_{p,q}(q)=1.
\]

La versión refinada estudia también:

\[
A_{p,q}(1),\ldots,A_{p,q}(q-1).
\]

Estos términos son recurrencias parciales o pre-resonantes.

---

# 7. Acción de \(p\) sobre el ciclo módulo \(q\)

Para \(p/q\) reducido, la multiplicación por \(p\) define una permutación de los residuos módulo \(q\):

\[
n\mapsto pn\pmod q.
\]

Como \(\gcd(p,q)=1\), el conjunto:

\[
\{p,2p,3p,\ldots,(q-1)p\}\pmod q
\]

es una permutación de:

\[
\{1,2,3,\ldots,q-1\}.
\]

Por tanto, el conjunto no ordenado de valores:

\[
\{A_{p,q}(n):1\leq n<q\}
\]

es el mismo para todos los \(p\) coprimos con \(q\). Lo que cambia es el orden temporal de aparición.

Así, el numerador \(p\) no cambia los valores disponibles dentro del ciclo; cambia la trayectoria que recorre esos valores.

Este hecho explica por qué muchas medidas simétricas no distinguen \(p\): si se pierde el orden temporal, la información del numerador se colapsa.

---

# 8. Firma pre-resonante

Definimos la firma pre-resonante como el conjunto ordenado de recurrencias parciales anteriores al cierre:

\[
\Phi(p,q)=\left(A_{p,q}(1),A_{p,q}(2),\ldots,A_{p,q}(q-1)\right).
\]

Esta firma conserva la información temporal que desaparece en la medida primaria \(1/q\).

Para obtener un escalar dependiente de esta firma, introducimos una resolución \(k\):

\[
\boxed{\Phi_k(p,q)=\sum_{n=1}^{q-1}\frac{A_{p,q}(n)^k}{n}}
\]

es decir:

\[
\boxed{\Phi_k(p,q)=\sum_{n=1}^{q-1}\frac{\cos^{2k}\left(\pi\frac{pn}{q}\right)}{n}}.
\]

El peso \(1/n\) conserva la prioridad temporal: una recurrencia parcial temprana pesa más que una tardía.

La sonancia de ciclo refinada queda:

\[
\boxed{S_k^{\mathrm{cycle}}(p,q)=\frac1q+\Phi_k(p,q)}.
\]

Equivalentemente:

\[
\boxed{S_k^{\mathrm{cycle}}(p,q)=\sum_{n=1}^{q}\frac{\cos^{2k}\left(\pi\frac{pn}{q}\right)}{n}}.
\]

Este objeto distingue, para \(k\) finito, diferencias de trayectoria generadas por \(p\), aunque todas ellas compartan el mismo cierre perfecto \(q\).

---

# 9. Resolución finita y pérdida de información en el límite

El parámetro \(k\) controla la selectividad del detector de recurrencias.

Si:

\[
0\leq A_{p,q}(n)<1,
\]

entonces:

\[
A_{p,q}(n)^k\to0\qquad\text{cuando }k\to\infty.
\]

Por tanto:

\[
\lim_{k\to\infty}\Phi_k(p,q)=0.
\]

Y:

\[
\lim_{k\to\infty}S_k^{\mathrm{cycle}}(p,q)=\frac1q.
\]

Esto muestra que la degeneración de numeradores aparece al tomar el límite de resolución infinita.

A resolución finita, los términos sub-resonantes sobreviven y contienen información sobre \(p\). Acústicamente, esto es relevante: los fenómenos musicales y físicos reales no operan solo con recurrencias perfectas, sino con aproximaciones, batidos, interferencias parciales y cierres incompletos.

---

# 10. Normalización de la firma pre-resonante

Para comparar denominadores distintos, se puede normalizar por la suma armónica:

\[
H_m=\sum_{n=1}^{m}\frac1n.
\]

Definimos:

\[
\boxed{\widetilde{S}_k^{\mathrm{cycle}}(p,q)=\frac{1}{H_q}\sum_{n=1}^{q}\frac{A_{p,q}(n)^k}{n}}.
\]

Y la firma pre-resonante normalizada:

\[
\boxed{\widetilde{\Phi}_k(p,q)=\frac{1}{H_{q-1}}\sum_{n=1}^{q-1}\frac{A_{p,q}(n)^k}{n}}.
\]

La sonancia completa puede representarse entonces como un par:

\[
\boxed{\operatorname{Son}_k(p,q)=\left(\frac1q,\widetilde{\Phi}_k(p,q)\right)}.
\]

El primer componente mide cierre perfecto; el segundo mide recurrencia parcial pre-resonante.

---

# 11. Simetría real: \(p\leftrightarrow q-p\)

La secuencia real satisface:

\[
A_{q-p,q}(n)=A_{p,q}(n).
\]

En efecto:

\[
A_{q-p,q}(n)=\cos^2\left(\pi\frac{(q-p)n}{q}\right)
\]

\[
=\cos^2\left(\pi n-\pi\frac{pn}{q}\right)
\]

\[
=\cos^2\left(\pi\frac{pn}{q}\right).
\]

Por tanto, cualquier modelo basado exclusivamente en \(\cos^2\) no distingue \(p\) de \(q-p\). Esta simetría no es un defecto técnico, sino una consecuencia de usar una magnitud real, par y energética.

Si se desea distinguir orientación, es necesario conservar información compleja de fase.

---

# 12. Generalización compleja

Introducimos el núcleo complejo:

\[
\boxed{z_r(n)=e^{2\pi i rn}}.
\]

Para \(r=p/q\):

\[
\boxed{z_{p,q}(n)=e^{2\pi i pn/q}}.
\]

Este objeto describe una órbita en el círculo unitario. Cada incremento de \(n\) multiplica por:

\[
e^{2\pi i p/q}.
\]

El cierre ocurre cuando:

\[
z_{p,q}(n)=1.
\]

Esto requiere:

\[
\frac{pn}{q}\in\mathbb Z,
\]

por lo que el primer cierre vuelve a ser:

\[
n=q.
\]

Sin embargo, el camino alrededor del círculo depende de \(p\). El denominador fija el número de vértices; el numerador fija el salto entre vértices.

---

# 13. El modelo real como proyección del modelo complejo

La relación con la autocorrelación real es:

\[
\cos(2\pi rn)=\operatorname{Re}\left(e^{2\pi i rn}\right).
\]

Por tanto:

\[
R(n)=\frac12+\frac12\operatorname{Re}(z_r(n)).
\]

También podemos escribir:

\[
R(n)=\frac12+\frac14z_r(n)+\frac14\overline{z_r(n)}.
\]

Así, \(R(n)\) no es el objeto más primitivo, sino una proyección real y energética del movimiento complejo.

La pérdida de orientación \(p\leftrightarrow q-p\) se explica porque:

\[
z_{q-p,q}(n)=\overline{z_{p,q}(n)}.
\]

Al tomar parte real, ambos quedan identificados:

\[
\operatorname{Re}(z)=\operatorname{Re}(\overline z).
\]

Por tanto, la complejificación permite distinguir entre:

1. trayectoria orientada compleja;
2. proyección real;
3. energía normalizada.

---

# 14. Conexión con Fourier discreto

Para \(r=p/q\), la función:

\[
n\mapsto e^{2\pi i pn/q}
\]

es un modo de Fourier discreto sobre el grupo cíclico:

\[
\mathbb Z/q\mathbb Z.
\]

El numerador \(p\) selecciona el modo de Fourier; el denominador \(q\) determina el tamaño del ciclo.

Así:

\[
q=\text{longitud del ciclo},
\]

\[
p=\text{modo/generador de la órbita}.
\]

Si \(\gcd(p,q)=1\), la órbita recorre todos los residuos antes del cierre. Si \(p\) y \(q\) no son coprimos, la fracción no está reducida y el ciclo efectivo es menor.

Esta observación vincula la sonancia con análisis armónico sobre grupos finitos.

---

# 15. Expansión compleja de \(S_k\)

Recordemos:

\[
R(n)=\frac12+\frac14e^{2\pi i rn}+\frac14e^{-2\pi i rn}.
\]

La función de sonancia finita es:

\[
S_k(r;N)=\sum_{n=1}^{N}\frac{R(n)^k}{n}.
\]

Expandiendo la potencia:

\[
R(n)^k=\sum_{\ell=-k}^{k}c_\ell^{(k)}e^{2\pi i\ell rn},
\]

con coeficientes simétricos:

\[
c_{-\ell}^{(k)}=c_\ell^{(k)}.
\]

Entonces:

\[
\boxed{S_k(r;N)=\sum_{\ell=-k}^{k}c_\ell^{(k)}\sum_{n=1}^{N}\frac{e^{2\pi i\ell rn}}{n}}.
\]

Esta fórmula muestra que \(S_k\) es una combinación finita de sumas exponenciales ponderadas.

---

# 16. Serie de Dirichlet compleja

Definimos la extensión analítica:

\[
\boxed{\mathcal S_k(r;s)=\sum_{n=1}^{\infty}\frac{R(n)^k}{n^s}}.
\]

Sustituyendo la expansión de Fourier:

\[
\boxed{\mathcal S_k(r;s)=\sum_{\ell=-k}^{k}c_\ell^{(k)}\operatorname{Li}_s(e^{2\pi i\ell r})},
\]

cuando la serie converge inicialmente para \(\operatorname{Re}(s)>1\), y por continuación analítica donde corresponda.

Para \(r=p/q\):

\[
\boxed{\mathcal S_k(p/q;s)=\sum_{\ell=-k}^{k}c_\ell^{(k)}\operatorname{Li}_s(e^{2\pi i\ell p/q})}.
\]

Los términos dependen de \(p\) mediante raíces de unidad:

\[
e^{2\pi i\ell p/q}.
\]

---

# 17. Residuo y parte finita

El polilogaritmo \(\operatorname{Li}_s(e^{2\pi i\ell p/q})\) tiene comportamiento singular en \(s=1\) cuando:

\[
e^{2\pi i\ell p/q}=1.
\]

Esto equivale a:

\[
q\mid \ell p.
\]

Como \(\gcd(p,q)=1\), equivale a:

\[
q\mid \ell.
\]

Por tanto, los términos singulares dependen de \(q\), no de \(p\). La parte principal en el polo captura la recurrencia perfecta.

La expansión de Laurent cerca de \(s=1\) toma la forma:

\[
\boxed{\mathcal S_k(p/q;s)=\frac{A_k(q)}{s-1}+C_k(p,q)+O(s-1)}.
\]

Aquí:

\[
A_k(q)=\operatorname{Res}_{s=1}\mathcal S_k(p/q;s)
\]

mide la densidad de recurrencias perfectas, mientras que:

\[
C_k(p,q)=\operatorname{FP}_{s=1}\mathcal S_k(p/q;s)
\]

es la parte finita y contiene información dependiente del numerador.

Esta es una formalización precisa de la jerarquía:

\[
\boxed{\text{residuo}=\text{cierre primario}}
\]

\[
\boxed{\text{parte finita}=\text{estructura pre-resonante}}
\]

---

# 18. Interpretación jerárquica de la sonancia

La sonancia completa no debe reducirse a un único número. Proponemos una jerarquía:

\[
\boxed{\operatorname{Son}_k(p,q)=\left(A_k(q),C_k(p,q),\Phi_k(p,q),\Pi(p,q)\right)}.
\]

Cada nivel contiene información distinta.

## 18.1 Recurrencia primaria

\[
A_k(q)\sim \frac1q.
\]

Mide el cierre perfecto.

## 18.2 Parte finita analítica

\[
C_k(p,q).
\]

Mide la estructura aritmético-temporal que queda después de separar el término dominante.

## 18.3 Firma pre-resonante

\[
\Phi_k(p,q)=\sum_{n=1}^{q-1}\frac{A_{p,q}(n)^k}{n}.
\]

Mide recurrencias parciales ponderadas por aparición temporal.

## 18.4 Perfil de ciclo

\[
\Pi(p,q)=\left(A_{p,q}(1),\ldots,A_{p,q}(q)\right).
\]

Conserva la trayectoria completa.

---

# 19. Recurrencias parciales y textura acústica

La recurrencia perfecta \(1/q\) es estructuralmente importante, pero acústicamente no agota el fenómeno.

La textura de un intervalo depende también de:

1. cómo se aproxima al cierre;
2. cuántas recurrencias parciales altas aparecen antes del cierre;
3. cuán temprano aparecen;
4. en qué orden temporal aparecen;
5. cómo estas recurrencias producen batidos, refuerzos o cancelaciones parciales.

Definimos eventos de recurrencia parcial mediante un umbral \(\varepsilon\):

\[
\mathcal R_\varepsilon(p,q)=\{n<q:A_{p,q}(n)>1-\varepsilon\}.
\]

No obstante, para evitar introducir un umbral externo, se prefiere usar \(k\) como resolución:

\[
A_{p,q}(n)^k.
\]

A mayor \(k\), solo sobreviven recurrencias muy cercanas a 1. A menor \(k\), la medida conserva una textura más amplia de recurrencias parciales.

---

# 20. Relación con batidos y rugosidad

Los batidos acústicos aparecen cuando dos componentes de frecuencia cercana interfieren. En una autocorrelación ideal de duración infinita, los términos cruzados entre frecuencias distintas se anulan. Esto simplifica la teoría, pero también elimina parte de la información asociada a batidos.

Para estudiar textura real, hay dos vías complementarias.

## 20.1 Vía de pares espectrales

Se suman contribuciones entre todos los pares de parciales. Cada par tiene un ratio efectivo y genera una contribución de sonancia.

## 20.2 Vía de autocorrelación finita

Se usa una autocorrelación con ventana temporal finita:

\[
R_T(\tau)=\frac1T\int_0^T s(t)s(t+\tau)\,dt.
\]

En este caso, los términos cruzados no desaparecen exactamente. Aparecen factores dependientes de \(T\) y de la diferencia de frecuencias. Cuando dos frecuencias son cercanas, esos términos persisten durante ventanas largas y se relacionan con batidos.

Así, la rugosidad puede entenderse como una manifestación de recurrencias parciales, interferencias persistentes y cierres incompletos en escalas temporales finitas.

---

# 21. Tonos compuestos

Un tono compuesto puede representarse como una suma finita de componentes sinusoidales:

\[
x(t)=\sum_{a=1}^{M}A_a\sin(2\pi \alpha_a f_1t+\phi_a).
\]

Otro tono compuesto, situado a un intervalo global \(r\), puede escribirse como:

\[
y_r(t)=\sum_{b=1}^{L}B_b\sin(2\pi \beta_b r f_1t+\psi_b).
\]

La señal total es:

\[
s_r(t)=x(t)+y_r(t).
\]

Físicamente, no hay una diferencia fundamental entre dos tonos puros, dos tonos compuestos, un acorde o un espectro inarmónico. En todos los casos tenemos una colección de componentes sinusoidales.

---

# 22. Autocorrelación ideal de tonos compuestos

Sea una señal general:

\[
s(t)=\sum_{j=1}^{M}C_j\sin(2\pi f_jt+\phi_j).
\]

En autocorrelación ideal de duración infinita, si las frecuencias \(f_j\) son distintas, los términos cruzados se anulan y queda:

\[
R_s(\tau)=\frac12\sum_{j=1}^{M}C_j^2\cos(2\pi f_j\tau).
\]

Para dos timbres:

\[
\{\alpha_a f_1\}_{a=1}^{M},\qquad \{\beta_b r f_1\}_{b=1}^{L},
\]

la autocorrelación es:

\[
R_{s_r}(\tau)=\frac12\sum_a A_a^2\cos(2\pi \alpha_a f_1\tau)+\frac12\sum_b B_b^2\cos(2\pi \beta_b r f_1\tau).
\]

Evaluando en \(\tau=nT_1\):

\[
R_{s_r}(n)=\frac12\sum_a A_a^2\cos(2\pi\alpha_a n)+\frac12\sum_b B_b^2\cos(2\pi\beta_b rn).
\]

Si el primer timbre es armónico, \(\alpha_a\in\mathbb Z\), entonces:

\[
\cos(2\pi\alpha_a n)=1.
\]

El primer timbre aporta una base constante y la estructura intervalar queda gobernada por:

\[
\sum_b B_b^2\cos(2\pi\beta_b rn).
\]

Esta es la generalización directa desde autocorrelación.

---

# 23. Modelo de sonancia por pares de parciales

La generalización más flexible consiste en aplicar la sonancia elemental a todos los pares de componentes.

Definimos la sonancia elemental:

\[
\boxed{\mathfrak s_k(\rho)=\sum_{n=1}^{N}\frac{\cos^{2k}(\pi\rho n)}{n}}.
\]

Dada una colección finita de parciales:

\[
\mathcal F=\{(f_i,A_i)\}_{i=1}^{M},
\]

definimos la sonancia total:

\[
\boxed{\operatorname{Son}_k(\mathcal F)=\sum_{i<j}A_i^2A_j^2\mathfrak s_k\left(\frac{f_j}{f_i}\right)}.
\]

Esta fórmula unifica:

1. dos tonos puros;
2. tonos compuestos;
3. acordes;
4. timbres armónicos;
5. timbres inarmónicos;
6. colecciones espectrales arbitrarias.

Para dos tonos puros, solo hay un par y se recupera el modelo original.

---

# 24. Sonancia de dos timbres a intervalo variable

Sea un timbre:

\[
T=\{(\alpha_a,A_a)\}_{a=1}^{M}
\]

respecto a su fundamental, y otro timbre:

\[
U=\{(\beta_b,B_b)\}_{b=1}^{L}.
\]

Si el segundo timbre está a intervalo \(r\), los pares de parciales tienen ratios efectivos:

\[
\rho_{ab}(r)=\frac{\beta_b r}{\alpha_a}.
\]

Definimos:

\[
\boxed{\operatorname{Son}_k^{T,U}(r)=\sum_{a=1}^{M}\sum_{b=1}^{L}A_a^2B_b^2\mathfrak s_k\left(\frac{\beta_b r}{\alpha_a}\right)}.
\]

Si \(T=U\):

\[
\boxed{\operatorname{Son}_k^T(r)=\sum_{a,b}A_a^2A_b^2\mathfrak s_k\left(\frac{\alpha_b r}{\alpha_a}\right)}.
\]

Esta función produce una curva de sonancia del timbre. Sus máximos indican intervalos favorecidos por recurrencia temporal.

---

# 25. Comparación estructural con Sethares

Sethares calcula curvas de disonancia para timbres compuestos sumando contribuciones entre pares de parciales. Su estructura general puede esquematizarse como:

\[
D(r)=\sum_{a,b}w_{ab}\,d(f_a,f_b;r),
\]

con una función elemental de rugosidad dependiente de frecuencias, amplitudes y separación espectral.

La propuesta de sonancia conserva la arquitectura de suma sobre pares, pero reemplaza la función elemental:

\[
\text{rugosidad psicoacústica}\quad\longrightarrow\quad\text{recurrencia ondulatoria}.
\]

Así:

\[
\boxed{D_{\mathrm{Sethares}}(r)=\sum_{a,b}w_{ab}\,\operatorname{Roughness}_{ab}(r)}
\]

mientras que:

\[
\boxed{\operatorname{Son}_k^{T,U}(r)=\sum_{a,b}w_{ab}\,\mathfrak s_k(\rho_{ab}(r))}.
\]

La diferencia conceptual es:

- Sethares: intervalos favorecidos como mínimos de disonancia;
- Sonancia: intervalos favorecidos como máximos de recurrencia.

Ambos enfoques reconocen que el timbre modifica la estructura intervalar, pero lo fundamentan de forma distinta.

---

# 26. Pesos de amplitud

La elección del peso \(A_i^2A_j^2\) responde a que la energía de una componente sinusoidal es proporcional al cuadrado de su amplitud. Sin embargo, existen varias posibilidades:

## 26.1 Peso energético

\[
w_{ij}=A_i^2A_j^2.
\]

Adecuado para un modelo físico de energía.

## 26.2 Peso lineal

\[
w_{ij}=A_iA_j.
\]

Más cercano a ciertos modelos de interacción perceptual entre amplitudes.

## 26.3 Peso normalizado

\[
w_{ij}=\frac{A_i^2A_j^2}{\sum_{m<n}A_m^2A_n^2}.
\]

Útil para comparar timbres de distinta energía total.

En esta versión se adopta provisionalmente el peso energético normalizable:

\[
\boxed{w_{ij}=\frac{A_i^2A_j^2}{\sum_{m<n}A_m^2A_n^2}}.
\]

La sonancia normalizada queda:

\[
\boxed{\operatorname{Son}_k^{\mathrm{norm}}(\mathcal F)=\sum_{i<j}w_{ij}\mathfrak s_k\left(\frac{f_j}{f_i}\right)}.
\]

---

# 27. Modelo temporal con autocorrelación finita

La suma por pares es flexible, pero no sustituye completamente el análisis temporal de la señal compuesta.

Definimos la autocorrelación finita:

\[
R_T(\tau)=\frac1T\int_0^T s(t)s(t+\tau)\,dt.
\]

A diferencia del límite \(T\to\infty\), aquí los términos cruzados entre frecuencias distintas no desaparecen por completo. Para dos componentes aparecen términos que dependen de:

\[
\omega_i-\omega_j.
\]

Cuando las frecuencias son cercanas, estos términos varían lentamente y contribuyen a batidos.

Podemos definir una sonancia temporal de ventana:

\[
\boxed{\operatorname{Son}_{k,T}(r)=\sum_{n=1}^{N}\frac{\widetilde R_T(nT_1)^k}{n}},
\]

con \(\widetilde R_T\) normalizada a \([0,1]\).

Este modelo captura textura temporal, batidos e interferencia parcial de manera más directa que el modelo ideal.

---

# 28. Dos modelos complementarios

La teoría puede organizarse en dos ramas.

## 28.1 Modelo espectral por pares

\[
\operatorname{Son}_k(\mathcal F)=\sum_{i<j}w_{ij}\mathfrak s_k\left(\frac{f_j}{f_i}\right).
\]

Ventajas:

- simple;
- computable;
- comparable con Sethares;
- válido para timbres armónicos e inarmónicos;
- conserva la idea de ratio como unidad elemental.

## 28.2 Modelo temporal por autocorrelación finita

\[
\operatorname{Son}_{k,T}(r)=\sum_{n=1}^{N}\frac{\widetilde R_T(nT_1)^k}{n}.
\]

Ventajas:

- captura batidos;
- conserva términos cruzados;
- se aproxima al modelo ideal cuando \(T\to\infty\);
- puede aplicarse directamente a señales reales.

Ambos modelos deben considerarse complementarios, no excluyentes.

---

# 29. Formulación compleja para timbres

Para una colección de parciales \(\mathcal F=\{(f_i,A_i)\}\), definimos la firma compleja:

\[
\boxed{\mathcal Z_{\mathcal F}(s)=\sum_{i<j}w_{ij}\operatorname{Li}_s\left(e^{2\pi i f_j/f_i}\right)}.
\]

La versión real de recurrencia se obtiene proyectando:

\[
\operatorname{Re}\mathcal Z_{\mathcal F}(s)
\]

o usando el núcleo energético:

\[
\frac12+\frac12\operatorname{Re}\left(e^{2\pi i(f_j/f_i)n}\right).
\]

Para ratios racionales, los polos y residuos de esta función capturan cierres perfectos; las partes finitas capturan estructura fina entre parciales.

---

# 30. Conjeturas y programa de investigación

## Conjetura 1: cierre primario

Para \(r=p/q\) reducido, la sonancia primaria está determinada por:

\[
\sigma_0(p,q)=\frac1q.
\]

Esto mide la rapidez del primer cierre perfecto.

## Conjetura 2: firma pre-resonante

Para \(k\) finito, la cantidad:

\[
\Phi_k(p,q)=\sum_{n=1}^{q-1}\frac{\cos^{2k}(\pi pn/q)}{n}
\]

contiene información significativa sobre la organización temporal de recurrencias parciales generada por \(p\).

## Conjetura 3: parte finita analítica

La parte finita:

\[
C_k(p,q)=\operatorname{FP}_{s=1}\mathcal S_k(p/q;s)
\]

es una versión analítica de la firma pre-resonante y distingue clases de numeradores módulo \(q\), salvo simetrías inducidas por proyección real.

## Conjetura 4: textura acústica

La percepción de textura, batido o rugosidad está más relacionada con la firma pre-resonante y la autocorrelación finita que con el cierre perfecto \(1/q\) aislado.

## Conjetura 5: timbre y escala

Para un timbre \(T\), los máximos de:

\[
\operatorname{Son}_k^T(r)
\]

predicen intervalos favorecidos por la estructura espectral del timbre, de manera análoga a como los mínimos de disonancia de Sethares predicen intervalos favorecidos perceptualmente.

---

# 31. Síntesis conceptual

La versión 2 del modelo puede resumirse así.

La autocorrelación de dos tonos puros conduce a:

\[
R(n)=\cos^2(\pi rn).
\]

Para \(r=p/q\), el cierre perfecto aparece por primera vez en:

\[
n=q.
\]

Esto produce la sonancia primaria:

\[
\frac1q.
\]

Pero la autocorrelación contiene más información que el cierre. La secuencia:

\[
\cos^2\left(\pi\frac{p}{q}\right),
\cos^2\left(2\pi\frac{p}{q}\right),
\ldots,
\cos^2\left((q-1)\pi\frac{p}{q}\right)
\]

contiene una firma temporal organizada por \(p\).

En dominio complejo, esta firma procede de la órbita:

\[
e^{2\pi i pn/q}.
\]

Así, \(q\) es el periodo de cierre y \(p\) es el generador de la trayectoria.

La sonancia completa debe entenderse como:

\[
\boxed{\text{sonancia}=\text{cierre perfecto}+\text{recurrencia parcial}+\text{trayectoria compleja}}.
\]

Para timbres compuestos, cualquier sonido se modela como una colección de parciales, y la sonancia total se obtiene sumando las sonancias elementales entre pares:

\[
\boxed{\operatorname{Son}_k(\mathcal F)=\sum_{i<j}w_{ij}\sum_{n=1}^{N}\frac{\cos^{2k}\left(\pi\frac{f_j}{f_i}n\right)}{n}}.
\]

Esta fórmula generaliza el modelo original, conecta con Sethares por su estructura de suma sobre parciales y mantiene un fundamento ondulatorio basado en recurrencia temporal.

---

# 32. Cadena deductiva final

\[
s(t)=\sin(2\pi f_1t)+\sin(2\pi f_2t)
\]

\[
\Downarrow
\]

\[
R(\tau)=\frac12\cos(2\pi f_1\tau)+\frac12\cos(2\pi f_2\tau)
\]

\[
\Downarrow \tau=nT_1
\]

\[
R(n)=\frac12+rac12\cos(2\pi rn)=\cos^2(\pi rn)
\]

\[
\Downarrow r=p/q
\]

\[
R_{p,q}(n)=\cos^2\left(\pi\frac{pn}{q}\right)
\]

\[
\Downarrow
\]

\[
\rho(p,q)=q,\\qquad \sigma_0(p,q)=\frac1q
\]

\[
\Downarrow \text{recurrencias parciales}
\]

\[
\Phi_k(p,q)=\sum_{n=1}^{q-1}\frac{\cos^{2k}(\pi pn/q)}{n}
\]

\[
\Downarrow \text{dominio complejo}
\]

\[
z_{p,q}(n)=e^{2\pi i pn/q}
\]

\[
\Downarrow \text{serie analítica}
\]

\[
\mathcal S_k(p/q;s)=\sum_{\ell=-k}^{k}c_\ell^{(k)}\operatorname{Li}_s(e^{2\pi i\ell p/q})
\]

\[
\Downarrow \text{expansión en }s=1
\]

\[
\mathcal S_k(p/q;s)=\frac{A_k(q)}{s-1}+C_k(p,q)+O(s-1)
\]

\[
\Downarrow \text{timbres compuestos}
\]

\[
\operatorname{Son}_k(\mathcal F)=\sum_{i<j}w_{ij}\mathfrak s_k(f_j/f_i)
\]

---

# 33. Cierre

La versión 2 no abandona el resultado \(1/q\), sino que lo reubica. \(1/q\) no es la sonancia completa: es la medida de recurrencia perfecta primaria. La información musical y acústica más fina vive en las recurrencias parciales, en su orden temporal y en la estructura compleja que las genera.

El modelo gana así una arquitectura jerárquica:

\[
\boxed{q\Rightarrow cierre}
\]

\[
\boxed{p\Rightarrow trayectoria}
\]

\[
\boxed{k\Rightarrow resolución}
\]

\[
\boxed{C_k(p,q)\Rightarrow estructura fina}
\]

\[
\boxed{\{(f_i,A_i)\}\Rightarrow timbre compuesto}
\]

Esta jerarquía permite responder a la crítica sobre ratios con igual denominador, abrir una vía hacia la textura acústica y construir un puente formal con las curvas de disonancia de Sethares sin abandonar el fundamento ondulatorio-temporal.

