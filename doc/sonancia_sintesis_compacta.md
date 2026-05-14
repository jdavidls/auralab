# Sonancia — Síntesis compacta del programa matemático

## 1. Idea central

La **sonancia** propone medir la consonancia como estabilidad recurrente del patrón de interferencia entre ondas.

Para dos tonos puros:

$$
s(t)=\sin(2\pi f_1t)+\sin(2\pi f_2t)
$$

con ratio:

$$
r=\frac{f_2}{f_1}
$$

la autocorrelación ideal evaluada en múltiplos enteros del periodo de $f_1$,

$$
\tau=nT_1,\qquad T_1=\frac1{f_1},
$$

conduce a:

$$
R(n)=\frac12+\frac12\cos(2\pi rn)
$$

y por tanto:

$$
\boxed{R(n)=\cos^2(\pi rn)}
$$

Esta identidad es el núcleo físico-matemático del modelo.

---

## 2. Recurrencia perfecta

Si:

$$
r=\frac pq,\qquad \gcd(p,q)=1,
$$

entonces:

$$
R(n)=\cos^2\left(\pi\frac{pn}{q}\right)
$$

La recurrencia perfecta ocurre cuando:

$$
\frac{pn}{q}\in\mathbb Z
$$

El primer $n$ positivo que cumple esto es:

$$
\boxed{n=q}
$$

Por tanto, el denominador $q$ mide el tiempo mínimo de recurrencia perfecta en periodos de la primera frecuencia.

La sonancia primaria es:

$$
\boxed{\sigma_0(p,q)=\frac1q}
$$

Interpretación:

$$
q \Rightarrow \text{tiempo de cierre perfecto}
$$

---

## 3. Limitación de $1/q$

La medida $1/q$ no distingue ratios con igual denominador.

Por ejemplo:

$$
\frac43,\frac53,\frac73
$$

comparten el mismo cierre primario $q=3$.

Esto no significa que la autocorrelación no contenga más información. Significa que el límite $1/q$ conserva solo el cierre perfecto y descarta la estructura pre-resonante.

La frase clave es:

$$
\boxed{
\text{El numerador }p\text{ no afecta al tiempo de cierre, pero sí organiza temporalmente las recurrencias parciales anteriores al cierre.}
}
$$

---

## 4. Firma pre-resonante

Definimos la secuencia de recurrencia:

$$
A_{p,q}(n)=\cos^2\left(\pi\frac{pn}{q}\right)
$$

y su firma dentro del ciclo:

$$
\Pi(p,q)=
\left(
A_{p,q}(1),
A_{p,q}(2),
\ldots,
A_{p,q}(q)
\right)
$$

La recurrencia perfecta es:

$$
A_{p,q}(q)=1
$$

pero la información fina está en:

$$
A_{p,q}(1),\ldots,A_{p,q}(q-1)
$$

Definimos la firma pre-resonante ponderada:

$$
\boxed{
\Phi_k(p,q)=
\sum_{n=1}^{q-1}
\frac{
\cos^{2k}\left(\pi\frac{pn}{q}\right)
}{n}
}
$$

Interpretación:

$$
\Phi_k(p,q)
\Rightarrow
\text{recurrencias parciales previas al cierre}
$$

---

## 5. Problema con argumentos $p,q$

Aunque $p,q$ son útiles para interpretar ratios racionales, no son buenos argumentos primarios para una función de sonancia.

La sonancia debe aceptar un ratio continuo:

$$
r\in\mathbb R^+
$$

y no depender de haber elegido previamente una fracción representativa.

Por tanto, $p,q$ deben aparecer como interpretación cuando $r$ sea racional, no como argumentos fundamentales.

---

## 6. Regularización de la sonancia direccional

Definimos:

$$
C_k(x)=\cos^{2k}(\pi x)
$$

La sonancia direccional original es:

$$
S_k(r;N)=
\sum_{n=1}^{N}
\frac{C_k(rn)}{n}
$$

Se propone una regularización:

$$
\boxed{
S_k^*(r;N)=
\sum_{n=1}^{N}
\frac{C_k(rn)}{n}
-\frac12C_k(r)
}
$$

Equivalente a:

$$
S_k^*(r;N)
=
\frac12C_k(r)
+
\sum_{n=2}^{N}
\frac{C_k(rn)}{n}
$$

Esto reduce el peso excesivo del primer lag.

Podemos escribir:

$$
S_k^*(r;N)=
\sum_{n=1}^{N}w_n C_k(rn)
$$

con:

$$
w_1=\frac12,
\qquad
w_n=\frac1n\quad(n\ge2)
$$

---

## 7. Sonancia recíproca regularizada

La consonancia física entre dos frecuencias debe ser recíproca:

$$
r\sim\frac1r
$$

Por tanto, la función principal propuesta es:

$$
\boxed{
\mathcal S_k(r;N)=
\frac12
\left[
S_k^*(r;N)+S_k^*(1/r;N)
\right]
}
$$

o explícitamente:

$$
\boxed{
\mathcal S_k(r;N)=
\frac12
\sum_{n=1}^{N}
w_n
\left[
\cos^{2k}(\pi rn)
+
\cos^{2k}\left(\frac{\pi n}{r}\right)
\right]
}
$$

con:

$$
w_1=\frac12,
\qquad
w_n=\frac1n\quad(n\ge2)
$$

Esta definición cumple:

$$
\boxed{
\mathcal S_k(r;N)=\mathcal S_k(1/r;N)
}
$$

y conserva la regularización propuesta.

---

## 8. Interpretación para $r=p/q$

Si:

$$
r=\frac pq
$$

entonces la rama $S_k^*(r;N)$ detecta el cierre en:

$$
n=q
$$

mientras que la rama recíproca $S_k^*(1/r;N)$ detecta el cierre en:

$$
n=p
$$

Así, $p$ y $q$ reaparecen de forma natural:

$$
S_k^*(p/q;N)
\Rightarrow q
$$

$$
S_k^*(q/p;N)
\Rightarrow p
$$

Por tanto, la función continua y recíproca recupera la información aritmética sin usar $p,q$ como argumentos primarios.

---

## 9. Dominio logarítmico

Como la reciprocidad es:

$$
r\mapsto\frac1r
$$

conviene usar:

$$
x=\log r
$$

Entonces:

$$
r=e^x,\qquad \frac1r=e^{-x}
$$

y la sonancia se vuelve una función par:

$$
\widehat{\mathcal S}_k(x;N)
=
\mathcal S_k(e^x;N)
$$

con:

$$
\boxed{
\widehat{\mathcal S}_k(x;N)
=
\widehat{\mathcal S}_k(-x;N)
}
$$

Esto es natural musicalmente, porque los intervalos son multiplicativos en frecuencia pero aditivos en log-frecuencia.

---

## 10. Fórmula fractal como alineamiento modular

Se exploró una fórmula dependiente de $p,q$:

$$
F_k(p,q)=
\frac1q
\sum_{a=0}^{q-1}
\cos^{2k}\left(\frac{\pi a}{q}\right)
\cos^{2k}\left(\frac{\pi pa}{q}\right)
$$

Definiendo:

$$
B_q(a)=\cos^{2k}\left(\frac{\pi a}{q}\right)
$$

y la permutación modular:

$$
P_p(a)=pa\pmod q
$$

se obtiene:

$$
\boxed{
F_k(p,q)=
\frac1q
\langle B_q,P_pB_q\rangle
}
$$

Interpretación:

$$
F_k(p,q)
\Rightarrow
\text{autoalineamiento del perfil de recurrencia bajo la permutación inducida por }p
$$

Aunque $P_pB_q$ contiene los mismos valores que $B_q$, el producto punto depende de cómo quedan emparejados.

Esta idea es útil conceptualmente, pero no debe sustituir a $\mathcal S_k(r;N)$ como función primaria porque depende explícitamente de $p,q$.

---

## 11. Formulación compleja

La transformación correcta de cosenos a exponenciales usa:

$$
\cos x=\frac{e^{ix}+e^{-ix}}2
$$

No se usa $e^x$, sino $e^{ix}$, porque el fenómeno es de fase y círculo unitario.

El núcleo complejo es:

$$
\boxed{
z_r(n)=e^{2\pi i rn}
}
$$

Entonces:

$$
\cos(2\pi rn)=\operatorname{Re}(z_r(n))
$$

y:

$$
\cos^2(\pi rn)
=
\frac12+
\frac14e^{2\pi i rn}
+
\frac14e^{-2\pi i rn}
$$

El modelo real es una proyección energética del modelo complejo.

---

## 12. Expansión de Fourier

La potencia:

$$
\cos^{2k}(\pi x)
$$

admite expansión finita:

$$
\boxed{
\cos^{2k}(\pi x)
=
\sum_{\ell=-k}^{k}
c_\ell^{(k)}
e^{2\pi i\ell x}
}
$$

con:

$$
\boxed{
c_\ell^{(k)}
=
2^{-2k}
\binom{2k}{k+\ell}
}
$$

para:

$$
-k\le \ell\le k
$$

Esto permite transportar la sonancia a Fourier.

---

## 13. Sonancia Fourier-Zeta

Definimos la versión analítica:

$$
\boxed{
S_k(r;s)
=
\sum_{n=1}^{\infty}
\frac{
\cos^{2k}(\pi rn)
}{n^s}
}
$$

Sustituyendo la expansión de Fourier:

$$
\boxed{
S_k(r;s)
=
\sum_{\ell=-k}^{k}
c_\ell^{(k)}
\operatorname{Li}_s
\left(
e^{2\pi i\ell r}
\right)
}
$$

donde:

$$
\operatorname{Li}_s(z)
=
\sum_{n=1}^{\infty}
\frac{z^n}{n^s}
$$

es el polilogaritmo.

Para $r=p/q$, los argumentos son raíces de unidad:

$$
e^{2\pi i\ell p/q}
$$

Los polos en $s=1$ detectan cierres perfectos.

---

## 14. Residuos y parte finita

Cerca de $s=1$, para $r=p/q$, se espera una expansión:

$$
S_k(p/q;s)
=
\frac{A_k(q)}{s-1}
+
C_k(p,q)
+
O(s-1)
$$

Interpretación:

$$
\boxed{
A_k(q)=\text{recurrencia primaria / cierre perfecto}
}
$$

$$
\boxed{
C_k(p,q)=\text{estructura fina / recurrencia parcial}
}
$$

Así:

$$
\text{residuo}
\Rightarrow q
$$

$$
\text{parte finita}
\Rightarrow p\text{ y estructura pre-resonante}
$$

---

## 15. Versión recíproca Fourier-Zeta

La función recíproca analítica puede escribirse como:

$$
\boxed{
\mathcal S_k(r;s)
=
\frac12
\left[
S_k(r;s)+S_k(1/r;s)
\right]
}
$$

o:

$$
\boxed{
\mathcal S_k(r;s)
=
\frac12
\sum_{\ell=-k}^{k}
c_\ell^{(k)}
\left[
\operatorname{Li}_s(e^{2\pi i\ell r})
+
\operatorname{Li}_s(e^{2\pi i\ell/r})
\right]
}
$$

Esta forma permite un análisis espectral de la sonancia respetando la reciprocidad:

$$
r\leftrightarrow1/r
$$

---

## 16. Timbres y espectros

Un timbre o espectro se representa como:

$$
\mathcal F=
\{(f_i,A_i)\}_{i=1}^{M}
$$

donde $f_i$ son frecuencias parciales y $A_i$ amplitudes.

La sonancia total del espectro se define como suma ponderada sobre pares:

$$
\boxed{
\operatorname{Son}_k(\mathcal F;N)
=
\sum_{i<j}
A_i^2A_j^2\,
\mathcal S_k\left(\frac{f_j}{f_i};N\right)
}
$$

Como:

$$
\mathcal S_k(r;N)=\mathcal S_k(1/r;N)
$$

no importa si se usa $f_j/f_i$ o $f_i/f_j$.

Esto permite tratar de forma unificada:

- tonos puros;
- tonos compuestos;
- acordes;
- timbres armónicos;
- timbres inarmónicos;
- espectros arbitrarios.

---

## 17. Timbres en Fourier-Zeta

La versión analítica de la sonancia de un timbre es:

$$
\boxed{
\operatorname{Son}_k(\mathcal F;s)
=
\sum_{i<j}
A_i^2A_j^2\,
\mathcal S_k\left(\frac{f_j}{f_i};s\right)
}
$$

Expandiendo:

$$
\operatorname{Son}_k(\mathcal F;s)
=
\frac12
\sum_{i<j}
A_i^2A_j^2
\sum_{\ell=-k}^{k}
c_\ell^{(k)}
\left[
\operatorname{Li}_s
\left(e^{2\pi i\ell f_j/f_i}\right)
+
\operatorname{Li}_s
\left(e^{2\pi i\ell f_i/f_j}\right)
\right]
$$

Esta es una función espectral compleja asociada al timbre.

Su residuo captura cierres perfectos ponderados por amplitud.

Su parte finita captura textura, recurrencias parciales y organización interna.

---

## 18. Relación con Sethares

Sethares modela la disonancia de timbres compuestos sumando contribuciones de rugosidad entre pares de parciales.

La arquitectura general es:

$$
D(r)=
\sum_{i<j}
w_{ij}
\operatorname{Roughness}(f_i,f_j)
$$

La propuesta de sonancia usa una arquitectura similar, pero reemplaza la rugosidad psicoacústica elemental por recurrencia ondulatoria:

$$
\operatorname{Son}_k(\mathcal F)
=
\sum_{i<j}
w_{ij}
\mathcal S_k(f_j/f_i)
$$

Diferencia conceptual:

| Sethares | Sonancia |
|---|---|
| Rugosidad perceptiva | Recurrencia ondulatoria |
| Dominio frecuencial/perceptivo | Dominio temporal/autocorrelativo |
| Mínimos de disonancia | Máximos de sonancia |
| Pares de parciales | Pares de parciales |
| Timbre determina escala | Timbre determina paisaje de recurrencias |

---

## 19. Interpretación física

La consonancia se interpreta como estabilidad recurrente del patrón de interferencia.

Un intervalo es más sonante si:

1. cierra pronto;
2. tiene recurrencias parciales fuertes;
3. esas recurrencias aparecen temprano;
4. el patrón es estable bajo inversión de referencia;
5. en timbres compuestos, muchos pares de parciales presentan relaciones recurrentes.

En forma compacta:

$$
\boxed{
\text{consonancia}
=
\text{recurrencia temprana}
+
\text{recurrencia parcial coherente}
+
\text{simetría recíproca}
}
$$

---

## 20. Definición recomendada para la siguiente versión

La función principal recomendada es:

$$
\boxed{
\mathcal S_k(r;N)=
\frac12
\left[
S_k^*(r;N)+S_k^*(1/r;N)
\right]
}
$$

con:

$$
\boxed{
S_k^*(r;N)=
\sum_{n=1}^{N}
\frac{\cos^{2k}(\pi rn)}{n}
-
\frac12\cos^{2k}(\pi r)
}
$$

o de forma equivalente:

$$
\boxed{
\mathcal S_k(r;N)=
\frac12
\sum_{n=1}^{N}
w_n
\left[
\cos^{2k}(\pi rn)
+
\cos^{2k}\left(\frac{\pi n}{r}\right)
\right]
}
$$

donde:

$$
w_1=\frac12,\qquad w_n=\frac1n\quad(n\ge2)
$$

Esta función:

- recibe $r$, no $p,q$;
- es continua en el ratio;
- es recíproca;
- incorpora regularización;
- recupera información aritmética cuando $r=p/q$;
- se transporta naturalmente a Fourier/Zeta;
- sirve como núcleo para timbres compuestos.

---

## 21. Cadena conceptual final

$$
s(t)
\Rightarrow
R(\tau)
\Rightarrow
R(n)=\cos^2(\pi rn)
$$

$$
\Rightarrow
S_k(r;N)
\Rightarrow
S_k^*(r;N)
\Rightarrow
\mathcal S_k(r;N)
$$

$$
\Rightarrow
\operatorname{Son}_k(\mathcal F;N)
=
\sum_{i<j}
A_i^2A_j^2
\mathcal S_k(f_j/f_i;N)
$$

$$
\Rightarrow
\mathcal S_k(r;s)
=
\frac12
\sum_{\ell=-k}^{k}
c_\ell^{(k)}
\left[
\operatorname{Li}_s(e^{2\pi i\ell r})
+
\operatorname{Li}_s(e^{2\pi i\ell/r})
\right]
$$

---

## 22. Tesis compacta

La sonancia no debe entenderse como una función aritmética de numeradores y denominadores, sino como una función recíproca y regularizada del ratio continuo $r$.

Los números $p,q$ emergen solo cuando $r$ es racional:

$$
r=\frac pq
$$

Entonces:

- $q$ aparece como cierre desde el marco de $f_1$;
- $p$ aparece como cierre desde el marco de $f_2$;
- la parte finita y las recurrencias parciales contienen la estructura fina;
- el transporte Fourier/Zeta permite análisis espectral de la sonancia.

La definición central queda:

$$
\boxed{
\mathcal S_k(r;N)=
\frac12
\left[
S_k^*(r;N)+S_k^*(1/r;N)
\right]
}
$$

y su extensión a timbres:

$$
\boxed{
\operatorname{Son}_k(\mathcal F;N)
=
\sum_{i<j}
A_i^2A_j^2
\mathcal S_k(f_j/f_i;N)
}
$$
