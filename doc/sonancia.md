# Sonancia: Formulación Continua Simplificada

## Resumen
Se introduce una formulación continua de la **sonancia** como medida de recurrencia autocorrelativa temprana entre relaciones frecuenciales. El objeto primario del modelo no es una fracción racional $p/q$, sino un ratio continuo $r \in \mathbb{R}^+$.

La estructura racional aparece solo como caso particular cuando $r=p/q$. El modelo principal se define mediante:

* **Núcleo autocorrelativo**: $K_k(r,n) = \cos^{2k}(\pi rn)$
* **Sonancia direccional**: $S_k(r;N) = \sum_{n=1}^{N} \frac{K_k(r,n)}{n}$
* **Sonancia recíproca continua**: $\Sigma_k(r;N) = \frac{1}{2} [S_k(r;N) + S_k(r^{-1};N)]$

---

## 1. Introducción
La teoría de la sonancia propone derivar una medida de consonancia desde la recurrencia temporal del patrón ondulatorio. La tesis principal es:

> **La sonancia es concentración temprana, continua y recíproca de autocorrelación en el espacio de ratios.**

---

## 2. Derivación Autocorrelativa Elemental
Consideremos dos tonos puros con ratio frecuencial $r = f_2/f_1$. La autocorrelación ideal $R(\tau)$ evaluada en retardos enteros del periodo de $f_1$ ($\tau = n T_1$) produce la identidad central:

$$R(n;r) = \cos^2(\pi rn)$$

---

## 3. Núcleo de Recurrencia y Resolución
Definimos el núcleo filtrado por un parámetro de resolución $k \in \mathbb{N}$:

$$K_k(r,n) = \cos^{2k}(\pi rn)$$

El parámetro $k$ controla la selectividad: a mayor $k$, las "ventanas resonantes" son más estrechas, comportándose cerca de un entero como una gaussiana:
$$K_k(r,n) \approx e^{-k\pi^2\varepsilon^2}$$

---

## 4. Sonancia Direccional
La sonancia acumulada hasta un horizonte $N$ pondera las recurrencias tempranas mediante $1/n$:

$$S_k(r;N) = \sum_{n=1}^{N} \frac{\cos^{2k}(\pi rn)}{n}$$

Propiedades:
* $0 \le S_k(r;N) \le H_N$ (donde $H_N$ es el número armónico).
* En el unísono: $S_k(1;N) = H_N$.

---

## 5. Sonancia Recíproca Continua
Para que la medida sea invariante ante la inversión del ratio ($r \leftrightarrow r^{-1}$), definimos la función principal:

$$\Sigma_k(r;N) = \frac{1}{2} \sum_{n=1}^{N} \frac{\cos^{2k}(\pi rn) + \cos^{2k}(\pi n/r)}{n}$$

### 5.1 Normalización
Para comparar distintos horizontes $N$, se define la versión normalizada en el rango $[0, 1]$:

$$\widehat{\Sigma}_k(r;N) = \frac{\Sigma_k(r;N)}{H_N}$$

---

## 6. Coordenada Logarítmica (Octavas)
Dado que los intervalos son aditivos en log-frecuencia, usamos $u = \log_2 r$:

$$\Sigma_k^{\text{oct}}(u;N) = \Sigma_k(2^u;N)$$

Esta función es par: $\Sigma_k^{\text{oct}}(u;N) = \Sigma_k^{\text{oct}}(-u;N)$.

---

## 7. Formulación Compleja y Fourier
El núcleo admite una expansión finita de Fourier:

$$K_k(r,n) = \sum_{\ell=-k}^{k} c_\ell^{(k)} e^{2\pi i\ell rn}$$

Donde los coeficientes son binomiales: $c_\ell^{(k)} = 2^{-2k} \binom{2k}{k+\ell}$. Esto permite expresar la sonancia mediante sumas exponenciales truncadas $D_N(x)$.

---

## 8. Disección Racional: $r = p/q$
Cuando el ratio es una fracción irreducible $p/q$:
* **$q$ (denominador)**: Determina el tiempo del **cierre directo**.
* **$p$ (numerador)**: Determina el tiempo del **cierre recíproco**.

La sonancia $\Sigma_k(p/q;N)$ incorpora ambos cierres simultáneamente. En el límite $k \to \infty$, la sonancia de ciclo tiende a $1/q$.

---

## 9. Extensión a Timbres Compuestos
Para un espectro $\mathcal{F} = \{(f_i, A_i)\}$, la sonancia total es la suma ponderada por energía de todos los pares de parciales:

$$\text{Son}_k(\mathcal{F};N) = \sum_{i<j} w_{ij} \Sigma_k\left(\frac{f_j}{f_i};N\right)$$

Donde los pesos energéticos son:
$$w_{ij} = \frac{A_i^2 A_j^2}{\sum_{m<n} A_m^2 A_n^2}$$

---

## 10. Comparación Estructural
| Característica | Modelo de Sethares | Modelo de Sonancia |
| :--- | :--- | :--- |
| **Base** | Rugosidad psicoacústica | Recurrencia física (autocorrelación) |
| **Optimización** | Mínimos de disonancia | Máximos de sonancia |
| **Naturaleza** | Modelo perceptivo | Modelo temporal-analítico |



---

## Anexo: Resumen de Ecuaciones Cerradas

1.  **Núcleo**: $K_k(r,n) = \cos^{2k}(\pi rn)$
2.  **Direccional**: $S_k(r;N) = \sum_{n=1}^{N} \frac{K_k(r,n)}{n}$
3.  **Principal**: $\Sigma_k(r;N) = \frac{1}{2} [S_k(r;N) + S_k(r^{-1};N)]$
4.  **Espectral**: $\text{Son}_k(\mathcal{F};N) = \sum_{i<j} w_{ij} \Sigma_k(\rho_{ij};N)$