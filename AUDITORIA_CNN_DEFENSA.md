# AUDITORÍA TÉCNICA PARA PREPARACIÓN DE DEFENSA — CNN / IA

> **Alcance (2026-10-05).** Auditoría estática de `CNNSmog`. No se reentrenó, alteró ni ejecutó el proyecto, no se accedió a Google Drive ni se presupone nada del proyecto web separado. El único añadido es este informe solicitado.
>
> **Etiquetas de evidencia:** `[CONFIRMADO POR CÓDIGO]`, `[CONFIRMADO POR RESULTADOS/ARCHIVOS]`, `[INFERIDO]`, `[NO ENCONTRADO]` y `[REQUIERE QUE EL AUTOR LO CONFIRME]`.

## 0. Resumen ejecutivo

- `[CONFIRMADO POR CÓDIGO]` Es un clasificador supervisado binario de imágenes: `noSmog` (índice 0) frente a `smog` (índice 1). No detecta/localiza objetos, no segmenta y no estima concentración de contaminantes.
- `[CONFIRMADO POR CÓDIGO]` Usa TensorFlow/Keras, EfficientNetB0 con pesos ImageNet, `GlobalAveragePooling2D`, Dense(128, ReLU), Dropout(0.5) y Dense(1, sigmoid).
- `[CONFIRMADO POR RESULTADOS/ARCHIVOS]` La ejecución incrustada usó TensorFlow 2.19.0 y 880 imágenes balanceadas: 704 train, 88 validation y 88 test (80/10/10; mitad por clase en cada split).
- `[CONFIRMADO POR CÓDIGO]` Entrena en dos fases: cabeza durante 10 épocas con backbone congelado y Adam `1e-3`; luego últimas 20 capas, salvo BatchNorm, durante 10 épocas con Adam `1e-4`.
- `[CONFIRMADO POR RESULTADOS/ARCHIVOS]` El máximo `val_auc` fue 0.99871 en época 19. En test: loss 0.11718965, 84/88 aciertos (0.954545), matriz `[[41,3],[1,43]]`, F1 redondeado 0.95 (`noSmog`) y 0.96 (`smog`).
- **Hallazgo crítico:** se encontraron 2 duplicados exactos train–validation y 1 train–test mediante SHA-1, pero no se retiraron. Hay fuga de datos y el test no es totalmente independiente.
- `[NO ENCONTRADO]` Dataset, modelos, checkpoints, reportes externos, requirements, documentación de procedencia, tests o scripts independientes. Solo estaba versionado el notebook con sus salidas incrustadas.
- El código llama “final” a `final.keras`, pero evalúa y demuestra `best.keras`. `[REQUIERE QUE EL AUTOR LO CONFIRME]` cuál se desplegó.

---

# 1. INVENTARIO

## Árbol simplificado

```text
CNNSmog/
├── .git/                                      # 2 commits
├── .gitkeep                                   # vacío
├── PiscoNawi_EfficientNetB0_Binario_Colab.ipynb
└── AUDITORIA_CNN_DEFENSA.md                   # este informe, no original
```

Antes del informe había un único artefacto funcional: notebook de ~2.4 MB, 36 celdas (21 código, 15 Markdown). El historial solo contiene el inicio y “Creado con Colab”; no documenta iteraciones.

| Ruta/archivo | Propósito | Importancia para defensa |
|---|---|---|
| `PiscoNawi_EfficientNetB0_Binario_Colab.ipynb` | Flujo de Drive, datos, modelo, entrenamiento, evaluación e inferencia por lote; conserva salidas. | **Crítica (A)**: fuente primaria única. |
| `.git/` | Historial mínimo. | **D**: confirma ausencia de evolución versionada. |
| `.gitkeep` | Marcador vacío. | **D**. |
| `AUDITORIA_CNN_DEFENSA.md` | Reconstrucción solicitada. | Guía secundaria; no sustituye evidencia primaria. |

El run observado fue `run_20260211_122709`, bajo `/content/drive/MyDrive/Data/guardado/experimentos/...`. El notebook genera/referencia, pero el repositorio **no contiene**:

```text
models/{best.keras,last.keras,final.keras}
logs/ (TensorBoard)
reports/{history.csv,history_full.json,dup_report.json,
         confusion_matrix.json,classification_report.txt}
figures/{loss,accuracy,auc,precision,recall,errors_FP,errors_FN}.png
snapshots/dataset_snapshot.json
```

`[NO ENCONTRADO]` Tampoco hay `.py`, dataset/imágenes, README, tesis, requirements/lockfile, configuración, pruebas automatizadas ni código de producción/integración web.

---

# 2. PROBLEMA Y RECONSTRUCCIÓN HISTÓRICA

## Qué resuelve

Una imagen RGB redimensionada a 224×224 produce un escalar sigmoide interpretado como `p(smog)`. El notebook decide `smog` si `p >= 0.5`; de lo contrario, `noSmog`. Es clasificación **a nivel de imagen**.

`[NO ENCONTRADO]` Definición operacional de smog, sensor/medición de referencia, lugar, fecha, clima, cámara, criterio de anotación o calibración probabilística. El score sigmoide no demuestra por sí solo una probabilidad calibrada ni una medición física de calidad del aire.

## Flujo demostrado

```text
Dataset ya organizado en Drive: smog/{train,val,test}/{smog,noSmog}
→ validar carpetas y contar
→ snapshot + hashes SHA-1 entre splits
→ etiquetas Keras alfabéticas: noSmog=0, smog=1
→ resize 224×224, batch 32
→ augmentation solo train + preprocess_input + cache/prefetch
→ EfficientNetB0/ImageNet sin top + cabeza binaria
→ fase 1 congelada, 10 épocas
→ cargar best y afinar últimas 20 capas, 10 épocas
→ checkpoint por val_auc + guardar final.keras
→ recargar best.keras y evaluar test
→ matriz, reporte, FP/FN y predicción por lote
```

`[NO ENCONTRADO]` Adquisición, limpieza, deduplicación, etiquetado o creación de splits: llegan ya hechos. Tampoco hay comparación con CNN propia, ResNet, MobileNet, YOLO u otros modelos, búsqueda de hiperparámetros o experimentos descartados. `[REQUIERE QUE EL AUTOR LO CONFIRME]` Si hubo pruebas previas y por qué se eligió EfficientNetB0.

---

# 3. DATASET

## Estructura y distribución observada

Raíz hardcodeada: `/content/drive/MyDrive/Data/smog`; splits `train`, `val`, `test`; carpetas esperadas `smog`, `noSmog`. Extensiones contadas: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`.

| Split | noSmog | smog | Total | Total global |
|---|---:|---:|---:|---:|
| Train | 352 | 352 | 704 | 80% |
| Validation | 44 | 44 | 88 | 10% |
| Test | 44 | 44 | 88 | 10% |
| **Total** | **440** | **440** | **880** | **100%** |

Las cifras están impresas tanto por el snapshot como por los loaders. No hay desbalance numérico. `[INFERIDO]` 80/10/10 describe lo presente, pero no prueba cómo se hizo la división, si fue aleatoria/estratificada o agrupó escenas relacionadas.

## Etiquetas, formatos y dimensiones

- `labels="inferred"`, `label_mode="binary"`: la carpeta asigna la etiqueta.
- Keras imprimió `['noSmog','smog']`; por ello 0=`noSmog`, 1=`smog`.
- `EXPECTED_CLASSES = ['smog','noSmog']` está en orden inverso, pero solo valida/cuenta; no controla los índices.
- `[NO ENCONTRADO]` Dimensiones, canales y metadatos originales. Loader → 224×224 y modelo → 3 canales.
- `[NO ENCONTRADO]` Política de relación de aspecto; el resize puede deformar imágenes no cuadradas.
- `[NO ENCONTRADO]` Autor/protocolo de etiquetas, acuerdo de anotadores, calidad, licencia y procedencia.

## Transformaciones

Train: `float32` → flip horizontal → rotación factor 0.03 (aprox. ±10.8°) → zoom 0.10 → contraste 0.10 → `efficientnet.preprocess_input`. Validation/test: cast + preprocess, sin augmentation. Después se usa `.cache().prefetch(AUTOTUNE)`.

**Matiz:** en TensorFlow/Keras EfficientNet actual el reescalado está incorporado en el modelo y `preprocess_input` se mantiene esencialmente como pass-through. No debe describirse como normalización por media/desviación sin evidencia de esta versión.

**Riesgo de pipeline:** augmentation ocurre antes de `.cache()`. `[INFERIDO]` La primera iteración queda materializada y se reutiliza, reduciendo diversidad entre épocas. Seed 42 se fija en Python, NumPy, TensorFlow y capas, pero no se demuestra determinismo completo.

## Integridad y posibles problemas

- `[CONFIRMADO POR RESULTADOS/ARCHIVOS]` 2 duplicados exactos train–val y 1 train–test; se reportan, no se eliminan.
- `[INFERIDO]` El resultado puede estar beneficiado; no puede cuantificarse sin datos/modelo.
- `[NO ENCONTRADO]` Comprobación val–test, near-duplicates, agrupación por cámara/lugar/ráfaga ni evaluación externa/geográfica/temporal.

---

# 4. MODELO / CNN

## Nivel 1 — explicación oral sugerida

“Usé transferencia de aprendizaje con EfficientNetB0. En vez de aprender todos los filtros desde cero con 704 imágenes de entrenamiento, partí de representaciones aprendidas en ImageNet. Quité el clasificador original, resumí los mapas mediante promedio global y añadí una cabeza de 128 neuronas ReLU, dropout y una salida sigmoide. Primero entrené solo la cabeza; luego ajusté las últimas 20 capas con una tasa menor. La salida es un score de smog y se umbraliza en 0.5.”

La arquitectura está confirmada; la motivación es rationale técnico razonable `[INFERIDO]`, no una decisión histórica documentada.

## Nivel 2 — arquitectura exacta

| Etapa | Configuración/salida | Parámetros |
|---|---|---:|
| Input | `(None,224,224,3)` | 0 |
| EfficientNetB0 | `include_top=False`, ImageNet; `(None,7,7,1280)` | 4,049,571 |
| GlobalAveragePooling2D | `(None,1280)` | 0 |
| Dense | 128, ReLU | 163,968 |
| Dropout | 0.5 | 0 |
| Dense output | 1, sigmoid | 129 |
| **Total** | `EfficientNetB0_binary_smog` | **4,213,668** |

Fase inicial: 164,097 entrenables y 4,049,571 no entrenables. Sigmoid + binary cross-entropy son coherentes con dos clases exclusivas; no hace falta softmax de dos salidas.

### Por qué funciona así

- Las convoluciones forman mapas desde patrones locales; EfficientNetB0 reutiliza rasgos visuales generales.
- Transfer learning reduce la necesidad de aprender desde cero con pocos datos.
- GAP resume cada canal y evita una cabeza Flatten enorme.
- ReLU añade no linealidad; Dropout 0.5 regulariza; sigmoid da score binario.
- Fine-tuning con LR menor adapta rasgos tardíos limitando cambios bruscos.
- BatchNorm queda congelada para no alterar estadísticas con dataset/lotes pequeños. Además el backbone se invoca con `training=False`, conservando comportamiento de inferencia para capas sensibles al modo.

`[NO ENCONTRADO]` L1/L2, label smoothing, class weights, mixup/cutmix o calibración.

---

# 5. NOTEBOOK POR BLOQUES LÓGICOS

## 1 — Entorno y destinos (celdas 3–4)
**Objetivo/entrada:** montar Drive y definir rutas. **Proceso/salida:** crea carpetas del run. **Dominar:** persistencia/rutas absolutas. **Pregunta:** ¿cómo reproducir sin esa cuenta? No se puede con el repo. Cambiar rutas afecta acceso, no la matemática.

## 2 — Reproducibilidad (5–6)
Seeds 42; importa TensorFlow/NumPy/Matplotlib; salida: TF 2.19.0, `GPUs: []`. Cambiar seed puede variar shuffle, aumento e inicialización. Que no haya GPU en esa sesión implica CPU visible, aunque el autor debe confirmar contexto.

## 3 — Verificación/snapshot (7–9)
Valida 3 splits/2 clases y escribe conteos. Conceptos: split, balance y etiqueta por carpeta. Pregunta clave: cómo se creó 80/10/10; no consta.

## 4 — Anti-leakage (10–11)
Calcula SHA-1 y escribe `dup_report.json`; resultado 2/1. Pregunta incómoda: por qué continuó sin retirarlos. Desactivarlo ahorra tiempo pero elimina un control crítico.

## 5 — Pipeline (12–14)
Loader, resize, batch, shuffle solo train, augmentation, cache/prefetch. Cambiar batch altera memoria/número de actualizaciones; augmentation excesiva puede destruir señales. Debe poder explicarse el riesgo cache-after-augmentation.

## 6 — Preview (15–16)
Muestra nueve imágenes raw y etiqueta numérica. Es sanity check, no auditoría sistemática de etiquetas.

## 7 — Construcción (17–18)
Descarga pesos ImageNet y compone el modelo. Pregunta: ¿por qué B0? No hay benchmark. Descongelar todo al inicio aumentaría riesgo de dañar representaciones con gradientes iniciales.

## 8 — Compilación/callbacks (19–22)
BCE, Adam, accuracy/AUC/precision/recall. `best.keras` monitoriza `val_auc`; `last.keras` cada época; TensorBoard; ReduceLR factor 0.5, paciencia 3, mínimo `1e-6`; EarlyStopping paciencia 6/restaura best; CSVLogger. Cambiar monitor cambia el compromiso seleccionado.

## 9 — Fase congelada (23–24)
10 épocas, LR `1e-3`; época 10: val AUC 0.99174. Comprender transferencia congelada y curvas.

## 10 — Fine-tuning (25)
Carga best; habilita últimas 20 capas excepto BatchNorm; recompila a `1e-4`; épocas 11–20. Máximo val AUC 0.99871 en 19, ReduceLR en 18, restaura 19 y guarda `final.keras`. “¿Por qué 20?” no está justificado. Más capas = más adaptación y riesgo/coste.

## 11 — Historial/gráficas (26–28)
Concatena ambos historiales y guarda JSON + curvas de loss, accuracy, AUC, precision y recall. Al interpretar, marcar cambio de régimen en época 11.

## 12 — Evaluación (29–31)
Recarga `best.keras`, evalúa test, umbraliza 0.5, genera matriz y reporte. Cambiar threshold altera FP/FN/precision/recall, no AUC. El diccionario de Keras agrupa métricas como `compile_metrics`, por lo que no rotula todas correctamente.

## 13 — Errores (32–33)
Alinea test raw/procesado; visualiza 3 FP y 1 FN. Sin dataset no se pueden diagnosticar sus patrones visuales.

## 14 — “Modelo listo” (34–35)
Carga best y predice primer lote preprocesado. No carga una imagen individual ni devuelve nombre de clase/API; es smoke test de lote.

---

# 6. ENTRENAMIENTO

## Fase A

| Elemento | Valor |
|---|---|
| Datos | 704 train / 88 val |
| Imagen / batch | 224×224×3 / 32; 22 batches impresos |
| Inicialización | EfficientNetB0 ImageNet |
| Entrenable | cabeza, 164,097 parámetros |
| Loss / optimizer | BinaryCrossentropy / Adam |
| LR / épocas | `1e-3` / 10 ejecutadas |
| Selección | máximo `val_auc` |

No intervino early stopping porque siguió mejorando el monitor. En época 10: train accuracy 0.9856, val accuracy 0.9091, val AUC 0.9917.

## Fase B

Carga best de A; localiza backbone; congela todo salvo últimas 20 y todas las BatchNorm; recompila Adam `1e-4`; ejecuta 10 épocas (`initial_epoch=10`); comparte callbacks/rutas; LR baja a ~`5e-5` tras época 18; restaura época 19 y guarda `final.keras`. `[INFERIDO]` `best.keras` conserva el mejor val AUC global de ambas fases.

`[NO ENCONTRADO]` Múltiples seeds/runs, validación cruzada, intervalos, búsqueda sistemática o prueba estadística. El test se consulta después de selección, pero incluye al menos un duplicado de train.

---

# 7. MÉTRICAS Y RESULTADOS

## Resultados finales

Con positivo=`smog`, filas reales y columnas predichas:

| | Pred. noSmog | Pred. smog |
|---|---:|---:|
| Real noSmog | TN=41 | FP=3 |
| Real smog | FN=1 | TP=43 |

| Métrica | Qué mide / por qué importa | Valor real encontrado |
|---|---|---:|
| Test BCE loss | Penaliza probabilidades erróneas confiadas | 0.1171896532 |
| Accuracy | Fracción correcta; interpretable aquí por balance | 0.9545454383 = 84/88 |
| Precision smog | TP/(TP+FP) | 43/46=0.9348; reporte 0.93 |
| Recall smog | TP/(TP+FN) | 43/44=0.9773; reporte 0.98 |
| F1 smog | Media armónica | 0.96 redondeado |
| Precision noSmog | Precisión de esa clase | 0.98 redondeado |
| Recall noSmog | 41/44 | 0.93 redondeado |
| F1 noSmog | Balance de ambas | 0.95 redondeado |
| Macro/weighted F1 | Promedios entre clases | 0.95 redondeado |
| Mejor val AUC | Ranking a través de umbrales; selección | 0.99871, época 19 |

La barra de `evaluate` muestra acumulados/intermedios (`accuracy 0.9460`, `auc 0.7451`, etc.), pero el print final solo da `{'loss': 0.117..., 'compile_metrics': 0.954545...}`. **No debe citarse 0.7451 como test AUC final**: no queda recuperado/rotulado de forma fiable. Precision/recall sí se reconstruyen de la matriz. `[NO ENCONTRADO]` test AUC final claro, ROC/PR curves, mAP (no central a esta tarea), calibración o intervalos.

Los valores son descriptivamente altos en este test interno, pero no prueban generalización: 88 casos, un duplicado con train, fuente desconocida y sin evaluación externa. No se califican como “buenos/malos” fuera de ese contexto.

---

# 8. MODELO FINAL

| Candidato | Generación/uso | Problema |
|---|---|---|
| `best.keras` | Checkpoint de máximo `val_auc`; recargado para test y demostración; ganador época 19. | Archivo ausente del repo. |
| `final.keras` | Guardado tras terminar y restaurar mejores pesos; print lo llama “Modelo final”. | No es el recargado para test; ausente. |
| `last.keras` | Sobrescrito cada época, no best-only. | No vuelve a usarse; puede ser época 20. |

`[CONFIRMADO POR CÓDIGO]` Formato Keras `.keras`, entrada `(224,224,3)`, salida sigmoid escalar, clases 0/1 y pipeline descrito. `[INFERIDO]` `final` debería tener pesos restaurados de época 19 como `best`, pero pueden diferir estados/serialización y no se compararon hashes/predicciones. `[REQUIERE QUE EL AUTOR LO CONFIRME]` producto efectivamente entregado/desplegado.

---

# 9. INFERENCIA

## Flujo real del notebook

```text
archivo test/{clase}/...
→ image_dataset_from_directory: decode + resize 224×224 + batch + label
→ cast float32 + efficientnet.preprocess_input (sin augmentation)
→ best_model.predict(batch)
→ vector sigmoid p(smog)
→ p >= 0.5
→ 0=noSmog / 1=smog
```

Intervienen loaders (celda 13), `preprocess` (14), `load_model/predict` (30–35) y umbral (31/33).

```python
# Pseudocódigo conceptual; NO existe como función standalone.
imagen = leer_y_decodificar(ruta)
imagen = resize(imagen, (224, 224))
imagen = cast_float32(imagen)
imagen = efficientnet.preprocess_input(imagen)
p_smog = modelo.predict(expand_dims(imagen, 0))[0, 0]
clase = "smog" if p_smog >= 0.5 else "noSmog"
```

`[NO ENCONTRADO]` Función real para archivo arbitrario, validación/manejo de errores, API/JSON o implementación web. Inferencia debe conservar resize, RGB, dtype, preprocesamiento, mapeo y threshold para evitar train-serving skew.

---

# 10. CONCEPTOS QUE SE DEBEN DOMINAR

| Concepto | Relación directa con el proyecto |
|---|---|
| Clasificación binaria supervisada | Define tarea, etiquetas y salida; no confundir con detección. |
| CNN, convolución, filtros, feature maps | EfficientNet transforma imagen en `(7,7,1280)`. |
| EfficientNet/compound scaling | Backbone concreto; no atribuir comparativas inexistentes. |
| Transfer learning/ImageNet | Explica pesos y utilidad con dataset pequeño. |
| Freeze/fine-tuning | Dos fases reales y capas que reciben gradiente. |
| GAP | Resume mapas con pocos parámetros. |
| ReLU/sigmoid/BCE | Cabeza no lineal y aprendizaje binario. |
| Adam/learning rate | `1e-3`, `1e-4` y reducción automática. |
| Epoch/batch | 20 épocas, batch 32; distinguir de cada actualización. |
| Augmentation/cache | Cuatro transformaciones y su problema de orden. |
| Dropout/overfitting | Regularización y brecha train-validation. |
| Train/validation/test | Entrenamiento, selección, estimación; leakage viola independencia. |
| Accuracy/precision/recall/F1 | Se reportan y derivan de 3 FP/1 FN. |
| AUC/threshold | AUC selecciona; 0.5 decide; cumplen roles distintos. |
| Matriz de confusión | Base para tipos/costes de error. |
| Callbacks | Checkpoint, early stopping, LR y trazabilidad. |
| BatchNorm | Congelada en fine-tuning por estadísticas internas. |
| Reproducibilidad/leakage | Semillas, versiones, artefactos y duplicados. |

---

# 11. PREGUNTAS DEL TRIBUNAL

## Básicas

- **¿Qué tarea resuelve y qué significa su salida?** Surge de sigmoid binaria. Estudiar celdas 13, 18, 31; clasificación, etiquetas y threshold.
- **¿Cuántas imágenes/clases hay?** Snapshot 8–13; balance/splits.
- **¿Qué arquitectura usó?** Celda 18; backbone, cabeza y transferencia.
- **¿Qué es un falso positivo aquí?** Celdas 31–33; matriz.

## Intermedias

- **¿Por qué congelar y luego afinar?** Celdas 24–25; transferencia, forgetting, LR.
- **¿Por qué BCE+sigmoid y no softmax?** Celdas 18/20; salida binaria.
- **¿Por qué 224 y batch 32?** Celda 13; compatibilidad/memoria, pero elección histórica no consta.
- **¿Para qué sirven los seis callbacks?** Celda 22; checkpoint/early stop/scheduler/logs.
- **¿Por qué seleccionar AUC y decidir a 0.5?** Celdas 22–31; ranking vs decisión.
- **¿Cómo se convierte una imagen en clase?** Celdas 13–14/31/35; pipeline.

## Difíciles

- **¿Por qué B0 frente a ResNet/MobileNet/CNN propia?** No hay comparación; estudiar celda 18 y reconocer ausencia.
- **¿Cómo sabe que aprende smog y no paisaje/iluminación?** Fuente desconocida; sesgo/confusores/validación externa.
- **¿Qué efecto tienen BatchNorm congelada y `training=False`?** Celdas 18/25; estadísticas train/inference.
- **¿Por qué test AUC no se reporta claramente?** Celda 30; API Keras/rotulado.
- **¿Cuál es final, best o final?** Celdas 22/25/30/35; selección/serialización.
- **¿Qué reproducibilidad hay?** Celdas 4/6/27; seed no sustituye datos/dependencias.

## Muy incómodas / puntos débiles

- **¿Por qué evaluar tras detectar duplicados train–test?** Celda 11 frente a 30–31; leakage.
- **¿Puede reproducir desde Git?** No: faltan datos/modelos/dependencias.
- **¿De dónde vienen las imágenes y quién etiquetó?** No consta; validez/licencia.
- **¿Qué incertidumbre tiene 95.45% con 88 casos?** Tamaño muestral/intervalos.
- **¿Por qué augmentation antes de cache?** Celda 14; estocasticidad de `tf.data`.
- **¿Hay overfitting?** Train llega a 1.0 y validation fluctúa; riesgo, no prueba definitiva.
- **¿Qué archivo/preprocesamiento usa la web?** Proyecto no disponible; no inventar.
- **¿Puede afirmar contaminación real?** No hay ground truth físico, solo etiquetas visuales.

---

# 12. PUNTOS DÉBILES

## Críticos
1. Fuga confirmada: 2 duplicados train–val y 1 train–test, no retirados.
2. No reproducible desde repo: faltan datos, modelos, reportes y dependencias fijadas.
3. Procedencia/etiquetado ausentes: impide valorar sesgo, licencia y validez.
4. Producto ambiguo: `best.keras` evaluado vs `final.keras` denominado final.

## Altos
5. Test interno pequeño (88) y sin intervalos/test externo.
6. Riesgo de overfitting: train llega a accuracy 1.0/loss 0.0066 y validation fluctúa.
7. Augmentation cacheada tras la primera materialización.
8. `evaluate` rotula métricas como `compile_metrics`; test AUC no claro.
9. No existe inferencia individual de producción.

## Medios
10. Rutas absolutas y dependencia Google Drive/Colab.
11. `pip install scikit-learn` sin versión; solo TF 2.19.0 quedó impreso.
12. Pesos ImageNet dependen de descarga/red.
13. Sin comparación/justificación registrada de hiperparámetros/threshold.
14. Solo SHA-1 exacto; no near-duplicates ni val–test.
15. Orden de `EXPECTED_CLASSES` contrario a índices Keras (riesgo humano).
16. Resize sin analizar relación de aspecto.
17. Historial Git mínimo, sin experimentos.
18. `execution_count=null`, aunque orden/salidas son coherentes.
19. Sin tests, validación de corruptos o contrato de entrada.
20. Sin análisis de coste FP vs FN ni optimización de threshold.

---

# 13. MAPA DE ESTUDIO (NO calendario)

## A — dominar y explicar sin ayuda
Tarea/mapeo; conteos y 80/10/10; leakage; arquitectura/dimensiones; transferencia y dos fases; loss/Adam/LR; inferencia/0.5; matriz y métricas; límites de generalización. Son el núcleo defendible.

## B — entender y explicar lógica
Augmentation/cache; callbacks; GAP/dropout/BatchNorm; EfficientNet conceptual; reproducibilidad; curvas/overfitting. Sostienen preguntas medias/difíciles.

## C — localizar y reconocer
Rutas/nombres de artefactos; conteo exacto de parámetros; detalles de plotting/JSON/CSV; formatos y SHA-1. Importan como referencia, no para memorizar.

## D — poco relevante
Badge Colab, `.gitkeep`, prints decorativos, HTML/PNG incrustado e historial mínimo, salvo al hablar de trazabilidad.

**Código prioritario:** celdas 13–14 (datos), 18 (modelo), 20/22 (compilación/selección), 24–25 (entrenamiento), 30–31 (evaluación), 35 (predicción). Secundario: 8–11 (integridad), 27–28 (historial), 33 (errores). Auxiliar: Drive/directorios/preview.

---

# 14. HECHOS VS INFERENCIAS

| Conclusión | Estado |
|---|---|
| Tarea binaria smog/noSmog | `[CONFIRMADO POR CÓDIGO]` |
| 880 imágenes balanceadas | `[CONFIRMADO POR RESULTADOS/ARCHIVOS]` |
| Reparto presente 80/10/10 | `[CONFIRMADO]`; método de creación `[NO ENCONTRADO]` |
| EfficientNetB0/ImageNet, TF 2.19 | `[CONFIRMADO]` |
| Dos fases de 10 épocas | `[CONFIRMADO]` |
| Selección por máximo val AUC | `[CONFIRMADO POR CÓDIGO]` |
| Época 19: val AUC 0.99871 | `[CONFIRMADO POR RESULTADOS/ARCHIVOS]` |
| Test 84/88, 3 FP, 1 FN | `[CONFIRMADO POR RESULTADOS/ARCHIVOS]` |
| Puede haber inflación por fuga | Duplicado `[CONFIRMADO]`; magnitud `[NO ENCONTRADO]` |
| best y final predicen idéntico | `[INFERIDO]`, no comparado |
| best fue desplegado en web | `[REQUIERE QUE EL AUTOR LO CONFIRME]` |
| B0 superó alternativas | `[NO ENCONTRADO]` |
| Dataset representa escenarios diversos | `[NO ENCONTRADO]` |
| Modelo mide contaminación física | `[NO ENCONTRADO]`; clasifica etiquetas visuales |

---

# PREGUNTAS PARA JOSÉ

1. ¿Fuente, licencia, lugar/fecha/dispositivo y criterio exacto de las 880 imágenes?
2. ¿Quién etiquetó y con qué ground truth/revisión/acuerdo?
3. ¿Cómo se creó 80/10/10: aleatorio, estratificado y agrupado por escena?
4. ¿Por qué quedaron los duplicados? ¿Existe evaluación limpia posterior?
5. ¿Hay datos originales, limpieza o scripts no versionados?
6. ¿Se probaron otras arquitecturas/hiperparámetros y dónde están resultados?
7. ¿Por qué augmentation, dropout 0.5, batch 32, 20 capas y threshold 0.5?
8. ¿`run_20260211_122709` es el run definitivo de tesis?
9. ¿Qué archivo se entregó/desplegó? Aportar copia, tamaño y hash.
10. ¿La web replica resize, RGB, dtype, preprocessing, clases y threshold? (No se inspeccionó.)
11. ¿Siguen en Drive snapshot, dup report, histories, logs, figuras y modelos?
12. ¿Versiones exactas de Python/Keras/NumPy/scikit-learn y hardware?
13. ¿El entrenamiento fue intencionalmente en CPU (`GPUs: []`)?
14. ¿Hay test externo, imágenes nuevas, intervalos o resultados posteriores?
15. ¿Cuál es el coste práctico relativo de FP y FN?
16. ¿Cómo define la tesis “smog”: presencia visual o contaminación medida?
17. ¿La tesis contiene cifras diferentes que deban reconciliarse?

---

# DATOS PARA PLANIFICACIÓN

> Caracterización para trabajo posterior; **no es un calendario de 15 días**.

## Cantidad
1 notebook/36 celdas; 14 bloques lógicos (8 centrales); 1 arquitectura; 2 fases; 6 callbacks; 880 muestras reportadas; 3 modelos y múltiples artefactos externos ausentes; 20 debilidades y 17 preguntas históricas.

## Dificultad

| Área | Nivel | Motivo |
|---|---|---|
| Problema/clases/pipeline | Media | Precisión conceptual; no confundir score con medición física. |
| CNN/EfficientNet/transfer | Alta | Convolución, representación y fine-tuning. |
| Entrenamiento/callbacks | Media-alta | Dos fases, recompilación, AUC. |
| Métricas/matriz | Media-alta | Derivación y umbrales. |
| Dataset/validez | Alta | Procedencia, leakage y generalización. |
| Reproducibilidad/despliegue | Alta | Faltan artefactos; best/final ambiguo. |

## Dependencias conceptuales
Imágenes/tensores y clasificación → CNN/activaciones/pooling → EfficientNet/transfer → BCE/gradiente/Adam/batch/LR → freeze/fine-tuning/regularización → splits/overfitting/leakage → matriz/métricas/AUC/threshold → inferencia reproducible/generalización.

## Prioridades
1. Narrar pipeline y arquitectura de memoria.
2. Explicar fuga, procedencia ausente y límites honestamente.
3. Derivar métricas desde TN=41, FP=3, FN=1, TP=43.
4. Diferenciar best/final/last y confirmar desplegado.
5. Practicar fine-tuning, AUC vs threshold y BatchNorm.
6. Recuperar de Drive evidencia separada sin fingir que estaba versionada.

## Conocimientos previos
Python básico; arrays/tensores; álgebra/probabilidad conceptual; aprendizaje supervisado y splits; gradiente; lectura de tablas/curvas.

## Requiere práctica, no solo lectura
Dibujar `224×224×3 → EfficientNet → 7×7×1280 → GAP → 128 → 1`; calcular métricas; explicar AUC vs 0.5; indicar capas con gradiente; defender limitaciones; recorrer inferencia y detectar preprocesamiento distinto; ensayar respuestas sobre duplicados, tamaño, ausencia de test externo/modelo.

## Diagnóstico final
El núcleo es compacto: un notebook, un modelo y un run. La dificultad no es memorizar código, sino sostener metodología y límites. Antes del futuro programa intensivo, deben responderse las preguntas a José y recuperarse dataset/snapshot, reportes, modelos y documentación de procedencia como evidencia separada.
