
## Training plain model
```python
es = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
if name == "plain":
    # training with early stopping and logging
    b = lgb.train(BASE_PARAMS, lgb.Dataset(X_tr, y_tr), rounds,
                    valid_sets=[lgb.Dataset(X_va, y_va)], callbacks=es)
    return lambda X: b.predict(X, num_iteration=b.best_iteration)
```

This code trains a **LightGBM** gradient boosting model with **early stopping** enabled and returns a predictor function that automatically uses the optimal number of boosting iterations.

---

### Step-by-Step Breakdown

#### 1. Setting Up Callbacks (`es`)

```python
es = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]

```

* **`lgb.early_stopping(stopping_rounds=50, verbose=False)`**:
* Halts training if the validation loss does not improve for **50 consecutive rounds**.
* `verbose=False` suppresses console printouts every time an early stopping check occurs.


* **`lgb.log_evaluation(period=0)`**:
* Setting `period=0` (or `False`) completely silences iteration log messages to keep terminal output clean.



---

#### 2. Model Training (`lgb.train`)

```python
b = lgb.train(
    BASE_PARAMS,
    lgb.Dataset(X_tr, y_tr),
    rounds,
    valid_sets=[lgb.Dataset(X_va, y_va)],
    callbacks=es,
)

```

* **`BASE_PARAMS`**: Dictionary containing hyperparameters (e.g., `objective`, `learning_rate`, `metric`).
* **`lgb.Dataset(X_tr, y_tr)`**: Wraps the training features and labels into LightGBM’s optimized internal dataset representation.
* **`rounds`**: Maximum number of boosting trees to train (often set high, e.g., `1000`, since early stopping controls when to stop).
* **`valid_sets=[lgb.Dataset(X_va, y_va)]`**: Provides the validation dataset. The early stopping callback evaluates performance on this set after each round.
* **`callbacks=es`**: Passes the list containing the early stopping and log suppression callbacks.

---

#### 3. Returning the Prediction Callable

```python
return lambda X: b.predict(X, num_iteration=b.best_iteration)

```

* Returns a lightweight anonymous function (`lambda`) instead of the raw `Booster` object `b`.
* **`num_iteration=b.best_iteration`**:
* Ensures predictions use the tree count from the iteration that scored best on the validation set, discarding any suboptimal trailing trees produced before early stopping triggered.


* Allows you to generate predictions simply by calling the returned function on new data:
```python
predict_fn = train_model(...)
y_pred = predict_fn(X_test)
```

<br><br>

## Training class_weights model
```python        
spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1) # giving more weight to the minority class
b = lgb.train({**BASE_PARAMS, "scale_pos_weight": spw}, lgb.Dataset(X_tr, y_tr), rounds,
                valid_sets=[lgb.Dataset(X_va, y_va)], callbacks=es)
return lambda X: b.predict(X, num_iteration=b.best_iteration)
```

This code trains a **LightGBM** binary classification model with **class imbalance handling** using the `scale_pos_weight` parameter, combined with early stopping.

---

### Step-by-Step Breakdown

#### 1. Calculating `scale_pos_weight` (`spw`)

```python
spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

```

* **Imbalance Ratio**: Calculates the ratio of negative examples (class `0`) to positive examples (class `1`):

$$\text{spw} = \frac{N_{\text{neg}}}{N_{\text{pos}}}$$


* **Division by Zero Protection**: `max((y_tr == 1).sum(), 1)` ensures that if there are zero positive samples in the training set, it divides by `1` instead of raising a `ZeroDivisionError`.
* **Effect**: In imbalanced binary classification (e.g., 95% class `0` and 5% class `1`), each positive instance is weighted higher during loss computation, forcing the model to pay equal attention to the minority class.

---

#### 2. Injecting `scale_pos_weight` into Training

```python
b = lgb.train(
    {**BASE_PARAMS, "scale_pos_weight": spw},
    lgb.Dataset(X_tr, y_tr),
    rounds,
    valid_sets=[lgb.Dataset(X_va, y_va)],
    callbacks=es,
)

```

* **`{**BASE_PARAMS, "scale_pos_weight": spw}`**: Unpacks the existing `BASE_PARAMS` dictionary and dynamically adds or overrides the `scale_pos_weight` parameter with the computed `spw`.
* **`lgb.Dataset(X_tr, y_tr)`**: Creates the LightGBM internal training dataset.
* **`valid_sets=[lgb.Dataset(X_va, y_va)]`**: Validates on `(X_va, y_va)` at each iteration to monitor validation loss or metric.
* **`callbacks=es`**: Applies callbacks such as `early_stopping` and `log_evaluation` to stop training when the validation score stops improving.

---

#### 3. Returning the Optimal Predictor Function

```python
return lambda X: b.predict(X, num_iteration=b.best_iteration)

```

* Returns a callable that accepts a feature matrix `X` and automatically calls `b.predict()` using `b.best_iteration` (the iteration that yielded the lowest validation loss before early stopping kicked in).

---

### Example Behavior

If your training labels `y_tr` consist of **900 zeros** and **100 ones**:

* `spw = 900 / 100 = 9.0`
* LightGBM will scale the gradients of the positive class by a factor of `9.0`, penalizing false negatives much more heavily than false positives.

<br><br>

## Training focal loss model
```python
base = float(np.log(max(y_tr.mean(), 1e-6) / max(1y_tr.mean(), 
1e-6)))
params = {**BASE_PARAMS, "objective": focal_objective()}
dtr = lgb.Dataset(X_tr, y_tr, init_score=np.full(len(y_tr), base))
dva = lgb.Dataset(X_va, y_va, init_score=np.full(len(y_va), base))
b = lgb.train(params, dtr, rounds, valid_sets=[dva], callbacks=[lgb.log_evaluation(0)])
return lambda X: sigmoid(b.predict(X, raw_score=True) + base)
```

This code trains a **LightGBM** model with a **custom Focal Loss objective** and initializes the training using the **prior log-odds (base margin)** of the training labels.

---

### Step-by-Step Breakdown

#### 1. Calculating Prior Log-Odds (`base`)

*(Note: There is a minor typo in the provided snippet: `1y_tr.mean()` is shorthand for `1 - y_tr.mean()`)*.

```python
base = float(np.log(max(y_tr.mean(), 1e-6) / max(1 - y_tr.mean(), 1e-6)))

```

* **Empirical Probability**: $p = \bar{y}_{\text{tr}} = \text{mean}(y_{\text{tr}})$ is the observed positive class proportion.
* **Logit / Log-Odds**:

$$\text{base} = \text{logit}(p) = \ln\left(\frac{p}{1 - p}\right)$$


* **Numerical Stability**: `max(..., 1e-6)` prevents division by zero or evaluating $\ln(0)$ if $p = 0$ or $p = 1$.
* **Why do this?** In standard logistic regression / gradient boosting, starting predictions at the prior log-odds instead of $0.0$ speeds up convergence significantly, especially for imbalanced data where $p \ll 0.5$.

---

#### 2. Configuring Custom Objective (`focal_objective`)

```python
params = {**BASE_PARAMS, "objective": focal_objective()}

```

* LightGBM allows custom loss functions by passing a callable to `"objective"` that computes first-order gradients and second-order hessians.
* **Focal Loss** is specifically designed for extreme class imbalance (e.g., object detection, rare fraud detection). It downweights well-classified easy examples so the model focuses primarily on hard examples.

---

#### 3. Initializing Datasets with `init_score`

```python
dtr = lgb.Dataset(X_tr, y_tr, init_score=np.full(len(y_tr), base))
dva = lgb.Dataset(X_va, y_va, init_score=np.full(len(y_va), base))

```

* **`init_score`**: Sets the initial margin (raw score before sigmoid) for each sample.
* By setting every sample's initial score to `base`, the boosting process starts from the baseline probability $p$ rather than $0.5$ (since $\text{sigmoid}(0) = 0.5$).
* The validation set `dva` is given the exact same baseline offset so metrics are evaluated properly.

---

#### 4. Training the Model

```python
b = lgb.train(
    params,
    dtr,
    rounds,
    valid_sets=[dva],
    callbacks=[lgb.log_evaluation(0)],
)

```

* Trains the tree ensemble to predict the *residual adjustments* on top of `init_score`.
* `callbacks=[lgb.log_evaluation(0)]` suppresses iteration log outputs to keep training quiet.

---

#### 5. Returning Predictor with Base Margin Re-added

```python
return lambda X: sigmoid(b.predict(X, raw_score=True) + base)

```

* **`raw_score=True`**: Tells LightGBM to output the raw accumulated margin values (log-odds delta from trees) rather than probabilities.
* **`+ base`**: Because `b.predict(X)` only predicts the residuals learned by the trees (it does *not* automatically know the initial score used during training), the prior log-odds `base` must be added back explicitly.
* **`sigmoid(...)`**: Maps the combined raw log-odds score back to a calibrated probability between $0$ and $1$:

$$\hat{y} = \frac{1}{1 + e^{-(\text{tree\_score} + \text{base})}}$$


<br><br>

## Training two phase model
```python
# phase 1(Pre-training phase)        
Xu, yu = undersample(X_tr, y_tr)
b1 = lgb.train(BASE_PARAMS, lgb.Dataset(Xu, yu), 150)
# phase 2: fine-tune on the ORIGINAL distribution, starting from the balanced model
b2 = lgb.train(BASE_PARAMS, lgb.Dataset(X_tr, y_tr), 150, init_model=b1,
                valid_sets=[lgb.Dataset(X_va, y_va)], callbacks=es)
return lambda X: b2.predict(X, num_iteration=b2.best_iteration)
```
This code implements a **two-phase transfer learning (pre-train then fine-tune) strategy** to combat class imbalance with LightGBM.

---

### Step-by-Step Breakdown

#### Phase 1: Pre-training on Balanced Data

```python
Xu, yu = undersample(X_tr, y_tr)
b1 = lgb.train(BASE_PARAMS, lgb.Dataset(Xu, yu), 150)

```

1. **`undersample(X_tr, y_tr)`**: Discards excess majority class examples so that the positive and negative classes are balanced (or close to 50/50).
2. **`b1 = lgb.train(..., 150)`**: Trains an initial LightGBM booster (`b1`) for 150 trees strictly on this balanced subset.
* **Why?** On severely imbalanced data, trees often default to predicting the majority class to minimize loss. Training on a balanced subset forces the initial trees to discover meaningful separating features for the minority class without getting swamped by the majority class.



---

#### Phase 2: Fine-Tuning on the Original Imbalanced Data

```python
b2 = lgb.train(
    BASE_PARAMS,
    lgb.Dataset(X_tr, y_tr),
    150,
    init_model=b1,
    valid_sets=[lgb.Dataset(X_va, y_va)],
    callbacks=es,
)

```

1. **`init_model=b1`**: Continues training by taking the 150 trees from `b1` as the starting foundation and boosting additional trees on top of them.
2. **`lgb.Dataset(X_tr, y_tr)`**: Switches back to the **full, natural training distribution** (all data, including all majority samples).
3. **`valid_sets=[...]`, `callbacks=es**`: Evaluates each newly added tree against the validation set and applies early stopping (`es`).
* **Why?** A model trained solely on undersampled data produces uncalibrated probabilities (it overestimates the minority class probability in the wild). Phase 2 recalibrates the predictions to the true base rate and learns finer boundaries on the full data.



---

#### Returning the Final Predictor

```python
return lambda X: b2.predict(X, num_iteration=b2.best_iteration)

```

* Returns a lambda that accepts new feature inputs `X`.
* **`num_iteration=b2.best_iteration`**: Restricts the prediction to the optimal iteration determined by early stopping on the validation set during Phase 2.

---

### Summary of Benefits

| Problem with Naive Undersampling | How this 2-Phase Approach Fixes It |
| --- | --- |
| Throws away valuable majority-class data. | Recovers and uses all training data during Phase 2. |
| Output probabilities are shifted/uncalibrated. | Phase 2 re-aligns probabilities with the real-world base rate. |
| Models on raw imbalanced data ignore the rare class. | Phase 1 establishes strong feature splits for the rare class first. |