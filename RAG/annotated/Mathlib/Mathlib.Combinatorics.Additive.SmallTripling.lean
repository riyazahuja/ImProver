@[to_additive]
private lemma inductive_claim_mul (hm : 3 ≤ m)
    (h : ∀ ε : Fin 3 → ℤ, (∀ i, |ε i| = 1) → #((finRange 3).map fun i ↦ A ^ ε i).prod ≤ k * #A)
    (ε : Fin m → ℤ) (hε : ∀ i, |ε i| = 1) :
    #((finRange m).map fun i ↦ A ^ ε i).prod ≤ k ^ (m - 2) * #A := by
  /-
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : Group G
    A : Finset G
    k : Real
    m : Nat
    hm : LE.le 3 m
    h : ∀ (ε : Fin 3 → Int), (∀ (i : Fin 3), Eq (abs (ε i)) 1) → LE.le (↑(List.map …
    ε : Fin m → Int
    hε : ∀ (i : Fin m), Eq (abs (ε i)) 1
    ⊢ LE.le (↑(List.map (fun i => HPow.hPow A (ε i)) (List.finRange m)).prod.card) …
  -/
  induction' m, hm using Nat.le_induction with m hm ih
    /-
      case base
      G : Type u_1
      inst✝¹ : DecidableEq G
      inst✝ : Group G
      A : Finset G
      k : Real
      m : Nat
      h : ∀ (ε : Fin 3 → Int), (∀ (i : Fin 3), Eq (abs (ε i)) 1) → LE.le (↑(List.map …
      ε : Fin 3 → Int
      hε : ∀ (i : Fin 3), Eq (abs (ε i)) 1
      ⊢ LE.le (↑(List.map (fun i => HPow.hPow A (ε i)) (List.finRange 3)).prod.card) …
    -/
  · simpa using h ε hε
    /-
      🎉 no goals
    -/
  /-
    case succ
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : Group G
    A : Finset G
    k : Real
    m✝ : Nat
    h : ∀ (ε : Fin 3 → Int), (∀ (i : Fin 3), Eq (abs (ε i)) 1) → LE.le (↑(List.map …
    m : Nat
    hm : LE.le 3 m
    ih : ∀ (ε : Fin m → Int), (∀ (i : Fin m), Eq (abs (ε i)) 1) → LE.le (↑(List.ma …
    ε : Fin (HAdd.hAdd m 1) → Int
    hε : ∀ (i : Fin (HAdd.hAdd m 1)), Eq (abs (ε i)) 1
    ⊢ LE.le (↑(List.map (fun i => HPow.hPow A (ε i)) (List.finRange (HAdd.hAdd m 1 …
  -/
  obtain _ | m := m
    /-
      case succ.zero
      G : Type u_1
      inst✝¹ : DecidableEq G
      inst✝ : Group G
      A : Finset G
      k : Real
      m : Nat
      h : ∀ (ε : Fin 3 → Int), (∀ (i : Fin 3), Eq (abs (ε i)) 1) → LE.le (↑(List.map …
      hm : LE.le 3 0
      ih : ∀ (ε : Fin 0 → Int), (∀ (i : Fin 0), Eq (abs (ε i)) 1) → LE.le (↑(List.ma …
      ε : Fin (HAdd.hAdd 0 1) → Int
      hε : ∀ (i : Fin (HAdd.hAdd 0 1)), Eq (abs (ε i)) 1
      ⊢ LE.le (↑(List.map (fun i => HPow.hPow A (ε i)) (List.finRange (HAdd.hAdd 0 1 …
    -/
  · simp at hm
    /-
      🎉 no goals
    -/
  /-
    case succ.succ
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : Group G
    A : Finset G
    k : Real
    m✝ : Nat
    h : ∀ (ε : Fin 3 → Int), (∀ (i : Fin 3), Eq (abs (ε i)) 1) → LE.le (↑(List.map …
    m : Nat
    hm : LE.le 3 (HAdd.hAdd m 1)
    ih : ∀ (ε : Fin (HAdd.hAdd m 1) → Int), (∀ (i : Fin (HAdd.hAdd m 1)), Eq (abs  …
    ε : Fin (HAdd.hAdd (HAdd.hAdd m 1) 1) → Int
    hε : ∀ (i : Fin (HAdd.hAdd (HAdd.hAdd m 1) 1)), Eq (abs (ε i)) 1
    ⊢ LE.le (↑(List.map (fun i => HPow.hPow A (ε i)) (List.finRange (HAdd.hAdd (HA …
  -/
  have hm₀ : m ≠ 0 := by simp at hm; positivity
  /-
    case succ.succ
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : Group G
    A : Finset G
    k : Real
    m✝ : Nat
    h : ∀ (ε : Fin 3 → Int), (∀ (i : Fin 3), Eq (abs (ε i)) 1) → LE.le (↑(List.map …
    m : Nat
    hm : LE.le 3 (HAdd.hAdd m 1)
    ih : ∀ (ε : Fin (HAdd.hAdd m 1) → Int), (∀ (i : Fin (HAdd.hAdd m 1)), Eq (abs  …
    ε : Fin (HAdd.hAdd (HAdd.hAdd m 1) 1) → Int
    hε : ∀ (i : Fin (HAdd.hAdd (HAdd.hAdd m 1) 1)), Eq (abs (ε i)) 1
    hm₀ : Ne m 0
    ⊢ LE.le (↑(List.map (fun i => HPow.hPow A (ε i)) (List.finRange (HAdd.hAdd (HA …
  -/
  have hε₀ i : ε i ≠ 0 := fun h ↦ by simpa [h] using hε i
  /-
    case succ.succ
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : Group G
    A : Finset G
    k : Real
    m✝ : Nat
    h : ∀ (ε : Fin 3 → Int), (∀ (i : Fin 3), Eq (abs (ε i)) 1) → LE.le (↑(List.map …
    m : Nat
    hm : LE.le 3 (HAdd.hAdd m 1)
    ih : ∀ (ε : Fin (HAdd.hAdd m 1) → Int), (∀ (i : Fin (HAdd.hAdd m 1)), Eq (abs  …
    ε : Fin (HAdd.hAdd (HAdd.hAdd m 1) 1) → Int
    hε : ∀ (i : Fin (HAdd.hAdd (HAdd.hAdd m 1) 1)), Eq (abs (ε i)) 1
    hm₀ : Ne m 0
    hε₀ : ∀ (i : Fin (HAdd.hAdd (HAdd.hAdd m 1) 1)), Ne (ε i) 0
    ⊢ LE.le (↑(List.map (fun i => HPow.hPow A (ε i)) (List.finRange (HAdd.hAdd (HA …
  -/
  obtain rfl | hA := A.eq_empty_or_nonempty
    /-
      case succ.succ.inl
      G : Type u_1
      inst✝¹ : DecidableEq G
      inst✝ : Group G
      k : Real
      m✝ m : Nat
      hm : LE.le 3 (HAdd.hAdd m 1)
      ε : Fin (HAdd.hAdd (HAdd.hAdd m 1) 1) → Int
      hε : ∀ (i : Fin (HAdd.hAdd (HAdd.hAdd m 1) 1)), Eq (abs (ε i)) 1
      hm₀ : Ne m 0
      hε₀ : ∀ (i : Fin (HAdd.hAdd (HAdd.hAdd m 1) 1)), Ne (ε i) 0
      h : ∀ (ε : Fin 3 → Int), (∀ (i : Fin 3), Eq (abs (ε i)) 1) → LE.le (↑(List.map …
      ih : ∀ (ε : Fin (HAdd.hAdd m 1) → Int), (∀ (i : Fin (HAdd.hAdd m 1)), Eq (abs  …
      ⊢ LE.le (↑(List.map (fun i => HPow.hPow EmptyCollection.emptyCollection (ε i)) …
    -/
  · simp [hε₀]
    /-
      🎉 no goals
    -/
  have hk : 0 ≤ k :=
    nonneg_of_mul_nonneg_left ((h 1 (by simp)).trans' (by positivity)) (by positivity)
  /-
    case succ.succ.inr
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : Group G
    A : Finset G
    k : Real
    m✝ : Nat
    h : ∀ (ε : Fin 3 → Int), (∀ (i : Fin 3), Eq (abs (ε i)) 1) → LE.le (↑(List.map …
    m : Nat
    hm : LE.le 3 (HAdd.hAdd m 1)
    ih : ∀ (ε : Fin (HAdd.hAdd m 1) → Int), (∀ (i : Fin (HAdd.hAdd m 1)), Eq (abs  …
    ε : Fin (HAdd.hAdd (HAdd.hAdd m 1) 1) → Int
    hε : ∀ (i : Fin (HAdd.hAdd (HAdd.hAdd m 1) 1)), Eq (abs (ε i)) 1
    hm₀ : Ne m 0
    hε₀ : ∀ (i : Fin (HAdd.hAdd (HAdd.hAdd m 1) 1)), Ne (ε i) 0
    hA : A.Nonempty
    hk : LE.le 0 k
    ⊢ LE.le (↑(List.map (fun i => HPow.hPow A (ε i)) (List.finRange (HAdd.hAdd (HA …
  -/
  let π {n} (δ : Fin n → ℤ) : Finset G := ((finRange _).map fun i ↦ A ^ δ i).prod
  /-
    case succ.succ.inr
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : Group G
    A : Finset G
    k : Real
    m✝ : Nat
    h : ∀ (ε : Fin 3 → Int), (∀ (i : Fin 3), Eq (abs (ε i)) 1) → LE.le (↑(List.map …
    m : Nat
    hm : LE.le 3 (HAdd.hAdd m 1)
    ih : ∀ (ε : Fin (HAdd.hAdd m 1) → Int), (∀ (i : Fin (HAdd.hAdd m 1)), Eq (abs  …
    ε : Fin (HAdd.hAdd (HAdd.hAdd m 1) 1) → Int
    hε : ∀ (i : Fin (HAdd.hAdd (HAdd.hAdd m 1) 1)), Eq (abs (ε i)) 1
    hm₀ : Ne m 0
    hε₀ : ∀ (i : Fin (HAdd.hAdd (HAdd.hAdd m 1) 1)), Ne (ε i) 0
    hA : A.Nonempty
    hk : LE.le 0 k
    π : {n : Nat} → (Fin n → Int) → Finset G := fun {n} δ => (List.map (fun i => H …
    ⊢ LE.le (↑(List.map (fun i => HPow.hPow A (ε i)) (List.finRange (HAdd.hAdd (HA …
  -/
  let V : Finset G := π ![-ε 1, -ε 0]
  /-
    case succ.succ.inr
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : Group G
    A : Finset G
    k : Real
    m✝ : Nat
    h : ∀ (ε : Fin 3 → Int), (∀ (i : Fin 3), Eq (abs (ε i)) 1) → LE.le (↑(List.map …
    m : Nat
    hm : LE.le 3 (HAdd.hAdd m 1)
    ih : ∀ (ε : Fin (HAdd.hAdd m 1) → Int), (∀ (i : Fin (HAdd.hAdd m 1)), Eq (abs  …
    ε : Fin (HAdd.hAdd (HAdd.hAdd m 1) 1) → Int
    hε : ∀ (i : Fin (HAdd.hAdd (HAdd.hAdd m 1) 1)), Eq (abs (ε i)) 1
    hm₀ : Ne m 0
    hε₀ : ∀ (i : Fin (HAdd.hAdd (HAdd.hAdd m 1) 1)), Ne (ε i) 0
    hA : A.Nonempty
    hk : LE.le 0 k
    π : {n : Nat} → (Fin n → Int) → Finset G := fun {n} δ => (List.map (fun i => H …
    V : Finset G := π (Matrix.vecCons (Neg.neg (ε 1)) (Matrix.vecCons (Neg.neg (ε  …
    ⊢ LE.le (↑(List.map (fun i => HPow.hPow A (ε i)) (List.finRange (HAdd.hAdd (HA …
  -/
  let W : Finset G := π <| tail <| tail ε
  /-
    case succ.succ.inr
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : Group G
    A : Finset G
    k : Real
    m✝ : Nat
    h : ∀ (ε : Fin 3 → Int), (∀ (i : Fin 3), Eq (abs (ε i)) 1) → LE.le (↑(List.map …
    m : Nat
    hm : LE.le 3 (HAdd.hAdd m 1)
    ih : ∀ (ε : Fin (HAdd.hAdd m 1) → Int), (∀ (i : Fin (HAdd.hAdd m 1)), Eq (abs  …
    ε : Fin (HAdd.hAdd (HAdd.hAdd m 1) 1) → Int
    hε : ∀ (i : Fin (HAdd.hAdd (HAdd.hAdd m 1) 1)), Eq (abs (ε i)) 1
    hm₀ : Ne m 0
    hε₀ : ∀ (i : Fin (HAdd.hAdd (HAdd.hAdd m 1) 1)), Ne (ε i) 0
    hA : A.Nonempty
    hk : LE.le 0 k
    π : {n : Nat} → (Fin n → Int) → Finset G := fun {n} δ => (List.map (fun i => H …
    V : Finset G := π (Matrix.vecCons (Neg.neg (ε 1)) (Matrix.vecCons (Neg.neg (ε  …
    W : Finset G := π (Fin.tail (Fin.tail ε))
    ⊢ LE.le (↑(List.map (fun i => HPow.hPow A (ε i)) (List.finRange (HAdd.hAdd (HA …
  -/
  refine le_of_mul_le_mul_left ?_ (by positivity : (0 : ℝ) < #A)
  calc
    (#A * #(π ε) : ℝ)
      = #A * #(V⁻¹ * W) := by
      simp [π, V, W, List.finRange_succ_eq_map, Fin.tail, Function.comp_def, mul_assoc]
    _ ≤ #(A * V) * #(A * W) := by norm_cast; exact ruzsa_triangle_inequality_invMul_mul_mul ..
    _ = #(π ![1, -ε 1, -ε 0]) * #(π <| Fin.cons 1 <| tail <| tail ε) := by
      simp [π, V, W, List.finRange_succ_eq_map, Fin.tail, Function.comp_def]
    _ ≤ (k * #A) * (k ^ (m - 1) * #A) := by
      gcongr
      · exact h ![1, -ε 1, -ε 0] fun i ↦ by fin_cases i <;> simp [hε]
      · exact ih (Fin.cons 1 <| tail <| tail ε) <| Fin.cons (by simp) (by simp [hε, Fin.tail])
    _ = #A * (k ^ m * #A) := by rw [← pow_sub_one_mul hm₀]; ring


@[to_additive]
private lemma small_neg_pos_pos_mul (hA : #(A ^ 3) ≤ K * #A) : #(A⁻¹ * A * A) ≤ K ^ 2 * #A := by
  /-
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : Group G
    A : Finset G
    K : Real
    hA : LE.le (↑(HPow.hPow A 3).card) (HMul.hMul K ↑A.card)
    ⊢ LE.le (↑(HMul.hMul (HMul.hMul (Inv.inv A) A) A).card) (HMul.hMul (HPow.hPow  …
  -/
  obtain rfl | hA₀ := A.eq_empty_or_nonempty
    /-
      case inl
      G : Type u_1
      inst✝¹ : DecidableEq G
      inst✝ : Group G
      K : Real
      hA : LE.le (↑(HPow.hPow EmptyCollection.emptyCollection 3).card) (HMul.hMul K  …
      ⊢ LE.le (↑(HMul.hMul (HMul.hMul (Inv.inv EmptyCollection.emptyCollection) Empt …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : Group G
    A : Finset G
    K : Real
    hA : LE.le (↑(HPow.hPow A 3).card) (HMul.hMul K ↑A.card)
    hA₀ : A.Nonempty
    ⊢ LE.le (↑(HMul.hMul (HMul.hMul (Inv.inv A) A) A).card) (HMul.hMul (HPow.hPow  …
  -/
  have : 0 ≤ K := nonneg_of_mul_nonneg_left (hA.trans' <| by positivity) (by positivity)
  /-
    case inr
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : Group G
    A : Finset G
    K : Real
    hA : LE.le (↑(HPow.hPow A 3).card) (HMul.hMul K ↑A.card)
    hA₀ : A.Nonempty
    this : LE.le 0 K
    ⊢ LE.le (↑(HMul.hMul (HMul.hMul (Inv.inv A) A) A).card) (HMul.hMul (HPow.hPow  …
  -/
  refine le_of_mul_le_mul_left ?_ (by positivity : (0 : ℝ) < #A)
  calc
    (#A * #(A⁻¹ * A * A) : ℝ) = #A * #(A⁻¹ * (A * A)) := by rw [mul_assoc]
    _ ≤ #(A * A) * #(A * (A * A)) := by
      norm_cast; exact ruzsa_triangle_inequality_invMul_mul_mul A A (A * A)
    _ = #(A ^ 2) * #(A ^ 3) := by simp [pow_succ']
    _ ≤ (K * #A) * (K * #A) := by
      gcongr
      calc
        (#(A ^ 2) : ℝ) ≤ #(A ^ 3) := mod_cast hA₀.card_pow_mono (by norm_num)
        _ ≤ K * #A := hA
    _ = #A * (K ^ 2 * #A) := by ring


@[to_additive]
private lemma small_neg_neg_pos_mul (hA : #(A ^ 3) ≤ K * #A) : #(A⁻¹ * A⁻¹ * A) ≤ K ^ 2 * #A := by
  /-
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : Group G
    A : Finset G
    K : Real
    hA : LE.le (↑(HPow.hPow A 3).card) (HMul.hMul K ↑A.card)
    ⊢ LE.le (↑(HMul.hMul (HMul.hMul (Inv.inv A) (Inv.inv A)) A).card) (HMul.hMul ( …
  -/
  rw [← card_inv]
  /-
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : Group G
    A : Finset G
    K : Real
    hA : LE.le (↑(HPow.hPow A 3).card) (HMul.hMul K ↑A.card)
    ⊢ LE.le (↑(Inv.inv (HMul.hMul (HMul.hMul (Inv.inv A) (Inv.inv A)) A)).card) (H …
  -/
  simpa [mul_assoc] using small_neg_pos_pos_mul (A := A) (K := K) (by simpa)
  /-
    🎉 no goals
  -/


@[to_additive]
private lemma small_pos_neg_neg_mul (hA : #(A ^ 3) ≤ K * #A) : #(A * A⁻¹ * A⁻¹) ≤ K ^ 2 * #A := by
  /-
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : Group G
    A : Finset G
    K : Real
    hA : LE.le (↑(HPow.hPow A 3).card) (HMul.hMul K ↑A.card)
    ⊢ LE.le (↑(HMul.hMul (HMul.hMul A (Inv.inv A)) (Inv.inv A)).card) (HMul.hMul ( …
  -/
  simpa using small_neg_pos_pos_mul (A := A⁻¹) (by simpa)
  /-
    🎉 no goals
  -/


@[to_additive]
private lemma small_pos_pos_neg_mul (hA : #(A ^ 3) ≤ K * #A) : #(A * A * A⁻¹) ≤ K ^ 2 * #A := by
  /-
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : Group G
    A : Finset G
    K : Real
    hA : LE.le (↑(HPow.hPow A 3).card) (HMul.hMul K ↑A.card)
    ⊢ LE.le (↑(HMul.hMul (HMul.hMul A A) (Inv.inv A)).card) (HMul.hMul (HPow.hPow  …
  -/
  rw [← card_inv]
  /-
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : Group G
    A : Finset G
    K : Real
    hA : LE.le (↑(HPow.hPow A 3).card) (HMul.hMul K ↑A.card)
    ⊢ LE.le (↑(Inv.inv (HMul.hMul (HMul.hMul A A) (Inv.inv A))).card) (HMul.hMul ( …
  -/
  simpa [mul_assoc] using small_pos_neg_neg_mul (A := A) (K := K) (by simpa)
  /-
    🎉 no goals
  -/


@[to_additive]
private lemma small_pos_neg_pos_mul (hA : #(A ^ 3) ≤ K * #A) : #(A * A⁻¹ * A) ≤ K ^ 3 * #A := by
  /-
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : Group G
    A : Finset G
    K : Real
    hA : LE.le (↑(HPow.hPow A 3).card) (HMul.hMul K ↑A.card)
    ⊢ LE.le (↑(HMul.hMul (HMul.hMul A (Inv.inv A)) A).card) (HMul.hMul (HPow.hPow  …
  -/
  obtain rfl | hA₀ := A.eq_empty_or_nonempty
    /-
      case inl
      G : Type u_1
      inst✝¹ : DecidableEq G
      inst✝ : Group G
      K : Real
      hA : LE.le (↑(HPow.hPow EmptyCollection.emptyCollection 3).card) (HMul.hMul K  …
      ⊢ LE.le (↑(HMul.hMul (HMul.hMul EmptyCollection.emptyCollection (Inv.inv Empty …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : Group G
    A : Finset G
    K : Real
    hA : LE.le (↑(HPow.hPow A 3).card) (HMul.hMul K ↑A.card)
    hA₀ : A.Nonempty
    ⊢ LE.le (↑(HMul.hMul (HMul.hMul A (Inv.inv A)) A).card) (HMul.hMul (HPow.hPow  …
  -/
  have : 0 ≤ K := nonneg_of_mul_nonneg_left (hA.trans' <| by positivity) (by positivity)
  /-
    case inr
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : Group G
    A : Finset G
    K : Real
    hA : LE.le (↑(HPow.hPow A 3).card) (HMul.hMul K ↑A.card)
    hA₀ : A.Nonempty
    this : LE.le 0 K
    ⊢ LE.le (↑(HMul.hMul (HMul.hMul A (Inv.inv A)) A).card) (HMul.hMul (HPow.hPow  …
  -/
  refine le_of_mul_le_mul_left ?_ (by positivity : (0 : ℝ) < #A)
  calc
    (#A * #(A * A⁻¹ * A) : ℝ) ≤ #(A * (A * A⁻¹)) * #(A * A) := by
      norm_cast; simpa using ruzsa_triangle_inequality_invMul_mul_mul (A * A⁻¹) A A
    _ = #(A  * A * A⁻¹) * #(A ^ 2) := by simp [pow_succ, mul_assoc]
    _ ≤ (K ^ 2 * #A) * (K * #A) := by
      gcongr
      · exact small_pos_pos_neg_mul hA
      calc
        (#(A ^ 2) : ℝ) ≤ #(A ^ 3) := mod_cast hA₀.card_pow_mono (by norm_num)
        _ ≤ K * #A := hA
    _ = #A * (K ^ 3 * #A) := by ring


@[to_additive]
private lemma small_neg_pos_neg_mul (hA : #(A ^ 3) ≤ K * #A) : #(A⁻¹ * A * A⁻¹) ≤ K ^ 3 * #A := by
  /-
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : Group G
    A : Finset G
    K : Real
    hA : LE.le (↑(HPow.hPow A 3).card) (HMul.hMul K ↑A.card)
    ⊢ LE.le (↑(HMul.hMul (HMul.hMul (Inv.inv A) A) (Inv.inv A)).card) (HMul.hMul ( …
  -/
  rw [← card_inv]
  /-
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : Group G
    A : Finset G
    K : Real
    hA : LE.le (↑(HPow.hPow A 3).card) (HMul.hMul K ↑A.card)
    ⊢ LE.le (↑(Inv.inv (HMul.hMul (HMul.hMul (Inv.inv A) A) (Inv.inv A))).card) (H …
  -/
  simpa [mul_assoc] using small_pos_neg_pos_mul (A := A) (K := K) (by simpa)
  /-
    🎉 no goals
  -/


/-- If `A` has small tripling, say with constant `K`, then `A` has small alternating powers, in the
sense that `|A^±1 * ... * A^±1|` is at most `|A|` times a constant exponential in the number of
terms in the product.

When `A` is symmetric (`A⁻¹ = A`), the base of the exponential can be lowered from `K ^ 3` to `K`,
where `K` is the tripling constant. See `Finset.small_pow_of_small_tripling`. -/
@[to_additive
"If `A` has small tripling, say with constant `K`, then `A` has small alternating powers, in the
sense that `|±A ± ... ± A|` is at most `|A|` times a constant exponential in the number of
terms in the product.

When `A` is symmetric (`-A = A`), the base of the exponential can be lowered from `K ^ 3` to `K`,
where `K` is the tripling constant. See `Finset.small_nsmul_of_small_tripling`."]
lemma small_alternating_pow_of_small_tripling' (hm : 3 ≤ m) (hA : #(A ^ 3) ≤ K * #A) (ε : Fin m → ℤ)
    (hε : ∀ i, |ε i| = 1) :
    #((finRange m).map fun i ↦ A ^ ε i).prod ≤ K ^ (3 * (m - 2)) * #A := by
  /-
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : Group G
    A : Finset G
    K : Real
    m : Nat
    hm : LE.le 3 m
    hA : LE.le (↑(HPow.hPow A 3).card) (HMul.hMul K ↑A.card)
    ε : Fin m → Int
    hε : ∀ (i : Fin m), Eq (abs (ε i)) 1
    ⊢ LE.le (↑(List.map (fun i => HPow.hPow A (ε i)) (List.finRange m)).prod.card) …
  -/
  have hm₀ : m ≠ 0 := by positivity
  /-
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : Group G
    A : Finset G
    K : Real
    m : Nat
    hm : LE.le 3 m
    hA : LE.le (↑(HPow.hPow A 3).card) (HMul.hMul K ↑A.card)
    ε : Fin m → Int
    hε : ∀ (i : Fin m), Eq (abs (ε i)) 1
    hm₀ : Ne m 0
    ⊢ LE.le (↑(List.map (fun i => HPow.hPow A (ε i)) (List.finRange m)).prod.card) …
  -/
  have hε₀ i : ε i ≠ 0 := fun h ↦ by simpa [h] using hε i
  /-
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : Group G
    A : Finset G
    K : Real
    m : Nat
    hm : LE.le 3 m
    hA : LE.le (↑(HPow.hPow A 3).card) (HMul.hMul K ↑A.card)
    ε : Fin m → Int
    hε : ∀ (i : Fin m), Eq (abs (ε i)) 1
    hm₀ : Ne m 0
    hε₀ : ∀ (i : Fin m), Ne (ε i) 0
    ⊢ LE.le (↑(List.map (fun i => HPow.hPow A (ε i)) (List.finRange m)).prod.card) …
  -/
  obtain rfl | hA₀ := A.eq_empty_or_nonempty
    /-
      case inl
      G : Type u_1
      inst✝¹ : DecidableEq G
      inst✝ : Group G
      K : Real
      m : Nat
      hm : LE.le 3 m
      ε : Fin m → Int
      hε : ∀ (i : Fin m), Eq (abs (ε i)) 1
      hm₀ : Ne m 0
      hε₀ : ∀ (i : Fin m), Ne (ε i) 0
      hA : LE.le (↑(HPow.hPow EmptyCollection.emptyCollection 3).card) (HMul.hMul K  …
      ⊢ LE.le (↑(List.map (fun i => HPow.hPow EmptyCollection.emptyCollection (ε i)) …
    -/
  · simp [hm₀, hε₀]
    /-
      🎉 no goals
    -/
  have hK₁ : 1 ≤ K :=
    one_le_of_le_mul_right₀ (by positivity)
      (hA.trans' <| by norm_cast; exact card_le_card_pow (by norm_num))
  /-
    case inr
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : Group G
    A : Finset G
    K : Real
    m : Nat
    hm : LE.le 3 m
    hA : LE.le (↑(HPow.hPow A 3).card) (HMul.hMul K ↑A.card)
    ε : Fin m → Int
    hε : ∀ (i : Fin m), Eq (abs (ε i)) 1
    hm₀ : Ne m 0
    hε₀ : ∀ (i : Fin m), Ne (ε i) 0
    hA₀ : A.Nonempty
    hK₁ : LE.le 1 K
    ⊢ LE.le (↑(List.map (fun i => HPow.hPow A (ε i)) (List.finRange m)).prod.card) …
  -/
  rw [pow_mul]
  /-
    case inr
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : Group G
    A : Finset G
    K : Real
    m : Nat
    hm : LE.le 3 m
    hA : LE.le (↑(HPow.hPow A 3).card) (HMul.hMul K ↑A.card)
    ε : Fin m → Int
    hε : ∀ (i : Fin m), Eq (abs (ε i)) 1
    hm₀ : Ne m 0
    hε₀ : ∀ (i : Fin m), Ne (ε i) 0
    hA₀ : A.Nonempty
    hK₁ : LE.le 1 K
    ⊢ LE.le (↑(List.map (fun i => HPow.hPow A (ε i)) (List.finRange m)).prod.card) …
  -/
  refine inductive_claim_mul hm (fun δ hδ ↦ ?_) ε hε
  simp only [finRange_succ_eq_map, Nat.reduceAdd, isValue, finRange_zero, map_nil, List.map_cons,
    succ_zero_eq_one, succ_one_eq_two, List.prod_cons, prod_nil, mul_one, ← mul_assoc]
  simp only [zero_le_one, abs_eq, Int.reduceNeg, forall_iff_succ, isValue, succ_zero_eq_one,
    succ_one_eq_two, IsEmpty.forall_iff, and_true] at hδ
  /-
    case inr
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : Group G
    A : Finset G
    K : Real
    m : Nat
    hm : LE.le 3 m
    hA : LE.le (↑(HPow.hPow A 3).card) (HMul.hMul K ↑A.card)
    ε : Fin m → Int
    hε : ∀ (i : Fin m), Eq (abs (ε i)) 1
    hm₀ : Ne m 0
    hε₀ : ∀ (i : Fin m), Ne (ε i) 0
    hA₀ : A.Nonempty
    hK₁ : LE.le 1 K
    δ : Fin 3 → Int
    hδ : And (Or (Eq (δ 0) 1) (Eq (δ 0) (-1))) (And (Or (Eq (δ 1) 1) (Eq (δ 1) (-1 …
    ⊢ LE.le (↑(HMul.hMul (HMul.hMul (HPow.hPow A (δ 0)) (HPow.hPow A (δ 1))) (HPow …
  -/
  have : K ≤ K ^ 3 := le_self_pow₀ hK₁ (by omega)
  have : K ^ 2 ≤ K ^ 3 := by
    gcongr
    · exact hK₁
    · norm_num
  /-
    case inr
    G : Type u_1
    inst✝¹ : DecidableEq G
    inst✝ : Group G
    A : Finset G
    K : Real
    m : Nat
    hm : LE.le 3 m
    hA : LE.le (↑(HPow.hPow A 3).card) (HMul.hMul K ↑A.card)
    ε : Fin m → Int
    hε : ∀ (i : Fin m), Eq (abs (ε i)) 1
    hm₀ : Ne m 0
    hε₀ : ∀ (i : Fin m), Ne (ε i) 0
    hA₀ : A.Nonempty
    hK₁ : LE.le 1 K
    δ : Fin 3 → Int
    hδ : And (Or (Eq (δ 0) 1) (Eq (δ 0) (-1))) (And (Or (Eq (δ 1) 1) (Eq (δ 1) (-1 …
    this✝ : LE.le K (HPow.hPow K 3)
    this : LE.le (HPow.hPow K 2) (HPow.hPow K 3)
    ⊢ LE.le (↑(HMul.hMul (HMul.hMul (HPow.hPow A (δ 0)) (HPow.hPow A (δ 1))) (HPow …
  -/
  obtain ⟨hδ₀ | hδ₀, hδ₁ | hδ₁, hδ₂ | hδ₂⟩ := hδ <;> simp [hδ₀, hδ₁, hδ₂]
    /-
      case inr.intro.inl.intro.inl.inl
      G : Type u_1
      inst✝¹ : DecidableEq G
      inst✝ : Group G
      A : Finset G
      K : Real
      m : Nat
      hm : LE.le 3 m
      hA : LE.le (↑(HPow.hPow A 3).card) (HMul.hMul K ↑A.card)
      ε : Fin m → Int
      hε : ∀ (i : Fin m), Eq (abs (ε i)) 1
      hm₀ : Ne m 0
      hε₀ : ∀ (i : Fin m), Ne (ε i) 0
      hA₀ : A.Nonempty
      hK₁ : LE.le 1 K
      δ : Fin 3 → Int
      this✝ : LE.le K (HPow.hPow K 3)
      this : LE.le (HPow.hPow K 2) (HPow.hPow K 3)
      hδ₀ : Eq (δ 0) 1
      hδ₁ : Eq (δ 1) 1
      hδ₂ : Eq (δ 2) 1
      ⊢ LE.le (↑(HMul.hMul (HMul.hMul A A) A).card) (HMul.hMul (HPow.hPow K 3) ↑A.ca …
    -/
  · simp [pow_succ] at hA
    /-
      case inr.intro.inl.intro.inl.inl
      G : Type u_1
      inst✝¹ : DecidableEq G
      inst✝ : Group G
      A : Finset G
      K : Real
      m : Nat
      hm : LE.le 3 m
      ε : Fin m → Int
      hε : ∀ (i : Fin m), Eq (abs (ε i)) 1
      hm₀ : Ne m 0
      hε₀ : ∀ (i : Fin m), Ne (ε i) 0
      hA₀ : A.Nonempty
      hK₁ : LE.le 1 K
      δ : Fin 3 → Int
      this✝ : LE.le K (HPow.hPow K 3)
      this : LE.le (HPow.hPow K 2) (HPow.hPow K 3)
      hδ₀ : Eq (δ 0) 1
      hδ₁ : Eq (δ 1) 1
      hδ₂ : Eq (δ 2) 1
      hA : LE.le (↑(HMul.hMul (HMul.hMul A A) A).card) (HMul.hMul K ↑A.card)
      ⊢ LE.le (↑(HMul.hMul (HMul.hMul A A) A).card) (HMul.hMul (HPow.hPow K 3) ↑A.ca …
    -/
    nlinarith
    /-
      🎉 no goals
    -/
    /-
      case inr.intro.inl.intro.inl.inr
      G : Type u_1
      inst✝¹ : DecidableEq G
      inst✝ : Group G
      A : Finset G
      K : Real
      m : Nat
      hm : LE.le 3 m
      hA : LE.le (↑(HPow.hPow A 3).card) (HMul.hMul K ↑A.card)
      ε : Fin m → Int
      hε : ∀ (i : Fin m), Eq (abs (ε i)) 1
      hm₀ : Ne m 0
      hε₀ : ∀ (i : Fin m), Ne (ε i) 0
      hA₀ : A.Nonempty
      hK₁ : LE.le 1 K
      δ : Fin 3 → Int
      this✝ : LE.le K (HPow.hPow K 3)
      this : LE.le (HPow.hPow K 2) (HPow.hPow K 3)
      hδ₀ : Eq (δ 0) 1
      hδ₁ : Eq (δ 1) 1
      hδ₂ : Eq (δ 2) (-1)
      ⊢ LE.le (↑(HMul.hMul (HMul.hMul A A) (Inv.inv A)).card) (HMul.hMul (HPow.hPow  …
    -/
  · nlinarith [small_pos_pos_neg_mul hA]
    /-
      🎉 no goals
    -/
    /-
      case inr.intro.inl.intro.inr.inl
      G : Type u_1
      inst✝¹ : DecidableEq G
      inst✝ : Group G
      A : Finset G
      K : Real
      m : Nat
      hm : LE.le 3 m
      hA : LE.le (↑(HPow.hPow A 3).card) (HMul.hMul K ↑A.card)
      ε : Fin m → Int
      hε : ∀ (i : Fin m), Eq (abs (ε i)) 1
      hm₀ : Ne m 0
      hε₀ : ∀ (i : Fin m), Ne (ε i) 0
      hA₀ : A.Nonempty
      hK₁ : LE.le 1 K
      δ : Fin 3 → Int
      this✝ : LE.le K (HPow.hPow K 3)
      this : LE.le (HPow.hPow K 2) (HPow.hPow K 3)
      hδ₀ : Eq (δ 0) 1
      hδ₁ : Eq (δ 1) (-1)
      hδ₂ : Eq (δ 2) 1
      ⊢ LE.le (↑(HMul.hMul (HMul.hMul A (Inv.inv A)) A).card) (HMul.hMul (HPow.hPow  …
    -/
  · nlinarith [small_pos_neg_pos_mul hA]
    /-
      🎉 no goals
    -/
    /-
      case inr.intro.inl.intro.inr.inr
      G : Type u_1
      inst✝¹ : DecidableEq G
      inst✝ : Group G
      A : Finset G
      K : Real
      m : Nat
      hm : LE.le 3 m
      hA : LE.le (↑(HPow.hPow A 3).card) (HMul.hMul K ↑A.card)
      ε : Fin m → Int
      hε : ∀ (i : Fin m), Eq (abs (ε i)) 1
      hm₀ : Ne m 0
      hε₀ : ∀ (i : Fin m), Ne (ε i) 0
      hA₀ : A.Nonempty
      hK₁ : LE.le 1 K
      δ : Fin 3 → Int
      this✝ : LE.le K (HPow.hPow K 3)
      this : LE.le (HPow.hPow K 2) (HPow.hPow K 3)
      hδ₀ : Eq (δ 0) 1
      hδ₁ : Eq (δ 1) (-1)
      hδ₂ : Eq (δ 2) (-1)
      ⊢ LE.le (↑(HMul.hMul (HMul.hMul A (Inv.inv A)) (Inv.inv A)).card) (HMul.hMul ( …
    -/
  · nlinarith [small_pos_neg_neg_mul hA]
    /-
      🎉 no goals
    -/
    /-
      case inr.intro.inr.intro.inl.inl
      G : Type u_1
      inst✝¹ : DecidableEq G
      inst✝ : Group G
      A : Finset G
      K : Real
      m : Nat
      hm : LE.le 3 m
      hA : LE.le (↑(HPow.hPow A 3).card) (HMul.hMul K ↑A.card)
      ε : Fin m → Int
      hε : ∀ (i : Fin m), Eq (abs (ε i)) 1
      hm₀ : Ne m 0
      hε₀ : ∀ (i : Fin m), Ne (ε i) 0
      hA₀ : A.Nonempty
      hK₁ : LE.le 1 K
      δ : Fin 3 → Int
      this✝ : LE.le K (HPow.hPow K 3)
      this : LE.le (HPow.hPow K 2) (HPow.hPow K 3)
      hδ₀ : Eq (δ 0) (-1)
      hδ₁ : Eq (δ 1) 1
      hδ₂ : Eq (δ 2) 1
      ⊢ LE.le (↑(HMul.hMul (HMul.hMul (Inv.inv A) A) A).card) (HMul.hMul (HPow.hPow  …
    -/
  · nlinarith [small_neg_pos_pos_mul hA]
    /-
      🎉 no goals
    -/
    /-
      case inr.intro.inr.intro.inl.inr
      G : Type u_1
      inst✝¹ : DecidableEq G
      inst✝ : Group G
      A : Finset G
      K : Real
      m : Nat
      hm : LE.le 3 m
      hA : LE.le (↑(HPow.hPow A 3).card) (HMul.hMul K ↑A.card)
      ε : Fin m → Int
      hε : ∀ (i : Fin m), Eq (abs (ε i)) 1
      hm₀ : Ne m 0
      hε₀ : ∀ (i : Fin m), Ne (ε i) 0
      hA₀ : A.Nonempty
      hK₁ : LE.le 1 K
      δ : Fin 3 → Int
      this✝ : LE.le K (HPow.hPow K 3)
      this : LE.le (HPow.hPow K 2) (HPow.hPow K 3)
      hδ₀ : Eq (δ 0) (-1)
      hδ₁ : Eq (δ 1) 1
      hδ₂ : Eq (δ 2) (-1)
      ⊢ LE.le (↑(HMul.hMul (HMul.hMul (Inv.inv A) A) (Inv.inv A)).card) (HMul.hMul ( …
    -/
  · nlinarith [small_neg_pos_neg_mul hA]
    /-
      🎉 no goals
    -/
    /-
      case inr.intro.inr.intro.inr.inl
      G : Type u_1
      inst✝¹ : DecidableEq G
      inst✝ : Group G
      A : Finset G
      K : Real
      m : Nat
      hm : LE.le 3 m
      hA : LE.le (↑(HPow.hPow A 3).card) (HMul.hMul K ↑A.card)
      ε : Fin m → Int
      hε : ∀ (i : Fin m), Eq (abs (ε i)) 1
      hm₀ : Ne m 0
      hε₀ : ∀ (i : Fin m), Ne (ε i) 0
      hA₀ : A.Nonempty
      hK₁ : LE.le 1 K
      δ : Fin 3 → Int
      this✝ : LE.le K (HPow.hPow K 3)
      this : LE.le (HPow.hPow K 2) (HPow.hPow K 3)
      hδ₀ : Eq (δ 0) (-1)
      hδ₁ : Eq (δ 1) (-1)
      hδ₂ : Eq (δ 2) 1
      ⊢ LE.le (↑(HMul.hMul (HMul.hMul (Inv.inv A) (Inv.inv A)) A).card) (HMul.hMul ( …
    -/
  · nlinarith [small_neg_neg_pos_mul hA]
    /-
      🎉 no goals
    -/
    /-
      case inr.intro.inr.intro.inr.inr
      G : Type u_1
      inst✝¹ : DecidableEq G
      inst✝ : Group G
      A : Finset G
      K : Real
      m : Nat
      hm : LE.le 3 m
      hA : LE.le (↑(HPow.hPow A 3).card) (HMul.hMul K ↑A.card)
      ε : Fin m → Int
      hε : ∀ (i : Fin m), Eq (abs (ε i)) 1
      hm₀ : Ne m 0
      hε₀ : ∀ (i : Fin m), Ne (ε i) 0
      hA₀ : A.Nonempty
      hK₁ : LE.le 1 K
      δ : Fin 3 → Int
      this✝ : LE.le K (HPow.hPow K 3)
      this : LE.le (HPow.hPow K 2) (HPow.hPow K 3)
      hδ₀ : Eq (δ 0) (-1)
      hδ₁ : Eq (δ 1) (-1)
      hδ₂ : Eq (δ 2) (-1)
      ⊢ LE.le (↑(HMul.hMul (HMul.hMul (Inv.inv A) (Inv.inv A)) (Inv.inv A)).card) (H …
    -/
  · simp [*, pow_succ', ← mul_inv_rev] at hA ⊢
    /-
      case inr.intro.inr.intro.inr.inr
      G : Type u_1
      inst✝¹ : DecidableEq G
      inst✝ : Group G
      A : Finset G
      K : Real
      m : Nat
      hm : LE.le 3 m
      ε : Fin m → Int
      hε : ∀ (i : Fin m), Eq (abs (ε i)) 1
      hm₀ : Ne m 0
      hε₀ : ∀ (i : Fin m), Ne (ε i) 0
      hA₀ : A.Nonempty
      hK₁ : LE.le 1 K
      δ : Fin 3 → Int
      this✝ : LE.le K (HPow.hPow K 3)
      this : LE.le (HPow.hPow K 2) (HPow.hPow K 3)
      hδ₀ : Eq (δ 0) (-1)
      hδ₁ : Eq (δ 1) (-1)
      hδ₂ : Eq (δ 2) (-1)
      hA : LE.le (↑(HMul.hMul A (HMul.hMul A A)).card) (HMul.hMul K ↑A.card)
      ⊢ LE.le (↑(HMul.hMul A (HMul.hMul A A)).card) (HMul.hMul (HMul.hMul K (HMul.hM …
    -/
    nlinarith
    /-
      🎉 no goals
    -/


/-- If `A` is symmetric (`A⁻¹ = A`) and has small tripling, then `A` has small powers,
in the sense that `|A ^ m|` is at most `|A|` times a constant exponential in `m`.

See also `Finset.small_alternating_pow_of_small_tripling` for a version with a weaker constant but
which encompasses non-symmetric sets. -/
@[to_additive
"If `A` is symmetric (`-A = A`) and has small tripling, then `A` has small powers,
in the sense that `|m • A|` is at most `|A|` times a constant exponential in `m`.

See also `Finset.small_alternating_nsmul_of_small_tripling` for a version with a weaker constant but
which encompasses non-symmetric sets."]
lemma small_pow_of_small_tripling' (hm : 3 ≤ m) (hA : #(A ^ 3) ≤ K * #A) (hAsymm : A⁻¹ = A) :
    #(A ^ m) ≤ K ^ (m - 2) * #A := by
  have (ε : ℤ) (hε : |ε| = 1) : A ^ ε = A := by
    obtain rfl | rfl := eq_or_eq_neg_of_abs_eq hε <;> simp [hAsymm]
  calc
    (#(A ^ m) : ℝ) = #((finRange m).map fun i ↦ A ^ 1).prod := by simp
    _ ≤ K ^ (m - 2) * #A :=
      inductive_claim_mul hm (fun δ hδ ↦ by simpa [this _ (hδ _), pow_succ'] using hA) _ (by simp)


