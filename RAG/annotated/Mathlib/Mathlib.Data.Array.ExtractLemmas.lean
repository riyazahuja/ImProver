@[simp]
theorem extract_eq_nil_of_start_eq_end {a : Array α} :
    a.extract i i = #[] := by
  /-
    α : Type u
    i : Nat
    a : Array α
    ⊢ Eq (a.extract i i) List.nil.toArray
  -/
  refine extract_empty_of_stop_le_start a ?h
  /-
    case h
    α : Type u
    i : Nat
    a : Array α
    ⊢ LE.le i i
  -/
  exact Nat.le_refl i
  /-
    🎉 no goals
  -/


theorem extract_append_left {a b : Array α} {i j : Nat} (h : j ≤ a.size) :
    (a ++ b).extract i j = a.extract i j := by
  /-
    α : Type u
    a b : Array α
    i j : Nat
    h : LE.le j a.size
    ⊢ Eq ((HAppend.hAppend a b).extract i j) (a.extract i j)
  -/
  apply ext
    /-
      case h₁
      α : Type u
      a b : Array α
      i j : Nat
      h : LE.le j a.size
      ⊢ Eq ((HAppend.hAppend a b).extract i j).size (a.extract i j).size
    -/
  · simp only [size_extract, size_append]
    /-
      case h₁
      α : Type u
      a b : Array α
      i j : Nat
      h : LE.le j a.size
      ⊢ Eq (HSub.hSub (Min.min j (HAdd.hAdd a.size b.size)) i) (HSub.hSub (Min.min j …
    -/
    omega
    /-
      🎉 no goals
    -/
    /-
      case h₂
      α : Type u
      a b : Array α
      i j : Nat
      h : LE.le j a.size
      ⊢ ∀ (i_1 : Nat) (hi₁ : LT.lt i_1 ((HAppend.hAppend a b).extract i j).size) (hi …
    -/
  · intro h1 h2 h3
    /-
      case h₂
      α : Type u
      a b : Array α
      i j : Nat
      h : LE.le j a.size
      h1 : Nat
      h2 : LT.lt h1 ((HAppend.hAppend a b).extract i j).size
      h3 : LT.lt h1 (a.extract i j).size
      ⊢ Eq (GetElem.getElem ((HAppend.hAppend a b).extract i j) h1 h2) (GetElem.getE …
    -/
    rw [getElem_extract, getElem_append_left, getElem_extract]
    /-
      🎉 no goals
    -/


theorem extract_append_right {a b : Array α} {i j : Nat} (h : a.size ≤ i) :
    (a ++ b).extract i j = b.extract (i - a.size) (j - a.size) := by
  /-
    α : Type u
    a b : Array α
    i j : Nat
    h : LE.le a.size i
    ⊢ Eq ((HAppend.hAppend a b).extract i j) (b.extract (HSub.hSub i a.size) (HSub …
  -/
  apply ext
    /-
      case h₁
      α : Type u
      a b : Array α
      i j : Nat
      h : LE.le a.size i
      ⊢ Eq ((HAppend.hAppend a b).extract i j).size (b.extract (HSub.hSub i a.size)  …
    -/
  · rw [size_extract, size_extract, size_append]
    /-
      case h₁
      α : Type u
      a b : Array α
      i j : Nat
      h : LE.le a.size i
      ⊢ Eq (HSub.hSub (Min.min j (HAdd.hAdd a.size b.size)) i) (HSub.hSub (Min.min ( …
    -/
    omega
    /-
      🎉 no goals
    -/
    /-
      case h₂
      α : Type u
      a b : Array α
      i j : Nat
      h : LE.le a.size i
      ⊢ ∀ (i_1 : Nat) (hi₁ : LT.lt i_1 ((HAppend.hAppend a b).extract i j).size) (hi …
    -/
  · intro k hi h2
    rw [getElem_extract, getElem_extract,
      getElem_append_right (show size a ≤ i + k by omega)]
    /-
      case h₂
      α : Type u
      a b : Array α
      i j : Nat
      h : LE.le a.size i
      k : Nat
      hi : LT.lt k ((HAppend.hAppend a b).extract i j).size
      h2 : LT.lt k (b.extract (HSub.hSub i a.size) (HSub.hSub j a.size)).size
      ⊢ Eq (GetElem.getElem b (HSub.hSub (HAdd.hAdd i k) a.size) ⋯) (GetElem.getElem …
    -/
    congr
    /-
      case h₂.e_i
      α : Type u
      a b : Array α
      i j : Nat
      h : LE.le a.size i
      k : Nat
      hi : LT.lt k ((HAppend.hAppend a b).extract i j).size
      h2 : LT.lt k (b.extract (HSub.hSub i a.size) (HSub.hSub j a.size)).size
      ⊢ Eq (HSub.hSub (HAdd.hAdd i k) a.size) (HAdd.hAdd (HSub.hSub i a.size) k)
    -/
    omega
    /-
      🎉 no goals
    -/


theorem extract_eq_of_size_le_end {l p : Nat} {a : Array α} (h : a.size ≤ l) :
    a.extract p l = a.extract p a.size := by
  /-
    α : Type u
    l p : Nat
    a : Array α
    h : LE.le a.size l
    ⊢ Eq (a.extract p l) (a.extract p a.size)
  -/
  simp only [extract, Nat.min_eq_right h, Nat.sub_eq, mkEmpty_eq, Nat.min_self]
  /-
    🎉 no goals
  -/


theorem extract_extract {s1 e2 e1 s2 : Nat} {a : Array α} (h : s1 + e2 ≤ e1) :
    (a.extract s1 e1).extract s2 e2 = a.extract (s1 + s2) (s1 + e2) := by
  /-
    α : Type u
    s1 e2 e1 s2 : Nat
    a : Array α
    h : LE.le (HAdd.hAdd s1 e2) e1
    ⊢ Eq ((a.extract s1 e1).extract s2 e2) (a.extract (HAdd.hAdd s1 s2) (HAdd.hAdd …
  -/
  apply ext
    /-
      case h₁
      α : Type u
      s1 e2 e1 s2 : Nat
      a : Array α
      h : LE.le (HAdd.hAdd s1 e2) e1
      ⊢ Eq ((a.extract s1 e1).extract s2 e2).size (a.extract (HAdd.hAdd s1 s2) (HAdd …
    -/
  · simp only [size_extract]
    /-
      case h₁
      α : Type u
      s1 e2 e1 s2 : Nat
      a : Array α
      h : LE.le (HAdd.hAdd s1 e2) e1
      ⊢ Eq (HSub.hSub (Min.min e2 (HSub.hSub (Min.min e1 a.size) s1)) s2) (HSub.hSub …
    -/
    omega
    /-
      🎉 no goals
    -/
    /-
      case h₂
      α : Type u
      s1 e2 e1 s2 : Nat
      a : Array α
      h : LE.le (HAdd.hAdd s1 e2) e1
      ⊢ ∀ (i : Nat) (hi₁ : LT.lt i ((a.extract s1 e1).extract s2 e2).size) (hi₂ : LT …
    -/
  · intro i h1 h2
    /-
      case h₂
      α : Type u
      s1 e2 e1 s2 : Nat
      a : Array α
      h : LE.le (HAdd.hAdd s1 e2) e1
      i : Nat
      h1 : LT.lt i ((a.extract s1 e1).extract s2 e2).size
      h2 : LT.lt i (a.extract (HAdd.hAdd s1 s2) (HAdd.hAdd s1 e2)).size
      ⊢ Eq (GetElem.getElem ((a.extract s1 e1).extract s2 e2) i h1) (GetElem.getElem …
    -/
    simp only [getElem_extract, Nat.add_assoc]
    /-
      🎉 no goals
    -/


