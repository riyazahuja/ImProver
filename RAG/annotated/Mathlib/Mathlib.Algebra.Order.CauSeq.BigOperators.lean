lemma of_abv_le (n : ℕ) (hm : ∀ m, n ≤ m → abv (f m) ≤ a m) :
    IsCauSeq abs (fun n ↦ ∑ i ∈ range n, a i) → IsCauSeq abv fun n ↦ ∑ i ∈ range n, f i := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : Nat → β
    a : Nat → α
    n : Nat
    hm : ∀ (m : Nat), LE.le n m → LE.le (abv (f m)) (a m)
    ⊢ (IsCauSeq abs fun n => (Finset.range n).sum fun i => a i) → IsCauSeq abv fun …
  -/
  intro hg ε ε0
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : Nat → β
    a : Nat → α
    n : Nat
    hm : ∀ (m : Nat), LE.le n m → LE.le (abv (f m)) (a m)
    hg : IsCauSeq abs fun n => (Finset.range n).sum fun i => a i
    ε : α
    ε0 : GT.gt ε 0
    ⊢ Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (abv (HSub.hSub ((fun n => (F …
  -/
  cases' hg (ε / 2) (div_pos ε0 (by norm_num)) with i hi
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : Nat → β
    a : Nat → α
    n : Nat
    hm : ∀ (m : Nat), LE.le n m → LE.le (abv (f m)) (a m)
    hg : IsCauSeq abs fun n => (Finset.range n).sum fun i => a i
    ε : α
    ε0 : GT.gt ε 0
    i : Nat
    hi : ∀ (j : Nat), GE.ge j i → LT.lt (abs (HSub.hSub ((fun n => (Finset.range n …
    ⊢ Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (abv (HSub.hSub ((fun n => (F …
  -/
  exists max n i
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : Nat → β
    a : Nat → α
    n : Nat
    hm : ∀ (m : Nat), LE.le n m → LE.le (abv (f m)) (a m)
    hg : IsCauSeq abs fun n => (Finset.range n).sum fun i => a i
    ε : α
    ε0 : GT.gt ε 0
    i : Nat
    hi : ∀ (j : Nat), GE.ge j i → LT.lt (abs (HSub.hSub ((fun n => (Finset.range n …
    ⊢ ∀ (j : Nat), GE.ge j (Max.max n i) → LT.lt (abv (HSub.hSub ((fun n => (Finse …
  -/
  intro j ji
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : Nat → β
    a : Nat → α
    n : Nat
    hm : ∀ (m : Nat), LE.le n m → LE.le (abv (f m)) (a m)
    hg : IsCauSeq abs fun n => (Finset.range n).sum fun i => a i
    ε : α
    ε0 : GT.gt ε 0
    i : Nat
    hi : ∀ (j : Nat), GE.ge j i → LT.lt (abs (HSub.hSub ((fun n => (Finset.range n …
    j : Nat
    ji : GE.ge j (Max.max n i)
    ⊢ LT.lt (abv (HSub.hSub ((fun n => (Finset.range n).sum fun i => f i) j) ((fun …
  -/
  have hi₁ := hi j (le_trans (le_max_right n i) ji)
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : Nat → β
    a : Nat → α
    n : Nat
    hm : ∀ (m : Nat), LE.le n m → LE.le (abv (f m)) (a m)
    hg : IsCauSeq abs fun n => (Finset.range n).sum fun i => a i
    ε : α
    ε0 : GT.gt ε 0
    i : Nat
    hi : ∀ (j : Nat), GE.ge j i → LT.lt (abs (HSub.hSub ((fun n => (Finset.range n …
    j : Nat
    ji : GE.ge j (Max.max n i)
    hi₁ : LT.lt (abs (HSub.hSub ((fun n => (Finset.range n).sum fun i => a i) j) ( …
    ⊢ LT.lt (abv (HSub.hSub ((fun n => (Finset.range n).sum fun i => f i) j) ((fun …
  -/
  have hi₂ := hi (max n i) (le_max_right n i)
  have sub_le :=
    abs_sub_le (∑ k ∈ range j, a k) (∑ k ∈ range i, a k) (∑ k ∈ range (max n i), a k)
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : Nat → β
    a : Nat → α
    n : Nat
    hm : ∀ (m : Nat), LE.le n m → LE.le (abv (f m)) (a m)
    hg : IsCauSeq abs fun n => (Finset.range n).sum fun i => a i
    ε : α
    ε0 : GT.gt ε 0
    i : Nat
    hi : ∀ (j : Nat), GE.ge j i → LT.lt (abs (HSub.hSub ((fun n => (Finset.range n …
    j : Nat
    ji : GE.ge j (Max.max n i)
    hi₁ : LT.lt (abs (HSub.hSub ((fun n => (Finset.range n).sum fun i => a i) j) ( …
    hi₂ : LT.lt (abs (HSub.hSub ((fun n => (Finset.range n).sum fun i => a i) (Max …
    sub_le : LE.le (abs (HSub.hSub ((Finset.range j).sum fun k => a k) ((Finset.ra …
    ⊢ LT.lt (abv (HSub.hSub ((fun n => (Finset.range n).sum fun i => f i) j) ((fun …
  -/
  have := add_lt_add hi₁ hi₂
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : Nat → β
    a : Nat → α
    n : Nat
    hm : ∀ (m : Nat), LE.le n m → LE.le (abv (f m)) (a m)
    hg : IsCauSeq abs fun n => (Finset.range n).sum fun i => a i
    ε : α
    ε0 : GT.gt ε 0
    i : Nat
    hi : ∀ (j : Nat), GE.ge j i → LT.lt (abs (HSub.hSub ((fun n => (Finset.range n …
    j : Nat
    ji : GE.ge j (Max.max n i)
    hi₁ : LT.lt (abs (HSub.hSub ((fun n => (Finset.range n).sum fun i => a i) j) ( …
    hi₂ : LT.lt (abs (HSub.hSub ((fun n => (Finset.range n).sum fun i => a i) (Max …
    sub_le : LE.le (abs (HSub.hSub ((Finset.range j).sum fun k => a k) ((Finset.ra …
    this : LT.lt (HAdd.hAdd (abs (HSub.hSub ((fun n => (Finset.range n).sum fun i  …
    ⊢ LT.lt (abv (HSub.hSub ((fun n => (Finset.range n).sum fun i => f i) j) ((fun …
  -/
  rw [abs_sub_comm (∑ k ∈ range (max n i), a k), add_halves ε] at this
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : Nat → β
    a : Nat → α
    n : Nat
    hm : ∀ (m : Nat), LE.le n m → LE.le (abv (f m)) (a m)
    hg : IsCauSeq abs fun n => (Finset.range n).sum fun i => a i
    ε : α
    ε0 : GT.gt ε 0
    i : Nat
    hi : ∀ (j : Nat), GE.ge j i → LT.lt (abs (HSub.hSub ((fun n => (Finset.range n …
    j : Nat
    ji : GE.ge j (Max.max n i)
    hi₁ : LT.lt (abs (HSub.hSub ((fun n => (Finset.range n).sum fun i => a i) j) ( …
    hi₂ : LT.lt (abs (HSub.hSub ((fun n => (Finset.range n).sum fun i => a i) (Max …
    sub_le : LE.le (abs (HSub.hSub ((Finset.range j).sum fun k => a k) ((Finset.ra …
    this : LT.lt (HAdd.hAdd (abs (HSub.hSub ((fun n => (Finset.range n).sum fun i  …
    ⊢ LT.lt (abv (HSub.hSub ((fun n => (Finset.range n).sum fun i => f i) j) ((fun …
  -/
  refine lt_of_le_of_lt (le_trans (le_trans ?_ (le_abs_self _)) sub_le) this
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : Nat → β
    a : Nat → α
    n : Nat
    hm : ∀ (m : Nat), LE.le n m → LE.le (abv (f m)) (a m)
    hg : IsCauSeq abs fun n => (Finset.range n).sum fun i => a i
    ε : α
    ε0 : GT.gt ε 0
    i : Nat
    hi : ∀ (j : Nat), GE.ge j i → LT.lt (abs (HSub.hSub ((fun n => (Finset.range n …
    j : Nat
    ji : GE.ge j (Max.max n i)
    hi₁ : LT.lt (abs (HSub.hSub ((fun n => (Finset.range n).sum fun i => a i) j) ( …
    hi₂ : LT.lt (abs (HSub.hSub ((fun n => (Finset.range n).sum fun i => a i) (Max …
    sub_le : LE.le (abs (HSub.hSub ((Finset.range j).sum fun k => a k) ((Finset.ra …
    this : LT.lt (HAdd.hAdd (abs (HSub.hSub ((fun n => (Finset.range n).sum fun i  …
    ⊢ LE.le (abv (HSub.hSub ((fun n => (Finset.range n).sum fun i => f i) j) ((fun …
  -/
  generalize hk : j - max n i = k
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : Nat → β
    a : Nat → α
    n : Nat
    hm : ∀ (m : Nat), LE.le n m → LE.le (abv (f m)) (a m)
    hg : IsCauSeq abs fun n => (Finset.range n).sum fun i => a i
    ε : α
    ε0 : GT.gt ε 0
    i : Nat
    hi : ∀ (j : Nat), GE.ge j i → LT.lt (abs (HSub.hSub ((fun n => (Finset.range n …
    j : Nat
    ji : GE.ge j (Max.max n i)
    hi₁ : LT.lt (abs (HSub.hSub ((fun n => (Finset.range n).sum fun i => a i) j) ( …
    hi₂ : LT.lt (abs (HSub.hSub ((fun n => (Finset.range n).sum fun i => a i) (Max …
    sub_le : LE.le (abs (HSub.hSub ((Finset.range j).sum fun k => a k) ((Finset.ra …
    this : LT.lt (HAdd.hAdd (abs (HSub.hSub ((fun n => (Finset.range n).sum fun i  …
    k : Nat
    hk : Eq (HSub.hSub j (Max.max n i)) k
    ⊢ LE.le (abv (HSub.hSub ((fun n => (Finset.range n).sum fun i => f i) j) ((fun …
  -/
  clear this hi₂ hi₁ hi ε0 ε hg sub_le
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : Nat → β
    a : Nat → α
    n : Nat
    hm : ∀ (m : Nat), LE.le n m → LE.le (abv (f m)) (a m)
    i j : Nat
    ji : GE.ge j (Max.max n i)
    k : Nat
    hk : Eq (HSub.hSub j (Max.max n i)) k
    ⊢ LE.le (abv (HSub.hSub ((fun n => (Finset.range n).sum fun i => f i) j) ((fun …
  -/
  rw [tsub_eq_iff_eq_add_of_le ji] at hk
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : Nat → β
    a : Nat → α
    n : Nat
    hm : ∀ (m : Nat), LE.le n m → LE.le (abv (f m)) (a m)
    i j : Nat
    ji : GE.ge j (Max.max n i)
    k : Nat
    hk : Eq j (HAdd.hAdd k (Max.max n i))
    ⊢ LE.le (abv (HSub.hSub ((fun n => (Finset.range n).sum fun i => f i) j) ((fun …
  -/
  rw [hk]
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : Nat → β
    a : Nat → α
    n : Nat
    hm : ∀ (m : Nat), LE.le n m → LE.le (abv (f m)) (a m)
    i j : Nat
    ji : GE.ge j (Max.max n i)
    k : Nat
    hk : Eq j (HAdd.hAdd k (Max.max n i))
    ⊢ LE.le (abv (HSub.hSub ((fun n => (Finset.range n).sum fun i => f i) (HAdd.hA …
  -/
  dsimp only
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : Nat → β
    a : Nat → α
    n : Nat
    hm : ∀ (m : Nat), LE.le n m → LE.le (abv (f m)) (a m)
    i j : Nat
    ji : GE.ge j (Max.max n i)
    k : Nat
    hk : Eq j (HAdd.hAdd k (Max.max n i))
    ⊢ LE.le (abv (HSub.hSub ((Finset.range (HAdd.hAdd k (Max.max n i))).sum fun i  …
  -/
  clear hk ji j
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : Nat → β
    a : Nat → α
    n : Nat
    hm : ∀ (m : Nat), LE.le n m → LE.le (abv (f m)) (a m)
    i k : Nat
    ⊢ LE.le (abv (HSub.hSub ((Finset.range (HAdd.hAdd k (Max.max n i))).sum fun i  …
  -/
  induction' k with k' hi
    /-
      case intro.zero
      α : Type u_1
      β : Type u_2
      inst✝² : LinearOrderedField α
      inst✝¹ : Ring β
      abv : β → α
      inst✝ : IsAbsoluteValue abv
      f : Nat → β
      a : Nat → α
      n : Nat
      hm : ∀ (m : Nat), LE.le n m → LE.le (abv (f m)) (a m)
      i : Nat
      ⊢ LE.le (abv (HSub.hSub ((Finset.range (HAdd.hAdd 0 (Max.max n i))).sum fun i  …
    -/
  · simp [abv_zero abv]
    /-
      🎉 no goals
    -/
  /-
    case intro.succ
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : Nat → β
    a : Nat → α
    n : Nat
    hm : ∀ (m : Nat), LE.le n m → LE.le (abv (f m)) (a m)
    i k' : Nat
    hi : LE.le (abv (HSub.hSub ((Finset.range (HAdd.hAdd k' (Max.max n i))).sum fu …
    ⊢ LE.le (abv (HSub.hSub ((Finset.range (HAdd.hAdd (HAdd.hAdd k' 1) (Max.max n  …
  -/
  simp only [Nat.succ_add, Nat.succ_eq_add_one, Finset.sum_range_succ_comm]
  /-
    case intro.succ
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : Nat → β
    a : Nat → α
    n : Nat
    hm : ∀ (m : Nat), LE.le n m → LE.le (abv (f m)) (a m)
    i k' : Nat
    hi : LE.le (abv (HSub.hSub ((Finset.range (HAdd.hAdd k' (Max.max n i))).sum fu …
    ⊢ LE.le (abv (HSub.hSub (HAdd.hAdd (f (HAdd.hAdd k' (Max.max n i))) ((Finset.r …
  -/
  simp only [add_assoc, sub_eq_add_neg]
  /-
    case intro.succ
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : Nat → β
    a : Nat → α
    n : Nat
    hm : ∀ (m : Nat), LE.le n m → LE.le (abv (f m)) (a m)
    i k' : Nat
    hi : LE.le (abv (HSub.hSub ((Finset.range (HAdd.hAdd k' (Max.max n i))).sum fu …
    ⊢ LE.le (abv (HAdd.hAdd (f (HAdd.hAdd k' (Max.max n i))) (HAdd.hAdd ((Finset.r …
  -/
  refine le_trans (abv_add _ _ _) ?_
  /-
    case intro.succ
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : Nat → β
    a : Nat → α
    n : Nat
    hm : ∀ (m : Nat), LE.le n m → LE.le (abv (f m)) (a m)
    i k' : Nat
    hi : LE.le (abv (HSub.hSub ((Finset.range (HAdd.hAdd k' (Max.max n i))).sum fu …
    ⊢ LE.le (HAdd.hAdd (abv (f (HAdd.hAdd k' (Max.max n i)))) (abv (HAdd.hAdd ((Fi …
  -/
  simp only [sub_eq_add_neg] at hi
  /-
    case intro.succ
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f : Nat → β
    a : Nat → α
    n : Nat
    hm : ∀ (m : Nat), LE.le n m → LE.le (abv (f m)) (a m)
    i k' : Nat
    hi : LE.le (abv (HAdd.hAdd ((Finset.range (HAdd.hAdd k' (Max.max n i))).sum fu …
    ⊢ LE.le (HAdd.hAdd (abv (f (HAdd.hAdd k' (Max.max n i)))) (abv (HAdd.hAdd ((Fi …
  -/
  exact add_le_add (hm _ (le_add_of_nonneg_of_le (Nat.zero_le _) (le_max_left _ _))) hi
  /-
    🎉 no goals
  -/


lemma of_abv (hf : IsCauSeq abs fun m ↦ ∑ n ∈ range m, abv (f n)) :
    IsCauSeq abv fun m ↦ ∑ n ∈ range m, f n :=
  hf.of_abv_le 0 fun _ _ ↦ le_rfl


theorem _root_.cauchy_product (ha : IsCauSeq abs fun m ↦ ∑ n ∈ range m, abv (f n))
    (hb : IsCauSeq abv fun m ↦ ∑ n ∈ range m, g n) (ε : α) (ε0 : 0 < ε) :
    ∃ i : ℕ, ∀ j ≥ i,
      abv ((∑ k ∈ range j, f k) * ∑ k ∈ range j, g k -
        ∑ n ∈ range j, ∑ m ∈ range (n + 1), f m * g (n - m)) < ε := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f g : Nat → β
    ha : IsCauSeq abs fun m => (Finset.range m).sum fun n => abv (f n)
    hb : IsCauSeq abv fun m => (Finset.range m).sum fun n => g n
    ε : α
    ε0 : LT.lt 0 ε
    ⊢ Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (abv (HSub.hSub (HMul.hMul (( …
  -/
  let ⟨P, hP⟩ := ha.bounded
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f g : Nat → β
    ha : IsCauSeq abs fun m => (Finset.range m).sum fun n => abv (f n)
    hb : IsCauSeq abv fun m => (Finset.range m).sum fun n => g n
    ε : α
    ε0 : LT.lt 0 ε
    P : α
    hP : ∀ (i : Nat), LT.lt (abs ((Finset.range i).sum fun n => abv (f n))) P
    ⊢ Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (abv (HSub.hSub (HMul.hMul (( …
  -/
  let ⟨Q, hQ⟩ := hb.bounded
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f g : Nat → β
    ha : IsCauSeq abs fun m => (Finset.range m).sum fun n => abv (f n)
    hb : IsCauSeq abv fun m => (Finset.range m).sum fun n => g n
    ε : α
    ε0 : LT.lt 0 ε
    P : α
    hP : ∀ (i : Nat), LT.lt (abs ((Finset.range i).sum fun n => abv (f n))) P
    Q : α
    hQ : ∀ (i : Nat), LT.lt (abv ((Finset.range i).sum fun n => g n)) Q
    ⊢ Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (abv (HSub.hSub (HMul.hMul (( …
  -/
  have hP0 : 0 < P := lt_of_le_of_lt (abs_nonneg _) (hP 0)
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f g : Nat → β
    ha : IsCauSeq abs fun m => (Finset.range m).sum fun n => abv (f n)
    hb : IsCauSeq abv fun m => (Finset.range m).sum fun n => g n
    ε : α
    ε0 : LT.lt 0 ε
    P : α
    hP : ∀ (i : Nat), LT.lt (abs ((Finset.range i).sum fun n => abv (f n))) P
    Q : α
    hQ : ∀ (i : Nat), LT.lt (abv ((Finset.range i).sum fun n => g n)) Q
    hP0 : LT.lt 0 P
    ⊢ Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (abv (HSub.hSub (HMul.hMul (( …
  -/
  have hPε0 : 0 < ε / (2 * P) := div_pos ε0 (mul_pos (show (2 : α) > 0 by norm_num) hP0)
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f g : Nat → β
    ha : IsCauSeq abs fun m => (Finset.range m).sum fun n => abv (f n)
    hb : IsCauSeq abv fun m => (Finset.range m).sum fun n => g n
    ε : α
    ε0 : LT.lt 0 ε
    P : α
    hP : ∀ (i : Nat), LT.lt (abs ((Finset.range i).sum fun n => abv (f n))) P
    Q : α
    hQ : ∀ (i : Nat), LT.lt (abv ((Finset.range i).sum fun n => g n)) Q
    hP0 : LT.lt 0 P
    hPε0 : LT.lt 0 (HDiv.hDiv ε (HMul.hMul 2 P))
    ⊢ Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (abv (HSub.hSub (HMul.hMul (( …
  -/
  let ⟨N, hN⟩ := hb.cauchy₂ hPε0
  have hQε0 : 0 < ε / (4 * Q) :=
    div_pos ε0 (mul_pos (show (0 : α) < 4 by norm_num) (lt_of_le_of_lt (abv_nonneg _ _) (hQ 0)))
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f g : Nat → β
    ha : IsCauSeq abs fun m => (Finset.range m).sum fun n => abv (f n)
    hb : IsCauSeq abv fun m => (Finset.range m).sum fun n => g n
    ε : α
    ε0 : LT.lt 0 ε
    P : α
    hP : ∀ (i : Nat), LT.lt (abs ((Finset.range i).sum fun n => abv (f n))) P
    Q : α
    hQ : ∀ (i : Nat), LT.lt (abv ((Finset.range i).sum fun n => g n)) Q
    hP0 : LT.lt 0 P
    hPε0 : LT.lt 0 (HDiv.hDiv ε (HMul.hMul 2 P))
    N : Nat
    hN : ∀ (j : Nat), GE.ge j N → ∀ (k : Nat), GE.ge k N → LT.lt (abv (HSub.hSub ( …
    hQε0 : LT.lt 0 (HDiv.hDiv ε (HMul.hMul 4 Q))
    ⊢ Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (abv (HSub.hSub (HMul.hMul (( …
  -/
  let ⟨M, hM⟩ := ha.cauchy₂ hQε0
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f g : Nat → β
    ha : IsCauSeq abs fun m => (Finset.range m).sum fun n => abv (f n)
    hb : IsCauSeq abv fun m => (Finset.range m).sum fun n => g n
    ε : α
    ε0 : LT.lt 0 ε
    P : α
    hP : ∀ (i : Nat), LT.lt (abs ((Finset.range i).sum fun n => abv (f n))) P
    Q : α
    hQ : ∀ (i : Nat), LT.lt (abv ((Finset.range i).sum fun n => g n)) Q
    hP0 : LT.lt 0 P
    hPε0 : LT.lt 0 (HDiv.hDiv ε (HMul.hMul 2 P))
    N : Nat
    hN : ∀ (j : Nat), GE.ge j N → ∀ (k : Nat), GE.ge k N → LT.lt (abv (HSub.hSub ( …
    hQε0 : LT.lt 0 (HDiv.hDiv ε (HMul.hMul 4 Q))
    M : Nat
    hM : ∀ (j : Nat), GE.ge j M → ∀ (k : Nat), GE.ge k M → LT.lt (abs (HSub.hSub ( …
    ⊢ Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (abv (HSub.hSub (HMul.hMul (( …
  -/
  refine ⟨2 * (max N M + 1), fun K hK ↦ ?_⟩
  have h₁ :
    (∑ m ∈ range K, ∑ k ∈ range (m + 1), f k * g (m - k)) =
      ∑ m ∈ range K, ∑ n ∈ range (K - m), f m * g n := by
    simpa using sum_range_diag_flip K fun m n ↦ f m * g n
  have h₂ :
    (fun i ↦ ∑ k ∈ range (K - i), f i * g k) = fun i ↦ f i * ∑ k ∈ range (K - i), g k := by
    simp [Finset.mul_sum]
  have h₃ :
    ∑ i ∈ range K, f i * ∑ k ∈ range (K - i), g k =
      ∑ i ∈ range K, f i * (∑ k ∈ range (K - i), g k - ∑ k ∈ range K, g k) +
        ∑ i ∈ range K, f i * ∑ k ∈ range K, g k := by
    rw [← sum_add_distrib]; simp [(mul_add _ _ _).symm]
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f g : Nat → β
    ha : IsCauSeq abs fun m => (Finset.range m).sum fun n => abv (f n)
    hb : IsCauSeq abv fun m => (Finset.range m).sum fun n => g n
    ε : α
    ε0 : LT.lt 0 ε
    P : α
    hP : ∀ (i : Nat), LT.lt (abs ((Finset.range i).sum fun n => abv (f n))) P
    Q : α
    hQ : ∀ (i : Nat), LT.lt (abv ((Finset.range i).sum fun n => g n)) Q
    hP0 : LT.lt 0 P
    hPε0 : LT.lt 0 (HDiv.hDiv ε (HMul.hMul 2 P))
    N : Nat
    hN : ∀ (j : Nat), GE.ge j N → ∀ (k : Nat), GE.ge k N → LT.lt (abv (HSub.hSub ( …
    hQε0 : LT.lt 0 (HDiv.hDiv ε (HMul.hMul 4 Q))
    M : Nat
    hM : ∀ (j : Nat), GE.ge j M → ∀ (k : Nat), GE.ge k M → LT.lt (abs (HSub.hSub ( …
    K : Nat
    hK : GE.ge K (HMul.hMul 2 (HAdd.hAdd (Max.max N M) 1))
    h₁ : Eq ((Finset.range K).sum fun m => (Finset.range (HAdd.hAdd m 1)).sum fun  …
    h₂ : Eq (fun i => (Finset.range (HSub.hSub K i)).sum fun k => HMul.hMul (f i)  …
    h₃ : Eq ((Finset.range K).sum fun i => HMul.hMul (f i) ((Finset.range (HSub.hS …
    ⊢ LT.lt (abv (HSub.hSub (HMul.hMul ((Finset.range K).sum fun k => f k) ((Finse …
  -/
  have two_mul_two : (4 : α) = 2 * 2 := by norm_num
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f g : Nat → β
    ha : IsCauSeq abs fun m => (Finset.range m).sum fun n => abv (f n)
    hb : IsCauSeq abv fun m => (Finset.range m).sum fun n => g n
    ε : α
    ε0 : LT.lt 0 ε
    P : α
    hP : ∀ (i : Nat), LT.lt (abs ((Finset.range i).sum fun n => abv (f n))) P
    Q : α
    hQ : ∀ (i : Nat), LT.lt (abv ((Finset.range i).sum fun n => g n)) Q
    hP0 : LT.lt 0 P
    hPε0 : LT.lt 0 (HDiv.hDiv ε (HMul.hMul 2 P))
    N : Nat
    hN : ∀ (j : Nat), GE.ge j N → ∀ (k : Nat), GE.ge k N → LT.lt (abv (HSub.hSub ( …
    hQε0 : LT.lt 0 (HDiv.hDiv ε (HMul.hMul 4 Q))
    M : Nat
    hM : ∀ (j : Nat), GE.ge j M → ∀ (k : Nat), GE.ge k M → LT.lt (abs (HSub.hSub ( …
    K : Nat
    hK : GE.ge K (HMul.hMul 2 (HAdd.hAdd (Max.max N M) 1))
    h₁ : Eq ((Finset.range K).sum fun m => (Finset.range (HAdd.hAdd m 1)).sum fun  …
    h₂ : Eq (fun i => (Finset.range (HSub.hSub K i)).sum fun k => HMul.hMul (f i)  …
    h₃ : Eq ((Finset.range K).sum fun i => HMul.hMul (f i) ((Finset.range (HSub.hS …
    two_mul_two : Eq 4 (HMul.hMul 2 2)
    ⊢ LT.lt (abv (HSub.hSub (HMul.hMul ((Finset.range K).sum fun k => f k) ((Finse …
  -/
  have hQ0 : Q ≠ 0 := fun h ↦ by simp [h, lt_irrefl] at hQε0
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f g : Nat → β
    ha : IsCauSeq abs fun m => (Finset.range m).sum fun n => abv (f n)
    hb : IsCauSeq abv fun m => (Finset.range m).sum fun n => g n
    ε : α
    ε0 : LT.lt 0 ε
    P : α
    hP : ∀ (i : Nat), LT.lt (abs ((Finset.range i).sum fun n => abv (f n))) P
    Q : α
    hQ : ∀ (i : Nat), LT.lt (abv ((Finset.range i).sum fun n => g n)) Q
    hP0 : LT.lt 0 P
    hPε0 : LT.lt 0 (HDiv.hDiv ε (HMul.hMul 2 P))
    N : Nat
    hN : ∀ (j : Nat), GE.ge j N → ∀ (k : Nat), GE.ge k N → LT.lt (abv (HSub.hSub ( …
    hQε0 : LT.lt 0 (HDiv.hDiv ε (HMul.hMul 4 Q))
    M : Nat
    hM : ∀ (j : Nat), GE.ge j M → ∀ (k : Nat), GE.ge k M → LT.lt (abs (HSub.hSub ( …
    K : Nat
    hK : GE.ge K (HMul.hMul 2 (HAdd.hAdd (Max.max N M) 1))
    h₁ : Eq ((Finset.range K).sum fun m => (Finset.range (HAdd.hAdd m 1)).sum fun  …
    h₂ : Eq (fun i => (Finset.range (HSub.hSub K i)).sum fun k => HMul.hMul (f i)  …
    h₃ : Eq ((Finset.range K).sum fun i => HMul.hMul (f i) ((Finset.range (HSub.hS …
    two_mul_two : Eq 4 (HMul.hMul 2 2)
    hQ0 : Ne Q 0
    ⊢ LT.lt (abv (HSub.hSub (HMul.hMul ((Finset.range K).sum fun k => f k) ((Finse …
  -/
  have h2Q0 : 2 * Q ≠ 0 := mul_ne_zero two_ne_zero hQ0
  have hε : ε / (2 * P) * P + ε / (4 * Q) * (2 * Q) = ε := by
    rw [← div_div, div_mul_cancel₀ _ (Ne.symm (ne_of_lt hP0)), two_mul_two, mul_assoc, ← div_div,
      div_mul_cancel₀ _ h2Q0, add_halves]
  have hNMK : max N M + 1 < K :=
    lt_of_lt_of_le (by rw [two_mul]; exact lt_add_of_pos_left _ (Nat.succ_pos _)) hK
  have hKN : N < K :=
    calc
      N ≤ max N M := le_max_left _ _
      _ < max N M + 1 := Nat.lt_succ_self _
      _ < K := hNMK
  have hsumlesum :
      (∑ i ∈ range (max N M + 1),
        abv (f i) * abv ((∑ k ∈ range (K - i), g k) - ∑ k ∈ range K, g k)) ≤
      ∑ i ∈ range (max N M + 1), abv (f i) * (ε / (2 * P)) := by
    gcongr with m hmJ
    refine le_of_lt <| hN (K - m) (le_tsub_of_add_le_left <| hK.trans' ?_) K hKN.le
    rw [two_mul]
    gcongr
    · exact (mem_range.1 hmJ).le
    · exact Nat.le_succ_of_le (le_max_left _ _)
  have hsumltP : (∑ n ∈ range (max N M + 1), abv (f n)) < P :=
    calc
      (∑ n ∈ range (max N M + 1), abv (f n)) = |∑ n ∈ range (max N M + 1), abv (f n)| :=
        Eq.symm (abs_of_nonneg (sum_nonneg fun x _ ↦ abv_nonneg abv (f x)))
      _ < P := hP (max N M + 1)

  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f g : Nat → β
    ha : IsCauSeq abs fun m => (Finset.range m).sum fun n => abv (f n)
    hb : IsCauSeq abv fun m => (Finset.range m).sum fun n => g n
    ε : α
    ε0 : LT.lt 0 ε
    P : α
    hP : ∀ (i : Nat), LT.lt (abs ((Finset.range i).sum fun n => abv (f n))) P
    Q : α
    hQ : ∀ (i : Nat), LT.lt (abv ((Finset.range i).sum fun n => g n)) Q
    hP0 : LT.lt 0 P
    hPε0 : LT.lt 0 (HDiv.hDiv ε (HMul.hMul 2 P))
    N : Nat
    hN : ∀ (j : Nat), GE.ge j N → ∀ (k : Nat), GE.ge k N → LT.lt (abv (HSub.hSub ( …
    hQε0 : LT.lt 0 (HDiv.hDiv ε (HMul.hMul 4 Q))
    M : Nat
    hM : ∀ (j : Nat), GE.ge j M → ∀ (k : Nat), GE.ge k M → LT.lt (abs (HSub.hSub ( …
    K : Nat
    hK : GE.ge K (HMul.hMul 2 (HAdd.hAdd (Max.max N M) 1))
    h₁ : Eq ((Finset.range K).sum fun m => (Finset.range (HAdd.hAdd m 1)).sum fun  …
    h₂ : Eq (fun i => (Finset.range (HSub.hSub K i)).sum fun k => HMul.hMul (f i)  …
    h₃ : Eq ((Finset.range K).sum fun i => HMul.hMul (f i) ((Finset.range (HSub.hS …
    two_mul_two : Eq 4 (HMul.hMul 2 2)
    hQ0 : Ne Q 0
    h2Q0 : Ne (HMul.hMul 2 Q) 0
    hε : Eq (HAdd.hAdd (HMul.hMul (HDiv.hDiv ε (HMul.hMul 2 P)) P) (HMul.hMul (HDi …
    hNMK : LT.lt (HAdd.hAdd (Max.max N M) 1) K
    hKN : LT.lt N K
    hsumlesum : LE.le ((Finset.range (HAdd.hAdd (Max.max N M) 1)).sum fun i => HMu …
    hsumltP : LT.lt ((Finset.range (HAdd.hAdd (Max.max N M) 1)).sum fun n => abv ( …
    ⊢ LT.lt (abv (HSub.hSub (HMul.hMul ((Finset.range K).sum fun k => f k) ((Finse …
  -/
  rw [h₁, h₂, h₃, sum_mul, ← sub_sub, sub_right_comm, sub_self, zero_sub, abv_neg abv]
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f g : Nat → β
    ha : IsCauSeq abs fun m => (Finset.range m).sum fun n => abv (f n)
    hb : IsCauSeq abv fun m => (Finset.range m).sum fun n => g n
    ε : α
    ε0 : LT.lt 0 ε
    P : α
    hP : ∀ (i : Nat), LT.lt (abs ((Finset.range i).sum fun n => abv (f n))) P
    Q : α
    hQ : ∀ (i : Nat), LT.lt (abv ((Finset.range i).sum fun n => g n)) Q
    hP0 : LT.lt 0 P
    hPε0 : LT.lt 0 (HDiv.hDiv ε (HMul.hMul 2 P))
    N : Nat
    hN : ∀ (j : Nat), GE.ge j N → ∀ (k : Nat), GE.ge k N → LT.lt (abv (HSub.hSub ( …
    hQε0 : LT.lt 0 (HDiv.hDiv ε (HMul.hMul 4 Q))
    M : Nat
    hM : ∀ (j : Nat), GE.ge j M → ∀ (k : Nat), GE.ge k M → LT.lt (abs (HSub.hSub ( …
    K : Nat
    hK : GE.ge K (HMul.hMul 2 (HAdd.hAdd (Max.max N M) 1))
    h₁ : Eq ((Finset.range K).sum fun m => (Finset.range (HAdd.hAdd m 1)).sum fun  …
    h₂ : Eq (fun i => (Finset.range (HSub.hSub K i)).sum fun k => HMul.hMul (f i)  …
    h₃ : Eq ((Finset.range K).sum fun i => HMul.hMul (f i) ((Finset.range (HSub.hS …
    two_mul_two : Eq 4 (HMul.hMul 2 2)
    hQ0 : Ne Q 0
    h2Q0 : Ne (HMul.hMul 2 Q) 0
    hε : Eq (HAdd.hAdd (HMul.hMul (HDiv.hDiv ε (HMul.hMul 2 P)) P) (HMul.hMul (HDi …
    hNMK : LT.lt (HAdd.hAdd (Max.max N M) 1) K
    hKN : LT.lt N K
    hsumlesum : LE.le ((Finset.range (HAdd.hAdd (Max.max N M) 1)).sum fun i => HMu …
    hsumltP : LT.lt ((Finset.range (HAdd.hAdd (Max.max N M) 1)).sum fun n => abv ( …
    ⊢ LT.lt (abv ((Finset.range K).sum fun i => HMul.hMul (f i) (HSub.hSub ((Finse …
  -/
  refine lt_of_le_of_lt (IsAbsoluteValue.abv_sum _ _ _) ?_
  suffices
    (∑ i ∈ range (max N M + 1),
          abv (f i) * abv ((∑ k ∈ range (K - i), g k) - ∑ k ∈ range K, g k)) +
        ((∑ i ∈ range K, abv (f i) * abv ((∑ k ∈ range (K - i), g k) - ∑ k ∈ range K, g k)) -
          ∑ i ∈ range (max N M + 1),
            abv (f i) * abv ((∑ k ∈ range (K - i), g k) - ∑ k ∈ range K, g k)) <
      ε / (2 * P) * P + ε / (4 * Q) * (2 * Q) by
    rw [hε] at this
    simpa [abv_mul abv] using this
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f g : Nat → β
    ha : IsCauSeq abs fun m => (Finset.range m).sum fun n => abv (f n)
    hb : IsCauSeq abv fun m => (Finset.range m).sum fun n => g n
    ε : α
    ε0 : LT.lt 0 ε
    P : α
    hP : ∀ (i : Nat), LT.lt (abs ((Finset.range i).sum fun n => abv (f n))) P
    Q : α
    hQ : ∀ (i : Nat), LT.lt (abv ((Finset.range i).sum fun n => g n)) Q
    hP0 : LT.lt 0 P
    hPε0 : LT.lt 0 (HDiv.hDiv ε (HMul.hMul 2 P))
    N : Nat
    hN : ∀ (j : Nat), GE.ge j N → ∀ (k : Nat), GE.ge k N → LT.lt (abv (HSub.hSub ( …
    hQε0 : LT.lt 0 (HDiv.hDiv ε (HMul.hMul 4 Q))
    M : Nat
    hM : ∀ (j : Nat), GE.ge j M → ∀ (k : Nat), GE.ge k M → LT.lt (abs (HSub.hSub ( …
    K : Nat
    hK : GE.ge K (HMul.hMul 2 (HAdd.hAdd (Max.max N M) 1))
    h₁ : Eq ((Finset.range K).sum fun m => (Finset.range (HAdd.hAdd m 1)).sum fun  …
    h₂ : Eq (fun i => (Finset.range (HSub.hSub K i)).sum fun k => HMul.hMul (f i)  …
    h₃ : Eq ((Finset.range K).sum fun i => HMul.hMul (f i) ((Finset.range (HSub.hS …
    two_mul_two : Eq 4 (HMul.hMul 2 2)
    hQ0 : Ne Q 0
    h2Q0 : Ne (HMul.hMul 2 Q) 0
    hε : Eq (HAdd.hAdd (HMul.hMul (HDiv.hDiv ε (HMul.hMul 2 P)) P) (HMul.hMul (HDi …
    hNMK : LT.lt (HAdd.hAdd (Max.max N M) 1) K
    hKN : LT.lt N K
    hsumlesum : LE.le ((Finset.range (HAdd.hAdd (Max.max N M) 1)).sum fun i => HMu …
    hsumltP : LT.lt ((Finset.range (HAdd.hAdd (Max.max N M) 1)).sum fun n => abv ( …
    ⊢ LT.lt (HAdd.hAdd ((Finset.range (HAdd.hAdd (Max.max N M) 1)).sum fun i => HM …
  -/
  gcongr
  · exact lt_of_le_of_lt hsumlesum
        (by rw [← sum_mul, mul_comm]; gcongr)
  /-
    case h₂
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrderedField α
    inst✝¹ : Ring β
    abv : β → α
    inst✝ : IsAbsoluteValue abv
    f g : Nat → β
    ha : IsCauSeq abs fun m => (Finset.range m).sum fun n => abv (f n)
    hb : IsCauSeq abv fun m => (Finset.range m).sum fun n => g n
    ε : α
    ε0 : LT.lt 0 ε
    P : α
    hP : ∀ (i : Nat), LT.lt (abs ((Finset.range i).sum fun n => abv (f n))) P
    Q : α
    hQ : ∀ (i : Nat), LT.lt (abv ((Finset.range i).sum fun n => g n)) Q
    hP0 : LT.lt 0 P
    hPε0 : LT.lt 0 (HDiv.hDiv ε (HMul.hMul 2 P))
    N : Nat
    hN : ∀ (j : Nat), GE.ge j N → ∀ (k : Nat), GE.ge k N → LT.lt (abv (HSub.hSub ( …
    hQε0 : LT.lt 0 (HDiv.hDiv ε (HMul.hMul 4 Q))
    M : Nat
    hM : ∀ (j : Nat), GE.ge j M → ∀ (k : Nat), GE.ge k M → LT.lt (abs (HSub.hSub ( …
    K : Nat
    hK : GE.ge K (HMul.hMul 2 (HAdd.hAdd (Max.max N M) 1))
    h₁ : Eq ((Finset.range K).sum fun m => (Finset.range (HAdd.hAdd m 1)).sum fun  …
    h₂ : Eq (fun i => (Finset.range (HSub.hSub K i)).sum fun k => HMul.hMul (f i)  …
    h₃ : Eq ((Finset.range K).sum fun i => HMul.hMul (f i) ((Finset.range (HSub.hS …
    two_mul_two : Eq 4 (HMul.hMul 2 2)
    hQ0 : Ne Q 0
    h2Q0 : Ne (HMul.hMul 2 Q) 0
    hε : Eq (HAdd.hAdd (HMul.hMul (HDiv.hDiv ε (HMul.hMul 2 P)) P) (HMul.hMul (HDi …
    hNMK : LT.lt (HAdd.hAdd (Max.max N M) 1) K
    hKN : LT.lt N K
    hsumlesum : LE.le ((Finset.range (HAdd.hAdd (Max.max N M) 1)).sum fun i => HMu …
    hsumltP : LT.lt ((Finset.range (HAdd.hAdd (Max.max N M) 1)).sum fun n => abv ( …
    ⊢ LT.lt (HSub.hSub ((Finset.range K).sum fun i => HMul.hMul (abv (f i)) (abv ( …
  -/
  rw [sum_range_sub_sum_range (le_of_lt hNMK)]
  calc
    (∑ i ∈ range K with max N M + 1 ≤ i,
          abv (f i) * abv ((∑ k ∈ range (K - i), g k) - ∑ k ∈ range K, g k)) ≤
        ∑ i ∈ range K with max N M + 1 ≤ i, abv (f i) * (2 * Q) := by
        gcongr
        rw [sub_eq_add_neg]
        refine le_trans (abv_add _ _ _) ?_
        rw [two_mul, abv_neg abv]
        gcongr <;> exact le_of_lt (hQ _)
    _ < ε / (4 * Q) * (2 * Q) := by
        rw [← sum_mul, ← sum_range_sub_sum_range (le_of_lt hNMK)]
        have := lt_of_le_of_lt (abv_nonneg _ _) (hQ 0)
        gcongr
        exact (le_abs_self _).trans_lt <|
          hM _ ((Nat.le_succ_of_le (le_max_right _ _)).trans hNMK.le) _ <|
            Nat.le_succ_of_le <| le_max_right _ _


lemma of_decreasing_bounded (f : ℕ → α) {a : α} {m : ℕ} (ham : ∀ n ≥ m, |f n| ≤ a)
    (hnm : ∀ n ≥ m, f n.succ ≤ f n) : IsCauSeq abs f := fun ε ε0 ↦ by
  classical
  let ⟨k, hk⟩ := Archimedean.arch a ε0
  have h : ∃ l, ∀ n ≥ m, a - l • ε < f n :=
    ⟨k + k + 1, fun n hnm ↦
      lt_of_lt_of_le (show a - (k + (k + 1)) • ε < -|f n| from
          lt_neg.1 <| (ham n hnm).trans_lt
              (by
                rw [neg_sub, lt_sub_iff_add_lt, add_nsmul, add_nsmul, one_nsmul]
                exact add_lt_add_of_le_of_lt hk (lt_of_le_of_lt hk (lt_add_of_pos_right _ ε0))))
        (neg_le.2 <| abs_neg (f n) ▸ le_abs_self _)⟩
  let l := Nat.find h
  have hl : ∀ n : ℕ, n ≥ m → f n > a - l • ε := Nat.find_spec h
  have hl0 : l ≠ 0 := fun hl0 ↦
    not_lt_of_ge (ham m le_rfl)
      (lt_of_lt_of_le (by have := hl m (le_refl m); simpa [hl0] using this) (le_abs_self (f m)))
  cases' not_forall.1 (Nat.find_min h (Nat.pred_lt hl0)) with i hi
  rw [Classical.not_imp, not_lt] at hi
  exists i
  intro j hj
  have hfij : f j ≤ f i := (Nat.rel_of_forall_rel_succ_of_le_of_le (· ≥ ·) hnm hi.1 hj).le
  rw [abs_of_nonpos (sub_nonpos.2 hfij), neg_sub, sub_lt_iff_lt_add']
  calc
    f i ≤ a - Nat.pred l • ε := hi.2
    _ = a - l • ε + ε := by
      conv =>
        rhs
        rw [← Nat.succ_pred_eq_of_pos (Nat.pos_of_ne_zero hl0), succ_nsmul, sub_add,
          add_sub_cancel_right]
    _ < f j + ε := add_lt_add_right (hl j (le_trans hi.1 hj)) _


lemma of_mono_bounded (f : ℕ → α) {a : α} {m : ℕ} (ham : ∀ n ≥ m, |f n| ≤ a)
    (hnm : ∀ n ≥ m, f n ≤ f n.succ) : IsCauSeq abs f :=
                                                    /-
                                                      α : Type u_1
                                                      inst✝¹ : LinearOrderedField α
                                                      inst✝ : Archimedean α
                                                      f : Nat → α
                                                      a : α
                                                      m : Nat
                                                      ham : ∀ (n : Nat), GE.ge n m → LE.le (abs (f n)) a
                                                      hnm : ∀ (n : Nat), GE.ge n m → LE.le (f n) (f n.succ)
                                                      ⊢ ∀ (n : Nat), GE.ge n m → LE.le (abs (Neg.neg f n)) a
                                                    -/
                                                    /-
                                                      🎉 no goals
                                                    -/
  (of_decreasing_bounded (-f) (a := a) (m := m) (by simpa using ham) <| by simpa using hnm).of_neg
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


lemma geo_series [Nontrivial β] (x : β) (hx1 : abv x < 1) :
    IsCauSeq abv fun n ↦ ∑ m ∈ range n, x ^ m := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : LinearOrderedField α
    inst✝³ : Ring β
    abv : β → α
    inst✝² : IsAbsoluteValue abv
    inst✝¹ : Archimedean α
    inst✝ : Nontrivial β
    x : β
    hx1 : LT.lt (abv x) 1
    ⊢ IsCauSeq abv fun n => (Finset.range n).sum fun m => HPow.hPow x m
  -/
  have hx1' : abv x ≠ 1 := fun h ↦ by simp [h, lt_irrefl] at hx1
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : LinearOrderedField α
    inst✝³ : Ring β
    abv : β → α
    inst✝² : IsAbsoluteValue abv
    inst✝¹ : Archimedean α
    inst✝ : Nontrivial β
    x : β
    hx1 : LT.lt (abv x) 1
    hx1' : Ne (abv x) 1
    ⊢ IsCauSeq abv fun n => (Finset.range n).sum fun m => HPow.hPow x m
  -/
  refine of_abv ?_
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : LinearOrderedField α
    inst✝³ : Ring β
    abv : β → α
    inst✝² : IsAbsoluteValue abv
    inst✝¹ : Archimedean α
    inst✝ : Nontrivial β
    x : β
    hx1 : LT.lt (abv x) 1
    hx1' : Ne (abv x) 1
    ⊢ IsCauSeq abs fun m => (Finset.range m).sum fun n => abv (HPow.hPow x n)
  -/
  simp only [abv_pow abv, geom_sum_eq hx1']
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : LinearOrderedField α
    inst✝³ : Ring β
    abv : β → α
    inst✝² : IsAbsoluteValue abv
    inst✝¹ : Archimedean α
    inst✝ : Nontrivial β
    x : β
    hx1 : LT.lt (abv x) 1
    hx1' : Ne (abv x) 1
    ⊢ IsCauSeq abs fun m => HDiv.hDiv (HSub.hSub (HPow.hPow (abv x) m) 1) (HSub.hS …
  -/
  conv in _ / _ => rw [← neg_div_neg_eq, neg_sub, neg_sub]
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : LinearOrderedField α
    inst✝³ : Ring β
    abv : β → α
    inst✝² : IsAbsoluteValue abv
    inst✝¹ : Archimedean α
    inst✝ : Nontrivial β
    x : β
    hx1 : LT.lt (abv x) 1
    hx1' : Ne (abv x) 1
    ⊢ IsCauSeq abs fun m => HDiv.hDiv (HSub.hSub 1 (HPow.hPow (abv x) m)) (HSub.hS …
  -/
  have : 0 < 1 - abv x := sub_pos.2 hx1
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : LinearOrderedField α
    inst✝³ : Ring β
    abv : β → α
    inst✝² : IsAbsoluteValue abv
    inst✝¹ : Archimedean α
    inst✝ : Nontrivial β
    x : β
    hx1 : LT.lt (abv x) 1
    hx1' : Ne (abv x) 1
    this : LT.lt 0 (HSub.hSub 1 (abv x))
    ⊢ IsCauSeq abs fun m => HDiv.hDiv (HSub.hSub 1 (HPow.hPow (abv x) m)) (HSub.hS …
  -/
  refine @of_mono_bounded _ _ _ _ ((1 : α) / (1 - abv x)) 0 ?_ ?_
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      inst✝⁴ : LinearOrderedField α
      inst✝³ : Ring β
      abv : β → α
      inst✝² : IsAbsoluteValue abv
      inst✝¹ : Archimedean α
      inst✝ : Nontrivial β
      x : β
      hx1 : LT.lt (abv x) 1
      hx1' : Ne (abv x) 1
      this : LT.lt 0 (HSub.hSub 1 (abv x))
      ⊢ ∀ (n : Nat), GE.ge n 0 → LE.le (abs (HDiv.hDiv (HSub.hSub 1 (HPow.hPow (abv  …
    -/
  · intro n _
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      inst✝⁴ : LinearOrderedField α
      inst✝³ : Ring β
      abv : β → α
      inst✝² : IsAbsoluteValue abv
      inst✝¹ : Archimedean α
      inst✝ : Nontrivial β
      x : β
      hx1 : LT.lt (abv x) 1
      hx1' : Ne (abv x) 1
      this : LT.lt 0 (HSub.hSub 1 (abv x))
      n : Nat
      a✝ : GE.ge n 0
      ⊢ LE.le (abs (HDiv.hDiv (HSub.hSub 1 (HPow.hPow (abv x) n)) (HSub.hSub 1 (abv  …
    -/
    rw [abs_of_nonneg]
      /-
        case refine_1
        α : Type u_1
        β : Type u_2
        inst✝⁴ : LinearOrderedField α
        inst✝³ : Ring β
        abv : β → α
        inst✝² : IsAbsoluteValue abv
        inst✝¹ : Archimedean α
        inst✝ : Nontrivial β
        x : β
        hx1 : LT.lt (abv x) 1
        hx1' : Ne (abv x) 1
        this : LT.lt 0 (HSub.hSub 1 (abv x))
        n : Nat
        a✝ : GE.ge n 0
        ⊢ LE.le (HDiv.hDiv (HSub.hSub 1 (HPow.hPow (abv x) n)) (HSub.hSub 1 (abv x)))  …
      -/
    · gcongr
      /-
        case refine_1.hab
        α : Type u_1
        β : Type u_2
        inst✝⁴ : LinearOrderedField α
        inst✝³ : Ring β
        abv : β → α
        inst✝² : IsAbsoluteValue abv
        inst✝¹ : Archimedean α
        inst✝ : Nontrivial β
        x : β
        hx1 : LT.lt (abv x) 1
        hx1' : Ne (abv x) 1
        this : LT.lt 0 (HSub.hSub 1 (abv x))
        n : Nat
        a✝ : GE.ge n 0
        ⊢ LE.le (HSub.hSub 1 (HPow.hPow (abv x) n)) 1
      -/
      exact sub_le_self _ (abv_pow abv x n ▸ abv_nonneg _ _)
      /-
        🎉 no goals
      -/
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      inst✝⁴ : LinearOrderedField α
      inst✝³ : Ring β
      abv : β → α
      inst✝² : IsAbsoluteValue abv
      inst✝¹ : Archimedean α
      inst✝ : Nontrivial β
      x : β
      hx1 : LT.lt (abv x) 1
      hx1' : Ne (abv x) 1
      this : LT.lt 0 (HSub.hSub 1 (abv x))
      n : Nat
      a✝ : GE.ge n 0
      ⊢ LE.le 0 (HDiv.hDiv (HSub.hSub 1 (HPow.hPow (abv x) n)) (HSub.hSub 1 (abv x)))
    -/
    refine div_nonneg (sub_nonneg.2 ?_) (sub_nonneg.2 <| le_of_lt hx1)
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      inst✝⁴ : LinearOrderedField α
      inst✝³ : Ring β
      abv : β → α
      inst✝² : IsAbsoluteValue abv
      inst✝¹ : Archimedean α
      inst✝ : Nontrivial β
      x : β
      hx1 : LT.lt (abv x) 1
      hx1' : Ne (abv x) 1
      this : LT.lt 0 (HSub.hSub 1 (abv x))
      n : Nat
      a✝ : GE.ge n 0
      ⊢ LE.le (HPow.hPow (abv x) n) 1
    -/
    exact pow_le_one₀ (by positivity) hx1.le
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      inst✝⁴ : LinearOrderedField α
      inst✝³ : Ring β
      abv : β → α
      inst✝² : IsAbsoluteValue abv
      inst✝¹ : Archimedean α
      inst✝ : Nontrivial β
      x : β
      hx1 : LT.lt (abv x) 1
      hx1' : Ne (abv x) 1
      this : LT.lt 0 (HSub.hSub 1 (abv x))
      ⊢ ∀ (n : Nat), GE.ge n 0 → LE.le (HDiv.hDiv (HSub.hSub 1 (HPow.hPow (abv x) n) …
    -/
  · intro n _
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      inst✝⁴ : LinearOrderedField α
      inst✝³ : Ring β
      abv : β → α
      inst✝² : IsAbsoluteValue abv
      inst✝¹ : Archimedean α
      inst✝ : Nontrivial β
      x : β
      hx1 : LT.lt (abv x) 1
      hx1' : Ne (abv x) 1
      this : LT.lt 0 (HSub.hSub 1 (abv x))
      n : Nat
      a✝ : GE.ge n 0
      ⊢ LE.le (HDiv.hDiv (HSub.hSub 1 (HPow.hPow (abv x) n)) (HSub.hSub 1 (abv x)))  …
    -/
    rw [← one_mul (abv x ^ n), pow_succ']
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      inst✝⁴ : LinearOrderedField α
      inst✝³ : Ring β
      abv : β → α
      inst✝² : IsAbsoluteValue abv
      inst✝¹ : Archimedean α
      inst✝ : Nontrivial β
      x : β
      hx1 : LT.lt (abv x) 1
      hx1' : Ne (abv x) 1
      this : LT.lt 0 (HSub.hSub 1 (abv x))
      n : Nat
      a✝ : GE.ge n 0
      ⊢ LE.le (HDiv.hDiv (HSub.hSub 1 (HMul.hMul 1 (HPow.hPow (abv x) n))) (HSub.hSu …
    -/
    gcongr
    /-
      🎉 no goals
    -/


lemma geo_series_const (a : α) {x : α} (hx1 : |x| < 1) :
    IsCauSeq abs fun m ↦ ∑ n ∈ range m, (a * x ^ n) := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrderedField α
    inst✝ : Archimedean α
    a x : α
    hx1 : LT.lt (abs x) 1
    ⊢ IsCauSeq abs fun m => (Finset.range m).sum fun n => HMul.hMul a (HPow.hPow x …
  -/
  simpa [mul_sum, Pi.mul_def] using (const a).mul (geo_series x hx1)
  /-
    🎉 no goals
  -/


lemma series_ratio_test {f : ℕ → β} (n : ℕ) (r : α) (hr0 : 0 ≤ r) (hr1 : r < 1)
    (h : ∀ m, n ≤ m → abv (f m.succ) ≤ r * abv (f m)) :
    IsCauSeq abv fun m ↦ ∑ n ∈ range m, f n := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : LinearOrderedField α
    inst✝² : Ring β
    abv : β → α
    inst✝¹ : IsAbsoluteValue abv
    inst✝ : Archimedean α
    f : Nat → β
    n : Nat
    r : α
    hr0 : LE.le 0 r
    hr1 : LT.lt r 1
    h : ∀ (m : Nat), LE.le n m → LE.le (abv (f m.succ)) (HMul.hMul r (abv (f m)))
    ⊢ IsCauSeq abv fun m => (Finset.range m).sum fun n => f n
  -/
  have har1 : |r| < 1 := by rwa [abs_of_nonneg hr0]
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : LinearOrderedField α
    inst✝² : Ring β
    abv : β → α
    inst✝¹ : IsAbsoluteValue abv
    inst✝ : Archimedean α
    f : Nat → β
    n : Nat
    r : α
    hr0 : LE.le 0 r
    hr1 : LT.lt r 1
    h : ∀ (m : Nat), LE.le n m → LE.le (abv (f m.succ)) (HMul.hMul r (abv (f m)))
    har1 : LT.lt (abs r) 1
    ⊢ IsCauSeq abv fun m => (Finset.range m).sum fun n => f n
  -/
  refine (geo_series_const (abv (f n.succ) * r⁻¹ ^ n.succ) har1).of_abv_le n.succ fun m hmn ↦ ?_
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : LinearOrderedField α
    inst✝² : Ring β
    abv : β → α
    inst✝¹ : IsAbsoluteValue abv
    inst✝ : Archimedean α
    f : Nat → β
    n : Nat
    r : α
    hr0 : LE.le 0 r
    hr1 : LT.lt r 1
    h : ∀ (m : Nat), LE.le n m → LE.le (abv (f m.succ)) (HMul.hMul r (abv (f m)))
    har1 : LT.lt (abs r) 1
    m : Nat
    hmn : LE.le n.succ m
    ⊢ LE.le (abv (f m)) (HMul.hMul (HMul.hMul (abv (f n.succ)) (HPow.hPow (Inv.inv …
  -/
  obtain rfl | hr := hr0.eq_or_lt
    /-
      case inl
      α : Type u_1
      β : Type u_2
      inst✝³ : LinearOrderedField α
      inst✝² : Ring β
      abv : β → α
      inst✝¹ : IsAbsoluteValue abv
      inst✝ : Archimedean α
      f : Nat → β
      n m : Nat
      hmn : LE.le n.succ m
      hr0 : LE.le 0 0
      hr1 : LT.lt 0 1
      h : ∀ (m : Nat), LE.le n m → LE.le (abv (f m.succ)) (HMul.hMul 0 (abv (f m)))
      har1 : LT.lt (abs 0) 1
      ⊢ LE.le (abv (f m)) (HMul.hMul (HMul.hMul (abv (f n.succ)) (HPow.hPow (Inv.inv …
    -/
  · have m_pos := lt_of_lt_of_le (Nat.succ_pos n) hmn
    /-
      case inl
      α : Type u_1
      β : Type u_2
      inst✝³ : LinearOrderedField α
      inst✝² : Ring β
      abv : β → α
      inst✝¹ : IsAbsoluteValue abv
      inst✝ : Archimedean α
      f : Nat → β
      n m : Nat
      hmn : LE.le n.succ m
      hr0 : LE.le 0 0
      hr1 : LT.lt 0 1
      h : ∀ (m : Nat), LE.le n m → LE.le (abv (f m.succ)) (HMul.hMul 0 (abv (f m)))
      har1 : LT.lt (abs 0) 1
      m_pos : LT.lt 0 m
      ⊢ LE.le (abv (f m)) (HMul.hMul (HMul.hMul (abv (f n.succ)) (HPow.hPow (Inv.inv …
    -/
    have := h m.pred (Nat.le_of_succ_le_succ (by rwa [Nat.succ_pred_eq_of_pos m_pos]))
    /-
      case inl
      α : Type u_1
      β : Type u_2
      inst✝³ : LinearOrderedField α
      inst✝² : Ring β
      abv : β → α
      inst✝¹ : IsAbsoluteValue abv
      inst✝ : Archimedean α
      f : Nat → β
      n m : Nat
      hmn : LE.le n.succ m
      hr0 : LE.le 0 0
      hr1 : LT.lt 0 1
      h : ∀ (m : Nat), LE.le n m → LE.le (abv (f m.succ)) (HMul.hMul 0 (abv (f m)))
      har1 : LT.lt (abs 0) 1
      m_pos : LT.lt 0 m
      this : LE.le (abv (f m.pred.succ)) (HMul.hMul 0 (abv (f m.pred)))
      ⊢ LE.le (abv (f m)) (HMul.hMul (HMul.hMul (abv (f n.succ)) (HPow.hPow (Inv.inv …
    -/
    simpa [Nat.sub_add_cancel m_pos, pow_succ] using this
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    β : Type u_2
    inst✝³ : LinearOrderedField α
    inst✝² : Ring β
    abv : β → α
    inst✝¹ : IsAbsoluteValue abv
    inst✝ : Archimedean α
    f : Nat → β
    n : Nat
    r : α
    hr0 : LE.le 0 r
    hr1 : LT.lt r 1
    h : ∀ (m : Nat), LE.le n m → LE.le (abv (f m.succ)) (HMul.hMul r (abv (f m)))
    har1 : LT.lt (abs r) 1
    m : Nat
    hmn : LE.le n.succ m
    hr : LT.lt 0 r
    ⊢ LE.le (abv (f m)) (HMul.hMul (HMul.hMul (abv (f n.succ)) (HPow.hPow (Inv.inv …
  -/
  generalize hk : m - n.succ = k
  /-
    case inr
    α : Type u_1
    β : Type u_2
    inst✝³ : LinearOrderedField α
    inst✝² : Ring β
    abv : β → α
    inst✝¹ : IsAbsoluteValue abv
    inst✝ : Archimedean α
    f : Nat → β
    n : Nat
    r : α
    hr0 : LE.le 0 r
    hr1 : LT.lt r 1
    h : ∀ (m : Nat), LE.le n m → LE.le (abv (f m.succ)) (HMul.hMul r (abv (f m)))
    har1 : LT.lt (abs r) 1
    m : Nat
    hmn : LE.le n.succ m
    hr : LT.lt 0 r
    k : Nat
    hk : Eq (HSub.hSub m n.succ) k
    ⊢ LE.le (abv (f m)) (HMul.hMul (HMul.hMul (abv (f n.succ)) (HPow.hPow (Inv.inv …
  -/
  replace hk : m = k + n.succ := (tsub_eq_iff_eq_add_of_le hmn).1 hk
  /-
    case inr
    α : Type u_1
    β : Type u_2
    inst✝³ : LinearOrderedField α
    inst✝² : Ring β
    abv : β → α
    inst✝¹ : IsAbsoluteValue abv
    inst✝ : Archimedean α
    f : Nat → β
    n : Nat
    r : α
    hr0 : LE.le 0 r
    hr1 : LT.lt r 1
    h : ∀ (m : Nat), LE.le n m → LE.le (abv (f m.succ)) (HMul.hMul r (abv (f m)))
    har1 : LT.lt (abs r) 1
    m : Nat
    hmn : LE.le n.succ m
    hr : LT.lt 0 r
    k : Nat
    hk : Eq m (HAdd.hAdd k n.succ)
    ⊢ LE.le (abv (f m)) (HMul.hMul (HMul.hMul (abv (f n.succ)) (HPow.hPow (Inv.inv …
  -/
  induction' k with k ih generalizing m n
    /-
      case inr.zero
      α : Type u_1
      β : Type u_2
      inst✝³ : LinearOrderedField α
      inst✝² : Ring β
      abv : β → α
      inst✝¹ : IsAbsoluteValue abv
      inst✝ : Archimedean α
      f : Nat → β
      r : α
      hr0 : LE.le 0 r
      hr1 : LT.lt r 1
      har1 : LT.lt (abs r) 1
      hr : LT.lt 0 r
      n : Nat
      h : ∀ (m : Nat), LE.le n m → LE.le (abv (f m.succ)) (HMul.hMul r (abv (f m)))
      m : Nat
      hmn : LE.le n.succ m
      hk : Eq m (HAdd.hAdd 0 n.succ)
      ⊢ LE.le (abv (f m)) (HMul.hMul (HMul.hMul (abv (f n.succ)) (HPow.hPow (Inv.inv …
    -/
  · rw [hk, Nat.zero_add, mul_right_comm, inv_pow _ _, ← div_eq_mul_inv, mul_div_cancel_right₀]
    /-
      case inr.zero.hb
      α : Type u_1
      β : Type u_2
      inst✝³ : LinearOrderedField α
      inst✝² : Ring β
      abv : β → α
      inst✝¹ : IsAbsoluteValue abv
      inst✝ : Archimedean α
      f : Nat → β
      r : α
      hr0 : LE.le 0 r
      hr1 : LT.lt r 1
      har1 : LT.lt (abs r) 1
      hr : LT.lt 0 r
      n : Nat
      h : ∀ (m : Nat), LE.le n m → LE.le (abv (f m.succ)) (HMul.hMul r (abv (f m)))
      m : Nat
      hmn : LE.le n.succ m
      hk : Eq m (HAdd.hAdd 0 n.succ)
      ⊢ Ne (HPow.hPow r n.succ) 0
    -/
    positivity
    /-
      🎉 no goals
    -/
  · have kn : k + n.succ ≥ n.succ := by
      rw [← zero_add n.succ]; exact add_le_add (Nat.zero_le _) (by simp)
    /-
      case inr.succ
      α : Type u_1
      β : Type u_2
      inst✝³ : LinearOrderedField α
      inst✝² : Ring β
      abv : β → α
      inst✝¹ : IsAbsoluteValue abv
      inst✝ : Archimedean α
      f : Nat → β
      r : α
      hr0 : LE.le 0 r
      hr1 : LT.lt r 1
      har1 : LT.lt (abs r) 1
      hr : LT.lt 0 r
      k : Nat
      ih : ∀ (n : Nat), (∀ (m : Nat), LE.le n m → LE.le (abv (f m.succ)) (HMul.hMul  …
      n : Nat
      h : ∀ (m : Nat), LE.le n m → LE.le (abv (f m.succ)) (HMul.hMul r (abv (f m)))
      m : Nat
      hmn : LE.le n.succ m
      hk : Eq m (HAdd.hAdd (HAdd.hAdd k 1) n.succ)
      kn : GE.ge (HAdd.hAdd k n.succ) n.succ
      ⊢ LE.le (abv (f m)) (HMul.hMul (HMul.hMul (abv (f n.succ)) (HPow.hPow (Inv.inv …
    -/
    rw [hk, Nat.succ_add, pow_succ r, ← mul_assoc]
    refine
      le_trans (by rw [mul_comm] <;> exact h _ (Nat.le_of_succ_le kn))
        (mul_le_mul_of_nonneg_right ?_ hr0)
    /-
      case inr.succ
      α : Type u_1
      β : Type u_2
      inst✝³ : LinearOrderedField α
      inst✝² : Ring β
      abv : β → α
      inst✝¹ : IsAbsoluteValue abv
      inst✝ : Archimedean α
      f : Nat → β
      r : α
      hr0 : LE.le 0 r
      hr1 : LT.lt r 1
      har1 : LT.lt (abs r) 1
      hr : LT.lt 0 r
      k : Nat
      ih : ∀ (n : Nat), (∀ (m : Nat), LE.le n m → LE.le (abv (f m.succ)) (HMul.hMul  …
      n : Nat
      h : ∀ (m : Nat), LE.le n m → LE.le (abv (f m.succ)) (HMul.hMul r (abv (f m)))
      m : Nat
      hmn : LE.le n.succ m
      hk : Eq m (HAdd.hAdd (HAdd.hAdd k 1) n.succ)
      kn : GE.ge (HAdd.hAdd k n.succ) n.succ
      ⊢ LE.le (abv (f (HAdd.hAdd k n.succ))) (HMul.hMul (HMul.hMul (abv (f n.succ))  …
    -/
    exact ih _ h _ (by simp) rfl
    /-
      🎉 no goals
    -/


