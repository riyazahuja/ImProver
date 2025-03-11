/-- Given a formal multilinear series `p`, a composition `c` of `n` and the index `i` of a
block of `c`, we may define a function on `Fin n → E` by picking the variables in the `i`-th block
of `n`, and applying the corresponding coefficient of `p` to these variables. This function is
called `p.applyComposition c v i` for `v : Fin n → E` and `i : Fin c.length`. -/
def applyComposition (p : FormalMultilinearSeries 𝕜 E F) {n : ℕ} (c : Composition n) :
    (Fin n → E) → Fin c.length → F := fun v i => p (c.blocksFun i) (v ∘ c.embedding i)


theorem applyComposition_ones (p : FormalMultilinearSeries 𝕜 E F) (n : ℕ) :
    p.applyComposition (Composition.ones n) = fun v i =>
      p 1 fun _ => v (Fin.castLE (Composition.length_le _) i) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹⁰ : CommRing 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : AddCommGroup F
    inst✝⁷ : Module 𝕜 E
    inst✝⁶ : Module 𝕜 F
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : TopologicalSpace F
    inst✝³ : TopologicalAddGroup E
    inst✝² : ContinuousConstSMul 𝕜 E
    inst✝¹ : TopologicalAddGroup F
    inst✝ : ContinuousConstSMul 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    ⊢ Eq (p.applyComposition (Composition.ones n)) fun v i => (p 1) fun x => v (Fi …
  -/
  funext v i
  /-
    case h.h
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹⁰ : CommRing 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : AddCommGroup F
    inst✝⁷ : Module 𝕜 E
    inst✝⁶ : Module 𝕜 F
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : TopologicalSpace F
    inst✝³ : TopologicalAddGroup E
    inst✝² : ContinuousConstSMul 𝕜 E
    inst✝¹ : TopologicalAddGroup F
    inst✝ : ContinuousConstSMul 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    v : Fin n → E
    i : Fin (Composition.ones n).length
    ⊢ Eq (p.applyComposition (Composition.ones n) v i) ((p 1) fun x => v (Fin.cast …
  -/
  apply p.congr (Composition.ones_blocksFun _ _)
  /-
    case h.h
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹⁰ : CommRing 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : AddCommGroup F
    inst✝⁷ : Module 𝕜 E
    inst✝⁶ : Module 𝕜 F
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : TopologicalSpace F
    inst✝³ : TopologicalAddGroup E
    inst✝² : ContinuousConstSMul 𝕜 E
    inst✝¹ : TopologicalAddGroup F
    inst✝ : ContinuousConstSMul 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    v : Fin n → E
    i : Fin (Composition.ones n).length
    ⊢ ∀ (i_1 : Nat) (him : LT.lt i_1 ((Composition.ones n).blocksFun i)), LT.lt i_ …
  -/
  intro j hjn hj1
  /-
    case h.h
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹⁰ : CommRing 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : AddCommGroup F
    inst✝⁷ : Module 𝕜 E
    inst✝⁶ : Module 𝕜 F
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : TopologicalSpace F
    inst✝³ : TopologicalAddGroup E
    inst✝² : ContinuousConstSMul 𝕜 E
    inst✝¹ : TopologicalAddGroup F
    inst✝ : ContinuousConstSMul 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    v : Fin n → E
    i : Fin (Composition.ones n).length
    j : Nat
    hjn : LT.lt j ((Composition.ones n).blocksFun i)
    hj1 : LT.lt j 1
    ⊢ Eq (Function.comp v ⇑((Composition.ones n).embedding i) ⟨j, hjn⟩) (v (Fin.ca …
  -/
  obtain rfl : j = 0 := by omega
  /-
    case h.h
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹⁰ : CommRing 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : AddCommGroup F
    inst✝⁷ : Module 𝕜 E
    inst✝⁶ : Module 𝕜 F
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : TopologicalSpace F
    inst✝³ : TopologicalAddGroup E
    inst✝² : ContinuousConstSMul 𝕜 E
    inst✝¹ : TopologicalAddGroup F
    inst✝ : ContinuousConstSMul 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    v : Fin n → E
    i : Fin (Composition.ones n).length
    hjn : LT.lt 0 ((Composition.ones n).blocksFun i)
    hj1 : LT.lt 0 1
    ⊢ Eq (Function.comp v ⇑((Composition.ones n).embedding i) ⟨0, hjn⟩) (v (Fin.ca …
  -/
  refine congr_arg v ?_
  /-
    case h.h
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹⁰ : CommRing 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : AddCommGroup F
    inst✝⁷ : Module 𝕜 E
    inst✝⁶ : Module 𝕜 F
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : TopologicalSpace F
    inst✝³ : TopologicalAddGroup E
    inst✝² : ContinuousConstSMul 𝕜 E
    inst✝¹ : TopologicalAddGroup F
    inst✝ : ContinuousConstSMul 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    v : Fin n → E
    i : Fin (Composition.ones n).length
    hjn : LT.lt 0 ((Composition.ones n).blocksFun i)
    hj1 : LT.lt 0 1
    ⊢ Eq (((Composition.ones n).embedding i) ⟨0, hjn⟩) (Fin.castLE ⋯ i)
  -/
  rw [Fin.ext_iff, Fin.coe_castLE, Composition.ones_embedding, Fin.val_mk]
  /-
    🎉 no goals
  -/


theorem applyComposition_single (p : FormalMultilinearSeries 𝕜 E F) {n : ℕ} (hn : 0 < n)
    (v : Fin n → E) : p.applyComposition (Composition.single n hn) v = fun _j => p n v := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹⁰ : CommRing 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : AddCommGroup F
    inst✝⁷ : Module 𝕜 E
    inst✝⁶ : Module 𝕜 F
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : TopologicalSpace F
    inst✝³ : TopologicalAddGroup E
    inst✝² : ContinuousConstSMul 𝕜 E
    inst✝¹ : TopologicalAddGroup F
    inst✝ : ContinuousConstSMul 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    hn : LT.lt 0 n
    v : Fin n → E
    ⊢ Eq (p.applyComposition (Composition.single n hn) v) fun _j => (p n) v
  -/
  ext j
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹⁰ : CommRing 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : AddCommGroup F
    inst✝⁷ : Module 𝕜 E
    inst✝⁶ : Module 𝕜 F
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : TopologicalSpace F
    inst✝³ : TopologicalAddGroup E
    inst✝² : ContinuousConstSMul 𝕜 E
    inst✝¹ : TopologicalAddGroup F
    inst✝ : ContinuousConstSMul 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    hn : LT.lt 0 n
    v : Fin n → E
    j : Fin (Composition.single n hn).length
    ⊢ Eq (p.applyComposition (Composition.single n hn) v j) ((p n) v)
  -/
  refine p.congr (by simp) fun i hi1 hi2 => ?_
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹⁰ : CommRing 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : AddCommGroup F
    inst✝⁷ : Module 𝕜 E
    inst✝⁶ : Module 𝕜 F
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : TopologicalSpace F
    inst✝³ : TopologicalAddGroup E
    inst✝² : ContinuousConstSMul 𝕜 E
    inst✝¹ : TopologicalAddGroup F
    inst✝ : ContinuousConstSMul 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    hn : LT.lt 0 n
    v : Fin n → E
    j : Fin (Composition.single n hn).length
    i : Nat
    hi1 : LT.lt i ((Composition.single n hn).blocksFun j)
    hi2 : LT.lt i n
    ⊢ Eq (Function.comp v ⇑((Composition.single n hn).embedding j) ⟨i, hi1⟩) (v ⟨i …
  -/
  dsimp
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹⁰ : CommRing 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : AddCommGroup F
    inst✝⁷ : Module 𝕜 E
    inst✝⁶ : Module 𝕜 F
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : TopologicalSpace F
    inst✝³ : TopologicalAddGroup E
    inst✝² : ContinuousConstSMul 𝕜 E
    inst✝¹ : TopologicalAddGroup F
    inst✝ : ContinuousConstSMul 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    hn : LT.lt 0 n
    v : Fin n → E
    j : Fin (Composition.single n hn).length
    i : Nat
    hi1 : LT.lt i ((Composition.single n hn).blocksFun j)
    hi2 : LT.lt i n
    ⊢ Eq (v (((Composition.single n hn).embedding j) ⟨i, hi1⟩)) (v ⟨i, hi2⟩)
  -/
  congr 1
  /-
    case h.e_a
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹⁰ : CommRing 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : AddCommGroup F
    inst✝⁷ : Module 𝕜 E
    inst✝⁶ : Module 𝕜 F
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : TopologicalSpace F
    inst✝³ : TopologicalAddGroup E
    inst✝² : ContinuousConstSMul 𝕜 E
    inst✝¹ : TopologicalAddGroup F
    inst✝ : ContinuousConstSMul 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    hn : LT.lt 0 n
    v : Fin n → E
    j : Fin (Composition.single n hn).length
    i : Nat
    hi1 : LT.lt i ((Composition.single n hn).blocksFun j)
    hi2 : LT.lt i n
    ⊢ Eq (((Composition.single n hn).embedding j) ⟨i, hi1⟩) ⟨i, hi2⟩
  -/
  convert Composition.single_embedding hn ⟨i, hi2⟩ using 1
  /-
    case h.e'_2
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹⁰ : CommRing 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : AddCommGroup F
    inst✝⁷ : Module 𝕜 E
    inst✝⁶ : Module 𝕜 F
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : TopologicalSpace F
    inst✝³ : TopologicalAddGroup E
    inst✝² : ContinuousConstSMul 𝕜 E
    inst✝¹ : TopologicalAddGroup F
    inst✝ : ContinuousConstSMul 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    hn : LT.lt 0 n
    v : Fin n → E
    j : Fin (Composition.single n hn).length
    i : Nat
    hi1 : LT.lt i ((Composition.single n hn).blocksFun j)
    hi2 : LT.lt i n
    ⊢ Eq (((Composition.single n hn).embedding j) ⟨i, hi1⟩) (((Composition.single  …
  -/
  cases' j with j_val j_property
  /-
    case h.e'_2.mk
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹⁰ : CommRing 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : AddCommGroup F
    inst✝⁷ : Module 𝕜 E
    inst✝⁶ : Module 𝕜 F
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : TopologicalSpace F
    inst✝³ : TopologicalAddGroup E
    inst✝² : ContinuousConstSMul 𝕜 E
    inst✝¹ : TopologicalAddGroup F
    inst✝ : ContinuousConstSMul 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    hn : LT.lt 0 n
    v : Fin n → E
    i : Nat
    hi2 : LT.lt i n
    j_val : Nat
    j_property : LT.lt j_val (Composition.single n hn).length
    hi1 : LT.lt i ((Composition.single n hn).blocksFun ⟨j_val, j_property⟩)
    ⊢ Eq (((Composition.single n hn).embedding ⟨j_val, j_property⟩) ⟨i, hi1⟩) (((C …
  -/
  have : j_val = 0 := le_bot_iff.1 (Nat.lt_succ_iff.1 j_property)
  /-
    case h.e'_2.mk
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹⁰ : CommRing 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : AddCommGroup F
    inst✝⁷ : Module 𝕜 E
    inst✝⁶ : Module 𝕜 F
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : TopologicalSpace F
    inst✝³ : TopologicalAddGroup E
    inst✝² : ContinuousConstSMul 𝕜 E
    inst✝¹ : TopologicalAddGroup F
    inst✝ : ContinuousConstSMul 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    hn : LT.lt 0 n
    v : Fin n → E
    i : Nat
    hi2 : LT.lt i n
    j_val : Nat
    j_property : LT.lt j_val (Composition.single n hn).length
    hi1 : LT.lt i ((Composition.single n hn).blocksFun ⟨j_val, j_property⟩)
    this : Eq j_val 0
    ⊢ Eq (((Composition.single n hn).embedding ⟨j_val, j_property⟩) ⟨i, hi1⟩) (((C …
  -/
  congr!
  /-
    case h.e'_2.mk.h.e'_6.e'_1
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹⁰ : CommRing 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : AddCommGroup F
    inst✝⁷ : Module 𝕜 E
    inst✝⁶ : Module 𝕜 F
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : TopologicalSpace F
    inst✝³ : TopologicalAddGroup E
    inst✝² : ContinuousConstSMul 𝕜 E
    inst✝¹ : TopologicalAddGroup F
    inst✝ : ContinuousConstSMul 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    hn : LT.lt 0 n
    v : Fin n → E
    i : Nat
    hi2 : LT.lt i n
    j_val : Nat
    j_property : LT.lt j_val (Composition.single n hn).length
    hi1 : LT.lt i ((Composition.single n hn).blocksFun ⟨j_val, j_property⟩)
    this : Eq j_val 0
    e_2✝ : Eq (Fin ((Composition.single n hn).blocksFun ⟨j_val, j_property⟩)) (Fin …
    ⊢ Eq ((Composition.single n hn).blocksFun ⟨j_val, j_property⟩) n
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem removeZero_applyComposition (p : FormalMultilinearSeries 𝕜 E F) {n : ℕ}
    (c : Composition n) : p.removeZero.applyComposition c = p.applyComposition c := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹⁰ : CommRing 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : AddCommGroup F
    inst✝⁷ : Module 𝕜 E
    inst✝⁶ : Module 𝕜 F
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : TopologicalSpace F
    inst✝³ : TopologicalAddGroup E
    inst✝² : ContinuousConstSMul 𝕜 E
    inst✝¹ : TopologicalAddGroup F
    inst✝ : ContinuousConstSMul 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    c : Composition n
    ⊢ Eq (p.removeZero.applyComposition c) (p.applyComposition c)
  -/
  ext v i
  /-
    case h.h
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹⁰ : CommRing 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : AddCommGroup F
    inst✝⁷ : Module 𝕜 E
    inst✝⁶ : Module 𝕜 F
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : TopologicalSpace F
    inst✝³ : TopologicalAddGroup E
    inst✝² : ContinuousConstSMul 𝕜 E
    inst✝¹ : TopologicalAddGroup F
    inst✝ : ContinuousConstSMul 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    c : Composition n
    v : Fin n → E
    i : Fin c.length
    ⊢ Eq (p.removeZero.applyComposition c v i) (p.applyComposition c v i)
  -/
  simp [applyComposition, zero_lt_one.trans_le (c.one_le_blocksFun i), removeZero_of_pos]
  /-
    🎉 no goals
  -/


/-- Technical lemma stating how `p.applyComposition` commutes with updating variables. This
will be the key point to show that functions constructed from `applyComposition` retain
multilinearity. -/
theorem applyComposition_update (p : FormalMultilinearSeries 𝕜 E F) {n : ℕ} (c : Composition n)
    (j : Fin n) (v : Fin n → E) (z : E) :
    p.applyComposition c (Function.update v j z) =
      Function.update (p.applyComposition c v) (c.index j)
        (p (c.blocksFun (c.index j))
          (Function.update (v ∘ c.embedding (c.index j)) (c.invEmbedding j) z)) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹⁰ : CommRing 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : AddCommGroup F
    inst✝⁷ : Module 𝕜 E
    inst✝⁶ : Module 𝕜 F
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : TopologicalSpace F
    inst✝³ : TopologicalAddGroup E
    inst✝² : ContinuousConstSMul 𝕜 E
    inst✝¹ : TopologicalAddGroup F
    inst✝ : ContinuousConstSMul 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    c : Composition n
    j : Fin n
    v : Fin n → E
    z : E
    ⊢ Eq (p.applyComposition c (Function.update v j z)) (Function.update (p.applyC …
  -/
  ext k
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹⁰ : CommRing 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : AddCommGroup F
    inst✝⁷ : Module 𝕜 E
    inst✝⁶ : Module 𝕜 F
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : TopologicalSpace F
    inst✝³ : TopologicalAddGroup E
    inst✝² : ContinuousConstSMul 𝕜 E
    inst✝¹ : TopologicalAddGroup F
    inst✝ : ContinuousConstSMul 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    c : Composition n
    j : Fin n
    v : Fin n → E
    z : E
    k : Fin c.length
    ⊢ Eq (p.applyComposition c (Function.update v j z) k) (Function.update (p.appl …
  -/
  by_cases h : k = c.index j
    /-
      case pos
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝¹⁰ : CommRing 𝕜
      inst✝⁹ : AddCommGroup E
      inst✝⁸ : AddCommGroup F
      inst✝⁷ : Module 𝕜 E
      inst✝⁶ : Module 𝕜 F
      inst✝⁵ : TopologicalSpace E
      inst✝⁴ : TopologicalSpace F
      inst✝³ : TopologicalAddGroup E
      inst✝² : ContinuousConstSMul 𝕜 E
      inst✝¹ : TopologicalAddGroup F
      inst✝ : ContinuousConstSMul 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      n : Nat
      c : Composition n
      j : Fin n
      v : Fin n → E
      z : E
      k : Fin c.length
      h : Eq k (c.index j)
      ⊢ Eq (p.applyComposition c (Function.update v j z) k) (Function.update (p.appl …
    -/
  · rw [h]
    /-
      case pos
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝¹⁰ : CommRing 𝕜
      inst✝⁹ : AddCommGroup E
      inst✝⁸ : AddCommGroup F
      inst✝⁷ : Module 𝕜 E
      inst✝⁶ : Module 𝕜 F
      inst✝⁵ : TopologicalSpace E
      inst✝⁴ : TopologicalSpace F
      inst✝³ : TopologicalAddGroup E
      inst✝² : ContinuousConstSMul 𝕜 E
      inst✝¹ : TopologicalAddGroup F
      inst✝ : ContinuousConstSMul 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      n : Nat
      c : Composition n
      j : Fin n
      v : Fin n → E
      z : E
      k : Fin c.length
      h : Eq k (c.index j)
      ⊢ Eq (p.applyComposition c (Function.update v j z) (c.index j)) (Function.upda …
    -/
    let r : Fin (c.blocksFun (c.index j)) → Fin n := c.embedding (c.index j)
    /-
      case pos
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝¹⁰ : CommRing 𝕜
      inst✝⁹ : AddCommGroup E
      inst✝⁸ : AddCommGroup F
      inst✝⁷ : Module 𝕜 E
      inst✝⁶ : Module 𝕜 F
      inst✝⁵ : TopologicalSpace E
      inst✝⁴ : TopologicalSpace F
      inst✝³ : TopologicalAddGroup E
      inst✝² : ContinuousConstSMul 𝕜 E
      inst✝¹ : TopologicalAddGroup F
      inst✝ : ContinuousConstSMul 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      n : Nat
      c : Composition n
      j : Fin n
      v : Fin n → E
      z : E
      k : Fin c.length
      h : Eq k (c.index j)
      r : Fin (c.blocksFun (c.index j)) → Fin n := ⇑(c.embedding (c.index j))
      ⊢ Eq (p.applyComposition c (Function.update v j z) (c.index j)) (Function.upda …
    -/
    simp only [Function.update_self]
    /-
      case pos
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝¹⁰ : CommRing 𝕜
      inst✝⁹ : AddCommGroup E
      inst✝⁸ : AddCommGroup F
      inst✝⁷ : Module 𝕜 E
      inst✝⁶ : Module 𝕜 F
      inst✝⁵ : TopologicalSpace E
      inst✝⁴ : TopologicalSpace F
      inst✝³ : TopologicalAddGroup E
      inst✝² : ContinuousConstSMul 𝕜 E
      inst✝¹ : TopologicalAddGroup F
      inst✝ : ContinuousConstSMul 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      n : Nat
      c : Composition n
      j : Fin n
      v : Fin n → E
      z : E
      k : Fin c.length
      h : Eq k (c.index j)
      r : Fin (c.blocksFun (c.index j)) → Fin n := ⇑(c.embedding (c.index j))
      ⊢ Eq (p.applyComposition c (Function.update v j z) (c.index j)) ((p (c.blocksF …
    -/
    change p (c.blocksFun (c.index j)) (Function.update v j z ∘ r) = _
    /-
      case pos
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝¹⁰ : CommRing 𝕜
      inst✝⁹ : AddCommGroup E
      inst✝⁸ : AddCommGroup F
      inst✝⁷ : Module 𝕜 E
      inst✝⁶ : Module 𝕜 F
      inst✝⁵ : TopologicalSpace E
      inst✝⁴ : TopologicalSpace F
      inst✝³ : TopologicalAddGroup E
      inst✝² : ContinuousConstSMul 𝕜 E
      inst✝¹ : TopologicalAddGroup F
      inst✝ : ContinuousConstSMul 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      n : Nat
      c : Composition n
      j : Fin n
      v : Fin n → E
      z : E
      k : Fin c.length
      h : Eq k (c.index j)
      r : Fin (c.blocksFun (c.index j)) → Fin n := ⇑(c.embedding (c.index j))
      ⊢ Eq ((p (c.blocksFun (c.index j))) (Function.comp (Function.update v j z) r)) …
    -/
    let j' := c.invEmbedding j
    /-
      case pos
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝¹⁰ : CommRing 𝕜
      inst✝⁹ : AddCommGroup E
      inst✝⁸ : AddCommGroup F
      inst✝⁷ : Module 𝕜 E
      inst✝⁶ : Module 𝕜 F
      inst✝⁵ : TopologicalSpace E
      inst✝⁴ : TopologicalSpace F
      inst✝³ : TopologicalAddGroup E
      inst✝² : ContinuousConstSMul 𝕜 E
      inst✝¹ : TopologicalAddGroup F
      inst✝ : ContinuousConstSMul 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      n : Nat
      c : Composition n
      j : Fin n
      v : Fin n → E
      z : E
      k : Fin c.length
      h : Eq k (c.index j)
      r : Fin (c.blocksFun (c.index j)) → Fin n := ⇑(c.embedding (c.index j))
      j' : Fin (c.blocksFun (c.index j)) := c.invEmbedding j
      ⊢ Eq ((p (c.blocksFun (c.index j))) (Function.comp (Function.update v j z) r)) …
    -/
    suffices B : Function.update v j z ∘ r = Function.update (v ∘ r) j' z by rw [B]
    suffices C : Function.update v (r j') z ∘ r = Function.update (v ∘ r) j' z by
      convert C; exact (c.embedding_comp_inv j).symm
    /-
      case pos
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝¹⁰ : CommRing 𝕜
      inst✝⁹ : AddCommGroup E
      inst✝⁸ : AddCommGroup F
      inst✝⁷ : Module 𝕜 E
      inst✝⁶ : Module 𝕜 F
      inst✝⁵ : TopologicalSpace E
      inst✝⁴ : TopologicalSpace F
      inst✝³ : TopologicalAddGroup E
      inst✝² : ContinuousConstSMul 𝕜 E
      inst✝¹ : TopologicalAddGroup F
      inst✝ : ContinuousConstSMul 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      n : Nat
      c : Composition n
      j : Fin n
      v : Fin n → E
      z : E
      k : Fin c.length
      h : Eq k (c.index j)
      r : Fin (c.blocksFun (c.index j)) → Fin n := ⇑(c.embedding (c.index j))
      j' : Fin (c.blocksFun (c.index j)) := c.invEmbedding j
      ⊢ Eq (Function.comp (Function.update v (r j') z) r) (Function.update (Function …
    -/
    exact Function.update_comp_eq_of_injective _ (c.embedding _).injective _ _
    /-
      🎉 no goals
    -/
    /-
      case neg
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝¹⁰ : CommRing 𝕜
      inst✝⁹ : AddCommGroup E
      inst✝⁸ : AddCommGroup F
      inst✝⁷ : Module 𝕜 E
      inst✝⁶ : Module 𝕜 F
      inst✝⁵ : TopologicalSpace E
      inst✝⁴ : TopologicalSpace F
      inst✝³ : TopologicalAddGroup E
      inst✝² : ContinuousConstSMul 𝕜 E
      inst✝¹ : TopologicalAddGroup F
      inst✝ : ContinuousConstSMul 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      n : Nat
      c : Composition n
      j : Fin n
      v : Fin n → E
      z : E
      k : Fin c.length
      h : Not (Eq k (c.index j))
      ⊢ Eq (p.applyComposition c (Function.update v j z) k) (Function.update (p.appl …
    -/
  · simp only [h, Function.update_eq_self, Function.update_of_ne, Ne, not_false_iff]
    /-
      case neg
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝¹⁰ : CommRing 𝕜
      inst✝⁹ : AddCommGroup E
      inst✝⁸ : AddCommGroup F
      inst✝⁷ : Module 𝕜 E
      inst✝⁶ : Module 𝕜 F
      inst✝⁵ : TopologicalSpace E
      inst✝⁴ : TopologicalSpace F
      inst✝³ : TopologicalAddGroup E
      inst✝² : ContinuousConstSMul 𝕜 E
      inst✝¹ : TopologicalAddGroup F
      inst✝ : ContinuousConstSMul 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      n : Nat
      c : Composition n
      j : Fin n
      v : Fin n → E
      z : E
      k : Fin c.length
      h : Not (Eq k (c.index j))
      ⊢ Eq (p.applyComposition c (Function.update v j z) k) (p.applyComposition c v k)
    -/
    let r : Fin (c.blocksFun k) → Fin n := c.embedding k
    /-
      case neg
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝¹⁰ : CommRing 𝕜
      inst✝⁹ : AddCommGroup E
      inst✝⁸ : AddCommGroup F
      inst✝⁷ : Module 𝕜 E
      inst✝⁶ : Module 𝕜 F
      inst✝⁵ : TopologicalSpace E
      inst✝⁴ : TopologicalSpace F
      inst✝³ : TopologicalAddGroup E
      inst✝² : ContinuousConstSMul 𝕜 E
      inst✝¹ : TopologicalAddGroup F
      inst✝ : ContinuousConstSMul 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      n : Nat
      c : Composition n
      j : Fin n
      v : Fin n → E
      z : E
      k : Fin c.length
      h : Not (Eq k (c.index j))
      r : Fin (c.blocksFun k) → Fin n := ⇑(c.embedding k)
      ⊢ Eq (p.applyComposition c (Function.update v j z) k) (p.applyComposition c v k)
    -/
    change p (c.blocksFun k) (Function.update v j z ∘ r) = p (c.blocksFun k) (v ∘ r)
    /-
      case neg
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝¹⁰ : CommRing 𝕜
      inst✝⁹ : AddCommGroup E
      inst✝⁸ : AddCommGroup F
      inst✝⁷ : Module 𝕜 E
      inst✝⁶ : Module 𝕜 F
      inst✝⁵ : TopologicalSpace E
      inst✝⁴ : TopologicalSpace F
      inst✝³ : TopologicalAddGroup E
      inst✝² : ContinuousConstSMul 𝕜 E
      inst✝¹ : TopologicalAddGroup F
      inst✝ : ContinuousConstSMul 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      n : Nat
      c : Composition n
      j : Fin n
      v : Fin n → E
      z : E
      k : Fin c.length
      h : Not (Eq k (c.index j))
      r : Fin (c.blocksFun k) → Fin n := ⇑(c.embedding k)
      ⊢ Eq ((p (c.blocksFun k)) (Function.comp (Function.update v j z) r)) ((p (c.bl …
    -/
    suffices B : Function.update v j z ∘ r = v ∘ r by rw [B]
    /-
      case neg
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝¹⁰ : CommRing 𝕜
      inst✝⁹ : AddCommGroup E
      inst✝⁸ : AddCommGroup F
      inst✝⁷ : Module 𝕜 E
      inst✝⁶ : Module 𝕜 F
      inst✝⁵ : TopologicalSpace E
      inst✝⁴ : TopologicalSpace F
      inst✝³ : TopologicalAddGroup E
      inst✝² : ContinuousConstSMul 𝕜 E
      inst✝¹ : TopologicalAddGroup F
      inst✝ : ContinuousConstSMul 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      n : Nat
      c : Composition n
      j : Fin n
      v : Fin n → E
      z : E
      k : Fin c.length
      h : Not (Eq k (c.index j))
      r : Fin (c.blocksFun k) → Fin n := ⇑(c.embedding k)
      ⊢ Eq (Function.comp (Function.update v j z) r) (Function.comp v r)
    -/
    apply Function.update_comp_eq_of_not_mem_range
    /-
      case neg.h
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝¹⁰ : CommRing 𝕜
      inst✝⁹ : AddCommGroup E
      inst✝⁸ : AddCommGroup F
      inst✝⁷ : Module 𝕜 E
      inst✝⁶ : Module 𝕜 F
      inst✝⁵ : TopologicalSpace E
      inst✝⁴ : TopologicalSpace F
      inst✝³ : TopologicalAddGroup E
      inst✝² : ContinuousConstSMul 𝕜 E
      inst✝¹ : TopologicalAddGroup F
      inst✝ : ContinuousConstSMul 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      n : Nat
      c : Composition n
      j : Fin n
      v : Fin n → E
      z : E
      k : Fin c.length
      h : Not (Eq k (c.index j))
      r : Fin (c.blocksFun k) → Fin n := ⇑(c.embedding k)
      ⊢ Not (Membership.mem (Set.range r) j)
    -/
    rwa [c.mem_range_embedding_iff']
    /-
      🎉 no goals
    -/


@[simp]
theorem compContinuousLinearMap_applyComposition {n : ℕ} (p : FormalMultilinearSeries 𝕜 F G)
    (f : E →L[𝕜] F) (c : Composition n) (v : Fin n → E) :
    (p.compContinuousLinearMap f).applyComposition c v = p.applyComposition c (f ∘ v) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝¹⁵ : CommRing 𝕜
    inst✝¹⁴ : AddCommGroup E
    inst✝¹³ : AddCommGroup F
    inst✝¹² : AddCommGroup G
    inst✝¹¹ : Module 𝕜 E
    inst✝¹⁰ : Module 𝕜 F
    inst✝⁹ : Module 𝕜 G
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : TopologicalSpace F
    inst✝⁶ : TopologicalSpace G
    inst✝⁵ : TopologicalAddGroup E
    inst✝⁴ : ContinuousConstSMul 𝕜 E
    inst✝³ : TopologicalAddGroup F
    inst✝² : ContinuousConstSMul 𝕜 F
    inst✝¹ : TopologicalAddGroup G
    inst✝ : ContinuousConstSMul 𝕜 G
    n : Nat
    p : FormalMultilinearSeries 𝕜 F G
    f : ContinuousLinearMap (RingHom.id 𝕜) E F
    c : Composition n
    v : Fin n → E
    ⊢ Eq ((p.compContinuousLinearMap f).applyComposition c v) (p.applyComposition  …
  -/
  simp (config := {unfoldPartialApp := true}) [applyComposition]; rfl
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


/-- Given a formal multilinear series `p`, a composition `c` of `n` and a continuous multilinear
map `f` in `c.length` variables, one may form a continuous multilinear map in `n` variables by
applying the right coefficient of `p` to each block of the composition, and then applying `f` to
the resulting vector. It is called `f.compAlongComposition p c`. -/
def compAlongComposition {n : ℕ} (p : FormalMultilinearSeries 𝕜 E F) (c : Composition n)
    (f : F [×c.length]→L[𝕜] G) : E [×n]→L[𝕜] G where
  toFun v := f (p.applyComposition c v)
  map_update_add' v i x y := by
    /-
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      H : Type u_5
      inst✝¹⁴ : CommRing 𝕜
      inst✝¹³ : AddCommGroup E
      inst✝¹² : AddCommGroup F
      inst✝¹¹ : AddCommGroup G
      inst✝¹⁰ : Module 𝕜 E
      inst✝⁹ : Module 𝕜 F
      inst✝⁸ : Module 𝕜 G
      inst✝⁷ : TopologicalSpace E
      inst✝⁶ : TopologicalSpace F
      inst✝⁵ : TopologicalSpace G
      inst✝⁴ : TopologicalAddGroup E
      inst✝³ : ContinuousConstSMul 𝕜 E
      inst✝² : TopologicalAddGroup F
      inst✝¹ : ContinuousConstSMul 𝕜 F
      n : Nat
      p : FormalMultilinearSeries 𝕜 E F
      c : Composition n
      f : ContinuousMultilinearMap 𝕜 (fun i => F) G
      inst✝ : DecidableEq (Fin n)
      v : Fin n → E
      i : Fin n
      x y : E
      ⊢ Eq ((fun v => f (p.applyComposition c v)) (Function.update v i (HAdd.hAdd x  …
    -/
    cases Subsingleton.elim ‹_› (instDecidableEqFin _)
    /-
      case refl
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      H : Type u_5
      inst✝¹³ : CommRing 𝕜
      inst✝¹² : AddCommGroup E
      inst✝¹¹ : AddCommGroup F
      inst✝¹⁰ : AddCommGroup G
      inst✝⁹ : Module 𝕜 E
      inst✝⁸ : Module 𝕜 F
      inst✝⁷ : Module 𝕜 G
      inst✝⁶ : TopologicalSpace E
      inst✝⁵ : TopologicalSpace F
      inst✝⁴ : TopologicalSpace G
      inst✝³ : TopologicalAddGroup E
      inst✝² : ContinuousConstSMul 𝕜 E
      inst✝¹ : TopologicalAddGroup F
      inst✝ : ContinuousConstSMul 𝕜 F
      n : Nat
      p : FormalMultilinearSeries 𝕜 E F
      c : Composition n
      f : ContinuousMultilinearMap 𝕜 (fun i => F) G
      v : Fin n → E
      i : Fin n
      x y : E
      ⊢ Eq ((fun v => f (p.applyComposition c v)) (Function.update v i (HAdd.hAdd x  …
    -/
    simp only [applyComposition_update, ContinuousMultilinearMap.map_update_add]
    /-
      🎉 no goals
    -/
  map_update_smul' v i c x := by
    /-
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      H : Type u_5
      inst✝¹⁴ : CommRing 𝕜
      inst✝¹³ : AddCommGroup E
      inst✝¹² : AddCommGroup F
      inst✝¹¹ : AddCommGroup G
      inst✝¹⁰ : Module 𝕜 E
      inst✝⁹ : Module 𝕜 F
      inst✝⁸ : Module 𝕜 G
      inst✝⁷ : TopologicalSpace E
      inst✝⁶ : TopologicalSpace F
      inst✝⁵ : TopologicalSpace G
      inst✝⁴ : TopologicalAddGroup E
      inst✝³ : ContinuousConstSMul 𝕜 E
      inst✝² : TopologicalAddGroup F
      inst✝¹ : ContinuousConstSMul 𝕜 F
      n : Nat
      p : FormalMultilinearSeries 𝕜 E F
      c✝ : Composition n
      f : ContinuousMultilinearMap 𝕜 (fun i => F) G
      inst✝ : DecidableEq (Fin n)
      v : Fin n → E
      i : Fin n
      c : 𝕜
      x : E
      ⊢ Eq ((fun v => f (p.applyComposition c✝ v)) (Function.update v i (HSMul.hSMul …
    -/
    cases Subsingleton.elim ‹_› (instDecidableEqFin _)
    /-
      case refl
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      H : Type u_5
      inst✝¹³ : CommRing 𝕜
      inst✝¹² : AddCommGroup E
      inst✝¹¹ : AddCommGroup F
      inst✝¹⁰ : AddCommGroup G
      inst✝⁹ : Module 𝕜 E
      inst✝⁸ : Module 𝕜 F
      inst✝⁷ : Module 𝕜 G
      inst✝⁶ : TopologicalSpace E
      inst✝⁵ : TopologicalSpace F
      inst✝⁴ : TopologicalSpace G
      inst✝³ : TopologicalAddGroup E
      inst✝² : ContinuousConstSMul 𝕜 E
      inst✝¹ : TopologicalAddGroup F
      inst✝ : ContinuousConstSMul 𝕜 F
      n : Nat
      p : FormalMultilinearSeries 𝕜 E F
      c✝ : Composition n
      f : ContinuousMultilinearMap 𝕜 (fun i => F) G
      v : Fin n → E
      i : Fin n
      c : 𝕜
      x : E
      ⊢ Eq ((fun v => f (p.applyComposition c✝ v)) (Function.update v i (HSMul.hSMul …
    -/
    simp only [applyComposition_update, ContinuousMultilinearMap.map_update_smul]
    /-
      🎉 no goals
    -/
  cont :=
    f.cont.comp <|
      continuous_pi fun _ => (coe_continuous _).comp <| continuous_pi fun _ => continuous_apply _


@[simp]
theorem compAlongComposition_apply {n : ℕ} (p : FormalMultilinearSeries 𝕜 E F) (c : Composition n)
    (f : F [×c.length]→L[𝕜] G) (v : Fin n → E) :
    (f.compAlongComposition p c) v = f (p.applyComposition c v) :=
  rfl


/-- Given two formal multilinear series `q` and `p` and a composition `c` of `n`, one may
form a continuous multilinear map in `n` variables by applying the right coefficient of `p` to each
block of the composition, and then applying `q c.length` to the resulting vector. It is
called `q.compAlongComposition p c`. -/
def compAlongComposition {n : ℕ} (q : FormalMultilinearSeries 𝕜 F G)
    (p : FormalMultilinearSeries 𝕜 E F) (c : Composition n) : (E [×n]→L[𝕜] G) :=
  (q c.length).compAlongComposition p c


@[simp]
theorem compAlongComposition_apply {n : ℕ} (q : FormalMultilinearSeries 𝕜 F G)
    (p : FormalMultilinearSeries 𝕜 E F) (c : Composition n) (v : Fin n → E) :
    (q.compAlongComposition p c) v = q c.length (p.applyComposition c v) :=
  rfl


/-- Formal composition of two formal multilinear series. The `n`-th coefficient in the composition
is defined to be the sum of `q.compAlongComposition p c` over all compositions of
`n`. In other words, this term (as a multilinear function applied to `v_0, ..., v_{n-1}`) is
`∑'_{k} ∑'_{i₁ + ... + iₖ = n} qₖ (p_{i_1} (...), ..., p_{i_k} (...))`, where one puts all variables
`v_0, ..., v_{n-1}` in increasing order in the dots.

In general, the composition `q ∘ p` only makes sense when the constant coefficient of `p` vanishes.
We give a general formula but which ignores the value of `p 0` instead.
-/
protected def comp (q : FormalMultilinearSeries 𝕜 F G) (p : FormalMultilinearSeries 𝕜 E F) :
    FormalMultilinearSeries 𝕜 E G := fun n => ∑ c : Composition n, q.compAlongComposition p c


/-- The `0`-th coefficient of `q.comp p` is `q 0`. Since these maps are multilinear maps in zero
variables, but on different spaces, we can not state this directly, so we state it when applied to
arbitrary vectors (which have to be the zero vector). -/
theorem comp_coeff_zero (q : FormalMultilinearSeries 𝕜 F G) (p : FormalMultilinearSeries 𝕜 E F)
    (v : Fin 0 → E) (v' : Fin 0 → F) : (q.comp p) 0 v = q 0 v' := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝¹⁵ : CommRing 𝕜
    inst✝¹⁴ : AddCommGroup E
    inst✝¹³ : AddCommGroup F
    inst✝¹² : AddCommGroup G
    inst✝¹¹ : Module 𝕜 E
    inst✝¹⁰ : Module 𝕜 F
    inst✝⁹ : Module 𝕜 G
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : TopologicalSpace F
    inst✝⁶ : TopologicalSpace G
    inst✝⁵ : TopologicalAddGroup E
    inst✝⁴ : ContinuousConstSMul 𝕜 E
    inst✝³ : TopologicalAddGroup F
    inst✝² : ContinuousConstSMul 𝕜 F
    inst✝¹ : TopologicalAddGroup G
    inst✝ : ContinuousConstSMul 𝕜 G
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    v : Fin 0 → E
    v' : Fin 0 → F
    ⊢ Eq ((q.comp p 0) v) ((q 0) v')
  -/
  let c : Composition 0 := Composition.ones 0
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝¹⁵ : CommRing 𝕜
    inst✝¹⁴ : AddCommGroup E
    inst✝¹³ : AddCommGroup F
    inst✝¹² : AddCommGroup G
    inst✝¹¹ : Module 𝕜 E
    inst✝¹⁰ : Module 𝕜 F
    inst✝⁹ : Module 𝕜 G
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : TopologicalSpace F
    inst✝⁶ : TopologicalSpace G
    inst✝⁵ : TopologicalAddGroup E
    inst✝⁴ : ContinuousConstSMul 𝕜 E
    inst✝³ : TopologicalAddGroup F
    inst✝² : ContinuousConstSMul 𝕜 F
    inst✝¹ : TopologicalAddGroup G
    inst✝ : ContinuousConstSMul 𝕜 G
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    v : Fin 0 → E
    v' : Fin 0 → F
    c : Composition 0 := Composition.ones 0
    ⊢ Eq ((q.comp p 0) v) ((q 0) v')
  -/
  dsimp [FormalMultilinearSeries.comp]
  have : {c} = (Finset.univ : Finset (Composition 0)) := by
    apply Finset.eq_of_subset_of_card_le <;> simp [Finset.card_univ, composition_card 0]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝¹⁵ : CommRing 𝕜
    inst✝¹⁴ : AddCommGroup E
    inst✝¹³ : AddCommGroup F
    inst✝¹² : AddCommGroup G
    inst✝¹¹ : Module 𝕜 E
    inst✝¹⁰ : Module 𝕜 F
    inst✝⁹ : Module 𝕜 G
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : TopologicalSpace F
    inst✝⁶ : TopologicalSpace G
    inst✝⁵ : TopologicalAddGroup E
    inst✝⁴ : ContinuousConstSMul 𝕜 E
    inst✝³ : TopologicalAddGroup F
    inst✝² : ContinuousConstSMul 𝕜 F
    inst✝¹ : TopologicalAddGroup G
    inst✝ : ContinuousConstSMul 𝕜 G
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    v : Fin 0 → E
    v' : Fin 0 → F
    c : Composition 0 := Composition.ones 0
    this : Eq (Singleton.singleton c) Finset.univ
    ⊢ Eq ((Finset.univ.sum fun c => q.compAlongComposition p c) v) ((q 0) v')
  -/
  rw [← this, Finset.sum_singleton, compAlongComposition_apply]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝¹⁵ : CommRing 𝕜
    inst✝¹⁴ : AddCommGroup E
    inst✝¹³ : AddCommGroup F
    inst✝¹² : AddCommGroup G
    inst✝¹¹ : Module 𝕜 E
    inst✝¹⁰ : Module 𝕜 F
    inst✝⁹ : Module 𝕜 G
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : TopologicalSpace F
    inst✝⁶ : TopologicalSpace G
    inst✝⁵ : TopologicalAddGroup E
    inst✝⁴ : ContinuousConstSMul 𝕜 E
    inst✝³ : TopologicalAddGroup F
    inst✝² : ContinuousConstSMul 𝕜 F
    inst✝¹ : TopologicalAddGroup G
    inst✝ : ContinuousConstSMul 𝕜 G
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    v : Fin 0 → E
    v' : Fin 0 → F
    c : Composition 0 := Composition.ones 0
    this : Eq (Singleton.singleton c) Finset.univ
    ⊢ Eq ((q c.length) (p.applyComposition c v)) ((q 0) v')
  -/
  symm; congr! -- Porting note: needed the stronger `congr!`!
        /-
          🎉 no goals
        -/


@[simp]
theorem comp_coeff_zero' (q : FormalMultilinearSeries 𝕜 F G) (p : FormalMultilinearSeries 𝕜 E F)
    (v : Fin 0 → E) : (q.comp p) 0 v = q 0 fun _i => 0 :=
  q.comp_coeff_zero p v _


/-- The `0`-th coefficient of `q.comp p` is `q 0`. When `p` goes from `E` to `E`, this can be
expressed as a direct equality -/
theorem comp_coeff_zero'' (q : FormalMultilinearSeries 𝕜 E F) (p : FormalMultilinearSeries 𝕜 E E) :
                             /-
                               𝕜 : Type u_1
                               E : Type u_2
                               F : Type u_3
                               inst✝¹⁰ : CommRing 𝕜
                               inst✝⁹ : AddCommGroup E
                               inst✝⁸ : AddCommGroup F
                               inst✝⁷ : Module 𝕜 E
                               inst✝⁶ : Module 𝕜 F
                               inst✝⁵ : TopologicalSpace E
                               inst✝⁴ : TopologicalSpace F
                               inst✝³ : TopologicalAddGroup E
                               inst✝² : ContinuousConstSMul 𝕜 E
                               inst✝¹ : TopologicalAddGroup F
                               inst✝ : ContinuousConstSMul 𝕜 F
                               q : FormalMultilinearSeries 𝕜 E F
                               p : FormalMultilinearSeries 𝕜 E E
                               ⊢ Eq (q.comp p 0) (q 0)
                             -/
    (q.comp p) 0 = q 0 := by ext v; exact q.comp_coeff_zero p _ _
                                    /-
                                      🎉 no goals
                                    -/


/-- The first coefficient of a composition of formal multilinear series is the composition of the
first coefficients seen as continuous linear maps. -/
theorem comp_coeff_one (q : FormalMultilinearSeries 𝕜 F G) (p : FormalMultilinearSeries 𝕜 E F)
    (v : Fin 1 → E) : (q.comp p) 1 v = q 1 fun _i => p 1 v := by
  have : {Composition.ones 1} = (Finset.univ : Finset (Composition 1)) :=
    Finset.eq_univ_of_card _ (by simp [composition_card])
  simp only [FormalMultilinearSeries.comp, compAlongComposition_apply, ← this,
    Finset.sum_singleton]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝¹⁵ : CommRing 𝕜
    inst✝¹⁴ : AddCommGroup E
    inst✝¹³ : AddCommGroup F
    inst✝¹² : AddCommGroup G
    inst✝¹¹ : Module 𝕜 E
    inst✝¹⁰ : Module 𝕜 F
    inst✝⁹ : Module 𝕜 G
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : TopologicalSpace F
    inst✝⁶ : TopologicalSpace G
    inst✝⁵ : TopologicalAddGroup E
    inst✝⁴ : ContinuousConstSMul 𝕜 E
    inst✝³ : TopologicalAddGroup F
    inst✝² : ContinuousConstSMul 𝕜 F
    inst✝¹ : TopologicalAddGroup G
    inst✝ : ContinuousConstSMul 𝕜 G
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    v : Fin 1 → E
    this : Eq (Singleton.singleton (Composition.ones 1)) Finset.univ
    ⊢ Eq ((q (Composition.ones 1).length) (p.applyComposition (Composition.ones 1) …
  -/
  refine q.congr (by simp) fun i hi1 hi2 => ?_
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝¹⁵ : CommRing 𝕜
    inst✝¹⁴ : AddCommGroup E
    inst✝¹³ : AddCommGroup F
    inst✝¹² : AddCommGroup G
    inst✝¹¹ : Module 𝕜 E
    inst✝¹⁰ : Module 𝕜 F
    inst✝⁹ : Module 𝕜 G
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : TopologicalSpace F
    inst✝⁶ : TopologicalSpace G
    inst✝⁵ : TopologicalAddGroup E
    inst✝⁴ : ContinuousConstSMul 𝕜 E
    inst✝³ : TopologicalAddGroup F
    inst✝² : ContinuousConstSMul 𝕜 F
    inst✝¹ : TopologicalAddGroup G
    inst✝ : ContinuousConstSMul 𝕜 G
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    v : Fin 1 → E
    this : Eq (Singleton.singleton (Composition.ones 1)) Finset.univ
    i : Nat
    hi1 : LT.lt i (Composition.ones 1).length
    hi2 : LT.lt i 1
    ⊢ Eq (p.applyComposition (Composition.ones 1) v ⟨i, hi1⟩) ((p 1) v)
  -/
  simp only [applyComposition_ones]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝¹⁵ : CommRing 𝕜
    inst✝¹⁴ : AddCommGroup E
    inst✝¹³ : AddCommGroup F
    inst✝¹² : AddCommGroup G
    inst✝¹¹ : Module 𝕜 E
    inst✝¹⁰ : Module 𝕜 F
    inst✝⁹ : Module 𝕜 G
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : TopologicalSpace F
    inst✝⁶ : TopologicalSpace G
    inst✝⁵ : TopologicalAddGroup E
    inst✝⁴ : ContinuousConstSMul 𝕜 E
    inst✝³ : TopologicalAddGroup F
    inst✝² : ContinuousConstSMul 𝕜 F
    inst✝¹ : TopologicalAddGroup G
    inst✝ : ContinuousConstSMul 𝕜 G
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    v : Fin 1 → E
    this : Eq (Singleton.singleton (Composition.ones 1)) Finset.univ
    i : Nat
    hi1 : LT.lt i (Composition.ones 1).length
    hi2 : LT.lt i 1
    ⊢ Eq ((p 1) fun x => v (Fin.castLE ⋯ ⟨i, hi1⟩)) ((p 1) v)
  -/
  exact p.congr rfl fun j _hj1 hj2 => by congr! -- Porting note: needed the stronger `congr!`
  /-
    🎉 no goals
  -/


/-- Only `0`-th coefficient of `q.comp p` depends on `q 0`. -/
theorem removeZero_comp_of_pos (q : FormalMultilinearSeries 𝕜 F G)
    (p : FormalMultilinearSeries 𝕜 E F) {n : ℕ} (hn : 0 < n) :
    q.removeZero.comp p n = q.comp p n := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝¹⁵ : CommRing 𝕜
    inst✝¹⁴ : AddCommGroup E
    inst✝¹³ : AddCommGroup F
    inst✝¹² : AddCommGroup G
    inst✝¹¹ : Module 𝕜 E
    inst✝¹⁰ : Module 𝕜 F
    inst✝⁹ : Module 𝕜 G
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : TopologicalSpace F
    inst✝⁶ : TopologicalSpace G
    inst✝⁵ : TopologicalAddGroup E
    inst✝⁴ : ContinuousConstSMul 𝕜 E
    inst✝³ : TopologicalAddGroup F
    inst✝² : ContinuousConstSMul 𝕜 F
    inst✝¹ : TopologicalAddGroup G
    inst✝ : ContinuousConstSMul 𝕜 G
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    hn : LT.lt 0 n
    ⊢ Eq (q.removeZero.comp p n) (q.comp p n)
  -/
  ext v
  simp only [FormalMultilinearSeries.comp, compAlongComposition,
    ContinuousMultilinearMap.compAlongComposition_apply, ContinuousMultilinearMap.sum_apply]
  /-
    case H
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝¹⁵ : CommRing 𝕜
    inst✝¹⁴ : AddCommGroup E
    inst✝¹³ : AddCommGroup F
    inst✝¹² : AddCommGroup G
    inst✝¹¹ : Module 𝕜 E
    inst✝¹⁰ : Module 𝕜 F
    inst✝⁹ : Module 𝕜 G
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : TopologicalSpace F
    inst✝⁶ : TopologicalSpace G
    inst✝⁵ : TopologicalAddGroup E
    inst✝⁴ : ContinuousConstSMul 𝕜 E
    inst✝³ : TopologicalAddGroup F
    inst✝² : ContinuousConstSMul 𝕜 F
    inst✝¹ : TopologicalAddGroup G
    inst✝ : ContinuousConstSMul 𝕜 G
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    hn : LT.lt 0 n
    v : Fin n → E
    ⊢ Eq (Finset.univ.sum fun x => (q.removeZero x.length) (p.applyComposition x v …
  -/
  refine Finset.sum_congr rfl fun c _hc => ?_
  /-
    case H
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝¹⁵ : CommRing 𝕜
    inst✝¹⁴ : AddCommGroup E
    inst✝¹³ : AddCommGroup F
    inst✝¹² : AddCommGroup G
    inst✝¹¹ : Module 𝕜 E
    inst✝¹⁰ : Module 𝕜 F
    inst✝⁹ : Module 𝕜 G
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : TopologicalSpace F
    inst✝⁶ : TopologicalSpace G
    inst✝⁵ : TopologicalAddGroup E
    inst✝⁴ : ContinuousConstSMul 𝕜 E
    inst✝³ : TopologicalAddGroup F
    inst✝² : ContinuousConstSMul 𝕜 F
    inst✝¹ : TopologicalAddGroup G
    inst✝ : ContinuousConstSMul 𝕜 G
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    hn : LT.lt 0 n
    v : Fin n → E
    c : Composition n
    _hc : Membership.mem Finset.univ c
    ⊢ Eq ((q.removeZero c.length) (p.applyComposition c v)) ((q c.length) (p.apply …
  -/
  rw [removeZero_of_pos _ (c.length_pos_of_pos hn)]
  /-
    🎉 no goals
  -/


@[simp]
theorem comp_removeZero (q : FormalMultilinearSeries 𝕜 F G) (p : FormalMultilinearSeries 𝕜 E F) :
                                         /-
                                           𝕜 : Type u_1
                                           E : Type u_2
                                           F : Type u_3
                                           G : Type u_4
                                           inst✝¹⁵ : CommRing 𝕜
                                           inst✝¹⁴ : AddCommGroup E
                                           inst✝¹³ : AddCommGroup F
                                           inst✝¹² : AddCommGroup G
                                           inst✝¹¹ : Module 𝕜 E
                                           inst✝¹⁰ : Module 𝕜 F
                                           inst✝⁹ : Module 𝕜 G
                                           inst✝⁸ : TopologicalSpace E
                                           inst✝⁷ : TopologicalSpace F
                                           inst✝⁶ : TopologicalSpace G
                                           inst✝⁵ : TopologicalAddGroup E
                                           inst✝⁴ : ContinuousConstSMul 𝕜 E
                                           inst✝³ : TopologicalAddGroup F
                                           inst✝² : ContinuousConstSMul 𝕜 F
                                           inst✝¹ : TopologicalAddGroup G
                                           inst✝ : ContinuousConstSMul 𝕜 G
                                           q : FormalMultilinearSeries 𝕜 F G
                                           p : FormalMultilinearSeries 𝕜 E F
                                           ⊢ Eq (q.comp p.removeZero) (q.comp p)
                                         -/
    q.comp p.removeZero = q.comp p := by ext n; simp [FormalMultilinearSeries.comp]
                                                /-
                                                  🎉 no goals
                                                -/


/-- The norm of `f.compAlongComposition p c` is controlled by the product of
the norms of the relevant bits of `f` and `p`. -/
theorem compAlongComposition_bound {n : ℕ} (p : FormalMultilinearSeries 𝕜 E F) (c : Composition n)
    (f : F [×c.length]→L[𝕜] G) (v : Fin n → E) :
    ‖f.compAlongComposition p c v‖ ≤ (‖f‖ * ∏ i, ‖p (c.blocksFun i)‖) * ∏ i : Fin n, ‖v i‖ :=
  calc
    ‖f.compAlongComposition p c v‖ = ‖f (p.applyComposition c v)‖ := rfl
    _ ≤ ‖f‖ * ∏ i, ‖p.applyComposition c v i‖ := ContinuousMultilinearMap.le_opNorm _ _
    _ ≤ ‖f‖ * ∏ i, ‖p (c.blocksFun i)‖ * ∏ j : Fin (c.blocksFun i), ‖(v ∘ c.embedding i) j‖ := by
      /-
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        G : Type u_4
        inst✝⁶ : NontriviallyNormedField 𝕜
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        n : Nat
        p : FormalMultilinearSeries 𝕜 E F
        c : Composition n
        f : ContinuousMultilinearMap 𝕜 (fun i => F) G
        v : Fin n → E
        ⊢ LE.le (HMul.hMul (Norm.norm f) (Finset.univ.prod fun i => Norm.norm (p.apply …
      -/
      apply mul_le_mul_of_nonneg_left _ (norm_nonneg _)
      /-
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        G : Type u_4
        inst✝⁶ : NontriviallyNormedField 𝕜
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        n : Nat
        p : FormalMultilinearSeries 𝕜 E F
        c : Composition n
        f : ContinuousMultilinearMap 𝕜 (fun i => F) G
        v : Fin n → E
        ⊢ LE.le (Finset.univ.prod fun i => Norm.norm (p.applyComposition c v i)) (Fins …
      -/
      refine Finset.prod_le_prod (fun i _hi => norm_nonneg _) fun i _hi => ?_
      /-
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        G : Type u_4
        inst✝⁶ : NontriviallyNormedField 𝕜
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        n : Nat
        p : FormalMultilinearSeries 𝕜 E F
        c : Composition n
        f : ContinuousMultilinearMap 𝕜 (fun i => F) G
        v : Fin n → E
        i : Fin c.length
        _hi : Membership.mem Finset.univ i
        ⊢ LE.le (Norm.norm (p.applyComposition c v i)) (HMul.hMul (Norm.norm (p (c.blo …
      -/
      apply ContinuousMultilinearMap.le_opNorm
      /-
        🎉 no goals
      -/
    _ = (‖f‖ * ∏ i, ‖p (c.blocksFun i)‖) *
        ∏ i, ∏ j : Fin (c.blocksFun i), ‖(v ∘ c.embedding i) j‖ := by
      /-
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        G : Type u_4
        inst✝⁶ : NontriviallyNormedField 𝕜
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        n : Nat
        p : FormalMultilinearSeries 𝕜 E F
        c : Composition n
        f : ContinuousMultilinearMap 𝕜 (fun i => F) G
        v : Fin n → E
        ⊢ Eq (HMul.hMul (Norm.norm f) (Finset.univ.prod fun i => HMul.hMul (Norm.norm  …
      -/
      rw [Finset.prod_mul_distrib, mul_assoc]
      /-
        🎉 no goals
      -/
    _ = (‖f‖ * ∏ i, ‖p (c.blocksFun i)‖) * ∏ i : Fin n, ‖v i‖ := by
      /-
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        G : Type u_4
        inst✝⁶ : NontriviallyNormedField 𝕜
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        n : Nat
        p : FormalMultilinearSeries 𝕜 E F
        c : Composition n
        f : ContinuousMultilinearMap 𝕜 (fun i => F) G
        v : Fin n → E
        ⊢ Eq (HMul.hMul (HMul.hMul (Norm.norm f) (Finset.univ.prod fun i => Norm.norm  …
      -/
      rw [← c.blocksFinEquiv.prod_comp, ← Finset.univ_sigma_univ, Finset.prod_sigma]
      /-
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        G : Type u_4
        inst✝⁶ : NontriviallyNormedField 𝕜
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        n : Nat
        p : FormalMultilinearSeries 𝕜 E F
        c : Composition n
        f : ContinuousMultilinearMap 𝕜 (fun i => F) G
        v : Fin n → E
        ⊢ Eq (HMul.hMul (HMul.hMul (Norm.norm f) (Finset.univ.prod fun i => Norm.norm  …
      -/
      congr
      /-
        🎉 no goals
      -/


/-- The norm of `q.compAlongComposition p c` is controlled by the product of
the norms of the relevant bits of `q` and `p`. -/
theorem compAlongComposition_norm {n : ℕ} (q : FormalMultilinearSeries 𝕜 F G)
    (p : FormalMultilinearSeries 𝕜 E F) (c : Composition n) :
    ‖q.compAlongComposition p c‖ ≤ ‖q c.length‖ * ∏ i, ‖p (c.blocksFun i)‖ :=
                                               /-
                                                 𝕜 : Type u_1
                                                 E : Type u_2
                                                 F : Type u_3
                                                 G : Type u_4
                                                 inst✝⁶ : NontriviallyNormedField 𝕜
                                                 inst✝⁵ : NormedAddCommGroup E
                                                 inst✝⁴ : NormedSpace 𝕜 E
                                                 inst✝³ : NormedAddCommGroup F
                                                 inst✝² : NormedSpace 𝕜 F
                                                 inst✝¹ : NormedAddCommGroup G
                                                 inst✝ : NormedSpace 𝕜 G
                                                 n : Nat
                                                 q : FormalMultilinearSeries 𝕜 F G
                                                 p : FormalMultilinearSeries 𝕜 E F
                                                 c : Composition n
                                                 ⊢ LE.le 0 (HMul.hMul (Norm.norm (q c.length)) (Finset.univ.prod fun i => Norm. …
                                               -/
  ContinuousMultilinearMap.opNorm_le_bound (by positivity) (compAlongComposition_bound _ _ _)
                                               /-
                                                 🎉 no goals
                                               -/


theorem compAlongComposition_nnnorm {n : ℕ} (q : FormalMultilinearSeries 𝕜 F G)
    (p : FormalMultilinearSeries 𝕜 E F) (c : Composition n) :
    ‖q.compAlongComposition p c‖₊ ≤ ‖q c.length‖₊ * ∏ i, ‖p (c.blocksFun i)‖₊ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    n : Nat
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    c : Composition n
    ⊢ LE.le (NNNorm.nnnorm (q.compAlongComposition p c)) (HMul.hMul (NNNorm.nnnorm …
  -/
  rw [← NNReal.coe_le_coe]; push_cast; exact q.compAlongComposition_norm p c
                                       /-
                                         🎉 no goals
                                       -/


/-- The identity formal multilinear series, with all coefficients equal to `0` except for `n = 1`
where it is (the continuous multilinear version of) the identity. We allow an arbitrary
constant coefficient `x`. -/
def id (x : E) : FormalMultilinearSeries 𝕜 E E
  | 0 => ContinuousMultilinearMap.uncurry0 𝕜 _ x
  | 1 => (continuousMultilinearCurryFin1 𝕜 E E).symm (ContinuousLinearMap.id 𝕜 E)
  | _ => 0


@[simp] theorem id_apply_zero (x : E) (v : Fin 0 → E) :
    (FormalMultilinearSeries.id 𝕜 E x) 0 v = x := rfl


/-- The first coefficient of `id 𝕜 E` is the identity. -/
@[simp]
theorem id_apply_one (x : E) (v : Fin 1 → E) : (FormalMultilinearSeries.id 𝕜 E x) 1 v = v 0 :=
  rfl


/-- The `n`th coefficient of `id 𝕜 E` is the identity when `n = 1`. We state this in a dependent
way, as it will often appear in this form. -/
theorem id_apply_one' (x : E) {n : ℕ} (h : n = 1) (v : Fin n → E) :
    (id 𝕜 E x) n v = v ⟨0, h.symm ▸ zero_lt_one⟩ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    n : Nat
    h : Eq n 1
    v : Fin n → E
    ⊢ Eq ((FormalMultilinearSeries.id 𝕜 E x n) v) (v ⟨0, ⋯⟩)
  -/
  subst n
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    v : Fin 1 → E
    ⊢ Eq ((FormalMultilinearSeries.id 𝕜 E x 1) v) (v ⟨0, ⋯⟩)
  -/
  apply id_apply_one
  /-
    🎉 no goals
  -/


/-- For `n ≠ 1`, the `n`-th coefficient of `id 𝕜 E` is zero, by definition. -/
@[simp]
theorem id_apply_of_one_lt (x : E) {n : ℕ} (h : 1 < n) :
    (FormalMultilinearSeries.id 𝕜 E x) n = 0 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    n : Nat
    h : LT.lt 1 n
    ⊢ Eq (FormalMultilinearSeries.id 𝕜 E x n) 0
  -/
  cases' n with n
    /-
      case zero
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      x : E
      h : LT.lt 1 0
      ⊢ Eq (FormalMultilinearSeries.id 𝕜 E x 0) 0
    -/
  · contradiction
    /-
      🎉 no goals
    -/
    /-
      case succ
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      x : E
      n : Nat
      h : LT.lt 1 (HAdd.hAdd n 1)
      ⊢ Eq (FormalMultilinearSeries.id 𝕜 E x (HAdd.hAdd n 1)) 0
    -/
  · cases n
      /-
        case succ.zero
        𝕜 : Type u_1
        E : Type u_2
        inst✝² : NontriviallyNormedField 𝕜
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace 𝕜 E
        x : E
        h : LT.lt 1 (HAdd.hAdd 0 1)
        ⊢ Eq (FormalMultilinearSeries.id 𝕜 E x (HAdd.hAdd 0 1)) 0
      -/
    · contradiction
      /-
        🎉 no goals
      -/
      /-
        case succ.succ
        𝕜 : Type u_1
        E : Type u_2
        inst✝² : NontriviallyNormedField 𝕜
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace 𝕜 E
        x : E
        n✝ : Nat
        h : LT.lt 1 (HAdd.hAdd (HAdd.hAdd n✝ 1) 1)
        ⊢ Eq (FormalMultilinearSeries.id 𝕜 E x (HAdd.hAdd (HAdd.hAdd n✝ 1) 1)) 0
      -/
    · rfl
      /-
        🎉 no goals
      -/


@[simp]
theorem comp_id (p : FormalMultilinearSeries 𝕜 E F) (x : E) : p.comp (id 𝕜 E x) = p := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    x : E
    ⊢ Eq (p.comp (FormalMultilinearSeries.id 𝕜 E x)) p
  -/
  ext1 n
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    x : E
    n : Nat
    ⊢ Eq (p.comp (FormalMultilinearSeries.id 𝕜 E x) n) (p n)
  -/
  dsimp [FormalMultilinearSeries.comp]
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    x : E
    n : Nat
    ⊢ Eq (Finset.univ.sum fun c => p.compAlongComposition (FormalMultilinearSeries …
  -/
  rw [Finset.sum_eq_single (Composition.ones n)]
    /-
      case h
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      x : E
      n : Nat
      ⊢ Eq (p.compAlongComposition (FormalMultilinearSeries.id 𝕜 E x) (Composition.o …
    -/
  · show compAlongComposition p (id 𝕜 E x) (Composition.ones n) = p n
    /-
      case h
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      x : E
      n : Nat
      ⊢ Eq (p.compAlongComposition (FormalMultilinearSeries.id 𝕜 E x) (Composition.o …
    -/
    ext v
    /-
      case h.H
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      x : E
      n : Nat
      v : Fin n → E
      ⊢ Eq ((p.compAlongComposition (FormalMultilinearSeries.id 𝕜 E x) (Composition. …
    -/
    rw [compAlongComposition_apply]
    /-
      case h.H
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      x : E
      n : Nat
      v : Fin n → E
      ⊢ Eq ((p (Composition.ones n).length) ((FormalMultilinearSeries.id 𝕜 E x).appl …
    -/
    apply p.congr (Composition.ones_length n)
    /-
      case h.H
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      x : E
      n : Nat
      v : Fin n → E
      ⊢ ∀ (i : Nat) (him : LT.lt i (Composition.ones n).length) (hin : LT.lt i n), E …
    -/
    intros
    /-
      case h.H
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      x : E
      n : Nat
      v : Fin n → E
      i✝ : Nat
      him✝ : LT.lt i✝ (Composition.ones n).length
      hin✝ : LT.lt i✝ n
      ⊢ Eq ((FormalMultilinearSeries.id 𝕜 E x).applyComposition (Composition.ones n) …
    -/
    rw [applyComposition_ones]
    /-
      case h.H
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      x : E
      n : Nat
      v : Fin n → E
      i✝ : Nat
      him✝ : LT.lt i✝ (Composition.ones n).length
      hin✝ : LT.lt i✝ n
      ⊢ Eq ((fun v i => (FormalMultilinearSeries.id 𝕜 E x 1) fun x => v (Fin.castLE  …
    -/
    refine congr_arg v ?_
    /-
      case h.H
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      x : E
      n : Nat
      v : Fin n → E
      i✝ : Nat
      him✝ : LT.lt i✝ (Composition.ones n).length
      hin✝ : LT.lt i✝ n
      ⊢ Eq (Fin.castLE ⋯ ⟨i✝, him✝⟩) ⟨i✝, hin✝⟩
    -/
    rw [Fin.ext_iff, Fin.coe_castLE, Fin.val_mk]
    /-
      🎉 no goals
    -/
  · show
    ∀ b : Composition n,
      b ∈ Finset.univ → b ≠ Composition.ones n → compAlongComposition p (id 𝕜 E x) b = 0
    /-
      case h.h₀
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      x : E
      n : Nat
      ⊢ ∀ (b : Composition n), Membership.mem Finset.univ b → Ne b (Composition.ones …
    -/
    intro b _ hb
    obtain ⟨k, hk, lt_k⟩ : ∃ (k : ℕ), k ∈ Composition.blocks b ∧ 1 < k :=
      Composition.ne_ones_iff.1 hb
    obtain ⟨i, hi⟩ : ∃ (i : Fin b.blocks.length), b.blocks[i] = k :=
      List.get_of_mem hk
    /-
      case h.h₀.intro.intro.intro
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      x : E
      n : Nat
      b : Composition n
      a✝ : Membership.mem Finset.univ b
      hb : Ne b (Composition.ones n)
      k : Nat
      hk : Membership.mem b.blocks k
      lt_k : LT.lt 1 k
      i : Fin b.blocks.length
      hi : Eq (GetElem.getElem b.blocks i ⋯) k
      ⊢ Eq (p.compAlongComposition (FormalMultilinearSeries.id 𝕜 E x) b) 0
    -/
    let j : Fin b.length := ⟨i.val, b.blocks_length ▸ i.prop⟩
    /-
      case h.h₀.intro.intro.intro
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      x : E
      n : Nat
      b : Composition n
      a✝ : Membership.mem Finset.univ b
      hb : Ne b (Composition.ones n)
      k : Nat
      hk : Membership.mem b.blocks k
      lt_k : LT.lt 1 k
      i : Fin b.blocks.length
      hi : Eq (GetElem.getElem b.blocks i ⋯) k
      j : Fin b.length := ⟨↑i, ⋯⟩
      ⊢ Eq (p.compAlongComposition (FormalMultilinearSeries.id 𝕜 E x) b) 0
    -/
    have A : 1 < b.blocksFun j := by convert lt_k
    /-
      case h.h₀.intro.intro.intro
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      x : E
      n : Nat
      b : Composition n
      a✝ : Membership.mem Finset.univ b
      hb : Ne b (Composition.ones n)
      k : Nat
      hk : Membership.mem b.blocks k
      lt_k : LT.lt 1 k
      i : Fin b.blocks.length
      hi : Eq (GetElem.getElem b.blocks i ⋯) k
      j : Fin b.length := ⟨↑i, ⋯⟩
      A : LT.lt 1 (b.blocksFun j)
      ⊢ Eq (p.compAlongComposition (FormalMultilinearSeries.id 𝕜 E x) b) 0
    -/
    ext v
    /-
      case h.h₀.intro.intro.intro.H
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      x : E
      n : Nat
      b : Composition n
      a✝ : Membership.mem Finset.univ b
      hb : Ne b (Composition.ones n)
      k : Nat
      hk : Membership.mem b.blocks k
      lt_k : LT.lt 1 k
      i : Fin b.blocks.length
      hi : Eq (GetElem.getElem b.blocks i ⋯) k
      j : Fin b.length := ⟨↑i, ⋯⟩
      A : LT.lt 1 (b.blocksFun j)
      v : Fin n → E
      ⊢ Eq ((p.compAlongComposition (FormalMultilinearSeries.id 𝕜 E x) b) v) (0 v)
    -/
    rw [compAlongComposition_apply, ContinuousMultilinearMap.zero_apply]
    /-
      case h.h₀.intro.intro.intro.H
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      x : E
      n : Nat
      b : Composition n
      a✝ : Membership.mem Finset.univ b
      hb : Ne b (Composition.ones n)
      k : Nat
      hk : Membership.mem b.blocks k
      lt_k : LT.lt 1 k
      i : Fin b.blocks.length
      hi : Eq (GetElem.getElem b.blocks i ⋯) k
      j : Fin b.length := ⟨↑i, ⋯⟩
      A : LT.lt 1 (b.blocksFun j)
      v : Fin n → E
      ⊢ Eq ((p b.length) ((FormalMultilinearSeries.id 𝕜 E x).applyComposition b v)) 0
    -/
    apply ContinuousMultilinearMap.map_coord_zero _ j
    /-
      case h.h₀.intro.intro.intro.H
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      x : E
      n : Nat
      b : Composition n
      a✝ : Membership.mem Finset.univ b
      hb : Ne b (Composition.ones n)
      k : Nat
      hk : Membership.mem b.blocks k
      lt_k : LT.lt 1 k
      i : Fin b.blocks.length
      hi : Eq (GetElem.getElem b.blocks i ⋯) k
      j : Fin b.length := ⟨↑i, ⋯⟩
      A : LT.lt 1 (b.blocksFun j)
      v : Fin n → E
      ⊢ Eq ((FormalMultilinearSeries.id 𝕜 E x).applyComposition b v j) 0
    -/
    dsimp [applyComposition]
    /-
      case h.h₀.intro.intro.intro.H
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      x : E
      n : Nat
      b : Composition n
      a✝ : Membership.mem Finset.univ b
      hb : Ne b (Composition.ones n)
      k : Nat
      hk : Membership.mem b.blocks k
      lt_k : LT.lt 1 k
      i : Fin b.blocks.length
      hi : Eq (GetElem.getElem b.blocks i ⋯) k
      j : Fin b.length := ⟨↑i, ⋯⟩
      A : LT.lt 1 (b.blocksFun j)
      v : Fin n → E
      ⊢ Eq ((FormalMultilinearSeries.id 𝕜 E x (b.blocksFun j)) (Function.comp v ⇑(b. …
    -/
    rw [id_apply_of_one_lt _ _ _ A]
    /-
      case h.h₀.intro.intro.intro.H
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      x : E
      n : Nat
      b : Composition n
      a✝ : Membership.mem Finset.univ b
      hb : Ne b (Composition.ones n)
      k : Nat
      hk : Membership.mem b.blocks k
      lt_k : LT.lt 1 k
      i : Fin b.blocks.length
      hi : Eq (GetElem.getElem b.blocks i ⋯) k
      j : Fin b.length := ⟨↑i, ⋯⟩
      A : LT.lt 1 (b.blocksFun j)
      v : Fin n → E
      ⊢ Eq (0 (Function.comp v ⇑(b.embedding j))) 0
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case h.h₁
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      x : E
      n : Nat
      ⊢ Not (Membership.mem Finset.univ (Composition.ones n)) → Eq (p.compAlongCompo …
    -/
  · simp
    /-
      🎉 no goals
    -/


@[simp]
theorem id_comp (p : FormalMultilinearSeries 𝕜 E F) (v0 : Fin 0 → E) :
    (id 𝕜 F (p 0 v0)).comp p = p := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    v0 : Fin 0 → E
    ⊢ Eq ((FormalMultilinearSeries.id 𝕜 F ((p 0) v0)).comp p) p
  -/
  ext1 n
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    v0 : Fin 0 → E
    n : Nat
    ⊢ Eq ((FormalMultilinearSeries.id 𝕜 F ((p 0) v0)).comp p n) (p n)
  -/
  by_cases hn : n = 0
    /-
      case pos
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      v0 : Fin 0 → E
      n : Nat
      hn : Eq n 0
      ⊢ Eq ((FormalMultilinearSeries.id 𝕜 F ((p 0) v0)).comp p n) (p n)
    -/
  · rw [hn]
    /-
      case pos
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      v0 : Fin 0 → E
      n : Nat
      hn : Eq n 0
      ⊢ Eq ((FormalMultilinearSeries.id 𝕜 F ((p 0) v0)).comp p 0) (p 0)
    -/
    ext v
    /-
      case pos.H
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      v0 : Fin 0 → E
      n : Nat
      hn : Eq n 0
      v : Fin 0 → E
      ⊢ Eq (((FormalMultilinearSeries.id 𝕜 F ((p 0) v0)).comp p 0) v) ((p 0) v)
    -/
    simp only [comp_coeff_zero', id_apply_zero]
    /-
      case pos.H
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      v0 : Fin 0 → E
      n : Nat
      hn : Eq n 0
      v : Fin 0 → E
      ⊢ Eq ((p 0) v0) ((p 0) v)
    -/
    congr with i
    /-
      case pos.H.h.e_6.h.h
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      v0 : Fin 0 → E
      n : Nat
      hn : Eq n 0
      v : Fin 0 → E
      i : Fin 0
      ⊢ Eq (v0 i) (v i)
    -/
    exact i.elim0
    /-
      🎉 no goals
    -/
    /-
      case neg
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      v0 : Fin 0 → E
      n : Nat
      hn : Not (Eq n 0)
      ⊢ Eq ((FormalMultilinearSeries.id 𝕜 F ((p 0) v0)).comp p n) (p n)
    -/
  · dsimp [FormalMultilinearSeries.comp]
    /-
      case neg
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      v0 : Fin 0 → E
      n : Nat
      hn : Not (Eq n 0)
      ⊢ Eq (Finset.univ.sum fun c => (FormalMultilinearSeries.id 𝕜 F ((p 0) v0)).com …
    -/
    have n_pos : 0 < n := bot_lt_iff_ne_bot.mpr hn
    /-
      case neg
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      v0 : Fin 0 → E
      n : Nat
      hn : Not (Eq n 0)
      n_pos : LT.lt 0 n
      ⊢ Eq (Finset.univ.sum fun c => (FormalMultilinearSeries.id 𝕜 F ((p 0) v0)).com …
    -/
    rw [Finset.sum_eq_single (Composition.single n n_pos)]
      /-
        case neg
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝⁴ : NontriviallyNormedField 𝕜
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        p : FormalMultilinearSeries 𝕜 E F
        v0 : Fin 0 → E
        n : Nat
        hn : Not (Eq n 0)
        n_pos : LT.lt 0 n
        ⊢ Eq ((FormalMultilinearSeries.id 𝕜 F ((p 0) v0)).compAlongComposition p (Comp …
      -/
    · show compAlongComposition (id 𝕜 F (p 0 v0)) p (Composition.single n n_pos) = p n
      /-
        case neg
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝⁴ : NontriviallyNormedField 𝕜
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        p : FormalMultilinearSeries 𝕜 E F
        v0 : Fin 0 → E
        n : Nat
        hn : Not (Eq n 0)
        n_pos : LT.lt 0 n
        ⊢ Eq ((FormalMultilinearSeries.id 𝕜 F ((p 0) v0)).compAlongComposition p (Comp …
      -/
      ext v
      /-
        case neg.H
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝⁴ : NontriviallyNormedField 𝕜
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        p : FormalMultilinearSeries 𝕜 E F
        v0 : Fin 0 → E
        n : Nat
        hn : Not (Eq n 0)
        n_pos : LT.lt 0 n
        v : Fin n → E
        ⊢ Eq (((FormalMultilinearSeries.id 𝕜 F ((p 0) v0)).compAlongComposition p (Com …
      -/
      rw [compAlongComposition_apply, id_apply_one' _ _ _ (Composition.single_length n_pos)]
      /-
        case neg.H
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝⁴ : NontriviallyNormedField 𝕜
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        p : FormalMultilinearSeries 𝕜 E F
        v0 : Fin 0 → E
        n : Nat
        hn : Not (Eq n 0)
        n_pos : LT.lt 0 n
        v : Fin n → E
        ⊢ Eq (p.applyComposition (Composition.single n n_pos) v ⟨0, ⋯⟩) ((p n) v)
      -/
      dsimp [applyComposition]
      /-
        case neg.H
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝⁴ : NontriviallyNormedField 𝕜
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        p : FormalMultilinearSeries 𝕜 E F
        v0 : Fin 0 → E
        n : Nat
        hn : Not (Eq n 0)
        n_pos : LT.lt 0 n
        v : Fin n → E
        ⊢ Eq ((p ((Composition.single n n_pos).blocksFun 0)) (Function.comp v ⇑((Compo …
      -/
      refine p.congr rfl fun i him hin => congr_arg v <| ?_
      /-
        case neg.H
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝⁴ : NontriviallyNormedField 𝕜
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        p : FormalMultilinearSeries 𝕜 E F
        v0 : Fin 0 → E
        n : Nat
        hn : Not (Eq n 0)
        n_pos : LT.lt 0 n
        v : Fin n → E
        i : Nat
        him : LT.lt i ((Composition.single n n_pos).blocksFun 0)
        hin : LT.lt i n
        ⊢ Eq (((Composition.single n n_pos).embedding 0) ⟨i, him⟩) ⟨i, hin⟩
      -/
      ext; simp
           /-
             🎉 no goals
           -/
    · show
      ∀ b : Composition n, b ∈ Finset.univ → b ≠ Composition.single n n_pos →
        compAlongComposition (id 𝕜 F (p 0 v0)) p b = 0
      /-
        case neg.h₀
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝⁴ : NontriviallyNormedField 𝕜
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        p : FormalMultilinearSeries 𝕜 E F
        v0 : Fin 0 → E
        n : Nat
        hn : Not (Eq n 0)
        n_pos : LT.lt 0 n
        ⊢ ∀ (b : Composition n), Membership.mem Finset.univ b → Ne b (Composition.sing …
      -/
      intro b _ hb
      have A : 1 < b.length := by
        have : b.length ≠ 1 := by simpa [Composition.eq_single_iff_length] using hb
        have : 0 < b.length := Composition.length_pos_of_pos b n_pos
        omega
      /-
        case neg.h₀
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝⁴ : NontriviallyNormedField 𝕜
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        p : FormalMultilinearSeries 𝕜 E F
        v0 : Fin 0 → E
        n : Nat
        hn : Not (Eq n 0)
        n_pos : LT.lt 0 n
        b : Composition n
        a✝ : Membership.mem Finset.univ b
        hb : Ne b (Composition.single n n_pos)
        A : LT.lt 1 b.length
        ⊢ Eq ((FormalMultilinearSeries.id 𝕜 F ((p 0) v0)).compAlongComposition p b) 0
      -/
      ext v
      /-
        case neg.h₀.H
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝⁴ : NontriviallyNormedField 𝕜
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        p : FormalMultilinearSeries 𝕜 E F
        v0 : Fin 0 → E
        n : Nat
        hn : Not (Eq n 0)
        n_pos : LT.lt 0 n
        b : Composition n
        a✝ : Membership.mem Finset.univ b
        hb : Ne b (Composition.single n n_pos)
        A : LT.lt 1 b.length
        v : Fin n → E
        ⊢ Eq (((FormalMultilinearSeries.id 𝕜 F ((p 0) v0)).compAlongComposition p b) v …
      -/
      rw [compAlongComposition_apply, id_apply_of_one_lt _ _ _ A]
      /-
        case neg.h₀.H
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝⁴ : NontriviallyNormedField 𝕜
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        p : FormalMultilinearSeries 𝕜 E F
        v0 : Fin 0 → E
        n : Nat
        hn : Not (Eq n 0)
        n_pos : LT.lt 0 n
        b : Composition n
        a✝ : Membership.mem Finset.univ b
        hb : Ne b (Composition.single n n_pos)
        A : LT.lt 1 b.length
        v : Fin n → E
        ⊢ Eq (0 (p.applyComposition b v)) (0 v)
      -/
      rfl
      /-
        🎉 no goals
      -/
      /-
        case neg.h₁
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝⁴ : NontriviallyNormedField 𝕜
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        p : FormalMultilinearSeries 𝕜 E F
        v0 : Fin 0 → E
        n : Nat
        hn : Not (Eq n 0)
        n_pos : LT.lt 0 n
        ⊢ Not (Membership.mem Finset.univ (Composition.single n n_pos)) → Eq ((FormalM …
      -/
    · simp
      /-
        🎉 no goals
      -/


/-- Variant of `id_comp` in which the zero coefficient is given by an equality hypothesis instead
of a definitional equality. Useful for rewriting or simplifying out in some situations. -/
theorem id_comp' (p : FormalMultilinearSeries 𝕜 E F) (x : F) (v0 : Fin 0 → E) (h : x = p 0 v0) :
    (id 𝕜 F x).comp p = p := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    x : F
    v0 : Fin 0 → E
    h : Eq x ((p 0) v0)
    ⊢ Eq ((FormalMultilinearSeries.id 𝕜 F x).comp p) p
  -/
  simp [h]
  /-
    🎉 no goals
  -/


/-- If two formal multilinear series have positive radius of convergence, then the terms appearing
in the definition of their composition are also summable (when multiplied by a suitable positive
geometric term). -/
theorem comp_summable_nnreal (q : FormalMultilinearSeries 𝕜 F G) (p : FormalMultilinearSeries 𝕜 E F)
    (hq : 0 < q.radius) (hp : 0 < p.radius) :
    ∃ r > (0 : ℝ≥0),
      Summable fun i : Σ n, Composition n => ‖q.compAlongComposition p i.2‖₊ * r ^ i.1 := by
  /- This follows from the fact that the growth rate of `‖qₙ‖` and `‖pₙ‖` is at most geometric,
    giving a geometric bound on each `‖q.compAlongComposition p op‖`, together with the
    fact that there are `2^(n-1)` compositions of `n`, giving at most a geometric loss. -/
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    hq : LT.lt 0 q.radius
    hp : LT.lt 0 p.radius
    ⊢ Exists fun r => And (GT.gt r 0) (Summable fun i => HMul.hMul (NNNorm.nnnorm  …
  -/
  rcases ENNReal.lt_iff_exists_nnreal_btwn.1 (lt_min zero_lt_one hq) with ⟨rq, rq_pos, hrq⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    hq : LT.lt 0 q.radius
    hp : LT.lt 0 p.radius
    rq : NNReal
    rq_pos : LT.lt 0 ↑rq
    hrq : LT.lt (↑rq) (Min.min 1 q.radius)
    ⊢ Exists fun r => And (GT.gt r 0) (Summable fun i => HMul.hMul (NNNorm.nnnorm  …
  -/
  rcases ENNReal.lt_iff_exists_nnreal_btwn.1 (lt_min zero_lt_one hp) with ⟨rp, rp_pos, hrp⟩
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    hq : LT.lt 0 q.radius
    hp : LT.lt 0 p.radius
    rq : NNReal
    rq_pos : LT.lt 0 ↑rq
    hrq : LT.lt (↑rq) (Min.min 1 q.radius)
    rp : NNReal
    rp_pos : LT.lt 0 ↑rp
    hrp : LT.lt (↑rp) (Min.min 1 p.radius)
    ⊢ Exists fun r => And (GT.gt r 0) (Summable fun i => HMul.hMul (NNNorm.nnnorm  …
  -/
  simp only [lt_min_iff, ENNReal.coe_lt_one_iff, ENNReal.coe_pos] at hrp hrq rp_pos rq_pos
  obtain ⟨Cq, _hCq0, hCq⟩ : ∃ Cq > 0, ∀ n, ‖q n‖₊ * rq ^ n ≤ Cq :=
    q.nnnorm_mul_pow_le_of_lt_radius hrq.2
  obtain ⟨Cp, hCp1, hCp⟩ : ∃ Cp ≥ 1, ∀ n, ‖p n‖₊ * rp ^ n ≤ Cp := by
    rcases p.nnnorm_mul_pow_le_of_lt_radius hrp.2 with ⟨Cp, -, hCp⟩
    exact ⟨max Cp 1, le_max_right _ _, fun n => (hCp n).trans (le_max_left _ _)⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    hq : LT.lt 0 q.radius
    hp : LT.lt 0 p.radius
    rq rp : NNReal
    hrp : And (LT.lt rp 1) (LT.lt (↑rp) p.radius)
    hrq : And (LT.lt rq 1) (LT.lt (↑rq) q.radius)
    rp_pos : LT.lt 0 rp
    rq_pos : LT.lt 0 rq
    Cq : NNReal
    _hCq0 : GT.gt Cq 0
    hCq : ∀ (n : Nat), LE.le (HMul.hMul (NNNorm.nnnorm (q n)) (HPow.hPow rq n)) Cq
    Cp : NNReal
    hCp1 : GE.ge Cp 1
    hCp : ∀ (n : Nat), LE.le (HMul.hMul (NNNorm.nnnorm (p n)) (HPow.hPow rp n)) Cp
    ⊢ Exists fun r => And (GT.gt r 0) (Summable fun i => HMul.hMul (NNNorm.nnnorm  …
  -/
  let r0 : ℝ≥0 := (4 * Cp)⁻¹
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    hq : LT.lt 0 q.radius
    hp : LT.lt 0 p.radius
    rq rp : NNReal
    hrp : And (LT.lt rp 1) (LT.lt (↑rp) p.radius)
    hrq : And (LT.lt rq 1) (LT.lt (↑rq) q.radius)
    rp_pos : LT.lt 0 rp
    rq_pos : LT.lt 0 rq
    Cq : NNReal
    _hCq0 : GT.gt Cq 0
    hCq : ∀ (n : Nat), LE.le (HMul.hMul (NNNorm.nnnorm (q n)) (HPow.hPow rq n)) Cq
    Cp : NNReal
    hCp1 : GE.ge Cp 1
    hCp : ∀ (n : Nat), LE.le (HMul.hMul (NNNorm.nnnorm (p n)) (HPow.hPow rp n)) Cp
    r0 : NNReal := Inv.inv (HMul.hMul 4 Cp)
    ⊢ Exists fun r => And (GT.gt r 0) (Summable fun i => HMul.hMul (NNNorm.nnnorm  …
  -/
  have r0_pos : 0 < r0 := inv_pos.2 (mul_pos zero_lt_four (zero_lt_one.trans_le hCp1))
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    hq : LT.lt 0 q.radius
    hp : LT.lt 0 p.radius
    rq rp : NNReal
    hrp : And (LT.lt rp 1) (LT.lt (↑rp) p.radius)
    hrq : And (LT.lt rq 1) (LT.lt (↑rq) q.radius)
    rp_pos : LT.lt 0 rp
    rq_pos : LT.lt 0 rq
    Cq : NNReal
    _hCq0 : GT.gt Cq 0
    hCq : ∀ (n : Nat), LE.le (HMul.hMul (NNNorm.nnnorm (q n)) (HPow.hPow rq n)) Cq
    Cp : NNReal
    hCp1 : GE.ge Cp 1
    hCp : ∀ (n : Nat), LE.le (HMul.hMul (NNNorm.nnnorm (p n)) (HPow.hPow rp n)) Cp
    r0 : NNReal := Inv.inv (HMul.hMul 4 Cp)
    r0_pos : LT.lt 0 r0
    ⊢ Exists fun r => And (GT.gt r 0) (Summable fun i => HMul.hMul (NNNorm.nnnorm  …
  -/
  set r : ℝ≥0 := rp * rq * r0
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    hq : LT.lt 0 q.radius
    hp : LT.lt 0 p.radius
    rq rp : NNReal
    hrp : And (LT.lt rp 1) (LT.lt (↑rp) p.radius)
    hrq : And (LT.lt rq 1) (LT.lt (↑rq) q.radius)
    rp_pos : LT.lt 0 rp
    rq_pos : LT.lt 0 rq
    Cq : NNReal
    _hCq0 : GT.gt Cq 0
    hCq : ∀ (n : Nat), LE.le (HMul.hMul (NNNorm.nnnorm (q n)) (HPow.hPow rq n)) Cq
    Cp : NNReal
    hCp1 : GE.ge Cp 1
    hCp : ∀ (n : Nat), LE.le (HMul.hMul (NNNorm.nnnorm (p n)) (HPow.hPow rp n)) Cp
    r0 : NNReal := Inv.inv (HMul.hMul 4 Cp)
    r0_pos : LT.lt 0 r0
    r : NNReal := HMul.hMul (HMul.hMul rp rq) r0
    ⊢ Exists fun r => And (GT.gt r 0) (Summable fun i => HMul.hMul (NNNorm.nnnorm  …
  -/
  have r_pos : 0 < r := mul_pos (mul_pos rp_pos rq_pos) r0_pos
  have I :
    ∀ i : Σ n : ℕ, Composition n, ‖q.compAlongComposition p i.2‖₊ * r ^ i.1 ≤ Cq / 4 ^ i.1 := by
    rintro ⟨n, c⟩
    have A := calc
      ‖q c.length‖₊ * rq ^ n ≤ ‖q c.length‖₊ * rq ^ c.length :=
        mul_le_mul' le_rfl (pow_le_pow_of_le_one rq.2 hrq.1.le c.length_le)
      _ ≤ Cq := hCq _
    have B := calc
      (∏ i, ‖p (c.blocksFun i)‖₊) * rp ^ n = ∏ i, ‖p (c.blocksFun i)‖₊ * rp ^ c.blocksFun i := by
        simp only [Finset.prod_mul_distrib, Finset.prod_pow_eq_pow_sum, c.sum_blocksFun]
      _ ≤ ∏ _i : Fin c.length, Cp := Finset.prod_le_prod' fun i _ => hCp _
      _ = Cp ^ c.length := by simp
      _ ≤ Cp ^ n := pow_right_mono₀ hCp1 c.length_le
    calc
      ‖q.compAlongComposition p c‖₊ * r ^ n ≤
          (‖q c.length‖₊ * ∏ i, ‖p (c.blocksFun i)‖₊) * r ^ n :=
        mul_le_mul' (q.compAlongComposition_nnnorm p c) le_rfl
      _ = ‖q c.length‖₊ * rq ^ n * ((∏ i, ‖p (c.blocksFun i)‖₊) * rp ^ n) * r0 ^ n := by
        ring
      _ ≤ Cq * Cp ^ n * r0 ^ n := mul_le_mul' (mul_le_mul' A B) le_rfl
      _ = Cq / 4 ^ n := by
        simp only [r0]
        field_simp [mul_pow, (zero_lt_one.trans_le hCp1).ne']
        ring
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    hq : LT.lt 0 q.radius
    hp : LT.lt 0 p.radius
    rq rp : NNReal
    hrp : And (LT.lt rp 1) (LT.lt (↑rp) p.radius)
    hrq : And (LT.lt rq 1) (LT.lt (↑rq) q.radius)
    rp_pos : LT.lt 0 rp
    rq_pos : LT.lt 0 rq
    Cq : NNReal
    _hCq0 : GT.gt Cq 0
    hCq : ∀ (n : Nat), LE.le (HMul.hMul (NNNorm.nnnorm (q n)) (HPow.hPow rq n)) Cq
    Cp : NNReal
    hCp1 : GE.ge Cp 1
    hCp : ∀ (n : Nat), LE.le (HMul.hMul (NNNorm.nnnorm (p n)) (HPow.hPow rp n)) Cp
    r0 : NNReal := Inv.inv (HMul.hMul 4 Cp)
    r0_pos : LT.lt 0 r0
    r : NNReal := HMul.hMul (HMul.hMul rp rq) r0
    r_pos : LT.lt 0 r
    I : ∀ (i : Sigma fun n => Composition n), LE.le (HMul.hMul (NNNorm.nnnorm (q.c …
    ⊢ Exists fun r => And (GT.gt r 0) (Summable fun i => HMul.hMul (NNNorm.nnnorm  …
  -/
  refine ⟨r, r_pos, NNReal.summable_of_le I ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    hq : LT.lt 0 q.radius
    hp : LT.lt 0 p.radius
    rq rp : NNReal
    hrp : And (LT.lt rp 1) (LT.lt (↑rp) p.radius)
    hrq : And (LT.lt rq 1) (LT.lt (↑rq) q.radius)
    rp_pos : LT.lt 0 rp
    rq_pos : LT.lt 0 rq
    Cq : NNReal
    _hCq0 : GT.gt Cq 0
    hCq : ∀ (n : Nat), LE.le (HMul.hMul (NNNorm.nnnorm (q n)) (HPow.hPow rq n)) Cq
    Cp : NNReal
    hCp1 : GE.ge Cp 1
    hCp : ∀ (n : Nat), LE.le (HMul.hMul (NNNorm.nnnorm (p n)) (HPow.hPow rp n)) Cp
    r0 : NNReal := Inv.inv (HMul.hMul 4 Cp)
    r0_pos : LT.lt 0 r0
    r : NNReal := HMul.hMul (HMul.hMul rp rq) r0
    r_pos : LT.lt 0 r
    I : ∀ (i : Sigma fun n => Composition n), LE.le (HMul.hMul (NNNorm.nnnorm (q.c …
    ⊢ Summable fun b => HDiv.hDiv Cq (HPow.hPow 4 b.fst)
  -/
  simp_rw [div_eq_mul_inv]
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    hq : LT.lt 0 q.radius
    hp : LT.lt 0 p.radius
    rq rp : NNReal
    hrp : And (LT.lt rp 1) (LT.lt (↑rp) p.radius)
    hrq : And (LT.lt rq 1) (LT.lt (↑rq) q.radius)
    rp_pos : LT.lt 0 rp
    rq_pos : LT.lt 0 rq
    Cq : NNReal
    _hCq0 : GT.gt Cq 0
    hCq : ∀ (n : Nat), LE.le (HMul.hMul (NNNorm.nnnorm (q n)) (HPow.hPow rq n)) Cq
    Cp : NNReal
    hCp1 : GE.ge Cp 1
    hCp : ∀ (n : Nat), LE.le (HMul.hMul (NNNorm.nnnorm (p n)) (HPow.hPow rp n)) Cp
    r0 : NNReal := Inv.inv (HMul.hMul 4 Cp)
    r0_pos : LT.lt 0 r0
    r : NNReal := HMul.hMul (HMul.hMul rp rq) r0
    r_pos : LT.lt 0 r
    I : ∀ (i : Sigma fun n => Composition n), LE.le (HMul.hMul (NNNorm.nnnorm (q.c …
    ⊢ Summable fun b => HMul.hMul Cq (Inv.inv (HPow.hPow 4 b.fst))
  -/
  refine Summable.mul_left _ ?_
  have : ∀ n : ℕ, HasSum (fun c : Composition n => (4 ^ n : ℝ≥0)⁻¹) (2 ^ (n - 1) / 4 ^ n) := by
    intro n
    convert hasSum_fintype fun c : Composition n => (4 ^ n : ℝ≥0)⁻¹
    simp [Finset.card_univ, composition_card, div_eq_mul_inv]
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    hq : LT.lt 0 q.radius
    hp : LT.lt 0 p.radius
    rq rp : NNReal
    hrp : And (LT.lt rp 1) (LT.lt (↑rp) p.radius)
    hrq : And (LT.lt rq 1) (LT.lt (↑rq) q.radius)
    rp_pos : LT.lt 0 rp
    rq_pos : LT.lt 0 rq
    Cq : NNReal
    _hCq0 : GT.gt Cq 0
    hCq : ∀ (n : Nat), LE.le (HMul.hMul (NNNorm.nnnorm (q n)) (HPow.hPow rq n)) Cq
    Cp : NNReal
    hCp1 : GE.ge Cp 1
    hCp : ∀ (n : Nat), LE.le (HMul.hMul (NNNorm.nnnorm (p n)) (HPow.hPow rp n)) Cp
    r0 : NNReal := Inv.inv (HMul.hMul 4 Cp)
    r0_pos : LT.lt 0 r0
    r : NNReal := HMul.hMul (HMul.hMul rp rq) r0
    r_pos : LT.lt 0 r
    I : ∀ (i : Sigma fun n => Composition n), LE.le (HMul.hMul (NNNorm.nnnorm (q.c …
    this : ∀ (n : Nat), HasSum (fun c => Inv.inv (HPow.hPow 4 n)) (HDiv.hDiv (HPow …
    ⊢ Summable fun b => Inv.inv (HPow.hPow 4 b.fst)
  -/
  refine NNReal.summable_sigma.2 ⟨fun n => (this n).summable, (NNReal.summable_nat_add_iff 1).1 ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    hq : LT.lt 0 q.radius
    hp : LT.lt 0 p.radius
    rq rp : NNReal
    hrp : And (LT.lt rp 1) (LT.lt (↑rp) p.radius)
    hrq : And (LT.lt rq 1) (LT.lt (↑rq) q.radius)
    rp_pos : LT.lt 0 rp
    rq_pos : LT.lt 0 rq
    Cq : NNReal
    _hCq0 : GT.gt Cq 0
    hCq : ∀ (n : Nat), LE.le (HMul.hMul (NNNorm.nnnorm (q n)) (HPow.hPow rq n)) Cq
    Cp : NNReal
    hCp1 : GE.ge Cp 1
    hCp : ∀ (n : Nat), LE.le (HMul.hMul (NNNorm.nnnorm (p n)) (HPow.hPow rp n)) Cp
    r0 : NNReal := Inv.inv (HMul.hMul 4 Cp)
    r0_pos : LT.lt 0 r0
    r : NNReal := HMul.hMul (HMul.hMul rp rq) r0
    r_pos : LT.lt 0 r
    I : ∀ (i : Sigma fun n => Composition n), LE.le (HMul.hMul (NNNorm.nnnorm (q.c …
    this : ∀ (n : Nat), HasSum (fun c => Inv.inv (HPow.hPow 4 n)) (HDiv.hDiv (HPow …
    ⊢ Summable fun i => tsum fun y => Inv.inv (HPow.hPow 4 ⟨HAdd.hAdd i 1, y⟩.fst)
  -/
  convert (NNReal.summable_geometric (NNReal.div_lt_one_of_lt one_lt_two)).mul_left (1 / 4) using 1
  /-
    case h.e'_5
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    hq : LT.lt 0 q.radius
    hp : LT.lt 0 p.radius
    rq rp : NNReal
    hrp : And (LT.lt rp 1) (LT.lt (↑rp) p.radius)
    hrq : And (LT.lt rq 1) (LT.lt (↑rq) q.radius)
    rp_pos : LT.lt 0 rp
    rq_pos : LT.lt 0 rq
    Cq : NNReal
    _hCq0 : GT.gt Cq 0
    hCq : ∀ (n : Nat), LE.le (HMul.hMul (NNNorm.nnnorm (q n)) (HPow.hPow rq n)) Cq
    Cp : NNReal
    hCp1 : GE.ge Cp 1
    hCp : ∀ (n : Nat), LE.le (HMul.hMul (NNNorm.nnnorm (p n)) (HPow.hPow rp n)) Cp
    r0 : NNReal := Inv.inv (HMul.hMul 4 Cp)
    r0_pos : LT.lt 0 r0
    r : NNReal := HMul.hMul (HMul.hMul rp rq) r0
    r_pos : LT.lt 0 r
    I : ∀ (i : Sigma fun n => Composition n), LE.le (HMul.hMul (NNNorm.nnnorm (q.c …
    this : ∀ (n : Nat), HasSum (fun c => Inv.inv (HPow.hPow 4 n)) (HDiv.hDiv (HPow …
    ⊢ Eq (fun i => tsum fun y => Inv.inv (HPow.hPow 4 ⟨HAdd.hAdd i 1, y⟩.fst)) fun …
  -/
  ext1 n
  /-
    case h.e'_5.h
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    hq : LT.lt 0 q.radius
    hp : LT.lt 0 p.radius
    rq rp : NNReal
    hrp : And (LT.lt rp 1) (LT.lt (↑rp) p.radius)
    hrq : And (LT.lt rq 1) (LT.lt (↑rq) q.radius)
    rp_pos : LT.lt 0 rp
    rq_pos : LT.lt 0 rq
    Cq : NNReal
    _hCq0 : GT.gt Cq 0
    hCq : ∀ (n : Nat), LE.le (HMul.hMul (NNNorm.nnnorm (q n)) (HPow.hPow rq n)) Cq
    Cp : NNReal
    hCp1 : GE.ge Cp 1
    hCp : ∀ (n : Nat), LE.le (HMul.hMul (NNNorm.nnnorm (p n)) (HPow.hPow rp n)) Cp
    r0 : NNReal := Inv.inv (HMul.hMul 4 Cp)
    r0_pos : LT.lt 0 r0
    r : NNReal := HMul.hMul (HMul.hMul rp rq) r0
    r_pos : LT.lt 0 r
    I : ∀ (i : Sigma fun n => Composition n), LE.le (HMul.hMul (NNNorm.nnnorm (q.c …
    this : ∀ (n : Nat), HasSum (fun c => Inv.inv (HPow.hPow 4 n)) (HDiv.hDiv (HPow …
    n : Nat
    ⊢ Eq (tsum fun y => Inv.inv (HPow.hPow 4 ⟨HAdd.hAdd n 1, y⟩.fst)) (HMul.hMul ( …
  -/
  rw [(this _).tsum_eq, add_tsub_cancel_right]
  field_simp [← mul_assoc, pow_succ, mul_pow, show (4 : ℝ≥0) = 2 * 2 by norm_num,
    mul_right_comm]


/-- Bounding below the radius of the composition of two formal multilinear series assuming
summability over all compositions. -/
theorem le_comp_radius_of_summable (q : FormalMultilinearSeries 𝕜 F G)
    (p : FormalMultilinearSeries 𝕜 E F) (r : ℝ≥0)
    (hr : Summable fun i : Σ n, Composition n => ‖q.compAlongComposition p i.2‖₊ * r ^ i.1) :
    (r : ℝ≥0∞) ≤ (q.comp p).radius := by
  refine
    le_radius_of_bound_nnreal _
      (∑' i : Σ n, Composition n, ‖compAlongComposition q p i.snd‖₊ * r ^ i.fst) fun n => ?_
  calc
    ‖FormalMultilinearSeries.comp q p n‖₊ * r ^ n ≤
        ∑' c : Composition n, ‖compAlongComposition q p c‖₊ * r ^ n := by
      rw [tsum_fintype, ← Finset.sum_mul]
      exact mul_le_mul' (nnnorm_sum_le _ _) le_rfl
    _ ≤ ∑' i : Σ n : ℕ, Composition n, ‖compAlongComposition q p i.snd‖₊ * r ^ i.fst :=
      NNReal.tsum_comp_le_tsum_of_inj hr sigma_mk_injective


/-- Source set in the change of variables to compute the composition of partial sums of formal
power series.
See also `comp_partialSum`. -/
def compPartialSumSource (m M N : ℕ) : Finset (Σ n, Fin n → ℕ) :=
  Finset.sigma (Finset.Ico m M) (fun n : ℕ => Fintype.piFinset fun _i : Fin n => Finset.Ico 1 N : _)


@[simp]
theorem mem_compPartialSumSource_iff (m M N : ℕ) (i : Σ n, Fin n → ℕ) :
    i ∈ compPartialSumSource m M N ↔
      (m ≤ i.1 ∧ i.1 < M) ∧ ∀ a : Fin i.1, 1 ≤ i.2 a ∧ i.2 a < N := by
  /-
    m M N : Nat
    i : Sigma fun n => Fin n → Nat
    ⊢ Iff (Membership.mem (FormalMultilinearSeries.compPartialSumSource m M N) i)  …
  -/
  simp only [compPartialSumSource, Finset.mem_Ico, Fintype.mem_piFinset, Finset.mem_sigma]
  /-
    🎉 no goals
  -/


/-- Change of variables appearing to compute the composition of partial sums of formal
power series -/
def compChangeOfVariables (m M N : ℕ) (i : Σ n, Fin n → ℕ) (hi : i ∈ compPartialSumSource m M N) :
    Σ n, Composition n := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    H : Type u_5
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    m M N : Nat
    i : Sigma fun n => Fin n → Nat
    hi : Membership.mem (FormalMultilinearSeries.compPartialSumSource m M N) i
    ⊢ Sigma fun n => Composition n
  -/
  rcases i with ⟨n, f⟩
  /-
    case mk
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    H : Type u_5
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    m M N n : Nat
    f : Fin n → Nat
    hi : Membership.mem (FormalMultilinearSeries.compPartialSumSource m M N) ⟨n, f⟩
    ⊢ Sigma fun n => Composition n
  -/
  rw [mem_compPartialSumSource_iff] at hi
  /-
    case mk
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    H : Type u_5
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    m M N n : Nat
    f : Fin n → Nat
    hi : And (And (LE.le m ⟨n, f⟩.fst) (LT.lt ⟨n, f⟩.fst M)) (∀ (a : Fin ⟨n, f⟩.fs …
    ⊢ Sigma fun n => Composition n
  -/
  refine ⟨∑ j, f j, ofFn fun a => f a, fun hi' => ?_, by simp [sum_ofFn]⟩
  /-
    case mk
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    H : Type u_5
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    m M N n : Nat
    f : Fin n → Nat
    hi : And (And (LE.le m ⟨n, f⟩.fst) (LT.lt ⟨n, f⟩.fst M)) (∀ (a : Fin ⟨n, f⟩.fs …
    i✝ : Nat
    hi' : Membership.mem (List.ofFn fun a => f a) i✝
    ⊢ LT.lt 0 i✝
  -/
  rename_i i
  /-
    case mk
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    H : Type u_5
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    m M N n : Nat
    f : Fin n → Nat
    hi : And (And (LE.le m ⟨n, f⟩.fst) (LT.lt ⟨n, f⟩.fst M)) (∀ (a : Fin ⟨n, f⟩.fs …
    i : Nat
    hi' : Membership.mem (List.ofFn fun a => f a) i
    ⊢ LT.lt 0 i
  -/
  obtain ⟨j, rfl⟩ : ∃ j : Fin n, f j = i := by rwa [mem_ofFn, Set.mem_range] at hi'
  /-
    case mk.intro
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    H : Type u_5
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    m M N n : Nat
    f : Fin n → Nat
    hi : And (And (LE.le m ⟨n, f⟩.fst) (LT.lt ⟨n, f⟩.fst M)) (∀ (a : Fin ⟨n, f⟩.fs …
    j : Fin n
    hi' : Membership.mem (List.ofFn fun a => f a) (f j)
    ⊢ LT.lt 0 (f j)
  -/
  exact (hi.2 j).1
  /-
    🎉 no goals
  -/


@[simp]
theorem compChangeOfVariables_length (m M N : ℕ) {i : Σ n, Fin n → ℕ}
    (hi : i ∈ compPartialSumSource m M N) :
    Composition.length (compChangeOfVariables m M N i hi).2 = i.1 := by
  /-
    m M N : Nat
    i : Sigma fun n => Fin n → Nat
    hi : Membership.mem (FormalMultilinearSeries.compPartialSumSource m M N) i
    ⊢ Eq (FormalMultilinearSeries.compChangeOfVariables m M N i hi).snd.length i.fst
  -/
  rcases i with ⟨k, blocks_fun⟩
  /-
    case mk
    m M N k : Nat
    blocks_fun : Fin k → Nat
    hi : Membership.mem (FormalMultilinearSeries.compPartialSumSource m M N) ⟨k, b …
    ⊢ Eq (FormalMultilinearSeries.compChangeOfVariables m M N ⟨k, blocks_fun⟩ hi). …
  -/
  dsimp [compChangeOfVariables]
  /-
    case mk
    m M N k : Nat
    blocks_fun : Fin k → Nat
    hi : Membership.mem (FormalMultilinearSeries.compPartialSumSource m M N) ⟨k, b …
    ⊢ Eq { blocks := List.ofFn fun a => blocks_fun a, blocks_pos := ⋯, blocks_sum  …
  -/
  simp only [Composition.length, map_ofFn, length_ofFn]
  /-
    🎉 no goals
  -/


theorem compChangeOfVariables_blocksFun (m M N : ℕ) {i : Σ n, Fin n → ℕ}
    (hi : i ∈ compPartialSumSource m M N) (j : Fin i.1) :
    (compChangeOfVariables m M N i hi).2.blocksFun
        ⟨j, (compChangeOfVariables_length m M N hi).symm ▸ j.2⟩ =
      i.2 j := by
  /-
    m M N : Nat
    i : Sigma fun n => Fin n → Nat
    hi : Membership.mem (FormalMultilinearSeries.compPartialSumSource m M N) i
    j : Fin i.fst
    ⊢ Eq ((FormalMultilinearSeries.compChangeOfVariables m M N i hi).snd.blocksFun …
  -/
  rcases i with ⟨n, f⟩
  /-
    case mk
    m M N n : Nat
    f : Fin n → Nat
    hi : Membership.mem (FormalMultilinearSeries.compPartialSumSource m M N) ⟨n, f⟩
    j : Fin ⟨n, f⟩.fst
    ⊢ Eq ((FormalMultilinearSeries.compChangeOfVariables m M N ⟨n, f⟩ hi).snd.bloc …
  -/
  dsimp [Composition.blocksFun, Composition.blocks, compChangeOfVariables]
  /-
    case mk
    m M N n : Nat
    f : Fin n → Nat
    hi : Membership.mem (FormalMultilinearSeries.compPartialSumSource m M N) ⟨n, f⟩
    j : Fin ⟨n, f⟩.fst
    ⊢ Eq (GetElem.getElem (List.ofFn fun a => f a) ↑j ⋯) (f j)
  -/
  simp only [map_ofFn, List.getElem_ofFn, Function.comp_apply]
  /-
    🎉 no goals
  -/


/-- Target set in the change of variables to compute the composition of partial sums of formal
power series, here given a a set. -/
def compPartialSumTargetSet (m M N : ℕ) : Set (Σ n, Composition n) :=
  {i | m ≤ i.2.length ∧ i.2.length < M ∧ ∀ j : Fin i.2.length, i.2.blocksFun j < N}


theorem compPartialSumTargetSet_image_compPartialSumSource (m M N : ℕ)
    (i : Σ n, Composition n) (hi : i ∈ compPartialSumTargetSet m M N) :
    ∃ (j : _) (hj : j ∈ compPartialSumSource m M N), compChangeOfVariables m M N j hj = i := by
  /-
    m M N : Nat
    i : Sigma fun n => Composition n
    hi : Membership.mem (FormalMultilinearSeries.compPartialSumTargetSet m M N) i
    ⊢ Exists fun j => Exists fun hj => Eq (FormalMultilinearSeries.compChangeOfVar …
  -/
  rcases i with ⟨n, c⟩
  /-
    case mk
    m M N n : Nat
    c : Composition n
    hi : Membership.mem (FormalMultilinearSeries.compPartialSumTargetSet m M N) ⟨n …
    ⊢ Exists fun j => Exists fun hj => Eq (FormalMultilinearSeries.compChangeOfVar …
  -/
  refine ⟨⟨c.length, c.blocksFun⟩, ?_, ?_⟩
    /-
      case mk.refine_1
      m M N n : Nat
      c : Composition n
      hi : Membership.mem (FormalMultilinearSeries.compPartialSumTargetSet m M N) ⟨n …
      ⊢ Membership.mem (FormalMultilinearSeries.compPartialSumSource m M N) ⟨c.lengt …
    -/
  · simp only [compPartialSumTargetSet, Set.mem_setOf_eq] at hi
    /-
      case mk.refine_1
      m M N n : Nat
      c : Composition n
      hi : And (LE.le m c.length) (And (LT.lt c.length M) (∀ (j : Fin c.length), LT. …
      ⊢ Membership.mem (FormalMultilinearSeries.compPartialSumSource m M N) ⟨c.lengt …
    -/
    simp only [mem_compPartialSumSource_iff, hi.left, hi.right, true_and, and_true]
    /-
      case mk.refine_1
      m M N n : Nat
      c : Composition n
      hi : And (LE.le m c.length) (And (LT.lt c.length M) (∀ (j : Fin c.length), LT. …
      ⊢ ∀ (a : Fin c.length), LE.le 1 (c.blocksFun a)
    -/
    exact fun a => c.one_le_blocks' _
    /-
      🎉 no goals
    -/
    /-
      case mk.refine_2
      m M N n : Nat
      c : Composition n
      hi : Membership.mem (FormalMultilinearSeries.compPartialSumTargetSet m M N) ⟨n …
      ⊢ Eq (FormalMultilinearSeries.compChangeOfVariables m M N ⟨c.length, c.blocksF …
    -/
  · dsimp [compChangeOfVariables]
    /-
      case mk.refine_2
      m M N n : Nat
      c : Composition n
      hi : Membership.mem (FormalMultilinearSeries.compPartialSumTargetSet m M N) ⟨n …
      ⊢ Eq ⟨Finset.univ.sum fun j => c.blocksFun j, { blocks := List.ofFn fun a => c …
    -/
    rw [Composition.sigma_eq_iff_blocks_eq]
    /-
      case mk.refine_2
      m M N n : Nat
      c : Composition n
      hi : Membership.mem (FormalMultilinearSeries.compPartialSumTargetSet m M N) ⟨n …
      ⊢ Eq ⟨Finset.univ.sum fun j => c.blocksFun j, { blocks := List.ofFn fun a => c …
    -/
    simp only [Composition.blocksFun, Composition.blocks, Subtype.coe_eta]
    /-
      case mk.refine_2
      m M N n : Nat
      c : Composition n
      hi : Membership.mem (FormalMultilinearSeries.compPartialSumTargetSet m M N) ⟨n …
      ⊢ Eq (List.ofFn fun a => c.blocks.get a) c.blocks
    -/
    conv_rhs => rw [← List.ofFn_get c.blocks]
    /-
      🎉 no goals
    -/


/-- Target set in the change of variables to compute the composition of partial sums of formal
power series, here given a a finset.
See also `comp_partialSum`. -/
def compPartialSumTarget (m M N : ℕ) : Finset (Σ n, Composition n) :=
  Set.Finite.toFinset <|
    ((Finset.finite_toSet _).dependent_image _).subset <|
      compPartialSumTargetSet_image_compPartialSumSource m M N


@[simp]
theorem mem_compPartialSumTarget_iff {m M N : ℕ} {a : Σ n, Composition n} :
    a ∈ compPartialSumTarget m M N ↔
      m ≤ a.2.length ∧ a.2.length < M ∧ ∀ j : Fin a.2.length, a.2.blocksFun j < N := by
  /-
    m M N : Nat
    a : Sigma fun n => Composition n
    ⊢ Iff (Membership.mem (FormalMultilinearSeries.compPartialSumTarget m M N) a)  …
  -/
  simp [compPartialSumTarget, compPartialSumTargetSet]
  /-
    🎉 no goals
  -/


/-- `compChangeOfVariables m M N` is a bijection between `compPartialSumSource m M N`
and `compPartialSumTarget m M N`, yielding equal sums for functions that correspond to each
other under the bijection. As `compChangeOfVariables m M N` is a dependent function, stating
that it is a bijection is not directly possible, but the consequence on sums can be stated
more easily. -/
theorem compChangeOfVariables_sum {α : Type*} [AddCommMonoid α] (m M N : ℕ)
    (f : (Σ n : ℕ, Fin n → ℕ) → α) (g : (Σ n, Composition n) → α)
    (h : ∀ (e) (he : e ∈ compPartialSumSource m M N), f e = g (compChangeOfVariables m M N e he)) :
    ∑ e ∈ compPartialSumSource m M N, f e = ∑ e ∈ compPartialSumTarget m M N, g e := by
  /-
    α : Type u_6
    inst✝ : AddCommMonoid α
    m M N : Nat
    f : (Sigma fun n => Fin n → Nat) → α
    g : (Sigma fun n => Composition n) → α
    h : ∀ (e : Sigma fun n => Fin n → Nat) (he : Membership.mem (FormalMultilinear …
    ⊢ Eq ((FormalMultilinearSeries.compPartialSumSource m M N).sum fun e => f e) ( …
  -/
  apply Finset.sum_bij (compChangeOfVariables m M N)
  -- We should show that the correspondence we have set up is indeed a bijection
  -- between the index sets of the two sums.
  -- 1 - show that the image belongs to `compPartialSumTarget m N N`
    /-
      case hi
      α : Type u_6
      inst✝ : AddCommMonoid α
      m M N : Nat
      f : (Sigma fun n => Fin n → Nat) → α
      g : (Sigma fun n => Composition n) → α
      h : ∀ (e : Sigma fun n => Fin n → Nat) (he : Membership.mem (FormalMultilinear …
      ⊢ ∀ (a : Sigma fun n => Fin n → Nat) (ha : Membership.mem (FormalMultilinearSe …
    -/
  · rintro ⟨k, blocks_fun⟩ H
    /-
      case hi.mk
      α : Type u_6
      inst✝ : AddCommMonoid α
      m M N : Nat
      f : (Sigma fun n => Fin n → Nat) → α
      g : (Sigma fun n => Composition n) → α
      h : ∀ (e : Sigma fun n => Fin n → Nat) (he : Membership.mem (FormalMultilinear …
      k : Nat
      blocks_fun : Fin k → Nat
      H : Membership.mem (FormalMultilinearSeries.compPartialSumSource m M N) ⟨k, bl …
      ⊢ Membership.mem (FormalMultilinearSeries.compPartialSumTarget m M N) (FormalM …
    -/
    rw [mem_compPartialSumSource_iff] at H
    -- Porting note: added
    /-
      case hi.mk
      α : Type u_6
      inst✝ : AddCommMonoid α
      m M N : Nat
      f : (Sigma fun n => Fin n → Nat) → α
      g : (Sigma fun n => Composition n) → α
      h : ∀ (e : Sigma fun n => Fin n → Nat) (he : Membership.mem (FormalMultilinear …
      k : Nat
      blocks_fun : Fin k → Nat
      H✝ : Membership.mem (FormalMultilinearSeries.compPartialSumSource m M N) ⟨k, b …
      H : And (And (LE.le m ⟨k, blocks_fun⟩.fst) (LT.lt ⟨k, blocks_fun⟩.fst M)) (∀ ( …
      ⊢ Membership.mem (FormalMultilinearSeries.compPartialSumTarget m M N) (FormalM …
    -/
    simp only at H
    simp only [mem_compPartialSumTarget_iff, Composition.length, Composition.blocks, H.left,
      map_ofFn, length_ofFn, true_and, compChangeOfVariables]
    /-
      case hi.mk
      α : Type u_6
      inst✝ : AddCommMonoid α
      m M N : Nat
      f : (Sigma fun n => Fin n → Nat) → α
      g : (Sigma fun n => Composition n) → α
      h : ∀ (e : Sigma fun n => Fin n → Nat) (he : Membership.mem (FormalMultilinear …
      k : Nat
      blocks_fun : Fin k → Nat
      H✝ : Membership.mem (FormalMultilinearSeries.compPartialSumSource m M N) ⟨k, b …
      H : And (And (LE.le m k) (LT.lt k M)) (∀ (a : Fin k), And (LE.le 1 (blocks_fun …
      ⊢ ∀ (j : Fin (List.ofFn fun a => blocks_fun a).length), LT.lt ({ blocks := Lis …
    -/
    intro j
    /-
      case hi.mk
      α : Type u_6
      inst✝ : AddCommMonoid α
      m M N : Nat
      f : (Sigma fun n => Fin n → Nat) → α
      g : (Sigma fun n => Composition n) → α
      h : ∀ (e : Sigma fun n => Fin n → Nat) (he : Membership.mem (FormalMultilinear …
      k : Nat
      blocks_fun : Fin k → Nat
      H✝ : Membership.mem (FormalMultilinearSeries.compPartialSumSource m M N) ⟨k, b …
      H : And (And (LE.le m k) (LT.lt k M)) (∀ (a : Fin k), And (LE.le 1 (blocks_fun …
      j : Fin (List.ofFn fun a => blocks_fun a).length
      ⊢ LT.lt ({ blocks := List.ofFn fun a => blocks_fun a, blocks_pos := ⋯, blocks_ …
    -/
    simp only [Composition.blocksFun, (H.right _).right, List.get_ofFn]
    /-
      🎉 no goals
    -/
  -- 2 - show that the map is injective
    /-
      case i_inj
      α : Type u_6
      inst✝ : AddCommMonoid α
      m M N : Nat
      f : (Sigma fun n => Fin n → Nat) → α
      g : (Sigma fun n => Composition n) → α
      h : ∀ (e : Sigma fun n => Fin n → Nat) (he : Membership.mem (FormalMultilinear …
      ⊢ ∀ (a₁ : Sigma fun n => Fin n → Nat) (ha₁ : Membership.mem (FormalMultilinear …
    -/
  · rintro ⟨k, blocks_fun⟩ H ⟨k', blocks_fun'⟩ H' heq
    obtain rfl : k = k' := by
      have := (compChangeOfVariables_length m M N H).symm
      rwa [heq, compChangeOfVariables_length] at this
    /-
      case i_inj.mk.mk
      α : Type u_6
      inst✝ : AddCommMonoid α
      m M N : Nat
      f : (Sigma fun n => Fin n → Nat) → α
      g : (Sigma fun n => Composition n) → α
      h : ∀ (e : Sigma fun n => Fin n → Nat) (he : Membership.mem (FormalMultilinear …
      k : Nat
      blocks_fun : Fin k → Nat
      H : Membership.mem (FormalMultilinearSeries.compPartialSumSource m M N) ⟨k, bl …
      blocks_fun' : Fin k → Nat
      H' : Membership.mem (FormalMultilinearSeries.compPartialSumSource m M N) ⟨k, b …
      heq : Eq (FormalMultilinearSeries.compChangeOfVariables m M N ⟨k, blocks_fun⟩  …
      ⊢ Eq ⟨k, blocks_fun⟩ ⟨k, blocks_fun'⟩
    -/
    congr
    /-
      case i_inj.mk.mk.e_snd
      α : Type u_6
      inst✝ : AddCommMonoid α
      m M N : Nat
      f : (Sigma fun n => Fin n → Nat) → α
      g : (Sigma fun n => Composition n) → α
      h : ∀ (e : Sigma fun n => Fin n → Nat) (he : Membership.mem (FormalMultilinear …
      k : Nat
      blocks_fun : Fin k → Nat
      H : Membership.mem (FormalMultilinearSeries.compPartialSumSource m M N) ⟨k, bl …
      blocks_fun' : Fin k → Nat
      H' : Membership.mem (FormalMultilinearSeries.compPartialSumSource m M N) ⟨k, b …
      heq : Eq (FormalMultilinearSeries.compChangeOfVariables m M N ⟨k, blocks_fun⟩  …
      ⊢ Eq blocks_fun blocks_fun'
    -/
    funext i
    calc
      blocks_fun i = (compChangeOfVariables m M N _ H).2.blocksFun _ :=
        (compChangeOfVariables_blocksFun m M N H i).symm
      _ = (compChangeOfVariables m M N _ H').2.blocksFun _ := by
        apply Composition.blocksFun_congr <;>
        first | rw [heq] | rfl
      _ = blocks_fun' i := compChangeOfVariables_blocksFun m M N H' i
  -- 3 - show that the map is surjective
    /-
      case i_surj
      α : Type u_6
      inst✝ : AddCommMonoid α
      m M N : Nat
      f : (Sigma fun n => Fin n → Nat) → α
      g : (Sigma fun n => Composition n) → α
      h : ∀ (e : Sigma fun n => Fin n → Nat) (he : Membership.mem (FormalMultilinear …
      ⊢ ∀ (b : Sigma fun n => Composition n), Membership.mem (FormalMultilinearSerie …
    -/
  · intro i hi
    /-
      case i_surj
      α : Type u_6
      inst✝ : AddCommMonoid α
      m M N : Nat
      f : (Sigma fun n => Fin n → Nat) → α
      g : (Sigma fun n => Composition n) → α
      h : ∀ (e : Sigma fun n => Fin n → Nat) (he : Membership.mem (FormalMultilinear …
      i : Sigma fun n => Composition n
      hi : Membership.mem (FormalMultilinearSeries.compPartialSumTarget m M N) i
      ⊢ Exists fun a => Exists fun ha => Eq (FormalMultilinearSeries.compChangeOfVar …
    -/
    apply compPartialSumTargetSet_image_compPartialSumSource m M N i
    /-
      case i_surj
      α : Type u_6
      inst✝ : AddCommMonoid α
      m M N : Nat
      f : (Sigma fun n => Fin n → Nat) → α
      g : (Sigma fun n => Composition n) → α
      h : ∀ (e : Sigma fun n => Fin n → Nat) (he : Membership.mem (FormalMultilinear …
      i : Sigma fun n => Composition n
      hi : Membership.mem (FormalMultilinearSeries.compPartialSumTarget m M N) i
      ⊢ Membership.mem (FormalMultilinearSeries.compPartialSumTargetSet m M N) i
    -/
    simpa [compPartialSumTarget] using hi
    /-
      🎉 no goals
    -/
  -- 4 - show that the composition gives the `compAlongComposition` application
    /-
      case h
      α : Type u_6
      inst✝ : AddCommMonoid α
      m M N : Nat
      f : (Sigma fun n => Fin n → Nat) → α
      g : (Sigma fun n => Composition n) → α
      h : ∀ (e : Sigma fun n => Fin n → Nat) (he : Membership.mem (FormalMultilinear …
      ⊢ ∀ (a : Sigma fun n => Fin n → Nat) (ha : Membership.mem (FormalMultilinearSe …
    -/
  · rintro ⟨k, blocks_fun⟩ H
    /-
      case h.mk
      α : Type u_6
      inst✝ : AddCommMonoid α
      m M N : Nat
      f : (Sigma fun n => Fin n → Nat) → α
      g : (Sigma fun n => Composition n) → α
      h : ∀ (e : Sigma fun n => Fin n → Nat) (he : Membership.mem (FormalMultilinear …
      k : Nat
      blocks_fun : Fin k → Nat
      H : Membership.mem (FormalMultilinearSeries.compPartialSumSource m M N) ⟨k, bl …
      ⊢ Eq (f ⟨k, blocks_fun⟩) (g (FormalMultilinearSeries.compChangeOfVariables m M …
    -/
    rw [h]
    /-
      🎉 no goals
    -/


/-- The auxiliary set corresponding to the composition of partial sums asymptotically contains
all possible compositions. -/
theorem compPartialSumTarget_tendsto_prod_atTop :
    Tendsto (fun (p : ℕ × ℕ) => compPartialSumTarget 0 p.1 p.2) atTop atTop := by
  /-
    ⊢ Filter.Tendsto (fun p => FormalMultilinearSeries.compPartialSumTarget 0 p.1  …
  -/
  apply Monotone.tendsto_atTop_finset
    /-
      case h
      ⊢ Monotone fun p => FormalMultilinearSeries.compPartialSumTarget 0 p.1 p.2
    -/
  · intro m n hmn a ha
    /-
      case h
      m n : Prod Nat Nat
      hmn : LE.le m n
      a : Sigma fun n => Composition n
      ha : Membership.mem ((fun p => FormalMultilinearSeries.compPartialSumTarget 0  …
      ⊢ Membership.mem ((fun p => FormalMultilinearSeries.compPartialSumTarget 0 p.1 …
    -/
    have : ∀ i, i < m.1 → i < n.1 := fun i hi => lt_of_lt_of_le hi hmn.1
    /-
      case h
      m n : Prod Nat Nat
      hmn : LE.le m n
      a : Sigma fun n => Composition n
      ha : Membership.mem ((fun p => FormalMultilinearSeries.compPartialSumTarget 0  …
      this : ∀ (i : Nat), LT.lt i m.1 → LT.lt i n.1
      ⊢ Membership.mem ((fun p => FormalMultilinearSeries.compPartialSumTarget 0 p.1 …
    -/
    have : ∀ i, i < m.2 → i < n.2 := fun i hi => lt_of_lt_of_le hi hmn.2
    /-
      case h
      m n : Prod Nat Nat
      hmn : LE.le m n
      a : Sigma fun n => Composition n
      ha : Membership.mem ((fun p => FormalMultilinearSeries.compPartialSumTarget 0  …
      this✝ : ∀ (i : Nat), LT.lt i m.1 → LT.lt i n.1
      this : ∀ (i : Nat), LT.lt i m.2 → LT.lt i n.2
      ⊢ Membership.mem ((fun p => FormalMultilinearSeries.compPartialSumTarget 0 p.1 …
    -/
    aesop
    /-
      🎉 no goals
    -/
    /-
      case h'
      ⊢ ∀ (x : Sigma fun n => Composition n), Exists fun n => Membership.mem (Formal …
    -/
  · rintro ⟨n, c⟩
    /-
      case h'.mk
      n : Nat
      c : Composition n
      ⊢ Exists fun n_1 => Membership.mem (FormalMultilinearSeries.compPartialSumTarg …
    -/
    simp only [mem_compPartialSumTarget_iff]
    obtain ⟨n, hn⟩ : BddAbove ((Finset.univ.image fun i : Fin c.length => c.blocksFun i) : Set ℕ) :=
      Finset.bddAbove _
    refine
      ⟨max n c.length + 1, bot_le, lt_of_le_of_lt (le_max_right n c.length) (lt_add_one _), fun j =>
        lt_of_le_of_lt (le_trans ?_ (le_max_left _ _)) (lt_add_one _)⟩
    /-
      case h'.mk.intro
      n✝ : Nat
      c : Composition n✝
      n : Nat
      hn : Membership.mem (upperBounds ↑(Finset.image (fun i => c.blocksFun i) Finse …
      j : Fin c.length
      ⊢ LE.le (c.blocksFun j) n
    -/
    apply hn
    /-
      case h'.mk.intro.a
      n✝ : Nat
      c : Composition n✝
      n : Nat
      hn : Membership.mem (upperBounds ↑(Finset.image (fun i => c.blocksFun i) Finse …
      j : Fin c.length
      ⊢ Membership.mem (↑(Finset.image (fun i => c.blocksFun i) Finset.univ)) (c.blo …
    -/
    simp only [Finset.mem_image_of_mem, Finset.mem_coe, Finset.mem_univ]
    /-
      🎉 no goals
    -/


/-- The auxiliary set corresponding to the composition of partial sums asymptotically contains
all possible compositions. -/
theorem compPartialSumTarget_tendsto_atTop :
    Tendsto (fun N => compPartialSumTarget 0 N N) atTop atTop := by
  /-
    ⊢ Filter.Tendsto (fun N => FormalMultilinearSeries.compPartialSumTarget 0 N N) …
  -/
  apply Tendsto.comp compPartialSumTarget_tendsto_prod_atTop tendsto_atTop_diagonal
  /-
    🎉 no goals
  -/


/-- Composing the partial sums of two multilinear series coincides with the sum over all
compositions in `compPartialSumTarget 0 N N`. This is precisely the motivation for the
definition of `compPartialSumTarget`. -/
theorem comp_partialSum (q : FormalMultilinearSeries 𝕜 F G) (p : FormalMultilinearSeries 𝕜 E F)
    (M N : ℕ) (z : E) :
    q.partialSum M (∑ i ∈ Finset.Ico 1 N, p i fun _j => z) =
      ∑ i ∈ compPartialSumTarget 0 M N, q.compAlongComposition p i.2 fun _j => z := by
  -- we expand the composition, using the multilinearity of `q` to expand along each coordinate.
  suffices H :
    (∑ n ∈ Finset.range M,
        ∑ r ∈ Fintype.piFinset fun i : Fin n => Finset.Ico 1 N,
          q n fun i : Fin n => p (r i) fun _j => z) =
      ∑ i ∈ compPartialSumTarget 0 M N, q.compAlongComposition p i.2 fun _j => z by
    simpa only [FormalMultilinearSeries.partialSum, ContinuousMultilinearMap.map_sum_finset] using H
  -- rewrite the first sum as a big sum over a sigma type, in the finset
  -- `compPartialSumTarget 0 N N`
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    M N : Nat
    z : E
    ⊢ Eq ((Finset.range M).sum fun n => (Fintype.piFinset fun i => Finset.Ico 1 N) …
  -/
  rw [Finset.range_eq_Ico, Finset.sum_sigma']
  -- use `compChangeOfVariables_sum`, saying that this change of variables respects sums
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    M N : Nat
    z : E
    ⊢ Eq (((Finset.Ico 0 M).sigma fun n => Fintype.piFinset fun i => Finset.Ico 1  …
  -/
  apply compChangeOfVariables_sum 0 M N
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    M N : Nat
    z : E
    ⊢ ∀ (e : Sigma fun n => Fin n → Nat) (he : Membership.mem (FormalMultilinearSe …
  -/
  rintro ⟨k, blocks_fun⟩ H
  /-
    case h.mk
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    M N : Nat
    z : E
    k : Nat
    blocks_fun : Fin k → Nat
    H : Membership.mem (FormalMultilinearSeries.compPartialSumSource 0 M N) ⟨k, bl …
    ⊢ Eq ((q ⟨k, blocks_fun⟩.fst) fun i => (p (⟨k, blocks_fun⟩.snd i)) fun _j => z …
  -/
  apply congr _ (compChangeOfVariables_length 0 M N H).symm
  /-
    case h.mk
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    M N : Nat
    z : E
    k : Nat
    blocks_fun : Fin k → Nat
    H : Membership.mem (FormalMultilinearSeries.compPartialSumSource 0 M N) ⟨k, bl …
    ⊢ ∀ (i : Nat) (him : LT.lt i ⟨k, blocks_fun⟩.fst) (hin : LT.lt i (FormalMultil …
  -/
  intros
  /-
    case h.mk
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    M N : Nat
    z : E
    k : Nat
    blocks_fun : Fin k → Nat
    H : Membership.mem (FormalMultilinearSeries.compPartialSumSource 0 M N) ⟨k, bl …
    i✝ : Nat
    him✝ : LT.lt i✝ ⟨k, blocks_fun⟩.fst
    hin✝ : LT.lt i✝ (FormalMultilinearSeries.compChangeOfVariables 0 M N ⟨k, block …
    ⊢ Eq ((p (⟨k, blocks_fun⟩.snd ⟨i✝, him✝⟩)) fun _j => z) (p.applyComposition (F …
  -/
  rw [← compChangeOfVariables_blocksFun 0 M N H]
  /-
    case h.mk
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    M N : Nat
    z : E
    k : Nat
    blocks_fun : Fin k → Nat
    H : Membership.mem (FormalMultilinearSeries.compPartialSumSource 0 M N) ⟨k, bl …
    i✝ : Nat
    him✝ : LT.lt i✝ ⟨k, blocks_fun⟩.fst
    hin✝ : LT.lt i✝ (FormalMultilinearSeries.compChangeOfVariables 0 M N ⟨k, block …
    ⊢ Eq ((p ((FormalMultilinearSeries.compChangeOfVariables 0 M N ⟨k, blocks_fun⟩ …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- If two functions `g` and `f` have power series `q` and `p` respectively at `f x` and `x`, within
two sets `s` and `t` such that `f` maps `s` to `t`, then `g ∘ f` admits the power
series `q.comp p` at `x` within `s`. -/
theorem HasFPowerSeriesWithinAt.comp {g : F → G} {f : E → F} {q : FormalMultilinearSeries 𝕜 F G}
    {p : FormalMultilinearSeries 𝕜 E F} {x : E} {t : Set F} {s : Set E}
    (hg : HasFPowerSeriesWithinAt g q t (f x)) (hf : HasFPowerSeriesWithinAt f p s x)
    (hs : Set.MapsTo f s t) : HasFPowerSeriesWithinAt (g ∘ f) (q.comp p) s x := by
  /- Consider `rf` and `rg` such that `f` and `g` have power series expansion on the disks
    of radius `rf` and `rg`. -/
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    g : F → G
    f : E → F
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    x : E
    t : Set F
    s : Set E
    hg : HasFPowerSeriesWithinAt g q t (f x)
    hf : HasFPowerSeriesWithinAt f p s x
    hs : Set.MapsTo f s t
    ⊢ HasFPowerSeriesWithinAt (Function.comp g f) (q.comp p) s x
  -/
  rcases hg with ⟨rg, Hg⟩
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    g : F → G
    f : E → F
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    x : E
    t : Set F
    s : Set E
    hf : HasFPowerSeriesWithinAt f p s x
    hs : Set.MapsTo f s t
    rg : ENNReal
    Hg : HasFPowerSeriesWithinOnBall g q t (f x) rg
    ⊢ HasFPowerSeriesWithinAt (Function.comp g f) (q.comp p) s x
  -/
  rcases hf with ⟨rf, Hf⟩
  -- The terms defining `q.comp p` are geometrically summable in a disk of some radius `r`.
  /-
    case intro.intro
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    g : F → G
    f : E → F
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    x : E
    t : Set F
    s : Set E
    hs : Set.MapsTo f s t
    rg : ENNReal
    Hg : HasFPowerSeriesWithinOnBall g q t (f x) rg
    rf : ENNReal
    Hf : HasFPowerSeriesWithinOnBall f p s x rf
    ⊢ HasFPowerSeriesWithinAt (Function.comp g f) (q.comp p) s x
  -/
  rcases q.comp_summable_nnreal p Hg.radius_pos Hf.radius_pos with ⟨r, r_pos : 0 < r, hr⟩
  /- We will consider `y` which is smaller than `r` and `rf`, and also small enough that
    `f (x + y)` is close enough to `f x` to be in the disk where `g` is well behaved. Let
    `min (r, rf, δ)` be this new radius. -/
  obtain ⟨δ, δpos, hδ⟩ :
    ∃ δ : ℝ≥0∞, 0 < δ ∧ ∀ {z : E}, z ∈ insert x s ∩ EMetric.ball x δ
      → f z ∈ insert (f x) t ∩ EMetric.ball (f x) rg := by
    have : insert (f x) t ∩ EMetric.ball (f x) rg ∈ 𝓝[insert (f x) t] (f x) := by
      apply inter_mem_nhdsWithin
      exact EMetric.ball_mem_nhds _ Hg.r_pos
    have := Hf.analyticWithinAt.continuousWithinAt_insert.tendsto_nhdsWithin (hs.insert x) this
    rcases EMetric.mem_nhdsWithin_iff.1 this with ⟨δ, δpos, Hδ⟩
    exact ⟨δ, δpos, fun {z} hz => Hδ (by rwa [Set.inter_comm])⟩
  /-
    case intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    g : F → G
    f : E → F
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    x : E
    t : Set F
    s : Set E
    hs : Set.MapsTo f s t
    rg : ENNReal
    Hg : HasFPowerSeriesWithinOnBall g q t (f x) rg
    rf : ENNReal
    Hf : HasFPowerSeriesWithinOnBall f p s x rf
    r : NNReal
    r_pos : LT.lt 0 r
    hr : Summable fun i => HMul.hMul (NNNorm.nnnorm (q.compAlongComposition p i.sn …
    δ : ENNReal
    δpos : LT.lt 0 δ
    hδ : ∀ {z : E}, Membership.mem (Inter.inter (Insert.insert x s) (EMetric.ball  …
    ⊢ HasFPowerSeriesWithinAt (Function.comp g f) (q.comp p) s x
  -/
  let rf' := min rf δ
  have min_pos : 0 < min rf' r := by
    simp only [rf', r_pos, Hf.r_pos, δpos, lt_min_iff, ENNReal.coe_pos, and_self_iff]
  /- We will show that `g ∘ f` admits the power series `q.comp p` in the disk of
    radius `min (r, rf', δ)`. -/
  /-
    case intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    g : F → G
    f : E → F
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    x : E
    t : Set F
    s : Set E
    hs : Set.MapsTo f s t
    rg : ENNReal
    Hg : HasFPowerSeriesWithinOnBall g q t (f x) rg
    rf : ENNReal
    Hf : HasFPowerSeriesWithinOnBall f p s x rf
    r : NNReal
    r_pos : LT.lt 0 r
    hr : Summable fun i => HMul.hMul (NNNorm.nnnorm (q.compAlongComposition p i.sn …
    δ : ENNReal
    δpos : LT.lt 0 δ
    hδ : ∀ {z : E}, Membership.mem (Inter.inter (Insert.insert x s) (EMetric.ball  …
    rf' : ENNReal := Min.min rf δ
    min_pos : LT.lt 0 (Min.min rf' ↑r)
    ⊢ HasFPowerSeriesWithinAt (Function.comp g f) (q.comp p) s x
  -/
  refine ⟨min rf' r, ?_⟩
  refine
    ⟨le_trans (min_le_right rf' r) (FormalMultilinearSeries.le_comp_radius_of_summable q p r hr),
      min_pos, fun {y} h'y hy ↦ ?_⟩
  /- Let `y` satisfy `‖y‖ < min (r, rf', δ)`. We want to show that `g (f (x + y))` is the sum of
    `q.comp p` applied to `y`. -/
  -- First, check that `y` is small enough so that estimates for `f` and `g` apply.
  have y_mem : y ∈ EMetric.ball (0 : E) rf :=
    (EMetric.ball_subset_ball (le_trans (min_le_left _ _) (min_le_left _ _))) hy
  have fy_mem : f (x + y) ∈ insert (f x) t ∩ EMetric.ball (f x) rg := by
    apply hδ
    have : y ∈ EMetric.ball (0 : E) δ :=
      (EMetric.ball_subset_ball (le_trans (min_le_left _ _) (min_le_right _ _))) hy
    simpa [-Set.mem_insert_iff, edist_eq_coe_nnnorm_sub, h'y]
  /- Now the proof starts. To show that the sum of `q.comp p` at `y` is `g (f (x + y))`,
    we will write `q.comp p` applied to `y` as a big sum over all compositions.
    Since the sum is summable, to get its convergence it suffices to get
    the convergence along some increasing sequence of sets.
    We will use the sequence of sets `compPartialSumTarget 0 n n`,
    along which the sum is exactly the composition of the partial sums of `q` and `p`, by design.
    To show that it converges to `g (f (x + y))`, pointwise convergence would not be enough,
    but we have uniform convergence to save the day. -/
  -- First step: the partial sum of `p` converges to `f (x + y)`.
  have A : Tendsto (fun n ↦ (n, ∑ a ∈ Finset.Ico 1 n, p a fun _ ↦ y))
      atTop (atTop ×ˢ 𝓝 (f (x + y) - f x)) := by
    apply Tendsto.prod_mk tendsto_id
    have L : ∀ᶠ n in atTop, (∑ a ∈ Finset.range n, p a fun _b ↦ y) - f x
        = ∑ a ∈ Finset.Ico 1 n, p a fun _b ↦ y := by
      rw [eventually_atTop]
      refine ⟨1, fun n hn => ?_⟩
      symm
      rw [eq_sub_iff_add_eq', Finset.range_eq_Ico, ← Hf.coeff_zero fun _i => y,
        Finset.sum_eq_sum_Ico_succ_bot hn]
    have :
      Tendsto (fun n => (∑ a ∈ Finset.range n, p a fun _b => y) - f x) atTop
        (𝓝 (f (x + y) - f x)) :=
      (Hf.hasSum h'y y_mem).tendsto_sum_nat.sub tendsto_const_nhds
    exact Tendsto.congr' L this
  -- Second step: the composition of the partial sums of `q` and `p` converges to `g (f (x + y))`.
  have B : Tendsto (fun n => q.partialSum n (∑ a ∈ Finset.Ico 1 n, p a fun _b ↦ y)) atTop
      (𝓝 (g (f (x + y)))) := by
    -- we use the fact that the partial sums of `q` converge to `g (f (x + y))`, uniformly on a
    -- neighborhood of `f (x + y)`.
    have : Tendsto (fun (z : ℕ × F) ↦ q.partialSum z.1 z.2)
        (atTop ×ˢ 𝓝 (f (x + y) - f x)) (𝓝 (g (f x + (f (x + y) - f x)))) := by
      apply Hg.tendsto_partialSum_prod (y := f (x + y) - f x)
      · simpa [edist_eq_coe_nnnorm_sub] using fy_mem.2
      · simpa using fy_mem.1
    simpa using this.comp A
  -- Third step: the sum over all compositions in `compPartialSumTarget 0 n n` converges to
  -- `g (f (x + y))`. As this sum is exactly the composition of the partial sum, this is a direct
  -- consequence of the second step
  have C :
    Tendsto
      (fun n => ∑ i ∈ compPartialSumTarget 0 n n, q.compAlongComposition p i.2 fun _j => y)
      atTop (𝓝 (g (f (x + y)))) := by
    simpa [comp_partialSum] using B
  -- Fourth step: the sum over all compositions is `g (f (x + y))`. This follows from the
  -- convergence along a subsequence proved in the third step, and the fact that the sum is Cauchy
  -- thanks to the summability properties.
  have D :
    HasSum (fun i : Σ n, Composition n => q.compAlongComposition p i.2 fun _j => y)
      (g (f (x + y))) :=
    haveI cau :
      CauchySeq fun s : Finset (Σ n, Composition n) =>
        ∑ i ∈ s, q.compAlongComposition p i.2 fun _j => y := by
      apply cauchySeq_finset_of_norm_bounded _ (NNReal.summable_coe.2 hr) _
      simp only [coe_nnnorm, NNReal.coe_mul, NNReal.coe_pow]
      rintro ⟨n, c⟩
      calc
        ‖(compAlongComposition q p c) fun _j : Fin n => y‖ ≤
            ‖compAlongComposition q p c‖ * ∏ _j : Fin n, ‖y‖ := by
          apply ContinuousMultilinearMap.le_opNorm
        _ ≤ ‖compAlongComposition q p c‖ * (r : ℝ) ^ n := by
          apply mul_le_mul_of_nonneg_left _ (norm_nonneg _)
          rw [Finset.prod_const, Finset.card_fin]
          gcongr
          rw [EMetric.mem_ball, edist_eq_coe_nnnorm] at hy
          have := le_trans (le_of_lt hy) (min_le_right _ _)
          rwa [ENNReal.coe_le_coe, ← NNReal.coe_le_coe, coe_nnnorm] at this
    tendsto_nhds_of_cauchySeq_of_subseq cau compPartialSumTarget_tendsto_atTop C
  -- Fifth step: the sum over `n` of `q.comp p n` can be expressed as a particular resummation of
  -- the sum over all compositions, by grouping together the compositions of the same
  -- integer `n`. The convergence of the whole sum therefore implies the converence of the sum
  -- of `q.comp p n`
  have E : HasSum (fun n => (q.comp p) n fun _j => y) (g (f (x + y))) := by
    apply D.sigma
    intro n
    dsimp [FormalMultilinearSeries.comp]
    convert hasSum_fintype (α := G) (β := Composition n) _
    simp only [ContinuousMultilinearMap.sum_apply]
    rfl
  /-
    case intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E✝ : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E✝
    inst✝⁴ : NormedSpace 𝕜 E✝
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    g : F → G
    f : E✝ → F
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E✝ F
    x : E✝
    t : Set F
    s : Set E✝
    hs : Set.MapsTo f s t
    rg : ENNReal
    Hg : HasFPowerSeriesWithinOnBall g q t (f x) rg
    rf : ENNReal
    Hf : HasFPowerSeriesWithinOnBall f p s x rf
    r : NNReal
    r_pos : LT.lt 0 r
    hr : Summable fun i => HMul.hMul (NNNorm.nnnorm (q.compAlongComposition p i.sn …
    δ : ENNReal
    δpos : LT.lt 0 δ
    hδ : ∀ {z : E✝}, Membership.mem (Inter.inter (Insert.insert x s) (EMetric.ball …
    rf' : ENNReal := Min.min rf δ
    min_pos : LT.lt 0 (Min.min rf' ↑r)
    y : E✝
    h'y : Membership.mem (Insert.insert x s) (HAdd.hAdd x y)
    hy : Membership.mem (EMetric.ball 0 (Min.min rf' ↑r)) y
    y_mem : Membership.mem (EMetric.ball 0 rf) y
    fy_mem : Membership.mem (Inter.inter (Insert.insert (f x) t) (EMetric.ball (f  …
    A : Filter.Tendsto (fun n => { fst := n, snd := (Finset.Ico 1 n).sum fun a =>  …
    B : Filter.Tendsto (fun n => q.partialSum n ((Finset.Ico 1 n).sum fun a => (p  …
    C : Filter.Tendsto (fun n => (FormalMultilinearSeries.compPartialSumTarget 0 n …
    D : HasSum (fun i => (q.compAlongComposition p i.snd) fun _j => y) (g (f (HAdd …
    E : HasSum (fun n => (q.comp p n) fun _j => y) (g (f (HAdd.hAdd x y)))
    ⊢ HasSum (fun n => (q.comp p n) fun x => y) (Function.comp g f (HAdd.hAdd x y))
  -/
  rw [Function.comp_apply]
  /-
    case intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E✝ : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E✝
    inst✝⁴ : NormedSpace 𝕜 E✝
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    g : F → G
    f : E✝ → F
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E✝ F
    x : E✝
    t : Set F
    s : Set E✝
    hs : Set.MapsTo f s t
    rg : ENNReal
    Hg : HasFPowerSeriesWithinOnBall g q t (f x) rg
    rf : ENNReal
    Hf : HasFPowerSeriesWithinOnBall f p s x rf
    r : NNReal
    r_pos : LT.lt 0 r
    hr : Summable fun i => HMul.hMul (NNNorm.nnnorm (q.compAlongComposition p i.sn …
    δ : ENNReal
    δpos : LT.lt 0 δ
    hδ : ∀ {z : E✝}, Membership.mem (Inter.inter (Insert.insert x s) (EMetric.ball …
    rf' : ENNReal := Min.min rf δ
    min_pos : LT.lt 0 (Min.min rf' ↑r)
    y : E✝
    h'y : Membership.mem (Insert.insert x s) (HAdd.hAdd x y)
    hy : Membership.mem (EMetric.ball 0 (Min.min rf' ↑r)) y
    y_mem : Membership.mem (EMetric.ball 0 rf) y
    fy_mem : Membership.mem (Inter.inter (Insert.insert (f x) t) (EMetric.ball (f  …
    A : Filter.Tendsto (fun n => { fst := n, snd := (Finset.Ico 1 n).sum fun a =>  …
    B : Filter.Tendsto (fun n => q.partialSum n ((Finset.Ico 1 n).sum fun a => (p  …
    C : Filter.Tendsto (fun n => (FormalMultilinearSeries.compPartialSumTarget 0 n …
    D : HasSum (fun i => (q.compAlongComposition p i.snd) fun _j => y) (g (f (HAdd …
    E : HasSum (fun n => (q.comp p n) fun _j => y) (g (f (HAdd.hAdd x y)))
    ⊢ HasSum (fun n => (q.comp p n) fun x => y) (g (f (HAdd.hAdd x y)))
  -/
  exact E
  /-
    🎉 no goals
  -/


/-- If two functions `g` and `f` have power series `q` and `p` respectively at `f x` and `x`,
then `g ∘ f` admits the power  series `q.comp p` at `x` within `s`. -/
theorem HasFPowerSeriesAt.comp {g : F → G} {f : E → F} {q : FormalMultilinearSeries 𝕜 F G}
    {p : FormalMultilinearSeries 𝕜 E F} {x : E}
    (hg : HasFPowerSeriesAt g q (f x)) (hf : HasFPowerSeriesAt f p x) :
    HasFPowerSeriesAt (g ∘ f) (q.comp p) x := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    g : F → G
    f : E → F
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    x : E
    hg : HasFPowerSeriesAt g q (f x)
    hf : HasFPowerSeriesAt f p x
    ⊢ HasFPowerSeriesAt (Function.comp g f) (q.comp p) x
  -/
  rw [← hasFPowerSeriesWithinAt_univ] at hf hg ⊢
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    g : F → G
    f : E → F
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    x : E
    hg : HasFPowerSeriesWithinAt g q Set.univ (f x)
    hf : HasFPowerSeriesWithinAt f p Set.univ x
    ⊢ HasFPowerSeriesWithinAt (Function.comp g f) (q.comp p) Set.univ x
  -/
  apply hg.comp hf (by simp)
  /-
    🎉 no goals
  -/


/-- If two functions `g` and `f` are analytic respectively at `f x` and `x`, within
two sets `s` and `t` such that `f` maps `s` to `t`, then `g ∘ f` is analytic at `x` within `s`. -/
theorem AnalyticWithinAt.comp {g : F → G} {f : E → F} {x : E} {t : Set F} {s : Set E}
    (hg : AnalyticWithinAt 𝕜 g t (f x)) (hf : AnalyticWithinAt 𝕜 f s x) (h : Set.MapsTo f s t) :
    AnalyticWithinAt 𝕜 (g ∘ f) s x := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    g : F → G
    f : E → F
    x : E
    t : Set F
    s : Set E
    hg : AnalyticWithinAt 𝕜 g t (f x)
    hf : AnalyticWithinAt 𝕜 f s x
    h : Set.MapsTo f s t
    ⊢ AnalyticWithinAt 𝕜 (Function.comp g f) s x
  -/
  let ⟨_q, hq⟩ := hg
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    g : F → G
    f : E → F
    x : E
    t : Set F
    s : Set E
    hg : AnalyticWithinAt 𝕜 g t (f x)
    hf : AnalyticWithinAt 𝕜 f s x
    h : Set.MapsTo f s t
    _q : FormalMultilinearSeries 𝕜 F G
    hq : HasFPowerSeriesWithinAt g _q t (f x)
    ⊢ AnalyticWithinAt 𝕜 (Function.comp g f) s x
  -/
  let ⟨_p, hp⟩ := hf
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    g : F → G
    f : E → F
    x : E
    t : Set F
    s : Set E
    hg : AnalyticWithinAt 𝕜 g t (f x)
    hf : AnalyticWithinAt 𝕜 f s x
    h : Set.MapsTo f s t
    _q : FormalMultilinearSeries 𝕜 F G
    hq : HasFPowerSeriesWithinAt g _q t (f x)
    _p : FormalMultilinearSeries 𝕜 E F
    hp : HasFPowerSeriesWithinAt f _p s x
    ⊢ AnalyticWithinAt 𝕜 (Function.comp g f) s x
  -/
  exact (hq.comp hp h).analyticWithinAt
  /-
    🎉 no goals
  -/


/-- Version of `AnalyticWithinAt.comp` where point equality is a separate hypothesis. -/
theorem AnalyticWithinAt.comp_of_eq {g : F → G} {f : E → F} {y : F} {x : E} {t : Set F} {s : Set E}
    (hg : AnalyticWithinAt 𝕜 g t y) (hf : AnalyticWithinAt 𝕜 f s x) (h : Set.MapsTo f s t)
    (hy : f x = y) :
    AnalyticWithinAt 𝕜 (g ∘ f) s x := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    g : F → G
    f : E → F
    y : F
    x : E
    t : Set F
    s : Set E
    hg : AnalyticWithinAt 𝕜 g t y
    hf : AnalyticWithinAt 𝕜 f s x
    h : Set.MapsTo f s t
    hy : Eq (f x) y
    ⊢ AnalyticWithinAt 𝕜 (Function.comp g f) s x
  -/
  rw [← hy] at hg
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    g : F → G
    f : E → F
    y : F
    x : E
    t : Set F
    s : Set E
    hg : AnalyticWithinAt 𝕜 g t (f x)
    hf : AnalyticWithinAt 𝕜 f s x
    h : Set.MapsTo f s t
    hy : Eq (f x) y
    ⊢ AnalyticWithinAt 𝕜 (Function.comp g f) s x
  -/
  exact hg.comp hf h
  /-
    🎉 no goals
  -/


lemma AnalyticOn.comp {f : F → G} {g : E → F} {s : Set F}
    {t : Set E} (hf : AnalyticOn 𝕜 f s) (hg : AnalyticOn 𝕜 g t) (h : Set.MapsTo g t s) :
    AnalyticOn 𝕜 (f ∘ g) t :=
  fun x m ↦ (hf _ (h m)).comp (hg x m) h


@[deprecated (since := "2024-09-26")]
alias AnalyticWithinOn.comp := AnalyticOn.comp


/-- If two functions `g` and `f` are analytic respectively at `f x` and `x`, then `g ∘ f` is
analytic at `x`. -/
theorem AnalyticAt.comp {g : F → G} {f : E → F} {x : E} (hg : AnalyticAt 𝕜 g (f x))
    (hf : AnalyticAt 𝕜 f x) : AnalyticAt 𝕜 (g ∘ f) x := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    g : F → G
    f : E → F
    x : E
    hg : AnalyticAt 𝕜 g (f x)
    hf : AnalyticAt 𝕜 f x
    ⊢ AnalyticAt 𝕜 (Function.comp g f) x
  -/
  rw [← analyticWithinAt_univ] at hg hf ⊢
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    g : F → G
    f : E → F
    x : E
    hg : AnalyticWithinAt 𝕜 g Set.univ (f x)
    hf : AnalyticWithinAt 𝕜 f Set.univ x
    ⊢ AnalyticWithinAt 𝕜 (Function.comp g f) Set.univ x
  -/
  apply hg.comp hf (by simp)
  /-
    🎉 no goals
  -/


/-- Version of `AnalyticAt.comp` where point equality is a separate hypothesis. -/
theorem AnalyticAt.comp_of_eq {g : F → G} {f : E → F} {y : F} {x : E} (hg : AnalyticAt 𝕜 g y)
    (hf : AnalyticAt 𝕜 f x) (hy : f x = y) : AnalyticAt 𝕜 (g ∘ f) x := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    g : F → G
    f : E → F
    y : F
    x : E
    hg : AnalyticAt 𝕜 g y
    hf : AnalyticAt 𝕜 f x
    hy : Eq (f x) y
    ⊢ AnalyticAt 𝕜 (Function.comp g f) x
  -/
  rw [← hy] at hg
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    g : F → G
    f : E → F
    y : F
    x : E
    hg : AnalyticAt 𝕜 g (f x)
    hf : AnalyticAt 𝕜 f x
    hy : Eq (f x) y
    ⊢ AnalyticAt 𝕜 (Function.comp g f) x
  -/
  exact hg.comp hf
  /-
    🎉 no goals
  -/


theorem AnalyticAt.comp_analyticWithinAt {g : F → G} {f : E → F} {x : E} {s : Set E}
    (hg : AnalyticAt 𝕜 g (f x)) (hf : AnalyticWithinAt 𝕜 f s x) :
    AnalyticWithinAt 𝕜 (g ∘ f) s x := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    g : F → G
    f : E → F
    x : E
    s : Set E
    hg : AnalyticAt 𝕜 g (f x)
    hf : AnalyticWithinAt 𝕜 f s x
    ⊢ AnalyticWithinAt 𝕜 (Function.comp g f) s x
  -/
  rw [← analyticWithinAt_univ] at hg
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    g : F → G
    f : E → F
    x : E
    s : Set E
    hg : AnalyticWithinAt 𝕜 g Set.univ (f x)
    hf : AnalyticWithinAt 𝕜 f s x
    ⊢ AnalyticWithinAt 𝕜 (Function.comp g f) s x
  -/
  exact hg.comp hf (Set.mapsTo_univ _ _)
  /-
    🎉 no goals
  -/


theorem AnalyticAt.comp_analyticWithinAt_of_eq {g : F → G} {f : E → F} {x : E} {y : F} {s : Set E}
    (hg : AnalyticAt 𝕜 g y) (hf : AnalyticWithinAt 𝕜 f s x) (h : f x = y) :
    AnalyticWithinAt 𝕜 (g ∘ f) s x := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    g : F → G
    f : E → F
    x : E
    y : F
    s : Set E
    hg : AnalyticAt 𝕜 g y
    hf : AnalyticWithinAt 𝕜 f s x
    h : Eq (f x) y
    ⊢ AnalyticWithinAt 𝕜 (Function.comp g f) s x
  -/
  rw [← h] at hg
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    g : F → G
    f : E → F
    x : E
    y : F
    s : Set E
    hg : AnalyticAt 𝕜 g (f x)
    hf : AnalyticWithinAt 𝕜 f s x
    h : Eq (f x) y
    ⊢ AnalyticWithinAt 𝕜 (Function.comp g f) s x
  -/
  exact hg.comp_analyticWithinAt hf
  /-
    🎉 no goals
  -/


/-- If two functions `g` and `f` are analytic respectively on `s.image f` and `s`, then `g ∘ f` is
analytic on `s`. -/
theorem AnalyticOnNhd.comp' {s : Set E} {g : F → G} {f : E → F} (hg : AnalyticOnNhd 𝕜 g (s.image f))
    (hf : AnalyticOnNhd 𝕜 f s) : AnalyticOnNhd 𝕜 (g ∘ f) s :=
  fun z hz => (hg (f z) (Set.mem_image_of_mem f hz)).comp (hf z hz)


@[deprecated (since := "2024-09-26")]
alias AnalyticOn.comp' := AnalyticOnNhd.comp'


theorem AnalyticOnNhd.comp {s : Set E} {t : Set F} {g : F → G} {f : E → F}
    (hg : AnalyticOnNhd 𝕜 g t) (hf : AnalyticOnNhd 𝕜 f s) (st : Set.MapsTo f s t) :
    AnalyticOnNhd 𝕜 (g ∘ f) s :=
  comp' (mono hg (Set.mapsTo'.mp st)) hf


lemma AnalyticOnNhd.comp_analyticOn {f : F → G} {g : E → F} {s : Set F}
    {t : Set E} (hf : AnalyticOnNhd 𝕜 f s) (hg : AnalyticOn 𝕜 g t) (h : Set.MapsTo g t s) :
    AnalyticOn 𝕜 (f ∘ g) t :=
  fun x m ↦ (hf _ (h m)).comp_analyticWithinAt (hg x m)


@[deprecated (since := "2024-09-26")]
alias AnalyticOn.comp_analyticWithinOn := AnalyticOnNhd.comp_analyticOn


/-- Rewriting equality in the dependent type `Σ (a : Composition n), Composition a.length)` in
non-dependent terms with lists, requiring that the blocks coincide. -/
theorem sigma_composition_eq_iff (i j : Σ a : Composition n, Composition a.length) :
    i = j ↔ i.1.blocks = j.1.blocks ∧ i.2.blocks = j.2.blocks := by
  /-
    n : Nat
    i j : Sigma fun a => Composition a.length
    ⊢ Iff (Eq i j) (And (Eq i.fst.blocks j.fst.blocks) (Eq i.snd.blocks j.snd.bloc …
  -/
  refine ⟨by rintro rfl; exact ⟨rfl, rfl⟩, ?_⟩
  /-
    n : Nat
    i j : Sigma fun a => Composition a.length
    ⊢ And (Eq i.fst.blocks j.fst.blocks) (Eq i.snd.blocks j.snd.blocks) → Eq i j
  -/
  rcases i with ⟨a, b⟩
  /-
    case mk
    n : Nat
    j : Sigma fun a => Composition a.length
    a : Composition n
    b : Composition a.length
    ⊢ And (Eq ⟨a, b⟩.fst.blocks j.fst.blocks) (Eq ⟨a, b⟩.snd.blocks j.snd.blocks)  …
  -/
  rcases j with ⟨a', b'⟩
  /-
    case mk.mk
    n : Nat
    a : Composition n
    b : Composition a.length
    a' : Composition n
    b' : Composition a'.length
    ⊢ And (Eq ⟨a, b⟩.fst.blocks ⟨a', b'⟩.fst.blocks) (Eq ⟨a, b⟩.snd.blocks ⟨a', b' …
  -/
  rintro ⟨h, h'⟩
  /-
    case mk.mk.intro
    n : Nat
    a : Composition n
    b : Composition a.length
    a' : Composition n
    b' : Composition a'.length
    h : Eq ⟨a, b⟩.fst.blocks ⟨a', b'⟩.fst.blocks
    h' : Eq ⟨a, b⟩.snd.blocks ⟨a', b'⟩.snd.blocks
    ⊢ Eq ⟨a, b⟩ ⟨a', b'⟩
  -/
  have H : a = a' := by ext1; exact h
  /-
    case mk.mk.intro
    n : Nat
    a : Composition n
    b : Composition a.length
    a' : Composition n
    b' : Composition a'.length
    h : Eq ⟨a, b⟩.fst.blocks ⟨a', b'⟩.fst.blocks
    h' : Eq ⟨a, b⟩.snd.blocks ⟨a', b'⟩.snd.blocks
    H : Eq a a'
    ⊢ Eq ⟨a, b⟩ ⟨a', b'⟩
  -/
  induction H; congr; ext1; exact h'
                            /-
                              🎉 no goals
                            -/


/-- Rewriting equality in the dependent type
`Σ (c : Composition n), Π (i : Fin c.length), Composition (c.blocksFun i)` in
non-dependent terms with lists, requiring that the lists of blocks coincide. -/
theorem sigma_pi_composition_eq_iff
    (u v : Σ c : Composition n, ∀ i : Fin c.length, Composition (c.blocksFun i)) :
    u = v ↔ (ofFn fun i => (u.2 i).blocks) = ofFn fun i => (v.2 i).blocks := by
  /-
    n : Nat
    u v : Sigma fun c => (i : Fin c.length) → Composition (c.blocksFun i)
    ⊢ Iff (Eq u v) (Eq (List.ofFn fun i => (u.snd i).blocks) (List.ofFn fun i => ( …
  -/
  refine ⟨fun H => by rw [H], fun H => ?_⟩
  /-
    n : Nat
    u v : Sigma fun c => (i : Fin c.length) → Composition (c.blocksFun i)
    H : Eq (List.ofFn fun i => (u.snd i).blocks) (List.ofFn fun i => (v.snd i).blo …
    ⊢ Eq u v
  -/
  rcases u with ⟨a, b⟩
  /-
    case mk
    n : Nat
    v : Sigma fun c => (i : Fin c.length) → Composition (c.blocksFun i)
    a : Composition n
    b : (i : Fin a.length) → Composition (a.blocksFun i)
    H : Eq (List.ofFn fun i => (⟨a, b⟩.snd i).blocks) (List.ofFn fun i => (v.snd i …
    ⊢ Eq ⟨a, b⟩ v
  -/
  rcases v with ⟨a', b'⟩
  /-
    case mk.mk
    n : Nat
    a : Composition n
    b : (i : Fin a.length) → Composition (a.blocksFun i)
    a' : Composition n
    b' : (i : Fin a'.length) → Composition (a'.blocksFun i)
    H : Eq (List.ofFn fun i => (⟨a, b⟩.snd i).blocks) (List.ofFn fun i => (⟨a', b' …
    ⊢ Eq ⟨a, b⟩ ⟨a', b'⟩
  -/
  dsimp at H
  have h : a = a' := by
    ext1
    have :
      map List.sum (ofFn fun i : Fin (Composition.length a) => (b i).blocks) =
        map List.sum (ofFn fun i : Fin (Composition.length a') => (b' i).blocks) := by
      rw [H]
    simp only [map_ofFn] at this
    change
      (ofFn fun i : Fin (Composition.length a) => (b i).blocks.sum) =
        ofFn fun i : Fin (Composition.length a') => (b' i).blocks.sum at this
    simpa [Composition.blocks_sum, Composition.ofFn_blocksFun] using this
  /-
    case mk.mk
    n : Nat
    a : Composition n
    b : (i : Fin a.length) → Composition (a.blocksFun i)
    a' : Composition n
    b' : (i : Fin a'.length) → Composition (a'.blocksFun i)
    H : Eq (List.ofFn fun i => (b i).blocks) (List.ofFn fun i => (b' i).blocks)
    h : Eq a a'
    ⊢ Eq ⟨a, b⟩ ⟨a', b'⟩
  -/
  induction h
  /-
    case mk.mk.refl
    n : Nat
    a : Composition n
    b : (i : Fin a.length) → Composition (a.blocksFun i)
    a' : Composition n
    b' : (i : Fin a.length) → Composition (a.blocksFun i)
    H : Eq (List.ofFn fun i => (b i).blocks) (List.ofFn fun i => (b' i).blocks)
    ⊢ Eq ⟨a, b⟩ ⟨a, b'⟩
  -/
  ext1
    /-
      case mk.mk.refl.fst
      n : Nat
      a : Composition n
      b : (i : Fin a.length) → Composition (a.blocksFun i)
      a' : Composition n
      b' : (i : Fin a.length) → Composition (a.blocksFun i)
      H : Eq (List.ofFn fun i => (b i).blocks) (List.ofFn fun i => (b' i).blocks)
      ⊢ Eq ⟨a, b⟩.fst ⟨a, b'⟩.fst
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case mk.mk.refl.snd
      n : Nat
      a : Composition n
      b : (i : Fin a.length) → Composition (a.blocksFun i)
      a' : Composition n
      b' : (i : Fin a.length) → Composition (a.blocksFun i)
      H : Eq (List.ofFn fun i => (b i).blocks) (List.ofFn fun i => (b' i).blocks)
      ⊢ HEq ⟨a, b⟩.snd ⟨a, b'⟩.snd
    -/
  · simp only [heq_eq_eq, ofFn_inj] at H ⊢
    /-
      case mk.mk.refl.snd
      n : Nat
      a : Composition n
      b : (i : Fin a.length) → Composition (a.blocksFun i)
      a' : Composition n
      b' : (i : Fin a.length) → Composition (a.blocksFun i)
      H : Eq (fun i => (b i).blocks) fun i => (b' i).blocks
      ⊢ Eq b b'
    -/
    ext1 i
    /-
      case mk.mk.refl.snd.h
      n : Nat
      a : Composition n
      b : (i : Fin a.length) → Composition (a.blocksFun i)
      a' : Composition n
      b' : (i : Fin a.length) → Composition (a.blocksFun i)
      H : Eq (fun i => (b i).blocks) fun i => (b' i).blocks
      i : Fin a.length
      ⊢ Eq (b i) (b' i)
    -/
    ext1
    /-
      case mk.mk.refl.snd.h.blocks
      n : Nat
      a : Composition n
      b : (i : Fin a.length) → Composition (a.blocksFun i)
      a' : Composition n
      b' : (i : Fin a.length) → Composition (a.blocksFun i)
      H : Eq (fun i => (b i).blocks) fun i => (b' i).blocks
      i : Fin a.length
      ⊢ Eq (b i).blocks (b' i).blocks
    -/
    exact congrFun H i
    /-
      🎉 no goals
    -/


/-- When `a` is a composition of `n` and `b` is a composition of `a.length`, `a.gather b` is the
composition of `n` obtained by gathering all the blocks of `a` corresponding to a block of `b`.
For instance, if `a = [6, 5, 3, 5, 2]` and `b = [2, 3]`, one should gather together
the first two blocks of `a` and its last three blocks, giving `a.gather b = [11, 10]`. -/
def gather (a : Composition n) (b : Composition a.length) : Composition n where
  blocks := (a.blocks.splitWrtComposition b).map sum
  blocks_pos := by
    /-
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      H : Type u_5
      inst✝⁸ : NontriviallyNormedField 𝕜
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      inst✝³ : NormedAddCommGroup G
      inst✝² : NormedSpace 𝕜 G
      inst✝¹ : NormedAddCommGroup H
      inst✝ : NormedSpace 𝕜 H
      n : Nat
      a : Composition n
      b : Composition a.length
      ⊢ ∀ {i : Nat}, Membership.mem (List.map List.sum (a.blocks.splitWrtComposition …
    -/
    rw [forall_mem_map]
    /-
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      H : Type u_5
      inst✝⁸ : NontriviallyNormedField 𝕜
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      inst✝³ : NormedAddCommGroup G
      inst✝² : NormedSpace 𝕜 G
      inst✝¹ : NormedAddCommGroup H
      inst✝ : NormedSpace 𝕜 H
      n : Nat
      a : Composition n
      b : Composition a.length
      ⊢ ∀ (j : List Nat), Membership.mem (a.blocks.splitWrtComposition b) j → LT.lt  …
    -/
    intro j hj
    suffices H : ∀ i ∈ j, 1 ≤ i by calc
      0 < j.length := length_pos_of_mem_splitWrtComposition hj
      _ ≤ j.sum := length_le_sum_of_one_le _ H
    /-
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      H : Type u_5
      inst✝⁸ : NontriviallyNormedField 𝕜
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      inst✝³ : NormedAddCommGroup G
      inst✝² : NormedSpace 𝕜 G
      inst✝¹ : NormedAddCommGroup H
      inst✝ : NormedSpace 𝕜 H
      n : Nat
      a : Composition n
      b : Composition a.length
      j : List Nat
      hj : Membership.mem (a.blocks.splitWrtComposition b) j
      ⊢ ∀ (i : Nat), Membership.mem j i → LE.le 1 i
    -/
    intro i hi
    /-
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      H : Type u_5
      inst✝⁸ : NontriviallyNormedField 𝕜
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      inst✝³ : NormedAddCommGroup G
      inst✝² : NormedSpace 𝕜 G
      inst✝¹ : NormedAddCommGroup H
      inst✝ : NormedSpace 𝕜 H
      n : Nat
      a : Composition n
      b : Composition a.length
      j : List Nat
      hj : Membership.mem (a.blocks.splitWrtComposition b) j
      i : Nat
      hi : Membership.mem j i
      ⊢ LE.le 1 i
    -/
    apply a.one_le_blocks
    /-
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      H : Type u_5
      inst✝⁸ : NontriviallyNormedField 𝕜
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      inst✝³ : NormedAddCommGroup G
      inst✝² : NormedSpace 𝕜 G
      inst✝¹ : NormedAddCommGroup H
      inst✝ : NormedSpace 𝕜 H
      n : Nat
      a : Composition n
      b : Composition a.length
      j : List Nat
      hj : Membership.mem (a.blocks.splitWrtComposition b) j
      i : Nat
      hi : Membership.mem j i
      ⊢ Membership.mem a.blocks i
    -/
    rw [← a.blocks.flatten_splitWrtComposition b]
    /-
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      H : Type u_5
      inst✝⁸ : NontriviallyNormedField 𝕜
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      inst✝³ : NormedAddCommGroup G
      inst✝² : NormedSpace 𝕜 G
      inst✝¹ : NormedAddCommGroup H
      inst✝ : NormedSpace 𝕜 H
      n : Nat
      a : Composition n
      b : Composition a.length
      j : List Nat
      hj : Membership.mem (a.blocks.splitWrtComposition b) j
      i : Nat
      hi : Membership.mem j i
      ⊢ Membership.mem (a.blocks.splitWrtComposition b).flatten i
    -/
    exact mem_flatten_of_mem hj hi
    /-
      🎉 no goals
    -/
                   /-
                     𝕜 : Type u_1
                     E : Type u_2
                     F : Type u_3
                     G : Type u_4
                     H : Type u_5
                     inst✝⁸ : NontriviallyNormedField 𝕜
                     inst✝⁷ : NormedAddCommGroup E
                     inst✝⁶ : NormedSpace 𝕜 E
                     inst✝⁵ : NormedAddCommGroup F
                     inst✝⁴ : NormedSpace 𝕜 F
                     inst✝³ : NormedAddCommGroup G
                     inst✝² : NormedSpace 𝕜 G
                     inst✝¹ : NormedAddCommGroup H
                     inst✝ : NormedSpace 𝕜 H
                     n : Nat
                     a : Composition n
                     b : Composition a.length
                     ⊢ Eq (List.map List.sum (a.blocks.splitWrtComposition b)).sum n
                   -/
  blocks_sum := by rw [← sum_flatten, flatten_splitWrtComposition, a.blocks_sum]
                   /-
                     🎉 no goals
                   -/


theorem length_gather (a : Composition n) (b : Composition a.length) :
    length (a.gather b) = b.length :=
  show (map List.sum (a.blocks.splitWrtComposition b)).length = b.blocks.length by
    /-
      n : Nat
      a : Composition n
      b : Composition a.length
      ⊢ Eq (List.map List.sum (a.blocks.splitWrtComposition b)).length b.blocks.length
    -/
    rw [length_map, length_splitWrtComposition]
    /-
      🎉 no goals
    -/


/-- An auxiliary function used in the definition of `sigmaEquivSigmaPi` below, associating to
two compositions `a` of `n` and `b` of `a.length`, and an index `i` bounded by the length of
`a.gather b`, the subcomposition of `a` made of those blocks belonging to the `i`-th block of
`a.gather b`. -/
def sigmaCompositionAux (a : Composition n) (b : Composition a.length)
    (i : Fin (a.gather b).length) : Composition ((a.gather b).blocksFun i) where
  blocks :=
    (a.blocks.splitWrtComposition b)[i.val]'(by
      /-
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        G : Type u_4
        H : Type u_5
        inst✝⁸ : NontriviallyNormedField 𝕜
        inst✝⁷ : NormedAddCommGroup E
        inst✝⁶ : NormedSpace 𝕜 E
        inst✝⁵ : NormedAddCommGroup F
        inst✝⁴ : NormedSpace 𝕜 F
        inst✝³ : NormedAddCommGroup G
        inst✝² : NormedSpace 𝕜 G
        inst✝¹ : NormedAddCommGroup H
        inst✝ : NormedSpace 𝕜 H
        n : Nat
        a : Composition n
        b : Composition a.length
        i : Fin (a.gather b).length
        ⊢ LT.lt (↑i) (a.blocks.splitWrtComposition b).length
      -/
      rw [length_splitWrtComposition, ← length_gather]; exact i.2)
                                                        /-
                                                          🎉 no goals
                                                        -/
  blocks_pos {i} hi :=
    a.blocks_pos
      (by
        /-
          𝕜 : Type u_1
          E : Type u_2
          F : Type u_3
          G : Type u_4
          H : Type u_5
          inst✝⁸ : NontriviallyNormedField 𝕜
          inst✝⁷ : NormedAddCommGroup E
          inst✝⁶ : NormedSpace 𝕜 E
          inst✝⁵ : NormedAddCommGroup F
          inst✝⁴ : NormedSpace 𝕜 F
          inst✝³ : NormedAddCommGroup G
          inst✝² : NormedSpace 𝕜 G
          inst✝¹ : NormedAddCommGroup H
          inst✝ : NormedSpace 𝕜 H
          n : Nat
          a : Composition n
          b : Composition a.length
          i✝ : Fin (a.gather b).length
          i : Nat
          hi : Membership.mem (GetElem.getElem (a.blocks.splitWrtComposition b) ↑i✝ ⋯) i
          ⊢ Membership.mem a.blocks i
        -/
        rw [← a.blocks.flatten_splitWrtComposition b]
        /-
          𝕜 : Type u_1
          E : Type u_2
          F : Type u_3
          G : Type u_4
          H : Type u_5
          inst✝⁸ : NontriviallyNormedField 𝕜
          inst✝⁷ : NormedAddCommGroup E
          inst✝⁶ : NormedSpace 𝕜 E
          inst✝⁵ : NormedAddCommGroup F
          inst✝⁴ : NormedSpace 𝕜 F
          inst✝³ : NormedAddCommGroup G
          inst✝² : NormedSpace 𝕜 G
          inst✝¹ : NormedAddCommGroup H
          inst✝ : NormedSpace 𝕜 H
          n : Nat
          a : Composition n
          b : Composition a.length
          i✝ : Fin (a.gather b).length
          i : Nat
          hi : Membership.mem (GetElem.getElem (a.blocks.splitWrtComposition b) ↑i✝ ⋯) i
          ⊢ Membership.mem (a.blocks.splitWrtComposition b).flatten i
        -/
        exact mem_flatten_of_mem (List.getElem_mem _) hi)
        /-
          🎉 no goals
        -/
                   /-
                     𝕜 : Type u_1
                     E : Type u_2
                     F : Type u_3
                     G : Type u_4
                     H : Type u_5
                     inst✝⁸ : NontriviallyNormedField 𝕜
                     inst✝⁷ : NormedAddCommGroup E
                     inst✝⁶ : NormedSpace 𝕜 E
                     inst✝⁵ : NormedAddCommGroup F
                     inst✝⁴ : NormedSpace 𝕜 F
                     inst✝³ : NormedAddCommGroup G
                     inst✝² : NormedSpace 𝕜 G
                     inst✝¹ : NormedAddCommGroup H
                     inst✝ : NormedSpace 𝕜 H
                     n : Nat
                     a : Composition n
                     b : Composition a.length
                     i : Fin (a.gather b).length
                     ⊢ Eq (GetElem.getElem (a.blocks.splitWrtComposition b) ↑i ⋯).sum ((a.gather b) …
                   -/
  blocks_sum := by simp [Composition.blocksFun, getElem_map, Composition.gather]
                   /-
                     🎉 no goals
                   -/


theorem length_sigmaCompositionAux (a : Composition n) (b : Composition a.length)
    (i : Fin b.length) :
    Composition.length (Composition.sigmaCompositionAux a b ⟨i, (length_gather a b).symm ▸ i.2⟩) =
      Composition.blocksFun b i :=
  show List.length ((splitWrtComposition a.blocks b)[i.1]) = blocksFun b i by
    /-
      n : Nat
      a : Composition n
      b : Composition a.length
      i : Fin b.length
      ⊢ Eq (GetElem.getElem (a.blocks.splitWrtComposition b) ↑i ⋯).length (b.blocksF …
    -/
    rw [getElem_map_rev List.length, getElem_of_eq (map_length_splitWrtComposition _ _)]; rfl
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


theorem blocksFun_sigmaCompositionAux (a : Composition n) (b : Composition a.length)
    (i : Fin b.length) (j : Fin (blocksFun b i)) :
    blocksFun (sigmaCompositionAux a b ⟨i, (length_gather a b).symm ▸ i.2⟩)
        ⟨j, (length_sigmaCompositionAux a b i).symm ▸ j.2⟩ =
      blocksFun a (embedding b i j) := by
  /-
    n : Nat
    a : Composition n
    b : Composition a.length
    i : Fin b.length
    j : Fin (b.blocksFun i)
    ⊢ Eq ((a.sigmaCompositionAux b ⟨↑i, ⋯⟩).blocksFun ⟨↑j, ⋯⟩) (a.blocksFun ((b.em …
  -/
  unfold sigmaCompositionAux
  rw [blocksFun, get_eq_getElem, getElem_of_eq (getElem_splitWrtComposition _ _ _ _),
                                 /-
                                   n : Nat
                                   a : Composition n
                                   b : Composition a.length
                                   i : Fin b.length
                                   j : Fin (b.blocksFun i)
                                   ⊢ Eq (GetElem.getElem a.blocks (HAdd.hAdd (b.sizeUpTo ↑⟨↑i, ⋯⟩) ↑⟨↑j, ⋯⟩) ⋯) ( …
                                 -/
    getElem_drop, getElem_take]; rfl
                                 /-
                                   🎉 no goals
                                 -/


/-- Auxiliary lemma to prove that the composition of formal multilinear series is associative.

Consider a composition `a` of `n` and a composition `b` of `a.length`. Grouping together some
blocks of `a` according to `b` as in `a.gather b`, one can compute the total size of the blocks
of `a` up to an index `sizeUpTo b i + j` (where the `j` corresponds to a set of blocks of `a`
that do not fill a whole block of `a.gather b`). The first part corresponds to a sum of blocks
in `a.gather b`, and the second one to a sum of blocks in the next block of
`sigmaCompositionAux a b`. This is the content of this lemma. -/
theorem sizeUpTo_sizeUpTo_add (a : Composition n) (b : Composition a.length) {i j : ℕ}
    (hi : i < b.length) (hj : j < blocksFun b ⟨i, hi⟩) :
    sizeUpTo a (sizeUpTo b i + j) =
      sizeUpTo (a.gather b) i +
        sizeUpTo (sigmaCompositionAux a b ⟨i, (length_gather a b).symm ▸ hi⟩) j := by
  -- Porting note: `induction'` left a spurious `hj` in the context
  induction j with
  | zero =>
    show
      sum (take (b.blocks.take i).sum a.blocks) =
        sum (take i (map sum (splitWrtComposition a.blocks b)))
    induction' i with i IH
    · rfl
    · have A : i < b.length := Nat.lt_of_succ_lt hi
      have B : i < List.length (map List.sum (splitWrtComposition a.blocks b)) := by simp [A]
      have C : 0 < blocksFun b ⟨i, A⟩ := Composition.blocks_pos' _ _ _
      rw [sum_take_succ _ _ B, ← IH A C]
      have :
        take (sum (take i b.blocks)) a.blocks =
          take (sum (take i b.blocks)) (take (sum (take (i + 1) b.blocks)) a.blocks) := by
        rw [take_take, min_eq_left]
        apply monotone_sum_take _ (Nat.le_succ _)
      rw [this, getElem_map, getElem_splitWrtComposition, ←
        take_append_drop (sum (take i b.blocks)) (take (sum (take (Nat.succ i) b.blocks)) a.blocks),
        sum_append]
      congr
      rw [take_append_drop]
  | succ j IHj =>
    have A : j < blocksFun b ⟨i, hi⟩ := lt_trans (lt_add_one j) hj
    have B : j < length (sigmaCompositionAux a b ⟨i, (length_gather a b).symm ▸ hi⟩) := by
      convert A; rw [← length_sigmaCompositionAux]
    have C : sizeUpTo b i + j < sizeUpTo b (i + 1) := by
      simp only [sizeUpTo_succ b hi, add_lt_add_iff_left]
      exact A
    have D : sizeUpTo b i + j < length a := lt_of_lt_of_le C (b.sizeUpTo_le _)
    have : sizeUpTo b i + Nat.succ j = (sizeUpTo b i + j).succ := rfl
    rw [this, sizeUpTo_succ _ D, IHj A, sizeUpTo_succ _ B]
    simp only [sigmaCompositionAux, add_assoc, add_left_inj, Fin.val_mk]
    rw [getElem_of_eq (getElem_splitWrtComposition _ _ _ _), getElem_drop, getElem_take' _ _ C]


/-- Natural equivalence between `(Σ (a : Composition n), Composition a.length)` and
`(Σ (c : Composition n), Π (i : Fin c.length), Composition (c.blocksFun i))`, that shows up as a
change of variables in the proof that composition of formal multilinear series is associative.

Consider a composition `a` of `n` and a composition `b` of `a.length`. Then `b` indicates how to
group together some blocks of `a`, giving altogether `b.length` blocks of blocks. These blocks of
blocks can be called `d₀, ..., d_{a.length - 1}`, and one obtains a composition `c` of `n` by
saying that each `dᵢ` is one single block. The map `⟨a, b⟩ → ⟨c, (d₀, ..., d_{a.length - 1})⟩` is
the direct map in the equiv.

Conversely, if one starts from `c` and the `dᵢ`s, one can join the `dᵢ`s to obtain a composition
`a` of `n`, and register the lengths of the `dᵢ`s in a composition `b` of `a.length`. This is the
inverse map of the equiv.
-/
def sigmaEquivSigmaPi (n : ℕ) :
    (Σ a : Composition n, Composition a.length) ≃
      Σ c : Composition n, ∀ i : Fin c.length, Composition (c.blocksFun i) where
  toFun i := ⟨i.1.gather i.2, i.1.sigmaCompositionAux i.2⟩
  invFun i :=
    ⟨{  blocks := (ofFn fun j => (i.2 j).blocks).flatten
        blocks_pos := by
          /-
            𝕜 : Type u_1
            E : Type u_2
            F : Type u_3
            G : Type u_4
            H : Type u_5
            inst✝⁸ : NontriviallyNormedField 𝕜
            inst✝⁷ : NormedAddCommGroup E
            inst✝⁶ : NormedSpace 𝕜 E
            inst✝⁵ : NormedAddCommGroup F
            inst✝⁴ : NormedSpace 𝕜 F
            inst✝³ : NormedAddCommGroup G
            inst✝² : NormedSpace 𝕜 G
            inst✝¹ : NormedAddCommGroup H
            inst✝ : NormedSpace 𝕜 H
            n✝ n : Nat
            i : Sigma fun c => (i : Fin c.length) → Composition (c.blocksFun i)
            ⊢ ∀ {i_1 : Nat}, Membership.mem (List.ofFn fun j => (i.snd j).blocks).flatten  …
          -/
          simp only [and_imp, List.mem_flatten, exists_imp, forall_mem_ofFn_iff]
          /-
            𝕜 : Type u_1
            E : Type u_2
            F : Type u_3
            G : Type u_4
            H : Type u_5
            inst✝⁸ : NontriviallyNormedField 𝕜
            inst✝⁷ : NormedAddCommGroup E
            inst✝⁶ : NormedSpace 𝕜 E
            inst✝⁵ : NormedAddCommGroup F
            inst✝⁴ : NormedSpace 𝕜 F
            inst✝³ : NormedAddCommGroup G
            inst✝² : NormedSpace 𝕜 G
            inst✝¹ : NormedAddCommGroup H
            inst✝ : NormedSpace 𝕜 H
            n✝ n : Nat
            i : Sigma fun c => (i : Fin c.length) → Composition (c.blocksFun i)
            ⊢ ∀ {i_1 : Nat} (j : Fin i.fst.length), Membership.mem (i.snd j).blocks i_1 →  …
          -/
          exact @fun i j hj => Composition.blocks_pos _ hj
          /-
            🎉 no goals
          -/
                         /-
                           𝕜 : Type u_1
                           E : Type u_2
                           F : Type u_3
                           G : Type u_4
                           H : Type u_5
                           inst✝⁸ : NontriviallyNormedField 𝕜
                           inst✝⁷ : NormedAddCommGroup E
                           inst✝⁶ : NormedSpace 𝕜 E
                           inst✝⁵ : NormedAddCommGroup F
                           inst✝⁴ : NormedSpace 𝕜 F
                           inst✝³ : NormedAddCommGroup G
                           inst✝² : NormedSpace 𝕜 G
                           inst✝¹ : NormedAddCommGroup H
                           inst✝ : NormedSpace 𝕜 H
                           n✝ n : Nat
                           i : Sigma fun c => (i : Fin c.length) → Composition (c.blocksFun i)
                           ⊢ Eq (List.ofFn fun j => (i.snd j).blocks).flatten.sum n
                         -/
        blocks_sum := by simp [sum_ofFn, Composition.blocks_sum, Composition.sum_blocksFun] },
                         /-
                           🎉 no goals
                         -/
      { blocks := ofFn fun j => (i.2 j).length
        blocks_pos := by
          /-
            𝕜 : Type u_1
            E : Type u_2
            F : Type u_3
            G : Type u_4
            H : Type u_5
            inst✝⁸ : NontriviallyNormedField 𝕜
            inst✝⁷ : NormedAddCommGroup E
            inst✝⁶ : NormedSpace 𝕜 E
            inst✝⁵ : NormedAddCommGroup F
            inst✝⁴ : NormedSpace 𝕜 F
            inst✝³ : NormedAddCommGroup G
            inst✝² : NormedSpace 𝕜 G
            inst✝¹ : NormedAddCommGroup H
            inst✝ : NormedSpace 𝕜 H
            n✝ n : Nat
            i : Sigma fun c => (i : Fin c.length) → Composition (c.blocksFun i)
            ⊢ ∀ {i_1 : Nat}, Membership.mem (List.ofFn fun j => (i.snd j).length) i_1 → LT …
          -/
          intro k hk
          /-
            𝕜 : Type u_1
            E : Type u_2
            F : Type u_3
            G : Type u_4
            H : Type u_5
            inst✝⁸ : NontriviallyNormedField 𝕜
            inst✝⁷ : NormedAddCommGroup E
            inst✝⁶ : NormedSpace 𝕜 E
            inst✝⁵ : NormedAddCommGroup F
            inst✝⁴ : NormedSpace 𝕜 F
            inst✝³ : NormedAddCommGroup G
            inst✝² : NormedSpace 𝕜 G
            inst✝¹ : NormedAddCommGroup H
            inst✝ : NormedSpace 𝕜 H
            n✝ n : Nat
            i : Sigma fun c => (i : Fin c.length) → Composition (c.blocksFun i)
            k : Nat
            hk : Membership.mem (List.ofFn fun j => (i.snd j).length) k
            ⊢ LT.lt 0 k
          -/
          refine ((forall_mem_ofFn_iff (P := fun i => 0 < i)).2 fun j => ?_) k hk
          /-
            𝕜 : Type u_1
            E : Type u_2
            F : Type u_3
            G : Type u_4
            H : Type u_5
            inst✝⁸ : NontriviallyNormedField 𝕜
            inst✝⁷ : NormedAddCommGroup E
            inst✝⁶ : NormedSpace 𝕜 E
            inst✝⁵ : NormedAddCommGroup F
            inst✝⁴ : NormedSpace 𝕜 F
            inst✝³ : NormedAddCommGroup G
            inst✝² : NormedSpace 𝕜 G
            inst✝¹ : NormedAddCommGroup H
            inst✝ : NormedSpace 𝕜 H
            n✝ n : Nat
            i : Sigma fun c => (i : Fin c.length) → Composition (c.blocksFun i)
            k : Nat
            hk : Membership.mem (List.ofFn fun j => (i.snd j).length) k
            j : Fin i.fst.length
            ⊢ LT.lt 0 (i.snd j).length
          -/
          exact Composition.length_pos_of_pos _ (Composition.blocks_pos' _ _ _)
          /-
            🎉 no goals
          -/
                         /-
                           𝕜 : Type u_1
                           E : Type u_2
                           F : Type u_3
                           G : Type u_4
                           H : Type u_5
                           inst✝⁸ : NontriviallyNormedField 𝕜
                           inst✝⁷ : NormedAddCommGroup E
                           inst✝⁶ : NormedSpace 𝕜 E
                           inst✝⁵ : NormedAddCommGroup F
                           inst✝⁴ : NormedSpace 𝕜 F
                           inst✝³ : NormedAddCommGroup G
                           inst✝² : NormedSpace 𝕜 G
                           inst✝¹ : NormedAddCommGroup H
                           inst✝ : NormedSpace 𝕜 H
                           n✝ n : Nat
                           i : Sigma fun c => (i : Fin c.length) → Composition (c.blocksFun i)
                           ⊢ Eq (List.ofFn fun j => (i.snd j).length).sum { blocks := (List.ofFn fun j => …
                         -/
        blocks_sum := by dsimp only [Composition.length]; simp [sum_ofFn] }⟩
                                                          /-
                                                            🎉 no goals
                                                          -/
  left_inv := by
    -- the fact that we have a left inverse is essentially `join_splitWrtComposition`,
    -- but we need to massage it to take care of the dependent setting.
    /-
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      H : Type u_5
      inst✝⁸ : NontriviallyNormedField 𝕜
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      inst✝³ : NormedAddCommGroup G
      inst✝² : NormedSpace 𝕜 G
      inst✝¹ : NormedAddCommGroup H
      inst✝ : NormedSpace 𝕜 H
      n✝ n : Nat
      ⊢ Function.LeftInverse (fun i => ⟨{ blocks := (List.ofFn fun j => (i.snd j).bl …
    -/
    rintro ⟨a, b⟩
    /-
      case mk
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      H : Type u_5
      inst✝⁸ : NontriviallyNormedField 𝕜
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      inst✝³ : NormedAddCommGroup G
      inst✝² : NormedSpace 𝕜 G
      inst✝¹ : NormedAddCommGroup H
      inst✝ : NormedSpace 𝕜 H
      n✝ n : Nat
      a : Composition n
      b : Composition a.length
      ⊢ Eq ((fun i => ⟨{ blocks := (List.ofFn fun j => (i.snd j).blocks).flatten, bl …
    -/
    rw [sigma_composition_eq_iff]
    /-
      case mk
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      H : Type u_5
      inst✝⁸ : NontriviallyNormedField 𝕜
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      inst✝³ : NormedAddCommGroup G
      inst✝² : NormedSpace 𝕜 G
      inst✝¹ : NormedAddCommGroup H
      inst✝ : NormedSpace 𝕜 H
      n✝ n : Nat
      a : Composition n
      b : Composition a.length
      ⊢ And (Eq ((fun i => ⟨{ blocks := (List.ofFn fun j => (i.snd j).blocks).flatte …
    -/
    dsimp
    /-
      case mk
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      H : Type u_5
      inst✝⁸ : NontriviallyNormedField 𝕜
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      inst✝³ : NormedAddCommGroup G
      inst✝² : NormedSpace 𝕜 G
      inst✝¹ : NormedAddCommGroup H
      inst✝ : NormedSpace 𝕜 H
      n✝ n : Nat
      a : Composition n
      b : Composition a.length
      ⊢ And (Eq (List.ofFn fun j => (a.sigmaCompositionAux b j).blocks).flatten a.bl …
    -/
    constructor
    · conv_rhs =>
        rw [← flatten_splitWrtComposition a.blocks b, ← ofFn_get (splitWrtComposition a.blocks b)]
      have A : length (gather a b) = List.length (splitWrtComposition a.blocks b) := by
        simp only [length, gather, length_map, length_splitWrtComposition]
      /-
        case mk.left
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        G : Type u_4
        H : Type u_5
        inst✝⁸ : NontriviallyNormedField 𝕜
        inst✝⁷ : NormedAddCommGroup E
        inst✝⁶ : NormedSpace 𝕜 E
        inst✝⁵ : NormedAddCommGroup F
        inst✝⁴ : NormedSpace 𝕜 F
        inst✝³ : NormedAddCommGroup G
        inst✝² : NormedSpace 𝕜 G
        inst✝¹ : NormedAddCommGroup H
        inst✝ : NormedSpace 𝕜 H
        n✝ n : Nat
        a : Composition n
        b : Composition a.length
        A : Eq (a.gather b).length (a.blocks.splitWrtComposition b).length
        ⊢ Eq (List.ofFn fun j => (a.sigmaCompositionAux b j).blocks).flatten (List.ofF …
      -/
      congr! 2
      /-
        case mk.left.h.e'_2.h.e'_3
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        G : Type u_4
        H : Type u_5
        inst✝⁸ : NontriviallyNormedField 𝕜
        inst✝⁷ : NormedAddCommGroup E
        inst✝⁶ : NormedSpace 𝕜 E
        inst✝⁵ : NormedAddCommGroup F
        inst✝⁴ : NormedSpace 𝕜 F
        inst✝³ : NormedAddCommGroup G
        inst✝² : NormedSpace 𝕜 G
        inst✝¹ : NormedAddCommGroup H
        inst✝ : NormedSpace 𝕜 H
        n✝ n : Nat
        a : Composition n
        b : Composition a.length
        A : Eq (a.gather b).length (a.blocks.splitWrtComposition b).length
        ⊢ HEq (fun j => (a.sigmaCompositionAux b j).blocks) (a.blocks.splitWrtComposit …
      -/
      exact (Fin.heq_fun_iff A (α := List ℕ)).2 fun i => rfl
      /-
        🎉 no goals
      -/
    · have B : Composition.length (Composition.gather a b) = List.length b.blocks :=
        Composition.length_gather _ _
      /-
        case mk.right
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        G : Type u_4
        H : Type u_5
        inst✝⁸ : NontriviallyNormedField 𝕜
        inst✝⁷ : NormedAddCommGroup E
        inst✝⁶ : NormedSpace 𝕜 E
        inst✝⁵ : NormedAddCommGroup F
        inst✝⁴ : NormedSpace 𝕜 F
        inst✝³ : NormedAddCommGroup G
        inst✝² : NormedSpace 𝕜 G
        inst✝¹ : NormedAddCommGroup H
        inst✝ : NormedSpace 𝕜 H
        n✝ n : Nat
        a : Composition n
        b : Composition a.length
        B : Eq (a.gather b).length b.blocks.length
        ⊢ Eq (List.ofFn fun j => (a.sigmaCompositionAux b j).length) b.blocks
      -/
      conv_rhs => rw [← ofFn_getElem b.blocks]
      /-
        case mk.right
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        G : Type u_4
        H : Type u_5
        inst✝⁸ : NontriviallyNormedField 𝕜
        inst✝⁷ : NormedAddCommGroup E
        inst✝⁶ : NormedSpace 𝕜 E
        inst✝⁵ : NormedAddCommGroup F
        inst✝⁴ : NormedSpace 𝕜 F
        inst✝³ : NormedAddCommGroup G
        inst✝² : NormedSpace 𝕜 G
        inst✝¹ : NormedAddCommGroup H
        inst✝ : NormedSpace 𝕜 H
        n✝ n : Nat
        a : Composition n
        b : Composition a.length
        B : Eq (a.gather b).length b.blocks.length
        ⊢ Eq (List.ofFn fun j => (a.sigmaCompositionAux b j).length) (List.ofFn fun i  …
      -/
      congr 1
      /-
        case mk.right.h.e_3
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        G : Type u_4
        H : Type u_5
        inst✝⁸ : NontriviallyNormedField 𝕜
        inst✝⁷ : NormedAddCommGroup E
        inst✝⁶ : NormedSpace 𝕜 E
        inst✝⁵ : NormedAddCommGroup F
        inst✝⁴ : NormedSpace 𝕜 F
        inst✝³ : NormedAddCommGroup G
        inst✝² : NormedSpace 𝕜 G
        inst✝¹ : NormedAddCommGroup H
        inst✝ : NormedSpace 𝕜 H
        n✝ n : Nat
        a : Composition n
        b : Composition a.length
        B : Eq (a.gather b).length b.blocks.length
        ⊢ HEq (fun j => (a.sigmaCompositionAux b j).length) fun i => GetElem.getElem b …
      -/
      refine (Fin.heq_fun_iff B).2 fun i => ?_
      rw [sigmaCompositionAux, Composition.length, List.getElem_map_rev List.length,
        List.getElem_of_eq (map_length_splitWrtComposition _ _)]
  right_inv := by
    -- the fact that we have a right inverse is essentially `splitWrtComposition_join`,
    -- but we need to massage it to take care of the dependent setting.
    /-
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      H : Type u_5
      inst✝⁸ : NontriviallyNormedField 𝕜
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      inst✝³ : NormedAddCommGroup G
      inst✝² : NormedSpace 𝕜 G
      inst✝¹ : NormedAddCommGroup H
      inst✝ : NormedSpace 𝕜 H
      n✝ n : Nat
      ⊢ Function.RightInverse (fun i => ⟨{ blocks := (List.ofFn fun j => (i.snd j).b …
    -/
    rintro ⟨c, d⟩
    have : map List.sum (ofFn fun i : Fin (Composition.length c) => (d i).blocks) = c.blocks := by
      simp [map_ofFn, Function.comp_def, Composition.blocks_sum, Composition.ofFn_blocksFun]
    /-
      case mk
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      H : Type u_5
      inst✝⁸ : NontriviallyNormedField 𝕜
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      inst✝³ : NormedAddCommGroup G
      inst✝² : NormedSpace 𝕜 G
      inst✝¹ : NormedAddCommGroup H
      inst✝ : NormedSpace 𝕜 H
      n✝ n : Nat
      c : Composition n
      d : (i : Fin c.length) → Composition (c.blocksFun i)
      this : Eq (List.map List.sum (List.ofFn fun i => (d i).blocks)) c.blocks
      ⊢ Eq ((fun i => ⟨i.fst.gather i.snd, i.fst.sigmaCompositionAux i.snd⟩) ((fun i …
    -/
    rw [sigma_pi_composition_eq_iff]
    /-
      case mk
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      H : Type u_5
      inst✝⁸ : NontriviallyNormedField 𝕜
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      inst✝³ : NormedAddCommGroup G
      inst✝² : NormedSpace 𝕜 G
      inst✝¹ : NormedAddCommGroup H
      inst✝ : NormedSpace 𝕜 H
      n✝ n : Nat
      c : Composition n
      d : (i : Fin c.length) → Composition (c.blocksFun i)
      this : Eq (List.map List.sum (List.ofFn fun i => (d i).blocks)) c.blocks
      ⊢ Eq (List.ofFn fun i => (((fun i => ⟨i.fst.gather i.snd, i.fst.sigmaCompositi …
    -/
    dsimp
    /-
      case mk
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      H : Type u_5
      inst✝⁸ : NontriviallyNormedField 𝕜
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      inst✝³ : NormedAddCommGroup G
      inst✝² : NormedSpace 𝕜 G
      inst✝¹ : NormedAddCommGroup H
      inst✝ : NormedSpace 𝕜 H
      n✝ n : Nat
      c : Composition n
      d : (i : Fin c.length) → Composition (c.blocksFun i)
      this : Eq (List.map List.sum (List.ofFn fun i => (d i).blocks)) c.blocks
      ⊢ Eq (List.ofFn fun i => ({ blocks := (List.ofFn fun j => (d j).blocks).flatte …
    -/
    congr! 1
      /-
        case mk.h.e'_2
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        G : Type u_4
        H : Type u_5
        inst✝⁸ : NontriviallyNormedField 𝕜
        inst✝⁷ : NormedAddCommGroup E
        inst✝⁶ : NormedSpace 𝕜 E
        inst✝⁵ : NormedAddCommGroup F
        inst✝⁴ : NormedSpace 𝕜 F
        inst✝³ : NormedAddCommGroup G
        inst✝² : NormedSpace 𝕜 G
        inst✝¹ : NormedAddCommGroup H
        inst✝ : NormedSpace 𝕜 H
        n✝ n : Nat
        c : Composition n
        d : (i : Fin c.length) → Composition (c.blocksFun i)
        this : Eq (List.map List.sum (List.ofFn fun i => (d i).blocks)) c.blocks
        ⊢ Eq ({ blocks := (List.ofFn fun j => (d j).blocks).flatten, blocks_pos := ⋯,  …
      -/
    · congr
      /-
        case mk.h.e'_2.e_c
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        G : Type u_4
        H : Type u_5
        inst✝⁸ : NontriviallyNormedField 𝕜
        inst✝⁷ : NormedAddCommGroup E
        inst✝⁶ : NormedSpace 𝕜 E
        inst✝⁵ : NormedAddCommGroup F
        inst✝⁴ : NormedSpace 𝕜 F
        inst✝³ : NormedAddCommGroup G
        inst✝² : NormedSpace 𝕜 G
        inst✝¹ : NormedAddCommGroup H
        inst✝ : NormedSpace 𝕜 H
        n✝ n : Nat
        c : Composition n
        d : (i : Fin c.length) → Composition (c.blocksFun i)
        this : Eq (List.map List.sum (List.ofFn fun i => (d i).blocks)) c.blocks
        ⊢ Eq ({ blocks := (List.ofFn fun j => (d j).blocks).flatten, blocks_pos := ⋯,  …
      -/
      ext1
      /-
        case mk.h.e'_2.e_c.blocks
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        G : Type u_4
        H : Type u_5
        inst✝⁸ : NontriviallyNormedField 𝕜
        inst✝⁷ : NormedAddCommGroup E
        inst✝⁶ : NormedSpace 𝕜 E
        inst✝⁵ : NormedAddCommGroup F
        inst✝⁴ : NormedSpace 𝕜 F
        inst✝³ : NormedAddCommGroup G
        inst✝² : NormedSpace 𝕜 G
        inst✝¹ : NormedAddCommGroup H
        inst✝ : NormedSpace 𝕜 H
        n✝ n : Nat
        c : Composition n
        d : (i : Fin c.length) → Composition (c.blocksFun i)
        this : Eq (List.map List.sum (List.ofFn fun i => (d i).blocks)) c.blocks
        ⊢ Eq ({ blocks := (List.ofFn fun j => (d j).blocks).flatten, blocks_pos := ⋯,  …
      -/
      dsimp [Composition.gather]
      /-
        case mk.h.e'_2.e_c.blocks
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        G : Type u_4
        H : Type u_5
        inst✝⁸ : NontriviallyNormedField 𝕜
        inst✝⁷ : NormedAddCommGroup E
        inst✝⁶ : NormedSpace 𝕜 E
        inst✝⁵ : NormedAddCommGroup F
        inst✝⁴ : NormedSpace 𝕜 F
        inst✝³ : NormedAddCommGroup G
        inst✝² : NormedSpace 𝕜 G
        inst✝¹ : NormedAddCommGroup H
        inst✝ : NormedSpace 𝕜 H
        n✝ n : Nat
        c : Composition n
        d : (i : Fin c.length) → Composition (c.blocksFun i)
        this : Eq (List.map List.sum (List.ofFn fun i => (d i).blocks)) c.blocks
        ⊢ Eq (List.map List.sum ((List.ofFn fun j => (d j).blocks).flatten.splitWrtCom …
      -/
      rwa [splitWrtComposition_flatten]
      /-
        case mk.h.e'_2.e_c.blocks.h
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        G : Type u_4
        H : Type u_5
        inst✝⁸ : NontriviallyNormedField 𝕜
        inst✝⁷ : NormedAddCommGroup E
        inst✝⁶ : NormedSpace 𝕜 E
        inst✝⁵ : NormedAddCommGroup F
        inst✝⁴ : NormedSpace 𝕜 F
        inst✝³ : NormedAddCommGroup G
        inst✝² : NormedSpace 𝕜 G
        inst✝¹ : NormedAddCommGroup H
        inst✝ : NormedSpace 𝕜 H
        n✝ n : Nat
        c : Composition n
        d : (i : Fin c.length) → Composition (c.blocksFun i)
        this : Eq (List.map List.sum (List.ofFn fun i => (d i).blocks)) c.blocks
        ⊢ Eq (List.map List.length (List.ofFn fun j => (d j).blocks)) { blocks := List …
      -/
      simp only [map_ofFn]
      /-
        case mk.h.e'_2.e_c.blocks.h
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        G : Type u_4
        H : Type u_5
        inst✝⁸ : NontriviallyNormedField 𝕜
        inst✝⁷ : NormedAddCommGroup E
        inst✝⁶ : NormedSpace 𝕜 E
        inst✝⁵ : NormedAddCommGroup F
        inst✝⁴ : NormedSpace 𝕜 F
        inst✝³ : NormedAddCommGroup G
        inst✝² : NormedSpace 𝕜 G
        inst✝¹ : NormedAddCommGroup H
        inst✝ : NormedSpace 𝕜 H
        n✝ n : Nat
        c : Composition n
        d : (i : Fin c.length) → Composition (c.blocksFun i)
        this : Eq (List.map List.sum (List.ofFn fun i => (d i).blocks)) c.blocks
        ⊢ Eq (List.ofFn (Function.comp List.length fun j => (d j).blocks)) (List.ofFn  …
      -/
      rfl
      /-
        🎉 no goals
      -/
      /-
        case mk.h.e'_3
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        G : Type u_4
        H : Type u_5
        inst✝⁸ : NontriviallyNormedField 𝕜
        inst✝⁷ : NormedAddCommGroup E
        inst✝⁶ : NormedSpace 𝕜 E
        inst✝⁵ : NormedAddCommGroup F
        inst✝⁴ : NormedSpace 𝕜 F
        inst✝³ : NormedAddCommGroup G
        inst✝² : NormedSpace 𝕜 G
        inst✝¹ : NormedAddCommGroup H
        inst✝ : NormedSpace 𝕜 H
        n✝ n : Nat
        c : Composition n
        d : (i : Fin c.length) → Composition (c.blocksFun i)
        this : Eq (List.map List.sum (List.ofFn fun i => (d i).blocks)) c.blocks
        e_2✝ : Eq ({ blocks := (List.ofFn fun j => (d j).blocks).flatten, blocks_pos : …
        ⊢ HEq (fun i => ({ blocks := (List.ofFn fun j => (d j).blocks).flatten, blocks …
      -/
    · rw [Fin.heq_fun_iff]
        /-
          case mk.h.e'_3
          𝕜 : Type u_1
          E : Type u_2
          F : Type u_3
          G : Type u_4
          H : Type u_5
          inst✝⁸ : NontriviallyNormedField 𝕜
          inst✝⁷ : NormedAddCommGroup E
          inst✝⁶ : NormedSpace 𝕜 E
          inst✝⁵ : NormedAddCommGroup F
          inst✝⁴ : NormedSpace 𝕜 F
          inst✝³ : NormedAddCommGroup G
          inst✝² : NormedSpace 𝕜 G
          inst✝¹ : NormedAddCommGroup H
          inst✝ : NormedSpace 𝕜 H
          n✝ n : Nat
          c : Composition n
          d : (i : Fin c.length) → Composition (c.blocksFun i)
          this : Eq (List.map List.sum (List.ofFn fun i => (d i).blocks)) c.blocks
          e_2✝ : Eq ({ blocks := (List.ofFn fun j => (d j).blocks).flatten, blocks_pos : …
          ⊢ ∀ (i : Fin ({ blocks := (List.ofFn fun j => (d j).blocks).flatten, blocks_po …
        -/
      · intro i
        /-
          case mk.h.e'_3
          𝕜 : Type u_1
          E : Type u_2
          F : Type u_3
          G : Type u_4
          H : Type u_5
          inst✝⁸ : NontriviallyNormedField 𝕜
          inst✝⁷ : NormedAddCommGroup E
          inst✝⁶ : NormedSpace 𝕜 E
          inst✝⁵ : NormedAddCommGroup F
          inst✝⁴ : NormedSpace 𝕜 F
          inst✝³ : NormedAddCommGroup G
          inst✝² : NormedSpace 𝕜 G
          inst✝¹ : NormedAddCommGroup H
          inst✝ : NormedSpace 𝕜 H
          n✝ n : Nat
          c : Composition n
          d : (i : Fin c.length) → Composition (c.blocksFun i)
          this : Eq (List.map List.sum (List.ofFn fun i => (d i).blocks)) c.blocks
          e_2✝ : Eq ({ blocks := (List.ofFn fun j => (d j).blocks).flatten, blocks_pos : …
          i : Fin ({ blocks := (List.ofFn fun j => (d j).blocks).flatten, blocks_pos :=  …
          ⊢ Eq ({ blocks := (List.ofFn fun j => (d j).blocks).flatten, blocks_pos := ⋯,  …
        -/
        dsimp [Composition.sigmaCompositionAux]
        /-
          case mk.h.e'_3
          𝕜 : Type u_1
          E : Type u_2
          F : Type u_3
          G : Type u_4
          H : Type u_5
          inst✝⁸ : NontriviallyNormedField 𝕜
          inst✝⁷ : NormedAddCommGroup E
          inst✝⁶ : NormedSpace 𝕜 E
          inst✝⁵ : NormedAddCommGroup F
          inst✝⁴ : NormedSpace 𝕜 F
          inst✝³ : NormedAddCommGroup G
          inst✝² : NormedSpace 𝕜 G
          inst✝¹ : NormedAddCommGroup H
          inst✝ : NormedSpace 𝕜 H
          n✝ n : Nat
          c : Composition n
          d : (i : Fin c.length) → Composition (c.blocksFun i)
          this : Eq (List.map List.sum (List.ofFn fun i => (d i).blocks)) c.blocks
          e_2✝ : Eq ({ blocks := (List.ofFn fun j => (d j).blocks).flatten, blocks_pos : …
          i : Fin ({ blocks := (List.ofFn fun j => (d j).blocks).flatten, blocks_pos :=  …
          ⊢ Eq (GetElem.getElem ((List.ofFn fun j => (d j).blocks).flatten.splitWrtCompo …
        -/
        rw [getElem_of_eq (splitWrtComposition_flatten _ _ _)]
          /-
            case mk.h.e'_3
            𝕜 : Type u_1
            E : Type u_2
            F : Type u_3
            G : Type u_4
            H : Type u_5
            inst✝⁸ : NontriviallyNormedField 𝕜
            inst✝⁷ : NormedAddCommGroup E
            inst✝⁶ : NormedSpace 𝕜 E
            inst✝⁵ : NormedAddCommGroup F
            inst✝⁴ : NormedSpace 𝕜 F
            inst✝³ : NormedAddCommGroup G
            inst✝² : NormedSpace 𝕜 G
            inst✝¹ : NormedAddCommGroup H
            inst✝ : NormedSpace 𝕜 H
            n✝ n : Nat
            c : Composition n
            d : (i : Fin c.length) → Composition (c.blocksFun i)
            this : Eq (List.map List.sum (List.ofFn fun i => (d i).blocks)) c.blocks
            e_2✝ : Eq ({ blocks := (List.ofFn fun j => (d j).blocks).flatten, blocks_pos : …
            i : Fin ({ blocks := (List.ofFn fun j => (d j).blocks).flatten, blocks_pos :=  …
            ⊢ Eq (GetElem.getElem (List.ofFn fun j => (d j).blocks) ↑i ⋯) (d ⟨↑i, ⋯⟩).blocks
          -/
        · simp only [List.getElem_ofFn]
          /-
            🎉 no goals
          -/
          /-
            𝕜 : Type u_1
            E : Type u_2
            F : Type u_3
            G : Type u_4
            H : Type u_5
            inst✝⁸ : NontriviallyNormedField 𝕜
            inst✝⁷ : NormedAddCommGroup E
            inst✝⁶ : NormedSpace 𝕜 E
            inst✝⁵ : NormedAddCommGroup F
            inst✝⁴ : NormedSpace 𝕜 F
            inst✝³ : NormedAddCommGroup G
            inst✝² : NormedSpace 𝕜 G
            inst✝¹ : NormedAddCommGroup H
            inst✝ : NormedSpace 𝕜 H
            n✝ n : Nat
            c : Composition n
            d : (i : Fin c.length) → Composition (c.blocksFun i)
            this : Eq (List.map List.sum (List.ofFn fun i => (d i).blocks)) c.blocks
            e_2✝ : Eq ({ blocks := (List.ofFn fun j => (d j).blocks).flatten, blocks_pos : …
            i : Fin ({ blocks := (List.ofFn fun j => (d j).blocks).flatten, blocks_pos :=  …
            ⊢ Eq (List.map List.length (List.ofFn fun j => (d j).blocks)) { blocks := List …
          -/
        · simp only [map_ofFn]
          /-
            𝕜 : Type u_1
            E : Type u_2
            F : Type u_3
            G : Type u_4
            H : Type u_5
            inst✝⁸ : NontriviallyNormedField 𝕜
            inst✝⁷ : NormedAddCommGroup E
            inst✝⁶ : NormedSpace 𝕜 E
            inst✝⁵ : NormedAddCommGroup F
            inst✝⁴ : NormedSpace 𝕜 F
            inst✝³ : NormedAddCommGroup G
            inst✝² : NormedSpace 𝕜 G
            inst✝¹ : NormedAddCommGroup H
            inst✝ : NormedSpace 𝕜 H
            n✝ n : Nat
            c : Composition n
            d : (i : Fin c.length) → Composition (c.blocksFun i)
            this : Eq (List.map List.sum (List.ofFn fun i => (d i).blocks)) c.blocks
            e_2✝ : Eq ({ blocks := (List.ofFn fun j => (d j).blocks).flatten, blocks_pos : …
            i : Fin ({ blocks := (List.ofFn fun j => (d j).blocks).flatten, blocks_pos :=  …
            ⊢ Eq (List.ofFn (Function.comp List.length fun j => (d j).blocks)) (List.ofFn  …
          -/
          rfl
          /-
            🎉 no goals
          -/
        /-
          case mk.h.e'_3.h
          𝕜 : Type u_1
          E : Type u_2
          F : Type u_3
          G : Type u_4
          H : Type u_5
          inst✝⁸ : NontriviallyNormedField 𝕜
          inst✝⁷ : NormedAddCommGroup E
          inst✝⁶ : NormedSpace 𝕜 E
          inst✝⁵ : NormedAddCommGroup F
          inst✝⁴ : NormedSpace 𝕜 F
          inst✝³ : NormedAddCommGroup G
          inst✝² : NormedSpace 𝕜 G
          inst✝¹ : NormedAddCommGroup H
          inst✝ : NormedSpace 𝕜 H
          n✝ n : Nat
          c : Composition n
          d : (i : Fin c.length) → Composition (c.blocksFun i)
          this : Eq (List.map List.sum (List.ofFn fun i => (d i).blocks)) c.blocks
          e_2✝ : Eq ({ blocks := (List.ofFn fun j => (d j).blocks).flatten, blocks_pos : …
          ⊢ Eq ({ blocks := (List.ofFn fun j => (d j).blocks).flatten, blocks_pos := ⋯,  …
        -/
      · congr
        /-
          🎉 no goals
        -/


theorem comp_assoc (r : FormalMultilinearSeries 𝕜 G H) (q : FormalMultilinearSeries 𝕜 F G)
    (p : FormalMultilinearSeries 𝕜 E F) : (r.comp q).comp p = r.comp (q.comp p) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    H : Type u_5
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    r : FormalMultilinearSeries 𝕜 G H
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    ⊢ Eq ((r.comp q).comp p) (r.comp (q.comp p))
  -/
  ext n v
  /- First, rewrite the two compositions appearing in the theorem as two sums over complicated
    sigma types, as in the description of the proof above. -/
  let f : (Σ a : Composition n, Composition a.length) → H := fun c =>
    r c.2.length (applyComposition q c.2 (applyComposition p c.1 v))
  let g : (Σ c : Composition n, ∀ i : Fin c.length, Composition (c.blocksFun i)) → H := fun c =>
    r c.1.length fun i : Fin c.1.length =>
      q (c.2 i).length (applyComposition p (c.2 i) (v ∘ c.1.embedding i))
  suffices ∑ c, f c = ∑ c, g c by
    simpa (config := { unfoldPartialApp := true }) only [FormalMultilinearSeries.comp,
      ContinuousMultilinearMap.sum_apply, compAlongComposition_apply, Finset.sum_sigma',
      applyComposition, ContinuousMultilinearMap.map_sum]
  /- Now, we use `Composition.sigmaEquivSigmaPi n` to change
    variables in the second sum, and check that we get exactly the same sums. -/
  /-
    case h.H
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    H : Type u_5
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    r : FormalMultilinearSeries 𝕜 G H
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    v : Fin n → E
    f : (Sigma fun a => Composition a.length) → H := fun c => (r c.snd.length) (q. …
    g : (Sigma fun c => (i : Fin c.length) → Composition (c.blocksFun i)) → H := f …
    ⊢ Eq (Finset.univ.sum fun c => f c) (Finset.univ.sum fun c => g c)
  -/
  rw [← (sigmaEquivSigmaPi n).sum_comp]
  /- To check that we have the same terms, we should check that we apply the same component of
    `r`, and the same component of `q`, and the same component of `p`, to the same coordinate of
    `v`. This is true by definition, but at each step one needs to convince Lean that the types
    one considers are the same, using a suitable congruence lemma to avoid dependent type issues.
    This dance has to be done three times, one for `r`, one for `q` and one for `p`. -/
  /-
    case h.H
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    H : Type u_5
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    r : FormalMultilinearSeries 𝕜 G H
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    v : Fin n → E
    f : (Sigma fun a => Composition a.length) → H := fun c => (r c.snd.length) (q. …
    g : (Sigma fun c => (i : Fin c.length) → Composition (c.blocksFun i)) → H := f …
    ⊢ Eq (Finset.univ.sum fun c => f c) (Finset.univ.sum fun i => g ((Composition. …
  -/
  apply Finset.sum_congr rfl
  /-
    case h.H
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    H : Type u_5
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    r : FormalMultilinearSeries 𝕜 G H
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    v : Fin n → E
    f : (Sigma fun a => Composition a.length) → H := fun c => (r c.snd.length) (q. …
    g : (Sigma fun c => (i : Fin c.length) → Composition (c.blocksFun i)) → H := f …
    ⊢ ∀ (x : Sigma fun a => Composition a.length), Membership.mem Finset.univ x →  …
  -/
  rintro ⟨a, b⟩ _
  /-
    case h.H.mk
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    H : Type u_5
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    r : FormalMultilinearSeries 𝕜 G H
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    v : Fin n → E
    f : (Sigma fun a => Composition a.length) → H := fun c => (r c.snd.length) (q. …
    g : (Sigma fun c => (i : Fin c.length) → Composition (c.blocksFun i)) → H := f …
    a : Composition n
    b : Composition a.length
    a✝ : Membership.mem Finset.univ ⟨a, b⟩
    ⊢ Eq (f ⟨a, b⟩) (g ((Composition.sigmaEquivSigmaPi n) ⟨a, b⟩))
  -/
  dsimp [sigmaEquivSigmaPi]
  -- check that the `r` components are the same. Based on `Composition.length_gather`
  /-
    case h.H.mk
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    H : Type u_5
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    r : FormalMultilinearSeries 𝕜 G H
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    v : Fin n → E
    f : (Sigma fun a => Composition a.length) → H := fun c => (r c.snd.length) (q. …
    g : (Sigma fun c => (i : Fin c.length) → Composition (c.blocksFun i)) → H := f …
    a : Composition n
    b : Composition a.length
    a✝ : Membership.mem Finset.univ ⟨a, b⟩
    ⊢ Eq (f ⟨a, b⟩) (g ⟨a.gather b, a.sigmaCompositionAux b⟩)
  -/
  apply r.congr (Composition.length_gather a b).symm
  /-
    case h.H.mk
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    H : Type u_5
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    r : FormalMultilinearSeries 𝕜 G H
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    v : Fin n → E
    f : (Sigma fun a => Composition a.length) → H := fun c => (r c.snd.length) (q. …
    g : (Sigma fun c => (i : Fin c.length) → Composition (c.blocksFun i)) → H := f …
    a : Composition n
    b : Composition a.length
    a✝ : Membership.mem Finset.univ ⟨a, b⟩
    ⊢ ∀ (i : Nat) (him : LT.lt i b.length) (hin : LT.lt i (a.gather b).length), Eq …
  -/
  intro i hi1 hi2
  -- check that the `q` components are the same. Based on `length_sigmaCompositionAux`
  /-
    case h.H.mk
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    H : Type u_5
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    r : FormalMultilinearSeries 𝕜 G H
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    v : Fin n → E
    f : (Sigma fun a => Composition a.length) → H := fun c => (r c.snd.length) (q. …
    g : (Sigma fun c => (i : Fin c.length) → Composition (c.blocksFun i)) → H := f …
    a : Composition n
    b : Composition a.length
    a✝ : Membership.mem Finset.univ ⟨a, b⟩
    i : Nat
    hi1 : LT.lt i b.length
    hi2 : LT.lt i (a.gather b).length
    ⊢ Eq (q.applyComposition ⟨a, b⟩.snd (p.applyComposition ⟨a, b⟩.fst v) ⟨i, hi1⟩ …
  -/
  apply q.congr (length_sigmaCompositionAux a b _).symm
  /-
    case h.H.mk
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    H : Type u_5
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    r : FormalMultilinearSeries 𝕜 G H
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    v : Fin n → E
    f : (Sigma fun a => Composition a.length) → H := fun c => (r c.snd.length) (q. …
    g : (Sigma fun c => (i : Fin c.length) → Composition (c.blocksFun i)) → H := f …
    a : Composition n
    b : Composition a.length
    a✝ : Membership.mem Finset.univ ⟨a, b⟩
    i : Nat
    hi1 : LT.lt i b.length
    hi2 : LT.lt i (a.gather b).length
    ⊢ ∀ (i_1 : Nat) (him : LT.lt i_1 (b.blocksFun ⟨i, hi1⟩)) (hin : LT.lt i_1 (a.s …
  -/
  intro j hj1 hj2
  -- check that the `p` components are the same. Based on `blocksFun_sigmaCompositionAux`
  /-
    case h.H.mk
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    H : Type u_5
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    r : FormalMultilinearSeries 𝕜 G H
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    v : Fin n → E
    f : (Sigma fun a => Composition a.length) → H := fun c => (r c.snd.length) (q. …
    g : (Sigma fun c => (i : Fin c.length) → Composition (c.blocksFun i)) → H := f …
    a : Composition n
    b : Composition a.length
    a✝ : Membership.mem Finset.univ ⟨a, b⟩
    i : Nat
    hi1 : LT.lt i b.length
    hi2 : LT.lt i (a.gather b).length
    j : Nat
    hj1 : LT.lt j (b.blocksFun ⟨i, hi1⟩)
    hj2 : LT.lt j (a.sigmaCompositionAux b ⟨↑⟨i, hi1⟩, ⋯⟩).length
    ⊢ Eq (Function.comp (p.applyComposition ⟨a, b⟩.fst v) ⇑(⟨a, b⟩.snd.embedding ⟨ …
  -/
  apply p.congr (blocksFun_sigmaCompositionAux a b _ _).symm
  /-
    case h.H.mk
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    H : Type u_5
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    r : FormalMultilinearSeries 𝕜 G H
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    v : Fin n → E
    f : (Sigma fun a => Composition a.length) → H := fun c => (r c.snd.length) (q. …
    g : (Sigma fun c => (i : Fin c.length) → Composition (c.blocksFun i)) → H := f …
    a : Composition n
    b : Composition a.length
    a✝ : Membership.mem Finset.univ ⟨a, b⟩
    i : Nat
    hi1 : LT.lt i b.length
    hi2 : LT.lt i (a.gather b).length
    j : Nat
    hj1 : LT.lt j (b.blocksFun ⟨i, hi1⟩)
    hj2 : LT.lt j (a.sigmaCompositionAux b ⟨↑⟨i, hi1⟩, ⋯⟩).length
    ⊢ ∀ (i_1 : Nat) (him : LT.lt i_1 (a.blocksFun ((b.embedding ⟨i, hi1⟩) ⟨j, hj1⟩ …
  -/
  intro k hk1 hk2
  -- finally, check that the coordinates of `v` one is using are the same. Based on
  -- `sizeUpTo_sizeUpTo_add`.
  /-
    case h.H.mk
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    H : Type u_5
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    r : FormalMultilinearSeries 𝕜 G H
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    v : Fin n → E
    f : (Sigma fun a => Composition a.length) → H := fun c => (r c.snd.length) (q. …
    g : (Sigma fun c => (i : Fin c.length) → Composition (c.blocksFun i)) → H := f …
    a : Composition n
    b : Composition a.length
    a✝ : Membership.mem Finset.univ ⟨a, b⟩
    i : Nat
    hi1 : LT.lt i b.length
    hi2 : LT.lt i (a.gather b).length
    j : Nat
    hj1 : LT.lt j (b.blocksFun ⟨i, hi1⟩)
    hj2 : LT.lt j (a.sigmaCompositionAux b ⟨↑⟨i, hi1⟩, ⋯⟩).length
    k : Nat
    hk1 : LT.lt k (a.blocksFun ((b.embedding ⟨i, hi1⟩) ⟨j, hj1⟩))
    hk2 : LT.lt k ((a.sigmaCompositionAux b ⟨↑⟨i, hi1⟩, ⋯⟩).blocksFun ⟨↑⟨j, hj1⟩,  …
    ⊢ Eq (Function.comp v ⇑(⟨a, b⟩.fst.embedding ((⟨a, b⟩.snd.embedding ⟨i, hi1⟩)  …
  -/
  refine congr_arg v (Fin.ext ?_)
  /-
    case h.H.mk
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    H : Type u_5
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    r : FormalMultilinearSeries 𝕜 G H
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    v : Fin n → E
    f : (Sigma fun a => Composition a.length) → H := fun c => (r c.snd.length) (q. …
    g : (Sigma fun c => (i : Fin c.length) → Composition (c.blocksFun i)) → H := f …
    a : Composition n
    b : Composition a.length
    a✝ : Membership.mem Finset.univ ⟨a, b⟩
    i : Nat
    hi1 : LT.lt i b.length
    hi2 : LT.lt i (a.gather b).length
    j : Nat
    hj1 : LT.lt j (b.blocksFun ⟨i, hi1⟩)
    hj2 : LT.lt j (a.sigmaCompositionAux b ⟨↑⟨i, hi1⟩, ⋯⟩).length
    k : Nat
    hk1 : LT.lt k (a.blocksFun ((b.embedding ⟨i, hi1⟩) ⟨j, hj1⟩))
    hk2 : LT.lt k ((a.sigmaCompositionAux b ⟨↑⟨i, hi1⟩, ⋯⟩).blocksFun ⟨↑⟨j, hj1⟩,  …
    ⊢ Eq ↑((⟨a, b⟩.fst.embedding ((⟨a, b⟩.snd.embedding ⟨i, hi1⟩) ⟨j, hj1⟩)) ⟨k, h …
  -/
  dsimp [Composition.embedding]
  /-
    case h.H.mk
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    H : Type u_5
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    r : FormalMultilinearSeries 𝕜 G H
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    v : Fin n → E
    f : (Sigma fun a => Composition a.length) → H := fun c => (r c.snd.length) (q. …
    g : (Sigma fun c => (i : Fin c.length) → Composition (c.blocksFun i)) → H := f …
    a : Composition n
    b : Composition a.length
    a✝ : Membership.mem Finset.univ ⟨a, b⟩
    i : Nat
    hi1 : LT.lt i b.length
    hi2 : LT.lt i (a.gather b).length
    j : Nat
    hj1 : LT.lt j (b.blocksFun ⟨i, hi1⟩)
    hj2 : LT.lt j (a.sigmaCompositionAux b ⟨↑⟨i, hi1⟩, ⋯⟩).length
    k : Nat
    hk1 : LT.lt k (a.blocksFun ((b.embedding ⟨i, hi1⟩) ⟨j, hj1⟩))
    hk2 : LT.lt k ((a.sigmaCompositionAux b ⟨↑⟨i, hi1⟩, ⋯⟩).blocksFun ⟨↑⟨j, hj1⟩,  …
    ⊢ Eq (HAdd.hAdd (a.sizeUpTo (HAdd.hAdd (b.sizeUpTo i) j)) k) (HAdd.hAdd ((a.ga …
  -/
  rw [sizeUpTo_sizeUpTo_add _ _ hi1 hj1, add_assoc]
  /-
    🎉 no goals
  -/


