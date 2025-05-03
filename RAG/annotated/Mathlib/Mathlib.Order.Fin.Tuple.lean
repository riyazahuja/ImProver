lemma pi_lex_lt_cons_cons {x₀ y₀ : α 0} {x y : ∀ i : Fin n, α i.succ}
    (s : ∀ {i : Fin n.succ}, α i → α i → Prop) :
    Pi.Lex (· < ·) (@s) (Fin.cons x₀ x) (Fin.cons y₀ y) ↔
      s x₀ y₀ ∨ x₀ = y₀ ∧ Pi.Lex (· < ·) (@fun i : Fin n ↦ @s i.succ) x y := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Type u_1
    x₀ y₀ : α 0
    x y : (i : Fin n) → α i.succ
    s : {i : Fin n.succ} → α i → α i → Prop
    ⊢ Iff (Pi.Lex (fun x1 x2 => LT.lt x1 x2) s (Fin.cons x₀ x) (Fin.cons y₀ y)) (O …
  -/
  simp_rw [Pi.Lex, Fin.exists_fin_succ, Fin.cons_succ, Fin.cons_zero, Fin.forall_iff_succ]
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Type u_1
    x₀ y₀ : α 0
    x y : (i : Fin n) → α i.succ
    s : {i : Fin n.succ} → α i → α i → Prop
    ⊢ Iff (Or (And (And (LT.lt 0 0 → Eq (Fin.cons x₀ x 0) (Fin.cons y₀ y 0)) (∀ (i …
  -/
  simp [and_assoc, exists_and_left]
  /-
    🎉 no goals
  -/


lemma insertNth_mem_Icc {i : Fin (n + 1)} {x : α i} {p : ∀ j, α (i.succAbove j)}
    {q₁ q₂ : ∀ j, α j} :
    i.insertNth x p ∈ Icc q₁ q₂ ↔
      x ∈ Icc (q₁ i) (q₂ i) ∧ p ∈ Icc (fun j ↦ q₁ (i.succAbove j)) fun j ↦ q₂ (i.succAbove j) := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Type u_1
    inst✝ : (i : Fin (HAdd.hAdd n 1)) → Preorder (α i)
    i : Fin (HAdd.hAdd n 1)
    x : α i
    p : (j : Fin n) → α (i.succAbove j)
    q₁ q₂ : (j : Fin (HAdd.hAdd n 1)) → α j
    ⊢ Iff (Membership.mem (Set.Icc q₁ q₂) (i.insertNth x p)) (And (Membership.mem  …
  -/
  simp only [mem_Icc, insertNth_le_iff, le_insertNth_iff, and_assoc, @and_left_comm (x ≤ q₂ i)]
  /-
    🎉 no goals
  -/


lemma preimage_insertNth_Icc_of_mem {i : Fin (n + 1)} {x : α i} {q₁ q₂ : ∀ j, α j}
    (hx : x ∈ Icc (q₁ i) (q₂ i)) :
    i.insertNth x ⁻¹' Icc q₁ q₂ = Icc (fun j ↦ q₁ (i.succAbove j)) fun j ↦ q₂ (i.succAbove j) :=
                     /-
                       n : Nat
                       α : Fin (HAdd.hAdd n 1) → Type u_1
                       inst✝ : (i : Fin (HAdd.hAdd n 1)) → Preorder (α i)
                       i : Fin (HAdd.hAdd n 1)
                       x : α i
                       q₁ q₂ : (j : Fin (HAdd.hAdd n 1)) → α j
                       hx : Membership.mem (Set.Icc (q₁ i) (q₂ i)) x
                       p : (j : Fin n) → α (i.succAbove j)
                       ⊢ Iff (Membership.mem (Set.preimage (i.insertNth x) (Set.Icc q₁ q₂)) p) (Membe …
                     -/
  Set.ext fun p ↦ by simp only [mem_preimage, insertNth_mem_Icc, hx, true_and]
                     /-
                       🎉 no goals
                     -/


lemma preimage_insertNth_Icc_of_not_mem {i : Fin (n + 1)} {x : α i} {q₁ q₂ : ∀ j, α j}
    (hx : x ∉ Icc (q₁ i) (q₂ i)) : i.insertNth x ⁻¹' Icc q₁ q₂ = ∅ :=
  Set.ext fun p ↦ by
    /-
      n : Nat
      α : Fin (HAdd.hAdd n 1) → Type u_1
      inst✝ : (i : Fin (HAdd.hAdd n 1)) → Preorder (α i)
      i : Fin (HAdd.hAdd n 1)
      x : α i
      q₁ q₂ : (j : Fin (HAdd.hAdd n 1)) → α j
      hx : Not (Membership.mem (Set.Icc (q₁ i) (q₂ i)) x)
      p : (j : Fin n) → α (i.succAbove j)
      ⊢ Iff (Membership.mem (Set.preimage (i.insertNth x) (Set.Icc q₁ q₂)) p) (Membe …
    -/
    simp only [mem_preimage, insertNth_mem_Icc, hx, false_and, mem_empty_iff_false]
    /-
      🎉 no goals
    -/


lemma liftFun_vecCons {n : ℕ} (r : α → α → Prop) [IsTrans α r] {f : Fin (n + 1) → α} {a : α} :
    ((· < ·) ⇒ r) (vecCons a f) (vecCons a f) ↔ r a (f 0) ∧ ((· < ·) ⇒ r) f f := by
  simp only [liftFun_iff_succ r, forall_iff_succ, cons_val_succ, cons_val_zero, ← succ_castSucc,
    castSucc_zero]


@[simp] lemma strictMono_vecCons : StrictMono (vecCons a f) ↔ a < f 0 ∧ StrictMono f :=
  liftFun_vecCons (· < ·)


@[simp]
lemma monotone_vecCons : Monotone (vecCons a f) ↔ a ≤ f 0 ∧ Monotone f := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    n : Nat
    f : Fin (HAdd.hAdd n 1) → α
    a : α
    ⊢ Iff (Monotone (Matrix.vecCons a f)) (And (LE.le a (f 0)) (Monotone f))
  -/
  simpa only [monotone_iff_forall_lt] using @liftFun_vecCons α n (· ≤ ·) _ f a
  /-
    🎉 no goals
  -/


@[simp] lemma monotone_vecEmpty : Monotone ![a]
  | ⟨0, _⟩, ⟨0, _⟩, _ => le_refl _


@[simp] lemma strictMono_vecEmpty : StrictMono ![a]
  | ⟨0, _⟩, ⟨0, _⟩, h => (irrefl _ h).elim


@[simp] lemma strictAnti_vecCons : StrictAnti (vecCons a f) ↔ f 0 < a ∧ StrictAnti f :=
  liftFun_vecCons (· > ·)


@[simp] lemma antitone_vecCons : Antitone (vecCons a f) ↔ f 0 ≤ a ∧ Antitone f :=
  monotone_vecCons (α := αᵒᵈ)


@[simp] lemma antitone_vecEmpty : Antitone (vecCons a vecEmpty)
  | ⟨0, _⟩, ⟨0, _⟩, _ => le_rfl


@[simp] lemma strictAnti_vecEmpty : StrictAnti (vecCons a vecEmpty)
  | ⟨0, _⟩, ⟨0, _⟩, h => (irrefl _ h).elim


lemma StrictMono.vecCons (hf : StrictMono f) (ha : a < f 0) : StrictMono (vecCons a f) :=
  strictMono_vecCons.2 ⟨ha, hf⟩


lemma StrictAnti.vecCons (hf : StrictAnti f) (ha : f 0 < a) : StrictAnti (vecCons a f) :=
  strictAnti_vecCons.2 ⟨ha, hf⟩


lemma Monotone.vecCons (hf : Monotone f) (ha : a ≤ f 0) : Monotone (vecCons a f) :=
  monotone_vecCons.2 ⟨ha, hf⟩


lemma Antitone.vecCons (hf : Antitone f) (ha : f 0 ≤ a) : Antitone (vecCons a f) :=
  antitone_vecCons.2 ⟨ha, hf⟩


/-- `Π i : Fin 2, α i` is order equivalent to `α 0 × α 1`. See also `OrderIso.finTwoArrowEquiv`
for a non-dependent version. -/
def OrderIso.piFinTwoIso (α : Fin 2 → Type*) [∀ i, Preorder (α i)] : (∀ i, α i) ≃o α 0 × α 1 where
  toEquiv := piFinTwoEquiv α
  map_rel_iff' := Iff.symm Fin.forall_fin_two


/-- The space of functions `Fin 2 → α` is order equivalent to `α × α`. See also
`OrderIso.piFinTwoIso`. -/
def OrderIso.finTwoArrowIso (α : Type*) [Preorder α] : (Fin 2 → α) ≃o α × α :=
  { OrderIso.piFinTwoIso fun _ => α with toEquiv := finTwoArrowEquiv α }


/-- Order isomorphism between tuples of length `n + 1` and pairs of an element and a tuple of length
`n` given by separating out the first element of the tuple.

This is `Fin.cons` as an `OrderIso`. -/
@[simps!, simps toEquiv]
def consOrderIso (α : Fin (n + 1) → Type*) [∀ i, LE (α i)] :
    α 0 × (∀ i, α (succ i)) ≃o ∀ i, α i where
  toEquiv := consEquiv α
  map_rel_iff' := forall_iff_succ


/-- Order isomorphism between tuples of length `n + 1` and pairs of an element and a tuple of length
`n` given by separating out the last element of the tuple.

This is `Fin.snoc` as an `OrderIso`. -/
@[simps!, simps toEquiv]
def snocOrderIso (α : Fin (n + 1) → Type*) [∀ i, LE (α i)] :
    α (last n) × (∀ i, α (castSucc i)) ≃o ∀ i, α i where
  toEquiv := snocEquiv α
                     /-
                       α✝ : Type u_1
                       inst✝¹ : Preorder α✝
                       n✝ : Nat
                       f : Fin (HAdd.hAdd n✝ 1) → α✝
                       a : α✝
                       n : Nat
                       α : Fin (HAdd.hAdd n 1) → Type u_2
                       inst✝ : (i : Fin (HAdd.hAdd n 1)) → LE (α i)
                       ⊢ ∀ {a b : Prod (α (Fin.last n)) ((i : Fin n) → α i.castSucc)}, Iff (LE.le ((F …
                     -/
  map_rel_iff' := by simp [Pi.le_def, Prod.le_def, forall_iff_castSucc]
                     /-
                       🎉 no goals
                     -/


/-- Order isomorphism between tuples of length `n + 1` and pairs of an element and a tuple of length
`n` given by separating out the `p`-th element of the tuple.

This is `Fin.insertNth` as an `OrderIso`. -/
@[simps!, simps toEquiv]
def insertNthOrderIso (α : Fin (n + 1) → Type*) [∀ i, LE (α i)] (p : Fin (n + 1)) :
    α p × (∀ i, α (p.succAbove i)) ≃o ∀ i, α i where
  toEquiv := insertNthEquiv α p
                     /-
                       α✝ : Type u_1
                       inst✝¹ : Preorder α✝
                       n✝ : Nat
                       f : Fin (HAdd.hAdd n✝ 1) → α✝
                       a : α✝
                       n : Nat
                       α : Fin (HAdd.hAdd n 1) → Type u_2
                       inst✝ : (i : Fin (HAdd.hAdd n 1)) → LE (α i)
                       p : Fin (HAdd.hAdd n 1)
                       ⊢ ∀ {a b : Prod (α p) ((i : Fin n) → α (p.succAbove i))}, Iff (LE.le ((Fin.ins …
                     -/
  map_rel_iff' := by simp [Pi.le_def, Prod.le_def, p.forall_iff_succAbove]
                     /-
                       🎉 no goals
                     -/


@[simp] lemma insertNthOrderIso_zero (α : Fin (n + 1) → Type*) [∀ i, LE (α i)] :
                                                 /-
                                                   n : Nat
                                                   α : Fin (HAdd.hAdd n 1) → Type u_2
                                                   inst✝ : (i : Fin (HAdd.hAdd n 1)) → LE (α i)
                                                   ⊢ Eq (Fin.insertNthOrderIso α 0) (Fin.consOrderIso α)
                                                 -/
    insertNthOrderIso α 0 = consOrderIso α := by ext; simp [insertNthOrderIso]
                                                      /-
                                                        🎉 no goals
                                                      -/


/-- Note this lemma can only be written about non-dependent tuples as `insertNth (last n) = snoc` is
not a definitional equality. -/
@[simp] lemma insertNthOrderIso_last (n : ℕ) (α : Type*) [LE α] :
                                                                            /-
                                                                              n : Nat
                                                                              α : Type u_2
                                                                              inst✝ : LE α
                                                                              ⊢ Eq (Fin.insertNthOrderIso (fun x => α) (Fin.last n)) (Fin.snocOrderIso fun x …
                                                                            -/
    insertNthOrderIso (fun _ ↦ α) (last n) = snocOrderIso (fun _ ↦ α) := by ext; simp
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


/-- Order isomorphism between `Π j : Fin (n + 1), α j` and
`α i × Π j : Fin n, α (Fin.succAbove i j)`. -/
@[deprecated Fin.insertNthOrderIso (since := "2024-07-12")]
def OrderIso.piFinSuccAboveIso (α : Fin (n + 1) → Type*) [∀ i, LE (α i)]
    (i : Fin (n + 1)) : (∀ j, α j) ≃o α i × ∀ j, α (i.succAbove j) where
  toEquiv := (Fin.insertNthEquiv α i).symm
  map_rel_iff' := Iff.symm i.forall_iff_succAbove


/-- `Fin.succAbove` as an order isomorphism between `Fin n` and `{x : Fin (n + 1) // x ≠ p}`. -/
def finSuccAboveOrderIso (p : Fin (n + 1)) : Fin n ≃o { x : Fin (n + 1) // x ≠ p } where
  __ := finSuccAboveEquiv p
  map_rel_iff' := p.succAboveOrderEmb.map_rel_iff'


lemma finSuccAboveOrderIso_apply (p : Fin (n + 1)) (i : Fin n) :
    finSuccAboveOrderIso p i = ⟨p.succAbove i, p.succAbove_ne i⟩ := rfl


lemma finSuccAboveOrderIso_symm_apply_last (x : { x : Fin (n + 1) // x ≠ Fin.last n }) :
    (finSuccAboveOrderIso (Fin.last n)).symm x = Fin.castLT x.1 (Fin.val_lt_last x.2) := by
  /-
    n : Nat
    x : Subtype fun x => Ne x (Fin.last n)
    ⊢ Eq ((finSuccAboveOrderIso (Fin.last n)).symm x) ((↑x).castLT ⋯)
  -/
  rw [← Option.some_inj]
  simpa [finSuccAboveOrderIso, finSuccAboveEquiv, OrderIso.symm]
    using finSuccEquiv'_last_apply x.property


lemma finSuccAboveOrderIso_symm_apply_ne_last {p : Fin (n + 1)} (h : p ≠ Fin.last n)
    (x : { x : Fin (n + 1) // x ≠ p }) :
    (finSuccAboveEquiv p).symm x = (p.castLT (Fin.val_lt_last h)).predAbove x := by
  /-
    n : Nat
    p : Fin (HAdd.hAdd n 1)
    h : Ne p (Fin.last n)
    x : Subtype fun x => Ne x p
    ⊢ Eq ((finSuccAboveEquiv p).symm x) ((p.castLT ⋯).predAbove ↑x)
  -/
  rw [← Option.some_inj]
  /-
    n : Nat
    p : Fin (HAdd.hAdd n 1)
    h : Ne p (Fin.last n)
    x : Subtype fun x => Ne x p
    ⊢ Eq (Option.some ((finSuccAboveEquiv p).symm x)) (Option.some ((p.castLT ⋯).p …
  -/
  simpa [finSuccAboveEquiv, OrderIso.symm] using finSuccEquiv'_ne_last_apply h x.property
  /-
    🎉 no goals
  -/


/-- Promote a `Fin n` into a larger `Fin m`, as a subtype where the underlying
values are retained. This is the `OrderIso` version of `Fin.castLE`. -/
@[simps apply symm_apply]
def Fin.castLEOrderIso {n m : ℕ} (h : n ≤ m) : Fin n ≃o { i : Fin m // (i : ℕ) < n } where
                                 /-
                                   α : Type u_1
                                   inst✝ : Preorder α
                                   n✝¹ : Nat
                                   f : Fin (HAdd.hAdd n✝¹ 1) → α
                                   a : α
                                   n✝ n m : Nat
                                   h : LE.le n m
                                   i : Fin n
                                   ⊢ LT.lt (↑(Fin.castLE h i)) n
                                 -/
  toFun i := ⟨Fin.castLE h i, by simp⟩
                                 /-
                                   🎉 no goals
                                 -/
  invFun i := ⟨i, i.prop⟩
                   /-
                     α : Type u_1
                     inst✝ : Preorder α
                     n✝¹ : Nat
                     f : Fin (HAdd.hAdd n✝¹ 1) → α
                     a : α
                     n✝ n m : Nat
                     h : LE.le n m
                     x✝ : Fin n
                     ⊢ Eq ((fun i => ⟨↑↑i, ⋯⟩) ((fun i => ⟨Fin.castLE h i, ⋯⟩) x✝)) x✝
                   -/
  left_inv _ := by simp
                   /-
                     🎉 no goals
                   -/
                    /-
                      α : Type u_1
                      inst✝ : Preorder α
                      n✝¹ : Nat
                      f : Fin (HAdd.hAdd n✝¹ 1) → α
                      a : α
                      n✝ n m : Nat
                      h : LE.le n m
                      x✝ : Subtype fun i => LT.lt (↑i) n
                      ⊢ Eq ((fun i => ⟨Fin.castLE h i, ⋯⟩) ((fun i => ⟨↑↑i, ⋯⟩) x✝)) x✝
                    -/
  right_inv _ := by simp
                    /-
                      🎉 no goals
                    -/
                     /-
                       α : Type u_1
                       inst✝ : Preorder α
                       n✝¹ : Nat
                       f : Fin (HAdd.hAdd n✝¹ 1) → α
                       a : α
                       n✝ n m : Nat
                       h : LE.le n m
                       ⊢ ∀ {a b : Fin n}, Iff (LE.le ({ toFun := fun i => ⟨Fin.castLE h i, ⋯⟩, invFun …
                     -/
  map_rel_iff' := by simp [(strictMono_castLE h).le_iff_le]
                     /-
                       🎉 no goals
                     -/

