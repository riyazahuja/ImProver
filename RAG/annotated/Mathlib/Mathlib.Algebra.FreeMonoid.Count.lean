/-- `List.countP` as a bundled additive monoid homomorphism. -/
def countP : FreeAddMonoid α →+ ℕ where
  toFun := List.countP p
  map_zero' := List.countP_nil _
  map_add' := List.countP_append _


theorem countP_of (x : α) : countP p (of x) = if p x = true then 1 else 0 := by
  /-
    α : Type u_1
    p : α → Prop
    inst✝ : DecidablePred p
    x : α
    ⊢ Eq ((FreeAddMonoid.countP p) (FreeAddMonoid.of x)) (ite (Eq (p x) (Eq Bool.t …
  -/
  change List.countP p [x] = _
  /-
    α : Type u_1
    p : α → Prop
    inst✝ : DecidablePred p
    x : α
    ⊢ Eq (List.countP (fun b => Decidable.decide (p b)) (List.cons x List.nil)) (i …
  -/
  simp [List.countP_cons]
  /-
    🎉 no goals
  -/


theorem countP_apply (l : FreeAddMonoid α) : countP p l = List.countP p l := rfl


/-- `List.count` as a bundled additive monoid homomorphism. -/
-- Porting note: was (x = ·)
def count [DecidableEq α] (x : α) : FreeAddMonoid α →+ ℕ := countP (· = x)


theorem count_of [DecidableEq α] (x y : α) : count x (of y) = (Pi.single x 1 : α → ℕ) y := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    x y : α
    ⊢ Eq ((FreeAddMonoid.count x) (FreeAddMonoid.of y)) (Pi.single x 1 y)
  -/
  change List.count x [y] = _
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    x y : α
    ⊢ Eq (List.count x (List.cons y List.nil)) (Pi.single x 1 y)
  -/
  simp [Pi.single, Function.update, List.count_cons]
  /-
    🎉 no goals
  -/


theorem count_apply [DecidableEq α] (x : α) (l : FreeAddMonoid α) : count x l = List.count x l :=
  rfl


/-- `List.countP` as a bundled multiplicative monoid homomorphism. -/
def countP : FreeMonoid α →* Multiplicative ℕ :=
    AddMonoidHom.toMultiplicative (FreeAddMonoid.countP p)


theorem countP_of' (x : α) :
    countP p (of x) = if p x then Multiplicative.ofAdd 1 else Multiplicative.ofAdd 0 := by
    /-
      α : Type u_1
      p : α → Prop
      inst✝ : DecidablePred p
      x : α
      ⊢ Eq ((FreeMonoid.countP p) (FreeMonoid.of x)) (ite (p x) (Multiplicative.ofAd …
    -/
    erw [FreeAddMonoid.countP_of]
    /-
      α : Type u_1
      p : α → Prop
      inst✝ : DecidablePred p
      x : α
      ⊢ Eq (ite (Eq (p x) (Eq Bool.true Bool.true)) 1 0) (ite (p x) (Multiplicative. …
    -/
    simp only [eq_iff_iff, iff_true, ofAdd_zero]; rfl
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem countP_of (x : α) : countP p (of x) = if p x then Multiplicative.ofAdd 1 else 1 := by
  /-
    α : Type u_1
    p : α → Prop
    inst✝ : DecidablePred p
    x : α
    ⊢ Eq ((FreeMonoid.countP p) (FreeMonoid.of x)) (ite (p x) (Multiplicative.ofAd …
  -/
  rw [countP_of', ofAdd_zero]
  /-
    🎉 no goals
  -/

-- `rfl` is not transitive

theorem countP_apply (l : FreeAddMonoid α) : countP p l = Multiplicative.ofAdd (List.countP p l) :=
  rfl


/-- `List.count` as a bundled additive monoid homomorphism. -/
def count [DecidableEq α] (x : α) : FreeMonoid α →* Multiplicative ℕ := countP (· = x)


theorem count_apply [DecidableEq α] (x : α) (l : FreeAddMonoid α) :
    count x l = Multiplicative.ofAdd (List.count x l) := rfl


theorem count_of [DecidableEq α] (x y : α) :
    count x (of y) = @Pi.mulSingle α (fun _ => Multiplicative ℕ) _ _ x (Multiplicative.ofAdd 1) y :=
     /-
       α : Type u_1
       inst✝ : DecidableEq α
       x y : α
       ⊢ Eq ((FreeMonoid.count x) (FreeMonoid.of y)) (Pi.mulSingle x (Multiplicative. …
     -/
  by simp [count, countP_of, Pi.mulSingle_apply, eq_comm, Bool.beq_eq_decide_eq]
     /-
       🎉 no goals
     -/


