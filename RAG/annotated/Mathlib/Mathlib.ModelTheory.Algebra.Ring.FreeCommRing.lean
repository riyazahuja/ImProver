private theorem exists_term_realize_eq_freeCommRing (p : FreeCommRing α) :
    ∃ t : Language.ring.Term α,
      (t.realize FreeCommRing.of : FreeCommRing α) = p :=
  FreeCommRing.induction_on p
            /-
              α : Type u_1
              p : FreeCommRing α
              ⊢ Eq (FirstOrder.Language.Term.realize FreeCommRing.of (-1)) (-1)
            -/
    ⟨-1, by simp [Term.realize]⟩
            /-
              🎉 no goals
            -/
                              /-
                                α : Type u_1
                                p : FreeCommRing α
                                a : α
                                ⊢ Eq (FirstOrder.Language.Term.realize FreeCommRing.of (FirstOrder.Language.Te …
                              -/
    (fun a => ⟨Term.var a, by simp [Term.realize]⟩)
                              /-
                                🎉 no goals
                              -/
    (fun x y ⟨t₁, ht₁⟩ ⟨t₂, ht₂⟩ =>
                   /-
                     α : Type u_1
                     p x y : FreeCommRing α
                     x✝¹ : Exists fun t => Eq (FirstOrder.Language.Term.realize FreeCommRing.of t) x
                     x✝ : Exists fun t => Eq (FirstOrder.Language.Term.realize FreeCommRing.of t) y
                     t₁ : FirstOrder.Language.ring.Term α
                     ht₁ : Eq (FirstOrder.Language.Term.realize FreeCommRing.of t₁) x
                     t₂ : FirstOrder.Language.ring.Term α
                     ht₂ : Eq (FirstOrder.Language.Term.realize FreeCommRing.of t₂) y
                     ⊢ Eq (FirstOrder.Language.Term.realize FreeCommRing.of (HAdd.hAdd t₁ t₂)) (HAd …
                   -/
      ⟨t₁ + t₂, by simp_all [Term.realize]⟩)
                   /-
                     🎉 no goals
                   -/
    (fun x y ⟨t₁, ht₁⟩ ⟨t₂, ht₂⟩ =>
                   /-
                     α : Type u_1
                     p x y : FreeCommRing α
                     x✝¹ : Exists fun t => Eq (FirstOrder.Language.Term.realize FreeCommRing.of t) x
                     x✝ : Exists fun t => Eq (FirstOrder.Language.Term.realize FreeCommRing.of t) y
                     t₁ : FirstOrder.Language.ring.Term α
                     ht₁ : Eq (FirstOrder.Language.Term.realize FreeCommRing.of t₁) x
                     t₂ : FirstOrder.Language.ring.Term α
                     ht₂ : Eq (FirstOrder.Language.Term.realize FreeCommRing.of t₂) y
                     ⊢ Eq (FirstOrder.Language.Term.realize FreeCommRing.of (HMul.hMul t₁ t₂)) (HMu …
                   -/
      ⟨t₁ * t₂, by simp_all [Term.realize]⟩)
                   /-
                     🎉 no goals
                   -/


/-- Make a `Language.ring.Term α` from an element of `FreeCommRing α` -/
noncomputable def termOfFreeCommRing (p : FreeCommRing α) : Language.ring.Term α :=
  Classical.choose (exists_term_realize_eq_freeCommRing p)


@[simp]
theorem realize_termOfFreeCommRing (p : FreeCommRing α) (v : α → R) :
    (termOfFreeCommRing p).realize v = FreeCommRing.lift v p := by
  /-
    α : Type u_1
    R : Type u_2
    inst✝¹ : CommRing R
    inst✝ : FirstOrder.Ring.CompatibleRing R
    p : FreeCommRing α
    v : α → R
    ⊢ Eq (FirstOrder.Language.Term.realize v (FirstOrder.Ring.termOfFreeCommRing p …
  -/
  let _ := compatibleRingOfRing (FreeCommRing α)
  /-
    α : Type u_1
    R : Type u_2
    inst✝¹ : CommRing R
    inst✝ : FirstOrder.Ring.CompatibleRing R
    p : FreeCommRing α
    v : α → R
    x✝ : FirstOrder.Ring.CompatibleRing (FreeCommRing α) := FirstOrder.Ring.compat …
    ⊢ Eq (FirstOrder.Language.Term.realize v (FirstOrder.Ring.termOfFreeCommRing p …
  -/
  rw [termOfFreeCommRing]
  /-
    α : Type u_1
    R : Type u_2
    inst✝¹ : CommRing R
    inst✝ : FirstOrder.Ring.CompatibleRing R
    p : FreeCommRing α
    v : α → R
    x✝ : FirstOrder.Ring.CompatibleRing (FreeCommRing α) := FirstOrder.Ring.compat …
    ⊢ Eq (FirstOrder.Language.Term.realize v (Classical.choose ⋯)) ((FreeCommRing. …
  -/
  conv_rhs => rw [← Classical.choose_spec (exists_term_realize_eq_freeCommRing p)]
  induction Classical.choose (exists_term_realize_eq_freeCommRing p) with
  | var _ => simp
  | func f a ih =>
    cases f <;>
    simp [ih]


