/-- The `FreeMonoid X` basis on the `FreeAlgebra R X`,
mapping `[x₁, x₂, ..., xₙ]` to the "monomial" `1 • x₁ * x₂ * ⋯ * xₙ` -/
-- @[simps]
noncomputable def basisFreeMonoid : Basis (FreeMonoid X) R (FreeAlgebra R X) :=
  Finsupp.basisSingleOne.map (equivMonoidAlgebraFreeMonoid (R := R) (X := X)).symm.toLinearEquiv


instance : Module.Free R (FreeAlgebra R X) :=
  have : Module.Free R (MonoidAlgebra R (FreeMonoid X)) := Module.Free.finsupp _ _ _
  Module.Free.of_equiv (equivMonoidAlgebraFreeMonoid (R := R) (X := X)).symm.toLinearEquiv


theorem rank_eq [CommRing R] [Nontrivial R] :
    Module.rank R (FreeAlgebra R X) = Cardinal.lift.{u} (Cardinal.mk (List X)) := by
  rw [← (Basis.mk_eq_rank'.{_,_,_,u} (basisFreeMonoid R X)).trans (Cardinal.lift_id _),
    Cardinal.lift_umax.{v, u}, FreeMonoid]


theorem Algebra.rank_adjoin_le {R : Type u} {S : Type v} [CommRing R] [Ring S] [Algebra R S]
    (s : Set S) : Module.rank R (adjoin R s) ≤ max #s ℵ₀ := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    s : Set S
    ⊢ LE.le (Module.rank R (Subtype fun x => Membership.mem (Algebra.adjoin R s) x …
  -/
  rw [adjoin_eq_range_freeAlgebra_lift]
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    s : Set S
    ⊢ LE.le (Module.rank R (Subtype fun x => Membership.mem ((FreeAlgebra.lift R)  …
  -/
  cases subsingleton_or_nontrivial R
    /-
      case inl
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : Ring S
      inst✝ : Algebra R S
      s : Set S
      h✝ : Subsingleton R
      ⊢ LE.le (Module.rank R (Subtype fun x => Membership.mem ((FreeAlgebra.lift R)  …
    -/
  · rw [rank_subsingleton]; exact one_le_aleph0.trans (le_max_right _ _)
                            /-
                              🎉 no goals
                            -/
  /-
    case inr
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    s : Set S
    h✝ : Nontrivial R
    ⊢ LE.le (Module.rank R (Subtype fun x => Membership.mem ((FreeAlgebra.lift R)  …
  -/
  rw [← lift_le.{max u v}]
  /-
    case inr
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    s : Set S
    h✝ : Nontrivial R
    ⊢ LE.le (Cardinal.lift.{max u v, v} (Module.rank R (Subtype fun x => Membershi …
  -/
  refine (lift_rank_range_le (FreeAlgebra.lift R ((↑) : s → S)).toLinearMap).trans ?_
  /-
    case inr
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    s : Set S
    h✝ : Nontrivial R
    ⊢ LE.le (Cardinal.lift.{v, max u v} (Module.rank R (FreeAlgebra R (Subtype fun …
  -/
  rw [FreeAlgebra.rank_eq, lift_id'.{v,u}, lift_umax.{v,u}, lift_le, max_comm]
  /-
    case inr
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    s : Set S
    h✝ : Nontrivial R
    ⊢ LE.le (Cardinal.mk (List (Subtype fun x => Membership.mem s x))) (Max.max Ca …
  -/
  exact mk_list_le_max _
  /-
    🎉 no goals
  -/

