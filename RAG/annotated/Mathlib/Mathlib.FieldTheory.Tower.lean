variable [Ring F] [Ring K] [Module F K]
  [AddCommGroup A] [Module K A] [NoZeroSMulDivisors K A]
  [Module F A] [IsNoetherian F A] [IsScalarTower F K A] in
/-- In a tower of field extensions `A / K / F`, if `A / F` is finite, so is `K / F`.

(In fact, it suffices that `A` is a nontrivial ring.)

Note this cannot be an instance as Lean cannot infer `A`.
-/
theorem left [Nontrivial A] : Module.Finite F K :=
  let ⟨x, hx⟩ := exists_ne (0 : A)
  Module.Finite.of_injective
    (LinearMap.ringLmapEquivSelf K ℕ A |>.symm x |>.restrictScalars F) (smul_left_injective K hx)


variable [Semiring F] [Semiring K] [Module F K]
  [AddCommMonoid A] [Module K A] [Module F A] [IsScalarTower F K A] in
@[stacks 09G5]
theorem right [hf : Module.Finite F A] : Module.Finite K A :=
  let ⟨⟨b, hb⟩⟩ := hf
  ⟨⟨b, Submodule.restrictScalars_injective F _ _ <| by
    /-
      F : Type u
      K : Type v
      A : Type w
      inst✝⁶ : Semiring F
      inst✝⁵ : Semiring K
      inst✝⁴ : Module F K
      inst✝³ : AddCommMonoid A
      inst✝² : Module K A
      inst✝¹ : Module F A
      inst✝ : IsScalarTower F K A
      hf : Module.Finite F A
      b : Finset A
      hb : Eq (Submodule.span F ↑b) Top.top
      ⊢ Eq (Submodule.restrictScalars F (Submodule.span K ↑b)) (Submodule.restrictSc …
    -/
    rw [Submodule.restrictScalars_top, eq_top_iff, ← hb, Submodule.span_le]
    /-
      F : Type u
      K : Type v
      A : Type w
      inst✝⁶ : Semiring F
      inst✝⁵ : Semiring K
      inst✝⁴ : Module F K
      inst✝³ : AddCommMonoid A
      inst✝² : Module K A
      inst✝¹ : Module F A
      inst✝ : IsScalarTower F K A
      hf : Module.Finite F A
      b : Finset A
      hb : Eq (Submodule.span F ↑b) Top.top
      ⊢ HasSubset.Subset ↑b ↑(Submodule.restrictScalars F (Submodule.span K ↑b))
    -/
    exact Submodule.subset_span⟩⟩
    /-
      🎉 no goals
    -/


alias FiniteDimensional.left := Module.Finite.left

alias FiniteDimensional.right := Module.Finite.right

