/-- The linear equivalence `(⨁ i₁, M₁ i₁) ⊗ (⨁ i₂, M₂ i₂) ≃ (⨁ i₁, ⨁ i₂, M₁ i₁ ⊗ M₂ i₂)`, i.e.
"tensor product distributes over direct sum". -/
protected def directSum :
    ((⨁ i₁, M₁ i₁) ⊗[R] ⨁ i₂, M₂ i₂) ≃ₗ[S] ⨁ i : ι₁ × ι₂, M₁ i.1 ⊗[R] M₂ i.2 := by
  -- Porting note: entirely rewritten to allow unification to happen one step at a time
  /-
    R : Type u
    inst✝¹⁴ : CommSemiring R
    S : Type ?u.2020
    inst✝¹³ : Semiring S
    inst✝¹² : Algebra R S
    ι₁ : Type v₁
    ι₂ : Type v₂
    inst✝¹¹ : DecidableEq ι₁
    inst✝¹⁰ : DecidableEq ι₂
    M₁ : ι₁ → Type w₁
    M₁' : Type w₁'
    M₂ : ι₂ → Type w₂
    M₂' : Type w₂'
    inst✝⁹ : (i₁ : ι₁) → AddCommMonoid (M₁ i₁)
    inst✝⁸ : AddCommMonoid M₁'
    inst✝⁷ : (i₂ : ι₂) → AddCommMonoid (M₂ i₂)
    inst✝⁶ : AddCommMonoid M₂'
    inst✝⁵ : (i₁ : ι₁) → Module R (M₁ i₁)
    inst✝⁴ : Module R M₁'
    inst✝³ : (i₂ : ι₂) → Module R (M₂ i₂)
    inst✝² : Module R M₂'
    inst✝¹ : (i₁ : ι₁) → Module S (M₁ i₁)
    inst✝ : ∀ (i₁ : ι₁), IsScalarTower R S (M₁ i₁)
    ⊢ LinearEquiv (RingHom.id S) (TensorProduct R (DirectSum ι₁ fun i₁ => M₁ i₁) ( …
  -/
  refine LinearEquiv.ofLinear (R := S) (R₂ := S) ?toFun ?invFun ?left ?right
    /-
      case toFun
      R : Type u
      inst✝¹⁴ : CommSemiring R
      S : Type ?u.2020
      inst✝¹³ : Semiring S
      inst✝¹² : Algebra R S
      ι₁ : Type v₁
      ι₂ : Type v₂
      inst✝¹¹ : DecidableEq ι₁
      inst✝¹⁰ : DecidableEq ι₂
      M₁ : ι₁ → Type w₁
      M₁' : Type w₁'
      M₂ : ι₂ → Type w₂
      M₂' : Type w₂'
      inst✝⁹ : (i₁ : ι₁) → AddCommMonoid (M₁ i₁)
      inst✝⁸ : AddCommMonoid M₁'
      inst✝⁷ : (i₂ : ι₂) → AddCommMonoid (M₂ i₂)
      inst✝⁶ : AddCommMonoid M₂'
      inst✝⁵ : (i₁ : ι₁) → Module R (M₁ i₁)
      inst✝⁴ : Module R M₁'
      inst✝³ : (i₂ : ι₂) → Module R (M₂ i₂)
      inst✝² : Module R M₂'
      inst✝¹ : (i₁ : ι₁) → Module S (M₁ i₁)
      inst✝ : ∀ (i₁ : ι₁), IsScalarTower R S (M₁ i₁)
      ⊢ LinearMap (RingHom.id S) (TensorProduct R (DirectSum ι₁ fun i₁ => M₁ i₁) (Di …
    -/
  · refine AlgebraTensorModule.lift ?_
    /-
      case toFun
      R : Type u
      inst✝¹⁴ : CommSemiring R
      S : Type ?u.2020
      inst✝¹³ : Semiring S
      inst✝¹² : Algebra R S
      ι₁ : Type v₁
      ι₂ : Type v₂
      inst✝¹¹ : DecidableEq ι₁
      inst✝¹⁰ : DecidableEq ι₂
      M₁ : ι₁ → Type w₁
      M₁' : Type w₁'
      M₂ : ι₂ → Type w₂
      M₂' : Type w₂'
      inst✝⁹ : (i₁ : ι₁) → AddCommMonoid (M₁ i₁)
      inst✝⁸ : AddCommMonoid M₁'
      inst✝⁷ : (i₂ : ι₂) → AddCommMonoid (M₂ i₂)
      inst✝⁶ : AddCommMonoid M₂'
      inst✝⁵ : (i₁ : ι₁) → Module R (M₁ i₁)
      inst✝⁴ : Module R M₁'
      inst✝³ : (i₂ : ι₂) → Module R (M₂ i₂)
      inst✝² : Module R M₂'
      inst✝¹ : (i₁ : ι₁) → Module S (M₁ i₁)
      inst✝ : ∀ (i₁ : ι₁), IsScalarTower R S (M₁ i₁)
      ⊢ LinearMap (RingHom.id S) (DirectSum ι₁ fun i₁ => M₁ i₁) (LinearMap (RingHom. …
    -/
    refine DirectSum.toModule S _ _ fun i₁ => ?_
    /-
      case toFun
      R : Type u
      inst✝¹⁴ : CommSemiring R
      S : Type ?u.2020
      inst✝¹³ : Semiring S
      inst✝¹² : Algebra R S
      ι₁ : Type v₁
      ι₂ : Type v₂
      inst✝¹¹ : DecidableEq ι₁
      inst✝¹⁰ : DecidableEq ι₂
      M₁ : ι₁ → Type w₁
      M₁' : Type w₁'
      M₂ : ι₂ → Type w₂
      M₂' : Type w₂'
      inst✝⁹ : (i₁ : ι₁) → AddCommMonoid (M₁ i₁)
      inst✝⁸ : AddCommMonoid M₁'
      inst✝⁷ : (i₂ : ι₂) → AddCommMonoid (M₂ i₂)
      inst✝⁶ : AddCommMonoid M₂'
      inst✝⁵ : (i₁ : ι₁) → Module R (M₁ i₁)
      inst✝⁴ : Module R M₁'
      inst✝³ : (i₂ : ι₂) → Module R (M₂ i₂)
      inst✝² : Module R M₂'
      inst✝¹ : (i₁ : ι₁) → Module S (M₁ i₁)
      inst✝ : ∀ (i₁ : ι₁), IsScalarTower R S (M₁ i₁)
      i₁ : ι₁
      ⊢ LinearMap (RingHom.id S) (M₁ i₁) (LinearMap (RingHom.id R) (DirectSum ι₂ fun …
    -/
    refine LinearMap.flip ?_
    /-
      case toFun
      R : Type u
      inst✝¹⁴ : CommSemiring R
      S : Type ?u.2020
      inst✝¹³ : Semiring S
      inst✝¹² : Algebra R S
      ι₁ : Type v₁
      ι₂ : Type v₂
      inst✝¹¹ : DecidableEq ι₁
      inst✝¹⁰ : DecidableEq ι₂
      M₁ : ι₁ → Type w₁
      M₁' : Type w₁'
      M₂ : ι₂ → Type w₂
      M₂' : Type w₂'
      inst✝⁹ : (i₁ : ι₁) → AddCommMonoid (M₁ i₁)
      inst✝⁸ : AddCommMonoid M₁'
      inst✝⁷ : (i₂ : ι₂) → AddCommMonoid (M₂ i₂)
      inst✝⁶ : AddCommMonoid M₂'
      inst✝⁵ : (i₁ : ι₁) → Module R (M₁ i₁)
      inst✝⁴ : Module R M₁'
      inst✝³ : (i₂ : ι₂) → Module R (M₂ i₂)
      inst✝² : Module R M₂'
      inst✝¹ : (i₁ : ι₁) → Module S (M₁ i₁)
      inst✝ : ∀ (i₁ : ι₁), IsScalarTower R S (M₁ i₁)
      i₁ : ι₁
      ⊢ LinearMap (RingHom.id R) (DirectSum ι₂ fun i₂ => M₂ i₂) (LinearMap (RingHom. …
    -/
    refine DirectSum.toModule R _ _ fun i₂ => LinearMap.flip <| ?_
    /-
      case toFun
      R : Type u
      inst✝¹⁴ : CommSemiring R
      S : Type ?u.2020
      inst✝¹³ : Semiring S
      inst✝¹² : Algebra R S
      ι₁ : Type v₁
      ι₂ : Type v₂
      inst✝¹¹ : DecidableEq ι₁
      inst✝¹⁰ : DecidableEq ι₂
      M₁ : ι₁ → Type w₁
      M₁' : Type w₁'
      M₂ : ι₂ → Type w₂
      M₂' : Type w₂'
      inst✝⁹ : (i₁ : ι₁) → AddCommMonoid (M₁ i₁)
      inst✝⁸ : AddCommMonoid M₁'
      inst✝⁷ : (i₂ : ι₂) → AddCommMonoid (M₂ i₂)
      inst✝⁶ : AddCommMonoid M₂'
      inst✝⁵ : (i₁ : ι₁) → Module R (M₁ i₁)
      inst✝⁴ : Module R M₁'
      inst✝³ : (i₂ : ι₂) → Module R (M₂ i₂)
      inst✝² : Module R M₂'
      inst✝¹ : (i₁ : ι₁) → Module S (M₁ i₁)
      inst✝ : ∀ (i₁ : ι₁), IsScalarTower R S (M₁ i₁)
      i₁ : ι₁
      i₂ : ι₂
      ⊢ LinearMap (RingHom.id S) (M₁ i₁) (LinearMap (RingHom.id R) (M₂ i₂) (DirectSu …
    -/
    refine AlgebraTensorModule.curry ?_
    /-
      case toFun
      R : Type u
      inst✝¹⁴ : CommSemiring R
      S : Type ?u.2020
      inst✝¹³ : Semiring S
      inst✝¹² : Algebra R S
      ι₁ : Type v₁
      ι₂ : Type v₂
      inst✝¹¹ : DecidableEq ι₁
      inst✝¹⁰ : DecidableEq ι₂
      M₁ : ι₁ → Type w₁
      M₁' : Type w₁'
      M₂ : ι₂ → Type w₂
      M₂' : Type w₂'
      inst✝⁹ : (i₁ : ι₁) → AddCommMonoid (M₁ i₁)
      inst✝⁸ : AddCommMonoid M₁'
      inst✝⁷ : (i₂ : ι₂) → AddCommMonoid (M₂ i₂)
      inst✝⁶ : AddCommMonoid M₂'
      inst✝⁵ : (i₁ : ι₁) → Module R (M₁ i₁)
      inst✝⁴ : Module R M₁'
      inst✝³ : (i₂ : ι₂) → Module R (M₂ i₂)
      inst✝² : Module R M₂'
      inst✝¹ : (i₁ : ι₁) → Module S (M₁ i₁)
      inst✝ : ∀ (i₁ : ι₁), IsScalarTower R S (M₁ i₁)
      i₁ : ι₁
      i₂ : ι₂
      ⊢ LinearMap (RingHom.id S) (TensorProduct R (M₁ i₁) (M₂ i₂)) (DirectSum (Prod  …
    -/
    exact DirectSum.lof S (ι₁ × ι₂) (fun i => M₁ i.1 ⊗[R] M₂ i.2) (i₁, i₂)
    /-
      🎉 no goals
    -/
    /-
      case invFun
      R : Type u
      inst✝¹⁴ : CommSemiring R
      S : Type ?u.2020
      inst✝¹³ : Semiring S
      inst✝¹² : Algebra R S
      ι₁ : Type v₁
      ι₂ : Type v₂
      inst✝¹¹ : DecidableEq ι₁
      inst✝¹⁰ : DecidableEq ι₂
      M₁ : ι₁ → Type w₁
      M₁' : Type w₁'
      M₂ : ι₂ → Type w₂
      M₂' : Type w₂'
      inst✝⁹ : (i₁ : ι₁) → AddCommMonoid (M₁ i₁)
      inst✝⁸ : AddCommMonoid M₁'
      inst✝⁷ : (i₂ : ι₂) → AddCommMonoid (M₂ i₂)
      inst✝⁶ : AddCommMonoid M₂'
      inst✝⁵ : (i₁ : ι₁) → Module R (M₁ i₁)
      inst✝⁴ : Module R M₁'
      inst✝³ : (i₂ : ι₂) → Module R (M₂ i₂)
      inst✝² : Module R M₂'
      inst✝¹ : (i₁ : ι₁) → Module S (M₁ i₁)
      inst✝ : ∀ (i₁ : ι₁), IsScalarTower R S (M₁ i₁)
      ⊢ LinearMap (RingHom.id S) (DirectSum (Prod ι₁ ι₂) fun i => TensorProduct R (M …
    -/
  · refine DirectSum.toModule S _ _ fun i => ?_
    /-
      case invFun
      R : Type u
      inst✝¹⁴ : CommSemiring R
      S : Type ?u.2020
      inst✝¹³ : Semiring S
      inst✝¹² : Algebra R S
      ι₁ : Type v₁
      ι₂ : Type v₂
      inst✝¹¹ : DecidableEq ι₁
      inst✝¹⁰ : DecidableEq ι₂
      M₁ : ι₁ → Type w₁
      M₁' : Type w₁'
      M₂ : ι₂ → Type w₂
      M₂' : Type w₂'
      inst✝⁹ : (i₁ : ι₁) → AddCommMonoid (M₁ i₁)
      inst✝⁸ : AddCommMonoid M₁'
      inst✝⁷ : (i₂ : ι₂) → AddCommMonoid (M₂ i₂)
      inst✝⁶ : AddCommMonoid M₂'
      inst✝⁵ : (i₁ : ι₁) → Module R (M₁ i₁)
      inst✝⁴ : Module R M₁'
      inst✝³ : (i₂ : ι₂) → Module R (M₂ i₂)
      inst✝² : Module R M₂'
      inst✝¹ : (i₁ : ι₁) → Module S (M₁ i₁)
      inst✝ : ∀ (i₁ : ι₁), IsScalarTower R S (M₁ i₁)
      i : Prod ι₁ ι₂
      ⊢ LinearMap (RingHom.id S) (TensorProduct R (M₁ i.1) (M₂ i.2)) (TensorProduct  …
    -/
    exact AlgebraTensorModule.map (DirectSum.lof S _ M₁ i.1) (DirectSum.lof R _ M₂ i.2)
    /-
      🎉 no goals
    -/
    /-
      case left
      R : Type u
      inst✝¹⁴ : CommSemiring R
      S : Type ?u.2020
      inst✝¹³ : Semiring S
      inst✝¹² : Algebra R S
      ι₁ : Type v₁
      ι₂ : Type v₂
      inst✝¹¹ : DecidableEq ι₁
      inst✝¹⁰ : DecidableEq ι₂
      M₁ : ι₁ → Type w₁
      M₁' : Type w₁'
      M₂ : ι₂ → Type w₂
      M₂' : Type w₂'
      inst✝⁹ : (i₁ : ι₁) → AddCommMonoid (M₁ i₁)
      inst✝⁸ : AddCommMonoid M₁'
      inst✝⁷ : (i₂ : ι₂) → AddCommMonoid (M₂ i₂)
      inst✝⁶ : AddCommMonoid M₂'
      inst✝⁵ : (i₁ : ι₁) → Module R (M₁ i₁)
      inst✝⁴ : Module R M₁'
      inst✝³ : (i₂ : ι₂) → Module R (M₂ i₂)
      inst✝² : Module R M₂'
      inst✝¹ : (i₁ : ι₁) → Module S (M₁ i₁)
      inst✝ : ∀ (i₁ : ι₁), IsScalarTower R S (M₁ i₁)
      ⊢ Eq ((TensorProduct.AlgebraTensorModule.lift (DirectSum.toModule S ι₁ (Linear …
    -/
  · refine DirectSum.linearMap_ext S fun ⟨i₁, i₂⟩ => ?_
    /-
      case left
      R : Type u
      inst✝¹⁴ : CommSemiring R
      S : Type ?u.2020
      inst✝¹³ : Semiring S
      inst✝¹² : Algebra R S
      ι₁ : Type v₁
      ι₂ : Type v₂
      inst✝¹¹ : DecidableEq ι₁
      inst✝¹⁰ : DecidableEq ι₂
      M₁ : ι₁ → Type w₁
      M₁' : Type w₁'
      M₂ : ι₂ → Type w₂
      M₂' : Type w₂'
      inst✝⁹ : (i₁ : ι₁) → AddCommMonoid (M₁ i₁)
      inst✝⁸ : AddCommMonoid M₁'
      inst✝⁷ : (i₂ : ι₂) → AddCommMonoid (M₂ i₂)
      inst✝⁶ : AddCommMonoid M₂'
      inst✝⁵ : (i₁ : ι₁) → Module R (M₁ i₁)
      inst✝⁴ : Module R M₁'
      inst✝³ : (i₂ : ι₂) → Module R (M₂ i₂)
      inst✝² : Module R M₂'
      inst✝¹ : (i₁ : ι₁) → Module S (M₁ i₁)
      inst✝ : ∀ (i₁ : ι₁), IsScalarTower R S (M₁ i₁)
      x✝ : Prod ι₁ ι₂
      i₁ : ι₁
      i₂ : ι₂
      ⊢ Eq (((TensorProduct.AlgebraTensorModule.lift (DirectSum.toModule S ι₁ (Linea …
    -/
    refine TensorProduct.AlgebraTensorModule.ext fun m₁ m₂ => ?_
    -- Porting note: seems much nicer than the `repeat` lean 3 proof.
    simp only [coe_comp, Function.comp_apply, toModule_lof, AlgebraTensorModule.map_tmul,
      AlgebraTensorModule.lift_apply, lift.tmul, coe_restrictScalars, flip_apply,
      AlgebraTensorModule.curry_apply, curry_apply, id_comp]
  · -- `(_)` prevents typeclass search timing out on problems that can be solved immediately by
    -- unification
    /-
      case right
      R : Type u
      inst✝¹⁴ : CommSemiring R
      S : Type ?u.2020
      inst✝¹³ : Semiring S
      inst✝¹² : Algebra R S
      ι₁ : Type v₁
      ι₂ : Type v₂
      inst✝¹¹ : DecidableEq ι₁
      inst✝¹⁰ : DecidableEq ι₂
      M₁ : ι₁ → Type w₁
      M₁' : Type w₁'
      M₂ : ι₂ → Type w₂
      M₂' : Type w₂'
      inst✝⁹ : (i₁ : ι₁) → AddCommMonoid (M₁ i₁)
      inst✝⁸ : AddCommMonoid M₁'
      inst✝⁷ : (i₂ : ι₂) → AddCommMonoid (M₂ i₂)
      inst✝⁶ : AddCommMonoid M₂'
      inst✝⁵ : (i₁ : ι₁) → Module R (M₁ i₁)
      inst✝⁴ : Module R M₁'
      inst✝³ : (i₂ : ι₂) → Module R (M₂ i₂)
      inst✝² : Module R M₂'
      inst✝¹ : (i₁ : ι₁) → Module S (M₁ i₁)
      inst✝ : ∀ (i₁ : ι₁), IsScalarTower R S (M₁ i₁)
      ⊢ Eq ((DirectSum.toModule S (Prod ι₁ ι₂) (TensorProduct R (DirectSum ι₁ fun i₁ …
    -/
    apply TensorProduct.AlgebraTensorModule.curry_injective
    /-
      case right.a
      R : Type u
      inst✝¹⁴ : CommSemiring R
      S : Type ?u.2020
      inst✝¹³ : Semiring S
      inst✝¹² : Algebra R S
      ι₁ : Type v₁
      ι₂ : Type v₂
      inst✝¹¹ : DecidableEq ι₁
      inst✝¹⁰ : DecidableEq ι₂
      M₁ : ι₁ → Type w₁
      M₁' : Type w₁'
      M₂ : ι₂ → Type w₂
      M₂' : Type w₂'
      inst✝⁹ : (i₁ : ι₁) → AddCommMonoid (M₁ i₁)
      inst✝⁸ : AddCommMonoid M₁'
      inst✝⁷ : (i₂ : ι₂) → AddCommMonoid (M₂ i₂)
      inst✝⁶ : AddCommMonoid M₂'
      inst✝⁵ : (i₁ : ι₁) → Module R (M₁ i₁)
      inst✝⁴ : Module R M₁'
      inst✝³ : (i₂ : ι₂) → Module R (M₂ i₂)
      inst✝² : Module R M₂'
      inst✝¹ : (i₁ : ι₁) → Module S (M₁ i₁)
      inst✝ : ∀ (i₁ : ι₁), IsScalarTower R S (M₁ i₁)
      ⊢ Eq (TensorProduct.AlgebraTensorModule.curry ((DirectSum.toModule S (Prod ι₁  …
    -/
    refine DirectSum.linearMap_ext _ fun i₁ => ?_
    /-
      case right.a
      R : Type u
      inst✝¹⁴ : CommSemiring R
      S : Type ?u.2020
      inst✝¹³ : Semiring S
      inst✝¹² : Algebra R S
      ι₁ : Type v₁
      ι₂ : Type v₂
      inst✝¹¹ : DecidableEq ι₁
      inst✝¹⁰ : DecidableEq ι₂
      M₁ : ι₁ → Type w₁
      M₁' : Type w₁'
      M₂ : ι₂ → Type w₂
      M₂' : Type w₂'
      inst✝⁹ : (i₁ : ι₁) → AddCommMonoid (M₁ i₁)
      inst✝⁸ : AddCommMonoid M₁'
      inst✝⁷ : (i₂ : ι₂) → AddCommMonoid (M₂ i₂)
      inst✝⁶ : AddCommMonoid M₂'
      inst✝⁵ : (i₁ : ι₁) → Module R (M₁ i₁)
      inst✝⁴ : Module R M₁'
      inst✝³ : (i₂ : ι₂) → Module R (M₂ i₂)
      inst✝² : Module R M₂'
      inst✝¹ : (i₁ : ι₁) → Module S (M₁ i₁)
      inst✝ : ∀ (i₁ : ι₁), IsScalarTower R S (M₁ i₁)
      i₁ : ι₁
      ⊢ Eq ((TensorProduct.AlgebraTensorModule.curry ((DirectSum.toModule S (Prod ι₁ …
    -/
    refine LinearMap.ext fun x₁ => ?_
    /-
      case right.a
      R : Type u
      inst✝¹⁴ : CommSemiring R
      S : Type ?u.2020
      inst✝¹³ : Semiring S
      inst✝¹² : Algebra R S
      ι₁ : Type v₁
      ι₂ : Type v₂
      inst✝¹¹ : DecidableEq ι₁
      inst✝¹⁰ : DecidableEq ι₂
      M₁ : ι₁ → Type w₁
      M₁' : Type w₁'
      M₂ : ι₂ → Type w₂
      M₂' : Type w₂'
      inst✝⁹ : (i₁ : ι₁) → AddCommMonoid (M₁ i₁)
      inst✝⁸ : AddCommMonoid M₁'
      inst✝⁷ : (i₂ : ι₂) → AddCommMonoid (M₂ i₂)
      inst✝⁶ : AddCommMonoid M₂'
      inst✝⁵ : (i₁ : ι₁) → Module R (M₁ i₁)
      inst✝⁴ : Module R M₁'
      inst✝³ : (i₂ : ι₂) → Module R (M₂ i₂)
      inst✝² : Module R M₂'
      inst✝¹ : (i₁ : ι₁) → Module S (M₁ i₁)
      inst✝ : ∀ (i₁ : ι₁), IsScalarTower R S (M₁ i₁)
      i₁ : ι₁
      x₁ : M₁ i₁
      ⊢ Eq (((TensorProduct.AlgebraTensorModule.curry ((DirectSum.toModule S (Prod ι …
    -/
    refine DirectSum.linearMap_ext _ fun i₂ => ?_
    /-
      case right.a
      R : Type u
      inst✝¹⁴ : CommSemiring R
      S : Type ?u.2020
      inst✝¹³ : Semiring S
      inst✝¹² : Algebra R S
      ι₁ : Type v₁
      ι₂ : Type v₂
      inst✝¹¹ : DecidableEq ι₁
      inst✝¹⁰ : DecidableEq ι₂
      M₁ : ι₁ → Type w₁
      M₁' : Type w₁'
      M₂ : ι₂ → Type w₂
      M₂' : Type w₂'
      inst✝⁹ : (i₁ : ι₁) → AddCommMonoid (M₁ i₁)
      inst✝⁸ : AddCommMonoid M₁'
      inst✝⁷ : (i₂ : ι₂) → AddCommMonoid (M₂ i₂)
      inst✝⁶ : AddCommMonoid M₂'
      inst✝⁵ : (i₁ : ι₁) → Module R (M₁ i₁)
      inst✝⁴ : Module R M₁'
      inst✝³ : (i₂ : ι₂) → Module R (M₂ i₂)
      inst✝² : Module R M₂'
      inst✝¹ : (i₁ : ι₁) → Module S (M₁ i₁)
      inst✝ : ∀ (i₁ : ι₁), IsScalarTower R S (M₁ i₁)
      i₁ : ι₁
      x₁ : M₁ i₁
      i₂ : ι₂
      ⊢ Eq ((((TensorProduct.AlgebraTensorModule.curry ((DirectSum.toModule S (Prod  …
    -/
    refine LinearMap.ext fun x₂ => ?_
    -- Porting note: seems much nicer than the `repeat` lean 3 proof.
    simp only [coe_comp, Function.comp_apply, AlgebraTensorModule.curry_apply, curry_apply,
      coe_restrictScalars, AlgebraTensorModule.lift_apply, lift.tmul, toModule_lof, flip_apply,
      AlgebraTensorModule.map_tmul, id_coe, id_eq]
  /- was:
    refine'
      LinearEquiv.ofLinear
        (lift <|
          DirectSum.toModule R _ _ fun i₁ => LinearMap.flip <| DirectSum.toModule R _ _ fun i₂ =>
                LinearMap.flip <| curry <|
                  DirectSum.lof R (ι₁ × ι₂) (fun i => M₁ i.1 ⊗[R] M₂ i.2) (i₁, i₂))
        (DirectSum.toModule R _ _ fun i => map (DirectSum.lof R _ _ _) (DirectSum.lof R _ _ _)) _
        _ <;>
    [ext ⟨i₁, i₂⟩ x₁ x₂ : 4, ext i₁ i₂ x₁ x₂ : 5]
  repeat'
    first
      |rw [compr₂_apply]|rw [comp_apply]|rw [id_apply]|rw [mk_apply]|rw [DirectSum.toModule_lof]
      |rw [map_tmul]|rw [lift.tmul]|rw [flip_apply]|rw [curry_apply]
  -/

/- alternative with explicit types:
  refine'
      LinearEquiv.ofLinear
        (lift <|
          DirectSum.toModule
            (R := R) (M := M₁) (N := (⨁ i₂, M₂ i₂) →ₗ[R] ⨁ i : ι₁ × ι₂, M₁ i.1 ⊗[R] M₂ i.2)
            (φ := fun i₁ => LinearMap.flip <|
              DirectSum.toModule (R := R) (M := M₂) (N := ⨁ i : ι₁ × ι₂, M₁ i.1 ⊗[R] M₂ i.2)
              (φ := fun i₂ => LinearMap.flip <| curry <|
                  DirectSum.lof R (ι₁ × ι₂) (fun i => M₁ i.1 ⊗[R] M₂ i.2) (i₁, i₂))))
        (DirectSum.toModule
          (R := R)
          (M := fun i : ι₁ × ι₂ => M₁ i.1 ⊗[R] M₂ i.2)
          (N := (⨁ i₁, M₁ i₁) ⊗[R] ⨁ i₂, M₂ i₂)
          (φ := fun i : ι₁ × ι₂ => map (DirectSum.lof R _ M₁ i.1) (DirectSum.lof R _ M₂ i.2))) _
        _ <;>
    [ext ⟨i₁, i₂⟩ x₁ x₂ : 4, ext i₁ i₂ x₁ x₂ : 5]
  repeat'
    first
      |rw [compr₂_apply]|rw [comp_apply]|rw [id_apply]|rw [mk_apply]|rw [DirectSum.toModule_lof]
      |rw [map_tmul]|rw [lift.tmul]|rw [flip_apply]|rw [curry_apply]
-/


/-- Tensor products distribute over a direct sum on the left . -/
def directSumLeft : (⨁ i₁, M₁ i₁) ⊗[R] M₂' ≃ₗ[R] ⨁ i, M₁ i ⊗[R] M₂' :=
  LinearEquiv.ofLinear
    (lift <|
      DirectSum.toModule R _ _ fun _ =>
        (mk R _ _).compr₂ <| DirectSum.lof R ι₁ (fun i => M₁ i ⊗[R] M₂') _)
    (DirectSum.toModule R _ _ fun _ => rTensor _ (DirectSum.lof R ι₁ _ _))
    (DirectSum.linearMap_ext R fun i =>
      TensorProduct.ext <|
        LinearMap.ext₂ fun m₁ m₂ => by
          /-
            R : Type u
            inst✝¹⁴ : CommSemiring R
            S : Type ?u.37376
            inst✝¹³ : Semiring S
            inst✝¹² : Algebra R S
            ι₁ : Type v₁
            ι₂ : Type v₂
            inst✝¹¹ : DecidableEq ι₁
            inst✝¹⁰ : DecidableEq ι₂
            M₁ : ι₁ → Type w₁
            M₁' : Type w₁'
            M₂ : ι₂ → Type w₂
            M₂' : Type w₂'
            inst✝⁹ : (i₁ : ι₁) → AddCommMonoid (M₁ i₁)
            inst✝⁸ : AddCommMonoid M₁'
            inst✝⁷ : (i₂ : ι₂) → AddCommMonoid (M₂ i₂)
            inst✝⁶ : AddCommMonoid M₂'
            inst✝⁵ : (i₁ : ι₁) → Module R (M₁ i₁)
            inst✝⁴ : Module R M₁'
            inst✝³ : (i₂ : ι₂) → Module R (M₂ i₂)
            inst✝² : Module R M₂'
            inst✝¹ : (i₁ : ι₁) → Module S (M₁ i₁)
            inst✝ : ∀ (i₁ : ι₁), IsScalarTower R S (M₁ i₁)
            i : ι₁
            m₁ : M₁ i
            m₂ : M₂'
            ⊢ Eq ((((TensorProduct.mk R (M₁ i) M₂').compr₂ (((TensorProduct.lift (DirectSu …
          -/
          dsimp only [comp_apply, compr₂_apply, id_apply, mk_apply]
          simp_rw [DirectSum.toModule_lof, rTensor_tmul, lift.tmul, DirectSum.toModule_lof,
            compr₂_apply, mk_apply])
    (TensorProduct.ext <|
      DirectSum.linearMap_ext R fun i =>
        LinearMap.ext₂ fun m₁ m₂ => by
          /-
            R : Type u
            inst✝¹⁴ : CommSemiring R
            S : Type ?u.37376
            inst✝¹³ : Semiring S
            inst✝¹² : Algebra R S
            ι₁ : Type v₁
            ι₂ : Type v₂
            inst✝¹¹ : DecidableEq ι₁
            inst✝¹⁰ : DecidableEq ι₂
            M₁ : ι₁ → Type w₁
            M₁' : Type w₁'
            M₂ : ι₂ → Type w₂
            M₂' : Type w₂'
            inst✝⁹ : (i₁ : ι₁) → AddCommMonoid (M₁ i₁)
            inst✝⁸ : AddCommMonoid M₁'
            inst✝⁷ : (i₂ : ι₂) → AddCommMonoid (M₂ i₂)
            inst✝⁶ : AddCommMonoid M₂'
            inst✝⁵ : (i₁ : ι₁) → Module R (M₁ i₁)
            inst✝⁴ : Module R M₁'
            inst✝³ : (i₂ : ι₂) → Module R (M₂ i₂)
            inst✝² : Module R M₂'
            inst✝¹ : (i₁ : ι₁) → Module S (M₁ i₁)
            inst✝ : ∀ (i₁ : ι₁), IsScalarTower R S (M₁ i₁)
            i : ι₁
            m₁ : M₁ i
            m₂ : M₂'
            ⊢ Eq (((((TensorProduct.mk R (DirectSum ι₁ fun i₁ => M₁ i₁) M₂').compr₂ ((Dire …
          -/
          dsimp only [comp_apply, compr₂_apply, id_apply, mk_apply]
          simp_rw [lift.tmul, DirectSum.toModule_lof, compr₂_apply,
            mk_apply, DirectSum.toModule_lof, rTensor_tmul])


/-- Tensor products distribute over a direct sum on the right. -/
def directSumRight : (M₁' ⊗[R] ⨁ i, M₂ i) ≃ₗ[R] ⨁ i, M₁' ⊗[R] M₂ i :=
  TensorProduct.comm R _ _ ≪≫ₗ directSumLeft R M₂ M₁' ≪≫ₗ
    DFinsupp.mapRange.linearEquiv fun _ => TensorProduct.comm R _ _


@[simp]
theorem directSum_lof_tmul_lof (i₁ : ι₁) (m₁ : M₁ i₁) (i₂ : ι₂) (m₂ : M₂ i₂) :
    TensorProduct.directSum R S M₁ M₂ (DirectSum.lof S ι₁ M₁ i₁ m₁ ⊗ₜ DirectSum.lof R ι₂ M₂ i₂ m₂) =
      DirectSum.lof S (ι₁ × ι₂) (fun i => M₁ i.1 ⊗[R] M₂ i.2) (i₁, i₂) (m₁ ⊗ₜ m₂) := by
  /-
    R : Type u
    inst✝¹⁰ : CommSemiring R
    S : Type u_1
    inst✝⁹ : Semiring S
    inst✝⁸ : Algebra R S
    ι₁ : Type v₁
    ι₂ : Type v₂
    inst✝⁷ : DecidableEq ι₁
    inst✝⁶ : DecidableEq ι₂
    M₁ : ι₁ → Type w₁
    M₂ : ι₂ → Type w₂
    inst✝⁵ : (i₁ : ι₁) → AddCommMonoid (M₁ i₁)
    inst✝⁴ : (i₂ : ι₂) → AddCommMonoid (M₂ i₂)
    inst✝³ : (i₁ : ι₁) → Module R (M₁ i₁)
    inst✝² : (i₂ : ι₂) → Module R (M₂ i₂)
    inst✝¹ : (i₁ : ι₁) → Module S (M₁ i₁)
    inst✝ : ∀ (i₁ : ι₁), IsScalarTower R S (M₁ i₁)
    i₁ : ι₁
    m₁ : M₁ i₁
    i₂ : ι₂
    m₂ : M₂ i₂
    ⊢ Eq ((TensorProduct.directSum R S M₁ M₂) (TensorProduct.tmul R ((DirectSum.lo …
  -/
  simp [TensorProduct.directSum]
  /-
    🎉 no goals
  -/


@[simp]
theorem directSum_symm_lof_tmul (i₁ : ι₁) (m₁ : M₁ i₁) (i₂ : ι₂) (m₂ : M₂ i₂) :
    (TensorProduct.directSum R S M₁ M₂).symm
      (DirectSum.lof S (ι₁ × ι₂) (fun i => M₁ i.1 ⊗[R] M₂ i.2) (i₁, i₂) (m₁ ⊗ₜ m₂)) =
      (DirectSum.lof S ι₁ M₁ i₁ m₁ ⊗ₜ DirectSum.lof R ι₂ M₂ i₂ m₂) := by
  /-
    R : Type u
    inst✝¹⁰ : CommSemiring R
    S : Type u_1
    inst✝⁹ : Semiring S
    inst✝⁸ : Algebra R S
    ι₁ : Type v₁
    ι₂ : Type v₂
    inst✝⁷ : DecidableEq ι₁
    inst✝⁶ : DecidableEq ι₂
    M₁ : ι₁ → Type w₁
    M₂ : ι₂ → Type w₂
    inst✝⁵ : (i₁ : ι₁) → AddCommMonoid (M₁ i₁)
    inst✝⁴ : (i₂ : ι₂) → AddCommMonoid (M₂ i₂)
    inst✝³ : (i₁ : ι₁) → Module R (M₁ i₁)
    inst✝² : (i₂ : ι₂) → Module R (M₂ i₂)
    inst✝¹ : (i₁ : ι₁) → Module S (M₁ i₁)
    inst✝ : ∀ (i₁ : ι₁), IsScalarTower R S (M₁ i₁)
    i₁ : ι₁
    m₁ : M₁ i₁
    i₂ : ι₂
    m₂ : M₂ i₂
    ⊢ Eq ((TensorProduct.directSum R S M₁ M₂).symm ((DirectSum.lof S (Prod ι₁ ι₂)  …
  -/
  rw [LinearEquiv.symm_apply_eq, directSum_lof_tmul_lof]
  /-
    🎉 no goals
  -/


@[simp]
theorem directSumLeft_tmul_lof (i : ι₁) (x : M₁ i) (y : M₂') :
    directSumLeft R M₁ M₂' (DirectSum.lof R _ _ i x ⊗ₜ[R] y) =
    DirectSum.lof R _ _ i (x ⊗ₜ[R] y) := by
  /-
    R : Type u
    inst✝⁵ : CommSemiring R
    ι₁ : Type v₁
    inst✝⁴ : DecidableEq ι₁
    M₁ : ι₁ → Type w₁
    M₂' : Type w₂'
    inst✝³ : (i₁ : ι₁) → AddCommMonoid (M₁ i₁)
    inst✝² : AddCommMonoid M₂'
    inst✝¹ : (i₁ : ι₁) → Module R (M₁ i₁)
    inst✝ : Module R M₂'
    i : ι₁
    x : M₁ i
    y : M₂'
    ⊢ Eq ((TensorProduct.directSumLeft R M₁ M₂') (TensorProduct.tmul R ((DirectSum …
  -/
  dsimp only [directSumLeft, LinearEquiv.ofLinear_apply, lift.tmul]
  /-
    R : Type u
    inst✝⁵ : CommSemiring R
    ι₁ : Type v₁
    inst✝⁴ : DecidableEq ι₁
    M₁ : ι₁ → Type w₁
    M₂' : Type w₂'
    inst✝³ : (i₁ : ι₁) → AddCommMonoid (M₁ i₁)
    inst✝² : AddCommMonoid M₂'
    inst✝¹ : (i₁ : ι₁) → Module R (M₁ i₁)
    inst✝ : Module R M₂'
    i : ι₁
    x : M₁ i
    y : M₂'
    ⊢ Eq (((DirectSum.toModule R ι₁ (LinearMap (RingHom.id R) M₂' (DirectSum ι₁ fu …
  -/
  rw [DirectSum.toModule_lof R i]
  /-
    R : Type u
    inst✝⁵ : CommSemiring R
    ι₁ : Type v₁
    inst✝⁴ : DecidableEq ι₁
    M₁ : ι₁ → Type w₁
    M₂' : Type w₂'
    inst✝³ : (i₁ : ι₁) → AddCommMonoid (M₁ i₁)
    inst✝² : AddCommMonoid M₂'
    inst✝¹ : (i₁ : ι₁) → Module R (M₁ i₁)
    inst✝ : Module R M₂'
    i : ι₁
    x : M₁ i
    y : M₂'
    ⊢ Eq ((((TensorProduct.mk R (M₁ i) M₂').compr₂ (DirectSum.lof R ι₁ (fun i => T …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem directSumLeft_symm_lof_tmul (i : ι₁) (x : M₁ i) (y : M₂') :
    (directSumLeft R M₁ M₂').symm (DirectSum.lof R _ _ i (x ⊗ₜ[R] y)) =
      DirectSum.lof R _ _ i x ⊗ₜ[R] y := by
  /-
    R : Type u
    inst✝⁵ : CommSemiring R
    ι₁ : Type v₁
    inst✝⁴ : DecidableEq ι₁
    M₁ : ι₁ → Type w₁
    M₂' : Type w₂'
    inst✝³ : (i₁ : ι₁) → AddCommMonoid (M₁ i₁)
    inst✝² : AddCommMonoid M₂'
    inst✝¹ : (i₁ : ι₁) → Module R (M₁ i₁)
    inst✝ : Module R M₂'
    i : ι₁
    x : M₁ i
    y : M₂'
    ⊢ Eq ((TensorProduct.directSumLeft R M₁ M₂').symm ((DirectSum.lof R ι₁ (fun i  …
  -/
  rw [LinearEquiv.symm_apply_eq, directSumLeft_tmul_lof]
  /-
    🎉 no goals
  -/


@[simp]
theorem directSumRight_tmul_lof (x : M₁') (i : ι₂) (y : M₂ i) :
    directSumRight R M₁' M₂ (x ⊗ₜ[R] DirectSum.lof R _ _ i y) =
    DirectSum.lof R _ _ i (x ⊗ₜ[R] y) := by
  /-
    R : Type u
    inst✝⁵ : CommSemiring R
    ι₂ : Type v₂
    inst✝⁴ : DecidableEq ι₂
    M₁' : Type w₁'
    M₂ : ι₂ → Type w₂
    inst✝³ : AddCommMonoid M₁'
    inst✝² : (i₂ : ι₂) → AddCommMonoid (M₂ i₂)
    inst✝¹ : Module R M₁'
    inst✝ : (i₂ : ι₂) → Module R (M₂ i₂)
    x : M₁'
    i : ι₂
    y : M₂ i
    ⊢ Eq ((TensorProduct.directSumRight R M₁' M₂) (TensorProduct.tmul R x ((Direct …
  -/
  dsimp only [directSumRight, LinearEquiv.trans_apply, TensorProduct.comm_tmul]
  /-
    R : Type u
    inst✝⁵ : CommSemiring R
    ι₂ : Type v₂
    inst✝⁴ : DecidableEq ι₂
    M₁' : Type w₁'
    M₂ : ι₂ → Type w₂
    inst✝³ : AddCommMonoid M₁'
    inst✝² : (i₂ : ι₂) → AddCommMonoid (M₂ i₂)
    inst✝¹ : Module R M₁'
    inst✝ : (i₂ : ι₂) → Module R (M₂ i₂)
    x : M₁'
    i : ι₂
    y : M₂ i
    ⊢ Eq ((DFinsupp.mapRange.linearEquiv fun x => TensorProduct.comm R (M₂ x) M₁') …
  -/
  rw [directSumLeft_tmul_lof]
  /-
    R : Type u
    inst✝⁵ : CommSemiring R
    ι₂ : Type v₂
    inst✝⁴ : DecidableEq ι₂
    M₁' : Type w₁'
    M₂ : ι₂ → Type w₂
    inst✝³ : AddCommMonoid M₁'
    inst✝² : (i₂ : ι₂) → AddCommMonoid (M₂ i₂)
    inst✝¹ : Module R M₁'
    inst✝ : (i₂ : ι₂) → Module R (M₂ i₂)
    x : M₁'
    i : ι₂
    y : M₂ i
    ⊢ Eq ((DFinsupp.mapRange.linearEquiv fun x => TensorProduct.comm R (M₂ x) M₁') …
  -/
  exact DFinsupp.mapRange_single (hf := fun _ => rfl)
  /-
    🎉 no goals
  -/


@[simp]
theorem directSumRight_symm_lof_tmul (x : M₁') (i : ι₂) (y : M₂ i) :
    (directSumRight R M₁' M₂).symm (DirectSum.lof R _ _ i (x ⊗ₜ[R] y)) =
      x ⊗ₜ[R] DirectSum.lof R _ _ i y := by
  /-
    R : Type u
    inst✝⁵ : CommSemiring R
    ι₂ : Type v₂
    inst✝⁴ : DecidableEq ι₂
    M₁' : Type w₁'
    M₂ : ι₂ → Type w₂
    inst✝³ : AddCommMonoid M₁'
    inst✝² : (i₂ : ι₂) → AddCommMonoid (M₂ i₂)
    inst✝¹ : Module R M₁'
    inst✝ : (i₂ : ι₂) → Module R (M₂ i₂)
    x : M₁'
    i : ι₂
    y : M₂ i
    ⊢ Eq ((TensorProduct.directSumRight R M₁' M₂).symm ((DirectSum.lof R ι₂ (fun i …
  -/
  rw [LinearEquiv.symm_apply_eq, directSumRight_tmul_lof]
  /-
    🎉 no goals
  -/


lemma directSumRight_comp_rTensor (f : M₁' →ₗ[R] M₂'):
    (directSumRight R M₂' M₁).toLinearMap ∘ₗ f.rTensor _ =
      (lmap fun _ ↦ f.rTensor _) ∘ₗ directSumRight R M₁' M₁ := by
  /-
    R : Type u
    inst✝⁷ : CommSemiring R
    ι₁ : Type v₁
    inst✝⁶ : DecidableEq ι₁
    M₁ : ι₁ → Type w₁
    M₁' : Type w₁'
    M₂' : Type w₂'
    inst✝⁵ : (i₁ : ι₁) → AddCommMonoid (M₁ i₁)
    inst✝⁴ : AddCommMonoid M₁'
    inst✝³ : AddCommMonoid M₂'
    inst✝² : (i₁ : ι₁) → Module R (M₁ i₁)
    inst✝¹ : Module R M₁'
    inst✝ : Module R M₂'
    f : LinearMap (RingHom.id R) M₁' M₂'
    ⊢ Eq ((↑(TensorProduct.directSumRight R M₂' M₁)).comp (LinearMap.rTensor (Dire …
  -/
  ext; simp
       /-
         🎉 no goals
       -/


