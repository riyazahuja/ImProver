/-- The kernel of a linear map taking values in a simple module over the base ring is closed or
dense. Applies, e.g., to the case when `R = N` is a division ring. -/
theorem LinearMap.isClosed_or_dense_ker (l : M →ₗ[R] N) :
    IsClosed (LinearMap.ker l : Set M) ∨ Dense (LinearMap.ker l : Set M) := by
  /-
    R : Type u
    M : Type v
    N : Type w
    inst✝⁹ : Ring R
    inst✝⁸ : TopologicalSpace R
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : Module R M
    inst✝³ : ContinuousSMul R M
    inst✝² : Module R N
    inst✝¹ : ContinuousAdd M
    inst✝ : IsSimpleModule R N
    l : LinearMap (RingHom.id R) M N
    ⊢ Or (IsClosed ↑(LinearMap.ker l)) (Dense ↑(LinearMap.ker l))
  -/
  rcases l.surjective_or_eq_zero with (hl | rfl)
    /-
      case inl
      R : Type u
      M : Type v
      N : Type w
      inst✝⁹ : Ring R
      inst✝⁸ : TopologicalSpace R
      inst✝⁷ : TopologicalSpace M
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R M
      inst✝³ : ContinuousSMul R M
      inst✝² : Module R N
      inst✝¹ : ContinuousAdd M
      inst✝ : IsSimpleModule R N
      l : LinearMap (RingHom.id R) M N
      hl : Function.Surjective ⇑l
      ⊢ Or (IsClosed ↑(LinearMap.ker l)) (Dense ↑(LinearMap.ker l))
    -/
  · exact l.ker.isClosed_or_dense_of_isCoatom (LinearMap.isCoatom_ker_of_surjective hl)
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u
      M : Type v
      N : Type w
      inst✝⁹ : Ring R
      inst✝⁸ : TopologicalSpace R
      inst✝⁷ : TopologicalSpace M
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R M
      inst✝³ : ContinuousSMul R M
      inst✝² : Module R N
      inst✝¹ : ContinuousAdd M
      inst✝ : IsSimpleModule R N
      ⊢ Or (IsClosed ↑(LinearMap.ker 0)) (Dense ↑(LinearMap.ker 0))
    -/
  · rw [LinearMap.ker_zero]
    /-
      case inr
      R : Type u
      M : Type v
      N : Type w
      inst✝⁹ : Ring R
      inst✝⁸ : TopologicalSpace R
      inst✝⁷ : TopologicalSpace M
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R M
      inst✝³ : ContinuousSMul R M
      inst✝² : Module R N
      inst✝¹ : ContinuousAdd M
      inst✝ : IsSimpleModule R N
      ⊢ Or (IsClosed ↑Top.top) (Dense ↑Top.top)
    -/
    left
    /-
      case inr.h
      R : Type u
      M : Type v
      N : Type w
      inst✝⁹ : Ring R
      inst✝⁸ : TopologicalSpace R
      inst✝⁷ : TopologicalSpace M
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R M
      inst✝³ : ContinuousSMul R M
      inst✝² : Module R N
      inst✝¹ : ContinuousAdd M
      inst✝ : IsSimpleModule R N
      ⊢ IsClosed ↑Top.top
    -/
    exact isClosed_univ
    /-
      🎉 no goals
    -/

