theorem isTorsion_of_aeval_eq_zero [CommSemiring R] [NoZeroDivisors R] [Semiring A] [Algebra R A]
    [AddCommMonoid M] [Module A M] [Module R M] [IsScalarTower R A M]
    {p : R[X]} (h : aeval a p = 0) (h' : p ≠ 0) :
    IsTorsion R[X] (AEval R M a) := by
  /-
    R : Type u_1
    M : Type u_3
    A : Type u_4
    a : A
    inst✝⁷ : CommSemiring R
    inst✝⁶ : NoZeroDivisors R
    inst✝⁵ : Semiring A
    inst✝⁴ : Algebra R A
    inst✝³ : AddCommMonoid M
    inst✝² : Module A M
    inst✝¹ : Module R M
    inst✝ : IsScalarTower R A M
    p : Polynomial R
    h : Eq ((Polynomial.aeval a) p) 0
    h' : Ne p 0
    ⊢ Module.IsTorsion (Polynomial R) (Module.AEval R M a)
  -/
  have hp : p ∈ nonZeroDivisors R[X] := fun q hq ↦ Or.resolve_right (mul_eq_zero.mp hq) h'
  /-
    R : Type u_1
    M : Type u_3
    A : Type u_4
    a : A
    inst✝⁷ : CommSemiring R
    inst✝⁶ : NoZeroDivisors R
    inst✝⁵ : Semiring A
    inst✝⁴ : Algebra R A
    inst✝³ : AddCommMonoid M
    inst✝² : Module A M
    inst✝¹ : Module R M
    inst✝ : IsScalarTower R A M
    p : Polynomial R
    h : Eq ((Polynomial.aeval a) p) 0
    h' : Ne p 0
    hp : Membership.mem (nonZeroDivisors (Polynomial R)) p
    ⊢ Module.IsTorsion (Polynomial R) (Module.AEval R M a)
  -/
  exact fun x ↦ ⟨⟨p, hp⟩, (of R M a).symm.injective <| by simp [h]⟩
  /-
    🎉 no goals
  -/


theorem isTorsion_of_finiteDimensional [Field K] [Ring A] [Algebra K A]
    [AddCommGroup M] [Module A M] [Module K M] [IsScalarTower K A M] [FiniteDimensional K A] :
    IsTorsion K[X] (AEval K M a) :=
  isTorsion_of_aeval_eq_zero (minpoly.aeval K a) (minpoly.ne_zero_of_finite K a)


