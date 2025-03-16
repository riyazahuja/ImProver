/-- Two bilinear maps are equal when they are equal on all basis vectors. -/
theorem ext_basis {B B' : M →ₛₗ[ρ₁₂] N →ₛₗ[σ₁₂] P} (h : ∀ i j, B (b₁ i) (b₂ j) = B' (b₁ i) (b₂ j)) :
    B = B' :=
  b₁.ext fun i => b₂.ext fun j => h i j


/-- Write out `B x y` as a sum over `B (b i) (b j)` if `b` is a basis.

Version for semi-bilinear maps, see `sum_repr_mul_repr_mul` for the bilinear version. -/
theorem sum_repr_mul_repr_mulₛₗ {B : M →ₛₗ[ρ₁₂] N →ₛₗ[σ₁₂] P} (x y) :
    ((b₁.repr x).sum fun i xi => (b₂.repr y).sum fun j yj => ρ₁₂ xi • σ₁₂ yj • B (b₁ i) (b₂ j)) =
      B x y := by
  /-
    ι₁ : Type u_1
    ι₂ : Type u_2
    R : Type u_3
    R₂ : Type u_4
    S : Type u_5
    S₂ : Type u_6
    M : Type u_7
    N : Type u_8
    P : Type u_9
    inst✝¹¹ : Semiring R
    inst✝¹⁰ : Semiring S
    inst✝⁹ : Semiring R₂
    inst✝⁸ : Semiring S₂
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : AddCommMonoid N
    inst✝⁵ : AddCommMonoid P
    inst✝⁴ : Module R M
    inst✝³ : Module S N
    inst✝² : Module R₂ P
    inst✝¹ : Module S₂ P
    inst✝ : SMulCommClass S₂ R₂ P
    ρ₁₂ : RingHom R R₂
    σ₁₂ : RingHom S S₂
    b₁ : Basis ι₁ R M
    b₂ : Basis ι₂ S N
    B : LinearMap ρ₁₂ M (LinearMap σ₁₂ N P)
    x : M
    y : N
    ⊢ Eq ((b₁.repr x).sum fun i xi => (b₂.repr y).sum fun j yj => HSMul.hSMul (ρ₁₂ …
  -/
  conv_rhs => rw [← b₁.linearCombination_repr x, ← b₂.linearCombination_repr y]
  simp_rw [Finsupp.linearCombination_apply, Finsupp.sum, map_sum₂, map_sum, LinearMap.map_smulₛₗ₂,
    LinearMap.map_smulₛₗ]


/-- Write out `B x y` as a sum over `B (b i) (b j)` if `b` is a basis.

Version for bilinear maps, see `sum_repr_mul_repr_mulₛₗ` for the semi-bilinear version. -/
theorem sum_repr_mul_repr_mul {B : Mₗ →ₗ[Rₗ] Nₗ →ₗ[Rₗ] Pₗ} (x y) :
    ((b₁'.repr x).sum fun i xi => (b₂'.repr y).sum fun j yj => xi • yj • B (b₁' i) (b₂' j)) =
      B x y := by
  /-
    ι₁ : Type u_1
    ι₂ : Type u_2
    Rₗ : Type u_10
    Mₗ : Type u_11
    Nₗ : Type u_12
    Pₗ : Type u_13
    inst✝⁶ : CommSemiring Rₗ
    inst✝⁵ : AddCommMonoid Mₗ
    inst✝⁴ : AddCommMonoid Nₗ
    inst✝³ : AddCommMonoid Pₗ
    inst✝² : Module Rₗ Mₗ
    inst✝¹ : Module Rₗ Nₗ
    inst✝ : Module Rₗ Pₗ
    b₁' : Basis ι₁ Rₗ Mₗ
    b₂' : Basis ι₂ Rₗ Nₗ
    B : LinearMap (RingHom.id Rₗ) Mₗ (LinearMap (RingHom.id Rₗ) Nₗ Pₗ)
    x : Mₗ
    y : Nₗ
    ⊢ Eq ((b₁'.repr x).sum fun i xi => (b₂'.repr y).sum fun j yj => HSMul.hSMul xi …
  -/
  conv_rhs => rw [← b₁'.linearCombination_repr x, ← b₂'.linearCombination_repr y]
  simp_rw [Finsupp.linearCombination_apply, Finsupp.sum, map_sum₂, map_sum, LinearMap.map_smul₂,
    LinearMap.map_smul]


