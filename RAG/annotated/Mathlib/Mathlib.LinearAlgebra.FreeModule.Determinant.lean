@[simp high]
theorem LinearMap.det_zero'' {R M : Type*} [CommRing R] [AddCommGroup M] [Module R M]
    [Module.Free R M] [Module.Finite R M] [Nontrivial M] : LinearMap.det (0 : M →ₗ[R] M) = 0 := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : Module.Free R M
    inst✝¹ : Module.Finite R M
    inst✝ : Nontrivial M
    ⊢ Eq (LinearMap.det 0) 0
  -/
  letI : Nonempty (Module.Free.ChooseBasisIndex R M) := (Module.Free.chooseBasis R M).index_nonempty
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : Module.Free R M
    inst✝¹ : Module.Finite R M
    inst✝ : Nontrivial M
    this : Nonempty (Module.Free.ChooseBasisIndex R M) := Basis.index_nonempty (Mo …
    ⊢ Eq (LinearMap.det 0) 0
  -/
  nontriviality R
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : Module.Free R M
    inst✝¹ : Module.Finite R M
    inst✝ : Nontrivial M
    this : Nonempty (Module.Free.ChooseBasisIndex R M) := Basis.index_nonempty (Mo …
    a✝ : Nontrivial R
    ⊢ Eq (LinearMap.det 0) 0
  -/
  exact LinearMap.det_zero' (Module.Free.chooseBasis R M)
  /-
    🎉 no goals
  -/

