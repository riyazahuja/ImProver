theorem eventuallyConst_of_isNoetherian [IsNoetherian R M] (f : ℕ →o Submodule R M) :
    atTop.EventuallyConst f := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : IsNoetherian R M
    f : OrderHom Nat (Submodule R M)
    ⊢ Filter.EventuallyConst (⇑f) Filter.atTop
  -/
  simp_rw [eventuallyConst_atTop, eq_comm]
  /-
    R : Type u_1
    M : Type u_2
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : IsNoetherian R M
    f : OrderHom Nat (Submodule R M)
    ⊢ Exists fun i => ∀ (j : Nat), LE.le i j → Eq (f i) (f j)
  -/
  exact (monotone_stabilizes_iff_noetherian.mpr inferInstance) f
  /-
    🎉 no goals
  -/


