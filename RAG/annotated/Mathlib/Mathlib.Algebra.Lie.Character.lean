/-- A character of a Lie algebra is a morphism to the scalars. -/
abbrev LieCharacter :=
  L →ₗ⁅R⁆ R


theorem lieCharacter_apply_lie (χ : LieCharacter R L) (x y : L) : χ ⁅x, y⁆ = 0 := by
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    χ : LieAlgebra.LieCharacter R L
    x y : L
    ⊢ Eq (χ (Bracket.bracket x y)) 0
  -/
  rw [LieHom.map_lie, LieRing.of_associative_ring_bracket, mul_comm, sub_self]
  /-
    🎉 no goals
  -/


@[simp]
theorem lieCharacter_apply_lie' (χ : LieCharacter R L) (x y : L) : ⁅χ x, χ y⁆ = 0 := by
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    χ : LieAlgebra.LieCharacter R L
    x y : L
    ⊢ Eq (Bracket.bracket (χ x) (χ y)) 0
  -/
  rw [LieRing.of_associative_ring_bracket, mul_comm, sub_self]
  /-
    🎉 no goals
  -/


theorem lieCharacter_apply_of_mem_derived (χ : LieCharacter R L) {x : L}
    (h : x ∈ derivedSeries R L 1) : χ x = 0 := by
  rw [derivedSeries_def, derivedSeriesOfIdeal_succ, derivedSeriesOfIdeal_zero, ←
    LieSubmodule.mem_toSubmodule, LieSubmodule.lieIdeal_oper_eq_linear_span] at h
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    χ : LieAlgebra.LieCharacter R L
    x : L
    h : Membership.mem (Submodule.span R (setOf fun m => Exists fun x => Exists fu …
    ⊢ Eq (χ x) 0
  -/
  refine Submodule.span_induction ?_ ?_ ?_ ?_ h
    /-
      case refine_1
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      χ : LieAlgebra.LieCharacter R L
      x : L
      h : Membership.mem (Submodule.span R (setOf fun m => Exists fun x => Exists fu …
      ⊢ ∀ (x : L), Membership.mem (setOf fun m => Exists fun x => Exists fun n => Eq …
    -/
  · rintro y ⟨⟨z, hz⟩, ⟨⟨w, hw⟩, rfl⟩⟩; apply lieCharacter_apply_lie
                                        /-
                                          🎉 no goals
                                        -/
    /-
      case refine_2
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      χ : LieAlgebra.LieCharacter R L
      x : L
      h : Membership.mem (Submodule.span R (setOf fun m => Exists fun x => Exists fu …
      ⊢ Eq (χ 0) 0
    -/
  · exact χ.map_zero
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      χ : LieAlgebra.LieCharacter R L
      x : L
      h : Membership.mem (Submodule.span R (setOf fun m => Exists fun x => Exists fu …
      ⊢ ∀ (x y : L), Membership.mem (Submodule.span R (setOf fun m => Exists fun x = …
    -/
  · intro y z _ _ hy hz; rw [LieHom.map_add, hy, hz, add_zero]
                         /-
                           🎉 no goals
                         -/
    /-
      case refine_4
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      χ : LieAlgebra.LieCharacter R L
      x : L
      h : Membership.mem (Submodule.span R (setOf fun m => Exists fun x => Exists fu …
      ⊢ ∀ (a : R) (x : L), Membership.mem (Submodule.span R (setOf fun m => Exists f …
    -/
  · intro t y _ hy; rw [LieHom.map_smul, hy, smul_zero]
                    /-
                      🎉 no goals
                    -/


/-- For an Abelian Lie algebra, characters are just linear forms. -/
@[simps! apply symm_apply]
def lieCharacterEquivLinearDual [IsLieAbelian L] : LieCharacter R L ≃ Module.Dual R L where
  toFun χ := (χ : L →ₗ[R] R)
  invFun ψ :=
    { ψ with
      map_lie' := fun {x y} => by
        rw [LieModule.IsTrivial.trivial, LieRing.of_associative_ring_bracket, mul_comm, sub_self,
          LinearMap.toFun_eq_coe, LinearMap.map_zero] }
                   /-
                     R : Type u
                     L : Type v
                     inst✝³ : CommRing R
                     inst✝² : LieRing L
                     inst✝¹ : LieAlgebra R L
                     inst✝ : IsLieAbelian L
                     χ : LieAlgebra.LieCharacter R L
                     ⊢ Eq ((fun ψ => { toLinearMap := ψ, map_lie' := ⋯ }) ((fun χ => ↑χ) χ)) χ
                   -/
  left_inv χ := by ext; rfl
                        /-
                          🎉 no goals
                        -/
                    /-
                      R : Type u
                      L : Type v
                      inst✝³ : CommRing R
                      inst✝² : LieRing L
                      inst✝¹ : LieAlgebra R L
                      inst✝ : IsLieAbelian L
                      ψ : Module.Dual R L
                      ⊢ Eq ((fun χ => ↑χ) ((fun ψ => { toLinearMap := ψ, map_lie' := ⋯ }) ψ)) ψ
                    -/
  right_inv ψ := by ext; rfl
                         /-
                           🎉 no goals
                         -/


