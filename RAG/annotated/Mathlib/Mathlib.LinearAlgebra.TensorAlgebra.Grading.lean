/-- A version of `TensorAlgebra.ι` that maps directly into the graded structure. This is
primarily an auxiliary construction used to provide `TensorAlgebra.gradedAlgebra`. -/
nonrec def GradedAlgebra.ι : M →ₗ[R] ⨁ i : ℕ, ↥(LinearMap.range (ι R : M →ₗ[_] _) ^ i) :=
  DirectSum.lof R ℕ (fun i => ↥(LinearMap.range (ι R : M →ₗ[_] _) ^ i)) 1 ∘ₗ
                                    /-
                                      R : Type u_1
                                      M : Type u_2
                                      inst✝² : CommSemiring R
                                      inst✝¹ : AddCommMonoid M
                                      inst✝ : Module R M
                                      m : M
                                      ⊢ Membership.mem (HPow.hPow (LinearMap.range (TensorAlgebra.ι R)) 1) ((TensorA …
                                    -/
    (ι R).codRestrict _ fun m => by simpa only [pow_one] using LinearMap.mem_range_self _ m
                                    /-
                                      🎉 no goals
                                    -/


theorem GradedAlgebra.ι_apply (m : M) :
    GradedAlgebra.ι R M m =
      DirectSum.of (fun (i : ℕ) => ↥(LinearMap.range (TensorAlgebra.ι R : M →ₗ[_] _) ^ i)) 1
                                 /-
                                   R : Type u_1
                                   M : Type u_2
                                   inst✝² : CommSemiring R
                                   inst✝¹ : AddCommMonoid M
                                   inst✝ : Module R M
                                   m : M
                                   ⊢ Membership.mem (HPow.hPow (LinearMap.range (TensorAlgebra.ι R)) 1) ((TensorA …
                                 -/
        ⟨TensorAlgebra.ι R m, by simpa only [pow_one] using LinearMap.mem_range_self _ m⟩ :=
                                 /-
                                   🎉 no goals
                                 -/
  rfl


/-- The tensor algebra is graded by the powers of the submodule `(TensorAlgebra.ι R).range`. -/
instance gradedAlgebra :
    GradedAlgebra ((LinearMap.range (ι R : M →ₗ[R] TensorAlgebra R M) ^ ·) : ℕ → Submodule R _) :=
  GradedAlgebra.ofAlgHom _ (lift R <| GradedAlgebra.ι R M)
    (by
      /-
        R : Type u_1
        M : Type u_2
        inst✝² : CommSemiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        ⊢ Eq ((DirectSum.coeAlgHom fun x => HPow.hPow (LinearMap.range (TensorAlgebra. …
      -/
      ext m
      dsimp only [LinearMap.comp_apply, AlgHom.toLinearMap_apply, AlgHom.comp_apply,
        AlgHom.id_apply]
      /-
        case w.h
        R : Type u_1
        M : Type u_2
        inst✝² : CommSemiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        m : M
        ⊢ Eq ((DirectSum.coeAlgHom fun x => HPow.hPow (LinearMap.range (TensorAlgebra. …
      -/
      rw [lift_ι_apply, GradedAlgebra.ι_apply R M, DirectSum.coeAlgHom_of, Subtype.coe_mk])
      /-
        🎉 no goals
      -/
    fun i x => by
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      i : Nat
      x : Subtype fun x => Membership.mem (HPow.hPow (LinearMap.range (TensorAlgebra …
      ⊢ Eq (((TensorAlgebra.lift R) (TensorAlgebra.GradedAlgebra.ι R M)) ↑x) ((Direc …
    -/
    cases' x with x hx
    /-
      case mk
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      i : Nat
      x : TensorAlgebra R M
      hx : Membership.mem (HPow.hPow (LinearMap.range (TensorAlgebra.ι R)) i) x
      ⊢ Eq (((TensorAlgebra.lift R) (TensorAlgebra.GradedAlgebra.ι R M)) ↑⟨x, hx⟩) ( …
    -/
    dsimp only [Subtype.coe_mk, DirectSum.lof_eq_of]
    -- Porting note: use new `induction using` support that failed in Lean 3
    induction hx using Submodule.pow_induction_on_left' with
    | algebraMap r =>
      rw [AlgHom.commutes, DirectSum.algebraMap_apply]; rfl
    | add x y i hx hy ihx ihy =>
      -- Note: https://github.com/leanprover-community/mathlib4/pull/8386 had to specialize `map_add` to avoid a timeout
      -- (the extra typeclass search seems to have pushed this already slow proof over the edge)
      rw [map_add, ihx, ihy, ← AddMonoidHom.map_add]
      rfl
    | mem_mul m hm i x hx ih =>
      obtain ⟨_, rfl⟩ := hm
      rw [map_mul, ih, lift_ι_apply, GradedAlgebra.ι_apply R M, DirectSum.of_mul_of]
      exact DirectSum.of_eq_of_gradedMonoid_eq (Sigma.subtype_ext (add_comm _ _) rfl)


