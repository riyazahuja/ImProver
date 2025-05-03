/-- A version of `ExteriorAlgebra.ι` that maps directly into the graded structure. This is
primarily an auxiliary construction used to provide `ExteriorAlgebra.gradedAlgebra`. -/
protected def GradedAlgebra.ι :
    M →ₗ[R] ⨁ i : ℕ, ⋀[R]^i M :=
  DirectSum.lof R ℕ (fun i => ⋀[R]^i M) 1 ∘ₗ
                                    /-
                                      R : Type u_1
                                      M : Type u_2
                                      inst✝² : CommRing R
                                      inst✝¹ : AddCommGroup M
                                      inst✝ : Module R M
                                      m : M
                                      ⊢ Membership.mem (ExteriorAlgebra.exteriorPower R 1 M) ((ExteriorAlgebra.ι R) m)
                                    -/
    (ι R).codRestrict _ fun m => by simpa only [pow_one] using LinearMap.mem_range_self _ m
                                    /-
                                      🎉 no goals
                                    -/


theorem GradedAlgebra.ι_apply (m : M) :
    GradedAlgebra.ι R M m =
      DirectSum.of (fun i : ℕ => ⋀[R]^i M) 1
                   /-
                     R : Type u_1
                     M : Type u_2
                     inst✝² : CommRing R
                     inst✝¹ : AddCommGroup M
                     inst✝ : Module R M
                     m : M
                     ⊢ Membership.mem (ExteriorAlgebra.exteriorPower R 1 M) ((ExteriorAlgebra.ι R) m)
                   -/
        ⟨ι R m, by simpa only [pow_one] using LinearMap.mem_range_self _ m⟩ :=
                   /-
                     🎉 no goals
                   -/
  rfl

-- Defining this instance manually, because Lean doesn't seem to be able to synthesize it.
-- Strangely, this problem only appears when we use the abbreviation or notation for the
-- exterior powers.

instance : SetLike.GradedMonoid fun i : ℕ ↦ ⋀[R]^i M :=
  Submodule.nat_power_gradedMonoid (LinearMap.range (ι R : M →ₗ[R] ExteriorAlgebra R M))

-- Porting note: Lean needs to be reminded of this instance otherwise it cannot
-- synthesize 0 in the next theorem

attribute [local instance 1100] MulZeroClass.toZero in
theorem GradedAlgebra.ι_sq_zero (m : M) : GradedAlgebra.ι R M m * GradedAlgebra.ι R M m = 0 := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    m : M
    ⊢ Eq (HMul.hMul ((ExteriorAlgebra.GradedAlgebra.ι R M) m) ((ExteriorAlgebra.Gr …
  -/
  rw [GradedAlgebra.ι_apply, DirectSum.of_mul_of]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    m : M
    ⊢ Eq ((DirectSum.of (fun i => Subtype fun x => Membership.mem (ExteriorAlgebra …
  -/
  exact DFinsupp.single_eq_zero.mpr (Subtype.ext <| ExteriorAlgebra.ι_sq_zero _)
  /-
    🎉 no goals
  -/


/-- `ExteriorAlgebra.GradedAlgebra.ι` lifted to exterior algebra. This is
primarily an auxiliary construction used to provide `ExteriorAlgebra.gradedAlgebra`. -/
def GradedAlgebra.liftι :
    ExteriorAlgebra R M →ₐ[R] ⨁ i : ℕ, ⋀[R]^i M :=
             /-
               R : Type u_1
               M : Type u_2
               inst✝² : CommRing R
               inst✝¹ : AddCommGroup M
               inst✝ : Module R M
               ⊢ LinearMap (RingHom.id R) M (DirectSum Nat fun i => Subtype fun x => Membersh …
             -/
  lift R ⟨by apply GradedAlgebra.ι R M, GradedAlgebra.ι_sq_zero R M⟩
             /-
               🎉 no goals
             -/


theorem GradedAlgebra.liftι_eq (i : ℕ) (x : ⋀[R]^i M) :
    GradedAlgebra.liftι R M x = DirectSum.of (fun i => ⋀[R]^i M) i x := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    i : Nat
    x : Subtype fun x => Membership.mem (ExteriorAlgebra.exteriorPower R i M) x
    ⊢ Eq ((ExteriorAlgebra.GradedAlgebra.liftι R M) ↑x) ((DirectSum.of (fun i => S …
  -/
  cases' x with x hx
  /-
    case mk
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    i : Nat
    x : ExteriorAlgebra R M
    hx : Membership.mem (ExteriorAlgebra.exteriorPower R i M) x
    ⊢ Eq ((ExteriorAlgebra.GradedAlgebra.liftι R M) ↑⟨x, hx⟩) ((DirectSum.of (fun  …
  -/
  dsimp only [Subtype.coe_mk, DirectSum.lof_eq_of]
  induction hx using Submodule.pow_induction_on_left' with
  | algebraMap => simp_rw [AlgHom.commutes, DirectSum.algebraMap_apply]; rfl
  | add _ _ _ _ _ ihx ihy => simp_rw [map_add, ihx, ihy, ← AddMonoidHom.map_add]; rfl
  | mem_mul _ hm _ _ _ ih =>
      obtain ⟨_, rfl⟩ := hm
      simp_rw [map_mul, ih, GradedAlgebra.liftι, lift_ι_apply, GradedAlgebra.ι_apply R M,
        DirectSum.of_mul_of]
      exact DirectSum.of_eq_of_gradedMonoid_eq (Sigma.subtype_ext (add_comm _ _) rfl)


/-- The exterior algebra is graded by the powers of the submodule `(ExteriorAlgebra.ι R).range`. -/
instance gradedAlgebra : GradedAlgebra (fun i : ℕ ↦ ⋀[R]^i M) :=
  GradedAlgebra.ofAlgHom _
    (-- while not necessary, the `by apply` makes this elaborate faster
       /-
         R : Type u_1
         M : Type u_2
         inst✝² : CommRing R
         inst✝¹ : AddCommGroup M
         inst✝ : Module R M
         ⊢ AlgHom R (ExteriorAlgebra R M) (DirectSum Nat fun i => Subtype fun x => Memb …
       -/
    by apply GradedAlgebra.liftι R M)
       /-
         🎉 no goals
       -/
    -- the proof from here onward is identical to the `TensorAlgebra` case
    (by
      /-
        R : Type u_1
        M : Type u_2
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        ⊢ Eq ((DirectSum.coeAlgHom fun i => ExteriorAlgebra.exteriorPower R i M).comp  …
      -/
      ext m
      dsimp only [LinearMap.comp_apply, AlgHom.toLinearMap_apply, AlgHom.comp_apply,
        AlgHom.id_apply, GradedAlgebra.liftι]
      /-
        case a.h
        R : Type u_1
        M : Type u_2
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        m : M
        ⊢ Eq ((DirectSum.coeAlgHom fun i => ExteriorAlgebra.exteriorPower R i M) (((Ex …
      -/
      rw [lift_ι_apply, GradedAlgebra.ι_apply R M, DirectSum.coeAlgHom_of, Subtype.coe_mk])
      /-
        🎉 no goals
      -/
        /-
          R : Type u_1
          M : Type u_2
          inst✝² : CommRing R
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          ⊢ ∀ (i : Nat) (x : Subtype fun x => Membership.mem (ExteriorAlgebra.exteriorPo …
        -/
    (by apply GradedAlgebra.liftι_eq R M)
        /-
          🎉 no goals
        -/


/-- The union of the images of the maps `ExteriorAlgebra.ιMulti R n` for `n` running through
all natural numbers spans the exterior algebra. -/
lemma ιMulti_span :
    Submodule.span R (Set.range fun x : Σ n, (Fin n → M) => ιMulti R x.1 x.2) = ⊤ := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ⊢ Eq (Submodule.span R (Set.range fun x => (ExteriorAlgebra.ιMulti R x.fst) x. …
  -/
  rw [Submodule.eq_top_iff']
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ⊢ ∀ (x : ExteriorAlgebra R M), Membership.mem (Submodule.span R (Set.range fun …
  -/
  intro x
  induction x using DirectSum.Decomposition.inductionOn fun i => ⋀[R]^i M with
  | h_zero => exact Submodule.zero_mem _
  | h_add _ _ hm hm' => exact Submodule.add_mem _ hm hm'
  | h_homogeneous hm =>
    let ⟨m, hm⟩ := hm
    apply Set.mem_of_mem_of_subset hm
    rw [← ιMulti_span_fixedDegree]
    refine Submodule.span_mono fun _ hx ↦ ?_
    obtain ⟨y, rfl⟩ := hx
    exact ⟨⟨_, y⟩, rfl⟩


