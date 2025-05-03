/-- If `m₁` and `m₂` are both homogeneous,
and the quadratic spaces `Q₁` and `Q₂` map into
orthogonal subspaces of `Qₙ` (for instance, when `Qₙ = Q₁.prod Q₂`),
then the product of the embedding in `CliffordAlgebra Q` commutes up to a sign factor. -/
nonrec theorem map_mul_map_of_isOrtho_of_mem_evenOdd
    {i₁ i₂ : ZMod 2} (hm₁ : m₁ ∈ evenOdd Q₁ i₁) (hm₂ : m₂ ∈ evenOdd Q₂ i₂) :
    map f₁ m₁ * map f₂ m₂ = (-1 : ℤˣ) ^ (i₂ * i₁) • (map f₂ m₂ * map f₁ m₁) := by
  -- the strategy; for each variable, induct on powers of `ι`, then on the exponent of each
  -- power.
  induction hm₁ using Submodule.iSup_induction' with
  | zero => rw [map_zero, zero_mul, mul_zero, smul_zero]
  | add _ _ _ _ ihx ihy => rw [map_add, add_mul, mul_add, ihx, ihy, smul_add]
  | mem i₁' m₁' hm₁ =>
    obtain ⟨i₁n, rfl⟩ := i₁'
    dsimp only at *
    induction hm₁ using Submodule.pow_induction_on_left' with
    | algebraMap =>
      rw [AlgHom.commutes, Nat.cast_zero, mul_zero, uzpow_zero, one_smul, Algebra.commutes]
    | add _ _ _ _ _ ihx ihy =>
      rw [map_add, add_mul, mul_add, ihx, ihy, smul_add]
    | mem_mul m₁ hm₁ i x₁ _hx₁ ih₁ =>
      obtain ⟨v₁, rfl⟩ := hm₁
      -- this is the first interesting goal
      rw [map_mul, mul_assoc, ih₁, mul_smul_comm, map_apply_ι, Nat.cast_succ, mul_add_one,
        uzpow_add, mul_smul, ← mul_assoc, ← mul_assoc, ← smul_mul_assoc ((-1) ^ i₂)]
      clear ih₁
      congr 2
      induction hm₂ using Submodule.iSup_induction' with
      | zero => rw [map_zero, zero_mul, mul_zero, smul_zero]
      | add _ _ _ _ ihx ihy => rw [map_add, add_mul, mul_add, ihx, ihy, smul_add]
      | mem i₂' m₂' hm₂ =>
        clear m₂
        obtain ⟨i₂n, rfl⟩ := i₂'
        dsimp only at *
        induction hm₂ using Submodule.pow_induction_on_left' with
        | algebraMap =>
          rw [AlgHom.commutes, Nat.cast_zero, uzpow_zero, one_smul, Algebra.commutes]
        | add _ _ _ _ _ ihx ihy =>
          rw [map_add, add_mul, mul_add, ihx, ihy, smul_add]
        | mem_mul m₂ hm₂ i x₂ _hx₂ ih₂ =>
          obtain ⟨v₂, rfl⟩ := hm₂
          -- this is the second interesting goal
          rw [map_mul, map_apply_ι, Nat.cast_succ, ← mul_assoc,
            ι_mul_ι_comm_of_isOrtho (hf _ _), neg_mul, mul_assoc, ih₂, mul_smul_comm,
            ← mul_assoc, ← Units.neg_smul, uzpow_add, uzpow_one, mul_neg_one]


theorem commute_map_mul_map_of_isOrtho_of_mem_evenOdd_zero_left
    {i₂ : ZMod 2} (hm₁ : m₁ ∈ evenOdd Q₁ 0) (hm₂ : m₂ ∈ evenOdd Q₂ i₂) :
    Commute (map f₁ m₁) (map f₂ m₂) :=
                                                                         /-
                                                                           R : Type u_1
                                                                           M₁ : Type u_2
                                                                           M₂ : Type u_3
                                                                           N : Type u_4
                                                                           inst✝⁶ : CommRing R
                                                                           inst✝⁵ : AddCommGroup M₁
                                                                           inst✝⁴ : AddCommGroup M₂
                                                                           inst✝³ : AddCommGroup N
                                                                           inst✝² : Module R M₁
                                                                           inst✝¹ : Module R M₂
                                                                           inst✝ : Module R N
                                                                           Q₁ : QuadraticForm R M₁
                                                                           Q₂ : QuadraticForm R M₂
                                                                           Qₙ : QuadraticForm R N
                                                                           f₁ : QuadraticMap.Isometry Q₁ Qₙ
                                                                           f₂ : QuadraticMap.Isometry Q₂ Qₙ
                                                                           hf : ∀ (x : M₁) (y : M₂), QuadraticMap.IsOrtho Qₙ (f₁ x) (f₂ y)
                                                                           m₁ : CliffordAlgebra Q₁
                                                                           m₂ : CliffordAlgebra Q₂
                                                                           i₂ : ZMod 2
                                                                           hm₁ : Membership.mem (CliffordAlgebra.evenOdd Q₁ 0) m₁
                                                                           hm₂ : Membership.mem (CliffordAlgebra.evenOdd Q₂ i₂) m₂
                                                                           ⊢ Eq (HSMul.hSMul (HPow.hPow (-1) (HMul.hMul i₂ 0)) (HMul.hMul ((CliffordAlgeb …
                                                                         -/
  (map_mul_map_of_isOrtho_of_mem_evenOdd _ _ hf _ _ hm₁ hm₂).trans <| by simp
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


theorem commute_map_mul_map_of_isOrtho_of_mem_evenOdd_zero_right
    {i₁ : ZMod 2} (hm₁ : m₁ ∈ evenOdd Q₁ i₁) (hm₂ : m₂ ∈ evenOdd Q₂ 0) :
    Commute (map f₁ m₁) (map f₂ m₂) :=
                                                                         /-
                                                                           R : Type u_1
                                                                           M₁ : Type u_2
                                                                           M₂ : Type u_3
                                                                           N : Type u_4
                                                                           inst✝⁶ : CommRing R
                                                                           inst✝⁵ : AddCommGroup M₁
                                                                           inst✝⁴ : AddCommGroup M₂
                                                                           inst✝³ : AddCommGroup N
                                                                           inst✝² : Module R M₁
                                                                           inst✝¹ : Module R M₂
                                                                           inst✝ : Module R N
                                                                           Q₁ : QuadraticForm R M₁
                                                                           Q₂ : QuadraticForm R M₂
                                                                           Qₙ : QuadraticForm R N
                                                                           f₁ : QuadraticMap.Isometry Q₁ Qₙ
                                                                           f₂ : QuadraticMap.Isometry Q₂ Qₙ
                                                                           hf : ∀ (x : M₁) (y : M₂), QuadraticMap.IsOrtho Qₙ (f₁ x) (f₂ y)
                                                                           m₁ : CliffordAlgebra Q₁
                                                                           m₂ : CliffordAlgebra Q₂
                                                                           i₁ : ZMod 2
                                                                           hm₁ : Membership.mem (CliffordAlgebra.evenOdd Q₁ i₁) m₁
                                                                           hm₂ : Membership.mem (CliffordAlgebra.evenOdd Q₂ 0) m₂
                                                                           ⊢ Eq (HSMul.hSMul (HPow.hPow (-1) (HMul.hMul 0 i₁)) (HMul.hMul ((CliffordAlgeb …
                                                                         -/
  (map_mul_map_of_isOrtho_of_mem_evenOdd _ _ hf _ _ hm₁ hm₂).trans <| by simp
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


theorem map_mul_map_eq_neg_of_isOrtho_of_mem_evenOdd_one
    (hm₁ : m₁ ∈ evenOdd Q₁ 1) (hm₂ : m₂ ∈ evenOdd Q₂ 1) :
    map f₁ m₁ * map f₂ m₂ = - map f₂ m₂ * map f₁ m₁ := by
  /-
    R : Type u_1
    M₁ : Type u_2
    M₂ : Type u_3
    N : Type u_4
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M₁
    inst✝⁴ : AddCommGroup M₂
    inst✝³ : AddCommGroup N
    inst✝² : Module R M₁
    inst✝¹ : Module R M₂
    inst✝ : Module R N
    Q₁ : QuadraticForm R M₁
    Q₂ : QuadraticForm R M₂
    Qₙ : QuadraticForm R N
    f₁ : QuadraticMap.Isometry Q₁ Qₙ
    f₂ : QuadraticMap.Isometry Q₂ Qₙ
    hf : ∀ (x : M₁) (y : M₂), QuadraticMap.IsOrtho Qₙ (f₁ x) (f₂ y)
    m₁ : CliffordAlgebra Q₁
    m₂ : CliffordAlgebra Q₂
    hm₁ : Membership.mem (CliffordAlgebra.evenOdd Q₁ 1) m₁
    hm₂ : Membership.mem (CliffordAlgebra.evenOdd Q₂ 1) m₂
    ⊢ Eq (HMul.hMul ((CliffordAlgebra.map f₁) m₁) ((CliffordAlgebra.map f₂) m₂)) ( …
  -/
  simp [map_mul_map_of_isOrtho_of_mem_evenOdd _ _ hf _ _ hm₁ hm₂]
  /-
    🎉 no goals
  -/


/-- The forward direction of `CliffordAlgebra.prodEquiv`. -/
def ofProd : CliffordAlgebra (Q₁.prod Q₂) →ₐ[R] (evenOdd Q₁ ᵍ⊗[R] evenOdd Q₂) :=
  lift _ ⟨
    LinearMap.coprod
      ((GradedTensorProduct.includeLeft (evenOdd Q₁) (evenOdd Q₂)).toLinearMap
          ∘ₗ (evenOdd Q₁ 1).subtype ∘ₗ (ι Q₁).codRestrict _ (ι_mem_evenOdd_one Q₁))
      ((GradedTensorProduct.includeRight (evenOdd Q₁) (evenOdd Q₂)).toLinearMap
          ∘ₗ (evenOdd Q₂ 1).subtype ∘ₗ (ι Q₂).codRestrict _ (ι_mem_evenOdd_one Q₂)),
    fun m => by
      simp_rw [LinearMap.coprod_apply, LinearMap.coe_comp, Function.comp_apply,
        AlgHom.toLinearMap_apply, QuadraticMap.prod_apply, Submodule.coe_subtype,
        GradedTensorProduct.includeLeft_apply, GradedTensorProduct.includeRight_apply, map_add,
        add_mul, mul_add, GradedTensorProduct.algebraMap_def,
        GradedTensorProduct.tmul_one_mul_one_tmul, GradedTensorProduct.tmul_one_mul_coe_tmul,
        GradedTensorProduct.tmul_coe_mul_one_tmul, GradedTensorProduct.tmul_coe_mul_coe_tmul,
        LinearMap.codRestrict_apply, one_mul, uzpow_one, Units.neg_smul, one_smul, ι_sq_scalar,
        mul_one, ← GradedTensorProduct.algebraMap_def, ← GradedTensorProduct.algebraMap_def']
      /-
        R : Type u_1
        M₁ : Type u_2
        M₂ : Type u_3
        N : Type u_4
        inst✝⁶ : CommRing R
        inst✝⁵ : AddCommGroup M₁
        inst✝⁴ : AddCommGroup M₂
        inst✝³ : AddCommGroup N
        inst✝² : Module R M₁
        inst✝¹ : Module R M₂
        inst✝ : Module R N
        Q₁ : QuadraticForm R M₁
        Q₂ : QuadraticForm R M₂
        Qₙ : QuadraticForm R N
        m : Prod M₁ M₂
        ⊢ Eq (HAdd.hAdd (HAdd.hAdd ((algebraMap R (GradedTensorProduct R (CliffordAlge …
      -/
      /-
        🎉 no goals
      -/
      abel⟩
      /-
        🎉 no goals
      -/


@[simp]
lemma ofProd_ι_mk (m₁ : M₁) (m₂ : M₂) :
    ofProd Q₁ Q₂ (ι _ (m₁, m₂)) = ι Q₁ m₁ ᵍ⊗ₜ 1 + 1 ᵍ⊗ₜ ι Q₂ m₂ := by
  /-
    R : Type u_1
    M₁ : Type u_2
    M₂ : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M₁
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R M₁
    inst✝ : Module R M₂
    Q₁ : QuadraticForm R M₁
    Q₂ : QuadraticForm R M₂
    m₁ : M₁
    m₂ : M₂
    ⊢ Eq ((CliffordAlgebra.ofProd Q₁ Q₂) ((CliffordAlgebra.ι (QuadraticMap.prod Q₁ …
  -/
  rw [ofProd, lift_ι_apply]
  /-
    R : Type u_1
    M₁ : Type u_2
    M₂ : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M₁
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R M₁
    inst✝ : Module R M₂
    Q₁ : QuadraticForm R M₁
    Q₂ : QuadraticForm R M₂
    m₁ : M₁
    m₂ : M₂
    ⊢ Eq ((((GradedTensorProduct.includeLeft (CliffordAlgebra.evenOdd Q₁) (Cliffor …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The reverse direction of `CliffordAlgebra.prodEquiv`. -/
def toProd : evenOdd Q₁ ᵍ⊗[R] evenOdd Q₂ →ₐ[R] CliffordAlgebra (Q₁.prod Q₂) :=
  GradedTensorProduct.lift _ _
    (CliffordAlgebra.map <| .inl _ _)
    (CliffordAlgebra.map <| .inr _ _)
    fun _i₁ _i₂ x₁ x₂ => map_mul_map_of_isOrtho_of_mem_evenOdd _ _ (QuadraticMap.IsOrtho.inl_inr) _
      _ x₁.prop x₂.prop


@[simp]
lemma toProd_ι_tmul_one (m₁ : M₁) : toProd Q₁ Q₂ (ι _ m₁ ᵍ⊗ₜ 1) = ι _ (m₁, 0) := by
  rw [toProd, GradedTensorProduct.lift_tmul, map_one, mul_one, map_apply_ι,
    QuadraticMap.Isometry.inl_apply]


@[simp]
lemma toProd_one_tmul_ι (m₂ : M₂) : toProd Q₁ Q₂ (1 ᵍ⊗ₜ ι _ m₂) = ι _ (0, m₂) := by
  rw [toProd, GradedTensorProduct.lift_tmul, map_one, one_mul, map_apply_ι,
    QuadraticMap.Isometry.inr_apply]


lemma toProd_comp_ofProd : (toProd Q₁ Q₂).comp (ofProd Q₁ Q₂) = AlgHom.id _ _ := by
  /-
    R : Type u_1
    M₁ : Type u_2
    M₂ : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M₁
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R M₁
    inst✝ : Module R M₂
    Q₁ : QuadraticForm R M₁
    Q₂ : QuadraticForm R M₂
    ⊢ Eq ((CliffordAlgebra.toProd Q₁ Q₂).comp (CliffordAlgebra.ofProd Q₁ Q₂)) (Alg …
  -/
  ext m <;> dsimp
  · rw [ofProd_ι_mk, map_add, toProd_one_tmul_ι, toProd_ι_tmul_one, Prod.mk_zero_zero,
      LinearMap.map_zero, add_zero]
  · rw [ofProd_ι_mk, map_add, toProd_one_tmul_ι, toProd_ι_tmul_one, Prod.mk_zero_zero,
      LinearMap.map_zero, zero_add]


lemma ofProd_comp_toProd : (ofProd Q₁ Q₂).comp (toProd Q₁ Q₂) = AlgHom.id _ _ := by
  /-
    R : Type u_1
    M₁ : Type u_2
    M₂ : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M₁
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R M₁
    inst✝ : Module R M₂
    Q₁ : QuadraticForm R M₁
    Q₂ : QuadraticForm R M₂
    ⊢ Eq ((CliffordAlgebra.ofProd Q₁ Q₂).comp (CliffordAlgebra.toProd Q₁ Q₂)) (Alg …
  -/
                  /-
                    🎉 no goals
                  -/
  ext <;> (dsimp; simp)
                  /-
                    🎉 no goals
                  -/


/-- The Clifford algebra over an orthogonal direct sum of quadratic vector spaces is isomorphic
as an algebra to the graded tensor product of the Clifford algebras of each space.

This is `CliffordAlgebra.toProd` and `CliffordAlgebra.ofProd` as an equivalence. -/
@[simps!]
def prodEquiv : CliffordAlgebra (Q₁.prod Q₂) ≃ₐ[R] (evenOdd Q₁ ᵍ⊗[R] evenOdd Q₂) :=
  AlgEquiv.ofAlgHom (ofProd Q₁ Q₂) (toProd Q₁ Q₂) (ofProd_comp_toProd _ _) (toProd_comp_ofProd _ _)


