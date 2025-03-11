/-- The kernel of the multiplication map `S ⊗[R] S →ₐ[R] S`. -/
abbrev KaehlerDifferential.ideal : Ideal (S ⊗[R] S) :=
  RingHom.ker (TensorProduct.lmul' R : S ⊗[R] S →ₐ[R] S)


theorem KaehlerDifferential.one_smul_sub_smul_one_mem_ideal (a : S) :
                                                                            /-
                                                                              R : Type u
                                                                              S : Type v
                                                                              inst✝² : CommRing R
                                                                              inst✝¹ : CommRing S
                                                                              inst✝ : Algebra R S
                                                                              a : S
                                                                              ⊢ Membership.mem (KaehlerDifferential.ideal R S) (HSub.hSub (TensorProduct.tmu …
                                                                            -/
    (1 : S) ⊗ₜ[R] a - a ⊗ₜ[R] (1 : S) ∈ KaehlerDifferential.ideal R S := by simp [RingHom.mem_ker]
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


/-- For a `R`-derivation `S → M`, this is the map `S ⊗[R] S →ₗ[S] M` sending `s ⊗ₜ t ↦ s • D t`. -/
def Derivation.tensorProductTo (D : Derivation R S M) : S ⊗[R] S →ₗ[S] M :=
  TensorProduct.AlgebraTensorModule.lift ((LinearMap.lsmul S (S →ₗ[R] M)).flip D.toLinearMap)


theorem Derivation.tensorProductTo_tmul (D : Derivation R S M) (s t : S) :
    D.tensorProductTo (s ⊗ₜ t) = s • D t := rfl


theorem Derivation.tensorProductTo_mul (D : Derivation R S M) (x y : S ⊗[R] S) :
    D.tensorProductTo (x * y) =
      TensorProduct.lmul' (S := S) R x • D.tensorProductTo y +
        TensorProduct.lmul' (S := S) R y • D.tensorProductTo x := by
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    M : Type u_1
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module S M
    inst✝ : IsScalarTower R S M
    D : Derivation R S M
    x y : TensorProduct R S S
    ⊢ Eq (D.tensorProductTo (HMul.hMul x y)) (HAdd.hAdd (HSMul.hSMul ((Algebra.Ten …
  -/
  refine TensorProduct.induction_on x ?_ ?_ ?_
    /-
      case refine_1
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      M : Type u_1
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : Module S M
      inst✝ : IsScalarTower R S M
      D : Derivation R S M
      x y : TensorProduct R S S
      ⊢ Eq (D.tensorProductTo (HMul.hMul 0 y)) (HAdd.hAdd (HSMul.hSMul ((Algebra.Ten …
    -/
  · rw [zero_mul, map_zero, map_zero, zero_smul, smul_zero, add_zero]
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    M : Type u_1
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module S M
    inst✝ : IsScalarTower R S M
    D : Derivation R S M
    x y : TensorProduct R S S
    ⊢ ∀ (x y_1 : S), Eq (D.tensorProductTo (HMul.hMul (TensorProduct.tmul R x y_1) …
  -/
  swap
    /-
      case refine_3
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      M : Type u_1
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : Module S M
      inst✝ : IsScalarTower R S M
      D : Derivation R S M
      x y : TensorProduct R S S
      ⊢ ∀ (x y_1 : TensorProduct R S S), Eq (D.tensorProductTo (HMul.hMul x y)) (HAd …
    -/
  · intro x₁ y₁ h₁ h₂
    /-
      case refine_3
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      M : Type u_1
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : Module S M
      inst✝ : IsScalarTower R S M
      D : Derivation R S M
      x y x₁ y₁ : TensorProduct R S S
      h₁ : Eq (D.tensorProductTo (HMul.hMul x₁ y)) (HAdd.hAdd (HSMul.hSMul ((Algebra …
      h₂ : Eq (D.tensorProductTo (HMul.hMul y₁ y)) (HAdd.hAdd (HSMul.hSMul ((Algebra …
      ⊢ Eq (D.tensorProductTo (HMul.hMul (HAdd.hAdd x₁ y₁) y)) (HAdd.hAdd (HSMul.hSM …
    -/
    rw [add_mul, map_add, map_add, map_add, add_smul, smul_add, h₁, h₂, add_add_add_comm]
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    M : Type u_1
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module S M
    inst✝ : IsScalarTower R S M
    D : Derivation R S M
    x y : TensorProduct R S S
    ⊢ ∀ (x y_1 : S), Eq (D.tensorProductTo (HMul.hMul (TensorProduct.tmul R x y_1) …
  -/
  intro x₁ x₂
  /-
    case refine_2
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    M : Type u_1
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module S M
    inst✝ : IsScalarTower R S M
    D : Derivation R S M
    x y : TensorProduct R S S
    x₁ x₂ : S
    ⊢ Eq (D.tensorProductTo (HMul.hMul (TensorProduct.tmul R x₁ x₂) y)) (HAdd.hAdd …
  -/
  refine TensorProduct.induction_on y ?_ ?_ ?_
    /-
      case refine_2.refine_1
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      M : Type u_1
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : Module S M
      inst✝ : IsScalarTower R S M
      D : Derivation R S M
      x y : TensorProduct R S S
      x₁ x₂ : S
      ⊢ Eq (D.tensorProductTo (HMul.hMul (TensorProduct.tmul R x₁ x₂) 0)) (HAdd.hAdd …
    -/
  · rw [mul_zero, map_zero, map_zero, zero_smul, smul_zero, add_zero]
    /-
      🎉 no goals
    -/
  /-
    case refine_2.refine_2
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    M : Type u_1
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module S M
    inst✝ : IsScalarTower R S M
    D : Derivation R S M
    x y : TensorProduct R S S
    x₁ x₂ : S
    ⊢ ∀ (x y : S), Eq (D.tensorProductTo (HMul.hMul (TensorProduct.tmul R x₁ x₂) ( …
  -/
  swap
    /-
      case refine_2.refine_3
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      M : Type u_1
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : Module S M
      inst✝ : IsScalarTower R S M
      D : Derivation R S M
      x y : TensorProduct R S S
      x₁ x₂ : S
      ⊢ ∀ (x y : TensorProduct R S S), Eq (D.tensorProductTo (HMul.hMul (TensorProdu …
    -/
  · intro x₁ y₁ h₁ h₂
    /-
      case refine_2.refine_3
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      M : Type u_1
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : Module S M
      inst✝ : IsScalarTower R S M
      D : Derivation R S M
      x y : TensorProduct R S S
      x₁✝ x₂ : S
      x₁ y₁ : TensorProduct R S S
      h₁ : Eq (D.tensorProductTo (HMul.hMul (TensorProduct.tmul R x₁✝ x₂) x₁)) (HAdd …
      h₂ : Eq (D.tensorProductTo (HMul.hMul (TensorProduct.tmul R x₁✝ x₂) y₁)) (HAdd …
      ⊢ Eq (D.tensorProductTo (HMul.hMul (TensorProduct.tmul R x₁✝ x₂) (HAdd.hAdd x₁ …
    -/
    rw [mul_add, map_add, map_add, map_add, add_smul, smul_add, h₁, h₂, add_add_add_comm]
    /-
      🎉 no goals
    -/
  /-
    case refine_2.refine_2
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    M : Type u_1
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module S M
    inst✝ : IsScalarTower R S M
    D : Derivation R S M
    x y : TensorProduct R S S
    x₁ x₂ : S
    ⊢ ∀ (x y : S), Eq (D.tensorProductTo (HMul.hMul (TensorProduct.tmul R x₁ x₂) ( …
  -/
  intro x y
  simp only [TensorProduct.tmul_mul_tmul, Derivation.tensorProductTo,
    TensorProduct.AlgebraTensorModule.lift_apply, TensorProduct.lift.tmul',
    TensorProduct.lmul'_apply_tmul]
  /-
    case refine_2.refine_2
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    M : Type u_1
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module S M
    inst✝ : IsScalarTower R S M
    D : Derivation R S M
    x✝ y✝ : TensorProduct R S S
    x₁ x₂ x y : S
    ⊢ Eq ((TensorProduct.lift (↑R ((LinearMap.lsmul S (LinearMap (RingHom.id R) S  …
  -/
  dsimp
  /-
    case refine_2.refine_2
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    M : Type u_1
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module S M
    inst✝ : IsScalarTower R S M
    D : Derivation R S M
    x✝ y✝ : TensorProduct R S S
    x₁ x₂ x y : S
    ⊢ Eq (HSMul.hSMul (HMul.hMul x₁ x) (D (HMul.hMul x₂ y))) (HAdd.hAdd (HSMul.hSM …
  -/
  rw [D.leibniz]
  /-
    case refine_2.refine_2
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    M : Type u_1
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module S M
    inst✝ : IsScalarTower R S M
    D : Derivation R S M
    x✝ y✝ : TensorProduct R S S
    x₁ x₂ x y : S
    ⊢ Eq (HSMul.hSMul (HMul.hMul x₁ x) (HAdd.hAdd (HSMul.hSMul x₂ (D y)) (HSMul.hS …
  -/
  simp only [smul_smul, smul_add, mul_comm (x * y) x₁, mul_right_comm x₁ x₂, ← mul_assoc]
  /-
    🎉 no goals
  -/


/-- The kernel of `S ⊗[R] S →ₐ[R] S` is generated by `1 ⊗ s - s ⊗ 1` as a `S`-module. -/
theorem KaehlerDifferential.submodule_span_range_eq_ideal :
    Submodule.span S (Set.range fun s : S => (1 : S) ⊗ₜ[R] s - s ⊗ₜ[R] (1 : S)) =
      (KaehlerDifferential.ideal R S).restrictScalars S := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    ⊢ Eq (Submodule.span S (Set.range fun s => HSub.hSub (TensorProduct.tmul R 1 s …
  -/
  apply le_antisymm
    /-
      case a
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      ⊢ LE.le (Submodule.span S (Set.range fun s => HSub.hSub (TensorProduct.tmul R  …
    -/
  · rw [Submodule.span_le]
    /-
      case a
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      ⊢ HasSubset.Subset (Set.range fun s => HSub.hSub (TensorProduct.tmul R 1 s) (T …
    -/
    rintro _ ⟨s, rfl⟩
    /-
      case a.intro
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      s : S
      ⊢ Membership.mem (↑(Submodule.restrictScalars S (KaehlerDifferential.ideal R S …
    -/
    exact KaehlerDifferential.one_smul_sub_smul_one_mem_ideal _ _
    /-
      🎉 no goals
    -/
    /-
      case a
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      ⊢ LE.le (Submodule.restrictScalars S (KaehlerDifferential.ideal R S)) (Submodu …
    -/
  · rintro x (hx : _ = _)
    have : x - TensorProduct.lmul' (S := S) R x ⊗ₜ[R] (1 : S) = x := by
      rw [hx, TensorProduct.zero_tmul, sub_zero]
    /-
      case a
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      x : TensorProduct R S S
      hx : Eq ((Algebra.TensorProduct.lmul' R) x) 0
      this : Eq (HSub.hSub x (TensorProduct.tmul R ((Algebra.TensorProduct.lmul' R)  …
      ⊢ Membership.mem (Submodule.span S (Set.range fun s => HSub.hSub (TensorProduc …
    -/
    rw [← this]
    /-
      case a
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      x : TensorProduct R S S
      hx : Eq ((Algebra.TensorProduct.lmul' R) x) 0
      this : Eq (HSub.hSub x (TensorProduct.tmul R ((Algebra.TensorProduct.lmul' R)  …
      ⊢ Membership.mem (Submodule.span S (Set.range fun s => HSub.hSub (TensorProduc …
    -/
    clear this hx
    /-
      case a
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      x : TensorProduct R S S
      ⊢ Membership.mem (Submodule.span S (Set.range fun s => HSub.hSub (TensorProduc …
    -/
    refine TensorProduct.induction_on x ?_ ?_ ?_
      /-
        case a.refine_1
        R : Type u
        S : Type v
        inst✝² : CommRing R
        inst✝¹ : CommRing S
        inst✝ : Algebra R S
        x : TensorProduct R S S
        ⊢ Membership.mem (Submodule.span S (Set.range fun s => HSub.hSub (TensorProduc …
      -/
    · rw [map_zero, TensorProduct.zero_tmul, sub_zero]; exact zero_mem _
                                                        /-
                                                          🎉 no goals
                                                        -/
      /-
        case a.refine_2
        R : Type u
        S : Type v
        inst✝² : CommRing R
        inst✝¹ : CommRing S
        inst✝ : Algebra R S
        x : TensorProduct R S S
        ⊢ ∀ (x y : S), Membership.mem (Submodule.span S (Set.range fun s => HSub.hSub  …
      -/
    · intro x y
      have : x ⊗ₜ[R] y - (x * y) ⊗ₜ[R] (1 : S) = x • ((1 : S) ⊗ₜ y - y ⊗ₜ (1 : S)) := by
        simp_rw [smul_sub, TensorProduct.smul_tmul', smul_eq_mul, mul_one]
      /-
        case a.refine_2
        R : Type u
        S : Type v
        inst✝² : CommRing R
        inst✝¹ : CommRing S
        inst✝ : Algebra R S
        x✝ : TensorProduct R S S
        x y : S
        this : Eq (HSub.hSub (TensorProduct.tmul R x y) (TensorProduct.tmul R (HMul.hM …
        ⊢ Membership.mem (Submodule.span S (Set.range fun s => HSub.hSub (TensorProduc …
      -/
      rw [TensorProduct.lmul'_apply_tmul, this]
      /-
        case a.refine_2
        R : Type u
        S : Type v
        inst✝² : CommRing R
        inst✝¹ : CommRing S
        inst✝ : Algebra R S
        x✝ : TensorProduct R S S
        x y : S
        this : Eq (HSub.hSub (TensorProduct.tmul R x y) (TensorProduct.tmul R (HMul.hM …
        ⊢ Membership.mem (Submodule.span S (Set.range fun s => HSub.hSub (TensorProduc …
      -/
      refine Submodule.smul_mem _ x ?_
      /-
        case a.refine_2
        R : Type u
        S : Type v
        inst✝² : CommRing R
        inst✝¹ : CommRing S
        inst✝ : Algebra R S
        x✝ : TensorProduct R S S
        x y : S
        this : Eq (HSub.hSub (TensorProduct.tmul R x y) (TensorProduct.tmul R (HMul.hM …
        ⊢ Membership.mem (Submodule.span S (Set.range fun s => HSub.hSub (TensorProduc …
      -/
      apply Submodule.subset_span
      /-
        case a.refine_2.a
        R : Type u
        S : Type v
        inst✝² : CommRing R
        inst✝¹ : CommRing S
        inst✝ : Algebra R S
        x✝ : TensorProduct R S S
        x y : S
        this : Eq (HSub.hSub (TensorProduct.tmul R x y) (TensorProduct.tmul R (HMul.hM …
        ⊢ Membership.mem (Set.range fun s => HSub.hSub (TensorProduct.tmul R 1 s) (Ten …
      -/
      exact Set.mem_range_self y
      /-
        🎉 no goals
      -/
      /-
        case a.refine_3
        R : Type u
        S : Type v
        inst✝² : CommRing R
        inst✝¹ : CommRing S
        inst✝ : Algebra R S
        x : TensorProduct R S S
        ⊢ ∀ (x y : TensorProduct R S S), Membership.mem (Submodule.span S (Set.range f …
      -/
    · intro x y hx hy
      /-
        case a.refine_3
        R : Type u
        S : Type v
        inst✝² : CommRing R
        inst✝¹ : CommRing S
        inst✝ : Algebra R S
        x✝ x y : TensorProduct R S S
        hx : Membership.mem (Submodule.span S (Set.range fun s => HSub.hSub (TensorPro …
        hy : Membership.mem (Submodule.span S (Set.range fun s => HSub.hSub (TensorPro …
        ⊢ Membership.mem (Submodule.span S (Set.range fun s => HSub.hSub (TensorProduc …
      -/
      rw [map_add, TensorProduct.add_tmul, ← sub_add_sub_comm]
      /-
        case a.refine_3
        R : Type u
        S : Type v
        inst✝² : CommRing R
        inst✝¹ : CommRing S
        inst✝ : Algebra R S
        x✝ x y : TensorProduct R S S
        hx : Membership.mem (Submodule.span S (Set.range fun s => HSub.hSub (TensorPro …
        hy : Membership.mem (Submodule.span S (Set.range fun s => HSub.hSub (TensorPro …
        ⊢ Membership.mem (Submodule.span S (Set.range fun s => HSub.hSub (TensorProduc …
      -/
      exact add_mem hx hy
      /-
        🎉 no goals
      -/


theorem KaehlerDifferential.span_range_eq_ideal :
    Ideal.span (Set.range fun s : S => (1 : S) ⊗ₜ[R] s - s ⊗ₜ[R] (1 : S)) =
      KaehlerDifferential.ideal R S := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    ⊢ Eq (Ideal.span (Set.range fun s => HSub.hSub (TensorProduct.tmul R 1 s) (Ten …
  -/
  apply le_antisymm
    /-
      case a
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      ⊢ LE.le (Ideal.span (Set.range fun s => HSub.hSub (TensorProduct.tmul R 1 s) ( …
    -/
  · rw [Ideal.span_le]
    /-
      case a
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      ⊢ HasSubset.Subset (Set.range fun s => HSub.hSub (TensorProduct.tmul R 1 s) (T …
    -/
    rintro _ ⟨s, rfl⟩
    /-
      case a.intro
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      s : S
      ⊢ Membership.mem (↑(KaehlerDifferential.ideal R S)) ((fun s => HSub.hSub (Tens …
    -/
    exact KaehlerDifferential.one_smul_sub_smul_one_mem_ideal _ _
    /-
      🎉 no goals
    -/
    /-
      case a
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      ⊢ LE.le (KaehlerDifferential.ideal R S) (Ideal.span (Set.range fun s => HSub.h …
    -/
  · change (KaehlerDifferential.ideal R S).restrictScalars S ≤ (Ideal.span _).restrictScalars S
    /-
      case a
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      ⊢ LE.le (Submodule.restrictScalars S (KaehlerDifferential.ideal R S)) (Submodu …
    -/
    rw [← KaehlerDifferential.submodule_span_range_eq_ideal, Ideal.span]
    /-
      case a
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      ⊢ LE.le (Submodule.span S (Set.range fun s => HSub.hSub (TensorProduct.tmul R  …
    -/
    conv_rhs => rw [← Submodule.span_span_of_tower S]
    /-
      case a
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      ⊢ LE.le (Submodule.span S (Set.range fun s => HSub.hSub (TensorProduct.tmul R  …
    -/
    exact Submodule.subset_span
    /-
      🎉 no goals
    -/


/-- The module of Kähler differentials (Kahler differentials, Kaehler differentials).
This is implemented as `I / I ^ 2` with `I` the kernel of the multiplication map `S ⊗[R] S →ₐ[R] S`.
To view elements as a linear combination of the form `s • D s'`, use
`KaehlerDifferential.tensorProductTo_surjective` and `Derivation.tensorProductTo_tmul`.

We also provide the notation `Ω[S⁄R]` for `KaehlerDifferential R S`.
Note that the slash is `\textfractionsolidus`.
-/
def KaehlerDifferential : Type v :=
  (KaehlerDifferential.ideal R S).Cotangent


instance : AddCommGroup (KaehlerDifferential R S) := inferInstanceAs <|
  AddCommGroup (KaehlerDifferential.ideal R S).Cotangent


instance KaehlerDifferential.module : Module (S ⊗[R] S) (KaehlerDifferential R S) :=
  Ideal.Cotangent.moduleOfTower _


@[inherit_doc KaehlerDifferential]
notation:100 "Ω[" S "⁄" R "]" => KaehlerDifferential R S


instance : Nonempty (Ω[S⁄R]) := ⟨0⟩


instance KaehlerDifferential.module' {R' : Type*} [CommRing R'] [Algebra R' S]
    [SMulCommClass R R' S] :
    Module R' (Ω[S⁄R]) :=
  Submodule.Quotient.module' _


instance : IsScalarTower S (S ⊗[R] S) (Ω[S⁄R]) :=
  Ideal.Cotangent.isScalarTower _


instance KaehlerDifferential.isScalarTower_of_tower {R₁ R₂ : Type*} [CommRing R₁] [CommRing R₂]
    [Algebra R₁ S] [Algebra R₂ S] [SMul R₁ R₂]
    [SMulCommClass R R₁ S] [SMulCommClass R R₂ S] [IsScalarTower R₁ R₂ S] :
    IsScalarTower R₁ R₂ (Ω[S⁄R]) :=
  Submodule.Quotient.isScalarTower _ _


instance KaehlerDifferential.isScalarTower' : IsScalarTower R (S ⊗[R] S) (Ω[S⁄R]) :=
  Submodule.Quotient.isScalarTower _ _


/-- The quotient map `I → Ω[S⁄R]` with `I` being the kernel of `S ⊗[R] S → S`. -/
def KaehlerDifferential.fromIdeal : KaehlerDifferential.ideal R S →ₗ[S ⊗[R] S] Ω[S⁄R] :=
  (KaehlerDifferential.ideal R S).toCotangent


/-- (Implementation) The underlying linear map of the derivation into `Ω[S⁄R]`. -/
def KaehlerDifferential.DLinearMap : S →ₗ[R] Ω[S⁄R] :=
  ((KaehlerDifferential.fromIdeal R S).restrictScalars R).comp
    ((TensorProduct.includeRight.toLinearMap - TensorProduct.includeLeft.toLinearMap :
            S →ₗ[R] S ⊗[R] S).codRestrict
        ((KaehlerDifferential.ideal R S).restrictScalars R)
        (KaehlerDifferential.one_smul_sub_smul_one_mem_ideal R) :
      _ →ₗ[R] _)


theorem KaehlerDifferential.DLinearMap_apply (s : S) :
    KaehlerDifferential.DLinearMap R S s =
      (KaehlerDifferential.ideal R S).toCotangent
        ⟨1 ⊗ₜ s - s ⊗ₜ 1, KaehlerDifferential.one_smul_sub_smul_one_mem_ideal R s⟩ := rfl


/-- The universal derivation into `Ω[S⁄R]`. -/
def KaehlerDifferential.D : Derivation R S (Ω[S⁄R]) :=
  { toLinearMap := KaehlerDifferential.DLinearMap R S
    map_one_eq_zero' := by
      /-
        R : Type u
        S : Type v
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        M : Type u_1
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : Module S M
        inst✝ : IsScalarTower R S M
        ⊢ Eq ((KaehlerDifferential.DLinearMap R S) 1) 0
      -/
      dsimp [KaehlerDifferential.DLinearMap_apply, Ideal.toCotangent_apply]
      /-
        R : Type u
        S : Type v
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        M : Type u_1
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : Module S M
        inst✝ : IsScalarTower R S M
        ⊢ Eq (Submodule.Quotient.mk ⟨HSub.hSub (TensorProduct.tmul R 1 1) (TensorProdu …
      -/
      congr
      /-
        case e_a.e_val
        R : Type u
        S : Type v
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        M : Type u_1
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : Module S M
        inst✝ : IsScalarTower R S M
        ⊢ Eq (HSub.hSub (TensorProduct.tmul R 1 1) (TensorProduct.tmul R 1 1)) 0
      -/
      rw [sub_self]
      /-
        🎉 no goals
      -/
    leibniz' := fun a b => by
      /-
        R : Type u
        S : Type v
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        M : Type u_1
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : Module S M
        inst✝ : IsScalarTower R S M
        a b : S
        ⊢ Eq ((KaehlerDifferential.DLinearMap R S) (HMul.hMul a b)) (HAdd.hAdd (HSMul. …
      -/
      have : LinearMap.CompatibleSMul { x // x ∈ ideal R S } (Ω[S⁄R]) S (S ⊗[R] S) := inferInstance
      /-
        R : Type u
        S : Type v
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        M : Type u_1
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : Module S M
        inst✝ : IsScalarTower R S M
        a b : S
        this : LinearMap.CompatibleSMul (Subtype fun x => Membership.mem (KaehlerDiffe …
        ⊢ Eq ((KaehlerDifferential.DLinearMap R S) (HMul.hMul a b)) (HAdd.hAdd (HSMul. …
      -/
      dsimp [KaehlerDifferential.DLinearMap_apply]
      -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
      erw [← LinearMap.map_smul_of_tower (M₂ := Ω[S⁄R]),
        ← LinearMap.map_smul_of_tower (M₂ := Ω[S⁄R]), ← map_add, Ideal.toCotangent_eq, pow_two]
      convert Submodule.mul_mem_mul (KaehlerDifferential.one_smul_sub_smul_one_mem_ideal R a : _)
        (KaehlerDifferential.one_smul_sub_smul_one_mem_ideal R b : _) using 1
      simp only [AddSubgroupClass.coe_sub, Submodule.coe_add, Submodule.coe_mk,
        TensorProduct.tmul_mul_tmul, mul_sub, sub_mul, mul_comm b, Submodule.coe_smul_of_tower,
        smul_sub, TensorProduct.smul_tmul', smul_eq_mul, mul_one]
      /-
        case h.e'_5
        R : Type u
        S : Type v
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        M : Type u_1
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : Module S M
        inst✝ : IsScalarTower R S M
        a b : S
        this : LinearMap.CompatibleSMul (Subtype fun x => Membership.mem (KaehlerDiffe …
        ⊢ Eq (HSub.hSub (HSub.hSub (TensorProduct.tmul R 1 (HMul.hMul a b)) (TensorPro …
      -/
      ring_nf }
      /-
        🎉 no goals
      -/


theorem KaehlerDifferential.D_apply (s : S) :
    KaehlerDifferential.D R S s =
      (KaehlerDifferential.ideal R S).toCotangent
        ⟨1 ⊗ₜ s - s ⊗ₜ 1, KaehlerDifferential.one_smul_sub_smul_one_mem_ideal R s⟩ := rfl


theorem KaehlerDifferential.span_range_derivation :
    Submodule.span S (Set.range <| KaehlerDifferential.D R S) = ⊤ := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    ⊢ Eq (Submodule.span S (Set.range ⇑(KaehlerDifferential.D R S))) Top.top
  -/
  rw [_root_.eq_top_iff]
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    ⊢ LE.le Top.top (Submodule.span S (Set.range ⇑(KaehlerDifferential.D R S)))
  -/
  rintro x -
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    x : KaehlerDifferential R S
    ⊢ Membership.mem (Submodule.span S (Set.range ⇑(KaehlerDifferential.D R S))) x
  -/
  obtain ⟨⟨x, hx⟩, rfl⟩ := Ideal.toCotangent_surjective _ x
  /-
    case intro.mk
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    x : TensorProduct R S S
    hx : Membership.mem (KaehlerDifferential.ideal R S) x
    ⊢ Membership.mem (Submodule.span S (Set.range ⇑(KaehlerDifferential.D R S))) ( …
  -/
  have : x ∈ (KaehlerDifferential.ideal R S).restrictScalars S := hx
  /-
    case intro.mk
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    x : TensorProduct R S S
    hx : Membership.mem (KaehlerDifferential.ideal R S) x
    this : Membership.mem (Submodule.restrictScalars S (KaehlerDifferential.ideal  …
    ⊢ Membership.mem (Submodule.span S (Set.range ⇑(KaehlerDifferential.D R S))) ( …
  -/
  rw [← KaehlerDifferential.submodule_span_range_eq_ideal] at this
  suffices ∃ hx, (KaehlerDifferential.ideal R S).toCotangent ⟨x, hx⟩ ∈
      Submodule.span S (Set.range <| KaehlerDifferential.D R S) by
    exact this.choose_spec
  /-
    case intro.mk
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    x : TensorProduct R S S
    hx : Membership.mem (KaehlerDifferential.ideal R S) x
    this : Membership.mem (Submodule.span S (Set.range fun s => HSub.hSub (TensorP …
    ⊢ Exists fun hx => Membership.mem (Submodule.span S (Set.range ⇑(KaehlerDiffer …
  -/
  refine Submodule.span_induction ?_ ?_ ?_ ?_ this
    /-
      case intro.mk.refine_1
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      x : TensorProduct R S S
      hx : Membership.mem (KaehlerDifferential.ideal R S) x
      this : Membership.mem (Submodule.span S (Set.range fun s => HSub.hSub (TensorP …
      ⊢ ∀ (x : TensorProduct R S S), Membership.mem (Set.range fun s => HSub.hSub (T …
    -/
  · rintro _ ⟨x, rfl⟩
    /-
      case intro.mk.refine_1.intro
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      x✝ : TensorProduct R S S
      hx : Membership.mem (KaehlerDifferential.ideal R S) x✝
      this : Membership.mem (Submodule.span S (Set.range fun s => HSub.hSub (TensorP …
      x : S
      ⊢ Exists fun hx => Membership.mem (Submodule.span S (Set.range ⇑(KaehlerDiffer …
    -/
    refine ⟨KaehlerDifferential.one_smul_sub_smul_one_mem_ideal R x, ?_⟩
    /-
      case intro.mk.refine_1.intro
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      x✝ : TensorProduct R S S
      hx : Membership.mem (KaehlerDifferential.ideal R S) x✝
      this : Membership.mem (Submodule.span S (Set.range fun s => HSub.hSub (TensorP …
      x : S
      ⊢ Membership.mem (Submodule.span S (Set.range ⇑(KaehlerDifferential.D R S))) ( …
    -/
    apply Submodule.subset_span
    /-
      case intro.mk.refine_1.intro.a
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      x✝ : TensorProduct R S S
      hx : Membership.mem (KaehlerDifferential.ideal R S) x✝
      this : Membership.mem (Submodule.span S (Set.range fun s => HSub.hSub (TensorP …
      x : S
      ⊢ Membership.mem (Set.range ⇑(KaehlerDifferential.D R S)) ((KaehlerDifferentia …
    -/
    exact ⟨x, KaehlerDifferential.DLinearMap_apply R S x⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.mk.refine_2
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      x : TensorProduct R S S
      hx : Membership.mem (KaehlerDifferential.ideal R S) x
      this : Membership.mem (Submodule.span S (Set.range fun s => HSub.hSub (TensorP …
      ⊢ Exists fun hx => Membership.mem (Submodule.span S (Set.range ⇑(KaehlerDiffer …
    -/
  · exact ⟨zero_mem _, Submodule.zero_mem _⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.mk.refine_3
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      x : TensorProduct R S S
      hx : Membership.mem (KaehlerDifferential.ideal R S) x
      this : Membership.mem (Submodule.span S (Set.range fun s => HSub.hSub (TensorP …
      ⊢ ∀ (x y : TensorProduct R S S), Membership.mem (Submodule.span S (Set.range f …
    -/
  · rintro x y - - ⟨hx₁, hx₂⟩ ⟨hy₁, hy₂⟩; exact ⟨add_mem hx₁ hy₁, Submodule.add_mem _ hx₂ hy₂⟩
                                          /-
                                            🎉 no goals
                                          -/
    /-
      case intro.mk.refine_4
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      x : TensorProduct R S S
      hx : Membership.mem (KaehlerDifferential.ideal R S) x
      this : Membership.mem (Submodule.span S (Set.range fun s => HSub.hSub (TensorP …
      ⊢ ∀ (a : S) (x : TensorProduct R S S), Membership.mem (Submodule.span S (Set.r …
    -/
  · rintro r x - ⟨hx₁, hx₂⟩
    exact ⟨((KaehlerDifferential.ideal R S).restrictScalars S).smul_mem r hx₁,
      Submodule.smul_mem _ r hx₂⟩


/-- `Ω[S⁄R]` is trivial if `R → S` is surjective.
Also see `Algebra.FormallyUnramified.iff_subsingleton_kaehlerDifferential`. -/
lemma KaehlerDifferential.subsingleton_of_surjective (h : Function.Surjective (algebraMap R S)) :
    Subsingleton (Ω[S⁄R]) := by
  suffices (⊤ : Submodule S (Ω[S⁄R])) ≤ ⊥ from
    (subsingleton_iff_forall_eq 0).mpr fun y ↦ this trivial
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    h : Function.Surjective ⇑(algebraMap R S)
    ⊢ LE.le Top.top Bot.bot
  -/
  rw [← KaehlerDifferential.span_range_derivation, Submodule.span_le]
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    h : Function.Surjective ⇑(algebraMap R S)
    ⊢ HasSubset.Subset (Set.range ⇑(KaehlerDifferential.D R S)) ↑Bot.bot
  -/
  rintro _ ⟨x, rfl⟩; obtain ⟨x, rfl⟩ := h x; simp
                                             /-
                                               🎉 no goals
                                             -/


/-- The linear map from `Ω[S⁄R]`, associated with a derivation. -/
def Derivation.liftKaehlerDifferential (D : Derivation R S M) : Ω[S⁄R] →ₗ[S] M := by
  refine LinearMap.comp ((((KaehlerDifferential.ideal R S) •
    (⊤ : Submodule (S ⊗[R] S) (KaehlerDifferential.ideal R S))).restrictScalars S).liftQ ?_ ?_)
    (Submodule.Quotient.restrictScalarsEquiv S _).symm.toLinearMap
    /-
      case refine_1
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      M : Type u_1
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : Module S M
      inst✝ : IsScalarTower R S M
      D : Derivation R S M
      ⊢ LinearMap (RingHom.id S) (Subtype fun x => Membership.mem (KaehlerDifferenti …
    -/
  · exact D.tensorProductTo.comp ((KaehlerDifferential.ideal R S).subtype.restrictScalars S)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      M : Type u_1
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : Module S M
      inst✝ : IsScalarTower R S M
      D : Derivation R S M
      ⊢ LE.le (Submodule.restrictScalars S (HSMul.hSMul (KaehlerDifferential.ideal R …
    -/
  · intro x hx
    /-
      case refine_2
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      M : Type u_1
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : Module S M
      inst✝ : IsScalarTower R S M
      D : Derivation R S M
      x : Subtype fun x => Membership.mem (KaehlerDifferential.ideal R S) x
      hx : Membership.mem (Submodule.restrictScalars S (HSMul.hSMul (KaehlerDifferen …
      ⊢ Membership.mem (LinearMap.ker (D.tensorProductTo.comp (↑S (Submodule.subtype …
    -/
    rw [LinearMap.mem_ker]
    /-
      case refine_2
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      M : Type u_1
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : Module S M
      inst✝ : IsScalarTower R S M
      D : Derivation R S M
      x : Subtype fun x => Membership.mem (KaehlerDifferential.ideal R S) x
      hx : Membership.mem (Submodule.restrictScalars S (HSMul.hSMul (KaehlerDifferen …
      ⊢ Eq ((D.tensorProductTo.comp (↑S (Submodule.subtype (KaehlerDifferential.idea …
    -/
    refine Submodule.smul_induction_on ((Submodule.restrictScalars_mem _ _ _).mp hx) ?_ ?_
      /-
        case refine_2.refine_1
        R : Type u
        S : Type v
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        M : Type u_1
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : Module S M
        inst✝ : IsScalarTower R S M
        D : Derivation R S M
        x : Subtype fun x => Membership.mem (KaehlerDifferential.ideal R S) x
        hx : Membership.mem (Submodule.restrictScalars S (HSMul.hSMul (KaehlerDifferen …
        ⊢ ∀ (r : TensorProduct R S S), Membership.mem (KaehlerDifferential.ideal R S)  …
      -/
    · rintro x hx y -
      /-
        case refine_2.refine_1
        R : Type u
        S : Type v
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        M : Type u_1
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : Module S M
        inst✝ : IsScalarTower R S M
        D : Derivation R S M
        x✝ : Subtype fun x => Membership.mem (KaehlerDifferential.ideal R S) x
        hx✝ : Membership.mem (Submodule.restrictScalars S (HSMul.hSMul (KaehlerDiffere …
        x : TensorProduct R S S
        hx : Membership.mem (KaehlerDifferential.ideal R S) x
        y : Subtype fun x => Membership.mem (KaehlerDifferential.ideal R S) x
        ⊢ Eq ((D.tensorProductTo.comp (↑S (Submodule.subtype (KaehlerDifferential.idea …
      -/
      rw [RingHom.mem_ker] at hx
      /-
        case refine_2.refine_1
        R : Type u
        S : Type v
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        M : Type u_1
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : Module S M
        inst✝ : IsScalarTower R S M
        D : Derivation R S M
        x✝ : Subtype fun x => Membership.mem (KaehlerDifferential.ideal R S) x
        hx✝ : Membership.mem (Submodule.restrictScalars S (HSMul.hSMul (KaehlerDiffere …
        x : TensorProduct R S S
        hx : Eq ((Algebra.TensorProduct.lmul' R) x) 0
        y : Subtype fun x => Membership.mem (KaehlerDifferential.ideal R S) x
        ⊢ Eq ((D.tensorProductTo.comp (↑S (Submodule.subtype (KaehlerDifferential.idea …
      -/
      dsimp
      /-
        case refine_2.refine_1
        R : Type u
        S : Type v
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        M : Type u_1
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : Module S M
        inst✝ : IsScalarTower R S M
        D : Derivation R S M
        x✝ : Subtype fun x => Membership.mem (KaehlerDifferential.ideal R S) x
        hx✝ : Membership.mem (Submodule.restrictScalars S (HSMul.hSMul (KaehlerDiffere …
        x : TensorProduct R S S
        hx : Eq ((Algebra.TensorProduct.lmul' R) x) 0
        y : Subtype fun x => Membership.mem (KaehlerDifferential.ideal R S) x
        ⊢ Eq (D.tensorProductTo (HMul.hMul x ↑y)) 0
      -/
      rw [Derivation.tensorProductTo_mul, hx, y.prop, zero_smul, zero_smul, zero_add]
      /-
        🎉 no goals
      -/
      /-
        case refine_2.refine_2
        R : Type u
        S : Type v
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        M : Type u_1
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : Module S M
        inst✝ : IsScalarTower R S M
        D : Derivation R S M
        x : Subtype fun x => Membership.mem (KaehlerDifferential.ideal R S) x
        hx : Membership.mem (Submodule.restrictScalars S (HSMul.hSMul (KaehlerDifferen …
        ⊢ ∀ (x y : Subtype fun x => Membership.mem (KaehlerDifferential.ideal R S) x), …
      -/
    · intro x y ex ey; rw [map_add, ex, ey, zero_add]
                       /-
                         🎉 no goals
                       -/


theorem Derivation.liftKaehlerDifferential_apply (D : Derivation R S M) (x) :
    D.liftKaehlerDifferential ((KaehlerDifferential.ideal R S).toCotangent x) =
      D.tensorProductTo x := rfl


theorem Derivation.liftKaehlerDifferential_comp (D : Derivation R S M) :
    D.liftKaehlerDifferential.compDer (KaehlerDifferential.D R S) = D := by
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    M : Type u_1
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module S M
    inst✝ : IsScalarTower R S M
    D : Derivation R S M
    ⊢ Eq (D.liftKaehlerDifferential.compDer (KaehlerDifferential.D R S)) D
  -/
  ext a
  /-
    case H
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    M : Type u_1
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module S M
    inst✝ : IsScalarTower R S M
    D : Derivation R S M
    a : S
    ⊢ Eq ((D.liftKaehlerDifferential.compDer (KaehlerDifferential.D R S)) a) (D a)
  -/
  dsimp [KaehlerDifferential.D_apply]
  /-
    case H
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    M : Type u_1
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module S M
    inst✝ : IsScalarTower R S M
    D : Derivation R S M
    a : S
    ⊢ Eq (D.liftKaehlerDifferential ((KaehlerDifferential.ideal R S).toCotangent ⟨ …
  -/
  refine (D.liftKaehlerDifferential_apply _).trans ?_
  rw [Subtype.coe_mk, map_sub, Derivation.tensorProductTo_tmul, Derivation.tensorProductTo_tmul,
    one_smul, D.map_one_eq_zero, smul_zero, sub_zero]


@[simp]
theorem Derivation.liftKaehlerDifferential_comp_D (D' : Derivation R S M) (x : S) :
    D'.liftKaehlerDifferential (KaehlerDifferential.D R S x) = D' x :=
  Derivation.congr_fun D'.liftKaehlerDifferential_comp x


@[ext]
theorem Derivation.liftKaehlerDifferential_unique (f f' : Ω[S⁄R] →ₗ[S] M)
    (hf : f.compDer (KaehlerDifferential.D R S) = f'.compDer (KaehlerDifferential.D R S)) :
    f = f' := by
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    M : Type u_1
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module S M
    inst✝ : IsScalarTower R S M
    f f' : LinearMap (RingHom.id S) (KaehlerDifferential R S) M
    hf : Eq (f.compDer (KaehlerDifferential.D R S)) (f'.compDer (KaehlerDifferenti …
    ⊢ Eq f f'
  -/
  apply LinearMap.ext
  /-
    case h
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    M : Type u_1
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module S M
    inst✝ : IsScalarTower R S M
    f f' : LinearMap (RingHom.id S) (KaehlerDifferential R S) M
    hf : Eq (f.compDer (KaehlerDifferential.D R S)) (f'.compDer (KaehlerDifferenti …
    ⊢ ∀ (x : KaehlerDifferential R S), Eq (f x) (f' x)
  -/
  intro x
  have : x ∈ Submodule.span S (Set.range <| KaehlerDifferential.D R S) := by
    rw [KaehlerDifferential.span_range_derivation]; trivial
  /-
    case h
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    M : Type u_1
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module S M
    inst✝ : IsScalarTower R S M
    f f' : LinearMap (RingHom.id S) (KaehlerDifferential R S) M
    hf : Eq (f.compDer (KaehlerDifferential.D R S)) (f'.compDer (KaehlerDifferenti …
    x : KaehlerDifferential R S
    this : Membership.mem (Submodule.span S (Set.range ⇑(KaehlerDifferential.D R S …
    ⊢ Eq (f x) (f' x)
  -/
  refine Submodule.span_induction ?_ ?_ ?_ ?_ this
    /-
      case h.refine_1
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      M : Type u_1
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : Module S M
      inst✝ : IsScalarTower R S M
      f f' : LinearMap (RingHom.id S) (KaehlerDifferential R S) M
      hf : Eq (f.compDer (KaehlerDifferential.D R S)) (f'.compDer (KaehlerDifferenti …
      x : KaehlerDifferential R S
      this : Membership.mem (Submodule.span S (Set.range ⇑(KaehlerDifferential.D R S …
      ⊢ ∀ (x : KaehlerDifferential R S), Membership.mem (Set.range ⇑(KaehlerDifferen …
    -/
  · rintro _ ⟨x, rfl⟩; exact congr_arg (fun D : Derivation R S M => D x) hf
                       /-
                         🎉 no goals
                       -/
    /-
      case h.refine_2
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      M : Type u_1
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : Module S M
      inst✝ : IsScalarTower R S M
      f f' : LinearMap (RingHom.id S) (KaehlerDifferential R S) M
      hf : Eq (f.compDer (KaehlerDifferential.D R S)) (f'.compDer (KaehlerDifferenti …
      x : KaehlerDifferential R S
      this : Membership.mem (Submodule.span S (Set.range ⇑(KaehlerDifferential.D R S …
      ⊢ Eq (f 0) (f' 0)
    -/
  · rw [map_zero, map_zero]
    /-
      🎉 no goals
    -/
    /-
      case h.refine_3
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      M : Type u_1
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : Module S M
      inst✝ : IsScalarTower R S M
      f f' : LinearMap (RingHom.id S) (KaehlerDifferential R S) M
      hf : Eq (f.compDer (KaehlerDifferential.D R S)) (f'.compDer (KaehlerDifferenti …
      x : KaehlerDifferential R S
      this : Membership.mem (Submodule.span S (Set.range ⇑(KaehlerDifferential.D R S …
      ⊢ ∀ (x y : KaehlerDifferential R S), Membership.mem (Submodule.span S (Set.ran …
    -/
  · intro x y _ _ hx hy; rw [map_add, map_add, hx, hy]
                         /-
                           🎉 no goals
                         -/
    /-
      case h.refine_4
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      M : Type u_1
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : Module S M
      inst✝ : IsScalarTower R S M
      f f' : LinearMap (RingHom.id S) (KaehlerDifferential R S) M
      hf : Eq (f.compDer (KaehlerDifferential.D R S)) (f'.compDer (KaehlerDifferenti …
      x : KaehlerDifferential R S
      this : Membership.mem (Submodule.span S (Set.range ⇑(KaehlerDifferential.D R S …
      ⊢ ∀ (a : S) (x : KaehlerDifferential R S), Membership.mem (Submodule.span S (S …
    -/
  · intro a x _ e; simp [e]
                   /-
                     🎉 no goals
                   -/


theorem Derivation.liftKaehlerDifferential_D :
    (KaehlerDifferential.D R S).liftKaehlerDifferential = LinearMap.id :=
  Derivation.liftKaehlerDifferential_unique _ _
    (KaehlerDifferential.D R S).liftKaehlerDifferential_comp


theorem KaehlerDifferential.D_tensorProductTo (x : KaehlerDifferential.ideal R S) :
    (KaehlerDifferential.D R S).tensorProductTo x =
      (KaehlerDifferential.ideal R S).toCotangent x := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    x : Subtype fun x => Membership.mem (KaehlerDifferential.ideal R S) x
    ⊢ Eq ((KaehlerDifferential.D R S).tensorProductTo ↑x) ((KaehlerDifferential.id …
  -/
  rw [← Derivation.liftKaehlerDifferential_apply, Derivation.liftKaehlerDifferential_D]
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    x : Subtype fun x => Membership.mem (KaehlerDifferential.ideal R S) x
    ⊢ Eq (LinearMap.id ((KaehlerDifferential.ideal R S).toCotangent x)) ((KaehlerD …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem KaehlerDifferential.tensorProductTo_surjective :
    Function.Surjective (KaehlerDifferential.D R S).tensorProductTo := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    ⊢ Function.Surjective ⇑(KaehlerDifferential.D R S).tensorProductTo
  -/
  intro x; obtain ⟨x, rfl⟩ := (KaehlerDifferential.ideal R S).toCotangent_surjective x
  /-
    case intro
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    x : Subtype fun x => Membership.mem (KaehlerDifferential.ideal R S) x
    ⊢ Exists fun a => Eq ((KaehlerDifferential.D R S).tensorProductTo a) ((Kaehler …
  -/
  exact ⟨x, KaehlerDifferential.D_tensorProductTo x⟩
  /-
    🎉 no goals
  -/


/-- The `S`-linear maps from `Ω[S⁄R]` to `M` are (`S`-linearly) equivalent to `R`-derivations
from `S` to `M`. -/
@[simps! symm_apply apply_apply]
def KaehlerDifferential.linearMapEquivDerivation : (Ω[S⁄R] →ₗ[S] M) ≃ₗ[S] Derivation R S M :=
  { Derivation.llcomp.flip <| KaehlerDifferential.D R S with
    invFun := Derivation.liftKaehlerDifferential
    left_inv := fun _ =>
      Derivation.liftKaehlerDifferential_unique _ _ (Derivation.liftKaehlerDifferential_comp _)
    right_inv := Derivation.liftKaehlerDifferential_comp }


/-- The quotient ring of `S ⊗ S ⧸ J ^ 2` by `Ω[S⁄R]` is isomorphic to `S`. -/
def KaehlerDifferential.quotientCotangentIdealRingEquiv :
    (S ⊗ S ⧸ KaehlerDifferential.ideal R S ^ 2) ⧸ (KaehlerDifferential.ideal R S).cotangentIdeal ≃+*
      S := by
  have : Function.RightInverse (TensorProduct.includeLeft (R := R) (S := R) (A := S) (B := S))
      (↑(TensorProduct.lmul' R : S ⊗[R] S →ₐ[R] S) : S ⊗[R] S →+* S) := by
    intro x; rw [AlgHom.coe_toRingHom, ← AlgHom.comp_apply, TensorProduct.lmul'_comp_includeLeft]
    rfl
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    M : Type u_1
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module S M
    inst✝ : IsScalarTower R S M
    this : Function.RightInverse ⇑Algebra.TensorProduct.includeLeft ⇑↑(Algebra.Ten …
    ⊢ RingEquiv (HasQuotient.Quotient (HasQuotient.Quotient (TensorProduct R S S)  …
  -/
  refine (Ideal.quotCotangent _).trans ?_
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    M : Type u_1
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module S M
    inst✝ : IsScalarTower R S M
    this : Function.RightInverse ⇑Algebra.TensorProduct.includeLeft ⇑↑(Algebra.Ten …
    ⊢ RingEquiv (HasQuotient.Quotient (TensorProduct R S S) (KaehlerDifferential.i …
  -/
  refine (Ideal.quotEquivOfEq ?_).trans (RingHom.quotientKerEquivOfRightInverse this)
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    M : Type u_1
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module S M
    inst✝ : IsScalarTower R S M
    this : Function.RightInverse ⇑Algebra.TensorProduct.includeLeft ⇑↑(Algebra.Ten …
    ⊢ Eq (KaehlerDifferential.ideal R S) (RingHom.ker ↑(Algebra.TensorProduct.lmul …
  -/
  ext; rfl
       /-
         🎉 no goals
       -/


/-- The quotient ring of `S ⊗ S ⧸ J ^ 2` by `Ω[S⁄R]` is isomorphic to `S` as an `S`-algebra. -/
def KaehlerDifferential.quotientCotangentIdeal :
    ((S ⊗ S ⧸ KaehlerDifferential.ideal R S ^ 2) ⧸
        (KaehlerDifferential.ideal R S).cotangentIdeal) ≃ₐ[S] S :=
  { KaehlerDifferential.quotientCotangentIdealRingEquiv R S with
    commutes' := (KaehlerDifferential.quotientCotangentIdealRingEquiv R S).apply_symm_apply }


theorem KaehlerDifferential.End_equiv_aux (f : S →ₐ[R] S ⊗ S ⧸ KaehlerDifferential.ideal R S ^ 2) :
    (Ideal.Quotient.mkₐ R (KaehlerDifferential.ideal R S).cotangentIdeal).comp f =
        IsScalarTower.toAlgHom R S _ ↔
      (TensorProduct.lmul' R : S ⊗[R] S →ₐ[R] S).kerSquareLift.comp f = AlgHom.id R S := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    f : AlgHom R S (HasQuotient.Quotient (TensorProduct R S S) (HPow.hPow (Kaehler …
    ⊢ Iff (Eq ((Ideal.Quotient.mkₐ R (KaehlerDifferential.ideal R S).cotangentIdea …
  -/
  rw [AlgHom.ext_iff, AlgHom.ext_iff]
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    f : AlgHom R S (HasQuotient.Quotient (TensorProduct R S S) (HPow.hPow (Kaehler …
    ⊢ Iff (∀ (x : S), Eq (((Ideal.Quotient.mkₐ R (KaehlerDifferential.ideal R S).c …
  -/
  apply forall_congr'
  /-
    case h
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    f : AlgHom R S (HasQuotient.Quotient (TensorProduct R S S) (HPow.hPow (Kaehler …
    ⊢ ∀ (a : S), Iff (Eq (((Ideal.Quotient.mkₐ R (KaehlerDifferential.ideal R S).c …
  -/
  intro x
  have e₁ : (TensorProduct.lmul' R : S ⊗[R] S →ₐ[R] S).kerSquareLift (f x) =
      KaehlerDifferential.quotientCotangentIdealRingEquiv R S
        (Ideal.Quotient.mk (KaehlerDifferential.ideal R S).cotangentIdeal <| f x) := by
    generalize f x = y; obtain ⟨y, rfl⟩ := Ideal.Quotient.mk_surjective y; rfl
  have e₂ :
    x = KaehlerDifferential.quotientCotangentIdealRingEquiv R S (IsScalarTower.toAlgHom R S _ x) :=
    (mul_one x).symm
  /-
    case h
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    f : AlgHom R S (HasQuotient.Quotient (TensorProduct R S S) (HPow.hPow (Kaehler …
    x : S
    e₁ : Eq ((Algebra.TensorProduct.lmul' R).kerSquareLift (f x)) ((KaehlerDiffere …
    e₂ : Eq x ((KaehlerDifferential.quotientCotangentIdealRingEquiv R S) ((IsScala …
    ⊢ Iff (Eq (((Ideal.Quotient.mkₐ R (KaehlerDifferential.ideal R S).cotangentIde …
  -/
  constructor
    /-
      case h.mp
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      f : AlgHom R S (HasQuotient.Quotient (TensorProduct R S S) (HPow.hPow (Kaehler …
      x : S
      e₁ : Eq ((Algebra.TensorProduct.lmul' R).kerSquareLift (f x)) ((KaehlerDiffere …
      e₂ : Eq x ((KaehlerDifferential.quotientCotangentIdealRingEquiv R S) ((IsScala …
      ⊢ Eq (((Ideal.Quotient.mkₐ R (KaehlerDifferential.ideal R S).cotangentIdeal).c …
    -/
  · intro e
    exact (e₁.trans (@RingEquiv.congr_arg _ _ _ _ _ _
      (KaehlerDifferential.quotientCotangentIdealRingEquiv R S) _ _ e)).trans e₂.symm
    /-
      case h.mpr
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      f : AlgHom R S (HasQuotient.Quotient (TensorProduct R S S) (HPow.hPow (Kaehler …
      x : S
      e₁ : Eq ((Algebra.TensorProduct.lmul' R).kerSquareLift (f x)) ((KaehlerDiffere …
      e₂ : Eq x ((KaehlerDifferential.quotientCotangentIdealRingEquiv R S) ((IsScala …
      ⊢ Eq (((Algebra.TensorProduct.lmul' R).kerSquareLift.comp f) x) ((AlgHom.id R  …
    -/
  · intro e; apply (KaehlerDifferential.quotientCotangentIdealRingEquiv R S).injective
    /-
      case h.mpr.a
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      f : AlgHom R S (HasQuotient.Quotient (TensorProduct R S S) (HPow.hPow (Kaehler …
      x : S
      e₁ : Eq ((Algebra.TensorProduct.lmul' R).kerSquareLift (f x)) ((KaehlerDiffere …
      e₂ : Eq x ((KaehlerDifferential.quotientCotangentIdealRingEquiv R S) ((IsScala …
      e : Eq (((Algebra.TensorProduct.lmul' R).kerSquareLift.comp f) x) ((AlgHom.id  …
      ⊢ Eq ((KaehlerDifferential.quotientCotangentIdealRingEquiv R S) (((Ideal.Quoti …
    -/
    exact e₁.symm.trans (e.trans e₂)
    /-
      🎉 no goals
    -/

/- Note: Lean is slow to synthesize these instances (times out).
  Without them the endEquivDerivation' and endEquivAuxEquiv both have significant timeouts.
  In Mathlib 3, it was slow but not this slow. -/

/-- A shortcut instance to prevent timing out. Hopefully to be removed in the future. -/
local instance smul_SSmod_SSmod : SMul (S ⊗[R] S ⧸ KaehlerDifferential.ideal R S ^ 2)
    (S ⊗[R] S ⧸ KaehlerDifferential.ideal R S ^ 2) := Mul.toSMul _


/-- A shortcut instance to prevent timing out. Hopefully to be removed in the future. -/
@[nolint defLemma]
local instance isScalarTower_S_right :
    IsScalarTower S (S ⊗[R] S ⧸ KaehlerDifferential.ideal R S ^ 2)
      (S ⊗[R] S ⧸ KaehlerDifferential.ideal R S ^ 2) := Ideal.Quotient.isScalarTower_right


/-- A shortcut instance to prevent timing out. Hopefully to be removed in the future. -/
@[nolint defLemma]
local instance isScalarTower_R_right :
    IsScalarTower R (S ⊗[R] S ⧸ KaehlerDifferential.ideal R S ^ 2)
      (S ⊗[R] S ⧸ KaehlerDifferential.ideal R S ^ 2) := Ideal.Quotient.isScalarTower_right


/-- A shortcut instance to prevent timing out. Hopefully to be removed in the future. -/
@[nolint defLemma]
local instance isScalarTower_SS_right : IsScalarTower (S ⊗[R] S)
    (S ⊗[R] S ⧸ KaehlerDifferential.ideal R S ^ 2) (S ⊗[R] S ⧸ KaehlerDifferential.ideal R S ^ 2) :=
  Ideal.Quotient.isScalarTower_right


/-- A shortcut instance to prevent timing out. Hopefully to be removed in the future. -/
local instance instS : Module S (KaehlerDifferential.ideal R S).cotangentIdeal :=
  Submodule.module' _


/-- A shortcut instance to prevent timing out. Hopefully to be removed in the future. -/
local instance instR : Module R (KaehlerDifferential.ideal R S).cotangentIdeal :=
  Submodule.module' _


/-- A shortcut instance to prevent timing out. Hopefully to be removed in the future. -/
local instance instSS : Module (S ⊗[R] S) (KaehlerDifferential.ideal R S).cotangentIdeal :=
  Submodule.module' _


/-- Derivations into `Ω[S⁄R]` is equivalent to derivations
into `(KaehlerDifferential.ideal R S).cotangentIdeal`. -/
noncomputable def KaehlerDifferential.endEquivDerivation' :
    Derivation R S (Ω[S⁄R]) ≃ₗ[R] Derivation R S (ideal R S).cotangentIdeal :=
  LinearEquiv.compDer ((KaehlerDifferential.ideal R S).cotangentEquivIdeal.restrictScalars S)


/-- (Implementation) An `Equiv` version of `KaehlerDifferential.End_equiv_aux`.
Used in `KaehlerDifferential.endEquiv`. -/
def KaehlerDifferential.endEquivAuxEquiv :
    { f //
        (Ideal.Quotient.mkₐ R (KaehlerDifferential.ideal R S).cotangentIdeal).comp f =
          IsScalarTower.toAlgHom R S _ } ≃
      { f // (TensorProduct.lmul' R : S ⊗[R] S →ₐ[R] S).kerSquareLift.comp f = AlgHom.id R S } :=
  (Equiv.refl _).subtypeEquiv (KaehlerDifferential.End_equiv_aux R S)


/--
The endomorphisms of `Ω[S⁄R]` corresponds to sections of the surjection `S ⊗[R] S ⧸ J ^ 2 →ₐ[R] S`,
with `J` being the kernel of the multiplication map `S ⊗[R] S →ₐ[R] S`.
-/
noncomputable def KaehlerDifferential.endEquiv :
    Module.End S (Ω[S⁄R]) ≃
      { f // (TensorProduct.lmul' R : S ⊗[R] S →ₐ[R] S).kerSquareLift.comp f = AlgHom.id R S } :=
  (KaehlerDifferential.linearMapEquivDerivation R S).toEquiv.trans <|
    (KaehlerDifferential.endEquivDerivation' R S).toEquiv.trans <|
      (derivationToSquareZeroEquivLift (KaehlerDifferential.ideal R S).cotangentIdeal
            (KaehlerDifferential.ideal R S).cotangentIdeal_square).trans <|
        KaehlerDifferential.endEquivAuxEquiv R S


theorem KaehlerDifferential.ideal_fg [EssFiniteType R S] :
    (KaehlerDifferential.ideal R S).FG := by
  classical
  use (EssFiniteType.finset R S).image (fun s ↦ (1 : S) ⊗ₜ[R] s - s ⊗ₜ[R] (1 : S))
  apply le_antisymm
  · rw [Finset.coe_image, Ideal.span_le]
    rintro _ ⟨x, _, rfl⟩
    exact KaehlerDifferential.one_smul_sub_smul_one_mem_ideal R x
  · rw [← KaehlerDifferential.span_range_eq_ideal, Ideal.span_le]
    rintro _ ⟨x, rfl⟩
    let I : Ideal (S ⊗[R] S) := Ideal.span
      ((EssFiniteType.finset R S).image (fun s ↦ (1 : S) ⊗ₜ[R] s - s ⊗ₜ[R] (1 : S)))
    show _ - _ ∈ I
    have : (IsScalarTower.toAlgHom R (S ⊗[R] S) (S ⊗[R] S ⧸ I)).comp TensorProduct.includeRight =
        (IsScalarTower.toAlgHom R (S ⊗[R] S) (S ⊗[R] S ⧸ I)).comp TensorProduct.includeLeft := by
      apply EssFiniteType.algHom_ext
      intro a ha
      simp only [AlgHom.coe_comp, IsScalarTower.coe_toAlgHom', Ideal.Quotient.algebraMap_eq,
        Function.comp_apply, TensorProduct.includeLeft_apply, TensorProduct.includeRight_apply,
        Ideal.Quotient.mk_eq_mk_iff_sub_mem]
      refine Ideal.subset_span ?_
      simp only [Finset.coe_image, Set.mem_image, Finset.mem_coe]
      exact ⟨a, ha, rfl⟩
    simpa [Ideal.Quotient.mk_eq_mk_iff_sub_mem] using AlgHom.congr_fun this x


instance KaehlerDifferential.finite [EssFiniteType R S] :
    Module.Finite S (Ω[S⁄R]) := by
  classical
  let s := (EssFiniteType.finset R S).image (fun s ↦ D R S s)
  refine ⟨⟨s, top_le_iff.mp ?_⟩⟩
  rw [← span_range_derivation, Submodule.span_le]
  rintro _ ⟨x, rfl⟩
  have : ∀ x ∈ adjoin R (EssFiniteType.finset R S).toSet,
      .D _ _ x ∈ Submodule.span S s.toSet := by
    intro x hx
    refine adjoin_induction ?_ ?_ ?_ ?_ hx
    · exact fun x hx ↦ Submodule.subset_span (Finset.mem_image_of_mem _ hx)
    · simp
    · exact fun x y _ _ hx hy ↦ (D R S).map_add x y ▸ add_mem hx hy
    · intro x y _ _ hx hy
      simp only [Derivation.leibniz]
      exact add_mem (Submodule.smul_mem _ _ hy) (Submodule.smul_mem _ _ hx)
  obtain ⟨t, ht, ht', hxt⟩ := (essFiniteType_cond_iff R S (EssFiniteType.finset R S)).mp
    EssFiniteType.cond.choose_spec x
  rw [show D R S x =
    ht'.unit⁻¹ • (D R S (x * t) - x • D R S t) by simp [smul_smul, Units.smul_def]]
  exact Submodule.smul_mem _ _ (sub_mem (this _ hxt) (Submodule.smul_mem _ _ (this _ ht)))


/-- The `S`-submodule of `S →₀ S` (the direct sum of copies of `S` indexed by `S`) generated by
the relations:
1. `dx + dy = d(x + y)`
2. `x dy + y dx = d(x * y)`
3. `dr = 0` for `r ∈ R`
where `db` is the unit in the copy of `S` with index `b`.

This is the kernel of the surjection
`Finsupp.linearCombination S Ω[S⁄R] S (KaehlerDifferential.D R S)`.
See `KaehlerDifferential.kerTotal_eq` and `KaehlerDifferential.linearCombination_surjective`.
-/
noncomputable def KaehlerDifferential.kerTotal : Submodule S (S →₀ S) :=
  Submodule.span S
    (((Set.range fun x : S × S => single x.1 1 + single x.2 1 - single (x.1 + x.2) 1) ∪
        Set.range fun x : S × S => single x.2 x.1 + single x.1 x.2 - single (x.1 * x.2) 1) ∪
      Set.range fun x : R => single (algebraMap R S x) 1)


unsuppress_compilation in
-- Porting note: was `local notation x "𝖣" y => (KaehlerDifferential.kerTotal R S).mkQ (single y x)`
-- but not having `DFunLike.coe` leads to `kerTotal_mkQ_single_smul` failing.
local notation3 x "𝖣" y => DFunLike.coe (KaehlerDifferential.kerTotal R S).mkQ (single y x)


theorem KaehlerDifferential.kerTotal_mkQ_single_add (x y z) : (z𝖣x + y) = (z𝖣x) + z𝖣y := by
  rw [← map_add, eq_comm, ← sub_eq_zero, ← map_sub (Submodule.mkQ (kerTotal R S)),
    Submodule.mkQ_apply, Submodule.Quotient.mk_eq_zero]
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    x y z : S
    ⊢ Membership.mem (KaehlerDifferential.kerTotal R S) (HSub.hSub (HAdd.hAdd (Fin …
  -/
  simp_rw [← Finsupp.smul_single_one _ z, ← smul_add, ← smul_sub]
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    x y z : S
    ⊢ Membership.mem (KaehlerDifferential.kerTotal R S) (HSMul.hSMul z (HSub.hSub  …
  -/
  exact Submodule.smul_mem _ _ (Submodule.subset_span (Or.inl <| Or.inl <| ⟨⟨_, _⟩, rfl⟩))
  /-
    🎉 no goals
  -/


theorem KaehlerDifferential.kerTotal_mkQ_single_mul (x y z) :
    (z𝖣x * y) = ((z * x)𝖣y) + (z * y)𝖣x := by
  rw [← map_add, eq_comm, ← sub_eq_zero, ← map_sub (Submodule.mkQ (kerTotal R S)),
    Submodule.mkQ_apply, Submodule.Quotient.mk_eq_zero]
  simp_rw [← Finsupp.smul_single_one _ z, ← @smul_eq_mul _ _ z, ← Finsupp.smul_single, ← smul_add,
    ← smul_sub]
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    x y z : S
    ⊢ Membership.mem (KaehlerDifferential.kerTotal R S) (HSMul.hSMul z (HSub.hSub  …
  -/
  exact Submodule.smul_mem _ _ (Submodule.subset_span (Or.inl <| Or.inr <| ⟨⟨_, _⟩, rfl⟩))
  /-
    🎉 no goals
  -/


theorem KaehlerDifferential.kerTotal_mkQ_single_algebraMap (x y) : (y𝖣algebraMap R S x) = 0 := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    x : R
    y : S
    ⊢ Eq ((KaehlerDifferential.kerTotal R S).mkQ (Finsupp.single ((algebraMap R S) …
  -/
  rw [Submodule.mkQ_apply, Submodule.Quotient.mk_eq_zero, ← Finsupp.smul_single_one _ y]
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    x : R
    y : S
    ⊢ Membership.mem (KaehlerDifferential.kerTotal R S) (HSMul.hSMul y (Finsupp.si …
  -/
  exact Submodule.smul_mem _ _ (Submodule.subset_span (Or.inr <| ⟨_, rfl⟩))
  /-
    🎉 no goals
  -/


theorem KaehlerDifferential.kerTotal_mkQ_single_algebraMap_one (x) : (x𝖣1) = 0 := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    x : S
    ⊢ Eq ((KaehlerDifferential.kerTotal R S).mkQ (Finsupp.single 1 x)) 0
  -/
  rw [← (algebraMap R S).map_one, KaehlerDifferential.kerTotal_mkQ_single_algebraMap]
  /-
    🎉 no goals
  -/


theorem KaehlerDifferential.kerTotal_mkQ_single_smul (r : R) (x y) : (y𝖣r • x) = r • y𝖣x := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    r : R
    x y : S
    ⊢ Eq ((KaehlerDifferential.kerTotal R S).mkQ (Finsupp.single (HSMul.hSMul r x) …
  -/
  letI : SMulZeroClass R S := inferInstance
  rw [Algebra.smul_def, KaehlerDifferential.kerTotal_mkQ_single_mul,
    KaehlerDifferential.kerTotal_mkQ_single_algebraMap, add_zero, ← LinearMap.map_smul_of_tower,
    Finsupp.smul_single, mul_comm, Algebra.smul_def]


/-- The (universal) derivation into `(S →₀ S) ⧸ KaehlerDifferential.kerTotal R S`. -/
noncomputable def KaehlerDifferential.derivationQuotKerTotal :
    Derivation R S ((S →₀ S) ⧸ KaehlerDifferential.kerTotal R S) where
  toFun x := 1𝖣x
  map_add' _ _ := KaehlerDifferential.kerTotal_mkQ_single_add _ _ _ _ _
  map_smul' _ _ := KaehlerDifferential.kerTotal_mkQ_single_smul _ _ _ _ _
  map_one_eq_zero' := KaehlerDifferential.kerTotal_mkQ_single_algebraMap_one _ _ _
  leibniz' a b :=
    (KaehlerDifferential.kerTotal_mkQ_single_mul _ _ _ _ _).trans
          /-
            R : Type u
            S : Type v
            inst✝⁶ : CommRing R
            inst✝⁵ : CommRing S
            inst✝⁴ : Algebra R S
            M : Type u_1
            inst✝³ : AddCommGroup M
            inst✝² : Module R M
            inst✝¹ : Module S M
            inst✝ : IsScalarTower R S M
            a b : S
            ⊢ Eq (HAdd.hAdd ((KaehlerDifferential.kerTotal R S).mkQ (Finsupp.single b (HMu …
          -/
      (by simp_rw [← Finsupp.smul_single_one _ (1 * _ : S)]; dsimp; simp)
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem KaehlerDifferential.derivationQuotKerTotal_apply (x) :
    KaehlerDifferential.derivationQuotKerTotal R S x = 1𝖣x :=
  rfl


theorem KaehlerDifferential.derivationQuotKerTotal_lift_comp_linearCombination :
    (KaehlerDifferential.derivationQuotKerTotal R S).liftKaehlerDifferential.comp
        (Finsupp.linearCombination S (KaehlerDifferential.D R S)) =
      Submodule.mkQ _ := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    ⊢ Eq ((KaehlerDifferential.derivationQuotKerTotal R S).liftKaehlerDifferential …
  -/
  apply Finsupp.lhom_ext
  /-
    case h
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    ⊢ ∀ (a b : S), Eq (((KaehlerDifferential.derivationQuotKerTotal R S).liftKaehl …
  -/
  intro a b
  /-
    case h
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    a b : S
    ⊢ Eq (((KaehlerDifferential.derivationQuotKerTotal R S).liftKaehlerDifferentia …
  -/
  conv_rhs => rw [← Finsupp.smul_single_one a b, LinearMap.map_smul]
  /-
    case h
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    a b : S
    ⊢ Eq (((KaehlerDifferential.derivationQuotKerTotal R S).liftKaehlerDifferentia …
  -/
  simp [KaehlerDifferential.derivationQuotKerTotal_apply]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-08-29")] alias
  KaehlerDifferential.derivationQuotKerTotal_lift_comp_total :=
  KaehlerDifferential.derivationQuotKerTotal_lift_comp_linearCombination


theorem KaehlerDifferential.kerTotal_eq :
    LinearMap.ker (Finsupp.linearCombination S (KaehlerDifferential.D R S)) =
      KaehlerDifferential.kerTotal R S := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    ⊢ Eq (LinearMap.ker (Finsupp.linearCombination S ⇑(KaehlerDifferential.D R S)) …
  -/
  apply le_antisymm
    /-
      case a
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      ⊢ LE.le (LinearMap.ker (Finsupp.linearCombination S ⇑(KaehlerDifferential.D R  …
    -/
  · conv_rhs => rw [← (KaehlerDifferential.kerTotal R S).ker_mkQ]
    /-
      case a
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      ⊢ LE.le (LinearMap.ker (Finsupp.linearCombination S ⇑(KaehlerDifferential.D R  …
    -/
    rw [← KaehlerDifferential.derivationQuotKerTotal_lift_comp_linearCombination]
    /-
      case a
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      ⊢ LE.le (LinearMap.ker (Finsupp.linearCombination S ⇑(KaehlerDifferential.D R  …
    -/
    exact LinearMap.ker_le_ker_comp _ _
    /-
      🎉 no goals
    -/
    /-
      case a
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      ⊢ LE.le (KaehlerDifferential.kerTotal R S) (LinearMap.ker (Finsupp.linearCombi …
    -/
  · rw [KaehlerDifferential.kerTotal, Submodule.span_le]
    /-
      case a
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      ⊢ HasSubset.Subset (Union.union (Union.union (Set.range fun x => HSub.hSub (HA …
    -/
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
    rintro _ ((⟨⟨x, y⟩, rfl⟩ | ⟨⟨x, y⟩, rfl⟩) | ⟨x, rfl⟩) <;> dsimp <;> simp [LinearMap.mem_ker]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


theorem KaehlerDifferential.linearCombination_surjective :
    Function.Surjective (Finsupp.linearCombination S (KaehlerDifferential.D R S)) := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    ⊢ Function.Surjective ⇑(Finsupp.linearCombination S ⇑(KaehlerDifferential.D R  …
  -/
  rw [← LinearMap.range_eq_top, range_linearCombination, span_range_derivation]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-08-29")] alias KaehlerDifferential.total_surjective :=
  KaehlerDifferential.linearCombination_surjective


/-- `Ω[S⁄R]` is isomorphic to `S` copies of `S` with kernel `KaehlerDifferential.kerTotal`. -/
@[simps!]
noncomputable def KaehlerDifferential.quotKerTotalEquiv :
    ((S →₀ S) ⧸ KaehlerDifferential.kerTotal R S) ≃ₗ[S] Ω[S⁄R] :=
  { (KaehlerDifferential.kerTotal R S).liftQ
      (Finsupp.linearCombination S (KaehlerDifferential.D R S))
      (KaehlerDifferential.kerTotal_eq R S).ge with
    invFun := (KaehlerDifferential.derivationQuotKerTotal R S).liftKaehlerDifferential
    left_inv := by
      /-
        R : Type u
        S : Type v
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        M : Type u_1
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : Module S M
        inst✝ : IsScalarTower R S M
        ⊢ Function.LeftInverse (⇑(KaehlerDifferential.derivationQuotKerTotal R S).lift …
      -/
      intro x
      /-
        R : Type u
        S : Type v
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        M : Type u_1
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : Module S M
        inst✝ : IsScalarTower R S M
        x : HasQuotient.Quotient (Finsupp S S) (KaehlerDifferential.kerTotal R S)
        ⊢ Eq ((KaehlerDifferential.derivationQuotKerTotal R S).liftKaehlerDifferential …
      -/
      obtain ⟨x, rfl⟩ := Submodule.mkQ_surjective _ x
      exact
        LinearMap.congr_fun
          (KaehlerDifferential.derivationQuotKerTotal_lift_comp_linearCombination R S : _) x
    right_inv := by
      /-
        R : Type u
        S : Type v
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        M : Type u_1
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : Module S M
        inst✝ : IsScalarTower R S M
        ⊢ Function.RightInverse (⇑(KaehlerDifferential.derivationQuotKerTotal R S).lif …
      -/
      intro x
      /-
        R : Type u
        S : Type v
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        M : Type u_1
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : Module S M
        inst✝ : IsScalarTower R S M
        x : KaehlerDifferential R S
        ⊢ Eq (__src✝.toFun ((KaehlerDifferential.derivationQuotKerTotal R S).liftKaehl …
      -/
      obtain ⟨x, rfl⟩ := KaehlerDifferential.linearCombination_surjective R S x
      have := LinearMap.congr_fun
        (KaehlerDifferential.derivationQuotKerTotal_lift_comp_linearCombination R S) x
      /-
        case intro
        R : Type u
        S : Type v
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        M : Type u_1
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : Module S M
        inst✝ : IsScalarTower R S M
        x : Finsupp S S
        this : Eq (((KaehlerDifferential.derivationQuotKerTotal R S).liftKaehlerDiffer …
        ⊢ Eq (__src✝.toFun ((KaehlerDifferential.derivationQuotKerTotal R S).liftKaehl …
      -/
      rw [LinearMap.comp_apply] at this
      /-
        case intro
        R : Type u
        S : Type v
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        M : Type u_1
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : Module S M
        inst✝ : IsScalarTower R S M
        x : Finsupp S S
        this : Eq ((KaehlerDifferential.derivationQuotKerTotal R S).liftKaehlerDiffere …
        ⊢ Eq (__src✝.toFun ((KaehlerDifferential.derivationQuotKerTotal R S).liftKaehl …
      -/
      rw [this]
      /-
        case intro
        R : Type u
        S : Type v
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        M : Type u_1
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : Module S M
        inst✝ : IsScalarTower R S M
        x : Finsupp S S
        this : Eq ((KaehlerDifferential.derivationQuotKerTotal R S).liftKaehlerDiffere …
        ⊢ Eq (__src✝.toFun ((KaehlerDifferential.kerTotal R S).mkQ x)) ((Finsupp.linea …
      -/
      rfl }
      /-
        🎉 no goals
      -/


theorem KaehlerDifferential.quotKerTotalEquiv_symm_comp_D :
    (KaehlerDifferential.quotKerTotalEquiv R S).symm.toLinearMap.compDer
        (KaehlerDifferential.D R S) =
      KaehlerDifferential.derivationQuotKerTotal R S := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    ⊢ Eq ((↑(KaehlerDifferential.quotKerTotalEquiv R S).symm).compDer (KaehlerDiff …
  -/
  convert (KaehlerDifferential.derivationQuotKerTotal R S).liftKaehlerDifferential_comp
  /-
    🎉 no goals
  -/


unsuppress_compilation in
-- The map `(A →₀ A) →ₗ[A] (B →₀ B)`
local macro "finsupp_map" : term =>
  `((Finsupp.mapRange.linearMap (Algebra.linearMap A B)).comp
    (Finsupp.lmapDomain A A (algebraMap A B)))


/--
Given the commutative diagram
```
A --→ B
↑     ↑
|     |
R --→ S
```
The kernel of the presentation `⊕ₓ B dx ↠ Ω_{B/S}` is spanned by the image of the
kernel of `⊕ₓ A dx ↠ Ω_{A/R}` and all `ds` with `s : S`.
See `kerTotal_map'` for the special case where `R = S`.
-/
theorem KaehlerDifferential.kerTotal_map [Algebra R B] [IsScalarTower R A B] [IsScalarTower R S B]
    (h : Function.Surjective (algebraMap A B)) :
    (KaehlerDifferential.kerTotal R A).map finsupp_map ⊔
        Submodule.span A (Set.range fun x : S => .single (algebraMap S B x) (1 : B)) =
      (KaehlerDifferential.kerTotal S B).restrictScalars _ := by
  rw [KaehlerDifferential.kerTotal, Submodule.map_span, KaehlerDifferential.kerTotal,
    Submodule.restrictScalars_span _ _ h]
  simp_rw [Set.image_union, Submodule.span_union, ← Set.image_univ, Set.image_image, Set.image_univ,
    map_sub, map_add]
  simp only [LinearMap.comp_apply, Finsupp.lmapDomain_apply, Finsupp.mapDomain_single,
    Finsupp.mapRange.linearMap_apply, Finsupp.mapRange_single, Algebra.linearMap_apply,
    map_one, map_add, map_mul]
  /-
    R : Type u
    S : Type v
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    A : Type u_2
    B : Type u_3
    inst✝⁷ : CommRing A
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra R A
    inst✝⁴ : Algebra A B
    inst✝³ : Algebra S B
    inst✝² : Algebra R B
    inst✝¹ : IsScalarTower R A B
    inst✝ : IsScalarTower R S B
    h : Function.Surjective ⇑(algebraMap A B)
    ⊢ Eq (Max.max (Max.max (Max.max (Submodule.span A (Set.range fun x => HSub.hSu …
  -/
  simp_rw [sup_assoc, ← (h.prodMap h).range_comp]
  /-
    R : Type u
    S : Type v
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    A : Type u_2
    B : Type u_3
    inst✝⁷ : CommRing A
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra R A
    inst✝⁴ : Algebra A B
    inst✝³ : Algebra S B
    inst✝² : Algebra R B
    inst✝¹ : IsScalarTower R A B
    inst✝ : IsScalarTower R S B
    h : Function.Surjective ⇑(algebraMap A B)
    ⊢ Eq (Max.max (Submodule.span A (Set.range fun x => HSub.hSub (HAdd.hAdd (Fins …
  -/
  congr!
  -- Porting note: new
  /-
    case h.e'_4.h.e'_4
    R : Type u
    S : Type v
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    A : Type u_2
    B : Type u_3
    inst✝⁷ : CommRing A
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra R A
    inst✝⁴ : Algebra A B
    inst✝³ : Algebra S B
    inst✝² : Algebra R B
    inst✝¹ : IsScalarTower R A B
    inst✝ : IsScalarTower R S B
    h : Function.Surjective ⇑(algebraMap A B)
    ⊢ Eq (Max.max (Submodule.span A (Set.range fun x => Finsupp.single ((algebraMa …
  -/
  simp_rw [← IsScalarTower.algebraMap_apply R A B]
  /-
    case h.e'_4.h.e'_4
    R : Type u
    S : Type v
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    A : Type u_2
    B : Type u_3
    inst✝⁷ : CommRing A
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra R A
    inst✝⁴ : Algebra A B
    inst✝³ : Algebra S B
    inst✝² : Algebra R B
    inst✝¹ : IsScalarTower R A B
    inst✝ : IsScalarTower R S B
    h : Function.Surjective ⇑(algebraMap A B)
    ⊢ Eq (Max.max (Submodule.span A (Set.range fun x => Finsupp.single ((algebraMa …
  -/
  rw [sup_eq_right]
  /-
    case h.e'_4.h.e'_4
    R : Type u
    S : Type v
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    A : Type u_2
    B : Type u_3
    inst✝⁷ : CommRing A
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra R A
    inst✝⁴ : Algebra A B
    inst✝³ : Algebra S B
    inst✝² : Algebra R B
    inst✝¹ : IsScalarTower R A B
    inst✝ : IsScalarTower R S B
    h : Function.Surjective ⇑(algebraMap A B)
    ⊢ LE.le (Submodule.span A (Set.range fun x => Finsupp.single ((algebraMap R B) …
  -/
  apply Submodule.span_mono
  /-
    case h.e'_4.h.e'_4.h
    R : Type u
    S : Type v
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    A : Type u_2
    B : Type u_3
    inst✝⁷ : CommRing A
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra R A
    inst✝⁴ : Algebra A B
    inst✝³ : Algebra S B
    inst✝² : Algebra R B
    inst✝¹ : IsScalarTower R A B
    inst✝ : IsScalarTower R S B
    h : Function.Surjective ⇑(algebraMap A B)
    ⊢ HasSubset.Subset (Set.range fun x => Finsupp.single ((algebraMap R B) x) 1)  …
  -/
  simp_rw [IsScalarTower.algebraMap_apply R S B]
  exact Set.range_comp_subset_range (algebraMap R S)
    fun x => Finsupp.single (algebraMap S B x) (1 : B)


/--
This is a special case of `kerTotal_map` where `R = S`.
The kernel of the presentation `⊕ₓ B dx ↠ Ω_{B/R}` is spanned by the image of the
kernel of `⊕ₓ A dx ↠ Ω_{A/R}` and all `da` with `a : A`.
-/
theorem KaehlerDifferential.kerTotal_map' [Algebra R B]
    [IsScalarTower R A B] (h : Function.Surjective (algebraMap A B)) :
    (KaehlerDifferential.kerTotal R A ⊔
      Submodule.span A (Set.range fun x ↦ .single (algebraMap R A x) 1)).map finsupp_map =
      (KaehlerDifferential.kerTotal R B).restrictScalars _ := by
  /-
    R : Type u
    inst✝⁶ : CommRing R
    A : Type u_2
    B : Type u_3
    inst✝⁵ : CommRing A
    inst✝⁴ : CommRing B
    inst✝³ : Algebra R A
    inst✝² : Algebra A B
    inst✝¹ : Algebra R B
    inst✝ : IsScalarTower R A B
    h : Function.Surjective ⇑(algebraMap A B)
    ⊢ Eq (Submodule.map ((Finsupp.mapRange.linearMap (Algebra.linearMap A B)).comp …
  -/
  rw [Submodule.map_sup, ← kerTotal_map R R A B h, Submodule.map_span, ← Set.range_comp]
  /-
    R : Type u
    inst✝⁶ : CommRing R
    A : Type u_2
    B : Type u_3
    inst✝⁵ : CommRing A
    inst✝⁴ : CommRing B
    inst✝³ : Algebra R A
    inst✝² : Algebra A B
    inst✝¹ : Algebra R B
    inst✝ : IsScalarTower R A B
    h : Function.Surjective ⇑(algebraMap A B)
    ⊢ Eq (Max.max (Submodule.map ((Finsupp.mapRange.linearMap (Algebra.linearMap A …
  -/
  congr
  /-
    case e_a.e_s
    R : Type u
    inst✝⁶ : CommRing R
    A : Type u_2
    B : Type u_3
    inst✝⁵ : CommRing A
    inst✝⁴ : CommRing B
    inst✝³ : Algebra R A
    inst✝² : Algebra A B
    inst✝¹ : Algebra R B
    inst✝ : IsScalarTower R A B
    h : Function.Surjective ⇑(algebraMap A B)
    ⊢ Eq (Set.range (Function.comp ⇑((Finsupp.mapRange.linearMap (Algebra.linearMa …
  -/
  refine congr_arg Set.range ?_
  /-
    case e_a.e_s
    R : Type u
    inst✝⁶ : CommRing R
    A : Type u_2
    B : Type u_3
    inst✝⁵ : CommRing A
    inst✝⁴ : CommRing B
    inst✝³ : Algebra R A
    inst✝² : Algebra A B
    inst✝¹ : Algebra R B
    inst✝ : IsScalarTower R A B
    h : Function.Surjective ⇑(algebraMap A B)
    ⊢ Eq (Function.comp ⇑((Finsupp.mapRange.linearMap (Algebra.linearMap A B)).com …
  -/
  ext; simp [IsScalarTower.algebraMap_eq R A B]
       /-
         🎉 no goals
       -/


/-- The map `Ω[A⁄R] →ₗ[A] Ω[B⁄S]` given a square
```
A --→ B
↑     ↑
|     |
R --→ S
```
-/
def KaehlerDifferential.map : Ω[A⁄R] →ₗ[A] Ω[B⁄S] :=
  Derivation.liftKaehlerDifferential
    (((KaehlerDifferential.D S B).restrictScalars R).compAlgebraMap A)


theorem KaehlerDifferential.map_compDer :
    (KaehlerDifferential.map R S A B).compDer (KaehlerDifferential.D R A) =
      ((KaehlerDifferential.D S B).restrictScalars R).compAlgebraMap A :=
  Derivation.liftKaehlerDifferential_comp _


@[simp]
theorem KaehlerDifferential.map_D (x : A) :
    KaehlerDifferential.map R S A B (KaehlerDifferential.D R A x) =
      KaehlerDifferential.D S B (algebraMap A B x) :=
  Derivation.congr_fun (KaehlerDifferential.map_compDer R S A B) x


theorem KaehlerDifferential.ker_map :
    LinearMap.ker (KaehlerDifferential.map R S A B) =
      (((kerTotal S B).restrictScalars A).comap finsupp_map).map
        (Finsupp.linearCombination (M := Ω[A⁄R]) A (D R A)) := by
  /-
    R : Type u
    S : Type v
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Algebra R S
    A : Type u_2
    B : Type u_3
    inst✝⁸ : CommRing A
    inst✝⁷ : CommRing B
    inst✝⁶ : Algebra R A
    inst✝⁵ : Algebra A B
    inst✝⁴ : Algebra S B
    inst✝³ : Algebra R B
    inst✝² : IsScalarTower R A B
    inst✝¹ : IsScalarTower R S B
    inst✝ : SMulCommClass S A B
    ⊢ Eq (LinearMap.ker (KaehlerDifferential.map R S A B)) (Submodule.map (Finsupp …
  -/
  rw [← Submodule.map_comap_eq_of_surjective (linearCombination_surjective R A) (LinearMap.ker _)]
  /-
    R : Type u
    S : Type v
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Algebra R S
    A : Type u_2
    B : Type u_3
    inst✝⁸ : CommRing A
    inst✝⁷ : CommRing B
    inst✝⁶ : Algebra R A
    inst✝⁵ : Algebra A B
    inst✝⁴ : Algebra S B
    inst✝³ : Algebra R B
    inst✝² : IsScalarTower R A B
    inst✝¹ : IsScalarTower R S B
    inst✝ : SMulCommClass S A B
    ⊢ Eq (Submodule.map (Finsupp.linearCombination A ⇑(KaehlerDifferential.D R A)) …
  -/
  congr 1
  /-
    case e_p
    R : Type u
    S : Type v
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Algebra R S
    A : Type u_2
    B : Type u_3
    inst✝⁸ : CommRing A
    inst✝⁷ : CommRing B
    inst✝⁶ : Algebra R A
    inst✝⁵ : Algebra A B
    inst✝⁴ : Algebra S B
    inst✝³ : Algebra R B
    inst✝² : IsScalarTower R A B
    inst✝¹ : IsScalarTower R S B
    inst✝ : SMulCommClass S A B
    ⊢ Eq (Submodule.comap (Finsupp.linearCombination A ⇑(KaehlerDifferential.D R A …
  -/
  ext x
  simp only [Submodule.mem_comap, LinearMap.mem_ker, Finsupp.apply_linearCombination, ← kerTotal_eq,
    Submodule.restrictScalars_mem]
  simp only [linearCombination_apply, Function.comp_apply, LinearMap.coe_comp, lmapDomain_apply,
    Finsupp.mapRange.linearMap_apply]
  /-
    case e_p.h
    R : Type u
    S : Type v
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Algebra R S
    A : Type u_2
    B : Type u_3
    inst✝⁸ : CommRing A
    inst✝⁷ : CommRing B
    inst✝⁶ : Algebra R A
    inst✝⁵ : Algebra A B
    inst✝⁴ : Algebra S B
    inst✝³ : Algebra R B
    inst✝² : IsScalarTower R A B
    inst✝¹ : IsScalarTower R S B
    inst✝ : SMulCommClass S A B
    x : Finsupp A A
    ⊢ Iff (Eq (x.sum fun i a => HSMul.hSMul a ((KaehlerDifferential.map R S A B) ( …
  -/
  rw [Finsupp.sum_mapRange_index, Finsupp.sum_mapDomain_index]
    /-
      case e_p.h
      R : Type u
      S : Type v
      inst✝¹¹ : CommRing R
      inst✝¹⁰ : CommRing S
      inst✝⁹ : Algebra R S
      A : Type u_2
      B : Type u_3
      inst✝⁸ : CommRing A
      inst✝⁷ : CommRing B
      inst✝⁶ : Algebra R A
      inst✝⁵ : Algebra A B
      inst✝⁴ : Algebra S B
      inst✝³ : Algebra R B
      inst✝² : IsScalarTower R A B
      inst✝¹ : IsScalarTower R S B
      inst✝ : SMulCommClass S A B
      x : Finsupp A A
      ⊢ Iff (Eq (x.sum fun i a => HSMul.hSMul a ((KaehlerDifferential.map R S A B) ( …
    -/
  · simp [ofId]
    /-
      🎉 no goals
    -/
    /-
      case e_p.h.h_zero
      R : Type u
      S : Type v
      inst✝¹¹ : CommRing R
      inst✝¹⁰ : CommRing S
      inst✝⁹ : Algebra R S
      A : Type u_2
      B : Type u_3
      inst✝⁸ : CommRing A
      inst✝⁷ : CommRing B
      inst✝⁶ : Algebra R A
      inst✝⁵ : Algebra A B
      inst✝⁴ : Algebra S B
      inst✝³ : Algebra R B
      inst✝² : IsScalarTower R A B
      inst✝¹ : IsScalarTower R S B
      inst✝ : SMulCommClass S A B
      x : Finsupp A A
      ⊢ ∀ (b : B), Eq (HSMul.hSMul ((Algebra.linearMap A B) 0) ((KaehlerDifferential …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case e_p.h.h_add
      R : Type u
      S : Type v
      inst✝¹¹ : CommRing R
      inst✝¹⁰ : CommRing S
      inst✝⁹ : Algebra R S
      A : Type u_2
      B : Type u_3
      inst✝⁸ : CommRing A
      inst✝⁷ : CommRing B
      inst✝⁶ : Algebra R A
      inst✝⁵ : Algebra A B
      inst✝⁴ : Algebra S B
      inst✝³ : Algebra R B
      inst✝² : IsScalarTower R A B
      inst✝¹ : IsScalarTower R S B
      inst✝ : SMulCommClass S A B
      x : Finsupp A A
      ⊢ ∀ (b : B) (m₁ m₂ : A), Eq (HSMul.hSMul ((Algebra.linearMap A B) (HAdd.hAdd m …
    -/
  · simp [add_smul]
    /-
      🎉 no goals
    -/
    /-
      case e_p.h
      R : Type u
      S : Type v
      inst✝¹¹ : CommRing R
      inst✝¹⁰ : CommRing S
      inst✝⁹ : Algebra R S
      A : Type u_2
      B : Type u_3
      inst✝⁸ : CommRing A
      inst✝⁷ : CommRing B
      inst✝⁶ : Algebra R A
      inst✝⁵ : Algebra A B
      inst✝⁴ : Algebra S B
      inst✝³ : Algebra R B
      inst✝² : IsScalarTower R A B
      inst✝¹ : IsScalarTower R S B
      inst✝ : SMulCommClass S A B
      x : Finsupp A A
      ⊢ ∀ (a : B), Eq (HSMul.hSMul 0 ((KaehlerDifferential.D S B) a)) 0
    -/
  · simp
    /-
      🎉 no goals
    -/


lemma KaehlerDifferential.ker_map_of_surjective (h : Function.Surjective (algebraMap A B)) :
    LinearMap.ker (map R R A B) =
      (LinearMap.ker finsupp_map).map (Finsupp.linearCombination A (D R A)) := by
  rw [ker_map, ← kerTotal_map' R A B h, Submodule.comap_map_eq, Submodule.map_sup,
    Submodule.map_sup, ← kerTotal_eq, ← Submodule.comap_bot,
    Submodule.map_comap_eq_of_surjective (linearCombination_surjective _ _),
    bot_sup_eq, Submodule.map_span, ← Set.range_comp]
  /-
    R : Type u
    inst✝⁶ : CommRing R
    A : Type u_2
    B : Type u_3
    inst✝⁵ : CommRing A
    inst✝⁴ : CommRing B
    inst✝³ : Algebra R A
    inst✝² : Algebra A B
    inst✝¹ : Algebra R B
    inst✝ : IsScalarTower R A B
    h : Function.Surjective ⇑(algebraMap A B)
    ⊢ Eq (Max.max (Submodule.span A (Set.range (Function.comp ⇑(Finsupp.linearComb …
  -/
  convert bot_sup_eq _
  /-
    case h.e'_2.h.e'_3
    R : Type u
    inst✝⁶ : CommRing R
    A : Type u_2
    B : Type u_3
    inst✝⁵ : CommRing A
    inst✝⁴ : CommRing B
    inst✝³ : Algebra R A
    inst✝² : Algebra A B
    inst✝¹ : Algebra R B
    inst✝ : IsScalarTower R A B
    h : Function.Surjective ⇑(algebraMap A B)
    ⊢ Eq (Submodule.span A (Set.range (Function.comp ⇑(Finsupp.linearCombination A …
  -/
  rw [Submodule.span_eq_bot]; simp
                              /-
                                🎉 no goals
                              -/


theorem KaehlerDifferential.map_surjective_of_surjective
    (h : Function.Surjective (algebraMap A B)) :
    Function.Surjective (KaehlerDifferential.map R S A B) := by
  rw [← LinearMap.range_eq_top, _root_.eq_top_iff,
    ← @Submodule.restrictScalars_top A B, ← span_range_derivation,
    Submodule.restrictScalars_span _ _ h, Submodule.span_le]
  /-
    R : Type u
    S : Type v
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Algebra R S
    A : Type u_2
    B : Type u_3
    inst✝⁸ : CommRing A
    inst✝⁷ : CommRing B
    inst✝⁶ : Algebra R A
    inst✝⁵ : Algebra A B
    inst✝⁴ : Algebra S B
    inst✝³ : Algebra R B
    inst✝² : IsScalarTower R A B
    inst✝¹ : IsScalarTower R S B
    inst✝ : SMulCommClass S A B
    h : Function.Surjective ⇑(algebraMap A B)
    ⊢ HasSubset.Subset (Set.range ⇑(KaehlerDifferential.D S B)) ↑(LinearMap.range  …
  -/
  rintro _ ⟨x, rfl⟩
  /-
    case intro
    R : Type u
    S : Type v
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Algebra R S
    A : Type u_2
    B : Type u_3
    inst✝⁸ : CommRing A
    inst✝⁷ : CommRing B
    inst✝⁶ : Algebra R A
    inst✝⁵ : Algebra A B
    inst✝⁴ : Algebra S B
    inst✝³ : Algebra R B
    inst✝² : IsScalarTower R A B
    inst✝¹ : IsScalarTower R S B
    inst✝ : SMulCommClass S A B
    h : Function.Surjective ⇑(algebraMap A B)
    x : B
    ⊢ Membership.mem (↑(LinearMap.range (KaehlerDifferential.map R S A B))) ((Kaeh …
  -/
  obtain ⟨y, rfl⟩ := h x
  /-
    case intro.intro
    R : Type u
    S : Type v
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Algebra R S
    A : Type u_2
    B : Type u_3
    inst✝⁸ : CommRing A
    inst✝⁷ : CommRing B
    inst✝⁶ : Algebra R A
    inst✝⁵ : Algebra A B
    inst✝⁴ : Algebra S B
    inst✝³ : Algebra R B
    inst✝² : IsScalarTower R A B
    inst✝¹ : IsScalarTower R S B
    inst✝ : SMulCommClass S A B
    h : Function.Surjective ⇑(algebraMap A B)
    y : A
    ⊢ Membership.mem (↑(LinearMap.range (KaehlerDifferential.map R S A B))) ((Kaeh …
  -/
  rw [← KaehlerDifferential.map_D R S A B]
  /-
    case intro.intro
    R : Type u
    S : Type v
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Algebra R S
    A : Type u_2
    B : Type u_3
    inst✝⁸ : CommRing A
    inst✝⁷ : CommRing B
    inst✝⁶ : Algebra R A
    inst✝⁵ : Algebra A B
    inst✝⁴ : Algebra S B
    inst✝³ : Algebra R B
    inst✝² : IsScalarTower R A B
    inst✝¹ : IsScalarTower R S B
    inst✝ : SMulCommClass S A B
    h : Function.Surjective ⇑(algebraMap A B)
    y : A
    ⊢ Membership.mem (↑(LinearMap.range (KaehlerDifferential.map R S A B))) ((Kaeh …
  -/
  exact ⟨_, rfl⟩
  /-
    🎉 no goals
  -/


theorem KaehlerDifferential.map_surjective :
    Function.Surjective (KaehlerDifferential.map R S B B) :=
  map_surjective_of_surjective R S B B Function.surjective_id


/-- The lift of the map `Ω[A⁄R] →ₗ[A] Ω[B⁄R]` to the base change along `A → B`.
This is the first map in the exact sequence `B ⊗[A] Ω[A⁄R] → Ω[B⁄R] → Ω[B⁄A] → 0`. -/
noncomputable def KaehlerDifferential.mapBaseChange : B ⊗[A] Ω[A⁄R] →ₗ[B] Ω[B⁄R] :=
  (TensorProduct.isBaseChange A (Ω[A⁄R]) B).lift (KaehlerDifferential.map R R A B)


@[simp]
theorem KaehlerDifferential.mapBaseChange_tmul (x : B) (y : Ω[A⁄R]) :
    KaehlerDifferential.mapBaseChange R A B (x ⊗ₜ y) = x • KaehlerDifferential.map R R A B y := by
  /-
    R : Type u
    inst✝⁶ : CommRing R
    A : Type u_2
    B : Type u_3
    inst✝⁵ : CommRing A
    inst✝⁴ : CommRing B
    inst✝³ : Algebra R A
    inst✝² : Algebra A B
    inst✝¹ : Algebra R B
    inst✝ : IsScalarTower R A B
    x : B
    y : KaehlerDifferential R A
    ⊢ Eq ((KaehlerDifferential.mapBaseChange R A B) (TensorProduct.tmul A x y)) (H …
  -/
  conv_lhs => rw [← mul_one x, ← smul_eq_mul, ← TensorProduct.smul_tmul', LinearMap.map_smul]
  /-
    R : Type u
    inst✝⁶ : CommRing R
    A : Type u_2
    B : Type u_3
    inst✝⁵ : CommRing A
    inst✝⁴ : CommRing B
    inst✝³ : Algebra R A
    inst✝² : Algebra A B
    inst✝¹ : Algebra R B
    inst✝ : IsScalarTower R A B
    x : B
    y : KaehlerDifferential R A
    ⊢ Eq (HSMul.hSMul x ((KaehlerDifferential.mapBaseChange R A B) (TensorProduct. …
  -/
  congr 1
  /-
    case e_a
    R : Type u
    inst✝⁶ : CommRing R
    A : Type u_2
    B : Type u_3
    inst✝⁵ : CommRing A
    inst✝⁴ : CommRing B
    inst✝³ : Algebra R A
    inst✝² : Algebra A B
    inst✝¹ : Algebra R B
    inst✝ : IsScalarTower R A B
    x : B
    y : KaehlerDifferential R A
    ⊢ Eq ((KaehlerDifferential.mapBaseChange R A B) (TensorProduct.tmul A 1 y)) (( …
  -/
  exact IsBaseChange.lift_eq _ _ _
  /-
    🎉 no goals
  -/


lemma KaehlerDifferential.range_mapBaseChange :
    LinearMap.range (mapBaseChange R A B) = LinearMap.ker (map R A B B) := by
  /-
    R : Type u
    inst✝⁶ : CommRing R
    A : Type u_2
    B : Type u_3
    inst✝⁵ : CommRing A
    inst✝⁴ : CommRing B
    inst✝³ : Algebra R A
    inst✝² : Algebra A B
    inst✝¹ : Algebra R B
    inst✝ : IsScalarTower R A B
    ⊢ Eq (LinearMap.range (KaehlerDifferential.mapBaseChange R A B)) (LinearMap.ke …
  -/
  apply le_antisymm
    /-
      case a
      R : Type u
      inst✝⁶ : CommRing R
      A : Type u_2
      B : Type u_3
      inst✝⁵ : CommRing A
      inst✝⁴ : CommRing B
      inst✝³ : Algebra R A
      inst✝² : Algebra A B
      inst✝¹ : Algebra R B
      inst✝ : IsScalarTower R A B
      ⊢ LE.le (LinearMap.range (KaehlerDifferential.mapBaseChange R A B)) (LinearMap …
    -/
  · rintro _ ⟨x, rfl⟩
    /-
      case a.intro
      R : Type u
      inst✝⁶ : CommRing R
      A : Type u_2
      B : Type u_3
      inst✝⁵ : CommRing A
      inst✝⁴ : CommRing B
      inst✝³ : Algebra R A
      inst✝² : Algebra A B
      inst✝¹ : Algebra R B
      inst✝ : IsScalarTower R A B
      x : TensorProduct A B (KaehlerDifferential R A)
      ⊢ Membership.mem (LinearMap.ker (KaehlerDifferential.map R A B B)) ((KaehlerDi …
    -/
    induction' x with r s
      /-
        case a.intro.zero
        R : Type u
        inst✝⁶ : CommRing R
        A : Type u_2
        B : Type u_3
        inst✝⁵ : CommRing A
        inst✝⁴ : CommRing B
        inst✝³ : Algebra R A
        inst✝² : Algebra A B
        inst✝¹ : Algebra R B
        inst✝ : IsScalarTower R A B
        ⊢ Membership.mem (LinearMap.ker (KaehlerDifferential.map R A B B)) ((KaehlerDi …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case a.intro.tmul
        R : Type u
        inst✝⁶ : CommRing R
        A : Type u_2
        B : Type u_3
        inst✝⁵ : CommRing A
        inst✝⁴ : CommRing B
        inst✝³ : Algebra R A
        inst✝² : Algebra A B
        inst✝¹ : Algebra R B
        inst✝ : IsScalarTower R A B
        r : B
        s : KaehlerDifferential R A
        ⊢ Membership.mem (LinearMap.ker (KaehlerDifferential.map R A B B)) ((KaehlerDi …
      -/
    · obtain ⟨x, rfl⟩ := linearCombination_surjective _ _ s
      /-
        case a.intro.tmul.intro
        R : Type u
        inst✝⁶ : CommRing R
        A : Type u_2
        B : Type u_3
        inst✝⁵ : CommRing A
        inst✝⁴ : CommRing B
        inst✝³ : Algebra R A
        inst✝² : Algebra A B
        inst✝¹ : Algebra R B
        inst✝ : IsScalarTower R A B
        r : B
        x : Finsupp A A
        ⊢ Membership.mem (LinearMap.ker (KaehlerDifferential.map R A B B)) ((KaehlerDi …
      -/
      simp only [mapBaseChange_tmul, LinearMap.mem_ker, map_smul]
      /-
        case a.intro.tmul.intro
        R : Type u
        inst✝⁶ : CommRing R
        A : Type u_2
        B : Type u_3
        inst✝⁵ : CommRing A
        inst✝⁴ : CommRing B
        inst✝³ : Algebra R A
        inst✝² : Algebra A B
        inst✝¹ : Algebra R B
        inst✝ : IsScalarTower R A B
        r : B
        x : Finsupp A A
        ⊢ Eq (HSMul.hSMul r ((KaehlerDifferential.map R A B B) ((KaehlerDifferential.m …
      -/
      induction x using Finsupp.induction_linear
        /-
          case a.intro.tmul.intro.h0
          R : Type u
          inst✝⁶ : CommRing R
          A : Type u_2
          B : Type u_3
          inst✝⁵ : CommRing A
          inst✝⁴ : CommRing B
          inst✝³ : Algebra R A
          inst✝² : Algebra A B
          inst✝¹ : Algebra R B
          inst✝ : IsScalarTower R A B
          r : B
          ⊢ Eq (HSMul.hSMul r ((KaehlerDifferential.map R A B B) ((KaehlerDifferential.m …
        -/
      · simp
        /-
          🎉 no goals
        -/
        /-
          case a.intro.tmul.intro.hadd
          R : Type u
          inst✝⁶ : CommRing R
          A : Type u_2
          B : Type u_3
          inst✝⁵ : CommRing A
          inst✝⁴ : CommRing B
          inst✝³ : Algebra R A
          inst✝² : Algebra A B
          inst✝¹ : Algebra R B
          inst✝ : IsScalarTower R A B
          r : B
          f✝ g✝ : Finsupp A A
          a✝¹ : Eq (HSMul.hSMul r ((KaehlerDifferential.map R A B B) ((KaehlerDifferenti …
          a✝ : Eq (HSMul.hSMul r ((KaehlerDifferential.map R A B B) ((KaehlerDifferentia …
          ⊢ Eq (HSMul.hSMul r ((KaehlerDifferential.map R A B B) ((KaehlerDifferential.m …
        -/
      · simp [smul_add, *]
        /-
          🎉 no goals
        -/
        /-
          case a.intro.tmul.intro.hsingle
          R : Type u
          inst✝⁶ : CommRing R
          A : Type u_2
          B : Type u_3
          inst✝⁵ : CommRing A
          inst✝⁴ : CommRing B
          inst✝³ : Algebra R A
          inst✝² : Algebra A B
          inst✝¹ : Algebra R B
          inst✝ : IsScalarTower R A B
          r : B
          a✝ b✝ : A
          ⊢ Eq (HSMul.hSMul r ((KaehlerDifferential.map R A B B) ((KaehlerDifferential.m …
        -/
      · simp
        /-
          🎉 no goals
        -/
      /-
        case a.intro.add
        R : Type u
        inst✝⁶ : CommRing R
        A : Type u_2
        B : Type u_3
        inst✝⁵ : CommRing A
        inst✝⁴ : CommRing B
        inst✝³ : Algebra R A
        inst✝² : Algebra A B
        inst✝¹ : Algebra R B
        inst✝ : IsScalarTower R A B
        x✝ y✝ : TensorProduct A B (KaehlerDifferential R A)
        a✝¹ : Membership.mem (LinearMap.ker (KaehlerDifferential.map R A B B)) ((Kaehl …
        a✝ : Membership.mem (LinearMap.ker (KaehlerDifferential.map R A B B)) ((Kaehle …
        ⊢ Membership.mem (LinearMap.ker (KaehlerDifferential.map R A B B)) ((KaehlerDi …
      -/
    · rw [map_add]; exact add_mem ‹_› ‹_›
                    /-
                      🎉 no goals
                    -/
    /-
      case a
      R : Type u
      inst✝⁶ : CommRing R
      A : Type u_2
      B : Type u_3
      inst✝⁵ : CommRing A
      inst✝⁴ : CommRing B
      inst✝³ : Algebra R A
      inst✝² : Algebra A B
      inst✝¹ : Algebra R B
      inst✝ : IsScalarTower R A B
      ⊢ LE.le (LinearMap.ker (KaehlerDifferential.map R A B B)) (LinearMap.range (Ka …
    -/
  · convert_to (kerTotal A B).map (Finsupp.linearCombination B (D R B)) ≤ _
      /-
        case h.e'_3
        R : Type u
        inst✝⁶ : CommRing R
        A : Type u_2
        B : Type u_3
        inst✝⁵ : CommRing A
        inst✝⁴ : CommRing B
        inst✝³ : Algebra R A
        inst✝² : Algebra A B
        inst✝¹ : Algebra R B
        inst✝ : IsScalarTower R A B
        ⊢ Eq (LinearMap.ker (KaehlerDifferential.map R A B B)) (Submodule.map (Finsupp …
      -/
    · rw [KaehlerDifferential.ker_map]
      /-
        case h.e'_3
        R : Type u
        inst✝⁶ : CommRing R
        A : Type u_2
        B : Type u_3
        inst✝⁵ : CommRing A
        inst✝⁴ : CommRing B
        inst✝³ : Algebra R A
        inst✝² : Algebra A B
        inst✝¹ : Algebra R B
        inst✝ : IsScalarTower R A B
        ⊢ Eq (Submodule.map (Finsupp.linearCombination B ⇑(KaehlerDifferential.D R B)) …
      -/
      congr 1
      /-
        case h.e'_3.e_p
        R : Type u
        inst✝⁶ : CommRing R
        A : Type u_2
        B : Type u_3
        inst✝⁵ : CommRing A
        inst✝⁴ : CommRing B
        inst✝³ : Algebra R A
        inst✝² : Algebra A B
        inst✝¹ : Algebra R B
        inst✝ : IsScalarTower R A B
        ⊢ Eq (Submodule.comap ((Finsupp.mapRange.linearMap (Algebra.linearMap B B)).co …
      -/
      convert Submodule.comap_id _
        /-
          case h.e'_2.h.e'_15
          R : Type u
          inst✝⁶ : CommRing R
          A : Type u_2
          B : Type u_3
          inst✝⁵ : CommRing A
          inst✝⁴ : CommRing B
          inst✝³ : Algebra R A
          inst✝² : Algebra A B
          inst✝¹ : Algebra R B
          inst✝ : IsScalarTower R A B
          ⊢ Eq ((Finsupp.mapRange.linearMap (Algebra.linearMap B B)).comp (Finsupp.lmapD …
        -/
      · ext; simp
             /-
               🎉 no goals
             -/
    /-
      case a.convert_2
      R : Type u
      inst✝⁶ : CommRing R
      A : Type u_2
      B : Type u_3
      inst✝⁵ : CommRing A
      inst✝⁴ : CommRing B
      inst✝³ : Algebra R A
      inst✝² : Algebra A B
      inst✝¹ : Algebra R B
      inst✝ : IsScalarTower R A B
      ⊢ LE.le (Submodule.map (Finsupp.linearCombination B ⇑(KaehlerDifferential.D R  …
    -/
    rw [Submodule.map_le_iff_le_comap, kerTotal, Submodule.span_le]
    /-
      case a.convert_2
      R : Type u
      inst✝⁶ : CommRing R
      A : Type u_2
      B : Type u_3
      inst✝⁵ : CommRing A
      inst✝⁴ : CommRing B
      inst✝³ : Algebra R A
      inst✝² : Algebra A B
      inst✝¹ : Algebra R B
      inst✝ : IsScalarTower R A B
      ⊢ HasSubset.Subset (Union.union (Union.union (Set.range fun x => HSub.hSub (HA …
    -/
    rintro f ((⟨⟨x, y⟩, rfl⟩|⟨⟨x, y⟩, rfl⟩)|⟨x, rfl⟩)
      /-
        case a.convert_2.inl.inl.intro.mk
        R : Type u
        inst✝⁶ : CommRing R
        A : Type u_2
        B : Type u_3
        inst✝⁵ : CommRing A
        inst✝⁴ : CommRing B
        inst✝³ : Algebra R A
        inst✝² : Algebra A B
        inst✝¹ : Algebra R B
        inst✝ : IsScalarTower R A B
        x y : B
        ⊢ Membership.mem (↑(Submodule.comap (Finsupp.linearCombination B ⇑(KaehlerDiff …
      -/
    · use 0; simp
             /-
               🎉 no goals
             -/
      /-
        case a.convert_2.inl.inr.intro.mk
        R : Type u
        inst✝⁶ : CommRing R
        A : Type u_2
        B : Type u_3
        inst✝⁵ : CommRing A
        inst✝⁴ : CommRing B
        inst✝³ : Algebra R A
        inst✝² : Algebra A B
        inst✝¹ : Algebra R B
        inst✝ : IsScalarTower R A B
        x y : B
        ⊢ Membership.mem (↑(Submodule.comap (Finsupp.linearCombination B ⇑(KaehlerDiff …
      -/
    · use 0; simp
             /-
               🎉 no goals
             -/
      /-
        case a.convert_2.inr.intro
        R : Type u
        inst✝⁶ : CommRing R
        A : Type u_2
        B : Type u_3
        inst✝⁵ : CommRing A
        inst✝⁴ : CommRing B
        inst✝³ : Algebra R A
        inst✝² : Algebra A B
        inst✝¹ : Algebra R B
        inst✝ : IsScalarTower R A B
        x : A
        ⊢ Membership.mem (↑(Submodule.comap (Finsupp.linearCombination B ⇑(KaehlerDiff …
      -/
    · use 1 ⊗ₜ D _ _ x; simp
                        /-
                          🎉 no goals
                        -/


/-- The sequence `B ⊗[A] Ω[A⁄R] → Ω[B⁄R] → Ω[B⁄A] → 0` is exact.
Also see `KaehlerDifferential.map_surjective`. -/
lemma KaehlerDifferential.exact_mapBaseChange_map :
    Function.Exact (mapBaseChange R A B) (map R A B B) :=
  SetLike.ext_iff.mp (range_mapBaseChange R A B).symm


/-- The map `I → B ⊗[A] Ω[A⁄R]` where `I = ker(A → B)`. -/
@[simps]
noncomputable
def KaehlerDifferential.kerToTensor :
    RingHom.ker (algebraMap A B) →ₗ[A] B ⊗[A] Ω[A⁄R] where
  toFun x := 1 ⊗ₜ D R A x
                     /-
                       R : Type u
                       S : Type v
                       inst✝¹¹ : CommRing R
                       inst✝¹⁰ : CommRing S
                       inst✝⁹ : Algebra R S
                       M : Type u_1
                       inst✝⁸ : AddCommGroup M
                       inst✝⁷ : Module R M
                       inst✝⁶ : Module S M
                       inst✝⁵ : IsScalarTower R S M
                       A : Type u_2
                       B : Type u_3
                       inst✝⁴ : CommRing A
                       inst✝³ : CommRing B
                       inst✝² : Algebra R A
                       inst✝¹ : Algebra A B
                       inst✝ : Algebra S B
                       x y : Subtype fun x => Membership.mem (RingHom.ker (algebraMap A B)) x
                       ⊢ Eq ((fun x => TensorProduct.tmul A 1 ((KaehlerDifferential.D R A) ↑x)) (HAdd …
                     -/
  map_add' x y := by simp only [Submodule.coe_add, map_add, TensorProduct.tmul_add]
                     /-
                       🎉 no goals
                     -/
  map_smul' r x := by simp only [SetLike.val_smul, smul_eq_mul, Derivation.leibniz,
    TensorProduct.tmul_add, TensorProduct.tmul_smul, TensorProduct.smul_tmul', ←
    algebraMap_eq_smul_one, RingHom.mem_ker.mp x.prop, TensorProduct.zero_tmul, add_zero,
    RingHom.id_apply]


/-- The map `I/I² → B ⊗[A] Ω[A⁄R]` where `I = ker(A → B)`. -/
noncomputable
def KaehlerDifferential.kerCotangentToTensor :
    (RingHom.ker (algebraMap A B)).Cotangent →ₗ[A] B ⊗[A] Ω[A⁄R] :=
  Submodule.liftQ _ (kerToTensor R A B) <| by
    /-
      R : Type u
      S : Type v
      inst✝¹¹ : CommRing R
      inst✝¹⁰ : CommRing S
      inst✝⁹ : Algebra R S
      M : Type u_1
      inst✝⁸ : AddCommGroup M
      inst✝⁷ : Module R M
      inst✝⁶ : Module S M
      inst✝⁵ : IsScalarTower R S M
      A : Type u_2
      B : Type u_3
      inst✝⁴ : CommRing A
      inst✝³ : CommRing B
      inst✝² : Algebra R A
      inst✝¹ : Algebra A B
      inst✝ : Algebra S B
      ⊢ LE.le (HSMul.hSMul (RingHom.ker (algebraMap A B)) Top.top) (LinearMap.ker (K …
    -/
    rw [Submodule.smul_eq_map₂]
    /-
      R : Type u
      S : Type v
      inst✝¹¹ : CommRing R
      inst✝¹⁰ : CommRing S
      inst✝⁹ : Algebra R S
      M : Type u_1
      inst✝⁸ : AddCommGroup M
      inst✝⁷ : Module R M
      inst✝⁶ : Module S M
      inst✝⁵ : IsScalarTower R S M
      A : Type u_2
      B : Type u_3
      inst✝⁴ : CommRing A
      inst✝³ : CommRing B
      inst✝² : Algebra R A
      inst✝¹ : Algebra A B
      inst✝ : Algebra S B
      ⊢ LE.le (Submodule.map₂ (LinearMap.lsmul A (Subtype fun x => Membership.mem (R …
    -/
    apply iSup_le_iff.mpr
    /-
      R : Type u
      S : Type v
      inst✝¹¹ : CommRing R
      inst✝¹⁰ : CommRing S
      inst✝⁹ : Algebra R S
      M : Type u_1
      inst✝⁸ : AddCommGroup M
      inst✝⁷ : Module R M
      inst✝⁶ : Module S M
      inst✝⁵ : IsScalarTower R S M
      A : Type u_2
      B : Type u_3
      inst✝⁴ : CommRing A
      inst✝³ : CommRing B
      inst✝² : Algebra R A
      inst✝¹ : Algebra A B
      inst✝ : Algebra S B
      ⊢ ∀ (i : Subtype fun x => Membership.mem (RingHom.ker (algebraMap A B)) x), LE …
    -/
    simp only [Submodule.map_le_iff_le_comap, Subtype.forall]
    /-
      R : Type u
      S : Type v
      inst✝¹¹ : CommRing R
      inst✝¹⁰ : CommRing S
      inst✝⁹ : Algebra R S
      M : Type u_1
      inst✝⁸ : AddCommGroup M
      inst✝⁷ : Module R M
      inst✝⁶ : Module S M
      inst✝⁵ : IsScalarTower R S M
      A : Type u_2
      B : Type u_3
      inst✝⁴ : CommRing A
      inst✝³ : CommRing B
      inst✝² : Algebra R A
      inst✝¹ : Algebra A B
      inst✝ : Algebra S B
      ⊢ ∀ (a : A), Membership.mem (RingHom.ker (algebraMap A B)) a → LE.le Top.top ( …
    -/
    rintro x hx y -
    simp only [Submodule.mem_comap, LinearMap.lsmul_apply, LinearMap.mem_ker, map_smul,
      kerToTensor_apply, TensorProduct.smul_tmul', ← algebraMap_eq_smul_one,
      RingHom.mem_ker.mp hx, TensorProduct.zero_tmul]


@[simp]
lemma KaehlerDifferential.kerCotangentToTensor_toCotangent (x) :
    kerCotangentToTensor R A B (Ideal.toCotangent _ x) = 1 ⊗ₜ D _ _ x.1 := rfl


theorem KaehlerDifferential.range_kerCotangentToTensor
    (h : Function.Surjective (algebraMap A B)) :
    LinearMap.range (kerCotangentToTensor R A B) =
      (LinearMap.ker (KaehlerDifferential.mapBaseChange R A B)).restrictScalars A := by
  classical
  ext x
  constructor
  · rintro ⟨x, rfl⟩
    obtain ⟨x, rfl⟩ := Ideal.toCotangent_surjective _ x
    simp only [kerCotangentToTensor_toCotangent, Submodule.restrictScalars_mem, LinearMap.mem_ker,
      mapBaseChange_tmul, map_D, RingHom.mem_ker.mp x.2, map_zero, smul_zero]
  · intro hx
    obtain ⟨x, rfl⟩ := LinearMap.rTensor_surjective (Ω[A⁄R]) (g := Algebra.linearMap A B) h x
    obtain ⟨x, rfl⟩ := (TensorProduct.lid _ _).symm.surjective x
    replace hx : x ∈ LinearMap.ker (KaehlerDifferential.map R R A B) := by simpa using hx
    rw [KaehlerDifferential.ker_map_of_surjective R A B h] at hx
    obtain ⟨x, hx, rfl⟩ := hx
    simp only [TensorProduct.lid_symm_apply, LinearMap.rTensor_tmul,
      Algebra.linearMap_apply, map_one]
    rw [← Finsupp.sum_single x, Finsupp.sum, ← Finset.sum_fiberwise_of_maps_to
      (fun _ ↦ Finset.mem_image_of_mem (algebraMap A B))]
    simp only [Function.comp_apply, map_sum (s := x.support.image (algebraMap A B)),
      TensorProduct.tmul_sum]
    apply sum_mem
    intro c _
    simp only [Finset.filter_congr_decidable, TensorProduct.lid_symm_apply, LinearMap.rTensor_tmul,
      AlgHom.toLinearMap_apply, map_one, LinearMap.mem_range]
    simp only [map_sum, Finsupp.linearCombination_single]
    have : (x.support.filter (algebraMap A B · = c)).sum x ∈ RingHom.ker (algebraMap A B) := by
      simpa [Finsupp.mapDomain, Finsupp.sum, Finsupp.finset_sum_apply, RingHom.mem_ker,
        Finsupp.single_apply, ← Finset.sum_filter] using DFunLike.congr_fun hx c
    obtain ⟨a, ha⟩ := h c
    use (x.support.filter (algebraMap A B · = c)).attach.sum
        fun i ↦ x i • Ideal.toCotangent _ ⟨i - a, ?_⟩; swap
    · have : x i ≠ 0 ∧ algebraMap A B i = c := by
        convert i.prop
        simp_rw [Finset.mem_filter, Finsupp.mem_support_iff]
      simp [RingHom.mem_ker, ha, this.2]
    · simp only [map_sum, LinearMapClass.map_smul, kerCotangentToTensor_toCotangent, map_sub]
      simp_rw [← TensorProduct.tmul_smul]
      -- was `simp [kerCotangentToTensor_toCotangent, RingHom.mem_ker.mp x.2]` and very slow
      -- (https://github.com/leanprover-community/mathlib4/issues/19751)
      simp only [smul_sub, TensorProduct.tmul_sub, Finset.sum_sub_distrib, ← TensorProduct.tmul_sum,
        ← Finset.sum_smul, Finset.sum_attach, sub_eq_self,
        Finset.sum_attach (f := fun i ↦ x i • KaehlerDifferential.D R A i)]
      rw [← TensorProduct.smul_tmul, ← Algebra.algebraMap_eq_smul_one, RingHom.mem_ker.mp this,
        TensorProduct.zero_tmul]


theorem KaehlerDifferential.exact_kerCotangentToTensor_mapBaseChange
    (h : Function.Surjective (algebraMap A B)) :
    Function.Exact (kerCotangentToTensor R A B) (KaehlerDifferential.mapBaseChange R A B) :=
  SetLike.ext_iff.mp (range_kerCotangentToTensor R A B h).symm


lemma KaehlerDifferential.mapBaseChange_surjective
    (h : Function.Surjective (algebraMap A B)) :
    Function.Surjective (KaehlerDifferential.mapBaseChange R A B) := by
  /-
    R : Type u
    inst✝⁶ : CommRing R
    A : Type u_2
    B : Type u_3
    inst✝⁵ : CommRing A
    inst✝⁴ : CommRing B
    inst✝³ : Algebra R A
    inst✝² : Algebra A B
    inst✝¹ : Algebra R B
    inst✝ : IsScalarTower R A B
    h : Function.Surjective ⇑(algebraMap A B)
    ⊢ Function.Surjective ⇑(KaehlerDifferential.mapBaseChange R A B)
  -/
  have := subsingleton_of_surjective A B h
  /-
    R : Type u
    inst✝⁶ : CommRing R
    A : Type u_2
    B : Type u_3
    inst✝⁵ : CommRing A
    inst✝⁴ : CommRing B
    inst✝³ : Algebra R A
    inst✝² : Algebra A B
    inst✝¹ : Algebra R B
    inst✝ : IsScalarTower R A B
    h : Function.Surjective ⇑(algebraMap A B)
    this : Subsingleton (KaehlerDifferential A B)
    ⊢ Function.Surjective ⇑(KaehlerDifferential.mapBaseChange R A B)
  -/
  rw [← LinearMap.range_eq_top, range_mapBaseChange, ← top_le_iff]
  /-
    R : Type u
    inst✝⁶ : CommRing R
    A : Type u_2
    B : Type u_3
    inst✝⁵ : CommRing A
    inst✝⁴ : CommRing B
    inst✝³ : Algebra R A
    inst✝² : Algebra A B
    inst✝¹ : Algebra R B
    inst✝ : IsScalarTower R A B
    h : Function.Surjective ⇑(algebraMap A B)
    this : Subsingleton (KaehlerDifferential A B)
    ⊢ LE.le Top.top (LinearMap.ker (KaehlerDifferential.map R A B B))
  -/
  exact fun x _ ↦ Subsingleton.elim _ _
  /-
    🎉 no goals
  -/


