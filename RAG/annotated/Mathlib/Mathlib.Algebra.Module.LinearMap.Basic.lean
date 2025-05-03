instance : SMul S'ᵈᵐᵃ (M →ₛₗ[σ₁₂] M') where
  smul a f :=
    { toFun := a • (f : M → M')
                               /-
                                 R : Type u_1
                                 R' : Type u_2
                                 S : Type u_3
                                 M : Type u_4
                                 M' : Type u_5
                                 inst✝¹¹ : Semiring R
                                 inst✝¹⁰ : Semiring R'
                                 inst✝⁹ : AddCommMonoid M
                                 inst✝⁸ : AddCommMonoid M'
                                 inst✝⁷ : Module R M
                                 inst✝⁶ : Module R' M'
                                 σ₁₂ : RingHom R R'
                                 S' : Type u_6
                                 T' : Type u_7
                                 inst✝⁵ : Monoid S'
                                 inst✝⁴ : DistribMulAction S' M
                                 inst✝³ : SMulCommClass R S' M
                                 inst✝² : Monoid T'
                                 inst✝¹ : DistribMulAction T' M
                                 inst✝ : SMulCommClass R T' M
                                 a : DomMulAct S'
                                 f : LinearMap σ₁₂ M M'
                                 x y : M
                                 ⊢ Eq (HSMul.hSMul a (⇑f) (HAdd.hAdd x y)) (HAdd.hAdd (HSMul.hSMul a (⇑f) x) (H …
                               -/
      map_add' := fun x y ↦ by simp only [DomMulAct.smul_apply, f.map_add, smul_add]
                               /-
                                 🎉 no goals
                               -/
                                /-
                                  R : Type u_1
                                  R' : Type u_2
                                  S : Type u_3
                                  M : Type u_4
                                  M' : Type u_5
                                  inst✝¹¹ : Semiring R
                                  inst✝¹⁰ : Semiring R'
                                  inst✝⁹ : AddCommMonoid M
                                  inst✝⁸ : AddCommMonoid M'
                                  inst✝⁷ : Module R M
                                  inst✝⁶ : Module R' M'
                                  σ₁₂ : RingHom R R'
                                  S' : Type u_6
                                  T' : Type u_7
                                  inst✝⁵ : Monoid S'
                                  inst✝⁴ : DistribMulAction S' M
                                  inst✝³ : SMulCommClass R S' M
                                  inst✝² : Monoid T'
                                  inst✝¹ : DistribMulAction T' M
                                  inst✝ : SMulCommClass R T' M
                                  a : DomMulAct S'
                                  f : LinearMap σ₁₂ M M'
                                  c : R
                                  x : M
                                  ⊢ Eq ({ toFun := HSMul.hSMul a ⇑f, map_add' := ⋯ }.toFun (HSMul.hSMul c x)) (H …
                                -/
      map_smul' := fun c x ↦ by simp_rw [DomMulAct.smul_apply, ← smul_comm, f.map_smulₛₗ] }
                                /-
                                  🎉 no goals
                                -/


theorem _root_.DomMulAct.smul_linearMap_apply (a : S'ᵈᵐᵃ) (f : M →ₛₗ[σ₁₂] M') (x : M) :
    (a • f) x = f (DomMulAct.mk.symm a • x) :=
  rfl


@[simp]
theorem _root_.DomMulAct.mk_smul_linearMap_apply (a : S') (f : M →ₛₗ[σ₁₂] M') (x : M) :
    (DomMulAct.mk a • f) x = f (a • x) :=
  rfl


theorem  _root_.DomMulAct.coe_smul_linearMap (a : S'ᵈᵐᵃ) (f : M →ₛₗ[σ₁₂] M') :
    (a • f : M →ₛₗ[σ₁₂] M') = a • (f : M → M') :=
  rfl


instance [SMulCommClass S' T' M] : SMulCommClass S'ᵈᵐᵃ T'ᵈᵐᵃ (M →ₛₗ[σ₁₂] M') :=
                              /-
                                R : Type u_1
                                R' : Type u_2
                                S : Type u_3
                                M : Type u_4
                                M' : Type u_5
                                inst✝¹² : Semiring R
                                inst✝¹¹ : Semiring R'
                                inst✝¹⁰ : AddCommMonoid M
                                inst✝⁹ : AddCommMonoid M'
                                inst✝⁸ : Module R M
                                inst✝⁷ : Module R' M'
                                σ₁₂ : RingHom R R'
                                S' : Type u_6
                                T' : Type u_7
                                inst✝⁶ : Monoid S'
                                inst✝⁵ : DistribMulAction S' M
                                inst✝⁴ : SMulCommClass R S' M
                                inst✝³ : Monoid T'
                                inst✝² : DistribMulAction T' M
                                inst✝¹ : SMulCommClass R T' M
                                inst✝ : SMulCommClass S' T' M
                                s : DomMulAct S'
                                t : DomMulAct T'
                                f : LinearMap σ₁₂ M M'
                                m : M
                                ⊢ Eq ((HSMul.hSMul s (HSMul.hSMul t f)) m) ((HSMul.hSMul t (HSMul.hSMul s f)) m)
                              -/
  ⟨fun s t f ↦ ext fun m ↦ by simp_rw [DomMulAct.smul_linearMap_apply, smul_comm]⟩
                              /-
                                🎉 no goals
                              -/


instance {S'} [Monoid S'] [DistribMulAction S' M] [SMulCommClass R S' M] :
    DistribMulAction S'ᵈᵐᵃ (M →ₛₗ[σ₁₂] M') where
  one_smul _ := ext fun _ ↦ congr_arg _ (one_smul _ _)
  mul_smul _ _ _ := ext fun _ ↦ congr_arg _ (mul_smul _ _ _)
  smul_add _ _ _ := ext fun _ ↦ rfl
  smul_zero _ := ext fun _ ↦ rfl


instance [NoZeroSMulDivisors S M'] : NoZeroSMulDivisors S (M →ₛₗ[σ₁₂] M') :=
  coe_injective.noZeroSMulDivisors _ rfl coe_smul


instance [SMulCommClass R S M] : Module Sᵈᵐᵃ (M →ₛₗ[σ₁₂] M') where
  add_smul _ _ _ := ext fun _ ↦ by
    /-
      R : Type u_1
      R' : Type u_2
      S : Type u_3
      M : Type u_4
      M' : Type u_5
      inst✝¹⁰ : Semiring R
      inst✝⁹ : Semiring R'
      inst✝⁸ : AddCommMonoid M
      inst✝⁷ : AddCommMonoid M'
      inst✝⁶ : Module R M
      inst✝⁵ : Module R' M'
      σ₁₂ : RingHom R R'
      inst✝⁴ : Semiring S
      inst✝³ : Module S M
      inst✝² : Module S M'
      inst✝¹ : SMulCommClass R' S M'
      inst✝ : SMulCommClass R S M
      x✝³ x✝² : DomMulAct S
      x✝¹ : LinearMap σ₁₂ M M'
      x✝ : M
      ⊢ Eq ((HSMul.hSMul (HAdd.hAdd x✝³ x✝²) x✝¹) x✝) ((HAdd.hAdd (HSMul.hSMul x✝³ x …
    -/
    simp_rw [add_apply, DomMulAct.smul_linearMap_apply, ← map_add, ← add_smul]; rfl
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
                                /-
                                  R : Type u_1
                                  R' : Type u_2
                                  S : Type u_3
                                  M : Type u_4
                                  M' : Type u_5
                                  inst✝¹⁰ : Semiring R
                                  inst✝⁹ : Semiring R'
                                  inst✝⁸ : AddCommMonoid M
                                  inst✝⁷ : AddCommMonoid M'
                                  inst✝⁶ : Module R M
                                  inst✝⁵ : Module R' M'
                                  σ₁₂ : RingHom R R'
                                  inst✝⁴ : Semiring S
                                  inst✝³ : Module S M
                                  inst✝² : Module S M'
                                  inst✝¹ : SMulCommClass R' S M'
                                  inst✝ : SMulCommClass R S M
                                  x✝¹ : LinearMap σ₁₂ M M'
                                  x✝ : M
                                  ⊢ Eq ((HSMul.hSMul 0 x✝¹) x✝) (0 x✝)
                                -/
  zero_smul _ := ext fun _ ↦ by erw [DomMulAct.smul_linearMap_apply, zero_smul, map_zero]; rfl
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/


