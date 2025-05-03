/-- If `f₁ f₂ : A →ₐ[R] B` are two lifts of the same `A →ₐ[R] B ⧸ I`,
  we may define a map `f₁ - f₂ : A →ₗ[R] I`. -/
def diffToIdealOfQuotientCompEq (f₁ f₂ : A →ₐ[R] B)
    (e : (Ideal.Quotient.mkₐ R I).comp f₁ = (Ideal.Quotient.mkₐ R I).comp f₂) : A →ₗ[R] I :=
  LinearMap.codRestrict (I.restrictScalars _) (f₁.toLinearMap - f₂.toLinearMap) (by
    /-
      R : Type u
      A : Type v
      B : Type w
      inst✝⁴ : CommSemiring R
      inst✝³ : CommSemiring A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      I : Ideal B
      f₁ f₂ : AlgHom R A B
      e : Eq ((Ideal.Quotient.mkₐ R I).comp f₁) ((Ideal.Quotient.mkₐ R I).comp f₂)
      ⊢ ∀ (c : A), Membership.mem (Submodule.restrictScalars R I) ((HSub.hSub f₁.toL …
    -/
    intro x
    /-
      R : Type u
      A : Type v
      B : Type w
      inst✝⁴ : CommSemiring R
      inst✝³ : CommSemiring A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      I : Ideal B
      f₁ f₂ : AlgHom R A B
      e : Eq ((Ideal.Quotient.mkₐ R I).comp f₁) ((Ideal.Quotient.mkₐ R I).comp f₂)
      x : A
      ⊢ Membership.mem (Submodule.restrictScalars R I) ((HSub.hSub f₁.toLinearMap f₂ …
    -/
    change f₁ x - f₂ x ∈ I
    /-
      R : Type u
      A : Type v
      B : Type w
      inst✝⁴ : CommSemiring R
      inst✝³ : CommSemiring A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      I : Ideal B
      f₁ f₂ : AlgHom R A B
      e : Eq ((Ideal.Quotient.mkₐ R I).comp f₁) ((Ideal.Quotient.mkₐ R I).comp f₂)
      x : A
      ⊢ Membership.mem I (HSub.hSub (f₁ x) (f₂ x))
    -/
    rw [← Ideal.Quotient.eq, ← Ideal.Quotient.mkₐ_eq_mk R, ← AlgHom.comp_apply, e]
    /-
      R : Type u
      A : Type v
      B : Type w
      inst✝⁴ : CommSemiring R
      inst✝³ : CommSemiring A
      inst✝² : CommRing B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      I : Ideal B
      f₁ f₂ : AlgHom R A B
      e : Eq ((Ideal.Quotient.mkₐ R I).comp f₁) ((Ideal.Quotient.mkₐ R I).comp f₂)
      x : A
      ⊢ Eq (((Ideal.Quotient.mkₐ R I).comp f₂) x) ((Ideal.Quotient.mkₐ R I) (f₂ x))
    -/
    rfl)
    /-
      🎉 no goals
    -/


@[simp]
theorem diffToIdealOfQuotientCompEq_apply (f₁ f₂ : A →ₐ[R] B)
    (e : (Ideal.Quotient.mkₐ R I).comp f₁ = (Ideal.Quotient.mkₐ R I).comp f₂) (x : A) :
    ((diffToIdealOfQuotientCompEq I f₁ f₂ e) x : B) = f₁ x - f₂ x :=
  rfl


/-- Given a tower of algebras `R → A → B`, and a square-zero `I : Ideal B`, each lift `A →ₐ[R] B`
of the canonical map `A →ₐ[R] B ⧸ I` corresponds to an `R`-derivation from `A` to `I`. -/
def derivationToSquareZeroOfLift [IsScalarTower R A B]  (hI : I ^ 2 = ⊥) (f : A →ₐ[R] B)
    (e : (Ideal.Quotient.mkₐ R I).comp f = IsScalarTower.toAlgHom R A (B ⧸ I)) :
    Derivation R A I := by
  refine
    { diffToIdealOfQuotientCompEq I f (IsScalarTower.toAlgHom R A B) ?_ with
      map_one_eq_zero' := ?_
      leibniz' := ?_ }
    /-
      case refine_1
      R : Type u
      A : Type v
      B : Type w
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommSemiring A
      inst✝⁴ : CommRing B
      inst✝³ : Algebra R A
      inst✝² : Algebra R B
      I : Ideal B
      inst✝¹ : Algebra A B
      inst✝ : IsScalarTower R A B
      hI : Eq (HPow.hPow I 2) Bot.bot
      f : AlgHom R A B
      e : Eq ((Ideal.Quotient.mkₐ R I).comp f) (IsScalarTower.toAlgHom R A (HasQuoti …
      ⊢ Eq ((Ideal.Quotient.mkₐ R I).comp f) ((Ideal.Quotient.mkₐ R I).comp (IsScala …
    -/
  · rw [e]; ext; rfl
                 /-
                   🎉 no goals
                 -/
    /-
      case refine_2
      R : Type u
      A : Type v
      B : Type w
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommSemiring A
      inst✝⁴ : CommRing B
      inst✝³ : Algebra R A
      inst✝² : Algebra R B
      I : Ideal B
      inst✝¹ : Algebra A B
      inst✝ : IsScalarTower R A B
      hI : Eq (HPow.hPow I 2) Bot.bot
      f : AlgHom R A B
      e : Eq ((Ideal.Quotient.mkₐ R I).comp f) (IsScalarTower.toAlgHom R A (HasQuoti …
      ⊢ Eq (__src✝ 1) 0
    -/
  · ext; change f 1 - algebraMap A B 1 = 0; rw [map_one, map_one, sub_self]
                                            /-
                                              🎉 no goals
                                            -/
    /-
      case refine_3
      R : Type u
      A : Type v
      B : Type w
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommSemiring A
      inst✝⁴ : CommRing B
      inst✝³ : Algebra R A
      inst✝² : Algebra R B
      I : Ideal B
      inst✝¹ : Algebra A B
      inst✝ : IsScalarTower R A B
      hI : Eq (HPow.hPow I 2) Bot.bot
      f : AlgHom R A B
      e : Eq ((Ideal.Quotient.mkₐ R I).comp f) (IsScalarTower.toAlgHom R A (HasQuoti …
      ⊢ ∀ (a b : A), Eq (__src✝ (HMul.hMul a b)) (HAdd.hAdd (HSMul.hSMul a (__src✝ b …
    -/
  · intro x y
    /-
      case refine_3
      R : Type u
      A : Type v
      B : Type w
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommSemiring A
      inst✝⁴ : CommRing B
      inst✝³ : Algebra R A
      inst✝² : Algebra R B
      I : Ideal B
      inst✝¹ : Algebra A B
      inst✝ : IsScalarTower R A B
      hI : Eq (HPow.hPow I 2) Bot.bot
      f : AlgHom R A B
      e : Eq ((Ideal.Quotient.mkₐ R I).comp f) (IsScalarTower.toAlgHom R A (HasQuoti …
      x y : A
      ⊢ Eq (__src✝ (HMul.hMul x y)) (HAdd.hAdd (HSMul.hSMul x (__src✝ y)) (HSMul.hSM …
    -/
    let F := diffToIdealOfQuotientCompEq I f (IsScalarTower.toAlgHom R A B) (by rw [e]; ext; rfl)
    have : (f x - algebraMap A B x) * (f y - algebraMap A B y) = 0 := by
      rw [← Ideal.mem_bot, ← hI, pow_two]
      convert Ideal.mul_mem_mul (F x).2 (F y).2 using 1
    /-
      case refine_3
      R : Type u
      A : Type v
      B : Type w
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommSemiring A
      inst✝⁴ : CommRing B
      inst✝³ : Algebra R A
      inst✝² : Algebra R B
      I : Ideal B
      inst✝¹ : Algebra A B
      inst✝ : IsScalarTower R A B
      hI : Eq (HPow.hPow I 2) Bot.bot
      f : AlgHom R A B
      e : Eq ((Ideal.Quotient.mkₐ R I).comp f) (IsScalarTower.toAlgHom R A (HasQuoti …
      x y : A
      F : LinearMap (RingHom.id R) A (Subtype fun x => Membership.mem I x) := diffTo …
      this : Eq (HMul.hMul (HSub.hSub (f x) ((algebraMap A B) x)) (HSub.hSub (f y) ( …
      ⊢ Eq (__src✝ (HMul.hMul x y)) (HAdd.hAdd (HSMul.hSMul x (__src✝ y)) (HSMul.hSM …
    -/
    ext
    dsimp only [Submodule.coe_add, Submodule.coe_mk, LinearMap.coe_mk,
      diffToIdealOfQuotientCompEq_apply, Submodule.coe_smul_of_tower, IsScalarTower.coe_toAlgHom',
      LinearMap.toFun_eq_coe]
    /-
      case refine_3.a
      R : Type u
      A : Type v
      B : Type w
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommSemiring A
      inst✝⁴ : CommRing B
      inst✝³ : Algebra R A
      inst✝² : Algebra R B
      I : Ideal B
      inst✝¹ : Algebra A B
      inst✝ : IsScalarTower R A B
      hI : Eq (HPow.hPow I 2) Bot.bot
      f : AlgHom R A B
      e : Eq ((Ideal.Quotient.mkₐ R I).comp f) (IsScalarTower.toAlgHom R A (HasQuoti …
      x y : A
      F : LinearMap (RingHom.id R) A (Subtype fun x => Membership.mem I x) := diffTo …
      this : Eq (HMul.hMul (HSub.hSub (f x) ((algebraMap A B) x)) (HSub.hSub (f y) ( …
      ⊢ Eq (HSub.hSub (f (HMul.hMul x y)) ((algebraMap A B) (HMul.hMul x y))) (HAdd. …
    -/
    simp only [map_mul, sub_mul, mul_sub, Algebra.smul_def] at this ⊢
    /-
      case refine_3.a
      R : Type u
      A : Type v
      B : Type w
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommSemiring A
      inst✝⁴ : CommRing B
      inst✝³ : Algebra R A
      inst✝² : Algebra R B
      I : Ideal B
      inst✝¹ : Algebra A B
      inst✝ : IsScalarTower R A B
      hI : Eq (HPow.hPow I 2) Bot.bot
      f : AlgHom R A B
      e : Eq ((Ideal.Quotient.mkₐ R I).comp f) (IsScalarTower.toAlgHom R A (HasQuoti …
      x y : A
      F : LinearMap (RingHom.id R) A (Subtype fun x => Membership.mem I x) := diffTo …
      this : Eq (HSub.hSub (HSub.hSub (HMul.hMul (f x) (f y)) (HMul.hMul ((algebraMa …
      ⊢ Eq (HSub.hSub (HMul.hMul (f x) (f y)) (HMul.hMul ((algebraMap A B) x) ((alge …
    -/
    rw [sub_eq_iff_eq_add, sub_eq_iff_eq_add] at this
    simp only [LinearMap.coe_toAddHom, diffToIdealOfQuotientCompEq_apply, map_mul, this,
      IsScalarTower.coe_toAlgHom']
    /-
      case refine_3.a
      R : Type u
      A : Type v
      B : Type w
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommSemiring A
      inst✝⁴ : CommRing B
      inst✝³ : Algebra R A
      inst✝² : Algebra R B
      I : Ideal B
      inst✝¹ : Algebra A B
      inst✝ : IsScalarTower R A B
      hI : Eq (HPow.hPow I 2) Bot.bot
      f : AlgHom R A B
      e : Eq ((Ideal.Quotient.mkₐ R I).comp f) (IsScalarTower.toAlgHom R A (HasQuoti …
      x y : A
      F : LinearMap (RingHom.id R) A (Subtype fun x => Membership.mem I x) := diffTo …
      this : Eq (HMul.hMul (f x) (f y)) (HAdd.hAdd (HAdd.hAdd 0 (HSub.hSub (HMul.hMu …
      ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd 0 (HSub.hSub (HMul.hMul (f x) ((algebraM …
    -/
    ring
    /-
      🎉 no goals
    -/


theorem derivationToSquareZeroOfLift_apply [IsScalarTower R A B] (f : A →ₐ[R] B)
    (e : (Ideal.Quotient.mkₐ R I).comp f = IsScalarTower.toAlgHom R A (B ⧸ I)) (x : A) :
    (derivationToSquareZeroOfLift I hI f e x : B) = f x - algebraMap A B x :=
  rfl


/-- Given a tower of algebras `R → A → B`, and a square-zero `I : Ideal B`, each `R`-derivation
from `A` to `I` corresponds to a lift `A →ₐ[R] B` of the canonical map `A →ₐ[R] B ⧸ I`. -/
@[simps (config := .lemmasOnly)]
def liftOfDerivationToSquareZero [IsScalarTower R A B]  (hI : I ^ 2 = ⊥) (f : Derivation R A I) :
    A →ₐ[R] B :=
  { ((I.restrictScalars R).subtype.comp f.toLinearMap + (IsScalarTower.toAlgHom R A B).toLinearMap :
      A →ₗ[R] B) with
    toFun := fun x => f x + algebraMap A B x
    map_one' := by
      /-
        R : Type u
        A : Type v
        B : Type w
        inst✝⁶ : CommSemiring R
        inst✝⁵ : CommSemiring A
        inst✝⁴ : CommRing B
        inst✝³ : Algebra R A
        inst✝² : Algebra R B
        I : Ideal B
        inst✝¹ : Algebra A B
        hI✝ : Eq (HPow.hPow I 2) Bot.bot
        inst✝ : IsScalarTower R A B
        hI : Eq (HPow.hPow I 2) Bot.bot
        f : Derivation R A (Subtype fun x => Membership.mem I x)
        ⊢ Eq ((fun x => HAdd.hAdd (↑(f x)) ((algebraMap A B) x)) 1) 1
      -/
      dsimp
      -- Note: added the `(algebraMap _ _)` hint because otherwise it would match `f 1`
      /-
        R : Type u
        A : Type v
        B : Type w
        inst✝⁶ : CommSemiring R
        inst✝⁵ : CommSemiring A
        inst✝⁴ : CommRing B
        inst✝³ : Algebra R A
        inst✝² : Algebra R B
        I : Ideal B
        inst✝¹ : Algebra A B
        hI✝ : Eq (HPow.hPow I 2) Bot.bot
        inst✝ : IsScalarTower R A B
        hI : Eq (HPow.hPow I 2) Bot.bot
        f : Derivation R A (Subtype fun x => Membership.mem I x)
        ⊢ Eq (HAdd.hAdd (↑(f 1)) ((algebraMap A B) 1)) 1
      -/
      rw [map_one (algebraMap _ _), f.map_one_eq_zero, Submodule.coe_zero, zero_add]
      /-
        🎉 no goals
      -/
    map_mul' := fun x y => by
      have : (f x : B) * f y = 0 := by
        rw [← Ideal.mem_bot, ← hI, pow_two]
        convert Ideal.mul_mem_mul (f x).2 (f y).2 using 1
      simp only [map_mul, f.leibniz, add_mul, mul_add, Submodule.coe_add,
        Submodule.coe_smul_of_tower, Algebra.smul_def, this]
      /-
        R : Type u
        A : Type v
        B : Type w
        inst✝⁶ : CommSemiring R
        inst✝⁵ : CommSemiring A
        inst✝⁴ : CommRing B
        inst✝³ : Algebra R A
        inst✝² : Algebra R B
        I : Ideal B
        inst✝¹ : Algebra A B
        hI✝ : Eq (HPow.hPow I 2) Bot.bot
        inst✝ : IsScalarTower R A B
        hI : Eq (HPow.hPow I 2) Bot.bot
        f : Derivation R A (Subtype fun x => Membership.mem I x)
        x y : A
        this : Eq (HMul.hMul ↑(f x) ↑(f y)) 0
        ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul ((algebraMap A B) x) ↑(f y)) (HMul.hMul  …
      -/
      ring
      /-
        🎉 no goals
      -/
    commutes' := fun r => by
      simp only [Derivation.map_algebraMap, eq_self_iff_true, zero_add, Submodule.coe_zero, ←
        IsScalarTower.algebraMap_apply R A B r]
    map_zero' := ((I.restrictScalars R).subtype.comp f.toLinearMap +
      (IsScalarTower.toAlgHom R A B).toLinearMap).map_zero }

-- @[simp] -- Porting note: simp normal form is `liftOfDerivationToSquareZero_mk_apply'`

theorem liftOfDerivationToSquareZero_mk_apply [IsScalarTower R A B] (d : Derivation R A I) (x : A) :
    Ideal.Quotient.mk I (liftOfDerivationToSquareZero I hI d x) = algebraMap A (B ⧸ I) x := by
  rw [liftOfDerivationToSquareZero_apply, map_add, Ideal.Quotient.eq_zero_iff_mem.mpr (d x).prop,
    zero_add]
  /-
    R : Type u
    A : Type v
    B : Type w
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommSemiring A
    inst✝⁴ : CommRing B
    inst✝³ : Algebra R A
    inst✝² : Algebra R B
    I : Ideal B
    inst✝¹ : Algebra A B
    hI : Eq (HPow.hPow I 2) Bot.bot
    inst✝ : IsScalarTower R A B
    d : Derivation R A (Subtype fun x => Membership.mem I x)
    x : A
    ⊢ Eq ((Ideal.Quotient.mk I) ((algebraMap A B) x)) ((algebraMap A (HasQuotient. …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem liftOfDerivationToSquareZero_mk_apply' (d : Derivation R A I) (x : A) :
    (Ideal.Quotient.mk I) (d x) + (algebraMap A (B ⧸ I)) x = algebraMap A (B ⧸ I) x := by
  /-
    R : Type u
    A : Type v
    B : Type w
    inst✝⁵ : CommSemiring R
    inst✝⁴ : CommSemiring A
    inst✝³ : CommRing B
    inst✝² : Algebra R A
    inst✝¹ : Algebra R B
    I : Ideal B
    inst✝ : Algebra A B
    d : Derivation R A (Subtype fun x => Membership.mem I x)
    x : A
    ⊢ Eq (HAdd.hAdd ((Ideal.Quotient.mk I) ↑(d x)) ((algebraMap A (HasQuotient.Quo …
  -/
  simp only [Ideal.Quotient.eq_zero_iff_mem.mpr (d x).prop, zero_add]
  /-
    🎉 no goals
  -/


/-- Given a tower of algebras `R → A → B`, and a square-zero `I : Ideal B`,
there is a 1-1 correspondence between `R`-derivations from `A` to `I` and
lifts `A →ₐ[R] B` of the canonical map `A →ₐ[R] B ⧸ I`. -/
@[simps!]
def derivationToSquareZeroEquivLift [IsScalarTower R A B] : Derivation R A I ≃
    { f : A →ₐ[R] B // (Ideal.Quotient.mkₐ R I).comp f = IsScalarTower.toAlgHom R A (B ⧸ I) } := by
  refine ⟨fun d => ⟨liftOfDerivationToSquareZero I hI d, ?_⟩, fun f =>
    (derivationToSquareZeroOfLift I hI f.1 f.2 : _), ?_, ?_⟩
    /-
      case refine_1
      R : Type u
      A : Type v
      B : Type w
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommSemiring A
      inst✝⁴ : CommRing B
      inst✝³ : Algebra R A
      inst✝² : Algebra R B
      I : Ideal B
      inst✝¹ : Algebra A B
      hI : Eq (HPow.hPow I 2) Bot.bot
      inst✝ : IsScalarTower R A B
      d : Derivation R A (Subtype fun x => Membership.mem I x)
      ⊢ Eq ((Ideal.Quotient.mkₐ R I).comp (liftOfDerivationToSquareZero I hI d)) (Is …
    -/
  · ext x; exact liftOfDerivationToSquareZero_mk_apply I hI d x
           /-
             🎉 no goals
           -/
    /-
      case refine_2
      R : Type u
      A : Type v
      B : Type w
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommSemiring A
      inst✝⁴ : CommRing B
      inst✝³ : Algebra R A
      inst✝² : Algebra R B
      I : Ideal B
      inst✝¹ : Algebra A B
      hI : Eq (HPow.hPow I 2) Bot.bot
      inst✝ : IsScalarTower R A B
      ⊢ Function.LeftInverse (fun f => derivationToSquareZeroOfLift I hI ↑f ⋯) fun d …
    -/
  · intro d; ext x; exact add_sub_cancel_right (d x : B) (algebraMap A B x)
                    /-
                      🎉 no goals
                    -/
    /-
      case refine_3
      R : Type u
      A : Type v
      B : Type w
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommSemiring A
      inst✝⁴ : CommRing B
      inst✝³ : Algebra R A
      inst✝² : Algebra R B
      I : Ideal B
      inst✝¹ : Algebra A B
      hI : Eq (HPow.hPow I 2) Bot.bot
      inst✝ : IsScalarTower R A B
      ⊢ Function.RightInverse (fun f => derivationToSquareZeroOfLift I hI ↑f ⋯) fun  …
    -/
  · rintro ⟨f, hf⟩; ext x; exact sub_add_cancel (f x) (algebraMap A B x)
                           /-
                             🎉 no goals
                           -/


