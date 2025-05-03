/-- The equivalence relation associated to a submodule `p`, defined by `x ≈ y` iff `-x + y ∈ p`.

Note this is equivalent to `y - x ∈ p`, but defined this way to be defeq to the `AddSubgroup`
version, where commutativity can't be assumed. -/
def quotientRel : Setoid M :=
  QuotientAddGroup.leftRel p.toAddSubgroup


theorem quotientRel_def {x y : M} : p.quotientRel x y ↔ x - y ∈ p :=
  Iff.trans
    (by
      /-
        R : Type u_1
        M : Type u_2
        inst✝² : Ring R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        p : Submodule R M
        x y : M
        ⊢ Iff (p.quotientRel x y) (Membership.mem p (Neg.neg (HSub.hSub x y)))
      -/
      rw [leftRel_apply, sub_eq_add_neg, neg_add, neg_neg]
      /-
        R : Type u_1
        M : Type u_2
        inst✝² : Ring R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        p : Submodule R M
        x y : M
        ⊢ Iff (Membership.mem p.toAddSubgroup (HAdd.hAdd (Neg.neg x) y)) (Membership.m …
      -/
      rfl)
      /-
        🎉 no goals
      -/
    neg_mem_iff


@[deprecated (since := "2024-08-29")] alias quotientRel_r_def := quotientRel_def


/-- The quotient of a module `M` by a submodule `p ⊆ M`. -/
instance hasQuotient : HasQuotient M (Submodule R M) :=
  ⟨fun p => Quotient (quotientRel p)⟩


/-- Map associating to an element of `M` the corresponding element of `M/p`,
when `p` is a submodule of `M`. -/
def mk {p : Submodule R M} : M → M ⧸ p :=
  Quotient.mk''


theorem mk'_eq_mk' {p : Submodule R M} (x : M) :
    @Quotient.mk' _ (quotientRel p) x = mk x :=
  rfl


theorem mk''_eq_mk {p : Submodule R M} (x : M) : (Quotient.mk'' x : M ⧸ p) = mk x :=
  rfl


theorem quot_mk_eq_mk {p : Submodule R M} (x : M) : (Quot.mk _ x : M ⧸ p) = mk x :=
  rfl


protected theorem eq' {x y : M} : (mk x : M ⧸ p) = mk y ↔ -x + y ∈ p :=
  QuotientAddGroup.eq


protected theorem eq {x y : M} : (mk x : M ⧸ p) = mk y ↔ x - y ∈ p :=
  (Submodule.Quotient.eq' p).trans (leftRel_apply.symm.trans p.quotientRel_def)


instance : Zero (M ⧸ p) where
  -- Use Quotient.mk'' instead of mk here because mk is not reducible.
  -- This would lead to non-defeq diamonds.
  -- See also the same comment at the One instance for Con.
  zero := Quotient.mk'' 0


instance : Inhabited (M ⧸ p) :=
  ⟨0⟩


@[simp]
theorem mk_zero : mk 0 = (0 : M ⧸ p) :=
  rfl


@[simp]
                                                      /-
                                                        R : Type u_1
                                                        M : Type u_2
                                                        x : M
                                                        inst✝² : Ring R
                                                        inst✝¹ : AddCommGroup M
                                                        inst✝ : Module R M
                                                        p : Submodule R M
                                                        ⊢ Iff (Eq (Submodule.Quotient.mk x) 0) (Membership.mem p x)
                                                      -/
theorem mk_eq_zero : (mk x : M ⧸ p) = 0 ↔ x ∈ p := by simpa using (Quotient.eq' p : mk x = 0 ↔ _)
                                                      /-
                                                        🎉 no goals
                                                      -/


instance addCommGroup : AddCommGroup (M ⧸ p) :=
  QuotientAddGroup.Quotient.addCommGroup p.toAddSubgroup


@[simp]
theorem mk_add : (mk (x + y) : M ⧸ p) = mk x + mk y :=
  rfl


@[simp]
theorem mk_neg : (mk (-x) : M ⧸ p) = -(mk x) :=
  rfl


@[simp]
theorem mk_sub : (mk (x - y) : M ⧸ p) = mk x - mk y :=
  rfl


protected nonrec lemma «forall» {P : M ⧸ p → Prop} : (∀ a, P a) ↔ ∀ a, P (mk a) := Quotient.forall


instance instSMul' : SMul S (M ⧸ P) :=
  ⟨fun a =>
    Quotient.map' (a • ·) fun x y h =>
                              /-
                                R : Type u_1
                                M : Type u_2
                                r : R
                                x✝ y✝ : M
                                inst✝⁵ : Ring R
                                inst✝⁴ : AddCommGroup M
                                inst✝³ : Module R M
                                p p' : Submodule R M
                                S : Type u_3
                                inst✝² : SMul S R
                                inst✝¹ : SMul S M
                                inst✝ : IsScalarTower S R M
                                P : Submodule R M
                                a : S
                                x y : M
                                h : P.quotientRel x y
                                ⊢ Membership.mem P.toAddSubgroup (HAdd.hAdd (Neg.neg ((fun x => HSMul.hSMul a  …
                              -/
      leftRel_apply.mpr <| by simpa using Submodule.smul_mem P (a • (1 : R)) (leftRel_apply.mp h)⟩
                              /-
                                🎉 no goals
                              -/

-- Porting note: should this be marked as a `@[default_instance]`?

/-- Shortcut to help the elaborator in the common case. -/
instance instSMul : SMul R (M ⧸ P) :=
  Quotient.instSMul' P


@[simp]
theorem mk_smul (r : S) (x : M) : (mk (r • x) : M ⧸ p) = r • mk x :=
  rfl


instance smulCommClass (T : Type*) [SMul T R] [SMul T M] [IsScalarTower T R M]
    [SMulCommClass S T M] : SMulCommClass S T (M ⧸ P) where
  smul_comm _x _y := Quotient.ind' fun _z => congr_arg mk (smul_comm _ _ _)


instance isScalarTower (T : Type*) [SMul T R] [SMul T M] [IsScalarTower T R M] [SMul S T]
    [IsScalarTower S T M] : IsScalarTower S T (M ⧸ P) where
  smul_assoc _x _y := Quotient.ind' fun _z => congr_arg mk (smul_assoc _ _ _)


instance isCentralScalar [SMul Sᵐᵒᵖ R] [SMul Sᵐᵒᵖ M] [IsScalarTower Sᵐᵒᵖ R M]
    [IsCentralScalar S M] : IsCentralScalar S (M ⧸ P) where
  op_smul_eq_smul _x := Quotient.ind' fun _z => congr_arg mk <| op_smul_eq_smul _ _


instance mulAction' [Monoid S] [SMul S R] [MulAction S M] [IsScalarTower S R M]
    (P : Submodule R M) : MulAction S (M ⧸ P) :=
  { Function.Surjective.mulAction mk Quot.mk_surjective <| Submodule.Quotient.mk_smul P with
    toSMul := instSMul' _ }

-- Porting note: should this be marked as a `@[default_instance]`?

instance mulAction (P : Submodule R M) : MulAction R (M ⧸ P) :=
  Quotient.mulAction' P


instance smulZeroClass' [SMul S R] [SMulZeroClass S M] [IsScalarTower S R M] (P : Submodule R M) :
    SMulZeroClass S (M ⧸ P) :=
  ZeroHom.smulZeroClass ⟨mk, mk_zero _⟩ <| Submodule.Quotient.mk_smul P

-- Porting note: should this be marked as a `@[default_instance]`?

instance smulZeroClass (P : Submodule R M) : SMulZeroClass R (M ⧸ P) :=
  Quotient.smulZeroClass' P

-- Performance of `Function.Surjective.distribSMul` is worse since it has to unify data to apply
-- TODO: https://github.com/leanprover-community/mathlib4/pull/7432

instance distribSMul' [SMul S R] [DistribSMul S M] [IsScalarTower S R M] (P : Submodule R M) :
    DistribSMul S (M ⧸ P) :=
  { Function.Surjective.distribSMul {toFun := mk, map_zero' := rfl, map_add' := fun _ _ => rfl}
    Quot.mk_surjective (Submodule.Quotient.mk_smul P) with
    toSMulZeroClass := smulZeroClass' _ }

-- Porting note: should this be marked as a `@[default_instance]`?

instance distribSMul (P : Submodule R M) : DistribSMul R (M ⧸ P) :=
  Quotient.distribSMul' P

-- Performance of `Function.Surjective.distribMulAction` is worse since it has to unify data
-- TODO: https://github.com/leanprover-community/mathlib4/pull/7432

instance distribMulAction' [Monoid S] [SMul S R] [DistribMulAction S M] [IsScalarTower S R M]
    (P : Submodule R M) : DistribMulAction S (M ⧸ P) :=
  { Function.Surjective.distribMulAction {toFun := mk, map_zero' := rfl, map_add' := fun _ _ => rfl}
    Quot.mk_surjective (Submodule.Quotient.mk_smul P) with
    toMulAction := mulAction' _ }

-- Porting note: should this be marked as a `@[default_instance]`?

instance distribMulAction (P : Submodule R M) : DistribMulAction R (M ⧸ P) :=
  Quotient.distribMulAction' P

-- Performance of `Function.Surjective.module` is worse since it has to unify data to apply
-- TODO: https://github.com/leanprover-community/mathlib4/pull/7432

instance module' [Semiring S] [SMul S R] [Module S M] [IsScalarTower S R M] (P : Submodule R M) :
    Module S (M ⧸ P) :=
                                                               /-
                                                                 R : Type u_1
                                                                 M : Type u_2
                                                                 r : R
                                                                 x y : M
                                                                 inst✝⁶ : Ring R
                                                                 inst✝⁵ : AddCommGroup M
                                                                 inst✝⁴ : Module R M
                                                                 p p' : Submodule R M
                                                                 S : Type u_3
                                                                 inst✝³ : Semiring S
                                                                 inst✝² : SMul S R
                                                                 inst✝¹ : Module S M
                                                                 inst✝ : IsScalarTower S R M
                                                                 P : Submodule R M
                                                                 ⊢ Eq (Submodule.Quotient.mk 0) 0
                                                               -/
                                                               /-
                                                                 🎉 no goals
                                                               -/
  { Function.Surjective.module _ {toFun := mk, map_zero' := by rfl, map_add' := fun _ _ => by rfl}
                                                                                              /-
                                                                                                🎉 no goals
                                                                                              -/
    Quot.mk_surjective (Submodule.Quotient.mk_smul P) with
    toDistribMulAction := distribMulAction' _ }

-- Porting note: should this be marked as a `@[default_instance]`?

instance module (P : Submodule R M) : Module R (M ⧸ P) :=
  Quotient.module' P


@[elab_as_elim]
theorem induction_on {C : M ⧸ p → Prop} (x : M ⧸ p) (H : ∀ z, C (Submodule.Quotient.mk z)) :
    C x := Quotient.inductionOn' x H


theorem mk_surjective : Function.Surjective (@mk _ _ _ _ _ p) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p : Submodule R M
    ⊢ Function.Surjective Submodule.Quotient.mk
  -/
  rintro ⟨x⟩
  /-
    case mk
    R : Type u_1
    M : Type u_2
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p : Submodule R M
    b✝ : HasQuotient.Quotient M p
    x : M
    ⊢ Exists fun a => Eq (Submodule.Quotient.mk a) (Quot.mk (⇑p.quotientRel) x)
  -/
  exact ⟨x, rfl⟩
  /-
    🎉 no goals
  -/


theorem quot_hom_ext (f g : (M ⧸ p) →ₗ[R] M₂) (h : ∀ x : M, f (Quotient.mk x) = g (Quotient.mk x)) :
    f = g :=
  LinearMap.ext fun x => Submodule.Quotient.induction_on _ x h


/-- The map from a module `M` to the quotient of `M` by a submodule `p` as a linear map. -/
def mkQ : M →ₗ[R] M ⧸ p where
  toFun := Quotient.mk
                 /-
                   R : Type u_1
                   M : Type u_2
                   r : R
                   x y : M
                   inst✝⁴ : Ring R
                   inst✝³ : AddCommGroup M
                   inst✝² : Module R M
                   p p' : Submodule R M
                   M₂ : Type u_3
                   inst✝¹ : AddCommGroup M₂
                   inst✝ : Module R M₂
                   ⊢ ∀ (x y : M), Eq (Submodule.Quotient.mk (HAdd.hAdd x y)) (HAdd.hAdd (Submodul …
                 -/
  map_add' := by simp
                 /-
                   🎉 no goals
                 -/
                  /-
                    R : Type u_1
                    M : Type u_2
                    r : R
                    x y : M
                    inst✝⁴ : Ring R
                    inst✝³ : AddCommGroup M
                    inst✝² : Module R M
                    p p' : Submodule R M
                    M₂ : Type u_3
                    inst✝¹ : AddCommGroup M₂
                    inst✝ : Module R M₂
                    ⊢ ∀ (m : R) (x : M), Eq ({ toFun := Submodule.Quotient.mk, map_add' := ⋯ }.toF …
                  -/
  map_smul' := by simp
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem mkQ_apply (x : M) : p.mkQ x = Quotient.mk x :=
  rfl


theorem mkQ_surjective : Function.Surjective p.mkQ := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p : Submodule R M
    ⊢ Function.Surjective ⇑p.mkQ
  -/
  rintro ⟨x⟩; exact ⟨x, rfl⟩
              /-
                🎉 no goals
              -/


/-- Two `LinearMap`s from a quotient module are equal if their compositions with
`submodule.mkQ` are equal.

See note [partially-applied ext lemmas]. -/
@[ext 1100] -- Porting note: increase priority so this applies before `LinearMap.ext`
theorem linearMap_qext ⦃f g : M ⧸ p →ₛₗ[τ₁₂] M₂⦄ (h : f.comp p.mkQ = g.comp p.mkQ) : f = g :=
  LinearMap.ext fun x => Submodule.Quotient.induction_on _ x <| (LinearMap.congr_fun h : _)


/-- Quotienting by equal submodules gives linearly equivalent quotients. -/
def quotEquivOfEq (h : p = p') : (M ⧸ p) ≃ₗ[R] M ⧸ p' :=
  { @Quotient.congr _ _ (quotientRel p) (quotientRel p') (Equiv.refl _) fun a b => by
      /-
        R : Type u_1
        M : Type u_2
        r : R
        x y : M
        inst✝⁵ : Ring R
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        p p' : Submodule R M
        R₂ : Type u_3
        M₂ : Type u_4
        inst✝² : Ring R₂
        inst✝¹ : AddCommGroup M₂
        inst✝ : Module R₂ M₂
        τ₁₂ : RingHom R R₂
        h : Eq p p'
        a b : M
        ⊢ Iff (p.quotientRel a b) (p'.quotientRel ((Equiv.refl M) a) ((Equiv.refl M) b))
      -/
      subst h
      /-
        R : Type u_1
        M : Type u_2
        r : R
        x y : M
        inst✝⁵ : Ring R
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        p : Submodule R M
        R₂ : Type u_3
        M₂ : Type u_4
        inst✝² : Ring R₂
        inst✝¹ : AddCommGroup M₂
        inst✝ : Module R₂ M₂
        τ₁₂ : RingHom R R₂
        a b : M
        ⊢ Iff (p.quotientRel a b) (p.quotientRel ((Equiv.refl M) a) ((Equiv.refl M) b))
      -/
      rfl with
      /-
        🎉 no goals
      -/
    map_add' := by
      /-
        R : Type u_1
        M : Type u_2
        r : R
        x y : M
        inst✝⁵ : Ring R
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        p p' : Submodule R M
        R₂ : Type u_3
        M₂ : Type u_4
        inst✝² : Ring R₂
        inst✝¹ : AddCommGroup M₂
        inst✝ : Module R₂ M₂
        τ₁₂ : RingHom R R₂
        h : Eq p p'
        ⊢ ∀ (x y : HasQuotient.Quotient M p), Eq (__src✝.toFun (HAdd.hAdd x y)) (HAdd. …
      -/
      rintro ⟨x⟩ ⟨y⟩
      /-
        case mk.mk
        R : Type u_1
        M : Type u_2
        r : R
        x✝¹ y✝¹ : M
        inst✝⁵ : Ring R
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        p p' : Submodule R M
        R₂ : Type u_3
        M₂ : Type u_4
        inst✝² : Ring R₂
        inst✝¹ : AddCommGroup M₂
        inst✝ : Module R₂ M₂
        τ₁₂ : RingHom R R₂
        h : Eq p p'
        x✝ : HasQuotient.Quotient M p
        x : M
        y✝ : HasQuotient.Quotient M p
        y : M
        ⊢ Eq (__src✝.toFun (HAdd.hAdd (Quot.mk (⇑p.quotientRel) x) (Quot.mk (⇑p.quotie …
      -/
      rfl
      /-
        🎉 no goals
      -/
    map_smul' := by
      /-
        R : Type u_1
        M : Type u_2
        r : R
        x y : M
        inst✝⁵ : Ring R
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        p p' : Submodule R M
        R₂ : Type u_3
        M₂ : Type u_4
        inst✝² : Ring R₂
        inst✝¹ : AddCommGroup M₂
        inst✝ : Module R₂ M₂
        τ₁₂ : RingHom R R₂
        h : Eq p p'
        ⊢ ∀ (m : R) (x : HasQuotient.Quotient M p), Eq ({ toFun := __src✝.toFun, map_a …
      -/
      rintro x ⟨y⟩
      /-
        case mk
        R : Type u_1
        M : Type u_2
        r : R
        x✝¹ y✝ : M
        inst✝⁵ : Ring R
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        p p' : Submodule R M
        R₂ : Type u_3
        M₂ : Type u_4
        inst✝² : Ring R₂
        inst✝¹ : AddCommGroup M₂
        inst✝ : Module R₂ M₂
        τ₁₂ : RingHom R R₂
        h : Eq p p'
        x : R
        x✝ : HasQuotient.Quotient M p
        y : M
        ⊢ Eq ({ toFun := __src✝.toFun, map_add' := ⋯ }.toFun (HSMul.hSMul x (Quot.mk ( …
      -/
      rfl }
      /-
        🎉 no goals
      -/


@[simp]
theorem quotEquivOfEq_mk (h : p = p') (x : M) :
    Submodule.quotEquivOfEq p p' h (Submodule.Quotient.mk x) =
      (Submodule.Quotient.mk x) :=
  rfl


