@[simp]
theorem star_natCast_smul [Semiring R] [AddCommMonoid M] [Module R M] [StarAddMonoid M] (n : ℕ)
    (x : M) : star ((n : R) • x) = (n : R) • star x :=
  map_natCast_smul (starAddEquiv : M ≃+ M) R R n x


@[deprecated (since := "2024-04-17")]
alias star_nat_cast_smul := star_natCast_smul


@[simp]
theorem star_intCast_smul [Ring R] [AddCommGroup M] [Module R M] [StarAddMonoid M] (n : ℤ)
    (x : M) : star ((n : R) • x) = (n : R) • star x :=
  map_intCast_smul (starAddEquiv : M ≃+ M) R R n x


@[deprecated (since := "2024-04-17")]
alias star_int_cast_smul := star_intCast_smul


@[simp]
theorem star_inv_natCast_smul [DivisionSemiring R] [AddCommMonoid M] [Module R M] [StarAddMonoid M]
    (n : ℕ) (x : M) : star ((n⁻¹ : R) • x) = (n⁻¹ : R) • star x :=
  map_inv_natCast_smul (starAddEquiv : M ≃+ M) R R n x


@[deprecated (since := "2024-04-17")]
alias star_inv_nat_cast_smul := star_inv_natCast_smul


@[simp]
theorem star_inv_intCast_smul [DivisionRing R] [AddCommGroup M] [Module R M] [StarAddMonoid M]
    (n : ℤ) (x : M) : star ((n⁻¹ : R) • x) = (n⁻¹ : R) • star x :=
  map_inv_intCast_smul (starAddEquiv : M ≃+ M) R R n x


@[deprecated (since := "2024-04-17")]
alias star_inv_int_cast_smul := star_inv_intCast_smul


@[simp]
theorem star_ratCast_smul [DivisionRing R] [AddCommGroup M] [Module R M] [StarAddMonoid M] (n : ℚ)
    (x : M) : star ((n : R) • x) = (n : R) • star x :=
  map_ratCast_smul (starAddEquiv : M ≃+ M) _ _ _ x


@[deprecated (since := "2024-04-17")]
alias star_rat_cast_smul := star_ratCast_smul


/-- Note that this lemma holds for an arbitrary `ℚ≥0`-action, rather than merely one coming from a
`DivisionSemiring`. We keep both the `nnqsmul` and `nnrat_smul` naming conventions for
discoverability. See `star_nnqsmul`. -/
@[simp high]
lemma star_nnrat_smul [AddCommMonoid R] [StarAddMonoid R] [Module ℚ≥0 R] (q : ℚ≥0) (x : R) :
    star (q • x) = q • star x := map_nnrat_smul (starAddEquiv : R ≃+ R) _ _


/-- Note that this lemma holds for an arbitrary `ℚ`-action, rather than merely one coming from a
`DivisionRing`. We keep both the `qsmul` and `rat_smul` naming conventions for discoverability.
See `star_qsmul`. -/
@[simp high] lemma star_rat_smul [AddCommGroup R] [StarAddMonoid R] [Module ℚ R] (q : ℚ) (x : R) :
    star (q • x) = q • star x :=
  map_rat_smul (starAddEquiv : R ≃+ R) _ _


/-- Note that this lemma holds for an arbitrary `ℚ≥0`-action, rather than merely one coming from a
`DivisionSemiring`. We keep both the `nnqsmul` and `nnrat_smul` naming conventions for
discoverability. See `star_nnrat_smul`. -/
alias star_nnqsmul := star_nnrat_smul


/-- Note that this lemma holds for an arbitrary `ℚ`-action, rather than merely one coming from a
`DivisionRing`. We keep both the `qsmul` and `rat_smul` naming conventions for
discoverability. See `star_rat_smul`. -/
alias star_qsmul := star_rat_smul


instance StarAddMonoid.toStarModuleNNRat [AddCommMonoid R] [Module ℚ≥0 R] [StarAddMonoid R] :
    StarModule ℚ≥0 R where star_smul := star_nnrat_smul


instance StarAddMonoid.toStarModuleRat [AddCommGroup R] [Module ℚ R] [StarAddMonoid R] :
    StarModule ℚ R where star_smul := star_rat_smul


/-- If `A` is a module over a commutative `R` with compatible actions,
then `star` is a semilinear equivalence. -/
@[simps]
def starLinearEquiv (R : Type*) {A : Type*} [CommSemiring R] [StarRing R] [AddCommMonoid A]
    [StarAddMonoid A] [Module R A] [StarModule R A] : A ≃ₗ⋆[R] A :=
  { starAddEquiv with
    toFun := star
    map_smul' := star_smul }


/-- The self-adjoint elements of a star module, as a submodule. -/
def selfAdjoint.submodule : Submodule R A :=
  { selfAdjoint A with smul_mem' := fun _ _ => (IsSelfAdjoint.all _).smul }


/-- The skew-adjoint elements of a star module, as a submodule. -/
def skewAdjoint.submodule : Submodule R A :=
  { skewAdjoint A with smul_mem' := skewAdjoint.smul_mem }


/-- The self-adjoint part of an element of a star module, as a linear map. -/
@[simps]
def selfAdjointPart : A →ₗ[R] selfAdjoint A where
  toFun x :=
    ⟨(⅟ 2 : R) • (x + star x), by
      /-
        R : Type u_1
        A : Type u_2
        inst✝⁷ : Semiring R
        inst✝⁶ : StarMul R
        inst✝⁵ : TrivialStar R
        inst✝⁴ : AddCommGroup A
        inst✝³ : Module R A
        inst✝² : StarAddMonoid A
        inst✝¹ : StarModule R A
        inst✝ : Invertible 2
        x : A
        ⊢ Membership.mem (selfAdjoint A) (HSMul.hSMul (Invertible.invOf 2) (HAdd.hAdd  …
      -/
      rw [selfAdjoint.mem_iff, star_smul, star_trivial, star_add, star_star, add_comm]⟩
      /-
        🎉 no goals
      -/
  map_add' x y := by
    /-
      R : Type u_1
      A : Type u_2
      inst✝⁷ : Semiring R
      inst✝⁶ : StarMul R
      inst✝⁵ : TrivialStar R
      inst✝⁴ : AddCommGroup A
      inst✝³ : Module R A
      inst✝² : StarAddMonoid A
      inst✝¹ : StarModule R A
      inst✝ : Invertible 2
      x y : A
      ⊢ Eq ((fun x => ⟨HSMul.hSMul (Invertible.invOf 2) (HAdd.hAdd x (Star.star x)), …
    -/
    ext
    /-
      case a
      R : Type u_1
      A : Type u_2
      inst✝⁷ : Semiring R
      inst✝⁶ : StarMul R
      inst✝⁵ : TrivialStar R
      inst✝⁴ : AddCommGroup A
      inst✝³ : Module R A
      inst✝² : StarAddMonoid A
      inst✝¹ : StarModule R A
      inst✝ : Invertible 2
      x y : A
      ⊢ Eq ↑((fun x => ⟨HSMul.hSMul (Invertible.invOf 2) (HAdd.hAdd x (Star.star x)) …
    -/
    simp [add_add_add_comm]
    /-
      🎉 no goals
    -/
  map_smul' r x := by
    /-
      R : Type u_1
      A : Type u_2
      inst✝⁷ : Semiring R
      inst✝⁶ : StarMul R
      inst✝⁵ : TrivialStar R
      inst✝⁴ : AddCommGroup A
      inst✝³ : Module R A
      inst✝² : StarAddMonoid A
      inst✝¹ : StarModule R A
      inst✝ : Invertible 2
      r : R
      x : A
      ⊢ Eq ({ toFun := fun x => ⟨HSMul.hSMul (Invertible.invOf 2) (HAdd.hAdd x (Star …
    -/
    ext
    /-
      case a
      R : Type u_1
      A : Type u_2
      inst✝⁷ : Semiring R
      inst✝⁶ : StarMul R
      inst✝⁵ : TrivialStar R
      inst✝⁴ : AddCommGroup A
      inst✝³ : Module R A
      inst✝² : StarAddMonoid A
      inst✝¹ : StarModule R A
      inst✝ : Invertible 2
      r : R
      x : A
      ⊢ Eq ↑({ toFun := fun x => ⟨HSMul.hSMul (Invertible.invOf 2) (HAdd.hAdd x (Sta …
    -/
    simp [← mul_smul, show ⅟ 2 * r = r * ⅟ 2 from Commute.invOf_left <| (2 : ℕ).cast_commute r]
    /-
      🎉 no goals
    -/


/-- The skew-adjoint part of an element of a star module, as a linear map. -/
@[simps]
def skewAdjointPart : A →ₗ[R] skewAdjoint A where
  toFun x :=
    ⟨(⅟ 2 : R) • (x - star x), by
      simp only [skewAdjoint.mem_iff, star_smul, star_sub, star_star, star_trivial, ← smul_neg,
        neg_sub]⟩
  map_add' x y := by
    /-
      R : Type u_1
      A : Type u_2
      inst✝⁷ : Semiring R
      inst✝⁶ : StarMul R
      inst✝⁵ : TrivialStar R
      inst✝⁴ : AddCommGroup A
      inst✝³ : Module R A
      inst✝² : StarAddMonoid A
      inst✝¹ : StarModule R A
      inst✝ : Invertible 2
      x y : A
      ⊢ Eq ((fun x => ⟨HSMul.hSMul (Invertible.invOf 2) (HSub.hSub x (Star.star x)), …
    -/
    ext
    simp only [sub_add, ← smul_add, sub_sub_eq_add_sub, star_add, AddSubgroup.coe_mk,
      AddSubgroup.coe_add]
  map_smul' r x := by
    /-
      R : Type u_1
      A : Type u_2
      inst✝⁷ : Semiring R
      inst✝⁶ : StarMul R
      inst✝⁵ : TrivialStar R
      inst✝⁴ : AddCommGroup A
      inst✝³ : Module R A
      inst✝² : StarAddMonoid A
      inst✝¹ : StarModule R A
      inst✝ : Invertible 2
      r : R
      x : A
      ⊢ Eq ({ toFun := fun x => ⟨HSMul.hSMul (Invertible.invOf 2) (HSub.hSub x (Star …
    -/
    ext
    simp [← mul_smul, ← smul_sub,
      show r * ⅟ 2 = ⅟ 2 * r from Commute.invOf_right <| (2 : ℕ).commute_cast r]


theorem StarModule.selfAdjointPart_add_skewAdjointPart (x : A) :
    (selfAdjointPart R x : A) + skewAdjointPart R x = x := by
  simp only [smul_sub, selfAdjointPart_apply_coe, smul_add, skewAdjointPart_apply_coe,
    add_add_sub_cancel, invOf_two_smul_add_invOf_two_smul]


theorem IsSelfAdjoint.coe_selfAdjointPart_apply {x : A} (hx : IsSelfAdjoint x) :
    (selfAdjointPart R x : A) = x := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝⁷ : Semiring R
    inst✝⁶ : StarMul R
    inst✝⁵ : TrivialStar R
    inst✝⁴ : AddCommGroup A
    inst✝³ : Module R A
    inst✝² : StarAddMonoid A
    inst✝¹ : StarModule R A
    inst✝ : Invertible 2
    x : A
    hx : IsSelfAdjoint x
    ⊢ Eq (↑((selfAdjointPart R) x)) x
  -/
  rw [selfAdjointPart_apply_coe, hx.star_eq, smul_add, invOf_two_smul_add_invOf_two_smul]
  /-
    🎉 no goals
  -/


theorem IsSelfAdjoint.selfAdjointPart_apply {x : A} (hx : IsSelfAdjoint x) :
    selfAdjointPart R x = ⟨x, hx⟩ :=
  Subtype.eq (hx.coe_selfAdjointPart_apply R)

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: make it a `simp`

theorem selfAdjointPart_comp_subtype_selfAdjoint :
    (selfAdjointPart R).comp (selfAdjoint.submodule R A).subtype = .id :=
  LinearMap.ext fun x ↦ x.2.selfAdjointPart_apply R


theorem IsSelfAdjoint.skewAdjointPart_apply {x : A} (hx : IsSelfAdjoint x) :
    skewAdjointPart R x = 0 := Subtype.eq <| by
  /-
    R : Type u_1
    A : Type u_2
    inst✝⁷ : Semiring R
    inst✝⁶ : StarMul R
    inst✝⁵ : TrivialStar R
    inst✝⁴ : AddCommGroup A
    inst✝³ : Module R A
    inst✝² : StarAddMonoid A
    inst✝¹ : StarModule R A
    inst✝ : Invertible 2
    x : A
    hx : IsSelfAdjoint x
    ⊢ Eq ↑((skewAdjointPart R) x) ↑0
  -/
  rw [skewAdjointPart_apply_coe, hx.star_eq, sub_self, smul_zero, ZeroMemClass.coe_zero]
  /-
    🎉 no goals
  -/

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: make it a `simp`

theorem skewAdjointPart_comp_subtype_selfAdjoint :
    (skewAdjointPart R).comp (selfAdjoint.submodule R A).subtype = 0 :=
  LinearMap.ext fun x ↦ x.2.skewAdjointPart_apply R

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: make it a `simp`

theorem selfAdjointPart_comp_subtype_skewAdjoint :
    (selfAdjointPart R).comp (skewAdjoint.submodule R A).subtype = 0 :=
                                                         /-
                                                           R : Type u_1
                                                           A : Type u_2
                                                           inst✝⁷ : Semiring R
                                                           inst✝⁶ : StarMul R
                                                           inst✝⁵ : TrivialStar R
                                                           inst✝⁴ : AddCommGroup A
                                                           inst✝³ : Module R A
                                                           inst✝² : StarAddMonoid A
                                                           inst✝¹ : StarModule R A
                                                           inst✝ : Invertible 2
                                                           x✝ : Subtype fun x => Membership.mem (skewAdjoint.submodule R A) x
                                                           x : A
                                                           hx : Eq (Star.star x) (Neg.neg x)
                                                           ⊢ Eq ↑(((selfAdjointPart R).comp (skewAdjoint.submodule R A).subtype) ⟨x, hx⟩) …
                                                         -/
  LinearMap.ext fun ⟨x, (hx : _ = _)⟩ ↦ Subtype.eq <| by simp [hx]
                                                         /-
                                                           🎉 no goals
                                                         -/

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: make it a `simp`

theorem skewAdjointPart_comp_subtype_skewAdjoint :
    (skewAdjointPart R).comp (skewAdjoint.submodule R A).subtype = .id :=
  LinearMap.ext fun ⟨x, (hx : _ = _)⟩ ↦ Subtype.eq <| by
    simp only [LinearMap.comp_apply, Submodule.subtype_apply, skewAdjointPart_apply_coe, hx,
                                                                    /-
                                                                      R : Type u_1
                                                                      A : Type u_2
                                                                      inst✝⁷ : Semiring R
                                                                      inst✝⁶ : StarMul R
                                                                      inst✝⁵ : TrivialStar R
                                                                      inst✝⁴ : AddCommGroup A
                                                                      inst✝³ : Module R A
                                                                      inst✝² : StarAddMonoid A
                                                                      inst✝¹ : StarModule R A
                                                                      inst✝ : Invertible 2
                                                                      x✝ : Subtype fun x => Membership.mem (skewAdjoint.submodule R A) x
                                                                      x : A
                                                                      hx : Eq (Star.star x) (Neg.neg x)
                                                                      ⊢ Eq x ↑(LinearMap.id ⟨x, hx⟩)
                                                                    -/
      sub_neg_eq_add, smul_add, invOf_two_smul_add_invOf_two_smul]; rfl
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


/-- The decomposition of elements of a star module into their self- and skew-adjoint parts,
as a linear equivalence. -/
-- Porting note: This attribute causes a `timeout at 'whnf'`.
@[simps!]
def StarModule.decomposeProdAdjoint : A ≃ₗ[R] selfAdjoint A × skewAdjoint A := by
  refine LinearEquiv.ofLinear ((selfAdjointPart R).prod (skewAdjointPart R))
    (LinearMap.coprod ((selfAdjoint.submodule R A).subtype) (skewAdjoint.submodule R A).subtype)
    ?_ (LinearMap.ext <| StarModule.selfAdjointPart_add_skewAdjointPart R)
  -- Note: with https://github.com/leanprover-community/mathlib4/pull/6965 `Submodule.coe_subtype` doesn't fire in `dsimp` or `simp`
  /-
    R : Type u_1
    A : Type u_2
    inst✝⁷ : Semiring R
    inst✝⁶ : StarMul R
    inst✝⁵ : TrivialStar R
    inst✝⁴ : AddCommGroup A
    inst✝³ : Module R A
    inst✝² : StarAddMonoid A
    inst✝¹ : StarModule R A
    inst✝ : Invertible 2
    ⊢ Eq (((selfAdjointPart R).prod (skewAdjointPart R)).comp ((selfAdjoint.submod …
  -/
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
  ext x <;> dsimp <;> erw [Submodule.coe_subtype, Submodule.coe_subtype] <;> simp
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


@[simp]
theorem algebraMap_star_comm (r : R) : algebraMap R A (star r) = star (algebraMap R A r) := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝⁵ : CommSemiring R
    inst✝⁴ : StarRing R
    inst✝³ : Semiring A
    inst✝² : StarMul A
    inst✝¹ : Algebra R A
    inst✝ : StarModule R A
    r : R
    ⊢ Eq ((algebraMap R A) (Star.star r)) (Star.star ((algebraMap R A) r))
  -/
  simp only [Algebra.algebraMap_eq_smul_one, star_smul, star_one]
  /-
    🎉 no goals
  -/


variable (A) in
protected lemma IsSelfAdjoint.algebraMap {r : R} (hr : IsSelfAdjoint r) :
    IsSelfAdjoint (algebraMap R A r) := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝⁵ : CommSemiring R
    inst✝⁴ : StarRing R
    inst✝³ : Semiring A
    inst✝² : StarMul A
    inst✝¹ : Algebra R A
    inst✝ : StarModule R A
    r : R
    hr : IsSelfAdjoint r
    ⊢ IsSelfAdjoint ((algebraMap R A) r)
  -/
  simpa using congr(algebraMap R A $(hr.star_eq))
  /-
    🎉 no goals
  -/


lemma isSelfAdjoint_algebraMap_iff {r : R} (h : Function.Injective (algebraMap R A)) :
    IsSelfAdjoint (algebraMap R A r) ↔ IsSelfAdjoint r :=
  ⟨fun hr ↦ h <| algebraMap_star_comm r (A := A) ▸ hr.star_eq, IsSelfAdjoint.algebraMap A⟩


