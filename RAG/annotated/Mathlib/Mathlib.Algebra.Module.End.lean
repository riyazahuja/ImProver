theorem AddMonoid.End.natCast_def (n : ℕ) :
    (↑n : AddMonoid.End M) = DistribMulAction.toAddMonoidEnd ℕ M n :=
  rfl


/-- `(•)` as an `AddMonoidHom`.

This is a stronger version of `DistribMulAction.toAddMonoidEnd` -/
@[simps! apply_apply]
def Module.toAddMonoidEnd : R →+* AddMonoid.End M :=
  { DistribMulAction.toAddMonoidEnd R M with
    -- Porting note: the two `show`s weren't needed in mathlib3.
    -- Somehow, now that `SMul` is heterogeneous, it can't unfold earlier fields of a definition for
    -- use in later fields.  See
    -- https://leanprover.zulipchat.com/#narrow/stream/287929-mathlib4/topic/Heterogeneous.20scalar.20multiplication
                                                                   /-
                                                                     R : Type u_1
                                                                     S : Type u_2
                                                                     M : Type u_3
                                                                     M₂ : Type u_4
                                                                     inst✝² : Semiring R
                                                                     inst✝¹ : AddCommMonoid M
                                                                     inst✝ : Module R M
                                                                     r✝ s : R
                                                                     x r : M
                                                                     ⊢ Eq (HSMul.hSMul 0 r) 0
                                                                   -/
    map_zero' := AddMonoidHom.ext fun r => show (0 : R) • r = 0 by simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
    map_add' := fun x y =>
                                                                    /-
                                                                      R : Type u_1
                                                                      S : Type u_2
                                                                      M : Type u_3
                                                                      M₂ : Type u_4
                                                                      inst✝² : Semiring R
                                                                      inst✝¹ : AddCommMonoid M
                                                                      inst✝ : Module R M
                                                                      r✝ s : R
                                                                      x✝ : M
                                                                      x y : R
                                                                      r : M
                                                                      ⊢ Eq (HSMul.hSMul (HAdd.hAdd x y) r) (HAdd.hAdd (HSMul.hSMul x r) (HSMul.hSMul …
                                                                    -/
      AddMonoidHom.ext fun r => show (x + y) • r = x • r + y • r by simp [add_smul] }
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


/-- A convenience alias for `Module.toAddMonoidEnd` as an `AddMonoidHom`, usually to allow the
use of `AddMonoidHom.flip`. -/
def smulAddHom : R →+ M →+ M :=
  (Module.toAddMonoidEnd R M).toAddMonoidHom


@[simp]
theorem smulAddHom_apply : smulAddHom R M r x = r • x :=
  rfl


lemma IsAddUnit.smul_left [Monoid S] [DistribMulAction S M] (hx : IsAddUnit x) (s : S) :
    IsAddUnit (s • x) :=
  hx.map (DistribMulAction.toAddMonoidHom M s)


lemma IsAddUnit.smul_right (hr : IsAddUnit r) : IsAddUnit (r • x) :=
  hr.map (AddMonoidHom.flip (smulAddHom R M) x)


theorem AddMonoid.End.intCast_def (z : ℤ) :
    (↑z : AddMonoid.End M) = DistribMulAction.toAddMonoidEnd ℤ M z :=
  rfl


/-- `zsmul` is equal to any other module structure via a cast. -/
@[norm_cast]
lemma Int.cast_smul_eq_zsmul (n : ℤ) (b : M) : (n : R) • b = n • b :=
  have : ((smulAddHom R M).flip b).comp (Int.castAddHom R) = (smulAddHom ℤ M).flip b := by
    /-
      R : Type u_1
      M : Type u_3
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      n : Int
      b : M
      ⊢ Eq (((smulAddHom R M).flip b).comp (Int.castAddHom R)) ((smulAddHom Int M).f …
    -/
    apply AddMonoidHom.ext_int
    /-
      case h1
      R : Type u_1
      M : Type u_3
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      n : Int
      b : M
      ⊢ Eq ((((smulAddHom R M).flip b).comp (Int.castAddHom R)) 1) (((smulAddHom Int …
    -/
    simp
    /-
      🎉 no goals
    -/
  DFunLike.congr_fun this n


@[deprecated (since := "2024-07-23")] alias intCast_smul := Int.cast_smul_eq_zsmul


/-- `zsmul` is equal to any other module structure via a cast. -/
@[deprecated Int.cast_smul_eq_zsmul (since := "2024-07-23")]
theorem zsmul_eq_smul_cast (n : ℤ) (b : M) : n • b = (n : R) • b := (Int.cast_smul_eq_zsmul ..).symm


/-- Convert back any exotic `ℤ`-smul to the canonical instance. This should not be needed since in
mathlib all `AddCommGroup`s should normally have exactly one `ℤ`-module structure by design. -/
theorem int_smul_eq_zsmul (h : Module ℤ M) (n : ℤ) (x : M) : @SMul.smul ℤ M h.toSMul n x = n • x :=
  Int.cast_smul_eq_zsmul ..


/-- All `ℤ`-module structures are equal. Not an instance since in mathlib all `AddCommGroup`
should normally have exactly one `ℤ`-module structure by design. -/
def AddCommGroup.uniqueIntModule : Unique (Module ℤ M) where
                /-
                  R : Type u_1
                  S : Type u_2
                  M : Type u_3
                  M₂ : Type u_4
                  inst✝² : Ring R
                  inst✝¹ : AddCommGroup M
                  inst✝ : Module R M
                  ⊢ Module Int M
                -/
  default := by infer_instance
                /-
                  🎉 no goals
                -/
                                          /-
                                            R : Type u_1
                                            S : Type u_2
                                            M : Type u_3
                                            M₂ : Type u_4
                                            inst✝² : Ring R
                                            inst✝¹ : AddCommGroup M
                                            inst✝ : Module R M
                                            P : Module Int M
                                            n : Int
                                            ⊢ ∀ (m : M), Eq (HSMul.hSMul n m) (HSMul.hSMul n m)
                                          -/
  uniq P := (Module.ext' P _) fun n => by convert int_smul_eq_zsmul P n
                                          /-
                                            🎉 no goals
                                          -/


theorem map_intCast_smul [AddCommGroup M] [AddCommGroup M₂] {F : Type*} [FunLike F M M₂]
    [AddMonoidHomClass F M M₂] (f : F) (R S : Type*) [Ring R] [Ring S] [Module R M] [Module S M₂]
    (x : ℤ) (a : M) :
                                          /-
                                            M : Type u_3
                                            M₂ : Type u_4
                                            inst✝⁷ : AddCommGroup M
                                            inst✝⁶ : AddCommGroup M₂
                                            F : Type u_5
                                            inst✝⁵ : FunLike F M M₂
                                            inst✝⁴ : AddMonoidHomClass F M M₂
                                            f : F
                                            R : Type u_6
                                            S : Type u_7
                                            inst✝³ : Ring R
                                            inst✝² : Ring S
                                            inst✝¹ : Module R M
                                            inst✝ : Module S M₂
                                            x : Int
                                            a : M
                                            ⊢ Eq (f (HSMul.hSMul (↑x) a)) (HSMul.hSMul (↑x) (f a))
                                          -/
    f ((x : R) • a) = (x : S) • f a := by simp only [Int.cast_smul_eq_zsmul, map_zsmul]
                                          /-
                                            🎉 no goals
                                          -/


instance AddCommGroup.intIsScalarTower {R : Type u} {M : Type v} [Ring R] [AddCommGroup M]
    [Module R M] : IsScalarTower ℤ R M where
  smul_assoc n x y := ((smulAddHom R M).flip y).map_zsmul x n

