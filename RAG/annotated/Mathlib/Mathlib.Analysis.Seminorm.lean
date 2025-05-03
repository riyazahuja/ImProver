/-- A seminorm on a module over a normed ring is a function to the reals that is positive
semidefinite, positive homogeneous, and subadditive. -/
structure Seminorm (𝕜 : Type*) (E : Type*) [SeminormedRing 𝕜] [AddGroup E] [SMul 𝕜 E] extends
  AddGroupSeminorm E where
  /-- The seminorm of a scalar multiplication is the product of the absolute value of the scalar
  and the original seminorm. -/
  smul' : ∀ (a : 𝕜) (x : E), toFun (a • x) = ‖a‖ * toFun x


/-- `SeminormClass F 𝕜 E` states that `F` is a type of seminorms on the `𝕜`-module `E`.

You should extend this class when you extend `Seminorm`. -/
class SeminormClass (F : Type*) (𝕜 E : outParam Type*) [SeminormedRing 𝕜] [AddGroup E]
  [SMul 𝕜 E] [FunLike F E ℝ] extends AddGroupSeminormClass F E ℝ : Prop where
  /-- The seminorm of a scalar multiplication is the product of the absolute value of the scalar
  and the original seminorm. -/
  map_smul_eq_mul (f : F) (a : 𝕜) (x : E) : f (a • x) = ‖a‖ * f x


/-- Alternative constructor for a `Seminorm` on an `AddCommGroup E` that is a module over a
`SeminormedRing 𝕜`. -/
def Seminorm.of [SeminormedRing 𝕜] [AddCommGroup E] [Module 𝕜 E] (f : E → ℝ)
    (add_le : ∀ x y : E, f (x + y) ≤ f x + f y) (smul : ∀ (a : 𝕜) (x : E), f (a • x) = ‖a‖ * f x) :
    Seminorm 𝕜 E where
  toFun := f
                  /-
                    R : Type u_1
                    R' : Type u_2
                    𝕜 : Type u_3
                    𝕜₂ : Type u_4
                    𝕜₃ : Type u_5
                    𝕝 : Type u_6
                    E : Type u_7
                    E₂ : Type u_8
                    E₃ : Type u_9
                    F : Type u_10
                    ι : Type u_11
                    inst✝² : SeminormedRing 𝕜
                    inst✝¹ : AddCommGroup E
                    inst✝ : Module 𝕜 E
                    f : E → Real
                    add_le : ∀ (x y : E), LE.le (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
                    smul : ∀ (a : 𝕜) (x : E), Eq (f (HSMul.hSMul a x)) (HMul.hMul (Norm.norm a) (f …
                    ⊢ Eq (f 0) 0
                  -/
  map_zero' := by rw [← zero_smul 𝕜 (0 : E), smul, norm_zero, zero_mul]
                  /-
                    🎉 no goals
                  -/
  add_le' := add_le
  smul' := smul
               /-
                 R : Type u_1
                 R' : Type u_2
                 𝕜 : Type u_3
                 𝕜₂ : Type u_4
                 𝕜₃ : Type u_5
                 𝕝 : Type u_6
                 E : Type u_7
                 E₂ : Type u_8
                 E₃ : Type u_9
                 F : Type u_10
                 ι : Type u_11
                 inst✝² : SeminormedRing 𝕜
                 inst✝¹ : AddCommGroup E
                 inst✝ : Module 𝕜 E
                 f : E → Real
                 add_le : ∀ (x y : E), LE.le (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
                 smul : ∀ (a : 𝕜) (x : E), Eq (f (HSMul.hSMul a x)) (HMul.hMul (Norm.norm a) (f …
                 x : E
                 ⊢ Eq (f (Neg.neg x)) (f x)
               -/
  neg' x := by rw [← neg_one_smul 𝕜, smul, norm_neg, ← smul, one_smul]
               /-
                 🎉 no goals
               -/


/-- Alternative constructor for a `Seminorm` over a normed field `𝕜` that only assumes `f 0 = 0`
and an inequality for the scalar multiplication. -/
def Seminorm.ofSMulLE [NormedField 𝕜] [AddCommGroup E] [Module 𝕜 E] (f : E → ℝ) (map_zero : f 0 = 0)
    (add_le : ∀ x y, f (x + y) ≤ f x + f y) (smul_le : ∀ (r : 𝕜) (x), f (r • x) ≤ ‖r‖ * f x) :
    Seminorm 𝕜 E :=
  Seminorm.of f add_le fun r x => by
    /-
      R : Type u_1
      R' : Type u_2
      𝕜 : Type u_3
      𝕜₂ : Type u_4
      𝕜₃ : Type u_5
      𝕝 : Type u_6
      E : Type u_7
      E₂ : Type u_8
      E₃ : Type u_9
      F : Type u_10
      ι : Type u_11
      inst✝² : NormedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      f : E → Real
      map_zero : Eq (f 0) 0
      add_le : ∀ (x y : E), LE.le (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
      smul_le : ∀ (r : 𝕜) (x : E), LE.le (f (HSMul.hSMul r x)) (HMul.hMul (Norm.norm …
      r : 𝕜
      x : E
      ⊢ Eq (f (HSMul.hSMul r x)) (HMul.hMul (Norm.norm r) (f x))
    -/
    refine le_antisymm (smul_le r x) ?_
    /-
      R : Type u_1
      R' : Type u_2
      𝕜 : Type u_3
      𝕜₂ : Type u_4
      𝕜₃ : Type u_5
      𝕝 : Type u_6
      E : Type u_7
      E₂ : Type u_8
      E₃ : Type u_9
      F : Type u_10
      ι : Type u_11
      inst✝² : NormedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      f : E → Real
      map_zero : Eq (f 0) 0
      add_le : ∀ (x y : E), LE.le (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
      smul_le : ∀ (r : 𝕜) (x : E), LE.le (f (HSMul.hSMul r x)) (HMul.hMul (Norm.norm …
      r : 𝕜
      x : E
      ⊢ LE.le (HMul.hMul (Norm.norm r) (f x)) (f (HSMul.hSMul r x))
    -/
    by_cases h : r = 0
      /-
        case pos
        R : Type u_1
        R' : Type u_2
        𝕜 : Type u_3
        𝕜₂ : Type u_4
        𝕜₃ : Type u_5
        𝕝 : Type u_6
        E : Type u_7
        E₂ : Type u_8
        E₃ : Type u_9
        F : Type u_10
        ι : Type u_11
        inst✝² : NormedField 𝕜
        inst✝¹ : AddCommGroup E
        inst✝ : Module 𝕜 E
        f : E → Real
        map_zero : Eq (f 0) 0
        add_le : ∀ (x y : E), LE.le (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
        smul_le : ∀ (r : 𝕜) (x : E), LE.le (f (HSMul.hSMul r x)) (HMul.hMul (Norm.norm …
        r : 𝕜
        x : E
        h : Eq r 0
        ⊢ LE.le (HMul.hMul (Norm.norm r) (f x)) (f (HSMul.hSMul r x))
      -/
    · simp [h, map_zero]
      /-
        🎉 no goals
      -/
    /-
      case neg
      R : Type u_1
      R' : Type u_2
      𝕜 : Type u_3
      𝕜₂ : Type u_4
      𝕜₃ : Type u_5
      𝕝 : Type u_6
      E : Type u_7
      E₂ : Type u_8
      E₃ : Type u_9
      F : Type u_10
      ι : Type u_11
      inst✝² : NormedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      f : E → Real
      map_zero : Eq (f 0) 0
      add_le : ∀ (x y : E), LE.le (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
      smul_le : ∀ (r : 𝕜) (x : E), LE.le (f (HSMul.hSMul r x)) (HMul.hMul (Norm.norm …
      r : 𝕜
      x : E
      h : Not (Eq r 0)
      ⊢ LE.le (HMul.hMul (Norm.norm r) (f x)) (f (HSMul.hSMul r x))
    -/
    rw [← mul_le_mul_left (inv_pos.mpr (norm_pos_iff.mpr h))]
    /-
      case neg
      R : Type u_1
      R' : Type u_2
      𝕜 : Type u_3
      𝕜₂ : Type u_4
      𝕜₃ : Type u_5
      𝕝 : Type u_6
      E : Type u_7
      E₂ : Type u_8
      E₃ : Type u_9
      F : Type u_10
      ι : Type u_11
      inst✝² : NormedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      f : E → Real
      map_zero : Eq (f 0) 0
      add_le : ∀ (x y : E), LE.le (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
      smul_le : ∀ (r : 𝕜) (x : E), LE.le (f (HSMul.hSMul r x)) (HMul.hMul (Norm.norm …
      r : 𝕜
      x : E
      h : Not (Eq r 0)
      ⊢ LE.le (HMul.hMul (Inv.inv (Norm.norm r)) (HMul.hMul (Norm.norm r) (f x))) (H …
    -/
    rw [inv_mul_cancel_left₀ (norm_ne_zero_iff.mpr h)]
    /-
      case neg
      R : Type u_1
      R' : Type u_2
      𝕜 : Type u_3
      𝕜₂ : Type u_4
      𝕜₃ : Type u_5
      𝕝 : Type u_6
      E : Type u_7
      E₂ : Type u_8
      E₃ : Type u_9
      F : Type u_10
      ι : Type u_11
      inst✝² : NormedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      f : E → Real
      map_zero : Eq (f 0) 0
      add_le : ∀ (x y : E), LE.le (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
      smul_le : ∀ (r : 𝕜) (x : E), LE.le (f (HSMul.hSMul r x)) (HMul.hMul (Norm.norm …
      r : 𝕜
      x : E
      h : Not (Eq r 0)
      ⊢ LE.le (f x) (HMul.hMul (Inv.inv (Norm.norm r)) (f (HSMul.hSMul r x)))
    -/
    specialize smul_le r⁻¹ (r • x)
    /-
      case neg
      R : Type u_1
      R' : Type u_2
      𝕜 : Type u_3
      𝕜₂ : Type u_4
      𝕜₃ : Type u_5
      𝕝 : Type u_6
      E : Type u_7
      E₂ : Type u_8
      E₃ : Type u_9
      F : Type u_10
      ι : Type u_11
      inst✝² : NormedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      f : E → Real
      map_zero : Eq (f 0) 0
      add_le : ∀ (x y : E), LE.le (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
      r : 𝕜
      x : E
      h : Not (Eq r 0)
      smul_le : LE.le (f (HSMul.hSMul (Inv.inv r) (HSMul.hSMul r x))) (HMul.hMul (No …
      ⊢ LE.le (f x) (HMul.hMul (Inv.inv (Norm.norm r)) (f (HSMul.hSMul r x)))
    -/
    rw [norm_inv] at smul_le
    /-
      case neg
      R : Type u_1
      R' : Type u_2
      𝕜 : Type u_3
      𝕜₂ : Type u_4
      𝕜₃ : Type u_5
      𝕝 : Type u_6
      E : Type u_7
      E₂ : Type u_8
      E₃ : Type u_9
      F : Type u_10
      ι : Type u_11
      inst✝² : NormedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      f : E → Real
      map_zero : Eq (f 0) 0
      add_le : ∀ (x y : E), LE.le (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
      r : 𝕜
      x : E
      h : Not (Eq r 0)
      smul_le : LE.le (f (HSMul.hSMul (Inv.inv r) (HSMul.hSMul r x))) (HMul.hMul (In …
      ⊢ LE.le (f x) (HMul.hMul (Inv.inv (Norm.norm r)) (f (HSMul.hSMul r x)))
    -/
    convert smul_le
    /-
      case h.e'_3.h.e'_1
      R : Type u_1
      R' : Type u_2
      𝕜 : Type u_3
      𝕜₂ : Type u_4
      𝕜₃ : Type u_5
      𝕝 : Type u_6
      E : Type u_7
      E₂ : Type u_8
      E₃ : Type u_9
      F : Type u_10
      ι : Type u_11
      inst✝² : NormedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      f : E → Real
      map_zero : Eq (f 0) 0
      add_le : ∀ (x y : E), LE.le (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
      r : 𝕜
      x : E
      h : Not (Eq r 0)
      smul_le : LE.le (f (HSMul.hSMul (Inv.inv r) (HSMul.hSMul r x))) (HMul.hMul (In …
      ⊢ Eq x (HSMul.hSMul (Inv.inv r) (HSMul.hSMul r x))
    -/
    simp [h]
    /-
      🎉 no goals
    -/


instance instFunLike : FunLike (Seminorm 𝕜 E) E ℝ where
  coe f := f.toFun
  coe_injective' f g h := by
    /-
      R : Type u_1
      R' : Type u_2
      𝕜 : Type u_3
      𝕜₂ : Type u_4
      𝕜₃ : Type u_5
      𝕝 : Type u_6
      E : Type u_7
      E₂ : Type u_8
      E₃ : Type u_9
      F : Type u_10
      ι : Type u_11
      inst✝² : SeminormedRing 𝕜
      inst✝¹ : AddGroup E
      inst✝ : SMul 𝕜 E
      f g : Seminorm 𝕜 E
      h : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g)
      ⊢ Eq f g
    -/
    rcases f with ⟨⟨_⟩⟩
    /-
      case mk.mk
      R : Type u_1
      R' : Type u_2
      𝕜 : Type u_3
      𝕜₂ : Type u_4
      𝕜₃ : Type u_5
      𝕝 : Type u_6
      E : Type u_7
      E₂ : Type u_8
      E₃ : Type u_9
      F : Type u_10
      ι : Type u_11
      inst✝² : SeminormedRing 𝕜
      inst✝¹ : AddGroup E
      inst✝ : SMul 𝕜 E
      g : Seminorm 𝕜 E
      toFun✝ : E → Real
      map_zero'✝ : Eq (toFun✝ 0) 0
      add_le'✝ : ∀ (r s : E), LE.le (toFun✝ (HAdd.hAdd r s)) (HAdd.hAdd (toFun✝ r) ( …
      neg'✝ : ∀ (r : E), Eq (toFun✝ (Neg.neg r)) (toFun✝ r)
      smul'✝ : ∀ (a : 𝕜) (x : E), Eq ({ toFun := toFun✝, map_zero' := map_zero'✝, ad …
      h : Eq ((fun f => f.toFun) { toFun := toFun✝, map_zero' := map_zero'✝, add_le' …
      ⊢ Eq { toFun := toFun✝, map_zero' := map_zero'✝, add_le' := add_le'✝, neg' :=  …
    -/
    rcases g with ⟨⟨_⟩⟩
    /-
      case mk.mk.mk.mk
      R : Type u_1
      R' : Type u_2
      𝕜 : Type u_3
      𝕜₂ : Type u_4
      𝕜₃ : Type u_5
      𝕝 : Type u_6
      E : Type u_7
      E₂ : Type u_8
      E₃ : Type u_9
      F : Type u_10
      ι : Type u_11
      inst✝² : SeminormedRing 𝕜
      inst✝¹ : AddGroup E
      inst✝ : SMul 𝕜 E
      toFun✝¹ : E → Real
      map_zero'✝¹ : Eq (toFun✝¹ 0) 0
      add_le'✝¹ : ∀ (r s : E), LE.le (toFun✝¹ (HAdd.hAdd r s)) (HAdd.hAdd (toFun✝¹ r …
      neg'✝¹ : ∀ (r : E), Eq (toFun✝¹ (Neg.neg r)) (toFun✝¹ r)
      smul'✝¹ : ∀ (a : 𝕜) (x : E), Eq ({ toFun := toFun✝¹, map_zero' := map_zero'✝¹, …
      toFun✝ : E → Real
      map_zero'✝ : Eq (toFun✝ 0) 0
      add_le'✝ : ∀ (r s : E), LE.le (toFun✝ (HAdd.hAdd r s)) (HAdd.hAdd (toFun✝ r) ( …
      neg'✝ : ∀ (r : E), Eq (toFun✝ (Neg.neg r)) (toFun✝ r)
      smul'✝ : ∀ (a : 𝕜) (x : E), Eq ({ toFun := toFun✝, map_zero' := map_zero'✝, ad …
      h : Eq ((fun f => f.toFun) { toFun := toFun✝¹, map_zero' := map_zero'✝¹, add_l …
      ⊢ Eq { toFun := toFun✝¹, map_zero' := map_zero'✝¹, add_le' := add_le'✝¹, neg'  …
    -/
    congr
    /-
      🎉 no goals
    -/


instance instSeminormClass : SeminormClass (Seminorm 𝕜 E) 𝕜 E where
  map_zero f := f.map_zero'
  map_add_le_add f := f.add_le'
  map_neg_eq_map f := f.neg'
  map_smul_eq_mul f := f.smul'


@[ext]
theorem ext {p q : Seminorm 𝕜 E} (h : ∀ x, (p : E → ℝ) x = q x) : p = q :=
  DFunLike.ext p q h


instance instZero : Zero (Seminorm 𝕜 E) :=
  ⟨{ AddGroupSeminorm.instZeroAddGroupSeminorm.zero with
    smul' := fun _ _ => (mul_zero _).symm }⟩


@[simp]
theorem coe_zero : ⇑(0 : Seminorm 𝕜 E) = 0 :=
  rfl


@[simp]
theorem zero_apply (x : E) : (0 : Seminorm 𝕜 E) x = 0 :=
  rfl


instance : Inhabited (Seminorm 𝕜 E) :=
  ⟨0⟩


/-- Any action on `ℝ` which factors through `ℝ≥0` applies to a seminorm. -/
instance instSMul [SMul R ℝ] [SMul R ℝ≥0] [IsScalarTower R ℝ≥0 ℝ] : SMul R (Seminorm 𝕜 E) where
  smul r p :=
    { r • p.toAddGroupSeminorm with
      toFun := fun x => r • p x
      smul' := fun _ _ => by
        /-
          R : Type u_1
          R' : Type u_2
          𝕜 : Type u_3
          𝕜₂ : Type u_4
          𝕜₃ : Type u_5
          𝕝 : Type u_6
          E : Type u_7
          E₂ : Type u_8
          E₃ : Type u_9
          F : Type u_10
          ι : Type u_11
          inst✝⁵ : SeminormedRing 𝕜
          inst✝⁴ : AddGroup E
          inst✝³ : SMul 𝕜 E
          p✝ : Seminorm 𝕜 E
          x : E
          r✝ : Real
          inst✝² : SMul R Real
          inst✝¹ : SMul R NNReal
          inst✝ : IsScalarTower R NNReal Real
          r : R
          p : Seminorm 𝕜 E
          x✝¹ : 𝕜
          x✝ : E
          ⊢ Eq ({ toFun := fun x => HSMul.hSMul r (p x), map_zero' := ⋯, add_le' := ⋯, n …
        -/
        simp only [← smul_one_smul ℝ≥0 r (_ : ℝ), NNReal.smul_def, smul_eq_mul]
        /-
          R : Type u_1
          R' : Type u_2
          𝕜 : Type u_3
          𝕜₂ : Type u_4
          𝕜₃ : Type u_5
          𝕝 : Type u_6
          E : Type u_7
          E₂ : Type u_8
          E₃ : Type u_9
          F : Type u_10
          ι : Type u_11
          inst✝⁵ : SeminormedRing 𝕜
          inst✝⁴ : AddGroup E
          inst✝³ : SMul 𝕜 E
          p✝ : Seminorm 𝕜 E
          x : E
          r✝ : Real
          inst✝² : SMul R Real
          inst✝¹ : SMul R NNReal
          inst✝ : IsScalarTower R NNReal Real
          r : R
          p : Seminorm 𝕜 E
          x✝¹ : 𝕜
          x✝ : E
          ⊢ Eq (HMul.hMul (↑(HSMul.hSMul r 1)) (p (HSMul.hSMul x✝¹ x✝))) (HMul.hMul (Nor …
        -/
        rw [map_smul_eq_mul, mul_left_comm] }
        /-
          🎉 no goals
        -/


instance [SMul R ℝ] [SMul R ℝ≥0] [IsScalarTower R ℝ≥0 ℝ] [SMul R' ℝ] [SMul R' ℝ≥0]
    [IsScalarTower R' ℝ≥0 ℝ] [SMul R R'] [IsScalarTower R R' ℝ] :
    IsScalarTower R R' (Seminorm 𝕜 E) where
  smul_assoc r a p := ext fun x => smul_assoc r a (p x)


theorem coe_smul [SMul R ℝ] [SMul R ℝ≥0] [IsScalarTower R ℝ≥0 ℝ] (r : R) (p : Seminorm 𝕜 E) :
    ⇑(r • p) = r • ⇑p :=
  rfl


@[simp]
theorem smul_apply [SMul R ℝ] [SMul R ℝ≥0] [IsScalarTower R ℝ≥0 ℝ] (r : R) (p : Seminorm 𝕜 E)
    (x : E) : (r • p) x = r • p x :=
  rfl


instance instAdd : Add (Seminorm 𝕜 E) where
  add p q :=
    { p.toAddGroupSeminorm + q.toAddGroupSeminorm with
      toFun := fun x => p x + q x
                             /-
                               R : Type u_1
                               R' : Type u_2
                               𝕜 : Type u_3
                               𝕜₂ : Type u_4
                               𝕜₃ : Type u_5
                               𝕝 : Type u_6
                               E : Type u_7
                               E₂ : Type u_8
                               E₃ : Type u_9
                               F : Type u_10
                               ι : Type u_11
                               inst✝² : SeminormedRing 𝕜
                               inst✝¹ : AddGroup E
                               inst✝ : SMul 𝕜 E
                               p✝ : Seminorm 𝕜 E
                               x✝ : E
                               r : Real
                               p q : Seminorm 𝕜 E
                               a : 𝕜
                               x : E
                               ⊢ Eq ({ toFun := fun x => HAdd.hAdd (p x) (q x), map_zero' := ⋯, add_le' := ⋯, …
                             -/
      smul' := fun a x => by simp only [map_smul_eq_mul, map_smul_eq_mul, mul_add] }
                             /-
                               🎉 no goals
                             -/


theorem coe_add (p q : Seminorm 𝕜 E) : ⇑(p + q) = p + q :=
  rfl


@[simp]
theorem add_apply (p q : Seminorm 𝕜 E) (x : E) : (p + q) x = p x + q x :=
  rfl


instance instAddMonoid : AddMonoid (Seminorm 𝕜 E) :=
                                                               /-
                                                                 R : Type u_1
                                                                 R' : Type u_2
                                                                 𝕜 : Type u_3
                                                                 𝕜₂ : Type u_4
                                                                 𝕜₃ : Type u_5
                                                                 𝕝 : Type u_6
                                                                 E : Type u_7
                                                                 E₂ : Type u_8
                                                                 E₃ : Type u_9
                                                                 F : Type u_10
                                                                 ι : Type u_11
                                                                 inst✝² : SeminormedRing 𝕜
                                                                 inst✝¹ : AddGroup E
                                                                 inst✝ : SMul 𝕜 E
                                                                 p : Seminorm 𝕜 E
                                                                 x : E
                                                                 r : Real
                                                                 x✝¹ : Seminorm 𝕜 E
                                                                 x✝ : Nat
                                                                 ⊢ Eq (⇑(HSMul.hSMul x✝ x✝¹)) (HSMul.hSMul x✝ ⇑x✝¹)
                                                               -/
  DFunLike.coe_injective.addMonoid _ rfl coe_add fun _ _ => by rfl
                                                               /-
                                                                 🎉 no goals
                                                               -/


instance instOrderedCancelAddCommMonoid : OrderedCancelAddCommMonoid (Seminorm 𝕜 E) :=
  DFunLike.coe_injective.orderedCancelAddCommMonoid _ rfl coe_add fun _ _ => rfl


instance instMulAction [Monoid R] [MulAction R ℝ] [SMul R ℝ≥0] [IsScalarTower R ℝ≥0 ℝ] :
    MulAction R (Seminorm 𝕜 E) :=
                                         /-
                                           R : Type u_1
                                           R' : Type u_2
                                           𝕜 : Type u_3
                                           𝕜₂ : Type u_4
                                           𝕜₃ : Type u_5
                                           𝕝 : Type u_6
                                           E : Type u_7
                                           E₂ : Type u_8
                                           E₃ : Type u_9
                                           F : Type u_10
                                           ι : Type u_11
                                           inst✝⁶ : SeminormedRing 𝕜
                                           inst✝⁵ : AddGroup E
                                           inst✝⁴ : SMul 𝕜 E
                                           p : Seminorm 𝕜 E
                                           x : E
                                           r : Real
                                           inst✝³ : Monoid R
                                           inst✝² : MulAction R Real
                                           inst✝¹ : SMul R NNReal
                                           inst✝ : IsScalarTower R NNReal Real
                                           ⊢ ∀ (c : R) (x : Seminorm 𝕜 E), Eq (⇑(HSMul.hSMul c x)) (HSMul.hSMul c ⇑x)
                                         -/
  DFunLike.coe_injective.mulAction _ (by intros; rfl)
                                                 /-
                                                   🎉 no goals
                                                 -/


/-- `coeFn` as an `AddMonoidHom`. Helper definition for showing that `Seminorm 𝕜 E` is a module. -/
@[simps]
def coeFnAddMonoidHom : AddMonoidHom (Seminorm 𝕜 E) (E → ℝ) where
  toFun := (↑)
  map_zero' := coe_zero
  map_add' := coe_add


theorem coeFnAddMonoidHom_injective : Function.Injective (coeFnAddMonoidHom 𝕜 E) :=
  show @Function.Injective (Seminorm 𝕜 E) (E → ℝ) (↑) from DFunLike.coe_injective


instance instDistribMulAction [Monoid R] [DistribMulAction R ℝ] [SMul R ℝ≥0]
    [IsScalarTower R ℝ≥0 ℝ] : DistribMulAction R (Seminorm 𝕜 E) :=
                                                           /-
                                                             R : Type u_1
                                                             R' : Type u_2
                                                             𝕜 : Type u_3
                                                             𝕜₂ : Type u_4
                                                             𝕜₃ : Type u_5
                                                             𝕝 : Type u_6
                                                             E : Type u_7
                                                             E₂ : Type u_8
                                                             E₃ : Type u_9
                                                             F : Type u_10
                                                             ι : Type u_11
                                                             inst✝⁶ : SeminormedRing 𝕜
                                                             inst✝⁵ : AddGroup E
                                                             inst✝⁴ : SMul 𝕜 E
                                                             p : Seminorm 𝕜 E
                                                             x : E
                                                             r : Real
                                                             inst✝³ : Monoid R
                                                             inst✝² : DistribMulAction R Real
                                                             inst✝¹ : SMul R NNReal
                                                             inst✝ : IsScalarTower R NNReal Real
                                                             ⊢ ∀ (c : R) (x : Seminorm 𝕜 E), Eq ((Seminorm.coeFnAddMonoidHom 𝕜 E) (HSMul.hS …
                                                           -/
  (coeFnAddMonoidHom_injective 𝕜 E).distribMulAction _ (by intros; rfl)
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


instance instModule [Semiring R] [Module R ℝ] [SMul R ℝ≥0] [IsScalarTower R ℝ≥0 ℝ] :
    Module R (Seminorm 𝕜 E) :=
                                                   /-
                                                     R : Type u_1
                                                     R' : Type u_2
                                                     𝕜 : Type u_3
                                                     𝕜₂ : Type u_4
                                                     𝕜₃ : Type u_5
                                                     𝕝 : Type u_6
                                                     E : Type u_7
                                                     E₂ : Type u_8
                                                     E₃ : Type u_9
                                                     F : Type u_10
                                                     ι : Type u_11
                                                     inst✝⁶ : SeminormedRing 𝕜
                                                     inst✝⁵ : AddGroup E
                                                     inst✝⁴ : SMul 𝕜 E
                                                     p : Seminorm 𝕜 E
                                                     x : E
                                                     r : Real
                                                     inst✝³ : Semiring R
                                                     inst✝² : Module R Real
                                                     inst✝¹ : SMul R NNReal
                                                     inst✝ : IsScalarTower R NNReal Real
                                                     ⊢ ∀ (c : R) (x : Seminorm 𝕜 E), Eq ((Seminorm.coeFnAddMonoidHom 𝕜 E) (HSMul.hS …
                                                   -/
  (coeFnAddMonoidHom_injective 𝕜 E).module R _ (by intros; rfl)
                                                           /-
                                                             🎉 no goals
                                                           -/


instance instSup : Max (Seminorm 𝕜 E) where
  max p q :=
    { p.toAddGroupSeminorm ⊔ q.toAddGroupSeminorm with
      toFun := p ⊔ q
      smul' := fun x v =>
        (congr_arg₂ max (map_smul_eq_mul p x v) (map_smul_eq_mul q x v)).trans <|
          (mul_max_of_nonneg _ _ <| norm_nonneg x).symm }


@[simp]
theorem coe_sup (p q : Seminorm 𝕜 E) : ⇑(p ⊔ q) = (p : E → ℝ) ⊔ (q : E → ℝ) :=
  rfl


theorem sup_apply (p q : Seminorm 𝕜 E) (x : E) : (p ⊔ q) x = p x ⊔ q x :=
  rfl


theorem smul_sup [SMul R ℝ] [SMul R ℝ≥0] [IsScalarTower R ℝ≥0 ℝ] (r : R) (p q : Seminorm 𝕜 E) :
    r • (p ⊔ q) = r • p ⊔ r • q :=
  have real.smul_max : ∀ x y : ℝ, r • max x y = max (r • x) (r • y) := fun x y => by
    simpa only [← smul_eq_mul, ← NNReal.smul_def, smul_one_smul ℝ≥0 r (_ : ℝ)] using
      mul_max_of_nonneg x y (r • (1 : ℝ≥0) : ℝ≥0).coe_nonneg
  ext fun _ => real.smul_max _ _


instance instPartialOrder : PartialOrder (Seminorm 𝕜 E) :=
  PartialOrder.lift _ DFunLike.coe_injective


@[simp, norm_cast]
theorem coe_le_coe {p q : Seminorm 𝕜 E} : (p : E → ℝ) ≤ q ↔ p ≤ q :=
  Iff.rfl


@[simp, norm_cast]
theorem coe_lt_coe {p q : Seminorm 𝕜 E} : (p : E → ℝ) < q ↔ p < q :=
  Iff.rfl


theorem le_def {p q : Seminorm 𝕜 E} : p ≤ q ↔ ∀ x, p x ≤ q x :=
  Iff.rfl


theorem lt_def {p q : Seminorm 𝕜 E} : p < q ↔ p ≤ q ∧ ∃ x, p x < q x :=
  @Pi.lt_def _ _ _ p q


instance instSemilatticeSup : SemilatticeSup (Seminorm 𝕜 E) :=
  Function.Injective.semilatticeSup _ DFunLike.coe_injective coe_sup


noncomputable instance smul_nnreal_real : SMul ℝ≥0 ℝ := inferInstance


/-- Composition of a seminorm with a linear map is a seminorm. -/
def comp (p : Seminorm 𝕜₂ E₂) (f : E →ₛₗ[σ₁₂] E₂) : Seminorm 𝕜 E :=
  { p.toAddGroupSeminorm.comp f.toAddMonoidHom with
    toFun := fun x => p (f x)
    -- Porting note: the `simp only` below used to be part of the `rw`.
    -- I'm not sure why this change was needed, and am worried by it!
    -- Note: https://github.com/leanprover-community/mathlib4/pull/8386 had to change `map_smulₛₗ` to `map_smulₛₗ _`
                           /-
                             R : Type u_1
                             R' : Type u_2
                             𝕜 : Type u_3
                             𝕜₂ : Type u_4
                             𝕜₃ : Type u_5
                             𝕝 : Type u_6
                             E : Type u_7
                             E₂ : Type u_8
                             E₃ : Type u_9
                             F : Type u_10
                             ι : Type u_11
                             inst✝¹⁴ : SeminormedRing 𝕜
                             inst✝¹³ : SeminormedRing 𝕜₂
                             inst✝¹² : SeminormedRing 𝕜₃
                             σ₁₂ : RingHom 𝕜 𝕜₂
                             inst✝¹¹ : RingHomIsometric σ₁₂
                             σ₂₃ : RingHom 𝕜₂ 𝕜₃
                             inst✝¹⁰ : RingHomIsometric σ₂₃
                             σ₁₃ : RingHom 𝕜 𝕜₃
                             inst✝⁹ : RingHomIsometric σ₁₃
                             inst✝⁸ : AddCommGroup E
                             inst✝⁷ : AddCommGroup E₂
                             inst✝⁶ : AddCommGroup E₃
                             inst✝⁵ : Module 𝕜 E
                             inst✝⁴ : Module 𝕜₂ E₂
                             inst✝³ : Module 𝕜₃ E₃
                             inst✝² : SMul R Real
                             inst✝¹ : SMul R NNReal
                             inst✝ : IsScalarTower R NNReal Real
                             p : Seminorm 𝕜₂ E₂
                             f : LinearMap σ₁₂ E E₂
                             x✝¹ : 𝕜
                             x✝ : E
                             ⊢ Eq ({ toFun := fun x => p (f x), map_zero' := ⋯, add_le' := ⋯, neg' := ⋯ }.t …
                           -/
    smul' := fun _ _ => by simp only [map_smulₛₗ _]; rw [map_smul_eq_mul, RingHomIsometric.is_iso] }
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem coe_comp (p : Seminorm 𝕜₂ E₂) (f : E →ₛₗ[σ₁₂] E₂) : ⇑(p.comp f) = p ∘ f :=
  rfl


@[simp]
theorem comp_apply (p : Seminorm 𝕜₂ E₂) (f : E →ₛₗ[σ₁₂] E₂) (x : E) : (p.comp f) x = p (f x) :=
  rfl


@[simp]
theorem comp_id (p : Seminorm 𝕜 E) : p.comp LinearMap.id = p :=
  ext fun _ => rfl


@[simp]
theorem comp_zero (p : Seminorm 𝕜₂ E₂) : p.comp (0 : E →ₛₗ[σ₁₂] E₂) = 0 :=
  ext fun _ => map_zero p


@[simp]
theorem zero_comp (f : E →ₛₗ[σ₁₂] E₂) : (0 : Seminorm 𝕜₂ E₂).comp f = 0 :=
  ext fun _ => rfl


theorem comp_comp [RingHomCompTriple σ₁₂ σ₂₃ σ₁₃] (p : Seminorm 𝕜₃ E₃) (g : E₂ →ₛₗ[σ₂₃] E₃)
    (f : E →ₛₗ[σ₁₂] E₂) : p.comp (g.comp f) = (p.comp g).comp f :=
  ext fun _ => rfl


theorem add_comp (p q : Seminorm 𝕜₂ E₂) (f : E →ₛₗ[σ₁₂] E₂) :
    (p + q).comp f = p.comp f + q.comp f :=
  ext fun _ => rfl


theorem comp_add_le (p : Seminorm 𝕜₂ E₂) (f g : E →ₛₗ[σ₁₂] E₂) :
    p.comp (f + g) ≤ p.comp f + p.comp g := fun _ => map_add_le_add p _ _


theorem smul_comp (p : Seminorm 𝕜₂ E₂) (f : E →ₛₗ[σ₁₂] E₂) (c : R) :
    (c • p).comp f = c • p.comp f :=
  ext fun _ => rfl


theorem comp_mono {p q : Seminorm 𝕜₂ E₂} (f : E →ₛₗ[σ₁₂] E₂) (hp : p ≤ q) : p.comp f ≤ q.comp f :=
  fun _ => hp _


/-- The composition as an `AddMonoidHom`. -/
@[simps]
def pullback (f : E →ₛₗ[σ₁₂] E₂) : Seminorm 𝕜₂ E₂ →+ Seminorm 𝕜 E where
  toFun := fun p => p.comp f
  map_zero' := zero_comp f
  map_add' := fun p q => add_comp p q f


instance instOrderBot : OrderBot (Seminorm 𝕜 E) where
  bot := 0
  bot_le := apply_nonneg


@[simp]
theorem coe_bot : ⇑(⊥ : Seminorm 𝕜 E) = 0 :=
  rfl


theorem bot_eq_zero : (⊥ : Seminorm 𝕜 E) = 0 :=
  rfl


theorem smul_le_smul {p q : Seminorm 𝕜 E} {a b : ℝ≥0} (hpq : p ≤ q) (hab : a ≤ b) :
    a • p ≤ b • q := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p q : Seminorm 𝕜 E
    a b : NNReal
    hpq : LE.le p q
    hab : LE.le a b
    ⊢ LE.le (HSMul.hSMul a p) (HSMul.hSMul b q)
  -/
  simp_rw [le_def]
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p q : Seminorm 𝕜 E
    a b : NNReal
    hpq : LE.le p q
    hab : LE.le a b
    ⊢ ∀ (x : E), LE.le ((HSMul.hSMul a p) x) ((HSMul.hSMul b q) x)
  -/
  intro x
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p q : Seminorm 𝕜 E
    a b : NNReal
    hpq : LE.le p q
    hab : LE.le a b
    x : E
    ⊢ LE.le ((HSMul.hSMul a p) x) ((HSMul.hSMul b q) x)
  -/
  exact mul_le_mul hab (hpq x) (apply_nonneg p x) (NNReal.coe_nonneg b)
  /-
    🎉 no goals
  -/


theorem finset_sup_apply (p : ι → Seminorm 𝕜 E) (s : Finset ι) (x : E) :
    s.sup p x = ↑(s.sup fun i => ⟨p i x, apply_nonneg (p i) x⟩ : ℝ≥0) := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    ι : Type u_11
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → Seminorm 𝕜 E
    s : Finset ι
    x : E
    ⊢ Eq ((s.sup p) x) ↑(s.sup fun i => ⟨(p i) x, ⋯⟩)
  -/
  induction' s using Finset.cons_induction_on with a s ha ih
    /-
      case h₁
      𝕜 : Type u_3
      E : Type u_7
      ι : Type u_11
      inst✝² : SeminormedRing 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      p : ι → Seminorm 𝕜 E
      x : E
      ⊢ Eq ((EmptyCollection.emptyCollection.sup p) x) ↑(EmptyCollection.emptyCollec …
    -/
  · rw [Finset.sup_empty, Finset.sup_empty, coe_bot, _root_.bot_eq_zero, Pi.zero_apply]
    /-
      case h₁
      𝕜 : Type u_3
      E : Type u_7
      ι : Type u_11
      inst✝² : SeminormedRing 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      p : ι → Seminorm 𝕜 E
      x : E
      ⊢ Eq 0 ↑0
    -/
    norm_cast
    /-
      🎉 no goals
    -/
    /-
      case h₂
      𝕜 : Type u_3
      E : Type u_7
      ι : Type u_11
      inst✝² : SeminormedRing 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      p : ι → Seminorm 𝕜 E
      x : E
      a : ι
      s : Finset ι
      ha : Not (Membership.mem s a)
      ih : Eq ((s.sup p) x) ↑(s.sup fun i => ⟨(p i) x, ⋯⟩)
      ⊢ Eq (((Finset.cons a s ha).sup p) x) ↑((Finset.cons a s ha).sup fun i => ⟨(p  …
    -/
  · rw [Finset.sup_cons, Finset.sup_cons, coe_sup, Pi.sup_apply, NNReal.coe_max, NNReal.coe_mk, ih]
    /-
      🎉 no goals
    -/


theorem exists_apply_eq_finset_sup (p : ι → Seminorm 𝕜 E) {s : Finset ι} (hs : s.Nonempty) (x : E) :
    ∃ i ∈ s, s.sup p x = p i x := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    ι : Type u_11
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → Seminorm 𝕜 E
    s : Finset ι
    hs : s.Nonempty
    x : E
    ⊢ Exists fun i => And (Membership.mem s i) (Eq ((s.sup p) x) ((p i) x))
  -/
  rcases Finset.exists_mem_eq_sup s hs (fun i ↦ (⟨p i x, apply_nonneg _ _⟩ : ℝ≥0)) with ⟨i, hi, hix⟩
  /-
    case intro.intro
    𝕜 : Type u_3
    E : Type u_7
    ι : Type u_11
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → Seminorm 𝕜 E
    s : Finset ι
    hs : s.Nonempty
    x : E
    i : ι
    hi : Membership.mem s i
    hix : Eq (s.sup fun i => ⟨(p i) x, ⋯⟩) ⟨(p i) x, ⋯⟩
    ⊢ Exists fun i => And (Membership.mem s i) (Eq ((s.sup p) x) ((p i) x))
  -/
  rw [finset_sup_apply]
  /-
    case intro.intro
    𝕜 : Type u_3
    E : Type u_7
    ι : Type u_11
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → Seminorm 𝕜 E
    s : Finset ι
    hs : s.Nonempty
    x : E
    i : ι
    hi : Membership.mem s i
    hix : Eq (s.sup fun i => ⟨(p i) x, ⋯⟩) ⟨(p i) x, ⋯⟩
    ⊢ Exists fun i => And (Membership.mem s i) (Eq (↑(s.sup fun i => ⟨(p i) x, ⋯⟩) …
  -/
  exact ⟨i, hi, congr_arg _ hix⟩
  /-
    🎉 no goals
  -/


theorem zero_or_exists_apply_eq_finset_sup (p : ι → Seminorm 𝕜 E) (s : Finset ι) (x : E) :
    s.sup p x = 0 ∨ ∃ i ∈ s, s.sup p x = p i x := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    ι : Type u_11
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → Seminorm 𝕜 E
    s : Finset ι
    x : E
    ⊢ Or (Eq ((s.sup p) x) 0) (Exists fun i => And (Membership.mem s i) (Eq ((s.su …
  -/
  rcases Finset.eq_empty_or_nonempty s with (rfl|hs)
    /-
      case inl
      𝕜 : Type u_3
      E : Type u_7
      ι : Type u_11
      inst✝² : SeminormedRing 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      p : ι → Seminorm 𝕜 E
      x : E
      ⊢ Or (Eq ((EmptyCollection.emptyCollection.sup p) x) 0) (Exists fun i => And ( …
    -/
  · left; rfl
          /-
            🎉 no goals
          -/
    /-
      case inr
      𝕜 : Type u_3
      E : Type u_7
      ι : Type u_11
      inst✝² : SeminormedRing 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      p : ι → Seminorm 𝕜 E
      s : Finset ι
      x : E
      hs : s.Nonempty
      ⊢ Or (Eq ((s.sup p) x) 0) (Exists fun i => And (Membership.mem s i) (Eq ((s.su …
    -/
  · right; exact exists_apply_eq_finset_sup p hs x
           /-
             🎉 no goals
           -/


theorem finset_sup_smul (p : ι → Seminorm 𝕜 E) (s : Finset ι) (C : ℝ≥0) :
    s.sup (C • p) = C • s.sup p := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    ι : Type u_11
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → Seminorm 𝕜 E
    s : Finset ι
    C : NNReal
    ⊢ Eq (s.sup (HSMul.hSMul C p)) (HSMul.hSMul C (s.sup p))
  -/
  ext x
  /-
    case h
    𝕜 : Type u_3
    E : Type u_7
    ι : Type u_11
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → Seminorm 𝕜 E
    s : Finset ι
    C : NNReal
    x : E
    ⊢ Eq ((s.sup (HSMul.hSMul C p)) x) ((HSMul.hSMul C (s.sup p)) x)
  -/
  rw [smul_apply, finset_sup_apply, finset_sup_apply]
  /-
    case h
    𝕜 : Type u_3
    E : Type u_7
    ι : Type u_11
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → Seminorm 𝕜 E
    s : Finset ι
    C : NNReal
    x : E
    ⊢ Eq (↑(s.sup fun i => ⟨(HSMul.hSMul C p i) x, ⋯⟩)) (HSMul.hSMul C ↑(s.sup fun …
  -/
  symm
  /-
    case h
    𝕜 : Type u_3
    E : Type u_7
    ι : Type u_11
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → Seminorm 𝕜 E
    s : Finset ι
    C : NNReal
    x : E
    ⊢ Eq (HSMul.hSMul C ↑(s.sup fun i => ⟨(p i) x, ⋯⟩)) ↑(s.sup fun i => ⟨(HSMul.h …
  -/
  exact congr_arg ((↑) : ℝ≥0 → ℝ) (NNReal.mul_finset_sup C s (fun i ↦ ⟨p i x, apply_nonneg _ _⟩))
  /-
    🎉 no goals
  -/


theorem finset_sup_le_sum (p : ι → Seminorm 𝕜 E) (s : Finset ι) : s.sup p ≤ ∑ i ∈ s, p i := by
  classical
  refine Finset.sup_le_iff.mpr ?_
  intro i hi
  rw [Finset.sum_eq_sum_diff_singleton_add hi, le_add_iff_nonneg_left]
  exact bot_le


theorem finset_sup_apply_le {p : ι → Seminorm 𝕜 E} {s : Finset ι} {x : E} {a : ℝ} (ha : 0 ≤ a)
    (h : ∀ i, i ∈ s → p i x ≤ a) : s.sup p x ≤ a := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    ι : Type u_11
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → Seminorm 𝕜 E
    s : Finset ι
    x : E
    a : Real
    ha : LE.le 0 a
    h : ∀ (i : ι), Membership.mem s i → LE.le ((p i) x) a
    ⊢ LE.le ((s.sup p) x) a
  -/
  lift a to ℝ≥0 using ha
  /-
    case intro
    𝕜 : Type u_3
    E : Type u_7
    ι : Type u_11
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → Seminorm 𝕜 E
    s : Finset ι
    x : E
    a : NNReal
    h : ∀ (i : ι), Membership.mem s i → LE.le ((p i) x) ↑a
    ⊢ LE.le ((s.sup p) x) ↑a
  -/
  rw [finset_sup_apply, NNReal.coe_le_coe]
  /-
    case intro
    𝕜 : Type u_3
    E : Type u_7
    ι : Type u_11
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → Seminorm 𝕜 E
    s : Finset ι
    x : E
    a : NNReal
    h : ∀ (i : ι), Membership.mem s i → LE.le ((p i) x) ↑a
    ⊢ LE.le (s.sup fun i => ⟨(p i) x, ⋯⟩) a
  -/
  exact Finset.sup_le h
  /-
    🎉 no goals
  -/


theorem le_finset_sup_apply {p : ι → Seminorm 𝕜 E} {s : Finset ι} {x : E} {i : ι}
    (hi : i ∈ s) : p i x ≤ s.sup p x :=
  (Finset.le_sup hi : p i ≤ s.sup p) x


theorem finset_sup_apply_lt {p : ι → Seminorm 𝕜 E} {s : Finset ι} {x : E} {a : ℝ} (ha : 0 < a)
    (h : ∀ i, i ∈ s → p i x < a) : s.sup p x < a := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    ι : Type u_11
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → Seminorm 𝕜 E
    s : Finset ι
    x : E
    a : Real
    ha : LT.lt 0 a
    h : ∀ (i : ι), Membership.mem s i → LT.lt ((p i) x) a
    ⊢ LT.lt ((s.sup p) x) a
  -/
  lift a to ℝ≥0 using ha.le
  /-
    case intro
    𝕜 : Type u_3
    E : Type u_7
    ι : Type u_11
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → Seminorm 𝕜 E
    s : Finset ι
    x : E
    a : NNReal
    ha : LT.lt 0 ↑a
    h : ∀ (i : ι), Membership.mem s i → LT.lt ((p i) x) ↑a
    ⊢ LT.lt ((s.sup p) x) ↑a
  -/
  rw [finset_sup_apply, NNReal.coe_lt_coe, Finset.sup_lt_iff]
    /-
      case intro
      𝕜 : Type u_3
      E : Type u_7
      ι : Type u_11
      inst✝² : SeminormedRing 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      p : ι → Seminorm 𝕜 E
      s : Finset ι
      x : E
      a : NNReal
      ha : LT.lt 0 ↑a
      h : ∀ (i : ι), Membership.mem s i → LT.lt ((p i) x) ↑a
      ⊢ ∀ (b : ι), Membership.mem s b → LT.lt ⟨(p b) x, ⋯⟩ a
    -/
  · exact h
    /-
      🎉 no goals
    -/
    /-
      case intro
      𝕜 : Type u_3
      E : Type u_7
      ι : Type u_11
      inst✝² : SeminormedRing 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      p : ι → Seminorm 𝕜 E
      s : Finset ι
      x : E
      a : NNReal
      ha : LT.lt 0 ↑a
      h : ∀ (i : ι), Membership.mem s i → LT.lt ((p i) x) ↑a
      ⊢ LT.lt Bot.bot a
    -/
  · exact NNReal.coe_pos.mpr ha
    /-
      🎉 no goals
    -/


theorem norm_sub_map_le_sub (p : Seminorm 𝕜 E) (x y : E) : ‖p x - p y‖ ≤ p (x - y) :=
  abs_sub_map_le_sub p x y


theorem comp_smul (p : Seminorm 𝕜₂ E₂) (f : E →ₛₗ[σ₁₂] E₂) (c : 𝕜₂) :
    p.comp (c • f) = ‖c‖₊ • p.comp f :=
  ext fun _ => by
    rw [comp_apply, smul_apply, LinearMap.smul_apply, map_smul_eq_mul, NNReal.smul_def, coe_nnnorm,
      smul_eq_mul, comp_apply]


theorem comp_smul_apply (p : Seminorm 𝕜₂ E₂) (f : E →ₛₗ[σ₁₂] E₂) (c : 𝕜₂) (x : E) :
    p.comp (c • f) x = ‖c‖ * p (f x) :=
  map_smul_eq_mul p _ _


/-- Auxiliary lemma to show that the infimum of seminorms is well-defined. -/
theorem bddBelow_range_add : BddBelow (range fun u => p u + q (x - u)) :=
  ⟨0, by
    /-
      𝕜 : Type u_3
      E : Type u_7
      inst✝² : NormedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      p q : Seminorm 𝕜 E
      x : E
      ⊢ Membership.mem (lowerBounds (Set.range fun u => HAdd.hAdd (p u) (q (HSub.hSu …
    -/
    rintro _ ⟨x, rfl⟩
    /-
      case intro
      𝕜 : Type u_3
      E : Type u_7
      inst✝² : NormedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      p q : Seminorm 𝕜 E
      x✝ x : E
      ⊢ LE.le 0 ((fun u => HAdd.hAdd (p u) (q (HSub.hSub x✝ u))) x)
    -/
    dsimp; positivity⟩
           /-
             🎉 no goals
           -/


noncomputable instance instInf : Min (Seminorm 𝕜 E) where
  min p q :=
    { p.toAddGroupSeminorm ⊓ q.toAddGroupSeminorm with
      toFun := fun x => ⨅ u : E, p u + q (x - u)
      smul' := by
        /-
          R : Type u_1
          R' : Type u_2
          𝕜 : Type u_3
          𝕜₂ : Type u_4
          𝕜₃ : Type u_5
          𝕝 : Type u_6
          E : Type u_7
          E₂ : Type u_8
          E₃ : Type u_9
          F : Type u_10
          ι : Type u_11
          inst✝² : NormedField 𝕜
          inst✝¹ : AddCommGroup E
          inst✝ : Module 𝕜 E
          p✝ q✝ : Seminorm 𝕜 E
          x : E
          p q : Seminorm 𝕜 E
          ⊢ ∀ (a : 𝕜) (x : E), Eq ({ toFun := fun x => iInf fun u => HAdd.hAdd (p u) (q  …
        -/
        intro a x
        /-
          R : Type u_1
          R' : Type u_2
          𝕜 : Type u_3
          𝕜₂ : Type u_4
          𝕜₃ : Type u_5
          𝕝 : Type u_6
          E : Type u_7
          E₂ : Type u_8
          E₃ : Type u_9
          F : Type u_10
          ι : Type u_11
          inst✝² : NormedField 𝕜
          inst✝¹ : AddCommGroup E
          inst✝ : Module 𝕜 E
          p✝ q✝ : Seminorm 𝕜 E
          x✝ : E
          p q : Seminorm 𝕜 E
          a : 𝕜
          x : E
          ⊢ Eq ({ toFun := fun x => iInf fun u => HAdd.hAdd (p u) (q (HSub.hSub x u)), m …
        -/
        obtain rfl | ha := eq_or_ne a 0
          /-
            case inl
            R : Type u_1
            R' : Type u_2
            𝕜 : Type u_3
            𝕜₂ : Type u_4
            𝕜₃ : Type u_5
            𝕝 : Type u_6
            E : Type u_7
            E₂ : Type u_8
            E₃ : Type u_9
            F : Type u_10
            ι : Type u_11
            inst✝² : NormedField 𝕜
            inst✝¹ : AddCommGroup E
            inst✝ : Module 𝕜 E
            p✝ q✝ : Seminorm 𝕜 E
            x✝ : E
            p q : Seminorm 𝕜 E
            x : E
            ⊢ Eq ({ toFun := fun x => iInf fun u => HAdd.hAdd (p u) (q (HSub.hSub x u)), m …
          -/
        · rw [norm_zero, zero_mul, zero_smul]
          refine
            ciInf_eq_of_forall_ge_of_forall_gt_exists_lt
              -- Porting note: the following was previously `fun i => by positivity`
              (fun i => add_nonneg (apply_nonneg _ _) (apply_nonneg _ _))
              fun x hx => ⟨0, by rwa [map_zero, sub_zero, map_zero, add_zero]⟩
        simp_rw [Real.mul_iInf_of_nonneg (norm_nonneg a), mul_add, ← map_smul_eq_mul p, ←
          map_smul_eq_mul q, smul_sub]
        refine
          Function.Surjective.iInf_congr ((a⁻¹ • ·) : E → E)
            (fun u => ⟨a • u, inv_smul_smul₀ ha u⟩) fun u => ?_
        /-
          case inr
          R : Type u_1
          R' : Type u_2
          𝕜 : Type u_3
          𝕜₂ : Type u_4
          𝕜₃ : Type u_5
          𝕝 : Type u_6
          E : Type u_7
          E₂ : Type u_8
          E₃ : Type u_9
          F : Type u_10
          ι : Type u_11
          inst✝² : NormedField 𝕜
          inst✝¹ : AddCommGroup E
          inst✝ : Module 𝕜 E
          p✝ q✝ : Seminorm 𝕜 E
          x✝ : E
          p q : Seminorm 𝕜 E
          a : 𝕜
          x : E
          ha : Ne a 0
          u : E
          ⊢ Eq (HAdd.hAdd (p (HSMul.hSMul a ((fun x => HSMul.hSMul (Inv.inv a) x) u))) ( …
        -/
        rw [smul_inv_smul₀ ha] }
        /-
          🎉 no goals
        -/


@[simp]
theorem inf_apply (p q : Seminorm 𝕜 E) (x : E) : (p ⊓ q) x = ⨅ u : E, p u + q (x - u) :=
  rfl


noncomputable instance instLattice : Lattice (Seminorm 𝕜 E) :=
  { Seminorm.instSemilatticeSup with
    inf := (· ⊓ ·)
    inf_le_left := fun p q x =>
      ciInf_le_of_le bddBelow_range_add x <| by
        /-
          R : Type u_1
          R' : Type u_2
          𝕜 : Type u_3
          𝕜₂ : Type u_4
          𝕜₃ : Type u_5
          𝕝 : Type u_6
          E : Type u_7
          E₂ : Type u_8
          E₃ : Type u_9
          F : Type u_10
          ι : Type u_11
          inst✝² : NormedField 𝕜
          inst✝¹ : AddCommGroup E
          inst✝ : Module 𝕜 E
          p✝ q✝ : Seminorm 𝕜 E
          x✝ : E
          p q : Seminorm 𝕜 E
          x : E
          ⊢ LE.le (HAdd.hAdd (p x) (q (HSub.hSub x x))) ((fun f => ⇑f) p x)
        -/
        simp only [sub_self, map_zero, add_zero]; rfl
                                                  /-
                                                    🎉 no goals
                                                  -/
    inf_le_right := fun p q x =>
      ciInf_le_of_le bddBelow_range_add 0 <| by
        /-
          R : Type u_1
          R' : Type u_2
          𝕜 : Type u_3
          𝕜₂ : Type u_4
          𝕜₃ : Type u_5
          𝕝 : Type u_6
          E : Type u_7
          E₂ : Type u_8
          E₃ : Type u_9
          F : Type u_10
          ι : Type u_11
          inst✝² : NormedField 𝕜
          inst✝¹ : AddCommGroup E
          inst✝ : Module 𝕜 E
          p✝ q✝ : Seminorm 𝕜 E
          x✝ : E
          p q : Seminorm 𝕜 E
          x : E
          ⊢ LE.le (HAdd.hAdd (p 0) (q (HSub.hSub x 0))) ((fun f => ⇑f) q x)
        -/
        simp only [sub_self, map_zero, zero_add, sub_zero]; rfl
                                                            /-
                                                              🎉 no goals
                                                            -/
    le_inf := fun a _ _ hab hac _ =>
      le_ciInf fun _ => (le_map_add_map_sub a _ _).trans <| add_le_add (hab _) (hac _) }


theorem smul_inf [SMul R ℝ] [SMul R ℝ≥0] [IsScalarTower R ℝ≥0 ℝ] (r : R) (p q : Seminorm 𝕜 E) :
    r • (p ⊓ q) = r • p ⊓ r • q := by
  /-
    R : Type u_1
    𝕜 : Type u_3
    E : Type u_7
    inst✝⁵ : NormedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : SMul R Real
    inst✝¹ : SMul R NNReal
    inst✝ : IsScalarTower R NNReal Real
    r : R
    p q : Seminorm 𝕜 E
    ⊢ Eq (HSMul.hSMul r (Min.min p q)) (Min.min (HSMul.hSMul r p) (HSMul.hSMul r q))
  -/
  ext
  simp_rw [smul_apply, inf_apply, smul_apply, ← smul_one_smul ℝ≥0 r (_ : ℝ), NNReal.smul_def,
    smul_eq_mul, Real.mul_iInf_of_nonneg (NNReal.coe_nonneg _), mul_add]


open Classical in
/-- We define the supremum of an arbitrary subset of `Seminorm 𝕜 E` as follows:
* if `s` is `BddAbove` *as a set of functions `E → ℝ`* (that is, if `s` is pointwise bounded
above), we take the pointwise supremum of all elements of `s`, and we prove that it is indeed a
seminorm.
* otherwise, we take the zero seminorm `⊥`.

There are two things worth mentioning here:
* First, it is not trivial at first that `s` being bounded above *by a function* implies
being bounded above *as a seminorm*. We show this in `Seminorm.bddAbove_iff` by using
that the `Sup s` as defined here is then a bounding seminorm for `s`. So it is important to make
the case disjunction on `BddAbove ((↑) '' s : Set (E → ℝ))` and not `BddAbove s`.
* Since the pointwise `Sup` already gives `0` at points where a family of functions is
not bounded above, one could hope that just using the pointwise `Sup` would work here, without the
need for an additional case disjunction. As discussed on Zulip, this doesn't work because this can
give a function which does *not* satisfy the seminorm axioms (typically sub-additivity).
-/
noncomputable instance instSupSet : SupSet (Seminorm 𝕜 E) where
  sSup s :=
    if h : BddAbove ((↑) '' s : Set (E → ℝ)) then
      { toFun := ⨆ p : s, ((p : Seminorm 𝕜 E) : E → ℝ)
        map_zero' := by
          /-
            R : Type u_1
            R' : Type u_2
            𝕜 : Type u_3
            𝕜₂ : Type u_4
            𝕜₃ : Type u_5
            𝕝 : Type u_6
            E : Type u_7
            E₂ : Type u_8
            E₃ : Type u_9
            F : Type u_10
            ι : Type u_11
            inst✝² : NormedField 𝕜
            inst✝¹ : AddCommGroup E
            inst✝ : Module 𝕜 E
            p q : Seminorm 𝕜 E
            x : E
            s : Set (Seminorm 𝕜 E)
            h : BddAbove (Set.image DFunLike.coe s)
            ⊢ Eq (iSup (fun p => ⇑↑p) 0) 0
          -/
          rw [iSup_apply, ← @Real.iSup_const_zero s]
          /-
            R : Type u_1
            R' : Type u_2
            𝕜 : Type u_3
            𝕜₂ : Type u_4
            𝕜₃ : Type u_5
            𝕝 : Type u_6
            E : Type u_7
            E₂ : Type u_8
            E₃ : Type u_9
            F : Type u_10
            ι : Type u_11
            inst✝² : NormedField 𝕜
            inst✝¹ : AddCommGroup E
            inst✝ : Module 𝕜 E
            p q : Seminorm 𝕜 E
            x : E
            s : Set (Seminorm 𝕜 E)
            h : BddAbove (Set.image DFunLike.coe s)
            ⊢ Eq (iSup fun i => ↑i 0) (iSup fun x => 0)
          -/
          congr!
          /-
            case h.e'_4.h
            R : Type u_1
            R' : Type u_2
            𝕜 : Type u_3
            𝕜₂ : Type u_4
            𝕜₃ : Type u_5
            𝕝 : Type u_6
            E : Type u_7
            E₂ : Type u_8
            E₃ : Type u_9
            F : Type u_10
            ι : Type u_11
            inst✝² : NormedField 𝕜
            inst✝¹ : AddCommGroup E
            inst✝ : Module 𝕜 E
            p q : Seminorm 𝕜 E
            x : E
            s : Set (Seminorm 𝕜 E)
            h : BddAbove (Set.image DFunLike.coe s)
            x✝ : ↑s
            ⊢ Eq (↑x✝ 0) 0
          -/
          rename_i _ _ _ i
          /-
            case h.e'_4.h
            R : Type u_1
            R' : Type u_2
            𝕜 : Type u_3
            𝕜₂ : Type u_4
            𝕜₃ : Type u_5
            𝕝 : Type u_6
            E : Type u_7
            E₂ : Type u_8
            E₃ : Type u_9
            F : Type u_10
            ι : Type u_11
            inst✝² : NormedField 𝕜
            inst✝¹ : AddCommGroup E
            inst✝ : Module 𝕜 E
            p q : Seminorm 𝕜 E
            x : E
            s : Set (Seminorm 𝕜 E)
            h : BddAbove (Set.image DFunLike.coe s)
            i : ↑s
            ⊢ Eq (↑i 0) 0
          -/
          exact map_zero i.1
          /-
            🎉 no goals
          -/
        add_le' := fun x y => by
          /-
            R : Type u_1
            R' : Type u_2
            𝕜 : Type u_3
            𝕜₂ : Type u_4
            𝕜₃ : Type u_5
            𝕝 : Type u_6
            E : Type u_7
            E₂ : Type u_8
            E₃ : Type u_9
            F : Type u_10
            ι : Type u_11
            inst✝² : NormedField 𝕜
            inst✝¹ : AddCommGroup E
            inst✝ : Module 𝕜 E
            p q : Seminorm 𝕜 E
            x✝ : E
            s : Set (Seminorm 𝕜 E)
            h : BddAbove (Set.image DFunLike.coe s)
            x y : E
            ⊢ LE.le (iSup (fun p => ⇑↑p) (HAdd.hAdd x y)) (HAdd.hAdd (iSup (fun p => ⇑↑p)  …
          -/
          rcases h with ⟨q, hq⟩
          /-
            case intro
            R : Type u_1
            R' : Type u_2
            𝕜 : Type u_3
            𝕜₂ : Type u_4
            𝕜₃ : Type u_5
            𝕝 : Type u_6
            E : Type u_7
            E₂ : Type u_8
            E₃ : Type u_9
            F : Type u_10
            ι : Type u_11
            inst✝² : NormedField 𝕜
            inst✝¹ : AddCommGroup E
            inst✝ : Module 𝕜 E
            p q✝ : Seminorm 𝕜 E
            x✝ : E
            s : Set (Seminorm 𝕜 E)
            x y : E
            q : E → Real
            hq : Membership.mem (upperBounds (Set.image DFunLike.coe s)) q
            ⊢ LE.le (iSup (fun p => ⇑↑p) (HAdd.hAdd x y)) (HAdd.hAdd (iSup (fun p => ⇑↑p)  …
          -/
          obtain rfl | h := s.eq_empty_or_nonempty
            /-
              case intro.inl
              R : Type u_1
              R' : Type u_2
              𝕜 : Type u_3
              𝕜₂ : Type u_4
              𝕜₃ : Type u_5
              𝕝 : Type u_6
              E : Type u_7
              E₂ : Type u_8
              E₃ : Type u_9
              F : Type u_10
              ι : Type u_11
              inst✝² : NormedField 𝕜
              inst✝¹ : AddCommGroup E
              inst✝ : Module 𝕜 E
              p q✝ : Seminorm 𝕜 E
              x✝ x y : E
              q : E → Real
              hq : Membership.mem (upperBounds (Set.image DFunLike.coe EmptyCollection.empty …
              ⊢ LE.le (iSup (fun p => ⇑↑p) (HAdd.hAdd x y)) (HAdd.hAdd (iSup (fun p => ⇑↑p)  …
            -/
          · simp [Real.iSup_of_isEmpty]
            /-
              🎉 no goals
            -/
          /-
            case intro.inr
            R : Type u_1
            R' : Type u_2
            𝕜 : Type u_3
            𝕜₂ : Type u_4
            𝕜₃ : Type u_5
            𝕝 : Type u_6
            E : Type u_7
            E₂ : Type u_8
            E₃ : Type u_9
            F : Type u_10
            ι : Type u_11
            inst✝² : NormedField 𝕜
            inst✝¹ : AddCommGroup E
            inst✝ : Module 𝕜 E
            p q✝ : Seminorm 𝕜 E
            x✝ : E
            s : Set (Seminorm 𝕜 E)
            x y : E
            q : E → Real
            hq : Membership.mem (upperBounds (Set.image DFunLike.coe s)) q
            h : s.Nonempty
            ⊢ LE.le (iSup (fun p => ⇑↑p) (HAdd.hAdd x y)) (HAdd.hAdd (iSup (fun p => ⇑↑p)  …
          -/
          haveI : Nonempty ↑s := h.coe_sort
          /-
            case intro.inr
            R : Type u_1
            R' : Type u_2
            𝕜 : Type u_3
            𝕜₂ : Type u_4
            𝕜₃ : Type u_5
            𝕝 : Type u_6
            E : Type u_7
            E₂ : Type u_8
            E₃ : Type u_9
            F : Type u_10
            ι : Type u_11
            inst✝² : NormedField 𝕜
            inst✝¹ : AddCommGroup E
            inst✝ : Module 𝕜 E
            p q✝ : Seminorm 𝕜 E
            x✝ : E
            s : Set (Seminorm 𝕜 E)
            x y : E
            q : E → Real
            hq : Membership.mem (upperBounds (Set.image DFunLike.coe s)) q
            h : s.Nonempty
            this : Nonempty ↑s
            ⊢ LE.le (iSup (fun p => ⇑↑p) (HAdd.hAdd x y)) (HAdd.hAdd (iSup (fun p => ⇑↑p)  …
          -/
          simp only [iSup_apply]
          refine ciSup_le fun i =>
            ((i : Seminorm 𝕜 E).add_le' x y).trans <| add_le_add
              -- Porting note: `f` is provided to force `Subtype.val` to appear.
              -- A type ascription on `_` would have also worked, but would have been more verbose.
              (le_ciSup (f := fun i => (Subtype.val i : Seminorm 𝕜 E).toFun x) ⟨q x, ?_⟩ i)
              (le_ciSup (f := fun i => (Subtype.val i : Seminorm 𝕜 E).toFun y) ⟨q y, ?_⟩ i)
              /-
                case intro.inr.refine_1
                R : Type u_1
                R' : Type u_2
                𝕜 : Type u_3
                𝕜₂ : Type u_4
                𝕜₃ : Type u_5
                𝕝 : Type u_6
                E : Type u_7
                E₂ : Type u_8
                E₃ : Type u_9
                F : Type u_10
                ι : Type u_11
                inst✝² : NormedField 𝕜
                inst✝¹ : AddCommGroup E
                inst✝ : Module 𝕜 E
                p q✝ : Seminorm 𝕜 E
                x✝ : E
                s : Set (Seminorm 𝕜 E)
                x y : E
                q : E → Real
                hq : Membership.mem (upperBounds (Set.image DFunLike.coe s)) q
                h : s.Nonempty
                this : Nonempty ↑s
                i : ↑s
                ⊢ Membership.mem (upperBounds (Set.range fun i => (↑i).toFun x)) (q x)
              -/
          <;> rw [mem_upperBounds, forall_mem_range]
              /-
                case intro.inr.refine_1
                R : Type u_1
                R' : Type u_2
                𝕜 : Type u_3
                𝕜₂ : Type u_4
                𝕜₃ : Type u_5
                𝕝 : Type u_6
                E : Type u_7
                E₂ : Type u_8
                E₃ : Type u_9
                F : Type u_10
                ι : Type u_11
                inst✝² : NormedField 𝕜
                inst✝¹ : AddCommGroup E
                inst✝ : Module 𝕜 E
                p q✝ : Seminorm 𝕜 E
                x✝ : E
                s : Set (Seminorm 𝕜 E)
                x y : E
                q : E → Real
                hq : Membership.mem (upperBounds (Set.image DFunLike.coe s)) q
                h : s.Nonempty
                this : Nonempty ↑s
                i : ↑s
                ⊢ ∀ (i : Subtype fun x => Membership.mem s x), LE.le ((↑i).toFun x) (q x)
              -/
              /-
                🎉 no goals
              -/
          <;> exact fun j => hq (mem_image_of_mem _ j.2) _
              /-
                🎉 no goals
              -/
        neg' := fun x => by
          /-
            R : Type u_1
            R' : Type u_2
            𝕜 : Type u_3
            𝕜₂ : Type u_4
            𝕜₃ : Type u_5
            𝕝 : Type u_6
            E : Type u_7
            E₂ : Type u_8
            E₃ : Type u_9
            F : Type u_10
            ι : Type u_11
            inst✝² : NormedField 𝕜
            inst✝¹ : AddCommGroup E
            inst✝ : Module 𝕜 E
            p q : Seminorm 𝕜 E
            x✝ : E
            s : Set (Seminorm 𝕜 E)
            h : BddAbove (Set.image DFunLike.coe s)
            x : E
            ⊢ Eq (iSup (fun p => ⇑↑p) (Neg.neg x)) (iSup (fun p => ⇑↑p) x)
          -/
          simp only [iSup_apply]
          /-
            R : Type u_1
            R' : Type u_2
            𝕜 : Type u_3
            𝕜₂ : Type u_4
            𝕜₃ : Type u_5
            𝕝 : Type u_6
            E : Type u_7
            E₂ : Type u_8
            E₃ : Type u_9
            F : Type u_10
            ι : Type u_11
            inst✝² : NormedField 𝕜
            inst✝¹ : AddCommGroup E
            inst✝ : Module 𝕜 E
            p q : Seminorm 𝕜 E
            x✝ : E
            s : Set (Seminorm 𝕜 E)
            h : BddAbove (Set.image DFunLike.coe s)
            x : E
            ⊢ Eq (iSup fun i => ↑i (Neg.neg x)) (iSup fun i => ↑i x)
          -/
          congr! 2
          /-
            case h.e'_4.h
            R : Type u_1
            R' : Type u_2
            𝕜 : Type u_3
            𝕜₂ : Type u_4
            𝕜₃ : Type u_5
            𝕝 : Type u_6
            E : Type u_7
            E₂ : Type u_8
            E₃ : Type u_9
            F : Type u_10
            ι : Type u_11
            inst✝² : NormedField 𝕜
            inst✝¹ : AddCommGroup E
            inst✝ : Module 𝕜 E
            p q : Seminorm 𝕜 E
            x✝¹ : E
            s : Set (Seminorm 𝕜 E)
            h : BddAbove (Set.image DFunLike.coe s)
            x : E
            x✝ : ↑s
            ⊢ Eq (↑x✝ (Neg.neg x)) (↑x✝ x)
          -/
          rename_i _ _ _ i
          /-
            case h.e'_4.h
            R : Type u_1
            R' : Type u_2
            𝕜 : Type u_3
            𝕜₂ : Type u_4
            𝕜₃ : Type u_5
            𝕝 : Type u_6
            E : Type u_7
            E₂ : Type u_8
            E₃ : Type u_9
            F : Type u_10
            ι : Type u_11
            inst✝² : NormedField 𝕜
            inst✝¹ : AddCommGroup E
            inst✝ : Module 𝕜 E
            p q : Seminorm 𝕜 E
            x✝ : E
            s : Set (Seminorm 𝕜 E)
            h : BddAbove (Set.image DFunLike.coe s)
            x : E
            i : ↑s
            ⊢ Eq (↑i (Neg.neg x)) (↑i x)
          -/
          exact i.1.neg' _
          /-
            🎉 no goals
          -/
        smul' := fun a x => by
          /-
            R : Type u_1
            R' : Type u_2
            𝕜 : Type u_3
            𝕜₂ : Type u_4
            𝕜₃ : Type u_5
            𝕝 : Type u_6
            E : Type u_7
            E₂ : Type u_8
            E₃ : Type u_9
            F : Type u_10
            ι : Type u_11
            inst✝² : NormedField 𝕜
            inst✝¹ : AddCommGroup E
            inst✝ : Module 𝕜 E
            p q : Seminorm 𝕜 E
            x✝ : E
            s : Set (Seminorm 𝕜 E)
            h : BddAbove (Set.image DFunLike.coe s)
            a : 𝕜
            x : E
            ⊢ Eq ({ toFun := iSup fun p => ⇑↑p, map_zero' := ⋯, add_le' := ⋯, neg' := ⋯ }. …
          -/
          simp only [iSup_apply]
          rw [← smul_eq_mul,
            Real.smul_iSup_of_nonneg (norm_nonneg a) fun i : s => (i : Seminorm 𝕜 E) x]
          /-
            R : Type u_1
            R' : Type u_2
            𝕜 : Type u_3
            𝕜₂ : Type u_4
            𝕜₃ : Type u_5
            𝕝 : Type u_6
            E : Type u_7
            E₂ : Type u_8
            E₃ : Type u_9
            F : Type u_10
            ι : Type u_11
            inst✝² : NormedField 𝕜
            inst✝¹ : AddCommGroup E
            inst✝ : Module 𝕜 E
            p q : Seminorm 𝕜 E
            x✝ : E
            s : Set (Seminorm 𝕜 E)
            h : BddAbove (Set.image DFunLike.coe s)
            a : 𝕜
            x : E
            ⊢ Eq (iSup fun i => ↑i (HSMul.hSMul a x)) (iSup fun i => HSMul.hSMul (Norm.nor …
          -/
          congr!
          /-
            case h.e'_4.h
            R : Type u_1
            R' : Type u_2
            𝕜 : Type u_3
            𝕜₂ : Type u_4
            𝕜₃ : Type u_5
            𝕝 : Type u_6
            E : Type u_7
            E₂ : Type u_8
            E₃ : Type u_9
            F : Type u_10
            ι : Type u_11
            inst✝² : NormedField 𝕜
            inst✝¹ : AddCommGroup E
            inst✝ : Module 𝕜 E
            p q : Seminorm 𝕜 E
            x✝¹ : E
            s : Set (Seminorm 𝕜 E)
            h : BddAbove (Set.image DFunLike.coe s)
            a : 𝕜
            x : E
            x✝ : ↑s
            ⊢ Eq (↑x✝ (HSMul.hSMul a x)) (HSMul.hSMul (Norm.norm a) (↑x✝ x))
          -/
          rename_i _ _ _ i
          /-
            case h.e'_4.h
            R : Type u_1
            R' : Type u_2
            𝕜 : Type u_3
            𝕜₂ : Type u_4
            𝕜₃ : Type u_5
            𝕝 : Type u_6
            E : Type u_7
            E₂ : Type u_8
            E₃ : Type u_9
            F : Type u_10
            ι : Type u_11
            inst✝² : NormedField 𝕜
            inst✝¹ : AddCommGroup E
            inst✝ : Module 𝕜 E
            p q : Seminorm 𝕜 E
            x✝ : E
            s : Set (Seminorm 𝕜 E)
            h : BddAbove (Set.image DFunLike.coe s)
            a : 𝕜
            x : E
            i : ↑s
            ⊢ Eq (↑i (HSMul.hSMul a x)) (HSMul.hSMul (Norm.norm a) (↑i x))
          -/
          exact i.1.smul' a x }
          /-
            🎉 no goals
          -/
    else ⊥


protected theorem coe_sSup_eq' {s : Set <| Seminorm 𝕜 E}
    (hs : BddAbove ((↑) '' s : Set (E → ℝ))) : ↑(sSup s) = ⨆ p : s, ((p : Seminorm 𝕜 E) : E → ℝ) :=
  congr_arg _ (dif_pos hs)


protected theorem bddAbove_iff {s : Set <| Seminorm 𝕜 E} :
    BddAbove s ↔ BddAbove ((↑) '' s : Set (E → ℝ)) :=
  ⟨fun ⟨q, hq⟩ => ⟨q, forall_mem_image.2 fun _ hp => hq hp⟩, fun H =>
    ⟨sSup s, fun p hp x => by
      /-
        𝕜 : Type u_3
        E : Type u_7
        inst✝² : NormedField 𝕜
        inst✝¹ : AddCommGroup E
        inst✝ : Module 𝕜 E
        s : Set (Seminorm 𝕜 E)
        H : BddAbove (Set.image DFunLike.coe s)
        p : Seminorm 𝕜 E
        hp : Membership.mem s p
        x : E
        ⊢ LE.le ((fun f => ⇑f) p x) ((fun f => ⇑f) (SupSet.sSup s) x)
      -/
      dsimp
      /-
        𝕜 : Type u_3
        E : Type u_7
        inst✝² : NormedField 𝕜
        inst✝¹ : AddCommGroup E
        inst✝ : Module 𝕜 E
        s : Set (Seminorm 𝕜 E)
        H : BddAbove (Set.image DFunLike.coe s)
        p : Seminorm 𝕜 E
        hp : Membership.mem s p
        x : E
        ⊢ LE.le (p x) ((SupSet.sSup s) x)
      -/
      rw [Seminorm.coe_sSup_eq' H, iSup_apply]
      /-
        𝕜 : Type u_3
        E : Type u_7
        inst✝² : NormedField 𝕜
        inst✝¹ : AddCommGroup E
        inst✝ : Module 𝕜 E
        s : Set (Seminorm 𝕜 E)
        H : BddAbove (Set.image DFunLike.coe s)
        p : Seminorm 𝕜 E
        hp : Membership.mem s p
        x : E
        ⊢ LE.le (p x) (iSup fun i => ↑i x)
      -/
      rcases H with ⟨q, hq⟩
      exact
        le_ciSup ⟨q x, forall_mem_range.mpr fun i : s => hq (mem_image_of_mem _ i.2) x⟩ ⟨p, hp⟩⟩⟩


protected theorem bddAbove_range_iff {ι : Sort*} {p : ι → Seminorm 𝕜 E} :
    BddAbove (range p) ↔ ∀ x, BddAbove (range fun i ↦ p i x) := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    ι : Sort u_12
    p : ι → Seminorm 𝕜 E
    ⊢ Iff (BddAbove (Set.range p)) (∀ (x : E), BddAbove (Set.range fun i => (p i)  …
  -/
  rw [Seminorm.bddAbove_iff, ← range_comp, bddAbove_range_pi]; rfl
                                                               /-
                                                                 🎉 no goals
                                                               -/


protected theorem coe_sSup_eq {s : Set <| Seminorm 𝕜 E} (hs : BddAbove s) :
    ↑(sSup s) = ⨆ p : s, ((p : Seminorm 𝕜 E) : E → ℝ) :=
  Seminorm.coe_sSup_eq' (Seminorm.bddAbove_iff.mp hs)


protected theorem coe_iSup_eq {ι : Sort*} {p : ι → Seminorm 𝕜 E} (hp : BddAbove (range p)) :
    ↑(⨆ i, p i) = ⨆ i, ((p i : Seminorm 𝕜 E) : E → ℝ) := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    ι : Sort u_12
    p : ι → Seminorm 𝕜 E
    hp : BddAbove (Set.range p)
    ⊢ Eq (⇑(iSup fun i => p i)) (iSup fun i => ⇑(p i))
  -/
  rw [← sSup_range, Seminorm.coe_sSup_eq hp]
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    ι : Sort u_12
    p : ι → Seminorm 𝕜 E
    hp : BddAbove (Set.range p)
    ⊢ Eq (iSup fun p_1 => ⇑↑p_1) (iSup fun i => ⇑(p i))
  -/
  exact iSup_range' (fun p : Seminorm 𝕜 E => (p : E → ℝ)) p
  /-
    🎉 no goals
  -/


protected theorem sSup_apply {s : Set (Seminorm 𝕜 E)} (hp : BddAbove s) {x : E} :
    (sSup s) x = ⨆ p : s, (p : E → ℝ) x := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set (Seminorm 𝕜 E)
    hp : BddAbove s
    x : E
    ⊢ Eq ((SupSet.sSup s) x) (iSup fun p => ↑p x)
  -/
  rw [Seminorm.coe_sSup_eq hp, iSup_apply]
  /-
    🎉 no goals
  -/


protected theorem iSup_apply {ι : Sort*} {p : ι → Seminorm 𝕜 E}
    (hp : BddAbove (range p)) {x : E} : (⨆ i, p i) x = ⨆ i, p i x := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    ι : Sort u_12
    p : ι → Seminorm 𝕜 E
    hp : BddAbove (Set.range p)
    x : E
    ⊢ Eq ((iSup fun i => p i) x) (iSup fun i => (p i) x)
  -/
  rw [Seminorm.coe_iSup_eq hp, iSup_apply]
  /-
    🎉 no goals
  -/


protected theorem sSup_empty : sSup (∅ : Set (Seminorm 𝕜 E)) = ⊥ := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    ⊢ Eq (SupSet.sSup EmptyCollection.emptyCollection) Bot.bot
  -/
  ext
  /-
    case h
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    x✝ : E
    ⊢ Eq ((SupSet.sSup EmptyCollection.emptyCollection) x✝) (Bot.bot x✝)
  -/
  rw [Seminorm.sSup_apply bddAbove_empty, Real.iSup_of_isEmpty]
  /-
    case h
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    x✝ : E
    ⊢ Eq 0 (Bot.bot x✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


private theorem isLUB_sSup (s : Set (Seminorm 𝕜 E)) (hs₁ : BddAbove s) (hs₂ : s.Nonempty) :
    IsLUB s (sSup s) := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set (Seminorm 𝕜 E)
    hs₁ : BddAbove s
    hs₂ : s.Nonempty
    ⊢ IsLUB s (SupSet.sSup s)
  -/
  refine ⟨fun p hp x => ?_, fun p hp x => ?_⟩ <;> haveI : Nonempty ↑s := hs₂.coe_sort <;>
    /-
      case refine_1
      𝕜 : Type u_3
      E : Type u_7
      inst✝² : NormedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set (Seminorm 𝕜 E)
      hs₁ : BddAbove s
      hs₂ : s.Nonempty
      p : Seminorm 𝕜 E
      hp : Membership.mem s p
      x : E
      this : Nonempty ↑s
      ⊢ LE.le ((fun f => ⇑f) p x) ((fun f => ⇑f) (SupSet.sSup s) x)
    -/
    dsimp <;> rw [Seminorm.coe_sSup_eq hs₁, iSup_apply]
    /-
      case refine_1
      𝕜 : Type u_3
      E : Type u_7
      inst✝² : NormedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set (Seminorm 𝕜 E)
      hs₁ : BddAbove s
      hs₂ : s.Nonempty
      p : Seminorm 𝕜 E
      hp : Membership.mem s p
      x : E
      this : Nonempty ↑s
      ⊢ LE.le (p x) (iSup fun i => ↑i x)
    -/
  · rcases hs₁ with ⟨q, hq⟩
    /-
      case refine_1.intro
      𝕜 : Type u_3
      E : Type u_7
      inst✝² : NormedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set (Seminorm 𝕜 E)
      hs₂ : s.Nonempty
      p : Seminorm 𝕜 E
      hp : Membership.mem s p
      x : E
      this : Nonempty ↑s
      q : Seminorm 𝕜 E
      hq : Membership.mem (upperBounds s) q
      ⊢ LE.le (p x) (iSup fun i => ↑i x)
    -/
    exact le_ciSup ⟨q x, forall_mem_range.mpr fun i : s => hq i.2 x⟩ ⟨p, hp⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      𝕜 : Type u_3
      E : Type u_7
      inst✝² : NormedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set (Seminorm 𝕜 E)
      hs₁ : BddAbove s
      hs₂ : s.Nonempty
      p : Seminorm 𝕜 E
      hp : Membership.mem (upperBounds s) p
      x : E
      this : Nonempty ↑s
      ⊢ LE.le (iSup fun i => ↑i x) (p x)
    -/
  · exact ciSup_le fun q => hp q.2 x
    /-
      🎉 no goals
    -/


/-- `Seminorm 𝕜 E` is a conditionally complete lattice.

Note that, while `inf`, `sup` and `sSup` have good definitional properties (corresponding to
the instances given here for `Inf`, `Sup` and `SupSet` respectively), `sInf s` is just
defined as the supremum of the lower bounds of `s`, which is not really useful in practice. If you
need to use `sInf` on seminorms, then you should probably provide a more workable definition first,
but this is unlikely to happen so we keep the "bad" definition for now. -/
noncomputable instance instConditionallyCompleteLattice :
    ConditionallyCompleteLattice (Seminorm 𝕜 E) :=
  conditionallyCompleteLatticeOfLatticeOfsSup (Seminorm 𝕜 E) Seminorm.isLUB_sSup


/-- The ball of radius `r` at `x` with respect to seminorm `p` is the set of elements `y` with
`p (y - x) < r`. -/
def ball (x : E) (r : ℝ) :=
  { y : E | p (y - x) < r }


/-- The closed ball of radius `r` at `x` with respect to seminorm `p` is the set of elements `y`
with `p (y - x) ≤ r`. -/
def closedBall (x : E) (r : ℝ) :=
  { y : E | p (y - x) ≤ r }


@[simp]
theorem mem_ball : y ∈ ball p x r ↔ p (y - x) < r :=
  Iff.rfl


@[simp]
theorem mem_closedBall : y ∈ closedBall p x r ↔ p (y - x) ≤ r :=
  Iff.rfl


                                                          /-
                                                            𝕜 : Type u_3
                                                            E : Type u_7
                                                            inst✝² : SeminormedRing 𝕜
                                                            inst✝¹ : AddCommGroup E
                                                            inst✝ : SMul 𝕜 E
                                                            p : Seminorm 𝕜 E
                                                            x : E
                                                            r : Real
                                                            hr : LT.lt 0 r
                                                            ⊢ Membership.mem (p.ball x r) x
                                                          -/
theorem mem_ball_self (hr : 0 < r) : x ∈ ball p x r := by simp [hr]
                                                          /-
                                                            🎉 no goals
                                                          -/


                                                                      /-
                                                                        𝕜 : Type u_3
                                                                        E : Type u_7
                                                                        inst✝² : SeminormedRing 𝕜
                                                                        inst✝¹ : AddCommGroup E
                                                                        inst✝ : SMul 𝕜 E
                                                                        p : Seminorm 𝕜 E
                                                                        x : E
                                                                        r : Real
                                                                        hr : LE.le 0 r
                                                                        ⊢ Membership.mem (p.closedBall x r) x
                                                                      -/
theorem mem_closedBall_self (hr : 0 ≤ r) : x ∈ closedBall p x r := by simp [hr]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


                                                       /-
                                                         𝕜 : Type u_3
                                                         E : Type u_7
                                                         inst✝² : SeminormedRing 𝕜
                                                         inst✝¹ : AddCommGroup E
                                                         inst✝ : SMul 𝕜 E
                                                         p : Seminorm 𝕜 E
                                                         y : E
                                                         r : Real
                                                         ⊢ Iff (Membership.mem (p.ball 0 r) y) (LT.lt (p y) r)
                                                       -/
theorem mem_ball_zero : y ∈ ball p 0 r ↔ p y < r := by rw [mem_ball, sub_zero]
                                                       /-
                                                         🎉 no goals
                                                       -/


                                                                   /-
                                                                     𝕜 : Type u_3
                                                                     E : Type u_7
                                                                     inst✝² : SeminormedRing 𝕜
                                                                     inst✝¹ : AddCommGroup E
                                                                     inst✝ : SMul 𝕜 E
                                                                     p : Seminorm 𝕜 E
                                                                     y : E
                                                                     r : Real
                                                                     ⊢ Iff (Membership.mem (p.closedBall 0 r) y) (LE.le (p y) r)
                                                                   -/
theorem mem_closedBall_zero : y ∈ closedBall p 0 r ↔ p y ≤ r := by rw [mem_closedBall, sub_zero]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


theorem ball_zero_eq : ball p 0 r = { y : E | p y < r } :=
  Set.ext fun _ => p.mem_ball_zero


theorem closedBall_zero_eq : closedBall p 0 r = { y : E | p y ≤ r } :=
  Set.ext fun _ => p.mem_closedBall_zero


theorem ball_subset_closedBall (x r) : ball p x r ⊆ closedBall p x r := fun _ h =>
  (mem_closedBall _).mpr ((mem_ball _).mp h).le


theorem closedBall_eq_biInter_ball (x r) : closedBall p x r = ⋂ ρ > r, ball p x ρ := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : SMul 𝕜 E
    p : Seminorm 𝕜 E
    x : E
    r : Real
    ⊢ Eq (p.closedBall x r) (Set.iInter fun ρ => Set.iInter fun h => p.ball x ρ)
  -/
  ext y; simp_rw [mem_closedBall, mem_iInter₂, mem_ball, ← forall_lt_iff_le']
         /-
           🎉 no goals
         -/


@[simp]
theorem ball_zero' (x : E) (hr : 0 < r) : ball (0 : Seminorm 𝕜 E) x r = Set.univ := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : SMul 𝕜 E
    r : Real
    x : E
    hr : LT.lt 0 r
    ⊢ Eq (Seminorm.ball 0 x r) Set.univ
  -/
  rw [Set.eq_univ_iff_forall, ball]
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : SMul 𝕜 E
    r : Real
    x : E
    hr : LT.lt 0 r
    ⊢ ∀ (x_1 : E), Membership.mem (setOf fun y => LT.lt (0 (HSub.hSub y x)) r) x_1
  -/
  simp [hr]
  /-
    🎉 no goals
  -/


@[simp]
theorem closedBall_zero' (x : E) (hr : 0 < r) : closedBall (0 : Seminorm 𝕜 E) x r = Set.univ :=
  eq_univ_of_subset (ball_subset_closedBall _ _ _) (ball_zero' x hr)


theorem ball_smul (p : Seminorm 𝕜 E) {c : NNReal} (hc : 0 < c) (r : ℝ) (x : E) :
    (c • p).ball x r = p.ball x (r / c) := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : SMul 𝕜 E
    p : Seminorm 𝕜 E
    c : NNReal
    hc : LT.lt 0 c
    r : Real
    x : E
    ⊢ Eq ((HSMul.hSMul c p).ball x r) (p.ball x (HDiv.hDiv r ↑c))
  -/
  ext
  rw [mem_ball, mem_ball, smul_apply, NNReal.smul_def, smul_eq_mul, mul_comm,
    lt_div_iff₀ (NNReal.coe_pos.mpr hc)]


theorem closedBall_smul (p : Seminorm 𝕜 E) {c : NNReal} (hc : 0 < c) (r : ℝ) (x : E) :
    (c • p).closedBall x r = p.closedBall x (r / c) := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : SMul 𝕜 E
    p : Seminorm 𝕜 E
    c : NNReal
    hc : LT.lt 0 c
    r : Real
    x : E
    ⊢ Eq ((HSMul.hSMul c p).closedBall x r) (p.closedBall x (HDiv.hDiv r ↑c))
  -/
  ext
  rw [mem_closedBall, mem_closedBall, smul_apply, NNReal.smul_def, smul_eq_mul, mul_comm,
    le_div_iff₀ (NNReal.coe_pos.mpr hc)]


theorem ball_sup (p : Seminorm 𝕜 E) (q : Seminorm 𝕜 E) (e : E) (r : ℝ) :
    ball (p ⊔ q) e r = ball p e r ∩ ball q e r := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : SMul 𝕜 E
    p q : Seminorm 𝕜 E
    e : E
    r : Real
    ⊢ Eq ((Max.max p q).ball e r) (Inter.inter (p.ball e r) (q.ball e r))
  -/
  simp_rw [ball, ← Set.setOf_and, coe_sup, Pi.sup_apply, sup_lt_iff]
  /-
    🎉 no goals
  -/


theorem closedBall_sup (p : Seminorm 𝕜 E) (q : Seminorm 𝕜 E) (e : E) (r : ℝ) :
    closedBall (p ⊔ q) e r = closedBall p e r ∩ closedBall q e r := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : SMul 𝕜 E
    p q : Seminorm 𝕜 E
    e : E
    r : Real
    ⊢ Eq ((Max.max p q).closedBall e r) (Inter.inter (p.closedBall e r) (q.closedB …
  -/
  simp_rw [closedBall, ← Set.setOf_and, coe_sup, Pi.sup_apply, sup_le_iff]
  /-
    🎉 no goals
  -/


theorem ball_finset_sup' (p : ι → Seminorm 𝕜 E) (s : Finset ι) (H : s.Nonempty) (e : E) (r : ℝ) :
    ball (s.sup' H p) e r = s.inf' H fun i => ball (p i) e r := by
  induction H using Finset.Nonempty.cons_induction with
  | singleton => simp
  | cons _ _ _ hs ih =>
    rw [Finset.sup'_cons hs, Finset.inf'_cons hs, ball_sup]
    -- Porting note: `rw` can't use `inf_eq_inter` here, but `simp` can?
    simp only [inf_eq_inter, ih]


theorem closedBall_finset_sup' (p : ι → Seminorm 𝕜 E) (s : Finset ι) (H : s.Nonempty) (e : E)
    (r : ℝ) : closedBall (s.sup' H p) e r = s.inf' H fun i => closedBall (p i) e r := by
  induction H using Finset.Nonempty.cons_induction with
  | singleton => simp
  | cons _ _ _ hs ih =>
    rw [Finset.sup'_cons hs, Finset.inf'_cons hs, closedBall_sup]
    -- Porting note: `rw` can't use `inf_eq_inter` here, but `simp` can?
    simp only [inf_eq_inter, ih]


theorem ball_mono {p : Seminorm 𝕜 E} {r₁ r₂ : ℝ} (h : r₁ ≤ r₂) : p.ball x r₁ ⊆ p.ball x r₂ :=
  fun _ (hx : _ < _) => hx.trans_le h


theorem closedBall_mono {p : Seminorm 𝕜 E} {r₁ r₂ : ℝ} (h : r₁ ≤ r₂) :
    p.closedBall x r₁ ⊆ p.closedBall x r₂ := fun _ (hx : _ ≤ _) => hx.trans h


theorem ball_antitone {p q : Seminorm 𝕜 E} (h : q ≤ p) : p.ball x r ⊆ q.ball x r := fun _ =>
  (h _).trans_lt


theorem closedBall_antitone {p q : Seminorm 𝕜 E} (h : q ≤ p) :
    p.closedBall x r ⊆ q.closedBall x r := fun _ => (h _).trans


theorem ball_add_ball_subset (p : Seminorm 𝕜 E) (r₁ r₂ : ℝ) (x₁ x₂ : E) :
    p.ball (x₁ : E) r₁ + p.ball (x₂ : E) r₂ ⊆ p.ball (x₁ + x₂) (r₁ + r₂) := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : SMul 𝕜 E
    p : Seminorm 𝕜 E
    r₁ r₂ : Real
    x₁ x₂ : E
    ⊢ HasSubset.Subset (HAdd.hAdd (p.ball x₁ r₁) (p.ball x₂ r₂)) (p.ball (HAdd.hAd …
  -/
  rintro x ⟨y₁, hy₁, y₂, hy₂, rfl⟩
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : SMul 𝕜 E
    p : Seminorm 𝕜 E
    r₁ r₂ : Real
    x₁ x₂ y₁ : E
    hy₁ : Membership.mem (p.ball x₁ r₁) y₁
    y₂ : E
    hy₂ : Membership.mem (p.ball x₂ r₂) y₂
    ⊢ Membership.mem (p.ball (HAdd.hAdd x₁ x₂) (HAdd.hAdd r₁ r₂)) ((fun x1 x2 => H …
  -/
  rw [mem_ball, add_sub_add_comm]
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : SMul 𝕜 E
    p : Seminorm 𝕜 E
    r₁ r₂ : Real
    x₁ x₂ y₁ : E
    hy₁ : Membership.mem (p.ball x₁ r₁) y₁
    y₂ : E
    hy₂ : Membership.mem (p.ball x₂ r₂) y₂
    ⊢ LT.lt (p (HAdd.hAdd (HSub.hSub y₁ x₁) (HSub.hSub y₂ x₂))) (HAdd.hAdd r₁ r₂)
  -/
  exact (map_add_le_add p _ _).trans_lt (add_lt_add hy₁ hy₂)
  /-
    🎉 no goals
  -/


theorem closedBall_add_closedBall_subset (p : Seminorm 𝕜 E) (r₁ r₂ : ℝ) (x₁ x₂ : E) :
    p.closedBall (x₁ : E) r₁ + p.closedBall (x₂ : E) r₂ ⊆ p.closedBall (x₁ + x₂) (r₁ + r₂) := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : SMul 𝕜 E
    p : Seminorm 𝕜 E
    r₁ r₂ : Real
    x₁ x₂ : E
    ⊢ HasSubset.Subset (HAdd.hAdd (p.closedBall x₁ r₁) (p.closedBall x₂ r₂)) (p.cl …
  -/
  rintro x ⟨y₁, hy₁, y₂, hy₂, rfl⟩
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : SMul 𝕜 E
    p : Seminorm 𝕜 E
    r₁ r₂ : Real
    x₁ x₂ y₁ : E
    hy₁ : Membership.mem (p.closedBall x₁ r₁) y₁
    y₂ : E
    hy₂ : Membership.mem (p.closedBall x₂ r₂) y₂
    ⊢ Membership.mem (p.closedBall (HAdd.hAdd x₁ x₂) (HAdd.hAdd r₁ r₂)) ((fun x1 x …
  -/
  rw [mem_closedBall, add_sub_add_comm]
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : SMul 𝕜 E
    p : Seminorm 𝕜 E
    r₁ r₂ : Real
    x₁ x₂ y₁ : E
    hy₁ : Membership.mem (p.closedBall x₁ r₁) y₁
    y₂ : E
    hy₂ : Membership.mem (p.closedBall x₂ r₂) y₂
    ⊢ LE.le (p (HAdd.hAdd (HSub.hSub y₁ x₁) (HSub.hSub y₂ x₂))) (HAdd.hAdd r₁ r₂)
  -/
  exact (map_add_le_add p _ _).trans (add_le_add hy₁ hy₂)
  /-
    🎉 no goals
  -/


theorem sub_mem_ball (p : Seminorm 𝕜 E) (x₁ x₂ y : E) (r : ℝ) :
                                                        /-
                                                          𝕜 : Type u_3
                                                          E : Type u_7
                                                          inst✝² : SeminormedRing 𝕜
                                                          inst✝¹ : AddCommGroup E
                                                          inst✝ : SMul 𝕜 E
                                                          p : Seminorm 𝕜 E
                                                          x₁ x₂ y : E
                                                          r : Real
                                                          ⊢ Iff (Membership.mem (p.ball y r) (HSub.hSub x₁ x₂)) (Membership.mem (p.ball  …
                                                        -/
    x₁ - x₂ ∈ p.ball y r ↔ x₁ ∈ p.ball (x₂ + y) r := by simp_rw [mem_ball, sub_sub]
                                                        /-
                                                          🎉 no goals
                                                        -/


/-- The image of a ball under addition with a singleton is another ball. -/
theorem vadd_ball (p : Seminorm 𝕜 E) : x +ᵥ p.ball y r = p.ball (x +ᵥ y) r :=
  letI := AddGroupSeminorm.toSeminormedAddCommGroup p.toAddGroupSeminorm
  Metric.vadd_ball x y r


/-- The image of a closed ball under addition with a singleton is another closed ball. -/
theorem vadd_closedBall (p : Seminorm 𝕜 E) : x +ᵥ p.closedBall y r = p.closedBall (x +ᵥ y) r :=
  letI := AddGroupSeminorm.toSeminormedAddCommGroup p.toAddGroupSeminorm
  Metric.vadd_closedBall x y r


theorem ball_comp (p : Seminorm 𝕜₂ E₂) (f : E →ₛₗ[σ₁₂] E₂) (x : E) (r : ℝ) :
    (p.comp f).ball x r = f ⁻¹' p.ball (f x) r := by
  /-
    𝕜 : Type u_3
    𝕜₂ : Type u_4
    E : Type u_7
    E₂ : Type u_8
    inst✝⁶ : SeminormedRing 𝕜
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module 𝕜 E
    inst✝³ : SeminormedRing 𝕜₂
    inst✝² : AddCommGroup E₂
    inst✝¹ : Module 𝕜₂ E₂
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝ : RingHomIsometric σ₁₂
    p : Seminorm 𝕜₂ E₂
    f : LinearMap σ₁₂ E E₂
    x : E
    r : Real
    ⊢ Eq ((p.comp f).ball x r) (Set.preimage (⇑f) (p.ball (f x) r))
  -/
  ext
  /-
    case h
    𝕜 : Type u_3
    𝕜₂ : Type u_4
    E : Type u_7
    E₂ : Type u_8
    inst✝⁶ : SeminormedRing 𝕜
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module 𝕜 E
    inst✝³ : SeminormedRing 𝕜₂
    inst✝² : AddCommGroup E₂
    inst✝¹ : Module 𝕜₂ E₂
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝ : RingHomIsometric σ₁₂
    p : Seminorm 𝕜₂ E₂
    f : LinearMap σ₁₂ E E₂
    x : E
    r : Real
    x✝ : E
    ⊢ Iff (Membership.mem ((p.comp f).ball x r) x✝) (Membership.mem (Set.preimage  …
  -/
  simp_rw [ball, mem_preimage, comp_apply, Set.mem_setOf_eq, map_sub]
  /-
    🎉 no goals
  -/


theorem closedBall_comp (p : Seminorm 𝕜₂ E₂) (f : E →ₛₗ[σ₁₂] E₂) (x : E) (r : ℝ) :
    (p.comp f).closedBall x r = f ⁻¹' p.closedBall (f x) r := by
  /-
    𝕜 : Type u_3
    𝕜₂ : Type u_4
    E : Type u_7
    E₂ : Type u_8
    inst✝⁶ : SeminormedRing 𝕜
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module 𝕜 E
    inst✝³ : SeminormedRing 𝕜₂
    inst✝² : AddCommGroup E₂
    inst✝¹ : Module 𝕜₂ E₂
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝ : RingHomIsometric σ₁₂
    p : Seminorm 𝕜₂ E₂
    f : LinearMap σ₁₂ E E₂
    x : E
    r : Real
    ⊢ Eq ((p.comp f).closedBall x r) (Set.preimage (⇑f) (p.closedBall (f x) r))
  -/
  ext
  /-
    case h
    𝕜 : Type u_3
    𝕜₂ : Type u_4
    E : Type u_7
    E₂ : Type u_8
    inst✝⁶ : SeminormedRing 𝕜
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module 𝕜 E
    inst✝³ : SeminormedRing 𝕜₂
    inst✝² : AddCommGroup E₂
    inst✝¹ : Module 𝕜₂ E₂
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝ : RingHomIsometric σ₁₂
    p : Seminorm 𝕜₂ E₂
    f : LinearMap σ₁₂ E E₂
    x : E
    r : Real
    x✝ : E
    ⊢ Iff (Membership.mem ((p.comp f).closedBall x r) x✝) (Membership.mem (Set.pre …
  -/
  simp_rw [closedBall, mem_preimage, comp_apply, Set.mem_setOf_eq, map_sub]
  /-
    🎉 no goals
  -/


theorem preimage_metric_ball {r : ℝ} : p ⁻¹' Metric.ball 0 r = { x | p x < r } := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    r : Real
    ⊢ Eq (Set.preimage (⇑p) (Metric.ball 0 r)) (setOf fun x => LT.lt (p x) r)
  -/
  ext x
  /-
    case h
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    r : Real
    x : E
    ⊢ Iff (Membership.mem (Set.preimage (⇑p) (Metric.ball 0 r)) x) (Membership.mem …
  -/
  simp only [mem_setOf, mem_preimage, mem_ball_zero_iff, Real.norm_of_nonneg (apply_nonneg p _)]
  /-
    🎉 no goals
  -/


theorem preimage_metric_closedBall {r : ℝ} : p ⁻¹' Metric.closedBall 0 r = { x | p x ≤ r } := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    r : Real
    ⊢ Eq (Set.preimage (⇑p) (Metric.closedBall 0 r)) (setOf fun x => LE.le (p x) r)
  -/
  ext x
  simp only [mem_setOf, mem_preimage, mem_closedBall_zero_iff,
    Real.norm_of_nonneg (apply_nonneg p _)]


theorem ball_zero_eq_preimage_ball {r : ℝ} : p.ball 0 r = p ⁻¹' Metric.ball 0 r := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    r : Real
    ⊢ Eq (p.ball 0 r) (Set.preimage (⇑p) (Metric.ball 0 r))
  -/
  rw [ball_zero_eq, preimage_metric_ball]
  /-
    🎉 no goals
  -/


theorem closedBall_zero_eq_preimage_closedBall {r : ℝ} :
    p.closedBall 0 r = p ⁻¹' Metric.closedBall 0 r := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    r : Real
    ⊢ Eq (p.closedBall 0 r) (Set.preimage (⇑p) (Metric.closedBall 0 r))
  -/
  rw [closedBall_zero_eq, preimage_metric_closedBall]
  /-
    🎉 no goals
  -/


@[simp]
theorem ball_bot {r : ℝ} (x : E) (hr : 0 < r) : ball (⊥ : Seminorm 𝕜 E) x r = Set.univ :=
  ball_zero' x hr


@[simp]
theorem closedBall_bot {r : ℝ} (x : E) (hr : 0 < r) :
    closedBall (⊥ : Seminorm 𝕜 E) x r = Set.univ :=
  closedBall_zero' x hr


/-- Seminorm-balls at the origin are balanced. -/
theorem balanced_ball_zero (r : ℝ) : Balanced 𝕜 (ball p 0 r) := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    r : Real
    ⊢ Balanced 𝕜 (p.ball 0 r)
  -/
  rintro a ha x ⟨y, hy, hx⟩
  /-
    case intro.intro
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    r : Real
    a : 𝕜
    ha : LE.le (Norm.norm a) 1
    x y : E
    hy : Membership.mem (p.ball 0 r) y
    hx : Eq ((fun x => HSMul.hSMul a x) y) x
    ⊢ Membership.mem (p.ball 0 r) x
  -/
  rw [mem_ball_zero, ← hx, map_smul_eq_mul]
  calc
    _ ≤ p y := mul_le_of_le_one_left (apply_nonneg p _) ha
    _ < r := by rwa [mem_ball_zero] at hy


/-- Closed seminorm-balls at the origin are balanced. -/
theorem balanced_closedBall_zero (r : ℝ) : Balanced 𝕜 (closedBall p 0 r) := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    r : Real
    ⊢ Balanced 𝕜 (p.closedBall 0 r)
  -/
  rintro a ha x ⟨y, hy, hx⟩
  /-
    case intro.intro
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    r : Real
    a : 𝕜
    ha : LE.le (Norm.norm a) 1
    x y : E
    hy : Membership.mem (p.closedBall 0 r) y
    hx : Eq ((fun x => HSMul.hSMul a x) y) x
    ⊢ Membership.mem (p.closedBall 0 r) x
  -/
  rw [mem_closedBall_zero, ← hx, map_smul_eq_mul]
  calc
    _ ≤ p y := mul_le_of_le_one_left (apply_nonneg p _) ha
    _ ≤ r := by rwa [mem_closedBall_zero] at hy


theorem ball_finset_sup_eq_iInter (p : ι → Seminorm 𝕜 E) (s : Finset ι) (x : E) {r : ℝ}
    (hr : 0 < r) : ball (s.sup p) x r = ⋂ i ∈ s, ball (p i) x r := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    ι : Type u_11
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → Seminorm 𝕜 E
    s : Finset ι
    x : E
    r : Real
    hr : LT.lt 0 r
    ⊢ Eq ((s.sup p).ball x r) (Set.iInter fun i => Set.iInter fun h => (p i).ball  …
  -/
  lift r to NNReal using hr.le
  simp_rw [ball, iInter_setOf, finset_sup_apply, NNReal.coe_lt_coe,
    Finset.sup_lt_iff (show ⊥ < r from hr), ← NNReal.coe_lt_coe, NNReal.coe_mk]


theorem closedBall_finset_sup_eq_iInter (p : ι → Seminorm 𝕜 E) (s : Finset ι) (x : E) {r : ℝ}
    (hr : 0 ≤ r) : closedBall (s.sup p) x r = ⋂ i ∈ s, closedBall (p i) x r := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    ι : Type u_11
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → Seminorm 𝕜 E
    s : Finset ι
    x : E
    r : Real
    hr : LE.le 0 r
    ⊢ Eq ((s.sup p).closedBall x r) (Set.iInter fun i => Set.iInter fun h => (p i) …
  -/
  lift r to NNReal using hr
  simp_rw [closedBall, iInter_setOf, finset_sup_apply, NNReal.coe_le_coe, Finset.sup_le_iff, ←
    NNReal.coe_le_coe, NNReal.coe_mk]


theorem ball_finset_sup (p : ι → Seminorm 𝕜 E) (s : Finset ι) (x : E) {r : ℝ} (hr : 0 < r) :
    ball (s.sup p) x r = s.inf fun i => ball (p i) x r := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    ι : Type u_11
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → Seminorm 𝕜 E
    s : Finset ι
    x : E
    r : Real
    hr : LT.lt 0 r
    ⊢ Eq ((s.sup p).ball x r) (s.inf fun i => (p i).ball x r)
  -/
  rw [Finset.inf_eq_iInf]
  /-
    𝕜 : Type u_3
    E : Type u_7
    ι : Type u_11
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → Seminorm 𝕜 E
    s : Finset ι
    x : E
    r : Real
    hr : LT.lt 0 r
    ⊢ Eq ((s.sup p).ball x r) (iInf fun a => iInf fun h => (p a).ball x r)
  -/
  exact ball_finset_sup_eq_iInter _ _ _ hr
  /-
    🎉 no goals
  -/


theorem closedBall_finset_sup (p : ι → Seminorm 𝕜 E) (s : Finset ι) (x : E) {r : ℝ} (hr : 0 ≤ r) :
    closedBall (s.sup p) x r = s.inf fun i => closedBall (p i) x r := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    ι : Type u_11
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → Seminorm 𝕜 E
    s : Finset ι
    x : E
    r : Real
    hr : LE.le 0 r
    ⊢ Eq ((s.sup p).closedBall x r) (s.inf fun i => (p i).closedBall x r)
  -/
  rw [Finset.inf_eq_iInf]
  /-
    𝕜 : Type u_3
    E : Type u_7
    ι : Type u_11
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → Seminorm 𝕜 E
    s : Finset ι
    x : E
    r : Real
    hr : LE.le 0 r
    ⊢ Eq ((s.sup p).closedBall x r) (iInf fun a => iInf fun h => (p a).closedBall  …
  -/
  exact closedBall_finset_sup_eq_iInter _ _ _ hr
  /-
    🎉 no goals
  -/


@[simp]
theorem ball_eq_emptyset (p : Seminorm 𝕜 E) {x : E} {r : ℝ} (hr : r ≤ 0) : p.ball x r = ∅ := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    x : E
    r : Real
    hr : LE.le r 0
    ⊢ Eq (p.ball x r) EmptyCollection.emptyCollection
  -/
  ext
  /-
    case h
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    x : E
    r : Real
    hr : LE.le r 0
    x✝ : E
    ⊢ Iff (Membership.mem (p.ball x r) x✝) (Membership.mem EmptyCollection.emptyCo …
  -/
  rw [Seminorm.mem_ball, Set.mem_empty_iff_false, iff_false, not_lt]
  /-
    case h
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    x : E
    r : Real
    hr : LE.le r 0
    x✝ : E
    ⊢ LE.le r (p (HSub.hSub x✝ x))
  -/
  exact hr.trans (apply_nonneg p _)
  /-
    🎉 no goals
  -/


@[simp]
theorem closedBall_eq_emptyset (p : Seminorm 𝕜 E) {x : E} {r : ℝ} (hr : r < 0) :
    p.closedBall x r = ∅ := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    x : E
    r : Real
    hr : LT.lt r 0
    ⊢ Eq (p.closedBall x r) EmptyCollection.emptyCollection
  -/
  ext
  /-
    case h
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    x : E
    r : Real
    hr : LT.lt r 0
    x✝ : E
    ⊢ Iff (Membership.mem (p.closedBall x r) x✝) (Membership.mem EmptyCollection.e …
  -/
  rw [Seminorm.mem_closedBall, Set.mem_empty_iff_false, iff_false, not_le]
  /-
    case h
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    x : E
    r : Real
    hr : LT.lt r 0
    x✝ : E
    ⊢ LT.lt r (p (HSub.hSub x✝ x))
  -/
  exact hr.trans_le (apply_nonneg _ _)
  /-
    🎉 no goals
  -/


theorem closedBall_smul_ball (p : Seminorm 𝕜 E) {r₁ : ℝ} (hr₁ : r₁ ≠ 0) (r₂ : ℝ) :
    Metric.closedBall (0 : 𝕜) r₁ • p.ball 0 r₂ ⊆ p.ball 0 (r₁ * r₂) := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    r₁ : Real
    hr₁ : Ne r₁ 0
    r₂ : Real
    ⊢ HasSubset.Subset (HSMul.hSMul (Metric.closedBall 0 r₁) (p.ball 0 r₂)) (p.bal …
  -/
  simp only [smul_subset_iff, mem_ball_zero, mem_closedBall_zero_iff, map_smul_eq_mul]
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    r₁ : Real
    hr₁ : Ne r₁ 0
    r₂ : Real
    ⊢ ∀ (a : 𝕜), LE.le (Norm.norm a) r₁ → ∀ (b : E), LT.lt (p b) r₂ → LT.lt (HMul. …
  -/
  refine fun a ha b hb ↦ mul_lt_mul' ha hb (apply_nonneg _ _) ?_
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    r₁ : Real
    hr₁ : Ne r₁ 0
    r₂ : Real
    a : 𝕜
    ha : LE.le (Norm.norm a) r₁
    b : E
    hb : LT.lt (p b) r₂
    ⊢ LT.lt 0 r₁
  -/
  exact hr₁.lt_or_lt.resolve_left <| ((norm_nonneg a).trans ha).not_lt
  /-
    🎉 no goals
  -/


theorem ball_smul_closedBall (p : Seminorm 𝕜 E) (r₁ : ℝ) {r₂ : ℝ} (hr₂ : r₂ ≠ 0) :
    Metric.ball (0 : 𝕜) r₁ • p.closedBall 0 r₂ ⊆ p.ball 0 (r₁ * r₂) := by
  simp only [smul_subset_iff, mem_ball_zero, mem_closedBall_zero, mem_ball_zero_iff,
    map_smul_eq_mul]
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    r₁ r₂ : Real
    hr₂ : Ne r₂ 0
    ⊢ ∀ (a : 𝕜), LT.lt (Norm.norm a) r₁ → ∀ (b : E), LE.le (p b) r₂ → LT.lt (HMul. …
  -/
  intro a ha b hb
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    r₁ r₂ : Real
    hr₂ : Ne r₂ 0
    a : 𝕜
    ha : LT.lt (Norm.norm a) r₁
    b : E
    hb : LE.le (p b) r₂
    ⊢ LT.lt (HMul.hMul (Norm.norm a) (p b)) (HMul.hMul r₁ r₂)
  -/
  rw [mul_comm, mul_comm r₁]
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    r₁ r₂ : Real
    hr₂ : Ne r₂ 0
    a : 𝕜
    ha : LT.lt (Norm.norm a) r₁
    b : E
    hb : LE.le (p b) r₂
    ⊢ LT.lt (HMul.hMul (p b) (Norm.norm a)) (HMul.hMul r₂ r₁)
  -/
  refine mul_lt_mul' hb ha (norm_nonneg _) (hr₂.lt_or_lt.resolve_left ?_)
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    r₁ r₂ : Real
    hr₂ : Ne r₂ 0
    a : 𝕜
    ha : LT.lt (Norm.norm a) r₁
    b : E
    hb : LE.le (p b) r₂
    ⊢ Not (LT.lt r₂ 0)
  -/
  exact ((apply_nonneg p b).trans hb).not_lt
  /-
    🎉 no goals
  -/


theorem ball_smul_ball (p : Seminorm 𝕜 E) (r₁ r₂ : ℝ) :
    Metric.ball (0 : 𝕜) r₁ • p.ball 0 r₂ ⊆ p.ball 0 (r₁ * r₂) := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    r₁ r₂ : Real
    ⊢ HasSubset.Subset (HSMul.hSMul (Metric.ball 0 r₁) (p.ball 0 r₂)) (p.ball 0 (H …
  -/
  rcases eq_or_ne r₂ 0 with rfl | hr₂
    /-
      case inl
      𝕜 : Type u_3
      E : Type u_7
      inst✝² : SeminormedRing 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      p : Seminorm 𝕜 E
      r₁ : Real
      ⊢ HasSubset.Subset (HSMul.hSMul (Metric.ball 0 r₁) (p.ball 0 0)) (p.ball 0 (HM …
    -/
  · simp
    /-
      🎉 no goals
    -/
  · exact (smul_subset_smul_left (ball_subset_closedBall _ _ _)).trans
      (ball_smul_closedBall _ _ hr₂)


theorem closedBall_smul_closedBall (p : Seminorm 𝕜 E) (r₁ r₂ : ℝ) :
    Metric.closedBall (0 : 𝕜) r₁ • p.closedBall 0 r₂ ⊆ p.closedBall 0 (r₁ * r₂) := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    r₁ r₂ : Real
    ⊢ HasSubset.Subset (HSMul.hSMul (Metric.closedBall 0 r₁) (p.closedBall 0 r₂))  …
  -/
  simp only [smul_subset_iff, mem_closedBall_zero, mem_closedBall_zero_iff, map_smul_eq_mul]
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    r₁ r₂ : Real
    ⊢ ∀ (a : 𝕜), LE.le (Norm.norm a) r₁ → ∀ (b : E), LE.le (p b) r₂ → LE.le (HMul. …
  -/
  intro a ha b hb
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    r₁ r₂ : Real
    a : 𝕜
    ha : LE.le (Norm.norm a) r₁
    b : E
    hb : LE.le (p b) r₂
    ⊢ LE.le (HMul.hMul (Norm.norm a) (p b)) (HMul.hMul r₁ r₂)
  -/
  gcongr
  /-
    case b0
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    r₁ r₂ : Real
    a : 𝕜
    ha : LE.le (Norm.norm a) r₁
    b : E
    hb : LE.le (p b) r₂
    ⊢ LE.le 0 r₁
  -/
  exact (norm_nonneg _).trans ha
  /-
    🎉 no goals
  -/


theorem neg_mem_ball_zero {r : ℝ} {x : E} : -x ∈ ball p 0 r ↔ x ∈ ball p 0 r := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    r : Real
    x : E
    ⊢ Iff (Membership.mem (p.ball 0 r) (Neg.neg x)) (Membership.mem (p.ball 0 r) x)
  -/
  simp only [mem_ball_zero, map_neg_eq_map]
  /-
    🎉 no goals
  -/


@[simp]
theorem neg_ball (p : Seminorm 𝕜 E) (r : ℝ) (x : E) : -ball p x r = ball p (-x) r := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    r : Real
    x : E
    ⊢ Eq (Neg.neg (p.ball x r)) (p.ball (Neg.neg x) r)
  -/
  ext
  /-
    case h
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : SeminormedRing 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    r : Real
    x x✝ : E
    ⊢ Iff (Membership.mem (Neg.neg (p.ball x r)) x✝) (Membership.mem (p.ball (Neg. …
  -/
  rw [Set.mem_neg, mem_ball, mem_ball, ← neg_add', sub_neg_eq_add, map_neg_eq_map]
  /-
    🎉 no goals
  -/


theorem closedBall_iSup {ι : Sort*} {p : ι → Seminorm 𝕜 E} (hp : BddAbove (range p)) (e : E)
    {r : ℝ} (hr : 0 < r) : closedBall (⨆ i, p i) e r = ⋂ i, closedBall (p i) e r := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    ι : Sort u_12
    p : ι → Seminorm 𝕜 E
    hp : BddAbove (Set.range p)
    e : E
    r : Real
    hr : LT.lt 0 r
    ⊢ Eq ((iSup fun i => p i).closedBall e r) (Set.iInter fun i => (p i).closedBal …
  -/
  cases isEmpty_or_nonempty ι
    /-
      case inl
      𝕜 : Type u_3
      E : Type u_7
      inst✝² : NormedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      ι : Sort u_12
      p : ι → Seminorm 𝕜 E
      hp : BddAbove (Set.range p)
      e : E
      r : Real
      hr : LT.lt 0 r
      h✝ : IsEmpty ι
      ⊢ Eq ((iSup fun i => p i).closedBall e r) (Set.iInter fun i => (p i).closedBal …
    -/
  · rw [iSup_of_empty', iInter_of_empty, Seminorm.sSup_empty]
    /-
      case inl
      𝕜 : Type u_3
      E : Type u_7
      inst✝² : NormedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      ι : Sort u_12
      p : ι → Seminorm 𝕜 E
      hp : BddAbove (Set.range p)
      e : E
      r : Real
      hr : LT.lt 0 r
      h✝ : IsEmpty ι
      ⊢ Eq (Bot.bot.closedBall e r) Set.univ
    -/
    exact closedBall_bot _ hr
    /-
      🎉 no goals
    -/
    /-
      case inr
      𝕜 : Type u_3
      E : Type u_7
      inst✝² : NormedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      ι : Sort u_12
      p : ι → Seminorm 𝕜 E
      hp : BddAbove (Set.range p)
      e : E
      r : Real
      hr : LT.lt 0 r
      h✝ : Nonempty ι
      ⊢ Eq ((iSup fun i => p i).closedBall e r) (Set.iInter fun i => (p i).closedBal …
    -/
  · ext x
    /-
      case inr.h
      𝕜 : Type u_3
      E : Type u_7
      inst✝² : NormedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      ι : Sort u_12
      p : ι → Seminorm 𝕜 E
      hp : BddAbove (Set.range p)
      e : E
      r : Real
      hr : LT.lt 0 r
      h✝ : Nonempty ι
      x : E
      ⊢ Iff (Membership.mem ((iSup fun i => p i).closedBall e r) x) (Membership.mem  …
    -/
    have := Seminorm.bddAbove_range_iff.mp hp (x - e)
    /-
      case inr.h
      𝕜 : Type u_3
      E : Type u_7
      inst✝² : NormedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      ι : Sort u_12
      p : ι → Seminorm 𝕜 E
      hp : BddAbove (Set.range p)
      e : E
      r : Real
      hr : LT.lt 0 r
      h✝ : Nonempty ι
      x : E
      this : BddAbove (Set.range fun i => (p i) (HSub.hSub x e))
      ⊢ Iff (Membership.mem ((iSup fun i => p i).closedBall e r) x) (Membership.mem  …
    -/
    simp only [mem_closedBall, mem_iInter, Seminorm.iSup_apply hp, ciSup_le_iff this]
    /-
      🎉 no goals
    -/


theorem ball_norm_mul_subset {p : Seminorm 𝕜 E} {k : 𝕜} {r : ℝ} :
    p.ball 0 (‖k‖ * r) ⊆ k • p.ball 0 r := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    k : 𝕜
    r : Real
    ⊢ HasSubset.Subset (p.ball 0 (HMul.hMul (Norm.norm k) r)) (HSMul.hSMul k (p.ba …
  -/
  rcases eq_or_ne k 0 with (rfl | hk)
    /-
      case inl
      𝕜 : Type u_3
      E : Type u_7
      inst✝² : NormedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      p : Seminorm 𝕜 E
      r : Real
      ⊢ HasSubset.Subset (p.ball 0 (HMul.hMul (Norm.norm 0) r)) (HSMul.hSMul 0 (p.ba …
    -/
  · rw [norm_zero, zero_mul, ball_eq_emptyset _ le_rfl]
    /-
      case inl
      𝕜 : Type u_3
      E : Type u_7
      inst✝² : NormedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      p : Seminorm 𝕜 E
      r : Real
      ⊢ HasSubset.Subset EmptyCollection.emptyCollection (HSMul.hSMul 0 (p.ball 0 r))
    -/
    exact empty_subset _
    /-
      🎉 no goals
    -/
    /-
      case inr
      𝕜 : Type u_3
      E : Type u_7
      inst✝² : NormedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      p : Seminorm 𝕜 E
      k : 𝕜
      r : Real
      hk : Ne k 0
      ⊢ HasSubset.Subset (p.ball 0 (HMul.hMul (Norm.norm k) r)) (HSMul.hSMul k (p.ba …
    -/
  · intro x
    /-
      case inr
      𝕜 : Type u_3
      E : Type u_7
      inst✝² : NormedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      p : Seminorm 𝕜 E
      k : 𝕜
      r : Real
      hk : Ne k 0
      x : E
      ⊢ Membership.mem (p.ball 0 (HMul.hMul (Norm.norm k) r)) x → Membership.mem (HS …
    -/
    rw [Set.mem_smul_set, Seminorm.mem_ball_zero]
    /-
      case inr
      𝕜 : Type u_3
      E : Type u_7
      inst✝² : NormedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      p : Seminorm 𝕜 E
      k : 𝕜
      r : Real
      hk : Ne k 0
      x : E
      ⊢ LT.lt (p x) (HMul.hMul (Norm.norm k) r) → Exists fun y => And (Membership.me …
    -/
    refine fun hx => ⟨k⁻¹ • x, ?_, ?_⟩
    · rwa [Seminorm.mem_ball_zero, map_smul_eq_mul, norm_inv, ←
        mul_lt_mul_left <| norm_pos_iff.mpr hk, ← mul_assoc, ← div_eq_mul_inv ‖k‖ ‖k‖,
        div_self (ne_of_gt <| norm_pos_iff.mpr hk), one_mul]
    /-
      case inr.refine_2
      𝕜 : Type u_3
      E : Type u_7
      inst✝² : NormedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      p : Seminorm 𝕜 E
      k : 𝕜
      r : Real
      hk : Ne k 0
      x : E
      hx : LT.lt (p x) (HMul.hMul (Norm.norm k) r)
      ⊢ Eq (HSMul.hSMul k (HSMul.hSMul (Inv.inv k) x)) x
    -/
    rw [← smul_assoc, smul_eq_mul, ← div_eq_mul_inv, div_self hk, one_smul]
    /-
      🎉 no goals
    -/


theorem smul_ball_zero {p : Seminorm 𝕜 E} {k : 𝕜} {r : ℝ} (hk : k ≠ 0) :
    k • p.ball 0 r = p.ball 0 (‖k‖ * r) := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    k : 𝕜
    r : Real
    hk : Ne k 0
    ⊢ Eq (HSMul.hSMul k (p.ball 0 r)) (p.ball 0 (HMul.hMul (Norm.norm k) r))
  -/
  ext
  rw [mem_smul_set_iff_inv_smul_mem₀ hk, p.mem_ball_zero, p.mem_ball_zero, map_smul_eq_mul,
    norm_inv, ← div_eq_inv_mul, div_lt_iff₀ (norm_pos_iff.2 hk), mul_comm]


theorem smul_closedBall_subset {p : Seminorm 𝕜 E} {k : 𝕜} {r : ℝ} :
    k • p.closedBall 0 r ⊆ p.closedBall 0 (‖k‖ * r) := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    k : 𝕜
    r : Real
    ⊢ HasSubset.Subset (HSMul.hSMul k (p.closedBall 0 r)) (p.closedBall 0 (HMul.hM …
  -/
  rintro x ⟨y, hy, h⟩
  /-
    case intro.intro
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    k : 𝕜
    r : Real
    x y : E
    hy : Membership.mem (p.closedBall 0 r) y
    h : Eq ((fun x => HSMul.hSMul k x) y) x
    ⊢ Membership.mem (p.closedBall 0 (HMul.hMul (Norm.norm k) r)) x
  -/
  rw [Seminorm.mem_closedBall_zero, ← h, map_smul_eq_mul]
  /-
    case intro.intro
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    k : 𝕜
    r : Real
    x y : E
    hy : Membership.mem (p.closedBall 0 r) y
    h : Eq ((fun x => HSMul.hSMul k x) y) x
    ⊢ LE.le (HMul.hMul (Norm.norm k) (p y)) (HMul.hMul (Norm.norm k) r)
  -/
  rw [Seminorm.mem_closedBall_zero] at hy
  /-
    case intro.intro
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    k : 𝕜
    r : Real
    x y : E
    hy : LE.le (p y) r
    h : Eq ((fun x => HSMul.hSMul k x) y) x
    ⊢ LE.le (HMul.hMul (Norm.norm k) (p y)) (HMul.hMul (Norm.norm k) r)
  -/
  gcongr
  /-
    🎉 no goals
  -/


theorem smul_closedBall_zero {p : Seminorm 𝕜 E} {k : 𝕜} {r : ℝ} (hk : 0 < ‖k‖) :
    k • p.closedBall 0 r = p.closedBall 0 (‖k‖ * r) := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    k : 𝕜
    r : Real
    hk : LT.lt 0 (Norm.norm k)
    ⊢ Eq (HSMul.hSMul k (p.closedBall 0 r)) (p.closedBall 0 (HMul.hMul (Norm.norm  …
  -/
  refine subset_antisymm smul_closedBall_subset ?_
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    k : 𝕜
    r : Real
    hk : LT.lt 0 (Norm.norm k)
    ⊢ HasSubset.Subset (p.closedBall 0 (HMul.hMul (Norm.norm k) r)) (HSMul.hSMul k …
  -/
  intro x
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    k : 𝕜
    r : Real
    hk : LT.lt 0 (Norm.norm k)
    x : E
    ⊢ Membership.mem (p.closedBall 0 (HMul.hMul (Norm.norm k) r)) x → Membership.m …
  -/
  rw [Set.mem_smul_set, Seminorm.mem_closedBall_zero]
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    k : 𝕜
    r : Real
    hk : LT.lt 0 (Norm.norm k)
    x : E
    ⊢ LE.le (p x) (HMul.hMul (Norm.norm k) r) → Exists fun y => And (Membership.me …
  -/
  refine fun hx => ⟨k⁻¹ • x, ?_, ?_⟩
  · rwa [Seminorm.mem_closedBall_zero, map_smul_eq_mul, norm_inv, ← mul_le_mul_left hk, ← mul_assoc,
      ← div_eq_mul_inv ‖k‖ ‖k‖, div_self (ne_of_gt hk), one_mul]
  /-
    case refine_2
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    k : 𝕜
    r : Real
    hk : LT.lt 0 (Norm.norm k)
    x : E
    hx : LE.le (p x) (HMul.hMul (Norm.norm k) r)
    ⊢ Eq (HSMul.hSMul k (HSMul.hSMul (Inv.inv k) x)) x
  -/
  rw [← smul_assoc, smul_eq_mul, ← div_eq_mul_inv, div_self (norm_pos_iff.mp hk), one_smul]
  /-
    🎉 no goals
  -/


theorem ball_zero_absorbs_ball_zero (p : Seminorm 𝕜 E) {r₁ r₂ : ℝ} (hr₁ : 0 < r₁) :
    Absorbs 𝕜 (p.ball 0 r₁) (p.ball 0 r₂) := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    r₁ r₂ : Real
    hr₁ : LT.lt 0 r₁
    ⊢ Absorbs 𝕜 (p.ball 0 r₁) (p.ball 0 r₂)
  -/
  rcases exists_pos_lt_mul hr₁ r₂ with ⟨r, hr₀, hr⟩
  /-
    case intro.intro
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    r₁ r₂ : Real
    hr₁ : LT.lt 0 r₁
    r : Real
    hr₀ : LT.lt 0 r
    hr : LT.lt r₂ (HMul.hMul r r₁)
    ⊢ Absorbs 𝕜 (p.ball 0 r₁) (p.ball 0 r₂)
  -/
  refine .of_norm ⟨r, fun a ha x hx => ?_⟩
  /-
    case intro.intro
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    r₁ r₂ : Real
    hr₁ : LT.lt 0 r₁
    r : Real
    hr₀ : LT.lt 0 r
    hr : LT.lt r₂ (HMul.hMul r r₁)
    a : 𝕜
    ha : LE.le r (Norm.norm a)
    x : E
    hx : Membership.mem (p.ball 0 r₂) x
    ⊢ Membership.mem (HSMul.hSMul a (p.ball 0 r₁)) x
  -/
  rw [smul_ball_zero (norm_pos_iff.1 <| hr₀.trans_le ha), p.mem_ball_zero]
  /-
    case intro.intro
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    r₁ r₂ : Real
    hr₁ : LT.lt 0 r₁
    r : Real
    hr₀ : LT.lt 0 r
    hr : LT.lt r₂ (HMul.hMul r r₁)
    a : 𝕜
    ha : LE.le r (Norm.norm a)
    x : E
    hx : Membership.mem (p.ball 0 r₂) x
    ⊢ LT.lt (p x) (HMul.hMul (Norm.norm a) r₁)
  -/
  rw [p.mem_ball_zero] at hx
  /-
    case intro.intro
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    r₁ r₂ : Real
    hr₁ : LT.lt 0 r₁
    r : Real
    hr₀ : LT.lt 0 r
    hr : LT.lt r₂ (HMul.hMul r r₁)
    a : 𝕜
    ha : LE.le r (Norm.norm a)
    x : E
    hx : LT.lt (p x) r₂
    ⊢ LT.lt (p x) (HMul.hMul (Norm.norm a) r₁)
  -/
  exact hx.trans (hr.trans_le <| by gcongr)
  /-
    🎉 no goals
  -/


/-- Seminorm-balls at the origin are absorbent. -/
protected theorem absorbent_ball_zero (hr : 0 < r) : Absorbent 𝕜 (ball p (0 : E) r) :=
  absorbent_iff_forall_absorbs_singleton.2 fun _ =>
    (p.ball_zero_absorbs_ball_zero hr).mono_right <|
      singleton_subset_iff.2 <| p.mem_ball_zero.2 <| lt_add_one _


/-- Closed seminorm-balls at the origin are absorbent. -/
protected theorem absorbent_closedBall_zero (hr : 0 < r) : Absorbent 𝕜 (closedBall p (0 : E) r) :=
  (p.absorbent_ball_zero hr).mono (p.ball_subset_closedBall _ _)


/-- Seminorm-balls containing the origin are absorbent. -/
protected theorem absorbent_ball (hpr : p x < r) : Absorbent 𝕜 (ball p x r) := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    r : Real
    x : E
    hpr : LT.lt (p x) r
    ⊢ Absorbent 𝕜 (p.ball x r)
  -/
  refine (p.absorbent_ball_zero <| sub_pos.2 hpr).mono fun y hy => ?_
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    r : Real
    x : E
    hpr : LT.lt (p x) r
    y : E
    hy : Membership.mem (p.ball 0 (HSub.hSub r (p x))) y
    ⊢ Membership.mem (p.ball x r) y
  -/
  rw [p.mem_ball_zero] at hy
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    r : Real
    x : E
    hpr : LT.lt (p x) r
    y : E
    hy : LT.lt (p y) (HSub.hSub r (p x))
    ⊢ Membership.mem (p.ball x r) y
  -/
  exact p.mem_ball.2 ((map_sub_le_add p _ _).trans_lt <| add_lt_of_lt_sub_right hy)
  /-
    🎉 no goals
  -/


/-- Seminorm-balls containing the origin are absorbent. -/
protected theorem absorbent_closedBall (hpr : p x < r) : Absorbent 𝕜 (closedBall p x r) := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    r : Real
    x : E
    hpr : LT.lt (p x) r
    ⊢ Absorbent 𝕜 (p.closedBall x r)
  -/
  refine (p.absorbent_closedBall_zero <| sub_pos.2 hpr).mono fun y hy => ?_
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    r : Real
    x : E
    hpr : LT.lt (p x) r
    y : E
    hy : Membership.mem (p.closedBall 0 (HSub.hSub r (p x))) y
    ⊢ Membership.mem (p.closedBall x r) y
  -/
  rw [p.mem_closedBall_zero] at hy
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    r : Real
    x : E
    hpr : LT.lt (p x) r
    y : E
    hy : LE.le (p y) (HSub.hSub r (p x))
    ⊢ Membership.mem (p.closedBall x r) y
  -/
  exact p.mem_closedBall.2 ((map_sub_le_add p _ _).trans <| add_le_of_le_sub_right hy)
  /-
    🎉 no goals
  -/


@[simp]
theorem smul_ball_preimage (p : Seminorm 𝕜 E) (y : E) (r : ℝ) (a : 𝕜) (ha : a ≠ 0) :
    (a • ·) ⁻¹' p.ball y r = p.ball (a⁻¹ • y) (r / ‖a‖) :=
  Set.ext fun _ => by
    rw [mem_preimage, mem_ball, mem_ball, lt_div_iff₀ (norm_pos_iff.mpr ha), mul_comm, ←
      map_smul_eq_mul p, smul_sub, smul_inv_smul₀ ha]


/-- A seminorm is convex. Also see `convexOn_norm`. -/
protected theorem convexOn : ConvexOn ℝ univ p := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝⁵ : NormedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : NormedSpace Real 𝕜
    inst✝² : Module 𝕜 E
    inst✝¹ : SMul Real E
    inst✝ : IsScalarTower Real 𝕜 E
    p : Seminorm 𝕜 E
    ⊢ ConvexOn Real Set.univ ⇑p
  -/
  refine ⟨convex_univ, fun x _ y _ a b ha hb _ => ?_⟩
  calc
    p (a • x + b • y) ≤ p (a • x) + p (b • y) := map_add_le_add p _ _
    _ = ‖a • (1 : 𝕜)‖ * p x + ‖b • (1 : 𝕜)‖ * p y := by
      rw [← map_smul_eq_mul p, ← map_smul_eq_mul p, smul_one_smul, smul_one_smul]
    _ = a * p x + b * p y := by
      rw [norm_smul, norm_smul, norm_one, mul_one, mul_one, Real.norm_of_nonneg ha,
        Real.norm_of_nonneg hb]


/-- Seminorm-balls are convex. -/
theorem convex_ball : Convex ℝ (ball p x r) := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝⁵ : NormedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : NormedSpace Real 𝕜
    inst✝² : Module 𝕜 E
    inst✝¹ : Module Real E
    inst✝ : IsScalarTower Real 𝕜 E
    p : Seminorm 𝕜 E
    x : E
    r : Real
    ⊢ Convex Real (p.ball x r)
  -/
  convert (p.convexOn.translate_left (-x)).convex_lt r
  /-
    case h.e'_6
    𝕜 : Type u_3
    E : Type u_7
    inst✝⁵ : NormedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : NormedSpace Real 𝕜
    inst✝² : Module 𝕜 E
    inst✝¹ : Module Real E
    inst✝ : IsScalarTower Real 𝕜 E
    p : Seminorm 𝕜 E
    x : E
    r : Real
    ⊢ Eq (p.ball x r) (setOf fun x_1 => And (Membership.mem (Set.preimage (fun z = …
  -/
  ext y
  /-
    case h.e'_6.h
    𝕜 : Type u_3
    E : Type u_7
    inst✝⁵ : NormedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : NormedSpace Real 𝕜
    inst✝² : Module 𝕜 E
    inst✝¹ : Module Real E
    inst✝ : IsScalarTower Real 𝕜 E
    p : Seminorm 𝕜 E
    x : E
    r : Real
    y : E
    ⊢ Iff (Membership.mem (p.ball x r) y) (Membership.mem (setOf fun x_1 => And (M …
  -/
  rw [preimage_univ, sep_univ, p.mem_ball, sub_eq_add_neg]
  /-
    case h.e'_6.h
    𝕜 : Type u_3
    E : Type u_7
    inst✝⁵ : NormedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : NormedSpace Real 𝕜
    inst✝² : Module 𝕜 E
    inst✝¹ : Module Real E
    inst✝ : IsScalarTower Real 𝕜 E
    p : Seminorm 𝕜 E
    x : E
    r : Real
    y : E
    ⊢ Iff (LT.lt (p (HAdd.hAdd y (Neg.neg x))) r) (Membership.mem (setOf fun x_1 = …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Closed seminorm-balls are convex. -/
theorem convex_closedBall : Convex ℝ (closedBall p x r) := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝⁵ : NormedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : NormedSpace Real 𝕜
    inst✝² : Module 𝕜 E
    inst✝¹ : Module Real E
    inst✝ : IsScalarTower Real 𝕜 E
    p : Seminorm 𝕜 E
    x : E
    r : Real
    ⊢ Convex Real (p.closedBall x r)
  -/
  rw [closedBall_eq_biInter_ball]
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝⁵ : NormedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : NormedSpace Real 𝕜
    inst✝² : Module 𝕜 E
    inst✝¹ : Module Real E
    inst✝ : IsScalarTower Real 𝕜 E
    p : Seminorm 𝕜 E
    x : E
    r : Real
    ⊢ Convex Real (Set.iInter fun ρ => Set.iInter fun h => p.ball x ρ)
  -/
  exact convex_iInter₂ fun _ _ => convex_ball _ _ _
  /-
    🎉 no goals
  -/


/-- Reinterpret a seminorm over a field `𝕜'` as a seminorm over a smaller field `𝕜`. This will
typically be used with `RCLike 𝕜'` and `𝕜 = ℝ`. -/
protected def restrictScalars (p : Seminorm 𝕜' E) : Seminorm 𝕜 E :=
  { p with
                           /-
                             R : Type u_1
                             R' : Type u_2
                             𝕜 : Type u_3
                             𝕜₂ : Type u_4
                             𝕜₃ : Type u_5
                             𝕝 : Type u_6
                             E : Type u_7
                             E₂ : Type u_8
                             E₃ : Type u_9
                             F : Type u_10
                             ι : Type u_11
                             𝕜' : Type u_12
                             inst✝⁷ : NormedField 𝕜
                             inst✝⁶ : SeminormedRing 𝕜'
                             inst✝⁵ : NormedAlgebra 𝕜 𝕜'
                             inst✝⁴ : NormOneClass 𝕜'
                             inst✝³ : AddCommGroup E
                             inst✝² : Module 𝕜' E
                             inst✝¹ : SMul 𝕜 E
                             inst✝ : IsScalarTower 𝕜 𝕜' E
                             p : Seminorm 𝕜' E
                             a : 𝕜
                             x : E
                             ⊢ Eq (p.toFun (HSMul.hSMul a x)) (HMul.hMul (Norm.norm a) (p.toFun x))
                           -/
    smul' := fun a x => by rw [← smul_one_smul 𝕜' a x, p.smul', norm_smul, norm_one, mul_one] }
                           /-
                             🎉 no goals
                           -/


@[simp]
theorem coe_restrictScalars (p : Seminorm 𝕜' E) : (p.restrictScalars 𝕜 : E → ℝ) = p :=
  rfl


@[simp]
theorem restrictScalars_ball (p : Seminorm 𝕜' E) : (p.restrictScalars 𝕜).ball = p.ball :=
  rfl


@[simp]
theorem restrictScalars_closedBall (p : Seminorm 𝕜' E) :
    (p.restrictScalars 𝕜).closedBall = p.closedBall :=
  rfl


/-- A seminorm is continuous at `0` if `p.closedBall 0 r ∈ 𝓝 0` for *all* `r > 0`.
Over a `NontriviallyNormedField` it is actually enough to check that this is true
for *some* `r`, see `Seminorm.continuousAt_zero'`. -/
theorem continuousAt_zero_of_forall' [TopologicalSpace E] {p : Seminorm 𝕝 E}
    (hp : ∀ r > 0, p.closedBall 0 r ∈ (𝓝 0 : Filter E)) :
    ContinuousAt p 0 := by
  /-
    𝕝 : Type u_6
    E : Type u_7
    inst✝³ : SeminormedRing 𝕝
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕝 E
    inst✝ : TopologicalSpace E
    p : Seminorm 𝕝 E
    hp : ∀ (r : Real), GT.gt r 0 → Membership.mem (nhds 0) (p.closedBall 0 r)
    ⊢ ContinuousAt (⇑p) 0
  -/
  simp_rw [Seminorm.closedBall_zero_eq_preimage_closedBall] at hp
  /-
    𝕝 : Type u_6
    E : Type u_7
    inst✝³ : SeminormedRing 𝕝
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕝 E
    inst✝ : TopologicalSpace E
    p : Seminorm 𝕝 E
    hp : ∀ (r : Real), GT.gt r 0 → Membership.mem (nhds 0) (Set.preimage (⇑p) (Met …
    ⊢ ContinuousAt (⇑p) 0
  -/
  rwa [ContinuousAt, Metric.nhds_basis_closedBall.tendsto_right_iff, map_zero]
  /-
    🎉 no goals
  -/


theorem continuousAt_zero' [TopologicalSpace E] [ContinuousConstSMul 𝕜 E] {p : Seminorm 𝕜 E}
    {r : ℝ} (hp : p.closedBall 0 r ∈ (𝓝 0 : Filter E)) : ContinuousAt p 0 := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : TopologicalSpace E
    inst✝ : ContinuousConstSMul 𝕜 E
    p : Seminorm 𝕜 E
    r : Real
    hp : Membership.mem (nhds 0) (p.closedBall 0 r)
    ⊢ ContinuousAt (⇑p) 0
  -/
  refine continuousAt_zero_of_forall' fun ε hε ↦ ?_
  obtain ⟨k, hk₀, hk⟩ : ∃ k : 𝕜, 0 < ‖k‖ ∧ ‖k‖ * r < ε := by
    rcases le_or_lt r 0 with hr | hr
    · use 1; simpa using hr.trans_lt hε
    · simpa [lt_div_iff₀ hr] using exists_norm_lt 𝕜 (div_pos hε hr)
  /-
    case intro.intro
    𝕜 : Type u_3
    E : Type u_7
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : TopologicalSpace E
    inst✝ : ContinuousConstSMul 𝕜 E
    p : Seminorm 𝕜 E
    r : Real
    hp : Membership.mem (nhds 0) (p.closedBall 0 r)
    ε : Real
    hε : GT.gt ε 0
    k : 𝕜
    hk₀ : LT.lt 0 (Norm.norm k)
    hk : LT.lt (HMul.hMul (Norm.norm k) r) ε
    ⊢ Membership.mem (nhds 0) (p.closedBall 0 ε)
  -/
  rw [← set_smul_mem_nhds_zero_iff (norm_pos_iff.1 hk₀), smul_closedBall_zero hk₀] at hp
  /-
    case intro.intro
    𝕜 : Type u_3
    E : Type u_7
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : TopologicalSpace E
    inst✝ : ContinuousConstSMul 𝕜 E
    p : Seminorm 𝕜 E
    r ε : Real
    hε : GT.gt ε 0
    k : 𝕜
    hp : Membership.mem (nhds 0) (p.closedBall 0 (HMul.hMul (Norm.norm k) r))
    hk₀ : LT.lt 0 (Norm.norm k)
    hk : LT.lt (HMul.hMul (Norm.norm k) r) ε
    ⊢ Membership.mem (nhds 0) (p.closedBall 0 ε)
  -/
  exact mem_of_superset hp <| p.closedBall_mono hk.le
  /-
    🎉 no goals
  -/


/-- A seminorm is continuous at `0` if `p.ball 0 r ∈ 𝓝 0` for *all* `r > 0`.
Over a `NontriviallyNormedField` it is actually enough to check that this is true
for *some* `r`, see `Seminorm.continuousAt_zero'`. -/
theorem continuousAt_zero_of_forall [TopologicalSpace E] {p : Seminorm 𝕝 E}
    (hp : ∀ r > 0, p.ball 0 r ∈ (𝓝 0 : Filter E)) :
    ContinuousAt p 0 :=
  continuousAt_zero_of_forall'
    (fun r hr ↦ Filter.mem_of_superset (hp r hr) <| p.ball_subset_closedBall _ _)


theorem continuousAt_zero [TopologicalSpace E] [ContinuousConstSMul 𝕜 E] {p : Seminorm 𝕜 E} {r : ℝ}
    (hp : p.ball 0 r ∈ (𝓝 0 : Filter E)) : ContinuousAt p 0 :=
  continuousAt_zero' (Filter.mem_of_superset hp <| p.ball_subset_closedBall _ _)


protected theorem uniformContinuous_of_continuousAt_zero [UniformSpace E] [UniformAddGroup E]
    {p : Seminorm 𝕝 E} (hp : ContinuousAt p 0) : UniformContinuous p := by
  /-
    𝕝 : Type u_6
    E : Type u_7
    inst✝⁴ : SeminormedRing 𝕝
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕝 E
    inst✝¹ : UniformSpace E
    inst✝ : UniformAddGroup E
    p : Seminorm 𝕝 E
    hp : ContinuousAt (⇑p) 0
    ⊢ UniformContinuous ⇑p
  -/
  have hp : Filter.Tendsto p (𝓝 0) (𝓝 0) := map_zero p ▸ hp
  rw [UniformContinuous, uniformity_eq_comap_nhds_zero_swapped,
    Metric.uniformity_eq_comap_nhds_zero, Filter.tendsto_comap_iff]
  exact
    tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds (hp.comp Filter.tendsto_comap)
      (fun xy => dist_nonneg) fun xy => p.norm_sub_map_le_sub _ _


protected theorem continuous_of_continuousAt_zero [TopologicalSpace E] [TopologicalAddGroup E]
    {p : Seminorm 𝕝 E} (hp : ContinuousAt p 0) : Continuous p := by
  /-
    𝕝 : Type u_6
    E : Type u_7
    inst✝⁴ : SeminormedRing 𝕝
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕝 E
    inst✝¹ : TopologicalSpace E
    inst✝ : TopologicalAddGroup E
    p : Seminorm 𝕝 E
    hp : ContinuousAt (⇑p) 0
    ⊢ Continuous ⇑p
  -/
  letI := TopologicalAddGroup.toUniformSpace E
  /-
    𝕝 : Type u_6
    E : Type u_7
    inst✝⁴ : SeminormedRing 𝕝
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕝 E
    inst✝¹ : TopologicalSpace E
    inst✝ : TopologicalAddGroup E
    p : Seminorm 𝕝 E
    hp : ContinuousAt (⇑p) 0
    this : UniformSpace E := TopologicalAddGroup.toUniformSpace E
    ⊢ Continuous ⇑p
  -/
  haveI : UniformAddGroup E := comm_topologicalAddGroup_is_uniform
  /-
    𝕝 : Type u_6
    E : Type u_7
    inst✝⁴ : SeminormedRing 𝕝
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕝 E
    inst✝¹ : TopologicalSpace E
    inst✝ : TopologicalAddGroup E
    p : Seminorm 𝕝 E
    hp : ContinuousAt (⇑p) 0
    this✝ : UniformSpace E := TopologicalAddGroup.toUniformSpace E
    this : UniformAddGroup E
    ⊢ Continuous ⇑p
  -/
  exact (Seminorm.uniformContinuous_of_continuousAt_zero hp).continuous
  /-
    🎉 no goals
  -/


/-- A seminorm is uniformly continuous if `p.ball 0 r ∈ 𝓝 0` for *all* `r > 0`.
Over a `NontriviallyNormedField` it is actually enough to check that this is true
for *some* `r`, see `Seminorm.uniformContinuous`. -/
protected theorem uniformContinuous_of_forall [UniformSpace E] [UniformAddGroup E]
    {p : Seminorm 𝕝 E} (hp : ∀ r > 0, p.ball 0 r ∈ (𝓝 0 : Filter E)) :
    UniformContinuous p :=
  Seminorm.uniformContinuous_of_continuousAt_zero (continuousAt_zero_of_forall hp)


protected theorem uniformContinuous [UniformSpace E] [UniformAddGroup E] [ContinuousConstSMul 𝕜 E]
    {p : Seminorm 𝕜 E} {r : ℝ} (hp : p.ball 0 r ∈ (𝓝 0 : Filter E)) :
    UniformContinuous p :=
  Seminorm.uniformContinuous_of_continuousAt_zero (continuousAt_zero hp)


/-- A seminorm is uniformly continuous if `p.closedBall 0 r ∈ 𝓝 0` for *all* `r > 0`.
Over a `NontriviallyNormedField` it is actually enough to check that this is true
for *some* `r`, see `Seminorm.uniformContinuous'`. -/
protected theorem uniformContinuous_of_forall' [UniformSpace E] [UniformAddGroup E]
    {p : Seminorm 𝕝 E} (hp : ∀ r > 0, p.closedBall 0 r ∈ (𝓝 0 : Filter E)) :
    UniformContinuous p :=
  Seminorm.uniformContinuous_of_continuousAt_zero (continuousAt_zero_of_forall' hp)


protected theorem uniformContinuous' [UniformSpace E] [UniformAddGroup E] [ContinuousConstSMul 𝕜 E]
    {p : Seminorm 𝕜 E} {r : ℝ} (hp : p.closedBall 0 r ∈ (𝓝 0 : Filter E)) :
    UniformContinuous p :=
  Seminorm.uniformContinuous_of_continuousAt_zero (continuousAt_zero' hp)


/-- A seminorm is continuous if `p.ball 0 r ∈ 𝓝 0` for *all* `r > 0`.
Over a `NontriviallyNormedField` it is actually enough to check that this is true
for *some* `r`, see `Seminorm.continuous`. -/
protected theorem continuous_of_forall [TopologicalSpace E] [TopologicalAddGroup E]
    {p : Seminorm 𝕝 E} (hp : ∀ r > 0, p.ball 0 r ∈ (𝓝 0 : Filter E)) :
    Continuous p :=
  Seminorm.continuous_of_continuousAt_zero (continuousAt_zero_of_forall hp)


protected theorem continuous [TopologicalSpace E] [TopologicalAddGroup E] [ContinuousConstSMul 𝕜 E]
    {p : Seminorm 𝕜 E} {r : ℝ} (hp : p.ball 0 r ∈ (𝓝 0 : Filter E)) : Continuous p :=
  Seminorm.continuous_of_continuousAt_zero (continuousAt_zero hp)


/-- A seminorm is continuous if `p.closedBall 0 r ∈ 𝓝 0` for *all* `r > 0`.
Over a `NontriviallyNormedField` it is actually enough to check that this is true
for *some* `r`, see `Seminorm.continuous'`. -/
protected theorem continuous_of_forall' [TopologicalSpace E] [TopologicalAddGroup E]
    {p : Seminorm 𝕝 E} (hp : ∀ r > 0, p.closedBall 0 r ∈ (𝓝 0 : Filter E)) :
    Continuous p :=
  Seminorm.continuous_of_continuousAt_zero (continuousAt_zero_of_forall' hp)


protected theorem continuous' [TopologicalSpace E] [TopologicalAddGroup E] [ContinuousConstSMul 𝕜 E]
    {p : Seminorm 𝕜 E} {r : ℝ} (hp : p.closedBall 0 r ∈ (𝓝 0 : Filter E)) :
    Continuous p :=
  Seminorm.continuous_of_continuousAt_zero (continuousAt_zero' hp)


theorem continuous_of_le [TopologicalSpace E] [TopologicalAddGroup E]
    {p q : Seminorm 𝕝 E} (hq : Continuous q) (hpq : p ≤ q) : Continuous p := by
  refine Seminorm.continuous_of_forall (fun r hr ↦ Filter.mem_of_superset
    (IsOpen.mem_nhds ?_ <| q.mem_ball_self hr) (ball_antitone hpq))
  /-
    𝕝 : Type u_6
    E : Type u_7
    inst✝⁴ : SeminormedRing 𝕝
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕝 E
    inst✝¹ : TopologicalSpace E
    inst✝ : TopologicalAddGroup E
    p q : Seminorm 𝕝 E
    hq : Continuous ⇑q
    hpq : LE.le p q
    r : Real
    hr : GT.gt r 0
    ⊢ IsOpen (q.ball 0 r)
  -/
  rw [ball_zero_eq]
  /-
    𝕝 : Type u_6
    E : Type u_7
    inst✝⁴ : SeminormedRing 𝕝
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕝 E
    inst✝¹ : TopologicalSpace E
    inst✝ : TopologicalAddGroup E
    p q : Seminorm 𝕝 E
    hq : Continuous ⇑q
    hpq : LE.le p q
    r : Real
    hr : GT.gt r 0
    ⊢ IsOpen (setOf fun y => LT.lt (q y) r)
  -/
  exact isOpen_lt hq continuous_const
  /-
    🎉 no goals
  -/


lemma ball_mem_nhds [TopologicalSpace E] {p : Seminorm 𝕝 E} (hp : Continuous p) {r : ℝ}
    (hr : 0 < r) : p.ball 0 r ∈ (𝓝 0 : Filter E) :=
  have this : Tendsto p (𝓝 0) (𝓝 0) := map_zero p ▸ hp.tendsto 0
     /-
       𝕝 : Type u_6
       E : Type u_7
       inst✝³ : SeminormedRing 𝕝
       inst✝² : AddCommGroup E
       inst✝¹ : Module 𝕝 E
       inst✝ : TopologicalSpace E
       p : Seminorm 𝕝 E
       hp : Continuous ⇑p
       r : Real
       hr : LT.lt 0 r
       this : Filter.Tendsto (⇑p) (nhds 0) (nhds 0)
       ⊢ Membership.mem (nhds 0) (p.ball 0 r)
     -/
  by simpa only [p.ball_zero_eq] using this (Iio_mem_nhds hr)
     /-
       🎉 no goals
     -/


lemma uniformSpace_eq_of_hasBasis
    {ι} [UniformSpace E] [UniformAddGroup E] [ContinuousConstSMul 𝕜 E]
    {p' : ι → Prop} {s : ι → Set E} (p : Seminorm 𝕜 E) (hb : (𝓝 0 : Filter E).HasBasis p' s)
    (h₁ : ∃ r, p.closedBall 0 r ∈ 𝓝 0) (h₂ : ∀ i, p' i → ∃ r > 0, p.ball 0 r ⊆ s i) :
    ‹UniformSpace E› = p.toAddGroupSeminorm.toSeminormedAddGroup.toUniformSpace := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    ι : Sort u_12
    inst✝² : UniformSpace E
    inst✝¹ : UniformAddGroup E
    inst✝ : ContinuousConstSMul 𝕜 E
    p' : ι → Prop
    s : ι → Set E
    p : Seminorm 𝕜 E
    hb : (nhds 0).HasBasis p' s
    h₁ : Exists fun r => Membership.mem (nhds 0) (p.closedBall 0 r)
    h₂ : ∀ (i : ι), p' i → Exists fun r => And (GT.gt r 0) (HasSubset.Subset (p.ba …
    ⊢ Eq inst✝² PseudoMetricSpace.toUniformSpace
  -/
  refine UniformAddGroup.ext ‹_› p.toAddGroupSeminorm.toSeminormedAddCommGroup.to_uniformAddGroup ?_
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    ι : Sort u_12
    inst✝² : UniformSpace E
    inst✝¹ : UniformAddGroup E
    inst✝ : ContinuousConstSMul 𝕜 E
    p' : ι → Prop
    s : ι → Set E
    p : Seminorm 𝕜 E
    hb : (nhds 0).HasBasis p' s
    h₁ : Exists fun r => Membership.mem (nhds 0) (p.closedBall 0 r)
    h₂ : ∀ (i : ι), p' i → Exists fun r => And (GT.gt r 0) (HasSubset.Subset (p.ba …
    ⊢ Eq (nhds 0) (nhds 0)
  -/
  apply le_antisymm
    /-
      case a
      𝕜 : Type u_3
      E : Type u_7
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module 𝕜 E
      ι : Sort u_12
      inst✝² : UniformSpace E
      inst✝¹ : UniformAddGroup E
      inst✝ : ContinuousConstSMul 𝕜 E
      p' : ι → Prop
      s : ι → Set E
      p : Seminorm 𝕜 E
      hb : (nhds 0).HasBasis p' s
      h₁ : Exists fun r => Membership.mem (nhds 0) (p.closedBall 0 r)
      h₂ : ∀ (i : ι), p' i → Exists fun r => And (GT.gt r 0) (HasSubset.Subset (p.ba …
      ⊢ LE.le (nhds 0) (nhds 0)
    -/
  · rw [← @comap_norm_nhds_zero E p.toAddGroupSeminorm.toSeminormedAddGroup, ← tendsto_iff_comap]
    /-
      case a
      𝕜 : Type u_3
      E : Type u_7
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module 𝕜 E
      ι : Sort u_12
      inst✝² : UniformSpace E
      inst✝¹ : UniformAddGroup E
      inst✝ : ContinuousConstSMul 𝕜 E
      p' : ι → Prop
      s : ι → Set E
      p : Seminorm 𝕜 E
      hb : (nhds 0).HasBasis p' s
      h₁ : Exists fun r => Membership.mem (nhds 0) (p.closedBall 0 r)
      h₂ : ∀ (i : ι), p' i → Exists fun r => And (GT.gt r 0) (HasSubset.Subset (p.ba …
      ⊢ Filter.Tendsto Norm.norm (nhds 0) (nhds 0)
    -/
    suffices Continuous p from this.tendsto' 0 _ (map_zero p)
    /-
      case a
      𝕜 : Type u_3
      E : Type u_7
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module 𝕜 E
      ι : Sort u_12
      inst✝² : UniformSpace E
      inst✝¹ : UniformAddGroup E
      inst✝ : ContinuousConstSMul 𝕜 E
      p' : ι → Prop
      s : ι → Set E
      p : Seminorm 𝕜 E
      hb : (nhds 0).HasBasis p' s
      h₁ : Exists fun r => Membership.mem (nhds 0) (p.closedBall 0 r)
      h₂ : ∀ (i : ι), p' i → Exists fun r => And (GT.gt r 0) (HasSubset.Subset (p.ba …
      ⊢ Continuous ⇑p
    -/
    rcases h₁ with ⟨r, hr⟩
    /-
      case a.intro
      𝕜 : Type u_3
      E : Type u_7
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module 𝕜 E
      ι : Sort u_12
      inst✝² : UniformSpace E
      inst✝¹ : UniformAddGroup E
      inst✝ : ContinuousConstSMul 𝕜 E
      p' : ι → Prop
      s : ι → Set E
      p : Seminorm 𝕜 E
      hb : (nhds 0).HasBasis p' s
      h₂ : ∀ (i : ι), p' i → Exists fun r => And (GT.gt r 0) (HasSubset.Subset (p.ba …
      r : Real
      hr : Membership.mem (nhds 0) (p.closedBall 0 r)
      ⊢ Continuous ⇑p
    -/
    exact p.continuous' hr
    /-
      🎉 no goals
    -/
  · rw [(@NormedAddCommGroup.nhds_zero_basis_norm_lt E
      p.toAddGroupSeminorm.toSeminormedAddGroup).le_basis_iff hb]
    /-
      case a
      𝕜 : Type u_3
      E : Type u_7
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module 𝕜 E
      ι : Sort u_12
      inst✝² : UniformSpace E
      inst✝¹ : UniformAddGroup E
      inst✝ : ContinuousConstSMul 𝕜 E
      p' : ι → Prop
      s : ι → Set E
      p : Seminorm 𝕜 E
      hb : (nhds 0).HasBasis p' s
      h₁ : Exists fun r => Membership.mem (nhds 0) (p.closedBall 0 r)
      h₂ : ∀ (i : ι), p' i → Exists fun r => And (GT.gt r 0) (HasSubset.Subset (p.ba …
      ⊢ ∀ (i' : ι), p' i' → Exists fun i => And (LT.lt 0 i) (HasSubset.Subset (setOf …
    -/
    simpa only [subset_def, mem_ball_zero] using h₂
    /-
      🎉 no goals
    -/


lemma uniformity_eq_of_hasBasis
    {ι} [UniformSpace E] [UniformAddGroup E] [ContinuousConstSMul 𝕜 E]
    {p' : ι → Prop} {s : ι → Set E} (p : Seminorm 𝕜 E) (hb : (𝓝 0 : Filter E).HasBasis p' s)
    (h₁ : ∃ r, p.closedBall 0 r ∈ 𝓝 0) (h₂ : ∀ i, p' i → ∃ r > 0, p.ball 0 r ⊆ s i) :
    𝓤 E = ⨅ r > 0, 𝓟 {x | p (x.1 - x.2) < r} := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    ι : Sort u_12
    inst✝² : UniformSpace E
    inst✝¹ : UniformAddGroup E
    inst✝ : ContinuousConstSMul 𝕜 E
    p' : ι → Prop
    s : ι → Set E
    p : Seminorm 𝕜 E
    hb : (nhds 0).HasBasis p' s
    h₁ : Exists fun r => Membership.mem (nhds 0) (p.closedBall 0 r)
    h₂ : ∀ (i : ι), p' i → Exists fun r => And (GT.gt r 0) (HasSubset.Subset (p.ba …
    ⊢ Eq (uniformity E) (iInf fun r => iInf fun h => Filter.principal (setOf fun x …
  -/
  rw [uniformSpace_eq_of_hasBasis p hb h₁ h₂]; rfl
                                               /-
                                                 🎉 no goals
                                               -/


/-- Let `p` be a seminorm on a vector space over a `NormedField`.
If there is a scalar `c` with `‖c‖>1`, then any `x` such that `p x ≠ 0` can be
moved by scalar multiplication to any `p`-shell of width `‖c‖`. Also recap information on the
value of `p` on the rescaling element that shows up in applications. -/
lemma rescale_to_shell_zpow (p : Seminorm 𝕜 E) {c : 𝕜} (hc : 1 < ‖c‖) {ε : ℝ}
    (εpos : 0 < ε) {x : E} (hx : p x ≠ 0) : ∃ n : ℤ, c^n ≠ 0 ∧
    p (c^n • x) < ε ∧ (ε / ‖c‖ ≤ p (c^n • x)) ∧ (‖c^n‖⁻¹ ≤ ε⁻¹ * ‖c‖ * p x) := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    ε : Real
    εpos : LT.lt 0 ε
    x : E
    hx : Ne (p x) 0
    ⊢ Exists fun n => And (Ne (HPow.hPow c n) 0) (And (LT.lt (p (HSMul.hSMul (HPow …
  -/
  have xεpos : 0 < (p x)/ε := by positivity
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    ε : Real
    εpos : LT.lt 0 ε
    x : E
    hx : Ne (p x) 0
    xεpos : LT.lt 0 (HDiv.hDiv (p x) ε)
    ⊢ Exists fun n => And (Ne (HPow.hPow c n) 0) (And (LT.lt (p (HSMul.hSMul (HPow …
  -/
  rcases exists_mem_Ico_zpow xεpos hc with ⟨n, hn⟩
  /-
    case intro
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    ε : Real
    εpos : LT.lt 0 ε
    x : E
    hx : Ne (p x) 0
    xεpos : LT.lt 0 (HDiv.hDiv (p x) ε)
    n : Int
    hn : Membership.mem (Set.Ico (HPow.hPow (Norm.norm c) n) (HPow.hPow (Norm.norm …
    ⊢ Exists fun n => And (Ne (HPow.hPow c n) 0) (And (LT.lt (p (HSMul.hSMul (HPow …
  -/
  have cpos : 0 < ‖c‖ := by positivity
  /-
    case intro
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    ε : Real
    εpos : LT.lt 0 ε
    x : E
    hx : Ne (p x) 0
    xεpos : LT.lt 0 (HDiv.hDiv (p x) ε)
    n : Int
    hn : Membership.mem (Set.Ico (HPow.hPow (Norm.norm c) n) (HPow.hPow (Norm.norm …
    cpos : LT.lt 0 (Norm.norm c)
    ⊢ Exists fun n => And (Ne (HPow.hPow c n) 0) (And (LT.lt (p (HSMul.hSMul (HPow …
  -/
  have cnpos : 0 < ‖c^(n+1)‖ := by rw [norm_zpow]; exact xεpos.trans hn.2
  /-
    case intro
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : Seminorm 𝕜 E
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    ε : Real
    εpos : LT.lt 0 ε
    x : E
    hx : Ne (p x) 0
    xεpos : LT.lt 0 (HDiv.hDiv (p x) ε)
    n : Int
    hn : Membership.mem (Set.Ico (HPow.hPow (Norm.norm c) n) (HPow.hPow (Norm.norm …
    cpos : LT.lt 0 (Norm.norm c)
    cnpos : LT.lt 0 (Norm.norm (HPow.hPow c (HAdd.hAdd n 1)))
    ⊢ Exists fun n => And (Ne (HPow.hPow c n) 0) (And (LT.lt (p (HSMul.hSMul (HPow …
  -/
  refine ⟨-(n+1), ?_, ?_, ?_, ?_⟩
    /-
      case intro.refine_1
      𝕜 : Type u_3
      E : Type u_7
      inst✝² : NormedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      p : Seminorm 𝕜 E
      c : 𝕜
      hc : LT.lt 1 (Norm.norm c)
      ε : Real
      εpos : LT.lt 0 ε
      x : E
      hx : Ne (p x) 0
      xεpos : LT.lt 0 (HDiv.hDiv (p x) ε)
      n : Int
      hn : Membership.mem (Set.Ico (HPow.hPow (Norm.norm c) n) (HPow.hPow (Norm.norm …
      cpos : LT.lt 0 (Norm.norm c)
      cnpos : LT.lt 0 (Norm.norm (HPow.hPow c (HAdd.hAdd n 1)))
      ⊢ Ne (HPow.hPow c (Neg.neg (HAdd.hAdd n 1))) 0
    -/
  · show c ^ (-(n + 1)) ≠ 0; exact zpow_ne_zero _ (norm_pos_iff.1 cpos)
                             /-
                               🎉 no goals
                             -/
    /-
      case intro.refine_2
      𝕜 : Type u_3
      E : Type u_7
      inst✝² : NormedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      p : Seminorm 𝕜 E
      c : 𝕜
      hc : LT.lt 1 (Norm.norm c)
      ε : Real
      εpos : LT.lt 0 ε
      x : E
      hx : Ne (p x) 0
      xεpos : LT.lt 0 (HDiv.hDiv (p x) ε)
      n : Int
      hn : Membership.mem (Set.Ico (HPow.hPow (Norm.norm c) n) (HPow.hPow (Norm.norm …
      cpos : LT.lt 0 (Norm.norm c)
      cnpos : LT.lt 0 (Norm.norm (HPow.hPow c (HAdd.hAdd n 1)))
      ⊢ LT.lt (p (HSMul.hSMul (HPow.hPow c (Neg.neg (HAdd.hAdd n 1))) x)) ε
    -/
  · show p ((c ^ (-(n + 1))) • x) < ε
    rw [map_smul_eq_mul, zpow_neg, norm_inv, ← div_eq_inv_mul, div_lt_iff₀ cnpos, mul_comm,
        norm_zpow]
    /-
      case intro.refine_2
      𝕜 : Type u_3
      E : Type u_7
      inst✝² : NormedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      p : Seminorm 𝕜 E
      c : 𝕜
      hc : LT.lt 1 (Norm.norm c)
      ε : Real
      εpos : LT.lt 0 ε
      x : E
      hx : Ne (p x) 0
      xεpos : LT.lt 0 (HDiv.hDiv (p x) ε)
      n : Int
      hn : Membership.mem (Set.Ico (HPow.hPow (Norm.norm c) n) (HPow.hPow (Norm.norm …
      cpos : LT.lt 0 (Norm.norm c)
      cnpos : LT.lt 0 (Norm.norm (HPow.hPow c (HAdd.hAdd n 1)))
      ⊢ LT.lt (p x) (HMul.hMul (HPow.hPow (Norm.norm c) (HAdd.hAdd n 1)) ε)
    -/
    exact (div_lt_iff₀ εpos).1 (hn.2)
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_3
      𝕜 : Type u_3
      E : Type u_7
      inst✝² : NormedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      p : Seminorm 𝕜 E
      c : 𝕜
      hc : LT.lt 1 (Norm.norm c)
      ε : Real
      εpos : LT.lt 0 ε
      x : E
      hx : Ne (p x) 0
      xεpos : LT.lt 0 (HDiv.hDiv (p x) ε)
      n : Int
      hn : Membership.mem (Set.Ico (HPow.hPow (Norm.norm c) n) (HPow.hPow (Norm.norm …
      cpos : LT.lt 0 (Norm.norm c)
      cnpos : LT.lt 0 (Norm.norm (HPow.hPow c (HAdd.hAdd n 1)))
      ⊢ LE.le (HDiv.hDiv ε (Norm.norm c)) (p (HSMul.hSMul (HPow.hPow c (Neg.neg (HAd …
    -/
  · show ε / ‖c‖ ≤ p (c ^ (-(n + 1)) • x)
    rw [zpow_neg, div_le_iff₀ cpos, map_smul_eq_mul, norm_inv, norm_zpow, zpow_add₀ (ne_of_gt cpos),
        zpow_one, mul_inv_rev, mul_comm, ← mul_assoc, ← mul_assoc, mul_inv_cancel₀ (ne_of_gt cpos),
        one_mul, ← div_eq_inv_mul, le_div_iff₀ (zpow_pos cpos _), mul_comm]
    /-
      case intro.refine_3
      𝕜 : Type u_3
      E : Type u_7
      inst✝² : NormedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      p : Seminorm 𝕜 E
      c : 𝕜
      hc : LT.lt 1 (Norm.norm c)
      ε : Real
      εpos : LT.lt 0 ε
      x : E
      hx : Ne (p x) 0
      xεpos : LT.lt 0 (HDiv.hDiv (p x) ε)
      n : Int
      hn : Membership.mem (Set.Ico (HPow.hPow (Norm.norm c) n) (HPow.hPow (Norm.norm …
      cpos : LT.lt 0 (Norm.norm c)
      cnpos : LT.lt 0 (Norm.norm (HPow.hPow c (HAdd.hAdd n 1)))
      ⊢ LE.le (HMul.hMul (HPow.hPow (Norm.norm c) n) ε) (p x)
    -/
    exact (le_div_iff₀ εpos).1 hn.1
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_4
      𝕜 : Type u_3
      E : Type u_7
      inst✝² : NormedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      p : Seminorm 𝕜 E
      c : 𝕜
      hc : LT.lt 1 (Norm.norm c)
      ε : Real
      εpos : LT.lt 0 ε
      x : E
      hx : Ne (p x) 0
      xεpos : LT.lt 0 (HDiv.hDiv (p x) ε)
      n : Int
      hn : Membership.mem (Set.Ico (HPow.hPow (Norm.norm c) n) (HPow.hPow (Norm.norm …
      cpos : LT.lt 0 (Norm.norm c)
      cnpos : LT.lt 0 (Norm.norm (HPow.hPow c (HAdd.hAdd n 1)))
      ⊢ LE.le (Inv.inv (Norm.norm (HPow.hPow c (Neg.neg (HAdd.hAdd n 1))))) (HMul.hM …
    -/
  · show ‖(c ^ (-(n + 1)))‖⁻¹ ≤ ε⁻¹ * ‖c‖ * p x
    /-
      case intro.refine_4
      𝕜 : Type u_3
      E : Type u_7
      inst✝² : NormedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      p : Seminorm 𝕜 E
      c : 𝕜
      hc : LT.lt 1 (Norm.norm c)
      ε : Real
      εpos : LT.lt 0 ε
      x : E
      hx : Ne (p x) 0
      xεpos : LT.lt 0 (HDiv.hDiv (p x) ε)
      n : Int
      hn : Membership.mem (Set.Ico (HPow.hPow (Norm.norm c) n) (HPow.hPow (Norm.norm …
      cpos : LT.lt 0 (Norm.norm c)
      cnpos : LT.lt 0 (Norm.norm (HPow.hPow c (HAdd.hAdd n 1)))
      ⊢ LE.le (Inv.inv (Norm.norm (HPow.hPow c (Neg.neg (HAdd.hAdd n 1))))) (HMul.hM …
    -/
    have : ε⁻¹ * ‖c‖ * p x = ε⁻¹ * p x * ‖c‖ := by ring
    rw [zpow_neg, norm_inv, inv_inv, norm_zpow, zpow_add₀ (ne_of_gt cpos), zpow_one, this,
        ← div_eq_inv_mul]
    /-
      case intro.refine_4
      𝕜 : Type u_3
      E : Type u_7
      inst✝² : NormedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      p : Seminorm 𝕜 E
      c : 𝕜
      hc : LT.lt 1 (Norm.norm c)
      ε : Real
      εpos : LT.lt 0 ε
      x : E
      hx : Ne (p x) 0
      xεpos : LT.lt 0 (HDiv.hDiv (p x) ε)
      n : Int
      hn : Membership.mem (Set.Ico (HPow.hPow (Norm.norm c) n) (HPow.hPow (Norm.norm …
      cpos : LT.lt 0 (Norm.norm c)
      cnpos : LT.lt 0 (Norm.norm (HPow.hPow c (HAdd.hAdd n 1)))
      this : Eq (HMul.hMul (HMul.hMul (Inv.inv ε) (Norm.norm c)) (p x)) (HMul.hMul ( …
      ⊢ LE.le (HMul.hMul (HPow.hPow (Norm.norm c) n) (Norm.norm c)) (HMul.hMul (HDiv …
    -/
    exact mul_le_mul_of_nonneg_right hn.1 (norm_nonneg _)
    /-
      🎉 no goals
    -/


/-- Let `p` be a seminorm on a vector space over a `NormedField`.
If there is a scalar `c` with `‖c‖>1`, then any `x` such that `p x ≠ 0` can be
moved by scalar multiplication to any `p`-shell of width `‖c‖`. Also recap information on the
value of `p` on the rescaling element that shows up in applications. -/
lemma rescale_to_shell (p : Seminorm 𝕜 E) {c : 𝕜} (hc : 1 < ‖c‖) {ε : ℝ} (εpos : 0 < ε) {x : E}
    (hx : p x ≠ 0) :
    ∃d : 𝕜, d ≠ 0 ∧ p (d • x) < ε ∧ (ε/‖c‖ ≤ p (d • x)) ∧ (‖d‖⁻¹ ≤ ε⁻¹ * ‖c‖ * p x) :=
let ⟨_, hn⟩ := p.rescale_to_shell_zpow hc εpos hx; ⟨_, hn⟩


/-- Let `p` and `q` be two seminorms on a vector space over a `NontriviallyNormedField`.
If we have `q x ≤ C * p x` on some shell of the form `{x | ε/‖c‖ ≤ p x < ε}` (where `ε > 0`
and `‖c‖ > 1`), then we also have `q x ≤ C * p x` for all `x` such that `p x ≠ 0`. -/
lemma bound_of_shell
    (p q : Seminorm 𝕜 E) {ε C : ℝ} (ε_pos : 0 < ε) {c : 𝕜} (hc : 1 < ‖c‖)
    (hf : ∀ x, ε / ‖c‖ ≤ p x → p x < ε → q x ≤ C * p x) {x : E} (hx : p x ≠ 0) :
    q x ≤ C * p x := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p q : Seminorm 𝕜 E
    ε C : Real
    ε_pos : LT.lt 0 ε
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    hf : ∀ (x : E), LE.le (HDiv.hDiv ε (Norm.norm c)) (p x) → LT.lt (p x) ε → LE.l …
    x : E
    hx : Ne (p x) 0
    ⊢ LE.le (q x) (HMul.hMul C (p x))
  -/
  rcases p.rescale_to_shell hc ε_pos hx with ⟨δ, hδ, δxle, leδx, -⟩
  simpa only [map_smul_eq_mul, mul_left_comm C, mul_le_mul_left (norm_pos_iff.2 hδ)]
    using hf (δ • x) leδx δxle


/-- A version of `Seminorm.bound_of_shell` expressed using pointwise scalar multiplication of
seminorms. -/
lemma bound_of_shell_smul
    (p q : Seminorm 𝕜 E) {ε : ℝ} {C : ℝ≥0} (ε_pos : 0 < ε) {c : 𝕜} (hc : 1 < ‖c‖)
    (hf : ∀ x, ε / ‖c‖ ≤ p x → p x < ε → q x ≤ (C • p) x) {x : E} (hx : p x ≠ 0) :
    q x ≤ (C • p) x :=
  Seminorm.bound_of_shell p q ε_pos hc hf hx


lemma bound_of_shell_sup (p : ι → Seminorm 𝕜 E) (s : Finset ι)
    (q : Seminorm 𝕜 E) {ε : ℝ} {C : ℝ≥0} (ε_pos : 0 < ε) {c : 𝕜} (hc : 1 < ‖c‖)
    (hf : ∀ x, (∀ i ∈ s, p i x < ε) → ∀ j ∈ s, ε / ‖c‖ ≤ p j x → q x ≤ (C • p j) x)
    {x : E} (hx : ∃ j, j ∈ s ∧ p j x ≠ 0) :
    q x ≤ (C • s.sup p) x := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    ι : Type u_11
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → Seminorm 𝕜 E
    s : Finset ι
    q : Seminorm 𝕜 E
    ε : Real
    C : NNReal
    ε_pos : LT.lt 0 ε
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    hf : ∀ (x : E), (∀ (i : ι), Membership.mem s i → LT.lt ((p i) x) ε) → ∀ (j : ι …
    x : E
    hx : Exists fun j => And (Membership.mem s j) (Ne ((p j) x) 0)
    ⊢ LE.le (q x) ((HSMul.hSMul C (s.sup p)) x)
  -/
  rcases hx with ⟨j, hj, hjx⟩
  have : (s.sup p) x ≠ 0 :=
    ne_of_gt ((hjx.symm.lt_of_le <| apply_nonneg _ _).trans_le (le_finset_sup_apply hj))
  /-
    case intro.intro
    𝕜 : Type u_3
    E : Type u_7
    ι : Type u_11
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → Seminorm 𝕜 E
    s : Finset ι
    q : Seminorm 𝕜 E
    ε : Real
    C : NNReal
    ε_pos : LT.lt 0 ε
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    hf : ∀ (x : E), (∀ (i : ι), Membership.mem s i → LT.lt ((p i) x) ε) → ∀ (j : ι …
    x : E
    j : ι
    hj : Membership.mem s j
    hjx : Ne ((p j) x) 0
    this : Ne ((s.sup p) x) 0
    ⊢ LE.le (q x) ((HSMul.hSMul C (s.sup p)) x)
  -/
  refine (s.sup p).bound_of_shell_smul q ε_pos hc (fun y hle hlt ↦ ?_) this
  /-
    case intro.intro
    𝕜 : Type u_3
    E : Type u_7
    ι : Type u_11
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → Seminorm 𝕜 E
    s : Finset ι
    q : Seminorm 𝕜 E
    ε : Real
    C : NNReal
    ε_pos : LT.lt 0 ε
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    hf : ∀ (x : E), (∀ (i : ι), Membership.mem s i → LT.lt ((p i) x) ε) → ∀ (j : ι …
    x : E
    j : ι
    hj : Membership.mem s j
    hjx : Ne ((p j) x) 0
    this : Ne ((s.sup p) x) 0
    y : E
    hle : LE.le (HDiv.hDiv ε (Norm.norm c)) ((s.sup p) y)
    hlt : LT.lt ((s.sup p) y) ε
    ⊢ LE.le (q y) ((HSMul.hSMul C (s.sup p)) y)
  -/
  rcases exists_apply_eq_finset_sup p ⟨j, hj⟩ y with ⟨i, hi, hiy⟩
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_3
    E : Type u_7
    ι : Type u_11
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → Seminorm 𝕜 E
    s : Finset ι
    q : Seminorm 𝕜 E
    ε : Real
    C : NNReal
    ε_pos : LT.lt 0 ε
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    hf : ∀ (x : E), (∀ (i : ι), Membership.mem s i → LT.lt ((p i) x) ε) → ∀ (j : ι …
    x : E
    j : ι
    hj : Membership.mem s j
    hjx : Ne ((p j) x) 0
    this : Ne ((s.sup p) x) 0
    y : E
    hle : LE.le (HDiv.hDiv ε (Norm.norm c)) ((s.sup p) y)
    hlt : LT.lt ((s.sup p) y) ε
    i : ι
    hi : Membership.mem s i
    hiy : Eq ((s.sup p) y) ((p i) y)
    ⊢ LE.le (q y) ((HSMul.hSMul C (s.sup p)) y)
  -/
  rw [smul_apply, hiy]
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_3
    E : Type u_7
    ι : Type u_11
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : ι → Seminorm 𝕜 E
    s : Finset ι
    q : Seminorm 𝕜 E
    ε : Real
    C : NNReal
    ε_pos : LT.lt 0 ε
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    hf : ∀ (x : E), (∀ (i : ι), Membership.mem s i → LT.lt ((p i) x) ε) → ∀ (j : ι …
    x : E
    j : ι
    hj : Membership.mem s j
    hjx : Ne ((p j) x) 0
    this : Ne ((s.sup p) x) 0
    y : E
    hle : LE.le (HDiv.hDiv ε (Norm.norm c)) ((s.sup p) y)
    hlt : LT.lt ((s.sup p) y) ε
    i : ι
    hi : Membership.mem s i
    hiy : Eq ((s.sup p) y) ((p i) y)
    ⊢ LE.le (q y) (HSMul.hSMul C ((p i) y))
  -/
  exact hf y (fun k hk ↦ (le_finset_sup_apply hk).trans_lt hlt) i hi (hiy ▸ hle)
  /-
    🎉 no goals
  -/


/-- Let `p i` be a family of seminorms on `E`. Let `s` be an absorbent set in `𝕜`.
If all seminorms are uniformly bounded at every point of `s`,
then they are bounded in the space of seminorms. -/
lemma bddAbove_of_absorbent {ι : Sort*} {p : ι → Seminorm 𝕜 E} {s : Set E} (hs : Absorbent 𝕜 s)
    (h : ∀ x ∈ s, BddAbove (range (p · x))) : BddAbove (range p) := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    ι : Sort u_12
    p : ι → Seminorm 𝕜 E
    s : Set E
    hs : Absorbent 𝕜 s
    h : ∀ (x : E), Membership.mem s x → BddAbove (Set.range fun x_1 => (p x_1) x)
    ⊢ BddAbove (Set.range p)
  -/
  rw [Seminorm.bddAbove_range_iff]
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    ι : Sort u_12
    p : ι → Seminorm 𝕜 E
    s : Set E
    hs : Absorbent 𝕜 s
    h : ∀ (x : E), Membership.mem s x → BddAbove (Set.range fun x_1 => (p x_1) x)
    ⊢ ∀ (x : E), BddAbove (Set.range fun i => (p i) x)
  -/
  intro x
  obtain ⟨c, hc₀, hc⟩ : ∃ c ≠ 0, (c : 𝕜) • x ∈ s :=
    (eventually_mem_nhdsWithin.and (hs.eventually_nhdsWithin_zero x)).exists
  /-
    case intro.intro
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    ι : Sort u_12
    p : ι → Seminorm 𝕜 E
    s : Set E
    hs : Absorbent 𝕜 s
    h : ∀ (x : E), Membership.mem s x → BddAbove (Set.range fun x_1 => (p x_1) x)
    x : E
    c : 𝕜
    hc₀ : Ne c 0
    hc : Membership.mem s (HSMul.hSMul c x)
    ⊢ BddAbove (Set.range fun i => (p i) x)
  -/
  rcases h _ hc with ⟨M, hM⟩
  /-
    case intro.intro.intro
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    ι : Sort u_12
    p : ι → Seminorm 𝕜 E
    s : Set E
    hs : Absorbent 𝕜 s
    h : ∀ (x : E), Membership.mem s x → BddAbove (Set.range fun x_1 => (p x_1) x)
    x : E
    c : 𝕜
    hc₀ : Ne c 0
    hc : Membership.mem s (HSMul.hSMul c x)
    M : Real
    hM : Membership.mem (upperBounds (Set.range fun x_1 => (p x_1) (HSMul.hSMul c  …
    ⊢ BddAbove (Set.range fun i => (p i) x)
  -/
  refine ⟨M / ‖c‖, forall_mem_range.mpr fun i ↦ (le_div_iff₀' (norm_pos_iff.2 hc₀)).2 ?_⟩
  /-
    case intro.intro.intro
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    ι : Sort u_12
    p : ι → Seminorm 𝕜 E
    s : Set E
    hs : Absorbent 𝕜 s
    h : ∀ (x : E), Membership.mem s x → BddAbove (Set.range fun x_1 => (p x_1) x)
    x : E
    c : 𝕜
    hc₀ : Ne c 0
    hc : Membership.mem s (HSMul.hSMul c x)
    M : Real
    hM : Membership.mem (upperBounds (Set.range fun x_1 => (p x_1) (HSMul.hSMul c  …
    i : ι
    ⊢ LE.le (HMul.hMul (Norm.norm c) ((p i) x)) M
  -/
  exact hM ⟨i, map_smul_eq_mul ..⟩
  /-
    🎉 no goals
  -/


/-- The norm of a seminormed group as a seminorm. -/
def normSeminorm : Seminorm 𝕜 E :=
  { normAddGroupSeminorm E with smul' := norm_smul }


@[simp]
theorem coe_normSeminorm : ⇑(normSeminorm 𝕜 E) = norm :=
  rfl


@[simp]
theorem ball_normSeminorm : (normSeminorm 𝕜 E).ball = Metric.ball := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    ⊢ Eq (normSeminorm 𝕜 E).ball Metric.ball
  -/
  ext x r y
  /-
    case h.h.h
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    r : Real
    y : E
    ⊢ Iff (Membership.mem ((normSeminorm 𝕜 E).ball x r) y) (Membership.mem (Metric …
  -/
  simp only [Seminorm.mem_ball, Metric.mem_ball, coe_normSeminorm, dist_eq_norm]
  /-
    🎉 no goals
  -/


/-- Balls at the origin are absorbent. -/
theorem absorbent_ball_zero (hr : 0 < r) : Absorbent 𝕜 (Metric.ball (0 : E) r) := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    r : Real
    hr : LT.lt 0 r
    ⊢ Absorbent 𝕜 (Metric.ball 0 r)
  -/
  rw [← ball_normSeminorm 𝕜]
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    r : Real
    hr : LT.lt 0 r
    ⊢ Absorbent 𝕜 ((normSeminorm 𝕜 E).ball 0 r)
  -/
  exact (normSeminorm _ _).absorbent_ball_zero hr
  /-
    🎉 no goals
  -/


/-- Balls containing the origin are absorbent. -/
theorem absorbent_ball (hx : ‖x‖ < r) : Absorbent 𝕜 (Metric.ball x r) := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    r : Real
    x : E
    hx : LT.lt (Norm.norm x) r
    ⊢ Absorbent 𝕜 (Metric.ball x r)
  -/
  rw [← ball_normSeminorm 𝕜]
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    r : Real
    x : E
    hx : LT.lt (Norm.norm x) r
    ⊢ Absorbent 𝕜 ((normSeminorm 𝕜 E).ball x r)
  -/
  exact (normSeminorm _ _).absorbent_ball hx
  /-
    🎉 no goals
  -/


/-- Balls at the origin are balanced. -/
theorem balanced_ball_zero : Balanced 𝕜 (Metric.ball (0 : E) r) := by
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    r : Real
    ⊢ Balanced 𝕜 (Metric.ball 0 r)
  -/
  rw [← ball_normSeminorm 𝕜]
  /-
    𝕜 : Type u_3
    E : Type u_7
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    r : Real
    ⊢ Balanced 𝕜 ((normSeminorm 𝕜 E).ball 0 r)
  -/
  exact (normSeminorm _ _).balanced_ball_zero r
  /-
    🎉 no goals
  -/


/-- If there is a scalar `c` with `‖c‖>1`, then any element with nonzero norm can be
moved by scalar multiplication to any shell of width `‖c‖`. Also recap information on the norm of
the rescaling element that shows up in applications. -/
lemma rescale_to_shell_semi_normed_zpow {c : 𝕜} (hc : 1 < ‖c‖) {ε : ℝ} (εpos : 0 < ε) {x : E}
    (hx : ‖x‖ ≠ 0) :
    ∃ n : ℤ, c^n ≠ 0 ∧ ‖c^n • x‖ < ε ∧ (ε / ‖c‖ ≤ ‖c^n • x‖) ∧ (‖c^n‖⁻¹ ≤ ε⁻¹ * ‖c‖ * ‖x‖) :=
  (normSeminorm 𝕜 E).rescale_to_shell_zpow hc εpos hx


/-- If there is a scalar `c` with `‖c‖>1`, then any element with nonzero norm can be
moved by scalar multiplication to any shell of width `‖c‖`. Also recap information on the norm of
the rescaling element that shows up in applications. -/
lemma rescale_to_shell_semi_normed {c : 𝕜} (hc : 1 < ‖c‖) {ε : ℝ} (εpos : 0 < ε)
    {x : E} (hx : ‖x‖ ≠ 0) :
    ∃d : 𝕜, d ≠ 0 ∧ ‖d • x‖ < ε ∧ (ε/‖c‖ ≤ ‖d • x‖) ∧ (‖d‖⁻¹ ≤ ε⁻¹ * ‖c‖ * ‖x‖) :=
  (normSeminorm 𝕜 E).rescale_to_shell hc εpos hx


lemma rescale_to_shell_zpow [NormedAddCommGroup F] [NormedSpace 𝕜 F] {c : 𝕜} (hc : 1 < ‖c‖)
    {ε : ℝ} (εpos : 0 < ε) {x : F} (hx : x ≠ 0) :
    ∃ n : ℤ, c^n ≠ 0 ∧ ‖c^n • x‖ < ε ∧ (ε / ‖c‖ ≤ ‖c^n • x‖) ∧ (‖c^n‖⁻¹ ≤ ε⁻¹ * ‖c‖ * ‖x‖) :=
  rescale_to_shell_semi_normed_zpow hc εpos (norm_ne_zero_iff.mpr hx)


/-- If there is a scalar `c` with `‖c‖>1`, then any element can be moved by scalar multiplication to
any shell of width `‖c‖`. Also recap information on the norm of the rescaling element that shows
up in applications. -/
lemma rescale_to_shell [NormedAddCommGroup F] [NormedSpace 𝕜 F] {c : 𝕜} (hc : 1 < ‖c‖)
    {ε : ℝ} (εpos : 0 < ε) {x : F} (hx : x ≠ 0) :
    ∃d : 𝕜, d ≠ 0 ∧ ‖d • x‖ < ε ∧ (ε/‖c‖ ≤ ‖d • x‖) ∧ (‖d‖⁻¹ ≤ ε⁻¹ * ‖c‖ * ‖x‖) :=
  rescale_to_shell_semi_normed hc εpos (norm_ne_zero_iff.mpr hx)


