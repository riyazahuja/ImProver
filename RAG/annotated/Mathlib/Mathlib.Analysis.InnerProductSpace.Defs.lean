/-- Syntactic typeclass for types endowed with an inner product -/
class Inner (𝕜 E : Type*) where
  /-- The inner product function. -/
  inner : E → E → 𝕜


/-- The inner product with values in `𝕜`. -/
scoped[InnerProductSpace] notation3:max "⟪" x ", " y "⟫_" 𝕜:max => @inner 𝕜 _ _ x y


/-- The inner product with values in `ℝ`. -/
scoped[RealInnerProductSpace] notation "⟪" x ", " y "⟫" => @inner ℝ _ _ x y


/-- The inner product with values in `ℂ`. -/
scoped[ComplexInnerProductSpace] notation "⟪" x ", " y "⟫" => @inner ℂ _ _ x y


/-- A (pre) inner product space is a vector space with an additional operation called inner product.
The (semi)norm could be derived from the inner product, instead we require the existence of a
seminorm and the fact that `‖x‖^2 = re ⟪x, x⟫` to be able to put instances on `𝕂` or product spaces.

Note that `NormedSpace` does not assume that `‖x‖=0` implies `x=0` (it is rather a seminorm).

To construct a seminorm from an inner product, see `PreInnerProductSpace.ofCore`.
-/
class InnerProductSpace (𝕜 : Type*) (E : Type*) [RCLike 𝕜] [SeminormedAddCommGroup E] extends
  NormedSpace 𝕜 E, Inner 𝕜 E where
  /-- The inner product induces the norm. -/
  norm_sq_eq_inner : ∀ x : E, ‖x‖ ^ 2 = re (inner x x)
  /-- The inner product is *hermitian*, taking the `conj` swaps the arguments. -/
  conj_symm : ∀ x y, conj (inner y x) = inner x y
  /-- The inner product is additive in the first coordinate. -/
  add_left : ∀ x y z, inner (x + y) z = inner x z + inner y z
  /-- The inner product is conjugate linear in the first coordinate. -/
  smul_left : ∀ x y r, inner (r • x) y = conj r * inner x y


/-- A structure requiring that a scalar product is positive semidefinite and symmetric. -/
structure PreInnerProductSpace.Core (𝕜 : Type*) (F : Type*) [RCLike 𝕜] [AddCommGroup F]
  [Module 𝕜 F] extends Inner 𝕜 F where
  /-- The inner product is *hermitian*, taking the `conj` swaps the arguments. -/
  conj_symm : ∀ x y, conj (inner y x) = inner x y
  /-- The inner product is positive (semi)definite. -/
  nonneg_re : ∀ x, 0 ≤ re (inner x x)
  /-- The inner product is additive in the first coordinate. -/
  add_left : ∀ x y z, inner (x + y) z = inner x z + inner y z
  /-- The inner product is conjugate linear in the first coordinate. -/
  smul_left : ∀ x y r, inner (r • x) y = conj r * inner x y


/-- A structure requiring that a scalar product is positive definite. Some theorems that
require this assumptions are put under section `InnerProductSpace.Core`. -/
-- @[nolint HasNonemptyInstance] porting note: I don't think we have this linter anymore
structure InnerProductSpace.Core (𝕜 : Type*) (F : Type*) [RCLike 𝕜] [AddCommGroup F]
  [Module 𝕜 F] extends PreInnerProductSpace.Core 𝕜 F where
  /-- The inner product is positive definite. -/
  definite : ∀ x, inner x x = 0 → x = 0

/- We set `InnerProductSpace.Core` to be a class as we will use it as such in the construction
of the normed space structure that it produces. However, all the instances we will use will be
local to this proof. -/

instance (𝕜 : Type*) (F : Type*) [RCLike 𝕜] [AddCommGroup F]
  [Module 𝕜 F] [cd : InnerProductSpace.Core 𝕜 F] : PreInnerProductSpace.Core 𝕜 F where
  inner := cd.inner
  conj_symm := cd.conj_symm
  nonneg_re := cd.nonneg_re
  add_left := cd.add_left
  smul_left := cd.smul_left


/-- Define `PreInnerProductSpace.Core` from `PreInnerProductSpace`. Defined to reuse lemmas about
`PreInnerProductSpace.Core` for `PreInnerProductSpace`s. Note that the `Seminorm` instance provided
by `PreInnerProductSpace.Core.norm` is propositionally but not definitionally equal to the original
norm. -/
def PreInnerProductSpace.toCore [SeminormedAddCommGroup E] [c : InnerProductSpace 𝕜 E] :
    PreInnerProductSpace.Core 𝕜 E :=
  { c with
    nonneg_re := fun x => by
      /-
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝¹ : RCLike 𝕜
        inst✝ : SeminormedAddCommGroup E
        c : InnerProductSpace 𝕜 E
        x : E
        ⊢ LE.le 0 (RCLike.re (Inner.inner x x))
      -/
      rw [← InnerProductSpace.norm_sq_eq_inner]
      /-
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝¹ : RCLike 𝕜
        inst✝ : SeminormedAddCommGroup E
        c : InnerProductSpace 𝕜 E
        x : E
        ⊢ LE.le 0 (HPow.hPow (Norm.norm x) 2)
      -/
      apply sq_nonneg }
      /-
        🎉 no goals
      -/


/-- Define `InnerProductSpace.Core` from `InnerProductSpace`. Defined to reuse lemmas about
`InnerProductSpace.Core` for `InnerProductSpace`s. Note that the `Norm` instance provided by
`InnerProductSpace.Core.norm` is propositionally but not definitionally equal to the original
norm. -/
def InnerProductSpace.toCore [NormedAddCommGroup E] [c : InnerProductSpace 𝕜 E] :
    InnerProductSpace.Core 𝕜 E :=
  { c with
    nonneg_re := fun x => by
      /-
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝¹ : RCLike 𝕜
        inst✝ : NormedAddCommGroup E
        c : InnerProductSpace 𝕜 E
        x : E
        ⊢ LE.le 0 (RCLike.re (Inner.inner x x))
      -/
      rw [← InnerProductSpace.norm_sq_eq_inner]
      /-
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝¹ : RCLike 𝕜
        inst✝ : NormedAddCommGroup E
        c : InnerProductSpace 𝕜 E
        x : E
        ⊢ LE.le 0 (HPow.hPow (Norm.norm x) 2)
      -/
      apply sq_nonneg
      /-
        🎉 no goals
      -/
    definite := fun x hx =>
      norm_eq_zero.1 <| pow_eq_zero (n := 2) <| by
        /-
          𝕜 : Type u_1
          E : Type u_2
          F : Type u_3
          inst✝¹ : RCLike 𝕜
          inst✝ : NormedAddCommGroup E
          c : InnerProductSpace 𝕜 E
          x : E
          hx : Eq (Inner.inner x x) 0
          ⊢ Eq (HPow.hPow (Norm.norm x) 2) 0
        -/
        rw [InnerProductSpace.norm_sq_eq_inner (𝕜 := 𝕜) x, hx, map_zero] }
        /-
          🎉 no goals
        -/


local notation "⟪" x ", " y "⟫" => @inner 𝕜 F _ x y


local notation "normSqK" => @RCLike.normSq 𝕜 _


local notation "reK" => @RCLike.re 𝕜 _


local notation "ext_iff" => @RCLike.ext_iff 𝕜 _


local postfix:90 "†" => starRingEnd _


/-- Inner product defined by the `PreInnerProductSpace.Core` structure. We can't reuse
`PreInnerProductSpace.Core.toInner` because it takes `PreInnerProductSpace.Core` as an explicit
argument. -/
def toPreInner' : Inner 𝕜 F :=
  c.toInner


/-- The norm squared function for `PreInnerProductSpace.Core` structure. -/
def normSq (x : F) :=
  reK ⟪x, x⟫


local notation "normSqF" => @normSq 𝕜 F _ _ _ _


theorem inner_conj_symm (x y : F) : ⟪y, x⟫† = ⟪x, y⟫ :=
  c.conj_symm x y


theorem inner_self_nonneg {x : F} : 0 ≤ re ⟪x, x⟫ :=
  c.nonneg_re _


theorem inner_self_im (x : F) : im ⟪x, x⟫ = 0 := by
  /-
    𝕜 : Type u_1
    F : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : AddCommGroup F
    inst✝ : Module 𝕜 F
    c : PreInnerProductSpace.Core 𝕜 F
    x : F
    ⊢ Eq (RCLike.im (Inner.inner x x)) 0
  -/
  rw [← @ofReal_inj 𝕜, im_eq_conj_sub]
  /-
    𝕜 : Type u_1
    F : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : AddCommGroup F
    inst✝ : Module 𝕜 F
    c : PreInnerProductSpace.Core 𝕜 F
    x : F
    ⊢ Eq (HDiv.hDiv (HMul.hMul RCLike.I (HSub.hSub ((starRingEnd 𝕜) (Inner.inner x …
  -/
  simp [inner_conj_symm]
  /-
    🎉 no goals
  -/


theorem inner_add_left (x y z : F) : ⟪x + y, z⟫ = ⟪x, z⟫ + ⟪y, z⟫ :=
  c.add_left _ _ _


theorem inner_add_right (x y z : F) : ⟪x, y + z⟫ = ⟪x, y⟫ + ⟪x, z⟫ := by
  /-
    𝕜 : Type u_1
    F : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : AddCommGroup F
    inst✝ : Module 𝕜 F
    c : PreInnerProductSpace.Core 𝕜 F
    x y z : F
    ⊢ Eq (Inner.inner x (HAdd.hAdd y z)) (HAdd.hAdd (Inner.inner x y) (Inner.inner …
  -/
  rw [← inner_conj_symm, inner_add_left, RingHom.map_add]; simp only [inner_conj_symm]
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem ofReal_normSq_eq_inner_self (x : F) : (normSqF x : 𝕜) = ⟪x, x⟫ := by
  /-
    𝕜 : Type u_1
    F : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : AddCommGroup F
    inst✝ : Module 𝕜 F
    c : PreInnerProductSpace.Core 𝕜 F
    x : F
    ⊢ Eq (↑(InnerProductSpace.Core.normSq x)) (Inner.inner x x)
  -/
  rw [ext_iff]
  /-
    𝕜 : Type u_1
    F : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : AddCommGroup F
    inst✝ : Module 𝕜 F
    c : PreInnerProductSpace.Core 𝕜 F
    x : F
    ⊢ And (Eq (RCLike.re ↑(InnerProductSpace.Core.normSq x)) (RCLike.re (Inner.inn …
  -/
  exact ⟨by simp only [ofReal_re]; rfl, by simp only [inner_self_im, ofReal_im]⟩
  /-
    🎉 no goals
  -/


                                                              /-
                                                                𝕜 : Type u_1
                                                                F : Type u_3
                                                                inst✝² : RCLike 𝕜
                                                                inst✝¹ : AddCommGroup F
                                                                inst✝ : Module 𝕜 F
                                                                c : PreInnerProductSpace.Core 𝕜 F
                                                                x y : F
                                                                ⊢ Eq (RCLike.re (Inner.inner x y)) (RCLike.re (Inner.inner y x))
                                                              -/
theorem inner_re_symm (x y : F) : re ⟪x, y⟫ = re ⟪y, x⟫ := by rw [← inner_conj_symm, conj_re]
                                                              /-
                                                                🎉 no goals
                                                              -/


                                                               /-
                                                                 𝕜 : Type u_1
                                                                 F : Type u_3
                                                                 inst✝² : RCLike 𝕜
                                                                 inst✝¹ : AddCommGroup F
                                                                 inst✝ : Module 𝕜 F
                                                                 c : PreInnerProductSpace.Core 𝕜 F
                                                                 x y : F
                                                                 ⊢ Eq (RCLike.im (Inner.inner x y)) (Neg.neg (RCLike.im (Inner.inner y x)))
                                                               -/
theorem inner_im_symm (x y : F) : im ⟪x, y⟫ = -im ⟪y, x⟫ := by rw [← inner_conj_symm, conj_im]
                                                               /-
                                                                 🎉 no goals
                                                               -/


theorem inner_smul_left (x y : F) {r : 𝕜} : ⟪r • x, y⟫ = r† * ⟪x, y⟫ :=
  c.smul_left _ _ _


theorem inner_smul_right (x y : F) {r : 𝕜} : ⟪x, r • y⟫ = r * ⟪x, y⟫ := by
  /-
    𝕜 : Type u_1
    F : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : AddCommGroup F
    inst✝ : Module 𝕜 F
    c : PreInnerProductSpace.Core 𝕜 F
    x y : F
    r : 𝕜
    ⊢ Eq (Inner.inner x (HSMul.hSMul r y)) (HMul.hMul r (Inner.inner x y))
  -/
  rw [← inner_conj_symm, inner_smul_left]
  /-
    𝕜 : Type u_1
    F : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : AddCommGroup F
    inst✝ : Module 𝕜 F
    c : PreInnerProductSpace.Core 𝕜 F
    x y : F
    r : 𝕜
    ⊢ Eq ((starRingEnd 𝕜) (HMul.hMul ((starRingEnd 𝕜) r) (Inner.inner y x))) (HMul …
  -/
  simp only [conj_conj, inner_conj_symm, RingHom.map_mul]
  /-
    🎉 no goals
  -/


theorem inner_zero_left (x : F) : ⟪0, x⟫ = 0 := by
  /-
    𝕜 : Type u_1
    F : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : AddCommGroup F
    inst✝ : Module 𝕜 F
    c : PreInnerProductSpace.Core 𝕜 F
    x : F
    ⊢ Eq (Inner.inner 0 x) 0
  -/
  rw [← zero_smul 𝕜 (0 : F), inner_smul_left]
  /-
    𝕜 : Type u_1
    F : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : AddCommGroup F
    inst✝ : Module 𝕜 F
    c : PreInnerProductSpace.Core 𝕜 F
    x : F
    ⊢ Eq (HMul.hMul ((starRingEnd 𝕜) 0) (Inner.inner 0 x)) 0
  -/
  simp only [zero_mul, RingHom.map_zero]
  /-
    🎉 no goals
  -/


theorem inner_zero_right (x : F) : ⟪x, 0⟫ = 0 := by
  /-
    𝕜 : Type u_1
    F : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : AddCommGroup F
    inst✝ : Module 𝕜 F
    c : PreInnerProductSpace.Core 𝕜 F
    x : F
    ⊢ Eq (Inner.inner x 0) 0
  -/
  rw [← inner_conj_symm, inner_zero_left]; simp only [RingHom.map_zero]
                                           /-
                                             🎉 no goals
                                           -/


theorem inner_self_of_eq_zero {x : F} : x = 0 → ⟪x, x⟫ = 0 := by
  /-
    𝕜 : Type u_1
    F : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : AddCommGroup F
    inst✝ : Module 𝕜 F
    c : PreInnerProductSpace.Core 𝕜 F
    x : F
    ⊢ Eq x 0 → Eq (Inner.inner x x) 0
  -/
  rintro rfl
  /-
    𝕜 : Type u_1
    F : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : AddCommGroup F
    inst✝ : Module 𝕜 F
    c : PreInnerProductSpace.Core 𝕜 F
    ⊢ Eq (Inner.inner 0 0) 0
  -/
  exact inner_zero_left _
  /-
    🎉 no goals
  -/


theorem normSq_eq_zero_of_eq_zero {x : F} : x = 0 → normSqF x = 0 := by
  /-
    𝕜 : Type u_1
    F : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : AddCommGroup F
    inst✝ : Module 𝕜 F
    c : PreInnerProductSpace.Core 𝕜 F
    x : F
    ⊢ Eq x 0 → Eq (InnerProductSpace.Core.normSq x) 0
  -/
  rintro rfl
  /-
    𝕜 : Type u_1
    F : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : AddCommGroup F
    inst✝ : Module 𝕜 F
    c : PreInnerProductSpace.Core 𝕜 F
    ⊢ Eq (InnerProductSpace.Core.normSq 0) 0
  -/
  simp [normSq, inner_self_of_eq_zero]
  /-
    🎉 no goals
  -/


theorem ne_zero_of_inner_self_ne_zero {x : F} : ⟪x, x⟫ ≠ 0 → x ≠ 0 :=
  mt inner_self_of_eq_zero

theorem inner_self_ofReal_re (x : F) : (re ⟪x, x⟫ : 𝕜) = ⟪x, x⟫ := by
  /-
    𝕜 : Type u_1
    F : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : AddCommGroup F
    inst✝ : Module 𝕜 F
    c : PreInnerProductSpace.Core 𝕜 F
    x : F
    ⊢ Eq (↑(RCLike.re (Inner.inner x x))) (Inner.inner x x)
  -/
  norm_num [ext_iff, inner_self_im]
  /-
    🎉 no goals
  -/


                                                              /-
                                                                𝕜 : Type u_1
                                                                F : Type u_3
                                                                inst✝² : RCLike 𝕜
                                                                inst✝¹ : AddCommGroup F
                                                                inst✝ : Module 𝕜 F
                                                                c : PreInnerProductSpace.Core 𝕜 F
                                                                x y : F
                                                                ⊢ Eq (Norm.norm (Inner.inner x y)) (Norm.norm (Inner.inner y x))
                                                              -/
theorem norm_inner_symm (x y : F) : ‖⟪x, y⟫‖ = ‖⟪y, x⟫‖ := by rw [← inner_conj_symm, norm_conj]
                                                              /-
                                                                🎉 no goals
                                                              -/


theorem inner_neg_left (x y : F) : ⟪-x, y⟫ = -⟪x, y⟫ := by
  /-
    𝕜 : Type u_1
    F : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : AddCommGroup F
    inst✝ : Module 𝕜 F
    c : PreInnerProductSpace.Core 𝕜 F
    x y : F
    ⊢ Eq (Inner.inner (Neg.neg x) y) (Neg.neg (Inner.inner x y))
  -/
  rw [← neg_one_smul 𝕜 x, inner_smul_left]
  /-
    𝕜 : Type u_1
    F : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : AddCommGroup F
    inst✝ : Module 𝕜 F
    c : PreInnerProductSpace.Core 𝕜 F
    x y : F
    ⊢ Eq (HMul.hMul ((starRingEnd 𝕜) (-1)) (Inner.inner x y)) (Neg.neg (Inner.inne …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem inner_neg_right (x y : F) : ⟪x, -y⟫ = -⟪x, y⟫ := by
  /-
    𝕜 : Type u_1
    F : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : AddCommGroup F
    inst✝ : Module 𝕜 F
    c : PreInnerProductSpace.Core 𝕜 F
    x y : F
    ⊢ Eq (Inner.inner x (Neg.neg y)) (Neg.neg (Inner.inner x y))
  -/
  rw [← inner_conj_symm, inner_neg_left]; simp only [RingHom.map_neg, inner_conj_symm]
                                          /-
                                            🎉 no goals
                                          -/


theorem inner_sub_left (x y z : F) : ⟪x - y, z⟫ = ⟪x, z⟫ - ⟪y, z⟫ := by
  /-
    𝕜 : Type u_1
    F : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : AddCommGroup F
    inst✝ : Module 𝕜 F
    c : PreInnerProductSpace.Core 𝕜 F
    x y z : F
    ⊢ Eq (Inner.inner (HSub.hSub x y) z) (HSub.hSub (Inner.inner x z) (Inner.inner …
  -/
  simp [sub_eq_add_neg, inner_add_left, inner_neg_left]
  /-
    🎉 no goals
  -/


theorem inner_sub_right (x y z : F) : ⟪x, y - z⟫ = ⟪x, y⟫ - ⟪x, z⟫ := by
  /-
    𝕜 : Type u_1
    F : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : AddCommGroup F
    inst✝ : Module 𝕜 F
    c : PreInnerProductSpace.Core 𝕜 F
    x y z : F
    ⊢ Eq (Inner.inner x (HSub.hSub y z)) (HSub.hSub (Inner.inner x y) (Inner.inner …
  -/
  simp [sub_eq_add_neg, inner_add_right, inner_neg_right]
  /-
    🎉 no goals
  -/


theorem inner_mul_symm_re_eq_norm (x y : F) : re (⟪x, y⟫ * ⟪y, x⟫) = ‖⟪x, y⟫ * ⟪y, x⟫‖ := by
  /-
    𝕜 : Type u_1
    F : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : AddCommGroup F
    inst✝ : Module 𝕜 F
    c : PreInnerProductSpace.Core 𝕜 F
    x y : F
    ⊢ Eq (RCLike.re (HMul.hMul (Inner.inner x y) (Inner.inner y x))) (Norm.norm (H …
  -/
  rw [← inner_conj_symm, mul_comm]
  /-
    𝕜 : Type u_1
    F : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : AddCommGroup F
    inst✝ : Module 𝕜 F
    c : PreInnerProductSpace.Core 𝕜 F
    x y : F
    ⊢ Eq (RCLike.re (HMul.hMul (Inner.inner y x) ((starRingEnd 𝕜) (Inner.inner y x …
  -/
  exact re_eq_norm_of_mul_conj (inner y x)
  /-
    🎉 no goals
  -/


/-- Expand `inner (x + y) (x + y)` -/
theorem inner_add_add_self (x y : F) : ⟪x + y, x + y⟫ = ⟪x, x⟫ + ⟪x, y⟫ + ⟪y, x⟫ + ⟪y, y⟫ := by
  /-
    𝕜 : Type u_1
    F : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : AddCommGroup F
    inst✝ : Module 𝕜 F
    c : PreInnerProductSpace.Core 𝕜 F
    x y : F
    ⊢ Eq (Inner.inner (HAdd.hAdd x y) (HAdd.hAdd x y)) (HAdd.hAdd (HAdd.hAdd (HAdd …
  -/
  simp only [inner_add_left, inner_add_right]; ring
                                               /-
                                                 🎉 no goals
                                               -/

-- Expand `inner (x - y) (x - y)`

theorem inner_sub_sub_self (x y : F) : ⟪x - y, x - y⟫ = ⟪x, x⟫ - ⟪x, y⟫ - ⟪y, x⟫ + ⟪y, y⟫ := by
  /-
    𝕜 : Type u_1
    F : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : AddCommGroup F
    inst✝ : Module 𝕜 F
    c : PreInnerProductSpace.Core 𝕜 F
    x y : F
    ⊢ Eq (Inner.inner (HSub.hSub x y) (HSub.hSub x y)) (HAdd.hAdd (HSub.hSub (HSub …
  -/
  simp only [inner_sub_left, inner_sub_right]; ring
                                               /-
                                                 🎉 no goals
                                               -/


theorem inner_smul_ofReal_left (x y : F) {t : ℝ} : ⟪(t : 𝕜) • x, y⟫ = ⟪x, y⟫ * t := by
  /-
    𝕜 : Type u_1
    F : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : AddCommGroup F
    inst✝ : Module 𝕜 F
    c : PreInnerProductSpace.Core 𝕜 F
    x y : F
    t : Real
    ⊢ Eq (Inner.inner (HSMul.hSMul (↑t) x) y) (HMul.hMul (Inner.inner x y) ↑t)
  -/
  rw [inner_smul_left, conj_ofReal, mul_comm]
  /-
    🎉 no goals
  -/


theorem inner_smul_ofReal_right (x y : F) {t : ℝ} : ⟪x, (t : 𝕜) • y⟫ = ⟪x, y⟫ * t := by
  /-
    𝕜 : Type u_1
    F : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : AddCommGroup F
    inst✝ : Module 𝕜 F
    c : PreInnerProductSpace.Core 𝕜 F
    x y : F
    t : Real
    ⊢ Eq (Inner.inner x (HSMul.hSMul (↑t) y)) (HMul.hMul (Inner.inner x y) ↑t)
  -/
  rw [inner_smul_right, mul_comm]
  /-
    🎉 no goals
  -/


theorem re_inner_smul_ofReal_smul_self (x : F) {t : ℝ} :
    re ⟪(t : 𝕜) • x, (t : 𝕜) • x⟫ = normSqF x * t * t := by
  /-
    𝕜 : Type u_1
    F : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : AddCommGroup F
    inst✝ : Module 𝕜 F
    c : PreInnerProductSpace.Core 𝕜 F
    x : F
    t : Real
    ⊢ Eq (RCLike.re (Inner.inner (HSMul.hSMul (↑t) x) (HSMul.hSMul (↑t) x))) (HMul …
  -/
  apply ofReal_injective (K := 𝕜)
  /-
    case a
    𝕜 : Type u_1
    F : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : AddCommGroup F
    inst✝ : Module 𝕜 F
    c : PreInnerProductSpace.Core 𝕜 F
    x : F
    t : Real
    ⊢ Eq ↑(RCLike.re (Inner.inner (HSMul.hSMul (↑t) x) (HSMul.hSMul (↑t) x))) ↑(HM …
  -/
  simp [inner_self_ofReal_re, inner_smul_ofReal_left, inner_smul_ofReal_right, normSq]
  /-
    🎉 no goals
  -/


/-- An auxiliary equality useful to prove the **Cauchy–Schwarz inequality**. Here we use the
standard argument involving the discriminant of quadratic form. -/
lemma cauchy_schwarz_aux' (x y : F) (t : ℝ) : 0 ≤ normSqF x * t * t + 2 * re ⟪x, y⟫ * t
    + normSqF y := by
  calc 0 ≤ re ⟪(ofReal t : 𝕜) • x + y, (ofReal t : 𝕜) • x + y⟫ := inner_self_nonneg
  _ = re (⟪(ofReal t : 𝕜) • x, (ofReal t : 𝕜) • x⟫ + ⟪(ofReal t : 𝕜) • x, y⟫
      + ⟪y, (ofReal t : 𝕜) • x⟫ + ⟪y, y⟫) := by rw [inner_add_add_self ((ofReal t : 𝕜) • x) y]
  _ = re ⟪(ofReal t : 𝕜) • x, (ofReal t : 𝕜) • x⟫
      + re ⟪(ofReal t : 𝕜) • x, y⟫ + re ⟪y, (ofReal t : 𝕜) • x⟫ + re ⟪y, y⟫ := by
      simp only [map_add]
  _ = normSq x * t * t + re (⟪x, y⟫ * t) + re (⟪y, x⟫ * t) + re ⟪y, y⟫ := by rw
    [re_inner_smul_ofReal_smul_self, inner_smul_ofReal_left, inner_smul_ofReal_right]
  _ = normSq x * t * t + re ⟪x, y⟫ * t + re ⟪y, x⟫ * t + re ⟪y, y⟫ := by rw [mul_comm ⟪x,y⟫ _,
    RCLike.re_ofReal_mul, mul_comm t _, mul_comm ⟪y,x⟫ _, RCLike.re_ofReal_mul, mul_comm t _]
  _ = normSq x * t * t + re ⟪x, y⟫ * t + re ⟪y, x⟫ * t + normSq y := by rw [← normSq]
  _ = normSq x * t * t + re ⟪x, y⟫ * t + re ⟪x, y⟫ * t + normSq y := by rw [inner_re_symm]
  _ = normSq x * t * t + 2 * re ⟪x, y⟫ * t + normSq y := by ring


/-- Another auxiliary equality related with the **Cauchy–Schwarz inequality**: the square of the
seminorm of `⟪x, y⟫ • x - ⟪x, x⟫ • y` is equal to `‖x‖ ^ 2 * (‖x‖ ^ 2 * ‖y‖ ^ 2 - ‖⟪x, y⟫‖ ^ 2)`.
We use `InnerProductSpace.ofCore.normSq x` etc (defeq to `is_R_or_C.re ⟪x, x⟫`) instead of `‖x‖ ^ 2`
etc to avoid extra rewrites when applying it to an `InnerProductSpace`. -/
theorem cauchy_schwarz_aux (x y : F) : normSqF (⟪x, y⟫ • x - ⟪x, x⟫ • y)
    = normSqF x * (normSqF x * normSqF y - ‖⟪x, y⟫‖ ^ 2) := by
  /-
    𝕜 : Type u_1
    F : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : AddCommGroup F
    inst✝ : Module 𝕜 F
    c : PreInnerProductSpace.Core 𝕜 F
    x y : F
    ⊢ Eq (InnerProductSpace.Core.normSq (HSub.hSub (HSMul.hSMul (Inner.inner x y)  …
  -/
  rw [← @ofReal_inj 𝕜, ofReal_normSq_eq_inner_self]
  simp only [inner_sub_sub_self, inner_smul_left, inner_smul_right, conj_ofReal, mul_sub, ←
    ofReal_normSq_eq_inner_self x, ← ofReal_normSq_eq_inner_self y]
  /-
    𝕜 : Type u_1
    F : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : AddCommGroup F
    inst✝ : Module 𝕜 F
    c : PreInnerProductSpace.Core 𝕜 F
    x y : F
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HSub.hSub (HMul.hMul (Inner.inner x y) (HMul.hMul  …
  -/
  rw [← mul_assoc, mul_conj, RCLike.conj_mul, mul_left_comm, ← inner_conj_symm y, mul_conj]
  /-
    𝕜 : Type u_1
    F : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : AddCommGroup F
    inst✝ : Module 𝕜 F
    c : PreInnerProductSpace.Core 𝕜 F
    x y : F
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HSub.hSub (HMul.hMul (HPow.hPow (↑(Norm.norm (Inne …
  -/
  push_cast
  /-
    𝕜 : Type u_1
    F : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : AddCommGroup F
    inst✝ : Module 𝕜 F
    c : PreInnerProductSpace.Core 𝕜 F
    x y : F
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HSub.hSub (HMul.hMul (HPow.hPow (↑(Norm.norm (Inne …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- **Cauchy–Schwarz inequality**.
We need this for the `PreInnerProductSpace.Core` structure to prove the triangle inequality below
when showing the core is a normed group and to take the quotient.

(This is not intended for general use; see `Analysis.InnerProductSpace.Basic` for a variety of
versions of Cauchy-Schwartz for an inner product space, rather than a `PreInnerProductSpace.Core`).
-/
theorem inner_mul_inner_self_le (x y : F) : ‖⟪x, y⟫‖ * ‖⟪y, x⟫‖ ≤ re ⟪x, x⟫ * re ⟪y, y⟫ := by
  suffices discrim (normSqF x) (2 * ‖⟪x, y⟫_𝕜‖) (normSqF y) ≤ 0 by
    rw [norm_inner_symm y x]
    rw [discrim, normSq, normSq, sq] at this
    linarith
  /-
    𝕜 : Type u_1
    F : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : AddCommGroup F
    inst✝ : Module 𝕜 F
    c : PreInnerProductSpace.Core 𝕜 F
    x y : F
    ⊢ LE.le (discrim (InnerProductSpace.Core.normSq x) (HMul.hMul 2 (Norm.norm (In …
  -/
  refine discrim_le_zero fun t ↦ ?_
  /-
    𝕜 : Type u_1
    F : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : AddCommGroup F
    inst✝ : Module 𝕜 F
    c : PreInnerProductSpace.Core 𝕜 F
    x y : F
    t : Real
    ⊢ LE.le 0 (HAdd.hAdd (HAdd.hAdd (HMul.hMul (InnerProductSpace.Core.normSq x) ( …
  -/
  by_cases hzero : ⟪x, y⟫ = 0
    /-
      case pos
      𝕜 : Type u_1
      F : Type u_3
      inst✝² : RCLike 𝕜
      inst✝¹ : AddCommGroup F
      inst✝ : Module 𝕜 F
      c : PreInnerProductSpace.Core 𝕜 F
      x y : F
      t : Real
      hzero : Eq (Inner.inner x y) 0
      ⊢ LE.le 0 (HAdd.hAdd (HAdd.hAdd (HMul.hMul (InnerProductSpace.Core.normSq x) ( …
    -/
  · simp only [mul_assoc, ← sq, hzero, norm_zero, mul_zero, zero_mul, add_zero, ge_iff_le]
    /-
      case pos
      𝕜 : Type u_1
      F : Type u_3
      inst✝² : RCLike 𝕜
      inst✝¹ : AddCommGroup F
      inst✝ : Module 𝕜 F
      c : PreInnerProductSpace.Core 𝕜 F
      x y : F
      t : Real
      hzero : Eq (Inner.inner x y) 0
      ⊢ LE.le 0 (HAdd.hAdd (HMul.hMul (InnerProductSpace.Core.normSq x) (HPow.hPow t …
    -/
    obtain ⟨hx, hy⟩ : (0 ≤ normSqF x ∧ 0 ≤ normSqF y) := ⟨inner_self_nonneg, inner_self_nonneg⟩
    /-
      case pos.intro
      𝕜 : Type u_1
      F : Type u_3
      inst✝² : RCLike 𝕜
      inst✝¹ : AddCommGroup F
      inst✝ : Module 𝕜 F
      c : PreInnerProductSpace.Core 𝕜 F
      x y : F
      t : Real
      hzero : Eq (Inner.inner x y) 0
      hx : LE.le 0 (InnerProductSpace.Core.normSq x)
      hy : LE.le 0 (InnerProductSpace.Core.normSq y)
      ⊢ LE.le 0 (HAdd.hAdd (HMul.hMul (InnerProductSpace.Core.normSq x) (HPow.hPow t …
    -/
    positivity
    /-
      🎉 no goals
    -/
    /-
      case neg
      𝕜 : Type u_1
      F : Type u_3
      inst✝² : RCLike 𝕜
      inst✝¹ : AddCommGroup F
      inst✝ : Module 𝕜 F
      c : PreInnerProductSpace.Core 𝕜 F
      x y : F
      t : Real
      hzero : Not (Eq (Inner.inner x y) 0)
      ⊢ LE.le 0 (HAdd.hAdd (HAdd.hAdd (HMul.hMul (InnerProductSpace.Core.normSq x) ( …
    -/
  · have hzero' : ‖⟪x, y⟫‖ ≠ 0 := norm_ne_zero_iff.2 hzero
    /-
      case neg
      𝕜 : Type u_1
      F : Type u_3
      inst✝² : RCLike 𝕜
      inst✝¹ : AddCommGroup F
      inst✝ : Module 𝕜 F
      c : PreInnerProductSpace.Core 𝕜 F
      x y : F
      t : Real
      hzero : Not (Eq (Inner.inner x y) 0)
      hzero' : Ne (Norm.norm (Inner.inner x y)) 0
      ⊢ LE.le 0 (HAdd.hAdd (HAdd.hAdd (HMul.hMul (InnerProductSpace.Core.normSq x) ( …
    -/
    convert cauchy_schwarz_aux' (𝕜 := 𝕜) (⟪x, y⟫ • x) y (t / ‖⟪x, y⟫‖) using 3
      /-
        case h.e'_4.h.e'_5.h.e'_5
        𝕜 : Type u_1
        F : Type u_3
        inst✝² : RCLike 𝕜
        inst✝¹ : AddCommGroup F
        inst✝ : Module 𝕜 F
        c : PreInnerProductSpace.Core 𝕜 F
        x y : F
        t : Real
        hzero : Not (Eq (Inner.inner x y) 0)
        hzero' : Ne (Norm.norm (Inner.inner x y)) 0
        ⊢ Eq (HMul.hMul (InnerProductSpace.Core.normSq x) (HMul.hMul t t)) (HMul.hMul  …
      -/
    · field_simp
      rw [← sq, normSq, normSq, inner_smul_right, inner_smul_left, ← mul_assoc _ _ ⟪x, x⟫,
        mul_conj]
      /-
        case h.e'_4.h.e'_5.h.e'_5
        𝕜 : Type u_1
        F : Type u_3
        inst✝² : RCLike 𝕜
        inst✝¹ : AddCommGroup F
        inst✝ : Module 𝕜 F
        c : PreInnerProductSpace.Core 𝕜 F
        x y : F
        t : Real
        hzero : Not (Eq (Inner.inner x y) 0)
        hzero' : Ne (Norm.norm (Inner.inner x y)) 0
        ⊢ Eq (HMul.hMul (HMul.hMul (RCLike.re (Inner.inner x x)) (HPow.hPow t 2)) (HMu …
      -/
      nth_rw 2 [sq]
      /-
        case h.e'_4.h.e'_5.h.e'_5
        𝕜 : Type u_1
        F : Type u_3
        inst✝² : RCLike 𝕜
        inst✝¹ : AddCommGroup F
        inst✝ : Module 𝕜 F
        c : PreInnerProductSpace.Core 𝕜 F
        x y : F
        t : Real
        hzero : Not (Eq (Inner.inner x y) 0)
        hzero' : Ne (Norm.norm (Inner.inner x y)) 0
        ⊢ Eq (HMul.hMul (HMul.hMul (RCLike.re (Inner.inner x x)) (HPow.hPow t 2)) (HMu …
      -/
      rw [← ofReal_mul, re_ofReal_mul]
      /-
        case h.e'_4.h.e'_5.h.e'_5
        𝕜 : Type u_1
        F : Type u_3
        inst✝² : RCLike 𝕜
        inst✝¹ : AddCommGroup F
        inst✝ : Module 𝕜 F
        c : PreInnerProductSpace.Core 𝕜 F
        x y : F
        t : Real
        hzero : Not (Eq (Inner.inner x y) 0)
        hzero' : Ne (Norm.norm (Inner.inner x y)) 0
        ⊢ Eq (HMul.hMul (HMul.hMul (RCLike.re (Inner.inner x x)) (HPow.hPow t 2)) (HMu …
      -/
      ring
      /-
        🎉 no goals
      -/
      /-
        case h.e'_4.h.e'_5.h.e'_6
        𝕜 : Type u_1
        F : Type u_3
        inst✝² : RCLike 𝕜
        inst✝¹ : AddCommGroup F
        inst✝ : Module 𝕜 F
        c : PreInnerProductSpace.Core 𝕜 F
        x y : F
        t : Real
        hzero : Not (Eq (Inner.inner x y) 0)
        hzero' : Ne (Norm.norm (Inner.inner x y)) 0
        ⊢ Eq (HMul.hMul (HMul.hMul 2 (Norm.norm (Inner.inner x y))) t) (HMul.hMul (HMu …
      -/
    · field_simp
      /-
        case h.e'_4.h.e'_5.h.e'_6
        𝕜 : Type u_1
        F : Type u_3
        inst✝² : RCLike 𝕜
        inst✝¹ : AddCommGroup F
        inst✝ : Module 𝕜 F
        c : PreInnerProductSpace.Core 𝕜 F
        x y : F
        t : Real
        hzero : Not (Eq (Inner.inner x y) 0)
        hzero' : Ne (Norm.norm (Inner.inner x y)) 0
        ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul 2 (Norm.norm (Inner.inner x y))) t) (Nor …
      -/
      rw [inner_smul_left, mul_comm _ ⟪x, y⟫_𝕜, mul_conj, ← ofReal_pow, ofReal_re]
      /-
        case h.e'_4.h.e'_5.h.e'_6
        𝕜 : Type u_1
        F : Type u_3
        inst✝² : RCLike 𝕜
        inst✝¹ : AddCommGroup F
        inst✝ : Module 𝕜 F
        c : PreInnerProductSpace.Core 𝕜 F
        x y : F
        t : Real
        hzero : Not (Eq (Inner.inner x y) 0)
        hzero' : Ne (Norm.norm (Inner.inner x y)) 0
        ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul 2 (Norm.norm (Inner.inner x y))) t) (Nor …
      -/
      ring
      /-
        🎉 no goals
      -/


/-- (Semi)norm constructed from an `PreInnerProductSpace.Core` structure, defined to be the square
root of the scalar product. -/
def toNorm : Norm F where norm x := √(re ⟪x, x⟫)


theorem norm_eq_sqrt_inner (x : F) : ‖x‖ = √(re ⟪x, x⟫) := rfl


theorem inner_self_eq_norm_mul_norm (x : F) : re ⟪x, x⟫ = ‖x‖ * ‖x‖ := by
  /-
    𝕜 : Type u_1
    F : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : AddCommGroup F
    inst✝ : Module 𝕜 F
    c : PreInnerProductSpace.Core 𝕜 F
    x : F
    ⊢ Eq (RCLike.re (Inner.inner x x)) (HMul.hMul (Norm.norm x) (Norm.norm x))
  -/
  rw [norm_eq_sqrt_inner, ← sqrt_mul inner_self_nonneg (re ⟪x, x⟫), sqrt_mul_self inner_self_nonneg]
  /-
    🎉 no goals
  -/


theorem sqrt_normSq_eq_norm (x : F) : √(normSqF x) = ‖x‖ := rfl


/-- Cauchy–Schwarz inequality with norm -/
theorem norm_inner_le_norm (x y : F) : ‖⟪x, y⟫‖ ≤ ‖x‖ * ‖y‖ :=
  nonneg_le_nonneg_of_sq_le_sq (mul_nonneg (sqrt_nonneg _) (sqrt_nonneg _)) <|
    calc
                                                      /-
                                                        𝕜 : Type u_1
                                                        F : Type u_3
                                                        inst✝² : RCLike 𝕜
                                                        inst✝¹ : AddCommGroup F
                                                        inst✝ : Module 𝕜 F
                                                        c : PreInnerProductSpace.Core 𝕜 F
                                                        x y : F
                                                        ⊢ Eq (HMul.hMul (Norm.norm (Inner.inner x y)) (Norm.norm (Inner.inner x y))) ( …
                                                      -/
      ‖⟪x, y⟫‖ * ‖⟪x, y⟫‖ = ‖⟪x, y⟫‖ * ‖⟪y, x⟫‖ := by rw [norm_inner_symm]
                                                      /-
                                                        🎉 no goals
                                                      -/
      _ ≤ re ⟪x, x⟫ * re ⟪y, y⟫ := inner_mul_inner_self_le x y
                                        /-
                                          𝕜 : Type u_1
                                          F : Type u_3
                                          inst✝² : RCLike 𝕜
                                          inst✝¹ : AddCommGroup F
                                          inst✝ : Module 𝕜 F
                                          c : PreInnerProductSpace.Core 𝕜 F
                                          x y : F
                                          ⊢ Eq (HMul.hMul (RCLike.re (Inner.inner x x)) (RCLike.re (Inner.inner y y))) ( …
                                        -/
      _ = ‖x‖ * ‖y‖ * (‖x‖ * ‖y‖) := by simp only [inner_self_eq_norm_mul_norm]; ring
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


/-- Seminormed group structure constructed from an `PreInnerProductSpace.Core` structure -/
def toSeminormedAddCommGroup : SeminormedAddCommGroup F :=
  AddGroupSeminorm.toSeminormedAddCommGroup
    { toFun := fun x => √(re ⟪x, x⟫)
                      /-
                        𝕜 : Type u_1
                        E : Type u_2
                        F : Type u_3
                        inst✝² : RCLike 𝕜
                        inst✝¹ : AddCommGroup F
                        inst✝ : Module 𝕜 F
                        c : PreInnerProductSpace.Core 𝕜 F
                        ⊢ Eq ((fun x => (RCLike.re (Inner.inner x x)).sqrt) 0) 0
                      -/
      map_zero' := by simp only [sqrt_zero, inner_zero_right, map_zero]
                      /-
                        🎉 no goals
                      -/
                          /-
                            𝕜 : Type u_1
                            E : Type u_2
                            F : Type u_3
                            inst✝² : RCLike 𝕜
                            inst✝¹ : AddCommGroup F
                            inst✝ : Module 𝕜 F
                            c : PreInnerProductSpace.Core 𝕜 F
                            x : F
                            ⊢ Eq ((fun x => (RCLike.re (Inner.inner x x)).sqrt) (Neg.neg x)) ((fun x => (R …
                          -/
      neg' := fun x => by simp only [inner_neg_left, neg_neg, inner_neg_right]
        /-
          𝕜 : Type u_1
          E : Type u_2
          F : Type u_3
          inst✝² : RCLike 𝕜
          inst✝¹ : AddCommGroup F
          inst✝ : Module 𝕜 F
          c : PreInnerProductSpace.Core 𝕜 F
          x y : F
          ⊢ LE.le ((fun x => (RCLike.re (Inner.inner x x)).sqrt) (HAdd.hAdd x y)) (HAdd. …
        -/
                          /-
                            🎉 no goals
                          -/
        /-
          𝕜 : Type u_1
          E : Type u_2
          F : Type u_3
          inst✝² : RCLike 𝕜
          inst✝¹ : AddCommGroup F
          inst✝ : Module 𝕜 F
          c : PreInnerProductSpace.Core 𝕜 F
          x y : F
          h₁ : LE.le (Norm.norm (Inner.inner x y)) (HMul.hMul (Norm.norm x) (Norm.norm y))
          ⊢ LE.le ((fun x => (RCLike.re (Inner.inner x x)).sqrt) (HAdd.hAdd x y)) (HAdd. …
        -/
      add_le' := fun x y => by
        /-
          𝕜 : Type u_1
          E : Type u_2
          F : Type u_3
          inst✝² : RCLike 𝕜
          inst✝¹ : AddCommGroup F
          inst✝ : Module 𝕜 F
          c : PreInnerProductSpace.Core 𝕜 F
          x y : F
          h₁ : LE.le (Norm.norm (Inner.inner x y)) (HMul.hMul (Norm.norm x) (Norm.norm y))
          h₂ : LE.le (RCLike.re (Inner.inner x y)) (Norm.norm (Inner.inner x y))
          ⊢ LE.le ((fun x => (RCLike.re (Inner.inner x x)).sqrt) (HAdd.hAdd x y)) (HAdd. …
        -/
        have h₁ : ‖⟪x, y⟫‖ ≤ ‖x‖ * ‖y‖ := norm_inner_le_norm _ _
        /-
          𝕜 : Type u_1
          E : Type u_2
          F : Type u_3
          inst✝² : RCLike 𝕜
          inst✝¹ : AddCommGroup F
          inst✝ : Module 𝕜 F
          c : PreInnerProductSpace.Core 𝕜 F
          x y : F
          h₁ : LE.le (Norm.norm (Inner.inner x y)) (HMul.hMul (Norm.norm x) (Norm.norm y))
          h₂ : LE.le (RCLike.re (Inner.inner x y)) (Norm.norm (Inner.inner x y))
          h₃ : LE.le (RCLike.re (Inner.inner x y)) (HMul.hMul (Norm.norm x) (Norm.norm y))
          ⊢ LE.le ((fun x => (RCLike.re (Inner.inner x x)).sqrt) (HAdd.hAdd x y)) (HAdd. …
        -/
        have h₂ : re ⟪x, y⟫ ≤ ‖⟪x, y⟫‖ := re_le_norm _
        have h₃ : re ⟪x, y⟫ ≤ ‖x‖ * ‖y‖ := h₂.trans h₁
        have h₄ : re ⟪y, x⟫ ≤ ‖x‖ * ‖y‖ := by rwa [← inner_conj_symm, conj_re]
        have : ‖x + y‖ * ‖x + y‖ ≤ (‖x‖ + ‖y‖) * (‖x‖ + ‖y‖) := by
        /-
          𝕜 : Type u_1
          E : Type u_2
          F : Type u_3
          inst✝² : RCLike 𝕜
          inst✝¹ : AddCommGroup F
          inst✝ : Module 𝕜 F
          c : PreInnerProductSpace.Core 𝕜 F
          x y : F
          h₁ : LE.le (Norm.norm (Inner.inner x y)) (HMul.hMul (Norm.norm x) (Norm.norm y))
          h₂ : LE.le (RCLike.re (Inner.inner x y)) (Norm.norm (Inner.inner x y))
          h₃ : LE.le (RCLike.re (Inner.inner x y)) (HMul.hMul (Norm.norm x) (Norm.norm y))
          h₄ : LE.le (RCLike.re (Inner.inner y x)) (HMul.hMul (Norm.norm x) (Norm.norm y))
          this : LE.le (HMul.hMul (Norm.norm (HAdd.hAdd x y)) (Norm.norm (HAdd.hAdd x y) …
          ⊢ LE.le ((fun x => (RCLike.re (Inner.inner x x)).sqrt) (HAdd.hAdd x y)) (HAdd. …
        -/
          simp only [← inner_self_eq_norm_mul_norm, inner_add_add_self, mul_add, mul_comm, map_add]
        /-
          🎉 no goals
        -/
          linarith
        exact nonneg_le_nonneg_of_sq_le_sq (add_nonneg (sqrt_nonneg _) (sqrt_nonneg _)) this }


/-- Normed space (which is actually a seminorm) structure constructed from an
`PreInnerProductSpace.Core` structure -/
def toSeminormedSpace : NormedSpace 𝕜 F where
  norm_smul_le r x := by
    /-
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝² : RCLike 𝕜
      inst✝¹ : AddCommGroup F
      inst✝ : Module 𝕜 F
      c : PreInnerProductSpace.Core 𝕜 F
      r : 𝕜
      x : F
      ⊢ LE.le (Norm.norm (HSMul.hSMul r x)) (HMul.hMul (Norm.norm r) (Norm.norm x))
    -/
    rw [norm_eq_sqrt_inner, inner_smul_left, inner_smul_right, ← mul_assoc]
    rw [RCLike.conj_mul, ← ofReal_pow, re_ofReal_mul, sqrt_mul, ← ofReal_normSq_eq_inner_self,
      ofReal_re]
      /-
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝² : RCLike 𝕜
        inst✝¹ : AddCommGroup F
        inst✝ : Module 𝕜 F
        c : PreInnerProductSpace.Core 𝕜 F
        r : 𝕜
        x : F
        ⊢ LE.le (HMul.hMul (HPow.hPow (Norm.norm r) 2).sqrt (InnerProductSpace.Core.no …
      -/
    · simp [sqrt_normSq_eq_norm, RCLike.sqrt_normSq_eq_norm]
      /-
        🎉 no goals
      -/
      /-
        case hx
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝² : RCLike 𝕜
        inst✝¹ : AddCommGroup F
        inst✝ : Module 𝕜 F
        c : PreInnerProductSpace.Core 𝕜 F
        r : 𝕜
        x : F
        ⊢ LE.le 0 (HPow.hPow (Norm.norm r) 2)
      -/
    · positivity
      /-
        🎉 no goals
      -/


/-- Inner product defined by the `InnerProductSpace.Core` structure. We can't reuse
`InnerProductSpace.Core.toInner` because it takes `InnerProductSpace.Core` as an explicit
argument. -/
def toInner' : Inner 𝕜 F :=
  cd.toInner


theorem inner_self_eq_zero {x : F} : ⟪x, x⟫ = 0 ↔ x = 0 :=
  ⟨cd.definite _, inner_self_of_eq_zero⟩


theorem normSq_eq_zero {x : F} : normSqF x = 0 ↔ x = 0 :=
  Iff.trans
        /-
          𝕜 : Type u_1
          F : Type u_3
          inst✝² : RCLike 𝕜
          inst✝¹ : AddCommGroup F
          inst✝ : Module 𝕜 F
          cd : InnerProductSpace.Core 𝕜 F
          x : F
          ⊢ Iff (Eq (InnerProductSpace.Core.normSq x) 0) (Eq (Inner.inner x x) 0)
        -/
    (by simp only [normSq, ext_iff, map_zero, inner_self_im, eq_self_iff_true, and_true])
        /-
          🎉 no goals
        -/
    (@inner_self_eq_zero 𝕜 _ _ _ _ _ x)


theorem inner_self_ne_zero {x : F} : ⟪x, x⟫ ≠ 0 ↔ x ≠ 0 :=
  inner_self_eq_zero.not


/-- Normed group structure constructed from an `InnerProductSpace.Core` structure -/
def toNormedAddCommGroup : NormedAddCommGroup F :=
  AddGroupNorm.toNormedAddCommGroup
    { toFun := fun x => √(re ⟪x, x⟫)
                      /-
                        𝕜 : Type u_1
                        E : Type u_2
                        F : Type u_3
                        inst✝² : RCLike 𝕜
                        inst✝¹ : AddCommGroup F
                        inst✝ : Module 𝕜 F
                        cd : InnerProductSpace.Core 𝕜 F
                        ⊢ Eq ((fun x => (RCLike.re (Inner.inner x x)).sqrt) 0) 0
                      -/
      map_zero' := by simp only [sqrt_zero, inner_zero_right, map_zero]
                      /-
                        🎉 no goals
                      -/
                          /-
                            𝕜 : Type u_1
                            E : Type u_2
                            F : Type u_3
                            inst✝² : RCLike 𝕜
                            inst✝¹ : AddCommGroup F
                            inst✝ : Module 𝕜 F
                            cd : InnerProductSpace.Core 𝕜 F
                            x : F
                            ⊢ Eq ((fun x => (RCLike.re (Inner.inner x x)).sqrt) (Neg.neg x)) ((fun x => (R …
                          -/
      neg' := fun x => by simp only [inner_neg_left, neg_neg, inner_neg_right]
        /-
          𝕜 : Type u_1
          E : Type u_2
          F : Type u_3
          inst✝² : RCLike 𝕜
          inst✝¹ : AddCommGroup F
          inst✝ : Module 𝕜 F
          cd : InnerProductSpace.Core 𝕜 F
          x y : F
          ⊢ LE.le ((fun x => (RCLike.re (Inner.inner x x)).sqrt) (HAdd.hAdd x y)) (HAdd. …
        -/
                          /-
                            🎉 no goals
                          -/
        /-
          𝕜 : Type u_1
          E : Type u_2
          F : Type u_3
          inst✝² : RCLike 𝕜
          inst✝¹ : AddCommGroup F
          inst✝ : Module 𝕜 F
          cd : InnerProductSpace.Core 𝕜 F
          x y : F
          h₁ : LE.le (Norm.norm (Inner.inner x y)) (HMul.hMul (Norm.norm x) (Norm.norm y))
          ⊢ LE.le ((fun x => (RCLike.re (Inner.inner x x)).sqrt) (HAdd.hAdd x y)) (HAdd. …
        -/
      add_le' := fun x y => by
        /-
          𝕜 : Type u_1
          E : Type u_2
          F : Type u_3
          inst✝² : RCLike 𝕜
          inst✝¹ : AddCommGroup F
          inst✝ : Module 𝕜 F
          cd : InnerProductSpace.Core 𝕜 F
          x y : F
          h₁ : LE.le (Norm.norm (Inner.inner x y)) (HMul.hMul (Norm.norm x) (Norm.norm y))
          h₂ : LE.le (RCLike.re (Inner.inner x y)) (Norm.norm (Inner.inner x y))
          ⊢ LE.le ((fun x => (RCLike.re (Inner.inner x x)).sqrt) (HAdd.hAdd x y)) (HAdd. …
        -/
        have h₁ : ‖⟪x, y⟫‖ ≤ ‖x‖ * ‖y‖ := norm_inner_le_norm _ _
        /-
          𝕜 : Type u_1
          E : Type u_2
          F : Type u_3
          inst✝² : RCLike 𝕜
          inst✝¹ : AddCommGroup F
          inst✝ : Module 𝕜 F
          cd : InnerProductSpace.Core 𝕜 F
          x y : F
          h₁ : LE.le (Norm.norm (Inner.inner x y)) (HMul.hMul (Norm.norm x) (Norm.norm y))
          h₂ : LE.le (RCLike.re (Inner.inner x y)) (Norm.norm (Inner.inner x y))
          h₃ : LE.le (RCLike.re (Inner.inner x y)) (HMul.hMul (Norm.norm x) (Norm.norm y))
          ⊢ LE.le ((fun x => (RCLike.re (Inner.inner x x)).sqrt) (HAdd.hAdd x y)) (HAdd. …
        -/
        have h₂ : re ⟪x, y⟫ ≤ ‖⟪x, y⟫‖ := re_le_norm _
        have h₃ : re ⟪x, y⟫ ≤ ‖x‖ * ‖y‖ := h₂.trans h₁
        have h₄ : re ⟪y, x⟫ ≤ ‖x‖ * ‖y‖ := by rwa [← inner_conj_symm, conj_re]
        have : ‖x + y‖ * ‖x + y‖ ≤ (‖x‖ + ‖y‖) * (‖x‖ + ‖y‖) := by
        /-
          𝕜 : Type u_1
          E : Type u_2
          F : Type u_3
          inst✝² : RCLike 𝕜
          inst✝¹ : AddCommGroup F
          inst✝ : Module 𝕜 F
          cd : InnerProductSpace.Core 𝕜 F
          x y : F
          h₁ : LE.le (Norm.norm (Inner.inner x y)) (HMul.hMul (Norm.norm x) (Norm.norm y))
          h₂ : LE.le (RCLike.re (Inner.inner x y)) (Norm.norm (Inner.inner x y))
          h₃ : LE.le (RCLike.re (Inner.inner x y)) (HMul.hMul (Norm.norm x) (Norm.norm y))
          h₄ : LE.le (RCLike.re (Inner.inner y x)) (HMul.hMul (Norm.norm x) (Norm.norm y))
          this : LE.le (HMul.hMul (Norm.norm (HAdd.hAdd x y)) (Norm.norm (HAdd.hAdd x y) …
          ⊢ LE.le ((fun x => (RCLike.re (Inner.inner x x)).sqrt) (HAdd.hAdd x y)) (HAdd. …
        -/
          simp only [← inner_self_eq_norm_mul_norm, inner_add_add_self, mul_add, mul_comm, map_add]
        /-
          🎉 no goals
        -/
          linarith
        exact nonneg_le_nonneg_of_sq_le_sq (add_nonneg (sqrt_nonneg _) (sqrt_nonneg _)) this
      eq_zero_of_map_eq_zero' := fun _ hx =>
        normSq_eq_zero.1 <| (sqrt_eq_zero inner_self_nonneg).1 hx }


/-- Normed space structure constructed from an `InnerProductSpace.Core` structure -/
def toNormedSpace : NormedSpace 𝕜 F where
  norm_smul_le r x := by
    /-
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝² : RCLike 𝕜
      inst✝¹ : AddCommGroup F
      inst✝ : Module 𝕜 F
      cd : InnerProductSpace.Core 𝕜 F
      r : 𝕜
      x : F
      ⊢ LE.le (Norm.norm (HSMul.hSMul r x)) (HMul.hMul (Norm.norm r) (Norm.norm x))
    -/
    rw [norm_eq_sqrt_inner, inner_smul_left, inner_smul_right, ← mul_assoc]
    rw [RCLike.conj_mul, ← ofReal_pow, re_ofReal_mul, sqrt_mul, ← ofReal_normSq_eq_inner_self,
      ofReal_re]
      /-
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝² : RCLike 𝕜
        inst✝¹ : AddCommGroup F
        inst✝ : Module 𝕜 F
        cd : InnerProductSpace.Core 𝕜 F
        r : 𝕜
        x : F
        ⊢ LE.le (HMul.hMul (HPow.hPow (Norm.norm r) 2).sqrt (InnerProductSpace.Core.no …
      -/
    · simp [sqrt_normSq_eq_norm, RCLike.sqrt_normSq_eq_norm]
      /-
        🎉 no goals
      -/
      /-
        case hx
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝² : RCLike 𝕜
        inst✝¹ : AddCommGroup F
        inst✝ : Module 𝕜 F
        cd : InnerProductSpace.Core 𝕜 F
        r : 𝕜
        x : F
        ⊢ LE.le 0 (HPow.hPow (Norm.norm r) 2)
      -/
    · positivity
      /-
        🎉 no goals
      -/


/-- Given an `InnerProductSpace.Core` structure on a space, one can use it to turn
the space into an inner product space. The `NormedAddCommGroup` structure is expected
to already be defined with `InnerProductSpace.ofCore.toNormedAddCommGroup`. -/
def InnerProductSpace.ofCore [AddCommGroup F] [Module 𝕜 F] (cd : InnerProductSpace.Core 𝕜 F) :
    InnerProductSpace 𝕜 F :=
  letI : NormedSpace 𝕜 F := @InnerProductSpace.Core.toNormedSpace 𝕜 F _ _ _ cd
  { cd with
    norm_sq_eq_inner := fun x => by
      /-
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝² : RCLike 𝕜
        inst✝¹ : AddCommGroup F
        inst✝ : Module 𝕜 F
        cd : InnerProductSpace.Core 𝕜 F
        this : NormedSpace 𝕜 F := InnerProductSpace.Core.toNormedSpace
        x : F
        ⊢ Eq (HPow.hPow (Norm.norm x) 2) (RCLike.re (Inner.inner x x))
      -/
      have h₁ : ‖x‖ ^ 2 = √(re (cd.inner x x)) ^ 2 := rfl
      /-
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝² : RCLike 𝕜
        inst✝¹ : AddCommGroup F
        inst✝ : Module 𝕜 F
        cd : InnerProductSpace.Core 𝕜 F
        this : NormedSpace 𝕜 F := InnerProductSpace.Core.toNormedSpace
        x : F
        h₁ : Eq (HPow.hPow (Norm.norm x) 2) (HPow.hPow (RCLike.re (Inner.inner x x)).s …
        ⊢ Eq (HPow.hPow (Norm.norm x) 2) (RCLike.re (Inner.inner x x))
      -/
      have h₂ : 0 ≤ re (cd.inner x x) := InnerProductSpace.Core.inner_self_nonneg
      /-
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝² : RCLike 𝕜
        inst✝¹ : AddCommGroup F
        inst✝ : Module 𝕜 F
        cd : InnerProductSpace.Core 𝕜 F
        this : NormedSpace 𝕜 F := InnerProductSpace.Core.toNormedSpace
        x : F
        h₁ : Eq (HPow.hPow (Norm.norm x) 2) (HPow.hPow (RCLike.re (Inner.inner x x)).s …
        h₂ : LE.le 0 (RCLike.re (Inner.inner x x))
        ⊢ Eq (HPow.hPow (Norm.norm x) 2) (RCLike.re (Inner.inner x x))
      -/
      simp [h₁, sq_sqrt, h₂] }
      /-
        🎉 no goals
      -/


