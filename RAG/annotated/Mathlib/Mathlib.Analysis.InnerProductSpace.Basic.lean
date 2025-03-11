local notation "⟪" x ", " y "⟫" => @inner 𝕜 _ _ x y


local postfix:90 "†" => starRingEnd _


@[simp]
theorem inner_conj_symm (x y : E) : ⟪y, x⟫† = ⟪x, y⟫ :=
  InnerProductSpace.conj_symm _ _


theorem real_inner_comm (x y : F) : ⟪y, x⟫_ℝ = ⟪x, y⟫_ℝ :=
  @inner_conj_symm ℝ _ _ _ _ x y


theorem inner_eq_zero_symm {x y : E} : ⟪x, y⟫ = 0 ↔ ⟪y, x⟫ = 0 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    ⊢ Iff (Eq (Inner.inner x y) 0) (Eq (Inner.inner y x) 0)
  -/
  rw [← inner_conj_symm]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    ⊢ Iff (Eq ((starRingEnd 𝕜) (Inner.inner y x)) 0) (Eq (Inner.inner y x) 0)
  -/
  exact star_eq_zero
  /-
    🎉 no goals
  -/


@[simp]
                                                    /-
                                                      𝕜 : Type u_1
                                                      E : Type u_2
                                                      inst✝² : RCLike 𝕜
                                                      inst✝¹ : SeminormedAddCommGroup E
                                                      inst✝ : InnerProductSpace 𝕜 E
                                                      x : E
                                                      ⊢ Eq (RCLike.im (Inner.inner x x)) 0
                                                    -/
theorem inner_self_im (x : E) : im ⟪x, x⟫ = 0 := by rw [← @ofReal_inj 𝕜, im_eq_conj_sub]; simp
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


theorem inner_add_left (x y z : E) : ⟪x + y, z⟫ = ⟪x, z⟫ + ⟪y, z⟫ :=
  InnerProductSpace.add_left _ _ _


theorem inner_add_right (x y z : E) : ⟪x, y + z⟫ = ⟪x, y⟫ + ⟪x, z⟫ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y z : E
    ⊢ Eq (Inner.inner x (HAdd.hAdd y z)) (HAdd.hAdd (Inner.inner x y) (Inner.inner …
  -/
  rw [← inner_conj_symm, inner_add_left, RingHom.map_add]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y z : E
    ⊢ Eq (HAdd.hAdd ((starRingEnd 𝕜) (Inner.inner y x)) ((starRingEnd 𝕜) (Inner.in …
  -/
  simp only [inner_conj_symm]
  /-
    🎉 no goals
  -/


                                                              /-
                                                                𝕜 : Type u_1
                                                                E : Type u_2
                                                                inst✝² : RCLike 𝕜
                                                                inst✝¹ : SeminormedAddCommGroup E
                                                                inst✝ : InnerProductSpace 𝕜 E
                                                                x y : E
                                                                ⊢ Eq (RCLike.re (Inner.inner x y)) (RCLike.re (Inner.inner y x))
                                                              -/
theorem inner_re_symm (x y : E) : re ⟪x, y⟫ = re ⟪y, x⟫ := by rw [← inner_conj_symm, conj_re]
                                                              /-
                                                                🎉 no goals
                                                              -/


                                                               /-
                                                                 𝕜 : Type u_1
                                                                 E : Type u_2
                                                                 inst✝² : RCLike 𝕜
                                                                 inst✝¹ : SeminormedAddCommGroup E
                                                                 inst✝ : InnerProductSpace 𝕜 E
                                                                 x y : E
                                                                 ⊢ Eq (RCLike.im (Inner.inner x y)) (Neg.neg (RCLike.im (Inner.inner y x)))
                                                               -/
theorem inner_im_symm (x y : E) : im ⟪x, y⟫ = -im ⟪y, x⟫ := by rw [← inner_conj_symm, conj_im]
                                                               /-
                                                                 🎉 no goals
                                                               -/


/-- See `inner_smul_left` for the common special when `𝕜 = 𝕝`. -/
lemma inner_smul_left_eq_star_smul (x y : E) (r : 𝕝) : ⟪r • x, y⟫ = r† • ⟪x, y⟫ := by
  rw [← algebraMap_smul 𝕜 r, InnerProductSpace.smul_left, starRingEnd_apply, starRingEnd_apply,
    ← algebraMap_star_comm, ← smul_eq_mul, algebraMap_smul]


/-- Special case of `inner_smul_left_eq_star_smul` when the acting ring has a trivial star
(eg `ℕ`, `ℤ`, `ℚ≥0`, `ℚ`, `ℝ`). -/
lemma inner_smul_left_eq_smul [TrivialStar 𝕝] (x y : E) (r : 𝕝) : ⟪r • x, y⟫ = r • ⟪x, y⟫ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁹ : RCLike 𝕜
    inst✝⁸ : SeminormedAddCommGroup E
    inst✝⁷ : InnerProductSpace 𝕜 E
    𝕝 : Type u_4
    inst✝⁶ : CommSemiring 𝕝
    inst✝⁵ : StarRing 𝕝
    inst✝⁴ : Algebra 𝕝 𝕜
    inst✝³ : Module 𝕝 E
    inst✝² : IsScalarTower 𝕝 𝕜 E
    inst✝¹ : StarModule 𝕝 𝕜
    inst✝ : TrivialStar 𝕝
    x y : E
    r : 𝕝
    ⊢ Eq (Inner.inner (HSMul.hSMul r x) y) (HSMul.hSMul r (Inner.inner x y))
  -/
  rw [inner_smul_left_eq_star_smul, starRingEnd_apply, star_trivial]
  /-
    🎉 no goals
  -/


/-- See `inner_smul_right` for the common special when `𝕜 = 𝕝`. -/
lemma inner_smul_right_eq_smul (x y : E) (r : 𝕝) : ⟪x, r • y⟫ = r • ⟪x, y⟫ := by
  rw [← inner_conj_symm, inner_smul_left_eq_star_smul, starRingEnd_apply, starRingEnd_apply,
    star_smul, star_star, ← starRingEnd_apply, inner_conj_symm]


/-- See `inner_smul_left_eq_star_smul` for the case of a general algebra action. -/
theorem inner_smul_left (x y : E) (r : 𝕜) : ⟪r • x, y⟫ = r† * ⟪x, y⟫ :=
  inner_smul_left_eq_star_smul ..


theorem real_inner_smul_left (x y : F) (r : ℝ) : ⟪r • x, y⟫_ℝ = r * ⟪x, y⟫_ℝ :=
  inner_smul_left _ _ _


theorem inner_smul_real_left (x y : E) (r : ℝ) : ⟪(r : 𝕜) • x, y⟫ = r • ⟪x, y⟫ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    r : Real
    ⊢ Eq (Inner.inner (HSMul.hSMul (↑r) x) y) (HSMul.hSMul r (Inner.inner x y))
  -/
  rw [inner_smul_left, conj_ofReal, Algebra.smul_def]
  /-
    🎉 no goals
  -/


/-- See `inner_smul_right_eq_smul` for the case of a general algebra action. -/
theorem inner_smul_right (x y : E) (r : 𝕜) : ⟪x, r • y⟫ = r * ⟪x, y⟫ :=
  inner_smul_right_eq_smul ..


theorem real_inner_smul_right (x y : F) (r : ℝ) : ⟪x, r • y⟫_ℝ = r * ⟪x, y⟫_ℝ :=
  inner_smul_right _ _ _


theorem inner_smul_real_right (x y : E) (r : ℝ) : ⟪x, (r : 𝕜) • y⟫ = r • ⟪x, y⟫ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    r : Real
    ⊢ Eq (Inner.inner x (HSMul.hSMul (↑r) y)) (HSMul.hSMul r (Inner.inner x y))
  -/
  rw [inner_smul_right, Algebra.smul_def]
  /-
    🎉 no goals
  -/


/-- The inner product as a sesquilinear form.

Note that in the case `𝕜 = ℝ` this is a bilinear form. -/
@[simps!]
def sesqFormOfInner : E →ₗ[𝕜] E →ₗ⋆[𝕜] 𝕜 :=
  LinearMap.mk₂'ₛₗ (RingHom.id 𝕜) (starRingEnd _) (fun x y => ⟪y, x⟫)
    (fun _x _y _z => inner_add_right _ _ _) (fun _r _x _y => inner_smul_right _ _ _)
    (fun _x _y _z => inner_add_left _ _ _) fun _r _x _y => inner_smul_left _ _ _


/-- The real inner product as a bilinear form.

Note that unlike `sesqFormOfInner`, this does not reverse the order of the arguments. -/
@[simps!]
def bilinFormOfRealInner : BilinForm ℝ F := sesqFormOfInner.flip


/-- An inner product with a sum on the left. -/
theorem sum_inner {ι : Type*} (s : Finset ι) (f : ι → E) (x : E) :
    ⟪∑ i ∈ s, f i, x⟫ = ∑ i ∈ s, ⟪f i, x⟫ :=
  map_sum (sesqFormOfInner (𝕜 := 𝕜) (E := E) x) _ _


/-- An inner product with a sum on the right. -/
theorem inner_sum {ι : Type*} (s : Finset ι) (f : ι → E) (x : E) :
    ⟪x, ∑ i ∈ s, f i⟫ = ∑ i ∈ s, ⟪x, f i⟫ :=
  map_sum (LinearMap.flip sesqFormOfInner x) _ _


/-- An inner product with a sum on the left, `Finsupp` version. -/
protected theorem Finsupp.sum_inner {ι : Type*} (l : ι →₀ 𝕜) (v : ι → E) (x : E) :
    ⟪l.sum fun (i : ι) (a : 𝕜) => a • v i, x⟫ = l.sum fun (i : ι) (a : 𝕜) => conj a • ⟪v i, x⟫ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    ι : Type u_4
    l : Finsupp ι 𝕜
    v : ι → E
    x : E
    ⊢ Eq (Inner.inner (l.sum fun i a => HSMul.hSMul a (v i)) x) (l.sum fun i a =>  …
  -/
  convert sum_inner (𝕜 := 𝕜) l.support (fun a => l a • v a) x
  /-
    case h.e'_3
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    ι : Type u_4
    l : Finsupp ι 𝕜
    v : ι → E
    x : E
    ⊢ Eq (l.sum fun i a => HSMul.hSMul ((starRingEnd 𝕜) a) (Inner.inner (v i) x))  …
  -/
  simp only [inner_smul_left, Finsupp.sum, smul_eq_mul]
  /-
    🎉 no goals
  -/


/-- An inner product with a sum on the right, `Finsupp` version. -/
protected theorem Finsupp.inner_sum {ι : Type*} (l : ι →₀ 𝕜) (v : ι → E) (x : E) :
    ⟪x, l.sum fun (i : ι) (a : 𝕜) => a • v i⟫ = l.sum fun (i : ι) (a : 𝕜) => a • ⟪x, v i⟫ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    ι : Type u_4
    l : Finsupp ι 𝕜
    v : ι → E
    x : E
    ⊢ Eq (Inner.inner x (l.sum fun i a => HSMul.hSMul a (v i))) (l.sum fun i a =>  …
  -/
  convert inner_sum (𝕜 := 𝕜) l.support (fun a => l a • v a) x
  /-
    case h.e'_3
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    ι : Type u_4
    l : Finsupp ι 𝕜
    v : ι → E
    x : E
    ⊢ Eq (l.sum fun i a => HSMul.hSMul a (Inner.inner x (v i))) (l.support.sum fun …
  -/
  simp only [inner_smul_right, Finsupp.sum, smul_eq_mul]
  /-
    🎉 no goals
  -/


protected theorem DFinsupp.sum_inner {ι : Type*} [DecidableEq ι] {α : ι → Type*}
    [∀ i, AddZeroClass (α i)] [∀ (i) (x : α i), Decidable (x ≠ 0)] (f : ∀ i, α i → E)
    (l : Π₀ i, α i) (x : E) : ⟪l.sum f, x⟫ = l.sum fun i a => ⟪f i a, x⟫ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : SeminormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_4
    inst✝² : DecidableEq ι
    α : ι → Type u_5
    inst✝¹ : (i : ι) → AddZeroClass (α i)
    inst✝ : (i : ι) → (x : α i) → Decidable (Ne x 0)
    f : (i : ι) → α i → E
    l : DFinsupp fun i => α i
    x : E
    ⊢ Eq (Inner.inner (l.sum f) x) (l.sum fun i a => Inner.inner (f i a) x)
  -/
  simp +contextual only [DFinsupp.sum, sum_inner, smul_eq_mul]
  /-
    🎉 no goals
  -/


protected theorem DFinsupp.inner_sum {ι : Type*} [DecidableEq ι] {α : ι → Type*}
    [∀ i, AddZeroClass (α i)] [∀ (i) (x : α i), Decidable (x ≠ 0)] (f : ∀ i, α i → E)
    (l : Π₀ i, α i) (x : E) : ⟪x, l.sum f⟫ = l.sum fun i a => ⟪x, f i a⟫ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : SeminormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    ι : Type u_4
    inst✝² : DecidableEq ι
    α : ι → Type u_5
    inst✝¹ : (i : ι) → AddZeroClass (α i)
    inst✝ : (i : ι) → (x : α i) → Decidable (Ne x 0)
    f : (i : ι) → α i → E
    l : DFinsupp fun i => α i
    x : E
    ⊢ Eq (Inner.inner x (l.sum f)) (l.sum fun i a => Inner.inner x (f i a))
  -/
  simp +contextual only [DFinsupp.sum, inner_sum, smul_eq_mul]
  /-
    🎉 no goals
  -/


@[simp]
theorem inner_zero_left (x : E) : ⟪0, x⟫ = 0 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x : E
    ⊢ Eq (Inner.inner 0 x) 0
  -/
  rw [← zero_smul 𝕜 (0 : E), inner_smul_left, RingHom.map_zero, zero_mul]
  /-
    🎉 no goals
  -/


theorem inner_re_zero_left (x : E) : re ⟪0, x⟫ = 0 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x : E
    ⊢ Eq (RCLike.re (Inner.inner 0 x)) 0
  -/
  simp only [inner_zero_left, AddMonoidHom.map_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem inner_zero_right (x : E) : ⟪x, 0⟫ = 0 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x : E
    ⊢ Eq (Inner.inner x 0) 0
  -/
  rw [← inner_conj_symm, inner_zero_left, RingHom.map_zero]
  /-
    🎉 no goals
  -/


theorem inner_re_zero_right (x : E) : re ⟪x, 0⟫ = 0 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x : E
    ⊢ Eq (RCLike.re (Inner.inner x 0)) 0
  -/
  simp only [inner_zero_right, AddMonoidHom.map_zero]
  /-
    🎉 no goals
  -/


theorem inner_self_nonneg {x : E} : 0 ≤ re ⟪x, x⟫ :=
  PreInnerProductSpace.toCore.nonneg_re x


theorem real_inner_self_nonneg {x : F} : 0 ≤ ⟪x, x⟫_ℝ :=
  @inner_self_nonneg ℝ F _ _ _ x


@[simp]
theorem inner_self_ofReal_re (x : E) : (re ⟪x, x⟫ : 𝕜) = ⟪x, x⟫ :=
   /-
     𝕜 : Type u_1
     E : Type u_2
     inst✝² : RCLike 𝕜
     inst✝¹ : SeminormedAddCommGroup E
     inst✝ : InnerProductSpace 𝕜 E
     x : E
     ⊢ Eq ((List.cons (Eq ((starRingEnd 𝕜) (Inner.inner x x)) (Inner.inner x x)) (L …
   -/
   /-
     🎉 no goals
   -/
  ((RCLike.is_real_TFAE (⟪x, x⟫ : 𝕜)).out 2 3).2 (inner_self_im (𝕜 := 𝕜) x)
   /-
     🎉 no goals
   -/


theorem inner_self_eq_norm_sq_to_K (x : E) : ⟪x, x⟫ = (‖x‖ : 𝕜) ^ 2 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x : E
    ⊢ Eq (Inner.inner x x) (HPow.hPow (↑(Norm.norm x)) 2)
  -/
  rw [← inner_self_ofReal_re, ← norm_sq_eq_inner, ofReal_pow]
  /-
    🎉 no goals
  -/


theorem inner_self_re_eq_norm (x : E) : re ⟪x, x⟫ = ‖⟪x, x⟫‖ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x : E
    ⊢ Eq (RCLike.re (Inner.inner x x)) (Norm.norm (Inner.inner x x))
  -/
  conv_rhs => rw [← inner_self_ofReal_re]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x : E
    ⊢ Eq (RCLike.re (Inner.inner x x)) (Norm.norm ↑(RCLike.re (Inner.inner x x)))
  -/
  symm
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x : E
    ⊢ Eq (Norm.norm ↑(RCLike.re (Inner.inner x x))) (RCLike.re (Inner.inner x x))
  -/
  exact norm_of_nonneg inner_self_nonneg
  /-
    🎉 no goals
  -/


theorem inner_self_ofReal_norm (x : E) : (‖⟪x, x⟫‖ : 𝕜) = ⟪x, x⟫ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x : E
    ⊢ Eq (↑(Norm.norm (Inner.inner x x))) (Inner.inner x x)
  -/
  rw [← inner_self_re_eq_norm]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x : E
    ⊢ Eq (↑(RCLike.re (Inner.inner x x))) (Inner.inner x x)
  -/
  exact inner_self_ofReal_re _
  /-
    🎉 no goals
  -/


theorem real_inner_self_abs (x : F) : |⟪x, x⟫_ℝ| = ⟪x, x⟫_ℝ :=
  @inner_self_ofReal_norm ℝ F _ _ _ x


                                                              /-
                                                                𝕜 : Type u_1
                                                                E : Type u_2
                                                                inst✝² : RCLike 𝕜
                                                                inst✝¹ : SeminormedAddCommGroup E
                                                                inst✝ : InnerProductSpace 𝕜 E
                                                                x y : E
                                                                ⊢ Eq (Norm.norm (Inner.inner x y)) (Norm.norm (Inner.inner y x))
                                                              -/
theorem norm_inner_symm (x y : E) : ‖⟪x, y⟫‖ = ‖⟪y, x⟫‖ := by rw [← inner_conj_symm, norm_conj]
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp]
theorem inner_neg_left (x y : E) : ⟪-x, y⟫ = -⟪x, y⟫ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    ⊢ Eq (Inner.inner (Neg.neg x) y) (Neg.neg (Inner.inner x y))
  -/
  rw [← neg_one_smul 𝕜 x, inner_smul_left]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    ⊢ Eq (HMul.hMul ((starRingEnd 𝕜) (-1)) (Inner.inner x y)) (Neg.neg (Inner.inne …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem inner_neg_right (x y : E) : ⟪x, -y⟫ = -⟪x, y⟫ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    ⊢ Eq (Inner.inner x (Neg.neg y)) (Neg.neg (Inner.inner x y))
  -/
  rw [← inner_conj_symm, inner_neg_left]; simp only [RingHom.map_neg, inner_conj_symm]
                                          /-
                                            🎉 no goals
                                          -/


                                                          /-
                                                            𝕜 : Type u_1
                                                            E : Type u_2
                                                            inst✝² : RCLike 𝕜
                                                            inst✝¹ : SeminormedAddCommGroup E
                                                            inst✝ : InnerProductSpace 𝕜 E
                                                            x y : E
                                                            ⊢ Eq (Inner.inner (Neg.neg x) (Neg.neg y)) (Inner.inner x y)
                                                          -/
theorem inner_neg_neg (x y : E) : ⟪-x, -y⟫ = ⟪x, y⟫ := by simp
                                                          /-
                                                            🎉 no goals
                                                          -/

-- Porting note: removed `simp` because it can prove it using `inner_conj_symm`

theorem inner_self_conj (x : E) : ⟪x, x⟫† = ⟪x, x⟫ := inner_conj_symm _ _


theorem inner_sub_left (x y z : E) : ⟪x - y, z⟫ = ⟪x, z⟫ - ⟪y, z⟫ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y z : E
    ⊢ Eq (Inner.inner (HSub.hSub x y) z) (HSub.hSub (Inner.inner x z) (Inner.inner …
  -/
  simp [sub_eq_add_neg, inner_add_left]
  /-
    🎉 no goals
  -/


theorem inner_sub_right (x y z : E) : ⟪x, y - z⟫ = ⟪x, y⟫ - ⟪x, z⟫ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y z : E
    ⊢ Eq (Inner.inner x (HSub.hSub y z)) (HSub.hSub (Inner.inner x y) (Inner.inner …
  -/
  simp [sub_eq_add_neg, inner_add_right]
  /-
    🎉 no goals
  -/


theorem inner_mul_symm_re_eq_norm (x y : E) : re (⟪x, y⟫ * ⟪y, x⟫) = ‖⟪x, y⟫ * ⟪y, x⟫‖ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    ⊢ Eq (RCLike.re (HMul.hMul (Inner.inner x y) (Inner.inner y x))) (Norm.norm (H …
  -/
  rw [← inner_conj_symm, mul_comm]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    ⊢ Eq (RCLike.re (HMul.hMul (Inner.inner y x) ((starRingEnd 𝕜) (Inner.inner y x …
  -/
  exact re_eq_norm_of_mul_conj (inner y x)
  /-
    🎉 no goals
  -/


/-- Expand `⟪x + y, x + y⟫` -/
theorem inner_add_add_self (x y : E) : ⟪x + y, x + y⟫ = ⟪x, x⟫ + ⟪x, y⟫ + ⟪y, x⟫ + ⟪y, y⟫ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    ⊢ Eq (Inner.inner (HAdd.hAdd x y) (HAdd.hAdd x y)) (HAdd.hAdd (HAdd.hAdd (HAdd …
  -/
  simp only [inner_add_left, inner_add_right]; ring
                                               /-
                                                 🎉 no goals
                                               -/


/-- Expand `⟪x + y, x + y⟫_ℝ` -/
theorem real_inner_add_add_self (x y : F) :
    ⟪x + y, x + y⟫_ℝ = ⟪x, x⟫_ℝ + 2 * ⟪x, y⟫_ℝ + ⟪y, y⟫_ℝ := by
  /-
    F : Type u_3
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    x y : F
    ⊢ Eq (Inner.inner (HAdd.hAdd x y) (HAdd.hAdd x y)) (HAdd.hAdd (HAdd.hAdd (Inne …
  -/
  have : ⟪y, x⟫_ℝ = ⟪x, y⟫_ℝ := by rw [← inner_conj_symm]; rfl
  /-
    F : Type u_3
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    x y : F
    this : Eq (Inner.inner y x) (Inner.inner x y)
    ⊢ Eq (Inner.inner (HAdd.hAdd x y) (HAdd.hAdd x y)) (HAdd.hAdd (HAdd.hAdd (Inne …
  -/
  simp only [inner_add_add_self, this, add_left_inj]
  /-
    F : Type u_3
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    x y : F
    this : Eq (Inner.inner y x) (Inner.inner x y)
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (Inner.inner x x) (Inner.inner x y)) (Inner.inner x …
  -/
  ring
  /-
    🎉 no goals
  -/

-- Expand `⟪x - y, x - y⟫`

theorem inner_sub_sub_self (x y : E) : ⟪x - y, x - y⟫ = ⟪x, x⟫ - ⟪x, y⟫ - ⟪y, x⟫ + ⟪y, y⟫ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    ⊢ Eq (Inner.inner (HSub.hSub x y) (HSub.hSub x y)) (HAdd.hAdd (HSub.hSub (HSub …
  -/
  simp only [inner_sub_left, inner_sub_right]; ring
                                               /-
                                                 🎉 no goals
                                               -/


/-- Expand `⟪x - y, x - y⟫_ℝ` -/
theorem real_inner_sub_sub_self (x y : F) :
    ⟪x - y, x - y⟫_ℝ = ⟪x, x⟫_ℝ - 2 * ⟪x, y⟫_ℝ + ⟪y, y⟫_ℝ := by
  /-
    F : Type u_3
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    x y : F
    ⊢ Eq (Inner.inner (HSub.hSub x y) (HSub.hSub x y)) (HAdd.hAdd (HSub.hSub (Inne …
  -/
  have : ⟪y, x⟫_ℝ = ⟪x, y⟫_ℝ := by rw [← inner_conj_symm]; rfl
  /-
    F : Type u_3
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    x y : F
    this : Eq (Inner.inner y x) (Inner.inner x y)
    ⊢ Eq (Inner.inner (HSub.hSub x y) (HSub.hSub x y)) (HAdd.hAdd (HSub.hSub (Inne …
  -/
  simp only [inner_sub_sub_self, this, add_left_inj]
  /-
    F : Type u_3
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    x y : F
    this : Eq (Inner.inner y x) (Inner.inner x y)
    ⊢ Eq (HSub.hSub (HSub.hSub (Inner.inner x x) (Inner.inner x y)) (Inner.inner x …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- Parallelogram law -/
theorem parallelogram_law {x y : E} : ⟪x + y, x + y⟫ + ⟪x - y, x - y⟫ = 2 * (⟪x, x⟫ + ⟪y, y⟫) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    ⊢ Eq (HAdd.hAdd (Inner.inner (HAdd.hAdd x y) (HAdd.hAdd x y)) (Inner.inner (HS …
  -/
  simp only [inner_add_add_self, inner_sub_sub_self]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (Inner.inner x x) (Inner.inne …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- **Cauchy–Schwarz inequality**. -/
theorem inner_mul_inner_self_le (x y : E) : ‖⟪x, y⟫‖ * ‖⟪y, x⟫‖ ≤ re ⟪x, x⟫ * re ⟪y, y⟫ :=
  letI cd : PreInnerProductSpace.Core 𝕜 E := PreInnerProductSpace.toCore
  InnerProductSpace.Core.inner_mul_inner_self_le x y


/-- Cauchy–Schwarz inequality for real inner products. -/
theorem real_inner_mul_inner_self_le (x y : F) : ⟪x, y⟫_ℝ * ⟪x, y⟫_ℝ ≤ ⟪x, x⟫_ℝ * ⟪y, y⟫_ℝ :=
  calc
    ⟪x, y⟫_ℝ * ⟪x, y⟫_ℝ ≤ ‖⟪x, y⟫_ℝ‖ * ‖⟪y, x⟫_ℝ‖ := by
      /-
        F : Type u_3
        inst✝¹ : SeminormedAddCommGroup F
        inst✝ : InnerProductSpace Real F
        x y : F
        ⊢ LE.le (HMul.hMul (Inner.inner x y) (Inner.inner x y)) (HMul.hMul (Norm.norm  …
      -/
      rw [real_inner_comm y, ← norm_mul]
      /-
        F : Type u_3
        inst✝¹ : SeminormedAddCommGroup F
        inst✝ : InnerProductSpace Real F
        x y : F
        ⊢ LE.le (HMul.hMul (Inner.inner y x) (Inner.inner y x)) (Norm.norm (HMul.hMul  …
      -/
      exact le_abs_self _
      /-
        🎉 no goals
      -/
    _ ≤ ⟪x, x⟫_ℝ * ⟪y, y⟫_ℝ := @inner_mul_inner_self_le ℝ _ _ _ _ x y


@[simp]
theorem inner_self_eq_zero {x : E} : ⟪x, x⟫ = 0 ↔ x = 0 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x : E
    ⊢ Iff (Eq (Inner.inner x x) 0) (Eq x 0)
  -/
  rw [inner_self_eq_norm_sq_to_K, sq_eq_zero_iff, ofReal_eq_zero, norm_eq_zero]
  /-
    🎉 no goals
  -/


theorem inner_self_ne_zero {x : E} : ⟪x, x⟫ ≠ 0 ↔ x ≠ 0 :=
  inner_self_eq_zero.not


theorem ext_inner_left {x y : E} (h : ∀ v, ⟪v, x⟫ = ⟪v, y⟫) : x = y := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    h : ∀ (v : E), Eq (Inner.inner v x) (Inner.inner v y)
    ⊢ Eq x y
  -/
  rw [← sub_eq_zero, ← @inner_self_eq_zero 𝕜, inner_sub_right, sub_eq_zero, h (x - y)]
  /-
    🎉 no goals
  -/


theorem ext_inner_right {x y : E} (h : ∀ v, ⟪x, v⟫ = ⟪y, v⟫) : x = y := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    h : ∀ (v : E), Eq (Inner.inner x v) (Inner.inner y v)
    ⊢ Eq x y
  -/
  rw [← sub_eq_zero, ← @inner_self_eq_zero 𝕜, inner_sub_left, sub_eq_zero, h (x - y)]
  /-
    🎉 no goals
  -/


@[simp]
theorem inner_self_nonpos {x : E} : re ⟪x, x⟫ ≤ 0 ↔ x = 0 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x : E
    ⊢ Iff (LE.le (RCLike.re (Inner.inner x x)) 0) (Eq x 0)
  -/
  rw [← norm_sq_eq_inner, (sq_nonneg _).le_iff_eq, sq_eq_zero_iff, norm_eq_zero]
  /-
    🎉 no goals
  -/


open scoped InnerProductSpace in
theorem real_inner_self_nonpos {x : F} : ⟪x, x⟫_ℝ ≤ 0 ↔ x = 0 :=
  @inner_self_nonpos ℝ F _ _ _ x


/-- A family of vectors is linearly independent if they are nonzero
and orthogonal. -/
theorem linearIndependent_of_ne_zero_of_inner_eq_zero {ι : Type*} {v : ι → E} (hz : ∀ i, v i ≠ 0)
    (ho : Pairwise fun i j => ⟪v i, v j⟫ = 0) : LinearIndependent 𝕜 v := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    ι : Type u_4
    v : ι → E
    hz : ∀ (i : ι), Ne (v i) 0
    ho : Pairwise fun i j => Eq (Inner.inner (v i) (v j)) 0
    ⊢ LinearIndependent 𝕜 v
  -/
  rw [linearIndependent_iff']
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    ι : Type u_4
    v : ι → E
    hz : ∀ (i : ι), Ne (v i) 0
    ho : Pairwise fun i j => Eq (Inner.inner (v i) (v j)) 0
    ⊢ ∀ (s : Finset ι) (g : ι → 𝕜), Eq (s.sum fun i => HSMul.hSMul (g i) (v i)) 0  …
  -/
  intro s g hg i hi
  have h' : g i * inner (v i) (v i) = inner (v i) (∑ j ∈ s, g j • v j) := by
    rw [inner_sum]
    symm
    convert Finset.sum_eq_single (β := 𝕜) i ?_ ?_
    · rw [inner_smul_right]
    · intro j _hj hji
      rw [inner_smul_right, ho hji.symm, mul_zero]
    · exact fun h => False.elim (h hi)
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    ι : Type u_4
    v : ι → E
    hz : ∀ (i : ι), Ne (v i) 0
    ho : Pairwise fun i j => Eq (Inner.inner (v i) (v j)) 0
    s : Finset ι
    g : ι → 𝕜
    hg : Eq (s.sum fun i => HSMul.hSMul (g i) (v i)) 0
    i : ι
    hi : Membership.mem s i
    h' : Eq (HMul.hMul (g i) (Inner.inner (v i) (v i))) (Inner.inner (v i) (s.sum  …
    ⊢ Eq (g i) 0
  -/
  simpa [hg, hz] using h'
  /-
    🎉 no goals
  -/


local notation "IK" => @RCLike.I 𝕜 _


theorem norm_eq_sqrt_inner (x : E) : ‖x‖ = √(re ⟪x, x⟫) :=
  calc
    ‖x‖ = √(‖x‖ ^ 2) := (sqrt_sq (norm_nonneg _)).symm
    _ = √(re ⟪x, x⟫) := congr_arg _ (norm_sq_eq_inner _)


theorem norm_eq_sqrt_real_inner (x : F) : ‖x‖ = √⟪x, x⟫_ℝ :=
  @norm_eq_sqrt_inner ℝ _ _ _ _ x


theorem inner_self_eq_norm_mul_norm (x : E) : re ⟪x, x⟫ = ‖x‖ * ‖x‖ := by
  rw [@norm_eq_sqrt_inner 𝕜, ← sqrt_mul inner_self_nonneg (re ⟪x, x⟫),
    sqrt_mul_self inner_self_nonneg]


theorem inner_self_eq_norm_sq (x : E) : re ⟪x, x⟫ = ‖x‖ ^ 2 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x : E
    ⊢ Eq (RCLike.re (Inner.inner x x)) (HPow.hPow (Norm.norm x) 2)
  -/
  rw [pow_two, inner_self_eq_norm_mul_norm]
  /-
    🎉 no goals
  -/


theorem real_inner_self_eq_norm_mul_norm (x : F) : ⟪x, x⟫_ℝ = ‖x‖ * ‖x‖ := by
  /-
    F : Type u_3
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    x : F
    ⊢ Eq (Inner.inner x x) (HMul.hMul (Norm.norm x) (Norm.norm x))
  -/
  have h := @inner_self_eq_norm_mul_norm ℝ F _ _ _ x
  /-
    F : Type u_3
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    x : F
    h : Eq (RCLike.re (Inner.inner x x)) (HMul.hMul (Norm.norm x) (Norm.norm x))
    ⊢ Eq (Inner.inner x x) (HMul.hMul (Norm.norm x) (Norm.norm x))
  -/
  simpa using h
  /-
    🎉 no goals
  -/


theorem real_inner_self_eq_norm_sq (x : F) : ⟪x, x⟫_ℝ = ‖x‖ ^ 2 := by
  /-
    F : Type u_3
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    x : F
    ⊢ Eq (Inner.inner x x) (HPow.hPow (Norm.norm x) 2)
  -/
  rw [pow_two, real_inner_self_eq_norm_mul_norm]
  /-
    🎉 no goals
  -/

-- Porting note: this was present in mathlib3 but seemingly didn't do anything.
-- variable (𝕜)


/-- Expand the square -/
theorem norm_add_sq (x y : E) : ‖x + y‖ ^ 2 = ‖x‖ ^ 2 + 2 * re ⟪x, y⟫ + ‖y‖ ^ 2 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    ⊢ Eq (HPow.hPow (Norm.norm (HAdd.hAdd x y)) 2) (HAdd.hAdd (HAdd.hAdd (HPow.hPo …
  -/
  repeat' rw [sq (M := ℝ), ← @inner_self_eq_norm_mul_norm 𝕜]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    ⊢ Eq (RCLike.re (Inner.inner (HAdd.hAdd x y) (HAdd.hAdd x y))) (HAdd.hAdd (HAd …
  -/
  rw [inner_add_add_self, two_mul]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    ⊢ Eq (RCLike.re (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (Inner.inner x x) (Inner.inne …
  -/
  simp only [add_assoc, add_left_inj, add_right_inj, AddMonoidHom.map_add]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    ⊢ Eq (RCLike.re (Inner.inner y x)) (RCLike.re (Inner.inner x y))
  -/
  rw [← inner_conj_symm, conj_re]
  /-
    🎉 no goals
  -/


alias norm_add_pow_two := norm_add_sq


/-- Expand the square -/
theorem norm_add_sq_real (x y : F) : ‖x + y‖ ^ 2 = ‖x‖ ^ 2 + 2 * ⟪x, y⟫_ℝ + ‖y‖ ^ 2 := by
  /-
    F : Type u_3
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    x y : F
    ⊢ Eq (HPow.hPow (Norm.norm (HAdd.hAdd x y)) 2) (HAdd.hAdd (HAdd.hAdd (HPow.hPo …
  -/
  have h := @norm_add_sq ℝ _ _ _ _ x y
  /-
    F : Type u_3
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    x y : F
    h : Eq (HPow.hPow (Norm.norm (HAdd.hAdd x y)) 2) (HAdd.hAdd (HAdd.hAdd (HPow.h …
    ⊢ Eq (HPow.hPow (Norm.norm (HAdd.hAdd x y)) 2) (HAdd.hAdd (HAdd.hAdd (HPow.hPo …
  -/
  simpa using h
  /-
    🎉 no goals
  -/


alias norm_add_pow_two_real := norm_add_sq_real


/-- Expand the square -/
theorem norm_add_mul_self (x y : E) :
    ‖x + y‖ * ‖x + y‖ = ‖x‖ * ‖x‖ + 2 * re ⟪x, y⟫ + ‖y‖ * ‖y‖ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    ⊢ Eq (HMul.hMul (Norm.norm (HAdd.hAdd x y)) (Norm.norm (HAdd.hAdd x y))) (HAdd …
  -/
  repeat' rw [← sq (M := ℝ)]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    ⊢ Eq (HPow.hPow (Norm.norm (HAdd.hAdd x y)) 2) (HAdd.hAdd (HAdd.hAdd (HPow.hPo …
  -/
  exact norm_add_sq _ _
  /-
    🎉 no goals
  -/


/-- Expand the square -/
theorem norm_add_mul_self_real (x y : F) :
    ‖x + y‖ * ‖x + y‖ = ‖x‖ * ‖x‖ + 2 * ⟪x, y⟫_ℝ + ‖y‖ * ‖y‖ := by
  /-
    F : Type u_3
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    x y : F
    ⊢ Eq (HMul.hMul (Norm.norm (HAdd.hAdd x y)) (Norm.norm (HAdd.hAdd x y))) (HAdd …
  -/
  have h := @norm_add_mul_self ℝ _ _ _ _ x y
  /-
    F : Type u_3
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    x y : F
    h : Eq (HMul.hMul (Norm.norm (HAdd.hAdd x y)) (Norm.norm (HAdd.hAdd x y))) (HA …
    ⊢ Eq (HMul.hMul (Norm.norm (HAdd.hAdd x y)) (Norm.norm (HAdd.hAdd x y))) (HAdd …
  -/
  simpa using h
  /-
    🎉 no goals
  -/


/-- Expand the square -/
theorem norm_sub_sq (x y : E) : ‖x - y‖ ^ 2 = ‖x‖ ^ 2 - 2 * re ⟪x, y⟫ + ‖y‖ ^ 2 := by
  rw [sub_eq_add_neg, @norm_add_sq 𝕜 _ _ _ _ x (-y), norm_neg, inner_neg_right, map_neg, mul_neg,
    sub_eq_add_neg]


alias norm_sub_pow_two := norm_sub_sq


/-- Expand the square -/
theorem norm_sub_sq_real (x y : F) : ‖x - y‖ ^ 2 = ‖x‖ ^ 2 - 2 * ⟪x, y⟫_ℝ + ‖y‖ ^ 2 :=
  @norm_sub_sq ℝ _ _ _ _ _ _


alias norm_sub_pow_two_real := norm_sub_sq_real


/-- Expand the square -/
theorem norm_sub_mul_self (x y : E) :
    ‖x - y‖ * ‖x - y‖ = ‖x‖ * ‖x‖ - 2 * re ⟪x, y⟫ + ‖y‖ * ‖y‖ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    ⊢ Eq (HMul.hMul (Norm.norm (HSub.hSub x y)) (Norm.norm (HSub.hSub x y))) (HAdd …
  -/
  repeat' rw [← sq (M := ℝ)]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    ⊢ Eq (HPow.hPow (Norm.norm (HSub.hSub x y)) 2) (HAdd.hAdd (HSub.hSub (HPow.hPo …
  -/
  exact norm_sub_sq _ _
  /-
    🎉 no goals
  -/


/-- Expand the square -/
theorem norm_sub_mul_self_real (x y : F) :
    ‖x - y‖ * ‖x - y‖ = ‖x‖ * ‖x‖ - 2 * ⟪x, y⟫_ℝ + ‖y‖ * ‖y‖ := by
  /-
    F : Type u_3
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    x y : F
    ⊢ Eq (HMul.hMul (Norm.norm (HSub.hSub x y)) (Norm.norm (HSub.hSub x y))) (HAdd …
  -/
  have h := @norm_sub_mul_self ℝ _ _ _ _ x y
  /-
    F : Type u_3
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    x y : F
    h : Eq (HMul.hMul (Norm.norm (HSub.hSub x y)) (Norm.norm (HSub.hSub x y))) (HA …
    ⊢ Eq (HMul.hMul (Norm.norm (HSub.hSub x y)) (Norm.norm (HSub.hSub x y))) (HAdd …
  -/
  simpa using h
  /-
    🎉 no goals
  -/


/-- Cauchy–Schwarz inequality with norm -/
theorem norm_inner_le_norm (x y : E) : ‖⟪x, y⟫‖ ≤ ‖x‖ * ‖y‖ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    ⊢ LE.le (Norm.norm (Inner.inner x y)) (HMul.hMul (Norm.norm x) (Norm.norm y))
  -/
  rw [norm_eq_sqrt_inner (𝕜 := 𝕜) x, norm_eq_sqrt_inner (𝕜 := 𝕜) y]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    ⊢ LE.le (Norm.norm (Inner.inner x y)) (HMul.hMul (RCLike.re (Inner.inner x x)) …
  -/
  letI : PreInnerProductSpace.Core 𝕜 E := PreInnerProductSpace.toCore
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    this : PreInnerProductSpace.Core 𝕜 E := PreInnerProductSpace.toCore
    ⊢ LE.le (Norm.norm (Inner.inner x y)) (HMul.hMul (RCLike.re (Inner.inner x x)) …
  -/
  exact InnerProductSpace.Core.norm_inner_le_norm x y
  /-
    🎉 no goals
  -/


theorem nnnorm_inner_le_nnnorm (x y : E) : ‖⟪x, y⟫‖₊ ≤ ‖x‖₊ * ‖y‖₊ :=
  norm_inner_le_norm x y


theorem re_inner_le_norm (x y : E) : re ⟪x, y⟫ ≤ ‖x‖ * ‖y‖ :=
  le_trans (re_le_norm (inner x y)) (norm_inner_le_norm x y)


/-- Cauchy–Schwarz inequality with norm -/
theorem abs_real_inner_le_norm (x y : F) : |⟪x, y⟫_ℝ| ≤ ‖x‖ * ‖y‖ :=
  (Real.norm_eq_abs _).ge.trans (norm_inner_le_norm x y)


/-- Cauchy–Schwarz inequality with norm -/
theorem real_inner_le_norm (x y : F) : ⟪x, y⟫_ℝ ≤ ‖x‖ * ‖y‖ :=
  le_trans (le_abs_self _) (abs_real_inner_le_norm _ _)


lemma inner_eq_zero_of_left {x : E} (y : E) (h : ‖x‖ = 0) : ⟪x, y⟫_𝕜 = 0 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    h : Eq (Norm.norm x) 0
    ⊢ Eq (Inner.inner x y) 0
  -/
  rw [← norm_eq_zero]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    h : Eq (Norm.norm x) 0
    ⊢ Eq (Norm.norm (Inner.inner x y)) 0
  -/
  refine le_antisymm ?_ (by positivity)
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    h : Eq (Norm.norm x) 0
    ⊢ LE.le (Norm.norm (Inner.inner x y)) 0
  -/
  exact norm_inner_le_norm _ _ |>.trans <| by simp [h]
  /-
    🎉 no goals
  -/


lemma inner_eq_zero_of_right (x : E) {y : E} (h : ‖y‖ = 0) : ⟪x, y⟫_𝕜 = 0 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    h : Eq (Norm.norm y) 0
    ⊢ Eq (Inner.inner x y) 0
  -/
  rw [inner_eq_zero_symm, inner_eq_zero_of_left _ h]
  /-
    🎉 no goals
  -/


include 𝕜 in
theorem parallelogram_law_with_norm (x y : E) :
    ‖x + y‖ * ‖x + y‖ + ‖x - y‖ * ‖x - y‖ = 2 * (‖x‖ * ‖x‖ + ‖y‖ * ‖y‖) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Norm.norm (HAdd.hAdd x y)) (Norm.norm (HAdd.hAdd x …
  -/
  simp only [← @inner_self_eq_norm_mul_norm 𝕜]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    ⊢ Eq (HAdd.hAdd (RCLike.re (Inner.inner (HAdd.hAdd x y) (HAdd.hAdd x y))) (RCL …
  -/
  rw [← re.map_add, parallelogram_law, two_mul, two_mul]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    ⊢ Eq (RCLike.re (HAdd.hAdd (HAdd.hAdd (Inner.inner x x) (Inner.inner y y)) (HA …
  -/
  simp only [re.map_add]
  /-
    🎉 no goals
  -/


include 𝕜 in
theorem parallelogram_law_with_nnnorm (x y : E) :
    ‖x + y‖₊ * ‖x + y‖₊ + ‖x - y‖₊ * ‖x - y‖₊ = 2 * (‖x‖₊ * ‖x‖₊ + ‖y‖₊ * ‖y‖₊) :=
  Subtype.ext <| parallelogram_law_with_norm 𝕜 x y


/-- Polarization identity: The real part of the inner product, in terms of the norm. -/
theorem re_inner_eq_norm_add_mul_self_sub_norm_mul_self_sub_norm_mul_self_div_two (x y : E) :
    re ⟪x, y⟫ = (‖x + y‖ * ‖x + y‖ - ‖x‖ * ‖x‖ - ‖y‖ * ‖y‖) / 2 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    ⊢ Eq (RCLike.re (Inner.inner x y)) (HDiv.hDiv (HSub.hSub (HSub.hSub (HMul.hMul …
  -/
  rw [@norm_add_mul_self 𝕜]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    ⊢ Eq (RCLike.re (Inner.inner x y)) (HDiv.hDiv (HSub.hSub (HSub.hSub (HAdd.hAdd …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- Polarization identity: The real part of the inner product, in terms of the norm. -/
theorem re_inner_eq_norm_mul_self_add_norm_mul_self_sub_norm_sub_mul_self_div_two (x y : E) :
    re ⟪x, y⟫ = (‖x‖ * ‖x‖ + ‖y‖ * ‖y‖ - ‖x - y‖ * ‖x - y‖) / 2 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    ⊢ Eq (RCLike.re (Inner.inner x y)) (HDiv.hDiv (HSub.hSub (HAdd.hAdd (HMul.hMul …
  -/
  rw [@norm_sub_mul_self 𝕜]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    ⊢ Eq (RCLike.re (Inner.inner x y)) (HDiv.hDiv (HSub.hSub (HAdd.hAdd (HMul.hMul …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- Polarization identity: The real part of the inner product, in terms of the norm. -/
theorem re_inner_eq_norm_add_mul_self_sub_norm_sub_mul_self_div_four (x y : E) :
    re ⟪x, y⟫ = (‖x + y‖ * ‖x + y‖ - ‖x - y‖ * ‖x - y‖) / 4 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    ⊢ Eq (RCLike.re (Inner.inner x y)) (HDiv.hDiv (HSub.hSub (HMul.hMul (Norm.norm …
  -/
  rw [@norm_add_mul_self 𝕜, @norm_sub_mul_self 𝕜]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    ⊢ Eq (RCLike.re (Inner.inner x y)) (HDiv.hDiv (HSub.hSub (HAdd.hAdd (HAdd.hAdd …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- Polarization identity: The imaginary part of the inner product, in terms of the norm. -/
theorem im_inner_eq_norm_sub_i_smul_mul_self_sub_norm_add_i_smul_mul_self_div_four (x y : E) :
    im ⟪x, y⟫ = (‖x - IK • y‖ * ‖x - IK • y‖ - ‖x + IK • y‖ * ‖x + IK • y‖) / 4 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    ⊢ Eq (RCLike.im (Inner.inner x y)) (HDiv.hDiv (HSub.hSub (HMul.hMul (Norm.norm …
  -/
  simp only [@norm_add_mul_self 𝕜, @norm_sub_mul_self 𝕜, inner_smul_right, I_mul_re]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    ⊢ Eq (RCLike.im (Inner.inner x y)) (HDiv.hDiv (HSub.hSub (HAdd.hAdd (HSub.hSub …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- Polarization identity: The inner product, in terms of the norm. -/
theorem inner_eq_sum_norm_sq_div_four (x y : E) :
    ⟪x, y⟫ = ((‖x + y‖ : 𝕜) ^ 2 - (‖x - y‖ : 𝕜) ^ 2 +
              ((‖x - IK • y‖ : 𝕜) ^ 2 - (‖x + IK • y‖ : 𝕜) ^ 2) * IK) / 4 := by
  rw [← re_add_im ⟪x, y⟫, re_inner_eq_norm_add_mul_self_sub_norm_sub_mul_self_div_four,
    im_inner_eq_norm_sub_i_smul_mul_self_sub_norm_add_i_smul_mul_self_div_four]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    ⊢ Eq (HAdd.hAdd (↑(HDiv.hDiv (HSub.hSub (HMul.hMul (Norm.norm (HAdd.hAdd x y)) …
  -/
  push_cast
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    ⊢ Eq (HAdd.hAdd (HDiv.hDiv (HSub.hSub (HMul.hMul ↑(Norm.norm (HAdd.hAdd x y))  …
  -/
  simp only [sq, ← mul_div_right_comm, ← add_div]
  /-
    🎉 no goals
  -/

-- See note [lower instance priority]

instance (priority := 100) InnerProductSpace.toUniformConvexSpace : UniformConvexSpace F :=
  ⟨fun ε hε => by
    refine
      ⟨2 - √(4 - ε ^ 2), sub_pos_of_lt <| (sqrt_lt' zero_lt_two).2 ?_, fun x hx y hy hxy => ?_⟩
      /-
        case refine_1
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝⁴ : RCLike 𝕜
        inst✝³ : SeminormedAddCommGroup E
        inst✝² : InnerProductSpace 𝕜 E
        inst✝¹ : SeminormedAddCommGroup F
        inst✝ : InnerProductSpace Real F
        ε : Real
        hε : LT.lt 0 ε
        ⊢ LT.lt (HSub.hSub 4 (HPow.hPow ε 2)) (HPow.hPow 2 2)
      -/
    · norm_num
      /-
        case refine_1
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝⁴ : RCLike 𝕜
        inst✝³ : SeminormedAddCommGroup E
        inst✝² : InnerProductSpace 𝕜 E
        inst✝¹ : SeminormedAddCommGroup F
        inst✝ : InnerProductSpace Real F
        ε : Real
        hε : LT.lt 0 ε
        ⊢ LT.lt 0 (HPow.hPow ε 2)
      -/
      exact pow_pos hε _
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : RCLike 𝕜
      inst✝³ : SeminormedAddCommGroup E
      inst✝² : InnerProductSpace 𝕜 E
      inst✝¹ : SeminormedAddCommGroup F
      inst✝ : InnerProductSpace Real F
      ε : Real
      hε : LT.lt 0 ε
      x : F
      hx : Eq (Norm.norm x) 1
      y : F
      hy : Eq (Norm.norm y) 1
      hxy : LE.le ε (Norm.norm (HSub.hSub x y))
      ⊢ LE.le (Norm.norm (HAdd.hAdd x y)) (HSub.hSub 2 (HSub.hSub 2 (HSub.hSub 4 (HP …
    -/
    rw [sub_sub_cancel]
    /-
      case refine_2
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : RCLike 𝕜
      inst✝³ : SeminormedAddCommGroup E
      inst✝² : InnerProductSpace 𝕜 E
      inst✝¹ : SeminormedAddCommGroup F
      inst✝ : InnerProductSpace Real F
      ε : Real
      hε : LT.lt 0 ε
      x : F
      hx : Eq (Norm.norm x) 1
      y : F
      hy : Eq (Norm.norm y) 1
      hxy : LE.le ε (Norm.norm (HSub.hSub x y))
      ⊢ LE.le (Norm.norm (HAdd.hAdd x y)) (HSub.hSub 4 (HPow.hPow ε 2)).sqrt
    -/
    refine le_sqrt_of_sq_le ?_
    /-
      case refine_2
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : RCLike 𝕜
      inst✝³ : SeminormedAddCommGroup E
      inst✝² : InnerProductSpace 𝕜 E
      inst✝¹ : SeminormedAddCommGroup F
      inst✝ : InnerProductSpace Real F
      ε : Real
      hε : LT.lt 0 ε
      x : F
      hx : Eq (Norm.norm x) 1
      y : F
      hy : Eq (Norm.norm y) 1
      hxy : LE.le ε (Norm.norm (HSub.hSub x y))
      ⊢ LE.le (HPow.hPow (Norm.norm (HAdd.hAdd x y)) 2) (HSub.hSub 4 (HPow.hPow ε 2))
    -/
    rw [sq, eq_sub_iff_add_eq.2 (parallelogram_law_with_norm ℝ x y), ← sq ‖x - y‖, hx, hy]
    /-
      case refine_2
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : RCLike 𝕜
      inst✝³ : SeminormedAddCommGroup E
      inst✝² : InnerProductSpace 𝕜 E
      inst✝¹ : SeminormedAddCommGroup F
      inst✝ : InnerProductSpace Real F
      ε : Real
      hε : LT.lt 0 ε
      x : F
      hx : Eq (Norm.norm x) 1
      y : F
      hy : Eq (Norm.norm y) 1
      hxy : LE.le ε (Norm.norm (HSub.hSub x y))
      ⊢ LE.le (HSub.hSub (HMul.hMul 2 (HAdd.hAdd (HMul.hMul 1 1) (HMul.hMul 1 1))) ( …
    -/
    ring_nf
    /-
      case refine_2
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : RCLike 𝕜
      inst✝³ : SeminormedAddCommGroup E
      inst✝² : InnerProductSpace 𝕜 E
      inst✝¹ : SeminormedAddCommGroup F
      inst✝ : InnerProductSpace Real F
      ε : Real
      hε : LT.lt 0 ε
      x : F
      hx : Eq (Norm.norm x) 1
      y : F
      hy : Eq (Norm.norm y) 1
      hxy : LE.le ε (Norm.norm (HSub.hSub x y))
      ⊢ LE.le (HSub.hSub 4 (HPow.hPow (Norm.norm (HSub.hSub x y)) 2)) (HSub.hSub 4 ( …
    -/
    gcongr⟩
    /-
      🎉 no goals
    -/


/-- Polarization identity: The real inner product, in terms of the norm. -/
theorem real_inner_eq_norm_add_mul_self_sub_norm_mul_self_sub_norm_mul_self_div_two (x y : F) :
    ⟪x, y⟫_ℝ = (‖x + y‖ * ‖x + y‖ - ‖x‖ * ‖x‖ - ‖y‖ * ‖y‖) / 2 :=
  re_to_real.symm.trans <|
    re_inner_eq_norm_add_mul_self_sub_norm_mul_self_sub_norm_mul_self_div_two x y


/-- Polarization identity: The real inner product, in terms of the norm. -/
theorem real_inner_eq_norm_mul_self_add_norm_mul_self_sub_norm_sub_mul_self_div_two (x y : F) :
    ⟪x, y⟫_ℝ = (‖x‖ * ‖x‖ + ‖y‖ * ‖y‖ - ‖x - y‖ * ‖x - y‖) / 2 :=
  re_to_real.symm.trans <|
    re_inner_eq_norm_mul_self_add_norm_mul_self_sub_norm_sub_mul_self_div_two x y


/-- Pythagorean theorem, if-and-only-if vector inner product form. -/
theorem norm_add_sq_eq_norm_sq_add_norm_sq_iff_real_inner_eq_zero (x y : F) :
    ‖x + y‖ * ‖x + y‖ = ‖x‖ * ‖x‖ + ‖y‖ * ‖y‖ ↔ ⟪x, y⟫_ℝ = 0 := by
  /-
    F : Type u_3
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    x y : F
    ⊢ Iff (Eq (HMul.hMul (Norm.norm (HAdd.hAdd x y)) (Norm.norm (HAdd.hAdd x y)))  …
  -/
  rw [@norm_add_mul_self ℝ, add_right_cancel_iff, add_right_eq_self, mul_eq_zero]
  /-
    F : Type u_3
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    x y : F
    ⊢ Iff (Or (Eq 2 0) (Eq (RCLike.re (Inner.inner x y)) 0)) (Eq (Inner.inner x y) …
  -/
  norm_num
  /-
    🎉 no goals
  -/


/-- Pythagorean theorem, if-and-if vector inner product form using square roots. -/
theorem norm_add_eq_sqrt_iff_real_inner_eq_zero {x y : F} :
    ‖x + y‖ = √(‖x‖ * ‖x‖ + ‖y‖ * ‖y‖) ↔ ⟪x, y⟫_ℝ = 0 := by
  rw [← norm_add_sq_eq_norm_sq_add_norm_sq_iff_real_inner_eq_zero, eq_comm, sqrt_eq_iff_mul_self_eq,
                 /-
                   case hx
                   F : Type u_3
                   inst✝¹ : SeminormedAddCommGroup F
                   inst✝ : InnerProductSpace Real F
                   x y : F
                   ⊢ LE.le 0 (HAdd.hAdd (HMul.hMul (Norm.norm x) (Norm.norm x)) (HMul.hMul (Norm. …
                 -/
                 /-
                   🎉 no goals
                 -/
    eq_comm] <;> positivity
                 /-
                   🎉 no goals
                 -/


/-- Pythagorean theorem, vector inner product form. -/
theorem norm_add_sq_eq_norm_sq_add_norm_sq_of_inner_eq_zero (x y : E) (h : ⟪x, y⟫ = 0) :
    ‖x + y‖ * ‖x + y‖ = ‖x‖ * ‖x‖ + ‖y‖ * ‖y‖ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    h : Eq (Inner.inner x y) 0
    ⊢ Eq (HMul.hMul (Norm.norm (HAdd.hAdd x y)) (Norm.norm (HAdd.hAdd x y))) (HAdd …
  -/
  rw [@norm_add_mul_self 𝕜, add_right_cancel_iff, add_right_eq_self, mul_eq_zero]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    h : Eq (Inner.inner x y) 0
    ⊢ Or (Eq 2 0) (Eq (RCLike.re (Inner.inner x y)) 0)
  -/
  apply Or.inr
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    h : Eq (Inner.inner x y) 0
    ⊢ Eq (RCLike.re (Inner.inner x y)) 0
  -/
  simp only [h, zero_re']
  /-
    🎉 no goals
  -/


/-- Pythagorean theorem, vector inner product form. -/
theorem norm_add_sq_eq_norm_sq_add_norm_sq_real {x y : F} (h : ⟪x, y⟫_ℝ = 0) :
    ‖x + y‖ * ‖x + y‖ = ‖x‖ * ‖x‖ + ‖y‖ * ‖y‖ :=
  (norm_add_sq_eq_norm_sq_add_norm_sq_iff_real_inner_eq_zero x y).2 h


/-- Pythagorean theorem, subtracting vectors, if-and-only-if vector
inner product form. -/
theorem norm_sub_sq_eq_norm_sq_add_norm_sq_iff_real_inner_eq_zero (x y : F) :
    ‖x - y‖ * ‖x - y‖ = ‖x‖ * ‖x‖ + ‖y‖ * ‖y‖ ↔ ⟪x, y⟫_ℝ = 0 := by
  rw [@norm_sub_mul_self ℝ, add_right_cancel_iff, sub_eq_add_neg, add_right_eq_self, neg_eq_zero,
    mul_eq_zero]
  /-
    F : Type u_3
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    x y : F
    ⊢ Iff (Or (Eq 2 0) (Eq (RCLike.re (Inner.inner x y)) 0)) (Eq (Inner.inner x y) …
  -/
  norm_num
  /-
    🎉 no goals
  -/


/-- Pythagorean theorem, subtracting vectors, if-and-if vector inner product form using square
roots. -/
theorem norm_sub_eq_sqrt_iff_real_inner_eq_zero {x y : F} :
    ‖x - y‖ = √(‖x‖ * ‖x‖ + ‖y‖ * ‖y‖) ↔ ⟪x, y⟫_ℝ = 0 := by
  rw [← norm_sub_sq_eq_norm_sq_add_norm_sq_iff_real_inner_eq_zero, eq_comm, sqrt_eq_iff_mul_self_eq,
                 /-
                   case hx
                   F : Type u_3
                   inst✝¹ : SeminormedAddCommGroup F
                   inst✝ : InnerProductSpace Real F
                   x y : F
                   ⊢ LE.le 0 (HAdd.hAdd (HMul.hMul (Norm.norm x) (Norm.norm x)) (HMul.hMul (Norm. …
                 -/
                 /-
                   🎉 no goals
                 -/
    eq_comm] <;> positivity
                 /-
                   🎉 no goals
                 -/


/-- Pythagorean theorem, subtracting vectors, vector inner product
form. -/
theorem norm_sub_sq_eq_norm_sq_add_norm_sq_real {x y : F} (h : ⟪x, y⟫_ℝ = 0) :
    ‖x - y‖ * ‖x - y‖ = ‖x‖ * ‖x‖ + ‖y‖ * ‖y‖ :=
  (norm_sub_sq_eq_norm_sq_add_norm_sq_iff_real_inner_eq_zero x y).2 h


/-- The sum and difference of two vectors are orthogonal if and only
if they have the same norm. -/
theorem real_inner_add_sub_eq_zero_iff (x y : F) : ⟪x + y, x - y⟫_ℝ = 0 ↔ ‖x‖ = ‖y‖ := by
  /-
    F : Type u_3
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    x y : F
    ⊢ Iff (Eq (Inner.inner (HAdd.hAdd x y) (HSub.hSub x y)) 0) (Eq (Norm.norm x) ( …
  -/
  conv_rhs => rw [← mul_self_inj_of_nonneg (norm_nonneg _) (norm_nonneg _)]
  simp only [← @inner_self_eq_norm_mul_norm ℝ, inner_add_left, inner_sub_right, real_inner_comm y x,
    sub_eq_zero, re_to_real]
  /-
    F : Type u_3
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    x y : F
    ⊢ Iff (Eq (HAdd.hAdd (Inner.inner x x) (Inner.inner y x)) (HAdd.hAdd (Inner.in …
  -/
  constructor
    /-
      case mp
      F : Type u_3
      inst✝¹ : SeminormedAddCommGroup F
      inst✝ : InnerProductSpace Real F
      x y : F
      ⊢ Eq (HAdd.hAdd (Inner.inner x x) (Inner.inner y x)) (HAdd.hAdd (Inner.inner y …
    -/
  · intro h
    /-
      case mp
      F : Type u_3
      inst✝¹ : SeminormedAddCommGroup F
      inst✝ : InnerProductSpace Real F
      x y : F
      h : Eq (HAdd.hAdd (Inner.inner x x) (Inner.inner y x)) (HAdd.hAdd (Inner.inner …
      ⊢ Eq (Inner.inner x x) (Inner.inner y y)
    -/
    rw [add_comm] at h
    /-
      case mp
      F : Type u_3
      inst✝¹ : SeminormedAddCommGroup F
      inst✝ : InnerProductSpace Real F
      x y : F
      h : Eq (HAdd.hAdd (Inner.inner y x) (Inner.inner x x)) (HAdd.hAdd (Inner.inner …
      ⊢ Eq (Inner.inner x x) (Inner.inner y y)
    -/
    linarith
    /-
      🎉 no goals
    -/
    /-
      case mpr
      F : Type u_3
      inst✝¹ : SeminormedAddCommGroup F
      inst✝ : InnerProductSpace Real F
      x y : F
      ⊢ Eq (Inner.inner x x) (Inner.inner y y) → Eq (HAdd.hAdd (Inner.inner x x) (In …
    -/
  · intro h
    /-
      case mpr
      F : Type u_3
      inst✝¹ : SeminormedAddCommGroup F
      inst✝ : InnerProductSpace Real F
      x y : F
      h : Eq (Inner.inner x x) (Inner.inner y y)
      ⊢ Eq (HAdd.hAdd (Inner.inner x x) (Inner.inner y x)) (HAdd.hAdd (Inner.inner y …
    -/
    linarith
    /-
      🎉 no goals
    -/


/-- Given two orthogonal vectors, their sum and difference have equal norms. -/
theorem norm_sub_eq_norm_add {v w : E} (h : ⟪v, w⟫ = 0) : ‖w - v‖ = ‖w + v‖ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    v w : E
    h : Eq (Inner.inner v w) 0
    ⊢ Eq (Norm.norm (HSub.hSub w v)) (Norm.norm (HAdd.hAdd w v))
  -/
  rw [← mul_self_inj_of_nonneg (norm_nonneg _) (norm_nonneg _)]
  simp only [h, ← @inner_self_eq_norm_mul_norm 𝕜, sub_neg_eq_add, sub_zero, map_sub, zero_re',
    zero_sub, add_zero, map_add, inner_add_right, inner_sub_left, inner_sub_right, inner_re_symm,
    zero_add]


/-- The real inner product of two vectors, divided by the product of their
norms, has absolute value at most 1. -/
theorem abs_real_inner_div_norm_mul_norm_le_one (x y : F) : |⟪x, y⟫_ℝ / (‖x‖ * ‖y‖)| ≤ 1 := by
  /-
    F : Type u_3
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    x y : F
    ⊢ LE.le (abs (HDiv.hDiv (Inner.inner x y) (HMul.hMul (Norm.norm x) (Norm.norm  …
  -/
  rw [abs_div, abs_mul, abs_norm, abs_norm]
  /-
    F : Type u_3
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    x y : F
    ⊢ LE.le (HDiv.hDiv (abs (Inner.inner x y)) (HMul.hMul (Norm.norm x) (Norm.norm …
  -/
  exact div_le_one_of_le₀ (abs_real_inner_le_norm x y) (by positivity)
  /-
    🎉 no goals
  -/


/-- The inner product of a vector with a multiple of itself. -/
theorem real_inner_smul_self_left (x : F) (r : ℝ) : ⟪r • x, x⟫_ℝ = r * (‖x‖ * ‖x‖) := by
  /-
    F : Type u_3
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    x : F
    r : Real
    ⊢ Eq (Inner.inner (HSMul.hSMul r x) x) (HMul.hMul r (HMul.hMul (Norm.norm x) ( …
  -/
  rw [real_inner_smul_left, ← real_inner_self_eq_norm_mul_norm]
  /-
    🎉 no goals
  -/


/-- The inner product of a vector with a multiple of itself. -/
theorem real_inner_smul_self_right (x : F) (r : ℝ) : ⟪x, r • x⟫_ℝ = r * (‖x‖ * ‖x‖) := by
  /-
    F : Type u_3
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    x : F
    r : Real
    ⊢ Eq (Inner.inner x (HSMul.hSMul r x)) (HMul.hMul r (HMul.hMul (Norm.norm x) ( …
  -/
  rw [inner_smul_right, ← real_inner_self_eq_norm_mul_norm]
  /-
    🎉 no goals
  -/


/-- When an inner product space `E` over `𝕜` is considered as a real normed space, its inner
product satisfies `IsBoundedBilinearMap`.

In order to state these results, we need a `NormedSpace ℝ E` instance. We will later establish
such an instance by restriction-of-scalars, `InnerProductSpace.rclikeToReal 𝕜 E`, but this
instance may be not definitionally equal to some other “natural” instance. So, we assume
`[NormedSpace ℝ E]`.
-/
theorem _root_.isBoundedBilinearMap_inner [NormedSpace ℝ E] [IsScalarTower ℝ 𝕜 E] :
    IsBoundedBilinearMap ℝ fun p : E × E => ⟪p.1, p.2⟫ :=
  { add_left := inner_add_left
    smul_left := fun r x y => by
      /-
        𝕜 : Type u_1
        E : Type u_2
        inst✝⁴ : RCLike 𝕜
        inst✝³ : SeminormedAddCommGroup E
        inst✝² : InnerProductSpace 𝕜 E
        inst✝¹ : NormedSpace Real E
        inst✝ : IsScalarTower Real 𝕜 E
        r : Real
        x y : E
        ⊢ Eq (Inner.inner { fst := HSMul.hSMul r x, snd := y }.1 { fst := HSMul.hSMul  …
      -/
      simp only [← algebraMap_smul 𝕜 r x, algebraMap_eq_ofReal, inner_smul_real_left]
      /-
        🎉 no goals
      -/
    add_right := inner_add_right
    smul_right := fun r x y => by
      /-
        𝕜 : Type u_1
        E : Type u_2
        inst✝⁴ : RCLike 𝕜
        inst✝³ : SeminormedAddCommGroup E
        inst✝² : InnerProductSpace 𝕜 E
        inst✝¹ : NormedSpace Real E
        inst✝ : IsScalarTower Real 𝕜 E
        r : Real
        x y : E
        ⊢ Eq (Inner.inner { fst := x, snd := HSMul.hSMul r y }.1 { fst := x, snd := HS …
      -/
      simp only [← algebraMap_smul 𝕜 r y, algebraMap_eq_ofReal, inner_smul_real_right]
      /-
        🎉 no goals
      -/
    bound :=
      ⟨1, zero_lt_one, fun x y => by
        /-
          𝕜 : Type u_1
          E : Type u_2
          inst✝⁴ : RCLike 𝕜
          inst✝³ : SeminormedAddCommGroup E
          inst✝² : InnerProductSpace 𝕜 E
          inst✝¹ : NormedSpace Real E
          inst✝ : IsScalarTower Real 𝕜 E
          x y : E
          ⊢ LE.le (Norm.norm (Inner.inner { fst := x, snd := y }.1 { fst := x, snd := y  …
        -/
        rw [one_mul]
        /-
          𝕜 : Type u_1
          E : Type u_2
          inst✝⁴ : RCLike 𝕜
          inst✝³ : SeminormedAddCommGroup E
          inst✝² : InnerProductSpace 𝕜 E
          inst✝¹ : NormedSpace Real E
          inst✝ : IsScalarTower Real 𝕜 E
          x y : E
          ⊢ LE.le (Norm.norm (Inner.inner { fst := x, snd := y }.1 { fst := x, snd := y  …
        -/
        exact norm_inner_le_norm x y⟩ }
        /-
          🎉 no goals
        -/


/-- The inner product of two weighted sums, where the weights in each
sum add to 0, in terms of the norms of pairwise differences. -/
theorem inner_sum_smul_sum_smul_of_sum_eq_zero {ι₁ : Type*} {s₁ : Finset ι₁} {w₁ : ι₁ → ℝ}
    (v₁ : ι₁ → F) (h₁ : ∑ i ∈ s₁, w₁ i = 0) {ι₂ : Type*} {s₂ : Finset ι₂} {w₂ : ι₂ → ℝ}
    (v₂ : ι₂ → F) (h₂ : ∑ i ∈ s₂, w₂ i = 0) :
    ⟪∑ i₁ ∈ s₁, w₁ i₁ • v₁ i₁, ∑ i₂ ∈ s₂, w₂ i₂ • v₂ i₂⟫_ℝ =
      (-∑ i₁ ∈ s₁, ∑ i₂ ∈ s₂, w₁ i₁ * w₂ i₂ * (‖v₁ i₁ - v₂ i₂‖ * ‖v₁ i₁ - v₂ i₂‖)) / 2 := by
  simp_rw [sum_inner, inner_sum, real_inner_smul_left, real_inner_smul_right,
    real_inner_eq_norm_mul_self_add_norm_mul_self_sub_norm_sub_mul_self_div_two, ← div_sub_div_same,
    ← div_add_div_same, mul_sub_left_distrib, left_distrib, Finset.sum_sub_distrib,
    Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.sum_mul, h₁, h₂, zero_mul,
    mul_zero, Finset.sum_const_zero, zero_add, zero_sub, Finset.mul_sum, neg_div,
    Finset.sum_div, mul_div_assoc, mul_assoc]


/-- Formula for the distance between the images of two nonzero points under an inversion with center
zero. See also `EuclideanGeometry.dist_inversion_inversion` for inversions around a general
point. -/
theorem dist_div_norm_sq_smul {x y : F} (hx : x ≠ 0) (hy : y ≠ 0) (R : ℝ) :
    dist ((R / ‖x‖) ^ 2 • x) ((R / ‖y‖) ^ 2 • y) = R ^ 2 / (‖x‖ * ‖y‖) * dist x y :=
  calc
    dist ((R / ‖x‖) ^ 2 • x) ((R / ‖y‖) ^ 2 • y) =
        √(‖(R / ‖x‖) ^ 2 • x - (R / ‖y‖) ^ 2 • y‖ ^ 2) := by
      /-
        F : Type u_3
        inst✝¹ : NormedAddCommGroup F
        inst✝ : InnerProductSpace Real F
        x y : F
        hx : Ne x 0
        hy : Ne y 0
        R : Real
        ⊢ Eq (Dist.dist (HSMul.hSMul (HPow.hPow (HDiv.hDiv R (Norm.norm x)) 2) x) (HSM …
      -/
      rw [dist_eq_norm, sqrt_sq (norm_nonneg _)]
      /-
        🎉 no goals
      -/
    _ = √((R ^ 2 / (‖x‖ * ‖y‖)) ^ 2 * ‖x - y‖ ^ 2) :=
      congr_arg sqrt <| by
        field_simp [sq, norm_sub_mul_self_real, norm_smul, real_inner_smul_left, inner_smul_right,
          Real.norm_of_nonneg (mul_self_nonneg _)]
        /-
          F : Type u_3
          inst✝¹ : NormedAddCommGroup F
          inst✝ : InnerProductSpace Real F
          x y : F
          hx : Ne x 0
          hy : Ne y 0
          R : Real
          ⊢ Eq (HMul.hMul (HAdd.hAdd (HMul.hMul (HSub.hSub (HMul.hMul (HMul.hMul (HMul.h …
        -/
        ring
        /-
          🎉 no goals
        -/
    _ = R ^ 2 / (‖x‖ * ‖y‖) * dist x y := by
      /-
        F : Type u_3
        inst✝¹ : NormedAddCommGroup F
        inst✝ : InnerProductSpace Real F
        x y : F
        hx : Ne x 0
        hy : Ne y 0
        R : Real
        ⊢ Eq (HMul.hMul (HPow.hPow (HDiv.hDiv (HPow.hPow R 2) (HMul.hMul (Norm.norm x) …
      -/
                                                        /-
                                                          🎉 no goals
                                                        -/
                                                        /-
                                                          🎉 no goals
                                                        -/
      rw [sqrt_mul, sqrt_sq, sqrt_sq, dist_eq_norm] <;> positivity
                                                        /-
                                                          🎉 no goals
                                                        -/


/-- The inner product of a nonzero vector with a nonzero multiple of
itself, divided by the product of their norms, has absolute value
1. -/
theorem norm_inner_div_norm_mul_norm_eq_one_of_ne_zero_of_ne_zero_mul {x : E} {r : 𝕜} (hx : x ≠ 0)
    (hr : r ≠ 0) : ‖⟪x, r • x⟫‖ / (‖x‖ * ‖r • x‖) = 1 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x : E
    r : 𝕜
    hx : Ne x 0
    hr : Ne r 0
    ⊢ Eq (HDiv.hDiv (Norm.norm (Inner.inner x (HSMul.hSMul r x))) (HMul.hMul (Norm …
  -/
  have hx' : ‖x‖ ≠ 0 := by simp [hx]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x : E
    r : 𝕜
    hx : Ne x 0
    hr : Ne r 0
    hx' : Ne (Norm.norm x) 0
    ⊢ Eq (HDiv.hDiv (Norm.norm (Inner.inner x (HSMul.hSMul r x))) (HMul.hMul (Norm …
  -/
  have hr' : ‖r‖ ≠ 0 := by simp [hr]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x : E
    r : 𝕜
    hx : Ne x 0
    hr : Ne r 0
    hx' : Ne (Norm.norm x) 0
    hr' : Ne (Norm.norm r) 0
    ⊢ Eq (HDiv.hDiv (Norm.norm (Inner.inner x (HSMul.hSMul r x))) (HMul.hMul (Norm …
  -/
  rw [inner_smul_right, norm_mul, ← inner_self_re_eq_norm, inner_self_eq_norm_mul_norm, norm_smul]
  rw [← mul_assoc, ← div_div, mul_div_cancel_right₀ _ hx', ← div_div, mul_comm,
    mul_div_cancel_right₀ _ hr', div_self hx']


/-- The inner product of a nonzero vector with a nonzero multiple of
itself, divided by the product of their norms, has absolute value
1. -/
theorem abs_real_inner_div_norm_mul_norm_eq_one_of_ne_zero_of_ne_zero_mul {x : F} {r : ℝ}
    (hx : x ≠ 0) (hr : r ≠ 0) : |⟪x, r • x⟫_ℝ| / (‖x‖ * ‖r • x‖) = 1 :=
  norm_inner_div_norm_mul_norm_eq_one_of_ne_zero_of_ne_zero_mul hx hr


/-- The inner product of a nonzero vector with a positive multiple of
itself, divided by the product of their norms, has value 1. -/
theorem real_inner_div_norm_mul_norm_eq_one_of_ne_zero_of_pos_mul {x : F} {r : ℝ} (hx : x ≠ 0)
    (hr : 0 < r) : ⟪x, r • x⟫_ℝ / (‖x‖ * ‖r • x‖) = 1 := by
  rw [real_inner_smul_self_right, norm_smul, Real.norm_eq_abs, ← mul_assoc ‖x‖, mul_comm _ |r|,
    mul_assoc, abs_of_nonneg hr.le, div_self]
  /-
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    x : F
    r : Real
    hx : Ne x 0
    hr : LT.lt 0 r
    ⊢ Ne (HMul.hMul r (HMul.hMul (Norm.norm x) (Norm.norm x))) 0
  -/
  exact mul_ne_zero hr.ne' (mul_self_ne_zero.2 (norm_ne_zero_iff.2 hx))
  /-
    🎉 no goals
  -/


/-- The inner product of a nonzero vector with a negative multiple of
itself, divided by the product of their norms, has value -1. -/
theorem real_inner_div_norm_mul_norm_eq_neg_one_of_ne_zero_of_neg_mul {x : F} {r : ℝ} (hx : x ≠ 0)
    (hr : r < 0) : ⟪x, r • x⟫_ℝ / (‖x‖ * ‖r • x‖) = -1 := by
  rw [real_inner_smul_self_right, norm_smul, Real.norm_eq_abs, ← mul_assoc ‖x‖, mul_comm _ |r|,
    mul_assoc, abs_of_neg hr, neg_mul, div_neg_eq_neg_div, div_self]
  /-
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    x : F
    r : Real
    hx : Ne x 0
    hr : LT.lt r 0
    ⊢ Ne (HMul.hMul r (HMul.hMul (Norm.norm x) (Norm.norm x))) 0
  -/
  exact mul_ne_zero hr.ne (mul_self_ne_zero.2 (norm_ne_zero_iff.2 hx))
  /-
    🎉 no goals
  -/


theorem norm_inner_eq_norm_tfae (x y : E) :
    List.TFAE [‖⟪x, y⟫‖ = ‖x‖ * ‖y‖,
      x = 0 ∨ y = (⟪x, y⟫ / ⟪x, x⟫) • x,
      x = 0 ∨ ∃ r : 𝕜, y = r • x,
      x = 0 ∨ y ∈ 𝕜 ∙ x] := by
  tfae_have 1 → 2 := by
    refine fun h => or_iff_not_imp_left.2 fun hx₀ => ?_
    have : ‖x‖ ^ 2 ≠ 0 := pow_ne_zero _ (norm_ne_zero_iff.2 hx₀)
    rw [← sq_eq_sq₀, mul_pow, ← mul_right_inj' this, eq_comm, ← sub_eq_zero, ← mul_sub] at h <;>
      try positivity
    simp only [@norm_sq_eq_inner 𝕜] at h
    letI : InnerProductSpace.Core 𝕜 E := InnerProductSpace.toCore
    erw [← InnerProductSpace.Core.cauchy_schwarz_aux (𝕜 := 𝕜) (F := E),
      InnerProductSpace.Core.normSq_eq_zero, sub_eq_zero] at h
    rw [div_eq_inv_mul, mul_smul, h, inv_smul_smul₀]
    rwa [inner_self_ne_zero]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    tfae_1_to_2 : Eq (Norm.norm (Inner.inner x y)) (HMul.hMul (Norm.norm x) (Norm. …
    ⊢ (List.cons (Eq (Norm.norm (Inner.inner x y)) (HMul.hMul (Norm.norm x) (Norm. …
  -/
  tfae_have 2 → 3 := fun h => h.imp_right fun h' => ⟨_, h'⟩
  tfae_have 3 → 1 := by
    rintro (rfl | ⟨r, rfl⟩) <;>
    simp [inner_smul_right, norm_smul, inner_self_eq_norm_sq_to_K, inner_self_eq_norm_mul_norm,
      sq, mul_left_comm]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    tfae_1_to_2 : Eq (Norm.norm (Inner.inner x y)) (HMul.hMul (Norm.norm x) (Norm. …
    tfae_2_to_3 : Or (Eq x 0) (Eq y (HSMul.hSMul (HDiv.hDiv (Inner.inner x y) (Inn …
    tfae_3_to_1 : Or (Eq x 0) (Exists fun r => Eq y (HSMul.hSMul r x)) → Eq (Norm. …
    ⊢ (List.cons (Eq (Norm.norm (Inner.inner x y)) (HMul.hMul (Norm.norm x) (Norm. …
  -/
  tfae_have 3 ↔ 4 := by simp only [Submodule.mem_span_singleton, eq_comm]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    tfae_1_to_2 : Eq (Norm.norm (Inner.inner x y)) (HMul.hMul (Norm.norm x) (Norm. …
    tfae_2_to_3 : Or (Eq x 0) (Eq y (HSMul.hSMul (HDiv.hDiv (Inner.inner x y) (Inn …
    tfae_3_to_1 : Or (Eq x 0) (Exists fun r => Eq y (HSMul.hSMul r x)) → Eq (Norm. …
    tfae_3_iff_4 : Iff (Or (Eq x 0) (Exists fun r => Eq y (HSMul.hSMul r x))) (Or  …
    ⊢ (List.cons (Eq (Norm.norm (Inner.inner x y)) (HMul.hMul (Norm.norm x) (Norm. …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


/-- If the inner product of two vectors is equal to the product of their norms, then the two vectors
are multiples of each other. One form of the equality case for Cauchy-Schwarz.
Compare `inner_eq_norm_mul_iff`, which takes the stronger hypothesis `⟪x, y⟫ = ‖x‖ * ‖y‖`. -/
theorem norm_inner_eq_norm_iff {x y : E} (hx₀ : x ≠ 0) (hy₀ : y ≠ 0) :
    ‖⟪x, y⟫‖ = ‖x‖ * ‖y‖ ↔ ∃ r : 𝕜, r ≠ 0 ∧ y = r • x :=
  calc
    ‖⟪x, y⟫‖ = ‖x‖ * ‖y‖ ↔ x = 0 ∨ ∃ r : 𝕜, y = r • x :=
      /-
        𝕜 : Type u_1
        E : Type u_2
        inst✝² : RCLike 𝕜
        inst✝¹ : NormedAddCommGroup E
        inst✝ : InnerProductSpace 𝕜 E
        x y : E
        hx₀ : Ne x 0
        hy₀ : Ne y 0
        ⊢ Eq ((List.cons (Eq (Norm.norm (Inner.inner x y)) (HMul.hMul (Norm.norm x) (N …
      -/
      /-
        🎉 no goals
      -/
      (@norm_inner_eq_norm_tfae 𝕜 _ _ _ _ x y).out 0 2
      /-
        🎉 no goals
      -/
    _ ↔ ∃ r : 𝕜, y = r • x := or_iff_right hx₀
    _ ↔ ∃ r : 𝕜, r ≠ 0 ∧ y = r • x :=
      ⟨fun ⟨r, h⟩ => ⟨r, fun hr₀ => hy₀ <| h.symm ▸ smul_eq_zero.2 <| Or.inl hr₀, h⟩,
        fun ⟨r, _hr₀, h⟩ => ⟨r, h⟩⟩


/-- The inner product of two vectors, divided by the product of their
norms, has absolute value 1 if and only if they are nonzero and one is
a multiple of the other. One form of equality case for Cauchy-Schwarz. -/
theorem norm_inner_div_norm_mul_norm_eq_one_iff (x y : E) :
    ‖⟪x, y⟫ / (‖x‖ * ‖y‖)‖ = 1 ↔ x ≠ 0 ∧ ∃ r : 𝕜, r ≠ 0 ∧ y = r • x := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    ⊢ Iff (Eq (Norm.norm (HDiv.hDiv (Inner.inner x y) (HMul.hMul ↑(Norm.norm x) ↑( …
  -/
  constructor
    /-
      case mp
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      x y : E
      ⊢ Eq (Norm.norm (HDiv.hDiv (Inner.inner x y) (HMul.hMul ↑(Norm.norm x) ↑(Norm. …
    -/
  · intro h
    /-
      case mp
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      x y : E
      h : Eq (Norm.norm (HDiv.hDiv (Inner.inner x y) (HMul.hMul ↑(Norm.norm x) ↑(Nor …
      ⊢ And (Ne x 0) (Exists fun r => And (Ne r 0) (Eq y (HSMul.hSMul r x)))
    -/
    have hx₀ : x ≠ 0 := fun h₀ => by simp [h₀] at h
    /-
      case mp
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      x y : E
      h : Eq (Norm.norm (HDiv.hDiv (Inner.inner x y) (HMul.hMul ↑(Norm.norm x) ↑(Nor …
      hx₀ : Ne x 0
      ⊢ And (Ne x 0) (Exists fun r => And (Ne r 0) (Eq y (HSMul.hSMul r x)))
    -/
    have hy₀ : y ≠ 0 := fun h₀ => by simp [h₀] at h
    /-
      case mp
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      x y : E
      h : Eq (Norm.norm (HDiv.hDiv (Inner.inner x y) (HMul.hMul ↑(Norm.norm x) ↑(Nor …
      hx₀ : Ne x 0
      hy₀ : Ne y 0
      ⊢ And (Ne x 0) (Exists fun r => And (Ne r 0) (Eq y (HSMul.hSMul r x)))
    -/
    refine ⟨hx₀, (norm_inner_eq_norm_iff hx₀ hy₀).1 <| eq_of_div_eq_one ?_⟩
    /-
      case mp
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      x y : E
      h : Eq (Norm.norm (HDiv.hDiv (Inner.inner x y) (HMul.hMul ↑(Norm.norm x) ↑(Nor …
      hx₀ : Ne x 0
      hy₀ : Ne y 0
      ⊢ Eq (HDiv.hDiv (Norm.norm (Inner.inner x y)) (HMul.hMul (Norm.norm x) (Norm.n …
    -/
    simpa using h
    /-
      🎉 no goals
    -/
    /-
      case mpr
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      x y : E
      ⊢ And (Ne x 0) (Exists fun r => And (Ne r 0) (Eq y (HSMul.hSMul r x))) → Eq (N …
    -/
  · rintro ⟨hx, ⟨r, ⟨hr, rfl⟩⟩⟩
    /-
      case mpr.intro.intro.intro
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      x : E
      hx : Ne x 0
      r : 𝕜
      hr : Ne r 0
      ⊢ Eq (Norm.norm (HDiv.hDiv (Inner.inner x (HSMul.hSMul r x)) (HMul.hMul ↑(Norm …
    -/
    simp only [norm_div, norm_mul, norm_ofReal, abs_norm]
    /-
      case mpr.intro.intro.intro
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      x : E
      hx : Ne x 0
      r : 𝕜
      hr : Ne r 0
      ⊢ Eq (HDiv.hDiv (Norm.norm (Inner.inner x (HSMul.hSMul r x))) (HMul.hMul (Norm …
    -/
    exact norm_inner_div_norm_mul_norm_eq_one_of_ne_zero_of_ne_zero_mul hx hr
    /-
      🎉 no goals
    -/


/-- The inner product of two vectors, divided by the product of their
norms, has absolute value 1 if and only if they are nonzero and one is
a multiple of the other. One form of equality case for Cauchy-Schwarz. -/
theorem abs_real_inner_div_norm_mul_norm_eq_one_iff (x y : F) :
    |⟪x, y⟫_ℝ / (‖x‖ * ‖y‖)| = 1 ↔ x ≠ 0 ∧ ∃ r : ℝ, r ≠ 0 ∧ y = r • x :=
  @norm_inner_div_norm_mul_norm_eq_one_iff ℝ F _ _ _ x y


theorem inner_eq_norm_mul_iff_div {x y : E} (h₀ : x ≠ 0) :
    ⟪x, y⟫ = (‖x‖ : 𝕜) * ‖y‖ ↔ (‖y‖ / ‖x‖ : 𝕜) • x = y := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    h₀ : Ne x 0
    ⊢ Iff (Eq (Inner.inner x y) (HMul.hMul ↑(Norm.norm x) ↑(Norm.norm y))) (Eq (HS …
  -/
  have h₀' := h₀
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    h₀ h₀' : Ne x 0
    ⊢ Iff (Eq (Inner.inner x y) (HMul.hMul ↑(Norm.norm x) ↑(Norm.norm y))) (Eq (HS …
  -/
  rw [← norm_ne_zero_iff, Ne, ← @ofReal_eq_zero 𝕜] at h₀'
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    h₀ : Ne x 0
    h₀' : Not (Eq (↑(Norm.norm x)) 0)
    ⊢ Iff (Eq (Inner.inner x y) (HMul.hMul ↑(Norm.norm x) ↑(Norm.norm y))) (Eq (HS …
  -/
  constructor <;> intro h
  · have : x = 0 ∨ y = (⟪x, y⟫ / ⟪x, x⟫ : 𝕜) • x :=
      ((@norm_inner_eq_norm_tfae 𝕜 _ _ _ _ x y).out 0 1).1 (by simp [h])
    /-
      case mp
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      x y : E
      h₀ : Ne x 0
      h₀' : Not (Eq (↑(Norm.norm x)) 0)
      h : Eq (Inner.inner x y) (HMul.hMul ↑(Norm.norm x) ↑(Norm.norm y))
      this : Or (Eq x 0) (Eq y (HSMul.hSMul (HDiv.hDiv (Inner.inner x y) (Inner.inne …
      ⊢ Eq (HSMul.hSMul (HDiv.hDiv ↑(Norm.norm y) ↑(Norm.norm x)) x) y
    -/
    rw [this.resolve_left h₀, h]
    /-
      case mp
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      x y : E
      h₀ : Ne x 0
      h₀' : Not (Eq (↑(Norm.norm x)) 0)
      h : Eq (Inner.inner x y) (HMul.hMul ↑(Norm.norm x) ↑(Norm.norm y))
      this : Or (Eq x 0) (Eq y (HSMul.hSMul (HDiv.hDiv (Inner.inner x y) (Inner.inne …
      ⊢ Eq (HSMul.hSMul (HDiv.hDiv ↑(Norm.norm (HSMul.hSMul (HDiv.hDiv (HMul.hMul ↑( …
    -/
    simp [norm_smul, inner_self_ofReal_norm, mul_div_cancel_right₀ _ h₀']
    /-
      🎉 no goals
    -/
    /-
      case mpr
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      x y : E
      h₀ : Ne x 0
      h₀' : Not (Eq (↑(Norm.norm x)) 0)
      h : Eq (HSMul.hSMul (HDiv.hDiv ↑(Norm.norm y) ↑(Norm.norm x)) x) y
      ⊢ Eq (Inner.inner x y) (HMul.hMul ↑(Norm.norm x) ↑(Norm.norm y))
    -/
  · conv_lhs => rw [← h, inner_smul_right, inner_self_eq_norm_sq_to_K]
    /-
      case mpr
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      x y : E
      h₀ : Ne x 0
      h₀' : Not (Eq (↑(Norm.norm x)) 0)
      h : Eq (HSMul.hSMul (HDiv.hDiv ↑(Norm.norm y) ↑(Norm.norm x)) x) y
      ⊢ Eq (HMul.hMul (HDiv.hDiv ↑(Norm.norm y) ↑(Norm.norm x)) (HPow.hPow (↑(Norm.n …
    -/
    field_simp [sq, mul_left_comm]
    /-
      🎉 no goals
    -/


/-- If the inner product of two vectors is equal to the product of their norms (i.e.,
`⟪x, y⟫ = ‖x‖ * ‖y‖`), then the two vectors are nonnegative real multiples of each other. One form
of the equality case for Cauchy-Schwarz.
Compare `norm_inner_eq_norm_iff`, which takes the weaker hypothesis `abs ⟪x, y⟫ = ‖x‖ * ‖y‖`. -/
theorem inner_eq_norm_mul_iff {x y : E} :
    ⟪x, y⟫ = (‖x‖ : 𝕜) * ‖y‖ ↔ (‖y‖ : 𝕜) • x = (‖x‖ : 𝕜) • y := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    ⊢ Iff (Eq (Inner.inner x y) (HMul.hMul ↑(Norm.norm x) ↑(Norm.norm y))) (Eq (HS …
  -/
  rcases eq_or_ne x 0 with (rfl | h₀)
    /-
      case inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      y : E
      ⊢ Iff (Eq (Inner.inner 0 y) (HMul.hMul ↑(Norm.norm 0) ↑(Norm.norm y))) (Eq (HS …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      x y : E
      h₀ : Ne x 0
      ⊢ Iff (Eq (Inner.inner x y) (HMul.hMul ↑(Norm.norm x) ↑(Norm.norm y))) (Eq (HS …
    -/
  · rw [inner_eq_norm_mul_iff_div h₀, div_eq_inv_mul, mul_smul, inv_smul_eq_iff₀]
    /-
      case inr.ha
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      x y : E
      h₀ : Ne x 0
      ⊢ Ne (↑(Norm.norm x)) 0
    -/
    rwa [Ne, ofReal_eq_zero, norm_eq_zero]
    /-
      🎉 no goals
    -/


/-- If the inner product of two vectors is equal to the product of their norms (i.e.,
`⟪x, y⟫ = ‖x‖ * ‖y‖`), then the two vectors are nonnegative real multiples of each other. One form
of the equality case for Cauchy-Schwarz.
Compare `norm_inner_eq_norm_iff`, which takes the weaker hypothesis `abs ⟪x, y⟫ = ‖x‖ * ‖y‖`. -/
theorem inner_eq_norm_mul_iff_real {x y : F} : ⟪x, y⟫_ℝ = ‖x‖ * ‖y‖ ↔ ‖y‖ • x = ‖x‖ • y :=
  inner_eq_norm_mul_iff


/-- The inner product of two vectors, divided by the product of their
norms, has value 1 if and only if they are nonzero and one is
a positive multiple of the other. -/
theorem real_inner_div_norm_mul_norm_eq_one_iff (x y : F) :
    ⟪x, y⟫_ℝ / (‖x‖ * ‖y‖) = 1 ↔ x ≠ 0 ∧ ∃ r : ℝ, 0 < r ∧ y = r • x := by
  /-
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    x y : F
    ⊢ Iff (Eq (HDiv.hDiv (Inner.inner x y) (HMul.hMul (Norm.norm x) (Norm.norm y)) …
  -/
  constructor
    /-
      case mp
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : InnerProductSpace Real F
      x y : F
      ⊢ Eq (HDiv.hDiv (Inner.inner x y) (HMul.hMul (Norm.norm x) (Norm.norm y))) 1 → …
    -/
  · intro h
    /-
      case mp
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : InnerProductSpace Real F
      x y : F
      h : Eq (HDiv.hDiv (Inner.inner x y) (HMul.hMul (Norm.norm x) (Norm.norm y))) 1
      ⊢ And (Ne x 0) (Exists fun r => And (LT.lt 0 r) (Eq y (HSMul.hSMul r x)))
    -/
    have hx₀ : x ≠ 0 := fun h₀ => by simp [h₀] at h
    /-
      case mp
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : InnerProductSpace Real F
      x y : F
      h : Eq (HDiv.hDiv (Inner.inner x y) (HMul.hMul (Norm.norm x) (Norm.norm y))) 1
      hx₀ : Ne x 0
      ⊢ And (Ne x 0) (Exists fun r => And (LT.lt 0 r) (Eq y (HSMul.hSMul r x)))
    -/
    have hy₀ : y ≠ 0 := fun h₀ => by simp [h₀] at h
    /-
      case mp
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : InnerProductSpace Real F
      x y : F
      h : Eq (HDiv.hDiv (Inner.inner x y) (HMul.hMul (Norm.norm x) (Norm.norm y))) 1
      hx₀ : Ne x 0
      hy₀ : Ne y 0
      ⊢ And (Ne x 0) (Exists fun r => And (LT.lt 0 r) (Eq y (HSMul.hSMul r x)))
    -/
    refine ⟨hx₀, ‖y‖ / ‖x‖, div_pos (norm_pos_iff.2 hy₀) (norm_pos_iff.2 hx₀), ?_⟩
    /-
      case mp
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : InnerProductSpace Real F
      x y : F
      h : Eq (HDiv.hDiv (Inner.inner x y) (HMul.hMul (Norm.norm x) (Norm.norm y))) 1
      hx₀ : Ne x 0
      hy₀ : Ne y 0
      ⊢ Eq y (HSMul.hSMul (HDiv.hDiv (Norm.norm y) (Norm.norm x)) x)
    -/
    exact ((inner_eq_norm_mul_iff_div hx₀).1 (eq_of_div_eq_one h)).symm
    /-
      🎉 no goals
    -/
    /-
      case mpr
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : InnerProductSpace Real F
      x y : F
      ⊢ And (Ne x 0) (Exists fun r => And (LT.lt 0 r) (Eq y (HSMul.hSMul r x))) → Eq …
    -/
  · rintro ⟨hx, ⟨r, ⟨hr, rfl⟩⟩⟩
    /-
      case mpr.intro.intro.intro
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : InnerProductSpace Real F
      x : F
      hx : Ne x 0
      r : Real
      hr : LT.lt 0 r
      ⊢ Eq (HDiv.hDiv (Inner.inner x (HSMul.hSMul r x)) (HMul.hMul (Norm.norm x) (No …
    -/
    exact real_inner_div_norm_mul_norm_eq_one_of_ne_zero_of_pos_mul hx hr
    /-
      🎉 no goals
    -/


/-- The inner product of two vectors, divided by the product of their
norms, has value -1 if and only if they are nonzero and one is
a negative multiple of the other. -/
theorem real_inner_div_norm_mul_norm_eq_neg_one_iff (x y : F) :
    ⟪x, y⟫_ℝ / (‖x‖ * ‖y‖) = -1 ↔ x ≠ 0 ∧ ∃ r : ℝ, r < 0 ∧ y = r • x := by
  rw [← neg_eq_iff_eq_neg, ← neg_div, ← inner_neg_right, ← norm_neg y,
    real_inner_div_norm_mul_norm_eq_one_iff, (@neg_surjective ℝ _).exists]
  /-
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    x y : F
    ⊢ Iff (And (Ne x 0) (Exists fun x_1 => And (LT.lt 0 (Neg.neg x_1)) (Eq (Neg.ne …
  -/
  refine Iff.rfl.and (exists_congr fun r => ?_)
  /-
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : InnerProductSpace Real F
    x y : F
    r : Real
    ⊢ Iff (And (LT.lt 0 (Neg.neg r)) (Eq (Neg.neg y) (HSMul.hSMul (Neg.neg r) x))) …
  -/
  rw [neg_pos, neg_smul, neg_inj]
  /-
    🎉 no goals
  -/


/-- If the inner product of two unit vectors is `1`, then the two vectors are equal. One form of
the equality case for Cauchy-Schwarz. -/
theorem inner_eq_one_iff_of_norm_one {x y : E} (hx : ‖x‖ = 1) (hy : ‖y‖ = 1) :
    ⟪x, y⟫ = 1 ↔ x = y := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    hx : Eq (Norm.norm x) 1
    hy : Eq (Norm.norm y) 1
    ⊢ Iff (Eq (Inner.inner x y) 1) (Eq x y)
  -/
                                                              /-
                                                                🎉 no goals
                                                              -/
                                                              /-
                                                                🎉 no goals
                                                              -/
  convert inner_eq_norm_mul_iff (𝕜 := 𝕜) (E := E) using 2 <;> simp [hx, hy]
                                                              /-
                                                                🎉 no goals
                                                              -/


theorem inner_lt_norm_mul_iff_real {x y : F} : ⟪x, y⟫_ℝ < ‖x‖ * ‖y‖ ↔ ‖y‖ • x ≠ ‖x‖ • y :=
  calc
    ⟪x, y⟫_ℝ < ‖x‖ * ‖y‖ ↔ ⟪x, y⟫_ℝ ≠ ‖x‖ * ‖y‖ :=
      ⟨ne_of_lt, lt_of_le_of_ne (real_inner_le_norm _ _)⟩
    _ ↔ ‖y‖ • x ≠ ‖x‖ • y := not_congr inner_eq_norm_mul_iff_real


/-- If the inner product of two unit vectors is strictly less than `1`, then the two vectors are
distinct. One form of the equality case for Cauchy-Schwarz. -/
theorem inner_lt_one_iff_real_of_norm_one {x y : F} (hx : ‖x‖ = 1) (hy : ‖y‖ = 1) :
                               /-
                                 F : Type u_3
                                 inst✝¹ : NormedAddCommGroup F
                                 inst✝ : InnerProductSpace Real F
                                 x y : F
                                 hx : Eq (Norm.norm x) 1
                                 hy : Eq (Norm.norm y) 1
                                 ⊢ Iff (LT.lt (Inner.inner x y) 1) (Ne x y)
                               -/
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
    ⟪x, y⟫_ℝ < 1 ↔ x ≠ y := by convert inner_lt_norm_mul_iff_real (F := F) <;> simp [hx, hy]
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


/-- The sphere of radius `r = ‖y‖` is tangent to the plane `⟪x, y⟫ = ‖y‖ ^ 2` at `x = y`. -/
theorem eq_of_norm_le_re_inner_eq_norm_sq {x y : E} (hle : ‖x‖ ≤ ‖y‖) (h : re ⟪x, y⟫ = ‖y‖ ^ 2) :
    x = y := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    hle : LE.le (Norm.norm x) (Norm.norm y)
    h : Eq (RCLike.re (Inner.inner x y)) (HPow.hPow (Norm.norm y) 2)
    ⊢ Eq x y
  -/
  suffices H : re ⟪x - y, x - y⟫ ≤ 0 by rwa [inner_self_nonpos, sub_eq_zero] at H
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    hle : LE.le (Norm.norm x) (Norm.norm y)
    h : Eq (RCLike.re (Inner.inner x y)) (HPow.hPow (Norm.norm y) 2)
    ⊢ LE.le (RCLike.re (Inner.inner (HSub.hSub x y) (HSub.hSub x y))) 0
  -/
  have H₁ : ‖x‖ ^ 2 ≤ ‖y‖ ^ 2 := by gcongr
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    hle : LE.le (Norm.norm x) (Norm.norm y)
    h : Eq (RCLike.re (Inner.inner x y)) (HPow.hPow (Norm.norm y) 2)
    H₁ : LE.le (HPow.hPow (Norm.norm x) 2) (HPow.hPow (Norm.norm y) 2)
    ⊢ LE.le (RCLike.re (Inner.inner (HSub.hSub x y) (HSub.hSub x y))) 0
  -/
  have H₂ : re ⟪y, x⟫ = ‖y‖ ^ 2 := by rwa [← inner_conj_symm, conj_re]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x y : E
    hle : LE.le (Norm.norm x) (Norm.norm y)
    h : Eq (RCLike.re (Inner.inner x y)) (HPow.hPow (Norm.norm y) 2)
    H₁ : LE.le (HPow.hPow (Norm.norm x) 2) (HPow.hPow (Norm.norm y) 2)
    H₂ : Eq (RCLike.re (Inner.inner y x)) (HPow.hPow (Norm.norm y) 2)
    ⊢ LE.le (RCLike.re (Inner.inner (HSub.hSub x y) (HSub.hSub x y))) 0
  -/
  simpa [inner_sub_left, inner_sub_right, ← norm_sq_eq_inner, h, H₂] using H₁
  /-
    🎉 no goals
  -/


/-- A field `𝕜` satisfying `RCLike` is itself a `𝕜`-inner product space. -/
instance RCLike.innerProductSpace : InnerProductSpace 𝕜 𝕜 where
  inner x y := conj x * y
                           /-
                             𝕜 : Type u_1
                             E : Type u_2
                             F : Type u_3
                             inst✝ : RCLike 𝕜
                             x : 𝕜
                             ⊢ Eq (HPow.hPow (Norm.norm x) 2) (RCLike.re (Inner.inner x x))
                           -/
  norm_sq_eq_inner x := by simp only [inner, conj_mul, ← ofReal_pow, ofReal_re]
                           /-
                             🎉 no goals
                           -/
                      /-
                        𝕜 : Type u_1
                        E : Type u_2
                        F : Type u_3
                        inst✝ : RCLike 𝕜
                        x y : 𝕜
                        ⊢ Eq ((starRingEnd 𝕜) (Inner.inner y x)) (Inner.inner x y)
                      -/
  conj_symm x y := by simp only [mul_comm, map_mul, starRingEnd_self_apply]
                      /-
                        🎉 no goals
                      -/
                       /-
                         𝕜 : Type u_1
                         E : Type u_2
                         F : Type u_3
                         inst✝ : RCLike 𝕜
                         x y z : 𝕜
                         ⊢ Eq (Inner.inner (HAdd.hAdd x y) z) (HAdd.hAdd (Inner.inner x z) (Inner.inner …
                       -/
  add_left x y z := by simp only [add_mul, map_add]
                       /-
                         🎉 no goals
                       -/
                        /-
                          𝕜 : Type u_1
                          E : Type u_2
                          F : Type u_3
                          inst✝ : RCLike 𝕜
                          x y z : 𝕜
                          ⊢ Eq (Inner.inner (HSMul.hSMul z x) y) (HMul.hMul ((starRingEnd 𝕜) z) (Inner.i …
                        -/
  smul_left x y z := by simp only [mul_assoc, smul_eq_mul, map_mul]
                        /-
                          🎉 no goals
                        -/


@[simp]
theorem RCLike.inner_apply (x y : 𝕜) : ⟪x, y⟫ = conj x * y :=
  rfl


/-- A general inner product implies a real inner product. This is not registered as an instance
since `𝕜` does not appear in the return type `Inner ℝ E`. -/
def Inner.rclikeToReal : Inner ℝ E where inner x y := re ⟪x, y⟫


/-- A general inner product space structure implies a real inner product structure.

This is not registered as an instance since
* `𝕜` does not appear in the return type `InnerProductSpace ℝ E`,
* It is likely to create instance diamonds, as it builds upon the diamond-prone
  `NormedSpace.restrictScalars`.

However, it can be used in a proof to obtain a real inner product space structure from a given
`𝕜`-inner product space structure. -/
-- See note [reducible non instances]
abbrev InnerProductSpace.rclikeToReal : InnerProductSpace ℝ E :=
  { Inner.rclikeToReal 𝕜 E,
    NormedSpace.restrictScalars ℝ 𝕜
      E with
    norm_sq_eq_inner := norm_sq_eq_inner
    conj_symm := fun _ _ => inner_re_symm _ _
    add_left := fun x y z => by
      /-
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝² : RCLike 𝕜
        G : Type u_4
        inst✝¹ : SeminormedAddCommGroup E
        inst✝ : InnerProductSpace 𝕜 E
        x y z : E
        ⊢ Eq (Inner.inner (HAdd.hAdd x y) z) (HAdd.hAdd (Inner.inner x z) (Inner.inner …
      -/
      change re ⟪x + y, z⟫ = re ⟪x, z⟫ + re ⟪y, z⟫
      /-
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝² : RCLike 𝕜
        G : Type u_4
        inst✝¹ : SeminormedAddCommGroup E
        inst✝ : InnerProductSpace 𝕜 E
        x y z : E
        ⊢ Eq (RCLike.re (Inner.inner (HAdd.hAdd x y) z)) (HAdd.hAdd (RCLike.re (Inner. …
      -/
      simp only [inner_add_left, map_add]
      /-
        🎉 no goals
      -/
    smul_left := fun x y r => by
      /-
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝² : RCLike 𝕜
        G : Type u_4
        inst✝¹ : SeminormedAddCommGroup E
        inst✝ : InnerProductSpace 𝕜 E
        x y : E
        r : Real
        ⊢ Eq (Inner.inner (HSMul.hSMul r x) y) (HMul.hMul ((starRingEnd Real) r) (Inne …
      -/
      change re ⟪(r : 𝕜) • x, y⟫ = r * re ⟪x, y⟫
      /-
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝² : RCLike 𝕜
        G : Type u_4
        inst✝¹ : SeminormedAddCommGroup E
        inst✝ : InnerProductSpace 𝕜 E
        x y : E
        r : Real
        ⊢ Eq (RCLike.re (Inner.inner (HSMul.hSMul (↑r) x) y)) (HMul.hMul r (RCLike.re  …
      -/
      simp only [inner_smul_left, conj_ofReal, re_ofReal_mul] }
      /-
        🎉 no goals
      -/


theorem real_inner_eq_re_inner (x y : E) :
    @Inner.inner ℝ E (Inner.rclikeToReal 𝕜 E) x y = re ⟪x, y⟫ :=
  rfl


theorem real_inner_I_smul_self (x : E) :
    @Inner.inner ℝ E (Inner.rclikeToReal 𝕜 E) x ((I : 𝕜) • x) = 0 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    x : E
    ⊢ Eq (Inner.inner x (HSMul.hSMul RCLike.I x)) 0
  -/
  simp [real_inner_eq_re_inner 𝕜, inner_smul_right]
  /-
    🎉 no goals
  -/


/-- A complex inner product implies a real inner product. This cannot be an instance since it
creates a diamond with `PiLp.innerProductSpace` because `re (sum i, inner (x i) (y i))` and
`sum i, re (inner (x i) (y i))` are not defeq. -/
def InnerProductSpace.complexToReal [SeminormedAddCommGroup G] [InnerProductSpace ℂ G] :
    InnerProductSpace ℝ G :=
  InnerProductSpace.rclikeToReal ℂ G


instance : InnerProductSpace ℝ ℂ := InnerProductSpace.complexToReal


@[simp]
protected theorem Complex.inner (w z : ℂ) : ⟪w, z⟫_ℝ = (conj w * z).re :=
  rfl


/-- An `RCLike` field is a real inner product space. -/
noncomputable instance RCLike.toInnerProductSpaceReal : InnerProductSpace ℝ 𝕜 where
  __ := Inner.rclikeToReal 𝕜 𝕜
  norm_sq_eq_inner := norm_sq_eq_inner
  conj_symm x y := inner_re_symm ..
  add_left x y z :=
                                                 /-
                                                   𝕜 : Type u_1
                                                   E : Type u_2
                                                   F : Type u_3
                                                   inst✝ : RCLike 𝕜
                                                   x y z : 𝕜
                                                   ⊢ Eq (RCLike.re (HMul.hMul ((starRingEnd 𝕜) (HAdd.hAdd x y)) z)) (HAdd.hAdd (R …
                                                 -/
    show re (_ * _) = re (_ * _) + re (_ * _) by simp only [map_add, mul_re, conj_re, conj_im]; ring
                                                                                                /-
                                                                                                  🎉 no goals
                                                                                                -/
  smul_left x y r :=
    show re (_ * _) = _ * re (_ * _) by
      /-
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝ : RCLike 𝕜
        x y : 𝕜
        r : Real
        ⊢ Eq (RCLike.re (HMul.hMul ((starRingEnd 𝕜) (HSMul.hSMul r x)) y)) (HMul.hMul  …
      -/
      simp only [mul_re, conj_re, conj_im, conj_trivial, smul_re, smul_im]; ring
                                                                            /-
                                                                              🎉 no goals
                                                                            -/

-- The instance above does not create diamonds for concrete `𝕜`:

theorem continuous_inner : Continuous fun p : E × E => ⟪p.1, p.2⟫ :=
  letI : InnerProductSpace ℝ E := InnerProductSpace.rclikeToReal 𝕜 E
  letI : IsScalarTower ℝ 𝕜 E := RestrictScalars.isScalarTower _ _ _
  isBoundedBilinearMap_inner.continuous


theorem Filter.Tendsto.inner {f g : α → E} {l : Filter α} {x y : E} (hf : Tendsto f l (𝓝 x))
    (hg : Tendsto g l (𝓝 y)) : Tendsto (fun t => ⟪f t, g t⟫) l (𝓝 ⟪x, y⟫) :=
  (continuous_inner.tendsto _).comp (hf.prod_mk_nhds hg)


theorem ContinuousWithinAt.inner (hf : ContinuousWithinAt f s x) (hg : ContinuousWithinAt g s x) :
    ContinuousWithinAt (fun t => ⟪f t, g t⟫) s x :=
  Filter.Tendsto.inner hf hg


theorem ContinuousAt.inner (hf : ContinuousAt f x) (hg : ContinuousAt g x) :
    ContinuousAt (fun t => ⟪f t, g t⟫) x :=
  Filter.Tendsto.inner hf hg


theorem ContinuousOn.inner (hf : ContinuousOn f s) (hg : ContinuousOn g s) :
    ContinuousOn (fun t => ⟪f t, g t⟫) s := fun x hx => (hf x hx).inner (hg x hx)


@[continuity]
theorem Continuous.inner (hf : Continuous f) (hg : Continuous g) : Continuous fun t => ⟪f t, g t⟫ :=
  continuous_iff_continuousAt.2 fun _x => hf.continuousAt.inner hg.continuousAt


