local notation "𝓚" => algebraMap ℝ _


/--
This typeclass captures properties shared by ℝ and ℂ, with an API that closely matches that of ℂ.
-/
class RCLike (K : semiOutParam Type*) extends DenselyNormedField K, StarRing K,
    NormedAlgebra ℝ K, CompleteSpace K where
  re : K →+ ℝ
  im : K →+ ℝ
  /-- Imaginary unit in `K`. Meant to be set to `0` for `K = ℝ`. -/
  I : K
  I_re_ax : re I = 0
  I_mul_I_ax : I = 0 ∨ I * I = -1
  re_add_im_ax : ∀ z : K, 𝓚 (re z) + 𝓚 (im z) * I = z
  ofReal_re_ax : ∀ r : ℝ, re (𝓚 r) = r
  ofReal_im_ax : ∀ r : ℝ, im (𝓚 r) = 0
  mul_re_ax : ∀ z w : K, re (z * w) = re z * re w - im z * im w
  mul_im_ax : ∀ z w : K, im (z * w) = re z * im w + im z * re w
  conj_re_ax : ∀ z : K, re (conj z) = re z
  conj_im_ax : ∀ z : K, im (conj z) = -im z
  conj_I_ax : conj I = -I
  norm_sq_eq_def_ax : ∀ z : K, ‖z‖ ^ 2 = re z * re z + im z * im z
  mul_im_I_ax : ∀ z : K, im z * im I = im z
  /-- only an instance in the `ComplexOrder` locale -/
  [toPartialOrder : PartialOrder K]
  le_iff_re_im {z w : K} : z ≤ w ↔ re z ≤ re w ∧ im z = im w
  -- note we cannot put this in the `extends` clause
  [toDecidableEq : DecidableEq K]


/-- Coercion from `ℝ` to an `RCLike` field. -/
@[coe] abbrev ofReal : ℝ → K := Algebra.cast

/- The priority must be set at 900 to ensure that coercions are tried in the right order.
See Note [coercion into rings], or `Mathlib/Data/Nat/Cast/Basic.lean` for more details. -/

noncomputable instance (priority := 900) algebraMapCoe : CoeTC ℝ K :=
  ⟨ofReal⟩


theorem ofReal_alg (x : ℝ) : (x : K) = x • (1 : K) :=
  Algebra.algebraMap_eq_smul_one x


theorem real_smul_eq_coe_mul (r : ℝ) (z : K) : r • z = (r : K) * z :=
  Algebra.smul_def r z


theorem real_smul_eq_coe_smul [AddCommGroup E] [Module K E] [Module ℝ E] [IsScalarTower ℝ K E]
                                                /-
                                                  K : Type u_1
                                                  E : Type u_2
                                                  inst✝⁴ : RCLike K
                                                  inst✝³ : AddCommGroup E
                                                  inst✝² : Module K E
                                                  inst✝¹ : Module Real E
                                                  inst✝ : IsScalarTower Real K E
                                                  r : Real
                                                  x : E
                                                  ⊢ Eq (HSMul.hSMul r x) (HSMul.hSMul (↑r) x)
                                                -/
    (r : ℝ) (x : E) : r • x = (r : K) • x := by rw [RCLike.ofReal_alg, smul_one_smul]
                                                /-
                                                  🎉 no goals
                                                -/


theorem algebraMap_eq_ofReal : ⇑(algebraMap ℝ K) = ofReal :=
  rfl


@[simp, rclike_simps]
theorem re_add_im (z : K) : (re z : K) + im z * I = z :=
  RCLike.re_add_im_ax z


@[simp, norm_cast, rclike_simps]
theorem ofReal_re : ∀ r : ℝ, re (r : K) = r :=
  RCLike.ofReal_re_ax


@[simp, norm_cast, rclike_simps]
theorem ofReal_im : ∀ r : ℝ, im (r : K) = 0 :=
  RCLike.ofReal_im_ax


@[simp, rclike_simps]
theorem mul_re : ∀ z w : K, re (z * w) = re z * re w - im z * im w :=
  RCLike.mul_re_ax


@[simp, rclike_simps]
theorem mul_im : ∀ z w : K, im (z * w) = re z * im w + im z * re w :=
  RCLike.mul_im_ax


theorem ext_iff {z w : K} : z = w ↔ re z = re w ∧ im z = im w :=
  ⟨fun h => h ▸ ⟨rfl, rfl⟩, fun ⟨h₁, h₂⟩ => re_add_im z ▸ re_add_im w ▸ h₁ ▸ h₂ ▸ rfl⟩


theorem ext {z w : K} (hre : re z = re w) (him : im z = im w) : z = w :=
  ext_iff.2 ⟨hre, him⟩


@[norm_cast]
theorem ofReal_zero : ((0 : ℝ) : K) = 0 :=
  algebraMap.coe_zero


@[rclike_simps]
theorem zero_re' : re (0 : K) = (0 : ℝ) :=
  map_zero re


@[norm_cast]
theorem ofReal_one : ((1 : ℝ) : K) = 1 :=
  map_one (algebraMap ℝ K)


@[simp, rclike_simps]
                                      /-
                                        K : Type u_1
                                        inst✝ : RCLike K
                                        ⊢ Eq (RCLike.re 1) 1
                                      -/
theorem one_re : re (1 : K) = 1 := by rw [← ofReal_one, ofReal_re]
                                      /-
                                        🎉 no goals
                                      -/


@[simp, rclike_simps]
                                      /-
                                        K : Type u_1
                                        inst✝ : RCLike K
                                        ⊢ Eq (RCLike.im 1) 0
                                      -/
theorem one_im : im (1 : K) = 0 := by rw [← ofReal_one, ofReal_im]
                                      /-
                                        🎉 no goals
                                      -/


theorem ofReal_injective : Function.Injective ((↑) : ℝ → K) :=
  (algebraMap ℝ K).injective


@[norm_cast]
theorem ofReal_inj {z w : ℝ} : (z : K) = (w : K) ↔ z = w :=
  algebraMap.coe_inj

-- replaced by `RCLike.ofNat_re`
-- replaced by `RCLike.ofNat_im`


theorem ofReal_eq_zero {x : ℝ} : (x : K) = 0 ↔ x = 0 :=
  algebraMap.lift_map_eq_zero_iff x


theorem ofReal_ne_zero {x : ℝ} : (x : K) ≠ 0 ↔ x ≠ 0 :=
  ofReal_eq_zero.not


@[rclike_simps, norm_cast]
theorem ofReal_add (r s : ℝ) : ((r + s : ℝ) : K) = r + s :=
  algebraMap.coe_add _ _

-- replaced by `RCLike.ofReal_ofNat`


@[rclike_simps, norm_cast]
theorem ofReal_neg (r : ℝ) : ((-r : ℝ) : K) = -r :=
  algebraMap.coe_neg r


@[rclike_simps, norm_cast]
theorem ofReal_sub (r s : ℝ) : ((r - s : ℝ) : K) = r - s :=
  map_sub (algebraMap ℝ K) r s


@[rclike_simps, norm_cast]
theorem ofReal_sum {α : Type*} (s : Finset α) (f : α → ℝ) :
    ((∑ i ∈ s, f i : ℝ) : K) = ∑ i ∈ s, (f i : K) :=
  map_sum (algebraMap ℝ K) _ _


@[simp, rclike_simps, norm_cast]
theorem ofReal_finsupp_sum {α M : Type*} [Zero M] (f : α →₀ M) (g : α → M → ℝ) :
    ((f.sum fun a b => g a b : ℝ) : K) = f.sum fun a b => (g a b : K) :=
  map_finsupp_sum (algebraMap ℝ K) f g


@[rclike_simps, norm_cast]
theorem ofReal_mul (r s : ℝ) : ((r * s : ℝ) : K) = r * s :=
  algebraMap.coe_mul _ _


@[rclike_simps, norm_cast]
theorem ofReal_pow (r : ℝ) (n : ℕ) : ((r ^ n : ℝ) : K) = (r : K) ^ n :=
  map_pow (algebraMap ℝ K) r n


@[rclike_simps, norm_cast]
theorem ofReal_prod {α : Type*} (s : Finset α) (f : α → ℝ) :
    ((∏ i ∈ s, f i : ℝ) : K) = ∏ i ∈ s, (f i : K) :=
  map_prod (algebraMap ℝ K) _ _


@[simp, rclike_simps, norm_cast]
theorem ofReal_finsupp_prod {α M : Type*} [Zero M] (f : α →₀ M) (g : α → M → ℝ) :
    ((f.prod fun a b => g a b : ℝ) : K) = f.prod fun a b => (g a b : K) :=
  map_finsupp_prod _ f g


@[simp, norm_cast, rclike_simps]
theorem real_smul_ofReal (r x : ℝ) : r • (x : K) = (r : K) * (x : K) :=
  real_smul_eq_coe_mul _ _


@[rclike_simps]
theorem re_ofReal_mul (r : ℝ) (z : K) : re (↑r * z) = r * re z := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    r : Real
    z : K
    ⊢ Eq (RCLike.re (HMul.hMul (↑r) z)) (HMul.hMul r (RCLike.re z))
  -/
  simp only [mul_re, ofReal_im, zero_mul, ofReal_re, sub_zero]
  /-
    🎉 no goals
  -/


@[rclike_simps]
theorem im_ofReal_mul (r : ℝ) (z : K) : im (↑r * z) = r * im z := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    r : Real
    z : K
    ⊢ Eq (RCLike.im (HMul.hMul (↑r) z)) (HMul.hMul r (RCLike.im z))
  -/
  simp only [add_zero, ofReal_im, zero_mul, ofReal_re, mul_im]
  /-
    🎉 no goals
  -/


@[rclike_simps]
theorem smul_re (r : ℝ) (z : K) : re (r • z) = r * re z := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    r : Real
    z : K
    ⊢ Eq (RCLike.re (HSMul.hSMul r z)) (HMul.hMul r (RCLike.re z))
  -/
  rw [real_smul_eq_coe_mul, re_ofReal_mul]
  /-
    🎉 no goals
  -/


@[rclike_simps]
theorem smul_im (r : ℝ) (z : K) : im (r • z) = r * im z := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    r : Real
    z : K
    ⊢ Eq (RCLike.im (HSMul.hSMul r z)) (HMul.hMul r (RCLike.im z))
  -/
  rw [real_smul_eq_coe_mul, im_ofReal_mul]
  /-
    🎉 no goals
  -/


@[rclike_simps, norm_cast]
theorem norm_ofReal (r : ℝ) : ‖(r : K)‖ = |r| :=
  norm_algebraMap' K r


/-- ℝ and ℂ are both of characteristic zero. -/
instance (priority := 100) charZero_rclike : CharZero K :=
  (RingHom.charZero_iff (algebraMap ℝ K).injective).1 inferInstance


@[rclike_simps, norm_cast]
lemma ofReal_expect {α : Type*} (s : Finset α) (f : α → ℝ) : 𝔼 i ∈ s, f i = 𝔼 i ∈ s, (f i : K) :=
  map_expect (algebraMap ..) ..


@[norm_cast]
lemma ofReal_balance {ι : Type*} [Fintype ι] (f : ι → ℝ) (i : ι) :
    ((balance f i : ℝ) : K) = balance ((↑) ∘ f) i := map_balance (algebraMap ..) ..


@[simp] lemma ofReal_comp_balance {ι : Type*} [Fintype ι] (f : ι → ℝ) :
    ofReal ∘ balance f = balance (ofReal ∘ f : ι → K) := funext <| ofReal_balance _


/-- The imaginary unit. -/
@[simp, rclike_simps]
theorem I_re : re (I : K) = 0 :=
  I_re_ax


@[simp, rclike_simps]
theorem I_im (z : K) : im z * im (I : K) = im z :=
  mul_im_I_ax z


@[simp, rclike_simps]
                                                       /-
                                                         K : Type u_1
                                                         inst✝ : RCLike K
                                                         z : K
                                                         ⊢ Eq (HMul.hMul (RCLike.im RCLike.I) (RCLike.im z)) (RCLike.im z)
                                                       -/
theorem I_im' (z : K) : im (I : K) * im z = im z := by rw [mul_comm, I_im]
                                                       /-
                                                         🎉 no goals
                                                       -/


@[rclike_simps] -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11119): was `simp`
theorem I_mul_re (z : K) : re (I * z) = -im z := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    z : K
    ⊢ Eq (RCLike.re (HMul.hMul RCLike.I z)) (Neg.neg (RCLike.im z))
  -/
  simp only [I_re, zero_sub, I_im', zero_mul, mul_re]
  /-
    🎉 no goals
  -/


theorem I_mul_I : (I : K) = 0 ∨ (I : K) * I = -1 :=
  I_mul_I_ax


variable (𝕜) in
lemma I_eq_zero_or_im_I_eq_one : (I : K) = 0 ∨ im (I : K) = 1 :=
                                           /-
                                             K : Type u_1
                                             inst✝ : RCLike K
                                             h : Eq (HMul.hMul RCLike.I RCLike.I) (-1)
                                             ⊢ Eq (RCLike.im RCLike.I) 1
                                           -/
  I_mul_I (K := K) |>.imp_right fun h ↦ by simpa [h] using (I_mul_re (I : K)).symm
                                           /-
                                             🎉 no goals
                                           -/


@[simp, rclike_simps]
theorem conj_re (z : K) : re (conj z) = re z :=
  RCLike.conj_re_ax z


@[simp, rclike_simps]
theorem conj_im (z : K) : im (conj z) = -im z :=
  RCLike.conj_im_ax z


@[simp, rclike_simps]
theorem conj_I : conj (I : K) = -I :=
  RCLike.conj_I_ax


@[simp, rclike_simps]
theorem conj_ofReal (r : ℝ) : conj (r : K) = (r : K) := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    r : Real
    ⊢ Eq ((starRingEnd K) ↑r) ↑r
  -/
  rw [ext_iff]
  /-
    K : Type u_1
    inst✝ : RCLike K
    r : Real
    ⊢ And (Eq (RCLike.re ((starRingEnd K) ↑r)) (RCLike.re ↑r)) (Eq (RCLike.im ((st …
  -/
  simp only [ofReal_im, conj_im, eq_self_iff_true, conj_re, and_self_iff, neg_zero]
  /-
    🎉 no goals
  -/

-- replaced by `RCLike.conj_ofNat`


theorem conj_nat_cast (n : ℕ) : conj (n : K) = n := map_natCast _ _


theorem conj_ofNat (n : ℕ) [n.AtLeastTwo] : conj (ofNat(n) : K) = ofNat(n) :=
  map_ofNat _ _


@[rclike_simps, simp]
                                               /-
                                                 K : Type u_1
                                                 inst✝ : RCLike K
                                                 ⊢ Eq ((starRingEnd K) (Neg.neg RCLike.I)) RCLike.I
                                               -/
theorem conj_neg_I : conj (-I) = (I : K) := by rw [map_neg, conj_I, neg_neg]
                                               /-
                                                 🎉 no goals
                                               -/


theorem conj_eq_re_sub_im (z : K) : conj z = re z - im z * I :=
  (congr_arg conj (re_add_im z).symm).trans <| by
    /-
      K : Type u_1
      inst✝ : RCLike K
      z : K
      ⊢ Eq ((starRingEnd K) (HAdd.hAdd (↑(RCLike.re z)) (HMul.hMul (↑(RCLike.im z))  …
    -/
    rw [map_add, map_mul, conj_I, conj_ofReal, conj_ofReal, mul_neg, sub_eq_add_neg]
    /-
      🎉 no goals
    -/


theorem sub_conj (z : K) : z - conj z = 2 * im z * I :=
  calc
                                                           /-
                                                             K : Type u_1
                                                             inst✝ : RCLike K
                                                             z : K
                                                             ⊢ Eq (HSub.hSub z ((starRingEnd K) z)) (HSub.hSub (HAdd.hAdd (↑(RCLike.re z))  …
                                                           -/
    z - conj z = re z + im z * I - (re z - im z * I) := by rw [re_add_im, ← conj_eq_re_sub_im]
                                                           /-
                                                             🎉 no goals
                                                           -/
                           /-
                             K : Type u_1
                             inst✝ : RCLike K
                             z : K
                             ⊢ Eq (HSub.hSub (HAdd.hAdd (↑(RCLike.re z)) (HMul.hMul (↑(RCLike.im z)) RCLike …
                           -/
    _ = 2 * im z * I := by rw [add_sub_sub_cancel, ← two_mul, mul_assoc]
                           /-
                             🎉 no goals
                           -/


@[rclike_simps]
theorem conj_smul (r : ℝ) (z : K) : conj (r • z) = r • conj z := by
  rw [conj_eq_re_sub_im, conj_eq_re_sub_im, smul_re, smul_im, ofReal_mul, ofReal_mul,
    real_smul_eq_coe_mul r (_ - _), mul_sub, mul_assoc]


theorem add_conj (z : K) : z + conj z = 2 * re z :=
  calc
                                                           /-
                                                             K : Type u_1
                                                             inst✝ : RCLike K
                                                             z : K
                                                             ⊢ Eq (HAdd.hAdd z ((starRingEnd K) z)) (HAdd.hAdd (HAdd.hAdd (↑(RCLike.re z))  …
                                                           -/
    z + conj z = re z + im z * I + (re z - im z * I) := by rw [re_add_im, conj_eq_re_sub_im]
                                                           /-
                                                             🎉 no goals
                                                           -/
                       /-
                         K : Type u_1
                         inst✝ : RCLike K
                         z : K
                         ⊢ Eq (HAdd.hAdd (HAdd.hAdd (↑(RCLike.re z)) (HMul.hMul (↑(RCLike.im z)) RCLike …
                       -/
    _ = 2 * re z := by rw [add_add_sub_cancel, two_mul]
                       /-
                         🎉 no goals
                       -/


theorem re_eq_add_conj (z : K) : ↑(re z) = (z + conj z) / 2 := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    z : K
    ⊢ Eq (↑(RCLike.re z)) (HDiv.hDiv (HAdd.hAdd z ((starRingEnd K) z)) 2)
  -/
  rw [add_conj, mul_div_cancel_left₀ (re z : K) two_ne_zero]
  /-
    🎉 no goals
  -/


theorem im_eq_conj_sub (z : K) : ↑(im z) = I * (conj z - z) / 2 := by
  rw [← neg_inj, ← ofReal_neg, ← I_mul_re, re_eq_add_conj, map_mul, conj_I, ← neg_div, ← mul_neg,
    neg_sub, mul_sub, neg_mul, sub_eq_add_neg]


open List in
/-- There are several equivalent ways to say that a number `z` is in fact a real number. -/
theorem is_real_TFAE (z : K) : TFAE [conj z = z, ∃ r : ℝ, (r : K) = z, ↑(re z) = z, im z = 0] := by
  tfae_have 1 → 4
  | h => by
    rw [← @ofReal_inj K, im_eq_conj_sub, h, sub_self, mul_zero, zero_div,
      ofReal_zero]
  tfae_have 4 → 3
  | h => by
    conv_rhs => rw [← re_add_im z, h, ofReal_zero, zero_mul, add_zero]
  /-
    K : Type u_1
    inst✝ : RCLike K
    z : K
    tfae_1_to_4 : Eq ((starRingEnd K) z) z → Eq (RCLike.im z) 0
    tfae_4_to_3 : Eq (RCLike.im z) 0 → Eq (↑(RCLike.re z)) z
    ⊢ (List.cons (Eq ((starRingEnd K) z) z) (List.cons (Exists fun r => Eq (↑r) z) …
  -/
  tfae_have 3 → 2 := fun h => ⟨_, h⟩
  /-
    K : Type u_1
    inst✝ : RCLike K
    z : K
    tfae_1_to_4 : Eq ((starRingEnd K) z) z → Eq (RCLike.im z) 0
    tfae_4_to_3 : Eq (RCLike.im z) 0 → Eq (↑(RCLike.re z)) z
    tfae_3_to_2 : Eq (↑(RCLike.re z)) z → Exists fun r => Eq (↑r) z
    ⊢ (List.cons (Eq ((starRingEnd K) z) z) (List.cons (Exists fun r => Eq (↑r) z) …
  -/
  tfae_have 2 → 1 := fun ⟨r, hr⟩ => hr ▸ conj_ofReal _
  /-
    K : Type u_1
    inst✝ : RCLike K
    z : K
    tfae_1_to_4 : Eq ((starRingEnd K) z) z → Eq (RCLike.im z) 0
    tfae_4_to_3 : Eq (RCLike.im z) 0 → Eq (↑(RCLike.re z)) z
    tfae_3_to_2 : Eq (↑(RCLike.re z)) z → Exists fun r => Eq (↑r) z
    tfae_2_to_1 : (Exists fun r => Eq (↑r) z) → Eq ((starRingEnd K) z) z
    ⊢ (List.cons (Eq ((starRingEnd K) z) z) (List.cons (Exists fun r => Eq (↑r) z) …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


theorem conj_eq_iff_real {z : K} : conj z = z ↔ ∃ r : ℝ, z = (r : K) :=
  calc
                                /-
                                  K : Type u_1
                                  inst✝ : RCLike K
                                  z : K
                                  ⊢ Eq ((List.cons (Eq ((starRingEnd K) z) z) (List.cons (Exists fun r => Eq (↑r …
                                -/
                                /-
                                  🎉 no goals
                                -/
    _ ↔ ∃ r : ℝ, (r : K) = z := (is_real_TFAE z).out 0 1
                                /-
                                  🎉 no goals
                                -/
                                   /-
                                     K : Type u_1
                                     inst✝ : RCLike K
                                     z : K
                                     ⊢ Iff (Exists fun r => Eq (↑r) z) (Exists fun r => Eq z ↑r)
                                   -/
    _ ↔ _                    := by simp only [eq_comm]
                                   /-
                                     🎉 no goals
                                   -/


theorem conj_eq_iff_re {z : K} : conj z = z ↔ (re z : K) = z :=
  /-
    K : Type u_1
    inst✝ : RCLike K
    z : K
    ⊢ Eq ((List.cons (Eq ((starRingEnd K) z) z) (List.cons (Exists fun r => Eq (↑r …
  -/
  /-
    🎉 no goals
  -/
  (is_real_TFAE z).out 0 2
  /-
    🎉 no goals
  -/


theorem conj_eq_iff_im {z : K} : conj z = z ↔ im z = 0 :=
  /-
    K : Type u_1
    inst✝ : RCLike K
    z : K
    ⊢ Eq ((List.cons (Eq ((starRingEnd K) z) z) (List.cons (Exists fun r => Eq (↑r …
  -/
  /-
    🎉 no goals
  -/
  (is_real_TFAE z).out 0 3
  /-
    🎉 no goals
  -/


@[simp]
theorem star_def : (Star.star : K → K) = conj :=
  rfl


/-- Conjugation as a ring equivalence. This is used to convert the inner product into a
sesquilinear product. -/
abbrev conjToRingEquiv : K ≃+* Kᵐᵒᵖ :=
  starRingEquiv


/-- The norm squared function. -/
def normSq : K →*₀ ℝ where
  toFun z := re z * re z + im z * im z
                  /-
                    K : Type u_1
                    E : Type u_2
                    inst✝ : RCLike K
                    z : K
                    ⊢ Eq ((fun z => HAdd.hAdd (HMul.hMul (RCLike.re z) (RCLike.re z)) (HMul.hMul ( …
                  -/
  map_zero' := by simp only [add_zero, mul_zero, map_zero]
                  /-
                    🎉 no goals
                  -/
                 /-
                   K : Type u_1
                   E : Type u_2
                   inst✝ : RCLike K
                   z : K
                   ⊢ Eq ({ toFun := fun z => HAdd.hAdd (HMul.hMul (RCLike.re z) (RCLike.re z)) (H …
                 -/
  map_one' := by simp only [one_im, add_zero, mul_one, one_re, mul_zero]
                 /-
                   🎉 no goals
                 -/
  map_mul' z w := by
    /-
      K : Type u_1
      E : Type u_2
      inst✝ : RCLike K
      z✝ z w : K
      ⊢ Eq ({ toFun := fun z => HAdd.hAdd (HMul.hMul (RCLike.re z) (RCLike.re z)) (H …
    -/
    simp only [mul_im, mul_re]
    /-
      K : Type u_1
      E : Type u_2
      inst✝ : RCLike K
      z✝ z w : K
      ⊢ Eq (HAdd.hAdd (HMul.hMul (HSub.hSub (HMul.hMul (RCLike.re z) (RCLike.re w))  …
    -/
    ring
    /-
      🎉 no goals
    -/


theorem normSq_apply (z : K) : normSq z = re z * re z + im z * im z :=
  rfl


theorem norm_sq_eq_def {z : K} : ‖z‖ ^ 2 = re z * re z + im z * im z :=
  norm_sq_eq_def_ax z


theorem normSq_eq_def' (z : K) : normSq z = ‖z‖ ^ 2 :=
  norm_sq_eq_def.symm


@[rclike_simps]
theorem normSq_zero : normSq (0 : K) = 0 :=
  normSq.map_zero


@[rclike_simps]
theorem normSq_one : normSq (1 : K) = 1 :=
  normSq.map_one


theorem normSq_nonneg (z : K) : 0 ≤ normSq z :=
  add_nonneg (mul_self_nonneg _) (mul_self_nonneg _)


@[rclike_simps] -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11119): was `simp`
theorem normSq_eq_zero {z : K} : normSq z = 0 ↔ z = 0 :=
  map_eq_zero _


@[simp, rclike_simps]
theorem normSq_pos {z : K} : 0 < normSq z ↔ z ≠ 0 := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    z : K
    ⊢ Iff (LT.lt 0 (RCLike.normSq z)) (Ne z 0)
  -/
  rw [lt_iff_le_and_ne, Ne, eq_comm]; simp [normSq_nonneg]
                                      /-
                                        🎉 no goals
                                      -/


@[simp, rclike_simps]
                                                          /-
                                                            K : Type u_1
                                                            inst✝ : RCLike K
                                                            z : K
                                                            ⊢ Eq (RCLike.normSq (Neg.neg z)) (RCLike.normSq z)
                                                          -/
theorem normSq_neg (z : K) : normSq (-z) = normSq z := by simp only [normSq_eq_def', norm_neg]
                                                          /-
                                                            🎉 no goals
                                                          -/


@[simp, rclike_simps]
theorem normSq_conj (z : K) : normSq (conj z) = normSq z := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    z : K
    ⊢ Eq (RCLike.normSq ((starRingEnd K) z)) (RCLike.normSq z)
  -/
  simp only [normSq_apply, neg_mul, mul_neg, neg_neg, rclike_simps]
  /-
    🎉 no goals
  -/


@[rclike_simps] -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11119): was `simp`
theorem normSq_mul (z w : K) : normSq (z * w) = normSq z * normSq w :=
  map_mul _ z w


theorem normSq_add (z w : K) : normSq (z + w) = normSq z + normSq w + 2 * re (z * conj w) := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    z w : K
    ⊢ Eq (RCLike.normSq (HAdd.hAdd z w)) (HAdd.hAdd (HAdd.hAdd (RCLike.normSq z) ( …
  -/
  simp only [normSq_apply, map_add, rclike_simps]
  /-
    K : Type u_1
    inst✝ : RCLike K
    z w : K
    ⊢ Eq (HAdd.hAdd (HMul.hMul (HAdd.hAdd (RCLike.re z) (RCLike.re w)) (HAdd.hAdd  …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem re_sq_le_normSq (z : K) : re z * re z ≤ normSq z :=
  le_add_of_nonneg_right (mul_self_nonneg _)


theorem im_sq_le_normSq (z : K) : im z * im z ≤ normSq z :=
  le_add_of_nonneg_left (mul_self_nonneg _)


theorem mul_conj (z : K) : z * conj z = ‖z‖ ^ 2 := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    z : K
    ⊢ Eq (HMul.hMul z ((starRingEnd K) z)) (HPow.hPow (↑(Norm.norm z)) 2)
  -/
                /-
                  🎉 no goals
                -/
  apply ext <;> simp [← ofReal_pow, norm_sq_eq_def, mul_comm]
                /-
                  🎉 no goals
                -/


                                                      /-
                                                        K : Type u_1
                                                        inst✝ : RCLike K
                                                        z : K
                                                        ⊢ Eq (HMul.hMul ((starRingEnd K) z) z) (HPow.hPow (↑(Norm.norm z)) 2)
                                                      -/
theorem conj_mul (z : K) : conj z * z = ‖z‖ ^ 2 := by rw [mul_comm, mul_conj]
                                                      /-
                                                        🎉 no goals
                                                      -/


lemma inv_eq_conj (hz : ‖z‖ = 1) : z⁻¹ = conj z :=
                                  /-
                                    K : Type u_1
                                    inst✝ : RCLike K
                                    z : K
                                    hz : Eq (Norm.norm z) 1
                                    ⊢ Eq (HMul.hMul ((starRingEnd K) z) z) 1
                                  -/
  inv_eq_of_mul_eq_one_left <| by simp_rw [conj_mul, hz, algebraMap.coe_one, one_pow]
                                  /-
                                    🎉 no goals
                                  -/


theorem normSq_sub (z w : K) : normSq (z - w) = normSq z + normSq w - 2 * re (z * conj w) := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    z w : K
    ⊢ Eq (RCLike.normSq (HSub.hSub z w)) (HSub.hSub (HAdd.hAdd (RCLike.normSq z) ( …
  -/
  simp only [normSq_add, sub_eq_add_neg, map_neg, mul_neg, normSq_neg, map_neg]
  /-
    🎉 no goals
  -/


theorem sqrt_normSq_eq_norm {z : K} : √(normSq z) = ‖z‖ := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    z : K
    ⊢ Eq (RCLike.normSq z).sqrt (Norm.norm z)
  -/
  rw [normSq_eq_def', Real.sqrt_sq (norm_nonneg _)]
  /-
    🎉 no goals
  -/


@[rclike_simps, norm_cast]
theorem ofReal_inv (r : ℝ) : ((r⁻¹ : ℝ) : K) = (r : K)⁻¹ :=
  map_inv₀ _ r


theorem inv_def (z : K) : z⁻¹ = conj z * ((‖z‖ ^ 2)⁻¹ : ℝ) := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    z : K
    ⊢ Eq (Inv.inv z) (HMul.hMul ((starRingEnd K) z) ↑(Inv.inv (HPow.hPow (Norm.nor …
  -/
  rcases eq_or_ne z 0 with (rfl | h₀)
    /-
      case inl
      K : Type u_1
      inst✝ : RCLike K
      ⊢ Eq (Inv.inv 0) (HMul.hMul ((starRingEnd K) 0) ↑(Inv.inv (HPow.hPow (Norm.nor …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      K : Type u_1
      inst✝ : RCLike K
      z : K
      h₀ : Ne z 0
      ⊢ Eq (Inv.inv z) (HMul.hMul ((starRingEnd K) z) ↑(Inv.inv (HPow.hPow (Norm.nor …
    -/
  · apply inv_eq_of_mul_eq_one_right
    /-
      case inr.a
      K : Type u_1
      inst✝ : RCLike K
      z : K
      h₀ : Ne z 0
      ⊢ Eq (HMul.hMul z (HMul.hMul ((starRingEnd K) z) ↑(Inv.inv (HPow.hPow (Norm.no …
    -/
    rw [← mul_assoc, mul_conj, ofReal_inv, ofReal_pow, mul_inv_cancel₀]
    /-
      case inr.a
      K : Type u_1
      inst✝ : RCLike K
      z : K
      h₀ : Ne z 0
      ⊢ Ne (HPow.hPow (↑(Norm.norm z)) 2) 0
    -/
    simpa
    /-
      🎉 no goals
    -/


@[simp, rclike_simps]
theorem inv_re (z : K) : re z⁻¹ = re z / normSq z := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    z : K
    ⊢ Eq (RCLike.re (Inv.inv z)) (HDiv.hDiv (RCLike.re z) (RCLike.normSq z))
  -/
  rw [inv_def, normSq_eq_def', mul_comm, re_ofReal_mul, conj_re, div_eq_inv_mul]
  /-
    🎉 no goals
  -/


@[simp, rclike_simps]
theorem inv_im (z : K) : im z⁻¹ = -im z / normSq z := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    z : K
    ⊢ Eq (RCLike.im (Inv.inv z)) (HDiv.hDiv (Neg.neg (RCLike.im z)) (RCLike.normSq …
  -/
  rw [inv_def, normSq_eq_def', mul_comm, im_ofReal_mul, conj_im, div_eq_inv_mul]
  /-
    🎉 no goals
  -/


theorem div_re (z w : K) : re (z / w) = re z * re w / normSq w + im z * im w / normSq w := by
  simp only [div_eq_mul_inv, mul_assoc, sub_eq_add_neg, neg_mul, mul_neg, neg_neg, map_neg,
    rclike_simps]


theorem div_im (z w : K) : im (z / w) = im z * re w / normSq w - re z * im w / normSq w := by
  simp only [div_eq_mul_inv, mul_assoc, sub_eq_add_neg, add_comm, neg_mul, mul_neg, map_neg,
    rclike_simps]


@[rclike_simps] -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11119): was `simp`
theorem conj_inv (x : K) : conj x⁻¹ = (conj x)⁻¹ :=
  star_inv₀ _


lemma conj_div (x y : K) : conj (x / y) = conj x / conj y := map_div' conj conj_inv _ _

--TODO: Do we rather want the map as an explicit definition?

lemma exists_norm_eq_mul_self (x : K) : ∃ c, ‖c‖ = 1 ∧ ↑‖x‖ = c * x := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    x : K
    ⊢ Exists fun c => And (Eq (Norm.norm c) 1) (Eq (↑(Norm.norm x)) (HMul.hMul c x))
  -/
  obtain rfl | hx := eq_or_ne x 0
    /-
      case inl
      K : Type u_1
      inst✝ : RCLike K
      ⊢ Exists fun c => And (Eq (Norm.norm c) 1) (Eq (↑(Norm.norm 0)) (HMul.hMul c 0))
    -/
  · exact ⟨1, by simp⟩
    /-
      🎉 no goals
    -/
    /-
      case inr
      K : Type u_1
      inst✝ : RCLike K
      x : K
      hx : Ne x 0
      ⊢ Exists fun c => And (Eq (Norm.norm c) 1) (Eq (↑(Norm.norm x)) (HMul.hMul c x))
    -/
  · exact ⟨‖x‖ / x, by simp [norm_ne_zero_iff.2, hx]⟩
    /-
      🎉 no goals
    -/


lemma exists_norm_mul_eq_self (x : K) : ∃ c, ‖c‖ = 1 ∧ c * ‖x‖ = x := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    x : K
    ⊢ Exists fun c => And (Eq (Norm.norm c) 1) (Eq (HMul.hMul c ↑(Norm.norm x)) x)
  -/
  obtain rfl | hx := eq_or_ne x 0
    /-
      case inl
      K : Type u_1
      inst✝ : RCLike K
      ⊢ Exists fun c => And (Eq (Norm.norm c) 1) (Eq (HMul.hMul c ↑(Norm.norm 0)) 0)
    -/
  · exact ⟨1, by simp⟩
    /-
      🎉 no goals
    -/
    /-
      case inr
      K : Type u_1
      inst✝ : RCLike K
      x : K
      hx : Ne x 0
      ⊢ Exists fun c => And (Eq (Norm.norm c) 1) (Eq (HMul.hMul c ↑(Norm.norm x)) x)
    -/
  · exact ⟨x / ‖x‖, by simp [norm_ne_zero_iff.2, hx]⟩
    /-
      🎉 no goals
    -/


@[rclike_simps, norm_cast]
theorem ofReal_div (r s : ℝ) : ((r / s : ℝ) : K) = r / s :=
  map_div₀ (algebraMap ℝ K) r s


theorem div_re_ofReal {z : K} {r : ℝ} : re (z / r) = re z / r := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    z : K
    r : Real
    ⊢ Eq (RCLike.re (HDiv.hDiv z ↑r)) (HDiv.hDiv (RCLike.re z) r)
  -/
  rw [div_eq_inv_mul, div_eq_inv_mul, ← ofReal_inv, re_ofReal_mul]
  /-
    🎉 no goals
  -/


@[rclike_simps, norm_cast]
theorem ofReal_zpow (r : ℝ) (n : ℤ) : ((r ^ n : ℝ) : K) = (r : K) ^ n :=
  map_zpow₀ (algebraMap ℝ K) r n


theorem I_mul_I_of_nonzero : (I : K) ≠ 0 → (I : K) * I = -1 :=
  I_mul_I_ax.resolve_left


@[simp, rclike_simps]
theorem inv_I : (I : K)⁻¹ = -I := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    ⊢ Eq (Inv.inv RCLike.I) (Neg.neg RCLike.I)
  -/
  by_cases h : (I : K) = 0
    /-
      case pos
      K : Type u_1
      inst✝ : RCLike K
      h : Eq RCLike.I 0
      ⊢ Eq (Inv.inv RCLike.I) (Neg.neg RCLike.I)
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
    /-
      case neg
      K : Type u_1
      inst✝ : RCLike K
      h : Not (Eq RCLike.I 0)
      ⊢ Eq (Inv.inv RCLike.I) (Neg.neg RCLike.I)
    -/
  · field_simp [I_mul_I_of_nonzero h]
    /-
      🎉 no goals
    -/


@[simp, rclike_simps]
                                               /-
                                                 K : Type u_1
                                                 inst✝ : RCLike K
                                                 z : K
                                                 ⊢ Eq (HDiv.hDiv z RCLike.I) (Neg.neg (HMul.hMul z RCLike.I))
                                               -/
theorem div_I (z : K) : z / I = -(z * I) := by rw [div_eq_mul_inv, inv_I, mul_neg]
                                               /-
                                                 🎉 no goals
                                               -/


@[rclike_simps] -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11119): was `simp`
theorem normSq_inv (z : K) : normSq z⁻¹ = (normSq z)⁻¹ :=
  map_inv₀ normSq z


@[rclike_simps] -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11119): was `simp`
theorem normSq_div (z w : K) : normSq (z / w) = normSq z / normSq w :=
  map_div₀ normSq z w


@[simp 1100, rclike_simps]
                                                 /-
                                                   K : Type u_1
                                                   inst✝ : RCLike K
                                                   z : K
                                                   ⊢ Eq (Norm.norm ((starRingEnd K) z)) (Norm.norm z)
                                                 -/
theorem norm_conj (z : K) : ‖conj z‖ = ‖z‖ := by simp only [← sqrt_normSq_eq_norm, normSq_conj]
                                                 /-
                                                   🎉 no goals
                                                 -/


                                                                         /-
                                                                           K : Type u_1
                                                                           inst✝ : RCLike K
                                                                           z : K
                                                                           ⊢ Eq (NNNorm.nnnorm ((starRingEnd K) z)) (NNNorm.nnnorm z)
                                                                         -/
@[simp, rclike_simps] lemma nnnorm_conj (z : K) : ‖conj z‖₊ = ‖z‖₊ := by simp [nnnorm]
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


instance (priority := 100) : CStarRing K where
  norm_mul_self_le x := le_of_eq <| ((norm_mul _ _).trans <| congr_arg (· * ‖x‖) (norm_conj _)).symm


@[rclike_simps, norm_cast]
theorem ofReal_natCast (n : ℕ) : ((n : ℝ) : K) = n :=
  map_natCast (algebraMap ℝ K) n


@[rclike_simps, norm_cast]
lemma ofReal_nnratCast (q : ℚ≥0) : ((q : ℝ) : K) = q := map_nnratCast (algebraMap ℝ K) _


@[simp, rclike_simps] -- Porting note: removed `norm_cast`
                                                  /-
                                                    K : Type u_1
                                                    inst✝ : RCLike K
                                                    n : Nat
                                                    ⊢ Eq (RCLike.re ↑n) ↑n
                                                  -/
theorem natCast_re (n : ℕ) : re (n : K) = n := by rw [← ofReal_natCast, ofReal_re]
                                                  /-
                                                    🎉 no goals
                                                  -/


@[simp, rclike_simps, norm_cast]
                                                  /-
                                                    K : Type u_1
                                                    inst✝ : RCLike K
                                                    n : Nat
                                                    ⊢ Eq (RCLike.im ↑n) 0
                                                  -/
theorem natCast_im (n : ℕ) : im (n : K) = 0 := by rw [← ofReal_natCast, ofReal_im]
                                                  /-
                                                    🎉 no goals
                                                  -/

@[simp, rclike_simps]
theorem ofNat_re (n : ℕ) [n.AtLeastTwo] : re (ofNat(n) : K) = ofNat(n) :=
  natCast_re n

@[simp, rclike_simps]
theorem ofNat_im (n : ℕ) [n.AtLeastTwo] : im (ofNat(n) : K) = 0 :=
  natCast_im n


@[rclike_simps, norm_cast]
theorem ofReal_ofNat (n : ℕ) [n.AtLeastTwo] : ((ofNat(n) : ℝ) : K) = ofNat(n) :=
  ofReal_natCast n


theorem ofNat_mul_re (n : ℕ) [n.AtLeastTwo] (z : K) :
    re (ofNat(n) * z) = ofNat(n) * re z := by
  /-
    K : Type u_1
    inst✝¹ : RCLike K
    n : Nat
    inst✝ : n.AtLeastTwo
    z : K
    ⊢ Eq (RCLike.re (HMul.hMul (OfNat.ofNat n) z)) (HMul.hMul (OfNat.ofNat n) (RCL …
  -/
  rw [← ofReal_ofNat, re_ofReal_mul]
  /-
    🎉 no goals
  -/


theorem ofNat_mul_im (n : ℕ) [n.AtLeastTwo] (z : K) :
    im (ofNat(n) * z) = ofNat(n) * im z := by
  /-
    K : Type u_1
    inst✝¹ : RCLike K
    n : Nat
    inst✝ : n.AtLeastTwo
    z : K
    ⊢ Eq (RCLike.im (HMul.hMul (OfNat.ofNat n) z)) (HMul.hMul (OfNat.ofNat n) (RCL …
  -/
  rw [← ofReal_ofNat, im_ofReal_mul]
  /-
    🎉 no goals
  -/


@[rclike_simps, norm_cast]
theorem ofReal_intCast (n : ℤ) : ((n : ℝ) : K) = n :=
  map_intCast _ n


@[simp, rclike_simps] -- Porting note: removed `norm_cast`
                                                  /-
                                                    K : Type u_1
                                                    inst✝ : RCLike K
                                                    n : Int
                                                    ⊢ Eq (RCLike.re ↑n) ↑n
                                                  -/
theorem intCast_re (n : ℤ) : re (n : K) = n := by rw [← ofReal_intCast, ofReal_re]
                                                  /-
                                                    🎉 no goals
                                                  -/


@[simp, rclike_simps, norm_cast]
                                                  /-
                                                    K : Type u_1
                                                    inst✝ : RCLike K
                                                    n : Int
                                                    ⊢ Eq (RCLike.im ↑n) 0
                                                  -/
theorem intCast_im (n : ℤ) : im (n : K) = 0 := by rw [← ofReal_intCast, ofReal_im]
                                                  /-
                                                    🎉 no goals
                                                  -/


@[rclike_simps, norm_cast]
theorem ofReal_ratCast (n : ℚ) : ((n : ℝ) : K) = n :=
  map_ratCast _ n


@[simp, rclike_simps] -- Porting note: removed `norm_cast`
                                                  /-
                                                    K : Type u_1
                                                    inst✝ : RCLike K
                                                    q : Rat
                                                    ⊢ Eq (RCLike.re ↑q) ↑q
                                                  -/
theorem ratCast_re (q : ℚ) : re (q : K) = q := by rw [← ofReal_ratCast, ofReal_re]
                                                  /-
                                                    🎉 no goals
                                                  -/


@[simp, rclike_simps, norm_cast]
                                                  /-
                                                    K : Type u_1
                                                    inst✝ : RCLike K
                                                    q : Rat
                                                    ⊢ Eq (RCLike.im ↑q) 0
                                                  -/
theorem ratCast_im (q : ℚ) : im (q : K) = 0 := by rw [← ofReal_ratCast, ofReal_im]
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem norm_of_nonneg {r : ℝ} (h : 0 ≤ r) : ‖(r : K)‖ = r :=
  (norm_ofReal _).trans (abs_of_nonneg h)


@[simp, rclike_simps, norm_cast]
theorem norm_natCast (n : ℕ) : ‖(n : K)‖ = n := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    n : Nat
    ⊢ Eq (Norm.norm ↑n) ↑n
  -/
  rw [← ofReal_natCast]
  /-
    K : Type u_1
    inst✝ : RCLike K
    n : Nat
    ⊢ Eq (Norm.norm ↑↑n) ↑n
  -/
  exact norm_of_nonneg (Nat.cast_nonneg n)
  /-
    🎉 no goals
  -/


                                                                                     /-
                                                                                       K : Type u_1
                                                                                       inst✝ : RCLike K
                                                                                       n : Nat
                                                                                       ⊢ Eq (NNNorm.nnnorm ↑n) ↑n
                                                                                     -/
@[simp, rclike_simps, norm_cast] lemma nnnorm_natCast (n : ℕ) : ‖(n : K)‖₊ = n := by simp [nnnorm]
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


@[simp, rclike_simps]
theorem norm_ofNat (n : ℕ) [n.AtLeastTwo] : ‖(ofNat(n) : K)‖ = ofNat(n) :=
  norm_natCast n


@[simp, rclike_simps]
lemma nnnorm_ofNat (n : ℕ) [n.AtLeastTwo] : ‖(ofNat(n) : K)‖₊ = ofNat(n) :=
  nnnorm_natCast n


lemma norm_two : ‖(2 : K)‖ = 2 := norm_ofNat 2

lemma nnnorm_two : ‖(2 : K)‖₊ = 2 := nnnorm_ofNat 2


@[simp, rclike_simps, norm_cast]
lemma norm_nnratCast (q : ℚ≥0) : ‖(q : K)‖ = q := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    q : NNRat
    ⊢ Eq (Norm.norm ↑q) ↑q
  -/
  rw [← ofReal_nnratCast]; exact norm_of_nonneg q.cast_nonneg
                           /-
                             🎉 no goals
                           -/


@[simp, rclike_simps, norm_cast]
                                                        /-
                                                          K : Type u_1
                                                          inst✝ : RCLike K
                                                          q : NNRat
                                                          ⊢ Eq (NNNorm.nnnorm ↑q) ↑q
                                                        -/
lemma nnnorm_nnratCast (q : ℚ≥0) : ‖(q : K)‖₊ = q := by simp [nnnorm]
                                                        /-
                                                          🎉 no goals
                                                        -/


variable (K) in
lemma norm_nsmul [NormedAddCommGroup E] [NormedSpace K E] (n : ℕ) (x : E) : ‖n • x‖ = n • ‖x‖ := by
  /-
    K : Type u_1
    E : Type u_2
    inst✝² : RCLike K
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace K E
    n : Nat
    x : E
    ⊢ Eq (Norm.norm (HSMul.hSMul n x)) (HSMul.hSMul n (Norm.norm x))
  -/
  simpa [Nat.cast_smul_eq_nsmul] using norm_smul (n : K) x
  /-
    🎉 no goals
  -/


variable (K) in
lemma nnnorm_nsmul [NormedAddCommGroup E] [NormedSpace K E] (n : ℕ) (x : E) :
                              /-
                                K : Type u_1
                                E : Type u_2
                                inst✝² : RCLike K
                                inst✝¹ : NormedAddCommGroup E
                                inst✝ : NormedSpace K E
                                n : Nat
                                x : E
                                ⊢ Eq (NNNorm.nnnorm (HSMul.hSMul n x)) (HSMul.hSMul n (NNNorm.nnnorm x))
                              -/
    ‖n • x‖₊ = n • ‖x‖₊ := by simpa [Nat.cast_smul_eq_nsmul] using nnnorm_smul (n : K) x
                              /-
                                🎉 no goals
                              -/


variable (K) in
lemma norm_nnqsmul (q : ℚ≥0) (x : E) : ‖q • x‖ = q • ‖x‖ := by
  /-
    K : Type u_1
    E : Type u_2
    inst✝³ : RCLike K
    inst✝² : NormedField E
    inst✝¹ : CharZero E
    inst✝ : NormedSpace K E
    q : NNRat
    x : E
    ⊢ Eq (Norm.norm (HSMul.hSMul q x)) (HSMul.hSMul q (Norm.norm x))
  -/
  simpa [NNRat.cast_smul_eq_nnqsmul] using norm_smul (q : K) x
  /-
    🎉 no goals
  -/


variable (K) in
lemma nnnorm_nnqsmul (q : ℚ≥0) (x : E) : ‖q • x‖₊ = q • ‖x‖₊ := by
  /-
    K : Type u_1
    E : Type u_2
    inst✝³ : RCLike K
    inst✝² : NormedField E
    inst✝¹ : CharZero E
    inst✝ : NormedSpace K E
    q : NNRat
    x : E
    ⊢ Eq (NNNorm.nnnorm (HSMul.hSMul q x)) (HSMul.hSMul q (NNNorm.nnnorm x))
  -/
  simpa [NNRat.cast_smul_eq_nnqsmul] using nnnorm_smul (q : K) x
  /-
    🎉 no goals
  -/


@[bound]
lemma norm_expect_le {ι : Type*} {s : Finset ι} {f : ι → E} : ‖𝔼 i ∈ s, f i‖ ≤ 𝔼 i ∈ s, ‖f i‖ :=
                                                                     /-
                                                                       K : Type u_1
                                                                       E : Type u_2
                                                                       inst✝³ : RCLike K
                                                                       inst✝² : NormedField E
                                                                       inst✝¹ : CharZero E
                                                                       inst✝ : NormedSpace K E
                                                                       ι : Type u_3
                                                                       s : Finset ι
                                                                       f : ι → E
                                                                       x✝¹ : Nat
                                                                       x✝ : E
                                                                       ⊢ Eq (Norm.norm (HSMul.hSMul (Inv.inv ↑x✝¹) x✝)) (HSMul.hSMul (Inv.inv ↑x✝¹) ( …
                                                                     -/
  Finset.le_expect_of_subadditive norm_zero norm_add_le fun _ _ ↦ by rw [norm_nnqsmul K]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


                                                           /-
                                                             K : Type u_1
                                                             inst✝ : RCLike K
                                                             z : K
                                                             ⊢ Eq (HMul.hMul (Norm.norm z) (Norm.norm z)) (RCLike.normSq z)
                                                           -/
theorem mul_self_norm (z : K) : ‖z‖ * ‖z‖ = normSq z := by rw [normSq_eq_def', sq]
                                                           /-
                                                             🎉 no goals
                                                           -/


attribute [rclike_simps] norm_zero norm_one norm_eq_zero abs_norm norm_inv norm_div


theorem abs_re_le_norm (z : K) : |re z| ≤ ‖z‖ := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    z : K
    ⊢ LE.le (abs (RCLike.re z)) (Norm.norm z)
  -/
  rw [mul_self_le_mul_self_iff (abs_nonneg _) (norm_nonneg _), abs_mul_abs_self, mul_self_norm]
  /-
    K : Type u_1
    inst✝ : RCLike K
    z : K
    ⊢ LE.le (HMul.hMul (RCLike.re z) (RCLike.re z)) (RCLike.normSq z)
  -/
  apply re_sq_le_normSq
  /-
    🎉 no goals
  -/


theorem abs_im_le_norm (z : K) : |im z| ≤ ‖z‖ := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    z : K
    ⊢ LE.le (abs (RCLike.im z)) (Norm.norm z)
  -/
  rw [mul_self_le_mul_self_iff (abs_nonneg _) (norm_nonneg _), abs_mul_abs_self, mul_self_norm]
  /-
    K : Type u_1
    inst✝ : RCLike K
    z : K
    ⊢ LE.le (HMul.hMul (RCLike.im z) (RCLike.im z)) (RCLike.normSq z)
  -/
  apply im_sq_le_normSq
  /-
    🎉 no goals
  -/


theorem norm_re_le_norm (z : K) : ‖re z‖ ≤ ‖z‖ :=
  abs_re_le_norm z


theorem norm_im_le_norm (z : K) : ‖im z‖ ≤ ‖z‖ :=
  abs_im_le_norm z


theorem re_le_norm (z : K) : re z ≤ ‖z‖ :=
  (abs_le.1 (abs_re_le_norm z)).2


theorem im_le_norm (z : K) : im z ≤ ‖z‖ :=
  (abs_le.1 (abs_im_le_norm _)).2


theorem im_eq_zero_of_le {a : K} (h : ‖a‖ ≤ re a) : im a = 0 := by
  simpa only [mul_self_norm a, normSq_apply, self_eq_add_right, mul_self_eq_zero]
    using congr_arg (fun z => z * z) ((re_le_norm a).antisymm h)


theorem re_eq_self_of_le {a : K} (h : ‖a‖ ≤ re a) : (re a : K) = a := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    a : K
    h : LE.le (Norm.norm a) (RCLike.re a)
    ⊢ Eq (↑(RCLike.re a)) a
  -/
  rw [← conj_eq_iff_re, conj_eq_iff_im, im_eq_zero_of_le h]
  /-
    🎉 no goals
  -/


theorem abs_re_div_norm_le_one (z : K) : |re z / ‖z‖| ≤ 1 := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    z : K
    ⊢ LE.le (abs (HDiv.hDiv (RCLike.re z) (Norm.norm z))) 1
  -/
  rw [abs_div, abs_norm]
  /-
    K : Type u_1
    inst✝ : RCLike K
    z : K
    ⊢ LE.le (HDiv.hDiv (abs (RCLike.re z)) (Norm.norm z)) 1
  -/
  exact div_le_one_of_le₀ (abs_re_le_norm _) (norm_nonneg _)
  /-
    🎉 no goals
  -/


theorem abs_im_div_norm_le_one (z : K) : |im z / ‖z‖| ≤ 1 := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    z : K
    ⊢ LE.le (abs (HDiv.hDiv (RCLike.im z) (Norm.norm z))) 1
  -/
  rw [abs_div, abs_norm]
  /-
    K : Type u_1
    inst✝ : RCLike K
    z : K
    ⊢ LE.le (HDiv.hDiv (abs (RCLike.im z)) (Norm.norm z)) 1
  -/
  exact div_le_one_of_le₀ (abs_im_le_norm _) (norm_nonneg _)
  /-
    🎉 no goals
  -/


theorem norm_I_of_ne_zero (hI : (I : K) ≠ 0) : ‖(I : K)‖ = 1 := by
  rw [← mul_self_inj_of_nonneg (norm_nonneg I) zero_le_one, one_mul, ← norm_mul,
    I_mul_I_of_nonzero hI, norm_neg, norm_one]


theorem re_eq_norm_of_mul_conj (x : K) : re (x * conj x) = ‖x * conj x‖ := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    x : K
    ⊢ Eq (RCLike.re (HMul.hMul x ((starRingEnd K) x))) (Norm.norm (HMul.hMul x ((s …
  -/
  rw [mul_conj, ← ofReal_pow]; simp [-map_pow]
                               /-
                                 🎉 no goals
                               -/


theorem norm_sq_re_add_conj (x : K) : ‖x + conj x‖ ^ 2 = re (x + conj x) ^ 2 := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    x : K
    ⊢ Eq (HPow.hPow (Norm.norm (HAdd.hAdd x ((starRingEnd K) x))) 2) (HPow.hPow (R …
  -/
  rw [add_conj, ← ofReal_ofNat, ← ofReal_mul, norm_ofReal, sq_abs, ofReal_re]
  /-
    🎉 no goals
  -/


theorem norm_sq_re_conj_add (x : K) : ‖conj x + x‖ ^ 2 = re (conj x + x) ^ 2 := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    x : K
    ⊢ Eq (HPow.hPow (Norm.norm (HAdd.hAdd ((starRingEnd K) x) x)) 2) (HPow.hPow (R …
  -/
  rw [add_comm, norm_sq_re_add_conj]
  /-
    🎉 no goals
  -/


theorem isCauSeq_re (f : CauSeq K norm) : IsCauSeq abs fun n => re (f n) := fun _ ε0 =>
  (f.cauchy ε0).imp fun i H j ij =>
                       /-
                         K : Type u_1
                         inst✝ : RCLike K
                         f : CauSeq K Norm.norm
                         x✝ : Real
                         ε0 : GT.gt x✝ 0
                         i : Nat
                         H : ∀ (j : Nat), GE.ge j i → LT.lt (Norm.norm (HSub.hSub (↑f j) (↑f i))) x✝
                         j : Nat
                         ij : GE.ge j i
                         ⊢ LE.le (abs (HSub.hSub ((fun n => RCLike.re (↑f n)) j) ((fun n => RCLike.re ( …
                       -/
    lt_of_le_of_lt (by simpa only [map_sub] using abs_re_le_norm (f j - f i)) (H _ ij)
                       /-
                         🎉 no goals
                       -/


theorem isCauSeq_im (f : CauSeq K norm) : IsCauSeq abs fun n => im (f n) := fun _ ε0 =>
  (f.cauchy ε0).imp fun i H j ij =>
                       /-
                         K : Type u_1
                         inst✝ : RCLike K
                         f : CauSeq K Norm.norm
                         x✝ : Real
                         ε0 : GT.gt x✝ 0
                         i : Nat
                         H : ∀ (j : Nat), GE.ge j i → LT.lt (Norm.norm (HSub.hSub (↑f j) (↑f i))) x✝
                         j : Nat
                         ij : GE.ge j i
                         ⊢ LE.le (abs (HSub.hSub ((fun n => RCLike.im (↑f n)) j) ((fun n => RCLike.im ( …
                       -/
    lt_of_le_of_lt (by simpa only [map_sub] using abs_im_le_norm (f j - f i)) (H _ ij)
                       /-
                         🎉 no goals
                       -/


/-- The real part of a K Cauchy sequence, as a real Cauchy sequence. -/
noncomputable def cauSeqRe (f : CauSeq K norm) : CauSeq ℝ abs :=
  ⟨_, isCauSeq_re f⟩


/-- The imaginary part of a K Cauchy sequence, as a real Cauchy sequence. -/
noncomputable def cauSeqIm (f : CauSeq K norm) : CauSeq ℝ abs :=
  ⟨_, isCauSeq_im f⟩


theorem isCauSeq_norm {f : ℕ → K} (hf : IsCauSeq norm f) : IsCauSeq abs (norm ∘ f) := fun ε ε0 =>
  let ⟨i, hi⟩ := hf ε ε0
  ⟨i, fun j hj => lt_of_le_of_lt (abs_norm_sub_norm_le _ _) (hi j hj)⟩


noncomputable instance Real.instRCLike : RCLike ℝ where
  re := AddMonoidHom.id ℝ
  im := 0
  I := 0
                /-
                  K : Type u_1
                  E : Type u_2
                  inst✝ : RCLike K
                  ⊢ Eq ((AddMonoidHom.id Real) 0) 0
                -/
  I_re_ax := by simp only [AddMonoidHom.map_zero]
                /-
                  🎉 no goals
                -/
  I_mul_I_ax := Or.intro_left _ rfl
  re_add_im_ax z := by
    /-
      K : Type u_1
      E : Type u_2
      inst✝ : RCLike K
      z : Real
      ⊢ Eq (HAdd.hAdd ((algebraMap Real Real) ((AddMonoidHom.id Real) z)) (HMul.hMul …
    -/
    simp only [add_zero, mul_zero, Algebra.id.map_eq_id, RingHom.id_apply, AddMonoidHom.id_apply]
    /-
      🎉 no goals
    -/
  ofReal_re_ax _ := rfl
  ofReal_im_ax _ := rfl
                      /-
                        K : Type u_1
                        E : Type u_2
                        inst✝ : RCLike K
                        z w : Real
                        ⊢ Eq ((AddMonoidHom.id Real) (HMul.hMul z w)) (HSub.hSub (HMul.hMul ((AddMonoi …
                      -/
  mul_re_ax z w := by simp only [sub_zero, mul_zero, AddMonoidHom.zero_apply, AddMonoidHom.id_apply]
                      /-
                        🎉 no goals
                      -/
                      /-
                        K : Type u_1
                        E : Type u_2
                        inst✝ : RCLike K
                        z w : Real
                        ⊢ Eq (0 (HMul.hMul z w)) (HAdd.hAdd (HMul.hMul ((AddMonoidHom.id Real) z) (0 w …
                      -/
  mul_im_ax z w := by simp only [add_zero, zero_mul, mul_zero, AddMonoidHom.zero_apply]
                      /-
                        🎉 no goals
                      -/
                     /-
                       K : Type u_1
                       E : Type u_2
                       inst✝ : RCLike K
                       z : Real
                       ⊢ Eq ((AddMonoidHom.id Real) ((starRingEnd Real) z)) ((AddMonoidHom.id Real) z)
                     -/
  conj_re_ax z := by simp only [starRingEnd_apply, star_id_of_comm]
                     /-
                       🎉 no goals
                     -/
                     /-
                       K : Type u_1
                       E : Type u_2
                       inst✝ : RCLike K
                       x✝ : Real
                       ⊢ Eq (0 ((starRingEnd Real) x✝)) (Neg.neg (0 x✝))
                     -/
  conj_im_ax _ := by simp only [neg_zero, AddMonoidHom.zero_apply]
                     /-
                       🎉 no goals
                     -/
                  /-
                    K : Type u_1
                    E : Type u_2
                    inst✝ : RCLike K
                    ⊢ Eq ((starRingEnd Real) 0) (-0)
                  -/
  conj_I_ax := by simp only [RingHom.map_zero, neg_zero]
                  /-
                    🎉 no goals
                  -/
  norm_sq_eq_def_ax z := by simp only [sq, Real.norm_eq_abs, ← abs_mul, abs_mul_self z, add_zero,
    mul_zero, AddMonoidHom.zero_apply, AddMonoidHom.id_apply]
                      /-
                        K : Type u_1
                        E : Type u_2
                        inst✝ : RCLike K
                        x✝ : Real
                        ⊢ Eq (HMul.hMul (0 x✝) (0 0)) (0 x✝)
                      -/
  mul_im_I_ax _ := by simp only [mul_zero, AddMonoidHom.zero_apply]
                      /-
                        🎉 no goals
                      -/
  le_iff_re_im := (and_iff_left rfl).symm


theorem lt_iff_re_im : z < w ↔ re z < re w ∧ im z = im w := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    z w : K
    ⊢ Iff (LT.lt z w) (And (LT.lt (RCLike.re z) (RCLike.re w)) (Eq (RCLike.im z) ( …
  -/
  simp_rw [lt_iff_le_and_ne, @RCLike.le_iff_re_im K]
  /-
    K : Type u_1
    inst✝ : RCLike K
    z w : K
    ⊢ Iff (And (And (LE.le (RCLike.re z) (RCLike.re w)) (Eq (RCLike.im z) (RCLike. …
  -/
  constructor
    /-
      case mp
      K : Type u_1
      inst✝ : RCLike K
      z w : K
      ⊢ And (And (LE.le (RCLike.re z) (RCLike.re w)) (Eq (RCLike.im z) (RCLike.im w) …
    -/
  · rintro ⟨⟨hr, hi⟩, heq⟩
    /-
      case mp.intro.intro
      K : Type u_1
      inst✝ : RCLike K
      z w : K
      heq : Ne z w
      hr : LE.le (RCLike.re z) (RCLike.re w)
      hi : Eq (RCLike.im z) (RCLike.im w)
      ⊢ And (And (LE.le (RCLike.re z) (RCLike.re w)) (Ne (RCLike.re z) (RCLike.re w) …
    -/
    exact ⟨⟨hr, mt (fun hreq => ext hreq hi) heq⟩, hi⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      K : Type u_1
      inst✝ : RCLike K
      z w : K
      ⊢ And (And (LE.le (RCLike.re z) (RCLike.re w)) (Ne (RCLike.re z) (RCLike.re w) …
    -/
  · rintro ⟨⟨hr, hrn⟩, hi⟩
    /-
      case mpr.intro.intro
      K : Type u_1
      inst✝ : RCLike K
      z w : K
      hi : Eq (RCLike.im z) (RCLike.im w)
      hr : LE.le (RCLike.re z) (RCLike.re w)
      hrn : Ne (RCLike.re z) (RCLike.re w)
      ⊢ And (And (LE.le (RCLike.re z) (RCLike.re w)) (Eq (RCLike.im z) (RCLike.im w) …
    -/
    exact ⟨⟨hr, hi⟩, ne_of_apply_ne _ hrn⟩
    /-
      🎉 no goals
    -/


theorem nonneg_iff : 0 ≤ z ↔ 0 ≤ re z ∧ im z = 0 := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    z : K
    ⊢ Iff (LE.le 0 z) (And (LE.le 0 (RCLike.re z)) (Eq (RCLike.im z) 0))
  -/
  simpa only [map_zero, eq_comm] using le_iff_re_im (z := 0) (w := z)
  /-
    🎉 no goals
  -/


theorem pos_iff : 0 < z ↔ 0 < re z ∧ im z = 0 := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    z : K
    ⊢ Iff (LT.lt 0 z) (And (LT.lt 0 (RCLike.re z)) (Eq (RCLike.im z) 0))
  -/
  simpa only [map_zero, eq_comm] using lt_iff_re_im (z := 0) (w := z)
  /-
    🎉 no goals
  -/


theorem nonpos_iff : z ≤ 0 ↔ re z ≤ 0 ∧ im z = 0 := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    z : K
    ⊢ Iff (LE.le z 0) (And (LE.le (RCLike.re z) 0) (Eq (RCLike.im z) 0))
  -/
  simpa only [map_zero] using le_iff_re_im (z := z) (w := 0)
  /-
    🎉 no goals
  -/


theorem neg_iff : z < 0 ↔ re z < 0 ∧ im z = 0 := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    z : K
    ⊢ Iff (LT.lt z 0) (And (LT.lt (RCLike.re z) 0) (Eq (RCLike.im z) 0))
  -/
  simpa only [map_zero] using lt_iff_re_im (z := z) (w := 0)
  /-
    🎉 no goals
  -/


lemma nonneg_iff_exists_ofReal : 0 ≤ z ↔ ∃ x ≥ (0 : ℝ), x = z := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    z : K
    ⊢ Iff (LE.le 0 z) (Exists fun x => And (GE.ge x 0) (Eq (↑x) z))
  -/
  simp_rw [nonneg_iff (K := K), ext_iff (K := K)]; aesop
                                                   /-
                                                     🎉 no goals
                                                   -/


lemma pos_iff_exists_ofReal : 0 < z ↔ ∃ x > (0 : ℝ), x = z := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    z : K
    ⊢ Iff (LT.lt 0 z) (Exists fun x => And (GT.gt x 0) (Eq (↑x) z))
  -/
  simp_rw [pos_iff (K := K), ext_iff (K := K)]; aesop
                                                /-
                                                  🎉 no goals
                                                -/


lemma nonpos_iff_exists_ofReal : z ≤ 0 ↔ ∃ x ≤ (0 : ℝ), x = z := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    z : K
    ⊢ Iff (LE.le z 0) (Exists fun x => And (LE.le x 0) (Eq (↑x) z))
  -/
  simp_rw [nonpos_iff (K := K), ext_iff (K := K)]; aesop
                                                   /-
                                                     🎉 no goals
                                                   -/


lemma neg_iff_exists_ofReal : z < 0 ↔ ∃ x < (0 : ℝ), x = z := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    z : K
    ⊢ Iff (LT.lt z 0) (Exists fun x => And (LT.lt x 0) (Eq (↑x) z))
  -/
  simp_rw [neg_iff (K := K), ext_iff (K := K)]; aesop
                                                /-
                                                  🎉 no goals
                                                -/


@[simp, norm_cast]
lemma ofReal_le_ofReal {x y : ℝ} : (x : K) ≤ (y : K) ↔ x ≤ y := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    x y : Real
    ⊢ Iff (LE.le ↑x ↑y) (LE.le x y)
  -/
  rw [le_iff_re_im]
  /-
    K : Type u_1
    inst✝ : RCLike K
    x y : Real
    ⊢ Iff (And (LE.le (RCLike.re ↑x) (RCLike.re ↑y)) (Eq (RCLike.im ↑x) (RCLike.im …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
lemma ofReal_lt_ofReal {x y : ℝ} : (x : K) < (y : K) ↔ x < y := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    x y : Real
    ⊢ Iff (LT.lt ↑x ↑y) (LT.lt x y)
  -/
  rw [lt_iff_re_im]
  /-
    K : Type u_1
    inst✝ : RCLike K
    x y : Real
    ⊢ Iff (And (LT.lt (RCLike.re ↑x) (RCLike.re ↑y)) (Eq (RCLike.im ↑x) (RCLike.im …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
lemma ofReal_nonneg {x : ℝ} : 0 ≤ (x : K) ↔ 0 ≤ x := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    x : Real
    ⊢ Iff (LE.le 0 ↑x) (LE.le 0 x)
  -/
  rw [← ofReal_zero, ofReal_le_ofReal]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
lemma ofReal_nonpos {x : ℝ} : (x : K) ≤ 0 ↔ x ≤ 0 := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    x : Real
    ⊢ Iff (LE.le (↑x) 0) (LE.le x 0)
  -/
  rw [← ofReal_zero, ofReal_le_ofReal]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
lemma ofReal_pos {x : ℝ} : 0 < (x : K) ↔ 0 < x := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    x : Real
    ⊢ Iff (LT.lt 0 ↑x) (LT.lt 0 x)
  -/
  rw [← ofReal_zero, ofReal_lt_ofReal]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
lemma ofReal_lt_zero {x : ℝ} : (x : K) < 0 ↔ x < 0 := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    x : Real
    ⊢ Iff (LT.lt (↑x) 0) (LT.lt x 0)
  -/
  rw [← ofReal_zero, ofReal_lt_ofReal]
  /-
    🎉 no goals
  -/


protected lemma inv_pos_of_pos (hz : 0 < z) : 0 < z⁻¹ := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    z : K
    hz : LT.lt 0 z
    ⊢ LT.lt 0 (Inv.inv z)
  -/
  rw [pos_iff_exists_ofReal] at hz
  /-
    K : Type u_1
    inst✝ : RCLike K
    z : K
    hz : Exists fun x => And (GT.gt x 0) (Eq (↑x) z)
    ⊢ LT.lt 0 (Inv.inv z)
  -/
  obtain ⟨x, hx, hx'⟩ := hz
  /-
    case intro.intro
    K : Type u_1
    inst✝ : RCLike K
    z : K
    x : Real
    hx : GT.gt x 0
    hx' : Eq (↑x) z
    ⊢ LT.lt 0 (Inv.inv z)
  -/
  rw [← hx', ← ofReal_inv, ofReal_pos]
  /-
    case intro.intro
    K : Type u_1
    inst✝ : RCLike K
    z : K
    x : Real
    hx : GT.gt x 0
    hx' : Eq (↑x) z
    ⊢ LT.lt 0 (Inv.inv x)
  -/
  exact inv_pos_of_pos hx
  /-
    🎉 no goals
  -/


protected lemma inv_pos : 0 < z⁻¹ ↔ 0 < z := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    z : K
    ⊢ Iff (LT.lt 0 (Inv.inv z)) (LT.lt 0 z)
  -/
  refine ⟨fun h => ?_, fun h => RCLike.inv_pos_of_pos h⟩
  /-
    K : Type u_1
    inst✝ : RCLike K
    z : K
    h : LT.lt 0 (Inv.inv z)
    ⊢ LT.lt 0 z
  -/
  rw [← inv_inv z]
  /-
    K : Type u_1
    inst✝ : RCLike K
    z : K
    h : LT.lt 0 (Inv.inv z)
    ⊢ LT.lt 0 (Inv.inv (Inv.inv z))
  -/
  exact RCLike.inv_pos_of_pos h
  /-
    🎉 no goals
  -/


/-- With `z ≤ w` iff `w - z` is real and nonnegative, `ℝ` and `ℂ` are star ordered rings.
(That is, a star ring in which the nonnegative elements are those of the form `star z * z`.)

Note this is only an instance with `open scoped ComplexOrder`. -/
lemma toStarOrderedRing : StarOrderedRing K :=
  StarOrderedRing.of_nonneg_iff'
    (h_add := fun {x y} hxy z => by
      /-
        K : Type u_1
        inst✝ : RCLike K
        x y : K
        hxy : LE.le x y
        z : K
        ⊢ LE.le (HAdd.hAdd z x) (HAdd.hAdd z y)
      -/
      rw [RCLike.le_iff_re_im] at *
      /-
        K : Type u_1
        inst✝ : RCLike K
        x y : K
        hxy : And (LE.le (RCLike.re x) (RCLike.re y)) (Eq (RCLike.im x) (RCLike.im y))
        z : K
        ⊢ And (LE.le (RCLike.re (HAdd.hAdd z x)) (RCLike.re (HAdd.hAdd z y))) (Eq (RCL …
      -/
      simpa [map_add, add_le_add_iff_left, add_right_inj] using hxy)
      /-
        🎉 no goals
      -/
    (h_nonneg_iff := fun x => by
      /-
        K : Type u_1
        inst✝ : RCLike K
        x : K
        ⊢ Iff (LE.le 0 x) (Exists fun s => Eq x (HMul.hMul (Star.star s) s))
      -/
      rw [nonneg_iff]
      /-
        K : Type u_1
        inst✝ : RCLike K
        x : K
        ⊢ Iff (And (LE.le 0 (RCLike.re x)) (Eq (RCLike.im x) 0)) (Exists fun s => Eq x …
      -/
      refine ⟨fun h ↦ ⟨√(re x), by simp [ext_iff (K := K), h.1, h.2]⟩, ?_⟩
      /-
        K : Type u_1
        inst✝ : RCLike K
        x : K
        ⊢ (Exists fun s => Eq x (HMul.hMul (Star.star s) s)) → And (LE.le 0 (RCLike.re …
      -/
      rintro ⟨s, rfl⟩
      /-
        case intro
        K : Type u_1
        inst✝ : RCLike K
        s : K
        ⊢ And (LE.le 0 (RCLike.re (HMul.hMul (Star.star s) s))) (Eq (RCLike.im (HMul.h …
      -/
      simp [mul_comm, mul_self_nonneg, add_nonneg])
      /-
        🎉 no goals
      -/


/-- With `z ≤ w` iff `w - z` is real and nonnegative, `ℝ` and `ℂ` are strictly ordered rings.

Note this is only an instance with `open scoped ComplexOrder`. -/
def toStrictOrderedCommRing : StrictOrderedCommRing K where
                    /-
                      K : Type u_1
                      E : Type u_2
                      inst✝ : RCLike K
                      z w : K
                      ⊢ LE.le 0 1
                    -/
  zero_le_one := by simp [@RCLike.le_iff_re_im K]
                    /-
                      🎉 no goals
                    -/
  add_le_add_left _ _ := add_le_add_left
  mul_pos z w hz hw := by
    /-
      K : Type u_1
      E : Type u_2
      inst✝ : RCLike K
      z✝ w✝ z w : K
      hz : LT.lt 0 z
      hw : LT.lt 0 w
      ⊢ LT.lt 0 (HMul.hMul z w)
    -/
    rw [lt_iff_re_im, map_zero] at hz hw ⊢
    /-
      K : Type u_1
      E : Type u_2
      inst✝ : RCLike K
      z✝ w✝ z w : K
      hz : And (LT.lt 0 (RCLike.re z)) (Eq (RCLike.im 0) (RCLike.im z))
      hw : And (LT.lt 0 (RCLike.re w)) (Eq (RCLike.im 0) (RCLike.im w))
      ⊢ And (LT.lt 0 (RCLike.re (HMul.hMul z w))) (Eq (RCLike.im 0) (RCLike.im (HMul …
    -/
    simp [mul_re, mul_im, ← hz.2, ← hw.2, mul_pos hz.1 hw.1]
    /-
      🎉 no goals
    -/
                 /-
                   K : Type u_1
                   E : Type u_2
                   inst✝ : RCLike K
                   z w : K
                   ⊢ ∀ (a b : K), Eq (HMul.hMul a b) (HMul.hMul b a)
                 -/
                                       /-
                                         🎉 no goals
                                       -/
  mul_comm := by intros; apply ext <;> ring_nf
                                       /-
                                         🎉 no goals
                                       -/


theorem toOrderedSMul : OrderedSMul ℝ K :=
  OrderedSMul.mk' fun a b r hab hr => by
    /-
      K : Type u_1
      inst✝ : RCLike K
      a b : K
      r : Real
      hab : LT.lt a b
      hr : LT.lt 0 r
      ⊢ LE.le (HSMul.hSMul r a) (HSMul.hSMul r b)
    -/
    replace hab := hab.le
    /-
      K : Type u_1
      inst✝ : RCLike K
      a b : K
      r : Real
      hr : LT.lt 0 r
      hab : LE.le a b
      ⊢ LE.le (HSMul.hSMul r a) (HSMul.hSMul r b)
    -/
    rw [RCLike.le_iff_re_im] at hab
    /-
      K : Type u_1
      inst✝ : RCLike K
      a b : K
      r : Real
      hr : LT.lt 0 r
      hab : And (LE.le (RCLike.re a) (RCLike.re b)) (Eq (RCLike.im a) (RCLike.im b))
      ⊢ LE.le (HSMul.hSMul r a) (HSMul.hSMul r b)
    -/
    rw [RCLike.le_iff_re_im, smul_re, smul_re, smul_im, smul_im]
    /-
      K : Type u_1
      inst✝ : RCLike K
      a b : K
      r : Real
      hr : LT.lt 0 r
      hab : And (LE.le (RCLike.re a) (RCLike.re b)) (Eq (RCLike.im a) (RCLike.im b))
      ⊢ And (LE.le (HMul.hMul r (RCLike.re a)) (HMul.hMul r (RCLike.re b))) (Eq (HMu …
    -/
    exact hab.imp (fun h => mul_le_mul_of_nonneg_left h hr.le) (congr_arg _)
    /-
      🎉 no goals
    -/


/-- A star algebra over `K` has a scalar multiplication that respects the order. -/
lemma _root_.StarModule.instOrderedSMul {A : Type*} [NonUnitalRing A] [StarRing A] [PartialOrder A]
    [StarOrderedRing A] [Module K A] [StarModule K A] [IsScalarTower K A A] [SMulCommClass K A A] :
    OrderedSMul K A where
  smul_lt_smul_of_pos {_ _ _} hxy hc := StarModule.smul_lt_smul_of_pos hxy hc
  lt_of_smul_lt_smul_of_pos {x y c} hxy hc := by
    have : c⁻¹ • c • x < c⁻¹ • c • y :=
      StarModule.smul_lt_smul_of_pos hxy (RCLike.inv_pos_of_pos hc)
    /-
      K : Type u_1
      inst✝⁸ : RCLike K
      A : Type u_3
      inst✝⁷ : NonUnitalRing A
      inst✝⁶ : StarRing A
      inst✝⁵ : PartialOrder A
      inst✝⁴ : StarOrderedRing A
      inst✝³ : Module K A
      inst✝² : StarModule K A
      inst✝¹ : IsScalarTower K A A
      inst✝ : SMulCommClass K A A
      x y : A
      c : K
      hxy : LT.lt (HSMul.hSMul c x) (HSMul.hSMul c y)
      hc : LT.lt 0 c
      this : LT.lt (HSMul.hSMul (Inv.inv c) (HSMul.hSMul c x)) (HSMul.hSMul (Inv.inv …
      ⊢ LT.lt x y
    -/
    simpa [smul_smul, inv_mul_cancel₀ hc.ne'] using this
    /-
      🎉 no goals
    -/


instance {A : Type*} [NonUnitalRing A] [StarRing A] [PartialOrder A] [StarOrderedRing A]
    [Module ℝ A] [StarModule ℝ A] [IsScalarTower ℝ A A] [SMulCommClass ℝ A A] :
    OrderedSMul ℝ A :=
  StarModule.instOrderedSMul


theorem ofReal_mul_pos_iff (x : ℝ) (z : K) :
    0 < x * z ↔ (x < 0 ∧ z < 0) ∨ (0 < x ∧ 0 < z) := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    x : Real
    z : K
    ⊢ Iff (LT.lt 0 (HMul.hMul (↑x) z)) (Or (And (LT.lt x 0) (LT.lt z 0)) (And (LT. …
  -/
  simp only [pos_iff (K := K), neg_iff (K := K), re_ofReal_mul, im_ofReal_mul]
  /-
    K : Type u_1
    inst✝ : RCLike K
    x : Real
    z : K
    ⊢ Iff (And (LT.lt 0 (HMul.hMul x (RCLike.re z))) (Eq (HMul.hMul x (RCLike.im z …
  -/
  obtain hx | hx | hx := lt_trichotomy x 0
  · simp only [mul_pos_iff, not_lt_of_gt hx, false_and, hx, true_and, false_or, mul_eq_zero, hx.ne,
      or_false]
    /-
      case inr.inl
      K : Type u_1
      inst✝ : RCLike K
      x : Real
      z : K
      hx : Eq x 0
      ⊢ Iff (And (LT.lt 0 (HMul.hMul x (RCLike.re z))) (Eq (HMul.hMul x (RCLike.im z …
    -/
  · simp only [hx, zero_mul, lt_self_iff_false, false_and, false_or]
    /-
      🎉 no goals
    -/
  · simp only [mul_pos_iff, hx, true_and, not_lt_of_gt hx, false_and, or_false, mul_eq_zero,
      hx.ne', false_or]


theorem ofReal_mul_neg_iff (x : ℝ) (z : K) :
    x * z < 0 ↔ (x < 0 ∧ 0 < z) ∨ (0 < x ∧ z < 0) := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    x : Real
    z : K
    ⊢ Iff (LT.lt (HMul.hMul (↑x) z) 0) (Or (And (LT.lt x 0) (LT.lt 0 z)) (And (LT. …
  -/
  simpa only [mul_neg, neg_pos, neg_neg_iff_pos] using ofReal_mul_pos_iff x (-z)
  /-
    🎉 no goals
  -/


local notation "reR" => @RCLike.re ℝ _

local notation "imR" => @RCLike.im ℝ _

local notation "IR" => @RCLike.I ℝ _

local notation "normSqR" => @RCLike.normSq ℝ _


@[simp, rclike_simps]
theorem re_to_real {x : ℝ} : reR x = x :=
  rfl


@[simp, rclike_simps]
theorem im_to_real {x : ℝ} : imR x = 0 :=
  rfl


@[rclike_simps]
theorem conj_to_real {x : ℝ} : conj x = x :=
  rfl


@[simp, rclike_simps]
theorem I_to_real : IR = 0 :=
  rfl


@[simp, rclike_simps]
                                                        /-
                                                          x : Real
                                                          ⊢ Eq (RCLike.normSq x) (HMul.hMul x x)
                                                        -/
theorem normSq_to_real {x : ℝ} : normSq x = x * x := by simp [RCLike.normSq]
                                                        /-
                                                          🎉 no goals
                                                        -/


@[simp]
theorem ofReal_real_eq_id : @ofReal ℝ _ = id :=
  rfl


/-- The real part in an `RCLike` field, as a linear map. -/
def reLm : K →ₗ[ℝ] ℝ :=
  { re with map_smul' := smul_re }


@[simp, rclike_simps]
theorem reLm_coe : (reLm : K → ℝ) = re :=
  rfl


/-- The real part in an `RCLike` field, as a continuous linear map. -/
noncomputable def reCLM : K →L[ℝ] ℝ :=
  reLm.mkContinuous 1 fun x => by
    /-
      K : Type u_1
      E : Type u_2
      inst✝ : RCLike K
      x : K
      ⊢ LE.le (Norm.norm (RCLike.reLm x)) (HMul.hMul 1 (Norm.norm x))
    -/
    rw [one_mul]
    /-
      K : Type u_1
      E : Type u_2
      inst✝ : RCLike K
      x : K
      ⊢ LE.le (Norm.norm (RCLike.reLm x)) (Norm.norm x)
    -/
    exact abs_re_le_norm x
    /-
      🎉 no goals
    -/


@[simp, rclike_simps, norm_cast]
theorem reCLM_coe : ((reCLM : K →L[ℝ] ℝ) : K →ₗ[ℝ] ℝ) = reLm :=
  rfl


@[simp, rclike_simps]
theorem reCLM_apply : ((reCLM : K →L[ℝ] ℝ) : K → ℝ) = re :=
  rfl


@[continuity, fun_prop]
theorem continuous_re : Continuous (re : K → ℝ) :=
  reCLM.continuous


/-- The imaginary part in an `RCLike` field, as a linear map. -/
def imLm : K →ₗ[ℝ] ℝ :=
  { im with map_smul' := smul_im }


@[simp, rclike_simps]
theorem imLm_coe : (imLm : K → ℝ) = im :=
  rfl


/-- The imaginary part in an `RCLike` field, as a continuous linear map. -/
noncomputable def imCLM : K →L[ℝ] ℝ :=
  imLm.mkContinuous 1 fun x => by
    /-
      K : Type u_1
      E : Type u_2
      inst✝ : RCLike K
      x : K
      ⊢ LE.le (Norm.norm (RCLike.imLm x)) (HMul.hMul 1 (Norm.norm x))
    -/
    rw [one_mul]
    /-
      K : Type u_1
      E : Type u_2
      inst✝ : RCLike K
      x : K
      ⊢ LE.le (Norm.norm (RCLike.imLm x)) (Norm.norm x)
    -/
    exact abs_im_le_norm x
    /-
      🎉 no goals
    -/


@[simp, rclike_simps, norm_cast]
theorem imCLM_coe : ((imCLM : K →L[ℝ] ℝ) : K →ₗ[ℝ] ℝ) = imLm :=
  rfl


@[simp, rclike_simps]
theorem imCLM_apply : ((imCLM : K →L[ℝ] ℝ) : K → ℝ) = im :=
  rfl


@[continuity, fun_prop]
theorem continuous_im : Continuous (im : K → ℝ) :=
  imCLM.continuous


/-- Conjugate as an `ℝ`-algebra equivalence -/
def conjAe : K ≃ₐ[ℝ] K :=
  { conj with
    invFun := conj
    left_inv := conj_conj
    right_inv := conj_conj
    commutes' := conj_ofReal }


@[simp, rclike_simps]
theorem conjAe_coe : (conjAe : K → K) = conj :=
  rfl


/-- Conjugate as a linear isometry -/
noncomputable def conjLIE : K ≃ₗᵢ[ℝ] K :=
  ⟨conjAe.toLinearEquiv, norm_conj⟩


@[simp, rclike_simps]
theorem conjLIE_apply : (conjLIE : K → K) = conj :=
  rfl


/-- Conjugate as a continuous linear equivalence -/
noncomputable def conjCLE : K ≃L[ℝ] K :=
  @conjLIE K _


@[simp, rclike_simps]
theorem conjCLE_coe : (@conjCLE K _).toLinearEquiv = conjAe.toLinearEquiv :=
  rfl


@[simp, rclike_simps]
theorem conjCLE_apply : (conjCLE : K → K) = conj :=
  rfl


instance (priority := 100) : ContinuousStar K :=
  ⟨conjLIE.continuous⟩


@[continuity]
theorem continuous_conj : Continuous (conj : K → K) :=
  continuous_star


/-- The `ℝ → K` coercion, as a linear map -/
noncomputable def ofRealAm : ℝ →ₐ[ℝ] K :=
  Algebra.ofId ℝ K


@[simp, rclike_simps]
theorem ofRealAm_coe : (ofRealAm : ℝ → K) = ofReal :=
  rfl


/-- The ℝ → K coercion, as a linear isometry -/
noncomputable def ofRealLI : ℝ →ₗᵢ[ℝ] K where
  toLinearMap := ofRealAm.toLinearMap
  norm_map' := norm_ofReal


@[simp, rclike_simps]
theorem ofRealLI_apply : (ofRealLI : ℝ → K) = ofReal :=
  rfl


/-- The `ℝ → K` coercion, as a continuous linear map -/
noncomputable def ofRealCLM : ℝ →L[ℝ] K :=
  ofRealLI.toContinuousLinearMap


@[simp, rclike_simps]
theorem ofRealCLM_coe : (@ofRealCLM K _ : ℝ →ₗ[ℝ] K) = ofRealAm.toLinearMap :=
  rfl


@[simp, rclike_simps]
theorem ofRealCLM_apply : (ofRealCLM : ℝ → K) = ofReal :=
  rfl


@[continuity, fun_prop]
theorem continuous_ofReal : Continuous (ofReal : ℝ → K) :=
  ofRealLI.continuous


@[continuity]
theorem continuous_normSq : Continuous (normSq : K → ℝ) :=
  (continuous_re.mul continuous_re).add (continuous_im.mul continuous_im)


lemma im_eq_zero (h : I = (0 : K)) (z : K) : im z = 0 := by
  /-
    K : Type u_1
    inst✝ : RCLike K
    h : Eq RCLike.I 0
    z : K
    ⊢ Eq (RCLike.im z) 0
  -/
  rw [← re_add_im z, h]
  /-
    K : Type u_1
    inst✝ : RCLike K
    h : Eq RCLike.I 0
    z : K
    ⊢ Eq (RCLike.im (HAdd.hAdd (↑(RCLike.re z)) (HMul.hMul (↑(RCLike.im z)) 0))) 0
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The natural isomorphism between `𝕜` satisfying `RCLike 𝕜` and `ℝ` when `RCLike.I = 0`. -/
@[simps]
def realRingEquiv (h : I = (0 : K)) : K ≃+* ℝ where
  toFun := re
  invFun := (↑)
                   /-
                     K : Type u_1
                     E : Type u_2
                     inst✝ : RCLike K
                     h : Eq RCLike.I 0
                     x : K
                     ⊢ Eq (↑(RCLike.re x)) x
                   -/
  left_inv x := by nth_rw 2 [← re_add_im x]; simp [h]
                                             /-
                                               🎉 no goals
                                             -/
  right_inv := ofReal_re
  map_add' := map_add re
                 /-
                   K : Type u_1
                   E : Type u_2
                   inst✝ : RCLike K
                   h : Eq RCLike.I 0
                   ⊢ ∀ (x y : K), Eq ({ toFun := ⇑RCLike.re, invFun := RCLike.ofReal, left_inv := …
                 -/
  map_mul' := by simp [im_eq_zero h]
                 /-
                   🎉 no goals
                 -/


/-- The natural `ℝ`-linear isometry equivalence between `𝕜` satisfying `RCLike 𝕜` and `ℝ` when
`RCLike.I = 0`. -/
@[simps]
noncomputable def realLinearIsometryEquiv (h : I = (0 : K)) : K ≃ₗᵢ[ℝ] ℝ where
  map_smul' := smul_re
                    /-
                      K : Type u_1
                      E : Type u_2
                      inst✝ : RCLike K
                      h : Eq RCLike.I 0
                      z : K
                      ⊢ Eq (Norm.norm ({ toFun := __spread✝⁻⁰.toFun, map_add' := ⋯, map_smul' := ⋯,  …
                    -/
  norm_map' z := by rw [← re_add_im z]; simp [- re_add_im, h]
                                        /-
                                          🎉 no goals
                                        -/
  __ := realRingEquiv h


lemma inv_apply_eq_conj [AddLeftCancelMonoid G] (ψ : AddChar G K) (x : G) : (ψ x)⁻¹ = conj (ψ x) :=
  RCLike.inv_eq_conj <| norm_apply _ _


lemma map_neg_eq_conj [AddCommGroup G] (ψ : AddChar G K) (x : G) : ψ (-x) = conj (ψ x) := by
  /-
    K : Type u_1
    inst✝² : RCLike K
    G : Type u_3
    inst✝¹ : Finite G
    inst✝ : AddCommGroup G
    ψ : AddChar G K
    x : G
    ⊢ Eq (ψ (Neg.neg x)) ((starRingEnd K) (ψ x))
  -/
  rw [map_neg_eq_inv, inv_apply_eq_conj]
  /-
    🎉 no goals
  -/


/-- A mixin over a normed field, saying that the norm field structure is the same as `ℝ` or `ℂ`.
To endow such a field with a compatible `RCLike` structure in a proof, use
`letI := IsRCLikeNormedField.rclike 𝕜`.-/
class IsRCLikeNormedField (𝕜 : Type*) [hk : NormedField 𝕜] : Prop where
  out : ∃ h : RCLike 𝕜, hk = h.toNormedField


instance (priority := 100) (𝕜 : Type*) [h : RCLike 𝕜] : IsRCLikeNormedField 𝕜 := ⟨⟨h, rfl⟩⟩


/-- A copy of an `RCLike` field in which the `NormedField` field is adjusted to be become defeq
to a propeq one. -/
noncomputable def RCLike.copy_of_normedField {𝕜 : Type*} (h : RCLike 𝕜) (hk : NormedField 𝕜)
    (h'' : hk = h.toNormedField) : RCLike 𝕜 where
  __ := hk
  toPartialOrder := h.toPartialOrder
  toDecidableEq := h.toDecidableEq
                 /-
                   K : Type u_1
                   E : Type u_2
                   inst✝ : RCLike K
                   𝕜 : Type u_3
                   h : RCLike 𝕜
                   hk : NormedField 𝕜
                   h'' : Eq hk DenselyNormedField.toNormedField
                   ⊢ ∀ {f : Filter 𝕜}, Cauchy f → Exists fun x => LE.le f (nhds x)
                 -/
                   /-
                     K : Type u_1
                     E : Type u_2
                     inst✝ : RCLike K
                     𝕜 : Type u_3
                     h : RCLike 𝕜
                     hk : NormedField 𝕜
                     h'' : Eq hk DenselyNormedField.toNormedField
                     ⊢ ∀ (x y : Real), LE.le 0 x → LT.lt x y → Exists fun a => And (LT.lt x (Norm.n …
                   -/
  complete := by subst h''; exact h.complete
                              /-
                                🎉 no goals
                              -/
                            /-
                              🎉 no goals
                            -/
  lt_norm_lt := by subst h''; exact h.lt_norm_lt
                        /-
                          K : Type u_1
                          E : Type u_2
                          inst✝ : RCLike K
                          𝕜 : Type u_3
                          h : RCLike 𝕜
                          hk : NormedField 𝕜
                          h'' : Eq hk DenselyNormedField.toNormedField
                          ⊢ Function.Involutive Star.star
                        -/
  -- star fields
                                   /-
                                     🎉 no goals
                                   -/
                 /-
                   K : Type u_1
                   E : Type u_2
                   inst✝ : RCLike K
                   𝕜 : Type u_3
                   h : RCLike 𝕜
                   hk : NormedField 𝕜
                   h'' : Eq hk DenselyNormedField.toNormedField
                   ⊢ ∀ (r s : 𝕜), Eq (Star.star (HMul.hMul r s)) (HMul.hMul (Star.star s) (Star.s …
                 -/
  star := (@StarMul.toInvolutiveStar _ (_) (@StarRing.toStarMul _ (_) h.toStarRing)).star
                            /-
                              🎉 no goals
                            -/
                 /-
                   K : Type u_1
                   E : Type u_2
                   inst✝ : RCLike K
                   𝕜 : Type u_3
                   h : RCLike 𝕜
                   hk : NormedField 𝕜
                   h'' : Eq hk DenselyNormedField.toNormedField
                   ⊢ ∀ (r s : 𝕜), Eq (Star.star (HAdd.hAdd r s)) (HAdd.hAdd (Star.star r) (Star.s …
                 -/
  star_involutive := by subst h''; exact h.star_involutive
                            /-
                              🎉 no goals
                            -/
  star_mul := by subst h''; exact h.star_mul
  star_add := by subst h''; exact h.star_add
  -- algebra fields
                 /-
                   K : Type u_1
                   E : Type u_2
                   inst✝ : RCLike K
                   𝕜 : Type u_3
                   h : RCLike 𝕜
                   hk : NormedField 𝕜
                   h'' : Eq hk DenselyNormedField.toNormedField
                   ⊢ Eq (Algebra.toRingHom 1) 1
                 -/
  smul := (@Algebra.toSMul _ _ _ (_) (@NormedAlgebra.toAlgebra _ _ _ (_) h.toNormedAlgebra)).smul
                            /-
                              🎉 no goals
                            -/
                 /-
                   K : Type u_1
                   E : Type u_2
                   inst✝ : RCLike K
                   𝕜 : Type u_3
                   h : RCLike 𝕜
                   hk : NormedField 𝕜
                   h'' : Eq hk DenselyNormedField.toNormedField
                   ⊢ ∀ (x y : Real), Eq ({ toFun := ⇑Algebra.toRingHom, map_one' := ⋯ }.toFun (HM …
                 -/
  toFun := @Algebra.toRingHom _ _ _ (_) (@NormedAlgebra.toAlgebra _ _ _ (_) h.toNormedAlgebra)
                            /-
                              🎉 no goals
                            -/
                  /-
                    K : Type u_1
                    E : Type u_2
                    inst✝ : RCLike K
                    𝕜 : Type u_3
                    h : RCLike 𝕜
                    hk : NormedField 𝕜
                    h'' : Eq hk DenselyNormedField.toNormedField
                    ⊢ Eq ((↑{ toFun := ⇑Algebra.toRingHom, map_one' := ⋯, map_mul' := ⋯ }).toFun 0 …
                  -/
  map_one' := by subst h''; exact h.map_one'
                             /-
                               🎉 no goals
                             -/
                 /-
                   K : Type u_1
                   E : Type u_2
                   inst✝ : RCLike K
                   𝕜 : Type u_3
                   h : RCLike 𝕜
                   hk : NormedField 𝕜
                   h'' : Eq hk DenselyNormedField.toNormedField
                   ⊢ ∀ (x y : Real), Eq ((↑{ toFun := ⇑Algebra.toRingHom, map_one' := ⋯, map_mul' …
                 -/
  map_mul' := by subst h''; exact h.map_mul'
                            /-
                              🎉 no goals
                            -/
                  /-
                    K : Type u_1
                    E : Type u_2
                    inst✝ : RCLike K
                    𝕜 : Type u_3
                    h : RCLike 𝕜
                    hk : NormedField 𝕜
                    h'' : Eq hk DenselyNormedField.toNormedField
                    ⊢ ∀ (r : Real) (x : 𝕜), Eq (HMul.hMul ({ toFun := ⇑Algebra.toRingHom, map_one' …
                  -/
  map_zero' := by subst h''; exact h.map_zero'
                             /-
                               🎉 no goals
                             -/
                  /-
                    K : Type u_1
                    E : Type u_2
                    inst✝ : RCLike K
                    𝕜 : Type u_3
                    h : RCLike 𝕜
                    hk : NormedField 𝕜
                    h'' : Eq hk DenselyNormedField.toNormedField
                    ⊢ ∀ (r : Real) (x : 𝕜), Eq (HSMul.hSMul r x) (HMul.hMul ({ toFun := ⇑Algebra.t …
                  -/
  map_add' := by subst h''; exact h.map_add'
                             /-
                               🎉 no goals
                             -/
                     /-
                       K : Type u_1
                       E : Type u_2
                       inst✝ : RCLike K
                       𝕜 : Type u_3
                       h : RCLike 𝕜
                       hk : NormedField 𝕜
                       h'' : Eq hk DenselyNormedField.toNormedField
                       ⊢ ∀ (r : Real) (x : 𝕜), LE.le (Norm.norm (HSMul.hSMul r x)) (HMul.hMul (Norm.n …
                     -/
  commutes' := by subst h''; exact h.commutes'
                                /-
                                  🎉 no goals
                                -/
  smul_def' := by subst h''; exact h.smul_def'
  norm_smul_le := by subst h''; exact h.norm_smul_le
  -- RCLike fields
           /-
             K : Type u_1
             E : Type u_2
             inst✝ : RCLike K
             𝕜 : Type u_3
             h : RCLike 𝕜
             hk : NormedField 𝕜
             h'' : Eq hk DenselyNormedField.toNormedField
             ⊢ AddMonoidHom 𝕜 Real
           -/
  re := by subst h''; exact h.re
                      /-
                        🎉 no goals
                      -/
           /-
             K : Type u_1
             E : Type u_2
             inst✝ : RCLike K
             𝕜 : Type u_3
             h : RCLike 𝕜
             hk : NormedField 𝕜
             h'' : Eq hk DenselyNormedField.toNormedField
             ⊢ AddMonoidHom 𝕜 Real
           -/
  im := by subst h''; exact h.im
                      /-
                        🎉 no goals
                      -/
  I := h.I
                /-
                  K : Type u_1
                  E : Type u_2
                  inst✝ : RCLike K
                  𝕜 : Type u_3
                  h : RCLike 𝕜
                  hk : NormedField 𝕜
                  h'' : Eq hk DenselyNormedField.toNormedField
                  ⊢ Eq
                      ((Eq.rec
                          (let __spread.0 := DenselyNormedField.toNormedField;
                          RCLike.re)
                          ⋯)
                        RCLike.I)
                      0
                -/
  I_re_ax := by subst h''; exact h.I_re_ax
                           /-
                             🎉 no goals
                           -/
                   /-
                     K : Type u_1
                     E : Type u_2
                     inst✝ : RCLike K
                     𝕜 : Type u_3
                     h : RCLike 𝕜
                     hk : NormedField 𝕜
                     h'' : Eq hk DenselyNormedField.toNormedField
                     ⊢ Or (Eq RCLike.I 0) (Eq (HMul.hMul RCLike.I RCLike.I) (-1))
                   -/
  I_mul_I_ax := by subst h''; exact h.I_mul_I_ax
                              /-
                                🎉 no goals
                              -/
                     /-
                       K : Type u_1
                       E : Type u_2
                       inst✝ : RCLike K
                       𝕜 : Type u_3
                       h : RCLike 𝕜
                       hk : NormedField 𝕜
                       h'' : Eq hk DenselyNormedField.toNormedField
                       ⊢ ∀ (z : 𝕜),
                           Eq
                             (HAdd.hAdd
                               ((algebraMap Real 𝕜)
                                 ((Eq.rec
                                     (let __spread.0 := DenselyNormedField.toNormedField;
                                     RCLike.re)
                                     ⋯)
                                   z))
                               (HMul.hMul
                                 ((algebraMap Real 𝕜)
                                   ((Eq.rec
                                       (let __spread.0 := DenselyNormedField.toNormedField;
                                       RCLike.im)
                                       ⋯)
                                     z))
                                 RCLike.I))
                             z
                     -/
  re_add_im_ax := by subst h''; exact h.re_add_im_ax
                                /-
                                  🎉 no goals
                                -/
                     /-
                       K : Type u_1
                       E : Type u_2
                       inst✝ : RCLike K
                       𝕜 : Type u_3
                       h : RCLike 𝕜
                       hk : NormedField 𝕜
                       h'' : Eq hk DenselyNormedField.toNormedField
                       ⊢ ∀ (r : Real),
                           Eq
                             ((Eq.rec
                                 (let __spread.0 := DenselyNormedField.toNormedField;
                                 RCLike.re)
                                 ⋯)
                               ((algebraMap Real 𝕜) r))
                             r
                     -/
  ofReal_re_ax := by subst h''; exact h.ofReal_re_ax
                                /-
                                  🎉 no goals
                                -/
                     /-
                       K : Type u_1
                       E : Type u_2
                       inst✝ : RCLike K
                       𝕜 : Type u_3
                       h : RCLike 𝕜
                       hk : NormedField 𝕜
                       h'' : Eq hk DenselyNormedField.toNormedField
                       ⊢ ∀ (r : Real),
                           Eq
                             ((Eq.rec
                                 (let __spread.0 := DenselyNormedField.toNormedField;
                                 RCLike.im)
                                 ⋯)
                               ((algebraMap Real 𝕜) r))
                             0
                     -/
  ofReal_im_ax := by subst h''; exact h.ofReal_im_ax
                                /-
                                  🎉 no goals
                                -/
                  /-
                    K : Type u_1
                    E : Type u_2
                    inst✝ : RCLike K
                    𝕜 : Type u_3
                    h : RCLike 𝕜
                    hk : NormedField 𝕜
                    h'' : Eq hk DenselyNormedField.toNormedField
                    ⊢ ∀ (z w : 𝕜),
                        Eq
                          ((Eq.rec
                              (let __spread.0 := DenselyNormedField.toNormedField;
                              RCLike.re)
                              ⋯)
                            (HMul.hMul z w))
                          (HSub.hSub
                            (HMul.hMul
                              ((Eq.rec
                                  (let __spread.0 := DenselyNormedField.toNormedField;
                                  RCLike.re)
                                  ⋯)
                                z)
                              ((Eq.rec
                                  (let __spread.0 := DenselyNormedField.toNormedField;
                                  RCLike.re)
                                  ⋯)
                                w))
                            (HMul.hMul
                              ((Eq.rec
                                  (let __spread.0 := DenselyNormedField.toNormedField;
                                  RCLike.im)
                                  ⋯)
                                z)
                              ((Eq.rec
                                  (let __spread.0 := DenselyNormedField.toNormedField;
                                  RCLike.im)
                                  ⋯)
                                w)))
                  -/
  mul_re_ax := by subst h''; exact h.mul_re_ax
                             /-
                               🎉 no goals
                             -/
                  /-
                    K : Type u_1
                    E : Type u_2
                    inst✝ : RCLike K
                    𝕜 : Type u_3
                    h : RCLike 𝕜
                    hk : NormedField 𝕜
                    h'' : Eq hk DenselyNormedField.toNormedField
                    ⊢ ∀ (z w : 𝕜),
                        Eq
                          ((Eq.rec
                              (let __spread.0 := DenselyNormedField.toNormedField;
                              RCLike.im)
                              ⋯)
                            (HMul.hMul z w))
                          (HAdd.hAdd
                            (HMul.hMul
                              ((Eq.rec
                                  (let __spread.0 := DenselyNormedField.toNormedField;
                                  RCLike.re)
                                  ⋯)
                                z)
                              ((Eq.rec
                                  (let __spread.0 := DenselyNormedField.toNormedField;
                                  RCLike.im)
                                  ⋯)
                                w))
                            (HMul.hMul
                              ((Eq.rec
                                  (let __spread.0 := DenselyNormedField.toNormedField;
                                  RCLike.im)
                                  ⋯)
                                z)
                              ((Eq.rec
                                  (let __spread.0 := DenselyNormedField.toNormedField;
                                  RCLike.re)
                                  ⋯)
                                w)))
                  -/
  mul_im_ax := by subst h''; exact h.mul_im_ax
                             /-
                               🎉 no goals
                             -/
                   /-
                     K : Type u_1
                     E : Type u_2
                     inst✝ : RCLike K
                     𝕜 : Type u_3
                     h : RCLike 𝕜
                     hk : NormedField 𝕜
                     h'' : Eq hk DenselyNormedField.toNormedField
                     ⊢ ∀ (z : 𝕜),
                         Eq
                           ((Eq.rec
                               (let __spread.0 := DenselyNormedField.toNormedField;
                               RCLike.re)
                               ⋯)
                             ((starRingEnd 𝕜) z))
                           ((Eq.rec
                               (let __spread.0 := DenselyNormedField.toNormedField;
                               RCLike.re)
                               ⋯)
                             z)
                   -/
  conj_re_ax := by subst h''; exact h.conj_re_ax
                              /-
                                🎉 no goals
                              -/
                   /-
                     K : Type u_1
                     E : Type u_2
                     inst✝ : RCLike K
                     𝕜 : Type u_3
                     h : RCLike 𝕜
                     hk : NormedField 𝕜
                     h'' : Eq hk DenselyNormedField.toNormedField
                     ⊢ ∀ (z : 𝕜),
                         Eq
                           ((Eq.rec
                               (let __spread.0 := DenselyNormedField.toNormedField;
                               RCLike.im)
                               ⋯)
                             ((starRingEnd 𝕜) z))
                           (Neg.neg
                             ((Eq.rec
                                 (let __spread.0 := DenselyNormedField.toNormedField;
                                 RCLike.im)
                                 ⋯)
                               z))
                   -/
  conj_im_ax := by subst h''; exact h.conj_im_ax
                              /-
                                🎉 no goals
                              -/
                  /-
                    K : Type u_1
                    E : Type u_2
                    inst✝ : RCLike K
                    𝕜 : Type u_3
                    h : RCLike 𝕜
                    hk : NormedField 𝕜
                    h'' : Eq hk DenselyNormedField.toNormedField
                    ⊢ Eq ((starRingEnd 𝕜) RCLike.I) (Neg.neg RCLike.I)
                  -/
  conj_I_ax := by subst h''; exact h.conj_I_ax
                             /-
                               🎉 no goals
                             -/
                          /-
                            K : Type u_1
                            E : Type u_2
                            inst✝ : RCLike K
                            𝕜 : Type u_3
                            h : RCLike 𝕜
                            hk : NormedField 𝕜
                            h'' : Eq hk DenselyNormedField.toNormedField
                            ⊢ ∀ (z : 𝕜),
                                Eq (HPow.hPow (Norm.norm z) 2)
                                  (HAdd.hAdd
                                    (HMul.hMul
                                      ((Eq.rec
                                          (let __spread.0 := DenselyNormedField.toNormedField;
                                          RCLike.re)
                                          ⋯)
                                        z)
                                      ((Eq.rec
                                          (let __spread.0 := DenselyNormedField.toNormedField;
                                          RCLike.re)
                                          ⋯)
                                        z))
                                    (HMul.hMul
                                      ((Eq.rec
                                          (let __spread.0 := DenselyNormedField.toNormedField;
                                          RCLike.im)
                                          ⋯)
                                        z)
                                      ((Eq.rec
                                          (let __spread.0 := DenselyNormedField.toNormedField;
                                          RCLike.im)
                                          ⋯)
                                        z)))
                          -/
  norm_sq_eq_def_ax := by subst h''; exact h.norm_sq_eq_def_ax
                                     /-
                                       🎉 no goals
                                     -/
                    /-
                      K : Type u_1
                      E : Type u_2
                      inst✝ : RCLike K
                      𝕜 : Type u_3
                      h : RCLike 𝕜
                      hk : NormedField 𝕜
                      h'' : Eq hk DenselyNormedField.toNormedField
                      ⊢ ∀ (z : 𝕜),
                          Eq
                            (HMul.hMul
                              ((Eq.rec
                                  (let __spread.0 := DenselyNormedField.toNormedField;
                                  RCLike.im)
                                  ⋯)
                                z)
                              ((Eq.rec
                                  (let __spread.0 := DenselyNormedField.toNormedField;
                                  RCLike.im)
                                  ⋯)
                                RCLike.I))
                            ((Eq.rec
                                (let __spread.0 := DenselyNormedField.toNormedField;
                                RCLike.im)
                                ⋯)
                              z)
                    -/
  mul_im_I_ax := by subst h''; exact h.mul_im_I_ax
                               /-
                                 🎉 no goals
                               -/
                     /-
                       K : Type u_1
                       E : Type u_2
                       inst✝ : RCLike K
                       𝕜 : Type u_3
                       h : RCLike 𝕜
                       hk : NormedField 𝕜
                       h'' : Eq hk DenselyNormedField.toNormedField
                       ⊢ ∀ {z w : 𝕜},
                           Iff (LE.le z w)
                             (And
                               (LE.le
                                 ((Eq.rec
                                     (let __spread.0 := DenselyNormedField.toNormedField;
                                     RCLike.re)
                                     ⋯)
                                   z)
                                 ((Eq.rec
                                     (let __spread.0 := DenselyNormedField.toNormedField;
                                     RCLike.re)
                                     ⋯)
                                   w))
                               (Eq
                                 ((Eq.rec
                                     (let __spread.0 := DenselyNormedField.toNormedField;
                                     RCLike.im)
                                     ⋯)
                                   z)
                                 ((Eq.rec
                                     (let __spread.0 := DenselyNormedField.toNormedField;
                                     RCLike.im)
                                     ⋯)
                                   w)))
                     -/
  le_iff_re_im := by subst h''; exact h.le_iff_re_im
                                /-
                                  🎉 no goals
                                -/


/-- Given a normed field `𝕜` satisfying `IsRCLikeNormedField 𝕜`, build an associated `RCLike 𝕜`
structure on `𝕜` which is definitionally compatible with the given normed field structure. -/
noncomputable def IsRCLikeNormedField.rclike (𝕜 : Type*)
    [hk : NormedField 𝕜] [h : IsRCLikeNormedField 𝕜] : RCLike 𝕜 := by
  /-
    K : Type u_1
    E : Type u_2
    inst✝ : RCLike K
    𝕜 : Type u_3
    hk : NormedField 𝕜
    h : IsRCLikeNormedField 𝕜
    ⊢ RCLike 𝕜
  -/
  choose p hp using h.out
  /-
    K : Type u_1
    E : Type u_2
    inst✝ : RCLike K
    𝕜 : Type u_3
    hk : NormedField 𝕜
    h : IsRCLikeNormedField 𝕜
    p : RCLike 𝕜
    hp : Eq hk DenselyNormedField.toNormedField
    ⊢ RCLike 𝕜
  -/
  exact p.copy_of_normedField hk hp
  /-
    🎉 no goals
  -/


