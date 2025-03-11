/-- Predicate for the parallelogram identity to hold in a normed group. This is a scalar-less
version of `InnerProductSpace`. If you have an `InnerProductSpaceable` assumption, you can
locally upgrade that to `InnerProductSpace 𝕜 E` using `casesI nonempty_innerProductSpace 𝕜 E`.
-/
class InnerProductSpaceable : Prop where
  parallelogram_identity :
    ∀ x y : E, ‖x + y‖ * ‖x + y‖ + ‖x - y‖ * ‖x - y‖ = 2 * (‖x‖ * ‖x‖ + ‖y‖ * ‖y‖)


theorem InnerProductSpace.toInnerProductSpaceable [InnerProductSpace 𝕜 E] :
    InnerProductSpaceable E :=
  ⟨parallelogram_law_with_norm 𝕜⟩

-- See note [lower instance priority]

instance (priority := 100) InnerProductSpace.toInnerProductSpaceable_ofReal
    [InnerProductSpace ℝ E] : InnerProductSpaceable E :=
  ⟨parallelogram_law_with_norm ℝ⟩


local notation "𝓚" => algebraMap ℝ 𝕜


/-- Auxiliary definition of the inner product derived from the norm. -/
private noncomputable def inner_ (x y : E) : 𝕜 :=
  4⁻¹ * (𝓚 ‖x + y‖ * 𝓚 ‖x + y‖ - 𝓚 ‖x - y‖ * 𝓚 ‖x - y‖ +
    (I : 𝕜) * 𝓚 ‖(I : 𝕜) • x + y‖ * 𝓚 ‖(I : 𝕜) • x + y‖ -
    (I : 𝕜) * 𝓚 ‖(I : 𝕜) • x - y‖ * 𝓚 ‖(I : 𝕜) • x - y‖)


/-- Auxiliary definition for the `add_left` property. -/
private def innerProp' (r : 𝕜) : Prop :=
  ∀ x y : E, inner_ 𝕜 (r • x) y = conj r * inner_ 𝕜 x y


theorem _root_.Continuous.inner_ {f g : ℝ → E} (hf : Continuous f) (hg : Continuous g) :
    Continuous fun x => inner_ 𝕜 (f x) (g x) := by
  /-
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f g : Real → E
    hf : Continuous f
    hg : Continuous g
    ⊢ Continuous fun x => _root_.inner_ 𝕜 (f x) (g x)
  -/
  unfold _root_.inner_
  /-
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f g : Real → E
    hf : Continuous f
    hg : Continuous g
    ⊢ Continuous fun x => HMul.hMul (Inv.inv 4) (HSub.hSub (HAdd.hAdd (HSub.hSub ( …
  -/
  fun_prop
  /-
    🎉 no goals
  -/


theorem inner_.norm_sq (x : E) : ‖x‖ ^ 2 = re (inner_ 𝕜 x x) := by
  simp only [inner_, normSq_apply, ofNat_re, ofNat_im, map_sub, map_add, map_zero, map_mul,
    ofReal_re, ofReal_im, mul_re, inv_re, mul_im, I_re, inv_im]
  /-
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    ⊢ Eq (HPow.hPow (Norm.norm x) 2) (HSub.hSub (HMul.hMul (HDiv.hDiv 4 (HAdd.hAdd …
  -/
  have h₁ : ‖x - x‖ = 0 := by simp
  /-
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    h₁ : Eq (Norm.norm (HSub.hSub x x)) 0
    ⊢ Eq (HPow.hPow (Norm.norm x) 2) (HSub.hSub (HMul.hMul (HDiv.hDiv 4 (HAdd.hAdd …
  -/
  have h₂ : ‖x + x‖ = 2 • ‖x‖ := by convert norm_nsmul 𝕜 2 x using 2; module
  /-
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    h₁ : Eq (Norm.norm (HSub.hSub x x)) 0
    h₂ : Eq (Norm.norm (HAdd.hAdd x x)) (HSMul.hSMul 2 (Norm.norm x))
    ⊢ Eq (HPow.hPow (Norm.norm x) 2) (HSub.hSub (HMul.hMul (HDiv.hDiv 4 (HAdd.hAdd …
  -/
  rw [h₁, h₂]
  /-
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    h₁ : Eq (Norm.norm (HSub.hSub x x)) 0
    h₂ : Eq (Norm.norm (HAdd.hAdd x x)) (HSMul.hSMul 2 (Norm.norm x))
    ⊢ Eq (HPow.hPow (Norm.norm x) 2) (HSub.hSub (HMul.hMul (HDiv.hDiv 4 (HAdd.hAdd …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem inner_.conj_symm (x y : E) : conj (inner_ 𝕜 y x) = inner_ 𝕜 x y := by
  /-
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x y : E
    ⊢ Eq ((starRingEnd 𝕜) (inner_ 𝕜 y x)) (inner_ 𝕜 x y)
  -/
  simp only [inner_, map_sub, map_add, map_mul, map_inv₀, map_ofNat, conj_ofReal, conj_I]
  /-
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x y : E
    ⊢ Eq (HMul.hMul (Inv.inv 4) (HSub.hSub (HAdd.hAdd (HSub.hSub (HMul.hMul ↑(Norm …
  -/
  rw [add_comm y x, norm_sub_rev]
  /-
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x y : E
    ⊢ Eq (HMul.hMul (Inv.inv 4) (HSub.hSub (HAdd.hAdd (HSub.hSub (HMul.hMul ↑(Norm …
  -/
  by_cases hI : (I : 𝕜) = 0
    /-
      case pos
      𝕜 : Type u_1
      inst✝² : RCLike 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      x y : E
      hI : Eq RCLike.I 0
      ⊢ Eq (HMul.hMul (Inv.inv 4) (HSub.hSub (HAdd.hAdd (HSub.hSub (HMul.hMul ↑(Norm …
    -/
  · simp only [hI, neg_zero, zero_mul]
    /-
      🎉 no goals
    -/
  /-
    case neg
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x y : E
    hI : Not (Eq RCLike.I 0)
    ⊢ Eq (HMul.hMul (Inv.inv 4) (HSub.hSub (HAdd.hAdd (HSub.hSub (HMul.hMul ↑(Norm …
  -/
  have hI' := I_mul_I_of_nonzero hI
  /-
    case neg
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x y : E
    hI : Not (Eq RCLike.I 0)
    hI' : Eq (HMul.hMul RCLike.I RCLike.I) (-1)
    ⊢ Eq (HMul.hMul (Inv.inv 4) (HSub.hSub (HAdd.hAdd (HSub.hSub (HMul.hMul ↑(Norm …
  -/
  have I_smul (v : E) : ‖(I : 𝕜) • v‖ = ‖v‖ := by rw [norm_smul, norm_I_of_ne_zero hI, one_mul]
  have h₁ : ‖(I : 𝕜) • y - x‖ = ‖(I : 𝕜) • x + y‖ := by
    convert I_smul ((I : 𝕜) • x + y) using 2
    linear_combination (norm := module) -hI' • x
  have h₂ : ‖(I : 𝕜) • y + x‖ = ‖(I : 𝕜) • x - y‖ := by
    convert (I_smul ((I : 𝕜) • y + x)).symm using 2
    linear_combination (norm := module) -hI' • y
  /-
    case neg
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x y : E
    hI : Not (Eq RCLike.I 0)
    hI' : Eq (HMul.hMul RCLike.I RCLike.I) (-1)
    I_smul : ∀ (v : E), Eq (Norm.norm (HSMul.hSMul RCLike.I v)) (Norm.norm v)
    h₁ : Eq (Norm.norm (HSub.hSub (HSMul.hSMul RCLike.I y) x)) (Norm.norm (HAdd.hA …
    h₂ : Eq (Norm.norm (HAdd.hAdd (HSMul.hSMul RCLike.I y) x)) (Norm.norm (HSub.hS …
    ⊢ Eq (HMul.hMul (Inv.inv 4) (HSub.hSub (HAdd.hAdd (HSub.hSub (HMul.hMul ↑(Norm …
  -/
  rw [h₁, h₂]
  /-
    case neg
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x y : E
    hI : Not (Eq RCLike.I 0)
    hI' : Eq (HMul.hMul RCLike.I RCLike.I) (-1)
    I_smul : ∀ (v : E), Eq (Norm.norm (HSMul.hSMul RCLike.I v)) (Norm.norm v)
    h₁ : Eq (Norm.norm (HSub.hSub (HSMul.hSMul RCLike.I y) x)) (Norm.norm (HAdd.hA …
    h₂ : Eq (Norm.norm (HAdd.hAdd (HSMul.hSMul RCLike.I y) x)) (Norm.norm (HSub.hS …
    ⊢ Eq (HMul.hMul (Inv.inv 4) (HSub.hSub (HAdd.hAdd (HSub.hSub (HMul.hMul ↑(Norm …
  -/
  ring
  /-
    🎉 no goals
  -/


private theorem add_left_aux1 (x y z : E) :
    ‖2 • x + y‖ * ‖2 • x + y‖ + ‖2 • z + y‖ * ‖2 • z + y‖
    = 2 * (‖x + y + z‖ * ‖x + y + z‖ + ‖x - z‖ * ‖x - z‖) := by
  /-
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpaceable E
    x y z : E
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Norm.norm (HAdd.hAdd (HSMul.hSMul 2 x) y)) (Norm.n …
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
                                                                 /-
                                                                   🎉 no goals
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
  convert parallelogram_identity (x + y + z) (x - z) using 4 <;> abel
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


private theorem add_left_aux2 (x y z : E) : ‖2 • x + y‖ * ‖2 • x + y‖ + ‖y - 2 • z‖ * ‖y - 2 • z‖
    = 2 * (‖x + y - z‖ * ‖x + y - z‖ + ‖x + z‖ * ‖x + z‖) := by
  /-
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpaceable E
    x y z : E
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Norm.norm (HAdd.hAdd (HSMul.hSMul 2 x) y)) (Norm.n …
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
                                                                 /-
                                                                   🎉 no goals
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
  convert parallelogram_identity (x + y - z) (x + z) using 4 <;> abel
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


private theorem add_left_aux3 (y z : E) :
    ‖2 • z + y‖ * ‖2 • z + y‖ + ‖y‖ * ‖y‖ = 2 * (‖y + z‖ * ‖y + z‖ + ‖z‖ * ‖z‖) := by
  /-
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpaceable E
    y z : E
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Norm.norm (HAdd.hAdd (HSMul.hSMul 2 z) y)) (Norm.n …
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
                                                       /-
                                                         🎉 no goals
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
  convert parallelogram_identity (y + z) z using 4 <;> abel
                                                       /-
                                                         🎉 no goals
                                                       -/


private theorem add_left_aux4 (y z : E) :
    ‖y‖ * ‖y‖ + ‖y - 2 • z‖ * ‖y - 2 • z‖ = 2 * (‖y - z‖ * ‖y - z‖ + ‖z‖ * ‖z‖) := by
  /-
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpaceable E
    y z : E
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Norm.norm y) (Norm.norm y)) (HMul.hMul (Norm.norm  …
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
                                                       /-
                                                         🎉 no goals
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
  convert parallelogram_identity (y - z) z using 4 <;> abel
                                                       /-
                                                         🎉 no goals
                                                       -/


private theorem add_left_aux5 (x y z : E) :
    ‖(I : 𝕜) • (2 • x + y)‖ * ‖(I : 𝕜) • (2 • x + y)‖
    + ‖(I : 𝕜) • y + 2 • z‖ * ‖(I : 𝕜) • y + 2 • z‖
    = 2 * (‖(I : 𝕜) • (x + y) + z‖ * ‖(I : 𝕜) • (x + y) + z‖
    + ‖(I : 𝕜) • x - z‖ * ‖(I : 𝕜) • x - z‖) := by
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : InnerProductSpaceable E
    x y z : E
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Norm.norm (HSMul.hSMul RCLike.I (HAdd.hAdd (HSMul. …
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
  convert parallelogram_identity ((I : 𝕜) • (x + y) + z) ((I : 𝕜) • x - z) using 4 <;> module
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/


private theorem add_left_aux6 (x y z : E) :
    (‖(I : 𝕜) • (2 • x + y)‖ * ‖(I : 𝕜) • (2 • x + y)‖ +
    ‖(I : 𝕜) • y - 2 • z‖ * ‖(I : 𝕜) • y - 2 • z‖)
    = 2 * (‖(I : 𝕜) • (x + y) - z‖ * ‖(I : 𝕜) • (x + y) - z‖ +
    ‖(I : 𝕜) • x + z‖ * ‖(I : 𝕜) • x + z‖) := by
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : InnerProductSpaceable E
    x y z : E
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Norm.norm (HSMul.hSMul RCLike.I (HAdd.hAdd (HSMul. …
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
  convert parallelogram_identity ((I : 𝕜) • (x + y) - z) ((I : 𝕜) • x + z) using 4 <;> module
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/


private theorem add_left_aux7 (y z : E) :
    ‖(I : 𝕜) • y + 2 • z‖ * ‖(I : 𝕜) • y + 2 • z‖ + ‖(I : 𝕜) • y‖ * ‖(I : 𝕜) • y‖ =
    2 * (‖(I : 𝕜) • y + z‖ * ‖(I : 𝕜) • y + z‖ + ‖z‖ * ‖z‖) := by
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : InnerProductSpaceable E
    y z : E
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Norm.norm (HAdd.hAdd (HSMul.hSMul RCLike.I y) (HSM …
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
  convert parallelogram_identity ((I : 𝕜) • y + z) z using 4 <;> module
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


private theorem add_left_aux8 (y z : E) :
    ‖(I : 𝕜) • y‖ * ‖(I : 𝕜) • y‖ + ‖(I : 𝕜) • y - 2 • z‖ * ‖(I : 𝕜) • y - 2 • z‖ =
    2 * (‖(I : 𝕜) • y - z‖ * ‖(I : 𝕜) • y - z‖ + ‖z‖ * ‖z‖) := by
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : InnerProductSpaceable E
    y z : E
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Norm.norm (HSMul.hSMul RCLike.I y)) (Norm.norm (HS …
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
  convert parallelogram_identity ((I : 𝕜) • y - z) z using 4 <;> module
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


theorem add_left (x y z : E) : inner_ 𝕜 (x + y) z = inner_ 𝕜 x z + inner_ 𝕜 y z := by
  have H_re := congr(- $(add_left_aux1 x y z) + $(add_left_aux2 x y z)
    + $(add_left_aux3 y z) - $(add_left_aux4 y z))
  have H_im := congr(- $(add_left_aux5 𝕜 x y z) + $(add_left_aux6 𝕜 x y z)
      + $(add_left_aux7 𝕜 y z) - $(add_left_aux8 𝕜 y z))
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : InnerProductSpaceable E
    x y z : E
    H_re : Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (Neg.neg (HAdd.hAdd (HMul.hMul (Nor …
    H_im : Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (Neg.neg (HAdd.hAdd (HMul.hMul (Nor …
    ⊢ Eq (inner_ 𝕜 (HAdd.hAdd x y) z) (HAdd.hAdd (inner_ 𝕜 x z) (inner_ 𝕜 y z))
  -/
  have H := congr(𝓚 $H_re + I * 𝓚 $H_im)
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : InnerProductSpaceable E
    x y z : E
    H_re : Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (Neg.neg (HAdd.hAdd (HMul.hMul (Nor …
    H_im : Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (Neg.neg (HAdd.hAdd (HMul.hMul (Nor …
    H : Eq (HAdd.hAdd ((algebraMap Real 𝕜) (HSub.hSub (HAdd.hAdd (HAdd.hAdd (Neg.n …
    ⊢ Eq (inner_ 𝕜 (HAdd.hAdd x y) z) (HAdd.hAdd (inner_ 𝕜 x z) (inner_ 𝕜 y z))
  -/
  simp only [inner_, map_add, map_sub, map_neg, map_mul, map_ofNat] at H ⊢
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : InnerProductSpaceable E
    x y z : E
    H_re : Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (Neg.neg (HAdd.hAdd (HMul.hMul (Nor …
    H_im : Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (Neg.neg (HAdd.hAdd (HMul.hMul (Nor …
    H : Eq (HAdd.hAdd (HSub.hSub (HAdd.hAdd (HAdd.hAdd (Neg.neg (HAdd.hAdd (HMul.h …
    ⊢ Eq (HMul.hMul (Inv.inv 4) (HSub.hSub (HAdd.hAdd (HSub.hSub (HMul.hMul ((alge …
  -/
  linear_combination H / 8
  /-
    🎉 no goals
  -/


private theorem rat_prop (r : ℚ) : innerProp' E (r : 𝕜) := by
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : InnerProductSpaceable E
    r : Rat
    ⊢ InnerProductSpaceable.innerProp' E ↑r
  -/
  intro x y
  let hom : 𝕜 →ₗ[ℚ] 𝕜 := AddMonoidHom.toRatLinearMap <|
    AddMonoidHom.mk' (fun r ↦ inner_ 𝕜 (r • x) y) <| fun a b ↦ by
      simpa [add_smul] using add_left (a • x) (b • x) y
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : InnerProductSpaceable E
    r : Rat
    x y : E
    hom : LinearMap (RingHom.id Rat) 𝕜 𝕜 := (AddMonoidHom.mk' (fun r => inner_ 𝕜 ( …
    ⊢ Eq (inner_ 𝕜 (HSMul.hSMul (↑r) x) y) (HMul.hMul ((starRingEnd 𝕜) ↑r) (inner_ …
  -/
  simpa [hom, Rat.smul_def] using map_smul hom r 1
  /-
    🎉 no goals
  -/


private theorem real_prop (r : ℝ) : innerProp' E (r : 𝕜) := by
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : InnerProductSpaceable E
    r : Real
    ⊢ InnerProductSpaceable.innerProp' E ↑r
  -/
  intro x y
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : InnerProductSpaceable E
    r : Real
    x y : E
    ⊢ Eq (inner_ 𝕜 (HSMul.hSMul (↑r) x) y) (HMul.hMul ((starRingEnd 𝕜) ↑r) (inner_ …
  -/
  revert r
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : InnerProductSpaceable E
    x y : E
    ⊢ ∀ (r : Real), Eq (inner_ 𝕜 (HSMul.hSMul (↑r) x) y) (HMul.hMul ((starRingEnd  …
  -/
  rw [← funext_iff]
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : InnerProductSpaceable E
    x y : E
    ⊢ Eq (fun r => inner_ 𝕜 (HSMul.hSMul (↑r) x) y) fun r => HMul.hMul ((starRingE …
  -/
  refine Rat.isDenseEmbedding_coe_real.dense.equalizer ?_ ?_ (funext fun X => ?_)
    /-
      case refine_1
      𝕜 : Type u_1
      inst✝³ : RCLike 𝕜
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : InnerProductSpaceable E
      x y : E
      ⊢ Continuous fun r => inner_ 𝕜 (HSMul.hSMul (↑r) x) y
    -/
  · exact (continuous_ofReal.smul continuous_const).inner_ continuous_const
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      𝕜 : Type u_1
      inst✝³ : RCLike 𝕜
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : InnerProductSpaceable E
      x y : E
      ⊢ Continuous fun r => HMul.hMul ((starRingEnd 𝕜) ↑r) (inner_ 𝕜 x y)
    -/
  · exact (continuous_conj.comp continuous_ofReal).mul continuous_const
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      𝕜 : Type u_1
      inst✝³ : RCLike 𝕜
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : InnerProductSpaceable E
      x y : E
      X : Rat
      ⊢ Eq (Function.comp (fun r => inner_ 𝕜 (HSMul.hSMul (↑r) x) y) Rat.cast X) (Fu …
    -/
  · simp only [Function.comp_apply, RCLike.ofReal_ratCast, rat_prop _ _]
    /-
      🎉 no goals
    -/


private theorem I_prop : innerProp' E (I : 𝕜) := by
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : InnerProductSpaceable E
    ⊢ InnerProductSpaceable.innerProp' E RCLike.I
  -/
  by_cases hI : (I : 𝕜) = 0
    /-
      case pos
      𝕜 : Type u_1
      inst✝³ : RCLike 𝕜
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : InnerProductSpaceable E
      hI : Eq RCLike.I 0
      ⊢ InnerProductSpaceable.innerProp' E RCLike.I
    -/
  · rw [hI]
    /-
      case pos
      𝕜 : Type u_1
      inst✝³ : RCLike 𝕜
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : InnerProductSpaceable E
      hI : Eq RCLike.I 0
      ⊢ InnerProductSpaceable.innerProp' E 0
    -/
    simpa using real_prop (𝕜 := 𝕜) 0
    /-
      🎉 no goals
    -/
  /-
    case neg
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : InnerProductSpaceable E
    hI : Not (Eq RCLike.I 0)
    ⊢ InnerProductSpaceable.innerProp' E RCLike.I
  -/
  intro x y
  /-
    case neg
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : InnerProductSpaceable E
    hI : Not (Eq RCLike.I 0)
    x y : E
    ⊢ Eq (inner_ 𝕜 (HSMul.hSMul RCLike.I x) y) (HMul.hMul ((starRingEnd 𝕜) RCLike. …
  -/
  have hI' := I_mul_I_of_nonzero hI
  /-
    case neg
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : InnerProductSpaceable E
    hI : Not (Eq RCLike.I 0)
    x y : E
    hI' : Eq (HMul.hMul RCLike.I RCLike.I) (-1)
    ⊢ Eq (inner_ 𝕜 (HSMul.hSMul RCLike.I x) y) (HMul.hMul ((starRingEnd 𝕜) RCLike. …
  -/
  rw [conj_I, inner_, inner_, mul_left_comm, smul_smul, hI', neg_one_smul]
  /-
    case neg
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : InnerProductSpaceable E
    hI : Not (Eq RCLike.I 0)
    x y : E
    hI' : Eq (HMul.hMul RCLike.I RCLike.I) (-1)
    ⊢ Eq (HMul.hMul (Inv.inv 4) (HSub.hSub (HAdd.hAdd (HSub.hSub (HMul.hMul ((alge …
  -/
  have h₁ : ‖-x - y‖ = ‖x + y‖ := by rw [← neg_add', norm_neg]
  /-
    case neg
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : InnerProductSpaceable E
    hI : Not (Eq RCLike.I 0)
    x y : E
    hI' : Eq (HMul.hMul RCLike.I RCLike.I) (-1)
    h₁ : Eq (Norm.norm (HSub.hSub (Neg.neg x) y)) (Norm.norm (HAdd.hAdd x y))
    ⊢ Eq (HMul.hMul (Inv.inv 4) (HSub.hSub (HAdd.hAdd (HSub.hSub (HMul.hMul ((alge …
  -/
  have h₂ : ‖-x + y‖ = ‖x - y‖ := by rw [← neg_sub, norm_neg, sub_eq_neg_add]
  /-
    case neg
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : InnerProductSpaceable E
    hI : Not (Eq RCLike.I 0)
    x y : E
    hI' : Eq (HMul.hMul RCLike.I RCLike.I) (-1)
    h₁ : Eq (Norm.norm (HSub.hSub (Neg.neg x) y)) (Norm.norm (HAdd.hAdd x y))
    h₂ : Eq (Norm.norm (HAdd.hAdd (Neg.neg x) y)) (Norm.norm (HSub.hSub x y))
    ⊢ Eq (HMul.hMul (Inv.inv 4) (HSub.hSub (HAdd.hAdd (HSub.hSub (HMul.hMul ((alge …
  -/
  rw [h₁, h₂]
  /-
    case neg
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : InnerProductSpaceable E
    hI : Not (Eq RCLike.I 0)
    x y : E
    hI' : Eq (HMul.hMul RCLike.I RCLike.I) (-1)
    h₁ : Eq (Norm.norm (HSub.hSub (Neg.neg x) y)) (Norm.norm (HAdd.hAdd x y))
    h₂ : Eq (Norm.norm (HAdd.hAdd (Neg.neg x) y)) (Norm.norm (HSub.hSub x y))
    ⊢ Eq (HMul.hMul (Inv.inv 4) (HSub.hSub (HAdd.hAdd (HSub.hSub (HMul.hMul ((alge …
  -/
  linear_combination (- 𝓚 ‖(I : 𝕜) • x - y‖ ^ 2 + 𝓚 ‖(I : 𝕜) • x + y‖ ^ 2) * hI' / 4
  /-
    🎉 no goals
  -/


theorem innerProp (r : 𝕜) : innerProp' E r := by
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : InnerProductSpaceable E
    r : 𝕜
    ⊢ InnerProductSpaceable.innerProp' E r
  -/
  intro x y
  rw [← re_add_im r, add_smul, add_left, real_prop _ x, ← smul_smul, real_prop _ _ y, I_prop,
    map_add, map_mul, conj_ofReal, conj_ofReal, conj_I]
  /-
    𝕜 : Type u_1
    inst✝³ : RCLike 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : InnerProductSpaceable E
    r : 𝕜
    x y : E
    ⊢ Eq (HAdd.hAdd (HMul.hMul (↑(RCLike.re r)) (inner_ 𝕜 x y)) (HMul.hMul (↑(RCLi …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- **Fréchet–von Neumann–Jordan Theorem**. A normed space `E` whose norm satisfies the
parallelogram identity can be given a compatible inner product. -/
noncomputable def InnerProductSpace.ofNorm
    (h : ∀ x y : E, ‖x + y‖ * ‖x + y‖ + ‖x - y‖ * ‖x - y‖ = 2 * (‖x‖ * ‖x‖ + ‖y‖ * ‖y‖)) :
    InnerProductSpace 𝕜 E :=
  haveI : InnerProductSpaceable E := ⟨h⟩
  { inner := inner_ 𝕜
    norm_sq_eq_inner := inner_.norm_sq
    conj_symm := inner_.conj_symm
    add_left := InnerProductSpaceable.add_left
    smul_left := fun _ _ _ => innerProp _ _ _ }


/-- **Fréchet–von Neumann–Jordan Theorem**. A normed space `E` whose norm satisfies the
parallelogram identity can be given a compatible inner product. Do
`casesI nonempty_innerProductSpace 𝕜 E` to locally upgrade `InnerProductSpaceable E` to
`InnerProductSpace 𝕜 E`. -/
theorem nonempty_innerProductSpace : Nonempty (InnerProductSpace 𝕜 E) :=
  ⟨{  inner := inner_ 𝕜
      norm_sq_eq_inner := inner_.norm_sq
      conj_symm := inner_.conj_symm
      add_left := add_left
      smul_left := fun _ _ _ => innerProp _ _ _ }⟩


instance (priority := 100) InnerProductSpaceable.to_uniformConvexSpace : UniformConvexSpace E := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : RCLike 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : InnerProductSpaceable E
    inst✝ : NormedSpace Real E
    ⊢ UniformConvexSpace E
  -/
  cases nonempty_innerProductSpace ℝ E; infer_instance
                                        /-
                                          🎉 no goals
                                        -/

