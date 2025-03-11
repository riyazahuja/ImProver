/-- **Hahn-Banach theorem** for continuous linear functions over `ℝ`.
See also `exists_extension_norm_eq` in the root namespace for a more general version
that works both for `ℝ` and `ℂ`. -/
theorem exists_extension_norm_eq (p : Subspace ℝ E) (f : p →L[ℝ] ℝ) :
    ∃ g : E →L[ℝ] ℝ, (∀ x : p, g x = f x) ∧ ‖g‖ = ‖f‖ := by
  rcases exists_extension_of_le_sublinear ⟨p, f⟩ (fun x => ‖f‖ * ‖x‖)
      (fun c hc x => by simp only [norm_smul c x, Real.norm_eq_abs, abs_of_pos hc, mul_left_comm])
      (fun x y => by -- Porting note: placeholder filled here
        rw [← left_distrib]
        exact mul_le_mul_of_nonneg_left (norm_add_le x y) (@norm_nonneg _ _ f))
      fun x => le_trans (le_abs_self _) (f.le_opNorm _) with ⟨g, g_eq, g_le⟩
  set g' :=
    g.mkContinuous ‖f‖ fun x => abs_le.2 ⟨neg_le.1 <| g.map_neg x ▸ norm_neg x ▸ g_le (-x), g_le x⟩
  /-
    case intro.intro
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    p : Subspace Real E
    f : ContinuousLinearMap (RingHom.id Real) (Subtype fun x => Membership.mem p x …
    g : LinearMap (RingHom.id Real) E Real
    g_eq : ∀ (x : Subtype fun x => Membership.mem { domain := p, toFun := ↑f }.dom …
    g_le : ∀ (x : E), LE.le (g x) (HMul.hMul (Norm.norm f) (Norm.norm x))
    g' : ContinuousLinearMap (RingHom.id Real) E Real := g.mkContinuous (Norm.norm …
    ⊢ Exists fun g => And (∀ (x : Subtype fun x => Membership.mem p x), Eq (g ↑x)  …
  -/
  refine ⟨g', g_eq, ?_⟩
  /-
    case intro.intro
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    p : Subspace Real E
    f : ContinuousLinearMap (RingHom.id Real) (Subtype fun x => Membership.mem p x …
    g : LinearMap (RingHom.id Real) E Real
    g_eq : ∀ (x : Subtype fun x => Membership.mem { domain := p, toFun := ↑f }.dom …
    g_le : ∀ (x : E), LE.le (g x) (HMul.hMul (Norm.norm f) (Norm.norm x))
    g' : ContinuousLinearMap (RingHom.id Real) E Real := g.mkContinuous (Norm.norm …
    ⊢ Eq (Norm.norm g') (Norm.norm f)
  -/
  apply le_antisymm (g.mkContinuous_norm_le (norm_nonneg f) _)
  /-
    case intro.intro
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    p : Subspace Real E
    f : ContinuousLinearMap (RingHom.id Real) (Subtype fun x => Membership.mem p x …
    g : LinearMap (RingHom.id Real) E Real
    g_eq : ∀ (x : Subtype fun x => Membership.mem { domain := p, toFun := ↑f }.dom …
    g_le : ∀ (x : E), LE.le (g x) (HMul.hMul (Norm.norm f) (Norm.norm x))
    g' : ContinuousLinearMap (RingHom.id Real) E Real := g.mkContinuous (Norm.norm …
    ⊢ LE.le (Norm.norm f) (Norm.norm (g.mkContinuous (Norm.norm f) ⋯))
  -/
  refine f.opNorm_le_bound (norm_nonneg _) fun x => ?_
  /-
    case intro.intro
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    p : Subspace Real E
    f : ContinuousLinearMap (RingHom.id Real) (Subtype fun x => Membership.mem p x …
    g : LinearMap (RingHom.id Real) E Real
    g_eq : ∀ (x : Subtype fun x => Membership.mem { domain := p, toFun := ↑f }.dom …
    g_le : ∀ (x : E), LE.le (g x) (HMul.hMul (Norm.norm f) (Norm.norm x))
    g' : ContinuousLinearMap (RingHom.id Real) E Real := g.mkContinuous (Norm.norm …
    x : Subtype fun x => Membership.mem p x
    ⊢ LE.le (Norm.norm (f x)) (HMul.hMul (Norm.norm (g.mkContinuous (Norm.norm f)  …
  -/
  dsimp at g_eq
  /-
    case intro.intro
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    p : Subspace Real E
    f : ContinuousLinearMap (RingHom.id Real) (Subtype fun x => Membership.mem p x …
    g : LinearMap (RingHom.id Real) E Real
    g_eq : ∀ (x : Subtype fun x => Membership.mem p x), Eq (g ↑x) (f x)
    g_le : ∀ (x : E), LE.le (g x) (HMul.hMul (Norm.norm f) (Norm.norm x))
    g' : ContinuousLinearMap (RingHom.id Real) E Real := g.mkContinuous (Norm.norm …
    x : Subtype fun x => Membership.mem p x
    ⊢ LE.le (Norm.norm (f x)) (HMul.hMul (Norm.norm (g.mkContinuous (Norm.norm f)  …
  -/
  rw [← g_eq]
  /-
    case intro.intro
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    p : Subspace Real E
    f : ContinuousLinearMap (RingHom.id Real) (Subtype fun x => Membership.mem p x …
    g : LinearMap (RingHom.id Real) E Real
    g_eq : ∀ (x : Subtype fun x => Membership.mem p x), Eq (g ↑x) (f x)
    g_le : ∀ (x : E), LE.le (g x) (HMul.hMul (Norm.norm f) (Norm.norm x))
    g' : ContinuousLinearMap (RingHom.id Real) E Real := g.mkContinuous (Norm.norm …
    x : Subtype fun x => Membership.mem p x
    ⊢ LE.le (Norm.norm (g ↑x)) (HMul.hMul (Norm.norm (g.mkContinuous (Norm.norm f) …
  -/
  apply g'.le_opNorm
  /-
    🎉 no goals
  -/


/-- **Hahn-Banach theorem** for continuous linear functions over `𝕜`
satisfying `IsRCLikeNormedField 𝕜`. -/
theorem exists_extension_norm_eq (p : Subspace 𝕜 E) (f : p →L[𝕜] 𝕜) :
    ∃ g : E →L[𝕜] 𝕜, (∀ x : p, g x = f x) ∧ ‖g‖ = ‖f‖ := by
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : IsRCLikeNormedField 𝕜
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : Subspace 𝕜 E
    f : ContinuousLinearMap (RingHom.id 𝕜) (Subtype fun x => Membership.mem p x) 𝕜
    ⊢ Exists fun g => And (∀ (x : Subtype fun x => Membership.mem p x), Eq (g ↑x)  …
  -/
  letI : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : IsRCLikeNormedField 𝕜
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : Subspace 𝕜 E
    f : ContinuousLinearMap (RingHom.id 𝕜) (Subtype fun x => Membership.mem p x) 𝕜
    this : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    ⊢ Exists fun g => And (∀ (x : Subtype fun x => Membership.mem p x), Eq (g ↑x)  …
  -/
  letI : Module ℝ E := RestrictScalars.module ℝ 𝕜 E
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : IsRCLikeNormedField 𝕜
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : Subspace 𝕜 E
    f : ContinuousLinearMap (RingHom.id 𝕜) (Subtype fun x => Membership.mem p x) 𝕜
    this✝ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    this : Module Real E := RestrictScalars.module Real 𝕜 E
    ⊢ Exists fun g => And (∀ (x : Subtype fun x => Membership.mem p x), Eq (g ↑x)  …
  -/
  letI : IsScalarTower ℝ 𝕜 E := RestrictScalars.isScalarTower _ _ _
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : IsRCLikeNormedField 𝕜
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : Subspace 𝕜 E
    f : ContinuousLinearMap (RingHom.id 𝕜) (Subtype fun x => Membership.mem p x) 𝕜
    this✝¹ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    this✝ : Module Real E := RestrictScalars.module Real 𝕜 E
    this : IsScalarTower Real 𝕜 E := RestrictScalars.isScalarTower Real 𝕜 E
    ⊢ Exists fun g => And (∀ (x : Subtype fun x => Membership.mem p x), Eq (g ↑x)  …
  -/
  letI : NormedSpace ℝ E := NormedSpace.restrictScalars _ 𝕜 _
  -- Let `fr: p →L[ℝ] ℝ` be the real part of `f`.
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : IsRCLikeNormedField 𝕜
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : Subspace 𝕜 E
    f : ContinuousLinearMap (RingHom.id 𝕜) (Subtype fun x => Membership.mem p x) 𝕜
    this✝² : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    this✝¹ : Module Real E := RestrictScalars.module Real 𝕜 E
    this✝ : IsScalarTower Real 𝕜 E := RestrictScalars.isScalarTower Real 𝕜 E
    this : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
    ⊢ Exists fun g => And (∀ (x : Subtype fun x => Membership.mem p x), Eq (g ↑x)  …
  -/
  let fr := reCLM.comp (f.restrictScalars ℝ)
  -- Use the real version to get a norm-preserving extension of `fr`, which
  -- we'll call `g : E →L[ℝ] ℝ`.
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : IsRCLikeNormedField 𝕜
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : Subspace 𝕜 E
    f : ContinuousLinearMap (RingHom.id 𝕜) (Subtype fun x => Membership.mem p x) 𝕜
    this✝² : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    this✝¹ : Module Real E := RestrictScalars.module Real 𝕜 E
    this✝ : IsScalarTower Real 𝕜 E := RestrictScalars.isScalarTower Real 𝕜 E
    this : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
    fr : ContinuousLinearMap (RingHom.id Real) (Subtype fun x => Membership.mem p  …
    ⊢ Exists fun g => And (∀ (x : Subtype fun x => Membership.mem p x), Eq (g ↑x)  …
  -/
  rcases Real.exists_extension_norm_eq (p.restrictScalars ℝ) fr with ⟨g, ⟨hextends, hnormeq⟩⟩
  -- Now `g` can be extended to the `E →L[𝕜] 𝕜` we need.
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : IsRCLikeNormedField 𝕜
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : Subspace 𝕜 E
    f : ContinuousLinearMap (RingHom.id 𝕜) (Subtype fun x => Membership.mem p x) 𝕜
    this✝² : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    this✝¹ : Module Real E := RestrictScalars.module Real 𝕜 E
    this✝ : IsScalarTower Real 𝕜 E := RestrictScalars.isScalarTower Real 𝕜 E
    this : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
    fr : ContinuousLinearMap (RingHom.id Real) (Subtype fun x => Membership.mem p  …
    g : ContinuousLinearMap (RingHom.id Real) E Real
    hextends : ∀ (x : Subtype fun x => Membership.mem (Submodule.restrictScalars R …
    hnormeq : Eq (Norm.norm g) (Norm.norm fr)
    ⊢ Exists fun g => And (∀ (x : Subtype fun x => Membership.mem p x), Eq (g ↑x)  …
  -/
  refine ⟨g.extendTo𝕜, ?_⟩
  -- It is an extension of `f`.
  have h : ∀ x : p, g.extendTo𝕜 x = f x := by
    intro x
    -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
    erw [ContinuousLinearMap.extendTo𝕜_apply, ← Submodule.coe_smul, hextends, hextends]
    have :
        (fr x : 𝕜) - I * ↑(fr ((I : 𝕜) • x)) = (re (f x) : 𝕜) - (I : 𝕜) * re (f ((I : 𝕜) • x)) := by
      rfl
    -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
    erw [this]
    apply ext
    · simp only [add_zero, Algebra.id.smul_eq_mul, I_re, ofReal_im, AddMonoidHom.map_add, zero_sub,
        I_im', zero_mul, ofReal_re, eq_self_iff_true, sub_zero, mul_neg, ofReal_neg,
        mul_re, mul_zero, sub_neg_eq_add, ContinuousLinearMap.map_smul]
    · simp only [Algebra.id.smul_eq_mul, I_re, ofReal_im, AddMonoidHom.map_add, zero_sub, I_im',
        zero_mul, ofReal_re, mul_neg, mul_im, zero_add, ofReal_neg, mul_re,
        sub_neg_eq_add, ContinuousLinearMap.map_smul]
  -- And we derive the equality of the norms by bounding on both sides.
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : IsRCLikeNormedField 𝕜
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : Subspace 𝕜 E
    f : ContinuousLinearMap (RingHom.id 𝕜) (Subtype fun x => Membership.mem p x) 𝕜
    this✝² : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    this✝¹ : Module Real E := RestrictScalars.module Real 𝕜 E
    this✝ : IsScalarTower Real 𝕜 E := RestrictScalars.isScalarTower Real 𝕜 E
    this : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
    fr : ContinuousLinearMap (RingHom.id Real) (Subtype fun x => Membership.mem p  …
    g : ContinuousLinearMap (RingHom.id Real) E Real
    hextends : ∀ (x : Subtype fun x => Membership.mem (Submodule.restrictScalars R …
    hnormeq : Eq (Norm.norm g) (Norm.norm fr)
    h : ∀ (x : Subtype fun x => Membership.mem p x), Eq (g.extendTo𝕜 ↑x) (f x)
    ⊢ And (∀ (x : Subtype fun x => Membership.mem p x), Eq (g.extendTo𝕜 ↑x) (f x)) …
  -/
  refine ⟨h, le_antisymm ?_ ?_⟩
  · calc
      ‖g.extendTo𝕜‖ = ‖g‖ := g.norm_extendTo𝕜
      _ = ‖fr‖ := hnormeq
      _ ≤ ‖reCLM‖ * ‖f‖ := ContinuousLinearMap.opNorm_comp_le _ _
      _ = ‖f‖ := by rw [reCLM_norm, one_mul]
    /-
      case intro.intro.refine_2
      𝕜 : Type u_1
      inst✝³ : NontriviallyNormedField 𝕜
      inst✝² : IsRCLikeNormedField 𝕜
      E : Type u_2
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      p : Subspace 𝕜 E
      f : ContinuousLinearMap (RingHom.id 𝕜) (Subtype fun x => Membership.mem p x) 𝕜
      this✝² : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
      this✝¹ : Module Real E := RestrictScalars.module Real 𝕜 E
      this✝ : IsScalarTower Real 𝕜 E := RestrictScalars.isScalarTower Real 𝕜 E
      this : NormedSpace Real E := NormedSpace.restrictScalars Real 𝕜 E
      fr : ContinuousLinearMap (RingHom.id Real) (Subtype fun x => Membership.mem p  …
      g : ContinuousLinearMap (RingHom.id Real) E Real
      hextends : ∀ (x : Subtype fun x => Membership.mem (Submodule.restrictScalars R …
      hnormeq : Eq (Norm.norm g) (Norm.norm fr)
      h : ∀ (x : Subtype fun x => Membership.mem p x), Eq (g.extendTo𝕜 ↑x) (f x)
      ⊢ LE.le (Norm.norm f) (Norm.norm g.extendTo𝕜)
    -/
  · exact f.opNorm_le_bound g.extendTo𝕜.opNorm_nonneg fun x => h x ▸ g.extendTo𝕜.le_opNorm x
    /-
      🎉 no goals
    -/


/-- Corollary of the **Hahn-Banach theorem**: if `f : p → F` is a continuous linear map
from a submodule of a normed space `E` over `𝕜`, `𝕜 = ℝ` or `𝕜 = ℂ`,
with a finite dimensional range, then `f` admits an extension to a continuous linear map `E → F`.

Note that contrary to the case `F = 𝕜`, see `exists_extension_norm_eq`,
we provide no estimates on the norm of the extension.
-/
lemma ContinuousLinearMap.exist_extension_of_finiteDimensional_range {p : Submodule 𝕜 E}
    (f : p →L[𝕜] F) [FiniteDimensional 𝕜 (LinearMap.range f)] :
    ∃ g : E →L[𝕜] F, f = g.comp p.subtypeL := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : IsRCLikeNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝⁴ : SeminormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    p : Submodule 𝕜 E
    f : ContinuousLinearMap (RingHom.id 𝕜) (Subtype fun x => Membership.mem p x) F
    inst✝ : FiniteDimensional 𝕜 (Subtype fun x => Membership.mem (LinearMap.range  …
    ⊢ Exists fun g => Eq f (g.comp p.subtypeL)
  -/
  letI : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : IsRCLikeNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝⁴ : SeminormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    p : Submodule 𝕜 E
    f : ContinuousLinearMap (RingHom.id 𝕜) (Subtype fun x => Membership.mem p x) F
    inst✝ : FiniteDimensional 𝕜 (Subtype fun x => Membership.mem (LinearMap.range  …
    this : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    ⊢ Exists fun g => Eq f (g.comp p.subtypeL)
  -/
  set b := Module.finBasis 𝕜 (LinearMap.range f)
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : IsRCLikeNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝⁴ : SeminormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    p : Submodule 𝕜 E
    f : ContinuousLinearMap (RingHom.id 𝕜) (Subtype fun x => Membership.mem p x) F
    inst✝ : FiniteDimensional 𝕜 (Subtype fun x => Membership.mem (LinearMap.range  …
    this : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    b : Basis (Fin (Module.finrank 𝕜 (Subtype fun x => Membership.mem (LinearMap.r …
    ⊢ Exists fun g => Eq f (g.comp p.subtypeL)
  -/
  set e := b.equivFunL
  set fi := fun i ↦ (LinearMap.toContinuousLinearMap (b.coord i)).comp
    (f.codRestrict _ <| LinearMap.mem_range_self _)
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : IsRCLikeNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝⁴ : SeminormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    p : Submodule 𝕜 E
    f : ContinuousLinearMap (RingHom.id 𝕜) (Subtype fun x => Membership.mem p x) F
    inst✝ : FiniteDimensional 𝕜 (Subtype fun x => Membership.mem (LinearMap.range  …
    this : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    b : Basis (Fin (Module.finrank 𝕜 (Subtype fun x => Membership.mem (LinearMap.r …
    e : ContinuousLinearEquiv (RingHom.id 𝕜) (Subtype fun x => Membership.mem (Lin …
    fi : Fin (Module.finrank 𝕜 (Subtype fun x => Membership.mem (LinearMap.range f …
    ⊢ Exists fun g => Eq f (g.comp p.subtypeL)
  -/
  choose gi hgf _ using fun i ↦ exists_extension_norm_eq p (fi i)
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : IsRCLikeNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝⁴ : SeminormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    p : Submodule 𝕜 E
    f : ContinuousLinearMap (RingHom.id 𝕜) (Subtype fun x => Membership.mem p x) F
    inst✝ : FiniteDimensional 𝕜 (Subtype fun x => Membership.mem (LinearMap.range  …
    this : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    b : Basis (Fin (Module.finrank 𝕜 (Subtype fun x => Membership.mem (LinearMap.r …
    e : ContinuousLinearEquiv (RingHom.id 𝕜) (Subtype fun x => Membership.mem (Lin …
    fi : Fin (Module.finrank 𝕜 (Subtype fun x => Membership.mem (LinearMap.range f …
    gi : Fin (Module.finrank 𝕜 (Subtype fun x => Membership.mem (LinearMap.range f …
    hgf : ∀ (i : Fin (Module.finrank 𝕜 (Subtype fun x => Membership.mem (LinearMap …
    a✝ : ∀ (i : Fin (Module.finrank 𝕜 (Subtype fun x => Membership.mem (LinearMap. …
    ⊢ Exists fun g => Eq f (g.comp p.subtypeL)
  -/
  use (LinearMap.range f).subtypeL.comp <| e.symm.toContinuousLinearMap.comp (.pi gi)
  /-
    case h
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : IsRCLikeNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝⁴ : SeminormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    p : Submodule 𝕜 E
    f : ContinuousLinearMap (RingHom.id 𝕜) (Subtype fun x => Membership.mem p x) F
    inst✝ : FiniteDimensional 𝕜 (Subtype fun x => Membership.mem (LinearMap.range  …
    this : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    b : Basis (Fin (Module.finrank 𝕜 (Subtype fun x => Membership.mem (LinearMap.r …
    e : ContinuousLinearEquiv (RingHom.id 𝕜) (Subtype fun x => Membership.mem (Lin …
    fi : Fin (Module.finrank 𝕜 (Subtype fun x => Membership.mem (LinearMap.range f …
    gi : Fin (Module.finrank 𝕜 (Subtype fun x => Membership.mem (LinearMap.range f …
    hgf : ∀ (i : Fin (Module.finrank 𝕜 (Subtype fun x => Membership.mem (LinearMap …
    a✝ : ∀ (i : Fin (Module.finrank 𝕜 (Subtype fun x => Membership.mem (LinearMap. …
    ⊢ Eq f (((LinearMap.range f).subtypeL.comp ((↑e.symm).comp (ContinuousLinearMa …
  -/
  ext x
  /-
    case h.h
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : IsRCLikeNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝⁴ : SeminormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    p : Submodule 𝕜 E
    f : ContinuousLinearMap (RingHom.id 𝕜) (Subtype fun x => Membership.mem p x) F
    inst✝ : FiniteDimensional 𝕜 (Subtype fun x => Membership.mem (LinearMap.range  …
    this : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    b : Basis (Fin (Module.finrank 𝕜 (Subtype fun x => Membership.mem (LinearMap.r …
    e : ContinuousLinearEquiv (RingHom.id 𝕜) (Subtype fun x => Membership.mem (Lin …
    fi : Fin (Module.finrank 𝕜 (Subtype fun x => Membership.mem (LinearMap.range f …
    gi : Fin (Module.finrank 𝕜 (Subtype fun x => Membership.mem (LinearMap.range f …
    hgf : ∀ (i : Fin (Module.finrank 𝕜 (Subtype fun x => Membership.mem (LinearMap …
    a✝ : ∀ (i : Fin (Module.finrank 𝕜 (Subtype fun x => Membership.mem (LinearMap. …
    x : Subtype fun x => Membership.mem p x
    ⊢ Eq (f x) ((((LinearMap.range f).subtypeL.comp ((↑e.symm).comp (ContinuousLin …
  -/
  simp [fi, e, hgf]
  /-
    🎉 no goals
  -/


/-- A finite dimensional submodule over `ℝ` or `ℂ` is `Submodule.ClosedComplemented`. -/
lemma Submodule.ClosedComplemented.of_finiteDimensional (p : Submodule 𝕜 F)
    [FiniteDimensional 𝕜 p] : p.ClosedComplemented :=
  let ⟨g, hg⟩ := (ContinuousLinearMap.id 𝕜 p).exist_extension_of_finiteDimensional_range
  ⟨g, DFunLike.congr_fun hg.symm⟩


theorem coord_norm' {x : E} (h : x ≠ 0) : ‖(‖x‖ : 𝕜) • coord 𝕜 x h‖ = 1 := by
  #adaptation_note
  /--
  `set_option maxSynthPendingDepth 2` required after https://github.com/leanprover/lean4/pull/4119
  Alternatively, we can add:
  ```
  let X : SeminormedAddCommGroup (↥(span 𝕜 {x}) →L[𝕜] 𝕜) := inferInstance
  have : BoundedSMul 𝕜 (↥(span 𝕜 {x}) →L[𝕜] 𝕜) := @NormedSpace.boundedSMul 𝕜 _ _ X _
  ```
  -/
  set_option maxSynthPendingDepth 2 in
  rw [norm_smul (α := 𝕜) (x := coord 𝕜 x h), RCLike.norm_coe_norm, coord_norm,
    mul_inv_cancel₀ (mt norm_eq_zero.mp h)]


/-- Corollary of Hahn-Banach. Given a nonzero element `x` of a normed space, there exists an
    element of the dual space, of norm `1`, whose value on `x` is `‖x‖`. -/
theorem exists_dual_vector (x : E) (h : x ≠ 0) : ∃ g : E →L[𝕜] 𝕜, ‖g‖ = 1 ∧ g x = ‖x‖ := by
  /-
    𝕜 : Type v
    inst✝² : RCLike 𝕜
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    h : Ne x 0
    ⊢ Exists fun g => And (Eq (Norm.norm g) 1) (Eq (g x) ↑(Norm.norm x))
  -/
  let p : Submodule 𝕜 E := 𝕜 ∙ x
  /-
    𝕜 : Type v
    inst✝² : RCLike 𝕜
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    h : Ne x 0
    p : Submodule 𝕜 E := Submodule.span 𝕜 (Singleton.singleton x)
    ⊢ Exists fun g => And (Eq (Norm.norm g) 1) (Eq (g x) ↑(Norm.norm x))
  -/
  let f := (‖x‖ : 𝕜) • coord 𝕜 x h
  /-
    𝕜 : Type v
    inst✝² : RCLike 𝕜
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    h : Ne x 0
    p : Submodule 𝕜 E := Submodule.span 𝕜 (Singleton.singleton x)
    f : ContinuousLinearMap (RingHom.id 𝕜) (Subtype fun x_1 => Membership.mem (Sub …
    ⊢ Exists fun g => And (Eq (Norm.norm g) 1) (Eq (g x) ↑(Norm.norm x))
  -/
  obtain ⟨g, hg⟩ := exists_extension_norm_eq p f
  /-
    case intro
    𝕜 : Type v
    inst✝² : RCLike 𝕜
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    h : Ne x 0
    p : Submodule 𝕜 E := Submodule.span 𝕜 (Singleton.singleton x)
    f : ContinuousLinearMap (RingHom.id 𝕜) (Subtype fun x_1 => Membership.mem (Sub …
    g : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
    hg : And (∀ (x : Subtype fun x => Membership.mem p x), Eq (g ↑x) (f x)) (Eq (N …
    ⊢ Exists fun g => And (Eq (Norm.norm g) 1) (Eq (g x) ↑(Norm.norm x))
  -/
  refine ⟨g, ?_, ?_⟩
    /-
      case intro.refine_1
      𝕜 : Type v
      inst✝² : RCLike 𝕜
      E : Type u
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      x : E
      h : Ne x 0
      p : Submodule 𝕜 E := Submodule.span 𝕜 (Singleton.singleton x)
      f : ContinuousLinearMap (RingHom.id 𝕜) (Subtype fun x_1 => Membership.mem (Sub …
      g : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
      hg : And (∀ (x : Subtype fun x => Membership.mem p x), Eq (g ↑x) (f x)) (Eq (N …
      ⊢ Eq (Norm.norm g) 1
    -/
  · rw [hg.2, coord_norm']
    /-
      🎉 no goals
    -/
  · calc
      g x = g (⟨x, mem_span_singleton_self x⟩ : 𝕜 ∙ x) := by rw [coe_mk]
      _ = ((‖x‖ : 𝕜) • coord 𝕜 x h) (⟨x, mem_span_singleton_self x⟩ : 𝕜 ∙ x) := by rw [← hg.1]
      _ = ‖x‖ := by simp


/-- Variant of Hahn-Banach, eliminating the hypothesis that `x` be nonzero, and choosing
    the dual element arbitrarily when `x = 0`. -/
theorem exists_dual_vector' [Nontrivial E] (x : E) : ∃ g : E →L[𝕜] 𝕜, ‖g‖ = 1 ∧ g x = ‖x‖ := by
  /-
    𝕜 : Type v
    inst✝³ : RCLike 𝕜
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : Nontrivial E
    x : E
    ⊢ Exists fun g => And (Eq (Norm.norm g) 1) (Eq (g x) ↑(Norm.norm x))
  -/
  by_cases hx : x = 0
    /-
      case pos
      𝕜 : Type v
      inst✝³ : RCLike 𝕜
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : Nontrivial E
      x : E
      hx : Eq x 0
      ⊢ Exists fun g => And (Eq (Norm.norm g) 1) (Eq (g x) ↑(Norm.norm x))
    -/
  · obtain ⟨y, hy⟩ := exists_ne (0 : E)
    /-
      case pos.intro
      𝕜 : Type v
      inst✝³ : RCLike 𝕜
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : Nontrivial E
      x : E
      hx : Eq x 0
      y : E
      hy : Ne y 0
      ⊢ Exists fun g => And (Eq (Norm.norm g) 1) (Eq (g x) ↑(Norm.norm x))
    -/
    obtain ⟨g, hg⟩ : ∃ g : E →L[𝕜] 𝕜, ‖g‖ = 1 ∧ g y = ‖y‖ := exists_dual_vector 𝕜 y hy
    /-
      case pos.intro.intro
      𝕜 : Type v
      inst✝³ : RCLike 𝕜
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : Nontrivial E
      x : E
      hx : Eq x 0
      y : E
      hy : Ne y 0
      g : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
      hg : And (Eq (Norm.norm g) 1) (Eq (g y) ↑(Norm.norm y))
      ⊢ Exists fun g => And (Eq (Norm.norm g) 1) (Eq (g x) ↑(Norm.norm x))
    -/
    refine ⟨g, hg.left, ?_⟩
    /-
      case pos.intro.intro
      𝕜 : Type v
      inst✝³ : RCLike 𝕜
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : Nontrivial E
      x : E
      hx : Eq x 0
      y : E
      hy : Ne y 0
      g : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
      hg : And (Eq (Norm.norm g) 1) (Eq (g y) ↑(Norm.norm y))
      ⊢ Eq (g x) ↑(Norm.norm x)
    -/
    simp [hx]
    /-
      🎉 no goals
    -/
    /-
      case neg
      𝕜 : Type v
      inst✝³ : RCLike 𝕜
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : Nontrivial E
      x : E
      hx : Not (Eq x 0)
      ⊢ Exists fun g => And (Eq (Norm.norm g) 1) (Eq (g x) ↑(Norm.norm x))
    -/
  · exact exists_dual_vector 𝕜 x hx
    /-
      🎉 no goals
    -/


/-- Variant of Hahn-Banach, eliminating the hypothesis that `x` be nonzero, but only ensuring that
    the dual element has norm at most `1` (this can not be improved for the trivial
    vector space). -/
theorem exists_dual_vector'' (x : E) : ∃ g : E →L[𝕜] 𝕜, ‖g‖ ≤ 1 ∧ g x = ‖x‖ := by
  /-
    𝕜 : Type v
    inst✝² : RCLike 𝕜
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    ⊢ Exists fun g => And (LE.le (Norm.norm g) 1) (Eq (g x) ↑(Norm.norm x))
  -/
  by_cases hx : x = 0
    /-
      case pos
      𝕜 : Type v
      inst✝² : RCLike 𝕜
      E : Type u
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      x : E
      hx : Eq x 0
      ⊢ Exists fun g => And (LE.le (Norm.norm g) 1) (Eq (g x) ↑(Norm.norm x))
    -/
  · refine ⟨0, by simp, ?_⟩
    /-
      case pos
      𝕜 : Type v
      inst✝² : RCLike 𝕜
      E : Type u
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      x : E
      hx : Eq x 0
      ⊢ Eq (0 x) ↑(Norm.norm x)
    -/
    symm
    /-
      case pos
      𝕜 : Type v
      inst✝² : RCLike 𝕜
      E : Type u
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      x : E
      hx : Eq x 0
      ⊢ Eq (↑(Norm.norm x)) (0 x)
    -/
    simp [hx]
    /-
      🎉 no goals
    -/
    /-
      case neg
      𝕜 : Type v
      inst✝² : RCLike 𝕜
      E : Type u
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      x : E
      hx : Not (Eq x 0)
      ⊢ Exists fun g => And (LE.le (Norm.norm g) 1) (Eq (g x) ↑(Norm.norm x))
    -/
  · rcases exists_dual_vector 𝕜 x hx with ⟨g, g_norm, g_eq⟩
    /-
      case neg.intro.intro
      𝕜 : Type v
      inst✝² : RCLike 𝕜
      E : Type u
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      x : E
      hx : Not (Eq x 0)
      g : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
      g_norm : Eq (Norm.norm g) 1
      g_eq : Eq (g x) ↑(Norm.norm x)
      ⊢ Exists fun g => And (LE.le (Norm.norm g) 1) (Eq (g x) ↑(Norm.norm x))
    -/
    exact ⟨g, g_norm.le, g_eq⟩
    /-
      🎉 no goals
    -/


