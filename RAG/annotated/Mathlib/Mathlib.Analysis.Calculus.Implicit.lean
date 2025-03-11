/-- Data for the general version of the implicit function theorem. It holds two functions
`f : E → F` and `g : E → G` (named `leftFun` and `rightFun`) and a point `a` (named `pt`) such that

* both functions are strictly differentiable at `a`;
* the derivatives are surjective;
* the kernels of the derivatives are complementary subspaces of `E`. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): linter not yet ported @[nolint has_nonempty_instance]
structure ImplicitFunctionData (𝕜 : Type*) [NontriviallyNormedField 𝕜] (E : Type*)
    [NormedAddCommGroup E] [NormedSpace 𝕜 E] [CompleteSpace E] (F : Type*) [NormedAddCommGroup F]
    [NormedSpace 𝕜 F] [CompleteSpace F] (G : Type*) [NormedAddCommGroup G] [NormedSpace 𝕜 G]
    [CompleteSpace G] where
  leftFun : E → F
  leftDeriv : E →L[𝕜] F
  rightFun : E → G
  rightDeriv : E →L[𝕜] G
  pt : E
  left_has_deriv : HasStrictFDerivAt leftFun leftDeriv pt
  right_has_deriv : HasStrictFDerivAt rightFun rightDeriv pt
  left_range : range leftDeriv = ⊤
  right_range : range rightDeriv = ⊤
  isCompl_ker : IsCompl (ker leftDeriv) (ker rightDeriv)


/-- The function given by `x ↦ (leftFun x, rightFun x)`. -/
def prodFun (x : E) : F × G :=
  (φ.leftFun x, φ.rightFun x)


@[simp]
theorem prodFun_apply (x : E) : φ.prodFun x = (φ.leftFun x, φ.rightFun x) :=
  rfl


protected theorem hasStrictFDerivAt :
    HasStrictFDerivAt φ.prodFun
      (φ.leftDeriv.equivProdOfSurjectiveOfIsCompl φ.rightDeriv φ.left_range φ.right_range
          φ.isCompl_ker :
        E →L[𝕜] F × G)
      φ.pt :=
  φ.left_has_deriv.prod φ.right_has_deriv


/-- Implicit function theorem. If `f : E → F` and `g : E → G` are two maps strictly differentiable
at `a`, their derivatives `f'`, `g'` are surjective, and the kernels of these derivatives are
complementary subspaces of `E`, then `x ↦ (f x, g x)` defines a partial homeomorphism between
`E` and `F × G`. In particular, `{x | f x = f a}` is locally homeomorphic to `G`. -/
def toPartialHomeomorph : PartialHomeomorph E (F × G) :=
  φ.hasStrictFDerivAt.toPartialHomeomorph _


/-- Implicit function theorem. If `f : E → F` and `g : E → G` are two maps strictly differentiable
at `a`, their derivatives `f'`, `g'` are surjective, and the kernels of these derivatives are
complementary subspaces of `E`, then `implicitFunction` is the unique (germ of a) map
`φ : F → G → E` such that `f (φ y z) = y` and `g (φ y z) = z`. -/
def implicitFunction : F → G → E :=
  Function.curry <| φ.toPartialHomeomorph.symm


@[simp]
theorem toPartialHomeomorph_coe : ⇑φ.toPartialHomeomorph = φ.prodFun :=
  rfl


theorem toPartialHomeomorph_apply (x : E) : φ.toPartialHomeomorph x = (φ.leftFun x, φ.rightFun x) :=
  rfl


theorem pt_mem_toPartialHomeomorph_source : φ.pt ∈ φ.toPartialHomeomorph.source :=
  φ.hasStrictFDerivAt.mem_toPartialHomeomorph_source


theorem map_pt_mem_toPartialHomeomorph_target :
    (φ.leftFun φ.pt, φ.rightFun φ.pt) ∈ φ.toPartialHomeomorph.target :=
  φ.toPartialHomeomorph.map_source <| φ.pt_mem_toPartialHomeomorph_source


theorem prod_map_implicitFunction :
    ∀ᶠ p : F × G in 𝓝 (φ.prodFun φ.pt), φ.prodFun (φ.implicitFunction p.1 p.2) = p :=
  φ.hasStrictFDerivAt.eventually_right_inverse.mono fun ⟨_, _⟩ h => h


theorem left_map_implicitFunction :
    ∀ᶠ p : F × G in 𝓝 (φ.prodFun φ.pt), φ.leftFun (φ.implicitFunction p.1 p.2) = p.1 :=
  φ.prod_map_implicitFunction.mono fun _ => congr_arg Prod.fst


theorem right_map_implicitFunction :
    ∀ᶠ p : F × G in 𝓝 (φ.prodFun φ.pt), φ.rightFun (φ.implicitFunction p.1 p.2) = p.2 :=
  φ.prod_map_implicitFunction.mono fun _ => congr_arg Prod.snd


theorem implicitFunction_apply_image :
    ∀ᶠ x in 𝓝 φ.pt, φ.implicitFunction (φ.leftFun x) (φ.rightFun x) = x :=
  φ.hasStrictFDerivAt.eventually_left_inverse


theorem map_nhds_eq : map φ.leftFun (𝓝 φ.pt) = 𝓝 (φ.leftFun φ.pt) :=
  show map (Prod.fst ∘ φ.prodFun) (𝓝 φ.pt) = 𝓝 (φ.prodFun φ.pt).1 by
    /-
      𝕜 : Type u_1
      inst✝⁹ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace 𝕜 E
      inst✝⁶ : CompleteSpace E
      F : Type u_3
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      inst✝³ : CompleteSpace F
      G : Type u_4
      inst✝² : NormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      inst✝ : CompleteSpace G
      φ : ImplicitFunctionData 𝕜 E F G
      ⊢ Eq (Filter.map (Function.comp Prod.fst φ.prodFun) (nhds φ.pt)) (nhds (φ.prod …
    -/
    rw [← map_map, φ.hasStrictFDerivAt.map_nhds_eq_of_equiv, map_fst_nhds]
    /-
      🎉 no goals
    -/


theorem implicitFunction_hasStrictFDerivAt (g'inv : G →L[𝕜] E)
    (hg'inv : φ.rightDeriv.comp g'inv = ContinuousLinearMap.id 𝕜 G)
    (hg'invf : φ.leftDeriv.comp g'inv = 0) :
    HasStrictFDerivAt (φ.implicitFunction (φ.leftFun φ.pt)) g'inv (φ.rightFun φ.pt) := by
  /-
    𝕜 : Type u_1
    inst✝⁹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : CompleteSpace E
    F : Type u_3
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    inst✝³ : CompleteSpace F
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    inst✝ : CompleteSpace G
    φ : ImplicitFunctionData 𝕜 E F G
    g'inv : ContinuousLinearMap (RingHom.id 𝕜) G E
    hg'inv : Eq (φ.rightDeriv.comp g'inv) (ContinuousLinearMap.id 𝕜 G)
    hg'invf : Eq (φ.leftDeriv.comp g'inv) 0
    ⊢ HasStrictFDerivAt (φ.implicitFunction (φ.leftFun φ.pt)) g'inv (φ.rightFun φ. …
  -/
  have := φ.hasStrictFDerivAt.to_localInverse
  /-
    𝕜 : Type u_1
    inst✝⁹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : CompleteSpace E
    F : Type u_3
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    inst✝³ : CompleteSpace F
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    inst✝ : CompleteSpace G
    φ : ImplicitFunctionData 𝕜 E F G
    g'inv : ContinuousLinearMap (RingHom.id 𝕜) G E
    hg'inv : Eq (φ.rightDeriv.comp g'inv) (ContinuousLinearMap.id 𝕜 G)
    hg'invf : Eq (φ.leftDeriv.comp g'inv) 0
    this : HasStrictFDerivAt (HasStrictFDerivAt.localInverse φ.prodFun (φ.leftDeri …
    ⊢ HasStrictFDerivAt (φ.implicitFunction (φ.leftFun φ.pt)) g'inv (φ.rightFun φ. …
  -/
  simp only [prodFun] at this
  /-
    𝕜 : Type u_1
    inst✝⁹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : CompleteSpace E
    F : Type u_3
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    inst✝³ : CompleteSpace F
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    inst✝ : CompleteSpace G
    φ : ImplicitFunctionData 𝕜 E F G
    g'inv : ContinuousLinearMap (RingHom.id 𝕜) G E
    hg'inv : Eq (φ.rightDeriv.comp g'inv) (ContinuousLinearMap.id 𝕜 G)
    hg'invf : Eq (φ.leftDeriv.comp g'inv) 0
    this : HasStrictFDerivAt (HasStrictFDerivAt.localInverse φ.prodFun (φ.leftDeri …
    ⊢ HasStrictFDerivAt (φ.implicitFunction (φ.leftFun φ.pt)) g'inv (φ.rightFun φ. …
  -/
  convert this.comp (φ.rightFun φ.pt) ((hasStrictFDerivAt_const _ _).prod (hasStrictFDerivAt_id _))
  -- Porting note: added parentheses to help `simp`
  /-
    case h.e'_12
    𝕜 : Type u_1
    inst✝⁹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : CompleteSpace E
    F : Type u_3
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    inst✝³ : CompleteSpace F
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    inst✝ : CompleteSpace G
    φ : ImplicitFunctionData 𝕜 E F G
    g'inv : ContinuousLinearMap (RingHom.id 𝕜) G E
    hg'inv : Eq (φ.rightDeriv.comp g'inv) (ContinuousLinearMap.id 𝕜 G)
    hg'invf : Eq (φ.leftDeriv.comp g'inv) 0
    this : HasStrictFDerivAt (HasStrictFDerivAt.localInverse φ.prodFun (φ.leftDeri …
    ⊢ Eq g'inv ((↑(φ.leftDeriv.equivProdOfSurjectiveOfIsCompl φ.rightDeriv ⋯ ⋯ ⋯). …
  -/
  simp only [ContinuousLinearMap.ext_iff, (ContinuousLinearMap.comp_apply)] at hg'inv hg'invf ⊢
  /-
    case h.e'_12
    𝕜 : Type u_1
    inst✝⁹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : CompleteSpace E
    F : Type u_3
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    inst✝³ : CompleteSpace F
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    inst✝ : CompleteSpace G
    φ : ImplicitFunctionData 𝕜 E F G
    g'inv : ContinuousLinearMap (RingHom.id 𝕜) G E
    this : HasStrictFDerivAt (HasStrictFDerivAt.localInverse φ.prodFun (φ.leftDeri …
    hg'inv : ∀ (x : G), Eq (φ.rightDeriv (g'inv x)) ((ContinuousLinearMap.id 𝕜 G) x)
    hg'invf : ∀ (x : G), Eq (φ.leftDeriv (g'inv x)) (0 x)
    ⊢ ∀ (x : G), Eq (g'inv x) (↑(φ.leftDeriv.equivProdOfSurjectiveOfIsCompl φ.righ …
  -/
  simp [ContinuousLinearEquiv.eq_symm_apply, *]
  /-
    🎉 no goals
  -/


/-- Data used to apply the generic implicit function theorem to the case of a strictly
differentiable map such that its derivative is surjective and has a complemented kernel. -/
@[simp]
def implicitFunctionDataOfComplemented (hf : HasStrictFDerivAt f f' a) (hf' : range f' = ⊤)
    (hker : (ker f').ClosedComplemented) : ImplicitFunctionData 𝕜 E F (ker f') where
  leftFun := f
  leftDeriv := f'
  rightFun x := Classical.choose hker (x - a)
  rightDeriv := Classical.choose hker
  pt := a
  left_has_deriv := hf
  right_has_deriv :=
    (Classical.choose hker).hasStrictFDerivAt.comp a ((hasStrictFDerivAt_id a).sub_const a)
  left_range := hf'
  right_range := LinearMap.range_eq_of_proj (Classical.choose_spec hker)
  isCompl_ker := LinearMap.isCompl_of_proj (Classical.choose_spec hker)


/-- A partial homeomorphism between `E` and `F × f'.ker` sending level surfaces of `f`
to vertical subspaces. -/
def implicitToPartialHomeomorphOfComplemented (hf : HasStrictFDerivAt f f' a) (hf' : range f' = ⊤)
    (hker : (ker f').ClosedComplemented) : PartialHomeomorph E (F × ker f') :=
  (implicitFunctionDataOfComplemented f f' hf hf' hker).toPartialHomeomorph


/-- Implicit function `g` defined by `f (g z y) = z`. -/
def implicitFunctionOfComplemented (hf : HasStrictFDerivAt f f' a) (hf' : range f' = ⊤)
    (hker : (ker f').ClosedComplemented) : F → ker f' → E :=
  (implicitFunctionDataOfComplemented f f' hf hf' hker).implicitFunction


@[simp]
theorem implicitToPartialHomeomorphOfComplemented_fst (hf : HasStrictFDerivAt f f' a)
    (hf' : range f' = ⊤) (hker : (ker f').ClosedComplemented) (x : E) :
    (hf.implicitToPartialHomeomorphOfComplemented f f' hf' hker x).fst = f x :=
  rfl


theorem implicitToPartialHomeomorphOfComplemented_apply (hf : HasStrictFDerivAt f f' a)
    (hf' : range f' = ⊤) (hker : (ker f').ClosedComplemented) (y : E) :
    hf.implicitToPartialHomeomorphOfComplemented f f' hf' hker y =
      (f y, Classical.choose hker (y - a)) :=
  rfl


@[simp]
theorem implicitToPartialHomeomorphOfComplemented_apply_ker (hf : HasStrictFDerivAt f f' a)
    (hf' : range f' = ⊤) (hker : (ker f').ClosedComplemented) (y : ker f') :
    hf.implicitToPartialHomeomorphOfComplemented f f' hf' hker (y + a) = (f (y + a), y) := by
  simp only [implicitToPartialHomeomorphOfComplemented_apply, add_sub_cancel_right,
    Classical.choose_spec hker]


@[simp]
theorem implicitToPartialHomeomorphOfComplemented_self (hf : HasStrictFDerivAt f f' a)
    (hf' : range f' = ⊤) (hker : (ker f').ClosedComplemented) :
    hf.implicitToPartialHomeomorphOfComplemented f f' hf' hker a = (f a, 0) := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : CompleteSpace E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    a : E
    hf : HasStrictFDerivAt f f' a
    hf' : Eq (LinearMap.range f') Top.top
    hker : (LinearMap.ker f').ClosedComplemented
    ⊢ Eq (↑(HasStrictFDerivAt.implicitToPartialHomeomorphOfComplemented f f' hf hf …
  -/
  simp [hf.implicitToPartialHomeomorphOfComplemented_apply]
  /-
    🎉 no goals
  -/


theorem mem_implicitToPartialHomeomorphOfComplemented_source (hf : HasStrictFDerivAt f f' a)
    (hf' : range f' = ⊤) (hker : (ker f').ClosedComplemented) :
    a ∈ (hf.implicitToPartialHomeomorphOfComplemented f f' hf' hker).source :=
  ImplicitFunctionData.pt_mem_toPartialHomeomorph_source _


theorem mem_implicitToPartialHomeomorphOfComplemented_target (hf : HasStrictFDerivAt f f' a)
    (hf' : range f' = ⊤) (hker : (ker f').ClosedComplemented) :
    (f a, (0 : ker f')) ∈ (hf.implicitToPartialHomeomorphOfComplemented f f' hf' hker).target := by
  simpa only [implicitToPartialHomeomorphOfComplemented_self] using
    (hf.implicitToPartialHomeomorphOfComplemented f f' hf' hker).map_source <|
      hf.mem_implicitToPartialHomeomorphOfComplemented_source hf' hker


/-- `HasStrictFDerivAt.implicitFunctionOfComplemented` sends `(z, y)` to a point in `f ⁻¹' z`. -/
theorem map_implicitFunctionOfComplemented_eq (hf : HasStrictFDerivAt f f' a) (hf' : range f' = ⊤)
    (hker : (ker f').ClosedComplemented) :
    ∀ᶠ p : F × ker f' in 𝓝 (f a, 0),
      f (hf.implicitFunctionOfComplemented f f' hf' hker p.1 p.2) = p.1 :=
  ((hf.implicitToPartialHomeomorphOfComplemented f f' hf' hker).eventually_right_inverse <|
        hf.mem_implicitToPartialHomeomorphOfComplemented_target hf' hker).mono
    fun ⟨_, _⟩ h => congr_arg Prod.fst h


/-- Any point in some neighborhood of `a` can be represented as
`HasStrictFDerivAt.implicitFunctionOfComplemented` of some point. -/
theorem eq_implicitFunctionOfComplemented (hf : HasStrictFDerivAt f f' a) (hf' : range f' = ⊤)
    (hker : (ker f').ClosedComplemented) :
    ∀ᶠ x in 𝓝 a, hf.implicitFunctionOfComplemented f f' hf' hker (f x)
      (hf.implicitToPartialHomeomorphOfComplemented f f' hf' hker x).snd = x :=
  (implicitFunctionDataOfComplemented f f' hf hf' hker).implicitFunction_apply_image


@[simp]
theorem implicitFunctionOfComplemented_apply_image (hf : HasStrictFDerivAt f f' a)
    (hf' : range f' = ⊤) (hker : (ker f').ClosedComplemented) :
    hf.implicitFunctionOfComplemented f f' hf' hker (f a) 0 = a := by
  simpa only [implicitToPartialHomeomorphOfComplemented_self] using
      (hf.implicitToPartialHomeomorphOfComplemented f f' hf' hker).left_inv
      (hf.mem_implicitToPartialHomeomorphOfComplemented_source hf' hker)


theorem to_implicitFunctionOfComplemented (hf : HasStrictFDerivAt f f' a) (hf' : range f' = ⊤)
    (hker : (ker f').ClosedComplemented) :
    HasStrictFDerivAt (hf.implicitFunctionOfComplemented f f' hf' hker (f a))
      (ker f').subtypeL 0 := by
  convert (implicitFunctionDataOfComplemented f f' hf hf' hker).implicitFunction_hasStrictFDerivAt
    (ker f').subtypeL _ _
  /-
    case h.e'_13
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : CompleteSpace E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    a : E
    hf : HasStrictFDerivAt f f' a
    hf' : Eq (LinearMap.range f') Top.top
    hker : (LinearMap.ker f').ClosedComplemented
    ⊢ Eq 0 ((HasStrictFDerivAt.implicitFunctionDataOfComplemented f f' hf hf' hker …
  -/
  swap
    /-
      case convert_1
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : CompleteSpace E
      F : Type u_3
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : CompleteSpace F
      f : E → F
      f' : ContinuousLinearMap (RingHom.id 𝕜) E F
      a : E
      hf : HasStrictFDerivAt f f' a
      hf' : Eq (LinearMap.range f') Top.top
      hker : (LinearMap.ker f').ClosedComplemented
      ⊢ Eq ((HasStrictFDerivAt.implicitFunctionDataOfComplemented f f' hf hf' hker). …
    -/
  · ext
    -- Porting note: added parentheses to help `simp`
    simp only [Classical.choose_spec hker, implicitFunctionDataOfComplemented,
      ContinuousLinearMap.comp_apply, Submodule.coe_subtypeL', Submodule.coe_subtype,
      ContinuousLinearMap.id_apply]
  /-
    case h.e'_13
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : CompleteSpace E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    a : E
    hf : HasStrictFDerivAt f f' a
    hf' : Eq (LinearMap.range f') Top.top
    hker : (LinearMap.ker f').ClosedComplemented
    ⊢ Eq 0 ((HasStrictFDerivAt.implicitFunctionDataOfComplemented f f' hf hf' hker …
  -/
  swap
    /-
      case convert_2
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : CompleteSpace E
      F : Type u_3
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : CompleteSpace F
      f : E → F
      f' : ContinuousLinearMap (RingHom.id 𝕜) E F
      a : E
      hf : HasStrictFDerivAt f f' a
      hf' : Eq (LinearMap.range f') Top.top
      hker : (LinearMap.ker f').ClosedComplemented
      ⊢ Eq ((HasStrictFDerivAt.implicitFunctionDataOfComplemented f f' hf hf' hker). …
    -/
  · ext
    -- Porting note: added parentheses to help `simp`
    simp only [(ContinuousLinearMap.comp_apply), Submodule.coe_subtypeL', Submodule.coe_subtype,
      LinearMap.map_coe_ker, (ContinuousLinearMap.zero_apply)]
  /-
    case h.e'_13
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : CompleteSpace E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    a : E
    hf : HasStrictFDerivAt f f' a
    hf' : Eq (LinearMap.range f') Top.top
    hker : (LinearMap.ker f').ClosedComplemented
    ⊢ Eq 0 ((HasStrictFDerivAt.implicitFunctionDataOfComplemented f f' hf hf' hker …
  -/
  simp only [implicitFunctionDataOfComplemented, map_sub, sub_self]
  /-
    🎉 no goals
  -/


/-- Given a map `f : E → F` to a finite dimensional space with a surjective derivative `f'`,
returns a partial homeomorphism between `E` and `F × ker f'`. -/
def implicitToPartialHomeomorph (hf : HasStrictFDerivAt f f' a) (hf' : range f' = ⊤) :
    PartialHomeomorph E (F × ker f') :=
  haveI := FiniteDimensional.complete 𝕜 F
  hf.implicitToPartialHomeomorphOfComplemented f f' hf'
    f'.ker_closedComplemented_of_finiteDimensional_range


/-- Implicit function `g` defined by `f (g z y) = z`. -/
def implicitFunction (hf : HasStrictFDerivAt f f' a) (hf' : range f' = ⊤) : F → ker f' → E :=
  Function.curry <| (hf.implicitToPartialHomeomorph f f' hf').symm


@[simp]
theorem implicitToPartialHomeomorph_fst (hf : HasStrictFDerivAt f f' a) (hf' : range f' = ⊤)
    (x : E) : (hf.implicitToPartialHomeomorph f f' hf' x).fst = f x :=
  rfl


@[simp]
theorem implicitToPartialHomeomorph_apply_ker (hf : HasStrictFDerivAt f f' a) (hf' : range f' = ⊤)
    (y : ker f') : hf.implicitToPartialHomeomorph f f' hf' (y + a) = (f (y + a), y) :=
  -- Porting note: had to add `haveI` (here and below)
  haveI := FiniteDimensional.complete 𝕜 F
  implicitToPartialHomeomorphOfComplemented_apply_ker ..


@[simp]
theorem implicitToPartialHomeomorph_self (hf : HasStrictFDerivAt f f' a) (hf' : range f' = ⊤) :
    hf.implicitToPartialHomeomorph f f' hf' a = (f a, 0) :=
  haveI := FiniteDimensional.complete 𝕜 F
  implicitToPartialHomeomorphOfComplemented_self ..


theorem mem_implicitToPartialHomeomorph_source (hf : HasStrictFDerivAt f f' a)
    (hf' : range f' = ⊤) : a ∈ (hf.implicitToPartialHomeomorph f f' hf').source :=
  haveI := FiniteDimensional.complete 𝕜 F
  ImplicitFunctionData.pt_mem_toPartialHomeomorph_source _


theorem mem_implicitToPartialHomeomorph_target (hf : HasStrictFDerivAt f f' a)
    (hf' : range f' = ⊤) : (f a, (0 : ker f')) ∈ (hf.implicitToPartialHomeomorph f f' hf').target :=
  haveI := FiniteDimensional.complete 𝕜 F
  mem_implicitToPartialHomeomorphOfComplemented_target ..


theorem tendsto_implicitFunction (hf : HasStrictFDerivAt f f' a) (hf' : range f' = ⊤) {α : Type*}
    {l : Filter α} {g₁ : α → F} {g₂ : α → ker f'} (h₁ : Tendsto g₁ l (𝓝 <| f a))
    (h₂ : Tendsto g₂ l (𝓝 0)) :
    Tendsto (fun t => hf.implicitFunction f f' hf' (g₁ t) (g₂ t)) l (𝓝 a) := by
  refine ((hf.implicitToPartialHomeomorph f f' hf').tendsto_symm
    (hf.mem_implicitToPartialHomeomorph_source hf')).comp ?_
  /-
    𝕜 : Type u_1
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : CompleteSpace 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : CompleteSpace E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : FiniteDimensional 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    a : E
    hf : HasStrictFDerivAt f f' a
    hf' : Eq (LinearMap.range f') Top.top
    α : Type u_4
    l : Filter α
    g₁ : α → F
    g₂ : α → Subtype fun x => Membership.mem (LinearMap.ker f') x
    h₁ : Filter.Tendsto g₁ l (nhds (f a))
    h₂ : Filter.Tendsto g₂ l (nhds 0)
    ⊢ Filter.Tendsto (fun t => { fst := g₁ t, snd := g₂ t }) l (nhds (↑(HasStrictF …
  -/
  rw [implicitToPartialHomeomorph_self]
  /-
    𝕜 : Type u_1
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : CompleteSpace 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : CompleteSpace E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : FiniteDimensional 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    a : E
    hf : HasStrictFDerivAt f f' a
    hf' : Eq (LinearMap.range f') Top.top
    α : Type u_4
    l : Filter α
    g₁ : α → F
    g₂ : α → Subtype fun x => Membership.mem (LinearMap.ker f') x
    h₁ : Filter.Tendsto g₁ l (nhds (f a))
    h₂ : Filter.Tendsto g₂ l (nhds 0)
    ⊢ Filter.Tendsto (fun t => { fst := g₁ t, snd := g₂ t }) l (nhds { fst := f a, …
  -/
  exact h₁.prod_mk_nhds h₂
  /-
    🎉 no goals
  -/


alias _root_.Filter.Tendsto.implicitFunction := tendsto_implicitFunction


/-- `HasStrictFDerivAt.implicitFunction` sends `(z, y)` to a point in `f ⁻¹' z`. -/
theorem map_implicitFunction_eq (hf : HasStrictFDerivAt f f' a) (hf' : range f' = ⊤) :
    ∀ᶠ p : F × ker f' in 𝓝 (f a, 0), f (hf.implicitFunction f f' hf' p.1 p.2) = p.1 :=
  haveI := FiniteDimensional.complete 𝕜 F
  map_implicitFunctionOfComplemented_eq ..


@[simp]
theorem implicitFunction_apply_image (hf : HasStrictFDerivAt f f' a) (hf' : range f' = ⊤) :
    hf.implicitFunction f f' hf' (f a) 0 = a := by
  /-
    𝕜 : Type u_1
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : CompleteSpace 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : CompleteSpace E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : FiniteDimensional 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    a : E
    hf : HasStrictFDerivAt f f' a
    hf' : Eq (LinearMap.range f') Top.top
    ⊢ Eq (HasStrictFDerivAt.implicitFunction f f' hf hf' (f a) 0) a
  -/
  haveI := FiniteDimensional.complete 𝕜 F
  /-
    𝕜 : Type u_1
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : CompleteSpace 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : CompleteSpace E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : FiniteDimensional 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    a : E
    hf : HasStrictFDerivAt f f' a
    hf' : Eq (LinearMap.range f') Top.top
    this : CompleteSpace F
    ⊢ Eq (HasStrictFDerivAt.implicitFunction f f' hf hf' (f a) 0) a
  -/
  apply implicitFunctionOfComplemented_apply_image
  /-
    🎉 no goals
  -/


/-- Any point in some neighborhood of `a` can be represented as `HasStrictFDerivAt.implicitFunction`
of some point. -/
theorem eq_implicitFunction (hf : HasStrictFDerivAt f f' a) (hf' : range f' = ⊤) :
    ∀ᶠ x in 𝓝 a,
      hf.implicitFunction f f' hf' (f x) (hf.implicitToPartialHomeomorph f f' hf' x).snd = x :=
  haveI := FiniteDimensional.complete 𝕜 F
  eq_implicitFunctionOfComplemented ..


theorem to_implicitFunction (hf : HasStrictFDerivAt f f' a) (hf' : range f' = ⊤) :
    HasStrictFDerivAt (hf.implicitFunction f f' hf' (f a)) (ker f').subtypeL 0 :=
  haveI := FiniteDimensional.complete 𝕜 F
  to_implicitFunctionOfComplemented ..


