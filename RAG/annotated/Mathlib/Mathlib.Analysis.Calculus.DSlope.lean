open Classical in
/-- `dslope f a b` is defined as `slope f a b = (b - a)⁻¹ • (f b - f a)` for `a ≠ b` and
`deriv f a` for `a = b`. -/
noncomputable def dslope (f : 𝕜 → E) (a : 𝕜) : 𝕜 → E :=
  update (slope f a) a (deriv f a)


@[simp]
theorem dslope_same (f : 𝕜 → E) (a : 𝕜) : dslope f a a = deriv f a := by
  classical
  exact update_self ..


theorem dslope_of_ne (f : 𝕜 → E) (h : b ≠ a) : dslope f a b = slope f a b := by
  classical
  exact update_of_ne h ..


theorem ContinuousLinearMap.dslope_comp {F : Type*} [NormedAddCommGroup F] [NormedSpace 𝕜 F]
    (f : E →L[𝕜] F) (g : 𝕜 → E) (a b : 𝕜) (H : a = b → DifferentiableAt 𝕜 g a) :
    dslope (f ∘ g) a b = f (dslope g a b) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : ContinuousLinearMap (RingHom.id 𝕜) E F
    g : 𝕜 → E
    a b : 𝕜
    H : Eq a b → DifferentiableAt 𝕜 g a
    ⊢ Eq (dslope (Function.comp (⇑f) g) a b) (f (dslope g a b))
  -/
  rcases eq_or_ne b a with (rfl | hne)
    /-
      case inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : ContinuousLinearMap (RingHom.id 𝕜) E F
      g : 𝕜 → E
      b : 𝕜
      H : Eq b b → DifferentiableAt 𝕜 g b
      ⊢ Eq (dslope (Function.comp (⇑f) g) b b) (f (dslope g b b))
    -/
  · simp only [dslope_same]
    /-
      case inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : ContinuousLinearMap (RingHom.id 𝕜) E F
      g : 𝕜 → E
      b : 𝕜
      H : Eq b b → DifferentiableAt 𝕜 g b
      ⊢ Eq (deriv (Function.comp (⇑f) g) b) (f (deriv g b))
    -/
    exact (f.hasFDerivAt.comp_hasDerivAt b (H rfl).hasDerivAt).deriv
    /-
      🎉 no goals
    -/
    /-
      case inr
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : ContinuousLinearMap (RingHom.id 𝕜) E F
      g : 𝕜 → E
      a b : 𝕜
      H : Eq a b → DifferentiableAt 𝕜 g a
      hne : Ne b a
      ⊢ Eq (dslope (Function.comp (⇑f) g) a b) (f (dslope g a b))
    -/
  · simpa only [dslope_of_ne _ hne] using f.toLinearMap.slope_comp g a b
    /-
      🎉 no goals
    -/


theorem eqOn_dslope_slope (f : 𝕜 → E) (a : 𝕜) : EqOn (dslope f a) (slope f a) {a}ᶜ := fun _ =>
  dslope_of_ne f


theorem dslope_eventuallyEq_slope_of_ne (f : 𝕜 → E) (h : b ≠ a) : dslope f a =ᶠ[𝓝 b] slope f a :=
  (eqOn_dslope_slope f a).eventuallyEq_of_mem (isOpen_ne.mem_nhds h)


theorem dslope_eventuallyEq_slope_punctured_nhds (f : 𝕜 → E) : dslope f a =ᶠ[𝓝[≠] a] slope f a :=
  (eqOn_dslope_slope f a).eventuallyEq_of_mem self_mem_nhdsWithin


@[simp]
theorem sub_smul_dslope (f : 𝕜 → E) (a b : 𝕜) : (b - a) • dslope f a b = f b - f a := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    a b : 𝕜
    ⊢ Eq (HSMul.hSMul (HSub.hSub b a) (dslope f a b)) (HSub.hSub (f b) (f a))
  -/
                                           /-
                                             🎉 no goals
                                           -/
  rcases eq_or_ne b a with (rfl | hne) <;> simp [dslope_of_ne, *]
                                           /-
                                             🎉 no goals
                                           -/


theorem dslope_sub_smul_of_ne (f : 𝕜 → E) (h : b ≠ a) :
    dslope (fun x => (x - a) • f x) a b = f b := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    a b : 𝕜
    f : 𝕜 → E
    h : Ne b a
    ⊢ Eq (dslope (fun x => HSMul.hSMul (HSub.hSub x a) (f x)) a b) (f b)
  -/
  rw [dslope_of_ne _ h, slope_sub_smul _ h.symm]
  /-
    🎉 no goals
  -/


theorem eqOn_dslope_sub_smul (f : 𝕜 → E) (a : 𝕜) :
    EqOn (dslope (fun x => (x - a) • f x) a) f {a}ᶜ := fun _ => dslope_sub_smul_of_ne f


theorem dslope_sub_smul [DecidableEq 𝕜] (f : 𝕜 → E) (a : 𝕜) :
    dslope (fun x => (x - a) • f x) a = update f a (deriv (fun x => (x - a) • f x) a) :=
  eq_update_iff.2 ⟨dslope_same _ _, eqOn_dslope_sub_smul f a⟩


@[simp]
theorem continuousAt_dslope_same : ContinuousAt (dslope f a) a ↔ DifferentiableAt 𝕜 f a := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    a : 𝕜
    ⊢ Iff (ContinuousAt (dslope f a) a) (DifferentiableAt 𝕜 f a)
  -/
  simp only [dslope, continuousAt_update_same, ← hasDerivAt_deriv_iff, hasDerivAt_iff_tendsto_slope]
  /-
    🎉 no goals
  -/


theorem ContinuousWithinAt.of_dslope (h : ContinuousWithinAt (dslope f a) s b) :
    ContinuousWithinAt f s b := by
  have : ContinuousWithinAt (fun x => (x - a) • dslope f a x + f a) s b :=
    ((continuousWithinAt_id.sub continuousWithinAt_const).smul h).add continuousWithinAt_const
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    a b : 𝕜
    s : Set 𝕜
    h : ContinuousWithinAt (dslope f a) s b
    this : ContinuousWithinAt (fun x => HAdd.hAdd (HSMul.hSMul (HSub.hSub x a) (ds …
    ⊢ ContinuousWithinAt f s b
  -/
  simpa only [sub_smul_dslope, sub_add_cancel] using this
  /-
    🎉 no goals
  -/


theorem ContinuousAt.of_dslope (h : ContinuousAt (dslope f a) b) : ContinuousAt f b :=
  (continuousWithinAt_univ _ _).1 h.continuousWithinAt.of_dslope


theorem ContinuousOn.of_dslope (h : ContinuousOn (dslope f a) s) : ContinuousOn f s := fun x hx =>
  (h x hx).of_dslope


theorem continuousWithinAt_dslope_of_ne (h : b ≠ a) :
    ContinuousWithinAt (dslope f a) s b ↔ ContinuousWithinAt f s b := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    a b : 𝕜
    s : Set 𝕜
    h : Ne b a
    ⊢ Iff (ContinuousWithinAt (dslope f a) s b) (ContinuousWithinAt f s b)
  -/
  refine ⟨ContinuousWithinAt.of_dslope, fun hc => ?_⟩
  classical
  simp only [dslope, continuousWithinAt_update_of_ne h]
  exact ((continuousWithinAt_id.sub continuousWithinAt_const).inv₀ (sub_ne_zero.2 h)).smul
    (hc.sub continuousWithinAt_const)


theorem continuousAt_dslope_of_ne (h : b ≠ a) : ContinuousAt (dslope f a) b ↔ ContinuousAt f b := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    a b : 𝕜
    h : Ne b a
    ⊢ Iff (ContinuousAt (dslope f a) b) (ContinuousAt f b)
  -/
  simp only [← continuousWithinAt_univ, continuousWithinAt_dslope_of_ne h]
  /-
    🎉 no goals
  -/


theorem continuousOn_dslope (h : s ∈ 𝓝 a) :
    ContinuousOn (dslope f a) s ↔ ContinuousOn f s ∧ DifferentiableAt 𝕜 f a := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    a : 𝕜
    s : Set 𝕜
    h : Membership.mem (nhds a) s
    ⊢ Iff (ContinuousOn (dslope f a) s) (And (ContinuousOn f s) (DifferentiableAt  …
  -/
  refine ⟨fun hc => ⟨hc.of_dslope, continuousAt_dslope_same.1 <| hc.continuousAt h⟩, ?_⟩
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    a : 𝕜
    s : Set 𝕜
    h : Membership.mem (nhds a) s
    ⊢ And (ContinuousOn f s) (DifferentiableAt 𝕜 f a) → ContinuousOn (dslope f a) s
  -/
  rintro ⟨hc, hd⟩ x hx
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    a : 𝕜
    s : Set 𝕜
    h : Membership.mem (nhds a) s
    hc : ContinuousOn f s
    hd : DifferentiableAt 𝕜 f a
    x : 𝕜
    hx : Membership.mem s x
    ⊢ ContinuousWithinAt (dslope f a) s x
  -/
  rcases eq_or_ne x a with (rfl | hne)
  exacts [(continuousAt_dslope_same.2 hd).continuousWithinAt,
    (continuousWithinAt_dslope_of_ne hne).2 (hc x hx)]


theorem DifferentiableWithinAt.of_dslope (h : DifferentiableWithinAt 𝕜 (dslope f a) s b) :
    DifferentiableWithinAt 𝕜 f s b := by
  simpa only [id, sub_smul_dslope f a, sub_add_cancel] using
    ((differentiableWithinAt_id.sub_const a).smul h).add_const (f a)


theorem DifferentiableAt.of_dslope (h : DifferentiableAt 𝕜 (dslope f a) b) :
    DifferentiableAt 𝕜 f b :=
  differentiableWithinAt_univ.1 h.differentiableWithinAt.of_dslope


theorem DifferentiableOn.of_dslope (h : DifferentiableOn 𝕜 (dslope f a) s) :
    DifferentiableOn 𝕜 f s := fun x hx => (h x hx).of_dslope


theorem differentiableWithinAt_dslope_of_ne (h : b ≠ a) :
    DifferentiableWithinAt 𝕜 (dslope f a) s b ↔ DifferentiableWithinAt 𝕜 f s b := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    a b : 𝕜
    s : Set 𝕜
    h : Ne b a
    ⊢ Iff (DifferentiableWithinAt 𝕜 (dslope f a) s b) (DifferentiableWithinAt 𝕜 f  …
  -/
  refine ⟨DifferentiableWithinAt.of_dslope, fun hd => ?_⟩
  refine (((differentiableWithinAt_id.sub_const a).inv (sub_ne_zero.2 h)).smul
    (hd.sub_const (f a))).congr_of_eventuallyEq ?_ (dslope_of_ne _ h)
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    a b : 𝕜
    s : Set 𝕜
    h : Ne b a
    hd : DifferentiableWithinAt 𝕜 f s b
    ⊢ (nhdsWithin b s).EventuallyEq (dslope f a) fun y => HSMul.hSMul (Inv.inv (HS …
  -/
  refine (eqOn_dslope_slope _ _).eventuallyEq_of_mem ?_
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    a b : 𝕜
    s : Set 𝕜
    h : Ne b a
    hd : DifferentiableWithinAt 𝕜 f s b
    ⊢ Membership.mem (nhdsWithin b s) (HasCompl.compl (Singleton.singleton a))
  -/
  exact mem_nhdsWithin_of_mem_nhds (isOpen_ne.mem_nhds h)
  /-
    🎉 no goals
  -/


theorem differentiableOn_dslope_of_nmem (h : a ∉ s) :
    DifferentiableOn 𝕜 (dslope f a) s ↔ DifferentiableOn 𝕜 f s :=
  forall_congr' fun _ =>
    forall_congr' fun hx => differentiableWithinAt_dslope_of_ne <| ne_of_mem_of_not_mem hx h


theorem differentiableAt_dslope_of_ne (h : b ≠ a) :
    DifferentiableAt 𝕜 (dslope f a) b ↔ DifferentiableAt 𝕜 f b := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    a b : 𝕜
    h : Ne b a
    ⊢ Iff (DifferentiableAt 𝕜 (dslope f a) b) (DifferentiableAt 𝕜 f b)
  -/
  simp only [← differentiableWithinAt_univ, differentiableWithinAt_dslope_of_ne h]
  /-
    🎉 no goals
  -/

