nonrec theorem HasDerivAtFilter.add (hf : HasDerivAtFilter f f' x L)
    (hg : HasDerivAtFilter g g' x L) : HasDerivAtFilter (fun y => f y + g y) (f' + g') x L := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f g : 𝕜 → F
    f' g' : F
    x : 𝕜
    L : Filter 𝕜
    hf : HasDerivAtFilter f f' x L
    hg : HasDerivAtFilter g g' x L
    ⊢ HasDerivAtFilter (fun y => HAdd.hAdd (f y) (g y)) (HAdd.hAdd f' g') x L
  -/
  simpa using (hf.add hg).hasDerivAtFilter
  /-
    🎉 no goals
  -/


nonrec theorem HasStrictDerivAt.add (hf : HasStrictDerivAt f f' x) (hg : HasStrictDerivAt g g' x) :
                                                            /-
                                                              𝕜 : Type u
                                                              inst✝² : NontriviallyNormedField 𝕜
                                                              F : Type v
                                                              inst✝¹ : NormedAddCommGroup F
                                                              inst✝ : NormedSpace 𝕜 F
                                                              f g : 𝕜 → F
                                                              f' g' : F
                                                              x : 𝕜
                                                              hf : HasStrictDerivAt f f' x
                                                              hg : HasStrictDerivAt g g' x
                                                              ⊢ HasStrictDerivAt (fun y => HAdd.hAdd (f y) (g y)) (HAdd.hAdd f' g') x
                                                            -/
    HasStrictDerivAt (fun y => f y + g y) (f' + g') x := by simpa using (hf.add hg).hasStrictDerivAt
                                                            /-
                                                              🎉 no goals
                                                            -/


nonrec theorem HasDerivWithinAt.add (hf : HasDerivWithinAt f f' s x)
    (hg : HasDerivWithinAt g g' s x) : HasDerivWithinAt (fun y => f y + g y) (f' + g') s x :=
  hf.add hg


nonrec theorem HasDerivAt.add (hf : HasDerivAt f f' x) (hg : HasDerivAt g g' x) :
    HasDerivAt (fun x => f x + g x) (f' + g') x :=
  hf.add hg


theorem derivWithin_add (hxs : UniqueDiffWithinAt 𝕜 s x) (hf : DifferentiableWithinAt 𝕜 f s x)
    (hg : DifferentiableWithinAt 𝕜 g s x) :
    derivWithin (fun y => f y + g y) s x = derivWithin f s x + derivWithin g s x :=
  (hf.hasDerivWithinAt.add hg.hasDerivWithinAt).derivWithin hxs


@[simp]
theorem deriv_add (hf : DifferentiableAt 𝕜 f x) (hg : DifferentiableAt 𝕜 g x) :
    deriv (fun y => f y + g y) x = deriv f x + deriv g x :=
  (hf.hasDerivAt.add hg.hasDerivAt).deriv


theorem HasStrictDerivAt.add_const (c : F) (hf : HasStrictDerivAt f f' x) :
    HasStrictDerivAt (fun y ↦ f y + c) f' x :=
  add_zero f' ▸ hf.add (hasStrictDerivAt_const x c)


theorem HasDerivAtFilter.add_const (hf : HasDerivAtFilter f f' x L) (c : F) :
    HasDerivAtFilter (fun y => f y + c) f' x L :=
  add_zero f' ▸ hf.add (hasDerivAtFilter_const x L c)


nonrec theorem HasDerivWithinAt.add_const (hf : HasDerivWithinAt f f' s x) (c : F) :
    HasDerivWithinAt (fun y => f y + c) f' s x :=
  hf.add_const c


nonrec theorem HasDerivAt.add_const (hf : HasDerivAt f f' x) (c : F) :
    HasDerivAt (fun x => f x + c) f' x :=
  hf.add_const c


theorem derivWithin_add_const (hxs : UniqueDiffWithinAt 𝕜 s x) (c : F) :
    derivWithin (fun y => f y + c) s x = derivWithin f s x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    x : 𝕜
    s : Set 𝕜
    hxs : UniqueDiffWithinAt 𝕜 s x
    c : F
    ⊢ Eq (derivWithin (fun y => HAdd.hAdd (f y) c) s x) (derivWithin f s x)
  -/
  simp only [derivWithin, fderivWithin_add_const hxs]
  /-
    🎉 no goals
  -/


theorem deriv_add_const (c : F) : deriv (fun y => f y + c) x = deriv f x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    x : 𝕜
    c : F
    ⊢ Eq (deriv (fun y => HAdd.hAdd (f y) c) x) (deriv f x)
  -/
  simp only [deriv, fderiv_add_const]
  /-
    🎉 no goals
  -/


@[simp]
theorem deriv_add_const' (c : F) : (deriv fun y => f y + c) = deriv f :=
  funext fun _ => deriv_add_const c


theorem HasStrictDerivAt.const_add (c : F) (hf : HasStrictDerivAt f f' x) :
    HasStrictDerivAt (fun y ↦ c + f y) f' x :=
  zero_add f' ▸ (hasStrictDerivAt_const x c).add hf


theorem HasDerivAtFilter.const_add (c : F) (hf : HasDerivAtFilter f f' x L) :
    HasDerivAtFilter (fun y => c + f y) f' x L :=
  zero_add f' ▸ (hasDerivAtFilter_const x L c).add hf


nonrec theorem HasDerivWithinAt.const_add (c : F) (hf : HasDerivWithinAt f f' s x) :
    HasDerivWithinAt (fun y => c + f y) f' s x :=
  hf.const_add c


nonrec theorem HasDerivAt.const_add (c : F) (hf : HasDerivAt f f' x) :
    HasDerivAt (fun x => c + f x) f' x :=
  hf.const_add c


theorem derivWithin_const_add (hxs : UniqueDiffWithinAt 𝕜 s x) (c : F) :
    derivWithin (fun y => c + f y) s x = derivWithin f s x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    x : 𝕜
    s : Set 𝕜
    hxs : UniqueDiffWithinAt 𝕜 s x
    c : F
    ⊢ Eq (derivWithin (fun y => HAdd.hAdd c (f y)) s x) (derivWithin f s x)
  -/
  simp only [derivWithin, fderivWithin_const_add hxs]
  /-
    🎉 no goals
  -/


theorem deriv_const_add (c : F) : deriv (fun y => c + f y) x = deriv f x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    x : 𝕜
    c : F
    ⊢ Eq (deriv (fun y => HAdd.hAdd c (f y)) x) (deriv f x)
  -/
  simp only [deriv, fderiv_const_add]
  /-
    🎉 no goals
  -/


@[simp]
theorem deriv_const_add' (c : F) : (deriv fun y => c + f y) = deriv f :=
  funext fun _ => deriv_const_add c


lemma differentiableAt_comp_const_add {a b : 𝕜} :
    DifferentiableAt 𝕜 (fun x ↦ f (b + x)) a ↔ DifferentiableAt 𝕜 f (b + a) := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    a b : 𝕜
    ⊢ Iff (DifferentiableAt 𝕜 (fun x => f (HAdd.hAdd b x)) a) (DifferentiableAt 𝕜  …
  -/
  refine ⟨fun H ↦ ?_, fun H ↦ H.comp _ (differentiable_id.const_add _).differentiableAt⟩
  convert DifferentiableAt.comp (b + a) (by simpa)
    (differentiable_id.const_add (-b)).differentiableAt
  /-
    case h.e'_11
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    a b : 𝕜
    H : DifferentiableAt 𝕜 (fun x => f (HAdd.hAdd b x)) a
    ⊢ Eq f (Function.comp (fun x => f (HAdd.hAdd b x)) fun y => HAdd.hAdd (Neg.neg …
  -/
  ext
  /-
    case h.e'_11.h
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    a b : 𝕜
    H : DifferentiableAt 𝕜 (fun x => f (HAdd.hAdd b x)) a
    x✝ : 𝕜
    ⊢ Eq (f x✝) (Function.comp (fun x => f (HAdd.hAdd b x)) (fun y => HAdd.hAdd (N …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma differentiableAt_comp_add_const {a b : 𝕜} :
    DifferentiableAt 𝕜 (fun x ↦ f (x + b)) a ↔ DifferentiableAt 𝕜 f (a + b) := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    a b : 𝕜
    ⊢ Iff (DifferentiableAt 𝕜 (fun x => f (HAdd.hAdd x b)) a) (DifferentiableAt 𝕜  …
  -/
  simpa [add_comm b] using differentiableAt_comp_const_add (f := f) (b := b)
  /-
    🎉 no goals
  -/


lemma differentiableAt_iff_comp_const_add {a b : 𝕜} :
    DifferentiableAt 𝕜 f a ↔ DifferentiableAt 𝕜 (fun x ↦ f (b + x)) (-b + a) := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    a b : 𝕜
    ⊢ Iff (DifferentiableAt 𝕜 f a) (DifferentiableAt 𝕜 (fun x => f (HAdd.hAdd b x) …
  -/
  simp [differentiableAt_comp_const_add]
  /-
    🎉 no goals
  -/


lemma differentiableAt_iff_comp_add_const {a b : 𝕜} :
    DifferentiableAt 𝕜 f a ↔ DifferentiableAt 𝕜 (fun x ↦ f (x + b)) (a - b) := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    a b : 𝕜
    ⊢ Iff (DifferentiableAt 𝕜 f a) (DifferentiableAt 𝕜 (fun x => f (HAdd.hAdd x b) …
  -/
  simp [differentiableAt_comp_add_const]
  /-
    🎉 no goals
  -/


theorem HasDerivAtFilter.sum (h : ∀ i ∈ u, HasDerivAtFilter (A i) (A' i) x L) :
    HasDerivAtFilter (fun y => ∑ i ∈ u, A i y) (∑ i ∈ u, A' i) x L := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    x : 𝕜
    L : Filter 𝕜
    ι : Type u_1
    u : Finset ι
    A : ι → 𝕜 → F
    A' : ι → F
    h : ∀ (i : ι), Membership.mem u i → HasDerivAtFilter (A i) (A' i) x L
    ⊢ HasDerivAtFilter (fun y => u.sum fun i => A i y) (u.sum fun i => A' i) x L
  -/
  simpa [ContinuousLinearMap.sum_apply] using (HasFDerivAtFilter.sum h).hasDerivAtFilter
  /-
    🎉 no goals
  -/


theorem HasStrictDerivAt.sum (h : ∀ i ∈ u, HasStrictDerivAt (A i) (A' i) x) :
    HasStrictDerivAt (fun y => ∑ i ∈ u, A i y) (∑ i ∈ u, A' i) x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    x : 𝕜
    ι : Type u_1
    u : Finset ι
    A : ι → 𝕜 → F
    A' : ι → F
    h : ∀ (i : ι), Membership.mem u i → HasStrictDerivAt (A i) (A' i) x
    ⊢ HasStrictDerivAt (fun y => u.sum fun i => A i y) (u.sum fun i => A' i) x
  -/
  simpa [ContinuousLinearMap.sum_apply] using (HasStrictFDerivAt.sum h).hasStrictDerivAt
  /-
    🎉 no goals
  -/


theorem HasDerivWithinAt.sum (h : ∀ i ∈ u, HasDerivWithinAt (A i) (A' i) s x) :
    HasDerivWithinAt (fun y => ∑ i ∈ u, A i y) (∑ i ∈ u, A' i) s x :=
  HasDerivAtFilter.sum h


theorem HasDerivAt.sum (h : ∀ i ∈ u, HasDerivAt (A i) (A' i) x) :
    HasDerivAt (fun y => ∑ i ∈ u, A i y) (∑ i ∈ u, A' i) x :=
  HasDerivAtFilter.sum h


theorem derivWithin_sum (hxs : UniqueDiffWithinAt 𝕜 s x)
    (h : ∀ i ∈ u, DifferentiableWithinAt 𝕜 (A i) s x) :
    derivWithin (fun y => ∑ i ∈ u, A i y) s x = ∑ i ∈ u, derivWithin (A i) s x :=
  (HasDerivWithinAt.sum fun i hi => (h i hi).hasDerivWithinAt).derivWithin hxs


@[simp]
theorem deriv_sum (h : ∀ i ∈ u, DifferentiableAt 𝕜 (A i) x) :
    deriv (fun y => ∑ i ∈ u, A i y) x = ∑ i ∈ u, deriv (A i) x :=
  (HasDerivAt.sum fun i hi => (h i hi).hasDerivAt).deriv


nonrec theorem HasDerivAtFilter.neg (h : HasDerivAtFilter f f' x L) :
                                                     /-
                                                       𝕜 : Type u
                                                       inst✝² : NontriviallyNormedField 𝕜
                                                       F : Type v
                                                       inst✝¹ : NormedAddCommGroup F
                                                       inst✝ : NormedSpace 𝕜 F
                                                       f : 𝕜 → F
                                                       f' : F
                                                       x : 𝕜
                                                       L : Filter 𝕜
                                                       h : HasDerivAtFilter f f' x L
                                                       ⊢ HasDerivAtFilter (fun x => Neg.neg (f x)) (Neg.neg f') x L
                                                     -/
    HasDerivAtFilter (fun x => -f x) (-f') x L := by simpa using h.neg.hasDerivAtFilter
                                                     /-
                                                       🎉 no goals
                                                     -/


nonrec theorem HasDerivWithinAt.neg (h : HasDerivWithinAt f f' s x) :
    HasDerivWithinAt (fun x => -f x) (-f') s x :=
  h.neg


nonrec theorem HasDerivAt.neg (h : HasDerivAt f f' x) : HasDerivAt (fun x => -f x) (-f') x :=
  h.neg


nonrec theorem HasStrictDerivAt.neg (h : HasStrictDerivAt f f' x) :
                                                   /-
                                                     𝕜 : Type u
                                                     inst✝² : NontriviallyNormedField 𝕜
                                                     F : Type v
                                                     inst✝¹ : NormedAddCommGroup F
                                                     inst✝ : NormedSpace 𝕜 F
                                                     f : 𝕜 → F
                                                     f' : F
                                                     x : 𝕜
                                                     h : HasStrictDerivAt f f' x
                                                     ⊢ HasStrictDerivAt (fun x => Neg.neg (f x)) (Neg.neg f') x
                                                   -/
    HasStrictDerivAt (fun x => -f x) (-f') x := by simpa using h.neg.hasStrictDerivAt
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem derivWithin.neg (hxs : UniqueDiffWithinAt 𝕜 s x) :
    derivWithin (fun y => -f y) s x = -derivWithin f s x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    x : 𝕜
    s : Set 𝕜
    hxs : UniqueDiffWithinAt 𝕜 s x
    ⊢ Eq (derivWithin (fun y => Neg.neg (f y)) s x) (Neg.neg (derivWithin f s x))
  -/
  simp only [derivWithin, fderivWithin_neg hxs, ContinuousLinearMap.neg_apply]
  /-
    🎉 no goals
  -/


theorem deriv.neg : deriv (fun y => -f y) x = -deriv f x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    x : 𝕜
    ⊢ Eq (deriv (fun y => Neg.neg (f y)) x) (Neg.neg (deriv f x))
  -/
  simp only [deriv, fderiv_neg, ContinuousLinearMap.neg_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem deriv.neg' : (deriv fun y => -f y) = fun x => -deriv f x :=
  funext fun _ => deriv.neg


theorem hasDerivAtFilter_neg : HasDerivAtFilter Neg.neg (-1) x L :=
  HasDerivAtFilter.neg <| hasDerivAtFilter_id _ _


theorem hasDerivWithinAt_neg : HasDerivWithinAt Neg.neg (-1) s x :=
  hasDerivAtFilter_neg _ _


theorem hasDerivAt_neg : HasDerivAt Neg.neg (-1) x :=
  hasDerivAtFilter_neg _ _


theorem hasDerivAt_neg' : HasDerivAt (fun x => -x) (-1) x :=
  hasDerivAtFilter_neg _ _


theorem hasStrictDerivAt_neg : HasStrictDerivAt Neg.neg (-1) x :=
  HasStrictDerivAt.neg <| hasStrictDerivAt_id _


theorem deriv_neg : deriv Neg.neg x = -1 :=
  HasDerivAt.deriv (hasDerivAt_neg x)


@[simp]
theorem deriv_neg' : deriv (Neg.neg : 𝕜 → 𝕜) = fun _ => -1 :=
  funext deriv_neg


@[simp]
theorem deriv_neg'' : deriv (fun x : 𝕜 => -x) x = -1 :=
  deriv_neg x


theorem derivWithin_neg (hxs : UniqueDiffWithinAt 𝕜 s x) : derivWithin Neg.neg s x = -1 :=
  (hasDerivWithinAt_neg x s).derivWithin hxs


theorem differentiable_neg : Differentiable 𝕜 (Neg.neg : 𝕜 → 𝕜) :=
  Differentiable.neg differentiable_id


theorem differentiableOn_neg : DifferentiableOn 𝕜 (Neg.neg : 𝕜 → 𝕜) s :=
  DifferentiableOn.neg differentiableOn_id


lemma differentiableAt_comp_neg {a : 𝕜} :
    DifferentiableAt 𝕜 (fun x ↦ f (-x)) a ↔ DifferentiableAt 𝕜 f (-a) := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    a : 𝕜
    ⊢ Iff (DifferentiableAt 𝕜 (fun x => f (Neg.neg x)) a) (DifferentiableAt 𝕜 f (N …
  -/
  refine ⟨fun H ↦ ?_, fun H ↦ H.comp a differentiable_neg.differentiableAt⟩
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    a : 𝕜
    H : DifferentiableAt 𝕜 (fun x => f (Neg.neg x)) a
    ⊢ DifferentiableAt 𝕜 f (Neg.neg a)
  -/
  convert ((neg_neg a).symm ▸ H).comp (-a) differentiable_neg.differentiableAt
  /-
    case h.e'_11
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    a : 𝕜
    H : DifferentiableAt 𝕜 (fun x => f (Neg.neg x)) a
    ⊢ Eq f (Function.comp (fun x => f (Neg.neg x)) Neg.neg)
  -/
  ext
  /-
    case h.e'_11.h
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    a : 𝕜
    H : DifferentiableAt 𝕜 (fun x => f (Neg.neg x)) a
    x✝ : 𝕜
    ⊢ Eq (f x✝) (Function.comp (fun x => f (Neg.neg x)) Neg.neg x✝)
  -/
  simp only [Function.comp_apply, neg_neg]
  /-
    🎉 no goals
  -/


lemma differentiableAt_iff_comp_neg {a : 𝕜} :
    DifferentiableAt 𝕜 f a ↔ DifferentiableAt 𝕜 (fun x ↦ f (-x)) (-a) := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    a : 𝕜
    ⊢ Iff (DifferentiableAt 𝕜 f a) (DifferentiableAt 𝕜 (fun x => f (Neg.neg x)) (N …
  -/
  simp_rw [← differentiableAt_comp_neg, neg_neg]
  /-
    🎉 no goals
  -/


theorem HasDerivAtFilter.sub (hf : HasDerivAtFilter f f' x L) (hg : HasDerivAtFilter g g' x L) :
    HasDerivAtFilter (fun x => f x - g x) (f' - g') x L := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f g : 𝕜 → F
    f' g' : F
    x : 𝕜
    L : Filter 𝕜
    hf : HasDerivAtFilter f f' x L
    hg : HasDerivAtFilter g g' x L
    ⊢ HasDerivAtFilter (fun x => HSub.hSub (f x) (g x)) (HSub.hSub f' g') x L
  -/
  simpa only [sub_eq_add_neg] using hf.add hg.neg
  /-
    🎉 no goals
  -/


nonrec theorem HasDerivWithinAt.sub (hf : HasDerivWithinAt f f' s x)
    (hg : HasDerivWithinAt g g' s x) : HasDerivWithinAt (fun x => f x - g x) (f' - g') s x :=
  hf.sub hg


nonrec theorem HasDerivAt.sub (hf : HasDerivAt f f' x) (hg : HasDerivAt g g' x) :
    HasDerivAt (fun x => f x - g x) (f' - g') x :=
  hf.sub hg


theorem HasStrictDerivAt.sub (hf : HasStrictDerivAt f f' x) (hg : HasStrictDerivAt g g' x) :
    HasStrictDerivAt (fun x => f x - g x) (f' - g') x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f g : 𝕜 → F
    f' g' : F
    x : 𝕜
    hf : HasStrictDerivAt f f' x
    hg : HasStrictDerivAt g g' x
    ⊢ HasStrictDerivAt (fun x => HSub.hSub (f x) (g x)) (HSub.hSub f' g') x
  -/
  simpa only [sub_eq_add_neg] using hf.add hg.neg
  /-
    🎉 no goals
  -/


theorem derivWithin_sub (hxs : UniqueDiffWithinAt 𝕜 s x) (hf : DifferentiableWithinAt 𝕜 f s x)
    (hg : DifferentiableWithinAt 𝕜 g s x) :
    derivWithin (fun y => f y - g y) s x = derivWithin f s x - derivWithin g s x :=
  (hf.hasDerivWithinAt.sub hg.hasDerivWithinAt).derivWithin hxs


@[simp]
theorem deriv_sub (hf : DifferentiableAt 𝕜 f x) (hg : DifferentiableAt 𝕜 g x) :
    deriv (fun y => f y - g y) x = deriv f x - deriv g x :=
  (hf.hasDerivAt.sub hg.hasDerivAt).deriv


theorem HasDerivAtFilter.sub_const (hf : HasDerivAtFilter f f' x L) (c : F) :
    HasDerivAtFilter (fun x => f x - c) f' x L := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    f' : F
    x : 𝕜
    L : Filter 𝕜
    hf : HasDerivAtFilter f f' x L
    c : F
    ⊢ HasDerivAtFilter (fun x => HSub.hSub (f x) c) f' x L
  -/
  simpa only [sub_eq_add_neg] using hf.add_const (-c)
  /-
    🎉 no goals
  -/


nonrec theorem HasDerivWithinAt.sub_const (hf : HasDerivWithinAt f f' s x) (c : F) :
    HasDerivWithinAt (fun x => f x - c) f' s x :=
  hf.sub_const c


nonrec theorem HasDerivAt.sub_const (hf : HasDerivAt f f' x) (c : F) :
    HasDerivAt (fun x => f x - c) f' x :=
  hf.sub_const c


theorem derivWithin_sub_const (hxs : UniqueDiffWithinAt 𝕜 s x) (c : F) :
    derivWithin (fun y => f y - c) s x = derivWithin f s x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    x : 𝕜
    s : Set 𝕜
    hxs : UniqueDiffWithinAt 𝕜 s x
    c : F
    ⊢ Eq (derivWithin (fun y => HSub.hSub (f y) c) s x) (derivWithin f s x)
  -/
  simp only [derivWithin, fderivWithin_sub_const hxs]
  /-
    🎉 no goals
  -/


theorem deriv_sub_const (c : F) : deriv (fun y => f y - c) x = deriv f x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    x : 𝕜
    c : F
    ⊢ Eq (deriv (fun y => HSub.hSub (f y) c) x) (deriv f x)
  -/
  simp only [deriv, fderiv_sub_const]
  /-
    🎉 no goals
  -/


theorem HasDerivAtFilter.const_sub (c : F) (hf : HasDerivAtFilter f f' x L) :
    HasDerivAtFilter (fun x => c - f x) (-f') x L := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    f' : F
    x : 𝕜
    L : Filter 𝕜
    c : F
    hf : HasDerivAtFilter f f' x L
    ⊢ HasDerivAtFilter (fun x => HSub.hSub c (f x)) (Neg.neg f') x L
  -/
  simpa only [sub_eq_add_neg] using hf.neg.const_add c
  /-
    🎉 no goals
  -/


nonrec theorem HasDerivWithinAt.const_sub (c : F) (hf : HasDerivWithinAt f f' s x) :
    HasDerivWithinAt (fun x => c - f x) (-f') s x :=
  hf.const_sub c


theorem HasStrictDerivAt.const_sub (c : F) (hf : HasStrictDerivAt f f' x) :
    HasStrictDerivAt (fun x => c - f x) (-f') x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    f' : F
    x : 𝕜
    c : F
    hf : HasStrictDerivAt f f' x
    ⊢ HasStrictDerivAt (fun x => HSub.hSub c (f x)) (Neg.neg f') x
  -/
  simpa only [sub_eq_add_neg] using hf.neg.const_add c
  /-
    🎉 no goals
  -/


nonrec theorem HasDerivAt.const_sub (c : F) (hf : HasDerivAt f f' x) :
    HasDerivAt (fun x => c - f x) (-f') x :=
  hf.const_sub c


theorem derivWithin_const_sub (hxs : UniqueDiffWithinAt 𝕜 s x) (c : F) :
    derivWithin (fun y => c - f y) s x = -derivWithin f s x := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    x : 𝕜
    s : Set 𝕜
    hxs : UniqueDiffWithinAt 𝕜 s x
    c : F
    ⊢ Eq (derivWithin (fun y => HSub.hSub c (f y)) s x) (Neg.neg (derivWithin f s  …
  -/
  simp [derivWithin, fderivWithin_const_sub hxs]
  /-
    🎉 no goals
  -/


theorem deriv_const_sub (c : F) : deriv (fun y => c - f y) x = -deriv f x := by
  simp only [← derivWithin_univ,
    derivWithin_const_sub (uniqueDiffWithinAt_univ : UniqueDiffWithinAt 𝕜 _ _)]


lemma differentiableAt_comp_sub_const {a b : 𝕜} :
    DifferentiableAt 𝕜 (fun x ↦ f (x - b)) a ↔ DifferentiableAt 𝕜 f (a - b) := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    a b : 𝕜
    ⊢ Iff (DifferentiableAt 𝕜 (fun x => f (HSub.hSub x b)) a) (DifferentiableAt 𝕜  …
  -/
  simp [sub_eq_add_neg, differentiableAt_comp_add_const]
  /-
    🎉 no goals
  -/


lemma differentiableAt_comp_const_sub {a b : 𝕜} :
    DifferentiableAt 𝕜 (fun x ↦ f (b - x)) a ↔ DifferentiableAt 𝕜 f (b - a) := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    a b : 𝕜
    ⊢ Iff (DifferentiableAt 𝕜 (fun x => f (HSub.hSub b x)) a) (DifferentiableAt 𝕜  …
  -/
  refine ⟨fun H ↦ ?_, fun H ↦ H.comp a (differentiable_id.const_sub _).differentiableAt⟩
  convert ((sub_sub_cancel _ a).symm ▸ H).comp (b - a)
    (differentiable_id.const_sub _).differentiableAt
  /-
    case h.e'_11
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    a b : 𝕜
    H : DifferentiableAt 𝕜 (fun x => f (HSub.hSub b x)) a
    ⊢ Eq f (Function.comp (fun x => f (HSub.hSub b x)) (HSub.hSub b))
  -/
  ext
  /-
    case h.e'_11.h
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    a b : 𝕜
    H : DifferentiableAt 𝕜 (fun x => f (HSub.hSub b x)) a
    x✝ : 𝕜
    ⊢ Eq (f x✝) (Function.comp (fun x => f (HSub.hSub b x)) (HSub.hSub b) x✝)
  -/
  simp
  /-
    🎉 no goals
  -/


lemma differentiableAt_iff_comp_sub_const {a b : 𝕜} :
    DifferentiableAt 𝕜 f a ↔ DifferentiableAt 𝕜 (fun x ↦ f (x - b)) (a + b) := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    a b : 𝕜
    ⊢ Iff (DifferentiableAt 𝕜 f a) (DifferentiableAt 𝕜 (fun x => f (HSub.hSub x b) …
  -/
  simp [sub_eq_add_neg, differentiableAt_comp_add_const]
  /-
    🎉 no goals
  -/


lemma differentiableAt_iff_comp_const_sub {a b : 𝕜} :
    DifferentiableAt 𝕜 f a ↔ DifferentiableAt 𝕜 (fun x ↦ f (b - x)) (b - a) := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    a b : 𝕜
    ⊢ Iff (DifferentiableAt 𝕜 f a) (DifferentiableAt 𝕜 (fun x => f (HSub.hSub b x) …
  -/
  simp [differentiableAt_comp_const_sub]
  /-
    🎉 no goals
  -/


