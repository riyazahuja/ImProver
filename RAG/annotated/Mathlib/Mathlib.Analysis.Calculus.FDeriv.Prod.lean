protected theorem HasStrictFDerivAt.prod (hf₁ : HasStrictFDerivAt f₁ f₁' x)
    (hf₂ : HasStrictFDerivAt f₂ f₂' x) :
    HasStrictFDerivAt (fun x => (f₁ x, f₂ x)) (f₁'.prod f₂') x :=
  .of_isLittleO <| hf₁.isLittleO.prod_left hf₂.isLittleO


theorem HasFDerivAtFilter.prod (hf₁ : HasFDerivAtFilter f₁ f₁' x L)
    (hf₂ : HasFDerivAtFilter f₂ f₂' x L) :
    HasFDerivAtFilter (fun x => (f₁ x, f₂ x)) (f₁'.prod f₂') x L :=
  .of_isLittleO <| hf₁.isLittleO.prod_left hf₂.isLittleO


@[fun_prop]
nonrec theorem HasFDerivWithinAt.prod (hf₁ : HasFDerivWithinAt f₁ f₁' s x)
    (hf₂ : HasFDerivWithinAt f₂ f₂' s x) :
    HasFDerivWithinAt (fun x => (f₁ x, f₂ x)) (f₁'.prod f₂') s x :=
  hf₁.prod hf₂


@[fun_prop]
nonrec theorem HasFDerivAt.prod (hf₁ : HasFDerivAt f₁ f₁' x) (hf₂ : HasFDerivAt f₂ f₂' x) :
    HasFDerivAt (fun x => (f₁ x, f₂ x)) (f₁'.prod f₂') x :=
  hf₁.prod hf₂


@[fun_prop]
theorem hasFDerivAt_prod_mk_left (e₀ : E) (f₀ : F) :
    HasFDerivAt (fun e : E => (e, f₀)) (inl 𝕜 E F) e₀ :=
  (hasFDerivAt_id e₀).prod (hasFDerivAt_const f₀ e₀)


@[fun_prop]
theorem hasFDerivAt_prod_mk_right (e₀ : E) (f₀ : F) :
    HasFDerivAt (fun f : F => (e₀, f)) (inr 𝕜 E F) f₀ :=
  (hasFDerivAt_const e₀ f₀).prod (hasFDerivAt_id f₀)


@[fun_prop]
theorem DifferentiableWithinAt.prod (hf₁ : DifferentiableWithinAt 𝕜 f₁ s x)
    (hf₂ : DifferentiableWithinAt 𝕜 f₂ s x) :
    DifferentiableWithinAt 𝕜 (fun x : E => (f₁ x, f₂ x)) s x :=
  (hf₁.hasFDerivWithinAt.prod hf₂.hasFDerivWithinAt).differentiableWithinAt


@[simp, fun_prop]
theorem DifferentiableAt.prod (hf₁ : DifferentiableAt 𝕜 f₁ x) (hf₂ : DifferentiableAt 𝕜 f₂ x) :
    DifferentiableAt 𝕜 (fun x : E => (f₁ x, f₂ x)) x :=
  (hf₁.hasFDerivAt.prod hf₂.hasFDerivAt).differentiableAt


@[fun_prop]
theorem DifferentiableOn.prod (hf₁ : DifferentiableOn 𝕜 f₁ s) (hf₂ : DifferentiableOn 𝕜 f₂ s) :
    DifferentiableOn 𝕜 (fun x : E => (f₁ x, f₂ x)) s := fun x hx =>
  DifferentiableWithinAt.prod (hf₁ x hx) (hf₂ x hx)


@[simp, fun_prop]
theorem Differentiable.prod (hf₁ : Differentiable 𝕜 f₁) (hf₂ : Differentiable 𝕜 f₂) :
    Differentiable 𝕜 fun x : E => (f₁ x, f₂ x) := fun x => DifferentiableAt.prod (hf₁ x) (hf₂ x)


theorem DifferentiableAt.fderiv_prod (hf₁ : DifferentiableAt 𝕜 f₁ x)
    (hf₂ : DifferentiableAt 𝕜 f₂ x) :
    fderiv 𝕜 (fun x : E => (f₁ x, f₂ x)) x = (fderiv 𝕜 f₁ x).prod (fderiv 𝕜 f₂ x) :=
  (hf₁.hasFDerivAt.prod hf₂.hasFDerivAt).fderiv


theorem DifferentiableWithinAt.fderivWithin_prod (hf₁ : DifferentiableWithinAt 𝕜 f₁ s x)
    (hf₂ : DifferentiableWithinAt 𝕜 f₂ s x) (hxs : UniqueDiffWithinAt 𝕜 s x) :
    fderivWithin 𝕜 (fun x : E => (f₁ x, f₂ x)) s x =
      (fderivWithin 𝕜 f₁ s x).prod (fderivWithin 𝕜 f₂ s x) :=
  (hf₁.hasFDerivWithinAt.prod hf₂.hasFDerivWithinAt).fderivWithin hxs


@[fun_prop]
theorem hasStrictFDerivAt_fst : HasStrictFDerivAt (@Prod.fst E F) (fst 𝕜 E F) p :=
  (fst 𝕜 E F).hasStrictFDerivAt


@[fun_prop]
protected theorem HasStrictFDerivAt.fst (h : HasStrictFDerivAt f₂ f₂' x) :
    HasStrictFDerivAt (fun x => (f₂ x).1) ((fst 𝕜 F G).comp f₂') x :=
  hasStrictFDerivAt_fst.comp x h


theorem hasFDerivAtFilter_fst {L : Filter (E × F)} :
    HasFDerivAtFilter (@Prod.fst E F) (fst 𝕜 E F) p L :=
  (fst 𝕜 E F).hasFDerivAtFilter


protected theorem HasFDerivAtFilter.fst (h : HasFDerivAtFilter f₂ f₂' x L) :
    HasFDerivAtFilter (fun x => (f₂ x).1) ((fst 𝕜 F G).comp f₂') x L :=
  hasFDerivAtFilter_fst.comp x h tendsto_map


@[fun_prop]
theorem hasFDerivAt_fst : HasFDerivAt (@Prod.fst E F) (fst 𝕜 E F) p :=
  hasFDerivAtFilter_fst


@[fun_prop]
protected nonrec theorem HasFDerivAt.fst (h : HasFDerivAt f₂ f₂' x) :
    HasFDerivAt (fun x => (f₂ x).1) ((fst 𝕜 F G).comp f₂') x :=
  h.fst


@[fun_prop]
theorem hasFDerivWithinAt_fst {s : Set (E × F)} :
    HasFDerivWithinAt (@Prod.fst E F) (fst 𝕜 E F) s p :=
  hasFDerivAtFilter_fst


@[fun_prop]
protected nonrec theorem HasFDerivWithinAt.fst (h : HasFDerivWithinAt f₂ f₂' s x) :
    HasFDerivWithinAt (fun x => (f₂ x).1) ((fst 𝕜 F G).comp f₂') s x :=
  h.fst


@[fun_prop]
theorem differentiableAt_fst : DifferentiableAt 𝕜 Prod.fst p :=
  hasFDerivAt_fst.differentiableAt


@[simp, fun_prop]
protected theorem DifferentiableAt.fst (h : DifferentiableAt 𝕜 f₂ x) :
    DifferentiableAt 𝕜 (fun x => (f₂ x).1) x :=
  differentiableAt_fst.comp x h


@[fun_prop]
theorem differentiable_fst : Differentiable 𝕜 (Prod.fst : E × F → E) := fun _ =>
  differentiableAt_fst


@[simp, fun_prop]
protected theorem Differentiable.fst (h : Differentiable 𝕜 f₂) :
    Differentiable 𝕜 fun x => (f₂ x).1 :=
  differentiable_fst.comp h


@[fun_prop]
theorem differentiableWithinAt_fst {s : Set (E × F)} : DifferentiableWithinAt 𝕜 Prod.fst s p :=
  differentiableAt_fst.differentiableWithinAt


@[fun_prop]
protected theorem DifferentiableWithinAt.fst (h : DifferentiableWithinAt 𝕜 f₂ s x) :
    DifferentiableWithinAt 𝕜 (fun x => (f₂ x).1) s x :=
  differentiableAt_fst.comp_differentiableWithinAt x h


@[fun_prop]
theorem differentiableOn_fst {s : Set (E × F)} : DifferentiableOn 𝕜 Prod.fst s :=
  differentiable_fst.differentiableOn


@[fun_prop]
protected theorem DifferentiableOn.fst (h : DifferentiableOn 𝕜 f₂ s) :
    DifferentiableOn 𝕜 (fun x => (f₂ x).1) s :=
  differentiable_fst.comp_differentiableOn h


theorem fderiv_fst : fderiv 𝕜 Prod.fst p = fst 𝕜 E F :=
  hasFDerivAt_fst.fderiv


theorem fderiv.fst (h : DifferentiableAt 𝕜 f₂ x) :
    fderiv 𝕜 (fun x => (f₂ x).1) x = (fst 𝕜 F G).comp (fderiv 𝕜 f₂ x) :=
  h.hasFDerivAt.fst.fderiv


theorem fderivWithin_fst {s : Set (E × F)} (hs : UniqueDiffWithinAt 𝕜 s p) :
    fderivWithin 𝕜 Prod.fst s p = fst 𝕜 E F :=
  hasFDerivWithinAt_fst.fderivWithin hs


theorem fderivWithin.fst (hs : UniqueDiffWithinAt 𝕜 s x) (h : DifferentiableWithinAt 𝕜 f₂ s x) :
    fderivWithin 𝕜 (fun x => (f₂ x).1) s x = (fst 𝕜 F G).comp (fderivWithin 𝕜 f₂ s x) :=
  h.hasFDerivWithinAt.fst.fderivWithin hs


@[fun_prop]
theorem hasStrictFDerivAt_snd : HasStrictFDerivAt (@Prod.snd E F) (snd 𝕜 E F) p :=
  (snd 𝕜 E F).hasStrictFDerivAt


@[fun_prop]
protected theorem HasStrictFDerivAt.snd (h : HasStrictFDerivAt f₂ f₂' x) :
    HasStrictFDerivAt (fun x => (f₂ x).2) ((snd 𝕜 F G).comp f₂') x :=
  hasStrictFDerivAt_snd.comp x h


theorem hasFDerivAtFilter_snd {L : Filter (E × F)} :
    HasFDerivAtFilter (@Prod.snd E F) (snd 𝕜 E F) p L :=
  (snd 𝕜 E F).hasFDerivAtFilter


protected theorem HasFDerivAtFilter.snd (h : HasFDerivAtFilter f₂ f₂' x L) :
    HasFDerivAtFilter (fun x => (f₂ x).2) ((snd 𝕜 F G).comp f₂') x L :=
  hasFDerivAtFilter_snd.comp x h tendsto_map


@[fun_prop]
theorem hasFDerivAt_snd : HasFDerivAt (@Prod.snd E F) (snd 𝕜 E F) p :=
  hasFDerivAtFilter_snd


@[fun_prop]
protected nonrec theorem HasFDerivAt.snd (h : HasFDerivAt f₂ f₂' x) :
    HasFDerivAt (fun x => (f₂ x).2) ((snd 𝕜 F G).comp f₂') x :=
  h.snd


@[fun_prop]
theorem hasFDerivWithinAt_snd {s : Set (E × F)} :
    HasFDerivWithinAt (@Prod.snd E F) (snd 𝕜 E F) s p :=
  hasFDerivAtFilter_snd


@[fun_prop]
protected nonrec theorem HasFDerivWithinAt.snd (h : HasFDerivWithinAt f₂ f₂' s x) :
    HasFDerivWithinAt (fun x => (f₂ x).2) ((snd 𝕜 F G).comp f₂') s x :=
  h.snd


@[fun_prop]
theorem differentiableAt_snd : DifferentiableAt 𝕜 Prod.snd p :=
  hasFDerivAt_snd.differentiableAt


@[simp, fun_prop]
protected theorem DifferentiableAt.snd (h : DifferentiableAt 𝕜 f₂ x) :
    DifferentiableAt 𝕜 (fun x => (f₂ x).2) x :=
  differentiableAt_snd.comp x h


@[fun_prop]
theorem differentiable_snd : Differentiable 𝕜 (Prod.snd : E × F → F) := fun _ =>
  differentiableAt_snd


@[simp, fun_prop]
protected theorem Differentiable.snd (h : Differentiable 𝕜 f₂) :
    Differentiable 𝕜 fun x => (f₂ x).2 :=
  differentiable_snd.comp h


@[fun_prop]
theorem differentiableWithinAt_snd {s : Set (E × F)} : DifferentiableWithinAt 𝕜 Prod.snd s p :=
  differentiableAt_snd.differentiableWithinAt


@[fun_prop]
protected theorem DifferentiableWithinAt.snd (h : DifferentiableWithinAt 𝕜 f₂ s x) :
    DifferentiableWithinAt 𝕜 (fun x => (f₂ x).2) s x :=
  differentiableAt_snd.comp_differentiableWithinAt x h


@[fun_prop]
theorem differentiableOn_snd {s : Set (E × F)} : DifferentiableOn 𝕜 Prod.snd s :=
  differentiable_snd.differentiableOn


@[fun_prop]
protected theorem DifferentiableOn.snd (h : DifferentiableOn 𝕜 f₂ s) :
    DifferentiableOn 𝕜 (fun x => (f₂ x).2) s :=
  differentiable_snd.comp_differentiableOn h


theorem fderiv_snd : fderiv 𝕜 Prod.snd p = snd 𝕜 E F :=
  hasFDerivAt_snd.fderiv


theorem fderiv.snd (h : DifferentiableAt 𝕜 f₂ x) :
    fderiv 𝕜 (fun x => (f₂ x).2) x = (snd 𝕜 F G).comp (fderiv 𝕜 f₂ x) :=
  h.hasFDerivAt.snd.fderiv


theorem fderivWithin_snd {s : Set (E × F)} (hs : UniqueDiffWithinAt 𝕜 s p) :
    fderivWithin 𝕜 Prod.snd s p = snd 𝕜 E F :=
  hasFDerivWithinAt_snd.fderivWithin hs


theorem fderivWithin.snd (hs : UniqueDiffWithinAt 𝕜 s x) (h : DifferentiableWithinAt 𝕜 f₂ s x) :
    fderivWithin 𝕜 (fun x => (f₂ x).2) s x = (snd 𝕜 F G).comp (fderivWithin 𝕜 f₂ s x) :=
  h.hasFDerivWithinAt.snd.fderivWithin hs


@[fun_prop]
protected theorem HasStrictFDerivAt.prodMap (hf : HasStrictFDerivAt f f' p.1)
    (hf₂ : HasStrictFDerivAt f₂ f₂' p.2) : HasStrictFDerivAt (Prod.map f f₂) (f'.prodMap f₂') p :=
  (hf.comp p hasStrictFDerivAt_fst).prod (hf₂.comp p hasStrictFDerivAt_snd)


@[fun_prop]
protected theorem HasFDerivAt.prodMap (hf : HasFDerivAt f f' p.1) (hf₂ : HasFDerivAt f₂ f₂' p.2) :
    HasFDerivAt (Prod.map f f₂) (f'.prodMap f₂') p :=
  (hf.comp p hasFDerivAt_fst).prod (hf₂.comp p hasFDerivAt_snd)


@[simp, fun_prop]
protected theorem DifferentiableAt.prod_map (hf : DifferentiableAt 𝕜 f p.1)
    (hf₂ : DifferentiableAt 𝕜 f₂ p.2) : DifferentiableAt 𝕜 (fun p : E × G => (f p.1, f₂ p.2)) p :=
  (hf.comp p differentiableAt_fst).prod (hf₂.comp p differentiableAt_snd)


@[simp]
theorem hasStrictFDerivAt_pi' :
    HasStrictFDerivAt Φ Φ' x ↔ ∀ i, HasStrictFDerivAt (fun x => Φ x i) ((proj i).comp Φ') x := by
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    x : E
    ι : Type u_6
    inst✝² : Fintype ι
    F' : ι → Type u_7
    inst✝¹ : (i : ι) → NormedAddCommGroup (F' i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (F' i)
    Φ : E → (i : ι) → F' i
    Φ' : ContinuousLinearMap (RingHom.id 𝕜) E ((i : ι) → F' i)
    ⊢ Iff (HasStrictFDerivAt Φ Φ' x) (∀ (i : ι), HasStrictFDerivAt (fun x => Φ x i …
  -/
  simp only [hasStrictFDerivAt_iff_isLittleO, ContinuousLinearMap.coe_pi]
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    x : E
    ι : Type u_6
    inst✝² : Fintype ι
    F' : ι → Type u_7
    inst✝¹ : (i : ι) → NormedAddCommGroup (F' i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (F' i)
    Φ : E → (i : ι) → F' i
    Φ' : ContinuousLinearMap (RingHom.id 𝕜) E ((i : ι) → F' i)
    ⊢ Iff (Asymptotics.IsLittleO (nhds { fst := x, snd := x }) (fun p => HSub.hSub …
  -/
  exact isLittleO_pi
  /-
    🎉 no goals
  -/


@[fun_prop]
theorem hasStrictFDerivAt_pi'' (hφ : ∀ i, HasStrictFDerivAt (fun x => Φ x i) ((proj i).comp Φ') x) :
    HasStrictFDerivAt Φ Φ' x := hasStrictFDerivAt_pi'.2 hφ


@[fun_prop]
theorem hasStrictFDerivAt_apply (i : ι) (f : ∀ i, F' i) :
    HasStrictFDerivAt (𝕜 := 𝕜) (fun f : ∀ i, F' i => f i) (proj i) f := by
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    ι : Type u_6
    inst✝² : Fintype ι
    F' : ι → Type u_7
    inst✝¹ : (i : ι) → NormedAddCommGroup (F' i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (F' i)
    i : ι
    f : (i : ι) → F' i
    ⊢ HasStrictFDerivAt (fun f => f i) (ContinuousLinearMap.proj i) f
  -/
  let id' := ContinuousLinearMap.id 𝕜 (∀ i, F' i)
  have h := ((hasStrictFDerivAt_pi'
             (Φ := fun (f : ∀ i, F' i) (i' : ι) => f i') (Φ' := id') (x := f))).1
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    ι : Type u_6
    inst✝² : Fintype ι
    F' : ι → Type u_7
    inst✝¹ : (i : ι) → NormedAddCommGroup (F' i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (F' i)
    i : ι
    f : (i : ι) → F' i
    id' : ContinuousLinearMap (RingHom.id 𝕜) ((i : ι) → F' i) ((i : ι) → F' i) :=  …
    h : HasStrictFDerivAt (fun f i' => f i') id' f → ∀ (i : ι), HasStrictFDerivAt  …
    ⊢ HasStrictFDerivAt (fun f => f i) (ContinuousLinearMap.proj i) f
  -/
  have h' : comp (proj i) id' = proj i := by rfl
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    ι : Type u_6
    inst✝² : Fintype ι
    F' : ι → Type u_7
    inst✝¹ : (i : ι) → NormedAddCommGroup (F' i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (F' i)
    i : ι
    f : (i : ι) → F' i
    id' : ContinuousLinearMap (RingHom.id 𝕜) ((i : ι) → F' i) ((i : ι) → F' i) :=  …
    h : HasStrictFDerivAt (fun f i' => f i') id' f → ∀ (i : ι), HasStrictFDerivAt  …
    h' : Eq ((ContinuousLinearMap.proj i).comp id') (ContinuousLinearMap.proj i)
    ⊢ HasStrictFDerivAt (fun f => f i) (ContinuousLinearMap.proj i) f
  -/
  rw [← h']; apply h; apply hasStrictFDerivAt_id
                      /-
                        🎉 no goals
                      -/


@[simp 1100] -- Porting note: increased priority to make lint happy
theorem hasStrictFDerivAt_pi :
    HasStrictFDerivAt (fun x i => φ i x) (ContinuousLinearMap.pi φ') x ↔
      ∀ i, HasStrictFDerivAt (φ i) (φ' i) x :=
  hasStrictFDerivAt_pi'


@[simp]
theorem hasFDerivAtFilter_pi' :
    HasFDerivAtFilter Φ Φ' x L ↔
      ∀ i, HasFDerivAtFilter (fun x => Φ x i) ((proj i).comp Φ') x L := by
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    x : E
    L : Filter E
    ι : Type u_6
    inst✝² : Fintype ι
    F' : ι → Type u_7
    inst✝¹ : (i : ι) → NormedAddCommGroup (F' i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (F' i)
    Φ : E → (i : ι) → F' i
    Φ' : ContinuousLinearMap (RingHom.id 𝕜) E ((i : ι) → F' i)
    ⊢ Iff (HasFDerivAtFilter Φ Φ' x L) (∀ (i : ι), HasFDerivAtFilter (fun x => Φ x …
  -/
  simp only [hasFDerivAtFilter_iff_isLittleO, ContinuousLinearMap.coe_pi]
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    x : E
    L : Filter E
    ι : Type u_6
    inst✝² : Fintype ι
    F' : ι → Type u_7
    inst✝¹ : (i : ι) → NormedAddCommGroup (F' i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (F' i)
    Φ : E → (i : ι) → F' i
    Φ' : ContinuousLinearMap (RingHom.id 𝕜) E ((i : ι) → F' i)
    ⊢ Iff (Asymptotics.IsLittleO L (fun x' => HSub.hSub (HSub.hSub (Φ x') (Φ x)) ( …
  -/
  exact isLittleO_pi
  /-
    🎉 no goals
  -/


theorem hasFDerivAtFilter_pi :
    HasFDerivAtFilter (fun x i => φ i x) (ContinuousLinearMap.pi φ') x L ↔
      ∀ i, HasFDerivAtFilter (φ i) (φ' i) x L :=
  hasFDerivAtFilter_pi'


@[simp]
theorem hasFDerivAt_pi' :
    HasFDerivAt Φ Φ' x ↔ ∀ i, HasFDerivAt (fun x => Φ x i) ((proj i).comp Φ') x :=
  hasFDerivAtFilter_pi'


@[fun_prop]
theorem hasFDerivAt_pi'' (hφ : ∀ i, HasFDerivAt (fun x => Φ x i) ((proj i).comp Φ') x) :
    HasFDerivAt Φ Φ' x := hasFDerivAt_pi'.2 hφ


@[fun_prop]
theorem hasFDerivAt_apply (i : ι) (f : ∀ i, F' i) :
    HasFDerivAt (𝕜 := 𝕜) (fun f : ∀ i, F' i => f i) (proj i) f := by
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    ι : Type u_6
    inst✝² : Fintype ι
    F' : ι → Type u_7
    inst✝¹ : (i : ι) → NormedAddCommGroup (F' i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (F' i)
    i : ι
    f : (i : ι) → F' i
    ⊢ HasFDerivAt (fun f => f i) (ContinuousLinearMap.proj i) f
  -/
  apply HasStrictFDerivAt.hasFDerivAt
  /-
    case hf
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    ι : Type u_6
    inst✝² : Fintype ι
    F' : ι → Type u_7
    inst✝¹ : (i : ι) → NormedAddCommGroup (F' i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (F' i)
    i : ι
    f : (i : ι) → F' i
    ⊢ HasStrictFDerivAt (fun f => f i) (ContinuousLinearMap.proj i) f
  -/
  apply hasStrictFDerivAt_apply
  /-
    🎉 no goals
  -/


theorem hasFDerivAt_pi :
    HasFDerivAt (fun x i => φ i x) (ContinuousLinearMap.pi φ') x ↔
      ∀ i, HasFDerivAt (φ i) (φ' i) x :=
  hasFDerivAtFilter_pi


@[simp]
theorem hasFDerivWithinAt_pi' :
    HasFDerivWithinAt Φ Φ' s x ↔ ∀ i, HasFDerivWithinAt (fun x => Φ x i) ((proj i).comp Φ') s x :=
  hasFDerivAtFilter_pi'


@[fun_prop]
theorem hasFDerivWithinAt_pi''
    (hφ : ∀ i, HasFDerivWithinAt (fun x => Φ x i) ((proj i).comp Φ') s x) :
    HasFDerivWithinAt Φ Φ' s x := hasFDerivWithinAt_pi'.2 hφ


@[fun_prop]
theorem hasFDerivWithinAt_apply (i : ι) (f : ∀ i, F' i) (s' : Set (∀ i, F' i)) :
    HasFDerivWithinAt (𝕜 := 𝕜) (fun f : ∀ i, F' i => f i) (proj i) s' f := by
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    ι : Type u_6
    inst✝² : Fintype ι
    F' : ι → Type u_7
    inst✝¹ : (i : ι) → NormedAddCommGroup (F' i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (F' i)
    i : ι
    f : (i : ι) → F' i
    s' : Set ((i : ι) → F' i)
    ⊢ HasFDerivWithinAt (fun f => f i) (ContinuousLinearMap.proj i) s' f
  -/
  let id' := ContinuousLinearMap.id 𝕜 (∀ i, F' i)
  have h := ((hasFDerivWithinAt_pi'
             (Φ := fun (f : ∀ i, F' i) (i' : ι) => f i') (Φ' := id') (x := f) (s := s'))).1
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    ι : Type u_6
    inst✝² : Fintype ι
    F' : ι → Type u_7
    inst✝¹ : (i : ι) → NormedAddCommGroup (F' i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (F' i)
    i : ι
    f : (i : ι) → F' i
    s' : Set ((i : ι) → F' i)
    id' : ContinuousLinearMap (RingHom.id 𝕜) ((i : ι) → F' i) ((i : ι) → F' i) :=  …
    h : HasFDerivWithinAt (fun f i' => f i') id' s' f → ∀ (i : ι), HasFDerivWithin …
    ⊢ HasFDerivWithinAt (fun f => f i) (ContinuousLinearMap.proj i) s' f
  -/
  have h' : comp (proj i) id' = proj i := by rfl
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    ι : Type u_6
    inst✝² : Fintype ι
    F' : ι → Type u_7
    inst✝¹ : (i : ι) → NormedAddCommGroup (F' i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (F' i)
    i : ι
    f : (i : ι) → F' i
    s' : Set ((i : ι) → F' i)
    id' : ContinuousLinearMap (RingHom.id 𝕜) ((i : ι) → F' i) ((i : ι) → F' i) :=  …
    h : HasFDerivWithinAt (fun f i' => f i') id' s' f → ∀ (i : ι), HasFDerivWithin …
    h' : Eq ((ContinuousLinearMap.proj i).comp id') (ContinuousLinearMap.proj i)
    ⊢ HasFDerivWithinAt (fun f => f i) (ContinuousLinearMap.proj i) s' f
  -/
  rw [← h']; apply h; apply hasFDerivWithinAt_id
                      /-
                        🎉 no goals
                      -/


theorem hasFDerivWithinAt_pi :
    HasFDerivWithinAt (fun x i => φ i x) (ContinuousLinearMap.pi φ') s x ↔
      ∀ i, HasFDerivWithinAt (φ i) (φ' i) s x :=
  hasFDerivAtFilter_pi


@[simp]
theorem differentiableWithinAt_pi :
    DifferentiableWithinAt 𝕜 Φ s x ↔ ∀ i, DifferentiableWithinAt 𝕜 (fun x => Φ x i) s x :=
  ⟨fun h i => (hasFDerivWithinAt_pi'.1 h.hasFDerivWithinAt i).differentiableWithinAt, fun h =>
    (hasFDerivWithinAt_pi.2 fun i => (h i).hasFDerivWithinAt).differentiableWithinAt⟩


@[fun_prop]
theorem differentiableWithinAt_pi'' (hφ : ∀ i, DifferentiableWithinAt 𝕜 (fun x => Φ x i) s x) :
    DifferentiableWithinAt 𝕜 Φ s x := differentiableWithinAt_pi.2 hφ


@[fun_prop]
theorem differentiableWithinAt_apply (i : ι) (f : ∀ i, F' i) (s' : Set (∀ i, F' i)) :
    DifferentiableWithinAt (𝕜 := 𝕜) (fun f : ∀ i, F' i => f i) s' f := by
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    ι : Type u_6
    inst✝² : Fintype ι
    F' : ι → Type u_7
    inst✝¹ : (i : ι) → NormedAddCommGroup (F' i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (F' i)
    i : ι
    f : (i : ι) → F' i
    s' : Set ((i : ι) → F' i)
    ⊢ DifferentiableWithinAt 𝕜 (fun f => f i) s' f
  -/
  apply HasFDerivWithinAt.differentiableWithinAt
  /-
    case h
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    ι : Type u_6
    inst✝² : Fintype ι
    F' : ι → Type u_7
    inst✝¹ : (i : ι) → NormedAddCommGroup (F' i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (F' i)
    i : ι
    f : (i : ι) → F' i
    s' : Set ((i : ι) → F' i)
    ⊢ HasFDerivWithinAt (fun f => f i) ?f' s' f
  -/
  fun_prop
  /-
    🎉 no goals
  -/


@[simp]
theorem differentiableAt_pi : DifferentiableAt 𝕜 Φ x ↔ ∀ i, DifferentiableAt 𝕜 (fun x => Φ x i) x :=
  ⟨fun h i => (hasFDerivAt_pi'.1 h.hasFDerivAt i).differentiableAt, fun h =>
    (hasFDerivAt_pi.2 fun i => (h i).hasFDerivAt).differentiableAt⟩


@[fun_prop]
theorem differentiableAt_pi'' (hφ : ∀ i, DifferentiableAt 𝕜 (fun x => Φ x i) x) :
    DifferentiableAt 𝕜 Φ x := differentiableAt_pi.2 hφ


@[fun_prop]
theorem differentiableAt_apply (i : ι) (f : ∀ i, F' i) :
    DifferentiableAt (𝕜 := 𝕜) (fun f : ∀ i, F' i => f i) f := by
  have h := ((differentiableAt_pi (𝕜 := 𝕜)
             (Φ := fun (f : ∀ i, F' i) (i' : ι) => f i') (x := f))).1
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    ι : Type u_6
    inst✝² : Fintype ι
    F' : ι → Type u_7
    inst✝¹ : (i : ι) → NormedAddCommGroup (F' i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (F' i)
    i : ι
    f : (i : ι) → F' i
    h : DifferentiableAt 𝕜 (fun f i' => f i') f → ∀ (i : ι), DifferentiableAt 𝕜 (f …
    ⊢ DifferentiableAt 𝕜 (fun f => f i) f
  -/
  apply h; apply differentiableAt_id
           /-
             🎉 no goals
           -/


theorem differentiableOn_pi : DifferentiableOn 𝕜 Φ s ↔ ∀ i, DifferentiableOn 𝕜 (fun x => Φ x i) s :=
  ⟨fun h i x hx => differentiableWithinAt_pi.1 (h x hx) i, fun h x hx =>
    differentiableWithinAt_pi.2 fun i => h i x hx⟩


@[fun_prop]
theorem differentiableOn_pi'' (hφ : ∀ i, DifferentiableOn 𝕜 (fun x => Φ x i) s) :
    DifferentiableOn 𝕜 Φ s := differentiableOn_pi.2 hφ


@[fun_prop]
theorem differentiableOn_apply (i : ι) (s' : Set (∀ i, F' i)) :
    DifferentiableOn (𝕜 := 𝕜) (fun f : ∀ i, F' i => f i) s' := by
  have h := ((differentiableOn_pi (𝕜 := 𝕜)
             (Φ := fun (f : ∀ i, F' i) (i' : ι) => f i') (s := s'))).1
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    ι : Type u_6
    inst✝² : Fintype ι
    F' : ι → Type u_7
    inst✝¹ : (i : ι) → NormedAddCommGroup (F' i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (F' i)
    i : ι
    s' : Set ((i : ι) → F' i)
    h : DifferentiableOn 𝕜 (fun f i' => f i') s' → ∀ (i : ι), DifferentiableOn 𝕜 ( …
    ⊢ DifferentiableOn 𝕜 (fun f => f i) s'
  -/
  apply h; apply differentiableOn_id
           /-
             🎉 no goals
           -/


theorem differentiable_pi : Differentiable 𝕜 Φ ↔ ∀ i, Differentiable 𝕜 fun x => Φ x i :=
  ⟨fun h i x => differentiableAt_pi.1 (h x) i, fun h x => differentiableAt_pi.2 fun i => h i x⟩


@[fun_prop]
theorem differentiable_pi'' (hφ : ∀ i, Differentiable 𝕜 fun x => Φ x i) :
    Differentiable 𝕜 Φ := differentiable_pi.2 hφ


@[fun_prop]
theorem differentiable_apply (i : ι) :
                                                             /-
                                                               𝕜 : Type u_1
                                                               inst✝³ : NontriviallyNormedField 𝕜
                                                               ι : Type u_6
                                                               inst✝² : Fintype ι
                                                               F' : ι → Type u_7
                                                               inst✝¹ : (i : ι) → NormedAddCommGroup (F' i)
                                                               inst✝ : (i : ι) → NormedSpace 𝕜 (F' i)
                                                               i : ι
                                                               ⊢ Differentiable 𝕜 fun f => f i
                                                             -/
    Differentiable (𝕜 := 𝕜) (fun f : ∀ i, F' i => f i) := by intro x; apply differentiableAt_apply
                                                                      /-
                                                                        🎉 no goals
                                                                      -/

-- TODO: find out which version (`φ` or `Φ`) works better with `rw`/`simp`

theorem fderivWithin_pi (h : ∀ i, DifferentiableWithinAt 𝕜 (φ i) s x)
    (hs : UniqueDiffWithinAt 𝕜 s x) :
    fderivWithin 𝕜 (fun x i => φ i x) s x = pi fun i => fderivWithin 𝕜 (φ i) s x :=
  (hasFDerivWithinAt_pi.2 fun i => (h i).hasFDerivWithinAt).fderivWithin hs


theorem fderiv_pi (h : ∀ i, DifferentiableAt 𝕜 (φ i) x) :
    fderiv 𝕜 (fun x i => φ i x) x = pi fun i => fderiv 𝕜 (φ i) x :=
  (hasFDerivAt_pi.2 fun i => (h i).hasFDerivAt).fderiv


