/-- A map `f : α → β` is called a *measurable embedding* if it is injective, measurable, and sends
measurable sets to measurable sets. The latter assumption can be replaced with “`f` has measurable
inverse `g : Set.range f → α`”, see `MeasurableEmbedding.measurable_rangeSplitting`,
`MeasurableEmbedding.of_measurable_inverse_range`, and
`MeasurableEmbedding.of_measurable_inverse`.

One more interpretation: `f` is a measurable embedding if it defines a measurable equivalence to its
range and the range is a measurable set. One implication is formalized as
`MeasurableEmbedding.equivRange`; the other one follows from
`MeasurableEquiv.measurableEmbedding`, `MeasurableEmbedding.subtype_coe`, and
`MeasurableEmbedding.comp`. -/
structure MeasurableEmbedding [MeasurableSpace α] [MeasurableSpace β] (f : α → β) : Prop where
  /-- A measurable embedding is injective. -/
  protected injective : Injective f
  /-- A measurable embedding is a measurable function. -/
  protected measurable : Measurable f
  /-- The image of a measurable set under a measurable embedding is a measurable set. -/
  protected measurableSet_image' : ∀ ⦃s⦄, MeasurableSet s → MeasurableSet (f '' s)


theorem measurableSet_image (hf : MeasurableEmbedding f) :
    MeasurableSet (f '' s) ↔ MeasurableSet s :=
               /-
                 α : Type u_1
                 β : Type u_2
                 s : Set α
                 mα : MeasurableSpace α
                 inst✝ : MeasurableSpace β
                 f : α → β
                 hf : MeasurableEmbedding f
                 h : MeasurableSet (Set.image f s)
                 ⊢ MeasurableSet s
               -/
  ⟨fun h => by simpa only [hf.injective.preimage_image] using hf.measurable h, fun h =>
               /-
                 🎉 no goals
               -/
    hf.measurableSet_image' h⟩


theorem id : MeasurableEmbedding (id : α → α) :=
                                               /-
                                                 α : Type u_1
                                                 mα : MeasurableSpace α
                                                 s : Set α
                                                 hs : MeasurableSet s
                                                 ⊢ MeasurableSet (Set.image _root_.id s)
                                               -/
  ⟨injective_id, measurable_id, fun s hs => by rwa [image_id]⟩
                                               /-
                                                 🎉 no goals
                                               -/


theorem comp (hg : MeasurableEmbedding g) (hf : MeasurableEmbedding f) :
    MeasurableEmbedding (g ∘ f) :=
  ⟨hg.injective.comp hf.injective, hg.measurable.comp hf.measurable, fun s hs => by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      inst✝ : MeasurableSpace γ
      f : α → β
      g : β → γ
      hg : MeasurableEmbedding g
      hf : MeasurableEmbedding f
      s : Set α
      hs : MeasurableSet s
      ⊢ MeasurableSet (Set.image (Function.comp g f) s)
    -/
    rwa [image_comp, hg.measurableSet_image, hf.measurableSet_image]⟩
    /-
      🎉 no goals
    -/


theorem subtype_coe (hs : MeasurableSet s) : MeasurableEmbedding ((↑) : s → α) where
  injective := Subtype.coe_injective
  measurable := measurable_subtype_coe
  measurableSet_image' := fun _ => MeasurableSet.subtype_image hs


theorem measurableSet_range (hf : MeasurableEmbedding f) : MeasurableSet (range f) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    inst✝ : MeasurableSpace β
    f : α → β
    hf : MeasurableEmbedding f
    ⊢ MeasurableSet (Set.range f)
  -/
  rw [← image_univ]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    inst✝ : MeasurableSpace β
    f : α → β
    hf : MeasurableEmbedding f
    ⊢ MeasurableSet (Set.image f Set.univ)
  -/
  exact hf.measurableSet_image' MeasurableSet.univ
  /-
    🎉 no goals
  -/


theorem measurableSet_preimage (hf : MeasurableEmbedding f) {s : Set β} :
    MeasurableSet (f ⁻¹' s) ↔ MeasurableSet (s ∩ range f) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    inst✝ : MeasurableSpace β
    f : α → β
    hf : MeasurableEmbedding f
    s : Set β
    ⊢ Iff (MeasurableSet (Set.preimage f s)) (MeasurableSet (Inter.inter s (Set.ra …
  -/
  rw [← image_preimage_eq_inter_range, hf.measurableSet_image]
  /-
    🎉 no goals
  -/


theorem measurable_rangeSplitting (hf : MeasurableEmbedding f) :
    Measurable (rangeSplitting f) := fun s hs => by
  rwa [preimage_rangeSplitting hf.injective,
    ← (subtype_coe hf.measurableSet_range).measurableSet_image, ← image_comp,
    coe_comp_rangeFactorization, hf.measurableSet_image]


theorem measurable_extend (hf : MeasurableEmbedding f) {g : α → γ} {g' : β → γ} (hg : Measurable g)
    (hg' : Measurable g') : Measurable (extend f g g') := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    inst✝ : MeasurableSpace γ
    f : α → β
    hf : MeasurableEmbedding f
    g : α → γ
    g' : β → γ
    hg : Measurable g
    hg' : Measurable g'
    ⊢ Measurable (Function.extend f g g')
  -/
  refine measurable_of_restrict_of_restrict_compl hf.measurableSet_range ?_ ?_
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      inst✝ : MeasurableSpace γ
      f : α → β
      hf : MeasurableEmbedding f
      g : α → γ
      g' : β → γ
      hg : Measurable g
      hg' : Measurable g'
      ⊢ Measurable ((Set.range f).restrict (Function.extend f g g'))
    -/
  · rw [restrict_extend_range]
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      inst✝ : MeasurableSpace γ
      f : α → β
      hf : MeasurableEmbedding f
      g : α → γ
      g' : β → γ
      hg : Measurable g
      hg' : Measurable g'
      ⊢ Measurable fun x => g (Exists.choose ⋯)
    -/
    simpa only [rangeSplitting] using hg.comp hf.measurable_rangeSplitting
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      inst✝ : MeasurableSpace γ
      f : α → β
      hf : MeasurableEmbedding f
      g : α → γ
      g' : β → γ
      hg : Measurable g
      hg' : Measurable g'
      ⊢ Measurable ((HasCompl.compl (Set.range f)).restrict (Function.extend f g g'))
    -/
  · rw [restrict_extend_compl_range]
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      inst✝ : MeasurableSpace γ
      f : α → β
      hf : MeasurableEmbedding f
      g : α → γ
      g' : β → γ
      hg : Measurable g
      hg' : Measurable g'
      ⊢ Measurable (Function.comp g' Subtype.val)
    -/
    exact hg'.comp measurable_subtype_coe
    /-
      🎉 no goals
    -/


theorem exists_measurable_extend (hf : MeasurableEmbedding f) {g : α → γ} (hg : Measurable g)
    (hne : β → Nonempty γ) : ∃ g' : β → γ, Measurable g' ∧ g' ∘ f = g :=
  ⟨extend f g fun x => Classical.choice (hne x),
    hf.measurable_extend hg (measurable_const' fun _ _ => rfl),
    funext fun _ => hf.injective.extend_apply _ _ _⟩


theorem measurable_comp_iff (hg : MeasurableEmbedding g) : Measurable (g ∘ f) ↔ Measurable f := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    inst✝ : MeasurableSpace γ
    f : α → β
    g : β → γ
    hg : MeasurableEmbedding g
    ⊢ Iff (Measurable (Function.comp g f)) (Measurable f)
  -/
  refine ⟨fun H => ?_, hg.measurable.comp⟩
  suffices Measurable ((rangeSplitting g ∘ rangeFactorization g) ∘ f) by
    rwa [(rightInverse_rangeSplitting hg.injective).comp_eq_id] at this
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    inst✝ : MeasurableSpace γ
    f : α → β
    g : β → γ
    hg : MeasurableEmbedding g
    H : Measurable (Function.comp g f)
    ⊢ Measurable (Function.comp (Function.comp (Set.rangeSplitting g) (Set.rangeFa …
  -/
  exact hg.measurable_rangeSplitting.comp H.subtype_mk
  /-
    🎉 no goals
  -/


lemma MeasurableSet.of_union_range_cover (hi₁ : MeasurableEmbedding i₁)
    (hi₂ : MeasurableEmbedding i₂) (h : univ ⊆ range i₁ ∪ range i₂)
    (hs₁ : MeasurableSet (i₁ ⁻¹' s)) (hs₂ : MeasurableSet (i₂ ⁻¹' s)) : MeasurableSet s := by
  /-
    α : Type u_1
    α₁ : Type u_6
    α₂ : Type u_7
    mα : MeasurableSpace α
    mα₁ : MeasurableSpace α₁
    mα₂ : MeasurableSpace α₂
    i₁ : α₁ → α
    i₂ : α₂ → α
    s : Set α
    hi₁ : MeasurableEmbedding i₁
    hi₂ : MeasurableEmbedding i₂
    h : HasSubset.Subset Set.univ (Union.union (Set.range i₁) (Set.range i₂))
    hs₁ : MeasurableSet (Set.preimage i₁ s)
    hs₂ : MeasurableSet (Set.preimage i₂ s)
    ⊢ MeasurableSet s
  -/
  convert (hi₁.measurableSet_image' hs₁).union (hi₂.measurableSet_image' hs₂)
  /-
    case h.e'_3
    α : Type u_1
    α₁ : Type u_6
    α₂ : Type u_7
    mα : MeasurableSpace α
    mα₁ : MeasurableSpace α₁
    mα₂ : MeasurableSpace α₂
    i₁ : α₁ → α
    i₂ : α₂ → α
    s : Set α
    hi₁ : MeasurableEmbedding i₁
    hi₂ : MeasurableEmbedding i₂
    h : HasSubset.Subset Set.univ (Union.union (Set.range i₁) (Set.range i₂))
    hs₁ : MeasurableSet (Set.preimage i₁ s)
    hs₂ : MeasurableSet (Set.preimage i₂ s)
    ⊢ Eq s (Union.union (Set.image i₁ (Set.preimage i₁ s)) (Set.image i₂ (Set.prei …
  -/
  simp [image_preimage_eq_range_inter, ← union_inter_distrib_right,univ_subset_iff.1 h]
  /-
    🎉 no goals
  -/


lemma MeasurableSet.of_union₃_range_cover (hi₁ : MeasurableEmbedding i₁)
    (hi₂ : MeasurableEmbedding i₂) (hi₃ : MeasurableEmbedding i₃)
    (h : univ ⊆ range i₁ ∪ range i₂ ∪ range i₃) (hs₁ : MeasurableSet (i₁ ⁻¹' s))
    (hs₂ : MeasurableSet (i₂ ⁻¹' s)) (hs₃ : MeasurableSet (i₃ ⁻¹' s)) : MeasurableSet s := by
  convert (hi₁.measurableSet_image' hs₁).union (hi₂.measurableSet_image' hs₂) |>.union
    (hi₃.measurableSet_image' hs₃)
  /-
    case h.e'_3
    α : Type u_1
    α₁ : Type u_6
    α₂ : Type u_7
    α₃ : Type u_8
    mα : MeasurableSpace α
    mα₁ : MeasurableSpace α₁
    mα₂ : MeasurableSpace α₂
    mα₃ : MeasurableSpace α₃
    i₁ : α₁ → α
    i₂ : α₂ → α
    i₃ : α₃ → α
    s : Set α
    hi₁ : MeasurableEmbedding i₁
    hi₂ : MeasurableEmbedding i₂
    hi₃ : MeasurableEmbedding i₃
    h : HasSubset.Subset Set.univ (Union.union (Union.union (Set.range i₁) (Set.ra …
    hs₁ : MeasurableSet (Set.preimage i₁ s)
    hs₂ : MeasurableSet (Set.preimage i₂ s)
    hs₃ : MeasurableSet (Set.preimage i₃ s)
    ⊢ Eq s (Union.union (Union.union (Set.image i₁ (Set.preimage i₁ s)) (Set.image …
  -/
  simp [image_preimage_eq_range_inter, ← union_inter_distrib_right, univ_subset_iff.1 h]
  /-
    🎉 no goals
  -/


lemma Measurable.of_union_range_cover (hi₁ : MeasurableEmbedding i₁)
    (hi₂ : MeasurableEmbedding i₂) (h : univ ⊆ range i₁ ∪ range i₂)
    (hf₁ : Measurable (f ∘ i₁)) (hf₂ : Measurable (f ∘ i₂)) : Measurable f :=
  fun _s hs ↦ .of_union_range_cover hi₁ hi₂ h (hf₁ hs) (hf₂ hs)


lemma Measurable.of_union₃_range_cover (hi₁ : MeasurableEmbedding i₁)
    (hi₂ : MeasurableEmbedding i₂) (hi₃ : MeasurableEmbedding i₃)
    (h : univ ⊆ range i₁ ∪ range i₂ ∪ range i₃) (hf₁ : Measurable (f ∘ i₁))
    (hf₂ : Measurable (f ∘ i₂)) (hf₃ : Measurable (f ∘ i₃)) : Measurable f :=
  fun _s hs ↦ .of_union₃_range_cover hi₁ hi₂ hi₃ h (hf₁ hs) (hf₂ hs) (hf₃ hs)


theorem MeasurableSet.exists_measurable_proj {_ : MeasurableSpace α}
    (hs : MeasurableSet s) (hne : s.Nonempty) : ∃ f : α → s, Measurable f ∧ ∀ x : s, f x = x :=
  let ⟨f, hfm, hf⟩ :=
    (MeasurableEmbedding.subtype_coe hs).exists_measurable_extend measurable_id fun _ =>
      hne.to_subtype
  ⟨f, hfm, congr_fun hf⟩


/-- Equivalences between measurable spaces. Main application is the simplification of measurability
statements along measurable equivalences. -/
structure MeasurableEquiv (α β : Type*) [MeasurableSpace α] [MeasurableSpace β] extends α ≃ β where
  /-- The forward function of a measurable equivalence is measurable. -/
  measurable_toFun : Measurable toEquiv
  /-- The inverse function of a measurable equivalence is measurable. -/
  measurable_invFun : Measurable toEquiv.symm


@[inherit_doc]
infixl:25 " ≃ᵐ " => MeasurableEquiv


theorem toEquiv_injective : Injective (toEquiv : α ≃ᵐ β → α ≃ β) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    ⊢ Function.Injective MeasurableEquiv.toEquiv
  -/
  rintro ⟨e₁, _, _⟩ ⟨e₂, _, _⟩ (rfl : e₁ = e₂)
  /-
    case mk.mk
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    e₁ : Equiv α β
    measurable_toFun✝¹ : Measurable ⇑e₁
    measurable_invFun✝¹ : Measurable ⇑e₁.symm
    measurable_toFun✝ : Measurable ⇑e₁
    measurable_invFun✝ : Measurable ⇑e₁.symm
    ⊢ Eq { toEquiv := e₁, measurable_toFun := measurable_toFun✝¹, measurable_invFu …
  -/
  rfl
  /-
    🎉 no goals
  -/


instance instEquivLike : EquivLike (α ≃ᵐ β) α β where
  coe e := e.toEquiv
  inv e := e.toEquiv.symm
  left_inv e := e.toEquiv.left_inv
  right_inv e := e.toEquiv.right_inv
  coe_injective' _ _ he _ := toEquiv_injective <| DFunLike.ext' he


@[simp]
theorem coe_toEquiv (e : α ≃ᵐ β) : (e.toEquiv : α → β) = e :=
  rfl


@[measurability, fun_prop]
protected theorem measurable (e : α ≃ᵐ β) : Measurable (e : α → β) :=
  e.measurable_toFun


@[simp]
theorem coe_mk (e : α ≃ β) (h1 : Measurable e) (h2 : Measurable e.symm) :
    ((⟨e, h1, h2⟩ : α ≃ᵐ β) : α → β) = e :=
  rfl


/-- Any measurable space is equivalent to itself. -/
def refl (α : Type*) [MeasurableSpace α] : α ≃ᵐ α where
  toEquiv := Equiv.refl α
  measurable_toFun := measurable_id
  measurable_invFun := measurable_id


instance instInhabited : Inhabited (α ≃ᵐ α) := ⟨refl α⟩


/-- The composition of equivalences between measurable spaces. -/
def trans (ab : α ≃ᵐ β) (bc : β ≃ᵐ γ) : α ≃ᵐ γ where
  toEquiv := ab.toEquiv.trans bc.toEquiv
  measurable_toFun := bc.measurable_toFun.comp ab.measurable_toFun
  measurable_invFun := ab.measurable_invFun.comp bc.measurable_invFun


theorem coe_trans (ab : α ≃ᵐ β) (bc : β ≃ᵐ γ) : ⇑(ab.trans bc) = bc ∘ ab := rfl


/-- The inverse of an equivalence between measurable spaces. -/
def symm (ab : α ≃ᵐ β) : β ≃ᵐ α where
  toEquiv := ab.toEquiv.symm
  measurable_toFun := ab.measurable_invFun
  measurable_invFun := ab.measurable_toFun


@[simp]
theorem coe_toEquiv_symm (e : α ≃ᵐ β) : (e.toEquiv.symm : β → α) = e.symm :=
  rfl


/-- See Note [custom simps projection]. We need to specify this projection explicitly in this case,
  because it is a composition of multiple projections. -/
def Simps.apply (h : α ≃ᵐ β) : α → β := h


/-- See Note [custom simps projection] -/
def Simps.symm_apply (h : α ≃ᵐ β) : β → α := h.symm


@[ext] theorem ext {e₁ e₂ : α ≃ᵐ β} (h : (e₁ : α → β) = e₂) : e₁ = e₂ := DFunLike.ext' h


@[simp]
theorem symm_mk (e : α ≃ β) (h1 : Measurable e) (h2 : Measurable e.symm) :
    (⟨e, h1, h2⟩ : α ≃ᵐ β).symm = ⟨e.symm, h2, h1⟩ :=
  rfl


attribute [simps! apply toEquiv] trans refl


@[simp]
theorem symm_symm (e : α ≃ᵐ β) : e.symm.symm = e := rfl


theorem symm_bijective :
    Function.Bijective (MeasurableEquiv.symm : (α ≃ᵐ β) → β ≃ᵐ α) :=
  Function.bijective_iff_has_inverse.mpr ⟨_, symm_symm, symm_symm⟩


@[simp]
theorem symm_refl (α : Type*) [MeasurableSpace α] : (refl α).symm = refl α :=
  rfl


@[simp]
theorem symm_comp_self (e : α ≃ᵐ β) : e.symm ∘ e = id :=
  funext e.left_inv


@[simp]
theorem self_comp_symm (e : α ≃ᵐ β) : e ∘ e.symm = id :=
  funext e.right_inv


@[simp]
theorem apply_symm_apply (e : α ≃ᵐ β) (y : β) : e (e.symm y) = y :=
  e.right_inv y


@[simp]
theorem symm_apply_apply (e : α ≃ᵐ β) (x : α) : e.symm (e x) = x :=
  e.left_inv x


@[simp]
theorem symm_trans_self (e : α ≃ᵐ β) : e.symm.trans e = refl β :=
  ext e.self_comp_symm


@[simp]
theorem self_trans_symm (e : α ≃ᵐ β) : e.trans e.symm = refl α :=
  ext e.symm_comp_self


protected theorem surjective (e : α ≃ᵐ β) : Surjective e :=
  e.toEquiv.surjective


protected theorem bijective (e : α ≃ᵐ β) : Bijective e :=
  e.toEquiv.bijective


protected theorem injective (e : α ≃ᵐ β) : Injective e :=
  e.toEquiv.injective


@[simp]
theorem symm_preimage_preimage (e : α ≃ᵐ β) (s : Set β) : e.symm ⁻¹' (e ⁻¹' s) = s :=
  e.toEquiv.symm_preimage_preimage s


theorem image_eq_preimage (e : α ≃ᵐ β) (s : Set α) : e '' s = e.symm ⁻¹' s :=
  e.toEquiv.image_eq_preimage s


lemma preimage_symm (e : α ≃ᵐ β) (s : Set α) : e.symm ⁻¹' s = e '' s := (image_eq_preimage _ _).symm


lemma image_symm (e : α ≃ᵐ β) (s : Set β) : e.symm '' s = e ⁻¹' s := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    e : MeasurableEquiv α β
    s : Set β
    ⊢ Eq (Set.image (⇑e.symm) s) (Set.preimage (⇑e) s)
  -/
  rw [← symm_symm e, preimage_symm, symm_symm]
  /-
    🎉 no goals
  -/


lemma eq_image_iff_symm_image_eq (e : α ≃ᵐ β) (s : Set β) (t : Set α) :
    s = e '' t ↔ e.symm '' s = t := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    e : MeasurableEquiv α β
    s : Set β
    t : Set α
    ⊢ Iff (Eq s (Set.image (⇑e) t)) (Eq (Set.image (⇑e.symm) s) t)
  -/
  rw [← coe_toEquiv, Equiv.eq_image_iff_symm_image_eq, coe_toEquiv_symm]
  /-
    🎉 no goals
  -/


@[simp]
lemma image_preimage (e : α ≃ᵐ β) (s : Set β) : e '' (e ⁻¹' s) = s := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    e : MeasurableEquiv α β
    s : Set β
    ⊢ Eq (Set.image (⇑e) (Set.preimage (⇑e) s)) s
  -/
  rw [← coe_toEquiv, Equiv.image_preimage]
  /-
    🎉 no goals
  -/


@[simp]
lemma preimage_image (e : α ≃ᵐ β) (s : Set α) : e ⁻¹' (e '' s) = s := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    e : MeasurableEquiv α β
    s : Set α
    ⊢ Eq (Set.preimage (⇑e) (Set.image (⇑e) s)) s
  -/
  rw [← coe_toEquiv, Equiv.preimage_image]
  /-
    🎉 no goals
  -/


@[simp]
theorem measurableSet_preimage (e : α ≃ᵐ β) {s : Set β} :
    MeasurableSet (e ⁻¹' s) ↔ MeasurableSet s :=
               /-
                 α : Type u_1
                 β : Type u_2
                 inst✝¹ : MeasurableSpace α
                 inst✝ : MeasurableSpace β
                 e : MeasurableEquiv α β
                 s : Set β
                 h : MeasurableSet (Set.preimage (⇑e) s)
                 ⊢ MeasurableSet s
               -/
  ⟨fun h => by simpa only [symm_preimage_preimage] using e.symm.measurable h, fun h =>
               /-
                 🎉 no goals
               -/
    e.measurable h⟩


@[simp]
theorem measurableSet_image (e : α ≃ᵐ β) :
                                                   /-
                                                     α : Type u_1
                                                     β : Type u_2
                                                     s : Set α
                                                     inst✝¹ : MeasurableSpace α
                                                     inst✝ : MeasurableSpace β
                                                     e : MeasurableEquiv α β
                                                     ⊢ Iff (MeasurableSet (Set.image (⇑e) s)) (MeasurableSet s)
                                                   -/
    MeasurableSet (e '' s) ↔ MeasurableSet s := by rw [image_eq_preimage, measurableSet_preimage]
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp] theorem map_eq (e : α ≃ᵐ β) : MeasurableSpace.map e ‹_› = ‹_› :=
  e.measurable.le_map.antisymm' fun _s ↦ e.measurableSet_preimage.1


/-- A measurable equivalence is a measurable embedding. -/
protected theorem measurableEmbedding (e : α ≃ᵐ β) : MeasurableEmbedding e where
  injective := e.injective
  measurable := e.measurable
  measurableSet_image' := fun _ => e.measurableSet_image.2


/-- Equal measurable spaces are equivalent. -/
protected def cast {α β} [i₁ : MeasurableSpace α] [i₂ : MeasurableSpace β] (h : α = β)
    (hi : HEq i₁ i₂) : α ≃ᵐ β where
  toEquiv := Equiv.cast h
  measurable_toFun := by
    /-
      α✝ : Type u_1
      β✝ : Type u_2
      γ : Type u_3
      δ : Type u_4
      δ' : Type u_5
      ι : Sort uι
      s t u : Set α✝
      inst✝² : MeasurableSpace α✝
      inst✝¹ : MeasurableSpace β✝
      inst✝ : MeasurableSpace γ
      α β : Type ?u.25310
      i₁ : MeasurableSpace α
      i₂ : MeasurableSpace β
      h : Eq α β
      hi : HEq i₁ i₂
      ⊢ Measurable ⇑(Equiv.cast h)
    -/
    subst h
    /-
      α✝ : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      δ' : Type u_5
      ι : Sort uι
      s t u : Set α✝
      inst✝² : MeasurableSpace α✝
      inst✝¹ : MeasurableSpace β
      inst✝ : MeasurableSpace γ
      α : Type ?u.25310
      i₁ i₂ : MeasurableSpace α
      hi : HEq i₁ i₂
      ⊢ Measurable ⇑(Equiv.cast ⋯)
    -/
    subst hi
    /-
      α✝ : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      δ' : Type u_5
      ι : Sort uι
      s t u : Set α✝
      inst✝² : MeasurableSpace α✝
      inst✝¹ : MeasurableSpace β
      inst✝ : MeasurableSpace γ
      α : Type ?u.25310
      i₁ : MeasurableSpace α
      ⊢ Measurable ⇑(Equiv.cast ⋯)
    -/
    exact measurable_id
    /-
      🎉 no goals
    -/
  measurable_invFun := by
    /-
      α✝ : Type u_1
      β✝ : Type u_2
      γ : Type u_3
      δ : Type u_4
      δ' : Type u_5
      ι : Sort uι
      s t u : Set α✝
      inst✝² : MeasurableSpace α✝
      inst✝¹ : MeasurableSpace β✝
      inst✝ : MeasurableSpace γ
      α β : Type ?u.25310
      i₁ : MeasurableSpace α
      i₂ : MeasurableSpace β
      h : Eq α β
      hi : HEq i₁ i₂
      ⊢ Measurable ⇑(Equiv.cast h).symm
    -/
    subst h
    /-
      α✝ : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      δ' : Type u_5
      ι : Sort uι
      s t u : Set α✝
      inst✝² : MeasurableSpace α✝
      inst✝¹ : MeasurableSpace β
      inst✝ : MeasurableSpace γ
      α : Type ?u.25310
      i₁ i₂ : MeasurableSpace α
      hi : HEq i₁ i₂
      ⊢ Measurable ⇑(Equiv.cast ⋯).symm
    -/
    subst hi
    /-
      α✝ : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      δ' : Type u_5
      ι : Sort uι
      s t u : Set α✝
      inst✝² : MeasurableSpace α✝
      inst✝¹ : MeasurableSpace β
      inst✝ : MeasurableSpace γ
      α : Type ?u.25310
      i₁ : MeasurableSpace α
      ⊢ Measurable ⇑(Equiv.cast ⋯).symm
    -/
    exact measurable_id
    /-
      🎉 no goals
    -/


/-- Measurable equivalence between `ULift α` and `α`. -/
def ulift.{u, v} {α : Type u} [MeasurableSpace α] : ULift.{v, u} α ≃ᵐ α :=
  ⟨Equiv.ulift, measurable_down, measurable_up⟩


protected theorem measurable_comp_iff {f : β → γ} (e : α ≃ᵐ β) :
    Measurable (f ∘ e) ↔ Measurable f :=
  Iff.intro
    (fun hfe => by
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        inst✝² : MeasurableSpace α
        inst✝¹ : MeasurableSpace β
        inst✝ : MeasurableSpace γ
        f : β → γ
        e : MeasurableEquiv α β
        hfe : Measurable (Function.comp f ⇑e)
        ⊢ Measurable f
      -/
      have : Measurable (f ∘ (e.symm.trans e).toEquiv) := hfe.comp e.symm.measurable
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        inst✝² : MeasurableSpace α
        inst✝¹ : MeasurableSpace β
        inst✝ : MeasurableSpace γ
        f : β → γ
        e : MeasurableEquiv α β
        hfe : Measurable (Function.comp f ⇑e)
        this : Measurable (Function.comp f ⇑(e.symm.trans e).toEquiv)
        ⊢ Measurable f
      -/
      rwa [coe_toEquiv, symm_trans_self] at this)
      /-
        🎉 no goals
      -/
    fun h => h.comp e.measurable


/-- Any two types with unique elements are measurably equivalent. -/
def ofUniqueOfUnique (α β : Type*) [MeasurableSpace α] [MeasurableSpace β] [Unique α] [Unique β] :
    α ≃ᵐ β where
  toEquiv := ofUnique α β
  measurable_toFun := Subsingleton.measurable
  measurable_invFun := Subsingleton.measurable


variable [MeasurableSpace δ] in
/-- Products of equivalent measurable spaces are equivalent. -/
def prodCongr (ab : α ≃ᵐ β) (cd : γ ≃ᵐ δ) : α × γ ≃ᵐ β × δ where
  toEquiv := .prodCongr ab.toEquiv cd.toEquiv
  measurable_toFun :=
    (ab.measurable_toFun.comp measurable_id.fst).prod_mk
      (cd.measurable_toFun.comp measurable_id.snd)
  measurable_invFun :=
    (ab.measurable_invFun.comp measurable_id.fst).prod_mk
      (cd.measurable_invFun.comp measurable_id.snd)


/-- Products of measurable spaces are symmetric. -/
def prodComm : α × β ≃ᵐ β × α where
  toEquiv := .prodComm α β
  measurable_toFun := measurable_id.snd.prod_mk measurable_id.fst
  measurable_invFun := measurable_id.snd.prod_mk measurable_id.fst


/-- Products of measurable spaces are associative. -/
def prodAssoc : (α × β) × γ ≃ᵐ α × β × γ where
  toEquiv := .prodAssoc α β γ
  measurable_toFun := measurable_fst.fst.prod_mk <| measurable_fst.snd.prod_mk measurable_snd
  measurable_invFun := (measurable_fst.prod_mk measurable_snd.fst).prod_mk measurable_snd.snd


/-- `PUnit` is a left identity for product of measurable spaces up to a measurable equivalence. -/
def punitProd : PUnit × α ≃ᵐ α where
  toEquiv := Equiv.punitProd α
  measurable_toFun := measurable_snd
  measurable_invFun := measurable_prod_mk_left


/-- `PUnit` is a right identity for product of measurable spaces up to a measurable equivalence. -/
def prodPUnit : α × PUnit ≃ᵐ α where
  toEquiv := Equiv.prodPUnit α
  measurable_toFun := measurable_fst
  measurable_invFun := measurable_prod_mk_right


variable [MeasurableSpace δ] in
/-- Sums of measurable spaces are symmetric. -/
def sumCongr (ab : α ≃ᵐ β) (cd : γ ≃ᵐ δ) : α ⊕ γ ≃ᵐ β ⊕ δ where
  toEquiv := .sumCongr ab.toEquiv cd.toEquiv
  measurable_toFun := ab.measurable.sumMap cd.measurable
  measurable_invFun := ab.symm.measurable.sumMap cd.symm.measurable


/-- `s ×ˢ t ≃ (s × t)` as measurable spaces. -/
def Set.prod (s : Set α) (t : Set β) : ↥(s ×ˢ t) ≃ᵐ s × t where
  toEquiv := Equiv.Set.prod s t
  measurable_toFun :=
    measurable_id.subtype_val.fst.subtype_mk.prod_mk measurable_id.subtype_val.snd.subtype_mk
  measurable_invFun :=
    Measurable.subtype_mk <| measurable_id.fst.subtype_val.prod_mk measurable_id.snd.subtype_val


/-- `univ α ≃ α` as measurable spaces. -/
def Set.univ (α : Type*) [MeasurableSpace α] : (univ : Set α) ≃ᵐ α where
  toEquiv := Equiv.Set.univ α
  measurable_toFun := measurable_id.subtype_val
  measurable_invFun := measurable_id.subtype_mk


/-- `{a} ≃ Unit` as measurable spaces. -/
def Set.singleton (a : α) : ({a} : Set α) ≃ᵐ Unit where
  toEquiv := Equiv.Set.singleton a
  measurable_toFun := measurable_const
  measurable_invFun := measurable_const


/-- `α` is equivalent to its image in `α ⊕ β` as measurable spaces. -/
def Set.rangeInl : (range Sum.inl : Set (α ⊕ β)) ≃ᵐ α where
  toEquiv := Equiv.Set.rangeInl α β
  measurable_toFun s (hs : MeasurableSet s) := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      δ' : Type u_5
      ι : Sort uι
      s✝ t u : Set α
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      inst✝ : MeasurableSpace γ
      s : Set α
      hs : MeasurableSet s
      ⊢ MeasurableSet (Set.preimage (⇑(Equiv.Set.rangeInl α β)) s)
    -/
    refine ⟨_, hs.inl_image, Set.ext ?_⟩
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      δ' : Type u_5
      ι : Sort uι
      s✝ t u : Set α
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      inst✝ : MeasurableSpace γ
      s : Set α
      hs : MeasurableSet s
      ⊢ ∀ (x : Subtype fun x => Membership.mem (Set.range Sum.inl) x), Iff (Membersh …
    -/
    rintro ⟨ab, a, rfl⟩
    /-
      case mk.intro
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      δ' : Type u_5
      ι : Sort uι
      s✝ t u : Set α
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      inst✝ : MeasurableSpace γ
      s : Set α
      hs : MeasurableSet s
      a : α
      ⊢ Iff (Membership.mem (Set.preimage Subtype.val (Set.image Sum.inl s)) ⟨Sum.in …
    -/
    simp [Set.range_inl]
    /-
      🎉 no goals
    -/
  measurable_invFun := Measurable.subtype_mk measurable_inl


/-- `β` is equivalent to its image in `α ⊕ β` as measurable spaces. -/
def Set.rangeInr : (range Sum.inr : Set (α ⊕ β)) ≃ᵐ β where
  toEquiv := Equiv.Set.rangeInr α β
  measurable_toFun s (hs : MeasurableSet s) := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      δ' : Type u_5
      ι : Sort uι
      s✝ t u : Set α
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      inst✝ : MeasurableSpace γ
      s : Set β
      hs : MeasurableSet s
      ⊢ MeasurableSet (Set.preimage (⇑(Equiv.Set.rangeInr α β)) s)
    -/
    refine ⟨_, hs.inr_image, Set.ext ?_⟩
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      δ' : Type u_5
      ι : Sort uι
      s✝ t u : Set α
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      inst✝ : MeasurableSpace γ
      s : Set β
      hs : MeasurableSet s
      ⊢ ∀ (x : Subtype fun x => Membership.mem (Set.range Sum.inr) x), Iff (Membersh …
    -/
    rintro ⟨ab, b, rfl⟩
    /-
      case mk.intro
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      δ' : Type u_5
      ι : Sort uι
      s✝ t u : Set α
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      inst✝ : MeasurableSpace γ
      s : Set β
      hs : MeasurableSet s
      b : β
      ⊢ Iff (Membership.mem (Set.preimage Subtype.val (Set.image Sum.inr s)) ⟨Sum.in …
    -/
    simp [Set.range_inr]
    /-
      🎉 no goals
    -/
  measurable_invFun := Measurable.subtype_mk measurable_inr


/-- Products distribute over sums (on the right) as measurable spaces. -/
def sumProdDistrib (α β γ) [MeasurableSpace α] [MeasurableSpace β] [MeasurableSpace γ] :
    (α ⊕ β) × γ ≃ᵐ (α × γ) ⊕ (β × γ) where
  toEquiv := .sumProdDistrib α β γ
  measurable_toFun := by
    refine
      measurable_of_measurable_union_cover (range Sum.inl ×ˢ (univ : Set γ))
        (range Sum.inr ×ˢ (univ : Set γ)) (measurableSet_range_inl.prod MeasurableSet.univ)
        (measurableSet_range_inr.prod MeasurableSet.univ)
        (by rintro ⟨a | b, c⟩ <;> simp [Set.prod_eq]) ?_ ?_
      /-
        case refine_1
        α✝ : Type u_1
        β✝ : Type u_2
        γ✝ : Type u_3
        δ : Type u_4
        δ' : Type u_5
        ι : Sort uι
        s t u : Set α✝
        inst✝⁵ : MeasurableSpace α✝
        inst✝⁴ : MeasurableSpace β✝
        inst✝³ : MeasurableSpace γ✝
        α : Type ?u.35914
        β : Type ?u.35917
        γ : Type ?u.36652
        inst✝² : MeasurableSpace α
        inst✝¹ : MeasurableSpace β
        inst✝ : MeasurableSpace γ
        ⊢ Measurable fun a => (Equiv.sumProdDistrib α β γ) ↑a
      -/
    · refine (Set.prod (range Sum.inl) univ).symm.measurable_comp_iff.1 ?_
      /-
        case refine_1
        α✝ : Type u_1
        β✝ : Type u_2
        γ✝ : Type u_3
        δ : Type u_4
        δ' : Type u_5
        ι : Sort uι
        s t u : Set α✝
        inst✝⁵ : MeasurableSpace α✝
        inst✝⁴ : MeasurableSpace β✝
        inst✝³ : MeasurableSpace γ✝
        α : Type ?u.35914
        β : Type ?u.35917
        γ : Type ?u.36652
        inst✝² : MeasurableSpace α
        inst✝¹ : MeasurableSpace β
        inst✝ : MeasurableSpace γ
        ⊢ Measurable (Function.comp (fun a => (Equiv.sumProdDistrib α β γ) ↑a) ⇑(Measu …
      -/
      refine (prodCongr Set.rangeInl (Set.univ _)).symm.measurable_comp_iff.1 ?_
      /-
        case refine_1
        α✝ : Type u_1
        β✝ : Type u_2
        γ✝ : Type u_3
        δ : Type u_4
        δ' : Type u_5
        ι : Sort uι
        s t u : Set α✝
        inst✝⁵ : MeasurableSpace α✝
        inst✝⁴ : MeasurableSpace β✝
        inst✝³ : MeasurableSpace γ✝
        α : Type ?u.35914
        β : Type ?u.35917
        γ : Type ?u.36652
        inst✝² : MeasurableSpace α
        inst✝¹ : MeasurableSpace β
        inst✝ : MeasurableSpace γ
        ⊢ Measurable (Function.comp (Function.comp (fun a => (Equiv.sumProdDistrib α β …
      -/
      exact measurable_inl
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        α✝ : Type u_1
        β✝ : Type u_2
        γ✝ : Type u_3
        δ : Type u_4
        δ' : Type u_5
        ι : Sort uι
        s t u : Set α✝
        inst✝⁵ : MeasurableSpace α✝
        inst✝⁴ : MeasurableSpace β✝
        inst✝³ : MeasurableSpace γ✝
        α : Type ?u.35914
        β : Type ?u.35917
        γ : Type ?u.36652
        inst✝² : MeasurableSpace α
        inst✝¹ : MeasurableSpace β
        inst✝ : MeasurableSpace γ
        ⊢ Measurable fun a => (Equiv.sumProdDistrib α β γ) ↑a
      -/
    · refine (Set.prod (range Sum.inr) univ).symm.measurable_comp_iff.1 ?_
      /-
        case refine_2
        α✝ : Type u_1
        β✝ : Type u_2
        γ✝ : Type u_3
        δ : Type u_4
        δ' : Type u_5
        ι : Sort uι
        s t u : Set α✝
        inst✝⁵ : MeasurableSpace α✝
        inst✝⁴ : MeasurableSpace β✝
        inst✝³ : MeasurableSpace γ✝
        α : Type ?u.35914
        β : Type ?u.35917
        γ : Type ?u.36652
        inst✝² : MeasurableSpace α
        inst✝¹ : MeasurableSpace β
        inst✝ : MeasurableSpace γ
        ⊢ Measurable (Function.comp (fun a => (Equiv.sumProdDistrib α β γ) ↑a) ⇑(Measu …
      -/
      refine (prodCongr Set.rangeInr (Set.univ _)).symm.measurable_comp_iff.1 ?_
      /-
        case refine_2
        α✝ : Type u_1
        β✝ : Type u_2
        γ✝ : Type u_3
        δ : Type u_4
        δ' : Type u_5
        ι : Sort uι
        s t u : Set α✝
        inst✝⁵ : MeasurableSpace α✝
        inst✝⁴ : MeasurableSpace β✝
        inst✝³ : MeasurableSpace γ✝
        α : Type ?u.35914
        β : Type ?u.35917
        γ : Type ?u.36652
        inst✝² : MeasurableSpace α
        inst✝¹ : MeasurableSpace β
        inst✝ : MeasurableSpace γ
        ⊢ Measurable (Function.comp (Function.comp (fun a => (Equiv.sumProdDistrib α β …
      -/
      exact measurable_inr
      /-
        🎉 no goals
      -/
  measurable_invFun :=
    measurable_sum ((measurable_inl.comp measurable_fst).prod_mk measurable_snd)
      ((measurable_inr.comp measurable_fst).prod_mk measurable_snd)


/-- Products distribute over sums (on the left) as measurable spaces. -/
def prodSumDistrib (α β γ) [MeasurableSpace α] [MeasurableSpace β] [MeasurableSpace γ] :
    α × (β ⊕ γ) ≃ᵐ (α × β) ⊕ (α × γ) :=
  prodComm.trans <| (sumProdDistrib _ _ _).trans <| sumCongr prodComm prodComm


/-- Products distribute over sums as measurable spaces. -/
def sumProdSum (α β γ δ) [MeasurableSpace α] [MeasurableSpace β] [MeasurableSpace γ]
    [MeasurableSpace δ] : (α ⊕ β) × (γ ⊕ δ) ≃ᵐ ((α × γ) ⊕ (α × δ)) ⊕ ((β × γ) ⊕ (β × δ)) :=
  (sumProdDistrib _ _ _).trans <| sumCongr (prodSumDistrib _ _ _) (prodSumDistrib _ _ _)


/-- A family of measurable equivalences `Π a, β₁ a ≃ᵐ β₂ a` generates a measurable equivalence
  between `Π a, β₁ a` and `Π a, β₂ a`. -/
def piCongrRight (e : ∀ a, π a ≃ᵐ π' a) : (∀ a, π a) ≃ᵐ ∀ a, π' a where
  toEquiv := .piCongrRight fun a => (e a).toEquiv
  measurable_toFun :=
    measurable_pi_lambda _ fun i => (e i).measurable_toFun.comp (measurable_pi_apply i)
  measurable_invFun :=
    measurable_pi_lambda _ fun i => (e i).measurable_invFun.comp (measurable_pi_apply i)


variable (π) in
/-- Moving a dependent type along an equivalence of coordinates, as a measurable equivalence. -/
def piCongrLeft (f : δ ≃ δ') : (∀ b, π (f b)) ≃ᵐ ∀ a, π a where
  __ := Equiv.piCongrLeft π f
  measurable_toFun := measurable_piCongrLeft f
  measurable_invFun := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      δ' : Type u_5
      ι : Sort uι
      s t u : Set α
      inst✝⁴ : MeasurableSpace α
      inst✝³ : MeasurableSpace β
      inst✝² : MeasurableSpace γ
      π : δ' → Type u_6
      π' : δ' → Type u_7
      inst✝¹ : (x : δ') → MeasurableSpace (π x)
      inst✝ : (x : δ') → MeasurableSpace (π' x)
      f : Equiv δ δ'
      ⊢ Measurable ⇑__spread✝⁻⁰.symm
    -/
    simp only [invFun_as_coe, coe_fn_symm_mk]
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      δ' : Type u_5
      ι : Sort uι
      s t u : Set α
      inst✝⁴ : MeasurableSpace α
      inst✝³ : MeasurableSpace β
      inst✝² : MeasurableSpace γ
      π : δ' → Type u_6
      π' : δ' → Type u_7
      inst✝¹ : (x : δ') → MeasurableSpace (π x)
      inst✝ : (x : δ') → MeasurableSpace (π' x)
      f : Equiv δ δ'
      ⊢ Measurable ⇑(Equiv.piCongrLeft π f).symm
    -/
    rw [measurable_pi_iff]
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      δ' : Type u_5
      ι : Sort uι
      s t u : Set α
      inst✝⁴ : MeasurableSpace α
      inst✝³ : MeasurableSpace β
      inst✝² : MeasurableSpace γ
      π : δ' → Type u_6
      π' : δ' → Type u_7
      inst✝¹ : (x : δ') → MeasurableSpace (π x)
      inst✝ : (x : δ') → MeasurableSpace (π' x)
      f : Equiv δ δ'
      ⊢ ∀ (a : δ), Measurable fun x => (Equiv.piCongrLeft π f).symm x a
    -/
    exact fun i => measurable_pi_apply (f i)
    /-
      🎉 no goals
    -/


theorem coe_piCongrLeft (f : δ ≃ δ') :
                                                               /-
                                                                 δ : Type u_4
                                                                 δ' : Type u_5
                                                                 π : δ' → Type u_6
                                                                 inst✝ : (x : δ') → MeasurableSpace (π x)
                                                                 f : Equiv δ δ'
                                                                 ⊢ Eq ⇑(MeasurableEquiv.piCongrLeft π f) ⇑(Equiv.piCongrLeft π f)
                                                               -/
    ⇑(MeasurableEquiv.piCongrLeft π f) = f.piCongrLeft π := by rfl
                                                               /-
                                                                 🎉 no goals
                                                               -/


lemma piCongrLeft_apply_apply {ι ι' : Type*} (e : ι ≃ ι') {β : ι' → Type*}
    [∀ i', MeasurableSpace (β i')] (x : (i : ι) → β (e i)) (i : ι) :
    piCongrLeft (fun i' ↦ β i') e x (e i) = x i := by
  /-
    ι : Type u_8
    ι' : Type u_9
    e : Equiv ι ι'
    β : ι' → Type u_10
    inst✝ : (i' : ι') → MeasurableSpace (β i')
    x : (i : ι) → β (e i)
    i : ι
    ⊢ Eq ((MeasurableEquiv.piCongrLeft (fun i' => β i') e) x (e i)) (x i)
  -/
  rw [piCongrLeft, coe_mk, Equiv.piCongrLeft_apply_apply]
  /-
    🎉 no goals
  -/


/-- The isomorphism `(γ → α × β) ≃ (γ → α) × (γ → β)` as a measurable equivalence. -/
def arrowProdEquivProdArrow (α β γ : Type*) [MeasurableSpace α] [MeasurableSpace β] :
    (γ → α × β) ≃ᵐ (γ → α) × (γ → β) where
  __ := Equiv.arrowProdEquivProdArrow α β γ
  measurable_toFun _ h := by
    /-
      α✝ : Type u_1
      β✝ : Type u_2
      γ✝ : Type u_3
      δ : Type u_4
      δ' : Type u_5
      ι : Sort uι
      s t u : Set α✝
      inst✝⁶ : MeasurableSpace α✝
      inst✝⁵ : MeasurableSpace β✝
      inst✝⁴ : MeasurableSpace γ✝
      π : δ' → Type u_6
      π' : δ' → Type u_7
      inst✝³ : (x : δ') → MeasurableSpace (π x)
      inst✝² : (x : δ') → MeasurableSpace (π' x)
      α : Type u_8
      β : Type u_9
      γ : Type u_10
      inst✝¹ : MeasurableSpace α
      inst✝ : MeasurableSpace β
      x✝ : Set (Prod (γ → α) (γ → β))
      h : MeasurableSet x✝
      ⊢ MeasurableSet (Set.preimage (⇑__spread✝⁻⁰) x✝)
    -/
    simp_rw [Equiv.arrowProdEquivProdArrow, coe_fn_mk]
    #adaptation_note
    /--
    After https://github.com/leanprover/lean4/pull/6024
    we need provide the type hints `(a : γ → α × β)`, to avoid unification issues.
    -/
    exact MeasurableSet.preimage h (Measurable.prod_mk
        (measurable_pi_lambda (fun (a : γ → α × β) c ↦ (a c).1)
          fun a ↦ (measurable_pi_apply a).fst)
        (measurable_pi_lambda (fun (a : γ → α × β) c ↦ (a c).2)
          fun a ↦ (measurable_pi_apply a).snd))
  measurable_invFun _ h := by
    /-
      α✝ : Type u_1
      β✝ : Type u_2
      γ✝ : Type u_3
      δ : Type u_4
      δ' : Type u_5
      ι : Sort uι
      s t u : Set α✝
      inst✝⁶ : MeasurableSpace α✝
      inst✝⁵ : MeasurableSpace β✝
      inst✝⁴ : MeasurableSpace γ✝
      π : δ' → Type u_6
      π' : δ' → Type u_7
      inst✝³ : (x : δ') → MeasurableSpace (π x)
      inst✝² : (x : δ') → MeasurableSpace (π' x)
      α : Type u_8
      β : Type u_9
      γ : Type u_10
      inst✝¹ : MeasurableSpace α
      inst✝ : MeasurableSpace β
      x✝ : Set (γ → Prod α β)
      h : MeasurableSet x✝
      ⊢ MeasurableSet (Set.preimage (⇑__spread✝⁻⁰.symm) x✝)
    -/
    simp_rw [Equiv.arrowProdEquivProdArrow, coe_fn_symm_mk]
    /-
      α✝ : Type u_1
      β✝ : Type u_2
      γ✝ : Type u_3
      δ : Type u_4
      δ' : Type u_5
      ι : Sort uι
      s t u : Set α✝
      inst✝⁶ : MeasurableSpace α✝
      inst✝⁵ : MeasurableSpace β✝
      inst✝⁴ : MeasurableSpace γ✝
      π : δ' → Type u_6
      π' : δ' → Type u_7
      inst✝³ : (x : δ') → MeasurableSpace (π x)
      inst✝² : (x : δ') → MeasurableSpace (π' x)
      α : Type u_8
      β : Type u_9
      γ : Type u_10
      inst✝¹ : MeasurableSpace α
      inst✝ : MeasurableSpace β
      x✝ : Set (γ → Prod α β)
      h : MeasurableSet x✝
      ⊢ MeasurableSet (Set.preimage (fun p c => { fst := p.1 c, snd := p.2 c }) x✝)
    -/
    exact MeasurableSet.preimage h (by measurability)
    /-
      🎉 no goals
    -/


/-- The measurable equivalence `(α₁ → β₁) ≃ᵐ (α₂ → β₂)` induced by `α₁ ≃ α₂` and `β₁ ≃ᵐ β₂`. -/
def arrowCongr' {α₁ β₁ α₂ β₂ : Type*} [MeasurableSpace β₁] [MeasurableSpace β₂]
    (hα : α₁ ≃ α₂) (hβ : β₁ ≃ᵐ β₂) :
    (α₁ → β₁) ≃ᵐ (α₂ → β₂) where
  __ := Equiv.arrowCongr' hα hβ
  measurable_toFun _ h := by
    exact MeasurableSet.preimage h <|
      measurable_pi_iff.mpr fun _ ↦ hβ.measurable.comp' (measurable_pi_apply _)
  measurable_invFun _ h := by
    exact MeasurableSet.preimage h <|
      measurable_pi_iff.mpr fun _ ↦ hβ.symm.measurable.comp' (measurable_pi_apply _)


/-- Pi-types are measurably equivalent to iterated products. -/
@[simps! (config := .asFn)]
def piMeasurableEquivTProd [DecidableEq δ'] {l : List δ'} (hnd : l.Nodup) (h : ∀ i, i ∈ l) :
    (∀ i, π i) ≃ᵐ List.TProd π l where
  toEquiv := List.TProd.piEquivTProd hnd h
  measurable_toFun := measurable_tProd_mk l
  measurable_invFun := measurable_tProd_elim' h


variable (π) in
/-- The measurable equivalence `(∀ i, π i) ≃ᵐ π ⋆` when the domain of `π` only contains `⋆` -/
@[simps! (config := .asFn)]
def piUnique [Unique δ'] : (∀ i, π i) ≃ᵐ π default where
  toEquiv := Equiv.piUnique π
  measurable_toFun := measurable_pi_apply _
  measurable_invFun := measurable_uniqueElim


/-- If `α` has a unique term, then the type of function `α → β` is measurably equivalent to `β`. -/
@[simps! (config := .asFn)]
def funUnique (α β : Type*) [Unique α] [MeasurableSpace β] : (α → β) ≃ᵐ β :=
  MeasurableEquiv.piUnique _


/-- The space `Π i : Fin 2, α i` is measurably equivalent to `α 0 × α 1`. -/
@[simps! (config := .asFn)]
def piFinTwo (α : Fin 2 → Type*) [∀ i, MeasurableSpace (α i)] : (∀ i, α i) ≃ᵐ α 0 × α 1 where
  toEquiv := piFinTwoEquiv α
  measurable_toFun := Measurable.prod (measurable_pi_apply _) (measurable_pi_apply _)
  measurable_invFun := measurable_pi_iff.2 <| Fin.forall_fin_two.2 ⟨measurable_fst, measurable_snd⟩


/-- The space `Fin 2 → α` is measurably equivalent to `α × α`. -/
@[simps! (config := .asFn)]
def finTwoArrow : (Fin 2 → α) ≃ᵐ α × α :=
  piFinTwo fun _ => α


/-- Measurable equivalence between `Π j : Fin (n + 1), α j` and
`α i × Π j : Fin n, α (Fin.succAbove i j)`.

Measurable version of `Fin.insertNthEquiv`. -/
@[simps! (config := .asFn)]
def piFinSuccAbove {n : ℕ} (α : Fin (n + 1) → Type*) [∀ i, MeasurableSpace (α i)]
    (i : Fin (n + 1)) : (∀ j, α j) ≃ᵐ α i × ∀ j, α (i.succAbove j) where
  toEquiv := (Fin.insertNthEquiv α i).symm
  measurable_toFun := (measurable_pi_apply i).prod_mk <| measurable_pi_iff.2 fun _ =>
    measurable_pi_apply _
  measurable_invFun := measurable_pi_iff.2 <| i.forall_iff_succAbove.2
        /-
          α✝ : Type u_1
          β : Type u_2
          γ : Type u_3
          δ : Type u_4
          δ' : Type u_5
          ι : Sort uι
          s t u : Set α✝
          inst✝⁵ : MeasurableSpace α✝
          inst✝⁴ : MeasurableSpace β
          inst✝³ : MeasurableSpace γ
          π : δ' → Type u_6
          π' : δ' → Type u_7
          inst✝² : (x : δ') → MeasurableSpace (π x)
          inst✝¹ : (x : δ') → MeasurableSpace (π' x)
          n : Nat
          α : Fin (HAdd.hAdd n 1) → Type u_8
          inst✝ : (i : Fin (HAdd.hAdd n 1)) → MeasurableSpace (α i)
          i : Fin (HAdd.hAdd n 1)
          ⊢ Measurable fun x => (Fin.insertNthEquiv α i).symm.symm x i
        -/
        /-
          🎉 no goals
        -/
    ⟨by simp [measurable_fst], fun j => by simpa using (measurable_pi_apply _).comp measurable_snd⟩
                                           /-
                                             🎉 no goals
                                           -/


/-- Measurable equivalence between (dependent) functions on a type and pairs of functions on
`{i // p i}` and `{i // ¬p i}`. See also `Equiv.piEquivPiSubtypeProd`. -/
@[simps! (config := .asFn)]
def piEquivPiSubtypeProd (p : δ' → Prop) [DecidablePred p] :
    (∀ i, π i) ≃ᵐ (∀ i : Subtype p, π i) × ∀ i : { i // ¬p i }, π i where
  toEquiv := .piEquivPiSubtypeProd p π
  measurable_toFun := measurable_piEquivPiSubtypeProd π p
  measurable_invFun := measurable_piEquivPiSubtypeProd_symm π p


/-- The measurable equivalence between the pi type over a sum type and a product of pi-types.
This is similar to `MeasurableEquiv.piEquivPiSubtypeProd`. -/
def sumPiEquivProdPi (α : δ ⊕ δ' → Type*) [∀ i, MeasurableSpace (α i)] :
    (∀ i, α i) ≃ᵐ (∀ i, α (.inl i)) × ∀ i', α (.inr i') where
  __ := Equiv.sumPiEquivProdPi α
  measurable_toFun := by
    /-
      α✝ : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      δ' : Type u_5
      ι : Sort uι
      s t u : Set α✝
      inst✝⁵ : MeasurableSpace α✝
      inst✝⁴ : MeasurableSpace β
      inst✝³ : MeasurableSpace γ
      π : δ' → Type u_6
      π' : δ' → Type u_7
      inst✝² : (x : δ') → MeasurableSpace (π x)
      inst✝¹ : (x : δ') → MeasurableSpace (π' x)
      α : Sum δ δ' → Type u_8
      inst✝ : (i : Sum δ δ') → MeasurableSpace (α i)
      ⊢ Measurable ⇑__spread✝⁻⁰
    -/
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
    apply Measurable.prod <;> rw [measurable_pi_iff] <;> rintro i <;> apply measurable_pi_apply
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
  measurable_invFun := by
    /-
      α✝ : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      δ' : Type u_5
      ι : Sort uι
      s t u : Set α✝
      inst✝⁵ : MeasurableSpace α✝
      inst✝⁴ : MeasurableSpace β
      inst✝³ : MeasurableSpace γ
      π : δ' → Type u_6
      π' : δ' → Type u_7
      inst✝² : (x : δ') → MeasurableSpace (π x)
      inst✝¹ : (x : δ') → MeasurableSpace (π' x)
      α : Sum δ δ' → Type u_8
      inst✝ : (i : Sum δ δ') → MeasurableSpace (α i)
      ⊢ Measurable ⇑__spread✝⁻⁰.symm
    -/
    rw [measurable_pi_iff]; rintro (i | i)
      /-
        case inl
        α✝ : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        δ' : Type u_5
        ι : Sort uι
        s t u : Set α✝
        inst✝⁵ : MeasurableSpace α✝
        inst✝⁴ : MeasurableSpace β
        inst✝³ : MeasurableSpace γ
        π : δ' → Type u_6
        π' : δ' → Type u_7
        inst✝² : (x : δ') → MeasurableSpace (π x)
        inst✝¹ : (x : δ') → MeasurableSpace (π' x)
        α : Sum δ δ' → Type u_8
        inst✝ : (i : Sum δ δ') → MeasurableSpace (α i)
        i : δ
        ⊢ Measurable fun x => __spread✝⁻⁰.symm x (Sum.inl i)
      -/
    · exact measurable_pi_iff.1 measurable_fst _
      /-
        🎉 no goals
      -/
      /-
        case inr
        α✝ : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        δ' : Type u_5
        ι : Sort uι
        s t u : Set α✝
        inst✝⁵ : MeasurableSpace α✝
        inst✝⁴ : MeasurableSpace β
        inst✝³ : MeasurableSpace γ
        π : δ' → Type u_6
        π' : δ' → Type u_7
        inst✝² : (x : δ') → MeasurableSpace (π x)
        inst✝¹ : (x : δ') → MeasurableSpace (π' x)
        α : Sum δ δ' → Type u_8
        inst✝ : (i : Sum δ δ') → MeasurableSpace (α i)
        i : δ'
        ⊢ Measurable fun x => __spread✝⁻⁰.symm x (Sum.inr i)
      -/
    · exact measurable_pi_iff.1 measurable_snd _
      /-
        🎉 no goals
      -/


theorem coe_sumPiEquivProdPi (α : δ ⊕ δ' → Type*) [∀ i, MeasurableSpace (α i)] :
                                                                           /-
                                                                             δ : Type u_4
                                                                             δ' : Type u_5
                                                                             α : Sum δ δ' → Type u_8
                                                                             inst✝ : (i : Sum δ δ') → MeasurableSpace (α i)
                                                                             ⊢ Eq ⇑(MeasurableEquiv.sumPiEquivProdPi α) ⇑(Equiv.sumPiEquivProdPi α)
                                                                           -/
    ⇑(MeasurableEquiv.sumPiEquivProdPi α) = Equiv.sumPiEquivProdPi α := by rfl
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


theorem coe_sumPiEquivProdPi_symm (α : δ ⊕ δ' → Type*) [∀ i, MeasurableSpace (α i)] :
                                                                                       /-
                                                                                         δ : Type u_4
                                                                                         δ' : Type u_5
                                                                                         α : Sum δ δ' → Type u_8
                                                                                         inst✝ : (i : Sum δ δ') → MeasurableSpace (α i)
                                                                                         ⊢ Eq ⇑(MeasurableEquiv.sumPiEquivProdPi α).symm ⇑(Equiv.sumPiEquivProdPi α).symm
                                                                                       -/
    ⇑(MeasurableEquiv.sumPiEquivProdPi α).symm = (Equiv.sumPiEquivProdPi α).symm := by rfl
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/


/-- The measurable equivalence for (dependent) functions on an Option type
  `(∀ i : Option δ, α i) ≃ᵐ (∀ (i : δ), α i) × α none`. -/
def piOptionEquivProd {δ : Type*} (α : Option δ → Type*) [∀ i, MeasurableSpace (α i)] :
    (∀ i, α i) ≃ᵐ (∀ (i : δ), α i) × α none :=
  let e : Option δ ≃ δ ⊕ Unit := Equiv.optionEquivSumPUnit δ
  let em1 : ((i : δ ⊕ Unit) → α (e.symm i)) ≃ᵐ ((a : Option δ) → α a) :=
    MeasurableEquiv.piCongrLeft α e.symm
  let em2 : ((i : δ ⊕ Unit) → α (e.symm i)) ≃ᵐ ((i : δ) → α (e.symm (Sum.inl i)))
      × ((i' : Unit) → α (e.symm (Sum.inr i'))) :=
    MeasurableEquiv.sumPiEquivProdPi (fun i ↦ α (e.symm i))
  let em3 : ((i : δ) → α (e.symm (Sum.inl i))) × ((i' : Unit) → α (e.symm (Sum.inr i')))
      ≃ᵐ ((i : δ) → α (some i)) × α none :=
    MeasurableEquiv.prodCongr (MeasurableEquiv.refl ((i : δ) → α (e.symm (Sum.inl i))))
      (MeasurableEquiv.piUnique fun i ↦ α (e.symm (Sum.inr i)))
  em1.symm.trans <| em2.trans em3


/-- The measurable equivalence `(∀ i : s, π i) × (∀ i : t, π i) ≃ᵐ (∀ i : s ∪ t, π i)`
  for disjoint finsets `s` and `t`. `Equiv.piFinsetUnion` as a measurable equivalence. -/
def piFinsetUnion [DecidableEq δ'] {s t : Finset δ'} (h : Disjoint s t) :
    ((∀ i : s, π i) × ∀ i : t, π i) ≃ᵐ ∀ i : (s ∪ t : Finset δ'), π i :=
  letI e := Finset.union s t h
  MeasurableEquiv.sumPiEquivProdPi (fun b ↦ π (e b)) |>.symm.trans <|
    .piCongrLeft (fun i : ↥(s ∪ t) ↦ π i) e


/-- If `s` is a measurable set in a measurable space, that space is equivalent
to the sum of `s` and `sᶜ`. -/
def sumCompl {s : Set α} [DecidablePred (· ∈ s)] (hs : MeasurableSet s) :
    s ⊕ (sᶜ : Set α) ≃ᵐ α where
  toEquiv := .sumCompl (· ∈ s)
  measurable_toFun := measurable_subtype_coe.sumElim measurable_subtype_coe
  measurable_invFun := Measurable.dite measurable_inl measurable_inr hs


/-- Convert a measurable involutive function `f` to a measurable permutation with
`toFun = invFun = f`. See also `Function.Involutive.toPerm`. -/
@[simps toEquiv]
def ofInvolutive (f : α → α) (hf : Involutive f) (hf' : Measurable f) : α ≃ᵐ α where
  toEquiv := hf.toPerm
  measurable_toFun := hf'
  measurable_invFun := hf'


@[simp] theorem ofInvolutive_apply (f : α → α) (hf : Involutive f) (hf' : Measurable f) (a : α) :
    ofInvolutive f hf hf' a = f a := rfl


@[simp] theorem ofInvolutive_symm (f : α → α) (hf : Involutive f) (hf' : Measurable f) :
    (ofInvolutive f hf hf').symm = ofInvolutive f hf hf' := rfl


@[simp] theorem comap_eq (hf : MeasurableEmbedding f) : MeasurableSpace.comap f ‹_› = ‹_› :=
  hf.measurable.comap_le.antisymm fun _s h ↦
    ⟨_, hf.measurableSet_image' h, hf.injective.preimage_image _⟩


theorem iff_comap_eq :
    MeasurableEmbedding f ↔
      Injective f ∧ MeasurableSpace.comap f ‹_› = ‹_› ∧ MeasurableSet (range f) :=
  ⟨fun hf ↦ ⟨hf.injective, hf.comap_eq, hf.measurableSet_range⟩, fun hf ↦
    { injective := hf.1
                       /-
                         α : Type u_1
                         β : Type u_2
                         inst✝¹ : MeasurableSpace α
                         inst✝ : MeasurableSpace β
                         f : α → β
                         hf : And (Function.Injective f) (And (Eq (MeasurableSpace.comap f inst✝) inst✝ …
                         ⊢ Measurable f
                       -/
      measurable := by rw [← hf.2.1]; exact comap_measurable f
                                      /-
                                        🎉 no goals
                                      -/
      measurableSet_image' := by
        /-
          α : Type u_1
          β : Type u_2
          inst✝¹ : MeasurableSpace α
          inst✝ : MeasurableSpace β
          f : α → β
          hf : And (Function.Injective f) (And (Eq (MeasurableSpace.comap f inst✝) inst✝ …
          ⊢ ∀ ⦃s : Set α⦄, MeasurableSet s → MeasurableSet (Set.image f s)
        -/
        rw [← hf.2.1]
        /-
          α : Type u_1
          β : Type u_2
          inst✝¹ : MeasurableSpace α
          inst✝ : MeasurableSpace β
          f : α → β
          hf : And (Function.Injective f) (And (Eq (MeasurableSpace.comap f inst✝) inst✝ …
          ⊢ ∀ ⦃s : Set α⦄, MeasurableSet s → MeasurableSet (Set.image f s)
        -/
        rintro _ ⟨s, hs, rfl⟩
        /-
          case intro.intro
          α : Type u_1
          β : Type u_2
          inst✝¹ : MeasurableSpace α
          inst✝ : MeasurableSpace β
          f : α → β
          hf : And (Function.Injective f) (And (Eq (MeasurableSpace.comap f inst✝) inst✝ …
          s : Set β
          hs : MeasurableSet s
          ⊢ MeasurableSet (Set.image f (Set.preimage f s))
        -/
        simpa only [image_preimage_eq_inter_range] using hs.inter hf.2.2 }⟩
        /-
          🎉 no goals
        -/


/-- A set is equivalent to its image under a function `f` as measurable spaces,
  if `f` is a measurable embedding -/
noncomputable def equivImage (s : Set α) (hf : MeasurableEmbedding f) : s ≃ᵐ f '' s where
  toEquiv := Equiv.Set.image f s hf.injective
  measurable_toFun := (hf.measurable.comp measurable_id.subtype_val).subtype_mk
  measurable_invFun := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      δ' : Type u_5
      ι : Sort uι
      s✝ t u : Set α
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      inst✝ : MeasurableSpace γ
      f : α → β
      g : β → α
      s : Set α
      hf : MeasurableEmbedding f
      ⊢ Measurable ⇑(Equiv.Set.image f s ⋯).symm
    -/
    rintro t ⟨u, hu, rfl⟩
    simpa [preimage_preimage, Set.image_symm_preimage hf.injective]
      using measurable_subtype_coe (hf.measurableSet_image' hu)


/-- The domain of `f` is equivalent to its range as measurable spaces,
  if `f` is a measurable embedding -/
noncomputable def equivRange (hf : MeasurableEmbedding f) : α ≃ᵐ range f :=
  (MeasurableEquiv.Set.univ _).symm.trans <|
                                                           /-
                                                             α : Type u_1
                                                             β : Type u_2
                                                             γ : Type u_3
                                                             δ : Type u_4
                                                             δ' : Type u_5
                                                             ι : Sort uι
                                                             s t u : Set α
                                                             inst✝² : MeasurableSpace α
                                                             inst✝¹ : MeasurableSpace β
                                                             inst✝ : MeasurableSpace γ
                                                             f : α → β
                                                             g : β → α
                                                             hf : MeasurableEmbedding f
                                                             ⊢ Eq ↑(Set.image f Set.univ) ↑(Set.range f)
                                                           -/
                                                           /-
                                                             🎉 no goals
                                                           -/
    (hf.equivImage univ).trans <| MeasurableEquiv.cast (by rw [image_univ]) (by rw [image_univ])
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


theorem of_measurable_inverse_on_range {g : range f → α} (hf₁ : Measurable f)
    (hf₂ : MeasurableSet (range f)) (hg : Measurable g) (H : LeftInverse g (rangeFactorization f)) :
    MeasurableEmbedding f := by
  set e : α ≃ᵐ range f :=
    ⟨⟨rangeFactorization f, g, H, H.rightInverse_of_surjective surjective_onto_range⟩,
      hf₁.subtype_mk, hg⟩
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    f : α → β
    g : ↑(Set.range f) → α
    hf₁ : Measurable f
    hf₂ : MeasurableSet (Set.range f)
    hg : Measurable g
    H : Function.LeftInverse g (Set.rangeFactorization f)
    e : MeasurableEquiv α ↑(Set.range f) := { toFun := Set.rangeFactorization f, i …
    ⊢ MeasurableEmbedding f
  -/
  exact (MeasurableEmbedding.subtype_coe hf₂).comp e.measurableEmbedding
  /-
    🎉 no goals
  -/


theorem of_measurable_inverse (hf₁ : Measurable f) (hf₂ : MeasurableSet (range f))
    (hg : Measurable g) (H : LeftInverse g f) : MeasurableEmbedding f :=
  of_measurable_inverse_on_range hf₁ hf₂ (hg.comp measurable_subtype_coe) H


/-- The **measurable Schröder-Bernstein Theorem**: given measurable embeddings
`α → β` and `β → α`, we can find a measurable equivalence `α ≃ᵐ β`. -/
noncomputable def schroederBernstein {f : α → β} {g : β → α} (hf : MeasurableEmbedding f)
    (hg : MeasurableEmbedding g) : α ≃ᵐ β := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    δ' : Type u_5
    ι : Sort uι
    s t u : Set α
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    inst✝ : MeasurableSpace γ
    f✝ : α → β
    g✝ : β → α
    f : α → β
    g : β → α
    hf : MeasurableEmbedding f
    hg : MeasurableEmbedding g
    ⊢ MeasurableEquiv α β
  -/
  let F : Set α → Set α := fun A => (g '' (f '' A)ᶜ)ᶜ
  -- We follow the proof of the usual SB theorem in mathlib,
  -- the crux of which is finding a fixed point of this F.
  -- However, we must find this fixed point manually instead of invoking Knaster-Tarski
  -- in order to make sure it is measurable.
  suffices Σ'A : Set α, MeasurableSet A ∧ F A = A by
    classical
    rcases this with ⟨A, Ameas, Afp⟩
    let B := f '' A
    have Bmeas : MeasurableSet B := hf.measurableSet_image' Ameas
    refine (MeasurableEquiv.sumCompl Ameas).symm.trans
      (MeasurableEquiv.trans ?_ (MeasurableEquiv.sumCompl Bmeas))
    apply MeasurableEquiv.sumCongr (hf.equivImage _)
    have : Aᶜ = g '' Bᶜ := by
      apply compl_injective
      rw [← Afp]
      simp [F, B]
    rw [this]
    exact (hg.equivImage _).symm
  have Fmono : ∀ {A B}, A ⊆ B → F A ⊆ F B := fun h =>
    compl_subset_compl.mpr <| Set.image_subset _ <| compl_subset_compl.mpr <| Set.image_subset _ h
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    δ' : Type u_5
    ι : Sort uι
    s t u : Set α
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    inst✝ : MeasurableSpace γ
    f✝ : α → β
    g✝ : β → α
    f : α → β
    g : β → α
    hf : MeasurableEmbedding f
    hg : MeasurableEmbedding g
    F : Set α → Set α := fun A => HasCompl.compl (Set.image g (HasCompl.compl (Set …
    Fmono : ∀ {A B : Set α}, HasSubset.Subset A B → HasSubset.Subset (F A) (F B)
    ⊢ PSigma fun A => And (MeasurableSet A) (Eq (F A) A)
  -/
  let X : ℕ → Set α := fun n => F^[n] univ
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    δ' : Type u_5
    ι : Sort uι
    s t u : Set α
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    inst✝ : MeasurableSpace γ
    f✝ : α → β
    g✝ : β → α
    f : α → β
    g : β → α
    hf : MeasurableEmbedding f
    hg : MeasurableEmbedding g
    F : Set α → Set α := fun A => HasCompl.compl (Set.image g (HasCompl.compl (Set …
    Fmono : ∀ {A B : Set α}, HasSubset.Subset A B → HasSubset.Subset (F A) (F B)
    X : Nat → Set α := fun n => Nat.iterate F n Set.univ
    ⊢ PSigma fun A => And (MeasurableSet A) (Eq (F A) A)
  -/
  refine ⟨iInter X, ?_, ?_⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      δ' : Type u_5
      ι : Sort uι
      s t u : Set α
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      inst✝ : MeasurableSpace γ
      f✝ : α → β
      g✝ : β → α
      f : α → β
      g : β → α
      hf : MeasurableEmbedding f
      hg : MeasurableEmbedding g
      F : Set α → Set α := fun A => HasCompl.compl (Set.image g (HasCompl.compl (Set …
      Fmono : ∀ {A B : Set α}, HasSubset.Subset A B → HasSubset.Subset (F A) (F B)
      X : Nat → Set α := fun n => Nat.iterate F n Set.univ
      ⊢ MeasurableSet (Set.iInter X)
    -/
  · apply MeasurableSet.iInter
    /-
      case refine_1.h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      δ' : Type u_5
      ι : Sort uι
      s t u : Set α
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      inst✝ : MeasurableSpace γ
      f✝ : α → β
      g✝ : β → α
      f : α → β
      g : β → α
      hf : MeasurableEmbedding f
      hg : MeasurableEmbedding g
      F : Set α → Set α := fun A => HasCompl.compl (Set.image g (HasCompl.compl (Set …
      Fmono : ∀ {A B : Set α}, HasSubset.Subset A B → HasSubset.Subset (F A) (F B)
      X : Nat → Set α := fun n => Nat.iterate F n Set.univ
      ⊢ ∀ (b : Nat), MeasurableSet (Nat.iterate F b Set.univ)
    -/
    intro n
    /-
      case refine_1.h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      δ' : Type u_5
      ι : Sort uι
      s t u : Set α
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      inst✝ : MeasurableSpace γ
      f✝ : α → β
      g✝ : β → α
      f : α → β
      g : β → α
      hf : MeasurableEmbedding f
      hg : MeasurableEmbedding g
      F : Set α → Set α := fun A => HasCompl.compl (Set.image g (HasCompl.compl (Set …
      Fmono : ∀ {A B : Set α}, HasSubset.Subset A B → HasSubset.Subset (F A) (F B)
      X : Nat → Set α := fun n => Nat.iterate F n Set.univ
      n : Nat
      ⊢ MeasurableSet (Nat.iterate F n Set.univ)
    -/
    induction' n with n ih
      /-
        case refine_1.h.zero
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        δ' : Type u_5
        ι : Sort uι
        s t u : Set α
        inst✝² : MeasurableSpace α
        inst✝¹ : MeasurableSpace β
        inst✝ : MeasurableSpace γ
        f✝ : α → β
        g✝ : β → α
        f : α → β
        g : β → α
        hf : MeasurableEmbedding f
        hg : MeasurableEmbedding g
        F : Set α → Set α := fun A => HasCompl.compl (Set.image g (HasCompl.compl (Set …
        Fmono : ∀ {A B : Set α}, HasSubset.Subset A B → HasSubset.Subset (F A) (F B)
        X : Nat → Set α := fun n => Nat.iterate F n Set.univ
        ⊢ MeasurableSet (Nat.iterate F 0 Set.univ)
      -/
    · exact MeasurableSet.univ
      /-
        🎉 no goals
      -/
    /-
      case refine_1.h.succ
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      δ' : Type u_5
      ι : Sort uι
      s t u : Set α
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      inst✝ : MeasurableSpace γ
      f✝ : α → β
      g✝ : β → α
      f : α → β
      g : β → α
      hf : MeasurableEmbedding f
      hg : MeasurableEmbedding g
      F : Set α → Set α := fun A => HasCompl.compl (Set.image g (HasCompl.compl (Set …
      Fmono : ∀ {A B : Set α}, HasSubset.Subset A B → HasSubset.Subset (F A) (F B)
      X : Nat → Set α := fun n => Nat.iterate F n Set.univ
      n : Nat
      ih : MeasurableSet (Nat.iterate F n Set.univ)
      ⊢ MeasurableSet (Nat.iterate F (HAdd.hAdd n 1) Set.univ)
    -/
    rw [Function.iterate_succ', Function.comp_apply]
    /-
      case refine_1.h.succ
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      δ' : Type u_5
      ι : Sort uι
      s t u : Set α
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      inst✝ : MeasurableSpace γ
      f✝ : α → β
      g✝ : β → α
      f : α → β
      g : β → α
      hf : MeasurableEmbedding f
      hg : MeasurableEmbedding g
      F : Set α → Set α := fun A => HasCompl.compl (Set.image g (HasCompl.compl (Set …
      Fmono : ∀ {A B : Set α}, HasSubset.Subset A B → HasSubset.Subset (F A) (F B)
      X : Nat → Set α := fun n => Nat.iterate F n Set.univ
      n : Nat
      ih : MeasurableSet (Nat.iterate F n Set.univ)
      ⊢ MeasurableSet (F (Nat.iterate F n Set.univ))
    -/
    exact (hg.measurableSet_image' (hf.measurableSet_image' ih).compl).compl
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    δ' : Type u_5
    ι : Sort uι
    s t u : Set α
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    inst✝ : MeasurableSpace γ
    f✝ : α → β
    g✝ : β → α
    f : α → β
    g : β → α
    hf : MeasurableEmbedding f
    hg : MeasurableEmbedding g
    F : Set α → Set α := fun A => HasCompl.compl (Set.image g (HasCompl.compl (Set …
    Fmono : ∀ {A B : Set α}, HasSubset.Subset A B → HasSubset.Subset (F A) (F B)
    X : Nat → Set α := fun n => Nat.iterate F n Set.univ
    ⊢ Eq (F (Set.iInter X)) (Set.iInter X)
  -/
  apply subset_antisymm
    /-
      case refine_2.a
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      δ' : Type u_5
      ι : Sort uι
      s t u : Set α
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      inst✝ : MeasurableSpace γ
      f✝ : α → β
      g✝ : β → α
      f : α → β
      g : β → α
      hf : MeasurableEmbedding f
      hg : MeasurableEmbedding g
      F : Set α → Set α := fun A => HasCompl.compl (Set.image g (HasCompl.compl (Set …
      Fmono : ∀ {A B : Set α}, HasSubset.Subset A B → HasSubset.Subset (F A) (F B)
      X : Nat → Set α := fun n => Nat.iterate F n Set.univ
      ⊢ HasSubset.Subset (F (Set.iInter X)) (Set.iInter X)
    -/
  · apply subset_iInter
    /-
      case refine_2.a.h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      δ' : Type u_5
      ι : Sort uι
      s t u : Set α
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      inst✝ : MeasurableSpace γ
      f✝ : α → β
      g✝ : β → α
      f : α → β
      g : β → α
      hf : MeasurableEmbedding f
      hg : MeasurableEmbedding g
      F : Set α → Set α := fun A => HasCompl.compl (Set.image g (HasCompl.compl (Set …
      Fmono : ∀ {A B : Set α}, HasSubset.Subset A B → HasSubset.Subset (F A) (F B)
      X : Nat → Set α := fun n => Nat.iterate F n Set.univ
      ⊢ ∀ (i : Nat), HasSubset.Subset (F (Set.iInter X)) (Nat.iterate F i Set.univ)
    -/
    intro n
    /-
      case refine_2.a.h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      δ' : Type u_5
      ι : Sort uι
      s t u : Set α
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      inst✝ : MeasurableSpace γ
      f✝ : α → β
      g✝ : β → α
      f : α → β
      g : β → α
      hf : MeasurableEmbedding f
      hg : MeasurableEmbedding g
      F : Set α → Set α := fun A => HasCompl.compl (Set.image g (HasCompl.compl (Set …
      Fmono : ∀ {A B : Set α}, HasSubset.Subset A B → HasSubset.Subset (F A) (F B)
      X : Nat → Set α := fun n => Nat.iterate F n Set.univ
      n : Nat
      ⊢ HasSubset.Subset (F (Set.iInter X)) (Nat.iterate F n Set.univ)
    -/
    cases n
      /-
        case refine_2.a.h.zero
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        δ' : Type u_5
        ι : Sort uι
        s t u : Set α
        inst✝² : MeasurableSpace α
        inst✝¹ : MeasurableSpace β
        inst✝ : MeasurableSpace γ
        f✝ : α → β
        g✝ : β → α
        f : α → β
        g : β → α
        hf : MeasurableEmbedding f
        hg : MeasurableEmbedding g
        F : Set α → Set α := fun A => HasCompl.compl (Set.image g (HasCompl.compl (Set …
        Fmono : ∀ {A B : Set α}, HasSubset.Subset A B → HasSubset.Subset (F A) (F B)
        X : Nat → Set α := fun n => Nat.iterate F n Set.univ
        ⊢ HasSubset.Subset (F (Set.iInter X)) (Nat.iterate F 0 Set.univ)
      -/
    · exact subset_univ _
      /-
        🎉 no goals
      -/
    /-
      case refine_2.a.h.succ
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      δ' : Type u_5
      ι : Sort uι
      s t u : Set α
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      inst✝ : MeasurableSpace γ
      f✝ : α → β
      g✝ : β → α
      f : α → β
      g : β → α
      hf : MeasurableEmbedding f
      hg : MeasurableEmbedding g
      F : Set α → Set α := fun A => HasCompl.compl (Set.image g (HasCompl.compl (Set …
      Fmono : ∀ {A B : Set α}, HasSubset.Subset A B → HasSubset.Subset (F A) (F B)
      X : Nat → Set α := fun n => Nat.iterate F n Set.univ
      n✝ : Nat
      ⊢ HasSubset.Subset (F (Set.iInter X)) (Nat.iterate F (HAdd.hAdd n✝ 1) Set.univ)
    -/
    rw [Function.iterate_succ', Function.comp_apply]
    /-
      case refine_2.a.h.succ
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      δ' : Type u_5
      ι : Sort uι
      s t u : Set α
      inst✝² : MeasurableSpace α
      inst✝¹ : MeasurableSpace β
      inst✝ : MeasurableSpace γ
      f✝ : α → β
      g✝ : β → α
      f : α → β
      g : β → α
      hf : MeasurableEmbedding f
      hg : MeasurableEmbedding g
      F : Set α → Set α := fun A => HasCompl.compl (Set.image g (HasCompl.compl (Set …
      Fmono : ∀ {A B : Set α}, HasSubset.Subset A B → HasSubset.Subset (F A) (F B)
      X : Nat → Set α := fun n => Nat.iterate F n Set.univ
      n✝ : Nat
      ⊢ HasSubset.Subset (F (Set.iInter X)) (F (Nat.iterate F n✝ Set.univ))
    -/
    exact Fmono (iInter_subset _ _)
    /-
      🎉 no goals
    -/
  /-
    case refine_2.a
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    δ' : Type u_5
    ι : Sort uι
    s t u : Set α
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    inst✝ : MeasurableSpace γ
    f✝ : α → β
    g✝ : β → α
    f : α → β
    g : β → α
    hf : MeasurableEmbedding f
    hg : MeasurableEmbedding g
    F : Set α → Set α := fun A => HasCompl.compl (Set.image g (HasCompl.compl (Set …
    Fmono : ∀ {A B : Set α}, HasSubset.Subset A B → HasSubset.Subset (F A) (F B)
    X : Nat → Set α := fun n => Nat.iterate F n Set.univ
    ⊢ HasSubset.Subset (Set.iInter X) (F (Set.iInter X))
  -/
  rintro x hx ⟨y, hy, rfl⟩
  /-
    case refine_2.a.intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    δ' : Type u_5
    ι : Sort uι
    s t u : Set α
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    inst✝ : MeasurableSpace γ
    f✝ : α → β
    g✝ : β → α
    f : α → β
    g : β → α
    hf : MeasurableEmbedding f
    hg : MeasurableEmbedding g
    F : Set α → Set α := fun A => HasCompl.compl (Set.image g (HasCompl.compl (Set …
    Fmono : ∀ {A B : Set α}, HasSubset.Subset A B → HasSubset.Subset (F A) (F B)
    X : Nat → Set α := fun n => Nat.iterate F n Set.univ
    y : β
    hy : Membership.mem (HasCompl.compl (Set.image f (Set.iInter X))) y
    hx : Membership.mem (Set.iInter X) (g y)
    ⊢ False
  -/
  rw [mem_iInter] at hx
  /-
    case refine_2.a.intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    δ' : Type u_5
    ι : Sort uι
    s t u : Set α
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    inst✝ : MeasurableSpace γ
    f✝ : α → β
    g✝ : β → α
    f : α → β
    g : β → α
    hf : MeasurableEmbedding f
    hg : MeasurableEmbedding g
    F : Set α → Set α := fun A => HasCompl.compl (Set.image g (HasCompl.compl (Set …
    Fmono : ∀ {A B : Set α}, HasSubset.Subset A B → HasSubset.Subset (F A) (F B)
    X : Nat → Set α := fun n => Nat.iterate F n Set.univ
    y : β
    hy : Membership.mem (HasCompl.compl (Set.image f (Set.iInter X))) y
    hx : ∀ (i : Nat), Membership.mem (Nat.iterate F i Set.univ) (g y)
    ⊢ False
  -/
  apply hy
  /-
    case refine_2.a.intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    δ' : Type u_5
    ι : Sort uι
    s t u : Set α
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    inst✝ : MeasurableSpace γ
    f✝ : α → β
    g✝ : β → α
    f : α → β
    g : β → α
    hf : MeasurableEmbedding f
    hg : MeasurableEmbedding g
    F : Set α → Set α := fun A => HasCompl.compl (Set.image g (HasCompl.compl (Set …
    Fmono : ∀ {A B : Set α}, HasSubset.Subset A B → HasSubset.Subset (F A) (F B)
    X : Nat → Set α := fun n => Nat.iterate F n Set.univ
    y : β
    hy : Membership.mem (HasCompl.compl (Set.image f (Set.iInter X))) y
    hx : ∀ (i : Nat), Membership.mem (Nat.iterate F i Set.univ) (g y)
    ⊢ Membership.mem (Set.image f (Set.iInter X)) y
  -/
  rw [hf.injective.injOn.image_iInter_eq]
  /-
    case refine_2.a.intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    δ' : Type u_5
    ι : Sort uι
    s t u : Set α
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    inst✝ : MeasurableSpace γ
    f✝ : α → β
    g✝ : β → α
    f : α → β
    g : β → α
    hf : MeasurableEmbedding f
    hg : MeasurableEmbedding g
    F : Set α → Set α := fun A => HasCompl.compl (Set.image g (HasCompl.compl (Set …
    Fmono : ∀ {A B : Set α}, HasSubset.Subset A B → HasSubset.Subset (F A) (F B)
    X : Nat → Set α := fun n => Nat.iterate F n Set.univ
    y : β
    hy : Membership.mem (HasCompl.compl (Set.image f (Set.iInter X))) y
    hx : ∀ (i : Nat), Membership.mem (Nat.iterate F i Set.univ) (g y)
    ⊢ Membership.mem (Set.iInter fun i => Set.image f (Nat.iterate F i Set.univ)) y
  -/
  rw [mem_iInter]
  /-
    case refine_2.a.intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    δ' : Type u_5
    ι : Sort uι
    s t u : Set α
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    inst✝ : MeasurableSpace γ
    f✝ : α → β
    g✝ : β → α
    f : α → β
    g : β → α
    hf : MeasurableEmbedding f
    hg : MeasurableEmbedding g
    F : Set α → Set α := fun A => HasCompl.compl (Set.image g (HasCompl.compl (Set …
    Fmono : ∀ {A B : Set α}, HasSubset.Subset A B → HasSubset.Subset (F A) (F B)
    X : Nat → Set α := fun n => Nat.iterate F n Set.univ
    y : β
    hy : Membership.mem (HasCompl.compl (Set.image f (Set.iInter X))) y
    hx : ∀ (i : Nat), Membership.mem (Nat.iterate F i Set.univ) (g y)
    ⊢ ∀ (i : Nat), Membership.mem (Set.image f (Nat.iterate F i Set.univ)) y
  -/
  intro n
  /-
    case refine_2.a.intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    δ' : Type u_5
    ι : Sort uι
    s t u : Set α
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    inst✝ : MeasurableSpace γ
    f✝ : α → β
    g✝ : β → α
    f : α → β
    g : β → α
    hf : MeasurableEmbedding f
    hg : MeasurableEmbedding g
    F : Set α → Set α := fun A => HasCompl.compl (Set.image g (HasCompl.compl (Set …
    Fmono : ∀ {A B : Set α}, HasSubset.Subset A B → HasSubset.Subset (F A) (F B)
    X : Nat → Set α := fun n => Nat.iterate F n Set.univ
    y : β
    hy : Membership.mem (HasCompl.compl (Set.image f (Set.iInter X))) y
    hx : ∀ (i : Nat), Membership.mem (Nat.iterate F i Set.univ) (g y)
    n : Nat
    ⊢ Membership.mem (Set.image f (Nat.iterate F n Set.univ)) y
  -/
  specialize hx n.succ
  /-
    case refine_2.a.intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    δ' : Type u_5
    ι : Sort uι
    s t u : Set α
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    inst✝ : MeasurableSpace γ
    f✝ : α → β
    g✝ : β → α
    f : α → β
    g : β → α
    hf : MeasurableEmbedding f
    hg : MeasurableEmbedding g
    F : Set α → Set α := fun A => HasCompl.compl (Set.image g (HasCompl.compl (Set …
    Fmono : ∀ {A B : Set α}, HasSubset.Subset A B → HasSubset.Subset (F A) (F B)
    X : Nat → Set α := fun n => Nat.iterate F n Set.univ
    y : β
    hy : Membership.mem (HasCompl.compl (Set.image f (Set.iInter X))) y
    n : Nat
    hx : Membership.mem (Nat.iterate F n.succ Set.univ) (g y)
    ⊢ Membership.mem (Set.image f (Nat.iterate F n Set.univ)) y
  -/
  rw [Function.iterate_succ', Function.comp_apply] at hx
  /-
    case refine_2.a.intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    δ' : Type u_5
    ι : Sort uι
    s t u : Set α
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    inst✝ : MeasurableSpace γ
    f✝ : α → β
    g✝ : β → α
    f : α → β
    g : β → α
    hf : MeasurableEmbedding f
    hg : MeasurableEmbedding g
    F : Set α → Set α := fun A => HasCompl.compl (Set.image g (HasCompl.compl (Set …
    Fmono : ∀ {A B : Set α}, HasSubset.Subset A B → HasSubset.Subset (F A) (F B)
    X : Nat → Set α := fun n => Nat.iterate F n Set.univ
    y : β
    hy : Membership.mem (HasCompl.compl (Set.image f (Set.iInter X))) y
    n : Nat
    hx : Membership.mem (F (Nat.iterate F n Set.univ)) (g y)
    ⊢ Membership.mem (Set.image f (Nat.iterate F n Set.univ)) y
  -/
  by_contra h
  /-
    case refine_2.a.intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    δ' : Type u_5
    ι : Sort uι
    s t u : Set α
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    inst✝ : MeasurableSpace γ
    f✝ : α → β
    g✝ : β → α
    f : α → β
    g : β → α
    hf : MeasurableEmbedding f
    hg : MeasurableEmbedding g
    F : Set α → Set α := fun A => HasCompl.compl (Set.image g (HasCompl.compl (Set …
    Fmono : ∀ {A B : Set α}, HasSubset.Subset A B → HasSubset.Subset (F A) (F B)
    X : Nat → Set α := fun n => Nat.iterate F n Set.univ
    y : β
    hy : Membership.mem (HasCompl.compl (Set.image f (Set.iInter X))) y
    n : Nat
    hx : Membership.mem (F (Nat.iterate F n Set.univ)) (g y)
    h : Not (Membership.mem (Set.image f (Nat.iterate F n Set.univ)) y)
    ⊢ False
  -/
  apply hx
  /-
    case refine_2.a.intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    δ' : Type u_5
    ι : Sort uι
    s t u : Set α
    inst✝² : MeasurableSpace α
    inst✝¹ : MeasurableSpace β
    inst✝ : MeasurableSpace γ
    f✝ : α → β
    g✝ : β → α
    f : α → β
    g : β → α
    hf : MeasurableEmbedding f
    hg : MeasurableEmbedding g
    F : Set α → Set α := fun A => HasCompl.compl (Set.image g (HasCompl.compl (Set …
    Fmono : ∀ {A B : Set α}, HasSubset.Subset A B → HasSubset.Subset (F A) (F B)
    X : Nat → Set α := fun n => Nat.iterate F n Set.univ
    y : β
    hy : Membership.mem (HasCompl.compl (Set.image f (Set.iInter X))) y
    n : Nat
    hx : Membership.mem (F (Nat.iterate F n Set.univ)) (g y)
    h : Not (Membership.mem (Set.image f (Nat.iterate F n Set.univ)) y)
    ⊢ Membership.mem (Set.image g (HasCompl.compl (Set.image f (Nat.iterate F n Se …
  -/
  exact ⟨y, h, rfl⟩
  /-
    🎉 no goals
  -/


theorem MeasurableSpace.comap_compl {m' : MeasurableSpace β} [BooleanAlgebra β]
    (h : Measurable (compl : β → β)) (f : α → β) :
    MeasurableSpace.comap (fun a => (f a)ᶜ) inferInstance =
      MeasurableSpace.comap f inferInstance := by
  /-
    α : Type u_1
    β : Type u_2
    m' : MeasurableSpace β
    inst✝ : BooleanAlgebra β
    h : Measurable HasCompl.compl
    f : α → β
    ⊢ Eq (MeasurableSpace.comap (fun a => HasCompl.compl (f a)) inferInstance) (Me …
  -/
  rw [← Function.comp_def, ← MeasurableSpace.comap_comp]
  /-
    α : Type u_1
    β : Type u_2
    m' : MeasurableSpace β
    inst✝ : BooleanAlgebra β
    h : Measurable HasCompl.compl
    f : α → β
    ⊢ Eq (MeasurableSpace.comap f (MeasurableSpace.comap HasCompl.compl inferInsta …
  -/
  congr
  /-
    case e_m
    α : Type u_1
    β : Type u_2
    m' : MeasurableSpace β
    inst✝ : BooleanAlgebra β
    h : Measurable HasCompl.compl
    f : α → β
    ⊢ Eq (MeasurableSpace.comap HasCompl.compl inferInstance) inferInstance
  -/
  exact (MeasurableEquiv.ofInvolutive _ compl_involutive h).measurableEmbedding.comap_eq
  /-
    🎉 no goals
  -/


@[simp] theorem MeasurableSpace.comap_not (p : α → Prop) :
    MeasurableSpace.comap (fun a ↦ ¬p a) inferInstance = MeasurableSpace.comap p inferInstance :=
  MeasurableSpace.comap_compl (fun _ _ ↦ measurableSet_top) _

