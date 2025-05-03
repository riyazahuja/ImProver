open Classical in
/-- Pullback of a `Measure` as a linear map. If `f` sends each measurable set to a measurable
set, then for each measurable set `s` we have `comapₗ f μ s = μ (f '' s)`.

Note that if `f` is not injective, this definition assigns `Set.univ` measure zero.

If the linearity is not needed, please use `comap` instead, which works for a larger class of
functions. `comapₗ` is an auxiliary definition and most lemmas deal with comap. -/
def comapₗ [MeasurableSpace α] [MeasurableSpace β] (f : α → β) : Measure β →ₗ[ℝ≥0∞] Measure α :=
  if hf : Injective f ∧ ∀ s, MeasurableSet s → MeasurableSet (f '' s) then
    liftLinear (OuterMeasure.comap f) fun μ s hs t => by
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        s✝ : Set α
        inst✝¹ : MeasurableSpace α
        inst✝ : MeasurableSpace β
        f : α → β
        hf : And (Function.Injective f) (∀ (s : Set α), MeasurableSet s → MeasurableSe …
        μ : MeasureTheory.Measure β
        s : Set α
        hs : MeasurableSet s
        t : Set α
        ⊢ Eq (((MeasureTheory.OuterMeasure.comap f) μ.toOuterMeasure) t) (HAdd.hAdd (( …
      -/
      simp only [OuterMeasure.comap_apply, image_inter hf.1, image_diff hf.1]
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        s✝ : Set α
        inst✝¹ : MeasurableSpace α
        inst✝ : MeasurableSpace β
        f : α → β
        hf : And (Function.Injective f) (∀ (s : Set α), MeasurableSet s → MeasurableSe …
        μ : MeasureTheory.Measure β
        s : Set α
        hs : MeasurableSet s
        t : Set α
        ⊢ Eq (μ.toOuterMeasure (Set.image f t)) (HAdd.hAdd (μ.toOuterMeasure (Inter.in …
      -/
      apply le_toOuterMeasure_caratheodory
      /-
        case a
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        s✝ : Set α
        inst✝¹ : MeasurableSpace α
        inst✝ : MeasurableSpace β
        f : α → β
        hf : And (Function.Injective f) (∀ (s : Set α), MeasurableSet s → MeasurableSe …
        μ : MeasureTheory.Measure β
        s : Set α
        hs : MeasurableSet s
        t : Set α
        ⊢ MeasurableSet (Set.image f s)
      -/
      exact hf.2 s hs
      /-
        🎉 no goals
      -/
  else 0


theorem comapₗ_apply {_ : MeasurableSpace α} {_ : MeasurableSpace β} (f : α → β)
    (hfi : Injective f) (hf : ∀ s, MeasurableSet s → MeasurableSet (f '' s)) (μ : Measure β)
    (hs : MeasurableSet s) : comapₗ f μ s = μ (f '' s) := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    x✝¹ : MeasurableSpace α
    x✝ : MeasurableSpace β
    f : α → β
    hfi : Function.Injective f
    hf : ∀ (s : Set α), MeasurableSet s → MeasurableSet (Set.image f s)
    μ : MeasureTheory.Measure β
    hs : MeasurableSet s
    ⊢ Eq (((MeasureTheory.Measure.comapₗ f) μ) s) (μ (Set.image f s))
  -/
  rw [comapₗ, dif_pos, liftLinear_apply _ hs, OuterMeasure.comap_apply, coe_toOuterMeasure]
  /-
    case hc
    α : Type u_1
    β : Type u_2
    s : Set α
    x✝¹ : MeasurableSpace α
    x✝ : MeasurableSpace β
    f : α → β
    hfi : Function.Injective f
    hf : ∀ (s : Set α), MeasurableSet s → MeasurableSet (Set.image f s)
    μ : MeasureTheory.Measure β
    hs : MeasurableSet s
    ⊢ And (Function.Injective f) (∀ (s : Set α), MeasurableSet s → MeasurableSet ( …
  -/
  exact ⟨hfi, hf⟩
  /-
    🎉 no goals
  -/


open Classical in
/-- Pullback of a `Measure`. If `f` sends each measurable set to a null-measurable set,
then for each measurable set `s` we have `comap f μ s = μ (f '' s)`.

Note that if `f` is not injective, this definition assigns `Set.univ` measure zero. -/
def comap [MeasurableSpace α] [MeasurableSpace β] (f : α → β) (μ : Measure β) : Measure α :=
  if hf : Injective f ∧ ∀ s, MeasurableSet s → NullMeasurableSet (f '' s) μ then
    (OuterMeasure.comap f μ.toOuterMeasure).toMeasure fun s hs t => by
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        s✝ : Set α
        inst✝¹ : MeasurableSpace α
        inst✝ : MeasurableSpace β
        f : α → β
        μ : MeasureTheory.Measure β
        hf : And (Function.Injective f) (∀ (s : Set α), MeasurableSet s → MeasureTheor …
        s : Set α
        hs : MeasurableSet s
        t : Set α
        ⊢ Eq (((MeasureTheory.OuterMeasure.comap f) μ.toOuterMeasure) t) (HAdd.hAdd (( …
      -/
      simp only [OuterMeasure.comap_apply, image_inter hf.1, image_diff hf.1]
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        s✝ : Set α
        inst✝¹ : MeasurableSpace α
        inst✝ : MeasurableSpace β
        f : α → β
        μ : MeasureTheory.Measure β
        hf : And (Function.Injective f) (∀ (s : Set α), MeasurableSet s → MeasureTheor …
        s : Set α
        hs : MeasurableSet s
        t : Set α
        ⊢ Eq (μ.toOuterMeasure (Set.image f t)) (HAdd.hAdd (μ.toOuterMeasure (Inter.in …
      -/
      exact (measure_inter_add_diff₀ _ (hf.2 s hs)).symm
      /-
        🎉 no goals
      -/
  else 0


theorem comap_apply₀ (f : α → β) (μ : Measure β) (hfi : Injective f)
    (hf : ∀ s, MeasurableSet s → NullMeasurableSet (f '' s) μ)
    (hs : NullMeasurableSet s (comap f μ)) : comap f μ s = μ (f '' s) := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β
    μ : MeasureTheory.Measure β
    hfi : Function.Injective f
    hf : ∀ (s : Set α), MeasurableSet s → MeasureTheory.NullMeasurableSet (Set.ima …
    hs : MeasureTheory.NullMeasurableSet s (MeasureTheory.Measure.comap f μ)
    ⊢ Eq ((MeasureTheory.Measure.comap f μ) s) (μ (Set.image f s))
  -/
  rw [comap, dif_pos (And.intro hfi hf)] at hs ⊢
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β
    μ : MeasureTheory.Measure β
    hfi : Function.Injective f
    hf : ∀ (s : Set α), MeasurableSet s → MeasureTheory.NullMeasurableSet (Set.ima …
    hs : MeasureTheory.NullMeasurableSet s (((MeasureTheory.OuterMeasure.comap f)  …
    ⊢ Eq ((((MeasureTheory.OuterMeasure.comap f) μ.toOuterMeasure).toMeasure ⋯) s) …
  -/
  rw [toMeasure_apply₀ _ _ hs, OuterMeasure.comap_apply, coe_toOuterMeasure]
  /-
    🎉 no goals
  -/


theorem le_comap_apply (f : α → β) (μ : Measure β) (hfi : Injective f)
    (hf : ∀ s, MeasurableSet s → NullMeasurableSet (f '' s) μ) (s : Set α) :
    μ (f '' s) ≤ comap f μ s := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β
    μ : MeasureTheory.Measure β
    hfi : Function.Injective f
    hf : ∀ (s : Set α), MeasurableSet s → MeasureTheory.NullMeasurableSet (Set.ima …
    s : Set α
    ⊢ LE.le (μ (Set.image f s)) ((MeasureTheory.Measure.comap f μ) s)
  -/
  rw [comap, dif_pos (And.intro hfi hf)]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β
    μ : MeasureTheory.Measure β
    hfi : Function.Injective f
    hf : ∀ (s : Set α), MeasurableSet s → MeasureTheory.NullMeasurableSet (Set.ima …
    s : Set α
    ⊢ LE.le (μ (Set.image f s)) ((((MeasureTheory.OuterMeasure.comap f) μ.toOuterM …
  -/
  exact le_toMeasure_apply _ _ _
  /-
    🎉 no goals
  -/


theorem comap_apply (f : α → β) (hfi : Injective f)
    (hf : ∀ s, MeasurableSet s → MeasurableSet (f '' s)) (μ : Measure β) (hs : MeasurableSet s) :
    comap f μ s = μ (f '' s) :=
  comap_apply₀ f μ hfi (fun s hs => (hf s hs).nullMeasurableSet) hs.nullMeasurableSet


theorem comapₗ_eq_comap (f : α → β) (hfi : Injective f)
    (hf : ∀ s, MeasurableSet s → MeasurableSet (f '' s)) (μ : Measure β) (hs : MeasurableSet s) :
    comapₗ f μ s = comap f μ s :=
  (comapₗ_apply f hfi hf μ hs).trans (comap_apply f hfi hf μ hs).symm


theorem measure_image_eq_zero_of_comap_eq_zero (f : α → β) (μ : Measure β) (hfi : Injective f)
    (hf : ∀ s, MeasurableSet s → NullMeasurableSet (f '' s) μ) {s : Set α} (hs : comap f μ s = 0) :
    μ (f '' s) = 0 :=
  le_antisymm ((le_comap_apply f μ hfi hf s).trans hs.le) (zero_le _)


theorem ae_eq_image_of_ae_eq_comap (f : α → β) (μ : Measure β) (hfi : Injective f)
    (hf : ∀ s, MeasurableSet s → NullMeasurableSet (f '' s) μ)
    {s t : Set α} (hst : s =ᵐ[comap f μ] t) : f '' s =ᵐ[μ] f '' t := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β
    μ : MeasureTheory.Measure β
    hfi : Function.Injective f
    hf : ∀ (s : Set α), MeasurableSet s → MeasureTheory.NullMeasurableSet (Set.ima …
    s t : Set α
    hst : (MeasureTheory.ae (MeasureTheory.Measure.comap f μ)).EventuallyEq s t
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Set.image f s) (Set.image f t)
  -/
  rw [EventuallyEq, ae_iff] at hst ⊢
  have h_eq_α : { a : α | ¬s a = t a } = s \ t ∪ t \ s := by
    ext1 x
    simp only [eq_iff_iff, mem_setOf_eq, mem_union, mem_diff]
    tauto
  have h_eq_β : { a : β | ¬(f '' s) a = (f '' t) a } = f '' s \ f '' t ∪ f '' t \ f '' s := by
    ext1 x
    simp only [eq_iff_iff, mem_setOf_eq, mem_union, mem_diff]
    tauto
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β
    μ : MeasureTheory.Measure β
    hfi : Function.Injective f
    hf : ∀ (s : Set α), MeasurableSet s → MeasureTheory.NullMeasurableSet (Set.ima …
    s t : Set α
    hst : Eq ((MeasureTheory.Measure.comap f μ) (setOf fun a => Not (Eq (s a) (t a …
    h_eq_α : Eq (setOf fun a => Not (Eq (s a) (t a))) (Union.union (SDiff.sdiff s  …
    h_eq_β : Eq (setOf fun a => Not (Eq (Set.image f s a) (Set.image f t a))) (Uni …
    ⊢ Eq (μ (setOf fun a => Not (Eq (Set.image f s a) (Set.image f t a)))) 0
  -/
  rw [← Set.image_diff hfi, ← Set.image_diff hfi, ← Set.image_union] at h_eq_β
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β
    μ : MeasureTheory.Measure β
    hfi : Function.Injective f
    hf : ∀ (s : Set α), MeasurableSet s → MeasureTheory.NullMeasurableSet (Set.ima …
    s t : Set α
    hst : Eq ((MeasureTheory.Measure.comap f μ) (setOf fun a => Not (Eq (s a) (t a …
    h_eq_α : Eq (setOf fun a => Not (Eq (s a) (t a))) (Union.union (SDiff.sdiff s  …
    h_eq_β : Eq (setOf fun a => Not (Eq (Set.image f s a) (Set.image f t a))) (Set …
    ⊢ Eq (μ (setOf fun a => Not (Eq (Set.image f s a) (Set.image f t a)))) 0
  -/
  rw [h_eq_β]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β
    μ : MeasureTheory.Measure β
    hfi : Function.Injective f
    hf : ∀ (s : Set α), MeasurableSet s → MeasureTheory.NullMeasurableSet (Set.ima …
    s t : Set α
    hst : Eq ((MeasureTheory.Measure.comap f μ) (setOf fun a => Not (Eq (s a) (t a …
    h_eq_α : Eq (setOf fun a => Not (Eq (s a) (t a))) (Union.union (SDiff.sdiff s  …
    h_eq_β : Eq (setOf fun a => Not (Eq (Set.image f s a) (Set.image f t a))) (Set …
    ⊢ Eq (μ (Set.image f (Union.union (SDiff.sdiff s t) (SDiff.sdiff t s)))) 0
  -/
  rw [h_eq_α] at hst
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β
    μ : MeasureTheory.Measure β
    hfi : Function.Injective f
    hf : ∀ (s : Set α), MeasurableSet s → MeasureTheory.NullMeasurableSet (Set.ima …
    s t : Set α
    hst : Eq ((MeasureTheory.Measure.comap f μ) (Union.union (SDiff.sdiff s t) (SD …
    h_eq_α : Eq (setOf fun a => Not (Eq (s a) (t a))) (Union.union (SDiff.sdiff s  …
    h_eq_β : Eq (setOf fun a => Not (Eq (Set.image f s a) (Set.image f t a))) (Set …
    ⊢ Eq (μ (Set.image f (Union.union (SDiff.sdiff s t) (SDiff.sdiff t s)))) 0
  -/
  exact measure_image_eq_zero_of_comap_eq_zero f μ hfi hf hst
  /-
    🎉 no goals
  -/


theorem NullMeasurableSet.image (f : α → β) (μ : Measure β) (hfi : Injective f)
    (hf : ∀ s, MeasurableSet s → NullMeasurableSet (f '' s) μ)
    (hs : NullMeasurableSet s (μ.comap f)) : NullMeasurableSet (f '' s) μ := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β
    μ : MeasureTheory.Measure β
    hfi : Function.Injective f
    hf : ∀ (s : Set α), MeasurableSet s → MeasureTheory.NullMeasurableSet (Set.ima …
    hs : MeasureTheory.NullMeasurableSet s (MeasureTheory.Measure.comap f μ)
    ⊢ MeasureTheory.NullMeasurableSet (Set.image f s) μ
  -/
  refine ⟨toMeasurable μ (f '' toMeasurable (μ.comap f) s), measurableSet_toMeasurable _ _, ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β
    μ : MeasureTheory.Measure β
    hfi : Function.Injective f
    hf : ∀ (s : Set α), MeasurableSet s → MeasureTheory.NullMeasurableSet (Set.ima …
    hs : MeasureTheory.NullMeasurableSet s (MeasureTheory.Measure.comap f μ)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Set.image f s) (MeasureTheory.toMeasurabl …
  -/
  refine EventuallyEq.trans ?_ (NullMeasurableSet.toMeasurable_ae_eq ?_).symm
  /-
    case refine_1
    α : Type u_1
    β : Type u_2
    s : Set α
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β
    μ : MeasureTheory.Measure β
    hfi : Function.Injective f
    hf : ∀ (s : Set α), MeasurableSet s → MeasureTheory.NullMeasurableSet (Set.ima …
    hs : MeasureTheory.NullMeasurableSet s (MeasureTheory.Measure.comap f μ)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Set.image f s) (Set.image f (MeasureTheor …
  -/
  swap
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      s : Set α
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      f : α → β
      μ : MeasureTheory.Measure β
      hfi : Function.Injective f
      hf : ∀ (s : Set α), MeasurableSet s → MeasureTheory.NullMeasurableSet (Set.ima …
      hs : MeasureTheory.NullMeasurableSet s (MeasureTheory.Measure.comap f μ)
      ⊢ MeasureTheory.NullMeasurableSet (Set.image f (MeasureTheory.toMeasurable (Me …
    -/
  · exact hf _ (measurableSet_toMeasurable _ _)
    /-
      🎉 no goals
    -/
  have h : toMeasurable (comap f μ) s =ᵐ[comap f μ] s :=
    NullMeasurableSet.toMeasurable_ae_eq hs
  /-
    case refine_1
    α : Type u_1
    β : Type u_2
    s : Set α
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β
    μ : MeasureTheory.Measure β
    hfi : Function.Injective f
    hf : ∀ (s : Set α), MeasurableSet s → MeasureTheory.NullMeasurableSet (Set.ima …
    hs : MeasureTheory.NullMeasurableSet s (MeasureTheory.Measure.comap f μ)
    h : (MeasureTheory.ae (MeasureTheory.Measure.comap f μ)).EventuallyEq (Measure …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Set.image f s) (Set.image f (MeasureTheor …
  -/
  exact ae_eq_image_of_ae_eq_comap f μ hfi hf h.symm
  /-
    🎉 no goals
  -/


theorem comap_preimage (f : α → β) (μ : Measure β) (hf : Injective f) (hf' : Measurable f)
    (h : ∀ t, MeasurableSet t → NullMeasurableSet (f '' t) μ) {s : Set β} (hs : MeasurableSet s) :
    μ.comap f (f ⁻¹' s) = μ (s ∩ range f) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β
    μ : MeasureTheory.Measure β
    hf : Function.Injective f
    hf' : Measurable f
    h : ∀ (t : Set α), MeasurableSet t → MeasureTheory.NullMeasurableSet (Set.imag …
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq ((MeasureTheory.Measure.comap f μ) (Set.preimage f s)) (μ (Inter.inter s  …
  -/
  rw [comap_apply₀ _ _ hf h (hf' hs).nullMeasurableSet, image_preimage_eq_inter_range]
  /-
    🎉 no goals
  -/


@[simp] lemma comap_zero (f : α → β) : (0 : Measure β).comap f = 0 := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β
    ⊢ Eq (MeasureTheory.Measure.comap f 0) 0
  -/
  by_cases hf : Injective f ∧ ∀ s, MeasurableSet s → NullMeasurableSet (f '' s) (0 : Measure β)
    /-
      case pos
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      f : α → β
      hf : And (Function.Injective f) (∀ (s : Set α), MeasurableSet s → MeasureTheor …
      ⊢ Eq (MeasureTheory.Measure.comap f 0) 0
    -/
  · simp [comap, hf]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      f : α → β
      hf : Not (And (Function.Injective f) (∀ (s : Set α), MeasurableSet s → Measure …
      ⊢ Eq (MeasureTheory.Measure.comap f 0) 0
    -/
  · simp [comap, hf]
    /-
      🎉 no goals
    -/


@[simp]
lemma comap_id (μ : Measure β) : comap (fun x ↦ x) μ = μ := by
  /-
    β : Type u_2
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure β
    ⊢ Eq (MeasureTheory.Measure.comap (fun x => x) μ) μ
  -/
  ext s hs
  /-
    case h
    β : Type u_2
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure β
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq ((MeasureTheory.Measure.comap (fun x => x) μ) s) (μ s)
  -/
  rw [comap_apply, image_id']
    /-
      case h.hfi
      β : Type u_2
      mβ : MeasurableSpace β
      μ : MeasureTheory.Measure β
      s : Set β
      hs : MeasurableSet s
      ⊢ Function.Injective fun x => x
    -/
  · exact injective_id
    /-
      🎉 no goals
    -/
  /-
    case h.hf
    β : Type u_2
    mβ : MeasurableSpace β
    μ : MeasureTheory.Measure β
    s : Set β
    hs : MeasurableSet s
    ⊢ ∀ (s : Set β), MeasurableSet s → MeasurableSet (Set.image (fun x => x) s)
  -/
  all_goals simp [*]
  /-
    🎉 no goals
  -/


lemma comap_comap (hf' : ∀ s, MeasurableSet s → MeasurableSet (f '' s)) (hg : Injective g)
    (hg' : ∀ s, MeasurableSet s → MeasurableSet (g '' s)) (μ : Measure γ) :
    comap f (comap g μ) = comap (g ∘ f) μ := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    f : α → β
    g : β → γ
    hf' : ∀ (s : Set α), MeasurableSet s → MeasurableSet (Set.image f s)
    hg : Function.Injective g
    hg' : ∀ (s : Set β), MeasurableSet s → MeasurableSet (Set.image g s)
    μ : MeasureTheory.Measure γ
    ⊢ Eq (MeasureTheory.Measure.comap f (MeasureTheory.Measure.comap g μ)) (Measur …
  -/
  by_cases hf : Injective f
    /-
      case pos
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      f : α → β
      g : β → γ
      hf' : ∀ (s : Set α), MeasurableSet s → MeasurableSet (Set.image f s)
      hg : Function.Injective g
      hg' : ∀ (s : Set β), MeasurableSet s → MeasurableSet (Set.image g s)
      μ : MeasureTheory.Measure γ
      hf : Function.Injective f
      ⊢ Eq (MeasureTheory.Measure.comap f (MeasureTheory.Measure.comap g μ)) (Measur …
    -/
  · ext s hs
    rw [comap_apply _ hf hf' _ hs, comap_apply _ hg hg' _ (hf' _ hs),
      comap_apply _ (hg.comp hf) (fun t ht ↦ image_comp g f _ ▸ hg' _ <| hf' _ ht) _ hs, image_comp]
    /-
      case neg
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      f : α → β
      g : β → γ
      hf' : ∀ (s : Set α), MeasurableSet s → MeasurableSet (Set.image f s)
      hg : Function.Injective g
      hg' : ∀ (s : Set β), MeasurableSet s → MeasurableSet (Set.image g s)
      μ : MeasureTheory.Measure γ
      hf : Not (Function.Injective f)
      ⊢ Eq (MeasureTheory.Measure.comap f (MeasureTheory.Measure.comap g μ)) (Measur …
    -/
  · rw [comap, dif_neg <| mt And.left hf, comap, dif_neg fun h ↦ hf h.1.of_comp]
    /-
      🎉 no goals
    -/



lemma MeasurableEmbedding.comap_add {f : α → β} (hf : MeasurableEmbedding f) (μ ν : Measure β) :
    (μ + ν).comap f = μ.comap f + ν.comap f := by
  /-
    α : Type u_1
    β : Type u_2
    ma : MeasurableSpace α
    mb : MeasurableSpace β
    f : α → β
    hf : MeasurableEmbedding f
    μ ν : MeasureTheory.Measure β
    ⊢ Eq (MeasureTheory.Measure.comap f (HAdd.hAdd μ ν)) (HAdd.hAdd (MeasureTheory …
  -/
  ext s hs
  simp only [← comapₗ_eq_comap _ hf.injective (fun _ ↦ hf.measurableSet_image.mpr) _ hs,
    _root_.map_add, add_apply]


lemma comap_symm {μ : Measure α} (e : α ≃ᵐ β) : μ.comap e.symm = μ.map e := by
  /-
    α : Type u_1
    β : Type u_2
    ma : MeasurableSpace α
    mb : MeasurableSpace β
    μ : MeasureTheory.Measure α
    e : MeasurableEquiv α β
    ⊢ Eq (MeasureTheory.Measure.comap (⇑e.symm) μ) (MeasureTheory.Measure.map (⇑e) …
  -/
  ext s hs
  /-
    case h
    α : Type u_1
    β : Type u_2
    ma : MeasurableSpace α
    mb : MeasurableSpace β
    μ : MeasureTheory.Measure α
    e : MeasurableEquiv α β
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq ((MeasureTheory.Measure.comap (⇑e.symm) μ) s) ((MeasureTheory.Measure.map …
  -/
  rw [e.map_apply, Measure.comap_apply _ e.symm.injective _ _ hs, image_symm]
  /-
    α : Type u_1
    β : Type u_2
    ma : MeasurableSpace α
    mb : MeasurableSpace β
    μ : MeasureTheory.Measure α
    e : MeasurableEquiv α β
    s : Set β
    hs : MeasurableSet s
    ⊢ ∀ (s : Set β), MeasurableSet s → MeasurableSet (Set.image (⇑e.symm) s)
  -/
  exact fun t ht ↦ e.symm.measurableSet_image.mpr ht
  /-
    🎉 no goals
  -/


lemma map_symm {μ : Measure α} (e : β ≃ᵐ α) : μ.map e.symm = μ.comap e := by
  /-
    α : Type u_1
    β : Type u_2
    ma : MeasurableSpace α
    mb : MeasurableSpace β
    μ : MeasureTheory.Measure α
    e : MeasurableEquiv β α
    ⊢ Eq (MeasureTheory.Measure.map (⇑e.symm) μ) (MeasureTheory.Measure.comap (⇑e) …
  -/
  rw [← comap_symm, symm_symm]
  /-
    🎉 no goals
  -/


lemma comap_swap (μ : Measure (α × β)) : μ.comap Prod.swap = μ.map Prod.swap :=
  (MeasurableEquiv.prodComm ..).comap_symm

