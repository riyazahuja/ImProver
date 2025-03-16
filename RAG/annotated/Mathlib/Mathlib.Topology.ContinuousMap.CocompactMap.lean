/-- A *cocompact continuous map* is a continuous function between topological spaces which
tends to the cocompact filter along the cocompact filter. Functions for which preimages of compact
sets are compact always satisfy this property, and the converse holds for cocompact continuous maps
when the codomain is Hausdorff (see `CocompactMap.tendsto_of_forall_preimage` and
`CocompactMap.isCompact_preimage`).

Cocompact maps thus generalise proper maps, with which they correspond when the codomain is
Hausdorff. -/
structure CocompactMap (α : Type u) (β : Type v) [TopologicalSpace α] [TopologicalSpace β] extends
  ContinuousMap α β : Type max u v where
  /-- The cocompact filter on `α` tends to the cocompact filter on `β` under the function -/
  cocompact_tendsto' : Tendsto toFun (cocompact α) (cocompact β)


/-- `CocompactMapClass F α β` states that `F` is a type of cocompact continuous maps.

You should also extend this typeclass when you extend `CocompactMap`. -/
class CocompactMapClass (F : Type*) (α β : outParam Type*) [TopologicalSpace α]
  [TopologicalSpace β] [FunLike F α β] extends ContinuousMapClass F α β : Prop where
  /-- The cocompact filter on `α` tends to the cocompact filter on `β` under the function -/
  cocompact_tendsto (f : F) : Tendsto f (cocompact α) (cocompact β)


/-- Turn an element of a type `F` satisfying `CocompactMapClass F α β` into an actual
`CocompactMap`. This is declared as the default coercion from `F` to `CocompactMap α β`. -/
@[coe]
def toCocompactMap (f : F) : CocompactMap α β :=
  { (f : C(α, β)) with
    cocompact_tendsto' := cocompact_tendsto f }


instance : CoeTC F (CocompactMap α β) :=
  ⟨toCocompactMap⟩


instance : FunLike (CocompactMap α β) α β where
  coe f := f.toFun
  coe_injective' f g h := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      inst✝³ : TopologicalSpace α
      inst✝² : TopologicalSpace β
      inst✝¹ : TopologicalSpace γ
      inst✝ : TopologicalSpace δ
      f g : CocompactMap α β
      h : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g)
      ⊢ Eq f g
    -/
    obtain ⟨⟨_, _⟩, _⟩ := f
    /-
      case mk.mk
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      inst✝³ : TopologicalSpace α
      inst✝² : TopologicalSpace β
      inst✝¹ : TopologicalSpace γ
      inst✝ : TopologicalSpace δ
      g : CocompactMap α β
      toFun✝ : α → β
      continuous_toFun✝ : Continuous toFun✝
      cocompact_tendsto'✝ : Filter.Tendsto { toFun := toFun✝, continuous_toFun := co …
      h : Eq ((fun f => f.toFun) { toFun := toFun✝, continuous_toFun := continuous_t …
      ⊢ Eq { toFun := toFun✝, continuous_toFun := continuous_toFun✝, cocompact_tends …
    -/
    obtain ⟨⟨_, _⟩, _⟩ := g
    /-
      case mk.mk.mk.mk
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      inst✝³ : TopologicalSpace α
      inst✝² : TopologicalSpace β
      inst✝¹ : TopologicalSpace γ
      inst✝ : TopologicalSpace δ
      toFun✝¹ : α → β
      continuous_toFun✝¹ : Continuous toFun✝¹
      cocompact_tendsto'✝¹ : Filter.Tendsto { toFun := toFun✝¹, continuous_toFun :=  …
      toFun✝ : α → β
      continuous_toFun✝ : Continuous toFun✝
      cocompact_tendsto'✝ : Filter.Tendsto { toFun := toFun✝, continuous_toFun := co …
      h : Eq ((fun f => f.toFun) { toFun := toFun✝¹, continuous_toFun := continuous_ …
      ⊢ Eq { toFun := toFun✝¹, continuous_toFun := continuous_toFun✝¹, cocompact_ten …
    -/
    congr
    /-
      🎉 no goals
    -/


instance : CocompactMapClass (CocompactMap α β) α β where
  map_continuous f := f.continuous_toFun
  cocompact_tendsto f := f.cocompact_tendsto'


@[simp]
theorem coe_toContinuousMap {f : CocompactMap α β} : (f.toContinuousMap : α → β) = f :=
  rfl


@[ext]
theorem ext {f g : CocompactMap α β} (h : ∀ x, f x = g x) : f = g :=
  DFunLike.ext _ _ h


/-- Copy of a `CocompactMap` with a new `toFun` equal to the old one. Useful
to fix definitional equalities. -/
protected def copy (f : CocompactMap α β) (f' : α → β) (h : f' = f) : CocompactMap α β where
  toFun := f'
  continuous_toFun := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      inst✝³ : TopologicalSpace α
      inst✝² : TopologicalSpace β
      inst✝¹ : TopologicalSpace γ
      inst✝ : TopologicalSpace δ
      f : CocompactMap α β
      f' : α → β
      h : Eq f' ⇑f
      ⊢ Continuous f'
    -/
    rw [h]
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      inst✝³ : TopologicalSpace α
      inst✝² : TopologicalSpace β
      inst✝¹ : TopologicalSpace γ
      inst✝ : TopologicalSpace δ
      f : CocompactMap α β
      f' : α → β
      h : Eq f' ⇑f
      ⊢ Continuous ⇑f
    -/
    exact f.continuous_toFun
    /-
      🎉 no goals
    -/
  cocompact_tendsto' := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      inst✝³ : TopologicalSpace α
      inst✝² : TopologicalSpace β
      inst✝¹ : TopologicalSpace γ
      inst✝ : TopologicalSpace δ
      f : CocompactMap α β
      f' : α → β
      h : Eq f' ⇑f
      ⊢ Filter.Tendsto { toFun := f', continuous_toFun := ⋯ }.toFun (Filter.cocompac …
    -/
    simp_rw [h]
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      inst✝³ : TopologicalSpace α
      inst✝² : TopologicalSpace β
      inst✝¹ : TopologicalSpace γ
      inst✝ : TopologicalSpace δ
      f : CocompactMap α β
      f' : α → β
      h : Eq f' ⇑f
      ⊢ Filter.Tendsto (⇑f) (Filter.cocompact α) (Filter.cocompact β)
    -/
    exact f.cocompact_tendsto'
    /-
      🎉 no goals
    -/


@[simp]
theorem coe_copy (f : CocompactMap α β) (f' : α → β) (h : f' = f) : ⇑(f.copy f' h) = f' :=
  rfl


theorem copy_eq (f : CocompactMap α β) (f' : α → β) (h : f' = f) : f.copy f' h = f :=
  DFunLike.ext' h


@[simp]
theorem coe_mk (f : C(α, β)) (h : Tendsto f (cocompact α) (cocompact β)) :
    ⇑(⟨f, h⟩ : CocompactMap α β) = f :=
  rfl


/-- The identity as a cocompact continuous map. -/
protected def id : CocompactMap α α :=
  ⟨ContinuousMap.id _, tendsto_id⟩


@[simp]
theorem coe_id : ⇑(CocompactMap.id α) = id :=
  rfl


instance : Inhabited (CocompactMap α α) :=
  ⟨CocompactMap.id α⟩


/-- The composition of cocompact continuous maps, as a cocompact continuous map. -/
def comp (f : CocompactMap β γ) (g : CocompactMap α β) : CocompactMap α γ :=
  ⟨f.toContinuousMap.comp g, (cocompact_tendsto f).comp (cocompact_tendsto g)⟩


@[simp]
theorem coe_comp (f : CocompactMap β γ) (g : CocompactMap α β) : ⇑(comp f g) = f ∘ g :=
  rfl


@[simp]
theorem comp_apply (f : CocompactMap β γ) (g : CocompactMap α β) (a : α) : comp f g a = f (g a) :=
  rfl


@[simp]
theorem comp_assoc (f : CocompactMap γ δ) (g : CocompactMap β γ) (h : CocompactMap α β) :
    (f.comp g).comp h = f.comp (g.comp h) :=
  rfl


@[simp]
theorem id_comp (f : CocompactMap α β) : (CocompactMap.id _).comp f = f :=
  ext fun _ => rfl


@[simp]
theorem comp_id (f : CocompactMap α β) : f.comp (CocompactMap.id _) = f :=
  ext fun _ => rfl


theorem tendsto_of_forall_preimage {f : α → β} (h : ∀ s, IsCompact s → IsCompact (f ⁻¹' s)) :
    Tendsto f (cocompact α) (cocompact β) := fun s hs =>
  match mem_cocompact.mp hs with
  | ⟨t, ht, hts⟩ =>
                                                        /-
                                                          α : Type u_1
                                                          β : Type u_2
                                                          inst✝¹ : TopologicalSpace α
                                                          inst✝ : TopologicalSpace β
                                                          f : α → β
                                                          h : ∀ (s : Set β), IsCompact s → IsCompact (Set.preimage f s)
                                                          s : Set β
                                                          hs : Membership.mem (Filter.cocompact β) s
                                                          t : Set β
                                                          ht : IsCompact t
                                                          hts : HasSubset.Subset (HasCompl.compl t) s
                                                          ⊢ HasSubset.Subset (HasCompl.compl (Set.preimage f t)) (Set.preimage f s)
                                                        -/
    mem_map.mpr (mem_cocompact.mpr ⟨f ⁻¹' t, h t ht, by simpa using preimage_mono hts⟩)
                                                        /-
                                                          🎉 no goals
                                                        -/


/-- Preimages of compact closed sets are compact under a cocompact continuous map. -/
theorem isCompact_preimage_of_isClosed (f : CocompactMap α β)
    ⦃s : Set β⦄ (hs : IsCompact s) (h's : IsClosed s) :
    IsCompact (f ⁻¹' s) := by
  obtain ⟨t, ht, hts⟩ :=
    mem_cocompact'.mp
      (by
        simpa only [preimage_image_preimage, preimage_compl] using
          mem_map.mp
            (cocompact_tendsto f <|
              mem_cocompact.mpr ⟨s, hs, compl_subset_compl.mpr (image_preimage_subset f _)⟩))
  exact
    ht.of_isClosed_subset (h's.preimage <| map_continuous f) (by simpa using hts)


/-- If the codomain is Hausdorff, preimages of compact sets are compact under a cocompact
continuous map. -/
theorem isCompact_preimage [T2Space β] (f : CocompactMap α β) ⦃s : Set β⦄ (hs : IsCompact s) :
    IsCompact (f ⁻¹' s) :=
  isCompact_preimage_of_isClosed f hs hs.isClosed


/-- A homeomorphism is a cocompact map. -/
@[simps]
def Homeomorph.toCocompactMap {α β : Type*} [TopologicalSpace α] [TopologicalSpace β]
    (f : α ≃ₜ β) : CocompactMap α β where
  toFun := f
  continuous_toFun := f.continuous
  cocompact_tendsto' := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : Homeomorph α β
      ⊢ Filter.Tendsto { toFun := ⇑f, continuous_toFun := ⋯ }.toFun (Filter.cocompac …
    -/
    refine CocompactMap.tendsto_of_forall_preimage fun K hK => ?_
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : Homeomorph α β
      K : Set β
      hK : IsCompact K
      ⊢ IsCompact (Set.preimage { toFun := ⇑f, continuous_toFun := ⋯ }.toFun K)
    -/
    erw [K.preimage_equiv_eq_image_symm]
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : Homeomorph α β
      K : Set β
      hK : IsCompact K
      ⊢ IsCompact (Set.image (⇑f.symm) K)
    -/
    exact hK.image f.symm.continuous
    /-
      🎉 no goals
    -/

