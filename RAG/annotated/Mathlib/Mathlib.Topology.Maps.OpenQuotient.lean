protected theorem id : IsOpenQuotientMap (id : X → X) := ⟨surjective_id, continuous_id, .id⟩


/-- An open quotient map is a quotient map. -/
theorem isQuotientMap (h : IsOpenQuotientMap f) : IsQuotientMap f :=
  h.isOpenMap.isQuotientMap h.continuous h.surjective


@[deprecated (since := "2024-10-22")]
alias quotientMap := isQuotientMap


theorem iff_isOpenMap_isQuotientMap : IsOpenQuotientMap f ↔ IsOpenMap f ∧ IsQuotientMap f :=
  ⟨fun h ↦ ⟨h.isOpenMap, h.isQuotientMap⟩, fun ⟨ho, hq⟩ ↦ ⟨hq.surjective, hq.continuous, ho⟩⟩


@[deprecated (since := "2024-10-22")]
alias iff_isOpenMap_quotientMap := iff_isOpenMap_isQuotientMap


theorem of_isOpenMap_isQuotientMap (ho : IsOpenMap f) (hq : IsQuotientMap f) :
    IsOpenQuotientMap f :=
  iff_isOpenMap_isQuotientMap.2 ⟨ho, hq⟩


@[deprecated (since := "2024-10-22")]
alias of_isOpenMap_quotientMap := of_isOpenMap_isQuotientMap


theorem comp {g : Y → Z} (hg : IsOpenQuotientMap g) (hf : IsOpenQuotientMap f) :
    IsOpenQuotientMap (g ∘ f) :=
  ⟨.comp hg.1 hf.1, .comp hg.2 hf.2, .comp hg.3 hf.3⟩


theorem map_nhds_eq (h : IsOpenQuotientMap f) (x : X) : map f (𝓝 x) = 𝓝 (f x) :=
  le_antisymm h.continuous.continuousAt <| h.isOpenMap.nhds_le _


theorem continuous_comp_iff (h : IsOpenQuotientMap f) {g : Y → Z} :
    Continuous (g ∘ f) ↔ Continuous g :=
  h.isQuotientMap.continuous_iff.symm


theorem continuousAt_comp_iff (h : IsOpenQuotientMap f) {g : Y → Z} {x : X} :
    ContinuousAt (g ∘ f) x ↔ ContinuousAt g (f x) := by
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    f : X → Y
    h : IsOpenQuotientMap f
    g : Y → Z
    x : X
    ⊢ Iff (ContinuousAt (Function.comp g f) x) (ContinuousAt g (f x))
  -/
  simp only [ContinuousAt, ← h.map_nhds_eq, tendsto_map'_iff, comp_def]
  /-
    🎉 no goals
  -/


theorem dense_preimage_iff (h : IsOpenQuotientMap f) {s : Set Y} : Dense (f ⁻¹' s) ↔ Dense s :=
  ⟨fun hs ↦ h.surjective.denseRange.dense_of_mapsTo h.continuous hs (mapsTo_preimage _ _),
    fun hs ↦ hs.preimage h.isOpenMap⟩


