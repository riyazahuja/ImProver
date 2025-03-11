/-- Homeomorphism between `X` and `Y`, also called topological isomorphism -/
structure Homeomorph (X : Type*) (Y : Type*) [TopologicalSpace X] [TopologicalSpace Y]
    extends X ≃ Y where
  /-- The forward map of a homeomorphism is a continuous function. -/
  continuous_toFun : Continuous toFun := by continuity
  /-- The inverse map of a homeomorphism is a continuous function. -/
  continuous_invFun : Continuous invFun := by continuity


@[inherit_doc]
infixl:25 " ≃ₜ " => Homeomorph


theorem toEquiv_injective : Function.Injective (toEquiv : X ≃ₜ Y → X ≃ Y)
  | ⟨_, _, _⟩, ⟨_, _, _⟩, rfl => rfl


instance : EquivLike (X ≃ₜ Y) X Y where
  coe h := h.toEquiv
  inv h := h.toEquiv.symm
  left_inv h := h.left_inv
  right_inv h := h.right_inv
  coe_injective' _ _ H _ := toEquiv_injective <| DFunLike.ext' H


@[simp] theorem homeomorph_mk_coe (a : X ≃ Y) (b c) : (Homeomorph.mk a b c : X → Y) = a :=
  rfl


/-- The unique homeomorphism between two empty types. -/
protected def empty [IsEmpty X] [IsEmpty Y] : X ≃ₜ Y where
  __ := Equiv.equivOfIsEmpty X Y


/-- Inverse of a homeomorphism. -/
@[symm]
protected def symm (h : X ≃ₜ Y) : Y ≃ₜ X where
  continuous_toFun := h.continuous_invFun
  continuous_invFun := h.continuous_toFun
  toEquiv := h.toEquiv.symm


@[simp] theorem symm_symm (h : X ≃ₜ Y) : h.symm.symm = h := rfl


theorem symm_bijective : Function.Bijective (Homeomorph.symm : (X ≃ₜ Y) → Y ≃ₜ X) :=
  Function.bijective_iff_has_inverse.mpr ⟨_, symm_symm, symm_symm⟩


/-- See Note [custom simps projection] -/
def Simps.symm_apply (h : X ≃ₜ Y) : Y → X :=
  h.symm


@[simp]
theorem coe_toEquiv (h : X ≃ₜ Y) : ⇑h.toEquiv = h :=
  rfl


@[simp]
theorem coe_symm_toEquiv (h : X ≃ₜ Y) : ⇑h.toEquiv.symm = h.symm :=
  rfl


@[ext]
theorem ext {h h' : X ≃ₜ Y} (H : ∀ x, h x = h' x) : h = h' :=
  DFunLike.ext _ _ H


/-- Identity map as a homeomorphism. -/
@[simps! (config := .asFn) apply]
protected def refl (X : Type*) [TopologicalSpace X] : X ≃ₜ X where
  continuous_toFun := continuous_id
  continuous_invFun := continuous_id
  toEquiv := Equiv.refl X


/-- Composition of two homeomorphisms. -/
@[trans]
protected def trans (h₁ : X ≃ₜ Y) (h₂ : Y ≃ₜ Z) : X ≃ₜ Z where
  continuous_toFun := h₂.continuous_toFun.comp h₁.continuous_toFun
  continuous_invFun := h₁.continuous_invFun.comp h₂.continuous_invFun
  toEquiv := Equiv.trans h₁.toEquiv h₂.toEquiv


@[simp]
theorem trans_apply (h₁ : X ≃ₜ Y) (h₂ : Y ≃ₜ Z) (x : X) : h₁.trans h₂ x = h₂ (h₁ x) :=
  rfl


@[simp]
theorem symm_trans_apply (f : X ≃ₜ Y) (g : Y ≃ₜ Z) (z : Z) :
    (f.trans g).symm z = f.symm (g.symm z) := rfl


@[simp]
theorem homeomorph_mk_coe_symm (a : X ≃ Y) (b c) :
    ((Homeomorph.mk a b c).symm : Y → X) = a.symm :=
  rfl


@[simp]
theorem refl_symm : (Homeomorph.refl X).symm = Homeomorph.refl X :=
  rfl


@[continuity, fun_prop]
protected theorem continuous (h : X ≃ₜ Y) : Continuous h :=
  h.continuous_toFun

-- otherwise `by continuity` can't prove continuity of `h.to_equiv.symm`

@[continuity]
protected theorem continuous_symm (h : X ≃ₜ Y) : Continuous h.symm :=
  h.continuous_invFun


@[simp]
theorem apply_symm_apply (h : X ≃ₜ Y) (y : Y) : h (h.symm y) = y :=
  h.toEquiv.apply_symm_apply y


@[simp]
theorem symm_apply_apply (h : X ≃ₜ Y) (x : X) : h.symm (h x) = x :=
  h.toEquiv.symm_apply_apply x


@[simp]
theorem self_trans_symm (h : X ≃ₜ Y) : h.trans h.symm = Homeomorph.refl X := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    h : Homeomorph X Y
    ⊢ Eq (h.trans h.symm) (Homeomorph.refl X)
  -/
  ext
  /-
    case H
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    h : Homeomorph X Y
    x✝ : X
    ⊢ Eq ((h.trans h.symm) x✝) ((Homeomorph.refl X) x✝)
  -/
  apply symm_apply_apply
  /-
    🎉 no goals
  -/


@[simp]
theorem symm_trans_self (h : X ≃ₜ Y) : h.symm.trans h = Homeomorph.refl Y := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    h : Homeomorph X Y
    ⊢ Eq (h.symm.trans h) (Homeomorph.refl Y)
  -/
  ext
  /-
    case H
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    h : Homeomorph X Y
    x✝ : Y
    ⊢ Eq ((h.symm.trans h) x✝) ((Homeomorph.refl Y) x✝)
  -/
  apply apply_symm_apply
  /-
    🎉 no goals
  -/


protected theorem bijective (h : X ≃ₜ Y) : Function.Bijective h :=
  h.toEquiv.bijective


protected theorem injective (h : X ≃ₜ Y) : Function.Injective h :=
  h.toEquiv.injective


protected theorem surjective (h : X ≃ₜ Y) : Function.Surjective h :=
  h.toEquiv.surjective


/-- Change the homeomorphism `f` to make the inverse function definitionally equal to `g`. -/
def changeInv (f : X ≃ₜ Y) (g : Y → X) (hg : Function.RightInverse g f) : X ≃ₜ Y :=
  haveI : g = f.symm := (f.left_inv.eq_rightInverse hg).symm
  { toFun := f
    invFun := g
                   /-
                     X : Type u_1
                     Y : Type u_2
                     W : Type u_3
                     Z : Type u_4
                     inst✝⁵ : TopologicalSpace X
                     inst✝⁴ : TopologicalSpace Y
                     inst✝³ : TopologicalSpace W
                     inst✝² : TopologicalSpace Z
                     X' : Type u_5
                     Y' : Type u_6
                     inst✝¹ : TopologicalSpace X'
                     inst✝ : TopologicalSpace Y'
                     f : Homeomorph X Y
                     g : Y → X
                     hg : Function.RightInverse g ⇑f
                     this : Eq g ⇑f.symm
                     ⊢ Function.LeftInverse g ⇑f
                   -/
    left_inv := by convert f.left_inv
                   /-
                     🎉 no goals
                   -/
                    /-
                      X : Type u_1
                      Y : Type u_2
                      W : Type u_3
                      Z : Type u_4
                      inst✝⁵ : TopologicalSpace X
                      inst✝⁴ : TopologicalSpace Y
                      inst✝³ : TopologicalSpace W
                      inst✝² : TopologicalSpace Z
                      X' : Type u_5
                      Y' : Type u_6
                      inst✝¹ : TopologicalSpace X'
                      inst✝ : TopologicalSpace Y'
                      f : Homeomorph X Y
                      g : Y → X
                      hg : Function.RightInverse g ⇑f
                      this : Eq g ⇑f.symm
                      ⊢ Function.RightInverse g ⇑f
                    -/
    right_inv := by convert f.right_inv using 1
                    /-
                      🎉 no goals
                    -/
    continuous_toFun := f.continuous
                            /-
                              X : Type u_1
                              Y : Type u_2
                              W : Type u_3
                              Z : Type u_4
                              inst✝⁵ : TopologicalSpace X
                              inst✝⁴ : TopologicalSpace Y
                              inst✝³ : TopologicalSpace W
                              inst✝² : TopologicalSpace Z
                              X' : Type u_5
                              Y' : Type u_6
                              inst✝¹ : TopologicalSpace X'
                              inst✝ : TopologicalSpace Y'
                              f : Homeomorph X Y
                              g : Y → X
                              hg : Function.RightInverse g ⇑f
                              this : Eq g ⇑f.symm
                              ⊢ Continuous { toFun := ⇑f, invFun := g, left_inv := ⋯, right_inv := ⋯ }.invFun
                            -/
    continuous_invFun := by convert f.symm.continuous }
                            /-
                              🎉 no goals
                            -/


@[simp]
theorem symm_comp_self (h : X ≃ₜ Y) : h.symm ∘ h = id :=
  funext h.symm_apply_apply


@[simp]
theorem self_comp_symm (h : X ≃ₜ Y) : h ∘ h.symm = id :=
  funext h.apply_symm_apply


@[simp]
theorem range_coe (h : X ≃ₜ Y) : range h = univ :=
  h.surjective.range_eq


theorem image_symm (h : X ≃ₜ Y) : image h.symm = preimage h :=
  funext h.symm.toEquiv.image_eq_preimage


theorem preimage_symm (h : X ≃ₜ Y) : preimage h.symm = image h :=
  (funext h.toEquiv.image_eq_preimage).symm


@[simp]
theorem image_preimage (h : X ≃ₜ Y) (s : Set Y) : h '' (h ⁻¹' s) = s :=
  h.toEquiv.image_preimage s


@[simp]
theorem preimage_image (h : X ≃ₜ Y) (s : Set X) : h ⁻¹' (h '' s) = s :=
  h.toEquiv.preimage_image s


theorem image_eq_preimage (h : X ≃ₜ Y) (s : Set X) : h '' s = h.symm ⁻¹' s :=
  h.toEquiv.image_eq_preimage s


lemma image_compl (h : X ≃ₜ Y) (s : Set X) : h '' (sᶜ) = (h '' s)ᶜ :=
  h.toEquiv.image_compl s


lemma isInducing (h : X ≃ₜ Y) : IsInducing h :=
                                                /-
                                                  X : Type u_1
                                                  Y : Type u_2
                                                  inst✝¹ : TopologicalSpace X
                                                  inst✝ : TopologicalSpace Y
                                                  h : Homeomorph X Y
                                                  ⊢ Topology.IsInducing (Function.comp ⇑h.symm ⇑h)
                                                -/
  .of_comp h.continuous h.symm.continuous <| by simp only [symm_comp_self, IsInducing.id]
                                                /-
                                                  🎉 no goals
                                                -/


@[deprecated (since := "2024-10-28")] alias inducing := isInducing


theorem induced_eq (h : X ≃ₜ Y) : TopologicalSpace.induced h ‹_› = ‹_› := h.isInducing.1.symm


theorem isQuotientMap (h : X ≃ₜ Y) : IsQuotientMap h :=
  IsQuotientMap.of_comp h.symm.continuous h.continuous <| by
    /-
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      h : Homeomorph X Y
      ⊢ Topology.IsQuotientMap (Function.comp ⇑h ⇑h.symm)
    -/
    simp only [self_comp_symm, IsQuotientMap.id]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-22")]
alias quotientMap := isQuotientMap


theorem coinduced_eq (h : X ≃ₜ Y) : TopologicalSpace.coinduced h ‹_› = ‹_› :=
  h.isQuotientMap.2.symm


theorem isEmbedding (h : X ≃ₜ Y) : IsEmbedding h := ⟨h.isInducing, h.injective⟩


@[deprecated (since := "2024-10-26")]
alias embedding := isEmbedding


/-- Homeomorphism given an embedding. -/
noncomputable def ofIsEmbedding (f : X → Y) (hf : IsEmbedding f) : X ≃ₜ Set.range f where
  continuous_toFun := hf.continuous.subtype_mk _
                                                 /-
                                                   X : Type u_1
                                                   Y : Type u_2
                                                   W : Type u_3
                                                   Z : Type u_4
                                                   inst✝⁵ : TopologicalSpace X
                                                   inst✝⁴ : TopologicalSpace Y
                                                   inst✝³ : TopologicalSpace W
                                                   inst✝² : TopologicalSpace Z
                                                   X' : Type u_5
                                                   Y' : Type u_6
                                                   inst✝¹ : TopologicalSpace X'
                                                   inst✝ : TopologicalSpace Y'
                                                   f : X → Y
                                                   hf : Topology.IsEmbedding f
                                                   ⊢ Continuous (Function.comp f (Equiv.ofInjective f ⋯).invFun)
                                                 -/
  continuous_invFun := hf.continuous_iff.2 <| by simp [continuous_subtype_val]
                                                 /-
                                                   🎉 no goals
                                                 -/
  toEquiv := Equiv.ofInjective f hf.injective


@[deprecated (since := "2024-10-26")]
alias ofEmbedding := ofIsEmbedding


protected theorem secondCountableTopology [SecondCountableTopology Y]
    (h : X ≃ₜ Y) : SecondCountableTopology X :=
  h.isInducing.secondCountableTopology


/-- If `h : X → Y` is a homeomorphism, `h(s)` is compact iff `s` is. -/
@[simp]
theorem isCompact_image {s : Set X} (h : X ≃ₜ Y) : IsCompact (h '' s) ↔ IsCompact s :=
  h.isEmbedding.isCompact_iff.symm


/-- If `h : X → Y` is a homeomorphism, `h⁻¹(s)` is compact iff `s` is. -/
@[simp]
theorem isCompact_preimage {s : Set Y} (h : X ≃ₜ Y) : IsCompact (h ⁻¹' s) ↔ IsCompact s := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set Y
    h : Homeomorph X Y
    ⊢ Iff (IsCompact (Set.preimage (⇑h) s)) (IsCompact s)
  -/
  rw [← image_symm]; exact h.symm.isCompact_image
                     /-
                       🎉 no goals
                     -/


/-- If `h : X → Y` is a homeomorphism, `s` is σ-compact iff `h(s)` is. -/
@[simp]
theorem isSigmaCompact_image {s : Set X} (h : X ≃ₜ Y) :
    IsSigmaCompact (h '' s) ↔ IsSigmaCompact s :=
  h.isEmbedding.isSigmaCompact_iff.symm


/-- If `h : X → Y` is a homeomorphism, `h⁻¹(s)` is σ-compact iff `s` is. -/
@[simp]
theorem isSigmaCompact_preimage {s : Set Y} (h : X ≃ₜ Y) :
    IsSigmaCompact (h ⁻¹' s) ↔ IsSigmaCompact s := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set Y
    h : Homeomorph X Y
    ⊢ Iff (IsSigmaCompact (Set.preimage (⇑h) s)) (IsSigmaCompact s)
  -/
  rw [← image_symm]; exact h.symm.isSigmaCompact_image
                     /-
                       🎉 no goals
                     -/


@[simp]
theorem isPreconnected_image {s : Set X} (h : X ≃ₜ Y) :
    IsPreconnected (h '' s) ↔ IsPreconnected s :=
  ⟨fun hs ↦ by simpa only [image_symm, preimage_image]
    using hs.image _ h.symm.continuous.continuousOn,
    fun hs ↦ hs.image _ h.continuous.continuousOn⟩


@[simp]
theorem isPreconnected_preimage {s : Set Y} (h : X ≃ₜ Y) :
    IsPreconnected (h ⁻¹' s) ↔ IsPreconnected s := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set Y
    h : Homeomorph X Y
    ⊢ Iff (IsPreconnected (Set.preimage (⇑h) s)) (IsPreconnected s)
  -/
  rw [← image_symm, isPreconnected_image]
  /-
    🎉 no goals
  -/


@[simp]
theorem isConnected_image {s : Set X} (h : X ≃ₜ Y) :
    IsConnected (h '' s) ↔ IsConnected s :=
  image_nonempty.and h.isPreconnected_image


@[simp]
theorem isConnected_preimage {s : Set Y} (h : X ≃ₜ Y) :
    IsConnected (h ⁻¹' s) ↔ IsConnected s := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set Y
    h : Homeomorph X Y
    ⊢ Iff (IsConnected (Set.preimage (⇑h) s)) (IsConnected s)
  -/
  rw [← image_symm, isConnected_image]
  /-
    🎉 no goals
  -/


theorem image_connectedComponentIn {s : Set X} (h : X ≃ₜ Y) {x : X} (hx : x ∈ s) :
    h '' connectedComponentIn s x = connectedComponentIn (h '' s) (h x) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set X
    h : Homeomorph X Y
    x : X
    hx : Membership.mem s x
    ⊢ Eq (Set.image (⇑h) (connectedComponentIn s x)) (connectedComponentIn (Set.im …
  -/
  refine (h.continuous.image_connectedComponentIn_subset hx).antisymm ?_
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set X
    h : Homeomorph X Y
    x : X
    hx : Membership.mem s x
    ⊢ HasSubset.Subset (connectedComponentIn (Set.image (⇑h) s) (h x)) (Set.image  …
  -/
  have := h.symm.continuous.image_connectedComponentIn_subset (mem_image_of_mem h hx)
  rwa [image_subset_iff, h.preimage_symm, h.image_symm, h.preimage_image, h.symm_apply_apply]
    at this


@[simp]
theorem comap_cocompact (h : X ≃ₜ Y) : comap h (cocompact Y) = cocompact X :=
  (comap_cocompact_le h.continuous).antisymm <|
    (hasBasis_cocompact.le_basis_iff (hasBasis_cocompact.comap h)).2 fun K hK =>
      ⟨h ⁻¹' K, h.isCompact_preimage.2 hK, Subset.rfl⟩


@[simp]
theorem map_cocompact (h : X ≃ₜ Y) : map h (cocompact X) = cocompact Y := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    h : Homeomorph X Y
    ⊢ Eq (Filter.map (⇑h) (Filter.cocompact X)) (Filter.cocompact Y)
  -/
  rw [← h.comap_cocompact, map_comap_of_surjective h.surjective]
  /-
    🎉 no goals
  -/


protected theorem compactSpace [CompactSpace X] (h : X ≃ₜ Y) : CompactSpace Y where
  isCompact_univ := h.symm.isCompact_preimage.2 isCompact_univ


protected theorem t0Space [T0Space X] (h : X ≃ₜ Y) : T0Space Y := h.symm.isEmbedding.t0Space

protected theorem t1Space [T1Space X] (h : X ≃ₜ Y) : T1Space Y := h.symm.isEmbedding.t1Space

protected theorem t2Space [T2Space X] (h : X ≃ₜ Y) : T2Space Y := h.symm.isEmbedding.t2Space

protected theorem t25Space [T25Space X] (h : X ≃ₜ Y) : T25Space Y := h.symm.isEmbedding.t25Space

protected theorem t3Space [T3Space X] (h : X ≃ₜ Y) : T3Space Y := h.symm.isEmbedding.t3Space


theorem isDenseEmbedding (h : X ≃ₜ Y) : IsDenseEmbedding h :=
  { h.isEmbedding with dense := h.surjective.denseRange }


protected lemma totallyDisconnectedSpace (h : X ≃ₜ Y) [tdc : TotallyDisconnectedSpace X] :
    TotallyDisconnectedSpace Y :=
  (totallyDisconnectedSpace_iff Y).mpr
    (h.range_coe ▸ ((IsEmbedding.isTotallyDisconnected_range h.isEmbedding).mpr tdc))


@[deprecated (since := "2024-09-30")]
alias denseEmbedding := isDenseEmbedding


@[simp]
theorem isOpen_preimage (h : X ≃ₜ Y) {s : Set Y} : IsOpen (h ⁻¹' s) ↔ IsOpen s :=
  h.isQuotientMap.isOpen_preimage


@[simp]
theorem isOpen_image (h : X ≃ₜ Y) {s : Set X} : IsOpen (h '' s) ↔ IsOpen s := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    h : Homeomorph X Y
    s : Set X
    ⊢ Iff (IsOpen (Set.image (⇑h) s)) (IsOpen s)
  -/
  rw [← preimage_symm, isOpen_preimage]
  /-
    🎉 no goals
  -/


protected theorem isOpenMap (h : X ≃ₜ Y) : IsOpenMap h := fun _ => h.isOpen_image.2


@[simp]
theorem isClosed_preimage (h : X ≃ₜ Y) {s : Set Y} : IsClosed (h ⁻¹' s) ↔ IsClosed s := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    h : Homeomorph X Y
    s : Set Y
    ⊢ Iff (IsClosed (Set.preimage (⇑h) s)) (IsClosed s)
  -/
  simp only [← isOpen_compl_iff, ← preimage_compl, isOpen_preimage]
  /-
    🎉 no goals
  -/


@[simp]
theorem isClosed_image (h : X ≃ₜ Y) {s : Set X} : IsClosed (h '' s) ↔ IsClosed s := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    h : Homeomorph X Y
    s : Set X
    ⊢ Iff (IsClosed (Set.image (⇑h) s)) (IsClosed s)
  -/
  rw [← preimage_symm, isClosed_preimage]
  /-
    🎉 no goals
  -/


protected theorem isClosedMap (h : X ≃ₜ Y) : IsClosedMap h := fun _ => h.isClosed_image.2


theorem isOpenEmbedding (h : X ≃ₜ Y) : IsOpenEmbedding h :=
  .of_isEmbedding_isOpenMap h.isEmbedding h.isOpenMap


@[deprecated (since := "2024-10-18")]
alias openEmbedding := isOpenEmbedding


theorem isClosedEmbedding (h : X ≃ₜ Y) : IsClosedEmbedding h :=
  .of_isEmbedding_isClosedMap h.isEmbedding h.isClosedMap


@[deprecated (since := "2024-10-20")]
alias closedEmbedding := isClosedEmbedding


protected theorem normalSpace [NormalSpace X] (h : X ≃ₜ Y) : NormalSpace Y :=
  h.symm.isClosedEmbedding.normalSpace


protected theorem t4Space [T4Space X] (h : X ≃ₜ Y) : T4Space Y := h.symm.isClosedEmbedding.t4Space

protected theorem t5Space [T5Space X] (h : X ≃ₜ Y) : T5Space Y := h.symm.isClosedEmbedding.t5Space


theorem preimage_closure (h : X ≃ₜ Y) (s : Set Y) : h ⁻¹' closure s = closure (h ⁻¹' s) :=
  h.isOpenMap.preimage_closure_eq_closure_preimage h.continuous _


theorem image_closure (h : X ≃ₜ Y) (s : Set X) : h '' closure s = closure (h '' s) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    h : Homeomorph X Y
    s : Set X
    ⊢ Eq (Set.image (⇑h) (closure s)) (closure (Set.image (⇑h) s))
  -/
  rw [← preimage_symm, preimage_closure]
  /-
    🎉 no goals
  -/


theorem preimage_interior (h : X ≃ₜ Y) (s : Set Y) : h ⁻¹' interior s = interior (h ⁻¹' s) :=
  h.isOpenMap.preimage_interior_eq_interior_preimage h.continuous _


theorem image_interior (h : X ≃ₜ Y) (s : Set X) : h '' interior s = interior (h '' s) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    h : Homeomorph X Y
    s : Set X
    ⊢ Eq (Set.image (⇑h) (interior s)) (interior (Set.image (⇑h) s))
  -/
  rw [← preimage_symm, preimage_interior]
  /-
    🎉 no goals
  -/


theorem preimage_frontier (h : X ≃ₜ Y) (s : Set Y) : h ⁻¹' frontier s = frontier (h ⁻¹' s) :=
  h.isOpenMap.preimage_frontier_eq_frontier_preimage h.continuous _


theorem image_frontier (h : X ≃ₜ Y) (s : Set X) : h '' frontier s = frontier (h '' s) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    h : Homeomorph X Y
    s : Set X
    ⊢ Eq (Set.image (⇑h) (frontier s)) (frontier (Set.image (⇑h) s))
  -/
  rw [← preimage_symm, preimage_frontier]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem _root_.HasCompactMulSupport.comp_homeomorph {M} [One M] {f : Y → M}
    (hf : HasCompactMulSupport f) (φ : X ≃ₜ Y) : HasCompactMulSupport (f ∘ φ) :=
  hf.comp_isClosedEmbedding φ.isClosedEmbedding


@[simp]
theorem map_nhds_eq (h : X ≃ₜ Y) (x : X) : map h (𝓝 x) = 𝓝 (h x) :=
                                      /-
                                        X : Type u_1
                                        Y : Type u_2
                                        inst✝¹ : TopologicalSpace X
                                        inst✝ : TopologicalSpace Y
                                        h : Homeomorph X Y
                                        x : X
                                        ⊢ Membership.mem (nhds (h x)) (Set.range ⇑h)
                                      -/
  h.isEmbedding.map_nhds_of_mem _ (by simp)
                                      /-
                                        🎉 no goals
                                      -/


@[simp]
theorem map_punctured_nhds_eq (h : X ≃ₜ Y) (x : X) : map h (𝓝[≠] x) = 𝓝[≠] (h x) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    h : Homeomorph X Y
    x : X
    ⊢ Eq (Filter.map (⇑h) (nhdsWithin x (HasCompl.compl (Singleton.singleton x)))) …
  -/
  convert h.isEmbedding.map_nhdsWithin_eq ({x}ᶜ) x
  /-
    case h.e'_3.h.e'_4
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    h : Homeomorph X Y
    x : X
    ⊢ Eq (HasCompl.compl (Singleton.singleton (h x))) (Set.image (⇑h) (HasCompl.co …
  -/
  rw [h.image_compl, Set.image_singleton]
  /-
    🎉 no goals
  -/


theorem symm_map_nhds_eq (h : X ≃ₜ Y) (x : X) : map h.symm (𝓝 (h x)) = 𝓝 x := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    h : Homeomorph X Y
    x : X
    ⊢ Eq (Filter.map (⇑h.symm) (nhds (h x))) (nhds x)
  -/
  rw [h.symm.map_nhds_eq, h.symm_apply_apply]
  /-
    🎉 no goals
  -/


theorem nhds_eq_comap (h : X ≃ₜ Y) (x : X) : 𝓝 x = comap h (𝓝 (h x)) :=
  h.isInducing.nhds_eq_comap x


@[simp]
theorem comap_nhds_eq (h : X ≃ₜ Y) (y : Y) : comap h (𝓝 y) = 𝓝 (h.symm y) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    h : Homeomorph X Y
    y : Y
    ⊢ Eq (Filter.comap (⇑h) (nhds y)) (nhds (h.symm y))
  -/
  rw [h.nhds_eq_comap, h.apply_symm_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem comap_coclosedCompact (h : X ≃ₜ Y) : comap h (coclosedCompact Y) = coclosedCompact X :=
  (hasBasis_coclosedCompact.comap h).eq_of_same_basis <| by
    /-
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      h : Homeomorph X Y
      ⊢ (Filter.coclosedCompact X).HasBasis (fun s => And (IsClosed s) (IsCompact s) …
    -/
    simpa [comp_def] using hasBasis_coclosedCompact.comp_surjective h.injective.preimage_surjective
    /-
      🎉 no goals
    -/


@[simp]
theorem map_coclosedCompact (h : X ≃ₜ Y) : map h (coclosedCompact X) = coclosedCompact Y := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    h : Homeomorph X Y
    ⊢ Eq (Filter.map (⇑h) (Filter.coclosedCompact X)) (Filter.coclosedCompact Y)
  -/
  rw [← h.comap_coclosedCompact, map_comap_of_surjective h.surjective]
  /-
    🎉 no goals
  -/


/-- If the codomain of a homeomorphism is a locally connected space, then the domain is also
a locally connected space. -/
theorem locallyConnectedSpace [i : LocallyConnectedSpace Y] (h : X ≃ₜ Y) :
    LocallyConnectedSpace X := by
  have : ∀ x, (𝓝 x).HasBasis (fun s ↦ IsOpen s ∧ h x ∈ s ∧ IsConnected s)
      (h.symm '' ·) := fun x ↦ by
    rw [← h.symm_map_nhds_eq]
    exact (i.1 _).map _
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    i : LocallyConnectedSpace Y
    h : Homeomorph X Y
    this : ∀ (x : X), (nhds x).HasBasis (fun s => And (IsOpen s) (And (Membership. …
    ⊢ LocallyConnectedSpace X
  -/
  refine locallyConnectedSpace_of_connected_bases _ _ this fun _ _ hs ↦ ?_
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    i : LocallyConnectedSpace Y
    h : Homeomorph X Y
    this : ∀ (x : X), (nhds x).HasBasis (fun s => And (IsOpen s) (And (Membership. …
    x✝¹ : X
    x✝ : Set Y
    hs : And (IsOpen x✝) (And (Membership.mem x✝ (h x✝¹)) (IsConnected x✝))
    ⊢ IsPreconnected (Set.image (⇑h.symm) x✝)
  -/
  exact hs.2.2.2.image _ h.symm.continuous.continuousOn
  /-
    🎉 no goals
  -/


/-- The codomain of a homeomorphism is a locally compact space if and only if
the domain is a locally compact space. -/
theorem locallyCompactSpace_iff (h : X ≃ₜ Y) :
    LocallyCompactSpace X ↔ LocallyCompactSpace Y := by
  exact ⟨fun _ => h.symm.isOpenEmbedding.locallyCompactSpace,
    fun _ => h.isClosedEmbedding.locallyCompactSpace⟩


/-- If a bijective map `e : X ≃ Y` is continuous and open, then it is a homeomorphism. -/
@[simps toEquiv]
def homeomorphOfContinuousOpen (e : X ≃ Y) (h₁ : Continuous e) (h₂ : IsOpenMap e) : X ≃ₜ Y where
  continuous_toFun := h₁
  continuous_invFun := e.continuous_symm_iff.2 h₂
  toEquiv := e


/-- If a bijective map `e : X ≃ Y` is continuous and closed, then it is a homeomorphism. -/
def homeomorphOfContinuousClosed (e : X ≃ Y) (h₁ : Continuous e) (h₂ : IsClosedMap e) : X ≃ₜ Y where
  continuous_toFun := h₁
  continuous_invFun := by
    /-
      X : Type u_1
      Y : Type u_2
      W : Type u_3
      Z : Type u_4
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : TopologicalSpace Y
      inst✝³ : TopologicalSpace W
      inst✝² : TopologicalSpace Z
      X' : Type u_5
      Y' : Type u_6
      inst✝¹ : TopologicalSpace X'
      inst✝ : TopologicalSpace Y'
      e : Equiv X Y
      h₁ : Continuous ⇑e
      h₂ : IsClosedMap ⇑e
      ⊢ Continuous e.invFun
    -/
    rw [continuous_iff_isClosed]
    /-
      X : Type u_1
      Y : Type u_2
      W : Type u_3
      Z : Type u_4
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : TopologicalSpace Y
      inst✝³ : TopologicalSpace W
      inst✝² : TopologicalSpace Z
      X' : Type u_5
      Y' : Type u_6
      inst✝¹ : TopologicalSpace X'
      inst✝ : TopologicalSpace Y'
      e : Equiv X Y
      h₁ : Continuous ⇑e
      h₂ : IsClosedMap ⇑e
      ⊢ ∀ (s : Set X), IsClosed s → IsClosed (Set.preimage e.invFun s)
    -/
    intro s hs
    /-
      X : Type u_1
      Y : Type u_2
      W : Type u_3
      Z : Type u_4
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : TopologicalSpace Y
      inst✝³ : TopologicalSpace W
      inst✝² : TopologicalSpace Z
      X' : Type u_5
      Y' : Type u_6
      inst✝¹ : TopologicalSpace X'
      inst✝ : TopologicalSpace Y'
      e : Equiv X Y
      h₁ : Continuous ⇑e
      h₂ : IsClosedMap ⇑e
      s : Set X
      hs : IsClosed s
      ⊢ IsClosed (Set.preimage e.invFun s)
    -/
    convert ← h₂ s hs using 1
    /-
      case h.e'_3
      X : Type u_1
      Y : Type u_2
      W : Type u_3
      Z : Type u_4
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : TopologicalSpace Y
      inst✝³ : TopologicalSpace W
      inst✝² : TopologicalSpace Z
      X' : Type u_5
      Y' : Type u_6
      inst✝¹ : TopologicalSpace X'
      inst✝ : TopologicalSpace Y'
      e : Equiv X Y
      h₁ : Continuous ⇑e
      h₂ : IsClosedMap ⇑e
      s : Set X
      hs : IsClosed s
      ⊢ Eq (Set.image (⇑e) s) (Set.preimage e.invFun s)
    -/
    apply e.image_eq_preimage
    /-
      🎉 no goals
    -/
  toEquiv := e


@[simp]
theorem homeomorphOfContinuousOpen_apply (e : X ≃ Y) (h₁ : Continuous e) (h₂ : IsOpenMap e) :
    ⇑(homeomorphOfContinuousOpen e h₁ h₂) = e := rfl


@[simp]
theorem homeomorphOfContinuousOpen_symm_apply (e : X ≃ Y) (h₁ : Continuous e) (h₂ : IsOpenMap e) :
    ⇑(homeomorphOfContinuousOpen e h₁ h₂).symm = e.symm := rfl


@[simp]
theorem comp_continuousOn_iff (h : X ≃ₜ Y) (f : Z → X) (s : Set Z) :
    ContinuousOn (h ∘ f) s ↔ ContinuousOn f s :=
  h.isInducing.continuousOn_iff.symm


@[simp]
theorem comp_continuous_iff (h : X ≃ₜ Y) {f : Z → X} : Continuous (h ∘ f) ↔ Continuous f :=
  h.isInducing.continuous_iff.symm


@[simp]
theorem comp_continuous_iff' (h : X ≃ₜ Y) {f : Y → Z} : Continuous (f ∘ h) ↔ Continuous f :=
  h.isQuotientMap.continuous_iff.symm


theorem comp_continuousAt_iff (h : X ≃ₜ Y) (f : Z → X) (z : Z) :
    ContinuousAt (h ∘ f) z ↔ ContinuousAt f z :=
  h.isInducing.continuousAt_iff.symm


theorem comp_continuousAt_iff' (h : X ≃ₜ Y) (f : Y → Z) (x : X) :
    ContinuousAt (f ∘ h) x ↔ ContinuousAt f (h x) :=
                                     /-
                                       X : Type u_1
                                       Y : Type u_2
                                       Z : Type u_4
                                       inst✝² : TopologicalSpace X
                                       inst✝¹ : TopologicalSpace Y
                                       inst✝ : TopologicalSpace Z
                                       h : Homeomorph X Y
                                       f : Y → Z
                                       x : X
                                       ⊢ Membership.mem (nhds (h x)) (Set.range ⇑h)
                                     -/
  h.isInducing.continuousAt_iff' (by simp)
                                     /-
                                       🎉 no goals
                                     -/


theorem comp_continuousWithinAt_iff (h : X ≃ₜ Y) (f : Z → X) (s : Set Z) (z : Z) :
    ContinuousWithinAt f s z ↔ ContinuousWithinAt (h ∘ f) s z :=
  h.isInducing.continuousWithinAt_iff


@[simp]
theorem comp_isOpenMap_iff (h : X ≃ₜ Y) {f : Z → X} : IsOpenMap (h ∘ f) ↔ IsOpenMap f := by
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_4
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    h : Homeomorph X Y
    f : Z → X
    ⊢ Iff (IsOpenMap (Function.comp (⇑h) f)) (IsOpenMap f)
  -/
  refine ⟨?_, fun hf => h.isOpenMap.comp hf⟩
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_4
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    h : Homeomorph X Y
    f : Z → X
    ⊢ IsOpenMap (Function.comp (⇑h) f) → IsOpenMap f
  -/
  intro hf
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_4
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    h : Homeomorph X Y
    f : Z → X
    hf : IsOpenMap (Function.comp (⇑h) f)
    ⊢ IsOpenMap f
  -/
  rw [← Function.id_comp f, ← h.symm_comp_self, Function.comp_assoc]
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_4
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    h : Homeomorph X Y
    f : Z → X
    hf : IsOpenMap (Function.comp (⇑h) f)
    ⊢ IsOpenMap (Function.comp (⇑h.symm) (Function.comp (⇑h) f))
  -/
  exact h.symm.isOpenMap.comp hf
  /-
    🎉 no goals
  -/


@[simp]
theorem comp_isOpenMap_iff' (h : X ≃ₜ Y) {f : Y → Z} : IsOpenMap (f ∘ h) ↔ IsOpenMap f := by
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_4
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    h : Homeomorph X Y
    f : Y → Z
    ⊢ Iff (IsOpenMap (Function.comp f ⇑h)) (IsOpenMap f)
  -/
  refine ⟨?_, fun hf => hf.comp h.isOpenMap⟩
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_4
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    h : Homeomorph X Y
    f : Y → Z
    ⊢ IsOpenMap (Function.comp f ⇑h) → IsOpenMap f
  -/
  intro hf
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_4
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    h : Homeomorph X Y
    f : Y → Z
    hf : IsOpenMap (Function.comp f ⇑h)
    ⊢ IsOpenMap f
  -/
  rw [← Function.comp_id f, ← h.self_comp_symm, ← Function.comp_assoc]
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_4
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    h : Homeomorph X Y
    f : Y → Z
    hf : IsOpenMap (Function.comp f ⇑h)
    ⊢ IsOpenMap (Function.comp (Function.comp f ⇑h) ⇑h.symm)
  -/
  exact hf.comp h.symm.isOpenMap
  /-
    🎉 no goals
  -/


/-- A homeomorphism `h : X ≃ₜ Y` lifts to a homeomorphism between subtypes corresponding to
predicates `p : X → Prop` and `q : Y → Prop` so long as `p = q ∘ h`. -/
@[simps!]
def subtype {p : X → Prop} {q : Y → Prop} (h : X ≃ₜ Y) (h_iff : ∀ x, p x ↔ q (h x)) :
    {x // p x} ≃ₜ {y // q y} where
                         /-
                           X : Type u_1
                           Y : Type u_2
                           W : Type u_3
                           Z : Type u_4
                           inst✝⁵ : TopologicalSpace X
                           inst✝⁴ : TopologicalSpace Y
                           inst✝³ : TopologicalSpace W
                           inst✝² : TopologicalSpace Z
                           X' : Type u_5
                           Y' : Type u_6
                           inst✝¹ : TopologicalSpace X'
                           inst✝ : TopologicalSpace Y'
                           p : X → Prop
                           q : Y → Prop
                           h : Homeomorph X Y
                           h_iff : ∀ (x : X), Iff (p x) (q (h x))
                           ⊢ Continuous __spread✝⁻⁰.toFun
                         -/
  continuous_toFun := by simpa [Equiv.coe_subtypeEquiv_eq_map] using h.continuous.subtype_map _
                         /-
                           🎉 no goals
                         -/
  continuous_invFun := by simpa [Equiv.coe_subtypeEquiv_eq_map] using
    h.symm.continuous.subtype_map _
  __ := h.subtypeEquiv h_iff


@[simp]
lemma subtype_toEquiv {p : X → Prop} {q : Y → Prop} (h : X ≃ₜ Y) (h_iff : ∀ x, p x ↔ q (h x)) :
    (h.subtype h_iff).toEquiv = h.toEquiv.subtypeEquiv h_iff :=
  rfl


/-- A homeomorphism `h : X ≃ₜ Y` lifts to a homeomorphism between sets `s : Set X` and `t : Set Y`
whenever `h` maps `s` onto `t`. -/
abbrev sets {s : Set X} {t : Set Y} (h : X ≃ₜ Y) (h_eq : s = h ⁻¹' t) : s ≃ₜ t :=
  h.subtype <| Set.ext_iff.mp h_eq


/-- If two sets are equal, then they are homeomorphic. -/
def setCongr {s t : Set X} (h : s = t) : s ≃ₜ t where
  continuous_toFun := continuous_inclusion h.subset
  continuous_invFun := continuous_inclusion h.symm.subset
  toEquiv := Equiv.setCongr h


/-- Sum of two homeomorphisms. -/
def sumCongr (h₁ : X ≃ₜ X') (h₂ : Y ≃ₜ Y') : X ⊕ Y ≃ₜ X' ⊕ Y' where
  continuous_toFun := h₁.continuous.sum_map h₂.continuous
  continuous_invFun := h₁.symm.continuous.sum_map h₂.symm.continuous
  toEquiv := h₁.toEquiv.sumCongr h₂.toEquiv


/-- Product of two homeomorphisms. -/
def prodCongr (h₁ : X ≃ₜ X') (h₂ : Y ≃ₜ Y') : X × Y ≃ₜ X' × Y' where
  toEquiv := h₁.toEquiv.prodCongr h₂.toEquiv


@[simp]
theorem prodCongr_symm (h₁ : X ≃ₜ X') (h₂ : Y ≃ₜ Y') :
    (h₁.prodCongr h₂).symm = h₁.symm.prodCongr h₂.symm :=
  rfl


@[simp]
theorem coe_prodCongr (h₁ : X ≃ₜ X') (h₂ : Y ≃ₜ Y') : ⇑(h₁.prodCongr h₂) = Prod.map h₁ h₂ :=
  rfl

-- Commutativity and associativity of the disjoint union of topological spaces,
-- and the sum with an empty space.

/-- `X ⊕ Y` is homeomorphic to `Y ⊕ X`. -/
def sumComm : X ⊕ Y ≃ₜ Y ⊕ X where
  toEquiv := Equiv.sumComm X Y
  continuous_toFun := continuous_sum_swap
  continuous_invFun := continuous_sum_swap


@[simp]
theorem sumComm_symm : (sumComm X Y).symm = sumComm Y X :=
  rfl


@[simp]
theorem coe_sumComm : ⇑(sumComm X Y) = Sum.swap :=
  rfl


@[continuity, fun_prop]
lemma continuous_sumAssoc : Continuous (Equiv.sumAssoc X Y Z) :=
                          /-
                            X : Type u_1
                            Y : Type u_2
                            Z : Type u_4
                            inst✝² : TopologicalSpace X
                            inst✝¹ : TopologicalSpace Y
                            inst✝ : TopologicalSpace Z
                            ⊢ Continuous (Sum.elim Sum.inl (Function.comp Sum.inr Sum.inl))
                          -/
                          /-
                            🎉 no goals
                          -/
  Continuous.sum_elim (by fun_prop) (by fun_prop)
                                        /-
                                          🎉 no goals
                                        -/


@[continuity, fun_prop]
lemma continuous_sumAssoc_symm : Continuous (Equiv.sumAssoc X Y Z).symm :=
                          /-
                            X : Type u_1
                            Y : Type u_2
                            Z : Type u_4
                            inst✝² : TopologicalSpace X
                            inst✝¹ : TopologicalSpace Y
                            inst✝ : TopologicalSpace Z
                            ⊢ Continuous (Function.comp Sum.inl Sum.inl)
                          -/
                          /-
                            🎉 no goals
                          -/
  Continuous.sum_elim (by fun_prop) (by fun_prop)
                                        /-
                                          🎉 no goals
                                        -/


/-- `(X ⊕ Y) ⊕ Z` is homeomorphic to `X ⊕ (Y ⊕ Z)`. -/
def sumAssoc : (X ⊕ Y) ⊕ Z ≃ₜ X ⊕ Y ⊕ Z where
  toEquiv := Equiv.sumAssoc X Y Z
  continuous_toFun := continuous_sumAssoc X Y Z
  continuous_invFun := continuous_sumAssoc_symm X Y Z


@[simp]
lemma sumAssoc_toEquiv : (sumAssoc X Y Z).toEquiv = Equiv.sumAssoc X Y Z := rfl


/-- Four-way commutativity of the disjoint union. The name matches `add_add_add_comm`. -/
def sumSumSumComm : (X ⊕ Y) ⊕ W ⊕ Z ≃ₜ (X ⊕ W) ⊕ Y ⊕ Z where
  toEquiv := Equiv.sumSumSumComm X Y W Z
  continuous_toFun := by
    /-
      X : Type u_1
      Y : Type u_2
      W : Type u_3
      Z : Type u_4
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : TopologicalSpace Y
      inst✝³ : TopologicalSpace W
      inst✝² : TopologicalSpace Z
      X' : Type u_5
      Y' : Type u_6
      inst✝¹ : TopologicalSpace X'
      inst✝ : TopologicalSpace Y'
      ⊢ Continuous (Equiv.sumSumSumComm X Y W Z).toFun
    -/
    unfold Equiv.sumSumSumComm
    /-
      X : Type u_1
      Y : Type u_2
      W : Type u_3
      Z : Type u_4
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : TopologicalSpace Y
      inst✝³ : TopologicalSpace W
      inst✝² : TopologicalSpace Z
      X' : Type u_5
      Y' : Type u_6
      inst✝¹ : TopologicalSpace X'
      inst✝ : TopologicalSpace Y'
      ⊢ Continuous { toFun := Function.comp (⇑(Equiv.sumAssoc (Sum X W) Y Z)) (Funct …
    -/
    dsimp only
    /-
      X : Type u_1
      Y : Type u_2
      W : Type u_3
      Z : Type u_4
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : TopologicalSpace Y
      inst✝³ : TopologicalSpace W
      inst✝² : TopologicalSpace Z
      X' : Type u_5
      Y' : Type u_6
      inst✝¹ : TopologicalSpace X'
      inst✝ : TopologicalSpace Y'
      ⊢ Continuous (Function.comp (⇑(Equiv.sumAssoc (Sum X W) Y Z)) (Function.comp ( …
    -/
    have : Continuous (Sum.map (Sum.map (@id X) ⇑(Equiv.sumComm Y W)) (@id Z)) := by continuity
    /-
      X : Type u_1
      Y : Type u_2
      W : Type u_3
      Z : Type u_4
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : TopologicalSpace Y
      inst✝³ : TopologicalSpace W
      inst✝² : TopologicalSpace Z
      X' : Type u_5
      Y' : Type u_6
      inst✝¹ : TopologicalSpace X'
      inst✝ : TopologicalSpace Y'
      this : Continuous (Sum.map (Sum.map id ⇑(Equiv.sumComm Y W)) id)
      ⊢ Continuous (Function.comp (⇑(Equiv.sumAssoc (Sum X W) Y Z)) (Function.comp ( …
    -/
    fun_prop
    /-
      🎉 no goals
    -/
  continuous_invFun := by
    /-
      X : Type u_1
      Y : Type u_2
      W : Type u_3
      Z : Type u_4
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : TopologicalSpace Y
      inst✝³ : TopologicalSpace W
      inst✝² : TopologicalSpace Z
      X' : Type u_5
      Y' : Type u_6
      inst✝¹ : TopologicalSpace X'
      inst✝ : TopologicalSpace Y'
      ⊢ Continuous (Equiv.sumSumSumComm X Y W Z).invFun
    -/
    unfold Equiv.sumSumSumComm
    /-
      X : Type u_1
      Y : Type u_2
      W : Type u_3
      Z : Type u_4
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : TopologicalSpace Y
      inst✝³ : TopologicalSpace W
      inst✝² : TopologicalSpace Z
      X' : Type u_5
      Y' : Type u_6
      inst✝¹ : TopologicalSpace X'
      inst✝ : TopologicalSpace Y'
      ⊢ Continuous { toFun := Function.comp (⇑(Equiv.sumAssoc (Sum X W) Y Z)) (Funct …
    -/
    dsimp only
    /-
      X : Type u_1
      Y : Type u_2
      W : Type u_3
      Z : Type u_4
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : TopologicalSpace Y
      inst✝³ : TopologicalSpace W
      inst✝² : TopologicalSpace Z
      X' : Type u_5
      Y' : Type u_6
      inst✝¹ : TopologicalSpace X'
      inst✝ : TopologicalSpace Y'
      ⊢ Continuous (Function.comp (⇑(Equiv.sumAssoc (Sum X Y) W Z)) (Function.comp ( …
    -/
    have : Continuous (Sum.map (Sum.map (@id X) (Equiv.sumComm Y W).symm) (@id Z)) := by continuity
    /-
      X : Type u_1
      Y : Type u_2
      W : Type u_3
      Z : Type u_4
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : TopologicalSpace Y
      inst✝³ : TopologicalSpace W
      inst✝² : TopologicalSpace Z
      X' : Type u_5
      Y' : Type u_6
      inst✝¹ : TopologicalSpace X'
      inst✝ : TopologicalSpace Y'
      this : Continuous (Sum.map (Sum.map id ⇑(Equiv.sumComm Y W).symm) id)
      ⊢ Continuous (Function.comp (⇑(Equiv.sumAssoc (Sum X Y) W Z)) (Function.comp ( …
    -/
    fun_prop
    /-
      🎉 no goals
    -/


@[simp]
lemma sumSumSumComm_toEquiv : (sumSumSumComm X Y W Z).toEquiv = (Equiv.sumSumSumComm X Y W Z) := rfl


@[simp]
lemma sumSumSumComm_symm : (sumSumSumComm X Y W Z).symm = (sumSumSumComm X W Y Z) := rfl


/-- The sum of `X` with any empty topological space is homeomorphic to `X`. -/
@[simps! (config := .asFn) apply]
def sumEmpty [IsEmpty Y] : X ⊕ Y ≃ₜ X where
  toEquiv := Equiv.sumEmpty X Y
                                                            /-
                                                              X : Type u_1
                                                              Y : Type u_2
                                                              W : Type u_3
                                                              Z : Type u_4
                                                              inst✝⁶ : TopologicalSpace X
                                                              inst✝⁵ : TopologicalSpace Y
                                                              inst✝⁴ : TopologicalSpace W
                                                              inst✝³ : TopologicalSpace Z
                                                              X' : Type u_5
                                                              Y' : Type u_6
                                                              inst✝² : TopologicalSpace X'
                                                              inst✝¹ : TopologicalSpace Y'
                                                              inst✝ : IsEmpty Y
                                                              ⊢ Continuous fun a => isEmptyElim a
                                                            -/
  continuous_toFun := Continuous.sum_elim continuous_id (by fun_prop)
                                                            /-
                                                              🎉 no goals
                                                            -/
  continuous_invFun := continuous_inl


/-- The sum of `X` with any empty topological space is homeomorphic to `X`. -/
def emptySum [IsEmpty Y] : Y ⊕ X ≃ₜ X := (sumComm Y X).trans (sumEmpty X Y)


@[simp] theorem coe_emptySum [IsEmpty Y] : (emptySum X Y).toEquiv = Equiv.emptySum Y X := rfl


/-- `X × Y` is homeomorphic to `Y × X`. -/
def prodComm : X × Y ≃ₜ Y × X where
  continuous_toFun := continuous_snd.prod_mk continuous_fst
  continuous_invFun := continuous_snd.prod_mk continuous_fst
  toEquiv := Equiv.prodComm X Y


@[simp]
theorem prodComm_symm : (prodComm X Y).symm = prodComm Y X :=
  rfl


@[simp]
theorem coe_prodComm : ⇑(prodComm X Y) = Prod.swap :=
  rfl


/-- `(X × Y) × Z` is homeomorphic to `X × (Y × Z)`. -/
def prodAssoc : (X × Y) × Z ≃ₜ X × Y × Z where
  continuous_toFun := continuous_fst.fst.prod_mk (continuous_fst.snd.prod_mk continuous_snd)
  continuous_invFun := (continuous_fst.prod_mk continuous_snd.fst).prod_mk continuous_snd.snd
  toEquiv := Equiv.prodAssoc X Y Z


@[simp]
lemma prodAssoc_toEquiv : (prodAssoc X Y Z).toEquiv = Equiv.prodAssoc X Y Z := rfl


/-- Four-way commutativity of `prod`. The name matches `mul_mul_mul_comm`. -/
def prodProdProdComm : (X × Y) × W × Z ≃ₜ (X × W) × Y × Z where
  toEquiv := Equiv.prodProdProdComm X Y W Z
  continuous_toFun := by
    /-
      X : Type u_1
      Y : Type u_2
      W : Type u_3
      Z : Type u_4
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : TopologicalSpace Y
      inst✝³ : TopologicalSpace W
      inst✝² : TopologicalSpace Z
      X' : Type u_5
      Y' : Type u_6
      inst✝¹ : TopologicalSpace X'
      inst✝ : TopologicalSpace Y'
      ⊢ Continuous (Equiv.prodProdProdComm X Y W Z).toFun
    -/
    unfold Equiv.prodProdProdComm
    /-
      X : Type u_1
      Y : Type u_2
      W : Type u_3
      Z : Type u_4
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : TopologicalSpace Y
      inst✝³ : TopologicalSpace W
      inst✝² : TopologicalSpace Z
      X' : Type u_5
      Y' : Type u_6
      inst✝¹ : TopologicalSpace X'
      inst✝ : TopologicalSpace Y'
      ⊢ Continuous { toFun := fun abcd => { fst := { fst := abcd.1.1, snd := abcd.2. …
    -/
    dsimp only
    /-
      X : Type u_1
      Y : Type u_2
      W : Type u_3
      Z : Type u_4
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : TopologicalSpace Y
      inst✝³ : TopologicalSpace W
      inst✝² : TopologicalSpace Z
      X' : Type u_5
      Y' : Type u_6
      inst✝¹ : TopologicalSpace X'
      inst✝ : TopologicalSpace Y'
      ⊢ Continuous fun abcd => { fst := { fst := abcd.1.1, snd := abcd.2.1 }, snd := …
    -/
    fun_prop
    /-
      🎉 no goals
    -/
  continuous_invFun := by
    /-
      X : Type u_1
      Y : Type u_2
      W : Type u_3
      Z : Type u_4
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : TopologicalSpace Y
      inst✝³ : TopologicalSpace W
      inst✝² : TopologicalSpace Z
      X' : Type u_5
      Y' : Type u_6
      inst✝¹ : TopologicalSpace X'
      inst✝ : TopologicalSpace Y'
      ⊢ Continuous (Equiv.prodProdProdComm X Y W Z).invFun
    -/
    unfold Equiv.prodProdProdComm
    /-
      X : Type u_1
      Y : Type u_2
      W : Type u_3
      Z : Type u_4
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : TopologicalSpace Y
      inst✝³ : TopologicalSpace W
      inst✝² : TopologicalSpace Z
      X' : Type u_5
      Y' : Type u_6
      inst✝¹ : TopologicalSpace X'
      inst✝ : TopologicalSpace Y'
      ⊢ Continuous { toFun := fun abcd => { fst := { fst := abcd.1.1, snd := abcd.2. …
    -/
    dsimp only
    /-
      X : Type u_1
      Y : Type u_2
      W : Type u_3
      Z : Type u_4
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : TopologicalSpace Y
      inst✝³ : TopologicalSpace W
      inst✝² : TopologicalSpace Z
      X' : Type u_5
      Y' : Type u_6
      inst✝¹ : TopologicalSpace X'
      inst✝ : TopologicalSpace Y'
      ⊢ Continuous fun acbd => { fst := { fst := acbd.1.1, snd := acbd.2.1 }, snd := …
    -/
    fun_prop
    /-
      🎉 no goals
    -/


@[simp]
theorem prodProdProdComm_symm : (prodProdProdComm X Y W Z).symm = prodProdProdComm X W Y Z :=
  rfl


/-- `X × {*}` is homeomorphic to `X`. -/
@[simps! (config := .asFn) apply]
def prodPUnit : X × PUnit ≃ₜ X where
  toEquiv := Equiv.prodPUnit X
  continuous_toFun := continuous_fst
  continuous_invFun := continuous_id.prod_mk continuous_const


/-- `{*} × X` is homeomorphic to `X`. -/
def punitProd : PUnit × X ≃ₜ X :=
  (prodComm _ _).trans (prodPUnit _)


@[simp] theorem coe_punitProd : ⇑(punitProd X) = Prod.snd := rfl


/-- If both `X` and `Y` have a unique element, then `X ≃ₜ Y`. -/
@[simps!]
def homeomorphOfUnique [Unique X] [Unique Y] : X ≃ₜ Y :=
  { Equiv.ofUnique X Y with
    continuous_toFun := continuous_const
    continuous_invFun := continuous_const }


/-- The product over `S ⊕ T` of a family of topological spaces
is homeomorphic to the product of (the product over `S`) and (the product over `T`).

This is `Equiv.sumPiEquivProdPi` as a `Homeomorph`.
-/
def sumPiEquivProdPi (S T : Type*) (A : S ⊕ T → Type*)
    [∀ st, TopologicalSpace (A st)] :
    (Π (st : S ⊕ T), A st) ≃ₜ (Π (s : S), A (.inl s)) × (Π (t : T), A (.inr t)) where
  __ := Equiv.sumPiEquivProdPi _
                                             /-
                                               X : Type u_1
                                               Y : Type u_2
                                               W : Type u_3
                                               Z : Type u_4
                                               inst✝⁶ : TopologicalSpace X
                                               inst✝⁵ : TopologicalSpace Y
                                               inst✝⁴ : TopologicalSpace W
                                               inst✝³ : TopologicalSpace Z
                                               X' : Type u_5
                                               Y' : Type u_6
                                               inst✝² : TopologicalSpace X'
                                               inst✝¹ : TopologicalSpace Y'
                                               S : Type u_7
                                               T : Type u_8
                                               A : Sum S T → Type u_9
                                               inst✝ : (st : Sum S T) → TopologicalSpace (A st)
                                               ⊢ Continuous fun x i => x (Sum.inl i)
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
  continuous_toFun := Continuous.prod_mk (by fun_prop) (by fun_prop)
                                                           /-
                                                             🎉 no goals
                                                           -/
                                           /-
                                             X : Type u_1
                                             Y : Type u_2
                                             W : Type u_3
                                             Z : Type u_4
                                             inst✝⁶ : TopologicalSpace X
                                             inst✝⁵ : TopologicalSpace Y
                                             inst✝⁴ : TopologicalSpace W
                                             inst✝³ : TopologicalSpace Z
                                             X' : Type u_5
                                             Y' : Type u_6
                                             inst✝² : TopologicalSpace X'
                                             inst✝¹ : TopologicalSpace Y'
                                             S : Type u_7
                                             T : Type u_8
                                             A : Sum S T → Type u_9
                                             inst✝ : (st : Sum S T) → TopologicalSpace (A st)
                                             ⊢ ∀ (i : Sum S T), Continuous fun a => __spread✝⁻⁰.invFun a i
                                           -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
  continuous_invFun := continuous_pi <| by rintro (s | t) <;> simp <;> fun_prop
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


/-- The product `Π t : α, f t` of a family of topological spaces is homeomorphic to the
space `f ⬝` when `α` only contains `⬝`.

This is `Equiv.piUnique` as a `Homeomorph`.
-/
@[simps! (config := .asFn)]
def piUnique {α : Type*} [Unique α] (f : α → Type*) [∀ x, TopologicalSpace (f x)] :
    (Π t, f t) ≃ₜ f default :=
  homeomorphOfContinuousOpen (Equiv.piUnique f) (continuous_apply default) (isOpenMap_eval _)


/-- `Equiv.piCongrLeft` as a homeomorphism: this is the natural homeomorphism
`Π i, Y (e i) ≃ₜ Π j, Y j` obtained from a bijection `ι ≃ ι'`. -/
@[simps! apply toEquiv]
def piCongrLeft {ι ι' : Type*} {Y : ι' → Type*} [∀ j, TopologicalSpace (Y j)]
    (e : ι ≃ ι') : (∀ i, Y (e i)) ≃ₜ ∀ j, Y j where
  continuous_toFun := continuous_pi <| e.forall_congr_right.mp fun i ↦ by
    /-
      X : Type u_1
      Y✝ : Type u_2
      W : Type u_3
      Z : Type u_4
      inst✝⁶ : TopologicalSpace X
      inst✝⁵ : TopologicalSpace Y✝
      inst✝⁴ : TopologicalSpace W
      inst✝³ : TopologicalSpace Z
      X' : Type u_5
      Y' : Type u_6
      inst✝² : TopologicalSpace X'
      inst✝¹ : TopologicalSpace Y'
      ι : Type u_7
      ι' : Type u_8
      Y : ι' → Type u_9
      inst✝ : (j : ι') → TopologicalSpace (Y j)
      e : Equiv ι ι'
      i : ι
      ⊢ Continuous fun a => (Equiv.piCongrLeft Y e).toFun a (e i)
    -/
    simpa only [Equiv.toFun_as_coe, Equiv.piCongrLeft_apply_apply] using continuous_apply i
    /-
      🎉 no goals
    -/
  continuous_invFun := Pi.continuous_precomp' e
  toEquiv := Equiv.piCongrLeft _ e


/-- `Equiv.piCongrRight` as a homeomorphism: this is the natural homeomorphism
`Π i, Y₁ i ≃ₜ Π j, Y₂ i` obtained from homeomorphisms `Y₁ i ≃ₜ Y₂ i` for each `i`. -/
@[simps! apply toEquiv]
def piCongrRight {ι : Type*} {Y₁ Y₂ : ι → Type*} [∀ i, TopologicalSpace (Y₁ i)]
    [∀ i, TopologicalSpace (Y₂ i)] (F : ∀ i, Y₁ i ≃ₜ Y₂ i) : (∀ i, Y₁ i) ≃ₜ ∀ i, Y₂ i where
  continuous_toFun := Pi.continuous_postcomp' fun i ↦ (F i).continuous
  continuous_invFun := Pi.continuous_postcomp' fun i ↦ (F i).symm.continuous
  toEquiv := Equiv.piCongrRight fun i => (F i).toEquiv


@[simp]
theorem piCongrRight_symm {ι : Type*} {Y₁ Y₂ : ι → Type*} [∀ i, TopologicalSpace (Y₁ i)]
    [∀ i, TopologicalSpace (Y₂ i)] (F : ∀ i, Y₁ i ≃ₜ Y₂ i) :
    (piCongrRight F).symm = piCongrRight fun i => (F i).symm :=
  rfl


/-- `Equiv.piCongr` as a homeomorphism: this is the natural homeomorphism
`Π i₁, Y₁ i ≃ₜ Π i₂, Y₂ i₂` obtained from a bijection `ι₁ ≃ ι₂` and homeomorphisms
`Y₁ i₁ ≃ₜ Y₂ (e i₁)` for each `i₁ : ι₁`. -/
@[simps! apply toEquiv]
def piCongr {ι₁ ι₂ : Type*} {Y₁ : ι₁ → Type*} {Y₂ : ι₂ → Type*}
    [∀ i₁, TopologicalSpace (Y₁ i₁)] [∀ i₂, TopologicalSpace (Y₂ i₂)]
    (e : ι₁ ≃ ι₂) (F : ∀ i₁, Y₁ i₁ ≃ₜ Y₂ (e i₁)) : (∀ i₁, Y₁ i₁) ≃ₜ ∀ i₂, Y₂ i₂ :=
  (Homeomorph.piCongrRight F).trans (Homeomorph.piCongrLeft e)

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: align the order of universes with `Equiv.ulift`

/-- `ULift X` is homeomorphic to `X`. -/
def ulift.{u, v} {X : Type u} [TopologicalSpace X] : ULift.{v, u} X ≃ₜ X where
  continuous_toFun := continuous_uLift_down
  continuous_invFun := continuous_uLift_up
  toEquiv := Equiv.ulift


/-- The natural homeomorphism `(ι ⊕ ι' → X) ≃ₜ (ι → X) × (ι' → X)`.
`Equiv.sumArrowEquivProdArrow` as a homeomorphism. -/
@[simps!]
def sumArrowHomeomorphProdArrow {ι ι' : Type*} : (ι ⊕ ι' → X) ≃ₜ (ι → X) × (ι' → X)  where
  toEquiv := Equiv.sumArrowEquivProdArrow _ _ _
  continuous_toFun := by
    /-
      X : Type u_1
      Y : Type u_2
      W : Type u_3
      Z : Type u_4
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : TopologicalSpace Y
      inst✝³ : TopologicalSpace W
      inst✝² : TopologicalSpace Z
      X' : Type u_5
      Y' : Type u_6
      inst✝¹ : TopologicalSpace X'
      inst✝ : TopologicalSpace Y'
      ι : Type u_7
      ι' : Type u_8
      ⊢ Continuous (Equiv.sumArrowEquivProdArrow ι ι' X).toFun
    -/
    simp only [Equiv.sumArrowEquivProdArrow, Equiv.coe_fn_mk, continuous_prod_mk]
    /-
      X : Type u_1
      Y : Type u_2
      W : Type u_3
      Z : Type u_4
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : TopologicalSpace Y
      inst✝³ : TopologicalSpace W
      inst✝² : TopologicalSpace Z
      X' : Type u_5
      Y' : Type u_6
      inst✝¹ : TopologicalSpace X'
      inst✝ : TopologicalSpace Y'
      ι : Type u_7
      ι' : Type u_8
      ⊢ And (Continuous fun x => Function.comp x Sum.inl) (Continuous fun x => Funct …
    -/
    continuity
    /-
      🎉 no goals
    -/
  continuous_invFun := continuous_pi fun i ↦ match i with
                   /-
                     X : Type u_1
                     Y : Type u_2
                     W : Type u_3
                     Z : Type u_4
                     inst✝⁵ : TopologicalSpace X
                     inst✝⁴ : TopologicalSpace Y
                     inst✝³ : TopologicalSpace W
                     inst✝² : TopologicalSpace Z
                     X' : Type u_5
                     Y' : Type u_6
                     inst✝¹ : TopologicalSpace X'
                     inst✝ : TopologicalSpace Y'
                     ι : Type u_7
                     ι' : Type u_8
                     i✝ : Sum ι ι'
                     i : ι
                     ⊢ Continuous fun a => (Equiv.sumArrowEquivProdArrow ι ι' X).invFun a (Sum.inl i)
                   -/
    | .inl i => by apply (continuous_apply _).comp' continuous_fst
                   /-
                     🎉 no goals
                   -/
                   /-
                     X : Type u_1
                     Y : Type u_2
                     W : Type u_3
                     Z : Type u_4
                     inst✝⁵ : TopologicalSpace X
                     inst✝⁴ : TopologicalSpace Y
                     inst✝³ : TopologicalSpace W
                     inst✝² : TopologicalSpace Z
                     X' : Type u_5
                     Y' : Type u_6
                     inst✝¹ : TopologicalSpace X'
                     inst✝ : TopologicalSpace Y'
                     ι : Type u_7
                     ι' : Type u_8
                     i✝ : Sum ι ι'
                     i : ι'
                     ⊢ Continuous fun a => (Equiv.sumArrowEquivProdArrow ι ι' X).invFun a (Sum.inr i)
                   -/
    | .inr i => by apply (continuous_apply _).comp' continuous_snd
                   /-
                     🎉 no goals
                   -/


private theorem _root_.Fin.appendEquiv_eq_Homeomorph (m n : ℕ) : Fin.appendEquiv m n =
    ((sumArrowHomeomorphProdArrow).symm.trans
    (piCongrLeft (Y := fun _ ↦ X) finSumFinEquiv)).toEquiv := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    m n : Nat
    ⊢ Eq (Fin.appendEquiv m n) (Homeomorph.sumArrowHomeomorphProdArrow.symm.trans  …
  -/
  ext ⟨x1, x2⟩ l
  simp only [sumArrowHomeomorphProdArrow, Equiv.sumArrowEquivProdArrow,
    finSumFinEquiv, Fin.addCases, Fin.appendEquiv, Fin.append, Equiv.coe_fn_mk]
  /-
    case H.mk.h
    X : Type u_1
    inst✝ : TopologicalSpace X
    m n : Nat
    x1 : Fin m → X
    x2 : Fin n → X
    l : Fin (HAdd.hAdd m n)
    ⊢ Eq (dite (LT.lt (↑l) m) (fun h => x1 (l.castLT ⋯)) fun h => Eq.rec (x2 (Fin. …
  -/
  by_cases h : l < m
    /-
      case pos
      X : Type u_1
      inst✝ : TopologicalSpace X
      m n : Nat
      x1 : Fin m → X
      x2 : Fin n → X
      l : Fin (HAdd.hAdd m n)
      h : LT.lt (↑l) m
      ⊢ Eq (dite (LT.lt (↑l) m) (fun h => x1 (l.castLT ⋯)) fun h => Eq.rec (x2 (Fin. …
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
    /-
      case neg
      X : Type u_1
      inst✝ : TopologicalSpace X
      m n : Nat
      x1 : Fin m → X
      x2 : Fin n → X
      l : Fin (HAdd.hAdd m n)
      h : Not (LT.lt (↑l) m)
      ⊢ Eq (dite (LT.lt (↑l) m) (fun h => x1 (l.castLT ⋯)) fun h => Eq.rec (x2 (Fin. …
    -/
  · simp [h]
    /-
      🎉 no goals
    -/


theorem _root_.Fin.continuous_append (m n : ℕ) :
    Continuous fun (p : (Fin m → X) × (Fin n → X)) ↦ Fin.append p.1 p.2 := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    m n : Nat
    ⊢ Continuous fun p => Fin.append p.1 p.2
  -/
  suffices Continuous (Fin.appendEquiv m n) by exact this
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    m n : Nat
    ⊢ Continuous ⇑(Fin.appendEquiv m n)
  -/
  rw [Fin.appendEquiv_eq_Homeomorph]
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    m n : Nat
    ⊢ Continuous ⇑(Homeomorph.sumArrowHomeomorphProdArrow.symm.trans (Homeomorph.p …
  -/
  exact Homeomorph.continuous_toFun _
  /-
    🎉 no goals
  -/


/-- The natural homeomorphism between `(Fin m → X) × (Fin n → X)` and `Fin (m + n) → X`.
`Fin.appendEquiv` as a homeomorphism.-/
@[simps!]
def _root_.Fin.appendHomeomorph (m n : ℕ) : (Fin m → X) × (Fin n → X) ≃ₜ (Fin (m + n) → X) where
  toEquiv := Fin.appendEquiv m n
  continuous_toFun := Fin.continuous_append m n
  continuous_invFun := by
    /-
      X : Type u_1
      Y : Type u_2
      W : Type u_3
      Z : Type u_4
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : TopologicalSpace Y
      inst✝³ : TopologicalSpace W
      inst✝² : TopologicalSpace Z
      X' : Type u_5
      Y' : Type u_6
      inst✝¹ : TopologicalSpace X'
      inst✝ : TopologicalSpace Y'
      m n : Nat
      ⊢ Continuous (Fin.appendEquiv m n).invFun
    -/
    rw [Fin.appendEquiv_eq_Homeomorph]
    /-
      X : Type u_1
      Y : Type u_2
      W : Type u_3
      Z : Type u_4
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : TopologicalSpace Y
      inst✝³ : TopologicalSpace W
      inst✝² : TopologicalSpace Z
      X' : Type u_5
      Y' : Type u_6
      inst✝¹ : TopologicalSpace X'
      inst✝ : TopologicalSpace Y'
      m n : Nat
      ⊢ Continuous (Homeomorph.sumArrowHomeomorphProdArrow.symm.trans (Homeomorph.pi …
    -/
    exact Homeomorph.continuous_invFun _
    /-
      🎉 no goals
    -/


@[simp]
theorem _root_.Fin.appendHomeomorph_toEquiv (m n : ℕ) :
    (Fin.appendHomeomorph (X := X) m n).toEquiv = Fin.appendEquiv m n :=
  rfl


/-- `(X ⊕ Y) × Z` is homeomorphic to `X × Z ⊕ Y × Z`. -/
@[simps!]
def sumProdDistrib : (X ⊕ Y) × Z ≃ₜ (X × Z) ⊕ (Y × Z) :=
  Homeomorph.symm <|
    homeomorphOfContinuousOpen (Equiv.sumProdDistrib X Y Z).symm
        ((continuous_inl.prodMap continuous_id).sum_elim
          (continuous_inr.prodMap continuous_id)) <|
      (isOpenMap_inl.prodMap IsOpenMap.id).sum_elim (isOpenMap_inr.prodMap IsOpenMap.id)


/-- `X × (Y ⊕ Z)` is homeomorphic to `X × Y ⊕ X × Z`. -/
def prodSumDistrib : X × (Y ⊕ Z) ≃ₜ (X × Y) ⊕ (X × Z) :=
  (prodComm _ _).trans <| sumProdDistrib.trans <| sumCongr (prodComm _ _) (prodComm _ _)


/-- `(Σ i, X i) × Y` is homeomorphic to `Σ i, (X i × Y)`. -/
@[simps! apply symm_apply toEquiv]
def sigmaProdDistrib : (Σ i, X i) × Y ≃ₜ Σ i, X i × Y :=
  Homeomorph.symm <|
    homeomorphOfContinuousOpen (Equiv.sigmaProdDistrib X Y).symm
      (continuous_sigma fun _ => continuous_sigmaMk.fst'.prod_mk continuous_snd)
      (isOpenMap_sigma.2 fun _ => isOpenMap_sigmaMk.prodMap IsOpenMap.id)


/-- If `ι` has a unique element, then `ι → X` is homeomorphic to `X`. -/
@[simps! (config := .asFn)]
def funUnique (ι X : Type*) [Unique ι] [TopologicalSpace X] : (ι → X) ≃ₜ X where
  toEquiv := Equiv.funUnique ι X
  continuous_toFun := continuous_apply _
  continuous_invFun := continuous_pi fun _ => continuous_id


/-- Homeomorphism between dependent functions `Π i : Fin 2, X i` and `X 0 × X 1`. -/
@[simps! (config := .asFn)]
def piFinTwo.{u} (X : Fin 2 → Type u) [∀ i, TopologicalSpace (X i)] : (∀ i, X i) ≃ₜ X 0 × X 1 where
  toEquiv := piFinTwoEquiv X
  continuous_toFun := (continuous_apply 0).prod_mk (continuous_apply 1)
  continuous_invFun := continuous_pi <| Fin.forall_fin_two.2 ⟨continuous_fst, continuous_snd⟩


/-- Homeomorphism between `X² = Fin 2 → X` and `X × X`. -/
@[simps! (config := .asFn)]
def finTwoArrow : (Fin 2 → X) ≃ₜ X × X :=
  { piFinTwo fun _ => X with toEquiv := finTwoArrowEquiv X }


/-- A subset of a topological space is homeomorphic to its image under a homeomorphism.
-/
@[simps!]
def image (e : X ≃ₜ Y) (s : Set X) : s ≃ₜ e '' s where
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: by continuity!
  continuous_toFun := e.continuous.continuousOn.restrict_mapsTo (mapsTo_image _ _)
  continuous_invFun := (e.symm.continuous.comp continuous_subtype_val).codRestrict _
  toEquiv := e.toEquiv.image s


/-- `Set.univ X` is homeomorphic to `X`. -/
@[simps! (config := .asFn)]
def Set.univ (X : Type*) [TopologicalSpace X] : (univ : Set X) ≃ₜ X where
  toEquiv := Equiv.Set.univ X
  continuous_toFun := continuous_subtype_val
  continuous_invFun := continuous_id.subtype_mk _


/-- `s ×ˢ t` is homeomorphic to `s × t`. -/
@[simps!]
def Set.prod (s : Set X) (t : Set Y) : ↥(s ×ˢ t) ≃ₜ s × t where
  toEquiv := Equiv.Set.prod s t
  continuous_toFun :=
    (continuous_subtype_val.fst.subtype_mk _).prod_mk (continuous_subtype_val.snd.subtype_mk _)
  continuous_invFun :=
    (continuous_subtype_val.fst'.prod_mk continuous_subtype_val.snd').subtype_mk _


/-- The topological space `Π i, Y i` can be split as a product by separating the indices in ι
  depending on whether they satisfy a predicate p or not. -/
@[simps!]
def piEquivPiSubtypeProd (p : ι → Prop) (Y : ι → Type*) [∀ i, TopologicalSpace (Y i)]
    [DecidablePred p] : (∀ i, Y i) ≃ₜ (∀ i : { x // p x }, Y i) × ∀ i : { x // ¬p x }, Y i where
  toEquiv := Equiv.piEquivPiSubtypeProd p Y
  continuous_toFun := by
    /-
      X : Type u_1
      Y✝ : Type u_2
      W : Type u_3
      Z : Type u_4
      inst✝⁷ : TopologicalSpace X
      inst✝⁶ : TopologicalSpace Y✝
      inst✝⁵ : TopologicalSpace W
      inst✝⁴ : TopologicalSpace Z
      X' : Type u_5
      Y' : Type u_6
      inst✝³ : TopologicalSpace X'
      inst✝² : TopologicalSpace Y'
      ι : Type u_7
      p : ι → Prop
      Y : ι → Type u_8
      inst✝¹ : (i : ι) → TopologicalSpace (Y i)
      inst✝ : DecidablePred p
      ⊢ Continuous (Equiv.piEquivPiSubtypeProd p Y).toFun
    -/
                                 /-
                                   🎉 no goals
                                 -/
    apply Continuous.prod_mk <;> exact continuous_pi fun j => continuous_apply j.1
                                 /-
                                   🎉 no goals
                                 -/
  continuous_invFun :=
    continuous_pi fun j => by
      /-
        X : Type u_1
        Y✝ : Type u_2
        W : Type u_3
        Z : Type u_4
        inst✝⁷ : TopologicalSpace X
        inst✝⁶ : TopologicalSpace Y✝
        inst✝⁵ : TopologicalSpace W
        inst✝⁴ : TopologicalSpace Z
        X' : Type u_5
        Y' : Type u_6
        inst✝³ : TopologicalSpace X'
        inst✝² : TopologicalSpace Y'
        ι : Type u_7
        p : ι → Prop
        Y : ι → Type u_8
        inst✝¹ : (i : ι) → TopologicalSpace (Y i)
        inst✝ : DecidablePred p
        j : ι
        ⊢ Continuous fun a => (Equiv.piEquivPiSubtypeProd p Y).invFun a j
      -/
      dsimp only [Equiv.piEquivPiSubtypeProd]; split_ifs
      /-
        case pos
        X : Type u_1
        Y✝ : Type u_2
        W : Type u_3
        Z : Type u_4
        inst✝⁷ : TopologicalSpace X
        inst✝⁶ : TopologicalSpace Y✝
        inst✝⁵ : TopologicalSpace W
        inst✝⁴ : TopologicalSpace Z
        X' : Type u_5
        Y' : Type u_6
        inst✝³ : TopologicalSpace X'
        inst✝² : TopologicalSpace Y'
        ι : Type u_7
        p : ι → Prop
        Y : ι → Type u_8
        inst✝¹ : (i : ι) → TopologicalSpace (Y i)
        inst✝ : DecidablePred p
        j : ι
        h✝ : p j
        ⊢ Continuous fun a => a.1 ⟨j, h✝⟩
      -/
      exacts [(continuous_apply _).comp continuous_fst, (continuous_apply _).comp continuous_snd]
      /-
        🎉 no goals
      -/


/-- A product of topological spaces can be split as the binary product of one of the spaces and
  the product of all the remaining spaces. -/
@[simps!]
def piSplitAt (Y : ι → Type*) [∀ j, TopologicalSpace (Y j)] :
    (∀ j, Y j) ≃ₜ Y i × ∀ j : { j // j ≠ i }, Y j where
  toEquiv := Equiv.piSplitAt i Y
  continuous_toFun := (continuous_apply i).prod_mk (continuous_pi fun j => continuous_apply j.1)
  continuous_invFun :=
    continuous_pi fun j => by
      /-
        X : Type u_1
        Y✝ : Type u_2
        W : Type u_3
        Z : Type u_4
        inst✝⁷ : TopologicalSpace X
        inst✝⁶ : TopologicalSpace Y✝
        inst✝⁵ : TopologicalSpace W
        inst✝⁴ : TopologicalSpace Z
        X' : Type u_5
        Y' : Type u_6
        inst✝³ : TopologicalSpace X'
        inst✝² : TopologicalSpace Y'
        ι : Type u_7
        inst✝¹ : DecidableEq ι
        i : ι
        Y : ι → Type u_8
        inst✝ : (j : ι) → TopologicalSpace (Y j)
        j : ι
        ⊢ Continuous fun a => (Equiv.piSplitAt i Y).invFun a j
      -/
      dsimp only [Equiv.piSplitAt]
      /-
        X : Type u_1
        Y✝ : Type u_2
        W : Type u_3
        Z : Type u_4
        inst✝⁷ : TopologicalSpace X
        inst✝⁶ : TopologicalSpace Y✝
        inst✝⁵ : TopologicalSpace W
        inst✝⁴ : TopologicalSpace Z
        X' : Type u_5
        Y' : Type u_6
        inst✝³ : TopologicalSpace X'
        inst✝² : TopologicalSpace Y'
        ι : Type u_7
        inst✝¹ : DecidableEq ι
        i : ι
        Y : ι → Type u_8
        inst✝ : (j : ι) → TopologicalSpace (Y j)
        j : ι
        ⊢ Continuous fun a => dite (Eq j i) (fun h => Eq.rec a.1 ⋯) fun h => a.2 ⟨j, h⟩
      -/
      split_ifs with h
        /-
          case pos
          X : Type u_1
          Y✝ : Type u_2
          W : Type u_3
          Z : Type u_4
          inst✝⁷ : TopologicalSpace X
          inst✝⁶ : TopologicalSpace Y✝
          inst✝⁵ : TopologicalSpace W
          inst✝⁴ : TopologicalSpace Z
          X' : Type u_5
          Y' : Type u_6
          inst✝³ : TopologicalSpace X'
          inst✝² : TopologicalSpace Y'
          ι : Type u_7
          inst✝¹ : DecidableEq ι
          i : ι
          Y : ι → Type u_8
          inst✝ : (j : ι) → TopologicalSpace (Y j)
          j : ι
          h : Eq j i
          ⊢ Continuous fun a => Eq.rec a.1 ⋯
        -/
      · subst h
        /-
          case pos
          X : Type u_1
          Y✝ : Type u_2
          W : Type u_3
          Z : Type u_4
          inst✝⁷ : TopologicalSpace X
          inst✝⁶ : TopologicalSpace Y✝
          inst✝⁵ : TopologicalSpace W
          inst✝⁴ : TopologicalSpace Z
          X' : Type u_5
          Y' : Type u_6
          inst✝³ : TopologicalSpace X'
          inst✝² : TopologicalSpace Y'
          ι : Type u_7
          inst✝¹ : DecidableEq ι
          Y : ι → Type u_8
          inst✝ : (j : ι) → TopologicalSpace (Y j)
          j : ι
          ⊢ Continuous fun a => Eq.rec a.1 ⋯
        -/
        exact continuous_fst
        /-
          🎉 no goals
        -/
        /-
          case neg
          X : Type u_1
          Y✝ : Type u_2
          W : Type u_3
          Z : Type u_4
          inst✝⁷ : TopologicalSpace X
          inst✝⁶ : TopologicalSpace Y✝
          inst✝⁵ : TopologicalSpace W
          inst✝⁴ : TopologicalSpace Z
          X' : Type u_5
          Y' : Type u_6
          inst✝³ : TopologicalSpace X'
          inst✝² : TopologicalSpace Y'
          ι : Type u_7
          inst✝¹ : DecidableEq ι
          i : ι
          Y : ι → Type u_8
          inst✝ : (j : ι) → TopologicalSpace (Y j)
          j : ι
          h : Not (Eq j i)
          ⊢ Continuous fun a => a.2 ⟨j, h⟩
        -/
      · exact (continuous_apply _).comp continuous_snd
        /-
          🎉 no goals
        -/


/-- A product of copies of a topological space can be split as the binary product of one copy and
  the product of all the remaining copies. -/
@[simps!]
def funSplitAt : (ι → Y) ≃ₜ Y × ({ j // j ≠ i } → Y) :=
  piSplitAt i _


/-- An equiv between topological spaces respecting openness is a homeomorphism. -/
@[simps toEquiv]
def toHomeomorph (e : X ≃ Y) (he : ∀ s, IsOpen (e ⁻¹' s) ↔ IsOpen s) : X ≃ₜ Y where
  toEquiv := e
  continuous_toFun := continuous_def.2 fun _ ↦ (he _).2
                                                   /-
                                                     X : Type u_1
                                                     Y : Type u_2
                                                     W : Type u_3
                                                     Z✝ : Type u_4
                                                     Z : Type u_5
                                                     inst✝² : TopologicalSpace X
                                                     inst✝¹ : TopologicalSpace Y
                                                     inst✝ : TopologicalSpace Z
                                                     e : Equiv X Y
                                                     he : ∀ (s : Set Y), Iff (IsOpen (Set.preimage (⇑e) s)) (IsOpen s)
                                                     s : Set X
                                                     ⊢ IsOpen s → IsOpen (Set.preimage e.invFun s)
                                                   -/
  continuous_invFun := continuous_def.2 fun s ↦ by convert (he _).1; simp
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[simp] lemma coe_toHomeomorph (e : X ≃ Y) (he) : ⇑(e.toHomeomorph he) = e := rfl

lemma toHomeomorph_apply (e : X ≃ Y) (he) (x : X) : e.toHomeomorph he x = e x := rfl


@[simp] lemma toHomeomorph_refl :
  (Equiv.refl X).toHomeomorph (fun _s ↦ Iff.rfl) = Homeomorph.refl _ := rfl


@[simp] lemma toHomeomorph_symm (e : X ≃ Y) (he) :
                                                            /-
                                                              X : Type u_1
                                                              Y : Type u_2
                                                              W : Type u_3
                                                              Z✝ : Type u_4
                                                              Z : Type u_5
                                                              inst✝² : TopologicalSpace X
                                                              inst✝¹ : TopologicalSpace Y
                                                              inst✝ : TopologicalSpace Z
                                                              e : Equiv X Y
                                                              he : ∀ (s : Set Y), Iff (IsOpen (Set.preimage (⇑e) s)) (IsOpen s)
                                                              s : Set X
                                                              ⊢ Iff (IsOpen (Set.preimage (⇑e.symm) s)) (IsOpen s)
                                                            -/
  (e.toHomeomorph he).symm = e.symm.toHomeomorph fun s ↦ by convert (he _).symm; simp := rfl
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


lemma toHomeomorph_trans (e : X ≃ Y) (f : Y ≃ Z) (he hf) :
    (e.trans f).toHomeomorph (fun _s ↦ (he _).trans (hf _)) =
    (e.toHomeomorph he).trans (f.toHomeomorph hf) := rfl


/-- An inducing equiv between topological spaces is a homeomorphism. -/
@[simps toEquiv] -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: was `@[simps]`
def toHomeomorphOfIsInducing (f : X ≃ Y) (hf : IsInducing f) : X ≃ₜ Y :=
  { f with
    continuous_toFun := hf.continuous
                                                   /-
                                                     X : Type u_1
                                                     Y : Type u_2
                                                     W : Type u_3
                                                     Z✝ : Type u_4
                                                     Z : Type u_5
                                                     inst✝² : TopologicalSpace X
                                                     inst✝¹ : TopologicalSpace Y
                                                     inst✝ : TopologicalSpace Z
                                                     f : Equiv X Y
                                                     hf : Topology.IsInducing ⇑f
                                                     ⊢ Continuous (Function.comp (⇑f) f.invFun)
                                                   -/
    continuous_invFun := hf.continuous_iff.2 <| by simpa using continuous_id }
                                                   /-
                                                     🎉 no goals
                                                   -/


@[deprecated (since := "2024-10-28")] alias toHomeomorphOfInducing := toHomeomorphOfIsInducing


theorem continuous_symm_of_equiv_compact_to_t2 [CompactSpace X] [T2Space Y] {f : X ≃ Y}
    (hf : Continuous f) : Continuous f.symm := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : CompactSpace X
    inst✝ : T2Space Y
    f : Equiv X Y
    hf : Continuous ⇑f
    ⊢ Continuous ⇑f.symm
  -/
  rw [continuous_iff_isClosed]
  /-
    X : Type u_1
    Y : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : CompactSpace X
    inst✝ : T2Space Y
    f : Equiv X Y
    hf : Continuous ⇑f
    ⊢ ∀ (s : Set X), IsClosed s → IsClosed (Set.preimage (⇑f.symm) s)
  -/
  intro C hC
  /-
    X : Type u_1
    Y : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : CompactSpace X
    inst✝ : T2Space Y
    f : Equiv X Y
    hf : Continuous ⇑f
    C : Set X
    hC : IsClosed C
    ⊢ IsClosed (Set.preimage (⇑f.symm) C)
  -/
  have hC' : IsClosed (f '' C) := (hC.isCompact.image hf).isClosed
  /-
    X : Type u_1
    Y : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : CompactSpace X
    inst✝ : T2Space Y
    f : Equiv X Y
    hf : Continuous ⇑f
    C : Set X
    hC : IsClosed C
    hC' : IsClosed (Set.image (⇑f) C)
    ⊢ IsClosed (Set.preimage (⇑f.symm) C)
  -/
  rwa [Equiv.image_eq_preimage] at hC'
  /-
    🎉 no goals
  -/


/-- Continuous equivalences from a compact space to a T2 space are homeomorphisms.

This is not true when T2 is weakened to T1
(see `Continuous.homeoOfEquivCompactToT2.t1_counterexample`). -/
@[simps toEquiv] -- Porting note: was `@[simps]`
def homeoOfEquivCompactToT2 [CompactSpace X] [T2Space Y] {f : X ≃ Y} (hf : Continuous f) : X ≃ₜ Y :=
  { f with
    continuous_toFun := hf
    continuous_invFun := hf.continuous_symm_of_equiv_compact_to_t2 }


/-- Predicate saying that `f` is a homeomorphism.

This should be used only when `f` is a concrete function whose continuous inverse is not easy to
write down. Otherwise, `Homeomorph` should be preferred as it bundles the continuous inverse.

Having both `Homeomorph` and `IsHomeomorph` is justified by the fact that so many function
properties are unbundled in the topology part of the library, and by the fact that a homeomorphism
is not merely a continuous bijection, that is `IsHomeomorph f` is not equivalent to
`Continuous f ∧ Bijective f` but to `Continuous f ∧ Bijective f ∧ IsOpenMap f`. -/
structure IsHomeomorph (f : X → Y) : Prop where
  continuous : Continuous f
  isOpenMap : IsOpenMap f
  bijective : Bijective f


protected theorem Homeomorph.isHomeomorph (h : X ≃ₜ Y) : IsHomeomorph h :=
  ⟨h.continuous, h.isOpenMap, h.bijective⟩


protected lemma injective : Function.Injective f := hf.bijective.injective

protected lemma surjective : Function.Surjective f := hf.bijective.surjective


variable (f) in
/-- Bundled homeomorphism constructed from a map that is a homeomorphism. -/
@[simps! toEquiv apply symm_apply]
noncomputable def homeomorph : X ≃ₜ Y where
  continuous_toFun := hf.1
  continuous_invFun := by
    /-
      X : Type u_1
      Y : Type u_2
      W✝ : Type u_3
      Z : Type u_4
      inst✝³ : TopologicalSpace X
      inst✝² : TopologicalSpace Y
      inst✝¹ : TopologicalSpace Z
      W : Type u_5
      inst✝ : TopologicalSpace W
      f : X → Y
      hf : IsHomeomorph f
      ⊢ Continuous (Equiv.ofBijective f ⋯).invFun
    -/
    rw [continuous_iff_continuousOn_univ, ← hf.bijective.2.range_eq]
    /-
      X : Type u_1
      Y : Type u_2
      W✝ : Type u_3
      Z : Type u_4
      inst✝³ : TopologicalSpace X
      inst✝² : TopologicalSpace Y
      inst✝¹ : TopologicalSpace Z
      W : Type u_5
      inst✝ : TopologicalSpace W
      f : X → Y
      hf : IsHomeomorph f
      ⊢ ContinuousOn (Equiv.ofBijective f ⋯).invFun (Set.range f)
    -/
    exact hf.isOpenMap.continuousOn_range_of_leftInverse (leftInverse_surjInv hf.bijective)
    /-
      🎉 no goals
    -/
  toEquiv := Equiv.ofBijective f hf.bijective


protected lemma isClosedMap : IsClosedMap f := (hf.homeomorph f).isClosedMap

lemma isInducing : IsInducing f := (hf.homeomorph f).isInducing

lemma isQuotientMap : IsQuotientMap f := (hf.homeomorph f).isQuotientMap

lemma isEmbedding : IsEmbedding f := (hf.homeomorph f).isEmbedding

lemma isOpenEmbedding : IsOpenEmbedding f := (hf.homeomorph f).isOpenEmbedding

lemma isClosedEmbedding : IsClosedEmbedding f := (hf.homeomorph f).isClosedEmbedding

lemma isDenseEmbedding : IsDenseEmbedding f := (hf.homeomorph f).isDenseEmbedding


@[deprecated (since := "2024-10-20")] alias closedEmbedding := isClosedEmbedding

/-- A map is a homeomorphism iff it is the map underlying a bundled homeomorphism `h : X ≃ₜ Y`. -/
lemma isHomeomorph_iff_exists_homeomorph : IsHomeomorph f ↔ ∃ h : X ≃ₜ Y, h = f :=
  ⟨fun hf => ⟨hf.homeomorph f, rfl⟩, fun ⟨h, h'⟩ => h' ▸ h.isHomeomorph⟩


/-- A map is a homeomorphism iff it is continuous and has a continuous inverse. -/
lemma isHomeomorph_iff_exists_inverse : IsHomeomorph f ↔ Continuous f ∧ ∃ g : Y → X,
    LeftInverse g f ∧ RightInverse g f ∧ Continuous g := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    ⊢ Iff (IsHomeomorph f) (And (Continuous f) (Exists fun g => And (Function.Left …
  -/
  refine ⟨fun hf ↦ ⟨hf.continuous, ?_⟩, fun ⟨hf, g, hg⟩ ↦ ?_⟩
    /-
      case refine_1
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      hf : IsHomeomorph f
      ⊢ Exists fun g => And (Function.LeftInverse g f) (And (Function.RightInverse g …
    -/
  · let h := hf.homeomorph f
    /-
      case refine_1
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      hf : IsHomeomorph f
      h : Homeomorph X Y := IsHomeomorph.homeomorph f hf
      ⊢ Exists fun g => And (Function.LeftInverse g f) (And (Function.RightInverse g …
    -/
    exact ⟨h.symm, h.left_inv, h.right_inv, h.continuous_invFun⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      x✝ : And (Continuous f) (Exists fun g => And (Function.LeftInverse g f) (And ( …
      hf : Continuous f
      g : Y → X
      hg : And (Function.LeftInverse g f) (And (Function.RightInverse g f) (Continuo …
      ⊢ IsHomeomorph f
    -/
  · exact (Homeomorph.mk ⟨f, g, hg.1, hg.2.1⟩ hf hg.2.2).isHomeomorph
    /-
      🎉 no goals
    -/


/-- A map is a homeomorphism iff it is a surjective embedding. -/
lemma isHomeomorph_iff_isEmbedding_surjective : IsHomeomorph f ↔ IsEmbedding f ∧ Surjective f where
  mp hf := ⟨hf.isEmbedding, hf.surjective⟩
  mpr h := ⟨h.1.continuous, ((isOpenEmbedding_iff f).2 ⟨h.1, h.2.range_eq ▸ isOpen_univ⟩).isOpenMap,
    h.1.injective, h.2⟩


@[deprecated (since := "2024-10-26")]
alias isHomeomorph_iff_embedding_surjective := isHomeomorph_iff_isEmbedding_surjective


/-- A map is a homeomorphism iff it is continuous, closed and bijective. -/
lemma isHomeomorph_iff_continuous_isClosedMap_bijective  : IsHomeomorph f ↔
    Continuous f ∧ IsClosedMap f ∧ Function.Bijective f :=
  ⟨fun hf => ⟨hf.continuous, hf.isClosedMap, hf.bijective⟩, fun ⟨hf, hf', hf''⟩ =>
    ⟨hf, fun _ hu => isClosed_compl_iff.1 (image_compl_eq hf'' ▸ hf' _ hu.isClosed_compl), hf''⟩⟩


/-- A map from a compact space to a T2 space is a homeomorphism iff it is continuous and
  bijective. -/
lemma isHomeomorph_iff_continuous_bijective [CompactSpace X] [T2Space Y] :
    IsHomeomorph f ↔ Continuous f ∧ Bijective f := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    f : X → Y
    inst✝¹ : CompactSpace X
    inst✝ : T2Space Y
    ⊢ Iff (IsHomeomorph f) (And (Continuous f) (Function.Bijective f))
  -/
  rw [isHomeomorph_iff_continuous_isClosedMap_bijective]
  /-
    X : Type u_1
    Y : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    f : X → Y
    inst✝¹ : CompactSpace X
    inst✝ : T2Space Y
    ⊢ Iff (And (Continuous f) (And (IsClosedMap f) (Function.Bijective f))) (And ( …
  -/
  refine and_congr_right fun hf ↦ ?_
  /-
    X : Type u_1
    Y : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    f : X → Y
    inst✝¹ : CompactSpace X
    inst✝ : T2Space Y
    hf : Continuous f
    ⊢ Iff (And (IsClosedMap f) (Function.Bijective f)) (Function.Bijective f)
  -/
  rw [eq_true hf.isClosedMap, true_and]
  /-
    🎉 no goals
  -/


protected lemma IsHomeomorph.id : IsHomeomorph (@id X) := ⟨continuous_id, .id, bijective_id⟩


lemma IsHomeomorph.comp {g : Y → Z} (hg : IsHomeomorph g) (hf : IsHomeomorph f) :
    IsHomeomorph (g ∘ f) := ⟨hg.1.comp hf.1, hg.2.comp hf.2, hg.3.comp hf.3⟩


lemma IsHomeomorph.sumMap {g : Z → W} (hf : IsHomeomorph f) (hg : IsHomeomorph g) :
    IsHomeomorph (Sum.map f g) := ⟨hf.1.sum_map hg.1, hf.2.sumMap hg.2, hf.3.sum_map hg.3⟩


lemma IsHomeomorph.prodMap {g : Z → W} (hf : IsHomeomorph f) (hg : IsHomeomorph g) :
    IsHomeomorph (Prod.map f g) := ⟨hf.1.prodMap hg.1, hf.2.prodMap hg.2, hf.3.prodMap hg.3⟩


lemma IsHomeomorph.sigmaMap {ι κ : Type*} {X : ι → Type*} {Y : κ → Type*}
    [∀ i, TopologicalSpace (X i)] [∀ i, TopologicalSpace (Y i)] {f : ι → κ}
    (hf : Bijective f) {g : (i : ι) → X i → Y (f i)} (hg : ∀ i, IsHomeomorph (g i)) :
    IsHomeomorph (Sigma.map f g) := by
  /-
    ι : Type u_6
    κ : Type u_7
    X : ι → Type u_8
    Y : κ → Type u_9
    inst✝¹ : (i : ι) → TopologicalSpace (X i)
    inst✝ : (i : κ) → TopologicalSpace (Y i)
    f : ι → κ
    hf : Function.Bijective f
    g : (i : ι) → X i → Y (f i)
    hg : ∀ (i : ι), IsHomeomorph (g i)
    ⊢ IsHomeomorph (Sigma.map f g)
  -/
  simp_rw [isHomeomorph_iff_isEmbedding_surjective,] at hg ⊢
  /-
    ι : Type u_6
    κ : Type u_7
    X : ι → Type u_8
    Y : κ → Type u_9
    inst✝¹ : (i : ι) → TopologicalSpace (X i)
    inst✝ : (i : κ) → TopologicalSpace (Y i)
    f : ι → κ
    hf : Function.Bijective f
    g : (i : ι) → X i → Y (f i)
    hg : ∀ (i : ι), And (Topology.IsEmbedding (g i)) (Function.Surjective (g i))
    ⊢ And (Topology.IsEmbedding (Sigma.map f g)) (Function.Surjective (Sigma.map f …
  -/
  exact ⟨(isEmbedding_sigmaMap hf.1).2 fun i ↦ (hg i).1, hf.2.sigma_map fun i ↦ (hg i).2⟩
  /-
    🎉 no goals
  -/


lemma IsHomeomorph.pi_map {ι : Type*} {X Y : ι → Type*} [∀ i, TopologicalSpace (X i)]
    [∀ i, TopologicalSpace (Y i)] {f : (i : ι) → X i → Y i} (h : ∀ i, IsHomeomorph (f i)) :
    IsHomeomorph (fun (x : ∀ i, X i) i ↦ f i (x i)) :=
  (Homeomorph.piCongrRight fun i ↦ (h i).homeomorph (f i)).isHomeomorph


/-- `HomeomorphClass F A B` states that `F` is a type of homeomorphisms.-/
class HomeomorphClass (F : Type*) (A B : outParam Type*)
    [TopologicalSpace A] [TopologicalSpace B] [h : EquivLike F A B] : Prop where
  map_continuous : ∀ (f : F), Continuous f
  inv_continuous : ∀ (f : F), Continuous (h.inv f)


/-- Turn an element of a type `F` satisfying `HomeomorphClass F α β` into an actual
`Homeomorph`. This is declared as the default coercion from `F` to `α ≃ₜ β`. -/
@[coe]
def toHomeomorph [h : HomeomorphClass F α β] (f : F) : α ≃ₜ β :=
  { (f : α ≃ β) with
    continuous_toFun := h.map_continuous f
    continuous_invFun := h.inv_continuous f }


@[simp]
theorem coe_coe [h : HomeomorphClass F α β] (f : F) : ⇑(h.toHomeomorph f) = ⇑f := rfl


instance [HomeomorphClass F α β] : CoeOut F (α ≃ₜ β) :=
  ⟨HomeomorphClass.toHomeomorph⟩


theorem toHomeomorph_injective [HomeomorphClass F α β] : Function.Injective ((↑) : F → α ≃ₜ β) :=
  fun _ _ e ↦ DFunLike.ext _ _ fun a ↦ congr_arg (fun e : α ≃ₜ β ↦ e.toFun a) e


instance [HomeomorphClass F α β] : ContinuousMapClass F α β where
  map_continuous  f := map_continuous f


instance : HomeomorphClass (α ≃ₜ β) α β where
  map_continuous e := e.continuous_toFun
  inv_continuous e := e.continuous_invFun


