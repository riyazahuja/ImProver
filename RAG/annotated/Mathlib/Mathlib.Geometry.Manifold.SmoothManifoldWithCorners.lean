/-- A structure containing information on the way a space `H` embeds in a
model vector space `E` over the field `𝕜`. This is all what is needed to
define a smooth manifold with model space `H`, and model vector space `E`.

We require two conditions `uniqueDiffOn'` and `target_subset_closure_interior`, which
are satisfied in the relevant cases (where `range I = univ` or a half space or a quadrant) and
useful for technical reasons. The former makes sure that manifold derivatives are uniquely
defined, the latter ensures that for `C^2` maps the second derivatives are symmetric even for points
on the boundary, as these are limit points of interior points where symmetry holds. If further
conditions turn out to be useful, they can be added here.
-/
@[ext] -- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): was nolint has_nonempty_instance
structure ModelWithCorners (𝕜 : Type*) [NontriviallyNormedField 𝕜] (E : Type*)
    [NormedAddCommGroup E] [NormedSpace 𝕜 E] (H : Type*) [TopologicalSpace H] extends
    PartialEquiv H E where
  source_eq : source = univ
  uniqueDiffOn' : UniqueDiffOn 𝕜 toPartialEquiv.target
  target_subset_closure_interior : toPartialEquiv.target ⊆ closure (interior toPartialEquiv.target)
  continuous_toFun : Continuous toFun := by continuity
  continuous_invFun : Continuous invFun := by continuity


/-- A vector space is a model with corners. -/
def modelWithCornersSelf (𝕜 : Type*) [NontriviallyNormedField 𝕜] (E : Type*)
    [NormedAddCommGroup E] [NormedSpace 𝕜 E] : ModelWithCorners 𝕜 E E where
  toPartialEquiv := PartialEquiv.refl E
  source_eq := rfl
  uniqueDiffOn' := uniqueDiffOn_univ
                                       /-
                                         𝕜 : Type u_1
                                         inst✝² : NontriviallyNormedField 𝕜
                                         E : Type u_2
                                         inst✝¹ : NormedAddCommGroup E
                                         inst✝ : NormedSpace 𝕜 E
                                         ⊢ HasSubset.Subset (PartialEquiv.refl E).target (closure (interior (PartialEqu …
                                       -/
  target_subset_closure_interior := by simp
                                       /-
                                         🎉 no goals
                                       -/
  continuous_toFun := continuous_id
  continuous_invFun := continuous_id


@[inherit_doc] scoped[Manifold] notation "𝓘(" 𝕜 ", " E ")" => modelWithCornersSelf 𝕜 E


/-- A normed field is a model with corners. -/
scoped[Manifold] notation "𝓘(" 𝕜 ")" => modelWithCornersSelf 𝕜 𝕜


/-- Coercion of a model with corners to a function. We don't use `e.toFun` because it is actually
`e.toPartialEquiv.toFun`, so `simp` will apply lemmas about `toPartialEquiv`. While we may want to
switch to this behavior later, doing it mid-port will break a lot of proofs. -/
@[coe] def toFun' (e : ModelWithCorners 𝕜 E H) : H → E := e.toFun


instance : CoeFun (ModelWithCorners 𝕜 E H) fun _ => H → E := ⟨toFun'⟩


/-- The inverse to a model with corners, only registered as a `PartialEquiv`. -/
protected def symm : PartialEquiv E H :=
  I.toPartialEquiv.symm


/-- See Note [custom simps projection]. We need to specify this projection explicitly in this case,
  because it is a composition of multiple projections. -/
def Simps.apply (𝕜 : Type*) [NontriviallyNormedField 𝕜] (E : Type*) [NormedAddCommGroup E]
    [NormedSpace 𝕜 E] (H : Type*) [TopologicalSpace H] (I : ModelWithCorners 𝕜 E H) : H → E :=
  I


/-- See Note [custom simps projection] -/
def Simps.symm_apply (𝕜 : Type*) [NontriviallyNormedField 𝕜] (E : Type*) [NormedAddCommGroup E]
    [NormedSpace 𝕜 E] (H : Type*) [TopologicalSpace H] (I : ModelWithCorners 𝕜 E H) : E → H :=
  I.symm


@[simp, mfld_simps]
theorem toPartialEquiv_coe : (I.toPartialEquiv : H → E) = I :=
  rfl


@[simp, mfld_simps]
theorem mk_coe (e : PartialEquiv H E) (a b c d d') :
    ((ModelWithCorners.mk e a b c d d' : ModelWithCorners 𝕜 E H) : H → E) = (e : H → E) :=
  rfl


@[simp, mfld_simps]
theorem toPartialEquiv_coe_symm : (I.toPartialEquiv.symm : E → H) = I.symm :=
  rfl


@[simp, mfld_simps]
theorem mk_symm (e : PartialEquiv H E) (a b c d d') :
    (ModelWithCorners.mk e a b c d d' : ModelWithCorners 𝕜 E H).symm = e.symm :=
  rfl


@[continuity]
protected theorem continuous : Continuous I :=
  I.continuous_toFun


protected theorem continuousAt {x} : ContinuousAt I x :=
  I.continuous.continuousAt


protected theorem continuousWithinAt {s x} : ContinuousWithinAt I s x :=
  I.continuousAt.continuousWithinAt


@[continuity]
theorem continuous_symm : Continuous I.symm :=
  I.continuous_invFun


theorem continuousAt_symm {x} : ContinuousAt I.symm x :=
  I.continuous_symm.continuousAt


theorem continuousWithinAt_symm {s x} : ContinuousWithinAt I.symm s x :=
  I.continuous_symm.continuousWithinAt


theorem continuousOn_symm {s} : ContinuousOn I.symm s :=
  I.continuous_symm.continuousOn


@[simp, mfld_simps]
theorem target_eq : I.target = range (I : H → E) := by
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    ⊢ Eq I.target (Set.range ↑I)
  -/
  rw [← image_univ, ← I.source_eq]
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    ⊢ Eq I.target (Set.image (↑I) I.source)
  -/
  exact I.image_source_eq_target.symm
  /-
    🎉 no goals
  -/


protected theorem uniqueDiffOn : UniqueDiffOn 𝕜 (range I) :=
  I.target_eq ▸ I.uniqueDiffOn'


@[deprecated (since := "2024-09-30")]
protected alias unique_diff := ModelWithCorners.uniqueDiffOn


theorem range_subset_closure_interior : range I ⊆ closure (interior (range I)) := by
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    ⊢ HasSubset.Subset (Set.range ↑I) (closure (interior (Set.range ↑I)))
  -/
  rw [← I.target_eq]
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    ⊢ HasSubset.Subset I.target (closure (interior I.target))
  -/
  exact I.target_subset_closure_interior
  /-
    🎉 no goals
  -/


@[simp, mfld_simps]
                                                            /-
                                                              𝕜 : Type u_1
                                                              inst✝³ : NontriviallyNormedField 𝕜
                                                              E : Type u_2
                                                              inst✝² : NormedAddCommGroup E
                                                              inst✝¹ : NormedSpace 𝕜 E
                                                              H : Type u_3
                                                              inst✝ : TopologicalSpace H
                                                              I : ModelWithCorners 𝕜 E H
                                                              x : H
                                                              ⊢ Eq (↑I.symm (↑I x)) x
                                                            -/
protected theorem left_inv (x : H) : I.symm (I x) = x := by refine I.left_inv' ?_; simp
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


protected theorem leftInverse : LeftInverse I.symm I :=
  I.left_inv


theorem injective : Injective I :=
  I.leftInverse.injective


@[simp, mfld_simps]
theorem symm_comp_self : I.symm ∘ I = id :=
  I.leftInverse.comp_eq_id


protected theorem rightInvOn : RightInvOn I.symm I (range I) :=
  I.leftInverse.rightInvOn_range


@[simp, mfld_simps]
protected theorem right_inv {x : E} (hx : x ∈ range I) : I (I.symm x) = x :=
  I.rightInvOn hx


theorem preimage_image (s : Set H) : I ⁻¹' (I '' s) = s :=
  I.injective.preimage_image s


protected theorem image_eq (s : Set H) : I '' s = I.symm ⁻¹' s ∩ range I := by
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    s : Set H
    ⊢ Eq (Set.image (↑I) s) (Inter.inter (Set.preimage (↑I.symm) s) (Set.range ↑I))
  -/
  refine (I.toPartialEquiv.image_eq_target_inter_inv_preimage ?_).trans ?_
    /-
      case refine_1
      𝕜 : Type u_1
      inst✝³ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      s : Set H
      ⊢ HasSubset.Subset s I.source
    -/
  · rw [I.source_eq]; exact subset_univ _
                      /-
                        🎉 no goals
                      -/
    /-
      case refine_2
      𝕜 : Type u_1
      inst✝³ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      s : Set H
      ⊢ Eq (Inter.inter I.target (Set.preimage (↑I.symm) s)) (Inter.inter (Set.preim …
    -/
  · rw [inter_comm, I.target_eq, I.toPartialEquiv_coe_symm]
    /-
      🎉 no goals
    -/


theorem isClosedEmbedding : IsClosedEmbedding I :=
  I.leftInverse.isClosedEmbedding I.continuous_symm I.continuous


@[deprecated (since := "2024-10-20")]
alias closedEmbedding := isClosedEmbedding


theorem isClosed_range : IsClosed (range I) :=
  I.isClosedEmbedding.isClosed_range


@[deprecated (since := "2024-03-17")] alias closed_range := isClosed_range


theorem range_eq_closure_interior : range I = closure (interior (range I)) :=
  Subset.antisymm I.range_subset_closure_interior I.isClosed_range.closure_interior_subset


theorem map_nhds_eq (x : H) : map I (𝓝 x) = 𝓝[range I] I x :=
  I.isClosedEmbedding.isEmbedding.map_nhds_eq x


theorem map_nhdsWithin_eq (s : Set H) (x : H) : map I (𝓝[s] x) = 𝓝[I '' s] I x :=
  I.isClosedEmbedding.isEmbedding.map_nhdsWithin_eq s x


theorem image_mem_nhdsWithin {x : H} {s : Set H} (hs : s ∈ 𝓝 x) : I '' s ∈ 𝓝[range I] I x :=
  I.map_nhds_eq x ▸ image_mem_map hs


theorem symm_map_nhdsWithin_image {x : H} {s : Set H} : map I.symm (𝓝[I '' s] I x) = 𝓝[s] x := by
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    x : H
    s : Set H
    ⊢ Eq (Filter.map (↑I.symm) (nhdsWithin (↑I x) (Set.image (↑I) s))) (nhdsWithin …
  -/
  rw [← I.map_nhdsWithin_eq, map_map, I.symm_comp_self, map_id]
  /-
    🎉 no goals
  -/


theorem symm_map_nhdsWithin_range (x : H) : map I.symm (𝓝[range I] I x) = 𝓝 x := by
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    x : H
    ⊢ Eq (Filter.map (↑I.symm) (nhdsWithin (↑I x) (Set.range ↑I))) (nhds x)
  -/
  rw [← I.map_nhds_eq, map_map, I.symm_comp_self, map_id]
  /-
    🎉 no goals
  -/


theorem uniqueDiffOn_preimage {s : Set H} (hs : IsOpen s) :
    UniqueDiffOn 𝕜 (I.symm ⁻¹' s ∩ range I) := by
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    s : Set H
    hs : IsOpen s
    ⊢ UniqueDiffOn 𝕜 (Inter.inter (Set.preimage (↑I.symm) s) (Set.range ↑I))
  -/
  rw [inter_comm]
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    s : Set H
    hs : IsOpen s
    ⊢ UniqueDiffOn 𝕜 (Inter.inter (Set.range ↑I) (Set.preimage (↑I.symm) s))
  -/
  exact I.uniqueDiffOn.inter (hs.preimage I.continuous_invFun)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-09-30")]
alias unique_diff_preimage := uniqueDiffOn_preimage


theorem uniqueDiffOn_preimage_source {β : Type*} [TopologicalSpace β] {e : PartialHomeomorph H β} :
    UniqueDiffOn 𝕜 (I.symm ⁻¹' e.source ∩ range I) :=
  I.uniqueDiffOn_preimage e.open_source


@[deprecated (since := "2024-09-30")]
alias unique_diff_preimage_source := uniqueDiffOn_preimage_source


theorem uniqueDiffWithinAt_image {x : H} : UniqueDiffWithinAt 𝕜 (range I) (I x) :=
  I.uniqueDiffOn _ (mem_range_self _)


@[deprecated (since := "2024-09-30")]
alias unique_diff_at_image := uniqueDiffWithinAt_image


theorem symm_continuousWithinAt_comp_right_iff {X} [TopologicalSpace X] {f : H → X} {s : Set H}
    {x : H} :
    ContinuousWithinAt (f ∘ I.symm) (I.symm ⁻¹' s ∩ range I) (I x) ↔ ContinuousWithinAt f s x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    X : Type u_4
    inst✝ : TopologicalSpace X
    f : H → X
    s : Set H
    x : H
    ⊢ Iff (ContinuousWithinAt (Function.comp f ↑I.symm) (Inter.inter (Set.preimage …
  -/
  refine ⟨fun h => ?_, fun h => ?_⟩
    /-
      case refine_1
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_3
      inst✝¹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      X : Type u_4
      inst✝ : TopologicalSpace X
      f : H → X
      s : Set H
      x : H
      h : ContinuousWithinAt (Function.comp f ↑I.symm) (Inter.inter (Set.preimage (↑ …
      ⊢ ContinuousWithinAt f s x
    -/
  · have := h.comp I.continuousWithinAt (mapsTo_preimage _ _)
    simp_rw [preimage_inter, preimage_preimage, I.left_inv, preimage_id', preimage_range,
      inter_univ] at this
    /-
      case refine_1
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_3
      inst✝¹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      X : Type u_4
      inst✝ : TopologicalSpace X
      f : H → X
      s : Set H
      x : H
      h : ContinuousWithinAt (Function.comp f ↑I.symm) (Inter.inter (Set.preimage (↑ …
      this : ContinuousWithinAt (Function.comp (Function.comp f ↑I.symm) ↑I) s x
      ⊢ ContinuousWithinAt f s x
    -/
    rwa [Function.comp_assoc, I.symm_comp_self] at this
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_3
      inst✝¹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      X : Type u_4
      inst✝ : TopologicalSpace X
      f : H → X
      s : Set H
      x : H
      h : ContinuousWithinAt f s x
      ⊢ ContinuousWithinAt (Function.comp f ↑I.symm) (Inter.inter (Set.preimage (↑I. …
    -/
  · rw [← I.left_inv x] at h; exact h.comp I.continuousWithinAt_symm inter_subset_left
                              /-
                                🎉 no goals
                              -/


protected theorem locallyCompactSpace [LocallyCompactSpace E] (I : ModelWithCorners 𝕜 E H) :
    LocallyCompactSpace H := by
  have : ∀ x : H, (𝓝 x).HasBasis (fun s => s ∈ 𝓝 (I x) ∧ IsCompact s)
      fun s => I.symm '' (s ∩ range I) := fun x ↦ by
    rw [← I.symm_map_nhdsWithin_range]
    exact ((compact_basis_nhds (I x)).inf_principal _).map _
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹ : TopologicalSpace H
    inst✝ : LocallyCompactSpace E
    I : ModelWithCorners 𝕜 E H
    this : ∀ (x : H), (nhds x).HasBasis (fun s => And (Membership.mem (nhds (↑I x) …
    ⊢ LocallyCompactSpace H
  -/
  refine .of_hasBasis this ?_
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹ : TopologicalSpace H
    inst✝ : LocallyCompactSpace E
    I : ModelWithCorners 𝕜 E H
    this : ∀ (x : H), (nhds x).HasBasis (fun s => And (Membership.mem (nhds (↑I x) …
    ⊢ ∀ (x : H) (i : Set E), And (Membership.mem (nhds (↑I x)) i) (IsCompact i) →  …
  -/
  rintro x s ⟨-, hsc⟩
  /-
    case intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    H : Type u_3
    inst✝¹ : TopologicalSpace H
    inst✝ : LocallyCompactSpace E
    I : ModelWithCorners 𝕜 E H
    this : ∀ (x : H), (nhds x).HasBasis (fun s => And (Membership.mem (nhds (↑I x) …
    x : H
    s : Set E
    hsc : IsCompact s
    ⊢ IsCompact (Set.image (↑I.symm) (Inter.inter s (Set.range ↑I)))
  -/
  exact (hsc.inter_right I.isClosed_range).image I.continuous_symm
  /-
    🎉 no goals
  -/


protected theorem secondCountableTopology [SecondCountableTopology E] (I : ModelWithCorners 𝕜 E H) :
    SecondCountableTopology H :=
  I.isClosedEmbedding.isEmbedding.secondCountableTopology


include I in
/-- Every manifold is a Fréchet space (T1 space) -- regardless of whether it is
Hausdorff. -/
protected theorem t1Space (M : Type*) [TopologicalSpace M] [ChartedSpace H M] : T1Space M := by
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝² : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    ⊢ T1Space M
  -/
  have : T2Space H := I.isClosedEmbedding.toIsEmbedding.t2Space
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝² : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    this : T2Space H
    ⊢ T1Space M
  -/
  exact ChartedSpace.t1Space H M
  /-
    🎉 no goals
  -/


/-- In the trivial model with corners, the associated `PartialEquiv` is the identity. -/
@[simp, mfld_simps]
theorem modelWithCornersSelf_partialEquiv : 𝓘(𝕜, E).toPartialEquiv = PartialEquiv.refl E :=
  rfl


@[simp, mfld_simps]
theorem modelWithCornersSelf_coe : (𝓘(𝕜, E) : E → E) = id :=
  rfl


@[simp, mfld_simps]
theorem modelWithCornersSelf_coe_symm : (𝓘(𝕜, E).symm : E → E) = id :=
  rfl


/-- Given two model_with_corners `I` on `(E, H)` and `I'` on `(E', H')`, we define the model with
corners `I.prod I'` on `(E × E', ModelProd H H')`. This appears in particular for the manifold
structure on the tangent bundle to a manifold modelled on `(E, H)`: it will be modelled on
`(E × E, H × E)`. See note [Manifold type tags] for explanation about `ModelProd H H'`
vs `H × H'`. -/
@[simps (config := .lemmasOnly)]
def ModelWithCorners.prod {𝕜 : Type u} [NontriviallyNormedField 𝕜] {E : Type v}
    [NormedAddCommGroup E] [NormedSpace 𝕜 E] {H : Type w} [TopologicalSpace H]
    (I : ModelWithCorners 𝕜 E H) {E' : Type v'} [NormedAddCommGroup E'] [NormedSpace 𝕜 E']
    {H' : Type w'} [TopologicalSpace H'] (I' : ModelWithCorners 𝕜 E' H') :
    ModelWithCorners 𝕜 (E × E') (ModelProd H H') :=
  { I.toPartialEquiv.prod I'.toPartialEquiv with
    toFun := fun x => (I x.1, I' x.2)
    invFun := fun x => (I.symm x.1, I'.symm x.2)
    source := { x | x.1 ∈ I.source ∧ x.2 ∈ I'.source }
                    /-
                      𝕜 : Type u
                      inst✝⁶ : NontriviallyNormedField 𝕜
                      E : Type v
                      inst✝⁵ : NormedAddCommGroup E
                      inst✝⁴ : NormedSpace 𝕜 E
                      H : Type w
                      inst✝³ : TopologicalSpace H
                      I : ModelWithCorners 𝕜 E H
                      E' : Type v'
                      inst✝² : NormedAddCommGroup E'
                      inst✝¹ : NormedSpace 𝕜 E'
                      H' : Type w'
                      inst✝ : TopologicalSpace H'
                      I' : ModelWithCorners 𝕜 E' H'
                      ⊢ Eq { toFun := fun x => { fst := ↑I x.1, snd := ↑I' x.2 }, invFun := fun x => …
                    -/
    source_eq := by simp only [setOf_true, mfld_simps]
                    /-
                      🎉 no goals
                    -/
    uniqueDiffOn' := I.uniqueDiffOn'.prod I'.uniqueDiffOn'
    target_subset_closure_interior := by
      /-
        𝕜 : Type u
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type v
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        H : Type w
        inst✝³ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        E' : Type v'
        inst✝² : NormedAddCommGroup E'
        inst✝¹ : NormedSpace 𝕜 E'
        H' : Type w'
        inst✝ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E' H'
        ⊢ HasSubset.Subset { toFun := fun x => { fst := ↑I x.1, snd := ↑I' x.2 }, invF …
      -/
      simp only [PartialEquiv.prod_target, target_eq, interior_prod_eq, closure_prod_eq]
      /-
        𝕜 : Type u
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type v
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        H : Type w
        inst✝³ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        E' : Type v'
        inst✝² : NormedAddCommGroup E'
        inst✝¹ : NormedSpace 𝕜 E'
        H' : Type w'
        inst✝ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E' H'
        ⊢ HasSubset.Subset (SProd.sprod (Set.range ↑I) (Set.range ↑I')) (SProd.sprod ( …
      -/
      exact Set.prod_mono I.range_subset_closure_interior I'.range_subset_closure_interior
      /-
        🎉 no goals
      -/
    continuous_toFun := I.continuous_toFun.prodMap I'.continuous_toFun
    continuous_invFun := I.continuous_invFun.prodMap I'.continuous_invFun }


/-- Given a finite family of `ModelWithCorners` `I i` on `(E i, H i)`, we define the model with
corners `pi I` on `(Π i, E i, ModelPi H)`. See note [Manifold type tags] for explanation about
`ModelPi H`. -/
def ModelWithCorners.pi {𝕜 : Type u} [NontriviallyNormedField 𝕜] {ι : Type v} [Fintype ι]
    {E : ι → Type w} [∀ i, NormedAddCommGroup (E i)] [∀ i, NormedSpace 𝕜 (E i)] {H : ι → Type u'}
    [∀ i, TopologicalSpace (H i)] (I : ∀ i, ModelWithCorners 𝕜 (E i) (H i)) :
    ModelWithCorners 𝕜 (∀ i, E i) (ModelPi H) where
  toPartialEquiv := PartialEquiv.pi fun i => (I i).toPartialEquiv
                  /-
                    𝕜 : Type u
                    inst✝⁴ : NontriviallyNormedField 𝕜
                    ι : Type v
                    inst✝³ : Fintype ι
                    E : ι → Type w
                    inst✝² : (i : ι) → NormedAddCommGroup (E i)
                    inst✝¹ : (i : ι) → NormedSpace 𝕜 (E i)
                    H : ι → Type u'
                    inst✝ : (i : ι) → TopologicalSpace (H i)
                    I : (i : ι) → ModelWithCorners 𝕜 (E i) (H i)
                    ⊢ Eq (PartialEquiv.pi fun i => (I i).toPartialEquiv).source Set.univ
                  -/
  source_eq := by simp only [pi_univ, mfld_simps]
                  /-
                    🎉 no goals
                  -/
  uniqueDiffOn' := UniqueDiffOn.pi ι E _ _ fun i _ => (I i).uniqueDiffOn'
  target_subset_closure_interior := by
    /-
      𝕜 : Type u
      inst✝⁴ : NontriviallyNormedField 𝕜
      ι : Type v
      inst✝³ : Fintype ι
      E : ι → Type w
      inst✝² : (i : ι) → NormedAddCommGroup (E i)
      inst✝¹ : (i : ι) → NormedSpace 𝕜 (E i)
      H : ι → Type u'
      inst✝ : (i : ι) → TopologicalSpace (H i)
      I : (i : ι) → ModelWithCorners 𝕜 (E i) (H i)
      ⊢ HasSubset.Subset (PartialEquiv.pi fun i => (I i).toPartialEquiv).target (clo …
    -/
    simp only [PartialEquiv.pi_target, target_eq, finite_univ, interior_pi_set, closure_pi_set]
    /-
      𝕜 : Type u
      inst✝⁴ : NontriviallyNormedField 𝕜
      ι : Type v
      inst✝³ : Fintype ι
      E : ι → Type w
      inst✝² : (i : ι) → NormedAddCommGroup (E i)
      inst✝¹ : (i : ι) → NormedSpace 𝕜 (E i)
      H : ι → Type u'
      inst✝ : (i : ι) → TopologicalSpace (H i)
      I : (i : ι) → ModelWithCorners 𝕜 (E i) (H i)
      ⊢ HasSubset.Subset (Set.univ.pi fun i => Set.range ↑(I i)) (Set.univ.pi fun i  …
    -/
    exact Set.pi_mono (fun i _ ↦ (I i).range_subset_closure_interior)
    /-
      🎉 no goals
    -/
  continuous_toFun := continuous_pi fun i => (I i).continuous.comp (continuous_apply i)
  continuous_invFun := continuous_pi fun i => (I i).continuous_symm.comp (continuous_apply i)


/-- Special case of product model with corners, which is trivial on the second factor. This shows up
as the model to tangent bundles. -/
abbrev ModelWithCorners.tangent {𝕜 : Type u} [NontriviallyNormedField 𝕜] {E : Type v}
    [NormedAddCommGroup E] [NormedSpace 𝕜 E] {H : Type w} [TopologicalSpace H]
    (I : ModelWithCorners 𝕜 E H) : ModelWithCorners 𝕜 (E × E) (ModelProd H E) :=
  I.prod 𝓘(𝕜, E)


@[simp, mfld_simps]
theorem modelWithCorners_prod_toPartialEquiv :
    (I.prod J).toPartialEquiv = I.toPartialEquiv.prod J.toPartialEquiv :=
  rfl


@[simp, mfld_simps]
theorem modelWithCorners_prod_coe (I : ModelWithCorners 𝕜 E H) (I' : ModelWithCorners 𝕜 E' H') :
    (I.prod I' : _ × _ → _ × _) = Prod.map I I' :=
  rfl


@[simp, mfld_simps]
theorem modelWithCorners_prod_coe_symm (I : ModelWithCorners 𝕜 E H)
    (I' : ModelWithCorners 𝕜 E' H') :
    ((I.prod I').symm : _ × _ → _ × _) = Prod.map I.symm I'.symm :=
  rfl


/-- This lemma should be erased, or at least burn in hell, as it uses bad defeq: the left model
with corners is for `E times F`, the right one for `ModelProd E F`, and there's a good reason
we are distinguishing them. -/
                                                                             /-
                                                                               𝕜 : Type u_1
                                                                               inst✝⁴ : NontriviallyNormedField 𝕜
                                                                               E : Type u_2
                                                                               inst✝³ : NormedAddCommGroup E
                                                                               inst✝² : NormedSpace 𝕜 E
                                                                               F : Type u_4
                                                                               inst✝¹ : NormedAddCommGroup F
                                                                               inst✝ : NormedSpace 𝕜 F
                                                                               ⊢ Eq (modelWithCornersSelf 𝕜 (Prod E F)) ((modelWithCornersSelf 𝕜 E).prod (mod …
                                                                             -/
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/
theorem modelWithCornersSelf_prod : 𝓘(𝕜, E × F) = 𝓘(𝕜, E).prod 𝓘(𝕜, F) := by ext1 <;> simp
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


theorem ModelWithCorners.range_prod : range (I.prod J) = range I ×ˢ range J := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_4
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    H : Type u_5
    inst✝¹ : TopologicalSpace H
    G : Type u_7
    inst✝ : TopologicalSpace G
    I : ModelWithCorners 𝕜 E H
    J : ModelWithCorners 𝕜 F G
    ⊢ Eq (Set.range ↑(I.prod J)) (SProd.sprod (Set.range ↑I) (Set.range ↑J))
  -/
  simp_rw [← ModelWithCorners.target_eq]; rfl
                                          /-
                                            🎉 no goals
                                          -/


/-- Property ensuring that the model with corners `I` defines manifolds without boundary. This
  differs from the more general `BoundarylessManifold`, which requires every point on the manifold
  to be an interior point. -/
class ModelWithCorners.Boundaryless {𝕜 : Type*} [NontriviallyNormedField 𝕜] {E : Type*}
    [NormedAddCommGroup E] [NormedSpace 𝕜 E] {H : Type*} [TopologicalSpace H]
    (I : ModelWithCorners 𝕜 E H) : Prop where
  range_eq_univ : range I = univ


theorem ModelWithCorners.range_eq_univ {𝕜 : Type*} [NontriviallyNormedField 𝕜] {E : Type*}
    [NormedAddCommGroup E] [NormedSpace 𝕜 E] {H : Type*} [TopologicalSpace H]
    (I : ModelWithCorners 𝕜 E H) [I.Boundaryless] :
    range I = univ := ModelWithCorners.Boundaryless.range_eq_univ


/-- If `I` is a `ModelWithCorners.Boundaryless` model, then it is a homeomorphism. -/
@[simps (config := {simpRhs := true})]
def ModelWithCorners.toHomeomorph {𝕜 : Type*} [NontriviallyNormedField 𝕜] {E : Type*}
    [NormedAddCommGroup E] [NormedSpace 𝕜 E] {H : Type*} [TopologicalSpace H]
    (I : ModelWithCorners 𝕜 E H) [I.Boundaryless] : H ≃ₜ E where
  __ := I
  left_inv := I.left_inv
  right_inv _ := I.right_inv <| I.range_eq_univ.symm ▸ mem_univ _


/-- The trivial model with corners has no boundary -/
instance modelWithCornersSelf_boundaryless (𝕜 : Type*) [NontriviallyNormedField 𝕜] (E : Type*)
    [NormedAddCommGroup E] [NormedSpace 𝕜 E] : (modelWithCornersSelf 𝕜 E).Boundaryless :=
      /-
        𝕜 : Type u_1
        inst✝² : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace 𝕜 E
        ⊢ Eq (Set.range ↑(modelWithCornersSelf 𝕜 E)) Set.univ
      -/
  ⟨by simp⟩
      /-
        🎉 no goals
      -/


/-- If two model with corners are boundaryless, their product also is -/
instance ModelWithCorners.range_eq_univ_prod {𝕜 : Type u} [NontriviallyNormedField 𝕜] {E : Type v}
    [NormedAddCommGroup E] [NormedSpace 𝕜 E] {H : Type w} [TopologicalSpace H]
    (I : ModelWithCorners 𝕜 E H) [I.Boundaryless] {E' : Type v'} [NormedAddCommGroup E']
    [NormedSpace 𝕜 E'] {H' : Type w'} [TopologicalSpace H'] (I' : ModelWithCorners 𝕜 E' H')
    [I'.Boundaryless] : (I.prod I').Boundaryless := by
  /-
    𝕜 : Type u
    inst✝⁸ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    H : Type w
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    inst✝⁴ : I.Boundaryless
    E' : Type v'
    inst✝³ : NormedAddCommGroup E'
    inst✝² : NormedSpace 𝕜 E'
    H' : Type w'
    inst✝¹ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    inst✝ : I'.Boundaryless
    ⊢ (I.prod I').Boundaryless
  -/
  constructor
  /-
    case range_eq_univ
    𝕜 : Type u
    inst✝⁸ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    H : Type w
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    inst✝⁴ : I.Boundaryless
    E' : Type v'
    inst✝³ : NormedAddCommGroup E'
    inst✝² : NormedSpace 𝕜 E'
    H' : Type w'
    inst✝¹ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    inst✝ : I'.Boundaryless
    ⊢ Eq (Set.range ↑(I.prod I')) Set.univ
  -/
  dsimp [ModelWithCorners.prod, ModelProd]
  rw [← prod_range_range_eq, ModelWithCorners.Boundaryless.range_eq_univ,
    ModelWithCorners.Boundaryless.range_eq_univ, univ_prod_univ]


variable (n I) in
/-- Given a model with corners `(E, H)`, we define the pregroupoid of `C^n` transformations of `H`
as the maps that are `C^n` when read in `E` through `I`. -/
def contDiffPregroupoid : Pregroupoid H where
  property f s := ContDiffOn 𝕜 n (I ∘ f ∘ I.symm) (I.symm ⁻¹' s ∩ range I)
  comp {f g u v} hf hg _ _ _ := by
    /-
      m n : WithTop ENat
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_3
      inst✝¹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝ : TopologicalSpace M
      f g : H → H
      u v : Set H
      hf : (fun f s => ContDiffOn 𝕜 n (Function.comp (↑I) (Function.comp f ↑I.symm)) …
      hg : (fun f s => ContDiffOn 𝕜 n (Function.comp (↑I) (Function.comp f ↑I.symm)) …
      x✝² : IsOpen u
      x✝¹ : IsOpen v
      x✝ : IsOpen (Inter.inter u (Set.preimage f v))
      ⊢ (fun f s => ContDiffOn 𝕜 n (Function.comp (↑I) (Function.comp f ↑I.symm)) (I …
    -/
    have : I ∘ (g ∘ f) ∘ I.symm = (I ∘ g ∘ I.symm) ∘ I ∘ f ∘ I.symm := by ext x; simp
    /-
      m n : WithTop ENat
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_3
      inst✝¹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝ : TopologicalSpace M
      f g : H → H
      u v : Set H
      hf : (fun f s => ContDiffOn 𝕜 n (Function.comp (↑I) (Function.comp f ↑I.symm)) …
      hg : (fun f s => ContDiffOn 𝕜 n (Function.comp (↑I) (Function.comp f ↑I.symm)) …
      x✝² : IsOpen u
      x✝¹ : IsOpen v
      x✝ : IsOpen (Inter.inter u (Set.preimage f v))
      this : Eq (Function.comp (↑I) (Function.comp (Function.comp g f) ↑I.symm)) (Fu …
      ⊢ (fun f s => ContDiffOn 𝕜 n (Function.comp (↑I) (Function.comp f ↑I.symm)) (I …
    -/
    simp only [this]
    /-
      m n : WithTop ENat
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_3
      inst✝¹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝ : TopologicalSpace M
      f g : H → H
      u v : Set H
      hf : (fun f s => ContDiffOn 𝕜 n (Function.comp (↑I) (Function.comp f ↑I.symm)) …
      hg : (fun f s => ContDiffOn 𝕜 n (Function.comp (↑I) (Function.comp f ↑I.symm)) …
      x✝² : IsOpen u
      x✝¹ : IsOpen v
      x✝ : IsOpen (Inter.inter u (Set.preimage f v))
      this : Eq (Function.comp (↑I) (Function.comp (Function.comp g f) ↑I.symm)) (Fu …
      ⊢ ContDiffOn 𝕜 n (Function.comp (Function.comp (↑I) (Function.comp g ↑I.symm)) …
    -/
    refine hg.comp (hf.mono fun x ⟨hx1, hx2⟩ ↦ ⟨hx1.1, hx2⟩) ?_
    /-
      m n : WithTop ENat
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_3
      inst✝¹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝ : TopologicalSpace M
      f g : H → H
      u v : Set H
      hf : (fun f s => ContDiffOn 𝕜 n (Function.comp (↑I) (Function.comp f ↑I.symm)) …
      hg : (fun f s => ContDiffOn 𝕜 n (Function.comp (↑I) (Function.comp f ↑I.symm)) …
      x✝² : IsOpen u
      x✝¹ : IsOpen v
      x✝ : IsOpen (Inter.inter u (Set.preimage f v))
      this : Eq (Function.comp (↑I) (Function.comp (Function.comp g f) ↑I.symm)) (Fu …
      ⊢ Set.MapsTo (Function.comp (↑I) (Function.comp f ↑I.symm)) (Inter.inter (Set. …
    -/
    rintro x ⟨hx1, _⟩
    /-
      case intro
      m n : WithTop ENat
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_3
      inst✝¹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝ : TopologicalSpace M
      f g : H → H
      u v : Set H
      hf : (fun f s => ContDiffOn 𝕜 n (Function.comp (↑I) (Function.comp f ↑I.symm)) …
      hg : (fun f s => ContDiffOn 𝕜 n (Function.comp (↑I) (Function.comp f ↑I.symm)) …
      x✝² : IsOpen u
      x✝¹ : IsOpen v
      x✝ : IsOpen (Inter.inter u (Set.preimage f v))
      this : Eq (Function.comp (↑I) (Function.comp (Function.comp g f) ↑I.symm)) (Fu …
      x : E
      hx1 : Membership.mem (Set.preimage (↑I.symm) (Inter.inter u (Set.preimage f v) …
      right✝ : Membership.mem (Set.range ↑I) x
      ⊢ Membership.mem (Inter.inter (Set.preimage (↑I.symm) v) (Set.range ↑I)) (Func …
    -/
    simp only [mfld_simps] at hx1 ⊢
    /-
      case intro
      m n : WithTop ENat
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_3
      inst✝¹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝ : TopologicalSpace M
      f g : H → H
      u v : Set H
      hf : (fun f s => ContDiffOn 𝕜 n (Function.comp (↑I) (Function.comp f ↑I.symm)) …
      hg : (fun f s => ContDiffOn 𝕜 n (Function.comp (↑I) (Function.comp f ↑I.symm)) …
      x✝² : IsOpen u
      x✝¹ : IsOpen v
      x✝ : IsOpen (Inter.inter u (Set.preimage f v))
      this : Eq (Function.comp (↑I) (Function.comp (Function.comp g f) ↑I.symm)) (Fu …
      x : E
      right✝ : Membership.mem (Set.range ↑I) x
      hx1 : And (Membership.mem u (↑I.symm x)) (Membership.mem v (f (↑I.symm x)))
      ⊢ Membership.mem v (f (↑I.symm x))
    -/
    exact hx1.2
    /-
      🎉 no goals
    -/
  id_mem := by
    /-
      m n : WithTop ENat
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_3
      inst✝¹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝ : TopologicalSpace M
      ⊢ (fun f s => ContDiffOn 𝕜 n (Function.comp (↑I) (Function.comp f ↑I.symm)) (I …
    -/
    apply ContDiffOn.congr contDiff_id.contDiffOn
    /-
      m n : WithTop ENat
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_3
      inst✝¹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝ : TopologicalSpace M
      ⊢ ∀ (x : E), Membership.mem (Inter.inter (Set.preimage (↑I.symm) Set.univ) (Se …
    -/
    rintro x ⟨_, hx2⟩
    /-
      case intro
      m n : WithTop ENat
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_3
      inst✝¹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝ : TopologicalSpace M
      x : E
      left✝ : Membership.mem (Set.preimage (↑I.symm) Set.univ) x
      hx2 : Membership.mem (Set.range ↑I) x
      ⊢ Eq (Function.comp (↑I) (Function.comp id ↑I.symm) x) (id x)
    -/
    rcases mem_range.1 hx2 with ⟨y, hy⟩
    /-
      case intro.intro
      m n : WithTop ENat
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_3
      inst✝¹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝ : TopologicalSpace M
      x : E
      left✝ : Membership.mem (Set.preimage (↑I.symm) Set.univ) x
      hx2 : Membership.mem (Set.range ↑I) x
      y : H
      hy : Eq (↑I y) x
      ⊢ Eq (Function.comp (↑I) (Function.comp id ↑I.symm) x) (id x)
    -/
    rw [← hy]
    /-
      case intro.intro
      m n : WithTop ENat
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_3
      inst✝¹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝ : TopologicalSpace M
      x : E
      left✝ : Membership.mem (Set.preimage (↑I.symm) Set.univ) x
      hx2 : Membership.mem (Set.range ↑I) x
      y : H
      hy : Eq (↑I y) x
      ⊢ Eq (Function.comp (↑I) (Function.comp id ↑I.symm) (↑I y)) (id (↑I y))
    -/
    simp only [mfld_simps]
    /-
      🎉 no goals
    -/
  locality {f u} _ H := by
    /-
      m n : WithTop ENat
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H✝ : Type u_3
      inst✝¹ : TopologicalSpace H✝
      I : ModelWithCorners 𝕜 E H✝
      M : Type u_4
      inst✝ : TopologicalSpace M
      f : H✝ → H✝
      u : Set H✝
      x✝ : IsOpen u
      H : ∀ (x : H✝), Membership.mem u x → Exists fun v => And (IsOpen v) (And (Memb …
      ⊢ (fun f s => ContDiffOn 𝕜 n (Function.comp (↑I) (Function.comp f ↑I.symm)) (I …
    -/
    apply contDiffOn_of_locally_contDiffOn
    /-
      case h
      m n : WithTop ENat
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H✝ : Type u_3
      inst✝¹ : TopologicalSpace H✝
      I : ModelWithCorners 𝕜 E H✝
      M : Type u_4
      inst✝ : TopologicalSpace M
      f : H✝ → H✝
      u : Set H✝
      x✝ : IsOpen u
      H : ∀ (x : H✝), Membership.mem u x → Exists fun v => And (IsOpen v) (And (Memb …
      ⊢ ∀ (x : E), Membership.mem (Inter.inter (Set.preimage (↑I.symm) u) (Set.range …
    -/
    rintro y ⟨hy1, hy2⟩
    /-
      case h.intro
      m n : WithTop ENat
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H✝ : Type u_3
      inst✝¹ : TopologicalSpace H✝
      I : ModelWithCorners 𝕜 E H✝
      M : Type u_4
      inst✝ : TopologicalSpace M
      f : H✝ → H✝
      u : Set H✝
      x✝ : IsOpen u
      H : ∀ (x : H✝), Membership.mem u x → Exists fun v => And (IsOpen v) (And (Memb …
      y : E
      hy1 : Membership.mem (Set.preimage (↑I.symm) u) y
      hy2 : Membership.mem (Set.range ↑I) y
      ⊢ Exists fun u_1 => And (IsOpen u_1) (And (Membership.mem u_1 y) (ContDiffOn 𝕜 …
    -/
    rcases mem_range.1 hy2 with ⟨x, hx⟩
    /-
      case h.intro.intro
      m n : WithTop ENat
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H✝ : Type u_3
      inst✝¹ : TopologicalSpace H✝
      I : ModelWithCorners 𝕜 E H✝
      M : Type u_4
      inst✝ : TopologicalSpace M
      f : H✝ → H✝
      u : Set H✝
      x✝ : IsOpen u
      H : ∀ (x : H✝), Membership.mem u x → Exists fun v => And (IsOpen v) (And (Memb …
      y : E
      hy1 : Membership.mem (Set.preimage (↑I.symm) u) y
      hy2 : Membership.mem (Set.range ↑I) y
      x : H✝
      hx : Eq (↑I x) y
      ⊢ Exists fun u_1 => And (IsOpen u_1) (And (Membership.mem u_1 y) (ContDiffOn 𝕜 …
    -/
    rw [← hx] at hy1 ⊢
    /-
      case h.intro.intro
      m n : WithTop ENat
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H✝ : Type u_3
      inst✝¹ : TopologicalSpace H✝
      I : ModelWithCorners 𝕜 E H✝
      M : Type u_4
      inst✝ : TopologicalSpace M
      f : H✝ → H✝
      u : Set H✝
      x✝ : IsOpen u
      H : ∀ (x : H✝), Membership.mem u x → Exists fun v => And (IsOpen v) (And (Memb …
      y : E
      hy2 : Membership.mem (Set.range ↑I) y
      x : H✝
      hy1 : Membership.mem (Set.preimage (↑I.symm) u) (↑I x)
      hx : Eq (↑I x) y
      ⊢ Exists fun u_1 => And (IsOpen u_1) (And (Membership.mem u_1 (↑I x)) (ContDif …
    -/
    simp only [mfld_simps] at hy1 ⊢
    /-
      case h.intro.intro
      m n : WithTop ENat
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H✝ : Type u_3
      inst✝¹ : TopologicalSpace H✝
      I : ModelWithCorners 𝕜 E H✝
      M : Type u_4
      inst✝ : TopologicalSpace M
      f : H✝ → H✝
      u : Set H✝
      x✝ : IsOpen u
      H : ∀ (x : H✝), Membership.mem u x → Exists fun v => And (IsOpen v) (And (Memb …
      y : E
      hy2 : Membership.mem (Set.range ↑I) y
      x : H✝
      hx : Eq (↑I x) y
      hy1 : Membership.mem u x
      ⊢ Exists fun u_1 => And (IsOpen u_1) (And (Membership.mem u_1 (↑I x)) (ContDif …
    -/
    rcases H x hy1 with ⟨v, v_open, xv, hv⟩
    have : I.symm ⁻¹' (u ∩ v) ∩ range I = I.symm ⁻¹' u ∩ range I ∩ I.symm ⁻¹' v := by
      rw [preimage_inter, inter_assoc, inter_assoc]
      congr 1
      rw [inter_comm]
    /-
      case h.intro.intro.intro.intro.intro
      m n : WithTop ENat
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H✝ : Type u_3
      inst✝¹ : TopologicalSpace H✝
      I : ModelWithCorners 𝕜 E H✝
      M : Type u_4
      inst✝ : TopologicalSpace M
      f : H✝ → H✝
      u : Set H✝
      x✝ : IsOpen u
      H : ∀ (x : H✝), Membership.mem u x → Exists fun v => And (IsOpen v) (And (Memb …
      y : E
      hy2 : Membership.mem (Set.range ↑I) y
      x : H✝
      hx : Eq (↑I x) y
      hy1 : Membership.mem u x
      v : Set H✝
      v_open : IsOpen v
      xv : Membership.mem v x
      hv : ContDiffOn 𝕜 n (Function.comp (↑I) (Function.comp f ↑I.symm)) (Inter.inte …
      this : Eq (Inter.inter (Set.preimage (↑I.symm) (Inter.inter u v)) (Set.range ↑ …
      ⊢ Exists fun u_1 => And (IsOpen u_1) (And (Membership.mem u_1 (↑I x)) (ContDif …
    -/
    rw [this] at hv
    /-
      case h.intro.intro.intro.intro.intro
      m n : WithTop ENat
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H✝ : Type u_3
      inst✝¹ : TopologicalSpace H✝
      I : ModelWithCorners 𝕜 E H✝
      M : Type u_4
      inst✝ : TopologicalSpace M
      f : H✝ → H✝
      u : Set H✝
      x✝ : IsOpen u
      H : ∀ (x : H✝), Membership.mem u x → Exists fun v => And (IsOpen v) (And (Memb …
      y : E
      hy2 : Membership.mem (Set.range ↑I) y
      x : H✝
      hx : Eq (↑I x) y
      hy1 : Membership.mem u x
      v : Set H✝
      v_open : IsOpen v
      xv : Membership.mem v x
      hv : ContDiffOn 𝕜 n (Function.comp (↑I) (Function.comp f ↑I.symm)) (Inter.inte …
      this : Eq (Inter.inter (Set.preimage (↑I.symm) (Inter.inter u v)) (Set.range ↑ …
      ⊢ Exists fun u_1 => And (IsOpen u_1) (And (Membership.mem u_1 (↑I x)) (ContDif …
    -/
    exact ⟨I.symm ⁻¹' v, v_open.preimage I.continuous_symm, by simpa, hv⟩
    /-
      🎉 no goals
    -/
  congr {f g u} _ fg hf := by
    /-
      m n : WithTop ENat
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_3
      inst✝¹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝ : TopologicalSpace M
      f g : H → H
      u : Set H
      x✝ : IsOpen u
      fg : ∀ (x : H), Membership.mem u x → Eq (g x) (f x)
      hf : (fun f s => ContDiffOn 𝕜 n (Function.comp (↑I) (Function.comp f ↑I.symm)) …
      ⊢ (fun f s => ContDiffOn 𝕜 n (Function.comp (↑I) (Function.comp f ↑I.symm)) (I …
    -/
    apply hf.congr
    /-
      m n : WithTop ENat
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_3
      inst✝¹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝ : TopologicalSpace M
      f g : H → H
      u : Set H
      x✝ : IsOpen u
      fg : ∀ (x : H), Membership.mem u x → Eq (g x) (f x)
      hf : (fun f s => ContDiffOn 𝕜 n (Function.comp (↑I) (Function.comp f ↑I.symm)) …
      ⊢ ∀ (x : E), Membership.mem (Inter.inter (Set.preimage (↑I.symm) u) (Set.range …
    -/
    rintro y ⟨hy1, hy2⟩
    /-
      case intro
      m n : WithTop ENat
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_3
      inst✝¹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝ : TopologicalSpace M
      f g : H → H
      u : Set H
      x✝ : IsOpen u
      fg : ∀ (x : H), Membership.mem u x → Eq (g x) (f x)
      hf : (fun f s => ContDiffOn 𝕜 n (Function.comp (↑I) (Function.comp f ↑I.symm)) …
      y : E
      hy1 : Membership.mem (Set.preimage (↑I.symm) u) y
      hy2 : Membership.mem (Set.range ↑I) y
      ⊢ Eq (Function.comp (↑I) (Function.comp g ↑I.symm) y) (Function.comp (↑I) (Fun …
    -/
    rcases mem_range.1 hy2 with ⟨x, hx⟩
    /-
      case intro.intro
      m n : WithTop ENat
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_3
      inst✝¹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝ : TopologicalSpace M
      f g : H → H
      u : Set H
      x✝ : IsOpen u
      fg : ∀ (x : H), Membership.mem u x → Eq (g x) (f x)
      hf : (fun f s => ContDiffOn 𝕜 n (Function.comp (↑I) (Function.comp f ↑I.symm)) …
      y : E
      hy1 : Membership.mem (Set.preimage (↑I.symm) u) y
      hy2 : Membership.mem (Set.range ↑I) y
      x : H
      hx : Eq (↑I x) y
      ⊢ Eq (Function.comp (↑I) (Function.comp g ↑I.symm) y) (Function.comp (↑I) (Fun …
    -/
    rw [← hx] at hy1 ⊢
    /-
      case intro.intro
      m n : WithTop ENat
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_3
      inst✝¹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝ : TopologicalSpace M
      f g : H → H
      u : Set H
      x✝ : IsOpen u
      fg : ∀ (x : H), Membership.mem u x → Eq (g x) (f x)
      hf : (fun f s => ContDiffOn 𝕜 n (Function.comp (↑I) (Function.comp f ↑I.symm)) …
      y : E
      hy2 : Membership.mem (Set.range ↑I) y
      x : H
      hy1 : Membership.mem (Set.preimage (↑I.symm) u) (↑I x)
      hx : Eq (↑I x) y
      ⊢ Eq (Function.comp (↑I) (Function.comp g ↑I.symm) (↑I x)) (Function.comp (↑I) …
    -/
    simp only [mfld_simps] at hy1 ⊢
    /-
      case intro.intro
      m n : WithTop ENat
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      H : Type u_3
      inst✝¹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝ : TopologicalSpace M
      f g : H → H
      u : Set H
      x✝ : IsOpen u
      fg : ∀ (x : H), Membership.mem u x → Eq (g x) (f x)
      hf : (fun f s => ContDiffOn 𝕜 n (Function.comp (↑I) (Function.comp f ↑I.symm)) …
      y : E
      hy2 : Membership.mem (Set.range ↑I) y
      x : H
      hx : Eq (↑I x) y
      hy1 : Membership.mem u x
      ⊢ Eq (↑I (g x)) (↑I (f x))
    -/
    rw [fg _ hy1]
    /-
      🎉 no goals
    -/


variable (n I) in
/-- Given a model with corners `(E, H)`, we define the groupoid of invertible `C^n` transformations
  of `H` as the invertible maps that are `C^n` when read in `E` through `I`. -/
def contDiffGroupoid : StructureGroupoid H :=
  Pregroupoid.groupoid (contDiffPregroupoid n I)


/-- Inclusion of the groupoid of `C^n` local diffeos in the groupoid of `C^m` local diffeos when
`m ≤ n` -/
theorem contDiffGroupoid_le (h : m ≤ n) : contDiffGroupoid n I ≤ contDiffGroupoid m I := by
  /-
    m n : WithTop ENat
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    h : LE.le m n
    ⊢ LE.le (contDiffGroupoid n I) (contDiffGroupoid m I)
  -/
  rw [contDiffGroupoid, contDiffGroupoid]
  /-
    m n : WithTop ENat
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    h : LE.le m n
    ⊢ LE.le (contDiffPregroupoid n I).groupoid (contDiffPregroupoid m I).groupoid
  -/
  apply groupoid_of_pregroupoid_le
  /-
    case h
    m n : WithTop ENat
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    h : LE.le m n
    ⊢ ∀ (f : H → H) (s : Set H), (contDiffPregroupoid n I).property f s → (contDif …
  -/
  intro f s hfs
  /-
    case h
    m n : WithTop ENat
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    h : LE.le m n
    f : H → H
    s : Set H
    hfs : (contDiffPregroupoid n I).property f s
    ⊢ (contDiffPregroupoid m I).property f s
  -/
  exact ContDiffOn.of_le hfs h
  /-
    🎉 no goals
  -/


/-- The groupoid of `0`-times continuously differentiable maps is just the groupoid of all
partial homeomorphisms -/
theorem contDiffGroupoid_zero_eq : contDiffGroupoid 0 I = continuousGroupoid H := by
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    ⊢ Eq (contDiffGroupoid 0 I) (continuousGroupoid H)
  -/
  apply le_antisymm le_top
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    ⊢ LE.le Top.top (contDiffGroupoid 0 I)
  -/
  intro u _
  -- we have to check that every partial homeomorphism belongs to `contDiffGroupoid 0 I`,
  -- by unfolding its definition
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    u : PartialHomeomorph H H
    a✝ : Membership.mem Top.top.members u
    ⊢ Membership.mem (contDiffGroupoid 0 I).members u
  -/
  change u ∈ contDiffGroupoid 0 I
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    u : PartialHomeomorph H H
    a✝ : Membership.mem Top.top.members u
    ⊢ Membership.mem (contDiffGroupoid 0 I) u
  -/
  rw [contDiffGroupoid, mem_groupoid_of_pregroupoid, contDiffPregroupoid]
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    u : PartialHomeomorph H H
    a✝ : Membership.mem Top.top.members u
    ⊢ And ({ property := fun f s => ContDiffOn 𝕜 0 (Function.comp (↑I) (Function.c …
  -/
  simp only [contDiffOn_zero]
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    u : PartialHomeomorph H H
    a✝ : Membership.mem Top.top.members u
    ⊢ And (ContinuousOn (Function.comp (↑I) (Function.comp ↑u ↑I.symm)) (Inter.int …
  -/
  constructor
    /-
      case left
      𝕜 : Type u_1
      inst✝³ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      u : PartialHomeomorph H H
      a✝ : Membership.mem Top.top.members u
      ⊢ ContinuousOn (Function.comp (↑I) (Function.comp ↑u ↑I.symm)) (Inter.inter (S …
    -/
  · refine I.continuous.comp_continuousOn (u.continuousOn.comp I.continuousOn_symm ?_)
    /-
      case left
      𝕜 : Type u_1
      inst✝³ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      u : PartialHomeomorph H H
      a✝ : Membership.mem Top.top.members u
      ⊢ Set.MapsTo (↑I.symm) (Inter.inter (Set.preimage (↑I.symm) u.source) (Set.ran …
    -/
    exact (mapsTo_preimage _ _).mono_left inter_subset_left
    /-
      🎉 no goals
    -/
    /-
      case right
      𝕜 : Type u_1
      inst✝³ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      u : PartialHomeomorph H H
      a✝ : Membership.mem Top.top.members u
      ⊢ ContinuousOn (Function.comp (↑I) (Function.comp ↑u.symm ↑I.symm)) (Inter.int …
    -/
  · refine I.continuous.comp_continuousOn (u.symm.continuousOn.comp I.continuousOn_symm ?_)
    /-
      case right
      𝕜 : Type u_1
      inst✝³ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      u : PartialHomeomorph H H
      a✝ : Membership.mem Top.top.members u
      ⊢ Set.MapsTo (↑I.symm) (Inter.inter (Set.preimage (↑I.symm) u.target) (Set.ran …
    -/
    exact (mapsTo_preimage _ _).mono_left inter_subset_left
    /-
      🎉 no goals
    -/


/-- An identity partial homeomorphism belongs to the `C^n` groupoid. -/
theorem ofSet_mem_contDiffGroupoid {s : Set H} (hs : IsOpen s) :
    PartialHomeomorph.ofSet s hs ∈ contDiffGroupoid n I := by
  /-
    n : WithTop ENat
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    s : Set H
    hs : IsOpen s
    ⊢ Membership.mem (contDiffGroupoid n I) (PartialHomeomorph.ofSet s hs)
  -/
  rw [contDiffGroupoid, mem_groupoid_of_pregroupoid]
  suffices h : ContDiffOn 𝕜 n (I ∘ I.symm) (I.symm ⁻¹' s ∩ range I) by
    simp [h, contDiffPregroupoid]
  /-
    n : WithTop ENat
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    s : Set H
    hs : IsOpen s
    ⊢ ContDiffOn 𝕜 n (Function.comp ↑I ↑I.symm) (Inter.inter (Set.preimage (↑I.sym …
  -/
  have : ContDiffOn 𝕜 n id (univ : Set E) := contDiff_id.contDiffOn
  /-
    n : WithTop ENat
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    s : Set H
    hs : IsOpen s
    this : ContDiffOn 𝕜 n id Set.univ
    ⊢ ContDiffOn 𝕜 n (Function.comp ↑I ↑I.symm) (Inter.inter (Set.preimage (↑I.sym …
  -/
  exact this.congr_mono (fun x hx => I.right_inv hx.2) (subset_univ _)
  /-
    🎉 no goals
  -/


/-- The composition of a partial homeomorphism from `H` to `M` and its inverse belongs to
the `C^n` groupoid. -/
theorem symm_trans_mem_contDiffGroupoid (e : PartialHomeomorph M H) :
    e.symm.trans e ∈ contDiffGroupoid n I :=
  haveI : e.symm.trans e ≈ PartialHomeomorph.ofSet e.target e.open_target :=
    PartialHomeomorph.symm_trans_self _
  StructureGroupoid.mem_of_eqOnSource _ (ofSet_mem_contDiffGroupoid e.open_target) this


/-- The product of two smooth partial homeomorphisms is smooth. -/
theorem contDiffGroupoid_prod {I : ModelWithCorners 𝕜 E H} {I' : ModelWithCorners 𝕜 E' H'}
    {e : PartialHomeomorph H H} {e' : PartialHomeomorph H' H'} (he : e ∈ contDiffGroupoid ∞ I)
    (he' : e' ∈ contDiffGroupoid ∞ I') : e.prod e' ∈ contDiffGroupoid ∞ (I.prod I') := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝³ : TopologicalSpace H
    E' : Type u_5
    H' : Type u_6
    inst✝² : NormedAddCommGroup E'
    inst✝¹ : NormedSpace 𝕜 E'
    inst✝ : TopologicalSpace H'
    I : ModelWithCorners 𝕜 E H
    I' : ModelWithCorners 𝕜 E' H'
    e : PartialHomeomorph H H
    e' : PartialHomeomorph H' H'
    he : Membership.mem (contDiffGroupoid (↑Top.top) I) e
    he' : Membership.mem (contDiffGroupoid (↑Top.top) I') e'
    ⊢ Membership.mem (contDiffGroupoid (↑Top.top) (I.prod I')) (e.prod e')
  -/
  cases' he with he he_symm
  /-
    case intro
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝³ : TopologicalSpace H
    E' : Type u_5
    H' : Type u_6
    inst✝² : NormedAddCommGroup E'
    inst✝¹ : NormedSpace 𝕜 E'
    inst✝ : TopologicalSpace H'
    I : ModelWithCorners 𝕜 E H
    I' : ModelWithCorners 𝕜 E' H'
    e : PartialHomeomorph H H
    e' : PartialHomeomorph H' H'
    he' : Membership.mem (contDiffGroupoid (↑Top.top) I') e'
    he : (contDiffPregroupoid (↑Top.top) I).property (↑e) e.source
    he_symm : (contDiffPregroupoid (↑Top.top) I).property (↑e.symm) e.target
    ⊢ Membership.mem (contDiffGroupoid (↑Top.top) (I.prod I')) (e.prod e')
  -/
  cases' he' with he' he'_symm
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝³ : TopologicalSpace H
    E' : Type u_5
    H' : Type u_6
    inst✝² : NormedAddCommGroup E'
    inst✝¹ : NormedSpace 𝕜 E'
    inst✝ : TopologicalSpace H'
    I : ModelWithCorners 𝕜 E H
    I' : ModelWithCorners 𝕜 E' H'
    e : PartialHomeomorph H H
    e' : PartialHomeomorph H' H'
    he : (contDiffPregroupoid (↑Top.top) I).property (↑e) e.source
    he_symm : (contDiffPregroupoid (↑Top.top) I).property (↑e.symm) e.target
    he' : (contDiffPregroupoid (↑Top.top) I').property (↑e') e'.source
    he'_symm : (contDiffPregroupoid (↑Top.top) I').property (↑e'.symm) e'.target
    ⊢ Membership.mem (contDiffGroupoid (↑Top.top) (I.prod I')) (e.prod e')
  -/
  simp only at he he_symm he' he'_symm
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝³ : TopologicalSpace H
    E' : Type u_5
    H' : Type u_6
    inst✝² : NormedAddCommGroup E'
    inst✝¹ : NormedSpace 𝕜 E'
    inst✝ : TopologicalSpace H'
    I : ModelWithCorners 𝕜 E H
    I' : ModelWithCorners 𝕜 E' H'
    e : PartialHomeomorph H H
    e' : PartialHomeomorph H' H'
    he : (contDiffPregroupoid (↑Top.top) I).property (↑e) e.source
    he_symm : (contDiffPregroupoid (↑Top.top) I).property (↑e.symm) e.target
    he' : (contDiffPregroupoid (↑Top.top) I').property (↑e') e'.source
    he'_symm : (contDiffPregroupoid (↑Top.top) I').property (↑e'.symm) e'.target
    ⊢ Membership.mem (contDiffGroupoid (↑Top.top) (I.prod I')) (e.prod e')
  -/
  constructor <;> simp only [PartialEquiv.prod_source, PartialHomeomorph.prod_toPartialEquiv,
    contDiffPregroupoid]
    /-
      case intro.intro.left
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝³ : TopologicalSpace H
      E' : Type u_5
      H' : Type u_6
      inst✝² : NormedAddCommGroup E'
      inst✝¹ : NormedSpace 𝕜 E'
      inst✝ : TopologicalSpace H'
      I : ModelWithCorners 𝕜 E H
      I' : ModelWithCorners 𝕜 E' H'
      e : PartialHomeomorph H H
      e' : PartialHomeomorph H' H'
      he : (contDiffPregroupoid (↑Top.top) I).property (↑e) e.source
      he_symm : (contDiffPregroupoid (↑Top.top) I).property (↑e.symm) e.target
      he' : (contDiffPregroupoid (↑Top.top) I').property (↑e') e'.source
      he'_symm : (contDiffPregroupoid (↑Top.top) I').property (↑e'.symm) e'.target
      ⊢ ContDiffOn 𝕜 (↑Top.top) (Function.comp (↑(I.prod I')) (Function.comp ↑(e.pro …
    -/
  · have h3 := ContDiffOn.prod_map he he'
    /-
      case intro.intro.left
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝³ : TopologicalSpace H
      E' : Type u_5
      H' : Type u_6
      inst✝² : NormedAddCommGroup E'
      inst✝¹ : NormedSpace 𝕜 E'
      inst✝ : TopologicalSpace H'
      I : ModelWithCorners 𝕜 E H
      I' : ModelWithCorners 𝕜 E' H'
      e : PartialHomeomorph H H
      e' : PartialHomeomorph H' H'
      he : (contDiffPregroupoid (↑Top.top) I).property (↑e) e.source
      he_symm : (contDiffPregroupoid (↑Top.top) I).property (↑e.symm) e.target
      he' : (contDiffPregroupoid (↑Top.top) I').property (↑e') e'.source
      he'_symm : (contDiffPregroupoid (↑Top.top) I').property (↑e'.symm) e'.target
      h3 : ContDiffOn 𝕜 (↑Top.top) (Prod.map (Function.comp (↑I) (Function.comp ↑e ↑ …
      ⊢ ContDiffOn 𝕜 (↑Top.top) (Function.comp (↑(I.prod I')) (Function.comp ↑(e.pro …
    -/
    rw [← I.image_eq, ← I'.image_eq, prod_image_image_eq] at h3
    /-
      case intro.intro.left
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝³ : TopologicalSpace H
      E' : Type u_5
      H' : Type u_6
      inst✝² : NormedAddCommGroup E'
      inst✝¹ : NormedSpace 𝕜 E'
      inst✝ : TopologicalSpace H'
      I : ModelWithCorners 𝕜 E H
      I' : ModelWithCorners 𝕜 E' H'
      e : PartialHomeomorph H H
      e' : PartialHomeomorph H' H'
      he : (contDiffPregroupoid (↑Top.top) I).property (↑e) e.source
      he_symm : (contDiffPregroupoid (↑Top.top) I).property (↑e.symm) e.target
      he' : (contDiffPregroupoid (↑Top.top) I').property (↑e') e'.source
      he'_symm : (contDiffPregroupoid (↑Top.top) I').property (↑e'.symm) e'.target
      h3 : ContDiffOn 𝕜 (↑Top.top) (Prod.map (Function.comp (↑I) (Function.comp ↑e ↑ …
      ⊢ ContDiffOn 𝕜 (↑Top.top) (Function.comp (↑(I.prod I')) (Function.comp ↑(e.pro …
    -/
    rw [← (I.prod I').image_eq]
    /-
      case intro.intro.left
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝³ : TopologicalSpace H
      E' : Type u_5
      H' : Type u_6
      inst✝² : NormedAddCommGroup E'
      inst✝¹ : NormedSpace 𝕜 E'
      inst✝ : TopologicalSpace H'
      I : ModelWithCorners 𝕜 E H
      I' : ModelWithCorners 𝕜 E' H'
      e : PartialHomeomorph H H
      e' : PartialHomeomorph H' H'
      he : (contDiffPregroupoid (↑Top.top) I).property (↑e) e.source
      he_symm : (contDiffPregroupoid (↑Top.top) I).property (↑e.symm) e.target
      he' : (contDiffPregroupoid (↑Top.top) I').property (↑e') e'.source
      he'_symm : (contDiffPregroupoid (↑Top.top) I').property (↑e'.symm) e'.target
      h3 : ContDiffOn 𝕜 (↑Top.top) (Prod.map (Function.comp (↑I) (Function.comp ↑e ↑ …
      ⊢ ContDiffOn 𝕜 (↑Top.top) (Function.comp (↑(I.prod I')) (Function.comp ↑(e.pro …
    -/
    exact h3
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.right
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝³ : TopologicalSpace H
      E' : Type u_5
      H' : Type u_6
      inst✝² : NormedAddCommGroup E'
      inst✝¹ : NormedSpace 𝕜 E'
      inst✝ : TopologicalSpace H'
      I : ModelWithCorners 𝕜 E H
      I' : ModelWithCorners 𝕜 E' H'
      e : PartialHomeomorph H H
      e' : PartialHomeomorph H' H'
      he : (contDiffPregroupoid (↑Top.top) I).property (↑e) e.source
      he_symm : (contDiffPregroupoid (↑Top.top) I).property (↑e.symm) e.target
      he' : (contDiffPregroupoid (↑Top.top) I').property (↑e') e'.source
      he'_symm : (contDiffPregroupoid (↑Top.top) I').property (↑e'.symm) e'.target
      ⊢ ContDiffOn 𝕜 (↑Top.top) (Function.comp (↑(I.prod I')) (Function.comp ↑(e.pro …
    -/
  · have h3 := ContDiffOn.prod_map he_symm he'_symm
    /-
      case intro.intro.right
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝³ : TopologicalSpace H
      E' : Type u_5
      H' : Type u_6
      inst✝² : NormedAddCommGroup E'
      inst✝¹ : NormedSpace 𝕜 E'
      inst✝ : TopologicalSpace H'
      I : ModelWithCorners 𝕜 E H
      I' : ModelWithCorners 𝕜 E' H'
      e : PartialHomeomorph H H
      e' : PartialHomeomorph H' H'
      he : (contDiffPregroupoid (↑Top.top) I).property (↑e) e.source
      he_symm : (contDiffPregroupoid (↑Top.top) I).property (↑e.symm) e.target
      he' : (contDiffPregroupoid (↑Top.top) I').property (↑e') e'.source
      he'_symm : (contDiffPregroupoid (↑Top.top) I').property (↑e'.symm) e'.target
      h3 : ContDiffOn 𝕜 (↑Top.top) (Prod.map (Function.comp (↑I) (Function.comp ↑e.s …
      ⊢ ContDiffOn 𝕜 (↑Top.top) (Function.comp (↑(I.prod I')) (Function.comp ↑(e.pro …
    -/
    rw [← I.image_eq, ← I'.image_eq, prod_image_image_eq] at h3
    /-
      case intro.intro.right
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝³ : TopologicalSpace H
      E' : Type u_5
      H' : Type u_6
      inst✝² : NormedAddCommGroup E'
      inst✝¹ : NormedSpace 𝕜 E'
      inst✝ : TopologicalSpace H'
      I : ModelWithCorners 𝕜 E H
      I' : ModelWithCorners 𝕜 E' H'
      e : PartialHomeomorph H H
      e' : PartialHomeomorph H' H'
      he : (contDiffPregroupoid (↑Top.top) I).property (↑e) e.source
      he_symm : (contDiffPregroupoid (↑Top.top) I).property (↑e.symm) e.target
      he' : (contDiffPregroupoid (↑Top.top) I').property (↑e') e'.source
      he'_symm : (contDiffPregroupoid (↑Top.top) I').property (↑e'.symm) e'.target
      h3 : ContDiffOn 𝕜 (↑Top.top) (Prod.map (Function.comp (↑I) (Function.comp ↑e.s …
      ⊢ ContDiffOn 𝕜 (↑Top.top) (Function.comp (↑(I.prod I')) (Function.comp ↑(e.pro …
    -/
    rw [← (I.prod I').image_eq]
    /-
      case intro.intro.right
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝³ : TopologicalSpace H
      E' : Type u_5
      H' : Type u_6
      inst✝² : NormedAddCommGroup E'
      inst✝¹ : NormedSpace 𝕜 E'
      inst✝ : TopologicalSpace H'
      I : ModelWithCorners 𝕜 E H
      I' : ModelWithCorners 𝕜 E' H'
      e : PartialHomeomorph H H
      e' : PartialHomeomorph H' H'
      he : (contDiffPregroupoid (↑Top.top) I).property (↑e) e.source
      he_symm : (contDiffPregroupoid (↑Top.top) I).property (↑e.symm) e.target
      he' : (contDiffPregroupoid (↑Top.top) I').property (↑e') e'.source
      he'_symm : (contDiffPregroupoid (↑Top.top) I').property (↑e'.symm) e'.target
      h3 : ContDiffOn 𝕜 (↑Top.top) (Prod.map (Function.comp (↑I) (Function.comp ↑e.s …
      ⊢ ContDiffOn 𝕜 (↑Top.top) (Function.comp (↑(I.prod I')) (Function.comp ↑(e.pro …
    -/
    exact h3
    /-
      🎉 no goals
    -/


/-- The `C^n` groupoid is closed under restriction. -/
instance : ClosedUnderRestriction (contDiffGroupoid n I) :=
  (closedUnderRestriction_iff_id_le _).mpr
    (by
      /-
        m n : WithTop ENat
        𝕜 : Type u_1
        inst✝⁷ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁶ : NormedAddCommGroup E
        inst✝⁵ : NormedSpace 𝕜 E
        H : Type u_3
        inst✝⁴ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        M : Type u_4
        inst✝³ : TopologicalSpace M
        E' : Type u_5
        H' : Type u_6
        inst✝² : NormedAddCommGroup E'
        inst✝¹ : NormedSpace 𝕜 E'
        inst✝ : TopologicalSpace H'
        ⊢ LE.le idRestrGroupoid (contDiffGroupoid n I)
      -/
      rw [StructureGroupoid.le_iff]
      /-
        m n : WithTop ENat
        𝕜 : Type u_1
        inst✝⁷ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁶ : NormedAddCommGroup E
        inst✝⁵ : NormedSpace 𝕜 E
        H : Type u_3
        inst✝⁴ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        M : Type u_4
        inst✝³ : TopologicalSpace M
        E' : Type u_5
        H' : Type u_6
        inst✝² : NormedAddCommGroup E'
        inst✝¹ : NormedSpace 𝕜 E'
        inst✝ : TopologicalSpace H'
        ⊢ ∀ (e : PartialHomeomorph H H), Membership.mem idRestrGroupoid e → Membership …
      -/
      rintro e ⟨s, hs, hes⟩
      /-
        case intro.intro
        m n : WithTop ENat
        𝕜 : Type u_1
        inst✝⁷ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁶ : NormedAddCommGroup E
        inst✝⁵ : NormedSpace 𝕜 E
        H : Type u_3
        inst✝⁴ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        M : Type u_4
        inst✝³ : TopologicalSpace M
        E' : Type u_5
        H' : Type u_6
        inst✝² : NormedAddCommGroup E'
        inst✝¹ : NormedSpace 𝕜 E'
        inst✝ : TopologicalSpace H'
        e : PartialHomeomorph H H
        s : Set H
        hs : IsOpen s
        hes : HasEquiv.Equiv e (PartialHomeomorph.ofSet s hs)
        ⊢ Membership.mem (contDiffGroupoid n I) e
      -/
      apply (contDiffGroupoid n I).mem_of_eqOnSource' _ _ _ hes
      /-
        m n : WithTop ENat
        𝕜 : Type u_1
        inst✝⁷ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁶ : NormedAddCommGroup E
        inst✝⁵ : NormedSpace 𝕜 E
        H : Type u_3
        inst✝⁴ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        M : Type u_4
        inst✝³ : TopologicalSpace M
        E' : Type u_5
        H' : Type u_6
        inst✝² : NormedAddCommGroup E'
        inst✝¹ : NormedSpace 𝕜 E'
        inst✝ : TopologicalSpace H'
        e : PartialHomeomorph H H
        s : Set H
        hs : IsOpen s
        hes : HasEquiv.Equiv e (PartialHomeomorph.ofSet s hs)
        ⊢ Membership.mem (contDiffGroupoid n I).members (PartialHomeomorph.ofSet s hs)
      -/
      exact ofSet_mem_contDiffGroupoid hs)
      /-
        🎉 no goals
      -/


/-- Typeclass defining smooth manifolds with corners with respect to a model with corners, over a
field `𝕜` and with infinite smoothness to simplify typeclass search and statements later on. -/
class SmoothManifoldWithCorners {𝕜 : Type*} [NontriviallyNormedField 𝕜] {E : Type*}
    [NormedAddCommGroup E] [NormedSpace 𝕜 E] {H : Type*} [TopologicalSpace H]
    (I : ModelWithCorners 𝕜 E H) (M : Type*) [TopologicalSpace M] [ChartedSpace H M] extends
    HasGroupoid M (contDiffGroupoid ∞ I) : Prop


theorem SmoothManifoldWithCorners.mk' {𝕜 : Type*} [NontriviallyNormedField 𝕜] {E : Type*}
    [NormedAddCommGroup E] [NormedSpace 𝕜 E] {H : Type*} [TopologicalSpace H]
    (I : ModelWithCorners 𝕜 E H) (M : Type*) [TopologicalSpace M] [ChartedSpace H M]
    [gr : HasGroupoid M (contDiffGroupoid ∞ I)] : SmoothManifoldWithCorners I M :=
  { gr with }


theorem smoothManifoldWithCorners_of_contDiffOn {𝕜 : Type*} [NontriviallyNormedField 𝕜]
    {E : Type*} [NormedAddCommGroup E] [NormedSpace 𝕜 E] {H : Type*} [TopologicalSpace H]
    (I : ModelWithCorners 𝕜 E H) (M : Type*) [TopologicalSpace M] [ChartedSpace H M]
    (h : ∀ e e' : PartialHomeomorph M H, e ∈ atlas H M → e' ∈ atlas H M →
      ContDiffOn 𝕜 ∞ (I ∘ e.symm ≫ₕ e' ∘ I.symm) (I.symm ⁻¹' (e.symm ≫ₕ e').source ∩ range I)) :
    SmoothManifoldWithCorners I M where
  compatible := by
    /-
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝² : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝¹ : TopologicalSpace M
      inst✝ : ChartedSpace H M
      h : ∀ (e e' : PartialHomeomorph M H), Membership.mem (atlas H M) e → Membershi …
      ⊢ ∀ {e e' : PartialHomeomorph M H}, Membership.mem (atlas H M) e → Membership. …
    -/
    haveI : HasGroupoid M (contDiffGroupoid ∞ I) := hasGroupoid_of_pregroupoid _ (h _ _)
    /-
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝² : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝¹ : TopologicalSpace M
      inst✝ : ChartedSpace H M
      h : ∀ (e e' : PartialHomeomorph M H), Membership.mem (atlas H M) e → Membershi …
      this : HasGroupoid M (contDiffGroupoid (↑Top.top) I)
      ⊢ ∀ {e e' : PartialHomeomorph M H}, Membership.mem (atlas H M) e → Membership. …
    -/
    apply StructureGroupoid.compatible
    /-
      🎉 no goals
    -/


/-- For any model with corners, the model space is a smooth manifold -/
instance model_space_smooth {𝕜 : Type*} [NontriviallyNormedField 𝕜] {E : Type*}
    [NormedAddCommGroup E] [NormedSpace 𝕜 E] {H : Type*} [TopologicalSpace H]
    {I : ModelWithCorners 𝕜 E H} : SmoothManifoldWithCorners I H :=
  { hasGroupoid_model_space _ _ with }


/-- The maximal atlas of `M` for the smooth manifold with corners structure corresponding to the
model with corners `I`. -/
def maximalAtlas :=
  (contDiffGroupoid ∞ I).maximalAtlas M


theorem subset_maximalAtlas [SmoothManifoldWithCorners I M] : atlas H M ⊆ maximalAtlas I M :=
  StructureGroupoid.subset_maximalAtlas _


theorem chart_mem_maximalAtlas [SmoothManifoldWithCorners I M] (x : M) :
    chartAt H x ∈ maximalAtlas I M :=
  StructureGroupoid.chart_mem_maximalAtlas _ x


theorem compatible_of_mem_maximalAtlas {e e' : PartialHomeomorph M H} (he : e ∈ maximalAtlas I M)
    (he' : e' ∈ maximalAtlas I M) : e.symm.trans e' ∈ contDiffGroupoid ∞ I :=
  StructureGroupoid.compatible_of_mem_maximalAtlas he he'


/-- The empty set is a smooth manifold w.r.t. any charted space and model. -/
instance empty [IsEmpty M] : SmoothManifoldWithCorners I M := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : IsEmpty M
    ⊢ SmoothManifoldWithCorners I M
  -/
  apply smoothManifoldWithCorners_of_contDiffOn
  /-
    case h
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : IsEmpty M
    ⊢ ∀ (e e' : PartialHomeomorph M H), Membership.mem (atlas H M) e → Membership. …
  -/
  intro e e' _ _ x hx
  /-
    case h
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : IsEmpty M
    e e' : PartialHomeomorph M H
    a✝¹ : Membership.mem (atlas H M) e
    a✝ : Membership.mem (atlas H M) e'
    x : E
    hx : Membership.mem (Inter.inter (Set.preimage (↑I.symm) (e.symm.trans e').sou …
    ⊢ ContDiffWithinAt 𝕜 (↑Top.top) (Function.comp (↑I) (Function.comp ↑(e.symm.tr …
  -/
  set t := I.symm ⁻¹' (e.symm ≫ₕ e').source ∩ range I
  -- Since `M` is empty, the condition about compatibility of transition maps is vacuous.
  have : (e.symm ≫ₕ e').source = ∅ := calc (e.symm ≫ₕ e').source
    _ = (e.symm.source) ∩ e.symm ⁻¹' e'.source := by rw [← PartialHomeomorph.trans_source]
    _ = (e.symm.source) ∩ e.symm ⁻¹' ∅ := by rw [eq_empty_of_isEmpty (e'.source)]
    _ = (e.symm.source) ∩ ∅ := by rw [preimage_empty]
    _ = ∅ := inter_empty e.symm.source
  have : t = ∅ := calc t
    _ = I.symm ⁻¹' (e.symm ≫ₕ e').source ∩ range I := by
      rw [← Subtype.preimage_val_eq_preimage_val_iff]
    _ = ∅ ∩ range I := by rw [this, preimage_empty]
    _ = ∅ := empty_inter (range I)
  /-
    case h
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : IsEmpty M
    e e' : PartialHomeomorph M H
    a✝¹ : Membership.mem (atlas H M) e
    a✝ : Membership.mem (atlas H M) e'
    x : E
    t : Set E := Inter.inter (Set.preimage (↑I.symm) (e.symm.trans e').source) (Se …
    hx : Membership.mem t x
    this✝ : Eq (e.symm.trans e').source EmptyCollection.emptyCollection
    this : Eq t EmptyCollection.emptyCollection
    ⊢ ContDiffWithinAt 𝕜 (↑Top.top) (Function.comp (↑I) (Function.comp ↑(e.symm.tr …
  -/
  apply (this ▸ hx).elim
  /-
    🎉 no goals
  -/


/-- The product of two smooth manifolds with corners is naturally a smooth manifold with corners. -/
instance prod {𝕜 : Type*} [NontriviallyNormedField 𝕜] {E : Type*} [NormedAddCommGroup E]
    [NormedSpace 𝕜 E] {E' : Type*} [NormedAddCommGroup E'] [NormedSpace 𝕜 E'] {H : Type*}
    [TopologicalSpace H] {I : ModelWithCorners 𝕜 E H} {H' : Type*} [TopologicalSpace H']
    {I' : ModelWithCorners 𝕜 E' H'} (M : Type*) [TopologicalSpace M] [ChartedSpace H M]
    [SmoothManifoldWithCorners I M] (M' : Type*) [TopologicalSpace M'] [ChartedSpace H' M']
    [SmoothManifoldWithCorners I' M'] : SmoothManifoldWithCorners (I.prod I') (M × M') where
  compatible := by
    /-
      𝕜✝ : Type u_1
      inst✝¹⁸ : NontriviallyNormedField 𝕜✝
      E✝ : Type u_2
      inst✝¹⁷ : NormedAddCommGroup E✝
      inst✝¹⁶ : NormedSpace 𝕜✝ E✝
      H✝ : Type u_3
      inst✝¹⁵ : TopologicalSpace H✝
      I✝ : ModelWithCorners 𝕜✝ E✝ H✝
      M✝ : Type u_4
      inst✝¹⁴ : TopologicalSpace M✝
      inst✝¹³ : ChartedSpace H✝ M✝
      𝕜 : Type u_5
      inst✝¹² : NontriviallyNormedField 𝕜
      E : Type u_6
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace 𝕜 E
      E' : Type u_7
      inst✝⁹ : NormedAddCommGroup E'
      inst✝⁸ : NormedSpace 𝕜 E'
      H : Type u_8
      inst✝⁷ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      H' : Type u_9
      inst✝⁶ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M : Type u_10
      inst✝⁵ : TopologicalSpace M
      inst✝⁴ : ChartedSpace H M
      inst✝³ : SmoothManifoldWithCorners I M
      M' : Type u_11
      inst✝² : TopologicalSpace M'
      inst✝¹ : ChartedSpace H' M'
      inst✝ : SmoothManifoldWithCorners I' M'
      ⊢ ∀ {e e' : PartialHomeomorph (Prod M M') (ModelProd H H')}, Membership.mem (a …
    -/
    rintro f g ⟨f1, hf1, f2, hf2, rfl⟩ ⟨g1, hg1, g2, hg2, rfl⟩
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro
      𝕜✝ : Type u_1
      inst✝¹⁸ : NontriviallyNormedField 𝕜✝
      E✝ : Type u_2
      inst✝¹⁷ : NormedAddCommGroup E✝
      inst✝¹⁶ : NormedSpace 𝕜✝ E✝
      H✝ : Type u_3
      inst✝¹⁵ : TopologicalSpace H✝
      I✝ : ModelWithCorners 𝕜✝ E✝ H✝
      M✝ : Type u_4
      inst✝¹⁴ : TopologicalSpace M✝
      inst✝¹³ : ChartedSpace H✝ M✝
      𝕜 : Type u_5
      inst✝¹² : NontriviallyNormedField 𝕜
      E : Type u_6
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace 𝕜 E
      E' : Type u_7
      inst✝⁹ : NormedAddCommGroup E'
      inst✝⁸ : NormedSpace 𝕜 E'
      H : Type u_8
      inst✝⁷ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      H' : Type u_9
      inst✝⁶ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M : Type u_10
      inst✝⁵ : TopologicalSpace M
      inst✝⁴ : ChartedSpace H M
      inst✝³ : SmoothManifoldWithCorners I M
      M' : Type u_11
      inst✝² : TopologicalSpace M'
      inst✝¹ : ChartedSpace H' M'
      inst✝ : SmoothManifoldWithCorners I' M'
      f1 : PartialHomeomorph M H
      hf1 : Membership.mem (atlas H M) f1
      f2 : PartialHomeomorph M' H'
      hf2 : Membership.mem (atlas H' M') f2
      g1 : PartialHomeomorph M H
      hg1 : Membership.mem (atlas H M) g1
      g2 : PartialHomeomorph M' H'
      hg2 : Membership.mem (atlas H' M') g2
      ⊢ Membership.mem (contDiffGroupoid (↑Top.top) (I.prod I')) ((f1.prod f2).symm. …
    -/
    rw [PartialHomeomorph.prod_symm, PartialHomeomorph.prod_trans]
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro
      𝕜✝ : Type u_1
      inst✝¹⁸ : NontriviallyNormedField 𝕜✝
      E✝ : Type u_2
      inst✝¹⁷ : NormedAddCommGroup E✝
      inst✝¹⁶ : NormedSpace 𝕜✝ E✝
      H✝ : Type u_3
      inst✝¹⁵ : TopologicalSpace H✝
      I✝ : ModelWithCorners 𝕜✝ E✝ H✝
      M✝ : Type u_4
      inst✝¹⁴ : TopologicalSpace M✝
      inst✝¹³ : ChartedSpace H✝ M✝
      𝕜 : Type u_5
      inst✝¹² : NontriviallyNormedField 𝕜
      E : Type u_6
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace 𝕜 E
      E' : Type u_7
      inst✝⁹ : NormedAddCommGroup E'
      inst✝⁸ : NormedSpace 𝕜 E'
      H : Type u_8
      inst✝⁷ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      H' : Type u_9
      inst✝⁶ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M : Type u_10
      inst✝⁵ : TopologicalSpace M
      inst✝⁴ : ChartedSpace H M
      inst✝³ : SmoothManifoldWithCorners I M
      M' : Type u_11
      inst✝² : TopologicalSpace M'
      inst✝¹ : ChartedSpace H' M'
      inst✝ : SmoothManifoldWithCorners I' M'
      f1 : PartialHomeomorph M H
      hf1 : Membership.mem (atlas H M) f1
      f2 : PartialHomeomorph M' H'
      hf2 : Membership.mem (atlas H' M') f2
      g1 : PartialHomeomorph M H
      hg1 : Membership.mem (atlas H M) g1
      g2 : PartialHomeomorph M' H'
      hg2 : Membership.mem (atlas H' M') g2
      ⊢ Membership.mem (contDiffGroupoid (↑Top.top) (I.prod I')) ((f1.symm.trans g1) …
    -/
    have h1 := (contDiffGroupoid ∞ I).compatible hf1 hg1
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro
      𝕜✝ : Type u_1
      inst✝¹⁸ : NontriviallyNormedField 𝕜✝
      E✝ : Type u_2
      inst✝¹⁷ : NormedAddCommGroup E✝
      inst✝¹⁶ : NormedSpace 𝕜✝ E✝
      H✝ : Type u_3
      inst✝¹⁵ : TopologicalSpace H✝
      I✝ : ModelWithCorners 𝕜✝ E✝ H✝
      M✝ : Type u_4
      inst✝¹⁴ : TopologicalSpace M✝
      inst✝¹³ : ChartedSpace H✝ M✝
      𝕜 : Type u_5
      inst✝¹² : NontriviallyNormedField 𝕜
      E : Type u_6
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace 𝕜 E
      E' : Type u_7
      inst✝⁹ : NormedAddCommGroup E'
      inst✝⁸ : NormedSpace 𝕜 E'
      H : Type u_8
      inst✝⁷ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      H' : Type u_9
      inst✝⁶ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M : Type u_10
      inst✝⁵ : TopologicalSpace M
      inst✝⁴ : ChartedSpace H M
      inst✝³ : SmoothManifoldWithCorners I M
      M' : Type u_11
      inst✝² : TopologicalSpace M'
      inst✝¹ : ChartedSpace H' M'
      inst✝ : SmoothManifoldWithCorners I' M'
      f1 : PartialHomeomorph M H
      hf1 : Membership.mem (atlas H M) f1
      f2 : PartialHomeomorph M' H'
      hf2 : Membership.mem (atlas H' M') f2
      g1 : PartialHomeomorph M H
      hg1 : Membership.mem (atlas H M) g1
      g2 : PartialHomeomorph M' H'
      hg2 : Membership.mem (atlas H' M') g2
      h1 : Membership.mem (contDiffGroupoid (↑Top.top) I) (f1.symm.trans g1)
      ⊢ Membership.mem (contDiffGroupoid (↑Top.top) (I.prod I')) ((f1.symm.trans g1) …
    -/
    have h2 := (contDiffGroupoid ∞ I').compatible hf2 hg2
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro
      𝕜✝ : Type u_1
      inst✝¹⁸ : NontriviallyNormedField 𝕜✝
      E✝ : Type u_2
      inst✝¹⁷ : NormedAddCommGroup E✝
      inst✝¹⁶ : NormedSpace 𝕜✝ E✝
      H✝ : Type u_3
      inst✝¹⁵ : TopologicalSpace H✝
      I✝ : ModelWithCorners 𝕜✝ E✝ H✝
      M✝ : Type u_4
      inst✝¹⁴ : TopologicalSpace M✝
      inst✝¹³ : ChartedSpace H✝ M✝
      𝕜 : Type u_5
      inst✝¹² : NontriviallyNormedField 𝕜
      E : Type u_6
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace 𝕜 E
      E' : Type u_7
      inst✝⁹ : NormedAddCommGroup E'
      inst✝⁸ : NormedSpace 𝕜 E'
      H : Type u_8
      inst✝⁷ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      H' : Type u_9
      inst✝⁶ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M : Type u_10
      inst✝⁵ : TopologicalSpace M
      inst✝⁴ : ChartedSpace H M
      inst✝³ : SmoothManifoldWithCorners I M
      M' : Type u_11
      inst✝² : TopologicalSpace M'
      inst✝¹ : ChartedSpace H' M'
      inst✝ : SmoothManifoldWithCorners I' M'
      f1 : PartialHomeomorph M H
      hf1 : Membership.mem (atlas H M) f1
      f2 : PartialHomeomorph M' H'
      hf2 : Membership.mem (atlas H' M') f2
      g1 : PartialHomeomorph M H
      hg1 : Membership.mem (atlas H M) g1
      g2 : PartialHomeomorph M' H'
      hg2 : Membership.mem (atlas H' M') g2
      h1 : Membership.mem (contDiffGroupoid (↑Top.top) I) (f1.symm.trans g1)
      h2 : Membership.mem (contDiffGroupoid (↑Top.top) I') (f2.symm.trans g2)
      ⊢ Membership.mem (contDiffGroupoid (↑Top.top) (I.prod I')) ((f1.symm.trans g1) …
    -/
    exact contDiffGroupoid_prod h1 h2
    /-
      🎉 no goals
    -/


theorem PartialHomeomorph.singleton_smoothManifoldWithCorners
    {𝕜 : Type*} [NontriviallyNormedField 𝕜] {E : Type*} [NormedAddCommGroup E] [NormedSpace 𝕜 E]
    {H : Type*} [TopologicalSpace H] {I : ModelWithCorners 𝕜 E H}
    {M : Type*} [TopologicalSpace M] (e : PartialHomeomorph M H) (h : e.source = Set.univ) :
    @SmoothManifoldWithCorners 𝕜 _ E _ _ H _ I M _ (e.singletonChartedSpace h) :=
  @SmoothManifoldWithCorners.mk' _ _ _ _ _ _ _ _ _ _ (id _) <|
    e.singleton_hasGroupoid h (contDiffGroupoid ∞ I)


theorem Topology.IsOpenEmbedding.singleton_smoothManifoldWithCorners {𝕜 E H : Type*}
    [NontriviallyNormedField 𝕜] [NormedAddCommGroup E] [NormedSpace 𝕜 E] [TopologicalSpace H]
    {I : ModelWithCorners 𝕜 E H} {M : Type*} [TopologicalSpace M] [Nonempty M] {f : M → H}
    (h : IsOpenEmbedding f) :
    @SmoothManifoldWithCorners 𝕜 _ E _ _ H _ I M _ h.singletonChartedSpace :=
                                                                    /-
                                                                      𝕜 : Type u_1
                                                                      E : Type u_2
                                                                      H : Type u_3
                                                                      inst✝⁵ : NontriviallyNormedField 𝕜
                                                                      inst✝⁴ : NormedAddCommGroup E
                                                                      inst✝³ : NormedSpace 𝕜 E
                                                                      inst✝² : TopologicalSpace H
                                                                      I : ModelWithCorners 𝕜 E H
                                                                      M : Type u_4
                                                                      inst✝¹ : TopologicalSpace M
                                                                      inst✝ : Nonempty M
                                                                      f : M → H
                                                                      h : Topology.IsOpenEmbedding f
                                                                      ⊢ Eq (Topology.IsOpenEmbedding.toPartialHomeomorph f h).source Set.univ
                                                                    -/
  (h.toPartialHomeomorph f).singleton_smoothManifoldWithCorners (by simp)
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[deprecated (since := "2024-10-18")]
alias OpenEmbedding.singleton_smoothManifoldWithCorners :=
  IsOpenEmbedding.singleton_smoothManifoldWithCorners


instance : SmoothManifoldWithCorners I s :=
  { s.instHasGroupoid (contDiffGroupoid ∞ I) with }


variable (I) in
/-- Given a chart `f` on a manifold with corners, `f.extend I` is the extended chart to the model
vector space. -/
@[simp, mfld_simps]
def extend : PartialEquiv M E :=
  f.toPartialEquiv ≫ I.toPartialEquiv


theorem extend_coe : ⇑(f.extend I) = I ∘ f :=
  rfl


theorem extend_coe_symm : ⇑(f.extend I).symm = f.symm ∘ I.symm :=
  rfl


theorem extend_source : (f.extend I).source = f.source := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    ⊢ Eq (f.extend I).source f.source
  -/
  rw [extend, PartialEquiv.trans_source, I.source_eq, preimage_univ, inter_univ]
  /-
    🎉 no goals
  -/


theorem isOpen_extend_source : IsOpen (f.extend I).source := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    ⊢ IsOpen (f.extend I).source
  -/
  rw [extend_source]
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    ⊢ IsOpen f.source
  -/
  exact f.open_source
  /-
    🎉 no goals
  -/


theorem extend_target : (f.extend I).target = I.symm ⁻¹' f.target ∩ range I := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    ⊢ Eq (f.extend I).target (Inter.inter (Set.preimage (↑I.symm) f.target) (Set.r …
  -/
  simp_rw [extend, PartialEquiv.trans_target, I.target_eq, I.toPartialEquiv_coe_symm, inter_comm]
  /-
    🎉 no goals
  -/


theorem extend_target' : (f.extend I).target = I '' f.target := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    ⊢ Eq (f.extend I).target (Set.image (↑I) f.target)
  -/
  rw [extend, PartialEquiv.trans_target'', I.source_eq, univ_inter, I.toPartialEquiv_coe]
  /-
    🎉 no goals
  -/


lemma isOpen_extend_target [I.Boundaryless] : IsOpen (f.extend I).target := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    inst✝ : I.Boundaryless
    ⊢ IsOpen (f.extend I).target
  -/
  rw [extend_target, I.range_eq_univ, inter_univ]
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    inst✝ : I.Boundaryless
    ⊢ IsOpen (Set.preimage (↑I.symm) f.target)
  -/
  exact I.continuous_symm.isOpen_preimage _ f.open_target
  /-
    🎉 no goals
  -/


theorem mapsTo_extend (hs : s ⊆ f.source) :
    MapsTo (f.extend I) s ((f.extend I).symm ⁻¹' s ∩ range I) := by
  rw [mapsTo', extend_coe, extend_coe_symm, preimage_comp, ← I.image_eq, image_comp,
    f.image_eq_target_inter_inv_preimage hs]
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    s : Set M
    hs : HasSubset.Subset s f.source
    ⊢ HasSubset.Subset (Set.image (↑I) (Inter.inter f.target (Set.preimage (↑f.sym …
  -/
  exact image_subset _ inter_subset_right
  /-
    🎉 no goals
  -/


theorem extend_left_inv {x : M} (hxf : x ∈ f.source) : (f.extend I).symm (f.extend I x) = x :=
                              /-
                                𝕜 : Type u_1
                                E : Type u_2
                                M : Type u_3
                                H : Type u_4
                                inst✝⁴ : NontriviallyNormedField 𝕜
                                inst✝³ : NormedAddCommGroup E
                                inst✝² : NormedSpace 𝕜 E
                                inst✝¹ : TopologicalSpace H
                                inst✝ : TopologicalSpace M
                                f : PartialHomeomorph M H
                                I : ModelWithCorners 𝕜 E H
                                x : M
                                hxf : Membership.mem f.source x
                                ⊢ Membership.mem (f.extend I).source x
                              -/
  (f.extend I).left_inv <| by rwa [f.extend_source]
                              /-
                                🎉 no goals
                              -/


/-- Variant of `f.extend_left_inv I`, stated in terms of images. -/
lemma extend_left_inv' (ht : t ⊆ f.source) : ((f.extend I).symm ∘ (f.extend I)) '' t = t :=
  EqOn.image_eq_self (fun _ hx ↦ f.extend_left_inv (ht hx))


theorem extend_source_mem_nhds {x : M} (h : x ∈ f.source) : (f.extend I).source ∈ 𝓝 x :=
                                          /-
                                            𝕜 : Type u_1
                                            E : Type u_2
                                            M : Type u_3
                                            H : Type u_4
                                            inst✝⁴ : NontriviallyNormedField 𝕜
                                            inst✝³ : NormedAddCommGroup E
                                            inst✝² : NormedSpace 𝕜 E
                                            inst✝¹ : TopologicalSpace H
                                            inst✝ : TopologicalSpace M
                                            f : PartialHomeomorph M H
                                            I : ModelWithCorners 𝕜 E H
                                            x : M
                                            h : Membership.mem f.source x
                                            ⊢ Membership.mem (f.extend I).source x
                                          -/
  (isOpen_extend_source f).mem_nhds <| by rwa [f.extend_source]
                                          /-
                                            🎉 no goals
                                          -/


theorem extend_source_mem_nhdsWithin {x : M} (h : x ∈ f.source) : (f.extend I).source ∈ 𝓝[s] x :=
  mem_nhdsWithin_of_mem_nhds <| extend_source_mem_nhds f h


theorem continuousOn_extend : ContinuousOn (f.extend I) (f.extend I).source := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    ⊢ ContinuousOn (↑(f.extend I)) (f.extend I).source
  -/
  refine I.continuous.comp_continuousOn ?_
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    ⊢ ContinuousOn (↑(f.symm.restr I.source).symm) (f.extend I).source
  -/
  rw [extend_source]
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    ⊢ ContinuousOn (↑(f.symm.restr I.source).symm) f.source
  -/
  exact f.continuousOn
  /-
    🎉 no goals
  -/


theorem continuousAt_extend {x : M} (h : x ∈ f.source) : ContinuousAt (f.extend I) x :=
  (continuousOn_extend f).continuousAt <| extend_source_mem_nhds f h


theorem map_extend_nhds {x : M} (hy : x ∈ f.source) :
    map (f.extend I) (𝓝 x) = 𝓝[range I] f.extend I x := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    x : M
    hy : Membership.mem f.source x
    ⊢ Eq (Filter.map (↑(f.extend I)) (nhds x)) (nhdsWithin (↑(f.extend I) x) (Set. …
  -/
  rwa [extend_coe, comp_apply, ← I.map_nhds_eq, ← f.map_nhds_eq, map_map]
  /-
    🎉 no goals
  -/


theorem map_extend_nhds_of_mem_interior_range {x : M} (hx : x ∈ f.source)
    (h'x : f.extend I x ∈ interior (range I)) :
    map (f.extend I) (𝓝 x) = 𝓝 (f.extend I x) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    x : M
    hx : Membership.mem f.source x
    h'x : Membership.mem (interior (Set.range ↑I)) (↑(f.extend I) x)
    ⊢ Eq (Filter.map (↑(f.extend I)) (nhds x)) (nhds (↑(f.extend I) x))
  -/
  rw [f.map_extend_nhds hx, nhdsWithin_eq_nhds]
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    x : M
    hx : Membership.mem f.source x
    h'x : Membership.mem (interior (Set.range ↑I)) (↑(f.extend I) x)
    ⊢ Membership.mem (nhds (↑(f.extend I) x)) (Set.range ↑I)
  -/
  exact mem_of_superset (isOpen_interior.mem_nhds h'x) interior_subset
  /-
    🎉 no goals
  -/


theorem map_extend_nhds_of_boundaryless [I.Boundaryless] {x : M} (hx : x ∈ f.source) :
    map (f.extend I) (𝓝 x) = 𝓝 (f.extend I x) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    inst✝ : I.Boundaryless
    x : M
    hx : Membership.mem f.source x
    ⊢ Eq (Filter.map (↑(f.extend I)) (nhds x)) (nhds (↑(f.extend I) x))
  -/
  rw [f.map_extend_nhds hx, I.range_eq_univ, nhdsWithin_univ]
  /-
    🎉 no goals
  -/


theorem extend_target_mem_nhdsWithin {y : M} (hy : y ∈ f.source) :
    (f.extend I).target ∈ 𝓝[range I] f.extend I y := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    y : M
    hy : Membership.mem f.source y
    ⊢ Membership.mem (nhdsWithin (↑(f.extend I) y) (Set.range ↑I)) (f.extend I).ta …
  -/
  rw [← PartialEquiv.image_source_eq_target, ← map_extend_nhds f hy]
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    y : M
    hy : Membership.mem f.source y
    ⊢ Membership.mem (Filter.map (↑(f.extend I)) (nhds y)) (Set.image (↑(f.extend  …
  -/
  exact image_mem_map (extend_source_mem_nhds _ hy)
  /-
    🎉 no goals
  -/


theorem extend_image_nhd_mem_nhds_of_boundaryless [I.Boundaryless] {x} (hx : x ∈ f.source)
    {s : Set M} (h : s ∈ 𝓝 x) : (f.extend I) '' s ∈ 𝓝 ((f.extend I) x) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    inst✝ : I.Boundaryless
    x : M
    hx : Membership.mem f.source x
    s : Set M
    h : Membership.mem (nhds x) s
    ⊢ Membership.mem (nhds (↑(f.extend I) x)) (Set.image (↑(f.extend I)) s)
  -/
  rw [← f.map_extend_nhds_of_boundaryless hx, Filter.mem_map]
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    inst✝ : I.Boundaryless
    x : M
    hx : Membership.mem f.source x
    s : Set M
    h : Membership.mem (nhds x) s
    ⊢ Membership.mem (nhds x) (Set.preimage (↑(f.extend I)) (Set.image (↑(f.extend …
  -/
  filter_upwards [h] using subset_preimage_image (f.extend I) s
  /-
    🎉 no goals
  -/


theorem extend_image_nhd_mem_nhds_of_mem_interior_range {x} (hx : x ∈ f.source)
    (h'x : f.extend I x ∈ interior (range I)) {s : Set M} (h : s ∈ 𝓝 x) :
    (f.extend I) '' s ∈ 𝓝 ((f.extend I) x) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    x : M
    hx : Membership.mem f.source x
    h'x : Membership.mem (interior (Set.range ↑I)) (↑(f.extend I) x)
    s : Set M
    h : Membership.mem (nhds x) s
    ⊢ Membership.mem (nhds (↑(f.extend I) x)) (Set.image (↑(f.extend I)) s)
  -/
  rw [← f.map_extend_nhds_of_mem_interior_range hx h'x, Filter.mem_map]
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    x : M
    hx : Membership.mem f.source x
    h'x : Membership.mem (interior (Set.range ↑I)) (↑(f.extend I) x)
    s : Set M
    h : Membership.mem (nhds x) s
    ⊢ Membership.mem (nhds x) (Set.preimage (↑(f.extend I)) (Set.image (↑(f.extend …
  -/
  filter_upwards [h] using subset_preimage_image (f.extend I) s
  /-
    🎉 no goals
  -/


                                                                         /-
                                                                           𝕜 : Type u_1
                                                                           E : Type u_2
                                                                           M : Type u_3
                                                                           H : Type u_4
                                                                           inst✝⁴ : NontriviallyNormedField 𝕜
                                                                           inst✝³ : NormedAddCommGroup E
                                                                           inst✝² : NormedSpace 𝕜 E
                                                                           inst✝¹ : TopologicalSpace H
                                                                           inst✝ : TopologicalSpace M
                                                                           f : PartialHomeomorph M H
                                                                           I : ModelWithCorners 𝕜 E H
                                                                           ⊢ HasSubset.Subset (f.extend I).target (Set.range ↑I)
                                                                         -/
theorem extend_target_subset_range : (f.extend I).target ⊆ range I := by simp only [mfld_simps]
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


lemma interior_extend_target_subset_interior_range :
    interior (f.extend I).target ⊆ interior (range I) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    ⊢ HasSubset.Subset (interior (f.extend I).target) (interior (Set.range ↑I))
  -/
  rw [f.extend_target, interior_inter, (f.open_target.preimage I.continuous_symm).interior_eq]
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    ⊢ HasSubset.Subset (Inter.inter (Set.preimage (↑I.symm) f.target) (interior (S …
  -/
  exact inter_subset_right
  /-
    🎉 no goals
  -/


/-- If `y ∈ f.target` and `I y ∈ interior (range I)`,
  then `I y` is an interior point of `(I ∘ f).target`. -/
lemma mem_interior_extend_target {y : H} (hy : y ∈ f.target)
    (hy' : I y ∈ interior (range I)) : I y ∈ interior (f.extend I).target := by
  rw [f.extend_target, interior_inter, (f.open_target.preimage I.continuous_symm).interior_eq,
    mem_inter_iff, mem_preimage]
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    y : H
    hy : Membership.mem f.target y
    hy' : Membership.mem (interior (Set.range ↑I)) (↑I y)
    ⊢ And (Membership.mem f.target (↑I.symm (↑I y))) (Membership.mem (interior (Se …
  -/
  exact ⟨mem_of_eq_of_mem (I.left_inv (y)) hy, hy'⟩
  /-
    🎉 no goals
  -/


theorem nhdsWithin_extend_target_eq {y : M} (hy : y ∈ f.source) :
    𝓝[(f.extend I).target] f.extend I y = 𝓝[range I] f.extend I y :=
  (nhdsWithin_mono _ (extend_target_subset_range _)).antisymm <|
    nhdsWithin_le_of_mem (extend_target_mem_nhdsWithin _ hy)


theorem extend_target_eventuallyEq {y : M} (hy : y ∈ f.source) :
    (f.extend I).target =ᶠ[𝓝 (f.extend I y)] range I :=
  nhdsWithin_eq_iff_eventuallyEq.1 (nhdsWithin_extend_target_eq _ hy)


theorem continuousAt_extend_symm' {x : E} (h : x ∈ (f.extend I).target) :
    ContinuousAt (f.extend I).symm x :=
  (f.continuousAt_symm h.2).comp I.continuous_symm.continuousAt


theorem continuousAt_extend_symm {x : M} (h : x ∈ f.source) :
    ContinuousAt (f.extend I).symm (f.extend I x) :=
                                                               /-
                                                                 𝕜 : Type u_1
                                                                 E : Type u_2
                                                                 M : Type u_3
                                                                 H : Type u_4
                                                                 inst✝⁴ : NontriviallyNormedField 𝕜
                                                                 inst✝³ : NormedAddCommGroup E
                                                                 inst✝² : NormedSpace 𝕜 E
                                                                 inst✝¹ : TopologicalSpace H
                                                                 inst✝ : TopologicalSpace M
                                                                 f : PartialHomeomorph M H
                                                                 I : ModelWithCorners 𝕜 E H
                                                                 x : M
                                                                 h : Membership.mem f.source x
                                                                 ⊢ Membership.mem (f.extend I).source x
                                                               -/
  continuousAt_extend_symm' f <| (f.extend I).map_source <| by rwa [f.extend_source]
                                                               /-
                                                                 🎉 no goals
                                                               -/


theorem continuousOn_extend_symm : ContinuousOn (f.extend I).symm (f.extend I).target := fun _ h =>
  (continuousAt_extend_symm' _ h).continuousWithinAt


theorem extend_symm_continuousWithinAt_comp_right_iff {X} [TopologicalSpace X] {g : M → X}
    {s : Set M} {x : M} :
    ContinuousWithinAt (g ∘ (f.extend I).symm) ((f.extend I).symm ⁻¹' s ∩ range I) (f.extend I x) ↔
      ContinuousWithinAt (g ∘ f.symm) (f.symm ⁻¹' s) (f x) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    X : Type u_8
    inst✝ : TopologicalSpace X
    g : M → X
    s : Set M
    x : M
    ⊢ Iff (ContinuousWithinAt (Function.comp g ↑(f.extend I).symm) (Inter.inter (S …
  -/
  rw [← I.symm_continuousWithinAt_comp_right_iff]; rfl
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem isOpen_extend_preimage' {s : Set E} (hs : IsOpen s) :
    IsOpen ((f.extend I).source ∩ f.extend I ⁻¹' s) :=
  (continuousOn_extend f).isOpen_inter_preimage (isOpen_extend_source _) hs


theorem isOpen_extend_preimage {s : Set E} (hs : IsOpen s) :
    IsOpen (f.source ∩ f.extend I ⁻¹' s) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    s : Set E
    hs : IsOpen s
    ⊢ IsOpen (Inter.inter f.source (Set.preimage (↑(f.extend I)) s))
  -/
  rw [← extend_source f (I := I)]; exact isOpen_extend_preimage' f hs
                                   /-
                                     🎉 no goals
                                   -/


theorem map_extend_nhdsWithin_eq_image {y : M} (hy : y ∈ f.source) :
    map (f.extend I) (𝓝[s] y) = 𝓝[f.extend I '' ((f.extend I).source ∩ s)] f.extend I y := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    s : Set M
    y : M
    hy : Membership.mem f.source y
    ⊢ Eq (Filter.map (↑(f.extend I)) (nhdsWithin y s)) (nhdsWithin (↑(f.extend I)  …
  -/
  set e := f.extend I
  calc
    map e (𝓝[s] y) = map e (𝓝[e.source ∩ s] y) :=
      congr_arg (map e) (nhdsWithin_inter_of_mem (extend_source_mem_nhdsWithin f hy)).symm
    _ = 𝓝[e '' (e.source ∩ s)] e y :=
      ((f.extend I).leftInvOn.mono inter_subset_left).map_nhdsWithin_eq
        ((f.extend I).left_inv <| by rwa [f.extend_source])
        (continuousAt_extend_symm f hy).continuousWithinAt
        (continuousAt_extend f hy).continuousWithinAt


theorem map_extend_nhdsWithin_eq_image_of_subset {y : M} (hy : y ∈ f.source) (hs : s ⊆ f.source) :
    map (f.extend I) (𝓝[s] y) = 𝓝[f.extend I '' s] f.extend I y := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    s : Set M
    y : M
    hy : Membership.mem f.source y
    hs : HasSubset.Subset s f.source
    ⊢ Eq (Filter.map (↑(f.extend I)) (nhdsWithin y s)) (nhdsWithin (↑(f.extend I)  …
  -/
  rw [map_extend_nhdsWithin_eq_image _ hy, inter_eq_self_of_subset_right]
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    s : Set M
    y : M
    hy : Membership.mem f.source y
    hs : HasSubset.Subset s f.source
    ⊢ HasSubset.Subset s (f.extend I).source
  -/
  rwa [extend_source]
  /-
    🎉 no goals
  -/


theorem map_extend_nhdsWithin {y : M} (hy : y ∈ f.source) :
    map (f.extend I) (𝓝[s] y) = 𝓝[(f.extend I).symm ⁻¹' s ∩ range I] f.extend I y := by
  rw [map_extend_nhdsWithin_eq_image f hy, nhdsWithin_inter, ←
    nhdsWithin_extend_target_eq _ hy, ← nhdsWithin_inter, (f.extend I).image_source_inter_eq',
    inter_comm]


theorem map_extend_symm_nhdsWithin {y : M} (hy : y ∈ f.source) :
    map (f.extend I).symm (𝓝[(f.extend I).symm ⁻¹' s ∩ range I] f.extend I y) = 𝓝[s] y := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    s : Set M
    y : M
    hy : Membership.mem f.source y
    ⊢ Eq (Filter.map (↑(f.extend I).symm) (nhdsWithin (↑(f.extend I) y) (Inter.int …
  -/
  rw [← map_extend_nhdsWithin f hy, map_map, Filter.map_congr, map_id]
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    s : Set M
    y : M
    hy : Membership.mem f.source y
    ⊢ (nhdsWithin y s).EventuallyEq (Function.comp ↑(f.extend I).symm ↑(f.extend I …
  -/
  exact (f.extend I).leftInvOn.eqOn.eventuallyEq_of_mem (extend_source_mem_nhdsWithin _ hy)
  /-
    🎉 no goals
  -/


theorem map_extend_symm_nhdsWithin_range {y : M} (hy : y ∈ f.source) :
    map (f.extend I).symm (𝓝[range I] f.extend I y) = 𝓝 y := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    y : M
    hy : Membership.mem f.source y
    ⊢ Eq (Filter.map (↑(f.extend I).symm) (nhdsWithin (↑(f.extend I) y) (Set.range …
  -/
  rw [← nhdsWithin_univ, ← map_extend_symm_nhdsWithin f (I := I) hy, preimage_univ, univ_inter]
  /-
    🎉 no goals
  -/


theorem tendsto_extend_comp_iff {α : Type*} {l : Filter α} {g : α → M}
    (hg : ∀ᶠ z in l, g z ∈ f.source) {y : M} (hy : y ∈ f.source) :
    Tendsto (f.extend I ∘ g) l (𝓝 (f.extend I y)) ↔ Tendsto g l (𝓝 y) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    α : Type u_8
    l : Filter α
    g : α → M
    hg : Filter.Eventually (fun z => Membership.mem f.source (g z)) l
    y : M
    hy : Membership.mem f.source y
    ⊢ Iff (Filter.Tendsto (Function.comp (↑(f.extend I)) g) l (nhds (↑(f.extend I) …
  -/
  refine ⟨fun h u hu ↦ mem_map.2 ?_, (continuousAt_extend _ hy).tendsto.comp⟩
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    α : Type u_8
    l : Filter α
    g : α → M
    hg : Filter.Eventually (fun z => Membership.mem f.source (g z)) l
    y : M
    hy : Membership.mem f.source y
    h : Filter.Tendsto (Function.comp (↑(f.extend I)) g) l (nhds (↑(f.extend I) y))
    u : Set M
    hu : Membership.mem (nhds y) u
    ⊢ Membership.mem l (Set.preimage g u)
  -/
  have := (f.continuousAt_extend_symm hy).tendsto.comp h
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    α : Type u_8
    l : Filter α
    g : α → M
    hg : Filter.Eventually (fun z => Membership.mem f.source (g z)) l
    y : M
    hy : Membership.mem f.source y
    h : Filter.Tendsto (Function.comp (↑(f.extend I)) g) l (nhds (↑(f.extend I) y))
    u : Set M
    hu : Membership.mem (nhds y) u
    this : Filter.Tendsto (Function.comp (↑(f.extend I).symm) (Function.comp (↑(f. …
    ⊢ Membership.mem l (Set.preimage g u)
  -/
  rw [extend_left_inv _ hy] at this
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    α : Type u_8
    l : Filter α
    g : α → M
    hg : Filter.Eventually (fun z => Membership.mem f.source (g z)) l
    y : M
    hy : Membership.mem f.source y
    h : Filter.Tendsto (Function.comp (↑(f.extend I)) g) l (nhds (↑(f.extend I) y))
    u : Set M
    hu : Membership.mem (nhds y) u
    this : Filter.Tendsto (Function.comp (↑(f.extend I).symm) (Function.comp (↑(f. …
    ⊢ Membership.mem l (Set.preimage g u)
  -/
  filter_upwards [hg, mem_map.1 (this hu)] with z hz hzu
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    α : Type u_8
    l : Filter α
    g : α → M
    hg : Filter.Eventually (fun z => Membership.mem f.source (g z)) l
    y : M
    hy : Membership.mem f.source y
    h : Filter.Tendsto (Function.comp (↑(f.extend I)) g) l (nhds (↑(f.extend I) y))
    u : Set M
    hu : Membership.mem (nhds y) u
    this : Filter.Tendsto (Function.comp (↑(f.extend I).symm) (Function.comp (↑(f. …
    z : α
    hz : Membership.mem f.source (g z)
    hzu : Membership.mem (Set.preimage (Function.comp (↑(f.extend I).symm) (Functi …
    ⊢ Membership.mem (Set.preimage g u) z
  -/
  simpa only [(· ∘ ·), extend_left_inv _ hz, mem_preimage] using hzu
  /-
    🎉 no goals
  -/

-- there is no definition `writtenInExtend` but we already use some made-up names in this file

theorem continuousWithinAt_writtenInExtend_iff {f' : PartialHomeomorph M' H'} {g : M → M'} {y : M}
    (hy : y ∈ f.source) (hgy : g y ∈ f'.source) (hmaps : MapsTo g s f'.source) :
    ContinuousWithinAt (f'.extend I' ∘ g ∘ (f.extend I).symm)
      ((f.extend I).symm ⁻¹' s ∩ range I) (f.extend I y) ↔ ContinuousWithinAt g s y := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    E' : Type u_5
    M' : Type u_6
    H' : Type u_7
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    inst✝³ : NormedAddCommGroup E'
    inst✝² : NormedSpace 𝕜 E'
    inst✝¹ : TopologicalSpace H'
    inst✝ : TopologicalSpace M'
    I' : ModelWithCorners 𝕜 E' H'
    s : Set M
    f' : PartialHomeomorph M' H'
    g : M → M'
    y : M
    hy : Membership.mem f.source y
    hgy : Membership.mem f'.source (g y)
    hmaps : Set.MapsTo g s f'.source
    ⊢ Iff (ContinuousWithinAt (Function.comp (↑(f'.extend I')) (Function.comp g ↑( …
  -/
  unfold ContinuousWithinAt
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    E' : Type u_5
    M' : Type u_6
    H' : Type u_7
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    inst✝³ : NormedAddCommGroup E'
    inst✝² : NormedSpace 𝕜 E'
    inst✝¹ : TopologicalSpace H'
    inst✝ : TopologicalSpace M'
    I' : ModelWithCorners 𝕜 E' H'
    s : Set M
    f' : PartialHomeomorph M' H'
    g : M → M'
    y : M
    hy : Membership.mem f.source y
    hgy : Membership.mem f'.source (g y)
    hmaps : Set.MapsTo g s f'.source
    ⊢ Iff (Filter.Tendsto (Function.comp (↑(f'.extend I')) (Function.comp g ↑(f.ex …
  -/
  simp only [comp_apply]
  rw [extend_left_inv _ hy, f'.tendsto_extend_comp_iff _ hgy,
    ← f.map_extend_symm_nhdsWithin (I := I) hy, tendsto_map'_iff]
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    E' : Type u_5
    M' : Type u_6
    H' : Type u_7
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    inst✝³ : NormedAddCommGroup E'
    inst✝² : NormedSpace 𝕜 E'
    inst✝¹ : TopologicalSpace H'
    inst✝ : TopologicalSpace M'
    I' : ModelWithCorners 𝕜 E' H'
    s : Set M
    f' : PartialHomeomorph M' H'
    g : M → M'
    y : M
    hy : Membership.mem f.source y
    hgy : Membership.mem f'.source (g y)
    hmaps : Set.MapsTo g s f'.source
    ⊢ Filter.Eventually (fun z => Membership.mem f'.source (Function.comp g (↑(f.e …
  -/
  rw [← f.map_extend_nhdsWithin (I := I) hy, eventually_map]
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    E' : Type u_5
    M' : Type u_6
    H' : Type u_7
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    inst✝³ : NormedAddCommGroup E'
    inst✝² : NormedSpace 𝕜 E'
    inst✝¹ : TopologicalSpace H'
    inst✝ : TopologicalSpace M'
    I' : ModelWithCorners 𝕜 E' H'
    s : Set M
    f' : PartialHomeomorph M' H'
    g : M → M'
    y : M
    hy : Membership.mem f.source y
    hgy : Membership.mem f'.source (g y)
    hmaps : Set.MapsTo g s f'.source
    ⊢ Filter.Eventually (fun a => Membership.mem f'.source (Function.comp g (↑(f.e …
  -/
  filter_upwards [inter_mem_nhdsWithin _ (f.open_source.mem_nhds hy)] with z hz
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    E' : Type u_5
    M' : Type u_6
    H' : Type u_7
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    inst✝³ : NormedAddCommGroup E'
    inst✝² : NormedSpace 𝕜 E'
    inst✝¹ : TopologicalSpace H'
    inst✝ : TopologicalSpace M'
    I' : ModelWithCorners 𝕜 E' H'
    s : Set M
    f' : PartialHomeomorph M' H'
    g : M → M'
    y : M
    hy : Membership.mem f.source y
    hgy : Membership.mem f'.source (g y)
    hmaps : Set.MapsTo g s f'.source
    z : M
    hz : Membership.mem (Inter.inter s f.source) z
    ⊢ Membership.mem f'.source (Function.comp g (↑(f.extend I).symm) (↑(f.extend I …
  -/
  rw [comp_apply, extend_left_inv _ hz.2]
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    E' : Type u_5
    M' : Type u_6
    H' : Type u_7
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    inst✝³ : NormedAddCommGroup E'
    inst✝² : NormedSpace 𝕜 E'
    inst✝¹ : TopologicalSpace H'
    inst✝ : TopologicalSpace M'
    I' : ModelWithCorners 𝕜 E' H'
    s : Set M
    f' : PartialHomeomorph M' H'
    g : M → M'
    y : M
    hy : Membership.mem f.source y
    hgy : Membership.mem f'.source (g y)
    hmaps : Set.MapsTo g s f'.source
    z : M
    hz : Membership.mem (Inter.inter s f.source) z
    ⊢ Membership.mem f'.source (g z)
  -/
  exact hmaps hz.1
  /-
    🎉 no goals
  -/

-- there is no definition `writtenInExtend` but we already use some made-up names in this file


/-- If `s ⊆ f.source` and `g x ∈ f'.source` whenever `x ∈ s`, then `g` is continuous on `s` if and
only if `g` written in charts `f.extend I` and `f'.extend I'` is continuous on `f.extend I '' s`. -/
theorem continuousOn_writtenInExtend_iff {f' : PartialHomeomorph M' H'} {g : M → M'}
    (hs : s ⊆ f.source) (hmaps : MapsTo g s f'.source) :
    ContinuousOn (f'.extend I' ∘ g ∘ (f.extend I).symm) (f.extend I '' s) ↔ ContinuousOn g s := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    E' : Type u_5
    M' : Type u_6
    H' : Type u_7
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    inst✝³ : NormedAddCommGroup E'
    inst✝² : NormedSpace 𝕜 E'
    inst✝¹ : TopologicalSpace H'
    inst✝ : TopologicalSpace M'
    I' : ModelWithCorners 𝕜 E' H'
    s : Set M
    f' : PartialHomeomorph M' H'
    g : M → M'
    hs : HasSubset.Subset s f.source
    hmaps : Set.MapsTo g s f'.source
    ⊢ Iff (ContinuousOn (Function.comp (↑(f'.extend I')) (Function.comp g ↑(f.exte …
  -/
  refine forall_mem_image.trans <| forall₂_congr fun x hx ↦ ?_
  refine (continuousWithinAt_congr_set ?_).trans
    (continuousWithinAt_writtenInExtend_iff _ (hs hx) (hmaps hx) hmaps)
  rw [← nhdsWithin_eq_iff_eventuallyEq, ← map_extend_nhdsWithin_eq_image_of_subset,
    ← map_extend_nhdsWithin]
  /-
    case hy
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    E' : Type u_5
    M' : Type u_6
    H' : Type u_7
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : TopologicalSpace H
    inst✝⁴ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    inst✝³ : NormedAddCommGroup E'
    inst✝² : NormedSpace 𝕜 E'
    inst✝¹ : TopologicalSpace H'
    inst✝ : TopologicalSpace M'
    I' : ModelWithCorners 𝕜 E' H'
    s : Set M
    f' : PartialHomeomorph M' H'
    g : M → M'
    hs : HasSubset.Subset s f.source
    hmaps : Set.MapsTo g s f'.source
    x : M
    hx : Membership.mem s x
    ⊢ Membership.mem f.source x
  -/
  exacts [hs hx, hs hx, hs]
  /-
    🎉 no goals
  -/


/-- Technical lemma ensuring that the preimage under an extended chart of a neighborhood of a point
in the source is a neighborhood of the preimage, within a set. -/
theorem extend_preimage_mem_nhdsWithin {x : M} (h : x ∈ f.source) (ht : t ∈ 𝓝[s] x) :
    (f.extend I).symm ⁻¹' t ∈ 𝓝[(f.extend I).symm ⁻¹' s ∩ range I] f.extend I x := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    s t : Set M
    x : M
    h : Membership.mem f.source x
    ht : Membership.mem (nhdsWithin x s) t
    ⊢ Membership.mem (nhdsWithin (↑(f.extend I) x) (Inter.inter (Set.preimage (↑(f …
  -/
  rwa [← map_extend_symm_nhdsWithin f (I := I) h, mem_map] at ht
  /-
    🎉 no goals
  -/


theorem extend_preimage_mem_nhds {x : M} (h : x ∈ f.source) (ht : t ∈ 𝓝 x) :
    (f.extend I).symm ⁻¹' t ∈ 𝓝 (f.extend I x) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    t : Set M
    x : M
    h : Membership.mem f.source x
    ht : Membership.mem (nhds x) t
    ⊢ Membership.mem (nhds (↑(f.extend I) x)) (Set.preimage (↑(f.extend I).symm) t)
  -/
  apply (continuousAt_extend_symm f h).preimage_mem_nhds
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    t : Set M
    x : M
    h : Membership.mem f.source x
    ht : Membership.mem (nhds x) t
    ⊢ Membership.mem (nhds (↑(f.extend I).symm (↑(f.extend I) x))) t
  -/
  rwa [(f.extend I).left_inv]
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    t : Set M
    x : M
    h : Membership.mem f.source x
    ht : Membership.mem (nhds x) t
    ⊢ Membership.mem (f.extend I).source x
  -/
  rwa [f.extend_source]
  /-
    🎉 no goals
  -/


/-- Technical lemma to rewrite suitably the preimage of an intersection under an extended chart, to
bring it into a convenient form to apply derivative lemmas. -/
theorem extend_preimage_inter_eq :
    (f.extend I).symm ⁻¹' (s ∩ t) ∩ range I =
      (f.extend I).symm ⁻¹' s ∩ range I ∩ (f.extend I).symm ⁻¹' t := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    s t : Set M
    ⊢ Eq (Inter.inter (Set.preimage (↑(f.extend I).symm) (Inter.inter s t)) (Set.r …
  -/
  mfld_set_tac
  /-
    🎉 no goals
  -/

-- Porting note: an `aux` lemma that is no longer needed. Delete?

theorem extend_symm_preimage_inter_range_eventuallyEq_aux {s : Set M} {x : M} (hx : x ∈ f.source) :
    ((f.extend I).symm ⁻¹' s ∩ range I : Set _) =ᶠ[𝓝 (f.extend I x)]
      ((f.extend I).target ∩ (f.extend I).symm ⁻¹' s : Set _) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    s : Set M
    x : M
    hx : Membership.mem f.source x
    ⊢ (nhds (↑(f.extend I) x)).EventuallyEq (Inter.inter (Set.preimage (↑(f.extend …
  -/
  rw [f.extend_target, inter_assoc, inter_comm (range I)]
  conv =>
    congr
    · skip
    rw [← univ_inter (_ ∩ range I)]
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    s : Set M
    x : M
    hx : Membership.mem f.source x
    ⊢ (nhds (↑(f.extend I) x)).EventuallyEq (Inter.inter Set.univ (Inter.inter (Se …
  -/
  refine (eventuallyEq_univ.mpr ?_).symm.inter EventuallyEq.rfl
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    s : Set M
    x : M
    hx : Membership.mem f.source x
    ⊢ Membership.mem (nhds (↑(f.extend I) x)) (Set.preimage (↑I.symm) f.target)
  -/
  refine I.continuousAt_symm.preimage_mem_nhds (f.open_target.mem_nhds ?_)
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    s : Set M
    x : M
    hx : Membership.mem f.source x
    ⊢ Membership.mem f.target (↑I.symm (↑(f.extend I) x))
  -/
  simp_rw [f.extend_coe, Function.comp_apply, I.left_inv, f.mapsTo hx]
  /-
    🎉 no goals
  -/


theorem extend_symm_preimage_inter_range_eventuallyEq {s : Set M} {x : M} (hs : s ⊆ f.source)
    (hx : x ∈ f.source) :
    ((f.extend I).symm ⁻¹' s ∩ range I : Set _) =ᶠ[𝓝 (f.extend I x)] f.extend I '' s := by
  rw [← nhdsWithin_eq_iff_eventuallyEq, ← map_extend_nhdsWithin _ hx,
    map_extend_nhdsWithin_eq_image_of_subset _ hx hs]


theorem extend_coord_change_source :
    ((f.extend I).symm ≫ f'.extend I).source = I '' (f.symm ≫ₕ f').source := by
  simp_rw [PartialEquiv.trans_source, I.image_eq, extend_source, PartialEquiv.symm_source,
    extend_target, inter_right_comm _ (range I)]
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f f' : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    ⊢ Eq (Inter.inter (Inter.inter (Set.preimage (↑I.symm) f.target) (Set.preimage …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem extend_image_source_inter :
    f.extend I '' (f.source ∩ f'.source) = ((f.extend I).symm ≫ f'.extend I).source := by
  simp_rw [f.extend_coord_change_source, f.extend_coe, image_comp I f, trans_source'', symm_symm,
    symm_target]


theorem extend_coord_change_source_mem_nhdsWithin {x : E}
    (hx : x ∈ ((f.extend I).symm ≫ f'.extend I).source) :
    ((f.extend I).symm ≫ f'.extend I).source ∈ 𝓝[range I] x := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f f' : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    x : E
    hx : Membership.mem ((f.extend I).symm.trans (f'.extend I)).source x
    ⊢ Membership.mem (nhdsWithin x (Set.range ↑I)) ((f.extend I).symm.trans (f'.ex …
  -/
  rw [f.extend_coord_change_source] at hx ⊢
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f f' : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    x : E
    hx : Membership.mem (Set.image (↑I) (f.symm.trans f').source) x
    ⊢ Membership.mem (nhdsWithin x (Set.range ↑I)) (Set.image (↑I) (f.symm.trans f …
  -/
  obtain ⟨x, hx, rfl⟩ := hx
  /-
    case intro.intro
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f f' : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    x : H
    hx : Membership.mem (f.symm.trans f').source x
    ⊢ Membership.mem (nhdsWithin (↑I x) (Set.range ↑I)) (Set.image (↑I) (f.symm.tr …
  -/
  refine I.image_mem_nhdsWithin ?_
  /-
    case intro.intro
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f f' : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    x : H
    hx : Membership.mem (f.symm.trans f').source x
    ⊢ Membership.mem (nhds x) (f.symm.trans f').source
  -/
  exact (PartialHomeomorph.open_source _).mem_nhds hx
  /-
    🎉 no goals
  -/


theorem extend_coord_change_source_mem_nhdsWithin' {x : M} (hxf : x ∈ f.source)
    (hxf' : x ∈ f'.source) :
    ((f.extend I).symm ≫ f'.extend I).source ∈ 𝓝[range I] f.extend I x := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f f' : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    x : M
    hxf : Membership.mem f.source x
    hxf' : Membership.mem f'.source x
    ⊢ Membership.mem (nhdsWithin (↑(f.extend I) x) (Set.range ↑I)) ((f.extend I).s …
  -/
  apply extend_coord_change_source_mem_nhdsWithin
  /-
    case hx
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f f' : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    x : M
    hxf : Membership.mem f.source x
    hxf' : Membership.mem f'.source x
    ⊢ Membership.mem ((f.extend I).symm.trans (f'.extend I)).source (↑(f.extend I) …
  -/
  rw [← extend_image_source_inter]
  /-
    case hx
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : TopologicalSpace H
    inst✝ : TopologicalSpace M
    f f' : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    x : M
    hxf : Membership.mem f.source x
    hxf' : Membership.mem f'.source x
    ⊢ Membership.mem (Set.image (↑(f.extend I)) (Inter.inter f.source f'.source))  …
  -/
  exact mem_image_of_mem _ ⟨hxf, hxf'⟩
  /-
    🎉 no goals
  -/


theorem contDiffOn_extend_coord_change [ChartedSpace H M] (hf : f ∈ maximalAtlas I M)
    (hf' : f' ∈ maximalAtlas I M) :
    ContDiffOn 𝕜 ∞ (f.extend I ∘ (f'.extend I).symm) ((f'.extend I).symm ≫ f.extend I).source := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    f f' : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    inst✝ : ChartedSpace H M
    hf : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I M) f
    hf' : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I M) f'
    ⊢ ContDiffOn 𝕜 (↑Top.top) (Function.comp ↑(f.extend I) ↑(f'.extend I).symm) (( …
  -/
  rw [extend_coord_change_source, I.image_eq]
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    f f' : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    inst✝ : ChartedSpace H M
    hf : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I M) f
    hf' : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I M) f'
    ⊢ ContDiffOn 𝕜 (↑Top.top) (Function.comp ↑(f.extend I) ↑(f'.extend I).symm) (I …
  -/
  exact (StructureGroupoid.compatible_of_mem_maximalAtlas hf' hf).1
  /-
    🎉 no goals
  -/


theorem contDiffWithinAt_extend_coord_change [ChartedSpace H M] (hf : f ∈ maximalAtlas I M)
    (hf' : f' ∈ maximalAtlas I M) {x : E} (hx : x ∈ ((f'.extend I).symm ≫ f.extend I).source) :
    ContDiffWithinAt 𝕜 ∞ (f.extend I ∘ (f'.extend I).symm) (range I) x := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    f f' : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    inst✝ : ChartedSpace H M
    hf : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I M) f
    hf' : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I M) f'
    x : E
    hx : Membership.mem ((f'.extend I).symm.trans (f.extend I)).source x
    ⊢ ContDiffWithinAt 𝕜 (↑Top.top) (Function.comp ↑(f.extend I) ↑(f'.extend I).sy …
  -/
  apply (contDiffOn_extend_coord_change hf hf' x hx).mono_of_mem_nhdsWithin
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    f f' : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    inst✝ : ChartedSpace H M
    hf : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I M) f
    hf' : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I M) f'
    x : E
    hx : Membership.mem ((f'.extend I).symm.trans (f.extend I)).source x
    ⊢ Membership.mem (nhdsWithin x (Set.range ↑I)) ((f'.extend I).symm.trans (f.ex …
  -/
  rw [extend_coord_change_source] at hx ⊢
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    f f' : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    inst✝ : ChartedSpace H M
    hf : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I M) f
    hf' : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I M) f'
    x : E
    hx : Membership.mem (Set.image (↑I) (f'.symm.trans f).source) x
    ⊢ Membership.mem (nhdsWithin x (Set.range ↑I)) (Set.image (↑I) (f'.symm.trans  …
  -/
  obtain ⟨z, hz, rfl⟩ := hx
  /-
    case intro.intro
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    f f' : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    inst✝ : ChartedSpace H M
    hf : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I M) f
    hf' : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I M) f'
    z : H
    hz : Membership.mem (f'.symm.trans f).source z
    ⊢ Membership.mem (nhdsWithin (↑I z) (Set.range ↑I)) (Set.image (↑I) (f'.symm.t …
  -/
  exact I.image_mem_nhdsWithin ((PartialHomeomorph.open_source _).mem_nhds hz)
  /-
    🎉 no goals
  -/


theorem contDiffWithinAt_extend_coord_change' [ChartedSpace H M] (hf : f ∈ maximalAtlas I M)
    (hf' : f' ∈ maximalAtlas I M) {x : M} (hxf : x ∈ f.source) (hxf' : x ∈ f'.source) :
    ContDiffWithinAt 𝕜 ∞ (f.extend I ∘ (f'.extend I).symm) (range I) (f'.extend I x) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    f f' : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    inst✝ : ChartedSpace H M
    hf : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I M) f
    hf' : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I M) f'
    x : M
    hxf : Membership.mem f.source x
    hxf' : Membership.mem f'.source x
    ⊢ ContDiffWithinAt 𝕜 (↑Top.top) (Function.comp ↑(f.extend I) ↑(f'.extend I).sy …
  -/
  refine contDiffWithinAt_extend_coord_change hf hf' ?_
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    f f' : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    inst✝ : ChartedSpace H M
    hf : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I M) f
    hf' : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I M) f'
    x : M
    hxf : Membership.mem f.source x
    hxf' : Membership.mem f'.source x
    ⊢ Membership.mem ((f'.extend I).symm.trans (f.extend I)).source (↑(f'.extend I …
  -/
  rw [← extend_image_source_inter]
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    f f' : PartialHomeomorph M H
    I : ModelWithCorners 𝕜 E H
    inst✝ : ChartedSpace H M
    hf : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I M) f
    hf' : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I M) f'
    x : M
    hxf : Membership.mem f.source x
    hxf' : Membership.mem f'.source x
    ⊢ Membership.mem (Set.image (↑(f'.extend I)) (Inter.inter f'.source f.source)) …
  -/
  exact mem_image_of_mem _ ⟨hxf', hxf⟩
  /-
    🎉 no goals
  -/


variable (I) in
/-- The preferred extended chart on a manifold with corners around a point `x`, from a neighborhood
of `x` to the model vector space. -/
@[simp, mfld_simps]
def extChartAt (x : M) : PartialEquiv M E :=
  (chartAt H x).extend I


theorem extChartAt_coe (x : M) : ⇑(extChartAt I x) = I ∘ chartAt H x :=
  rfl


theorem extChartAt_coe_symm (x : M) : ⇑(extChartAt I x).symm = (chartAt H x).symm ∘ I.symm :=
  rfl


variable (I) in
theorem extChartAt_source (x : M) : (extChartAt I x).source = (chartAt H x).source :=
  extend_source _


theorem isOpen_extChartAt_source (x : M) : IsOpen (extChartAt I x).source :=
  isOpen_extend_source _


theorem mem_extChartAt_source (x : M) : x ∈ (extChartAt I x).source := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    inst✝ : ChartedSpace H M
    x : M
    ⊢ Membership.mem (extChartAt I x).source x
  -/
  simp only [extChartAt_source, mem_chart_source]
  /-
    🎉 no goals
  -/


theorem mem_extChartAt_target (x : M) : extChartAt I x x ∈ (extChartAt I x).target :=
  (extChartAt I x).map_source <| mem_extChartAt_source _


variable (I) in
theorem extChartAt_target (x : M) :
    (extChartAt I x).target = I.symm ⁻¹' (chartAt H x).target ∩ range I :=
  extend_target _


theorem uniqueDiffOn_extChartAt_target (x : M) : UniqueDiffOn 𝕜 (extChartAt I x).target := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    inst✝ : ChartedSpace H M
    x : M
    ⊢ UniqueDiffOn 𝕜 (extChartAt I x).target
  -/
  rw [extChartAt_target]
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    inst✝ : ChartedSpace H M
    x : M
    ⊢ UniqueDiffOn 𝕜 (Inter.inter (Set.preimage (↑I.symm) (chartAt H x).target) (S …
  -/
  exact I.uniqueDiffOn_preimage (chartAt H x).open_target
  /-
    🎉 no goals
  -/


theorem uniqueDiffWithinAt_extChartAt_target (x : M) :
    UniqueDiffWithinAt 𝕜 (extChartAt I x).target (extChartAt I x x) :=
  uniqueDiffOn_extChartAt_target x _ <| mem_extChartAt_target x


theorem extChartAt_to_inv (x : M) : (extChartAt I x).symm ((extChartAt I x) x) = x :=
  (extChartAt I x).left_inv (mem_extChartAt_source x)


theorem mapsTo_extChartAt {x : M} (hs : s ⊆ (chartAt H x).source) :
    MapsTo (extChartAt I x) s ((extChartAt I x).symm ⁻¹' s ∩ range I) :=
  mapsTo_extend _ hs


theorem extChartAt_source_mem_nhds' {x x' : M} (h : x' ∈ (extChartAt I x).source) :
    (extChartAt I x).source ∈ 𝓝 x' :=
                                 /-
                                   𝕜 : Type u_1
                                   E : Type u_2
                                   M : Type u_3
                                   H : Type u_4
                                   inst✝⁵ : NontriviallyNormedField 𝕜
                                   inst✝⁴ : NormedAddCommGroup E
                                   inst✝³ : NormedSpace 𝕜 E
                                   inst✝² : TopologicalSpace H
                                   inst✝¹ : TopologicalSpace M
                                   I : ModelWithCorners 𝕜 E H
                                   inst✝ : ChartedSpace H M
                                   x x' : M
                                   h : Membership.mem (extChartAt I x).source x'
                                   ⊢ Membership.mem (chartAt H x).source x'
                                 -/
  extend_source_mem_nhds _ <| by rwa [← extChartAt_source I]
                                 /-
                                   🎉 no goals
                                 -/


theorem extChartAt_source_mem_nhds (x : M) : (extChartAt I x).source ∈ 𝓝 x :=
  extChartAt_source_mem_nhds' (mem_extChartAt_source x)


theorem extChartAt_source_mem_nhdsWithin' {x x' : M} (h : x' ∈ (extChartAt I x).source) :
    (extChartAt I x).source ∈ 𝓝[s] x' :=
  mem_nhdsWithin_of_mem_nhds (extChartAt_source_mem_nhds' h)


theorem extChartAt_source_mem_nhdsWithin (x : M) : (extChartAt I x).source ∈ 𝓝[s] x :=
  mem_nhdsWithin_of_mem_nhds (extChartAt_source_mem_nhds x)


theorem continuousOn_extChartAt (x : M) : ContinuousOn (extChartAt I x) (extChartAt I x).source :=
  continuousOn_extend _


theorem continuousAt_extChartAt' {x x' : M} (h : x' ∈ (extChartAt I x).source) :
    ContinuousAt (extChartAt I x) x' :=
                              /-
                                𝕜 : Type u_1
                                E : Type u_2
                                M : Type u_3
                                H : Type u_4
                                inst✝⁵ : NontriviallyNormedField 𝕜
                                inst✝⁴ : NormedAddCommGroup E
                                inst✝³ : NormedSpace 𝕜 E
                                inst✝² : TopologicalSpace H
                                inst✝¹ : TopologicalSpace M
                                I : ModelWithCorners 𝕜 E H
                                inst✝ : ChartedSpace H M
                                x x' : M
                                h : Membership.mem (extChartAt I x).source x'
                                ⊢ Membership.mem (chartAt H x).source x'
                              -/
  continuousAt_extend _ <| by rwa [← extChartAt_source I]
                              /-
                                🎉 no goals
                              -/


theorem continuousAt_extChartAt (x : M) : ContinuousAt (extChartAt I x) x :=
  continuousAt_extChartAt' (mem_extChartAt_source x)


theorem map_extChartAt_nhds' {x y : M} (hy : y ∈ (extChartAt I x).source) :
    map (extChartAt I x) (𝓝 y) = 𝓝[range I] extChartAt I x y :=
                          /-
                            𝕜 : Type u_1
                            E : Type u_2
                            M : Type u_3
                            H : Type u_4
                            inst✝⁵ : NontriviallyNormedField 𝕜
                            inst✝⁴ : NormedAddCommGroup E
                            inst✝³ : NormedSpace 𝕜 E
                            inst✝² : TopologicalSpace H
                            inst✝¹ : TopologicalSpace M
                            I : ModelWithCorners 𝕜 E H
                            inst✝ : ChartedSpace H M
                            x y : M
                            hy : Membership.mem (extChartAt I x).source y
                            ⊢ Membership.mem (chartAt H x).source y
                          -/
  map_extend_nhds _ <| by rwa [← extChartAt_source I]
                          /-
                            🎉 no goals
                          -/


theorem map_extChartAt_nhds (x : M) : map (extChartAt I x) (𝓝 x) = 𝓝[range I] extChartAt I x x :=
  map_extChartAt_nhds' <| mem_extChartAt_source x


theorem map_extChartAt_nhds_of_boundaryless [I.Boundaryless] (x : M) :
    map (extChartAt I x) (𝓝 x) = 𝓝 (extChartAt I x x) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : TopologicalSpace H
    inst✝² : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    inst✝¹ : ChartedSpace H M
    inst✝ : I.Boundaryless
    x : M
    ⊢ Eq (Filter.map (↑(extChartAt I x)) (nhds x)) (nhds (↑(extChartAt I x) x))
  -/
  rw [extChartAt]
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : TopologicalSpace H
    inst✝² : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    inst✝¹ : ChartedSpace H M
    inst✝ : I.Boundaryless
    x : M
    ⊢ Eq (Filter.map (↑((chartAt H x).extend I)) (nhds x)) (nhds (↑((chartAt H x). …
  -/
  exact map_extend_nhds_of_boundaryless (chartAt H x) (mem_chart_source H x)
  /-
    🎉 no goals
  -/


theorem extChartAt_image_nhd_mem_nhds_of_mem_interior_range {x y} (hx : y ∈ (extChartAt I x).source)
    (h'x : extChartAt I x y ∈ interior (range I)) {s : Set M} (h : s ∈ 𝓝 y) :
    (extChartAt I x) '' s ∈ 𝓝 (extChartAt I x y) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    inst✝ : ChartedSpace H M
    x y : M
    hx : Membership.mem (extChartAt I x).source y
    h'x : Membership.mem (interior (Set.range ↑I)) (↑(extChartAt I x) y)
    s : Set M
    h : Membership.mem (nhds y) s
    ⊢ Membership.mem (nhds (↑(extChartAt I x) y)) (Set.image (↑(extChartAt I x)) s)
  -/
  rw [extChartAt]
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    inst✝ : ChartedSpace H M
    x y : M
    hx : Membership.mem (extChartAt I x).source y
    h'x : Membership.mem (interior (Set.range ↑I)) (↑(extChartAt I x) y)
    s : Set M
    h : Membership.mem (nhds y) s
    ⊢ Membership.mem (nhds (↑((chartAt H x).extend I) y)) (Set.image (↑((chartAt H …
  -/
  exact extend_image_nhd_mem_nhds_of_mem_interior_range _ (by simpa using hx) h'x h
  /-
    🎉 no goals
  -/


variable {x} in
theorem extChartAt_image_nhd_mem_nhds_of_boundaryless [I.Boundaryless]
    {x : M} (hx : s ∈ 𝓝 x) : extChartAt I x '' s ∈ 𝓝 (extChartAt I x x) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : TopologicalSpace H
    inst✝² : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    s : Set M
    inst✝¹ : ChartedSpace H M
    inst✝ : I.Boundaryless
    x : M
    hx : Membership.mem (nhds x) s
    ⊢ Membership.mem (nhds (↑(extChartAt I x) x)) (Set.image (↑(extChartAt I x)) s)
  -/
  rw [extChartAt]
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : TopologicalSpace H
    inst✝² : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    s : Set M
    inst✝¹ : ChartedSpace H M
    inst✝ : I.Boundaryless
    x : M
    hx : Membership.mem (nhds x) s
    ⊢ Membership.mem (nhds (↑((chartAt H x).extend I) x)) (Set.image (↑((chartAt H …
  -/
  exact extend_image_nhd_mem_nhds_of_boundaryless _ (mem_chart_source H x) hx
  /-
    🎉 no goals
  -/


theorem extChartAt_target_mem_nhdsWithin' {x y : M} (hy : y ∈ (extChartAt I x).source) :
    (extChartAt I x).target ∈ 𝓝[range I] extChartAt I x y :=
                                       /-
                                         𝕜 : Type u_1
                                         E : Type u_2
                                         M : Type u_3
                                         H : Type u_4
                                         inst✝⁵ : NontriviallyNormedField 𝕜
                                         inst✝⁴ : NormedAddCommGroup E
                                         inst✝³ : NormedSpace 𝕜 E
                                         inst✝² : TopologicalSpace H
                                         inst✝¹ : TopologicalSpace M
                                         I : ModelWithCorners 𝕜 E H
                                         inst✝ : ChartedSpace H M
                                         x y : M
                                         hy : Membership.mem (extChartAt I x).source y
                                         ⊢ Membership.mem (chartAt H x).source y
                                       -/
  extend_target_mem_nhdsWithin _ <| by rwa [← extChartAt_source I]
                                       /-
                                         🎉 no goals
                                       -/


theorem extChartAt_target_mem_nhdsWithin (x : M) :
    (extChartAt I x).target ∈ 𝓝[range I] extChartAt I x x :=
  extChartAt_target_mem_nhdsWithin' (mem_extChartAt_source x)


theorem extChartAt_target_mem_nhdsWithin_of_mem {x : M} {y : E} (hy : y ∈ (extChartAt I x).target) :
    (extChartAt I x).target ∈ 𝓝[range I] y := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    inst✝ : ChartedSpace H M
    x : M
    y : E
    hy : Membership.mem (extChartAt I x).target y
    ⊢ Membership.mem (nhdsWithin y (Set.range ↑I)) (extChartAt I x).target
  -/
  rw [← (extChartAt I x).right_inv hy]
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    inst✝ : ChartedSpace H M
    x : M
    y : E
    hy : Membership.mem (extChartAt I x).target y
    ⊢ Membership.mem (nhdsWithin (↑(extChartAt I x) (↑(extChartAt I x).symm y)) (S …
  -/
  apply extChartAt_target_mem_nhdsWithin'
  /-
    case hy
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    inst✝ : ChartedSpace H M
    x : M
    y : E
    hy : Membership.mem (extChartAt I x).target y
    ⊢ Membership.mem (extChartAt I x).source (↑(extChartAt I x).symm y)
  -/
  exact (extChartAt I x).map_target hy
  /-
    🎉 no goals
  -/


theorem extChartAt_target_union_compl_range_mem_nhds_of_mem {y : E} {x : M}
    (hy : y ∈ (extChartAt I x).target) : (extChartAt I x).target ∪ (range I)ᶜ ∈ 𝓝 y := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    inst✝ : ChartedSpace H M
    y : E
    x : M
    hy : Membership.mem (extChartAt I x).target y
    ⊢ Membership.mem (nhds y) (Union.union (extChartAt I x).target (HasCompl.compl …
  -/
  rw [← nhdsWithin_univ, ← union_compl_self (range I), nhdsWithin_union]
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    inst✝ : ChartedSpace H M
    y : E
    x : M
    hy : Membership.mem (extChartAt I x).target y
    ⊢ Membership.mem (Max.max (nhdsWithin y (Set.range ↑I)) (nhdsWithin y (HasComp …
  -/
  exact Filter.union_mem_sup (extChartAt_target_mem_nhdsWithin_of_mem hy) self_mem_nhdsWithin
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-27")] alias
extChartAt_target_union_comp_range_mem_nhds_of_mem :=
extChartAt_target_union_compl_range_mem_nhds_of_mem


/-- If we're boundaryless, `extChartAt` has open target -/
theorem isOpen_extChartAt_target [I.Boundaryless] (x : M) : IsOpen (extChartAt I x).target := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : TopologicalSpace H
    inst✝² : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    inst✝¹ : ChartedSpace H M
    inst✝ : I.Boundaryless
    x : M
    ⊢ IsOpen (extChartAt I x).target
  -/
  simp_rw [extChartAt_target, I.range_eq_univ, inter_univ]
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : TopologicalSpace H
    inst✝² : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    inst✝¹ : ChartedSpace H M
    inst✝ : I.Boundaryless
    x : M
    ⊢ IsOpen (Set.preimage (↑I.symm) (chartAt H x).target)
  -/
  exact (PartialHomeomorph.open_target _).preimage I.continuous_symm
  /-
    🎉 no goals
  -/


/-- If we're boundaryless, `(extChartAt I x).target` is a neighborhood of the key point -/
theorem extChartAt_target_mem_nhds [I.Boundaryless] (x : M) :
    (extChartAt I x).target ∈ 𝓝 (extChartAt I x x) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : TopologicalSpace H
    inst✝² : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    inst✝¹ : ChartedSpace H M
    inst✝ : I.Boundaryless
    x : M
    ⊢ Membership.mem (nhds (↑(extChartAt I x) x)) (extChartAt I x).target
  -/
  convert extChartAt_target_mem_nhdsWithin x
  /-
    case h.e'_4
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : TopologicalSpace H
    inst✝² : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    inst✝¹ : ChartedSpace H M
    inst✝ : I.Boundaryless
    x : M
    ⊢ Eq (nhds (↑(extChartAt I x) x)) (nhdsWithin (↑(extChartAt I x) x) (Set.range …
  -/
  simp only [I.range_eq_univ, nhdsWithin_univ]
  /-
    🎉 no goals
  -/


/-- If we're boundaryless, `(extChartAt I x).target` is a neighborhood of any of its points -/
theorem extChartAt_target_mem_nhds' [I.Boundaryless] {x : M} {y : E}
    (m : y ∈ (extChartAt I x).target) : (extChartAt I x).target ∈ 𝓝 y :=
  (isOpen_extChartAt_target x).mem_nhds m


theorem extChartAt_target_subset_range (x : M) : (extChartAt I x).target ⊆ range I := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    inst✝ : ChartedSpace H M
    x : M
    ⊢ HasSubset.Subset (extChartAt I x).target (Set.range ↑I)
  -/
  simp only [mfld_simps]
  /-
    🎉 no goals
  -/


/-- Around the image of a point in the source, the neighborhoods are the same
within `(extChartAt I x).target` and within `range I`. -/
theorem nhdsWithin_extChartAt_target_eq' {x y : M} (hy : y ∈ (extChartAt I x).source) :
    𝓝[(extChartAt I x).target] extChartAt I x y = 𝓝[range I] extChartAt I x y :=
                                      /-
                                        𝕜 : Type u_1
                                        E : Type u_2
                                        M : Type u_3
                                        H : Type u_4
                                        inst✝⁵ : NontriviallyNormedField 𝕜
                                        inst✝⁴ : NormedAddCommGroup E
                                        inst✝³ : NormedSpace 𝕜 E
                                        inst✝² : TopologicalSpace H
                                        inst✝¹ : TopologicalSpace M
                                        I : ModelWithCorners 𝕜 E H
                                        inst✝ : ChartedSpace H M
                                        x y : M
                                        hy : Membership.mem (extChartAt I x).source y
                                        ⊢ Membership.mem (chartAt H x).source y
                                      -/
  nhdsWithin_extend_target_eq _ <| by rwa [← extChartAt_source I]
                                      /-
                                        🎉 no goals
                                      -/


/-- Around a point in the target, the neighborhoods are the same within `(extChartAt I x).target`
and within `range I`. -/
theorem nhdsWithin_extChartAt_target_eq_of_mem {x : M} {z : E} (hz : z ∈ (extChartAt I x).target) :
    𝓝[(extChartAt I x).target] z = 𝓝[range I] z := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    inst✝ : ChartedSpace H M
    x : M
    z : E
    hz : Membership.mem (extChartAt I x).target z
    ⊢ Eq (nhdsWithin z (extChartAt I x).target) (nhdsWithin z (Set.range ↑I))
  -/
  rw [← PartialEquiv.right_inv (extChartAt I x) hz]
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    inst✝ : ChartedSpace H M
    x : M
    z : E
    hz : Membership.mem (extChartAt I x).target z
    ⊢ Eq (nhdsWithin (↑(extChartAt I x) (↑(extChartAt I x).symm z)) (extChartAt I  …
  -/
  exact nhdsWithin_extChartAt_target_eq' ((extChartAt I x).map_target hz)
  /-
    🎉 no goals
  -/


/-- Around the image of the base point, the neighborhoods are the same
within `(extChartAt I x).target` and within `range I`. -/
theorem nhdsWithin_extChartAt_target_eq (x : M) :
    𝓝[(extChartAt I x).target] (extChartAt I x) x = 𝓝[range I] (extChartAt I x) x :=
  nhdsWithin_extChartAt_target_eq' (mem_extChartAt_source x)


/-- Around the image of a point in the source, `(extChartAt I x).target` and `range I`
coincide locally. -/
theorem extChartAt_target_eventuallyEq' {x y : M} (hy : y ∈ (extChartAt I x).source) :
    (extChartAt I x).target =ᶠ[𝓝 (extChartAt I x y)] range I :=
  nhdsWithin_eq_iff_eventuallyEq.1 (nhdsWithin_extChartAt_target_eq' hy)


/-- Around a point in the target, `(extChartAt I x).target` and `range I` coincide locally. -/
theorem extChartAt_target_eventuallyEq_of_mem {x : M} {z : E} (hz : z ∈ (extChartAt I x).target) :
    (extChartAt I x).target =ᶠ[𝓝 z] range I :=
  nhdsWithin_eq_iff_eventuallyEq.1 (nhdsWithin_extChartAt_target_eq_of_mem hz)


/-- Around the image of the base point, `(extChartAt I x).target` and `range I` coincide locally. -/
theorem extChartAt_target_eventuallyEq {x : M} :
    (extChartAt I x).target =ᶠ[𝓝 (extChartAt I x x)] range I :=
  nhdsWithin_eq_iff_eventuallyEq.1 (nhdsWithin_extChartAt_target_eq x)


theorem continuousAt_extChartAt_symm'' {x : M} {y : E} (h : y ∈ (extChartAt I x).target) :
    ContinuousAt (extChartAt I x).symm y :=
  continuousAt_extend_symm' _ h


theorem continuousAt_extChartAt_symm' {x x' : M} (h : x' ∈ (extChartAt I x).source) :
    ContinuousAt (extChartAt I x).symm (extChartAt I x x') :=
  continuousAt_extChartAt_symm'' <| (extChartAt I x).map_source h


theorem continuousAt_extChartAt_symm (x : M) :
    ContinuousAt (extChartAt I x).symm ((extChartAt I x) x) :=
  continuousAt_extChartAt_symm' (mem_extChartAt_source x)


theorem continuousOn_extChartAt_symm (x : M) :
    ContinuousOn (extChartAt I x).symm (extChartAt I x).target :=
  fun _y hy => (continuousAt_extChartAt_symm'' hy).continuousWithinAt


lemma extChartAt_target_subset_closure_interior {x : M} :
    (extChartAt I x).target ⊆ closure (interior (extChartAt I x).target) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    inst✝ : ChartedSpace H M
    x : M
    ⊢ HasSubset.Subset (extChartAt I x).target (closure (interior (extChartAt I x) …
  -/
  intro y hy
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    inst✝ : ChartedSpace H M
    x : M
    y : E
    hy : Membership.mem (extChartAt I x).target y
    ⊢ Membership.mem (closure (interior (extChartAt I x).target)) y
  -/
  rw [mem_closure_iff_nhds]
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    inst✝ : ChartedSpace H M
    x : M
    y : E
    hy : Membership.mem (extChartAt I x).target y
    ⊢ ∀ (t : Set E), Membership.mem (nhds y) t → (Inter.inter t (interior (extChar …
  -/
  intro t ht
  have A : t ∩ ((extChartAt I x).target ∪ (range I)ᶜ) ∈ 𝓝 y :=
    inter_mem ht (extChartAt_target_union_compl_range_mem_nhds_of_mem hy)
  have B : y ∈ closure (interior (range I)) := by
    apply I.range_subset_closure_interior (extChartAt_target_subset_range x hy)
  obtain ⟨z, ⟨tz, h'z⟩, hz⟩ :
      (t ∩ ((extChartAt I x).target ∪ (range ↑I)ᶜ) ∩ interior (range I)).Nonempty :=
    mem_closure_iff_nhds.1 B _ A
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    inst✝ : ChartedSpace H M
    x : M
    y : E
    hy : Membership.mem (extChartAt I x).target y
    t : Set E
    ht : Membership.mem (nhds y) t
    A : Membership.mem (nhds y) (Inter.inter t (Union.union (extChartAt I x).targe …
    B : Membership.mem (closure (interior (Set.range ↑I))) y
    z : E
    hz : Membership.mem (interior (Set.range ↑I)) z
    tz : Membership.mem t z
    h'z : Membership.mem (Union.union (extChartAt I x).target (HasCompl.compl (Set …
    ⊢ (Inter.inter t (interior (extChartAt I x).target)).Nonempty
  -/
  refine ⟨z, ⟨tz, ?_⟩⟩
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    inst✝ : ChartedSpace H M
    x : M
    y : E
    hy : Membership.mem (extChartAt I x).target y
    t : Set E
    ht : Membership.mem (nhds y) t
    A : Membership.mem (nhds y) (Inter.inter t (Union.union (extChartAt I x).targe …
    B : Membership.mem (closure (interior (Set.range ↑I))) y
    z : E
    hz : Membership.mem (interior (Set.range ↑I)) z
    tz : Membership.mem t z
    h'z : Membership.mem (Union.union (extChartAt I x).target (HasCompl.compl (Set …
    ⊢ Membership.mem (interior (extChartAt I x).target) z
  -/
  have h''z : z ∈ (extChartAt I x).target := by simpa [interior_subset hz] using h'z
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    inst✝ : ChartedSpace H M
    x : M
    y : E
    hy : Membership.mem (extChartAt I x).target y
    t : Set E
    ht : Membership.mem (nhds y) t
    A : Membership.mem (nhds y) (Inter.inter t (Union.union (extChartAt I x).targe …
    B : Membership.mem (closure (interior (Set.range ↑I))) y
    z : E
    hz : Membership.mem (interior (Set.range ↑I)) z
    tz : Membership.mem t z
    h'z : Membership.mem (Union.union (extChartAt I x).target (HasCompl.compl (Set …
    h''z : Membership.mem (extChartAt I x).target z
    ⊢ Membership.mem (interior (extChartAt I x).target) z
  -/
  exact (extChartAt_target_eventuallyEq_of_mem h''z).symm.mem_interior hz
  /-
    🎉 no goals
  -/


variable (I) in
theorem interior_extChartAt_target_nonempty (x : M) :
    (interior (extChartAt I x).target).Nonempty := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    inst✝ : ChartedSpace H M
    x : M
    ⊢ (interior (extChartAt I x).target).Nonempty
  -/
  by_contra! H
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H✝ : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H✝
    inst✝¹ : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H✝
    inst✝ : ChartedSpace H✝ M
    x : M
    H : Eq (interior (extChartAt I x).target) EmptyCollection.emptyCollection
    ⊢ False
  -/
  have := extChartAt_target_subset_closure_interior (mem_extChartAt_target (I := I) x)
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H✝ : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H✝
    inst✝¹ : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H✝
    inst✝ : ChartedSpace H✝ M
    x : M
    H : Eq (interior (extChartAt I x).target) EmptyCollection.emptyCollection
    this : Membership.mem (closure (interior (extChartAt I x).target)) (↑(extChart …
    ⊢ False
  -/
  simp only [H, closure_empty, mem_empty_iff_false] at this
  /-
    🎉 no goals
  -/


lemma extChartAt_mem_closure_interior {x₀ x : M}
    (hx : x ∈ closure (interior s)) (h'x : x ∈ (extChartAt I x₀).source) :
    extChartAt I x₀ x ∈
      closure (interior ((extChartAt I x₀).symm ⁻¹' s ∩ (extChartAt I x₀).target)) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    s : Set M
    inst✝ : ChartedSpace H M
    x₀ x : M
    hx : Membership.mem (closure (interior s)) x
    h'x : Membership.mem (extChartAt I x₀).source x
    ⊢ Membership.mem (closure (interior (Inter.inter (Set.preimage (↑(extChartAt I …
  -/
  simp_rw [mem_closure_iff, interior_inter, ← inter_assoc]
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    s : Set M
    inst✝ : ChartedSpace H M
    x₀ x : M
    hx : Membership.mem (closure (interior s)) x
    h'x : Membership.mem (extChartAt I x₀).source x
    ⊢ ∀ (o : Set E), IsOpen o → Membership.mem o (↑(extChartAt I x₀) x) → (Inter.i …
  -/
  intro o o_open ho
  obtain ⟨y, ⟨yo, hy⟩, ys⟩ :
      ((extChartAt I x₀) ⁻¹' o ∩ (extChartAt I x₀).source ∩ interior s).Nonempty := by
    have : (extChartAt I x₀) ⁻¹' o ∈ 𝓝 x := by
      apply (continuousAt_extChartAt' h'x).preimage_mem_nhds (o_open.mem_nhds ho)
    refine (mem_closure_iff_nhds.1 hx) _ (inter_mem this ?_)
    apply (isOpen_extChartAt_source x₀).mem_nhds h'x
  have A : interior (↑(extChartAt I x₀).symm ⁻¹' s) ∈ 𝓝 (extChartAt I x₀ y) := by
    simp only [interior_mem_nhds]
    apply (continuousAt_extChartAt_symm' hy).preimage_mem_nhds
    simp only [hy, PartialEquiv.left_inv]
    exact mem_interior_iff_mem_nhds.mp ys
  have B : (extChartAt I x₀) y ∈ closure (interior (extChartAt I x₀).target) := by
    apply extChartAt_target_subset_closure_interior (x := x₀)
    exact (extChartAt I x₀).map_source hy
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    s : Set M
    inst✝ : ChartedSpace H M
    x₀ x : M
    hx : Membership.mem (closure (interior s)) x
    h'x : Membership.mem (extChartAt I x₀).source x
    o : Set E
    o_open : IsOpen o
    ho : Membership.mem o (↑(extChartAt I x₀) x)
    y : M
    ys : Membership.mem (interior s) y
    yo : Membership.mem (Set.preimage (↑(extChartAt I x₀)) o) y
    hy : Membership.mem (extChartAt I x₀).source y
    A : Membership.mem (nhds (↑(extChartAt I x₀) y)) (interior (Set.preimage (↑(ex …
    B : Membership.mem (closure (interior (extChartAt I x₀).target)) (↑(extChartAt …
    ⊢ (Inter.inter (Inter.inter o (interior (Set.preimage (↑(extChartAt I x₀).symm …
  -/
  exact mem_closure_iff_nhds.1 B _ (inter_mem (o_open.mem_nhds yo) A)
  /-
    🎉 no goals
  -/


theorem isOpen_extChartAt_preimage' (x : M) {s : Set E} (hs : IsOpen s) :
    IsOpen ((extChartAt I x).source ∩ extChartAt I x ⁻¹' s) :=
  isOpen_extend_preimage' _ hs


theorem isOpen_extChartAt_preimage (x : M) {s : Set E} (hs : IsOpen s) :
    IsOpen ((chartAt H x).source ∩ extChartAt I x ⁻¹' s) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    inst✝ : ChartedSpace H M
    x : M
    s : Set E
    hs : IsOpen s
    ⊢ IsOpen (Inter.inter (chartAt H x).source (Set.preimage (↑(extChartAt I x)) s))
  -/
  rw [← extChartAt_source I]
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    inst✝ : ChartedSpace H M
    x : M
    s : Set E
    hs : IsOpen s
    ⊢ IsOpen (Inter.inter (extChartAt I x).source (Set.preimage (↑(extChartAt I x) …
  -/
  exact isOpen_extChartAt_preimage' x hs
  /-
    🎉 no goals
  -/


theorem map_extChartAt_nhdsWithin_eq_image' {x y : M} (hy : y ∈ (extChartAt I x).source) :
    map (extChartAt I x) (𝓝[s] y) =
      𝓝[extChartAt I x '' ((extChartAt I x).source ∩ s)] extChartAt I x y :=
                                         /-
                                           𝕜 : Type u_1
                                           E : Type u_2
                                           M : Type u_3
                                           H : Type u_4
                                           inst✝⁵ : NontriviallyNormedField 𝕜
                                           inst✝⁴ : NormedAddCommGroup E
                                           inst✝³ : NormedSpace 𝕜 E
                                           inst✝² : TopologicalSpace H
                                           inst✝¹ : TopologicalSpace M
                                           I : ModelWithCorners 𝕜 E H
                                           s : Set M
                                           inst✝ : ChartedSpace H M
                                           x y : M
                                           hy : Membership.mem (extChartAt I x).source y
                                           ⊢ Membership.mem (chartAt H x).source y
                                         -/
  map_extend_nhdsWithin_eq_image _ <| by rwa [← extChartAt_source I]
                                         /-
                                           🎉 no goals
                                         -/


theorem map_extChartAt_nhdsWithin_eq_image (x : M) :
    map (extChartAt I x) (𝓝[s] x) =
      𝓝[extChartAt I x '' ((extChartAt I x).source ∩ s)] extChartAt I x x :=
  map_extChartAt_nhdsWithin_eq_image' (mem_extChartAt_source x)


theorem map_extChartAt_nhdsWithin' {x y : M} (hy : y ∈ (extChartAt I x).source) :
    map (extChartAt I x) (𝓝[s] y) = 𝓝[(extChartAt I x).symm ⁻¹' s ∩ range I] extChartAt I x y :=
                                /-
                                  𝕜 : Type u_1
                                  E : Type u_2
                                  M : Type u_3
                                  H : Type u_4
                                  inst✝⁵ : NontriviallyNormedField 𝕜
                                  inst✝⁴ : NormedAddCommGroup E
                                  inst✝³ : NormedSpace 𝕜 E
                                  inst✝² : TopologicalSpace H
                                  inst✝¹ : TopologicalSpace M
                                  I : ModelWithCorners 𝕜 E H
                                  s : Set M
                                  inst✝ : ChartedSpace H M
                                  x y : M
                                  hy : Membership.mem (extChartAt I x).source y
                                  ⊢ Membership.mem (chartAt H x).source y
                                -/
  map_extend_nhdsWithin _ <| by rwa [← extChartAt_source I]
                                /-
                                  🎉 no goals
                                -/


theorem map_extChartAt_nhdsWithin (x : M) :
    map (extChartAt I x) (𝓝[s] x) = 𝓝[(extChartAt I x).symm ⁻¹' s ∩ range I] extChartAt I x x :=
  map_extChartAt_nhdsWithin' (mem_extChartAt_source x)


theorem map_extChartAt_symm_nhdsWithin' {x y : M} (hy : y ∈ (extChartAt I x).source) :
    map (extChartAt I x).symm (𝓝[(extChartAt I x).symm ⁻¹' s ∩ range I] extChartAt I x y) =
      𝓝[s] y :=
                                     /-
                                       𝕜 : Type u_1
                                       E : Type u_2
                                       M : Type u_3
                                       H : Type u_4
                                       inst✝⁵ : NontriviallyNormedField 𝕜
                                       inst✝⁴ : NormedAddCommGroup E
                                       inst✝³ : NormedSpace 𝕜 E
                                       inst✝² : TopologicalSpace H
                                       inst✝¹ : TopologicalSpace M
                                       I : ModelWithCorners 𝕜 E H
                                       s : Set M
                                       inst✝ : ChartedSpace H M
                                       x y : M
                                       hy : Membership.mem (extChartAt I x).source y
                                       ⊢ Membership.mem (chartAt H x).source y
                                     -/
  map_extend_symm_nhdsWithin _ <| by rwa [← extChartAt_source I]
                                     /-
                                       🎉 no goals
                                     -/


theorem map_extChartAt_symm_nhdsWithin_range' {x y : M} (hy : y ∈ (extChartAt I x).source) :
    map (extChartAt I x).symm (𝓝[range I] extChartAt I x y) = 𝓝 y :=
                                           /-
                                             𝕜 : Type u_1
                                             E : Type u_2
                                             M : Type u_3
                                             H : Type u_4
                                             inst✝⁵ : NontriviallyNormedField 𝕜
                                             inst✝⁴ : NormedAddCommGroup E
                                             inst✝³ : NormedSpace 𝕜 E
                                             inst✝² : TopologicalSpace H
                                             inst✝¹ : TopologicalSpace M
                                             I : ModelWithCorners 𝕜 E H
                                             inst✝ : ChartedSpace H M
                                             x y : M
                                             hy : Membership.mem (extChartAt I x).source y
                                             ⊢ Membership.mem (chartAt H x).source y
                                           -/
  map_extend_symm_nhdsWithin_range _ <| by rwa [← extChartAt_source I]
                                           /-
                                             🎉 no goals
                                           -/


theorem map_extChartAt_symm_nhdsWithin (x : M) :
    map (extChartAt I x).symm (𝓝[(extChartAt I x).symm ⁻¹' s ∩ range I] extChartAt I x x) =
      𝓝[s] x :=
  map_extChartAt_symm_nhdsWithin' (mem_extChartAt_source x)


theorem map_extChartAt_symm_nhdsWithin_range (x : M) :
    map (extChartAt I x).symm (𝓝[range I] extChartAt I x x) = 𝓝 x :=
  map_extChartAt_symm_nhdsWithin_range' (mem_extChartAt_source x)


/-- Technical lemma ensuring that the preimage under an extended chart of a neighborhood of a point
in the source is a neighborhood of the preimage, within a set. -/
theorem extChartAt_preimage_mem_nhdsWithin' {x x' : M} (h : x' ∈ (extChartAt I x).source)
    (ht : t ∈ 𝓝[s] x') :
    (extChartAt I x).symm ⁻¹' t ∈ 𝓝[(extChartAt I x).symm ⁻¹' s ∩ range I] (extChartAt I x) x' := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    s t : Set M
    inst✝ : ChartedSpace H M
    x x' : M
    h : Membership.mem (extChartAt I x).source x'
    ht : Membership.mem (nhdsWithin x' s) t
    ⊢ Membership.mem (nhdsWithin (↑(extChartAt I x) x') (Inter.inter (Set.preimage …
  -/
  rwa [← map_extChartAt_symm_nhdsWithin' h, mem_map] at ht
  /-
    🎉 no goals
  -/


/-- Technical lemma ensuring that the preimage under an extended chart of a neighborhood of the
base point is a neighborhood of the preimage, within a set. -/
theorem extChartAt_preimage_mem_nhdsWithin {x : M} (ht : t ∈ 𝓝[s] x) :
    (extChartAt I x).symm ⁻¹' t ∈ 𝓝[(extChartAt I x).symm ⁻¹' s ∩ range I] (extChartAt I x) x :=
  extChartAt_preimage_mem_nhdsWithin' (mem_extChartAt_source x) ht


theorem extChartAt_preimage_mem_nhds' {x x' : M} (h : x' ∈ (extChartAt I x).source)
    (ht : t ∈ 𝓝 x') : (extChartAt I x).symm ⁻¹' t ∈ 𝓝 (extChartAt I x x') :=
                                 /-
                                   𝕜 : Type u_1
                                   E : Type u_2
                                   M : Type u_3
                                   H : Type u_4
                                   inst✝⁵ : NontriviallyNormedField 𝕜
                                   inst✝⁴ : NormedAddCommGroup E
                                   inst✝³ : NormedSpace 𝕜 E
                                   inst✝² : TopologicalSpace H
                                   inst✝¹ : TopologicalSpace M
                                   I : ModelWithCorners 𝕜 E H
                                   t : Set M
                                   inst✝ : ChartedSpace H M
                                   x x' : M
                                   h : Membership.mem (extChartAt I x).source x'
                                   ht : Membership.mem (nhds x') t
                                   ⊢ Membership.mem (chartAt H x).source x'
                                 -/
  extend_preimage_mem_nhds _ (by rwa [← extChartAt_source I]) ht
                                 /-
                                   🎉 no goals
                                 -/


/-- Technical lemma ensuring that the preimage under an extended chart of a neighborhood of a point
is a neighborhood of the preimage. -/
theorem extChartAt_preimage_mem_nhds {x : M} (ht : t ∈ 𝓝 x) :
    (extChartAt I x).symm ⁻¹' t ∈ 𝓝 ((extChartAt I x) x) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    t : Set M
    inst✝ : ChartedSpace H M
    x : M
    ht : Membership.mem (nhds x) t
    ⊢ Membership.mem (nhds (↑(extChartAt I x) x)) (Set.preimage (↑(extChartAt I x) …
  -/
  apply (continuousAt_extChartAt_symm x).preimage_mem_nhds
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    t : Set M
    inst✝ : ChartedSpace H M
    x : M
    ht : Membership.mem (nhds x) t
    ⊢ Membership.mem (nhds (↑(extChartAt I x).symm (↑(extChartAt I x) x))) t
  -/
  rwa [(extChartAt I x).left_inv (mem_extChartAt_source _)]
  /-
    🎉 no goals
  -/


/-- Technical lemma to rewrite suitably the preimage of an intersection under an extended chart, to
bring it into a convenient form to apply derivative lemmas. -/
theorem extChartAt_preimage_inter_eq (x : M) :
    (extChartAt I x).symm ⁻¹' (s ∩ t) ∩ range I =
      (extChartAt I x).symm ⁻¹' s ∩ range I ∩ (extChartAt I x).symm ⁻¹' t := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    s t : Set M
    inst✝ : ChartedSpace H M
    x : M
    ⊢ Eq (Inter.inter (Set.preimage (↑(extChartAt I x).symm) (Inter.inter s t)) (S …
  -/
  mfld_set_tac
  /-
    🎉 no goals
  -/


theorem ContinuousWithinAt.nhdsWithin_extChartAt_symm_preimage_inter_range
    {f : M → M'} {x : M} (hc : ContinuousWithinAt f s x) :
    𝓝[(extChartAt I x).symm ⁻¹' s ∩ range I] (extChartAt I x x) =
      𝓝[(extChartAt I x).target ∩
        (extChartAt I x).symm ⁻¹' (s ∩ f ⁻¹' (extChartAt I' (f x)).source)] (extChartAt I x x) := by
  rw [← (extChartAt I x).image_source_inter_eq', ← map_extChartAt_nhdsWithin_eq_image,
    ← map_extChartAt_nhdsWithin, nhdsWithin_inter_of_mem']
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    E' : Type u_5
    M' : Type u_6
    H' : Type u_7
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    inst✝⁷ : TopologicalSpace H
    inst✝⁶ : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    inst✝⁵ : NormedAddCommGroup E'
    inst✝⁴ : NormedSpace 𝕜 E'
    inst✝³ : TopologicalSpace H'
    inst✝² : TopologicalSpace M'
    I' : ModelWithCorners 𝕜 E' H'
    s : Set M
    inst✝¹ : ChartedSpace H M
    inst✝ : ChartedSpace H' M'
    f : M → M'
    x : M
    hc : ContinuousWithinAt f s x
    ⊢ Membership.mem (nhdsWithin x s) (Set.preimage f (extChartAt I' (f x)).source)
  -/
  exact hc (extChartAt_source_mem_nhds _)
  /-
    🎉 no goals
  -/


theorem ContinuousWithinAt.extChartAt_symm_preimage_inter_range_eventuallyEq
    {f : M → M'} {x : M} (hc : ContinuousWithinAt f s x) :
    ((extChartAt I x).symm ⁻¹' s ∩ range I : Set E) =ᶠ[𝓝 (extChartAt I x x)]
      ((extChartAt I x).target ∩
        (extChartAt I x).symm ⁻¹' (s ∩ f ⁻¹' (extChartAt I' (f x)).source) : Set E) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    E' : Type u_5
    M' : Type u_6
    H' : Type u_7
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    inst✝⁷ : TopologicalSpace H
    inst✝⁶ : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    inst✝⁵ : NormedAddCommGroup E'
    inst✝⁴ : NormedSpace 𝕜 E'
    inst✝³ : TopologicalSpace H'
    inst✝² : TopologicalSpace M'
    I' : ModelWithCorners 𝕜 E' H'
    s : Set M
    inst✝¹ : ChartedSpace H M
    inst✝ : ChartedSpace H' M'
    f : M → M'
    x : M
    hc : ContinuousWithinAt f s x
    ⊢ (nhds (↑(extChartAt I x) x)).EventuallyEq (Inter.inter (Set.preimage (↑(extC …
  -/
  rw [← nhdsWithin_eq_iff_eventuallyEq]
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    E' : Type u_5
    M' : Type u_6
    H' : Type u_7
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    inst✝⁷ : TopologicalSpace H
    inst✝⁶ : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    inst✝⁵ : NormedAddCommGroup E'
    inst✝⁴ : NormedSpace 𝕜 E'
    inst✝³ : TopologicalSpace H'
    inst✝² : TopologicalSpace M'
    I' : ModelWithCorners 𝕜 E' H'
    s : Set M
    inst✝¹ : ChartedSpace H M
    inst✝ : ChartedSpace H' M'
    f : M → M'
    x : M
    hc : ContinuousWithinAt f s x
    ⊢ Eq (nhdsWithin (↑(extChartAt I x) x) (Inter.inter (Set.preimage (↑(extChartA …
  -/
  exact hc.nhdsWithin_extChartAt_symm_preimage_inter_range
  /-
    🎉 no goals
  -/


theorem ext_coord_change_source (x x' : M) :
    ((extChartAt I x').symm ≫ extChartAt I x).source =
      I '' ((chartAt H x').symm ≫ₕ chartAt H x).source :=
  extend_coord_change_source _ _


theorem contDiffOn_ext_coord_change [SmoothManifoldWithCorners I M] (x x' : M) :
    ContDiffOn 𝕜 ∞ (extChartAt I x ∘ (extChartAt I x').symm)
      ((extChartAt I x').symm ≫ extChartAt I x).source :=
  contDiffOn_extend_coord_change (chart_mem_maximalAtlas x) (chart_mem_maximalAtlas x')


theorem contDiffWithinAt_ext_coord_change [SmoothManifoldWithCorners I M] (x x' : M) {y : E}
    (hy : y ∈ ((extChartAt I x').symm ≫ extChartAt I x).source) :
    ContDiffWithinAt 𝕜 ∞ (extChartAt I x ∘ (extChartAt I x').symm) (range I) y :=
  contDiffWithinAt_extend_coord_change (chart_mem_maximalAtlas x) (chart_mem_maximalAtlas x') hy


variable (I I') in
/-- Conjugating a function to write it in the preferred charts around `x`.
The manifold derivative of `f` will just be the derivative of this conjugated function. -/
@[simp, mfld_simps]
def writtenInExtChartAt (x : M) (f : M → M') : E → E' :=
  extChartAt I' (f x) ∘ f ∘ (extChartAt I x).symm


theorem writtenInExtChartAt_chartAt {x : M} {y : E} (h : y ∈ (extChartAt I x).target) :
                                                        /-
                                                          𝕜 : Type u_1
                                                          E : Type u_2
                                                          M : Type u_3
                                                          H : Type u_4
                                                          inst✝⁵ : NontriviallyNormedField 𝕜
                                                          inst✝⁴ : NormedAddCommGroup E
                                                          inst✝³ : NormedSpace 𝕜 E
                                                          inst✝² : TopologicalSpace H
                                                          inst✝¹ : TopologicalSpace M
                                                          I : ModelWithCorners 𝕜 E H
                                                          inst✝ : ChartedSpace H M
                                                          x : M
                                                          y : E
                                                          h : Membership.mem (extChartAt I x).target y
                                                          ⊢ Eq (writtenInExtChartAt I I x (↑(chartAt H x)) y) y
                                                        -/
    writtenInExtChartAt I I x (chartAt H x) y = y := by simp_all only [mfld_simps]
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem writtenInExtChartAt_chartAt_symm {x : M} {y : E} (h : y ∈ (extChartAt I x).target) :
    writtenInExtChartAt I I (chartAt H x x) (chartAt H x).symm y = y := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    inst✝ : ChartedSpace H M
    x : M
    y : E
    h : Membership.mem (extChartAt I x).target y
    ⊢ Eq (writtenInExtChartAt I I (↑(chartAt H x) x) (↑(chartAt H x).symm) y) y
  -/
  simp_all only [mfld_simps]
  /-
    🎉 no goals
  -/


theorem writtenInExtChartAt_extChartAt {x : M} {y : E} (h : y ∈ (extChartAt I x).target) :
    writtenInExtChartAt I 𝓘(𝕜, E) x (extChartAt I x) y = y := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    inst✝ : ChartedSpace H M
    x : M
    y : E
    h : Membership.mem (extChartAt I x).target y
    ⊢ Eq (writtenInExtChartAt I (modelWithCornersSelf 𝕜 E) x (↑(extChartAt I x)) y …
  -/
  simp_all only [mfld_simps]
  /-
    🎉 no goals
  -/


theorem writtenInExtChartAt_extChartAt_symm {x : M} {y : E} (h : y ∈ (extChartAt I x).target) :
    writtenInExtChartAt 𝓘(𝕜, E) I (extChartAt I x x) (extChartAt I x).symm y = y := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : TopologicalSpace H
    inst✝¹ : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    inst✝ : ChartedSpace H M
    x : M
    y : E
    h : Membership.mem (extChartAt I x).target y
    ⊢ Eq (writtenInExtChartAt (modelWithCornersSelf 𝕜 E) I (↑(extChartAt I x) x) ( …
  -/
  simp_all only [mfld_simps]
  /-
    🎉 no goals
  -/


theorem extChartAt_self_eq {x : H} : ⇑(extChartAt I x) = I :=
  rfl


theorem extChartAt_self_apply {x y : H} : extChartAt I x y = I y :=
  rfl


/-- In the case of the manifold structure on a vector space, the extended charts are just the
identity. -/
theorem extChartAt_model_space_eq_id (x : E) : extChartAt 𝓘(𝕜, E) x = PartialEquiv.refl E := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    ⊢ Eq (extChartAt (modelWithCornersSelf 𝕜 E) x) (PartialEquiv.refl E)
  -/
  simp only [mfld_simps]
  /-
    🎉 no goals
  -/


theorem ext_chart_model_space_apply {x y : E} : extChartAt 𝓘(𝕜, E) x y = y :=
  rfl


theorem extChartAt_prod (x : M × M') :
    extChartAt (I.prod I') x = (extChartAt I x.1).prod (extChartAt I' x.2) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    E' : Type u_5
    M' : Type u_6
    H' : Type u_7
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    inst✝⁷ : TopologicalSpace H
    inst✝⁶ : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    inst✝⁵ : NormedAddCommGroup E'
    inst✝⁴ : NormedSpace 𝕜 E'
    inst✝³ : TopologicalSpace H'
    inst✝² : TopologicalSpace M'
    I' : ModelWithCorners 𝕜 E' H'
    inst✝¹ : ChartedSpace H M
    inst✝ : ChartedSpace H' M'
    x : Prod M M'
    ⊢ Eq (extChartAt (I.prod I') x) ((extChartAt I x.1).prod (extChartAt I' x.2))
  -/
  simp only [mfld_simps]
  -- Porting note: `simp` can't use `PartialEquiv.prod_trans` here because of a type
  -- synonym
  /-
    𝕜 : Type u_1
    E : Type u_2
    M : Type u_3
    H : Type u_4
    E' : Type u_5
    M' : Type u_6
    H' : Type u_7
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    inst✝⁷ : TopologicalSpace H
    inst✝⁶ : TopologicalSpace M
    I : ModelWithCorners 𝕜 E H
    inst✝⁵ : NormedAddCommGroup E'
    inst✝⁴ : NormedSpace 𝕜 E'
    inst✝³ : TopologicalSpace H'
    inst✝² : TopologicalSpace M'
    I' : ModelWithCorners 𝕜 E' H'
    inst✝¹ : ChartedSpace H M
    inst✝ : ChartedSpace H' M'
    x : Prod M M'
    ⊢ Eq (((chartAt H x.1).prod (chartAt H' x.2).toPartialEquiv).trans (I.prod I'. …
  -/
  rw [PartialEquiv.prod_trans]
  /-
    🎉 no goals
  -/


theorem extChartAt_comp [ChartedSpace H H'] (x : M') :
    (letI := ChartedSpace.comp H H' M'; extChartAt I x) =
      (chartAt H' x).toPartialEquiv ≫ extChartAt I (chartAt H' x x) :=
  PartialEquiv.trans_assoc ..


theorem writtenInExtChartAt_chartAt_comp [ChartedSpace H H'] (x : M') {y}
    (hy : y ∈ letI := ChartedSpace.comp H H' M'; (extChartAt I x).target) :
    (letI := ChartedSpace.comp H H' M'; writtenInExtChartAt I I x (chartAt H' x) y) = y := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    H : Type u_4
    M' : Type u_6
    H' : Type u_7
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    inst✝³ : TopologicalSpace H'
    inst✝² : TopologicalSpace M'
    inst✝¹ : ChartedSpace H' M'
    inst✝ : ChartedSpace H H'
    x : M'
    y : E
    hy : Membership.mem (extChartAt I x).target y
    ⊢ Eq (writtenInExtChartAt I I x (↑(chartAt H' x)) y) y
  -/
  letI := ChartedSpace.comp H H' M'
  /-
    𝕜 : Type u_1
    E : Type u_2
    H : Type u_4
    M' : Type u_6
    H' : Type u_7
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    inst✝³ : TopologicalSpace H'
    inst✝² : TopologicalSpace M'
    inst✝¹ : ChartedSpace H' M'
    inst✝ : ChartedSpace H H'
    x : M'
    y : E
    hy : Membership.mem (extChartAt I x).target y
    this : ChartedSpace H M' := ChartedSpace.comp H H' M'
    ⊢ Eq (writtenInExtChartAt I I x (↑(chartAt H' x)) y) y
  -/
  simp_all only [mfld_simps, chartAt_comp]
  /-
    🎉 no goals
  -/


theorem writtenInExtChartAt_chartAt_symm_comp [ChartedSpace H H'] (x : M') {y}
    (hy : y ∈ letI := ChartedSpace.comp H H' M'; (extChartAt I x).target) :
    ( letI := ChartedSpace.comp H H' M'
      writtenInExtChartAt I I (chartAt H' x x) (chartAt H' x).symm y) = y := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    H : Type u_4
    M' : Type u_6
    H' : Type u_7
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    inst✝³ : TopologicalSpace H'
    inst✝² : TopologicalSpace M'
    inst✝¹ : ChartedSpace H' M'
    inst✝ : ChartedSpace H H'
    x : M'
    y : E
    hy : Membership.mem (extChartAt I x).target y
    ⊢ Eq (writtenInExtChartAt I I (↑(chartAt H' x) x) (↑(chartAt H' x).symm) y) y
  -/
  letI := ChartedSpace.comp H H' M'
  /-
    𝕜 : Type u_1
    E : Type u_2
    H : Type u_4
    M' : Type u_6
    H' : Type u_7
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    inst✝³ : TopologicalSpace H'
    inst✝² : TopologicalSpace M'
    inst✝¹ : ChartedSpace H' M'
    inst✝ : ChartedSpace H H'
    x : M'
    y : E
    hy : Membership.mem (extChartAt I x).target y
    this : ChartedSpace H M' := ChartedSpace.comp H H' M'
    ⊢ Eq (writtenInExtChartAt I I (↑(chartAt H' x) x) (↑(chartAt H' x).symm) y) y
  -/
  simp_all only [mfld_simps, chartAt_comp]
  /-
    🎉 no goals
  -/


/-- A finite-dimensional manifold modelled on a locally compact field
  (such as ℝ, ℂ or the `p`-adic numbers) is locally compact. -/
lemma Manifold.locallyCompact_of_finiteDimensional
    (I : ModelWithCorners 𝕜 E H) [LocallyCompactSpace 𝕜] [FiniteDimensional 𝕜 E] :
    LocallyCompactSpace M := by
  /-
    E : Type u_1
    𝕜 : Type u_2
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁴ : TopologicalSpace H
    M : Type u_4
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    I : ModelWithCorners 𝕜 E H
    inst✝¹ : LocallyCompactSpace 𝕜
    inst✝ : FiniteDimensional 𝕜 E
    ⊢ LocallyCompactSpace M
  -/
  have : ProperSpace E := FiniteDimensional.proper 𝕜 E
  /-
    E : Type u_1
    𝕜 : Type u_2
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁴ : TopologicalSpace H
    M : Type u_4
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    I : ModelWithCorners 𝕜 E H
    inst✝¹ : LocallyCompactSpace 𝕜
    inst✝ : FiniteDimensional 𝕜 E
    this : ProperSpace E
    ⊢ LocallyCompactSpace M
  -/
  have : LocallyCompactSpace H := I.locallyCompactSpace
  /-
    E : Type u_1
    𝕜 : Type u_2
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁴ : TopologicalSpace H
    M : Type u_4
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    I : ModelWithCorners 𝕜 E H
    inst✝¹ : LocallyCompactSpace 𝕜
    inst✝ : FiniteDimensional 𝕜 E
    this✝ : ProperSpace E
    this : LocallyCompactSpace H
    ⊢ LocallyCompactSpace M
  -/
  exact ChartedSpace.locallyCompactSpace H M
  /-
    🎉 no goals
  -/


/-- A locally compact manifold must be modelled on a locally compact space. -/
lemma LocallyCompactSpace.of_locallyCompact_manifold (I : ModelWithCorners 𝕜 E H)
    [h : Nonempty M] [LocallyCompactSpace M] :
    LocallyCompactSpace E := by
  /-
    E : Type u_1
    𝕜 : Type u_2
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝³ : TopologicalSpace H
    M : Type u_4
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    I : ModelWithCorners 𝕜 E H
    h : Nonempty M
    inst✝ : LocallyCompactSpace M
    ⊢ LocallyCompactSpace E
  -/
  rcases h with ⟨x⟩
  /-
    case intro
    E : Type u_1
    𝕜 : Type u_2
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝³ : TopologicalSpace H
    M : Type u_4
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    I : ModelWithCorners 𝕜 E H
    inst✝ : LocallyCompactSpace M
    x : M
    ⊢ LocallyCompactSpace E
  -/
  obtain ⟨y, hy⟩ := interior_extChartAt_target_nonempty I x
  /-
    case intro.intro
    E : Type u_1
    𝕜 : Type u_2
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝³ : TopologicalSpace H
    M : Type u_4
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    I : ModelWithCorners 𝕜 E H
    inst✝ : LocallyCompactSpace M
    x : M
    y : E
    hy : Membership.mem (interior (extChartAt I x).target) y
    ⊢ LocallyCompactSpace E
  -/
  have h'y : y ∈ (extChartAt I x).target := interior_subset hy
  obtain ⟨s, hmem, hss, hcom⟩ :=
    LocallyCompactSpace.local_compact_nhds ((extChartAt I x).symm y) (extChartAt I x).source
      ((isOpen_extChartAt_source x).mem_nhds ((extChartAt I x).map_target h'y))
  have : IsCompact <| (extChartAt I x) '' s :=
    hcom.image_of_continuousOn <| (continuousOn_extChartAt x).mono hss
  /-
    case intro.intro.intro.intro.intro
    E : Type u_1
    𝕜 : Type u_2
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝³ : TopologicalSpace H
    M : Type u_4
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    I : ModelWithCorners 𝕜 E H
    inst✝ : LocallyCompactSpace M
    x : M
    y : E
    hy : Membership.mem (interior (extChartAt I x).target) y
    h'y : Membership.mem (extChartAt I x).target y
    s : Set M
    hmem : Membership.mem (nhds (↑(extChartAt I x).symm y)) s
    hss : HasSubset.Subset s (extChartAt I x).source
    hcom : IsCompact s
    this : IsCompact (Set.image (↑(extChartAt I x)) s)
    ⊢ LocallyCompactSpace E
  -/
  apply this.locallyCompactSpace_of_mem_nhds_of_addGroup (x := y)
  /-
    case intro.intro.intro.intro.intro
    E : Type u_1
    𝕜 : Type u_2
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝³ : TopologicalSpace H
    M : Type u_4
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    I : ModelWithCorners 𝕜 E H
    inst✝ : LocallyCompactSpace M
    x : M
    y : E
    hy : Membership.mem (interior (extChartAt I x).target) y
    h'y : Membership.mem (extChartAt I x).target y
    s : Set M
    hmem : Membership.mem (nhds (↑(extChartAt I x).symm y)) s
    hss : HasSubset.Subset s (extChartAt I x).source
    hcom : IsCompact s
    this : IsCompact (Set.image (↑(extChartAt I x)) s)
    ⊢ Membership.mem (nhds y) (Set.image (↑(extChartAt I x)) s)
  -/
  rw [← (extChartAt I x).right_inv h'y]
  apply extChartAt_image_nhd_mem_nhds_of_mem_interior_range
    (PartialEquiv.map_target (extChartAt I x) h'y) _ hmem
  /-
    E : Type u_1
    𝕜 : Type u_2
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝³ : TopologicalSpace H
    M : Type u_4
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    I : ModelWithCorners 𝕜 E H
    inst✝ : LocallyCompactSpace M
    x : M
    y : E
    hy : Membership.mem (interior (extChartAt I x).target) y
    h'y : Membership.mem (extChartAt I x).target y
    s : Set M
    hmem : Membership.mem (nhds (↑(extChartAt I x).symm y)) s
    hss : HasSubset.Subset s (extChartAt I x).source
    hcom : IsCompact s
    this : IsCompact (Set.image (↑(extChartAt I x)) s)
    ⊢ Membership.mem (interior (Set.range ↑I)) (↑(extChartAt I x) (↑(extChartAt I  …
  -/
  simp only [(extChartAt I x).right_inv h'y]
  /-
    E : Type u_1
    𝕜 : Type u_2
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝³ : TopologicalSpace H
    M : Type u_4
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    I : ModelWithCorners 𝕜 E H
    inst✝ : LocallyCompactSpace M
    x : M
    y : E
    hy : Membership.mem (interior (extChartAt I x).target) y
    h'y : Membership.mem (extChartAt I x).target y
    s : Set M
    hmem : Membership.mem (nhds (↑(extChartAt I x).symm y)) s
    hss : HasSubset.Subset s (extChartAt I x).source
    hcom : IsCompact s
    this : IsCompact (Set.image (↑(extChartAt I x)) s)
    ⊢ Membership.mem (interior (Set.range ↑I)) y
  -/
  exact interior_mono (extChartAt_target_subset_range x) hy
  /-
    🎉 no goals
  -/


/-- Riesz's theorem applied to manifolds: a locally compact manifolds must be modelled on a
  finite-dimensional space. This is the converse to
  `Manifold.locallyCompact_of_finiteDimensional`. -/
theorem FiniteDimensional.of_locallyCompact_manifold
    [CompleteSpace 𝕜] (I : ModelWithCorners 𝕜 E H) [Nonempty M] [LocallyCompactSpace M] :
    FiniteDimensional 𝕜 E := by
  /-
    E : Type u_1
    𝕜 : Type u_2
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁵ : TopologicalSpace H
    M : Type u_4
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : CompleteSpace 𝕜
    I : ModelWithCorners 𝕜 E H
    inst✝¹ : Nonempty M
    inst✝ : LocallyCompactSpace M
    ⊢ FiniteDimensional 𝕜 E
  -/
  have := LocallyCompactSpace.of_locallyCompact_manifold M I
  /-
    E : Type u_1
    𝕜 : Type u_2
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁵ : TopologicalSpace H
    M : Type u_4
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : CompleteSpace 𝕜
    I : ModelWithCorners 𝕜 E H
    inst✝¹ : Nonempty M
    inst✝ : LocallyCompactSpace M
    this : LocallyCompactSpace E
    ⊢ FiniteDimensional 𝕜 E
  -/
  exact FiniteDimensional.of_locallyCompactSpace 𝕜
  /-
    🎉 no goals
  -/


set_option linter.unusedVariables false in
/-- The tangent space at a point of the manifold `M`. It is just `E`. We could use instead
`(tangentBundleCore I M).to_topological_vector_bundle_core.fiber x`, but we use `E` to help the
kernel.
-/
@[nolint unusedArguments]
def TangentSpace {𝕜 : Type*} [NontriviallyNormedField 𝕜]
    {E : Type u} [NormedAddCommGroup E] [NormedSpace 𝕜 E]
    {H : Type*} [TopologicalSpace H] (I : ModelWithCorners 𝕜 E H)
    {M : Type*} [TopologicalSpace M] [ChartedSpace H M] (_x : M) : Type u := E
-- Porting note: was deriving TopologicalSpace, AddCommGroup, TopologicalAddGroup

/- In general, the definition of `TangentSpace` is not reducible, so that type class inference
does not pick wrong instances. We record the right instances for them. -/


instance : TopologicalSpace (TangentSpace I x) := inferInstanceAs (TopologicalSpace E)

instance : AddCommGroup (TangentSpace I x) := inferInstanceAs (AddCommGroup E)

instance : TopologicalAddGroup (TangentSpace I x) := inferInstanceAs (TopologicalAddGroup E)

instance : Module 𝕜 (TangentSpace I x) := inferInstanceAs (Module 𝕜 E)

instance : Inhabited (TangentSpace I x) := ⟨0⟩


variable (M) in
-- is empty if the base manifold is empty
/-- The tangent bundle to a smooth manifold, as a Sigma type. Defined in terms of
`Bundle.TotalSpace` to be able to put a suitable topology on it. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): was nolint has_nonempty_instance
abbrev TangentBundle :=
  Bundle.TotalSpace E (TangentSpace I : M → Type _)


instance : PathConnectedSpace (TangentSpace I x) := inferInstanceAs (PathConnectedSpace E)


