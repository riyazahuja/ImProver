/-- A subset `s` of a topological space is quasi-separated if the intersections of any pairs of
compact open subsets of `s` are still compact.

Note that this is equivalent to `s` being a `QuasiSeparatedSpace` only when `s` is open. -/
def IsQuasiSeparated (s : Set α) : Prop :=
  ∀ U V : Set α, U ⊆ s → IsOpen U → IsCompact U → V ⊆ s → IsOpen V → IsCompact V → IsCompact (U ∩ V)


/-- A topological space is quasi-separated if the intersections of any pairs of compact open
subsets are still compact. -/
@[mk_iff]
class QuasiSeparatedSpace (α : Type*) [TopologicalSpace α] : Prop where
  /-- The intersection of two open compact subsets of a quasi-separated space is compact. -/
  inter_isCompact :
    ∀ U V : Set α, IsOpen U → IsCompact U → IsOpen V → IsCompact V → IsCompact (U ∩ V)


theorem isQuasiSeparated_univ_iff {α : Type*} [TopologicalSpace α] :
    IsQuasiSeparated (Set.univ : Set α) ↔ QuasiSeparatedSpace α := by
  /-
    α : Type u_3
    inst✝ : TopologicalSpace α
    ⊢ Iff (IsQuasiSeparated Set.univ) (QuasiSeparatedSpace α)
  -/
  rw [quasiSeparatedSpace_iff]
  /-
    α : Type u_3
    inst✝ : TopologicalSpace α
    ⊢ Iff (IsQuasiSeparated Set.univ) (∀ (U V : Set α), IsOpen U → IsCompact U → I …
  -/
  simp [IsQuasiSeparated]
  /-
    🎉 no goals
  -/


theorem isQuasiSeparated_univ {α : Type*} [TopologicalSpace α] [QuasiSeparatedSpace α] :
    IsQuasiSeparated (Set.univ : Set α) :=
  isQuasiSeparated_univ_iff.mpr inferInstance


theorem IsQuasiSeparated.image_of_isEmbedding {s : Set α} (H : IsQuasiSeparated s)
    (h : IsEmbedding f) : IsQuasiSeparated (f '' s) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    s : Set α
    H : IsQuasiSeparated s
    h : Topology.IsEmbedding f
    ⊢ IsQuasiSeparated (Set.image f s)
  -/
  intro U V hU hU' hU'' hV hV' hV''
  convert
    (H (f ⁻¹' U) (f ⁻¹' V)
      ?_ (h.continuous.1 _ hU') ?_ ?_ (h.continuous.1 _ hV') ?_).image h.continuous
    /-
      case h.e'_3
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      s : Set α
      H : IsQuasiSeparated s
      h : Topology.IsEmbedding f
      U V : Set β
      hU : HasSubset.Subset U (Set.image f s)
      hU' : IsOpen U
      hU'' : IsCompact U
      hV : HasSubset.Subset V (Set.image f s)
      hV' : IsOpen V
      hV'' : IsCompact V
      ⊢ Eq (Inter.inter U V) (Set.image f (Inter.inter (Set.preimage f U) (Set.preim …
    -/
  · symm
    /-
      case h.e'_3
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      s : Set α
      H : IsQuasiSeparated s
      h : Topology.IsEmbedding f
      U V : Set β
      hU : HasSubset.Subset U (Set.image f s)
      hU' : IsOpen U
      hU'' : IsCompact U
      hV : HasSubset.Subset V (Set.image f s)
      hV' : IsOpen V
      hV'' : IsCompact V
      ⊢ Eq (Set.image f (Inter.inter (Set.preimage f U) (Set.preimage f V))) (Inter. …
    -/
    rw [← Set.preimage_inter, Set.image_preimage_eq_inter_range, Set.inter_eq_left]
    /-
      case h.e'_3
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      s : Set α
      H : IsQuasiSeparated s
      h : Topology.IsEmbedding f
      U V : Set β
      hU : HasSubset.Subset U (Set.image f s)
      hU' : IsOpen U
      hU'' : IsCompact U
      hV : HasSubset.Subset V (Set.image f s)
      hV' : IsOpen V
      hV'' : IsCompact V
      ⊢ HasSubset.Subset (Inter.inter U V) (Set.range f)
    -/
    exact Set.inter_subset_left.trans (hU.trans (Set.image_subset_range _ _))
    /-
      🎉 no goals
    -/
    /-
      case convert_1
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      s : Set α
      H : IsQuasiSeparated s
      h : Topology.IsEmbedding f
      U V : Set β
      hU : HasSubset.Subset U (Set.image f s)
      hU' : IsOpen U
      hU'' : IsCompact U
      hV : HasSubset.Subset V (Set.image f s)
      hV' : IsOpen V
      hV'' : IsCompact V
      ⊢ HasSubset.Subset (Set.preimage f U) s
    -/
  · intro x hx
    /-
      case convert_1
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      s : Set α
      H : IsQuasiSeparated s
      h : Topology.IsEmbedding f
      U V : Set β
      hU : HasSubset.Subset U (Set.image f s)
      hU' : IsOpen U
      hU'' : IsCompact U
      hV : HasSubset.Subset V (Set.image f s)
      hV' : IsOpen V
      hV'' : IsCompact V
      x : α
      hx : Membership.mem (Set.preimage f U) x
      ⊢ Membership.mem s x
    -/
    rw [← h.injective.injOn.mem_image_iff (Set.subset_univ _) trivial]
    /-
      case convert_1
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      s : Set α
      H : IsQuasiSeparated s
      h : Topology.IsEmbedding f
      U V : Set β
      hU : HasSubset.Subset U (Set.image f s)
      hU' : IsOpen U
      hU'' : IsCompact U
      hV : HasSubset.Subset V (Set.image f s)
      hV' : IsOpen V
      hV'' : IsCompact V
      x : α
      hx : Membership.mem (Set.preimage f U) x
      ⊢ Membership.mem (Set.image f s) (f x)
    -/
    exact hU hx
    /-
      🎉 no goals
    -/
    /-
      case convert_2
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      s : Set α
      H : IsQuasiSeparated s
      h : Topology.IsEmbedding f
      U V : Set β
      hU : HasSubset.Subset U (Set.image f s)
      hU' : IsOpen U
      hU'' : IsCompact U
      hV : HasSubset.Subset V (Set.image f s)
      hV' : IsOpen V
      hV'' : IsCompact V
      ⊢ IsCompact (Set.preimage f U)
    -/
  · rw [h.isCompact_iff]
    /-
      case convert_2
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      s : Set α
      H : IsQuasiSeparated s
      h : Topology.IsEmbedding f
      U V : Set β
      hU : HasSubset.Subset U (Set.image f s)
      hU' : IsOpen U
      hU'' : IsCompact U
      hV : HasSubset.Subset V (Set.image f s)
      hV' : IsOpen V
      hV'' : IsCompact V
      ⊢ IsCompact (Set.image f (Set.preimage f U))
    -/
    convert hU''
    /-
      case h.e'_3
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      s : Set α
      H : IsQuasiSeparated s
      h : Topology.IsEmbedding f
      U V : Set β
      hU : HasSubset.Subset U (Set.image f s)
      hU' : IsOpen U
      hU'' : IsCompact U
      hV : HasSubset.Subset V (Set.image f s)
      hV' : IsOpen V
      hV'' : IsCompact V
      ⊢ Eq (Set.image f (Set.preimage f U)) U
    -/
    rw [Set.image_preimage_eq_inter_range, Set.inter_eq_left]
    /-
      case h.e'_3
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      s : Set α
      H : IsQuasiSeparated s
      h : Topology.IsEmbedding f
      U V : Set β
      hU : HasSubset.Subset U (Set.image f s)
      hU' : IsOpen U
      hU'' : IsCompact U
      hV : HasSubset.Subset V (Set.image f s)
      hV' : IsOpen V
      hV'' : IsCompact V
      ⊢ HasSubset.Subset U (Set.range f)
    -/
    exact hU.trans (Set.image_subset_range _ _)
    /-
      🎉 no goals
    -/
    /-
      case convert_3
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      s : Set α
      H : IsQuasiSeparated s
      h : Topology.IsEmbedding f
      U V : Set β
      hU : HasSubset.Subset U (Set.image f s)
      hU' : IsOpen U
      hU'' : IsCompact U
      hV : HasSubset.Subset V (Set.image f s)
      hV' : IsOpen V
      hV'' : IsCompact V
      ⊢ HasSubset.Subset (Set.preimage f V) s
    -/
  · intro x hx
    /-
      case convert_3
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      s : Set α
      H : IsQuasiSeparated s
      h : Topology.IsEmbedding f
      U V : Set β
      hU : HasSubset.Subset U (Set.image f s)
      hU' : IsOpen U
      hU'' : IsCompact U
      hV : HasSubset.Subset V (Set.image f s)
      hV' : IsOpen V
      hV'' : IsCompact V
      x : α
      hx : Membership.mem (Set.preimage f V) x
      ⊢ Membership.mem s x
    -/
    rw [← h.injective.injOn.mem_image_iff (Set.subset_univ _) trivial]
    /-
      case convert_3
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      s : Set α
      H : IsQuasiSeparated s
      h : Topology.IsEmbedding f
      U V : Set β
      hU : HasSubset.Subset U (Set.image f s)
      hU' : IsOpen U
      hU'' : IsCompact U
      hV : HasSubset.Subset V (Set.image f s)
      hV' : IsOpen V
      hV'' : IsCompact V
      x : α
      hx : Membership.mem (Set.preimage f V) x
      ⊢ Membership.mem (Set.image f s) (f x)
    -/
    exact hV hx
    /-
      🎉 no goals
    -/
    /-
      case convert_4
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      s : Set α
      H : IsQuasiSeparated s
      h : Topology.IsEmbedding f
      U V : Set β
      hU : HasSubset.Subset U (Set.image f s)
      hU' : IsOpen U
      hU'' : IsCompact U
      hV : HasSubset.Subset V (Set.image f s)
      hV' : IsOpen V
      hV'' : IsCompact V
      ⊢ IsCompact (Set.preimage f V)
    -/
  · rw [h.isCompact_iff]
    /-
      case convert_4
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      s : Set α
      H : IsQuasiSeparated s
      h : Topology.IsEmbedding f
      U V : Set β
      hU : HasSubset.Subset U (Set.image f s)
      hU' : IsOpen U
      hU'' : IsCompact U
      hV : HasSubset.Subset V (Set.image f s)
      hV' : IsOpen V
      hV'' : IsCompact V
      ⊢ IsCompact (Set.image f (Set.preimage f V))
    -/
    convert hV''
    /-
      case h.e'_3
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      s : Set α
      H : IsQuasiSeparated s
      h : Topology.IsEmbedding f
      U V : Set β
      hU : HasSubset.Subset U (Set.image f s)
      hU' : IsOpen U
      hU'' : IsCompact U
      hV : HasSubset.Subset V (Set.image f s)
      hV' : IsOpen V
      hV'' : IsCompact V
      ⊢ Eq (Set.image f (Set.preimage f V)) V
    -/
    rw [Set.image_preimage_eq_inter_range, Set.inter_eq_left]
    /-
      case h.e'_3
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      s : Set α
      H : IsQuasiSeparated s
      h : Topology.IsEmbedding f
      U V : Set β
      hU : HasSubset.Subset U (Set.image f s)
      hU' : IsOpen U
      hU'' : IsCompact U
      hV : HasSubset.Subset V (Set.image f s)
      hV' : IsOpen V
      hV'' : IsCompact V
      ⊢ HasSubset.Subset V (Set.range f)
    -/
    exact hV.trans (Set.image_subset_range _ _)
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-26")]
alias IsQuasiSeparated.image_of_embedding := IsQuasiSeparated.image_of_isEmbedding


theorem Topology.IsOpenEmbedding.isQuasiSeparated_iff (h : IsOpenEmbedding f) {s : Set α} :
    IsQuasiSeparated s ↔ IsQuasiSeparated (f '' s) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    h : Topology.IsOpenEmbedding f
    s : Set α
    ⊢ Iff (IsQuasiSeparated s) (IsQuasiSeparated (Set.image f s))
  -/
  refine ⟨fun hs => hs.image_of_isEmbedding h.isEmbedding, ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    h : Topology.IsOpenEmbedding f
    s : Set α
    ⊢ IsQuasiSeparated (Set.image f s) → IsQuasiSeparated s
  -/
  intro H U V hU hU' hU'' hV hV' hV''
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    h : Topology.IsOpenEmbedding f
    s : Set α
    H : IsQuasiSeparated (Set.image f s)
    U V : Set α
    hU : HasSubset.Subset U s
    hU' : IsOpen U
    hU'' : IsCompact U
    hV : HasSubset.Subset V s
    hV' : IsOpen V
    hV'' : IsCompact V
    ⊢ IsCompact (Inter.inter U V)
  -/
  rw [h.isEmbedding.isCompact_iff, Set.image_inter h.injective]
  exact
    H (f '' U) (f '' V) (Set.image_subset _ hU) (h.isOpenMap _ hU') (hU''.image h.continuous)
      (Set.image_subset _ hV) (h.isOpenMap _ hV') (hV''.image h.continuous)


@[deprecated (since := "2024-10-18")]
alias OpenEmbedding.isQuasiSeparated_iff := IsOpenEmbedding.isQuasiSeparated_iff


theorem isQuasiSeparated_iff_quasiSeparatedSpace (s : Set α) (hs : IsOpen s) :
    IsQuasiSeparated s ↔ QuasiSeparatedSpace s := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    s : Set α
    hs : IsOpen s
    ⊢ Iff (IsQuasiSeparated s) (QuasiSeparatedSpace ↑s)
  -/
  rw [← isQuasiSeparated_univ_iff]
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    s : Set α
    hs : IsOpen s
    ⊢ Iff (IsQuasiSeparated s) (IsQuasiSeparated Set.univ)
  -/
  convert (hs.isOpenEmbedding_subtypeVal.isQuasiSeparated_iff (s := Set.univ)).symm
  /-
    case h.e'_1.h.e'_3
    α : Type u_1
    inst✝ : TopologicalSpace α
    s : Set α
    hs : IsOpen s
    ⊢ Eq s (Set.image Subtype.val Set.univ)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem IsQuasiSeparated.of_subset {s t : Set α} (ht : IsQuasiSeparated t) (h : s ⊆ t) :
    IsQuasiSeparated s := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    s t : Set α
    ht : IsQuasiSeparated t
    h : HasSubset.Subset s t
    ⊢ IsQuasiSeparated s
  -/
  intro U V hU hU' hU'' hV hV' hV''
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    s t : Set α
    ht : IsQuasiSeparated t
    h : HasSubset.Subset s t
    U V : Set α
    hU : HasSubset.Subset U s
    hU' : IsOpen U
    hU'' : IsCompact U
    hV : HasSubset.Subset V s
    hV' : IsOpen V
    hV'' : IsCompact V
    ⊢ IsCompact (Inter.inter U V)
  -/
  exact ht U V (hU.trans h) hU' hU'' (hV.trans h) hV' hV''
  /-
    🎉 no goals
  -/


instance (priority := 100) T2Space.to_quasiSeparatedSpace [T2Space α] : QuasiSeparatedSpace α :=
  ⟨fun _ _ _ hU' _ hV' => hU'.inter hV'⟩


instance (priority := 100) NoetherianSpace.to_quasiSeparatedSpace [NoetherianSpace α] :
    QuasiSeparatedSpace α :=
  ⟨fun _ _ _ _ _ _ => NoetherianSpace.isCompact _⟩


theorem IsQuasiSeparated.of_quasiSeparatedSpace (s : Set α) [QuasiSeparatedSpace α] :
    IsQuasiSeparated s :=
  isQuasiSeparated_univ.of_subset (Set.subset_univ _)


theorem QuasiSeparatedSpace.of_isOpenEmbedding (h : IsOpenEmbedding f) [QuasiSeparatedSpace β] :
    QuasiSeparatedSpace α :=
  isQuasiSeparated_univ_iff.mp
    (h.isQuasiSeparated_iff.mpr <| IsQuasiSeparated.of_quasiSeparatedSpace _)


@[deprecated (since := "2024-10-18")]
alias QuasiSeparatedSpace.of_openEmbedding := QuasiSeparatedSpace.of_isOpenEmbedding

