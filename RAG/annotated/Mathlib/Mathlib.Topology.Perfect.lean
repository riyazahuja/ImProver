/-- If `x` is an accumulation point of a set `C` and `U` is a neighborhood of `x`,
then `x` is an accumulation point of `U ∩ C`. -/
theorem AccPt.nhds_inter {x : α} {U : Set α} (h_acc : AccPt x (𝓟 C)) (hU : U ∈ 𝓝 x) :
    AccPt x (𝓟 (U ∩ C)) := by
  have : 𝓝[≠] x ≤ 𝓟 U := by
    rw [le_principal_iff]
    exact mem_nhdsWithin_of_mem_nhds hU
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    C : Set α
    x : α
    U : Set α
    h_acc : AccPt x (Filter.principal C)
    hU : Membership.mem (nhds x) U
    this : LE.le (nhdsWithin x (HasCompl.compl (Singleton.singleton x))) (Filter.p …
    ⊢ AccPt x (Filter.principal (Inter.inter U C))
  -/
  rw [AccPt, ← inf_principal, ← inf_assoc, inf_of_le_left this]
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    C : Set α
    x : α
    U : Set α
    h_acc : AccPt x (Filter.principal C)
    hU : Membership.mem (nhds x) U
    this : LE.le (nhdsWithin x (HasCompl.compl (Singleton.singleton x))) (Filter.p …
    ⊢ (Min.min (nhdsWithin x (HasCompl.compl (Singleton.singleton x))) (Filter.pri …
  -/
  exact h_acc
  /-
    🎉 no goals
  -/


/-- A set `C` is preperfect if all of its points are accumulation points of itself.
If `C` is nonempty and `α` is a T1 space, this is equivalent to the closure of `C` being perfect.
See `preperfect_iff_perfect_closure`. -/
def Preperfect (C : Set α) : Prop :=
  ∀ x ∈ C, AccPt x (𝓟 C)


/-- A set `C` is called perfect if it is closed and all of its
points are accumulation points of itself.
Note that we do not require `C` to be nonempty. -/
@[mk_iff perfect_def]
structure Perfect (C : Set α) : Prop where
  closed : IsClosed C
  acc : Preperfect C


theorem preperfect_iff_nhds : Preperfect C ↔ ∀ x ∈ C, ∀ U ∈ 𝓝 x, ∃ y ∈ U ∩ C, y ≠ x := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    C : Set α
    ⊢ Iff (Preperfect C) (∀ (x : α), Membership.mem C x → ∀ (U : Set α), Membershi …
  -/
  simp only [Preperfect, accPt_iff_nhds]
  /-
    🎉 no goals
  -/


/--
A topological space `X` is said to be perfect if its universe is a perfect set.
Equivalently, this means that `𝓝[≠] x ≠ ⊥` for every point `x : X`.
-/
@[mk_iff perfectSpace_def]
class PerfectSpace : Prop where
  univ_preperfect : Preperfect (Set.univ : Set α)


theorem PerfectSpace.univ_perfect [PerfectSpace α] : Perfect (Set.univ : Set α) :=
  ⟨isClosed_univ, PerfectSpace.univ_preperfect⟩


/-- The intersection of a preperfect set and an open set is preperfect. -/
theorem Preperfect.open_inter {U : Set α} (hC : Preperfect C) (hU : IsOpen U) :
    Preperfect (U ∩ C) := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    C U : Set α
    hC : Preperfect C
    hU : IsOpen U
    ⊢ Preperfect (Inter.inter U C)
  -/
  rintro x ⟨xU, xC⟩
  /-
    case intro
    α : Type u_1
    inst✝ : TopologicalSpace α
    C U : Set α
    hC : Preperfect C
    hU : IsOpen U
    x : α
    xU : Membership.mem U x
    xC : Membership.mem C x
    ⊢ AccPt x (Filter.principal (Inter.inter U C))
  -/
  apply (hC _ xC).nhds_inter
  /-
    case intro
    α : Type u_1
    inst✝ : TopologicalSpace α
    C U : Set α
    hC : Preperfect C
    hU : IsOpen U
    x : α
    xU : Membership.mem U x
    xC : Membership.mem C x
    ⊢ Membership.mem (nhds x) U
  -/
  exact hU.mem_nhds xU
  /-
    🎉 no goals
  -/


/-- The closure of a preperfect set is perfect.
For a converse, see `preperfect_iff_perfect_closure`. -/
theorem Preperfect.perfect_closure (hC : Preperfect C) : Perfect (closure C) := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    C : Set α
    hC : Preperfect C
    ⊢ Perfect (closure C)
  -/
  constructor; · exact isClosed_closure
                 /-
                   🎉 no goals
                 -/
  /-
    case acc
    α : Type u_1
    inst✝ : TopologicalSpace α
    C : Set α
    hC : Preperfect C
    ⊢ Preperfect (closure C)
  -/
  intro x hx
  /-
    case acc
    α : Type u_1
    inst✝ : TopologicalSpace α
    C : Set α
    hC : Preperfect C
    x : α
    hx : Membership.mem (closure C) x
    ⊢ AccPt x (Filter.principal (closure C))
  -/
  by_cases h : x ∈ C <;> apply AccPt.mono _ (principal_mono.mpr subset_closure)
    /-
      α : Type u_1
      inst✝ : TopologicalSpace α
      C : Set α
      hC : Preperfect C
      x : α
      hx : Membership.mem (closure C) x
      h : Membership.mem C x
      ⊢ AccPt x (Filter.principal C)
    -/
  · exact hC _ h
    /-
      🎉 no goals
    -/
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    C : Set α
    hC : Preperfect C
    x : α
    hx : Membership.mem (closure C) x
    h : Not (Membership.mem C x)
    ⊢ AccPt x (Filter.principal C)
  -/
  have : {x}ᶜ ∩ C = C := by simp [h]
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    C : Set α
    hC : Preperfect C
    x : α
    hx : Membership.mem (closure C) x
    h : Not (Membership.mem C x)
    this : Eq (Inter.inter (HasCompl.compl (Singleton.singleton x)) C) C
    ⊢ AccPt x (Filter.principal C)
  -/
  rw [AccPt, nhdsWithin, inf_assoc, inf_principal, this]
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    C : Set α
    hC : Preperfect C
    x : α
    hx : Membership.mem (closure C) x
    h : Not (Membership.mem C x)
    this : Eq (Inter.inter (HasCompl.compl (Singleton.singleton x)) C) C
    ⊢ (Min.min (nhds x) (Filter.principal C)).NeBot
  -/
  rw [closure_eq_cluster_pts] at hx
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    C : Set α
    hC : Preperfect C
    x : α
    hx : Membership.mem (setOf fun a => ClusterPt a (Filter.principal C)) x
    h : Not (Membership.mem C x)
    this : Eq (Inter.inter (HasCompl.compl (Singleton.singleton x)) C) C
    ⊢ (Min.min (nhds x) (Filter.principal C)).NeBot
  -/
  exact hx
  /-
    🎉 no goals
  -/


/-- In a T1 space, being preperfect is equivalent to having perfect closure. -/
theorem preperfect_iff_perfect_closure [T1Space α] : Preperfect C ↔ Perfect (closure C) := by
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    C : Set α
    inst✝ : T1Space α
    ⊢ Iff (Preperfect C) (Perfect (closure C))
  -/
  constructor <;> intro h
    /-
      case mp
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      C : Set α
      inst✝ : T1Space α
      h : Preperfect C
      ⊢ Perfect (closure C)
    -/
  · exact h.perfect_closure
    /-
      🎉 no goals
    -/
  /-
    case mpr
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    C : Set α
    inst✝ : T1Space α
    h : Perfect (closure C)
    ⊢ Preperfect C
  -/
  intro x xC
  /-
    case mpr
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    C : Set α
    inst✝ : T1Space α
    h : Perfect (closure C)
    x : α
    xC : Membership.mem C x
    ⊢ AccPt x (Filter.principal C)
  -/
  have H : AccPt x (𝓟 (closure C)) := h.acc _ (subset_closure xC)
  /-
    case mpr
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    C : Set α
    inst✝ : T1Space α
    h : Perfect (closure C)
    x : α
    xC : Membership.mem C x
    H : AccPt x (Filter.principal (closure C))
    ⊢ AccPt x (Filter.principal C)
  -/
  rw [accPt_iff_frequently] at *
  have : ∀ y, y ≠ x ∧ y ∈ closure C → ∃ᶠ z in 𝓝 y, z ≠ x ∧ z ∈ C := by
    rintro y ⟨hyx, yC⟩
    simp only [← mem_compl_singleton_iff, and_comm, ← frequently_nhdsWithin_iff,
      hyx.nhdsWithin_compl_singleton, ← mem_closure_iff_frequently]
    exact yC
  /-
    case mpr
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    C : Set α
    inst✝ : T1Space α
    h : Perfect (closure C)
    x : α
    xC : Membership.mem C x
    H : Filter.Frequently (fun y => And (Ne y x) (Membership.mem (closure C) y)) ( …
    this : ∀ (y : α), And (Ne y x) (Membership.mem (closure C) y) → Filter.Frequen …
    ⊢ Filter.Frequently (fun y => And (Ne y x) (Membership.mem C y)) (nhds x)
  -/
  rw [← frequently_frequently_nhds]
  /-
    case mpr
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    C : Set α
    inst✝ : T1Space α
    h : Perfect (closure C)
    x : α
    xC : Membership.mem C x
    H : Filter.Frequently (fun y => And (Ne y x) (Membership.mem (closure C) y)) ( …
    this : ∀ (y : α), And (Ne y x) (Membership.mem (closure C) y) → Filter.Frequen …
    ⊢ Filter.Frequently (fun x' => Filter.Frequently (fun x'' => And (Ne x'' x) (M …
  -/
  exact H.mono this
  /-
    🎉 no goals
  -/


theorem Perfect.closure_nhds_inter {U : Set α} (hC : Perfect C) (x : α) (xC : x ∈ C) (xU : x ∈ U)
    (Uop : IsOpen U) : Perfect (closure (U ∩ C)) ∧ (closure (U ∩ C)).Nonempty := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    C U : Set α
    hC : Perfect C
    x : α
    xC : Membership.mem C x
    xU : Membership.mem U x
    Uop : IsOpen U
    ⊢ And (Perfect (closure (Inter.inter U C))) (closure (Inter.inter U C)).Nonempty
  -/
  constructor
    /-
      case left
      α : Type u_1
      inst✝ : TopologicalSpace α
      C U : Set α
      hC : Perfect C
      x : α
      xC : Membership.mem C x
      xU : Membership.mem U x
      Uop : IsOpen U
      ⊢ Perfect (closure (Inter.inter U C))
    -/
  · apply Preperfect.perfect_closure
    /-
      case left.hC
      α : Type u_1
      inst✝ : TopologicalSpace α
      C U : Set α
      hC : Perfect C
      x : α
      xC : Membership.mem C x
      xU : Membership.mem U x
      Uop : IsOpen U
      ⊢ Preperfect (Inter.inter U C)
    -/
    exact hC.acc.open_inter Uop
    /-
      🎉 no goals
    -/
  /-
    case right
    α : Type u_1
    inst✝ : TopologicalSpace α
    C U : Set α
    hC : Perfect C
    x : α
    xC : Membership.mem C x
    xU : Membership.mem U x
    Uop : IsOpen U
    ⊢ (closure (Inter.inter U C)).Nonempty
  -/
  apply Nonempty.closure
  /-
    case right.a
    α : Type u_1
    inst✝ : TopologicalSpace α
    C U : Set α
    hC : Perfect C
    x : α
    xC : Membership.mem C x
    xU : Membership.mem U x
    Uop : IsOpen U
    ⊢ (Inter.inter U C).Nonempty
  -/
  exact ⟨x, ⟨xU, xC⟩⟩
  /-
    🎉 no goals
  -/


/-- Given a perfect nonempty set in a T2.5 space, we can find two disjoint perfect subsets.
This is the main inductive step in the proof of the Cantor-Bendixson Theorem. -/
theorem Perfect.splitting [T25Space α] (hC : Perfect C) (hnonempty : C.Nonempty) :
    ∃ C₀ C₁ : Set α,
    (Perfect C₀ ∧ C₀.Nonempty ∧ C₀ ⊆ C) ∧ (Perfect C₁ ∧ C₁.Nonempty ∧ C₁ ⊆ C) ∧ Disjoint C₀ C₁ := by
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    C : Set α
    inst✝ : T25Space α
    hC : Perfect C
    hnonempty : C.Nonempty
    ⊢ Exists fun C₀ => Exists fun C₁ => And (And (Perfect C₀) (And C₀.Nonempty (Ha …
  -/
  cases' hnonempty with y yC
  obtain ⟨x, xC, hxy⟩ : ∃ x ∈ C, x ≠ y := by
    have := hC.acc _ yC
    rw [accPt_iff_nhds] at this
    rcases this univ univ_mem with ⟨x, xC, hxy⟩
    exact ⟨x, xC.2, hxy⟩
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    C : Set α
    inst✝ : T25Space α
    hC : Perfect C
    y : α
    yC : Membership.mem C y
    x : α
    xC : Membership.mem C x
    hxy : Ne x y
    ⊢ Exists fun C₀ => Exists fun C₁ => And (And (Perfect C₀) (And C₀.Nonempty (Ha …
  -/
  obtain ⟨U, xU, Uop, V, yV, Vop, hUV⟩ := exists_open_nhds_disjoint_closure hxy
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    C : Set α
    inst✝ : T25Space α
    hC : Perfect C
    y : α
    yC : Membership.mem C y
    x : α
    xC : Membership.mem C x
    hxy : Ne x y
    U : Set α
    xU : Membership.mem U x
    Uop : IsOpen U
    V : Set α
    yV : Membership.mem V y
    Vop : IsOpen V
    hUV : Disjoint (closure U) (closure V)
    ⊢ Exists fun C₀ => Exists fun C₁ => And (And (Perfect C₀) (And C₀.Nonempty (Ha …
  -/
  use closure (U ∩ C), closure (V ∩ C)
  /-
    case h
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    C : Set α
    inst✝ : T25Space α
    hC : Perfect C
    y : α
    yC : Membership.mem C y
    x : α
    xC : Membership.mem C x
    hxy : Ne x y
    U : Set α
    xU : Membership.mem U x
    Uop : IsOpen U
    V : Set α
    yV : Membership.mem V y
    Vop : IsOpen V
    hUV : Disjoint (closure U) (closure V)
    ⊢ And (And (Perfect (closure (Inter.inter U C))) (And (closure (Inter.inter U  …
  -/
  constructor <;> rw [← and_assoc]
    /-
      case h.left
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      C : Set α
      inst✝ : T25Space α
      hC : Perfect C
      y : α
      yC : Membership.mem C y
      x : α
      xC : Membership.mem C x
      hxy : Ne x y
      U : Set α
      xU : Membership.mem U x
      Uop : IsOpen U
      V : Set α
      yV : Membership.mem V y
      Vop : IsOpen V
      hUV : Disjoint (closure U) (closure V)
      ⊢ And (And (Perfect (closure (Inter.inter U C))) (closure (Inter.inter U C)).N …
    -/
  · refine ⟨hC.closure_nhds_inter x xC xU Uop, ?_⟩
    /-
      case h.left
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      C : Set α
      inst✝ : T25Space α
      hC : Perfect C
      y : α
      yC : Membership.mem C y
      x : α
      xC : Membership.mem C x
      hxy : Ne x y
      U : Set α
      xU : Membership.mem U x
      Uop : IsOpen U
      V : Set α
      yV : Membership.mem V y
      Vop : IsOpen V
      hUV : Disjoint (closure U) (closure V)
      ⊢ HasSubset.Subset (closure (Inter.inter U C)) C
    -/
    rw [hC.closed.closure_subset_iff]
    /-
      case h.left
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      C : Set α
      inst✝ : T25Space α
      hC : Perfect C
      y : α
      yC : Membership.mem C y
      x : α
      xC : Membership.mem C x
      hxy : Ne x y
      U : Set α
      xU : Membership.mem U x
      Uop : IsOpen U
      V : Set α
      yV : Membership.mem V y
      Vop : IsOpen V
      hUV : Disjoint (closure U) (closure V)
      ⊢ HasSubset.Subset (Inter.inter U C) C
    -/
    exact inter_subset_right
    /-
      🎉 no goals
    -/
  /-
    case h.right
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    C : Set α
    inst✝ : T25Space α
    hC : Perfect C
    y : α
    yC : Membership.mem C y
    x : α
    xC : Membership.mem C x
    hxy : Ne x y
    U : Set α
    xU : Membership.mem U x
    Uop : IsOpen U
    V : Set α
    yV : Membership.mem V y
    Vop : IsOpen V
    hUV : Disjoint (closure U) (closure V)
    ⊢ And (And (And (Perfect (closure (Inter.inter V C))) (closure (Inter.inter V  …
  -/
  constructor
    /-
      case h.right.left
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      C : Set α
      inst✝ : T25Space α
      hC : Perfect C
      y : α
      yC : Membership.mem C y
      x : α
      xC : Membership.mem C x
      hxy : Ne x y
      U : Set α
      xU : Membership.mem U x
      Uop : IsOpen U
      V : Set α
      yV : Membership.mem V y
      Vop : IsOpen V
      hUV : Disjoint (closure U) (closure V)
      ⊢ And (And (Perfect (closure (Inter.inter V C))) (closure (Inter.inter V C)).N …
    -/
  · refine ⟨hC.closure_nhds_inter y yC yV Vop, ?_⟩
    /-
      case h.right.left
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      C : Set α
      inst✝ : T25Space α
      hC : Perfect C
      y : α
      yC : Membership.mem C y
      x : α
      xC : Membership.mem C x
      hxy : Ne x y
      U : Set α
      xU : Membership.mem U x
      Uop : IsOpen U
      V : Set α
      yV : Membership.mem V y
      Vop : IsOpen V
      hUV : Disjoint (closure U) (closure V)
      ⊢ HasSubset.Subset (closure (Inter.inter V C)) C
    -/
    rw [hC.closed.closure_subset_iff]
    /-
      case h.right.left
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      C : Set α
      inst✝ : T25Space α
      hC : Perfect C
      y : α
      yC : Membership.mem C y
      x : α
      xC : Membership.mem C x
      hxy : Ne x y
      U : Set α
      xU : Membership.mem U x
      Uop : IsOpen U
      V : Set α
      yV : Membership.mem V y
      Vop : IsOpen V
      hUV : Disjoint (closure U) (closure V)
      ⊢ HasSubset.Subset (Inter.inter V C) C
    -/
    exact inter_subset_right
    /-
      🎉 no goals
    -/
  /-
    case h.right.right
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    C : Set α
    inst✝ : T25Space α
    hC : Perfect C
    y : α
    yC : Membership.mem C y
    x : α
    xC : Membership.mem C x
    hxy : Ne x y
    U : Set α
    xU : Membership.mem U x
    Uop : IsOpen U
    V : Set α
    yV : Membership.mem V y
    Vop : IsOpen V
    hUV : Disjoint (closure U) (closure V)
    ⊢ Disjoint (closure (Inter.inter U C)) (closure (Inter.inter V C))
  -/
                                                         /-
                                                           🎉 no goals
                                                         -/
  apply Disjoint.mono _ _ hUV <;> apply closure_mono <;> exact inter_subset_left
                                                         /-
                                                           🎉 no goals
                                                         -/


lemma IsPreconnected.preperfect_of_nontrivial [T1Space α] {U : Set α} (hu : U.Nontrivial)
    (h : IsPreconnected U) : Preperfect U := by
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : T1Space α
    U : Set α
    hu : U.Nontrivial
    h : IsPreconnected U
    ⊢ Preperfect U
  -/
  intro x hx
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : T1Space α
    U : Set α
    hu : U.Nontrivial
    h : IsPreconnected U
    x : α
    hx : Membership.mem U x
    ⊢ AccPt x (Filter.principal U)
  -/
  rw [isPreconnected_closed_iff] at h
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : T1Space α
    U : Set α
    hu : U.Nontrivial
    h : ∀ (t t' : Set α), IsClosed t → IsClosed t' → HasSubset.Subset U (Union.uni …
    x : α
    hx : Membership.mem U x
    ⊢ AccPt x (Filter.principal U)
  -/
  specialize h {x} (closure (U \ {x})) isClosed_singleton isClosed_closure ?_ ?_ ?_
    /-
      case specialize_1
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : T1Space α
      U : Set α
      hu : U.Nontrivial
      h : ∀ (t t' : Set α), IsClosed t → IsClosed t' → HasSubset.Subset U (Union.uni …
      x : α
      hx : Membership.mem U x
      ⊢ HasSubset.Subset U (Union.union (Singleton.singleton x) (closure (SDiff.sdif …
    -/
  · trans {x} ∪ (U \ {x})
      /-
        α : Type u_1
        inst✝¹ : TopologicalSpace α
        inst✝ : T1Space α
        U : Set α
        hu : U.Nontrivial
        h : ∀ (t t' : Set α), IsClosed t → IsClosed t' → HasSubset.Subset U (Union.uni …
        x : α
        hx : Membership.mem U x
        ⊢ HasSubset.Subset U (Union.union (Singleton.singleton x) (SDiff.sdiff U (Sing …
      -/
    · simp
      /-
        🎉 no goals
      -/
    /-
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : T1Space α
      U : Set α
      hu : U.Nontrivial
      h : ∀ (t t' : Set α), IsClosed t → IsClosed t' → HasSubset.Subset U (Union.uni …
      x : α
      hx : Membership.mem U x
      ⊢ HasSubset.Subset (Union.union (Singleton.singleton x) (SDiff.sdiff U (Single …
    -/
    apply Set.union_subset_union_right
    /-
      case h
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : T1Space α
      U : Set α
      hu : U.Nontrivial
      h : ∀ (t t' : Set α), IsClosed t → IsClosed t' → HasSubset.Subset U (Union.uni …
      x : α
      hx : Membership.mem U x
      ⊢ HasSubset.Subset (SDiff.sdiff U (Singleton.singleton x)) (closure (SDiff.sdi …
    -/
    exact subset_closure
    /-
      🎉 no goals
    -/
    /-
      case specialize_2
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : T1Space α
      U : Set α
      hu : U.Nontrivial
      h : ∀ (t t' : Set α), IsClosed t → IsClosed t' → HasSubset.Subset U (Union.uni …
      x : α
      hx : Membership.mem U x
      ⊢ (Inter.inter U (Singleton.singleton x)).Nonempty
    -/
  · exact Set.inter_singleton_nonempty.mpr hx
    /-
      🎉 no goals
    -/
    /-
      case specialize_3
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : T1Space α
      U : Set α
      hu : U.Nontrivial
      h : ∀ (t t' : Set α), IsClosed t → IsClosed t' → HasSubset.Subset U (Union.uni …
      x : α
      hx : Membership.mem U x
      ⊢ (Inter.inter U (closure (SDiff.sdiff U (Singleton.singleton x)))).Nonempty
    -/
  · obtain ⟨y, hy⟩ := Set.Nontrivial.exists_ne hu x
    /-
      case specialize_3.intro
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : T1Space α
      U : Set α
      hu : U.Nontrivial
      h : ∀ (t t' : Set α), IsClosed t → IsClosed t' → HasSubset.Subset U (Union.uni …
      x : α
      hx : Membership.mem U x
      y : α
      hy : And (Membership.mem U y) (Ne y x)
      ⊢ (Inter.inter U (closure (SDiff.sdiff U (Singleton.singleton x)))).Nonempty
    -/
    use y
    /-
      case h
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : T1Space α
      U : Set α
      hu : U.Nontrivial
      h : ∀ (t t' : Set α), IsClosed t → IsClosed t' → HasSubset.Subset U (Union.uni …
      x : α
      hx : Membership.mem U x
      y : α
      hy : And (Membership.mem U y) (Ne y x)
      ⊢ Membership.mem (Inter.inter U (closure (SDiff.sdiff U (Singleton.singleton x …
    -/
    simp only [Set.mem_inter_iff, hy, true_and]
    /-
      case h
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : T1Space α
      U : Set α
      hu : U.Nontrivial
      h : ∀ (t t' : Set α), IsClosed t → IsClosed t' → HasSubset.Subset U (Union.uni …
      x : α
      hx : Membership.mem U x
      y : α
      hy : And (Membership.mem U y) (Ne y x)
      ⊢ Membership.mem (closure (SDiff.sdiff U (Singleton.singleton x))) y
    -/
    apply subset_closure
    /-
      case h.a
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : T1Space α
      U : Set α
      hu : U.Nontrivial
      h : ∀ (t t' : Set α), IsClosed t → IsClosed t' → HasSubset.Subset U (Union.uni …
      x : α
      hx : Membership.mem U x
      y : α
      hy : And (Membership.mem U y) (Ne y x)
      ⊢ Membership.mem (SDiff.sdiff U (Singleton.singleton x)) y
    -/
    simp [hy]
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : T1Space α
      U : Set α
      hu : U.Nontrivial
      x : α
      hx : Membership.mem U x
      h : (Inter.inter U (Inter.inter (Singleton.singleton x) (closure (SDiff.sdiff  …
      ⊢ AccPt x (Filter.principal U)
    -/
  · apply Set.Nonempty.right at h
    /-
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : T1Space α
      U : Set α
      hu : U.Nontrivial
      x : α
      hx : Membership.mem U x
      h : (Inter.inter (Singleton.singleton x) (closure (SDiff.sdiff U (Singleton.si …
      ⊢ AccPt x (Filter.principal U)
    -/
    rw [Set.singleton_inter_nonempty, mem_closure_iff_clusterPt, ← acc_principal_iff_cluster] at h
    /-
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      inst✝ : T1Space α
      U : Set α
      hu : U.Nontrivial
      x : α
      hx : Membership.mem U x
      h : AccPt x (Filter.principal U)
      ⊢ AccPt x (Filter.principal U)
    -/
    exact h
    /-
      🎉 no goals
    -/


/-- The **Cantor-Bendixson Theorem**: Any closed subset of a second countable space
can be written as the union of a countable set and a perfect set. -/
theorem exists_countable_union_perfect_of_isClosed [SecondCountableTopology α]
    (hclosed : IsClosed C) : ∃ V D : Set α, V.Countable ∧ Perfect D ∧ C = V ∪ D := by
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    C : Set α
    inst✝ : SecondCountableTopology α
    hclosed : IsClosed C
    ⊢ Exists fun V => Exists fun D => And V.Countable (And (Perfect D) (Eq C (Unio …
  -/
  obtain ⟨b, bct, _, bbasis⟩ := TopologicalSpace.exists_countable_basis α
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    C : Set α
    inst✝ : SecondCountableTopology α
    hclosed : IsClosed C
    b : Set (Set α)
    bct : b.Countable
    left✝ : Not (Membership.mem b EmptyCollection.emptyCollection)
    bbasis : TopologicalSpace.IsTopologicalBasis b
    ⊢ Exists fun V => Exists fun D => And V.Countable (And (Perfect D) (Eq C (Unio …
  -/
  let v := { U ∈ b | (U ∩ C).Countable }
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    C : Set α
    inst✝ : SecondCountableTopology α
    hclosed : IsClosed C
    b : Set (Set α)
    bct : b.Countable
    left✝ : Not (Membership.mem b EmptyCollection.emptyCollection)
    bbasis : TopologicalSpace.IsTopologicalBasis b
    v : Set (Set α) := setOf fun U => And (Membership.mem b U) (Inter.inter U C).C …
    ⊢ Exists fun V => Exists fun D => And V.Countable (And (Perfect D) (Eq C (Unio …
  -/
  let V := ⋃ U ∈ v, U
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    C : Set α
    inst✝ : SecondCountableTopology α
    hclosed : IsClosed C
    b : Set (Set α)
    bct : b.Countable
    left✝ : Not (Membership.mem b EmptyCollection.emptyCollection)
    bbasis : TopologicalSpace.IsTopologicalBasis b
    v : Set (Set α) := setOf fun U => And (Membership.mem b U) (Inter.inter U C).C …
    V : Set α := Set.iUnion fun U => Set.iUnion fun h => U
    ⊢ Exists fun V => Exists fun D => And V.Countable (And (Perfect D) (Eq C (Unio …
  -/
  let D := C \ V
  have Vct : (V ∩ C).Countable := by
    simp only [V, iUnion_inter, mem_sep_iff]
    apply Countable.biUnion
    · exact Countable.mono inter_subset_left bct
    · exact inter_subset_right
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    C : Set α
    inst✝ : SecondCountableTopology α
    hclosed : IsClosed C
    b : Set (Set α)
    bct : b.Countable
    left✝ : Not (Membership.mem b EmptyCollection.emptyCollection)
    bbasis : TopologicalSpace.IsTopologicalBasis b
    v : Set (Set α) := setOf fun U => And (Membership.mem b U) (Inter.inter U C).C …
    V : Set α := Set.iUnion fun U => Set.iUnion fun h => U
    D : Set α := SDiff.sdiff C V
    Vct : (Inter.inter V C).Countable
    ⊢ Exists fun V => Exists fun D => And V.Countable (And (Perfect D) (Eq C (Unio …
  -/
  refine ⟨V ∩ C, D, Vct, ⟨?_, ?_⟩, ?_⟩
    /-
      case intro.intro.intro.refine_1
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      C : Set α
      inst✝ : SecondCountableTopology α
      hclosed : IsClosed C
      b : Set (Set α)
      bct : b.Countable
      left✝ : Not (Membership.mem b EmptyCollection.emptyCollection)
      bbasis : TopologicalSpace.IsTopologicalBasis b
      v : Set (Set α) := setOf fun U => And (Membership.mem b U) (Inter.inter U C).C …
      V : Set α := Set.iUnion fun U => Set.iUnion fun h => U
      D : Set α := SDiff.sdiff C V
      Vct : (Inter.inter V C).Countable
      ⊢ IsClosed D
    -/
  · refine hclosed.sdiff (isOpen_biUnion fun _ ↦ ?_)
    /-
      case intro.intro.intro.refine_1
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      C : Set α
      inst✝ : SecondCountableTopology α
      hclosed : IsClosed C
      b : Set (Set α)
      bct : b.Countable
      left✝ : Not (Membership.mem b EmptyCollection.emptyCollection)
      bbasis : TopologicalSpace.IsTopologicalBasis b
      v : Set (Set α) := setOf fun U => And (Membership.mem b U) (Inter.inter U C).C …
      V : Set α := Set.iUnion fun U => Set.iUnion fun h => U
      D : Set α := SDiff.sdiff C V
      Vct : (Inter.inter V C).Countable
      x✝ : Set α
      ⊢ Membership.mem v x✝ → IsOpen x✝
    -/
    exact fun ⟨Ub, _⟩ ↦ IsTopologicalBasis.isOpen bbasis Ub
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.refine_2
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      C : Set α
      inst✝ : SecondCountableTopology α
      hclosed : IsClosed C
      b : Set (Set α)
      bct : b.Countable
      left✝ : Not (Membership.mem b EmptyCollection.emptyCollection)
      bbasis : TopologicalSpace.IsTopologicalBasis b
      v : Set (Set α) := setOf fun U => And (Membership.mem b U) (Inter.inter U C).C …
      V : Set α := Set.iUnion fun U => Set.iUnion fun h => U
      D : Set α := SDiff.sdiff C V
      Vct : (Inter.inter V C).Countable
      ⊢ Preperfect D
    -/
  · rw [preperfect_iff_nhds]
    /-
      case intro.intro.intro.refine_2
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      C : Set α
      inst✝ : SecondCountableTopology α
      hclosed : IsClosed C
      b : Set (Set α)
      bct : b.Countable
      left✝ : Not (Membership.mem b EmptyCollection.emptyCollection)
      bbasis : TopologicalSpace.IsTopologicalBasis b
      v : Set (Set α) := setOf fun U => And (Membership.mem b U) (Inter.inter U C).C …
      V : Set α := Set.iUnion fun U => Set.iUnion fun h => U
      D : Set α := SDiff.sdiff C V
      Vct : (Inter.inter V C).Countable
      ⊢ ∀ (x : α), Membership.mem D x → ∀ (U : Set α), Membership.mem (nhds x) U → E …
    -/
    intro x xD E xE
    have : ¬(E ∩ D).Countable := by
      intro h
      obtain ⟨U, hUb, xU, hU⟩ : ∃ U ∈ b, x ∈ U ∧ U ⊆ E :=
        (IsTopologicalBasis.mem_nhds_iff bbasis).mp xE
      have hU_cnt : (U ∩ C).Countable := by
        apply @Countable.mono _ _ (E ∩ D ∪ V ∩ C)
        · rintro y ⟨yU, yC⟩
          by_cases h : y ∈ V
          · exact mem_union_right _ (mem_inter h yC)
          · exact mem_union_left _ (mem_inter (hU yU) ⟨yC, h⟩)
        exact Countable.union h Vct
      have : U ∈ v := ⟨hUb, hU_cnt⟩
      apply xD.2
      exact mem_biUnion this xU
    /-
      case intro.intro.intro.refine_2
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      C : Set α
      inst✝ : SecondCountableTopology α
      hclosed : IsClosed C
      b : Set (Set α)
      bct : b.Countable
      left✝ : Not (Membership.mem b EmptyCollection.emptyCollection)
      bbasis : TopologicalSpace.IsTopologicalBasis b
      v : Set (Set α) := setOf fun U => And (Membership.mem b U) (Inter.inter U C).C …
      V : Set α := Set.iUnion fun U => Set.iUnion fun h => U
      D : Set α := SDiff.sdiff C V
      Vct : (Inter.inter V C).Countable
      x : α
      xD : Membership.mem D x
      E : Set α
      xE : Membership.mem (nhds x) E
      this : Not (Inter.inter E D).Countable
      ⊢ Exists fun y => And (Membership.mem (Inter.inter E D) y) (Ne y x)
    -/
    by_contra! h
    /-
      case intro.intro.intro.refine_2
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      C : Set α
      inst✝ : SecondCountableTopology α
      hclosed : IsClosed C
      b : Set (Set α)
      bct : b.Countable
      left✝ : Not (Membership.mem b EmptyCollection.emptyCollection)
      bbasis : TopologicalSpace.IsTopologicalBasis b
      v : Set (Set α) := setOf fun U => And (Membership.mem b U) (Inter.inter U C).C …
      V : Set α := Set.iUnion fun U => Set.iUnion fun h => U
      D : Set α := SDiff.sdiff C V
      Vct : (Inter.inter V C).Countable
      x : α
      xD : Membership.mem D x
      E : Set α
      xE : Membership.mem (nhds x) E
      this : Not (Inter.inter E D).Countable
      h : ∀ (y : α), Membership.mem (Inter.inter E D) y → Eq y x
      ⊢ False
    -/
    exact absurd (Countable.mono h (Set.countable_singleton _)) this
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.refine_3
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      C : Set α
      inst✝ : SecondCountableTopology α
      hclosed : IsClosed C
      b : Set (Set α)
      bct : b.Countable
      left✝ : Not (Membership.mem b EmptyCollection.emptyCollection)
      bbasis : TopologicalSpace.IsTopologicalBasis b
      v : Set (Set α) := setOf fun U => And (Membership.mem b U) (Inter.inter U C).C …
      V : Set α := Set.iUnion fun U => Set.iUnion fun h => U
      D : Set α := SDiff.sdiff C V
      Vct : (Inter.inter V C).Countable
      ⊢ Eq C (Union.union (Inter.inter V C) D)
    -/
  · rw [inter_comm, inter_union_diff]
    /-
      🎉 no goals
    -/


/-- Any uncountable closed set in a second countable space contains a nonempty perfect subset. -/
theorem exists_perfect_nonempty_of_isClosed_of_not_countable [SecondCountableTopology α]
    (hclosed : IsClosed C) (hunc : ¬C.Countable) : ∃ D : Set α, Perfect D ∧ D.Nonempty ∧ D ⊆ C := by
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    C : Set α
    inst✝ : SecondCountableTopology α
    hclosed : IsClosed C
    hunc : Not C.Countable
    ⊢ Exists fun D => And (Perfect D) (And D.Nonempty (HasSubset.Subset D C))
  -/
  rcases exists_countable_union_perfect_of_isClosed hclosed with ⟨V, D, Vct, Dperf, VD⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    C : Set α
    inst✝ : SecondCountableTopology α
    hclosed : IsClosed C
    hunc : Not C.Countable
    V D : Set α
    Vct : V.Countable
    Dperf : Perfect D
    VD : Eq C (Union.union V D)
    ⊢ Exists fun D => And (Perfect D) (And D.Nonempty (HasSubset.Subset D C))
  -/
  refine ⟨D, ⟨Dperf, ?_⟩⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    C : Set α
    inst✝ : SecondCountableTopology α
    hclosed : IsClosed C
    hunc : Not C.Countable
    V D : Set α
    Vct : V.Countable
    Dperf : Perfect D
    VD : Eq C (Union.union V D)
    ⊢ And D.Nonempty (HasSubset.Subset D C)
  -/
  constructor
    /-
      case intro.intro.intro.intro.left
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      C : Set α
      inst✝ : SecondCountableTopology α
      hclosed : IsClosed C
      hunc : Not C.Countable
      V D : Set α
      Vct : V.Countable
      Dperf : Perfect D
      VD : Eq C (Union.union V D)
      ⊢ D.Nonempty
    -/
  · rw [nonempty_iff_ne_empty]
    /-
      case intro.intro.intro.intro.left
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      C : Set α
      inst✝ : SecondCountableTopology α
      hclosed : IsClosed C
      hunc : Not C.Countable
      V D : Set α
      Vct : V.Countable
      Dperf : Perfect D
      VD : Eq C (Union.union V D)
      ⊢ Ne D EmptyCollection.emptyCollection
    -/
    by_contra h
    /-
      case intro.intro.intro.intro.left
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      C : Set α
      inst✝ : SecondCountableTopology α
      hclosed : IsClosed C
      hunc : Not C.Countable
      V D : Set α
      Vct : V.Countable
      Dperf : Perfect D
      VD : Eq C (Union.union V D)
      h : Eq D EmptyCollection.emptyCollection
      ⊢ False
    -/
    rw [h, union_empty] at VD
    /-
      case intro.intro.intro.intro.left
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      C : Set α
      inst✝ : SecondCountableTopology α
      hclosed : IsClosed C
      hunc : Not C.Countable
      V D : Set α
      Vct : V.Countable
      Dperf : Perfect D
      VD : Eq C V
      h : Eq D EmptyCollection.emptyCollection
      ⊢ False
    -/
    rw [VD] at hunc
    /-
      case intro.intro.intro.intro.left
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      C : Set α
      inst✝ : SecondCountableTopology α
      hclosed : IsClosed C
      V : Set α
      hunc : Not V.Countable
      D : Set α
      Vct : V.Countable
      Dperf : Perfect D
      VD : Eq C V
      h : Eq D EmptyCollection.emptyCollection
      ⊢ False
    -/
    contradiction
    /-
      🎉 no goals
    -/
  /-
    case intro.intro.intro.intro.right
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    C : Set α
    inst✝ : SecondCountableTopology α
    hclosed : IsClosed C
    hunc : Not C.Countable
    V D : Set α
    Vct : V.Countable
    Dperf : Perfect D
    VD : Eq C (Union.union V D)
    ⊢ HasSubset.Subset D C
  -/
  rw [VD]
  /-
    case intro.intro.intro.intro.right
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    C : Set α
    inst✝ : SecondCountableTopology α
    hclosed : IsClosed C
    hunc : Not C.Countable
    V D : Set α
    Vct : V.Countable
    Dperf : Perfect D
    VD : Eq C (Union.union V D)
    ⊢ HasSubset.Subset D (Union.union V D)
  -/
  exact subset_union_right
  /-
    🎉 no goals
  -/


theorem perfectSpace_iff_forall_not_isolated : PerfectSpace X ↔ ∀ x : X, Filter.NeBot (𝓝[≠] x) := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    ⊢ Iff (PerfectSpace X) (∀ (x : X), (nhdsWithin x (HasCompl.compl (Singleton.si …
  -/
  simp [perfectSpace_def, Preperfect, AccPt]
  /-
    🎉 no goals
  -/


instance PerfectSpace.not_isolated [PerfectSpace X] (x : X) : Filter.NeBot (𝓝[≠] x) :=
  perfectSpace_iff_forall_not_isolated.mp ‹_› x


