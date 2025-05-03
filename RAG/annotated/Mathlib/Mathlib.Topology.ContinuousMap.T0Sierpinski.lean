theorem eq_induced_by_maps_to_sierpinski (X : Type*) [t : TopologicalSpace X] :
    t = ⨅ u : Opens X, sierpinskiSpace.induced (· ∈ u) := by
  /-
    X : Type u_1
    t : TopologicalSpace X
    ⊢ Eq t (iInf fun u => TopologicalSpace.induced (fun x => Membership.mem u x) s …
  -/
  apply le_antisymm
    /-
      case a
      X : Type u_1
      t : TopologicalSpace X
      ⊢ LE.le t (iInf fun u => TopologicalSpace.induced (fun x => Membership.mem u x …
    -/
  · rw [le_iInf_iff]
    /-
      case a
      X : Type u_1
      t : TopologicalSpace X
      ⊢ ∀ (i : TopologicalSpace.Opens X), LE.le t (TopologicalSpace.induced (fun x = …
    -/
    exact fun u => Continuous.le_induced (isOpen_iff_continuous_mem.mp u.2)
    /-
      🎉 no goals
    -/
    /-
      case a
      X : Type u_1
      t : TopologicalSpace X
      ⊢ LE.le (iInf fun u => TopologicalSpace.induced (fun x => Membership.mem u x)  …
    -/
  · intro u h
    /-
      case a
      X : Type u_1
      t : TopologicalSpace X
      u : Set X
      h : IsOpen u
      ⊢ IsOpen u
    -/
    rw [← generateFrom_iUnion_isOpen]
    /-
      case a
      X : Type u_1
      t : TopologicalSpace X
      u : Set X
      h : IsOpen u
      ⊢ IsOpen u
    -/
    apply isOpen_generateFrom_of_mem
    /-
      case a.hs
      X : Type u_1
      t : TopologicalSpace X
      u : Set X
      h : IsOpen u
      ⊢ Membership.mem (Set.iUnion fun i => setOf fun s => IsOpen s) u
    -/
    simp only [Set.mem_iUnion, Set.mem_setOf_eq, isOpen_induced_iff]
    /-
      case a.hs
      X : Type u_1
      t : TopologicalSpace X
      u : Set X
      h : IsOpen u
      ⊢ Exists fun i => Exists fun t_1 => And (IsOpen t_1) (Eq (Set.preimage (fun x  …
    -/
    exact ⟨⟨u, h⟩, {True}, isOpen_singleton_true, by simp [Set.preimage]⟩
    /-
      🎉 no goals
    -/


/-- The continuous map from `X` to the product of copies of the Sierpinski space, (one copy for each
open subset `u` of `X`). The `u` coordinate of `productOfMemOpens x` is given by `x ∈ u`.
-/
def productOfMemOpens : C(X, Opens X → Prop) where
  toFun x u := x ∈ u
  continuous_toFun := continuous_pi_iff.2 fun u => continuous_Prop.2 u.isOpen


theorem productOfMemOpens_isInducing : IsInducing (productOfMemOpens X) := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    ⊢ Topology.IsInducing ⇑(TopologicalSpace.productOfMemOpens X)
  -/
  convert inducing_iInf_to_pi fun (u : Opens X) (x : X) => x ∈ u
  /-
    case h.e'_3
    X : Type u_1
    inst✝ : TopologicalSpace X
    ⊢ Eq inst✝ (iInf fun i => TopologicalSpace.induced (fun x => Membership.mem i  …
  -/
  apply eq_induced_by_maps_to_sierpinski
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-28")]
alias productOfMemOpens_inducing := productOfMemOpens_isInducing


theorem productOfMemOpens_injective [T0Space X] : Function.Injective (productOfMemOpens X) := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T0Space X
    ⊢ Function.Injective ⇑(TopologicalSpace.productOfMemOpens X)
  -/
  intro x1 x2 h
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T0Space X
    x1 x2 : X
    h : Eq ((TopologicalSpace.productOfMemOpens X) x1) ((TopologicalSpace.productO …
    ⊢ Eq x1 x2
  -/
  apply Inseparable.eq
  /-
    case h
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T0Space X
    x1 x2 : X
    h : Eq ((TopologicalSpace.productOfMemOpens X) x1) ((TopologicalSpace.productO …
    ⊢ Inseparable x1 x2
  -/
  rw [← IsInducing.inseparable_iff (productOfMemOpens_isInducing X), h]
  /-
    🎉 no goals
  -/


theorem productOfMemOpens_isEmbedding [T0Space X] : IsEmbedding (productOfMemOpens X) :=
  .mk (productOfMemOpens_isInducing X) (productOfMemOpens_injective X)


@[deprecated (since := "2024-10-26")]
alias productOfMemOpens_embedding := productOfMemOpens_isEmbedding


