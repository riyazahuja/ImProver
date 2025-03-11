/-- Let `L` be a topological space and let `L×L` be equipped with the product topology and let
`⊓:L×L → L` be an infimum. Then `L` is said to have *(jointly) continuous infimum* if the map
`⊓:L×L → L` is continuous.
-/
class ContinuousInf (L : Type*) [TopologicalSpace L] [Min L] : Prop where
  /-- The infimum is continuous -/
  continuous_inf : Continuous fun p : L × L => p.1 ⊓ p.2


/-- Let `L` be a topological space and let `L×L` be equipped with the product topology and let
`⊓:L×L → L` be a supremum. Then `L` is said to have *(jointly) continuous supremum* if the map
`⊓:L×L → L` is continuous.
-/
class ContinuousSup (L : Type*) [TopologicalSpace L] [Max L] : Prop where
  /-- The supremum is continuous -/
  continuous_sup : Continuous fun p : L × L => p.1 ⊔ p.2

-- see Note [lower instance priority]

instance (priority := 100) OrderDual.continuousSup (L : Type*) [TopologicalSpace L] [Min L]
    [ContinuousInf L] : ContinuousSup Lᵒᵈ where
  continuous_sup := @ContinuousInf.continuous_inf L _ _ _

-- see Note [lower instance priority]

instance (priority := 100) OrderDual.continuousInf (L : Type*) [TopologicalSpace L] [Max L]
    [ContinuousSup L] : ContinuousInf Lᵒᵈ where
  continuous_inf := @ContinuousSup.continuous_sup L _ _ _


/-- Let `L` be a lattice equipped with a topology such that `L` has continuous infimum and supremum.
Then `L` is said to be a *topological lattice*.
-/
class TopologicalLattice (L : Type*) [TopologicalSpace L] [Lattice L]
  extends ContinuousInf L, ContinuousSup L : Prop

-- see Note [lower instance priority]

instance (priority := 100) OrderDual.topologicalLattice (L : Type*) [TopologicalSpace L]
    [Lattice L] [TopologicalLattice L] : TopologicalLattice Lᵒᵈ where

-- see Note [lower instance priority]

instance (priority := 100) LinearOrder.topologicalLattice {L : Type*} [TopologicalSpace L]
    [LinearOrder L] [OrderClosedTopology L] : TopologicalLattice L where
  continuous_inf := continuous_min
  continuous_sup := continuous_max


@[continuity]
theorem continuous_inf [Min L] [ContinuousInf L] : Continuous fun p : L × L => p.1 ⊓ p.2 :=
  ContinuousInf.continuous_inf


@[continuity, fun_prop]
theorem Continuous.inf [Min L] [ContinuousInf L] {f g : X → L} (hf : Continuous f)
    (hg : Continuous g) : Continuous fun x => f x ⊓ g x :=
  continuous_inf.comp (hf.prod_mk hg : _)


@[continuity]
theorem continuous_sup [Max L] [ContinuousSup L] : Continuous fun p : L × L => p.1 ⊔ p.2 :=
  ContinuousSup.continuous_sup


@[continuity, fun_prop]
theorem Continuous.sup [Max L] [ContinuousSup L] {f g : X → L} (hf : Continuous f)
    (hg : Continuous g) : Continuous fun x => f x ⊔ g x :=
  continuous_sup.comp (hf.prod_mk hg : _)


lemma sup_nhds' [Max L] [ContinuousSup L] (hf : Tendsto f l (𝓝 x)) (hg : Tendsto g l (𝓝 y)) :
    Tendsto (f ⊔ g) l (𝓝 (x ⊔ y)) :=
  (continuous_sup.tendsto _).comp (Tendsto.prod_mk_nhds hf hg)


lemma sup_nhds [Max L] [ContinuousSup L] (hf : Tendsto f l (𝓝 x)) (hg : Tendsto g l (𝓝 y)) :
    Tendsto (fun i => f i ⊔ g i) l (𝓝 (x ⊔ y)) :=
  hf.sup_nhds' hg


lemma inf_nhds' [Min L] [ContinuousInf L] (hf : Tendsto f l (𝓝 x)) (hg : Tendsto g l (𝓝 y)) :
    Tendsto (f ⊓ g) l (𝓝 (x ⊓ y)) :=
  (continuous_inf.tendsto _).comp (Tendsto.prod_mk_nhds hf hg)


lemma inf_nhds [Min L] [ContinuousInf L] (hf : Tendsto f l (𝓝 x)) (hg : Tendsto g l (𝓝 y)) :
    Tendsto (fun i => f i ⊓ g i) l (𝓝 (x ⊓ y)) :=
  hf.inf_nhds' hg


lemma finset_sup'_nhds [SemilatticeSup L] [ContinuousSup L]
    (hne : s.Nonempty) (hs : ∀ i ∈ s, Tendsto (f i) l (𝓝 (g i))) :
    Tendsto (s.sup' hne f) l (𝓝 (s.sup' hne g)) := by
  induction hne using Finset.Nonempty.cons_induction with
  | singleton => simpa using hs
  | cons a s ha hne ihs =>
    rw [forall_mem_cons] at hs
    simp only [sup'_cons, hne]
    exact hs.1.sup_nhds (ihs hs.2)


lemma finset_sup'_nhds_apply [SemilatticeSup L] [ContinuousSup L]
    (hne : s.Nonempty) (hs : ∀ i ∈ s, Tendsto (f i) l (𝓝 (g i))) :
    Tendsto (fun a ↦ s.sup' hne (f · a)) l (𝓝 (s.sup' hne g)) := by
  /-
    L : Type u_1
    inst✝² : TopologicalSpace L
    ι : Type u_3
    α : Type u_4
    s : Finset ι
    f : ι → α → L
    l : Filter α
    g : ι → L
    inst✝¹ : SemilatticeSup L
    inst✝ : ContinuousSup L
    hne : s.Nonempty
    hs : ∀ (i : ι), Membership.mem s i → Filter.Tendsto (f i) l (nhds (g i))
    ⊢ Filter.Tendsto (fun a => s.sup' hne fun x => f x a) l (nhds (s.sup' hne g))
  -/
  simpa only [← Finset.sup'_apply] using finset_sup'_nhds hne hs
  /-
    🎉 no goals
  -/


lemma finset_inf'_nhds [SemilatticeInf L] [ContinuousInf L]
    (hne : s.Nonempty) (hs : ∀ i ∈ s, Tendsto (f i) l (𝓝 (g i))) :
    Tendsto (s.inf' hne f) l (𝓝 (s.inf' hne g)) :=
  finset_sup'_nhds (L := Lᵒᵈ) hne hs


lemma finset_inf'_nhds_apply [SemilatticeInf L] [ContinuousInf L]
    (hne : s.Nonempty) (hs : ∀ i ∈ s, Tendsto (f i) l (𝓝 (g i))) :
    Tendsto (fun a ↦ s.inf' hne (f · a)) l (𝓝 (s.inf' hne g)) :=
  finset_sup'_nhds_apply (L := Lᵒᵈ) hne hs


lemma finset_sup_nhds [SemilatticeSup L] [OrderBot L] [ContinuousSup L]
    (hs : ∀ i ∈ s, Tendsto (f i) l (𝓝 (g i))) : Tendsto (s.sup f) l (𝓝 (s.sup g)) := by
  /-
    L : Type u_1
    inst✝³ : TopologicalSpace L
    ι : Type u_3
    α : Type u_4
    s : Finset ι
    f : ι → α → L
    l : Filter α
    g : ι → L
    inst✝² : SemilatticeSup L
    inst✝¹ : OrderBot L
    inst✝ : ContinuousSup L
    hs : ∀ (i : ι), Membership.mem s i → Filter.Tendsto (f i) l (nhds (g i))
    ⊢ Filter.Tendsto (s.sup f) l (nhds (s.sup g))
  -/
  rcases s.eq_empty_or_nonempty with rfl | hne
    /-
      case inl
      L : Type u_1
      inst✝³ : TopologicalSpace L
      ι : Type u_3
      α : Type u_4
      f : ι → α → L
      l : Filter α
      g : ι → L
      inst✝² : SemilatticeSup L
      inst✝¹ : OrderBot L
      inst✝ : ContinuousSup L
      hs : ∀ (i : ι), Membership.mem EmptyCollection.emptyCollection i → Filter.Tend …
      ⊢ Filter.Tendsto (EmptyCollection.emptyCollection.sup f) l (nhds (EmptyCollect …
    -/
  · simpa using tendsto_const_nhds
    /-
      🎉 no goals
    -/
    /-
      case inr
      L : Type u_1
      inst✝³ : TopologicalSpace L
      ι : Type u_3
      α : Type u_4
      s : Finset ι
      f : ι → α → L
      l : Filter α
      g : ι → L
      inst✝² : SemilatticeSup L
      inst✝¹ : OrderBot L
      inst✝ : ContinuousSup L
      hs : ∀ (i : ι), Membership.mem s i → Filter.Tendsto (f i) l (nhds (g i))
      hne : s.Nonempty
      ⊢ Filter.Tendsto (s.sup f) l (nhds (s.sup g))
    -/
  · simp only [← sup'_eq_sup hne]
    /-
      case inr
      L : Type u_1
      inst✝³ : TopologicalSpace L
      ι : Type u_3
      α : Type u_4
      s : Finset ι
      f : ι → α → L
      l : Filter α
      g : ι → L
      inst✝² : SemilatticeSup L
      inst✝¹ : OrderBot L
      inst✝ : ContinuousSup L
      hs : ∀ (i : ι), Membership.mem s i → Filter.Tendsto (f i) l (nhds (g i))
      hne : s.Nonempty
      ⊢ Filter.Tendsto (s.sup' hne f) l (nhds (s.sup' hne g))
    -/
    exact finset_sup'_nhds hne hs
    /-
      🎉 no goals
    -/


lemma finset_sup_nhds_apply [SemilatticeSup L] [OrderBot L] [ContinuousSup L]
    (hs : ∀ i ∈ s, Tendsto (f i) l (𝓝 (g i))) :
    Tendsto (fun a ↦ s.sup (f · a)) l (𝓝 (s.sup g)) := by
  /-
    L : Type u_1
    inst✝³ : TopologicalSpace L
    ι : Type u_3
    α : Type u_4
    s : Finset ι
    f : ι → α → L
    l : Filter α
    g : ι → L
    inst✝² : SemilatticeSup L
    inst✝¹ : OrderBot L
    inst✝ : ContinuousSup L
    hs : ∀ (i : ι), Membership.mem s i → Filter.Tendsto (f i) l (nhds (g i))
    ⊢ Filter.Tendsto (fun a => s.sup fun x => f x a) l (nhds (s.sup g))
  -/
  simpa only [← Finset.sup_apply] using finset_sup_nhds hs
  /-
    🎉 no goals
  -/


lemma finset_inf_nhds [SemilatticeInf L] [OrderTop L] [ContinuousInf L]
    (hs : ∀ i ∈ s, Tendsto (f i) l (𝓝 (g i))) : Tendsto (s.inf f) l (𝓝 (s.inf g)) :=
  finset_sup_nhds (L := Lᵒᵈ) hs


lemma finset_inf_nhds_apply [SemilatticeInf L] [OrderTop L] [ContinuousInf L]
    (hs : ∀ i ∈ s, Tendsto (f i) l (𝓝 (g i))) :
    Tendsto (fun a ↦ s.inf (f · a)) l (𝓝 (s.inf g)) :=
  finset_sup_nhds_apply (L := Lᵒᵈ) hs


lemma ContinuousAt.sup' (hf : ContinuousAt f x) (hg : ContinuousAt g x) :
    ContinuousAt (f ⊔ g) x :=
  hf.sup_nhds' hg


@[fun_prop]
lemma ContinuousAt.sup (hf : ContinuousAt f x) (hg : ContinuousAt g x) :
    ContinuousAt (fun a ↦ f a ⊔ g a) x :=
  hf.sup' hg


lemma ContinuousWithinAt.sup' (hf : ContinuousWithinAt f s x) (hg : ContinuousWithinAt g s x) :
    ContinuousWithinAt (f ⊔ g) s x :=
  hf.sup_nhds' hg


lemma ContinuousWithinAt.sup (hf : ContinuousWithinAt f s x) (hg : ContinuousWithinAt g s x) :
    ContinuousWithinAt (fun a ↦ f a ⊔ g a) s x :=
  hf.sup' hg


lemma ContinuousOn.sup' (hf : ContinuousOn f s) (hg : ContinuousOn g s) :
    ContinuousOn (f ⊔ g) s := fun x hx ↦
  (hf x hx).sup' (hg x hx)


@[fun_prop]
lemma ContinuousOn.sup (hf : ContinuousOn f s) (hg : ContinuousOn g s) :
    ContinuousOn (fun a ↦ f a ⊔ g a) s :=
  hf.sup' hg


lemma Continuous.sup' (hf : Continuous f) (hg : Continuous g) : Continuous (f ⊔ g) := hf.sup hg


lemma ContinuousAt.inf' (hf : ContinuousAt f x) (hg : ContinuousAt g x) :
    ContinuousAt (f ⊓ g) x :=
  hf.inf_nhds' hg


@[fun_prop]
lemma ContinuousAt.inf (hf : ContinuousAt f x) (hg : ContinuousAt g x) :
    ContinuousAt (fun a ↦ f a ⊓ g a) x :=
  hf.inf' hg


lemma ContinuousWithinAt.inf' (hf : ContinuousWithinAt f s x) (hg : ContinuousWithinAt g s x) :
    ContinuousWithinAt (f ⊓ g) s x :=
  hf.inf_nhds' hg


lemma ContinuousWithinAt.inf (hf : ContinuousWithinAt f s x) (hg : ContinuousWithinAt g s x) :
    ContinuousWithinAt (fun a ↦ f a ⊓ g a) s x :=
  hf.inf' hg


lemma ContinuousOn.inf' (hf : ContinuousOn f s) (hg : ContinuousOn g s) :
    ContinuousOn (f ⊓ g) s := fun x hx ↦
  (hf x hx).inf' (hg x hx)


@[fun_prop]
lemma ContinuousOn.inf (hf : ContinuousOn f s) (hg : ContinuousOn g s) :
    ContinuousOn (fun a ↦ f a ⊓ g a) s :=
  hf.inf' hg


lemma Continuous.inf' (hf : Continuous f) (hg : Continuous g) : Continuous (f ⊓ g) := hf.inf hg


lemma ContinuousAt.finset_sup'_apply (hne : s.Nonempty) (hs : ∀ i ∈ s, ContinuousAt (f i) x) :
    ContinuousAt (fun a ↦ s.sup' hne (f · a)) x :=
  Tendsto.finset_sup'_nhds_apply hne hs


lemma ContinuousAt.finset_sup' (hne : s.Nonempty) (hs : ∀ i ∈ s, ContinuousAt (f i) x) :
    ContinuousAt (s.sup' hne f) x := by
  /-
    L : Type u_1
    X : Type u_2
    inst✝³ : TopologicalSpace L
    inst✝² : TopologicalSpace X
    ι : Type u_3
    inst✝¹ : SemilatticeSup L
    inst✝ : ContinuousSup L
    s : Finset ι
    f : ι → X → L
    x : X
    hne : s.Nonempty
    hs : ∀ (i : ι), Membership.mem s i → ContinuousAt (f i) x
    ⊢ ContinuousAt (s.sup' hne f) x
  -/
  simpa only [← Finset.sup'_apply] using finset_sup'_apply hne hs
  /-
    🎉 no goals
  -/


lemma ContinuousWithinAt.finset_sup'_apply (hne : s.Nonempty)
    (hs : ∀ i ∈ s, ContinuousWithinAt (f i) t x) :
    ContinuousWithinAt (fun a ↦ s.sup' hne (f · a)) t x :=
  Tendsto.finset_sup'_nhds_apply hne hs


lemma ContinuousWithinAt.finset_sup' (hne : s.Nonempty)
    (hs : ∀ i ∈ s, ContinuousWithinAt (f i) t x) : ContinuousWithinAt (s.sup' hne f) t x := by
  /-
    L : Type u_1
    X : Type u_2
    inst✝³ : TopologicalSpace L
    inst✝² : TopologicalSpace X
    ι : Type u_3
    inst✝¹ : SemilatticeSup L
    inst✝ : ContinuousSup L
    s : Finset ι
    f : ι → X → L
    t : Set X
    x : X
    hne : s.Nonempty
    hs : ∀ (i : ι), Membership.mem s i → ContinuousWithinAt (f i) t x
    ⊢ ContinuousWithinAt (s.sup' hne f) t x
  -/
  simpa only [← Finset.sup'_apply] using finset_sup'_apply hne hs
  /-
    🎉 no goals
  -/


lemma ContinuousOn.finset_sup'_apply (hne : s.Nonempty) (hs : ∀ i ∈ s, ContinuousOn (f i) t) :
    ContinuousOn (fun a ↦ s.sup' hne (f · a)) t := fun x hx ↦
  ContinuousWithinAt.finset_sup'_apply hne fun i hi ↦ hs i hi x hx


lemma ContinuousOn.finset_sup' (hne : s.Nonempty) (hs : ∀ i ∈ s, ContinuousOn (f i) t) :
    ContinuousOn (s.sup' hne f) t := fun x hx ↦
  ContinuousWithinAt.finset_sup' hne fun i hi ↦ hs i hi x hx


lemma Continuous.finset_sup'_apply (hne : s.Nonempty) (hs : ∀ i ∈ s, Continuous (f i)) :
    Continuous (fun a ↦ s.sup' hne (f · a)) :=
  continuous_iff_continuousAt.2 fun _ ↦ ContinuousAt.finset_sup'_apply _ fun i hi ↦
    (hs i hi).continuousAt


lemma Continuous.finset_sup' (hne : s.Nonempty) (hs : ∀ i ∈ s, Continuous (f i)) :
    Continuous (s.sup' hne f) :=
  continuous_iff_continuousAt.2 fun _ ↦ ContinuousAt.finset_sup' _ fun i hi ↦ (hs i hi).continuousAt


lemma ContinuousAt.finset_sup_apply (hs : ∀ i ∈ s, ContinuousAt (f i) x) :
    ContinuousAt (fun a ↦ s.sup (f · a)) x :=
  Tendsto.finset_sup_nhds_apply hs


lemma ContinuousAt.finset_sup (hs : ∀ i ∈ s, ContinuousAt (f i) x) :
    ContinuousAt (s.sup f) x := by
  /-
    L : Type u_1
    X : Type u_2
    inst✝⁴ : TopologicalSpace L
    inst✝³ : TopologicalSpace X
    ι : Type u_3
    inst✝² : SemilatticeSup L
    inst✝¹ : OrderBot L
    inst✝ : ContinuousSup L
    s : Finset ι
    f : ι → X → L
    x : X
    hs : ∀ (i : ι), Membership.mem s i → ContinuousAt (f i) x
    ⊢ ContinuousAt (s.sup f) x
  -/
  simpa only [← Finset.sup_apply] using finset_sup_apply hs
  /-
    🎉 no goals
  -/


lemma ContinuousWithinAt.finset_sup_apply
    (hs : ∀ i ∈ s, ContinuousWithinAt (f i) t x) :
    ContinuousWithinAt (fun a ↦ s.sup (f · a)) t x :=
  Tendsto.finset_sup_nhds_apply hs


lemma ContinuousWithinAt.finset_sup
    (hs : ∀ i ∈ s, ContinuousWithinAt (f i) t x) : ContinuousWithinAt (s.sup f) t x := by
  /-
    L : Type u_1
    X : Type u_2
    inst✝⁴ : TopologicalSpace L
    inst✝³ : TopologicalSpace X
    ι : Type u_3
    inst✝² : SemilatticeSup L
    inst✝¹ : OrderBot L
    inst✝ : ContinuousSup L
    s : Finset ι
    f : ι → X → L
    t : Set X
    x : X
    hs : ∀ (i : ι), Membership.mem s i → ContinuousWithinAt (f i) t x
    ⊢ ContinuousWithinAt (s.sup f) t x
  -/
  simpa only [← Finset.sup_apply] using finset_sup_apply hs
  /-
    🎉 no goals
  -/


lemma ContinuousOn.finset_sup_apply (hs : ∀ i ∈ s, ContinuousOn (f i) t) :
    ContinuousOn (fun a ↦ s.sup (f · a)) t := fun x hx ↦
  ContinuousWithinAt.finset_sup_apply fun i hi ↦ hs i hi x hx


lemma ContinuousOn.finset_sup (hs : ∀ i ∈ s, ContinuousOn (f i) t) :
    ContinuousOn (s.sup f) t := fun x hx ↦
  ContinuousWithinAt.finset_sup fun i hi ↦ hs i hi x hx


lemma Continuous.finset_sup_apply (hs : ∀ i ∈ s, Continuous (f i)) :
    Continuous (fun a ↦ s.sup (f · a)) :=
  continuous_iff_continuousAt.2 fun _ ↦ ContinuousAt.finset_sup_apply fun i hi ↦
    (hs i hi).continuousAt


lemma Continuous.finset_sup (hs : ∀ i ∈ s, Continuous (f i)) : Continuous (s.sup f) :=
  continuous_iff_continuousAt.2 fun _ ↦ ContinuousAt.finset_sup fun i hi ↦ (hs i hi).continuousAt


lemma ContinuousAt.finset_inf'_apply (hne : s.Nonempty) (hs : ∀ i ∈ s, ContinuousAt (f i) x) :
    ContinuousAt (fun a ↦ s.inf' hne (f · a)) x :=
  Tendsto.finset_inf'_nhds_apply hne hs


lemma ContinuousAt.finset_inf' (hne : s.Nonempty) (hs : ∀ i ∈ s, ContinuousAt (f i) x) :
    ContinuousAt (s.inf' hne f) x := by
  /-
    L : Type u_1
    X : Type u_2
    inst✝³ : TopologicalSpace L
    inst✝² : TopologicalSpace X
    ι : Type u_3
    inst✝¹ : SemilatticeInf L
    inst✝ : ContinuousInf L
    s : Finset ι
    f : ι → X → L
    x : X
    hne : s.Nonempty
    hs : ∀ (i : ι), Membership.mem s i → ContinuousAt (f i) x
    ⊢ ContinuousAt (s.inf' hne f) x
  -/
  simpa only [← Finset.inf'_apply] using finset_inf'_apply hne hs
  /-
    🎉 no goals
  -/


lemma ContinuousWithinAt.finset_inf'_apply (hne : s.Nonempty)
    (hs : ∀ i ∈ s, ContinuousWithinAt (f i) t x) :
    ContinuousWithinAt (fun a ↦ s.inf' hne (f · a)) t x :=
  Tendsto.finset_inf'_nhds_apply hne hs


lemma ContinuousWithinAt.finset_inf' (hne : s.Nonempty)
    (hs : ∀ i ∈ s, ContinuousWithinAt (f i) t x) : ContinuousWithinAt (s.inf' hne f) t x := by
  /-
    L : Type u_1
    X : Type u_2
    inst✝³ : TopologicalSpace L
    inst✝² : TopologicalSpace X
    ι : Type u_3
    inst✝¹ : SemilatticeInf L
    inst✝ : ContinuousInf L
    s : Finset ι
    f : ι → X → L
    t : Set X
    x : X
    hne : s.Nonempty
    hs : ∀ (i : ι), Membership.mem s i → ContinuousWithinAt (f i) t x
    ⊢ ContinuousWithinAt (s.inf' hne f) t x
  -/
  simpa only [← Finset.inf'_apply] using finset_inf'_apply hne hs
  /-
    🎉 no goals
  -/


lemma ContinuousOn.finset_inf'_apply (hne : s.Nonempty) (hs : ∀ i ∈ s, ContinuousOn (f i) t) :
    ContinuousOn (fun a ↦ s.inf' hne (f · a)) t := fun x hx ↦
  ContinuousWithinAt.finset_inf'_apply hne fun i hi ↦ hs i hi x hx


lemma ContinuousOn.finset_inf' (hne : s.Nonempty) (hs : ∀ i ∈ s, ContinuousOn (f i) t) :
    ContinuousOn (s.inf' hne f) t := fun x hx ↦
  ContinuousWithinAt.finset_inf' hne fun i hi ↦ hs i hi x hx


lemma Continuous.finset_inf'_apply (hne : s.Nonempty) (hs : ∀ i ∈ s, Continuous (f i)) :
    Continuous (fun a ↦ s.inf' hne (f · a)) :=
  continuous_iff_continuousAt.2 fun _ ↦ ContinuousAt.finset_inf'_apply _ fun i hi ↦
    (hs i hi).continuousAt


lemma Continuous.finset_inf' (hne : s.Nonempty) (hs : ∀ i ∈ s, Continuous (f i)) :
    Continuous (s.inf' hne f) :=
  continuous_iff_continuousAt.2 fun _ ↦ ContinuousAt.finset_inf' _ fun i hi ↦ (hs i hi).continuousAt


lemma ContinuousAt.finset_inf_apply (hs : ∀ i ∈ s, ContinuousAt (f i) x) :
    ContinuousAt (fun a ↦ s.inf (f · a)) x :=
  Tendsto.finset_inf_nhds_apply hs


lemma ContinuousAt.finset_inf (hs : ∀ i ∈ s, ContinuousAt (f i) x) :
    ContinuousAt (s.inf f) x := by
  /-
    L : Type u_1
    X : Type u_2
    inst✝⁴ : TopologicalSpace L
    inst✝³ : TopologicalSpace X
    ι : Type u_3
    inst✝² : SemilatticeInf L
    inst✝¹ : OrderTop L
    inst✝ : ContinuousInf L
    s : Finset ι
    f : ι → X → L
    x : X
    hs : ∀ (i : ι), Membership.mem s i → ContinuousAt (f i) x
    ⊢ ContinuousAt (s.inf f) x
  -/
  simpa only [← Finset.inf_apply] using finset_inf_apply hs
  /-
    🎉 no goals
  -/


lemma ContinuousWithinAt.finset_inf_apply
    (hs : ∀ i ∈ s, ContinuousWithinAt (f i) t x) :
    ContinuousWithinAt (fun a ↦ s.inf (f · a)) t x :=
  Tendsto.finset_inf_nhds_apply hs


lemma ContinuousWithinAt.finset_inf
    (hs : ∀ i ∈ s, ContinuousWithinAt (f i) t x) : ContinuousWithinAt (s.inf f) t x := by
  /-
    L : Type u_1
    X : Type u_2
    inst✝⁴ : TopologicalSpace L
    inst✝³ : TopologicalSpace X
    ι : Type u_3
    inst✝² : SemilatticeInf L
    inst✝¹ : OrderTop L
    inst✝ : ContinuousInf L
    s : Finset ι
    f : ι → X → L
    t : Set X
    x : X
    hs : ∀ (i : ι), Membership.mem s i → ContinuousWithinAt (f i) t x
    ⊢ ContinuousWithinAt (s.inf f) t x
  -/
  simpa only [← Finset.inf_apply] using finset_inf_apply hs
  /-
    🎉 no goals
  -/


lemma ContinuousOn.finset_inf_apply (hs : ∀ i ∈ s, ContinuousOn (f i) t) :
    ContinuousOn (fun a ↦ s.inf (f · a)) t := fun x hx ↦
  ContinuousWithinAt.finset_inf_apply fun i hi ↦ hs i hi x hx


lemma ContinuousOn.finset_inf (hs : ∀ i ∈ s, ContinuousOn (f i) t) :
    ContinuousOn (s.inf f) t := fun x hx ↦
  ContinuousWithinAt.finset_inf fun i hi ↦ hs i hi x hx


lemma Continuous.finset_inf_apply (hs : ∀ i ∈ s, Continuous (f i)) :
    Continuous (fun a ↦ s.inf (f · a)) :=
  continuous_iff_continuousAt.2 fun _ ↦ ContinuousAt.finset_inf_apply fun i hi ↦
    (hs i hi).continuousAt


lemma Continuous.finset_inf (hs : ∀ i ∈ s, Continuous (f i)) : Continuous (s.inf f) :=
  continuous_iff_continuousAt.2 fun _ ↦ ContinuousAt.finset_inf fun i hi ↦ (hs i hi).continuousAt


