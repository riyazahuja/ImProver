/-- An extremally disconnected topological space is a space
in which the closure of every open set is open. -/
class ExtremallyDisconnected : Prop where
  /-- The closure of every open set is open. -/
  open_closure : ∀ U : Set X, IsOpen U → IsOpen (closure U)


theorem extremallyDisconnected_of_homeo {X Y : Type*} [TopologicalSpace X] [TopologicalSpace Y]
    [ExtremallyDisconnected X] (e : X ≃ₜ Y) : ExtremallyDisconnected Y where
  open_closure U hU := by
    /-
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : ExtremallyDisconnected X
      e : Homeomorph X Y
      U : Set Y
      hU : IsOpen U
      ⊢ IsOpen (closure U)
    -/
    rw [e.symm.isInducing.closure_eq_preimage_closure_image, Homeomorph.isOpen_preimage]
    /-
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : ExtremallyDisconnected X
      e : Homeomorph X Y
      U : Set Y
      hU : IsOpen U
      ⊢ IsOpen (closure (Set.image (⇑e.symm) U))
    -/
    exact ExtremallyDisconnected.open_closure _ (e.symm.isOpen_image.mpr hU)
    /-
      🎉 no goals
    -/


/-- Extremally disconnected spaces are totally separated. -/
instance [ExtremallyDisconnected X] [T2Space X] : TotallySeparatedSpace X :=
{ isTotallySeparated_univ := by
    /-
      X : Type u
      inst✝² : TopologicalSpace X
      inst✝¹ : ExtremallyDisconnected X
      inst✝ : T2Space X
      ⊢ IsTotallySeparated Set.univ
    -/
    intro x _ y _ hxy
    /-
      X : Type u
      inst✝² : TopologicalSpace X
      inst✝¹ : ExtremallyDisconnected X
      inst✝ : T2Space X
      x : X
      a✝¹ : Membership.mem Set.univ x
      y : X
      a✝ : Membership.mem Set.univ y
      hxy : Ne x y
      ⊢ Exists fun u => Exists fun v => And (IsOpen u) (And (IsOpen v) (And (Members …
    -/
    obtain ⟨U, V, hUV⟩ := T2Space.t2 hxy
    refine ⟨closure U, (closure U)ᶜ, ExtremallyDisconnected.open_closure U hUV.1,
      by simp only [isOpen_compl_iff, isClosed_closure], subset_closure hUV.2.2.1, ?_,
      by simp only [Set.union_compl_self, Set.subset_univ], disjoint_compl_right⟩
    /-
      case intro.intro
      X : Type u
      inst✝² : TopologicalSpace X
      inst✝¹ : ExtremallyDisconnected X
      inst✝ : T2Space X
      x : X
      a✝¹ : Membership.mem Set.univ x
      y : X
      a✝ : Membership.mem Set.univ y
      hxy : Ne x y
      U V : Set X
      hUV : And (IsOpen U) (And (IsOpen V) (And (Membership.mem U x) (And (Membershi …
      ⊢ Membership.mem (HasCompl.compl (closure U)) y
    -/
    rw [Set.mem_compl_iff, mem_closure_iff]
    /-
      case intro.intro
      X : Type u
      inst✝² : TopologicalSpace X
      inst✝¹ : ExtremallyDisconnected X
      inst✝ : T2Space X
      x : X
      a✝¹ : Membership.mem Set.univ x
      y : X
      a✝ : Membership.mem Set.univ y
      hxy : Ne x y
      U V : Set X
      hUV : And (IsOpen U) (And (IsOpen V) (And (Membership.mem U x) (And (Membershi …
      ⊢ Not (∀ (o : Set X), IsOpen o → Membership.mem o y → (Inter.inter o U).Nonemp …
    -/
    push_neg
    /-
      case intro.intro
      X : Type u
      inst✝² : TopologicalSpace X
      inst✝¹ : ExtremallyDisconnected X
      inst✝ : T2Space X
      x : X
      a✝¹ : Membership.mem Set.univ x
      y : X
      a✝ : Membership.mem Set.univ y
      hxy : Ne x y
      U V : Set X
      hUV : And (IsOpen U) (And (IsOpen V) (And (Membership.mem U x) (And (Membershi …
      ⊢ Exists fun o => And (IsOpen o) (And (Membership.mem o y) (Eq (Inter.inter o  …
    -/
    refine ⟨V, ⟨hUV.2.1, hUV.2.2.2.1, ?_⟩⟩
    /-
      case intro.intro
      X : Type u
      inst✝² : TopologicalSpace X
      inst✝¹ : ExtremallyDisconnected X
      inst✝ : T2Space X
      x : X
      a✝¹ : Membership.mem Set.univ x
      y : X
      a✝ : Membership.mem Set.univ y
      hxy : Ne x y
      U V : Set X
      hUV : And (IsOpen U) (And (IsOpen V) (And (Membership.mem U x) (And (Membershi …
      ⊢ Eq (Inter.inter V U) EmptyCollection.emptyCollection
    -/
    rw [← Set.disjoint_iff_inter_eq_empty, disjoint_comm]
    /-
      case intro.intro
      X : Type u
      inst✝² : TopologicalSpace X
      inst✝¹ : ExtremallyDisconnected X
      inst✝ : T2Space X
      x : X
      a✝¹ : Membership.mem Set.univ x
      y : X
      a✝ : Membership.mem Set.univ y
      hxy : Ne x y
      U V : Set X
      hUV : And (IsOpen U) (And (IsOpen V) (And (Membership.mem U x) (And (Membershi …
      ⊢ Disjoint U V
    -/
    exact hUV.2.2.2.2 }
    /-
      🎉 no goals
    -/


/-- The assertion `CompactT2.Projective` states that given continuous maps
`f : X → Z` and `g : Y → Z` with `g` surjective between `t_2`, compact topological spaces,
there exists a continuous lift `h : X → Y`, such that `f = g ∘ h`. -/
def CompactT2.Projective : Prop :=
  ∀ {Y Z : Type u} [TopologicalSpace Y] [TopologicalSpace Z],
    ∀ [CompactSpace Y] [T2Space Y] [CompactSpace Z] [T2Space Z],
      ∀ {f : X → Z} {g : Y → Z} (_ : Continuous f) (_ : Continuous g) (_ : Surjective g),
        ∃ h : X → Y, Continuous h ∧ g ∘ h = f


theorem StoneCech.projective [DiscreteTopology X] : CompactT2.Projective (StoneCech X) := by
  /-
    X : Type u
    inst✝¹ : TopologicalSpace X
    inst✝ : DiscreteTopology X
    ⊢ CompactT2.Projective (StoneCech X)
  -/
  intro Y Z _tsY _tsZ _csY _t2Y _csZ _csZ f g hf hg g_sur
  /-
    X : Type u
    inst✝¹ : TopologicalSpace X
    inst✝ : DiscreteTopology X
    Y Z : Type u
    _tsY : TopologicalSpace Y
    _tsZ : TopologicalSpace Z
    _csY : CompactSpace Y
    _t2Y : T2Space Y
    _csZ✝ : CompactSpace Z
    _csZ : T2Space Z
    f : StoneCech X → Z
    g : Y → Z
    hf : Continuous f
    hg : Continuous g
    g_sur : Function.Surjective g
    ⊢ Exists fun h => And (Continuous h) (Eq (Function.comp g h) f)
  -/
  let s : Z → Y := fun z => Classical.choose <| g_sur z
  /-
    X : Type u
    inst✝¹ : TopologicalSpace X
    inst✝ : DiscreteTopology X
    Y Z : Type u
    _tsY : TopologicalSpace Y
    _tsZ : TopologicalSpace Z
    _csY : CompactSpace Y
    _t2Y : T2Space Y
    _csZ✝ : CompactSpace Z
    _csZ : T2Space Z
    f : StoneCech X → Z
    g : Y → Z
    hf : Continuous f
    hg : Continuous g
    g_sur : Function.Surjective g
    s : Z → Y := fun z => Classical.choose ⋯
    ⊢ Exists fun h => And (Continuous h) (Eq (Function.comp g h) f)
  -/
  have hs : g ∘ s = id := funext fun z => Classical.choose_spec (g_sur z)
  /-
    X : Type u
    inst✝¹ : TopologicalSpace X
    inst✝ : DiscreteTopology X
    Y Z : Type u
    _tsY : TopologicalSpace Y
    _tsZ : TopologicalSpace Z
    _csY : CompactSpace Y
    _t2Y : T2Space Y
    _csZ✝ : CompactSpace Z
    _csZ : T2Space Z
    f : StoneCech X → Z
    g : Y → Z
    hf : Continuous f
    hg : Continuous g
    g_sur : Function.Surjective g
    s : Z → Y := fun z => Classical.choose ⋯
    hs : Eq (Function.comp g s) id
    ⊢ Exists fun h => And (Continuous h) (Eq (Function.comp g h) f)
  -/
  let t := s ∘ f ∘ stoneCechUnit
  /-
    X : Type u
    inst✝¹ : TopologicalSpace X
    inst✝ : DiscreteTopology X
    Y Z : Type u
    _tsY : TopologicalSpace Y
    _tsZ : TopologicalSpace Z
    _csY : CompactSpace Y
    _t2Y : T2Space Y
    _csZ✝ : CompactSpace Z
    _csZ : T2Space Z
    f : StoneCech X → Z
    g : Y → Z
    hf : Continuous f
    hg : Continuous g
    g_sur : Function.Surjective g
    s : Z → Y := fun z => Classical.choose ⋯
    hs : Eq (Function.comp g s) id
    t : X → Y := Function.comp s (Function.comp f stoneCechUnit)
    ⊢ Exists fun h => And (Continuous h) (Eq (Function.comp g h) f)
  -/
  have ht : Continuous t := continuous_of_discreteTopology
  /-
    X : Type u
    inst✝¹ : TopologicalSpace X
    inst✝ : DiscreteTopology X
    Y Z : Type u
    _tsY : TopologicalSpace Y
    _tsZ : TopologicalSpace Z
    _csY : CompactSpace Y
    _t2Y : T2Space Y
    _csZ✝ : CompactSpace Z
    _csZ : T2Space Z
    f : StoneCech X → Z
    g : Y → Z
    hf : Continuous f
    hg : Continuous g
    g_sur : Function.Surjective g
    s : Z → Y := fun z => Classical.choose ⋯
    hs : Eq (Function.comp g s) id
    t : X → Y := Function.comp s (Function.comp f stoneCechUnit)
    ht : Continuous t
    ⊢ Exists fun h => And (Continuous h) (Eq (Function.comp g h) f)
  -/
  let h : StoneCech X → Y := stoneCechExtend ht
  /-
    X : Type u
    inst✝¹ : TopologicalSpace X
    inst✝ : DiscreteTopology X
    Y Z : Type u
    _tsY : TopologicalSpace Y
    _tsZ : TopologicalSpace Z
    _csY : CompactSpace Y
    _t2Y : T2Space Y
    _csZ✝ : CompactSpace Z
    _csZ : T2Space Z
    f : StoneCech X → Z
    g : Y → Z
    hf : Continuous f
    hg : Continuous g
    g_sur : Function.Surjective g
    s : Z → Y := fun z => Classical.choose ⋯
    hs : Eq (Function.comp g s) id
    t : X → Y := Function.comp s (Function.comp f stoneCechUnit)
    ht : Continuous t
    h : StoneCech X → Y := stoneCechExtend ht
    ⊢ Exists fun h => And (Continuous h) (Eq (Function.comp g h) f)
  -/
  have hh : Continuous h := continuous_stoneCechExtend ht
  /-
    X : Type u
    inst✝¹ : TopologicalSpace X
    inst✝ : DiscreteTopology X
    Y Z : Type u
    _tsY : TopologicalSpace Y
    _tsZ : TopologicalSpace Z
    _csY : CompactSpace Y
    _t2Y : T2Space Y
    _csZ✝ : CompactSpace Z
    _csZ : T2Space Z
    f : StoneCech X → Z
    g : Y → Z
    hf : Continuous f
    hg : Continuous g
    g_sur : Function.Surjective g
    s : Z → Y := fun z => Classical.choose ⋯
    hs : Eq (Function.comp g s) id
    t : X → Y := Function.comp s (Function.comp f stoneCechUnit)
    ht : Continuous t
    h : StoneCech X → Y := stoneCechExtend ht
    hh : Continuous h
    ⊢ Exists fun h => And (Continuous h) (Eq (Function.comp g h) f)
  -/
  refine ⟨h, hh, denseRange_stoneCechUnit.equalizer (hg.comp hh) hf ?_⟩
  /-
    X : Type u
    inst✝¹ : TopologicalSpace X
    inst✝ : DiscreteTopology X
    Y Z : Type u
    _tsY : TopologicalSpace Y
    _tsZ : TopologicalSpace Z
    _csY : CompactSpace Y
    _t2Y : T2Space Y
    _csZ✝ : CompactSpace Z
    _csZ : T2Space Z
    f : StoneCech X → Z
    g : Y → Z
    hf : Continuous f
    hg : Continuous g
    g_sur : Function.Surjective g
    s : Z → Y := fun z => Classical.choose ⋯
    hs : Eq (Function.comp g s) id
    t : X → Y := Function.comp s (Function.comp f stoneCechUnit)
    ht : Continuous t
    h : StoneCech X → Y := stoneCechExtend ht
    hh : Continuous h
    ⊢ Eq (Function.comp (Function.comp g h) stoneCechUnit) (Function.comp f stoneC …
  -/
  rw [comp_assoc, stoneCechExtend_extends ht, ← comp_assoc, hs, id_comp]
  /-
    🎉 no goals
  -/


protected theorem CompactT2.Projective.extremallyDisconnected [CompactSpace X] [T2Space X]
    (h : CompactT2.Projective X) : ExtremallyDisconnected X := by
  /-
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : CompactSpace X
    inst✝ : T2Space X
    h : CompactT2.Projective X
    ⊢ ExtremallyDisconnected X
  -/
  refine { open_closure := fun U hU => ?_ }
  /-
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : CompactSpace X
    inst✝ : T2Space X
    h : CompactT2.Projective X
    U : Set X
    hU : IsOpen U
    ⊢ IsOpen (closure U)
  -/
  let Z₁ : Set (X × Bool) := Uᶜ ×ˢ {true}
  /-
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : CompactSpace X
    inst✝ : T2Space X
    h : CompactT2.Projective X
    U : Set X
    hU : IsOpen U
    Z₁ : Set (Prod X Bool) := SProd.sprod (HasCompl.compl U) (Singleton.singleton  …
    ⊢ IsOpen (closure U)
  -/
  let Z₂ : Set (X × Bool) := closure U ×ˢ {false}
  /-
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : CompactSpace X
    inst✝ : T2Space X
    h : CompactT2.Projective X
    U : Set X
    hU : IsOpen U
    Z₁ : Set (Prod X Bool) := SProd.sprod (HasCompl.compl U) (Singleton.singleton  …
    Z₂ : Set (Prod X Bool) := SProd.sprod (closure U) (Singleton.singleton Bool.fa …
    ⊢ IsOpen (closure U)
  -/
  let Z : Set (X × Bool) := Z₁ ∪ Z₂
  /-
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : CompactSpace X
    inst✝ : T2Space X
    h : CompactT2.Projective X
    U : Set X
    hU : IsOpen U
    Z₁ : Set (Prod X Bool) := SProd.sprod (HasCompl.compl U) (Singleton.singleton  …
    Z₂ : Set (Prod X Bool) := SProd.sprod (closure U) (Singleton.singleton Bool.fa …
    Z : Set (Prod X Bool) := Union.union Z₁ Z₂
    ⊢ IsOpen (closure U)
  -/
  have hZ₁₂ : Disjoint Z₁ Z₂ := disjoint_left.2 fun x hx₁ hx₂ => by cases hx₁.2.symm.trans hx₂.2
  /-
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : CompactSpace X
    inst✝ : T2Space X
    h : CompactT2.Projective X
    U : Set X
    hU : IsOpen U
    Z₁ : Set (Prod X Bool) := SProd.sprod (HasCompl.compl U) (Singleton.singleton  …
    Z₂ : Set (Prod X Bool) := SProd.sprod (closure U) (Singleton.singleton Bool.fa …
    Z : Set (Prod X Bool) := Union.union Z₁ Z₂
    hZ₁₂ : Disjoint Z₁ Z₂
    ⊢ IsOpen (closure U)
  -/
  have hZ₁ : IsClosed Z₁ := hU.isClosed_compl.prod (T1Space.t1 _)
  /-
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : CompactSpace X
    inst✝ : T2Space X
    h : CompactT2.Projective X
    U : Set X
    hU : IsOpen U
    Z₁ : Set (Prod X Bool) := SProd.sprod (HasCompl.compl U) (Singleton.singleton  …
    Z₂ : Set (Prod X Bool) := SProd.sprod (closure U) (Singleton.singleton Bool.fa …
    Z : Set (Prod X Bool) := Union.union Z₁ Z₂
    hZ₁₂ : Disjoint Z₁ Z₂
    hZ₁ : IsClosed Z₁
    ⊢ IsOpen (closure U)
  -/
  have hZ₂ : IsClosed Z₂ := isClosed_closure.prod (T1Space.t1 false)
  /-
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : CompactSpace X
    inst✝ : T2Space X
    h : CompactT2.Projective X
    U : Set X
    hU : IsOpen U
    Z₁ : Set (Prod X Bool) := SProd.sprod (HasCompl.compl U) (Singleton.singleton  …
    Z₂ : Set (Prod X Bool) := SProd.sprod (closure U) (Singleton.singleton Bool.fa …
    Z : Set (Prod X Bool) := Union.union Z₁ Z₂
    hZ₁₂ : Disjoint Z₁ Z₂
    hZ₁ : IsClosed Z₁
    hZ₂ : IsClosed Z₂
    ⊢ IsOpen (closure U)
  -/
  have hZ : IsClosed Z := hZ₁.union hZ₂
  /-
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : CompactSpace X
    inst✝ : T2Space X
    h : CompactT2.Projective X
    U : Set X
    hU : IsOpen U
    Z₁ : Set (Prod X Bool) := SProd.sprod (HasCompl.compl U) (Singleton.singleton  …
    Z₂ : Set (Prod X Bool) := SProd.sprod (closure U) (Singleton.singleton Bool.fa …
    Z : Set (Prod X Bool) := Union.union Z₁ Z₂
    hZ₁₂ : Disjoint Z₁ Z₂
    hZ₁ : IsClosed Z₁
    hZ₂ : IsClosed Z₂
    hZ : IsClosed Z
    ⊢ IsOpen (closure U)
  -/
  let f : Z → X := Prod.fst ∘ Subtype.val
  /-
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : CompactSpace X
    inst✝ : T2Space X
    h : CompactT2.Projective X
    U : Set X
    hU : IsOpen U
    Z₁ : Set (Prod X Bool) := SProd.sprod (HasCompl.compl U) (Singleton.singleton  …
    Z₂ : Set (Prod X Bool) := SProd.sprod (closure U) (Singleton.singleton Bool.fa …
    Z : Set (Prod X Bool) := Union.union Z₁ Z₂
    hZ₁₂ : Disjoint Z₁ Z₂
    hZ₁ : IsClosed Z₁
    hZ₂ : IsClosed Z₂
    hZ : IsClosed Z
    f : ↑Z → X := Function.comp Prod.fst Subtype.val
    ⊢ IsOpen (closure U)
  -/
  have f_cont : Continuous f := continuous_fst.comp continuous_subtype_val
  have f_sur : Surjective f := by
    intro x
    by_cases hx : x ∈ U
    · exact ⟨⟨(x, false), Or.inr ⟨subset_closure hx, mem_singleton _⟩⟩, rfl⟩
    · exact ⟨⟨(x, true), Or.inl ⟨hx, mem_singleton _⟩⟩, rfl⟩
  /-
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : CompactSpace X
    inst✝ : T2Space X
    h : CompactT2.Projective X
    U : Set X
    hU : IsOpen U
    Z₁ : Set (Prod X Bool) := SProd.sprod (HasCompl.compl U) (Singleton.singleton  …
    Z₂ : Set (Prod X Bool) := SProd.sprod (closure U) (Singleton.singleton Bool.fa …
    Z : Set (Prod X Bool) := Union.union Z₁ Z₂
    hZ₁₂ : Disjoint Z₁ Z₂
    hZ₁ : IsClosed Z₁
    hZ₂ : IsClosed Z₂
    hZ : IsClosed Z
    f : ↑Z → X := Function.comp Prod.fst Subtype.val
    f_cont : Continuous f
    f_sur : Function.Surjective f
    ⊢ IsOpen (closure U)
  -/
  haveI : CompactSpace Z := isCompact_iff_compactSpace.mp hZ.isCompact
  /-
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : CompactSpace X
    inst✝ : T2Space X
    h : CompactT2.Projective X
    U : Set X
    hU : IsOpen U
    Z₁ : Set (Prod X Bool) := SProd.sprod (HasCompl.compl U) (Singleton.singleton  …
    Z₂ : Set (Prod X Bool) := SProd.sprod (closure U) (Singleton.singleton Bool.fa …
    Z : Set (Prod X Bool) := Union.union Z₁ Z₂
    hZ₁₂ : Disjoint Z₁ Z₂
    hZ₁ : IsClosed Z₁
    hZ₂ : IsClosed Z₂
    hZ : IsClosed Z
    f : ↑Z → X := Function.comp Prod.fst Subtype.val
    f_cont : Continuous f
    f_sur : Function.Surjective f
    this : CompactSpace ↑Z
    ⊢ IsOpen (closure U)
  -/
  obtain ⟨g, hg, g_sec⟩ := h continuous_id f_cont f_sur
  /-
    case intro.intro
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : CompactSpace X
    inst✝ : T2Space X
    h : CompactT2.Projective X
    U : Set X
    hU : IsOpen U
    Z₁ : Set (Prod X Bool) := SProd.sprod (HasCompl.compl U) (Singleton.singleton  …
    Z₂ : Set (Prod X Bool) := SProd.sprod (closure U) (Singleton.singleton Bool.fa …
    Z : Set (Prod X Bool) := Union.union Z₁ Z₂
    hZ₁₂ : Disjoint Z₁ Z₂
    hZ₁ : IsClosed Z₁
    hZ₂ : IsClosed Z₂
    hZ : IsClosed Z
    f : ↑Z → X := Function.comp Prod.fst Subtype.val
    f_cont : Continuous f
    f_sur : Function.Surjective f
    this : CompactSpace ↑Z
    g : X → ↑Z
    hg : Continuous g
    g_sec : Eq (Function.comp f g) id
    ⊢ IsOpen (closure U)
  -/
  let φ := Subtype.val ∘ g
  /-
    case intro.intro
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : CompactSpace X
    inst✝ : T2Space X
    h : CompactT2.Projective X
    U : Set X
    hU : IsOpen U
    Z₁ : Set (Prod X Bool) := SProd.sprod (HasCompl.compl U) (Singleton.singleton  …
    Z₂ : Set (Prod X Bool) := SProd.sprod (closure U) (Singleton.singleton Bool.fa …
    Z : Set (Prod X Bool) := Union.union Z₁ Z₂
    hZ₁₂ : Disjoint Z₁ Z₂
    hZ₁ : IsClosed Z₁
    hZ₂ : IsClosed Z₂
    hZ : IsClosed Z
    f : ↑Z → X := Function.comp Prod.fst Subtype.val
    f_cont : Continuous f
    f_sur : Function.Surjective f
    this : CompactSpace ↑Z
    g : X → ↑Z
    hg : Continuous g
    g_sec : Eq (Function.comp f g) id
    φ : X → Prod X Bool := Function.comp Subtype.val g
    ⊢ IsOpen (closure U)
  -/
  have hφ : Continuous φ := continuous_subtype_val.comp hg
  /-
    case intro.intro
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : CompactSpace X
    inst✝ : T2Space X
    h : CompactT2.Projective X
    U : Set X
    hU : IsOpen U
    Z₁ : Set (Prod X Bool) := SProd.sprod (HasCompl.compl U) (Singleton.singleton  …
    Z₂ : Set (Prod X Bool) := SProd.sprod (closure U) (Singleton.singleton Bool.fa …
    Z : Set (Prod X Bool) := Union.union Z₁ Z₂
    hZ₁₂ : Disjoint Z₁ Z₂
    hZ₁ : IsClosed Z₁
    hZ₂ : IsClosed Z₂
    hZ : IsClosed Z
    f : ↑Z → X := Function.comp Prod.fst Subtype.val
    f_cont : Continuous f
    f_sur : Function.Surjective f
    this : CompactSpace ↑Z
    g : X → ↑Z
    hg : Continuous g
    g_sec : Eq (Function.comp f g) id
    φ : X → Prod X Bool := Function.comp Subtype.val g
    hφ : Continuous φ
    ⊢ IsOpen (closure U)
  -/
  have hφ₁ : ∀ x, (φ x).1 = x := congr_fun g_sec
  suffices closure U = φ ⁻¹' Z₂ by
    rw [this, preimage_comp, ← isClosed_compl_iff, ← preimage_compl,
      ← preimage_subtype_coe_eq_compl Subset.rfl]
    · exact hZ₁.preimage hφ
    · rw [hZ₁₂.inter_eq, inter_empty]
  /-
    case intro.intro
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : CompactSpace X
    inst✝ : T2Space X
    h : CompactT2.Projective X
    U : Set X
    hU : IsOpen U
    Z₁ : Set (Prod X Bool) := SProd.sprod (HasCompl.compl U) (Singleton.singleton  …
    Z₂ : Set (Prod X Bool) := SProd.sprod (closure U) (Singleton.singleton Bool.fa …
    Z : Set (Prod X Bool) := Union.union Z₁ Z₂
    hZ₁₂ : Disjoint Z₁ Z₂
    hZ₁ : IsClosed Z₁
    hZ₂ : IsClosed Z₂
    hZ : IsClosed Z
    f : ↑Z → X := Function.comp Prod.fst Subtype.val
    f_cont : Continuous f
    f_sur : Function.Surjective f
    this : CompactSpace ↑Z
    g : X → ↑Z
    hg : Continuous g
    g_sec : Eq (Function.comp f g) id
    φ : X → Prod X Bool := Function.comp Subtype.val g
    hφ : Continuous φ
    hφ₁ : ∀ (x : X), Eq (φ x).1 x
    ⊢ Eq (closure U) (Set.preimage φ Z₂)
  -/
  refine (closure_minimal ?_ <| hZ₂.preimage hφ).antisymm fun x hx => ?_
    /-
      case intro.intro.refine_1
      X : Type u
      inst✝² : TopologicalSpace X
      inst✝¹ : CompactSpace X
      inst✝ : T2Space X
      h : CompactT2.Projective X
      U : Set X
      hU : IsOpen U
      Z₁ : Set (Prod X Bool) := SProd.sprod (HasCompl.compl U) (Singleton.singleton  …
      Z₂ : Set (Prod X Bool) := SProd.sprod (closure U) (Singleton.singleton Bool.fa …
      Z : Set (Prod X Bool) := Union.union Z₁ Z₂
      hZ₁₂ : Disjoint Z₁ Z₂
      hZ₁ : IsClosed Z₁
      hZ₂ : IsClosed Z₂
      hZ : IsClosed Z
      f : ↑Z → X := Function.comp Prod.fst Subtype.val
      f_cont : Continuous f
      f_sur : Function.Surjective f
      this : CompactSpace ↑Z
      g : X → ↑Z
      hg : Continuous g
      g_sec : Eq (Function.comp f g) id
      φ : X → Prod X Bool := Function.comp Subtype.val g
      hφ : Continuous φ
      hφ₁ : ∀ (x : X), Eq (φ x).1 x
      ⊢ HasSubset.Subset U (Set.preimage φ Z₂)
    -/
  · intro x hx
    /-
      case intro.intro.refine_1
      X : Type u
      inst✝² : TopologicalSpace X
      inst✝¹ : CompactSpace X
      inst✝ : T2Space X
      h : CompactT2.Projective X
      U : Set X
      hU : IsOpen U
      Z₁ : Set (Prod X Bool) := SProd.sprod (HasCompl.compl U) (Singleton.singleton  …
      Z₂ : Set (Prod X Bool) := SProd.sprod (closure U) (Singleton.singleton Bool.fa …
      Z : Set (Prod X Bool) := Union.union Z₁ Z₂
      hZ₁₂ : Disjoint Z₁ Z₂
      hZ₁ : IsClosed Z₁
      hZ₂ : IsClosed Z₂
      hZ : IsClosed Z
      f : ↑Z → X := Function.comp Prod.fst Subtype.val
      f_cont : Continuous f
      f_sur : Function.Surjective f
      this : CompactSpace ↑Z
      g : X → ↑Z
      hg : Continuous g
      g_sec : Eq (Function.comp f g) id
      φ : X → Prod X Bool := Function.comp Subtype.val g
      hφ : Continuous φ
      hφ₁ : ∀ (x : X), Eq (φ x).1 x
      x : X
      hx : Membership.mem U x
      ⊢ Membership.mem (Set.preimage φ Z₂) x
    -/
    have : φ x ∈ Z₁ ∪ Z₂ := (g x).2
    -- Porting note: Originally `simpa [hx, hφ₁] using this`
    /-
      case intro.intro.refine_1
      X : Type u
      inst✝² : TopologicalSpace X
      inst✝¹ : CompactSpace X
      inst✝ : T2Space X
      h : CompactT2.Projective X
      U : Set X
      hU : IsOpen U
      Z₁ : Set (Prod X Bool) := SProd.sprod (HasCompl.compl U) (Singleton.singleton  …
      Z₂ : Set (Prod X Bool) := SProd.sprod (closure U) (Singleton.singleton Bool.fa …
      Z : Set (Prod X Bool) := Union.union Z₁ Z₂
      hZ₁₂ : Disjoint Z₁ Z₂
      hZ₁ : IsClosed Z₁
      hZ₂ : IsClosed Z₂
      hZ : IsClosed Z
      f : ↑Z → X := Function.comp Prod.fst Subtype.val
      f_cont : Continuous f
      f_sur : Function.Surjective f
      this✝ : CompactSpace ↑Z
      g : X → ↑Z
      hg : Continuous g
      g_sec : Eq (Function.comp f g) id
      φ : X → Prod X Bool := Function.comp Subtype.val g
      hφ : Continuous φ
      hφ₁ : ∀ (x : X), Eq (φ x).1 x
      x : X
      hx : Membership.mem U x
      this : Membership.mem (Union.union Z₁ Z₂) (φ x)
      ⊢ Membership.mem (Set.preimage φ Z₂) x
    -/
    cases' this with hφ hφ
      /-
        case intro.intro.refine_1.inl
        X : Type u
        inst✝² : TopologicalSpace X
        inst✝¹ : CompactSpace X
        inst✝ : T2Space X
        h : CompactT2.Projective X
        U : Set X
        hU : IsOpen U
        Z₁ : Set (Prod X Bool) := SProd.sprod (HasCompl.compl U) (Singleton.singleton  …
        Z₂ : Set (Prod X Bool) := SProd.sprod (closure U) (Singleton.singleton Bool.fa …
        Z : Set (Prod X Bool) := Union.union Z₁ Z₂
        hZ₁₂ : Disjoint Z₁ Z₂
        hZ₁ : IsClosed Z₁
        hZ₂ : IsClosed Z₂
        hZ : IsClosed Z
        f : ↑Z → X := Function.comp Prod.fst Subtype.val
        f_cont : Continuous f
        f_sur : Function.Surjective f
        this : CompactSpace ↑Z
        g : X → ↑Z
        hg : Continuous g
        g_sec : Eq (Function.comp f g) id
        φ : X → Prod X Bool := Function.comp Subtype.val g
        hφ✝ : Continuous φ
        hφ₁ : ∀ (x : X), Eq (φ x).1 x
        x : X
        hx : Membership.mem U x
        hφ : Membership.mem Z₁ (φ x)
        ⊢ Membership.mem (Set.preimage φ Z₂) x
      -/
    · exact ((hφ₁ x ▸ hφ.1) hx).elim
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.refine_1.inr
        X : Type u
        inst✝² : TopologicalSpace X
        inst✝¹ : CompactSpace X
        inst✝ : T2Space X
        h : CompactT2.Projective X
        U : Set X
        hU : IsOpen U
        Z₁ : Set (Prod X Bool) := SProd.sprod (HasCompl.compl U) (Singleton.singleton  …
        Z₂ : Set (Prod X Bool) := SProd.sprod (closure U) (Singleton.singleton Bool.fa …
        Z : Set (Prod X Bool) := Union.union Z₁ Z₂
        hZ₁₂ : Disjoint Z₁ Z₂
        hZ₁ : IsClosed Z₁
        hZ₂ : IsClosed Z₂
        hZ : IsClosed Z
        f : ↑Z → X := Function.comp Prod.fst Subtype.val
        f_cont : Continuous f
        f_sur : Function.Surjective f
        this : CompactSpace ↑Z
        g : X → ↑Z
        hg : Continuous g
        g_sec : Eq (Function.comp f g) id
        φ : X → Prod X Bool := Function.comp Subtype.val g
        hφ✝ : Continuous φ
        hφ₁ : ∀ (x : X), Eq (φ x).1 x
        x : X
        hx : Membership.mem U x
        hφ : Membership.mem Z₂ (φ x)
        ⊢ Membership.mem (Set.preimage φ Z₂) x
      -/
    · exact hφ
      /-
        🎉 no goals
      -/
    /-
      case intro.intro.refine_2
      X : Type u
      inst✝² : TopologicalSpace X
      inst✝¹ : CompactSpace X
      inst✝ : T2Space X
      h : CompactT2.Projective X
      U : Set X
      hU : IsOpen U
      Z₁ : Set (Prod X Bool) := SProd.sprod (HasCompl.compl U) (Singleton.singleton  …
      Z₂ : Set (Prod X Bool) := SProd.sprod (closure U) (Singleton.singleton Bool.fa …
      Z : Set (Prod X Bool) := Union.union Z₁ Z₂
      hZ₁₂ : Disjoint Z₁ Z₂
      hZ₁ : IsClosed Z₁
      hZ₂ : IsClosed Z₂
      hZ : IsClosed Z
      f : ↑Z → X := Function.comp Prod.fst Subtype.val
      f_cont : Continuous f
      f_sur : Function.Surjective f
      this : CompactSpace ↑Z
      g : X → ↑Z
      hg : Continuous g
      g_sec : Eq (Function.comp f g) id
      φ : X → Prod X Bool := Function.comp Subtype.val g
      hφ : Continuous φ
      hφ₁ : ∀ (x : X), Eq (φ x).1 x
      x : X
      hx : Membership.mem (Set.preimage φ Z₂) x
      ⊢ Membership.mem (closure U) x
    -/
  · rw [← hφ₁ x]
    /-
      case intro.intro.refine_2
      X : Type u
      inst✝² : TopologicalSpace X
      inst✝¹ : CompactSpace X
      inst✝ : T2Space X
      h : CompactT2.Projective X
      U : Set X
      hU : IsOpen U
      Z₁ : Set (Prod X Bool) := SProd.sprod (HasCompl.compl U) (Singleton.singleton  …
      Z₂ : Set (Prod X Bool) := SProd.sprod (closure U) (Singleton.singleton Bool.fa …
      Z : Set (Prod X Bool) := Union.union Z₁ Z₂
      hZ₁₂ : Disjoint Z₁ Z₂
      hZ₁ : IsClosed Z₁
      hZ₂ : IsClosed Z₂
      hZ : IsClosed Z
      f : ↑Z → X := Function.comp Prod.fst Subtype.val
      f_cont : Continuous f
      f_sur : Function.Surjective f
      this : CompactSpace ↑Z
      g : X → ↑Z
      hg : Continuous g
      g_sec : Eq (Function.comp f g) id
      φ : X → Prod X Bool := Function.comp Subtype.val g
      hφ : Continuous φ
      hφ₁ : ∀ (x : X), Eq (φ x).1 x
      x : X
      hx : Membership.mem (Set.preimage φ Z₂) x
      ⊢ Membership.mem (closure U) (φ x).1
    -/
    exact hx.1
    /-
      🎉 no goals
    -/


/-- Lemma 2.4 in [Gleason, *Projective topological spaces*][gleason1958]:
a continuous surjection $\pi$ from a compact space $D$ to a Fréchet space $A$ restricts to
a compact subset $E$ of $D$, such that $\pi$ maps $E$ onto $A$ and satisfies the
"Zorn subset condition", where $\pi(E_0) \ne A$ for any proper closed subset $E_0 \subsetneq E$. -/
lemma exists_compact_surjective_zorn_subset [T1Space A] [CompactSpace D] {π : D → A}
    (π_cont : Continuous π) (π_surj : π.Surjective) : ∃ E : Set D, CompactSpace E ∧ π '' E = univ ∧
    ∀ E₀ : Set E, E₀ ≠ univ → IsClosed E₀ → E.restrict π '' E₀ ≠ univ := by
  -- suffices to apply Zorn's lemma on the subsets of $D$ that are closed and mapped onto $A$
  /-
    A D : Type u
    inst✝³ : TopologicalSpace A
    inst✝² : TopologicalSpace D
    inst✝¹ : T1Space A
    inst✝ : CompactSpace D
    π : D → A
    π_cont : Continuous π
    π_surj : Function.Surjective π
    ⊢ Exists fun E => And (CompactSpace ↑E) (And (Eq (Set.image π E) Set.univ) (∀  …
  -/
  let S : Set <| Set D := {E : Set D | IsClosed E ∧ π '' E = univ}
  suffices ∀ (C : Set <| Set D) (_ : C ⊆ S) (_ : IsChain (· ⊆ ·) C), ∃ s ∈ S, ∀ c ∈ C, s ⊆ c by
    rcases zorn_superset S this with ⟨E, E_min⟩
    obtain ⟨E_closed, E_surj⟩ := E_min.prop
    refine ⟨E, isCompact_iff_compactSpace.mp E_closed.isCompact, E_surj, ?_⟩
    intro E₀ E₀_min E₀_closed
    contrapose! E₀_min
    exact eq_univ_of_image_val_eq <|
      E_min.eq_of_subset ⟨E₀_closed.trans E_closed, image_image_val_eq_restrict_image ▸ E₀_min⟩
        image_val_subset
  -- suffices to prove intersection of chain is minimal
  /-
    A D : Type u
    inst✝³ : TopologicalSpace A
    inst✝² : TopologicalSpace D
    inst✝¹ : T1Space A
    inst✝ : CompactSpace D
    π : D → A
    π_cont : Continuous π
    π_surj : Function.Surjective π
    S : Set (Set D) := setOf fun E => And (IsClosed E) (Eq (Set.image π E) Set.univ)
    ⊢ ∀ (C : Set (Set D)), HasSubset.Subset C S → IsChain (fun x1 x2 => HasSubset. …
  -/
  intro C C_sub C_chain
  -- prove intersection of chain is closed
  refine ⟨iInter (fun c : C => c), ⟨isClosed_iInter fun ⟨_, h⟩ => (C_sub h).left, ?_⟩,
    fun c hc _ h => mem_iInter.mp h ⟨c, hc⟩⟩
  -- prove intersection of chain is mapped onto $A$
  /-
    A D : Type u
    inst✝³ : TopologicalSpace A
    inst✝² : TopologicalSpace D
    inst✝¹ : T1Space A
    inst✝ : CompactSpace D
    π : D → A
    π_cont : Continuous π
    π_surj : Function.Surjective π
    S : Set (Set D) := setOf fun E => And (IsClosed E) (Eq (Set.image π E) Set.univ)
    C : Set (Set D)
    C_sub : HasSubset.Subset C S
    C_chain : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) C
    ⊢ Eq (Set.image π (Set.iInter fun c => ↑c)) Set.univ
  -/
  by_cases hC : Nonempty C
    /-
      case pos
      A D : Type u
      inst✝³ : TopologicalSpace A
      inst✝² : TopologicalSpace D
      inst✝¹ : T1Space A
      inst✝ : CompactSpace D
      π : D → A
      π_cont : Continuous π
      π_surj : Function.Surjective π
      S : Set (Set D) := setOf fun E => And (IsClosed E) (Eq (Set.image π E) Set.univ)
      C : Set (Set D)
      C_sub : HasSubset.Subset C S
      C_chain : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) C
      hC : Nonempty ↑C
      ⊢ Eq (Set.image π (Set.iInter fun c => ↑c)) Set.univ
    -/
  · refine eq_univ_of_forall fun a => inter_nonempty_iff_exists_left.mp ?_
    -- apply Cantor's intersection theorem
    refine iInter_inter (ι := C) (π ⁻¹' {a}) _ ▸
      IsCompact.nonempty_iInter_of_directed_nonempty_isCompact_isClosed _
      ?_ (fun c => ?_) (fun c => IsClosed.isCompact ?_) (fun c => ?_)
      /-
        case pos.refine_1
        A D : Type u
        inst✝³ : TopologicalSpace A
        inst✝² : TopologicalSpace D
        inst✝¹ : T1Space A
        inst✝ : CompactSpace D
        π : D → A
        π_cont : Continuous π
        π_surj : Function.Surjective π
        S : Set (Set D) := setOf fun E => And (IsClosed E) (Eq (Set.image π E) Set.univ)
        C : Set (Set D)
        C_sub : HasSubset.Subset C S
        C_chain : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) C
        hC : Nonempty ↑C
        a : A
        ⊢ Directed (fun x1 x2 => Superset x1 x2) fun i => Inter.inter (↑i) (Set.preima …
      -/
    · replace C_chain : IsChain (· ⊇ ·) C := C_chain.symm
      /-
        case pos.refine_1
        A D : Type u
        inst✝³ : TopologicalSpace A
        inst✝² : TopologicalSpace D
        inst✝¹ : T1Space A
        inst✝ : CompactSpace D
        π : D → A
        π_cont : Continuous π
        π_surj : Function.Surjective π
        S : Set (Set D) := setOf fun E => And (IsClosed E) (Eq (Set.image π E) Set.univ)
        C : Set (Set D)
        C_sub : HasSubset.Subset C S
        hC : Nonempty ↑C
        a : A
        C_chain : IsChain (fun x1 x2 => Superset x1 x2) C
        ⊢ Directed (fun x1 x2 => Superset x1 x2) fun i => Inter.inter (↑i) (Set.preima …
      -/
      have : ∀ s t : Set D, s ⊇ t → _ ⊇ _ := fun _ _ => inter_subset_inter_left <| π ⁻¹' {a}
      /-
        case pos.refine_1
        A D : Type u
        inst✝³ : TopologicalSpace A
        inst✝² : TopologicalSpace D
        inst✝¹ : T1Space A
        inst✝ : CompactSpace D
        π : D → A
        π_cont : Continuous π
        π_surj : Function.Surjective π
        S : Set (Set D) := setOf fun E => And (IsClosed E) (Eq (Set.image π E) Set.univ)
        C : Set (Set D)
        C_sub : HasSubset.Subset C S
        hC : Nonempty ↑C
        a : A
        C_chain : IsChain (fun x1 x2 => Superset x1 x2) C
        this : ∀ (s t : Set D), Superset s t → Superset (Inter.inter s (Set.preimage π …
        ⊢ Directed (fun x1 x2 => Superset x1 x2) fun i => Inter.inter (↑i) (Set.preima …
      -/
      exact (directedOn_iff_directed.mp C_chain.directedOn).mono_comp (· ⊇ ·) this
      /-
        🎉 no goals
      -/
      /-
        case pos.refine_2
        A D : Type u
        inst✝³ : TopologicalSpace A
        inst✝² : TopologicalSpace D
        inst✝¹ : T1Space A
        inst✝ : CompactSpace D
        π : D → A
        π_cont : Continuous π
        π_surj : Function.Surjective π
        S : Set (Set D) := setOf fun E => And (IsClosed E) (Eq (Set.image π E) Set.univ)
        C : Set (Set D)
        C_sub : HasSubset.Subset C S
        C_chain : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) C
        hC : Nonempty ↑C
        a : A
        c : ↑C
        ⊢ (Inter.inter (↑c) (Set.preimage π (Singleton.singleton a))).Nonempty
      -/
    · rw [← image_inter_nonempty_iff, (C_sub c.mem).right, univ_inter]
      /-
        case pos.refine_2
        A D : Type u
        inst✝³ : TopologicalSpace A
        inst✝² : TopologicalSpace D
        inst✝¹ : T1Space A
        inst✝ : CompactSpace D
        π : D → A
        π_cont : Continuous π
        π_surj : Function.Surjective π
        S : Set (Set D) := setOf fun E => And (IsClosed E) (Eq (Set.image π E) Set.univ)
        C : Set (Set D)
        C_sub : HasSubset.Subset C S
        C_chain : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) C
        hC : Nonempty ↑C
        a : A
        c : ↑C
        ⊢ (Singleton.singleton a).Nonempty
      -/
      exact singleton_nonempty a
      /-
        🎉 no goals
      -/
    /-
      case pos.refine_3
      A D : Type u
      inst✝³ : TopologicalSpace A
      inst✝² : TopologicalSpace D
      inst✝¹ : T1Space A
      inst✝ : CompactSpace D
      π : D → A
      π_cont : Continuous π
      π_surj : Function.Surjective π
      S : Set (Set D) := setOf fun E => And (IsClosed E) (Eq (Set.image π E) Set.univ)
      C : Set (Set D)
      C_sub : HasSubset.Subset C S
      C_chain : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) C
      hC : Nonempty ↑C
      a : A
      c : ↑C
      ⊢ IsClosed (Inter.inter (↑c) (Set.preimage π (Singleton.singleton a)))
    -/
    all_goals exact (C_sub c.mem).left.inter <| (T1Space.t1 a).preimage π_cont
    /-
      🎉 no goals
    -/
    /-
      case neg
      A D : Type u
      inst✝³ : TopologicalSpace A
      inst✝² : TopologicalSpace D
      inst✝¹ : T1Space A
      inst✝ : CompactSpace D
      π : D → A
      π_cont : Continuous π
      π_surj : Function.Surjective π
      S : Set (Set D) := setOf fun E => And (IsClosed E) (Eq (Set.image π E) Set.univ)
      C : Set (Set D)
      C_sub : HasSubset.Subset C S
      C_chain : IsChain (fun x1 x2 => HasSubset.Subset x1 x2) C
      hC : Not (Nonempty ↑C)
      ⊢ Eq (Set.image π (Set.iInter fun c => ↑c)) Set.univ
    -/
  · rw [@iInter_of_empty _ _ <| not_nonempty_iff.mp hC, image_univ_of_surjective π_surj]
    /-
      🎉 no goals
    -/


/-- Lemma 2.1 in [Gleason, *Projective topological spaces*][gleason1958]:
if $\rho$ is a continuous surjection from a topological space $E$ to a topological space $A$
satisfying the "Zorn subset condition", then $\rho(G)$ is contained in
the closure of $A \setminus \rho(E \setminus G)$ for any open set $G$ of $E$. -/
lemma image_subset_closure_compl_image_compl_of_isOpen {ρ : E → A} (ρ_cont : Continuous ρ)
    (ρ_surj : ρ.Surjective) (zorn_subset : ∀ E₀ : Set E, E₀ ≠ univ → IsClosed E₀ → ρ '' E₀ ≠ univ)
    {G : Set E} (hG : IsOpen G) : ρ '' G ⊆ closure ((ρ '' Gᶜ)ᶜ) := by
  -- suffices to prove for nonempty $G$
  /-
    A E : Type u
    inst✝¹ : TopologicalSpace A
    inst✝ : TopologicalSpace E
    ρ : E → A
    ρ_cont : Continuous ρ
    ρ_surj : Function.Surjective ρ
    zorn_subset : ∀ (E₀ : Set E), Ne E₀ Set.univ → IsClosed E₀ → Ne (Set.image ρ E …
    G : Set E
    hG : IsOpen G
    ⊢ HasSubset.Subset (Set.image ρ G) (closure (HasCompl.compl (Set.image ρ (HasC …
  -/
  by_cases G_empty : G = ∅
    /-
      case pos
      A E : Type u
      inst✝¹ : TopologicalSpace A
      inst✝ : TopologicalSpace E
      ρ : E → A
      ρ_cont : Continuous ρ
      ρ_surj : Function.Surjective ρ
      zorn_subset : ∀ (E₀ : Set E), Ne E₀ Set.univ → IsClosed E₀ → Ne (Set.image ρ E …
      G : Set E
      hG : IsOpen G
      G_empty : Eq G EmptyCollection.emptyCollection
      ⊢ HasSubset.Subset (Set.image ρ G) (closure (HasCompl.compl (Set.image ρ (HasC …
    -/
  · simpa only [G_empty, image_empty] using empty_subset _
    /-
      🎉 no goals
    -/
  · -- let $a \in \rho(G)$
    /-
      case neg
      A E : Type u
      inst✝¹ : TopologicalSpace A
      inst✝ : TopologicalSpace E
      ρ : E → A
      ρ_cont : Continuous ρ
      ρ_surj : Function.Surjective ρ
      zorn_subset : ∀ (E₀ : Set E), Ne E₀ Set.univ → IsClosed E₀ → Ne (Set.image ρ E …
      G : Set E
      hG : IsOpen G
      G_empty : Not (Eq G EmptyCollection.emptyCollection)
      ⊢ HasSubset.Subset (Set.image ρ G) (closure (HasCompl.compl (Set.image ρ (HasC …
    -/
    intro a ha
    /-
      case neg
      A E : Type u
      inst✝¹ : TopologicalSpace A
      inst✝ : TopologicalSpace E
      ρ : E → A
      ρ_cont : Continuous ρ
      ρ_surj : Function.Surjective ρ
      zorn_subset : ∀ (E₀ : Set E), Ne E₀ Set.univ → IsClosed E₀ → Ne (Set.image ρ E …
      G : Set E
      hG : IsOpen G
      G_empty : Not (Eq G EmptyCollection.emptyCollection)
      a : A
      ha : Membership.mem (Set.image ρ G) a
      ⊢ Membership.mem (closure (HasCompl.compl (Set.image ρ (HasCompl.compl G)))) a
    -/
    rw [mem_closure_iff]
    -- let $N$ be a neighbourhood of $a$
    /-
      case neg
      A E : Type u
      inst✝¹ : TopologicalSpace A
      inst✝ : TopologicalSpace E
      ρ : E → A
      ρ_cont : Continuous ρ
      ρ_surj : Function.Surjective ρ
      zorn_subset : ∀ (E₀ : Set E), Ne E₀ Set.univ → IsClosed E₀ → Ne (Set.image ρ E …
      G : Set E
      hG : IsOpen G
      G_empty : Not (Eq G EmptyCollection.emptyCollection)
      a : A
      ha : Membership.mem (Set.image ρ G) a
      ⊢ ∀ (o : Set A), IsOpen o → Membership.mem o a → (Inter.inter o (HasCompl.comp …
    -/
    intro N N_open hN
    -- get $x \in A$ from nonempty open $G \cap \rho^{-1}(N)$
    /-
      case neg
      A E : Type u
      inst✝¹ : TopologicalSpace A
      inst✝ : TopologicalSpace E
      ρ : E → A
      ρ_cont : Continuous ρ
      ρ_surj : Function.Surjective ρ
      zorn_subset : ∀ (E₀ : Set E), Ne E₀ Set.univ → IsClosed E₀ → Ne (Set.image ρ E …
      G : Set E
      hG : IsOpen G
      G_empty : Not (Eq G EmptyCollection.emptyCollection)
      a : A
      ha : Membership.mem (Set.image ρ G) a
      N : Set A
      N_open : IsOpen N
      hN : Membership.mem N a
      ⊢ (Inter.inter N (HasCompl.compl (Set.image ρ (HasCompl.compl G)))).Nonempty
    -/
    rcases (G.mem_image ρ a).mp ha with ⟨e, he, rfl⟩
    /-
      case neg.intro.intro
      A E : Type u
      inst✝¹ : TopologicalSpace A
      inst✝ : TopologicalSpace E
      ρ : E → A
      ρ_cont : Continuous ρ
      ρ_surj : Function.Surjective ρ
      zorn_subset : ∀ (E₀ : Set E), Ne E₀ Set.univ → IsClosed E₀ → Ne (Set.image ρ E …
      G : Set E
      hG : IsOpen G
      G_empty : Not (Eq G EmptyCollection.emptyCollection)
      N : Set A
      N_open : IsOpen N
      e : E
      he : Membership.mem G e
      ha : Membership.mem (Set.image ρ G) (ρ e)
      hN : Membership.mem N (ρ e)
      ⊢ (Inter.inter N (HasCompl.compl (Set.image ρ (HasCompl.compl G)))).Nonempty
    -/
    have nonempty : (G ∩ ρ⁻¹' N).Nonempty := ⟨e, mem_inter he <| mem_preimage.mpr hN⟩
    /-
      case neg.intro.intro
      A E : Type u
      inst✝¹ : TopologicalSpace A
      inst✝ : TopologicalSpace E
      ρ : E → A
      ρ_cont : Continuous ρ
      ρ_surj : Function.Surjective ρ
      zorn_subset : ∀ (E₀ : Set E), Ne E₀ Set.univ → IsClosed E₀ → Ne (Set.image ρ E …
      G : Set E
      hG : IsOpen G
      G_empty : Not (Eq G EmptyCollection.emptyCollection)
      N : Set A
      N_open : IsOpen N
      e : E
      he : Membership.mem G e
      ha : Membership.mem (Set.image ρ G) (ρ e)
      hN : Membership.mem N (ρ e)
      nonempty : (Inter.inter G (Set.preimage ρ N)).Nonempty
      ⊢ (Inter.inter N (HasCompl.compl (Set.image ρ (HasCompl.compl G)))).Nonempty
    -/
    have is_open : IsOpen <| G ∩ ρ⁻¹' N := hG.inter <| N_open.preimage ρ_cont
    have ne_univ : ρ '' (G ∩ ρ⁻¹' N)ᶜ ≠ univ :=
      zorn_subset _ (compl_ne_univ.mpr nonempty) is_open.isClosed_compl
    /-
      case neg.intro.intro
      A E : Type u
      inst✝¹ : TopologicalSpace A
      inst✝ : TopologicalSpace E
      ρ : E → A
      ρ_cont : Continuous ρ
      ρ_surj : Function.Surjective ρ
      zorn_subset : ∀ (E₀ : Set E), Ne E₀ Set.univ → IsClosed E₀ → Ne (Set.image ρ E …
      G : Set E
      hG : IsOpen G
      G_empty : Not (Eq G EmptyCollection.emptyCollection)
      N : Set A
      N_open : IsOpen N
      e : E
      he : Membership.mem G e
      ha : Membership.mem (Set.image ρ G) (ρ e)
      hN : Membership.mem N (ρ e)
      nonempty : (Inter.inter G (Set.preimage ρ N)).Nonempty
      is_open : IsOpen (Inter.inter G (Set.preimage ρ N))
      ne_univ : Ne (Set.image ρ (HasCompl.compl (Inter.inter G (Set.preimage ρ N)))) …
      ⊢ (Inter.inter N (HasCompl.compl (Set.image ρ (HasCompl.compl G)))).Nonempty
    -/
    rcases nonempty_compl.mpr ne_univ with ⟨x, hx⟩
    -- prove $x \in N \cap (A \setminus \rho(E \setminus G))$
    /-
      case neg.intro.intro.intro
      A E : Type u
      inst✝¹ : TopologicalSpace A
      inst✝ : TopologicalSpace E
      ρ : E → A
      ρ_cont : Continuous ρ
      ρ_surj : Function.Surjective ρ
      zorn_subset : ∀ (E₀ : Set E), Ne E₀ Set.univ → IsClosed E₀ → Ne (Set.image ρ E …
      G : Set E
      hG : IsOpen G
      G_empty : Not (Eq G EmptyCollection.emptyCollection)
      N : Set A
      N_open : IsOpen N
      e : E
      he : Membership.mem G e
      ha : Membership.mem (Set.image ρ G) (ρ e)
      hN : Membership.mem N (ρ e)
      nonempty : (Inter.inter G (Set.preimage ρ N)).Nonempty
      is_open : IsOpen (Inter.inter G (Set.preimage ρ N))
      ne_univ : Ne (Set.image ρ (HasCompl.compl (Inter.inter G (Set.preimage ρ N)))) …
      x : A
      hx : Membership.mem (HasCompl.compl (Set.image ρ (HasCompl.compl (Inter.inter  …
      ⊢ (Inter.inter N (HasCompl.compl (Set.image ρ (HasCompl.compl G)))).Nonempty
    -/
    have hx' : x ∈ (ρ '' Gᶜ)ᶜ := fun h => hx <| image_subset ρ (by simp) h
    /-
      case neg.intro.intro.intro
      A E : Type u
      inst✝¹ : TopologicalSpace A
      inst✝ : TopologicalSpace E
      ρ : E → A
      ρ_cont : Continuous ρ
      ρ_surj : Function.Surjective ρ
      zorn_subset : ∀ (E₀ : Set E), Ne E₀ Set.univ → IsClosed E₀ → Ne (Set.image ρ E …
      G : Set E
      hG : IsOpen G
      G_empty : Not (Eq G EmptyCollection.emptyCollection)
      N : Set A
      N_open : IsOpen N
      e : E
      he : Membership.mem G e
      ha : Membership.mem (Set.image ρ G) (ρ e)
      hN : Membership.mem N (ρ e)
      nonempty : (Inter.inter G (Set.preimage ρ N)).Nonempty
      is_open : IsOpen (Inter.inter G (Set.preimage ρ N))
      ne_univ : Ne (Set.image ρ (HasCompl.compl (Inter.inter G (Set.preimage ρ N)))) …
      x : A
      hx : Membership.mem (HasCompl.compl (Set.image ρ (HasCompl.compl (Inter.inter  …
      hx' : Membership.mem (HasCompl.compl (Set.image ρ (HasCompl.compl G))) x
      ⊢ (Inter.inter N (HasCompl.compl (Set.image ρ (HasCompl.compl G)))).Nonempty
    -/
    rcases ρ_surj x with ⟨y, rfl⟩
    /-
      case neg.intro.intro.intro.intro
      A E : Type u
      inst✝¹ : TopologicalSpace A
      inst✝ : TopologicalSpace E
      ρ : E → A
      ρ_cont : Continuous ρ
      ρ_surj : Function.Surjective ρ
      zorn_subset : ∀ (E₀ : Set E), Ne E₀ Set.univ → IsClosed E₀ → Ne (Set.image ρ E …
      G : Set E
      hG : IsOpen G
      G_empty : Not (Eq G EmptyCollection.emptyCollection)
      N : Set A
      N_open : IsOpen N
      e : E
      he : Membership.mem G e
      ha : Membership.mem (Set.image ρ G) (ρ e)
      hN : Membership.mem N (ρ e)
      nonempty : (Inter.inter G (Set.preimage ρ N)).Nonempty
      is_open : IsOpen (Inter.inter G (Set.preimage ρ N))
      ne_univ : Ne (Set.image ρ (HasCompl.compl (Inter.inter G (Set.preimage ρ N)))) …
      y : E
      hx : Membership.mem (HasCompl.compl (Set.image ρ (HasCompl.compl (Inter.inter  …
      hx' : Membership.mem (HasCompl.compl (Set.image ρ (HasCompl.compl G))) (ρ y)
      ⊢ (Inter.inter N (HasCompl.compl (Set.image ρ (HasCompl.compl G)))).Nonempty
    -/
    have hy : y ∈ G ∩ ρ⁻¹' N := by simpa using mt (mem_image_of_mem ρ) <| mem_compl hx
    /-
      case neg.intro.intro.intro.intro
      A E : Type u
      inst✝¹ : TopologicalSpace A
      inst✝ : TopologicalSpace E
      ρ : E → A
      ρ_cont : Continuous ρ
      ρ_surj : Function.Surjective ρ
      zorn_subset : ∀ (E₀ : Set E), Ne E₀ Set.univ → IsClosed E₀ → Ne (Set.image ρ E …
      G : Set E
      hG : IsOpen G
      G_empty : Not (Eq G EmptyCollection.emptyCollection)
      N : Set A
      N_open : IsOpen N
      e : E
      he : Membership.mem G e
      ha : Membership.mem (Set.image ρ G) (ρ e)
      hN : Membership.mem N (ρ e)
      nonempty : (Inter.inter G (Set.preimage ρ N)).Nonempty
      is_open : IsOpen (Inter.inter G (Set.preimage ρ N))
      ne_univ : Ne (Set.image ρ (HasCompl.compl (Inter.inter G (Set.preimage ρ N)))) …
      y : E
      hx : Membership.mem (HasCompl.compl (Set.image ρ (HasCompl.compl (Inter.inter  …
      hx' : Membership.mem (HasCompl.compl (Set.image ρ (HasCompl.compl G))) (ρ y)
      hy : Membership.mem (Inter.inter G (Set.preimage ρ N)) y
      ⊢ (Inter.inter N (HasCompl.compl (Set.image ρ (HasCompl.compl G)))).Nonempty
    -/
    exact ⟨ρ y, mem_inter (mem_preimage.mp <| mem_of_mem_inter_right hy) hx'⟩
    /-
      🎉 no goals
    -/


/-- Lemma 2.2 in [Gleason, *Projective topological spaces*][gleason1958]:
in an extremally disconnected space, if $U_1$ and $U_2$ are disjoint open sets,
then $\overline{U_1}$ and $\overline{U_2}$ are also disjoint. -/
lemma ExtremallyDisconnected.disjoint_closure_of_disjoint_isOpen [ExtremallyDisconnected A]
    {U₁ U₂ : Set A} (h : Disjoint U₁ U₂) (hU₁ : IsOpen U₁) (hU₂ : IsOpen U₂) :
    Disjoint (closure U₁) (closure U₂) :=
  (h.closure_right hU₁).closure_left <| open_closure U₂ hU₂


private lemma ExtremallyDisconnected.homeoCompactToT2_injective [ExtremallyDisconnected A]
    [T2Space A] [T2Space E] [CompactSpace E] {ρ : E → A} (ρ_cont : Continuous ρ)
    (ρ_surj : ρ.Surjective) (zorn_subset : ∀ E₀ : Set E, E₀ ≠ univ → IsClosed E₀ → ρ '' E₀ ≠ univ) :
    ρ.Injective := by
  -- let $x_1, x_2 \in E$ be distinct points such that $\rho(x_1) = \rho(x_2)$
  /-
    A E : Type u
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : TopologicalSpace E
    inst✝³ : ExtremallyDisconnected A
    inst✝² : T2Space A
    inst✝¹ : T2Space E
    inst✝ : CompactSpace E
    ρ : E → A
    ρ_cont : Continuous ρ
    ρ_surj : Function.Surjective ρ
    zorn_subset : ∀ (E₀ : Set E), Ne E₀ Set.univ → IsClosed E₀ → Ne (Set.image ρ E …
    ⊢ Function.Injective ρ
  -/
  intro x₁ x₂ hρx
  /-
    A E : Type u
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : TopologicalSpace E
    inst✝³ : ExtremallyDisconnected A
    inst✝² : T2Space A
    inst✝¹ : T2Space E
    inst✝ : CompactSpace E
    ρ : E → A
    ρ_cont : Continuous ρ
    ρ_surj : Function.Surjective ρ
    zorn_subset : ∀ (E₀ : Set E), Ne E₀ Set.univ → IsClosed E₀ → Ne (Set.image ρ E …
    x₁ x₂ : E
    hρx : Eq (ρ x₁) (ρ x₂)
    ⊢ Eq x₁ x₂
  -/
  by_contra hx
  -- let $G_1$ and $G_2$ be disjoint open neighbourhoods of $x_1$ and $x_2$ respectively
  /-
    A E : Type u
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : TopologicalSpace E
    inst✝³ : ExtremallyDisconnected A
    inst✝² : T2Space A
    inst✝¹ : T2Space E
    inst✝ : CompactSpace E
    ρ : E → A
    ρ_cont : Continuous ρ
    ρ_surj : Function.Surjective ρ
    zorn_subset : ∀ (E₀ : Set E), Ne E₀ Set.univ → IsClosed E₀ → Ne (Set.image ρ E …
    x₁ x₂ : E
    hρx : Eq (ρ x₁) (ρ x₂)
    hx : Not (Eq x₁ x₂)
    ⊢ False
  -/
  rcases t2_separation hx with ⟨G₁, G₂, G₁_open, G₂_open, hx₁, hx₂, disj⟩
  -- prove $A \setminus \rho(E - G_1)$ and $A \setminus \rho(E - G_2)$ are disjoint
  /-
    case intro.intro.intro.intro.intro.intro
    A E : Type u
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : TopologicalSpace E
    inst✝³ : ExtremallyDisconnected A
    inst✝² : T2Space A
    inst✝¹ : T2Space E
    inst✝ : CompactSpace E
    ρ : E → A
    ρ_cont : Continuous ρ
    ρ_surj : Function.Surjective ρ
    zorn_subset : ∀ (E₀ : Set E), Ne E₀ Set.univ → IsClosed E₀ → Ne (Set.image ρ E …
    x₁ x₂ : E
    hρx : Eq (ρ x₁) (ρ x₂)
    hx : Not (Eq x₁ x₂)
    G₁ G₂ : Set E
    G₁_open : IsOpen G₁
    G₂_open : IsOpen G₂
    hx₁ : Membership.mem G₁ x₁
    hx₂ : Membership.mem G₂ x₂
    disj : Disjoint G₁ G₂
    ⊢ False
  -/
  have G₁_comp : IsCompact G₁ᶜ := IsClosed.isCompact G₁_open.isClosed_compl
  /-
    case intro.intro.intro.intro.intro.intro
    A E : Type u
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : TopologicalSpace E
    inst✝³ : ExtremallyDisconnected A
    inst✝² : T2Space A
    inst✝¹ : T2Space E
    inst✝ : CompactSpace E
    ρ : E → A
    ρ_cont : Continuous ρ
    ρ_surj : Function.Surjective ρ
    zorn_subset : ∀ (E₀ : Set E), Ne E₀ Set.univ → IsClosed E₀ → Ne (Set.image ρ E …
    x₁ x₂ : E
    hρx : Eq (ρ x₁) (ρ x₂)
    hx : Not (Eq x₁ x₂)
    G₁ G₂ : Set E
    G₁_open : IsOpen G₁
    G₂_open : IsOpen G₂
    hx₁ : Membership.mem G₁ x₁
    hx₂ : Membership.mem G₂ x₂
    disj : Disjoint G₁ G₂
    G₁_comp : IsCompact (HasCompl.compl G₁)
    ⊢ False
  -/
  have G₂_comp : IsCompact G₂ᶜ := IsClosed.isCompact G₂_open.isClosed_compl
  /-
    case intro.intro.intro.intro.intro.intro
    A E : Type u
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : TopologicalSpace E
    inst✝³ : ExtremallyDisconnected A
    inst✝² : T2Space A
    inst✝¹ : T2Space E
    inst✝ : CompactSpace E
    ρ : E → A
    ρ_cont : Continuous ρ
    ρ_surj : Function.Surjective ρ
    zorn_subset : ∀ (E₀ : Set E), Ne E₀ Set.univ → IsClosed E₀ → Ne (Set.image ρ E …
    x₁ x₂ : E
    hρx : Eq (ρ x₁) (ρ x₂)
    hx : Not (Eq x₁ x₂)
    G₁ G₂ : Set E
    G₁_open : IsOpen G₁
    G₂_open : IsOpen G₂
    hx₁ : Membership.mem G₁ x₁
    hx₂ : Membership.mem G₂ x₂
    disj : Disjoint G₁ G₂
    G₁_comp : IsCompact (HasCompl.compl G₁)
    G₂_comp : IsCompact (HasCompl.compl G₂)
    ⊢ False
  -/
  have G₁_open' : IsOpen (ρ '' G₁ᶜ)ᶜ := (G₁_comp.image ρ_cont).isClosed.isOpen_compl
  /-
    case intro.intro.intro.intro.intro.intro
    A E : Type u
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : TopologicalSpace E
    inst✝³ : ExtremallyDisconnected A
    inst✝² : T2Space A
    inst✝¹ : T2Space E
    inst✝ : CompactSpace E
    ρ : E → A
    ρ_cont : Continuous ρ
    ρ_surj : Function.Surjective ρ
    zorn_subset : ∀ (E₀ : Set E), Ne E₀ Set.univ → IsClosed E₀ → Ne (Set.image ρ E …
    x₁ x₂ : E
    hρx : Eq (ρ x₁) (ρ x₂)
    hx : Not (Eq x₁ x₂)
    G₁ G₂ : Set E
    G₁_open : IsOpen G₁
    G₂_open : IsOpen G₂
    hx₁ : Membership.mem G₁ x₁
    hx₂ : Membership.mem G₂ x₂
    disj : Disjoint G₁ G₂
    G₁_comp : IsCompact (HasCompl.compl G₁)
    G₂_comp : IsCompact (HasCompl.compl G₂)
    G₁_open' : IsOpen (HasCompl.compl (Set.image ρ (HasCompl.compl G₁)))
    ⊢ False
  -/
  have G₂_open' : IsOpen (ρ '' G₂ᶜ)ᶜ := (G₂_comp.image ρ_cont).isClosed.isOpen_compl
  have disj' : Disjoint (ρ '' G₁ᶜ)ᶜ (ρ '' G₂ᶜ)ᶜ := by
    rw [disjoint_iff_inter_eq_empty, ← compl_union, ← image_union, ← compl_inter,
      disjoint_iff_inter_eq_empty.mp disj, compl_empty, compl_empty_iff,
      image_univ_of_surjective ρ_surj]
  -- apply Lemma 2.2 to prove their closures are disjoint
  have disj'' : Disjoint (closure (ρ '' G₁ᶜ)ᶜ) (closure (ρ '' G₂ᶜ)ᶜ) :=
    disjoint_closure_of_disjoint_isOpen disj' G₁_open' G₂_open'
  -- apply Lemma 2.1 to prove $\rho(x_1) = \rho(x_2)$ lies in their intersection
  have hx₁' := image_subset_closure_compl_image_compl_of_isOpen ρ_cont ρ_surj zorn_subset G₁_open <|
    mem_image_of_mem ρ hx₁
  have hx₂' := image_subset_closure_compl_image_compl_of_isOpen ρ_cont ρ_surj zorn_subset G₂_open <|
    mem_image_of_mem ρ hx₂
  /-
    case intro.intro.intro.intro.intro.intro
    A E : Type u
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : TopologicalSpace E
    inst✝³ : ExtremallyDisconnected A
    inst✝² : T2Space A
    inst✝¹ : T2Space E
    inst✝ : CompactSpace E
    ρ : E → A
    ρ_cont : Continuous ρ
    ρ_surj : Function.Surjective ρ
    zorn_subset : ∀ (E₀ : Set E), Ne E₀ Set.univ → IsClosed E₀ → Ne (Set.image ρ E …
    x₁ x₂ : E
    hρx : Eq (ρ x₁) (ρ x₂)
    hx : Not (Eq x₁ x₂)
    G₁ G₂ : Set E
    G₁_open : IsOpen G₁
    G₂_open : IsOpen G₂
    hx₁ : Membership.mem G₁ x₁
    hx₂ : Membership.mem G₂ x₂
    disj : Disjoint G₁ G₂
    G₁_comp : IsCompact (HasCompl.compl G₁)
    G₂_comp : IsCompact (HasCompl.compl G₂)
    G₁_open' : IsOpen (HasCompl.compl (Set.image ρ (HasCompl.compl G₁)))
    G₂_open' : IsOpen (HasCompl.compl (Set.image ρ (HasCompl.compl G₂)))
    disj' : Disjoint (HasCompl.compl (Set.image ρ (HasCompl.compl G₁))) (HasCompl. …
    disj'' : Disjoint (closure (HasCompl.compl (Set.image ρ (HasCompl.compl G₁)))) …
    hx₁' : Membership.mem (closure (HasCompl.compl (Set.image ρ (HasCompl.compl G₁ …
    hx₂' : Membership.mem (closure (HasCompl.compl (Set.image ρ (HasCompl.compl G₂ …
    ⊢ False
  -/
  exact disj''.ne_of_mem hx₁' hx₂' hρx
  /-
    🎉 no goals
  -/


/-- Lemma 2.3 in [Gleason, *Projective topological spaces*][gleason1958]:
a continuous surjection from a compact Hausdorff space to an extremally disconnected Hausdorff space
satisfying the "Zorn subset condition" is a homeomorphism. -/
noncomputable def ExtremallyDisconnected.homeoCompactToT2 [ExtremallyDisconnected A] [T2Space A]
    [T2Space E] [CompactSpace E] {ρ : E → A} (ρ_cont : Continuous ρ) (ρ_surj : ρ.Surjective)
    (zorn_subset : ∀ E₀ : Set E, E₀ ≠ univ → IsClosed E₀ → ρ '' E₀ ≠ univ) : E ≃ₜ A :=
  ρ_cont.homeoOfEquivCompactToT2
    (f := Equiv.ofBijective ρ ⟨homeoCompactToT2_injective ρ_cont ρ_surj zorn_subset, ρ_surj⟩)


/-- Theorem 2.5 in [Gleason, *Projective topological spaces*][gleason1958]:
in the category of compact spaces and continuous maps,
the projective spaces are precisely the extremally disconnected spaces. -/
protected theorem CompactT2.ExtremallyDisconnected.projective [ExtremallyDisconnected A]
    [CompactSpace A] [T2Space A] : CompactT2.Projective A := by
  -- let $B$ and $C$ be compact; let $f : B \twoheadrightarrow C$ and $\phi : A \to C$ be continuous
  /-
    A : Type u
    inst✝³ : TopologicalSpace A
    inst✝² : ExtremallyDisconnected A
    inst✝¹ : CompactSpace A
    inst✝ : T2Space A
    ⊢ CompactT2.Projective A
  -/
  intro B C _ _ _ _ _ _ φ f φ_cont f_cont f_surj
  -- let $D := \{(a, b) : \phi(a) = f(b)\}$ with projections $\pi_1 : D \to A$ and $\pi_2 : D \to B$
  /-
    A : Type u
    inst✝⁹ : TopologicalSpace A
    inst✝⁸ : ExtremallyDisconnected A
    inst✝⁷ : CompactSpace A
    inst✝⁶ : T2Space A
    B C : Type u
    inst✝⁵ : TopologicalSpace B
    inst✝⁴ : TopologicalSpace C
    inst✝³ : CompactSpace B
    inst✝² : T2Space B
    inst✝¹ : CompactSpace C
    inst✝ : T2Space C
    φ : A → C
    f : B → C
    φ_cont : Continuous φ
    f_cont : Continuous f
    f_surj : Function.Surjective f
    ⊢ Exists fun h => And (Continuous h) (Eq (Function.comp f h) φ)
  -/
  let D : Set <| A × B := {x | φ x.fst = f x.snd}
  have D_comp : CompactSpace D := isCompact_iff_compactSpace.mp
    (isClosed_eq (φ_cont.comp continuous_fst) (f_cont.comp continuous_snd)).isCompact
  -- apply Lemma 2.4 to get closed $E$ satisfying "Zorn subset condition"
  /-
    A : Type u
    inst✝⁹ : TopologicalSpace A
    inst✝⁸ : ExtremallyDisconnected A
    inst✝⁷ : CompactSpace A
    inst✝⁶ : T2Space A
    B C : Type u
    inst✝⁵ : TopologicalSpace B
    inst✝⁴ : TopologicalSpace C
    inst✝³ : CompactSpace B
    inst✝² : T2Space B
    inst✝¹ : CompactSpace C
    inst✝ : T2Space C
    φ : A → C
    f : B → C
    φ_cont : Continuous φ
    f_cont : Continuous f
    f_surj : Function.Surjective f
    D : Set (Prod A B) := setOf fun x => Eq (φ x.1) (f x.2)
    D_comp : CompactSpace ↑D
    ⊢ Exists fun h => And (Continuous h) (Eq (Function.comp f h) φ)
  -/
  let π₁ : D → A := Prod.fst ∘ Subtype.val
  /-
    A : Type u
    inst✝⁹ : TopologicalSpace A
    inst✝⁸ : ExtremallyDisconnected A
    inst✝⁷ : CompactSpace A
    inst✝⁶ : T2Space A
    B C : Type u
    inst✝⁵ : TopologicalSpace B
    inst✝⁴ : TopologicalSpace C
    inst✝³ : CompactSpace B
    inst✝² : T2Space B
    inst✝¹ : CompactSpace C
    inst✝ : T2Space C
    φ : A → C
    f : B → C
    φ_cont : Continuous φ
    f_cont : Continuous f
    f_surj : Function.Surjective f
    D : Set (Prod A B) := setOf fun x => Eq (φ x.1) (f x.2)
    D_comp : CompactSpace ↑D
    π₁ : ↑D → A := Function.comp Prod.fst Subtype.val
    ⊢ Exists fun h => And (Continuous h) (Eq (Function.comp f h) φ)
  -/
  have π₁_cont : Continuous π₁ := continuous_fst.comp continuous_subtype_val
  /-
    A : Type u
    inst✝⁹ : TopologicalSpace A
    inst✝⁸ : ExtremallyDisconnected A
    inst✝⁷ : CompactSpace A
    inst✝⁶ : T2Space A
    B C : Type u
    inst✝⁵ : TopologicalSpace B
    inst✝⁴ : TopologicalSpace C
    inst✝³ : CompactSpace B
    inst✝² : T2Space B
    inst✝¹ : CompactSpace C
    inst✝ : T2Space C
    φ : A → C
    f : B → C
    φ_cont : Continuous φ
    f_cont : Continuous f
    f_surj : Function.Surjective f
    D : Set (Prod A B) := setOf fun x => Eq (φ x.1) (f x.2)
    D_comp : CompactSpace ↑D
    π₁ : ↑D → A := Function.comp Prod.fst Subtype.val
    π₁_cont : Continuous π₁
    ⊢ Exists fun h => And (Continuous h) (Eq (Function.comp f h) φ)
  -/
  have π₁_surj : π₁.Surjective := fun a => ⟨⟨⟨a, _⟩, (f_surj <| φ a).choose_spec.symm⟩, rfl⟩
  /-
    A : Type u
    inst✝⁹ : TopologicalSpace A
    inst✝⁸ : ExtremallyDisconnected A
    inst✝⁷ : CompactSpace A
    inst✝⁶ : T2Space A
    B C : Type u
    inst✝⁵ : TopologicalSpace B
    inst✝⁴ : TopologicalSpace C
    inst✝³ : CompactSpace B
    inst✝² : T2Space B
    inst✝¹ : CompactSpace C
    inst✝ : T2Space C
    φ : A → C
    f : B → C
    φ_cont : Continuous φ
    f_cont : Continuous f
    f_surj : Function.Surjective f
    D : Set (Prod A B) := setOf fun x => Eq (φ x.1) (f x.2)
    D_comp : CompactSpace ↑D
    π₁ : ↑D → A := Function.comp Prod.fst Subtype.val
    π₁_cont : Continuous π₁
    π₁_surj : Function.Surjective π₁
    ⊢ Exists fun h => And (Continuous h) (Eq (Function.comp f h) φ)
  -/
  rcases exists_compact_surjective_zorn_subset π₁_cont π₁_surj with ⟨E, _, E_onto, E_min⟩
  -- apply Lemma 2.3 to get homeomorphism $\pi_1|_E : E \to A$
  /-
    case intro.intro.intro
    A : Type u
    inst✝⁹ : TopologicalSpace A
    inst✝⁸ : ExtremallyDisconnected A
    inst✝⁷ : CompactSpace A
    inst✝⁶ : T2Space A
    B C : Type u
    inst✝⁵ : TopologicalSpace B
    inst✝⁴ : TopologicalSpace C
    inst✝³ : CompactSpace B
    inst✝² : T2Space B
    inst✝¹ : CompactSpace C
    inst✝ : T2Space C
    φ : A → C
    f : B → C
    φ_cont : Continuous φ
    f_cont : Continuous f
    f_surj : Function.Surjective f
    D : Set (Prod A B) := setOf fun x => Eq (φ x.1) (f x.2)
    D_comp : CompactSpace ↑D
    π₁ : ↑D → A := Function.comp Prod.fst Subtype.val
    π₁_cont : Continuous π₁
    π₁_surj : Function.Surjective π₁
    E : Set ↑D
    left✝ : CompactSpace ↑E
    E_onto : Eq (Set.image π₁ E) Set.univ
    E_min : ∀ (E₀ : Set ↑E), Ne E₀ Set.univ → IsClosed E₀ → Ne (Set.image (E.restr …
    ⊢ Exists fun h => And (Continuous h) (Eq (Function.comp f h) φ)
  -/
  let ρ : E → A := E.restrict π₁
  /-
    case intro.intro.intro
    A : Type u
    inst✝⁹ : TopologicalSpace A
    inst✝⁸ : ExtremallyDisconnected A
    inst✝⁷ : CompactSpace A
    inst✝⁶ : T2Space A
    B C : Type u
    inst✝⁵ : TopologicalSpace B
    inst✝⁴ : TopologicalSpace C
    inst✝³ : CompactSpace B
    inst✝² : T2Space B
    inst✝¹ : CompactSpace C
    inst✝ : T2Space C
    φ : A → C
    f : B → C
    φ_cont : Continuous φ
    f_cont : Continuous f
    f_surj : Function.Surjective f
    D : Set (Prod A B) := setOf fun x => Eq (φ x.1) (f x.2)
    D_comp : CompactSpace ↑D
    π₁ : ↑D → A := Function.comp Prod.fst Subtype.val
    π₁_cont : Continuous π₁
    π₁_surj : Function.Surjective π₁
    E : Set ↑D
    left✝ : CompactSpace ↑E
    E_onto : Eq (Set.image π₁ E) Set.univ
    E_min : ∀ (E₀ : Set ↑E), Ne E₀ Set.univ → IsClosed E₀ → Ne (Set.image (E.restr …
    ρ : ↑E → A := E.restrict π₁
    ⊢ Exists fun h => And (Continuous h) (Eq (Function.comp f h) φ)
  -/
  have ρ_cont : Continuous ρ := π₁_cont.continuousOn.restrict
  have ρ_surj : ρ.Surjective := fun a => by
    rcases (E_onto ▸ mem_univ a : a ∈ π₁ '' E) with ⟨d, ⟨hd, rfl⟩⟩; exact ⟨⟨d, hd⟩, rfl⟩
  /-
    case intro.intro.intro
    A : Type u
    inst✝⁹ : TopologicalSpace A
    inst✝⁸ : ExtremallyDisconnected A
    inst✝⁷ : CompactSpace A
    inst✝⁶ : T2Space A
    B C : Type u
    inst✝⁵ : TopologicalSpace B
    inst✝⁴ : TopologicalSpace C
    inst✝³ : CompactSpace B
    inst✝² : T2Space B
    inst✝¹ : CompactSpace C
    inst✝ : T2Space C
    φ : A → C
    f : B → C
    φ_cont : Continuous φ
    f_cont : Continuous f
    f_surj : Function.Surjective f
    D : Set (Prod A B) := setOf fun x => Eq (φ x.1) (f x.2)
    D_comp : CompactSpace ↑D
    π₁ : ↑D → A := Function.comp Prod.fst Subtype.val
    π₁_cont : Continuous π₁
    π₁_surj : Function.Surjective π₁
    E : Set ↑D
    left✝ : CompactSpace ↑E
    E_onto : Eq (Set.image π₁ E) Set.univ
    E_min : ∀ (E₀ : Set ↑E), Ne E₀ Set.univ → IsClosed E₀ → Ne (Set.image (E.restr …
    ρ : ↑E → A := E.restrict π₁
    ρ_cont : Continuous ρ
    ρ_surj : Function.Surjective ρ
    ⊢ Exists fun h => And (Continuous h) (Eq (Function.comp f h) φ)
  -/
  let ρ' := ExtremallyDisconnected.homeoCompactToT2 ρ_cont ρ_surj E_min
  -- prove $\rho := \pi_2|_E \circ \pi_1|_E^{-1}$ satisfies $\phi = f \circ \rho$
  /-
    case intro.intro.intro
    A : Type u
    inst✝⁹ : TopologicalSpace A
    inst✝⁸ : ExtremallyDisconnected A
    inst✝⁷ : CompactSpace A
    inst✝⁶ : T2Space A
    B C : Type u
    inst✝⁵ : TopologicalSpace B
    inst✝⁴ : TopologicalSpace C
    inst✝³ : CompactSpace B
    inst✝² : T2Space B
    inst✝¹ : CompactSpace C
    inst✝ : T2Space C
    φ : A → C
    f : B → C
    φ_cont : Continuous φ
    f_cont : Continuous f
    f_surj : Function.Surjective f
    D : Set (Prod A B) := setOf fun x => Eq (φ x.1) (f x.2)
    D_comp : CompactSpace ↑D
    π₁ : ↑D → A := Function.comp Prod.fst Subtype.val
    π₁_cont : Continuous π₁
    π₁_surj : Function.Surjective π₁
    E : Set ↑D
    left✝ : CompactSpace ↑E
    E_onto : Eq (Set.image π₁ E) Set.univ
    E_min : ∀ (E₀ : Set ↑E), Ne E₀ Set.univ → IsClosed E₀ → Ne (Set.image (E.restr …
    ρ : ↑E → A := E.restrict π₁
    ρ_cont : Continuous ρ
    ρ_surj : Function.Surjective ρ
    ρ' : Homeomorph (↑E) A := ExtremallyDisconnected.homeoCompactToT2 ρ_cont ρ_sur …
    ⊢ Exists fun h => And (Continuous h) (Eq (Function.comp f h) φ)
  -/
  let π₂ : D → B := Prod.snd ∘ Subtype.val
  /-
    case intro.intro.intro
    A : Type u
    inst✝⁹ : TopologicalSpace A
    inst✝⁸ : ExtremallyDisconnected A
    inst✝⁷ : CompactSpace A
    inst✝⁶ : T2Space A
    B C : Type u
    inst✝⁵ : TopologicalSpace B
    inst✝⁴ : TopologicalSpace C
    inst✝³ : CompactSpace B
    inst✝² : T2Space B
    inst✝¹ : CompactSpace C
    inst✝ : T2Space C
    φ : A → C
    f : B → C
    φ_cont : Continuous φ
    f_cont : Continuous f
    f_surj : Function.Surjective f
    D : Set (Prod A B) := setOf fun x => Eq (φ x.1) (f x.2)
    D_comp : CompactSpace ↑D
    π₁ : ↑D → A := Function.comp Prod.fst Subtype.val
    π₁_cont : Continuous π₁
    π₁_surj : Function.Surjective π₁
    E : Set ↑D
    left✝ : CompactSpace ↑E
    E_onto : Eq (Set.image π₁ E) Set.univ
    E_min : ∀ (E₀ : Set ↑E), Ne E₀ Set.univ → IsClosed E₀ → Ne (Set.image (E.restr …
    ρ : ↑E → A := E.restrict π₁
    ρ_cont : Continuous ρ
    ρ_surj : Function.Surjective ρ
    ρ' : Homeomorph (↑E) A := ExtremallyDisconnected.homeoCompactToT2 ρ_cont ρ_sur …
    π₂ : ↑D → B := Function.comp Prod.snd Subtype.val
    ⊢ Exists fun h => And (Continuous h) (Eq (Function.comp f h) φ)
  -/
  have π₂_cont : Continuous π₂ := continuous_snd.comp continuous_subtype_val
  /-
    case intro.intro.intro
    A : Type u
    inst✝⁹ : TopologicalSpace A
    inst✝⁸ : ExtremallyDisconnected A
    inst✝⁷ : CompactSpace A
    inst✝⁶ : T2Space A
    B C : Type u
    inst✝⁵ : TopologicalSpace B
    inst✝⁴ : TopologicalSpace C
    inst✝³ : CompactSpace B
    inst✝² : T2Space B
    inst✝¹ : CompactSpace C
    inst✝ : T2Space C
    φ : A → C
    f : B → C
    φ_cont : Continuous φ
    f_cont : Continuous f
    f_surj : Function.Surjective f
    D : Set (Prod A B) := setOf fun x => Eq (φ x.1) (f x.2)
    D_comp : CompactSpace ↑D
    π₁ : ↑D → A := Function.comp Prod.fst Subtype.val
    π₁_cont : Continuous π₁
    π₁_surj : Function.Surjective π₁
    E : Set ↑D
    left✝ : CompactSpace ↑E
    E_onto : Eq (Set.image π₁ E) Set.univ
    E_min : ∀ (E₀ : Set ↑E), Ne E₀ Set.univ → IsClosed E₀ → Ne (Set.image (E.restr …
    ρ : ↑E → A := E.restrict π₁
    ρ_cont : Continuous ρ
    ρ_surj : Function.Surjective ρ
    ρ' : Homeomorph (↑E) A := ExtremallyDisconnected.homeoCompactToT2 ρ_cont ρ_sur …
    π₂ : ↑D → B := Function.comp Prod.snd Subtype.val
    π₂_cont : Continuous π₂
    ⊢ Exists fun h => And (Continuous h) (Eq (Function.comp f h) φ)
  -/
  refine ⟨E.restrict π₂ ∘ ρ'.symm, ⟨π₂_cont.continuousOn.restrict.comp ρ'.symm.continuous, ?_⟩⟩
  suffices f ∘ E.restrict π₂ = φ ∘ ρ' by
    rw [← comp_assoc, this, comp_assoc, Homeomorph.self_comp_symm, comp_id]
  /-
    case intro.intro.intro
    A : Type u
    inst✝⁹ : TopologicalSpace A
    inst✝⁸ : ExtremallyDisconnected A
    inst✝⁷ : CompactSpace A
    inst✝⁶ : T2Space A
    B C : Type u
    inst✝⁵ : TopologicalSpace B
    inst✝⁴ : TopologicalSpace C
    inst✝³ : CompactSpace B
    inst✝² : T2Space B
    inst✝¹ : CompactSpace C
    inst✝ : T2Space C
    φ : A → C
    f : B → C
    φ_cont : Continuous φ
    f_cont : Continuous f
    f_surj : Function.Surjective f
    D : Set (Prod A B) := setOf fun x => Eq (φ x.1) (f x.2)
    D_comp : CompactSpace ↑D
    π₁ : ↑D → A := Function.comp Prod.fst Subtype.val
    π₁_cont : Continuous π₁
    π₁_surj : Function.Surjective π₁
    E : Set ↑D
    left✝ : CompactSpace ↑E
    E_onto : Eq (Set.image π₁ E) Set.univ
    E_min : ∀ (E₀ : Set ↑E), Ne E₀ Set.univ → IsClosed E₀ → Ne (Set.image (E.restr …
    ρ : ↑E → A := E.restrict π₁
    ρ_cont : Continuous ρ
    ρ_surj : Function.Surjective ρ
    ρ' : Homeomorph (↑E) A := ExtremallyDisconnected.homeoCompactToT2 ρ_cont ρ_sur …
    π₂ : ↑D → B := Function.comp Prod.snd Subtype.val
    π₂_cont : Continuous π₂
    ⊢ Eq (Function.comp f (E.restrict π₂)) (Function.comp φ ⇑ρ')
  -/
  ext x
  /-
    case intro.intro.intro.h
    A : Type u
    inst✝⁹ : TopologicalSpace A
    inst✝⁸ : ExtremallyDisconnected A
    inst✝⁷ : CompactSpace A
    inst✝⁶ : T2Space A
    B C : Type u
    inst✝⁵ : TopologicalSpace B
    inst✝⁴ : TopologicalSpace C
    inst✝³ : CompactSpace B
    inst✝² : T2Space B
    inst✝¹ : CompactSpace C
    inst✝ : T2Space C
    φ : A → C
    f : B → C
    φ_cont : Continuous φ
    f_cont : Continuous f
    f_surj : Function.Surjective f
    D : Set (Prod A B) := setOf fun x => Eq (φ x.1) (f x.2)
    D_comp : CompactSpace ↑D
    π₁ : ↑D → A := Function.comp Prod.fst Subtype.val
    π₁_cont : Continuous π₁
    π₁_surj : Function.Surjective π₁
    E : Set ↑D
    left✝ : CompactSpace ↑E
    E_onto : Eq (Set.image π₁ E) Set.univ
    E_min : ∀ (E₀ : Set ↑E), Ne E₀ Set.univ → IsClosed E₀ → Ne (Set.image (E.restr …
    ρ : ↑E → A := E.restrict π₁
    ρ_cont : Continuous ρ
    ρ_surj : Function.Surjective ρ
    ρ' : Homeomorph (↑E) A := ExtremallyDisconnected.homeoCompactToT2 ρ_cont ρ_sur …
    π₂ : ↑D → B := Function.comp Prod.snd Subtype.val
    π₂_cont : Continuous π₂
    x : ↑E
    ⊢ Eq (Function.comp f (E.restrict π₂) x) (Function.comp φ (⇑ρ') x)
  -/
  exact x.val.mem.symm
  /-
    🎉 no goals
  -/


protected theorem CompactT2.projective_iff_extremallyDisconnected [CompactSpace A] [T2Space A] :
    Projective A ↔ ExtremallyDisconnected A :=
  ⟨Projective.extremallyDisconnected, fun _ => ExtremallyDisconnected.projective⟩


@[deprecated (since := "2024-05-26")]
alias CompactT2.projective_iff_extremallyDisconnnected :=
  CompactT2.projective_iff_extremallyDisconnected


/-- The sigma-type of extremally disconnected spaces is extremally disconnected. -/
instance instExtremallyDisconnected {ι : Type*} {π : ι → Type*} [∀ i, TopologicalSpace (π i)]
    [h₀ : ∀ i, ExtremallyDisconnected (π i)] : ExtremallyDisconnected (Σ i, π i) := by
  /-
    X : Type u
    inst✝¹ : TopologicalSpace X
    ι : Type u_1
    π : ι → Type u_2
    inst✝ : (i : ι) → TopologicalSpace (π i)
    h₀ : ∀ (i : ι), ExtremallyDisconnected (π i)
    ⊢ ExtremallyDisconnected (Sigma fun i => π i)
  -/
  constructor
  /-
    case open_closure
    X : Type u
    inst✝¹ : TopologicalSpace X
    ι : Type u_1
    π : ι → Type u_2
    inst✝ : (i : ι) → TopologicalSpace (π i)
    h₀ : ∀ (i : ι), ExtremallyDisconnected (π i)
    ⊢ ∀ (U : Set (Sigma fun i => π i)), IsOpen U → IsOpen (closure U)
  -/
  intro s hs
  /-
    case open_closure
    X : Type u
    inst✝¹ : TopologicalSpace X
    ι : Type u_1
    π : ι → Type u_2
    inst✝ : (i : ι) → TopologicalSpace (π i)
    h₀ : ∀ (i : ι), ExtremallyDisconnected (π i)
    s : Set (Sigma fun i => π i)
    hs : IsOpen s
    ⊢ IsOpen (closure s)
  -/
  rw [isOpen_sigma_iff] at hs ⊢
  /-
    case open_closure
    X : Type u
    inst✝¹ : TopologicalSpace X
    ι : Type u_1
    π : ι → Type u_2
    inst✝ : (i : ι) → TopologicalSpace (π i)
    h₀ : ∀ (i : ι), ExtremallyDisconnected (π i)
    s : Set (Sigma fun i => π i)
    hs : ∀ (i : ι), IsOpen (Set.preimage (Sigma.mk i) s)
    ⊢ ∀ (i : ι), IsOpen (Set.preimage (Sigma.mk i) (closure s))
  -/
  intro i
  /-
    case open_closure
    X : Type u
    inst✝¹ : TopologicalSpace X
    ι : Type u_1
    π : ι → Type u_2
    inst✝ : (i : ι) → TopologicalSpace (π i)
    h₀ : ∀ (i : ι), ExtremallyDisconnected (π i)
    s : Set (Sigma fun i => π i)
    hs : ∀ (i : ι), IsOpen (Set.preimage (Sigma.mk i) s)
    i : ι
    ⊢ IsOpen (Set.preimage (Sigma.mk i) (closure s))
  -/
  rcases h₀ i with ⟨h₀⟩
  suffices h : Sigma.mk i ⁻¹' closure s = closure (Sigma.mk i ⁻¹' s) by
    rw [h]
    exact h₀ _ (hs i)
  /-
    case open_closure.mk
    X : Type u
    inst✝¹ : TopologicalSpace X
    ι : Type u_1
    π : ι → Type u_2
    inst✝ : (i : ι) → TopologicalSpace (π i)
    h₀✝ : ∀ (i : ι), ExtremallyDisconnected (π i)
    s : Set (Sigma fun i => π i)
    hs : ∀ (i : ι), IsOpen (Set.preimage (Sigma.mk i) s)
    i : ι
    h₀ : ∀ (U : Set (π i)), IsOpen U → IsOpen (closure U)
    ⊢ Eq (Set.preimage (Sigma.mk i) (closure s)) (closure (Set.preimage (Sigma.mk  …
  -/
  apply IsOpenMap.preimage_closure_eq_closure_preimage
    /-
      case open_closure.mk.hf
      X : Type u
      inst✝¹ : TopologicalSpace X
      ι : Type u_1
      π : ι → Type u_2
      inst✝ : (i : ι) → TopologicalSpace (π i)
      h₀✝ : ∀ (i : ι), ExtremallyDisconnected (π i)
      s : Set (Sigma fun i => π i)
      hs : ∀ (i : ι), IsOpen (Set.preimage (Sigma.mk i) s)
      i : ι
      h₀ : ∀ (U : Set (π i)), IsOpen U → IsOpen (closure U)
      ⊢ IsOpenMap (Sigma.mk i)
    -/
  · intro U _
    /-
      case open_closure.mk.hf
      X : Type u
      inst✝¹ : TopologicalSpace X
      ι : Type u_1
      π : ι → Type u_2
      inst✝ : (i : ι) → TopologicalSpace (π i)
      h₀✝ : ∀ (i : ι), ExtremallyDisconnected (π i)
      s : Set (Sigma fun i => π i)
      hs : ∀ (i : ι), IsOpen (Set.preimage (Sigma.mk i) s)
      i : ι
      h₀ : ∀ (U : Set (π i)), IsOpen U → IsOpen (closure U)
      U : Set (π i)
      a✝ : IsOpen U
      ⊢ IsOpen (Set.image (Sigma.mk i) U)
    -/
    rw [isOpen_sigma_iff]
    /-
      case open_closure.mk.hf
      X : Type u
      inst✝¹ : TopologicalSpace X
      ι : Type u_1
      π : ι → Type u_2
      inst✝ : (i : ι) → TopologicalSpace (π i)
      h₀✝ : ∀ (i : ι), ExtremallyDisconnected (π i)
      s : Set (Sigma fun i => π i)
      hs : ∀ (i : ι), IsOpen (Set.preimage (Sigma.mk i) s)
      i : ι
      h₀ : ∀ (U : Set (π i)), IsOpen U → IsOpen (closure U)
      U : Set (π i)
      a✝ : IsOpen U
      ⊢ ∀ (i_1 : ι), IsOpen (Set.preimage (Sigma.mk i_1) (Set.image (Sigma.mk i) U))
    -/
    intro j
    /-
      case open_closure.mk.hf
      X : Type u
      inst✝¹ : TopologicalSpace X
      ι : Type u_1
      π : ι → Type u_2
      inst✝ : (i : ι) → TopologicalSpace (π i)
      h₀✝ : ∀ (i : ι), ExtremallyDisconnected (π i)
      s : Set (Sigma fun i => π i)
      hs : ∀ (i : ι), IsOpen (Set.preimage (Sigma.mk i) s)
      i : ι
      h₀ : ∀ (U : Set (π i)), IsOpen U → IsOpen (closure U)
      U : Set (π i)
      a✝ : IsOpen U
      j : ι
      ⊢ IsOpen (Set.preimage (Sigma.mk j) (Set.image (Sigma.mk i) U))
    -/
    by_cases ij : i = j
      /-
        case pos
        X : Type u
        inst✝¹ : TopologicalSpace X
        ι : Type u_1
        π : ι → Type u_2
        inst✝ : (i : ι) → TopologicalSpace (π i)
        h₀✝ : ∀ (i : ι), ExtremallyDisconnected (π i)
        s : Set (Sigma fun i => π i)
        hs : ∀ (i : ι), IsOpen (Set.preimage (Sigma.mk i) s)
        i : ι
        h₀ : ∀ (U : Set (π i)), IsOpen U → IsOpen (closure U)
        U : Set (π i)
        a✝ : IsOpen U
        j : ι
        ij : Eq i j
        ⊢ IsOpen (Set.preimage (Sigma.mk j) (Set.image (Sigma.mk i) U))
      -/
    · rwa [← ij, sigma_mk_preimage_image_eq_self]
      /-
        🎉 no goals
      -/
      /-
        case neg
        X : Type u
        inst✝¹ : TopologicalSpace X
        ι : Type u_1
        π : ι → Type u_2
        inst✝ : (i : ι) → TopologicalSpace (π i)
        h₀✝ : ∀ (i : ι), ExtremallyDisconnected (π i)
        s : Set (Sigma fun i => π i)
        hs : ∀ (i : ι), IsOpen (Set.preimage (Sigma.mk i) s)
        i : ι
        h₀ : ∀ (U : Set (π i)), IsOpen U → IsOpen (closure U)
        U : Set (π i)
        a✝ : IsOpen U
        j : ι
        ij : Not (Eq i j)
        ⊢ IsOpen (Set.preimage (Sigma.mk j) (Set.image (Sigma.mk i) U))
      -/
    · rw [sigma_mk_preimage_image' ij]
      /-
        case neg
        X : Type u
        inst✝¹ : TopologicalSpace X
        ι : Type u_1
        π : ι → Type u_2
        inst✝ : (i : ι) → TopologicalSpace (π i)
        h₀✝ : ∀ (i : ι), ExtremallyDisconnected (π i)
        s : Set (Sigma fun i => π i)
        hs : ∀ (i : ι), IsOpen (Set.preimage (Sigma.mk i) s)
        i : ι
        h₀ : ∀ (U : Set (π i)), IsOpen U → IsOpen (closure U)
        U : Set (π i)
        a✝ : IsOpen U
        j : ι
        ij : Not (Eq i j)
        ⊢ IsOpen EmptyCollection.emptyCollection
      -/
      exact isOpen_empty
      /-
        🎉 no goals
      -/
    /-
      case open_closure.mk.hfc
      X : Type u
      inst✝¹ : TopologicalSpace X
      ι : Type u_1
      π : ι → Type u_2
      inst✝ : (i : ι) → TopologicalSpace (π i)
      h₀✝ : ∀ (i : ι), ExtremallyDisconnected (π i)
      s : Set (Sigma fun i => π i)
      hs : ∀ (i : ι), IsOpen (Set.preimage (Sigma.mk i) s)
      i : ι
      h₀ : ∀ (U : Set (π i)), IsOpen U → IsOpen (closure U)
      ⊢ Continuous (Sigma.mk i)
    -/
  · continuity
    /-
      🎉 no goals
    -/


