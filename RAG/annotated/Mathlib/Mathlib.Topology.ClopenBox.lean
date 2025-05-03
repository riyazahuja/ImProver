theorem exists_prod_subset (W : Clopens (X × Y)) {a : X × Y} (h : a ∈ W) :
    ∃ U : Clopens X, a.1 ∈ U ∧ ∃ V : Clopens Y, a.2 ∈ V ∧ U ×ˢ V ≤ W := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : CompactSpace Y
    W : TopologicalSpace.Clopens (Prod X Y)
    a : Prod X Y
    h : Membership.mem W a
    ⊢ Exists fun U => And (Membership.mem U a.1) (Exists fun V => And (Membership. …
  -/
  have hp : Continuous (fun y : Y ↦ (a.1, y)) := Continuous.Prod.mk _
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : CompactSpace Y
    W : TopologicalSpace.Clopens (Prod X Y)
    a : Prod X Y
    h : Membership.mem W a
    hp : Continuous fun y => { fst := a.1, snd := y }
    ⊢ Exists fun U => And (Membership.mem U a.1) (Exists fun V => And (Membership. …
  -/
  let V : Set Y := {y | (a.1, y) ∈ W}
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : CompactSpace Y
    W : TopologicalSpace.Clopens (Prod X Y)
    a : Prod X Y
    h : Membership.mem W a
    hp : Continuous fun y => { fst := a.1, snd := y }
    V : Set Y := setOf fun y => Membership.mem W { fst := a.1, snd := y }
    ⊢ Exists fun U => And (Membership.mem U a.1) (Exists fun V => And (Membership. …
  -/
  have hV : IsCompact V := (W.2.1.preimage hp).isCompact
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : CompactSpace Y
    W : TopologicalSpace.Clopens (Prod X Y)
    a : Prod X Y
    h : Membership.mem W a
    hp : Continuous fun y => { fst := a.1, snd := y }
    V : Set Y := setOf fun y => Membership.mem W { fst := a.1, snd := y }
    hV : IsCompact V
    ⊢ Exists fun U => And (Membership.mem U a.1) (Exists fun V => And (Membership. …
  -/
  let U : Set X := {x | MapsTo (Prod.mk x) V W}
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : CompactSpace Y
    W : TopologicalSpace.Clopens (Prod X Y)
    a : Prod X Y
    h : Membership.mem W a
    hp : Continuous fun y => { fst := a.1, snd := y }
    V : Set Y := setOf fun y => Membership.mem W { fst := a.1, snd := y }
    hV : IsCompact V
    U : Set X := setOf fun x => Set.MapsTo (Prod.mk x) V ↑W
    ⊢ Exists fun U => And (Membership.mem U a.1) (Exists fun V => And (Membership. …
  -/
  have hUV : U ×ˢ V ⊆ W := fun ⟨_, _⟩ hw ↦ hw.1 hw.2
  exact ⟨⟨U, (ContinuousMap.isClopen_setOf_mapsTo hV W.2).preimage
    (ContinuousMap.id (X × Y)).curry.2⟩, by simp [U, V, MapsTo], ⟨V, W.2.preimage hp⟩, h, hUV⟩


/-- Every clopen set in a product of two compact spaces
is a union of finitely many clopen boxes. -/
theorem exists_finset_eq_sup_prod (W : Clopens (X × Y)) :
    ∃ (I : Finset (Clopens X × Clopens Y)), W = I.sup fun i ↦ i.1 ×ˢ i.2 := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : CompactSpace Y
    inst✝ : CompactSpace X
    W : TopologicalSpace.Clopens (Prod X Y)
    ⊢ Exists fun I => Eq W (I.sup fun i => SProd.sprod i.1 i.2)
  -/
  choose! U hxU V hxV hUV using fun x ↦ W.exists_prod_subset (a := x)
  rcases W.2.1.isCompact.elim_nhds_subcover (fun x ↦ U x ×ˢ V x) (fun x hx ↦
    (U x ×ˢ V x).2.isOpen.mem_nhds ⟨hxU x hx, hxV x hx⟩) with ⟨I, hIW, hWI⟩
  classical
  use I.image fun x ↦ (U x, V x)
  rw [Finset.sup_image]
  refine le_antisymm (fun x hx ↦ ?_) (Finset.sup_le fun x hx ↦ ?_)
  · rcases Set.mem_iUnion₂.1 (hWI hx) with ⟨i, hi, hxi⟩
    exact SetLike.le_def.1 (Finset.le_sup hi) hxi
  · exact hUV _ <| hIW _ hx


lemma surjective_finset_sup_prod :
    Surjective fun I : Finset (Clopens X × Clopens Y) ↦ I.sup fun i ↦ i.1 ×ˢ i.2 := fun W ↦
  let ⟨I, hI⟩ := W.exists_finset_eq_sup_prod; ⟨I, hI.symm⟩


instance countable_prod [Countable (Clopens X)]
    [Countable (Clopens Y)] : Countable (Clopens (X × Y)) :=
  surjective_finset_sup_prod.countable


instance finite_prod [Finite (Clopens X)] [Finite (Clopens Y)] :
    Finite (Clopens (X × Y)) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝⁵ : TopologicalSpace X
    inst✝⁴ : TopologicalSpace Y
    inst✝³ : CompactSpace Y
    inst✝² : CompactSpace X
    inst✝¹ : Finite (TopologicalSpace.Clopens X)
    inst✝ : Finite (TopologicalSpace.Clopens Y)
    ⊢ Finite (TopologicalSpace.Clopens (Prod X Y))
  -/
  cases nonempty_fintype (Clopens X)
  /-
    case intro
    X : Type u_1
    Y : Type u_2
    inst✝⁵ : TopologicalSpace X
    inst✝⁴ : TopologicalSpace Y
    inst✝³ : CompactSpace Y
    inst✝² : CompactSpace X
    inst✝¹ : Finite (TopologicalSpace.Clopens X)
    inst✝ : Finite (TopologicalSpace.Clopens Y)
    val✝ : Fintype (TopologicalSpace.Clopens X)
    ⊢ Finite (TopologicalSpace.Clopens (Prod X Y))
  -/
  cases nonempty_fintype (Clopens Y)
  /-
    case intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝⁵ : TopologicalSpace X
    inst✝⁴ : TopologicalSpace Y
    inst✝³ : CompactSpace Y
    inst✝² : CompactSpace X
    inst✝¹ : Finite (TopologicalSpace.Clopens X)
    inst✝ : Finite (TopologicalSpace.Clopens Y)
    val✝¹ : Fintype (TopologicalSpace.Clopens X)
    val✝ : Fintype (TopologicalSpace.Clopens Y)
    ⊢ Finite (TopologicalSpace.Clopens (Prod X Y))
  -/
  exact .of_surjective _ surjective_finset_sup_prod
  /-
    🎉 no goals
  -/


lemma countable_iff_secondCountable [T2Space X]
    [TotallyDisconnectedSpace X] : Countable (Clopens X) ↔ SecondCountableTopology X := by
  /-
    X : Type u_1
    inst✝³ : TopologicalSpace X
    inst✝² : CompactSpace X
    inst✝¹ : T2Space X
    inst✝ : TotallyDisconnectedSpace X
    ⊢ Iff (Countable (TopologicalSpace.Clopens X)) (SecondCountableTopology X)
  -/
  refine ⟨fun h ↦ ⟨{s : Set X | IsClopen s}, ?_, ?_⟩, fun h ↦ ?_⟩
    /-
      case refine_1
      X : Type u_1
      inst✝³ : TopologicalSpace X
      inst✝² : CompactSpace X
      inst✝¹ : T2Space X
      inst✝ : TotallyDisconnectedSpace X
      h : Countable (TopologicalSpace.Clopens X)
      ⊢ (setOf fun s => IsClopen s).Countable
    -/
  · let f : {s : Set X | IsClopen s} → Clopens X := fun s ↦ ⟨s.1, s.2⟩
    /-
      case refine_1
      X : Type u_1
      inst✝³ : TopologicalSpace X
      inst✝² : CompactSpace X
      inst✝¹ : T2Space X
      inst✝ : TotallyDisconnectedSpace X
      h : Countable (TopologicalSpace.Clopens X)
      f : ↑(setOf fun s => IsClopen s) → TopologicalSpace.Clopens X := fun s => { ca …
      ⊢ (setOf fun s => IsClopen s).Countable
    -/
    exact (injective_of_le_imp_le f fun a ↦ a).countable
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      X : Type u_1
      inst✝³ : TopologicalSpace X
      inst✝² : CompactSpace X
      inst✝¹ : T2Space X
      inst✝ : TotallyDisconnectedSpace X
      h : Countable (TopologicalSpace.Clopens X)
      ⊢ Eq inst✝³ (TopologicalSpace.generateFrom (setOf fun s => IsClopen s))
    -/
  · apply IsTopologicalBasis.eq_generateFrom
    /-
      case refine_2.self
      X : Type u_1
      inst✝³ : TopologicalSpace X
      inst✝² : CompactSpace X
      inst✝¹ : T2Space X
      inst✝ : TotallyDisconnectedSpace X
      h : Countable (TopologicalSpace.Clopens X)
      ⊢ TopologicalSpace.IsTopologicalBasis (setOf fun s => IsClopen s)
    -/
    exact loc_compact_Haus_tot_disc_of_zero_dim
    /-
      🎉 no goals
    -/
  · have : ∀ (s : Clopens X), ∃ (t : Finset (countableBasis X)), s.1 = t.toSet.sUnion :=
      fun s ↦ eq_sUnion_finset_of_isTopologicalBasis_of_isCompact_open _
        (isBasis_countableBasis X) s.1 s.2.1.isCompact s.2.2
    /-
      case refine_3
      X : Type u_1
      inst✝³ : TopologicalSpace X
      inst✝² : CompactSpace X
      inst✝¹ : T2Space X
      inst✝ : TotallyDisconnectedSpace X
      h : SecondCountableTopology X
      this : ∀ (s : TopologicalSpace.Clopens X), Exists fun t => Eq s.carrier (Set.i …
      ⊢ Countable (TopologicalSpace.Clopens X)
    -/
    let f : Clopens X → Finset (countableBasis X) := fun s ↦ (this s).choose
    have hf : f.Injective := by
      intro s t (h : Exists.choose _ = Exists.choose _)
      ext1; change s.carrier = t.carrier
      rw [(this s).choose_spec, (this t).choose_spec, h]
    /-
      case refine_3
      X : Type u_1
      inst✝³ : TopologicalSpace X
      inst✝² : CompactSpace X
      inst✝¹ : T2Space X
      inst✝ : TotallyDisconnectedSpace X
      h : SecondCountableTopology X
      this : ∀ (s : TopologicalSpace.Clopens X), Exists fun t => Eq s.carrier (Set.i …
      f : TopologicalSpace.Clopens X → Finset ↑(TopologicalSpace.countableBasis X) : …
      hf : Function.Injective f
      ⊢ Countable (TopologicalSpace.Clopens X)
    -/
    exact hf.countable
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-11-12")]
alias countable_iff_second_countable := countable_iff_secondCountable


