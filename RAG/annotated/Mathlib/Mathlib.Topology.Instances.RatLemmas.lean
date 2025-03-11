local notation "ℚ∞" => OnePoint ℚ


theorem interior_compact_eq_empty (hs : IsCompact s) : interior s = ∅ :=
  isDenseEmbedding_coe_real.isDenseInducing.interior_compact_eq_empty dense_irrational hs


theorem dense_compl_compact (hs : IsCompact s) : Dense sᶜ :=
  interior_eq_empty_iff_dense_compl.1 (interior_compact_eq_empty hs)


instance cocompact_inf_nhds_neBot : NeBot (cocompact ℚ ⊓ 𝓝 p) := by
  /-
    p : Rat
    s : Set Rat
    ⊢ (Min.min (Filter.cocompact Rat) (nhds p)).NeBot
  -/
  refine (hasBasis_cocompact.inf (nhds_basis_opens _)).neBot_iff.2 ?_
  /-
    p : Rat
    s : Set Rat
    ⊢ ∀ {i : Prod (Set Rat) (Set Rat)}, And (IsCompact i.1) (And (Membership.mem i …
  -/
  rintro ⟨s, o⟩ ⟨hs, hpo, ho⟩; rw [inter_comm]
  /-
    case mk.intro.intro
    p : Rat
    s✝ s o : Set Rat
    hs : IsCompact { fst := s, snd := o }.1
    hpo : Membership.mem { fst := s, snd := o }.2 p
    ho : IsOpen { fst := s, snd := o }.2
    ⊢ (Inter.inter { fst := s, snd := o }.2 (HasCompl.compl { fst := s, snd := o } …
  -/
  exact (dense_compl_compact hs).inter_open_nonempty _ ho ⟨p, hpo⟩
  /-
    🎉 no goals
  -/


theorem not_countably_generated_cocompact : ¬IsCountablyGenerated (cocompact ℚ) := by
  /-
    ⊢ Not (Filter.cocompact Rat).IsCountablyGenerated
  -/
  intro H
  /-
    H : (Filter.cocompact Rat).IsCountablyGenerated
    ⊢ False
  -/
  rcases exists_seq_tendsto (cocompact ℚ ⊓ 𝓝 0) with ⟨x, hx⟩
  /-
    case intro
    H : (Filter.cocompact Rat).IsCountablyGenerated
    x : Nat → Rat
    hx : Filter.Tendsto x Filter.atTop (Min.min (Filter.cocompact Rat) (nhds 0))
    ⊢ False
  -/
  rw [tendsto_inf] at hx; rcases hx with ⟨hxc, hx0⟩
  obtain ⟨n, hn⟩ : ∃ n : ℕ, x n ∉ insert (0 : ℚ) (range x) :=
    (hxc.eventually hx0.isCompact_insert_range.compl_mem_cocompact).exists
  /-
    case intro.intro.intro
    H : (Filter.cocompact Rat).IsCountablyGenerated
    x : Nat → Rat
    hxc : Filter.Tendsto x Filter.atTop (Filter.cocompact Rat)
    hx0 : Filter.Tendsto x Filter.atTop (nhds 0)
    n : Nat
    hn : Not (Membership.mem (Insert.insert 0 (Set.range x)) (x n))
    ⊢ False
  -/
  exact hn (Or.inr ⟨n, rfl⟩)
  /-
    🎉 no goals
  -/


theorem not_countably_generated_nhds_infty_opc : ¬IsCountablyGenerated (𝓝 (∞ : ℚ∞)) := by
  /-
    ⊢ Not (nhds OnePoint.infty).IsCountablyGenerated
  -/
  intro
  /-
    a✝ : (nhds OnePoint.infty).IsCountablyGenerated
    ⊢ False
  -/
  have : IsCountablyGenerated (comap (OnePoint.some : ℚ → ℚ∞) (𝓝 ∞)) := by infer_instance
  /-
    a✝ : (nhds OnePoint.infty).IsCountablyGenerated
    this : (Filter.comap OnePoint.some (nhds OnePoint.infty)).IsCountablyGenerated
    ⊢ False
  -/
  rw [OnePoint.comap_coe_nhds_infty, coclosedCompact_eq_cocompact] at this
  /-
    a✝ : (nhds OnePoint.infty).IsCountablyGenerated
    this : (Filter.cocompact Rat).IsCountablyGenerated
    ⊢ False
  -/
  exact not_countably_generated_cocompact this
  /-
    🎉 no goals
  -/


theorem not_firstCountableTopology_opc : ¬FirstCountableTopology ℚ∞ := by
  /-
    ⊢ Not (FirstCountableTopology (OnePoint Rat))
  -/
  intro
  /-
    a✝ : FirstCountableTopology (OnePoint Rat)
    ⊢ False
  -/
  exact not_countably_generated_nhds_infty_opc inferInstance
  /-
    🎉 no goals
  -/


theorem not_secondCountableTopology_opc : ¬SecondCountableTopology ℚ∞ := by
  /-
    ⊢ Not (SecondCountableTopology (OnePoint Rat))
  -/
  intro
  /-
    a✝ : SecondCountableTopology (OnePoint Rat)
    ⊢ False
  -/
  exact not_firstCountableTopology_opc inferInstance
  /-
    🎉 no goals
  -/


instance : TotallyDisconnectedSpace ℚ := by
  /-
    p : Rat
    s : Set Rat
    ⊢ TotallyDisconnectedSpace Rat
  -/
  clear p s
  /-
    ⊢ TotallyDisconnectedSpace Rat
  -/
  refine ⟨fun s hsu hs x hx y hy => ?_⟩; clear hsu
  /-
    s : Set Rat
    hs : IsPreconnected s
    x : Rat
    hx : Membership.mem s x
    y : Rat
    hy : Membership.mem s y
    ⊢ Eq x y
  -/
  by_contra! H : x ≠ y
  /-
    s : Set Rat
    hs : IsPreconnected s
    x : Rat
    hx : Membership.mem s x
    y : Rat
    hy : Membership.mem s y
    H : Ne x y
    ⊢ False
  -/
  wlog hlt : x < y
    /-
      case inr
      s : Set Rat
      hs : IsPreconnected s
      x : Rat
      hx : Membership.mem s x
      y : Rat
      hy : Membership.mem s y
      H : Ne x y
      this : ∀ (s : Set Rat), IsPreconnected s → ∀ (x : Rat), Membership.mem s x → ∀ …
      hlt : Not (LT.lt x y)
      ⊢ False
    -/
  · apply this s hs y hy x hx H.symm <| H.lt_or_lt.resolve_left hlt
    /-
      🎉 no goals
    -/
  /-
    s : Set Rat
    hs : IsPreconnected s
    x : Rat
    hx : Membership.mem s x
    y : Rat
    hy : Membership.mem s y
    H : Ne x y
    hlt : LT.lt x y
    ⊢ False
  -/
  rcases exists_irrational_btwn (Rat.cast_lt.2 hlt) with ⟨z, hz, hxz, hzy⟩
  /-
    case intro.intro.intro
    s : Set Rat
    hs : IsPreconnected s
    x : Rat
    hx : Membership.mem s x
    y : Rat
    hy : Membership.mem s y
    H : Ne x y
    hlt : LT.lt x y
    z : Real
    hz : Irrational z
    hxz : LT.lt (↑x) z
    hzy : LT.lt z ↑y
    ⊢ False
  -/
  have := hs.image _ continuous_coe_real.continuousOn
  /-
    case intro.intro.intro
    s : Set Rat
    hs : IsPreconnected s
    x : Rat
    hx : Membership.mem s x
    y : Rat
    hy : Membership.mem s y
    H : Ne x y
    hlt : LT.lt x y
    z : Real
    hz : Irrational z
    hxz : LT.lt (↑x) z
    hzy : LT.lt z ↑y
    this : IsPreconnected (Set.image Rat.cast s)
    ⊢ False
  -/
  rw [isPreconnected_iff_ordConnected] at this
  have : z ∈ Rat.cast '' s :=
    this.out (mem_image_of_mem _ hx) (mem_image_of_mem _ hy) ⟨hxz.le, hzy.le⟩
  /-
    case intro.intro.intro
    s : Set Rat
    hs : IsPreconnected s
    x : Rat
    hx : Membership.mem s x
    y : Rat
    hy : Membership.mem s y
    H : Ne x y
    hlt : LT.lt x y
    z : Real
    hz : Irrational z
    hxz : LT.lt (↑x) z
    hzy : LT.lt z ↑y
    this✝ : (Set.image Rat.cast s).OrdConnected
    this : Membership.mem (Set.image Rat.cast s) z
    ⊢ False
  -/
  exact hz (image_subset_range _ _ this)
  /-
    🎉 no goals
  -/


