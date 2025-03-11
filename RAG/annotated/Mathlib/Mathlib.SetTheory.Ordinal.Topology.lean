instance : TopologicalSpace Ordinal.{u} := Preorder.topology Ordinal.{u}

instance : OrderTopology Ordinal.{u} := ⟨rfl⟩


theorem isOpen_singleton_iff : IsOpen ({a} : Set Ordinal) ↔ ¬IsLimit a := by
  /-
    a : Ordinal.{u}
    ⊢ Iff (IsOpen (Singleton.singleton a)) (Not a.IsLimit)
  -/
  refine ⟨fun h ha => ?_, fun ha => ?_⟩
  · obtain ⟨b, c, hbc, hbc'⟩ :=
      (mem_nhds_iff_exists_Ioo_subset' ⟨0, ha.pos⟩ ⟨_, lt_succ a⟩).1
        (h.mem_nhds rfl)
    /-
      case refine_1.intro.intro.intro
      a : Ordinal.{u}
      h : IsOpen (Singleton.singleton a)
      ha : a.IsLimit
      b c : Ordinal.{u}
      hbc : Membership.mem (Set.Ioo b c) a
      hbc' : HasSubset.Subset (Set.Ioo b c) (Singleton.singleton a)
      ⊢ False
    -/
    have hba := ha.succ_lt hbc.1
    /-
      case refine_1.intro.intro.intro
      a : Ordinal.{u}
      h : IsOpen (Singleton.singleton a)
      ha : a.IsLimit
      b c : Ordinal.{u}
      hbc : Membership.mem (Set.Ioo b c) a
      hbc' : HasSubset.Subset (Set.Ioo b c) (Singleton.singleton a)
      hba : LT.lt (Order.succ b) a
      ⊢ False
    -/
    exact hba.ne (hbc' ⟨lt_succ b, hba.trans hbc.2⟩)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      a : Ordinal.{u}
      ha : Not a.IsLimit
      ⊢ IsOpen (Singleton.singleton a)
    -/
  · rcases zero_or_succ_or_limit a with (rfl | ⟨b, rfl⟩ | ha')
      /-
        case refine_2.inl
        ha : Not (Ordinal.IsLimit 0)
        ⊢ IsOpen (Singleton.singleton 0)
      -/
    · rw [← bot_eq_zero, ← Set.Iic_bot, ← Iio_succ]
      /-
        case refine_2.inl
        ha : Not (Ordinal.IsLimit 0)
        ⊢ IsOpen (Set.Iio (Order.succ Bot.bot))
      -/
      exact isOpen_Iio
      /-
        🎉 no goals
      -/
      /-
        case refine_2.inr.inl.intro
        b : Ordinal.{u}
        ha : Not (Order.succ b).IsLimit
        ⊢ IsOpen (Singleton.singleton (Order.succ b))
      -/
    · rw [← Set.Icc_self, Icc_succ_left, ← Ioo_succ_right]
      /-
        case refine_2.inr.inl.intro
        b : Ordinal.{u}
        ha : Not (Order.succ b).IsLimit
        ⊢ IsOpen (Set.Ioo b (Order.succ (Order.succ b)))
      -/
      exact isOpen_Ioo
      /-
        🎉 no goals
      -/
      /-
        case refine_2.inr.inr
        a : Ordinal.{u}
        ha : Not a.IsLimit
        ha' : a.IsLimit
        ⊢ IsOpen (Singleton.singleton a)
      -/
    · exact (ha ha').elim
      /-
        🎉 no goals
      -/


protected theorem nhdsGT (a : Ordinal) : 𝓝[>] a = ⊥ := SuccOrder.nhdsGT


@[deprecated (since := "2024-12-22")] alias nhds_right' := Ordinal.nhdsGT

-- todo: generalize to a `SuccOrder`

theorem nhdsLT_eq_nhdsNE (a : Ordinal) : 𝓝[<] a = 𝓝[≠] a := by
  /-
    a : Ordinal.{u_1}
    ⊢ Eq (nhdsWithin a (Set.Iio a)) (nhdsWithin a (HasCompl.compl (Singleton.singl …
  -/
  rw [← nhdsLT_sup_nhdsGT, Ordinal.nhdsGT, sup_bot_eq]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-22")] alias nhds_left'_eq_nhds_ne := nhdsLT_eq_nhdsNE

-- todo: generalize to a `SuccOrder`

theorem nhdsLE_eq_nhds (a : Ordinal) : 𝓝[≤] a = 𝓝 a := by
  /-
    a : Ordinal.{u_1}
    ⊢ Eq (nhdsWithin a (Set.Iic a)) (nhds a)
  -/
  rw [← nhdsLE_sup_nhdsGT, SuccOrder.nhdsGT, sup_bot_eq]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-22")] alias nhds_left_eq_nhds := nhdsLE_eq_nhds

-- todo: generalize to a `SuccOrder`

theorem hasBasis_nhds_Ioc (h : a ≠ 0) : (𝓝 a).HasBasis (· < a) (Set.Ioc · a) :=
  nhdsLE_eq_nhds a ▸ nhdsLE_basis_of_exists_lt ⟨0, h.bot_lt⟩


@[deprecated (since := "2024-12-22")] alias nhdsBasis_Ioc := hasBasis_nhds_Ioc

-- todo: generalize to a `SuccOrder`

theorem nhds_eq_pure : 𝓝 a = pure a ↔ ¬IsLimit a :=
  (isOpen_singleton_iff_nhds_eq_pure _).symm.trans isOpen_singleton_iff

-- todo: generalize `Ordinal.IsLimit` and this lemma to a `SuccOrder`

theorem isOpen_iff : IsOpen s ↔ ∀ o ∈ s, IsLimit o → ∃ a < o, Set.Ioo a o ⊆ s := by
  /-
    s : Set Ordinal.{u}
    ⊢ Iff (IsOpen s) (∀ (o : Ordinal.{u}), Membership.mem s o → o.IsLimit → Exists …
  -/
  refine isOpen_iff_mem_nhds.trans <| forall₂_congr fun o ho => ?_
  /-
    s : Set Ordinal.{u}
    o : Ordinal.{u}
    ho : Membership.mem s o
    ⊢ Iff (Membership.mem (nhds o) s) (o.IsLimit → Exists fun a => And (LT.lt a o) …
  -/
  by_cases ho' : IsLimit o
    /-
      case pos
      s : Set Ordinal.{u}
      o : Ordinal.{u}
      ho : Membership.mem s o
      ho' : o.IsLimit
      ⊢ Iff (Membership.mem (nhds o) s) (o.IsLimit → Exists fun a => And (LT.lt a o) …
    -/
  · simp only [(hasBasis_nhds_Ioc ho'.ne_zero).mem_iff, ho', true_implies]
    /-
      case pos
      s : Set Ordinal.{u}
      o : Ordinal.{u}
      ho : Membership.mem s o
      ho' : o.IsLimit
      ⊢ Iff (Exists fun i => And (LT.lt i o) (HasSubset.Subset (Set.Ioc i o) s)) (Ex …
    -/
    refine exists_congr fun a => and_congr_right fun ha => ?_
    /-
      case pos
      s : Set Ordinal.{u}
      o : Ordinal.{u}
      ho : Membership.mem s o
      ho' : o.IsLimit
      a : Ordinal.{u}
      ha : LT.lt a o
      ⊢ Iff (HasSubset.Subset (Set.Ioc a o) s) (HasSubset.Subset (Set.Ioo a o) s)
    -/
    simp only [← Set.Ioo_insert_right ha, Set.insert_subset_iff, ho, true_and]
    /-
      🎉 no goals
    -/
    /-
      case neg
      s : Set Ordinal.{u}
      o : Ordinal.{u}
      ho : Membership.mem s o
      ho' : Not o.IsLimit
      ⊢ Iff (Membership.mem (nhds o) s) (o.IsLimit → Exists fun a => And (LT.lt a o) …
    -/
  · simp [nhds_eq_pure.2 ho', ho, ho']
    /-
      🎉 no goals
    -/


open List Set in
theorem mem_closure_tfae (a : Ordinal.{u}) (s : Set Ordinal) :
    TFAE [a ∈ closure s,
      a ∈ closure (s ∩ Iic a),
      (s ∩ Iic a).Nonempty ∧ sSup (s ∩ Iic a) = a,
      ∃ t, t ⊆ s ∧ t.Nonempty ∧ BddAbove t ∧ sSup t = a,
      ∃ (o : Ordinal.{u}), o ≠ 0 ∧ ∃ (f : ∀ x < o, Ordinal),
        (∀ x hx, f x hx ∈ s) ∧ bsup.{u, u} o f = a,
      ∃ (ι : Type u), Nonempty ι ∧ ∃ f : ι → Ordinal, (∀ i, f i ∈ s) ∧ ⨆ i, f i = a] := by
  tfae_have 1 → 2 := by
    simp only [mem_closure_iff_nhdsWithin_neBot, inter_comm s, nhdsWithin_inter', nhdsLE_eq_nhds]
    exact id
  tfae_have 2 → 3
  | h => by
    rcases (s ∩ Iic a).eq_empty_or_nonempty with he | hne
    · simp [he] at h
    · refine ⟨hne, (isLUB_of_mem_closure ?_ h).csSup_eq hne⟩
      exact fun x hx => hx.2
  tfae_have 3 → 4
  | h => ⟨_, inter_subset_left, h.1, bddAbove_Iic.mono inter_subset_right, h.2⟩
  tfae_have 4 → 5 := by
    rintro ⟨t, hts, hne, hbdd, rfl⟩
    have hlub : IsLUB t (sSup t) := isLUB_csSup hne hbdd
    let ⟨y, hyt⟩ := hne
    classical
      refine ⟨succ (sSup t), succ_ne_zero _, fun x _ => if x ∈ t then x else y, fun x _ => ?_, ?_⟩
      · simp only
        split_ifs with h <;> exact hts ‹_›
      · refine le_antisymm (bsup_le fun x _ => ?_) (csSup_le hne fun x hx => ?_)
        · split_ifs <;> exact hlub.1 ‹_›
        · refine (if_pos hx).symm.trans_le (le_bsup _ _ <| (hlub.1 hx).trans_lt (lt_succ _))
  tfae_have 5 → 6 := by
    rintro ⟨o, h₀, f, hfs, rfl⟩
    exact ⟨_, toType_nonempty_iff_ne_zero.2 h₀, familyOfBFamily o f, fun _ => hfs _ _, rfl⟩
  tfae_have 6 → 1 := by
    rintro ⟨ι, hne, f, hfs, rfl⟩
    exact closure_mono (range_subset_iff.2 hfs) <| csSup_mem_closure (range_nonempty f)
      (bddAbove_range.{u, u} f)
  /-
    a : Ordinal.{u}
    s : Set Ordinal.{u}
    tfae_1_to_2 : Membership.mem (closure s) a → Membership.mem (closure (Inter.in …
    tfae_2_to_3 : Membership.mem (closure (Inter.inter s (Set.Iic a))) a → And (In …
    tfae_3_to_4 : And (Inter.inter s (Set.Iic a)).Nonempty (Eq (SupSet.sSup (Inter …
    tfae_4_to_5 : (Exists fun t => And (HasSubset.Subset t s) (And t.Nonempty (And …
    tfae_5_to_6 : (Exists fun o => And (Ne o 0) (Exists fun f => And (∀ (x : Ordin …
    tfae_6_to_1 : (Exists fun ι => And (Nonempty ι) (Exists fun f => And (∀ (i : ι …
    ⊢ (List.cons (Membership.mem (closure s) a) (List.cons (Membership.mem (closur …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


theorem mem_closure_iff_iSup :
    a ∈ closure s ↔
      ∃ (ι : Type u) (_ : Nonempty ι) (f : ι → Ordinal), (∀ i, f i ∈ s) ∧ ⨆ i, f i = a := by
  /-
    s : Set Ordinal.{u}
    a : Ordinal.{u}
    ⊢ Iff (Membership.mem (closure s) a) (Exists fun ι => Exists fun x => Exists f …
  -/
  apply ((mem_closure_tfae a s).out 0 5).trans
  /-
    s : Set Ordinal.{u}
    a : Ordinal.{u}
    ⊢ Iff (Exists fun ι => And (Nonempty ι) (Exists fun f => And (∀ (i : ι), Membe …
  -/
  simp_rw [exists_prop]
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[deprecated mem_closure_iff_iSup (since := "2024-08-27")]
theorem mem_closure_iff_sup :
    a ∈ closure s ↔
      ∃ (ι : Type u) (_ : Nonempty ι) (f : ι → Ordinal), (∀ i, f i ∈ s) ∧ sup f = a :=
  mem_closure_iff_iSup


theorem mem_iff_iSup_of_isClosed (hs : IsClosed s) :
    a ∈ s ↔ ∃ (ι : Type u) (_hι : Nonempty ι) (f : ι → Ordinal),
      (∀ i, f i ∈ s) ∧ ⨆ i, f i = a := by
  /-
    s : Set Ordinal.{u}
    a : Ordinal.{u}
    hs : IsClosed s
    ⊢ Iff (Membership.mem s a) (Exists fun ι => Exists fun _hι => Exists fun f =>  …
  -/
  rw [← mem_closure_iff_iSup, hs.closure_eq]
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[deprecated mem_iff_iSup_of_isClosed (since := "2024-08-27")]
theorem mem_closed_iff_sup (hs : IsClosed s) :
    a ∈ s ↔ ∃ (ι : Type u) (_hι : Nonempty ι) (f : ι → Ordinal),
      (∀ i, f i ∈ s) ∧ sup f = a :=
  mem_iff_iSup_of_isClosed hs


theorem mem_closure_iff_bsup :
    a ∈ closure s ↔
      ∃ (o : Ordinal) (_ho : o ≠ 0) (f : ∀ a < o, Ordinal),
        (∀ i hi, f i hi ∈ s) ∧ bsup.{u, u} o f = a := by
  /-
    s : Set Ordinal.{u}
    a : Ordinal.{u}
    ⊢ Iff (Membership.mem (closure s) a) (Exists fun o => Exists fun _ho => Exists …
  -/
  apply ((mem_closure_tfae a s).out 0 4).trans
  /-
    s : Set Ordinal.{u}
    a : Ordinal.{u}
    ⊢ Iff (Exists fun o => And (Ne o 0) (Exists fun f => And (∀ (x : Ordinal.{u})  …
  -/
  simp_rw [exists_prop]
  /-
    🎉 no goals
  -/


theorem mem_closed_iff_bsup (hs : IsClosed s) :
    a ∈ s ↔
      ∃ (o : Ordinal) (_ho : o ≠ 0) (f : ∀ a < o, Ordinal),
        (∀ i hi, f i hi ∈ s) ∧ bsup.{u, u} o f = a := by
  /-
    s : Set Ordinal.{u}
    a : Ordinal.{u}
    hs : IsClosed s
    ⊢ Iff (Membership.mem s a) (Exists fun o => Exists fun _ho => Exists fun f =>  …
  -/
  rw [← mem_closure_iff_bsup, hs.closure_eq]
  /-
    🎉 no goals
  -/


theorem isClosed_iff_iSup :
    IsClosed s ↔
      ∀ {ι : Type u}, Nonempty ι → ∀ f : ι → Ordinal, (∀ i, f i ∈ s) → ⨆ i, f i ∈ s := by
  /-
    s : Set Ordinal.{u}
    ⊢ Iff (IsClosed s) (∀ {ι : Type u}, Nonempty ι → ∀ (f : ι → Ordinal.{u}), (∀ ( …
  -/
  use fun hs ι hι f hf => (mem_iff_iSup_of_isClosed hs).2 ⟨ι, hι, f, hf, rfl⟩
  /-
    case mpr
    s : Set Ordinal.{u}
    ⊢ (∀ {ι : Type u}, Nonempty ι → ∀ (f : ι → Ordinal.{u}), (∀ (i : ι), Membershi …
  -/
  rw [← closure_subset_iff_isClosed]
  /-
    case mpr
    s : Set Ordinal.{u}
    ⊢ (∀ {ι : Type u}, Nonempty ι → ∀ (f : ι → Ordinal.{u}), (∀ (i : ι), Membershi …
  -/
  intro h x hx
  /-
    case mpr
    s : Set Ordinal.{u}
    h : ∀ {ι : Type u}, Nonempty ι → ∀ (f : ι → Ordinal.{u}), (∀ (i : ι), Membersh …
    x : Ordinal.{u}
    hx : Membership.mem (closure s) x
    ⊢ Membership.mem s x
  -/
  rcases mem_closure_iff_iSup.1 hx with ⟨ι, hι, f, hf, rfl⟩
  /-
    case mpr.intro.intro.intro.intro
    s : Set Ordinal.{u}
    h : ∀ {ι : Type u}, Nonempty ι → ∀ (f : ι → Ordinal.{u}), (∀ (i : ι), Membersh …
    ι : Type u
    hι : Nonempty ι
    f : ι → Ordinal.{u}
    hf : ∀ (i : ι), Membership.mem s (f i)
    hx : Membership.mem (closure s) (iSup fun i => f i)
    ⊢ Membership.mem s (iSup fun i => f i)
  -/
  exact h hι f hf
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[deprecated mem_iff_iSup_of_isClosed (since := "2024-08-27")]
theorem isClosed_iff_sup :
    IsClosed s ↔
      ∀ {ι : Type u}, Nonempty ι → ∀ f : ι → Ordinal, (∀ i, f i ∈ s) → ⨆ i, f i ∈ s := by
  /-
    s : Set Ordinal.{u}
    ⊢ Iff (IsClosed s) (∀ {ι : Type u}, Nonempty ι → ∀ (f : ι → Ordinal.{u}), (∀ ( …
  -/
  use fun hs ι hι f hf => (mem_closed_iff_sup hs).2 ⟨ι, hι, f, hf, rfl⟩
  /-
    case mpr
    s : Set Ordinal.{u}
    ⊢ (∀ {ι : Type u}, Nonempty ι → ∀ (f : ι → Ordinal.{u}), (∀ (i : ι), Membershi …
  -/
  rw [← closure_subset_iff_isClosed]
  /-
    case mpr
    s : Set Ordinal.{u}
    ⊢ (∀ {ι : Type u}, Nonempty ι → ∀ (f : ι → Ordinal.{u}), (∀ (i : ι), Membershi …
  -/
  intro h x hx
  /-
    case mpr
    s : Set Ordinal.{u}
    h : ∀ {ι : Type u}, Nonempty ι → ∀ (f : ι → Ordinal.{u}), (∀ (i : ι), Membersh …
    x : Ordinal.{u}
    hx : Membership.mem (closure s) x
    ⊢ Membership.mem s x
  -/
  rcases mem_closure_iff_sup.1 hx with ⟨ι, hι, f, hf, rfl⟩
  /-
    case mpr.intro.intro.intro.intro
    s : Set Ordinal.{u}
    h : ∀ {ι : Type u}, Nonempty ι → ∀ (f : ι → Ordinal.{u}), (∀ (i : ι), Membersh …
    ι : Type u
    hι : Nonempty ι
    f : ι → Ordinal.{u}
    hf : ∀ (i : ι), Membership.mem s (f i)
    hx : Membership.mem (closure s) (Ordinal.sup f)
    ⊢ Membership.mem s (Ordinal.sup f)
  -/
  exact h hι f hf
  /-
    🎉 no goals
  -/


theorem isClosed_iff_bsup :
    IsClosed s ↔
      ∀ {o : Ordinal}, o ≠ 0 → ∀ f : ∀ a < o, Ordinal,
        (∀ i hi, f i hi ∈ s) → bsup.{u, u} o f ∈ s := by
  /-
    s : Set Ordinal.{u}
    ⊢ Iff (IsClosed s) (∀ {o : Ordinal.{u}}, Ne o 0 → ∀ (f : (a : Ordinal.{u}) → L …
  -/
  rw [isClosed_iff_iSup]
  /-
    s : Set Ordinal.{u}
    ⊢ Iff (∀ {ι : Type u}, Nonempty ι → ∀ (f : ι → Ordinal.{u}), (∀ (i : ι), Membe …
  -/
  refine ⟨fun H o ho f hf => H (toType_nonempty_iff_ne_zero.2 ho) _ ?_, fun H ι hι f hf => ?_⟩
    /-
      case refine_1
      s : Set Ordinal.{u}
      H : ∀ {ι : Type u}, Nonempty ι → ∀ (f : ι → Ordinal.{u}), (∀ (i : ι), Membersh …
      o : Ordinal.{u}
      ho : Ne o 0
      f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{u}
      hf : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), Membership.mem s (f i hi)
      ⊢ ∀ (i : o.toType), Membership.mem s (o.familyOfBFamily f i)
    -/
  · exact fun i => hf _ _
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      s : Set Ordinal.{u}
      H : ∀ {o : Ordinal.{u}}, Ne o 0 → ∀ (f : (a : Ordinal.{u}) → LT.lt a o → Ordin …
      ι : Type u
      hι : Nonempty ι
      f : ι → Ordinal.{u}
      hf : ∀ (i : ι), Membership.mem s (f i)
      ⊢ Membership.mem s (iSup fun i => f i)
    -/
  · rw [← Ordinal.sup, ← bsup_eq_sup]
    /-
      case refine_2
      s : Set Ordinal.{u}
      H : ∀ {o : Ordinal.{u}}, Ne o 0 → ∀ (f : (a : Ordinal.{u}) → LT.lt a o → Ordin …
      ι : Type u
      hι : Nonempty ι
      f : ι → Ordinal.{u}
      hf : ∀ (i : ι), Membership.mem s (f i)
      ⊢ Membership.mem s ((Ordinal.type WellOrderingRel).bsup (Ordinal.bfamilyOfFami …
    -/
    apply H (type_ne_zero_iff_nonempty.2 hι)
    /-
      case refine_2.a
      s : Set Ordinal.{u}
      H : ∀ {o : Ordinal.{u}}, Ne o 0 → ∀ (f : (a : Ordinal.{u}) → LT.lt a o → Ordin …
      ι : Type u
      hι : Nonempty ι
      f : ι → Ordinal.{u}
      hf : ∀ (i : ι), Membership.mem s (f i)
      ⊢ ∀ (i : Ordinal.{u}) (hi : LT.lt i (Ordinal.type WellOrderingRel)), Membershi …
    -/
    exact fun i hi => hf _
    /-
      🎉 no goals
    -/


theorem isLimit_of_mem_frontier (ha : a ∈ frontier s) : IsLimit a := by
  /-
    s : Set Ordinal.{u}
    a : Ordinal.{u}
    ha : Membership.mem (frontier s) a
    ⊢ a.IsLimit
  -/
  simp only [frontier_eq_closure_inter_closure, Set.mem_inter_iff, mem_closure_iff] at ha
  /-
    s : Set Ordinal.{u}
    a : Ordinal.{u}
    ha : And (∀ (o : Set Ordinal.{u}), IsOpen o → Membership.mem o a → (Inter.inte …
    ⊢ a.IsLimit
  -/
  by_contra h
  /-
    s : Set Ordinal.{u}
    a : Ordinal.{u}
    ha : And (∀ (o : Set Ordinal.{u}), IsOpen o → Membership.mem o a → (Inter.inte …
    h : Not a.IsLimit
    ⊢ False
  -/
  rw [← isOpen_singleton_iff] at h
  /-
    s : Set Ordinal.{u}
    a : Ordinal.{u}
    ha : And (∀ (o : Set Ordinal.{u}), IsOpen o → Membership.mem o a → (Inter.inte …
    h : IsOpen (Singleton.singleton a)
    ⊢ False
  -/
  rcases ha.1 _ h rfl with ⟨b, hb, hb'⟩
  /-
    case intro.intro
    s : Set Ordinal.{u}
    a : Ordinal.{u}
    ha : And (∀ (o : Set Ordinal.{u}), IsOpen o → Membership.mem o a → (Inter.inte …
    h : IsOpen (Singleton.singleton a)
    b : Ordinal.{u}
    hb : Membership.mem (Singleton.singleton a) b
    hb' : Membership.mem s b
    ⊢ False
  -/
  rcases ha.2 _ h rfl with ⟨c, hc, hc'⟩
  /-
    case intro.intro.intro.intro
    s : Set Ordinal.{u}
    a : Ordinal.{u}
    ha : And (∀ (o : Set Ordinal.{u}), IsOpen o → Membership.mem o a → (Inter.inte …
    h : IsOpen (Singleton.singleton a)
    b : Ordinal.{u}
    hb : Membership.mem (Singleton.singleton a) b
    hb' : Membership.mem s b
    c : Ordinal.{u}
    hc : Membership.mem (Singleton.singleton a) c
    hc' : Membership.mem (HasCompl.compl s) c
    ⊢ False
  -/
  rw [Set.mem_singleton_iff] at *
  /-
    case intro.intro.intro.intro
    s : Set Ordinal.{u}
    a : Ordinal.{u}
    ha : And (∀ (o : Set Ordinal.{u}), IsOpen o → Membership.mem o a → (Inter.inte …
    h : IsOpen (Singleton.singleton a)
    b : Ordinal.{u}
    hb : Eq b a
    hb' : Membership.mem s b
    c : Ordinal.{u}
    hc : Eq c a
    hc' : Membership.mem (HasCompl.compl s) c
    ⊢ False
  -/
  subst hb; subst hc
  /-
    case intro.intro.intro.intro
    s : Set Ordinal.{u}
    c : Ordinal.{u}
    hc' : Membership.mem (HasCompl.compl s) c
    hb' : Membership.mem s c
    ha : And (∀ (o : Set Ordinal.{u}), IsOpen o → Membership.mem o c → (Inter.inte …
    h : IsOpen (Singleton.singleton c)
    ⊢ False
  -/
  exact hc' hb'
  /-
    🎉 no goals
  -/


theorem isNormal_iff_strictMono_and_continuous (f : Ordinal.{u} → Ordinal.{u}) :
    IsNormal f ↔ StrictMono f ∧ Continuous f := by
  /-
    f : Ordinal.{u} → Ordinal.{u}
    ⊢ Iff (Ordinal.IsNormal f) (And (StrictMono f) (Continuous f))
  -/
  refine ⟨fun h => ⟨h.strictMono, ?_⟩, ?_⟩
    /-
      case refine_1
      f : Ordinal.{u} → Ordinal.{u}
      h : Ordinal.IsNormal f
      ⊢ Continuous f
    -/
  · rw [continuous_def]
    /-
      case refine_1
      f : Ordinal.{u} → Ordinal.{u}
      h : Ordinal.IsNormal f
      ⊢ ∀ (s : Set Ordinal.{u}), IsOpen s → IsOpen (Set.preimage f s)
    -/
    intro s hs
    /-
      case refine_1
      f : Ordinal.{u} → Ordinal.{u}
      h : Ordinal.IsNormal f
      s : Set Ordinal.{u}
      hs : IsOpen s
      ⊢ IsOpen (Set.preimage f s)
    -/
    rw [isOpen_iff] at *
    /-
      case refine_1
      f : Ordinal.{u} → Ordinal.{u}
      h : Ordinal.IsNormal f
      s : Set Ordinal.{u}
      hs : ∀ (o : Ordinal.{u}), Membership.mem s o → o.IsLimit → Exists fun a => And …
      ⊢ ∀ (o : Ordinal.{u}), Membership.mem (Set.preimage f s) o → o.IsLimit → Exist …
    -/
    intro o ho ho'
    /-
      case refine_1
      f : Ordinal.{u} → Ordinal.{u}
      h : Ordinal.IsNormal f
      s : Set Ordinal.{u}
      hs : ∀ (o : Ordinal.{u}), Membership.mem s o → o.IsLimit → Exists fun a => And …
      o : Ordinal.{u}
      ho : Membership.mem (Set.preimage f s) o
      ho' : o.IsLimit
      ⊢ Exists fun a => And (LT.lt a o) (HasSubset.Subset (Set.Ioo a o) (Set.preimag …
    -/
    rcases hs _ ho (h.isLimit ho') with ⟨a, ha, has⟩
    /-
      case refine_1.intro.intro
      f : Ordinal.{u} → Ordinal.{u}
      h : Ordinal.IsNormal f
      s : Set Ordinal.{u}
      hs : ∀ (o : Ordinal.{u}), Membership.mem s o → o.IsLimit → Exists fun a => And …
      o : Ordinal.{u}
      ho : Membership.mem (Set.preimage f s) o
      ho' : o.IsLimit
      a : Ordinal.{u}
      ha : LT.lt a (f o)
      has : HasSubset.Subset (Set.Ioo a (f o)) s
      ⊢ Exists fun a => And (LT.lt a o) (HasSubset.Subset (Set.Ioo a o) (Set.preimag …
    -/
    rw [← IsNormal.bsup_eq.{u, u} h ho', lt_bsup] at ha
    /-
      case refine_1.intro.intro
      f : Ordinal.{u} → Ordinal.{u}
      h : Ordinal.IsNormal f
      s : Set Ordinal.{u}
      hs : ∀ (o : Ordinal.{u}), Membership.mem s o → o.IsLimit → Exists fun a => And …
      o : Ordinal.{u}
      ho : Membership.mem (Set.preimage f s) o
      ho' : o.IsLimit
      a : Ordinal.{u}
      ha : Exists fun i => Exists fun hi => LT.lt a (f i)
      has : HasSubset.Subset (Set.Ioo a (f o)) s
      ⊢ Exists fun a => And (LT.lt a o) (HasSubset.Subset (Set.Ioo a o) (Set.preimag …
    -/
    rcases ha with ⟨b, hb, hab⟩
    exact
      ⟨b, hb, fun c hc =>
        Set.mem_preimage.2 (has ⟨hab.trans (h.strictMono hc.1), h.strictMono hc.2⟩)⟩
    /-
      case refine_2
      f : Ordinal.{u} → Ordinal.{u}
      ⊢ And (StrictMono f) (Continuous f) → Ordinal.IsNormal f
    -/
  · rw [isNormal_iff_strictMono_limit]
    /-
      case refine_2
      f : Ordinal.{u} → Ordinal.{u}
      ⊢ And (StrictMono f) (Continuous f) → And (StrictMono f) (∀ (o : Ordinal.{u}), …
    -/
    rintro ⟨h, h'⟩
    /-
      case refine_2.intro
      f : Ordinal.{u} → Ordinal.{u}
      h : StrictMono f
      h' : Continuous f
      ⊢ And (StrictMono f) (∀ (o : Ordinal.{u}), o.IsLimit → ∀ (a : Ordinal.{u}), (∀ …
    -/
    refine ⟨h, fun o ho a h => ?_⟩
    /-
      case refine_2.intro
      f : Ordinal.{u} → Ordinal.{u}
      h✝ : StrictMono f
      h' : Continuous f
      o : Ordinal.{u}
      ho : o.IsLimit
      a : Ordinal.{u}
      h : ∀ (b : Ordinal.{u}), LT.lt b o → LE.le (f b) a
      ⊢ LE.le (f o) a
    -/
    suffices o ∈ f ⁻¹' Set.Iic a from Set.mem_preimage.1 this
    /-
      case refine_2.intro
      f : Ordinal.{u} → Ordinal.{u}
      h✝ : StrictMono f
      h' : Continuous f
      o : Ordinal.{u}
      ho : o.IsLimit
      a : Ordinal.{u}
      h : ∀ (b : Ordinal.{u}), LT.lt b o → LE.le (f b) a
      ⊢ Membership.mem (Set.preimage f (Set.Iic a)) o
    -/
    rw [mem_iff_iSup_of_isClosed (IsClosed.preimage h' (@isClosed_Iic _ _ _ _ a))]
    exact
      ⟨_, toType_nonempty_iff_ne_zero.2 ho.ne_zero, typein (· < ·), fun i => h _ (typein_lt_self i),
        sup_typein_limit fun _ ↦ ho.succ_lt⟩


theorem enumOrd_isNormal_iff_isClosed (hs : ¬ BddAbove s) :
    IsNormal (enumOrd s) ↔ IsClosed s := by
  /-
    s : Set Ordinal.{u}
    hs : Not (BddAbove s)
    ⊢ Iff (Ordinal.IsNormal (Ordinal.enumOrd s)) (IsClosed s)
  -/
  have Hs := enumOrd_strictMono hs
  refine
    ⟨fun h => isClosed_iff_iSup.2 fun {ι} hι f hf => ?_, fun h =>
      (isNormal_iff_strictMono_limit _).2 ⟨Hs, fun a ha o H => ?_⟩⟩
    /-
      case refine_1
      s : Set Ordinal.{u}
      hs : Not (BddAbove s)
      Hs : StrictMono (Ordinal.enumOrd s)
      h : Ordinal.IsNormal (Ordinal.enumOrd s)
      ι : Type u
      hι : Nonempty ι
      f : ι → Ordinal.{u}
      hf : ∀ (i : ι), Membership.mem s (f i)
      ⊢ Membership.mem s (iSup fun i => f i)
    -/
  · let g : ι → Ordinal.{u} := fun i => (enumOrdOrderIso s hs).symm ⟨_, hf i⟩
    suffices enumOrd s (⨆ i, g i) = ⨆ i, f i by
      rw [← this]
      exact enumOrd_mem hs _
    /-
      case refine_1
      s : Set Ordinal.{u}
      hs : Not (BddAbove s)
      Hs : StrictMono (Ordinal.enumOrd s)
      h : Ordinal.IsNormal (Ordinal.enumOrd s)
      ι : Type u
      hι : Nonempty ι
      f : ι → Ordinal.{u}
      hf : ∀ (i : ι), Membership.mem s (f i)
      g : ι → Ordinal.{u} := fun i => (Ordinal.enumOrdOrderIso s hs).symm ⟨f i, ⋯⟩
      ⊢ Eq (Ordinal.enumOrd s (iSup fun i => g i)) (iSup fun i => f i)
    -/
    rw [IsNormal.map_iSup h g]
    /-
      case refine_1
      s : Set Ordinal.{u}
      hs : Not (BddAbove s)
      Hs : StrictMono (Ordinal.enumOrd s)
      h : Ordinal.IsNormal (Ordinal.enumOrd s)
      ι : Type u
      hι : Nonempty ι
      f : ι → Ordinal.{u}
      hf : ∀ (i : ι), Membership.mem s (f i)
      g : ι → Ordinal.{u} := fun i => (Ordinal.enumOrdOrderIso s hs).symm ⟨f i, ⋯⟩
      ⊢ Eq (iSup fun i => Ordinal.enumOrd s (g i)) (iSup fun i => f i)
    -/
    congr
    /-
      case refine_1.e_s
      s : Set Ordinal.{u}
      hs : Not (BddAbove s)
      Hs : StrictMono (Ordinal.enumOrd s)
      h : Ordinal.IsNormal (Ordinal.enumOrd s)
      ι : Type u
      hι : Nonempty ι
      f : ι → Ordinal.{u}
      hf : ∀ (i : ι), Membership.mem s (f i)
      g : ι → Ordinal.{u} := fun i => (Ordinal.enumOrdOrderIso s hs).symm ⟨f i, ⋯⟩
      ⊢ Eq (fun i => Ordinal.enumOrd s (g i)) fun i => f i
    -/
    ext x
    /-
      case refine_1.e_s.h
      s : Set Ordinal.{u}
      hs : Not (BddAbove s)
      Hs : StrictMono (Ordinal.enumOrd s)
      h : Ordinal.IsNormal (Ordinal.enumOrd s)
      ι : Type u
      hι : Nonempty ι
      f : ι → Ordinal.{u}
      hf : ∀ (i : ι), Membership.mem s (f i)
      g : ι → Ordinal.{u} := fun i => (Ordinal.enumOrdOrderIso s hs).symm ⟨f i, ⋯⟩
      x : ι
      ⊢ Eq (Ordinal.enumOrd s (g x)) (f x)
    -/
    change (enumOrdOrderIso s hs _).val = f x
    /-
      case refine_1.e_s.h
      s : Set Ordinal.{u}
      hs : Not (BddAbove s)
      Hs : StrictMono (Ordinal.enumOrd s)
      h : Ordinal.IsNormal (Ordinal.enumOrd s)
      ι : Type u
      hι : Nonempty ι
      f : ι → Ordinal.{u}
      hf : ∀ (i : ι), Membership.mem s (f i)
      g : ι → Ordinal.{u} := fun i => (Ordinal.enumOrdOrderIso s hs).symm ⟨f i, ⋯⟩
      x : ι
      ⊢ Eq (↑((Ordinal.enumOrdOrderIso s hs) (g x))) (f x)
    -/
    rw [OrderIso.apply_symm_apply]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      s : Set Ordinal.{u}
      hs : Not (BddAbove s)
      Hs : StrictMono (Ordinal.enumOrd s)
      h : IsClosed s
      a : Ordinal.{u}
      ha : a.IsLimit
      o : Ordinal.{u}
      H : ∀ (b : Ordinal.{u}), LT.lt b a → LE.le (Ordinal.enumOrd s b) o
      ⊢ LE.le (Ordinal.enumOrd s a) o
    -/
  · rw [isClosed_iff_bsup] at h
    suffices enumOrd s a ≤ bsup.{u, u} a fun b (_ : b < a) => enumOrd s b from
      this.trans (bsup_le H)
    obtain ⟨b, hb⟩ := enumOrd_surjective hs (h ha.ne_zero (fun b _ => enumOrd s b)
      fun b _ => enumOrd_mem hs b)
    /-
      case refine_2.intro
      s : Set Ordinal.{u}
      hs : Not (BddAbove s)
      Hs : StrictMono (Ordinal.enumOrd s)
      h : ∀ {o : Ordinal.{u}}, Ne o 0 → ∀ (f : (a : Ordinal.{u}) → LT.lt a o → Ordin …
      a : Ordinal.{u}
      ha : a.IsLimit
      o : Ordinal.{u}
      H : ∀ (b : Ordinal.{u}), LT.lt b a → LE.le (Ordinal.enumOrd s b) o
      b : Ordinal.{u}
      hb : Eq (Ordinal.enumOrd s b) (a.bsup fun b x => Ordinal.enumOrd s b)
      ⊢ LE.le (Ordinal.enumOrd s a) (a.bsup fun b x => Ordinal.enumOrd s b)
    -/
    rw [← hb]
    /-
      case refine_2.intro
      s : Set Ordinal.{u}
      hs : Not (BddAbove s)
      Hs : StrictMono (Ordinal.enumOrd s)
      h : ∀ {o : Ordinal.{u}}, Ne o 0 → ∀ (f : (a : Ordinal.{u}) → LT.lt a o → Ordin …
      a : Ordinal.{u}
      ha : a.IsLimit
      o : Ordinal.{u}
      H : ∀ (b : Ordinal.{u}), LT.lt b a → LE.le (Ordinal.enumOrd s b) o
      b : Ordinal.{u}
      hb : Eq (Ordinal.enumOrd s b) (a.bsup fun b x => Ordinal.enumOrd s b)
      ⊢ LE.le (Ordinal.enumOrd s a) (Ordinal.enumOrd s b)
    -/
    apply Hs.monotone
    /-
      case refine_2.intro.a
      s : Set Ordinal.{u}
      hs : Not (BddAbove s)
      Hs : StrictMono (Ordinal.enumOrd s)
      h : ∀ {o : Ordinal.{u}}, Ne o 0 → ∀ (f : (a : Ordinal.{u}) → LT.lt a o → Ordin …
      a : Ordinal.{u}
      ha : a.IsLimit
      o : Ordinal.{u}
      H : ∀ (b : Ordinal.{u}), LT.lt b a → LE.le (Ordinal.enumOrd s b) o
      b : Ordinal.{u}
      hb : Eq (Ordinal.enumOrd s b) (a.bsup fun b x => Ordinal.enumOrd s b)
      ⊢ LE.le a b
    -/
    by_contra! hba
    /-
      case refine_2.intro.a
      s : Set Ordinal.{u}
      hs : Not (BddAbove s)
      Hs : StrictMono (Ordinal.enumOrd s)
      h : ∀ {o : Ordinal.{u}}, Ne o 0 → ∀ (f : (a : Ordinal.{u}) → LT.lt a o → Ordin …
      a : Ordinal.{u}
      ha : a.IsLimit
      o : Ordinal.{u}
      H : ∀ (b : Ordinal.{u}), LT.lt b a → LE.le (Ordinal.enumOrd s b) o
      b : Ordinal.{u}
      hb : Eq (Ordinal.enumOrd s b) (a.bsup fun b x => Ordinal.enumOrd s b)
      hba : LT.lt b a
      ⊢ False
    -/
    apply (Hs (lt_succ b)).not_le
    /-
      case refine_2.intro.a
      s : Set Ordinal.{u}
      hs : Not (BddAbove s)
      Hs : StrictMono (Ordinal.enumOrd s)
      h : ∀ {o : Ordinal.{u}}, Ne o 0 → ∀ (f : (a : Ordinal.{u}) → LT.lt a o → Ordin …
      a : Ordinal.{u}
      ha : a.IsLimit
      o : Ordinal.{u}
      H : ∀ (b : Ordinal.{u}), LT.lt b a → LE.le (Ordinal.enumOrd s b) o
      b : Ordinal.{u}
      hb : Eq (Ordinal.enumOrd s b) (a.bsup fun b x => Ordinal.enumOrd s b)
      hba : LT.lt b a
      ⊢ LE.le (Ordinal.enumOrd s (Order.succ b)) (Ordinal.enumOrd s b)
    -/
    rw [hb]
    /-
      case refine_2.intro.a
      s : Set Ordinal.{u}
      hs : Not (BddAbove s)
      Hs : StrictMono (Ordinal.enumOrd s)
      h : ∀ {o : Ordinal.{u}}, Ne o 0 → ∀ (f : (a : Ordinal.{u}) → LT.lt a o → Ordin …
      a : Ordinal.{u}
      ha : a.IsLimit
      o : Ordinal.{u}
      H : ∀ (b : Ordinal.{u}), LT.lt b a → LE.le (Ordinal.enumOrd s b) o
      b : Ordinal.{u}
      hb : Eq (Ordinal.enumOrd s b) (a.bsup fun b x => Ordinal.enumOrd s b)
      hba : LT.lt b a
      ⊢ LE.le (Ordinal.enumOrd s (Order.succ b)) (a.bsup fun b x => Ordinal.enumOrd  …
    -/
    exact le_bsup.{u, u} _ _ (ha.succ_lt hba)
    /-
      🎉 no goals
    -/


/-- An ordinal is an accumulation point of a set of ordinals if it is positive and there
are elements in the set arbitrarily close to the ordinal from below. -/
def IsAcc (o : Ordinal) (S : Set Ordinal) : Prop :=
  AccPt o (𝓟 S)


/-- A set of ordinals is closed below an ordinal if it contains all of
its accumulation points below the ordinal. -/
def IsClosedBelow (S : Set Ordinal) (o : Ordinal) : Prop :=
  IsClosed (Iio o ↓∩ S)


theorem isAcc_iff (o : Ordinal) (S : Set Ordinal) : o.IsAcc S ↔
    o ≠ 0 ∧ ∀ p < o, (S ∩ Ioo p o).Nonempty := by
  /-
    o : Ordinal.{u_1}
    S : Set Ordinal.{u_1}
    ⊢ Iff (o.IsAcc S) (And (Ne o 0) (∀ (p : Ordinal.{u_1}), LT.lt p o → (Inter.int …
  -/
  dsimp [IsAcc]
  /-
    o : Ordinal.{u_1}
    S : Set Ordinal.{u_1}
    ⊢ Iff (AccPt o (Filter.principal S)) (And (Not (Eq o 0)) (∀ (p : Ordinal.{u_1} …
  -/
  constructor
    /-
      case mp
      o : Ordinal.{u_1}
      S : Set Ordinal.{u_1}
      ⊢ AccPt o (Filter.principal S) → And (Not (Eq o 0)) (∀ (p : Ordinal.{u_1}), LT …
    -/
  · rw [accPt_iff_nhds]
    /-
      case mp
      o : Ordinal.{u_1}
      S : Set Ordinal.{u_1}
      ⊢ (∀ (U : Set Ordinal.{u_1}), Membership.mem (nhds o) U → Exists fun y => And  …
    -/
    intro h
    /-
      case mp
      o : Ordinal.{u_1}
      S : Set Ordinal.{u_1}
      h : ∀ (U : Set Ordinal.{u_1}), Membership.mem (nhds o) U → Exists fun y => And …
      ⊢ And (Not (Eq o 0)) (∀ (p : Ordinal.{u_1}), LT.lt p o → (Inter.inter S (Set.I …
    -/
    constructor
      /-
        case mp.left
        o : Ordinal.{u_1}
        S : Set Ordinal.{u_1}
        h : ∀ (U : Set Ordinal.{u_1}), Membership.mem (nhds o) U → Exists fun y => And …
        ⊢ Not (Eq o 0)
      -/
    · rintro rfl
      /-
        case mp.left
        S : Set Ordinal.{u_1}
        h : ∀ (U : Set Ordinal.{u_1}), Membership.mem (nhds 0) U → Exists fun y => And …
        ⊢ False
      -/
      obtain ⟨x, hx⟩ := h (Iio 1) (Iio_mem_nhds zero_lt_one)
      /-
        case mp.left.intro
        S : Set Ordinal.{u_1}
        h : ∀ (U : Set Ordinal.{u_1}), Membership.mem (nhds 0) U → Exists fun y => And …
        x : Ordinal.{u_1}
        hx : And (Membership.mem (Inter.inter (Set.Iio 1) S) x) (Ne x 0)
        ⊢ False
      -/
      exact hx.2 <| lt_one_iff_zero.mp hx.1.1
      /-
        🎉 no goals
      -/
      /-
        case mp.right
        o : Ordinal.{u_1}
        S : Set Ordinal.{u_1}
        h : ∀ (U : Set Ordinal.{u_1}), Membership.mem (nhds o) U → Exists fun y => And …
        ⊢ ∀ (p : Ordinal.{u_1}), LT.lt p o → (Inter.inter S (Set.Ioo p o)).Nonempty
      -/
    · intro p plt
      /-
        case mp.right
        o : Ordinal.{u_1}
        S : Set Ordinal.{u_1}
        h : ∀ (U : Set Ordinal.{u_1}), Membership.mem (nhds o) U → Exists fun y => And …
        p : Ordinal.{u_1}
        plt : LT.lt p o
        ⊢ (Inter.inter S (Set.Ioo p o)).Nonempty
      -/
      obtain ⟨x, hx⟩ := h (Ioo p (o + 1)) <| Ioo_mem_nhds plt (lt_succ o)
      /-
        case mp.right.intro
        o : Ordinal.{u_1}
        S : Set Ordinal.{u_1}
        h : ∀ (U : Set Ordinal.{u_1}), Membership.mem (nhds o) U → Exists fun y => And …
        p : Ordinal.{u_1}
        plt : LT.lt p o
        x : Ordinal.{u_1}
        hx : And (Membership.mem (Inter.inter (Set.Ioo p (HAdd.hAdd o 1)) S) x) (Ne x o)
        ⊢ (Inter.inter S (Set.Ioo p o)).Nonempty
      -/
      use x
      /-
        case h
        o : Ordinal.{u_1}
        S : Set Ordinal.{u_1}
        h : ∀ (U : Set Ordinal.{u_1}), Membership.mem (nhds o) U → Exists fun y => And …
        p : Ordinal.{u_1}
        plt : LT.lt p o
        x : Ordinal.{u_1}
        hx : And (Membership.mem (Inter.inter (Set.Ioo p (HAdd.hAdd o 1)) S) x) (Ne x o)
        ⊢ Membership.mem (Inter.inter S (Set.Ioo p o)) x
      -/
      refine ⟨hx.1.2, ⟨hx.1.1.1, lt_of_le_of_ne ?_ hx.2⟩⟩
      /-
        case h
        o : Ordinal.{u_1}
        S : Set Ordinal.{u_1}
        h : ∀ (U : Set Ordinal.{u_1}), Membership.mem (nhds o) U → Exists fun y => And …
        p : Ordinal.{u_1}
        plt : LT.lt p o
        x : Ordinal.{u_1}
        hx : And (Membership.mem (Inter.inter (Set.Ioo p (HAdd.hAdd o 1)) S) x) (Ne x o)
        ⊢ LE.le x o
      -/
      have := hx.1.1.2
      /-
        case h
        o : Ordinal.{u_1}
        S : Set Ordinal.{u_1}
        h : ∀ (U : Set Ordinal.{u_1}), Membership.mem (nhds o) U → Exists fun y => And …
        p : Ordinal.{u_1}
        plt : LT.lt p o
        x : Ordinal.{u_1}
        hx : And (Membership.mem (Inter.inter (Set.Ioo p (HAdd.hAdd o 1)) S) x) (Ne x o)
        this : LT.lt x (HAdd.hAdd o 1)
        ⊢ LE.le x o
      -/
      rwa [← succ_eq_add_one, lt_succ_iff] at this
      /-
        🎉 no goals
      -/
    /-
      case mpr
      o : Ordinal.{u_1}
      S : Set Ordinal.{u_1}
      ⊢ And (Not (Eq o 0)) (∀ (p : Ordinal.{u_1}), LT.lt p o → (Inter.inter S (Set.I …
    -/
  · rw [accPt_iff_nhds]
    /-
      case mpr
      o : Ordinal.{u_1}
      S : Set Ordinal.{u_1}
      ⊢ And (Not (Eq o 0)) (∀ (p : Ordinal.{u_1}), LT.lt p o → (Inter.inter S (Set.I …
    -/
    intro h u umem
    /-
      case mpr
      o : Ordinal.{u_1}
      S : Set Ordinal.{u_1}
      h : And (Not (Eq o 0)) (∀ (p : Ordinal.{u_1}), LT.lt p o → (Inter.inter S (Set …
      u : Set Ordinal.{u_1}
      umem : Membership.mem (nhds o) u
      ⊢ Exists fun y => And (Membership.mem (Inter.inter u S) y) (Ne y o)
    -/
    obtain ⟨l, hl⟩ := exists_Ioc_subset_of_mem_nhds umem ⟨0, Ordinal.pos_iff_ne_zero.mpr h.1⟩
    /-
      case mpr.intro
      o : Ordinal.{u_1}
      S : Set Ordinal.{u_1}
      h : And (Not (Eq o 0)) (∀ (p : Ordinal.{u_1}), LT.lt p o → (Inter.inter S (Set …
      u : Set Ordinal.{u_1}
      umem : Membership.mem (nhds o) u
      l : Ordinal.{u_1}
      hl : And (LT.lt l o) (HasSubset.Subset (Set.Ioc l o) u)
      ⊢ Exists fun y => And (Membership.mem (Inter.inter u S) y) (Ne y o)
    -/
    obtain ⟨x, hx⟩ := h.2 l hl.1
    /-
      case mpr.intro.intro
      o : Ordinal.{u_1}
      S : Set Ordinal.{u_1}
      h : And (Not (Eq o 0)) (∀ (p : Ordinal.{u_1}), LT.lt p o → (Inter.inter S (Set …
      u : Set Ordinal.{u_1}
      umem : Membership.mem (nhds o) u
      l : Ordinal.{u_1}
      hl : And (LT.lt l o) (HasSubset.Subset (Set.Ioc l o) u)
      x : Ordinal.{u_1}
      hx : Membership.mem (Inter.inter S (Set.Ioo l o)) x
      ⊢ Exists fun y => And (Membership.mem (Inter.inter u S) y) (Ne y o)
    -/
    use x
    /-
      case h
      o : Ordinal.{u_1}
      S : Set Ordinal.{u_1}
      h : And (Not (Eq o 0)) (∀ (p : Ordinal.{u_1}), LT.lt p o → (Inter.inter S (Set …
      u : Set Ordinal.{u_1}
      umem : Membership.mem (nhds o) u
      l : Ordinal.{u_1}
      hl : And (LT.lt l o) (HasSubset.Subset (Set.Ioc l o) u)
      x : Ordinal.{u_1}
      hx : Membership.mem (Inter.inter S (Set.Ioo l o)) x
      ⊢ And (Membership.mem (Inter.inter u S) x) (Ne x o)
    -/
    exact ⟨⟨hl.2 ⟨hx.2.1, hx.2.2.le⟩, hx.1⟩, hx.2.2.ne⟩
    /-
      🎉 no goals
    -/


theorem IsAcc.forall_lt {o : Ordinal} {S : Set Ordinal} (h : o.IsAcc S) :
    ∀ p < o, (S ∩ Ioo p o).Nonempty := ((isAcc_iff _ _).mp h).2


theorem IsAcc.pos {o : Ordinal} {S : Set Ordinal} (h : o.IsAcc S) :
    0 < o := Ordinal.pos_iff_ne_zero.mpr ((isAcc_iff _ _).mp h).1


theorem IsAcc.isLimit {o : Ordinal} {S : Set Ordinal} (h : o.IsAcc S) : IsLimit o := by
  /-
    o : Ordinal.{u_1}
    S : Set Ordinal.{u_1}
    h : o.IsAcc S
    ⊢ o.IsLimit
  -/
  rw [isAcc_iff] at h
  /-
    o : Ordinal.{u_1}
    S : Set Ordinal.{u_1}
    h : And (Ne o 0) (∀ (p : Ordinal.{u_1}), LT.lt p o → (Inter.inter S (Set.Ioo p …
    ⊢ o.IsLimit
  -/
  refine isLimit_of_not_succ_of_ne_zero (fun ⟨x, hx⟩ ↦ ?_) h.1
  /-
    o : Ordinal.{u_1}
    S : Set Ordinal.{u_1}
    h : And (Ne o 0) (∀ (p : Ordinal.{u_1}), LT.lt p o → (Inter.inter S (Set.Ioo p …
    x✝ : Exists fun a => Eq o (Order.succ a)
    x : Ordinal.{u_1}
    hx : Eq o (Order.succ x)
    ⊢ False
  -/
  rcases h.2 x (lt_of_lt_of_le (lt_succ x) hx.symm.le) with ⟨p, hp⟩
  /-
    case intro
    o : Ordinal.{u_1}
    S : Set Ordinal.{u_1}
    h : And (Ne o 0) (∀ (p : Ordinal.{u_1}), LT.lt p o → (Inter.inter S (Set.Ioo p …
    x✝ : Exists fun a => Eq o (Order.succ a)
    x : Ordinal.{u_1}
    hx : Eq o (Order.succ x)
    p : Ordinal.{u_1}
    hp : Membership.mem (Inter.inter S (Set.Ioo x o)) p
    ⊢ False
  -/
  exact (hx.symm ▸ (succ_le_iff.mpr hp.2.1)).not_lt hp.2.2
  /-
    🎉 no goals
  -/


theorem IsAcc.mono {o : Ordinal} {S T : Set Ordinal} (h : S ⊆ T) (ho : o.IsAcc S) :
    o.IsAcc T := by
  /-
    o : Ordinal.{u_1}
    S T : Set Ordinal.{u_1}
    h : HasSubset.Subset S T
    ho : o.IsAcc S
    ⊢ o.IsAcc T
  -/
  rw [isAcc_iff] at *
  /-
    o : Ordinal.{u_1}
    S T : Set Ordinal.{u_1}
    h : HasSubset.Subset S T
    ho : And (Ne o 0) (∀ (p : Ordinal.{u_1}), LT.lt p o → (Inter.inter S (Set.Ioo  …
    ⊢ And (Ne o 0) (∀ (p : Ordinal.{u_1}), LT.lt p o → (Inter.inter T (Set.Ioo p o …
  -/
  exact ⟨ho.1, fun p plto ↦ (ho.2 p plto).casesOn fun s hs ↦ ⟨s, h hs.1, hs.2⟩⟩
  /-
    🎉 no goals
  -/


theorem IsAcc.inter_Ioo_nonempty {o : Ordinal} {S : Set Ordinal} (hS : o.IsAcc S)
    {p : Ordinal} (hp : p < o) : (S ∩ Ioo p o).Nonempty := hS.forall_lt p hp

-- todo: prove this for a general linear `SuccOrder`.

theorem accPt_subtype {p o : Ordinal} (S : Set Ordinal) (hpo : p < o) :
    AccPt p (𝓟 S) ↔ AccPt ⟨p, hpo⟩ (𝓟 (Iio o ↓∩ S)) := by
  /-
    p o : Ordinal.{u_1}
    S : Set Ordinal.{u_1}
    hpo : LT.lt p o
    ⊢ Iff (AccPt p (Filter.principal S)) (AccPt ⟨p, hpo⟩ (Filter.principal (Set.pr …
  -/
  constructor
    /-
      case mp
      p o : Ordinal.{u_1}
      S : Set Ordinal.{u_1}
      hpo : LT.lt p o
      ⊢ AccPt p (Filter.principal S) → AccPt ⟨p, hpo⟩ (Filter.principal (Set.preimag …
    -/
  · intro h
    /-
      case mp
      p o : Ordinal.{u_1}
      S : Set Ordinal.{u_1}
      hpo : LT.lt p o
      h : AccPt p (Filter.principal S)
      ⊢ AccPt ⟨p, hpo⟩ (Filter.principal (Set.preimage Subtype.val S))
    -/
    have plim : p.IsLimit := IsAcc.isLimit h
    /-
      case mp
      p o : Ordinal.{u_1}
      S : Set Ordinal.{u_1}
      hpo : LT.lt p o
      h : AccPt p (Filter.principal S)
      plim : p.IsLimit
      ⊢ AccPt ⟨p, hpo⟩ (Filter.principal (Set.preimage Subtype.val S))
    -/
    rw [accPt_iff_nhds] at *
    /-
      case mp
      p o : Ordinal.{u_1}
      S : Set Ordinal.{u_1}
      hpo : LT.lt p o
      h : ∀ (U : Set Ordinal.{u_1}), Membership.mem (nhds p) U → Exists fun y => And …
      plim : p.IsLimit
      ⊢ ∀ (U : Set ↑(Set.Iio o)), Membership.mem (nhds ⟨p, hpo⟩) U → Exists fun y => …
    -/
    intro u hu
    /-
      case mp
      p o : Ordinal.{u_1}
      S : Set Ordinal.{u_1}
      hpo : LT.lt p o
      h : ∀ (U : Set Ordinal.{u_1}), Membership.mem (nhds p) U → Exists fun y => And …
      plim : p.IsLimit
      u : Set ↑(Set.Iio o)
      hu : Membership.mem (nhds ⟨p, hpo⟩) u
      ⊢ Exists fun y => And (Membership.mem (Inter.inter u (Set.preimage Subtype.val …
    -/
    obtain ⟨l, hl⟩ := exists_Ioc_subset_of_mem_nhds hu ⟨⟨0, plim.pos.trans hpo⟩, plim.pos⟩
    /-
      case mp.intro
      p o : Ordinal.{u_1}
      S : Set Ordinal.{u_1}
      hpo : LT.lt p o
      h : ∀ (U : Set Ordinal.{u_1}), Membership.mem (nhds p) U → Exists fun y => And …
      plim : p.IsLimit
      u : Set ↑(Set.Iio o)
      hu : Membership.mem (nhds ⟨p, hpo⟩) u
      l : ↑(Set.Iio o)
      hl : And (LT.lt l ⟨p, hpo⟩) (HasSubset.Subset (Set.Ioc l ⟨p, hpo⟩) u)
      ⊢ Exists fun y => And (Membership.mem (Inter.inter u (Set.preimage Subtype.val …
    -/
    obtain ⟨x, hx⟩ := h (Ioo l (p + 1)) (Ioo_mem_nhds hl.1 (lt_add_one _))
    /-
      case mp.intro.intro
      p o : Ordinal.{u_1}
      S : Set Ordinal.{u_1}
      hpo : LT.lt p o
      h : ∀ (U : Set Ordinal.{u_1}), Membership.mem (nhds p) U → Exists fun y => And …
      plim : p.IsLimit
      u : Set ↑(Set.Iio o)
      hu : Membership.mem (nhds ⟨p, hpo⟩) u
      l : ↑(Set.Iio o)
      hl : And (LT.lt l ⟨p, hpo⟩) (HasSubset.Subset (Set.Ioc l ⟨p, hpo⟩) u)
      x : Ordinal.{u_1}
      hx : And (Membership.mem (Inter.inter (Set.Ioo (↑l) (HAdd.hAdd p 1)) S) x) (Ne …
      ⊢ Exists fun y => And (Membership.mem (Inter.inter u (Set.preimage Subtype.val …
    -/
    use ⟨x, lt_of_le_of_lt (lt_succ_iff.mp hx.1.1.2) hpo⟩
    /-
      case h
      p o : Ordinal.{u_1}
      S : Set Ordinal.{u_1}
      hpo : LT.lt p o
      h : ∀ (U : Set Ordinal.{u_1}), Membership.mem (nhds p) U → Exists fun y => And …
      plim : p.IsLimit
      u : Set ↑(Set.Iio o)
      hu : Membership.mem (nhds ⟨p, hpo⟩) u
      l : ↑(Set.Iio o)
      hl : And (LT.lt l ⟨p, hpo⟩) (HasSubset.Subset (Set.Ioc l ⟨p, hpo⟩) u)
      x : Ordinal.{u_1}
      hx : And (Membership.mem (Inter.inter (Set.Ioo (↑l) (HAdd.hAdd p 1)) S) x) (Ne …
      ⊢ And (Membership.mem (Inter.inter u (Set.preimage Subtype.val S)) ⟨x, ⋯⟩) (Ne …
    -/
    refine ⟨?_, Subtype.coe_ne_coe.mp hx.2⟩
    /-
      case h
      p o : Ordinal.{u_1}
      S : Set Ordinal.{u_1}
      hpo : LT.lt p o
      h : ∀ (U : Set Ordinal.{u_1}), Membership.mem (nhds p) U → Exists fun y => And …
      plim : p.IsLimit
      u : Set ↑(Set.Iio o)
      hu : Membership.mem (nhds ⟨p, hpo⟩) u
      l : ↑(Set.Iio o)
      hl : And (LT.lt l ⟨p, hpo⟩) (HasSubset.Subset (Set.Ioc l ⟨p, hpo⟩) u)
      x : Ordinal.{u_1}
      hx : And (Membership.mem (Inter.inter (Set.Ioo (↑l) (HAdd.hAdd p 1)) S) x) (Ne …
      ⊢ Membership.mem (Inter.inter u (Set.preimage Subtype.val S)) ⟨x, ⋯⟩
    -/
    exact ⟨hl.2 ⟨hx.1.1.1, by exact_mod_cast lt_succ_iff.mp hx.1.1.2⟩, hx.1.2⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      p o : Ordinal.{u_1}
      S : Set Ordinal.{u_1}
      hpo : LT.lt p o
      ⊢ AccPt ⟨p, hpo⟩ (Filter.principal (Set.preimage Subtype.val S)) → AccPt p (Fi …
    -/
  · intro h
    /-
      case mpr
      p o : Ordinal.{u_1}
      S : Set Ordinal.{u_1}
      hpo : LT.lt p o
      h : AccPt ⟨p, hpo⟩ (Filter.principal (Set.preimage Subtype.val S))
      ⊢ AccPt p (Filter.principal S)
    -/
    rw [accPt_iff_nhds] at *
    /-
      case mpr
      p o : Ordinal.{u_1}
      S : Set Ordinal.{u_1}
      hpo : LT.lt p o
      h : ∀ (U : Set ↑(Set.Iio o)), Membership.mem (nhds ⟨p, hpo⟩) U → Exists fun y  …
      ⊢ ∀ (U : Set Ordinal.{u_1}), Membership.mem (nhds p) U → Exists fun y => And ( …
    -/
    intro u hu
    /-
      case mpr
      p o : Ordinal.{u_1}
      S : Set Ordinal.{u_1}
      hpo : LT.lt p o
      h : ∀ (U : Set ↑(Set.Iio o)), Membership.mem (nhds ⟨p, hpo⟩) U → Exists fun y  …
      u : Set Ordinal.{u_1}
      hu : Membership.mem (nhds p) u
      ⊢ Exists fun y => And (Membership.mem (Inter.inter u S) y) (Ne y p)
    -/
    by_cases ho : p + 1 < o
    · have ppos : p ≠ 0 := by
        rintro rfl
        rw [zero_add] at ho
        specialize h (Iio ⟨1, ho⟩) (Iio_mem_nhds (Subtype.mk_lt_mk.mpr zero_lt_one))
        obtain ⟨_, h⟩ := h
        exact h.2 <| Subtype.mk_eq_mk.mpr (lt_one_iff_zero.mp h.1.1)
      have plim : p.IsLimit := by
        contrapose! h
        obtain ⟨q, hq⟩ := ((zero_or_succ_or_limit p).resolve_left ppos).resolve_right h
        use (Ioo ⟨q, ((hq ▸ lt_succ q).trans hpo)⟩ ⟨p + 1, ho⟩)
        constructor
        · exact Ioo_mem_nhds (by simp only [hq, Subtype.mk_lt_mk, lt_succ]) (lt_succ p)
        · intro _ mem
          have aux1 := Subtype.mk_lt_mk.mp mem.1.1
          have aux2 := Subtype.mk_lt_mk.mp mem.1.2
          rw [Subtype.mk_eq_mk]
          rw [hq] at aux2 ⊢
          exact ((succ_le_iff.mpr aux1).antisymm (le_of_lt_succ aux2)).symm
      /-
        case pos
        p o : Ordinal.{u_1}
        S : Set Ordinal.{u_1}
        hpo : LT.lt p o
        h : ∀ (U : Set ↑(Set.Iio o)), Membership.mem (nhds ⟨p, hpo⟩) U → Exists fun y  …
        u : Set Ordinal.{u_1}
        hu : Membership.mem (nhds p) u
        ho : LT.lt (HAdd.hAdd p 1) o
        ppos : Ne p 0
        plim : p.IsLimit
        ⊢ Exists fun y => And (Membership.mem (Inter.inter u S) y) (Ne y p)
      -/
      obtain ⟨l, hl⟩ := exists_Ioc_subset_of_mem_nhds hu ⟨0, plim.pos⟩
      /-
        case pos.intro
        p o : Ordinal.{u_1}
        S : Set Ordinal.{u_1}
        hpo : LT.lt p o
        h : ∀ (U : Set ↑(Set.Iio o)), Membership.mem (nhds ⟨p, hpo⟩) U → Exists fun y  …
        u : Set Ordinal.{u_1}
        hu : Membership.mem (nhds p) u
        ho : LT.lt (HAdd.hAdd p 1) o
        ppos : Ne p 0
        plim : p.IsLimit
        l : Ordinal.{u_1}
        hl : And (LT.lt l p) (HasSubset.Subset (Set.Ioc l p) u)
        ⊢ Exists fun y => And (Membership.mem (Inter.inter u S) y) (Ne y p)
      -/
      obtain ⟨x, hx⟩ := h (Ioo ⟨l, hl.1.trans hpo⟩ ⟨p + 1, ho⟩) (Ioo_mem_nhds hl.1 (lt_add_one p))
      /-
        case pos.intro.intro
        p o : Ordinal.{u_1}
        S : Set Ordinal.{u_1}
        hpo : LT.lt p o
        h : ∀ (U : Set ↑(Set.Iio o)), Membership.mem (nhds ⟨p, hpo⟩) U → Exists fun y  …
        u : Set Ordinal.{u_1}
        hu : Membership.mem (nhds p) u
        ho : LT.lt (HAdd.hAdd p 1) o
        ppos : Ne p 0
        plim : p.IsLimit
        l : Ordinal.{u_1}
        hl : And (LT.lt l p) (HasSubset.Subset (Set.Ioc l p) u)
        x : ↑(Set.Iio o)
        hx : And (Membership.mem (Inter.inter (Set.Ioo ⟨l, ⋯⟩ ⟨HAdd.hAdd p 1, ho⟩) (Se …
        ⊢ Exists fun y => And (Membership.mem (Inter.inter u S) y) (Ne y p)
      -/
      use x
      /-
        case h
        p o : Ordinal.{u_1}
        S : Set Ordinal.{u_1}
        hpo : LT.lt p o
        h : ∀ (U : Set ↑(Set.Iio o)), Membership.mem (nhds ⟨p, hpo⟩) U → Exists fun y  …
        u : Set Ordinal.{u_1}
        hu : Membership.mem (nhds p) u
        ho : LT.lt (HAdd.hAdd p 1) o
        ppos : Ne p 0
        plim : p.IsLimit
        l : Ordinal.{u_1}
        hl : And (LT.lt l p) (HasSubset.Subset (Set.Ioc l p) u)
        x : ↑(Set.Iio o)
        hx : And (Membership.mem (Inter.inter (Set.Ioo ⟨l, ⋯⟩ ⟨HAdd.hAdd p 1, ho⟩) (Se …
        ⊢ And (Membership.mem (Inter.inter u S) ↑x) (Ne (↑x) p)
      -/
      exact ⟨⟨hl.2 ⟨hx.1.1.1, lt_succ_iff.mp hx.1.1.2⟩, hx.1.2⟩, fun h ↦ hx.2 (SetCoe.ext h)⟩
      /-
        🎉 no goals
      -/
    have hp : o = p + 1 := (le_succ_iff_eq_or_le.mp (le_of_not_lt ho)).resolve_right
      (not_le_of_lt hpo)
    have ppos : p ≠ 0 := by
      rintro rfl
      obtain ⟨x, hx⟩ := h Set.univ univ_mem
      have : ↑x < o := x.2
      simp_rw [hp, zero_add, lt_one_iff_zero] at this
      exact hx.2 (SetCoe.ext this)
    /-
      case neg
      p o : Ordinal.{u_1}
      S : Set Ordinal.{u_1}
      hpo : LT.lt p o
      h : ∀ (U : Set ↑(Set.Iio o)), Membership.mem (nhds ⟨p, hpo⟩) U → Exists fun y  …
      u : Set Ordinal.{u_1}
      hu : Membership.mem (nhds p) u
      ho : Not (LT.lt (HAdd.hAdd p 1) o)
      hp : Eq o (HAdd.hAdd p 1)
      ppos : Ne p 0
      ⊢ Exists fun y => And (Membership.mem (Inter.inter u S) y) (Ne y p)
    -/
    obtain ⟨l, hl⟩ := exists_Ioc_subset_of_mem_nhds hu ⟨0, Ordinal.pos_iff_ne_zero.mpr ppos⟩
    /-
      case neg.intro
      p o : Ordinal.{u_1}
      S : Set Ordinal.{u_1}
      hpo : LT.lt p o
      h : ∀ (U : Set ↑(Set.Iio o)), Membership.mem (nhds ⟨p, hpo⟩) U → Exists fun y  …
      u : Set Ordinal.{u_1}
      hu : Membership.mem (nhds p) u
      ho : Not (LT.lt (HAdd.hAdd p 1) o)
      hp : Eq o (HAdd.hAdd p 1)
      ppos : Ne p 0
      l : Ordinal.{u_1}
      hl : And (LT.lt l p) (HasSubset.Subset (Set.Ioc l p) u)
      ⊢ Exists fun y => And (Membership.mem (Inter.inter u S) y) (Ne y p)
    -/
    obtain ⟨x, hx⟩ := h (Ioi ⟨l, hl.1.trans hpo⟩) (Ioi_mem_nhds hl.1)
    /-
      case neg.intro.intro
      p o : Ordinal.{u_1}
      S : Set Ordinal.{u_1}
      hpo : LT.lt p o
      h : ∀ (U : Set ↑(Set.Iio o)), Membership.mem (nhds ⟨p, hpo⟩) U → Exists fun y  …
      u : Set Ordinal.{u_1}
      hu : Membership.mem (nhds p) u
      ho : Not (LT.lt (HAdd.hAdd p 1) o)
      hp : Eq o (HAdd.hAdd p 1)
      ppos : Ne p 0
      l : Ordinal.{u_1}
      hl : And (LT.lt l p) (HasSubset.Subset (Set.Ioc l p) u)
      x : ↑(Set.Iio o)
      hx : And (Membership.mem (Inter.inter (Set.Ioi ⟨l, ⋯⟩) (Set.preimage Subtype.v …
      ⊢ Exists fun y => And (Membership.mem (Inter.inter u S) y) (Ne y p)
    -/
    use x
    /-
      case h
      p o : Ordinal.{u_1}
      S : Set Ordinal.{u_1}
      hpo : LT.lt p o
      h : ∀ (U : Set ↑(Set.Iio o)), Membership.mem (nhds ⟨p, hpo⟩) U → Exists fun y  …
      u : Set Ordinal.{u_1}
      hu : Membership.mem (nhds p) u
      ho : Not (LT.lt (HAdd.hAdd p 1) o)
      hp : Eq o (HAdd.hAdd p 1)
      ppos : Ne p 0
      l : Ordinal.{u_1}
      hl : And (LT.lt l p) (HasSubset.Subset (Set.Ioc l p) u)
      x : ↑(Set.Iio o)
      hx : And (Membership.mem (Inter.inter (Set.Ioi ⟨l, ⋯⟩) (Set.preimage Subtype.v …
      ⊢ And (Membership.mem (Inter.inter u S) ↑x) (Ne (↑x) p)
    -/
    refine ⟨⟨hl.2 ⟨hx.1.1, ?_⟩, hx.1.2⟩, fun h ↦ hx.2 (SetCoe.ext h)⟩
    /-
      case h
      p o : Ordinal.{u_1}
      S : Set Ordinal.{u_1}
      hpo : LT.lt p o
      h : ∀ (U : Set ↑(Set.Iio o)), Membership.mem (nhds ⟨p, hpo⟩) U → Exists fun y  …
      u : Set Ordinal.{u_1}
      hu : Membership.mem (nhds p) u
      ho : Not (LT.lt (HAdd.hAdd p 1) o)
      hp : Eq o (HAdd.hAdd p 1)
      ppos : Ne p 0
      l : Ordinal.{u_1}
      hl : And (LT.lt l p) (HasSubset.Subset (Set.Ioc l p) u)
      x : ↑(Set.Iio o)
      hx : And (Membership.mem (Inter.inter (Set.Ioi ⟨l, ⋯⟩) (Set.preimage Subtype.v …
      ⊢ LE.le (↑x) p
    -/
    rw [← lt_add_one_iff, ← hp]
    /-
      case h
      p o : Ordinal.{u_1}
      S : Set Ordinal.{u_1}
      hpo : LT.lt p o
      h : ∀ (U : Set ↑(Set.Iio o)), Membership.mem (nhds ⟨p, hpo⟩) U → Exists fun y  …
      u : Set Ordinal.{u_1}
      hu : Membership.mem (nhds p) u
      ho : Not (LT.lt (HAdd.hAdd p 1) o)
      hp : Eq o (HAdd.hAdd p 1)
      ppos : Ne p 0
      l : Ordinal.{u_1}
      hl : And (LT.lt l p) (HasSubset.Subset (Set.Ioc l p) u)
      x : ↑(Set.Iio o)
      hx : And (Membership.mem (Inter.inter (Set.Ioi ⟨l, ⋯⟩) (Set.preimage Subtype.v …
      ⊢ LT.lt (↑x) o
    -/
    exact x.2
    /-
      🎉 no goals
    -/


theorem isClosedBelow_iff {S : Set Ordinal} {o : Ordinal} : IsClosedBelow S o ↔
    ∀ p < o, IsAcc p S → p ∈ S := by
  /-
    S : Set Ordinal.{u_1}
    o : Ordinal.{u_1}
    ⊢ Iff (Ordinal.IsClosedBelow S o) (∀ (p : Ordinal.{u_1}), LT.lt p o → p.IsAcc  …
  -/
  dsimp [IsClosedBelow]
  /-
    S : Set Ordinal.{u_1}
    o : Ordinal.{u_1}
    ⊢ Iff (IsClosed (Set.preimage Subtype.val S)) (∀ (p : Ordinal.{u_1}), LT.lt p  …
  -/
  constructor
    /-
      case mp
      S : Set Ordinal.{u_1}
      o : Ordinal.{u_1}
      ⊢ IsClosed (Set.preimage Subtype.val S) → ∀ (p : Ordinal.{u_1}), LT.lt p o → p …
    -/
  · intro h p plto hp
    /-
      case mp
      S : Set Ordinal.{u_1}
      o : Ordinal.{u_1}
      h : IsClosed (Set.preimage Subtype.val S)
      p : Ordinal.{u_1}
      plto : LT.lt p o
      hp : p.IsAcc S
      ⊢ Membership.mem S p
    -/
    have : AccPt ⟨p, plto⟩ (𝓟 (Iio o ↓∩ S)) := (accPt_subtype _ _).mp hp
    /-
      case mp
      S : Set Ordinal.{u_1}
      o : Ordinal.{u_1}
      h : IsClosed (Set.preimage Subtype.val S)
      p : Ordinal.{u_1}
      plto : LT.lt p o
      hp : p.IsAcc S
      this : AccPt ⟨p, plto⟩ (Filter.principal (Set.preimage Subtype.val S))
      ⊢ Membership.mem S p
    -/
    rw [isClosed_iff_clusterPt] at h
    /-
      case mp
      S : Set Ordinal.{u_1}
      o : Ordinal.{u_1}
      h : ∀ (a : ↑(Set.Iio o)), ClusterPt a (Filter.principal (Set.preimage Subtype. …
      p : Ordinal.{u_1}
      plto : LT.lt p o
      hp : p.IsAcc S
      this : AccPt ⟨p, plto⟩ (Filter.principal (Set.preimage Subtype.val S))
      ⊢ Membership.mem S p
    -/
    exact h ⟨p, plto⟩ this.clusterPt
    /-
      🎉 no goals
    -/
    /-
      case mpr
      S : Set Ordinal.{u_1}
      o : Ordinal.{u_1}
      ⊢ (∀ (p : Ordinal.{u_1}), LT.lt p o → p.IsAcc S → Membership.mem S p) → IsClos …
    -/
  · intro h
    /-
      case mpr
      S : Set Ordinal.{u_1}
      o : Ordinal.{u_1}
      h : ∀ (p : Ordinal.{u_1}), LT.lt p o → p.IsAcc S → Membership.mem S p
      ⊢ IsClosed (Set.preimage Subtype.val S)
    -/
    rw [isClosed_iff_clusterPt]
    /-
      case mpr
      S : Set Ordinal.{u_1}
      o : Ordinal.{u_1}
      h : ∀ (p : Ordinal.{u_1}), LT.lt p o → p.IsAcc S → Membership.mem S p
      ⊢ ∀ (a : ↑(Set.Iio o)), ClusterPt a (Filter.principal (Set.preimage Subtype.va …
    -/
    intro r hr
    match clusterPt_principal.mp hr with
    | .inl h => exact h
    | .inr h' => exact h r.1 r.2 <| (accPt_subtype _ _).mpr h'


alias ⟨IsClosedBelow.forall_lt, _⟩ := isClosedBelow_iff


theorem IsClosedBelow.sInter {o : Ordinal} {S : Set (Set Ordinal)}
    (h : ∀ C ∈ S, IsClosedBelow C o) : IsClosedBelow (⋂₀ S) o := by
  /-
    o : Ordinal.{u_1}
    S : Set (Set Ordinal.{u_1})
    h : ∀ (C : Set Ordinal.{u_1}), Membership.mem S C → Ordinal.IsClosedBelow C o
    ⊢ Ordinal.IsClosedBelow S.sInter o
  -/
  rw [isClosedBelow_iff]
  /-
    o : Ordinal.{u_1}
    S : Set (Set Ordinal.{u_1})
    h : ∀ (C : Set Ordinal.{u_1}), Membership.mem S C → Ordinal.IsClosedBelow C o
    ⊢ ∀ (p : Ordinal.{u_1}), LT.lt p o → p.IsAcc S.sInter → Membership.mem S.sInte …
  -/
  intro p plto pAcc C CmemS
  /-
    o : Ordinal.{u_1}
    S : Set (Set Ordinal.{u_1})
    h : ∀ (C : Set Ordinal.{u_1}), Membership.mem S C → Ordinal.IsClosedBelow C o
    p : Ordinal.{u_1}
    plto : LT.lt p o
    pAcc : p.IsAcc S.sInter
    C : Set Ordinal.{u_1}
    CmemS : Membership.mem S C
    ⊢ Membership.mem C p
  -/
  exact (h C CmemS).forall_lt p plto (pAcc.mono (sInter_subset_of_mem CmemS))
  /-
    🎉 no goals
  -/


theorem IsClosedBelow.iInter {ι : Type u} {f : ι → Set Ordinal} {o : Ordinal}
    (h : ∀ i, IsClosedBelow (f i) o) : IsClosedBelow (⋂ i, f i) o :=
  IsClosedBelow.sInter fun _ ⟨i, hi⟩ ↦ hi ▸ (h i)


