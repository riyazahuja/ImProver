/-- Produce an association list for the finsupp over its support using choice. -/
@[simps]
noncomputable def toAList (f : α →₀ M) : AList fun _x : α => M :=
  ⟨f.graph.toList.map Prod.toSigma,
    by
      /-
        α : Type u_1
        M : Type u_2
        inst✝ : Zero M
        f : Finsupp α M
        ⊢ (List.map Prod.toSigma f.graph.toList).NodupKeys
      -/
      rw [List.NodupKeys, List.keys, List.map_map, Prod.fst_comp_toSigma, List.nodup_map_iff_inj_on]
        /-
          α : Type u_1
          M : Type u_2
          inst✝ : Zero M
          f : Finsupp α M
          ⊢ ∀ (x : Prod α M), Membership.mem f.graph.toList x → ∀ (y : Prod α M), Member …
        -/
      · rintro ⟨b, m⟩ hb ⟨c, n⟩ hc (rfl : b = c)
        /-
          case mk.mk
          α : Type u_1
          M : Type u_2
          inst✝ : Zero M
          f : Finsupp α M
          b : α
          m : M
          hb : Membership.mem f.graph.toList { fst := b, snd := m }
          n : M
          hc : Membership.mem f.graph.toList { fst := b, snd := n }
          ⊢ Eq { fst := b, snd := m } { fst := b, snd := n }
        -/
        rw [Finset.mem_toList, Finsupp.mem_graph_iff] at hb hc
        /-
          case mk.mk
          α : Type u_1
          M : Type u_2
          inst✝ : Zero M
          f : Finsupp α M
          b : α
          m : M
          hb : And (Eq (f { fst := b, snd := m }.1) { fst := b, snd := m }.2) (Ne { fst  …
          n : M
          hc : And (Eq (f { fst := b, snd := n }.1) { fst := b, snd := n }.2) (Ne { fst  …
          ⊢ Eq { fst := b, snd := m } { fst := b, snd := n }
        -/
        dsimp at hb hc
        /-
          case mk.mk
          α : Type u_1
          M : Type u_2
          inst✝ : Zero M
          f : Finsupp α M
          b : α
          m : M
          hb : And (Eq (f b) m) (Not (Eq m 0))
          n : M
          hc : And (Eq (f b) n) (Not (Eq n 0))
          ⊢ Eq { fst := b, snd := m } { fst := b, snd := n }
        -/
        rw [← hc.1, hb.1]
        /-
          🎉 no goals
        -/
        /-
          α : Type u_1
          M : Type u_2
          inst✝ : Zero M
          f : Finsupp α M
          ⊢ f.graph.toList.Nodup
        -/
      · apply Finset.nodup_toList⟩
        /-
          🎉 no goals
        -/


@[simp]
theorem toAList_keys_toFinset [DecidableEq α] (f : α →₀ M) :
    f.toAList.keys.toFinset = f.support := by
  /-
    α : Type u_1
    M : Type u_2
    inst✝¹ : Zero M
    inst✝ : DecidableEq α
    f : Finsupp α M
    ⊢ Eq f.toAList.keys.toFinset f.support
  -/
  ext
  /-
    case h
    α : Type u_1
    M : Type u_2
    inst✝¹ : Zero M
    inst✝ : DecidableEq α
    f : Finsupp α M
    a✝ : α
    ⊢ Iff (Membership.mem f.toAList.keys.toFinset a✝) (Membership.mem f.support a✝)
  -/
  simp [toAList, AList.mem_keys, AList.keys, List.keys]
  /-
    🎉 no goals
  -/


@[simp]
theorem mem_toAlist {f : α →₀ M} {x : α} : x ∈ f.toAList ↔ f x ≠ 0 := by
  /-
    α : Type u_1
    M : Type u_2
    inst✝ : Zero M
    f : Finsupp α M
    x : α
    ⊢ Iff (Membership.mem f.toAList x) (Ne (f x) 0)
  -/
  classical rw [AList.mem_keys, ← List.mem_toFinset, toAList_keys_toFinset, mem_support_iff]
  /-
    🎉 no goals
  -/


/-- Converts an association list into a finitely supported function via `AList.lookup`, sending
absent keys to zero. -/
noncomputable def lookupFinsupp (l : AList fun _x : α => M) : α →₀ M where
  support := by
    /-
      α : Type u_1
      M : Type u_2
      inst✝ : Zero M
      l : AList fun _x => M
      ⊢ Finset α
    -/
    haveI := Classical.decEq α; haveI := Classical.decEq M
    /-
      α : Type u_1
      M : Type u_2
      inst✝ : Zero M
      l : AList fun _x => M
      this✝ : DecidableEq α
      this : DecidableEq M
      ⊢ Finset α
    -/
    exact (l.1.filter fun x => Sigma.snd x ≠ 0).keys.toFinset
    /-
      🎉 no goals
    -/
  toFun a :=
    haveI := Classical.decEq α
    (l.lookup a).getD 0
  mem_support_toFun a := by
    classical
      simp_rw [@mem_toFinset _ _, List.mem_keys, List.mem_filter, ← mem_lookup_iff]
      cases lookup a l <;> simp


@[simp]
theorem lookupFinsupp_apply [DecidableEq α] (l : AList fun _x : α => M) (a : α) :
    l.lookupFinsupp a = (l.lookup a).getD 0 := by
    /-
      α : Type u_1
      M : Type u_2
      inst✝¹ : Zero M
      inst✝ : DecidableEq α
      l : AList fun _x => M
      a : α
      ⊢ Eq (l.lookupFinsupp a) ((AList.lookup a l).getD 0)
    -/
    convert rfl; congr
                 /-
                   🎉 no goals
                 -/


@[simp]
theorem lookupFinsupp_support [DecidableEq α] [DecidableEq M] (l : AList fun _x : α => M) :
    l.lookupFinsupp.support = (l.1.filter fun x => Sigma.snd x ≠ 0).keys.toFinset := by
  /-
    α : Type u_1
    M : Type u_2
    inst✝² : Zero M
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq M
    l : AList fun _x => M
    ⊢ Eq l.lookupFinsupp.support (List.filter (fun x => Decidable.decide (Ne x.snd …
  -/
  dsimp only [lookupFinsupp]
  /-
    α : Type u_1
    M : Type u_2
    inst✝² : Zero M
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq M
    l : AList fun _x => M
    ⊢ Eq (List.filter (fun x => Decidable.decide (Ne x.snd 0)) l.entries).keys.toF …
  -/
  congr!
  /-
    🎉 no goals
  -/


theorem lookupFinsupp_eq_iff_of_ne_zero [DecidableEq α] {l : AList fun _x : α => M} {a : α} {x : M}
    (hx : x ≠ 0) : l.lookupFinsupp a = x ↔ x ∈ l.lookup a := by
  /-
    α : Type u_1
    M : Type u_2
    inst✝¹ : Zero M
    inst✝ : DecidableEq α
    l : AList fun _x => M
    a : α
    x : M
    hx : Ne x 0
    ⊢ Iff (Eq (l.lookupFinsupp a) x) (Membership.mem (AList.lookup a l) x)
  -/
  rw [lookupFinsupp_apply]
  /-
    α : Type u_1
    M : Type u_2
    inst✝¹ : Zero M
    inst✝ : DecidableEq α
    l : AList fun _x => M
    a : α
    x : M
    hx : Ne x 0
    ⊢ Iff (Eq ((AList.lookup a l).getD 0) x) (Membership.mem (AList.lookup a l) x)
  -/
                               /-
                                 🎉 no goals
                               -/
  cases' lookup a l with m <;> simp [hx.symm]
                               /-
                                 🎉 no goals
                               -/


theorem lookupFinsupp_eq_zero_iff [DecidableEq α] {l : AList fun _x : α => M} {a : α} :
    l.lookupFinsupp a = 0 ↔ a ∉ l ∨ (0 : M) ∈ l.lookup a := by
  /-
    α : Type u_1
    M : Type u_2
    inst✝¹ : Zero M
    inst✝ : DecidableEq α
    l : AList fun _x => M
    a : α
    ⊢ Iff (Eq (l.lookupFinsupp a) 0) (Or (Not (Membership.mem l a)) (Membership.me …
  -/
  rw [lookupFinsupp_apply, ← lookup_eq_none]
  /-
    α : Type u_1
    M : Type u_2
    inst✝¹ : Zero M
    inst✝ : DecidableEq α
    l : AList fun _x => M
    a : α
    ⊢ Iff (Eq ((AList.lookup a l).getD 0) 0) (Or (Eq (AList.lookup a l) Option.non …
  -/
                               /-
                                 🎉 no goals
                               -/
  cases' lookup a l with m <;> simp
                               /-
                                 🎉 no goals
                               -/


@[simp]
theorem empty_lookupFinsupp : lookupFinsupp (∅ : AList fun _x : α => M) = 0 := by
  classical
    ext
    simp


@[simp]
theorem insert_lookupFinsupp [DecidableEq α] (l : AList fun _x : α => M) (a : α) (m : M) :
    (l.insert a m).lookupFinsupp = l.lookupFinsupp.update a m := by
  /-
    α : Type u_1
    M : Type u_2
    inst✝¹ : Zero M
    inst✝ : DecidableEq α
    l : AList fun _x => M
    a : α
    m : M
    ⊢ Eq (AList.insert a m l).lookupFinsupp (l.lookupFinsupp.update a m)
  -/
  ext b
  /-
    case h
    α : Type u_1
    M : Type u_2
    inst✝¹ : Zero M
    inst✝ : DecidableEq α
    l : AList fun _x => M
    a : α
    m : M
    b : α
    ⊢ Eq ((AList.insert a m l).lookupFinsupp b) ((l.lookupFinsupp.update a m) b)
  -/
                         /-
                           🎉 no goals
                         -/
  by_cases h : b = a <;> simp [h]
                         /-
                           🎉 no goals
                         -/


@[simp]
theorem singleton_lookupFinsupp (a : α) (m : M) :
    (singleton a m).lookupFinsupp = Finsupp.single a m := by
  classical
  simp [← AList.insert_empty]


@[simp]
theorem _root_.Finsupp.toAList_lookupFinsupp (f : α →₀ M) : f.toAList.lookupFinsupp = f := by
  /-
    α : Type u_1
    M : Type u_2
    inst✝ : Zero M
    f : Finsupp α M
    ⊢ Eq f.toAList.lookupFinsupp f
  -/
  ext a
  classical
    by_cases h : f a = 0
    · suffices f.toAList.lookup a = none by simp [h, this]
      simp [lookup_eq_none, h]
    · suffices f.toAList.lookup a = some (f a) by simp [h, this]
      apply mem_lookup_iff.2
      simpa using h


theorem lookupFinsupp_surjective : Function.Surjective (@lookupFinsupp α M _) := fun f =>
  ⟨_, Finsupp.toAList_lookupFinsupp f⟩


