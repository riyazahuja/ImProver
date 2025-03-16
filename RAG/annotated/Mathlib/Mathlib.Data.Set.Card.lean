/-- The cardinality of a set as a term in `ℕ∞` -/
noncomputable def encard (s : Set α) : ℕ∞ := ENat.card s


@[simp] theorem encard_univ_coe (s : Set α) : encard (univ : Set s) = encard s := by
  /-
    α : Type u_1
    s : Set α
    ⊢ Eq Set.univ.encard s.encard
  -/
  rw [encard, encard, ENat.card_congr (Equiv.Set.univ ↑s)]
  /-
    🎉 no goals
  -/


theorem encard_univ (α : Type*) :
    encard (univ : Set α) = ENat.card α := by
  /-
    α : Type u_3
    ⊢ Eq Set.univ.encard (ENat.card α)
  -/
  rw [encard, ENat.card_congr (Equiv.Set.univ α)]
  /-
    🎉 no goals
  -/


theorem Finite.encard_eq_coe_toFinset_card (h : s.Finite) : s.encard = h.toFinset.card := by
  /-
    α : Type u_1
    s : Set α
    h : s.Finite
    ⊢ Eq s.encard ↑h.toFinset.card
  -/
  have := h.fintype
  /-
    α : Type u_1
    s : Set α
    h : s.Finite
    this : Fintype ↑s
    ⊢ Eq s.encard ↑h.toFinset.card
  -/
  rw [encard, ENat.card_eq_coe_fintype_card, toFinite_toFinset, toFinset_card]
  /-
    🎉 no goals
  -/


theorem encard_eq_coe_toFinset_card (s : Set α) [Fintype s] : encard s = s.toFinset.card := by
  /-
    α : Type u_1
    s : Set α
    inst✝ : Fintype ↑s
    ⊢ Eq s.encard ↑s.toFinset.card
  -/
  have h := toFinite s
  /-
    α : Type u_1
    s : Set α
    inst✝ : Fintype ↑s
    h : s.Finite
    ⊢ Eq s.encard ↑s.toFinset.card
  -/
  rw [h.encard_eq_coe_toFinset_card, toFinite_toFinset]
  /-
    🎉 no goals
  -/


@[simp, norm_cast] theorem encard_coe_eq_coe_finsetCard (s : Finset α) :
    encard (s : Set α) = s.card := by
  /-
    α : Type u_1
    s : Finset α
    ⊢ Eq (↑s).encard ↑s.card
  -/
  rw [Finite.encard_eq_coe_toFinset_card (Finset.finite_toSet s)]; simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


theorem Infinite.encard_eq {s : Set α} (h : s.Infinite) : s.encard = ⊤ := by
  /-
    α : Type u_1
    s : Set α
    h : s.Infinite
    ⊢ Eq s.encard Top.top
  -/
  have := h.to_subtype
  /-
    α : Type u_1
    s : Set α
    h : s.Infinite
    this : Infinite ↑s
    ⊢ Eq s.encard Top.top
  -/
  rw [encard, ENat.card_eq_top_of_infinite]
  /-
    🎉 no goals
  -/


@[simp] theorem encard_eq_zero : s.encard = 0 ↔ s = ∅ := by
  /-
    α : Type u_1
    s : Set α
    ⊢ Iff (Eq s.encard 0) (Eq s EmptyCollection.emptyCollection)
  -/
  rw [encard, ENat.card_eq_zero_iff_empty, isEmpty_subtype, eq_empty_iff_forall_not_mem]
  /-
    🎉 no goals
  -/


@[simp] theorem encard_empty : (∅ : Set α).encard = 0 := by
  /-
    α : Type u_1
    ⊢ Eq EmptyCollection.emptyCollection.encard 0
  -/
  rw [encard_eq_zero]
  /-
    🎉 no goals
  -/


theorem nonempty_of_encard_ne_zero (h : s.encard ≠ 0) : s.Nonempty := by
  /-
    α : Type u_1
    s : Set α
    h : Ne s.encard 0
    ⊢ s.Nonempty
  -/
  rwa [nonempty_iff_ne_empty, Ne, ← encard_eq_zero]
  /-
    🎉 no goals
  -/


theorem encard_ne_zero : s.encard ≠ 0 ↔ s.Nonempty := by
  /-
    α : Type u_1
    s : Set α
    ⊢ Iff (Ne s.encard 0) s.Nonempty
  -/
  rw [ne_eq, encard_eq_zero, nonempty_iff_ne_empty]
  /-
    🎉 no goals
  -/


@[simp] theorem encard_pos : 0 < s.encard ↔ s.Nonempty := by
  /-
    α : Type u_1
    s : Set α
    ⊢ Iff (LT.lt 0 s.encard) s.Nonempty
  -/
  rw [pos_iff_ne_zero, encard_ne_zero]
  /-
    🎉 no goals
  -/


protected alias ⟨_, Nonempty.encard_pos⟩ := encard_pos


@[simp] theorem encard_singleton (e : α) : ({e} : Set α).encard = 1 := by
  /-
    α : Type u_1
    e : α
    ⊢ Eq (Singleton.singleton e).encard 1
  -/
  rw [encard, ENat.card_eq_coe_fintype_card, Fintype.card_ofSubsingleton, Nat.cast_one]
  /-
    🎉 no goals
  -/


theorem encard_union_eq (h : Disjoint s t) : (s ∪ t).encard = s.encard + t.encard := by
  classical
  simp [encard, ENat.card_congr (Equiv.Set.union h)]


theorem encard_insert_of_not_mem {a : α} (has : a ∉ s) : (insert a s).encard = s.encard + 1 := by
  /-
    α : Type u_1
    s : Set α
    a : α
    has : Not (Membership.mem s a)
    ⊢ Eq (Insert.insert a s).encard (HAdd.hAdd s.encard 1)
  -/
  rw [← union_singleton, encard_union_eq (by simpa), encard_singleton]
  /-
    🎉 no goals
  -/


theorem Finite.encard_lt_top (h : s.Finite) : s.encard < ⊤ := by
  /-
    α : Type u_1
    s : Set α
    h : s.Finite
    ⊢ LT.lt s.encard Top.top
  -/
  refine h.induction_on (by simp) ?_
  /-
    α : Type u_1
    s : Set α
    h : s.Finite
    ⊢ ∀ {a : α} {s : Set α}, Not (Membership.mem s a) → s.Finite → LT.lt s.encard  …
  -/
  rintro a t hat _ ht'
  /-
    α : Type u_1
    s : Set α
    h : s.Finite
    a : α
    t : Set α
    hat : Not (Membership.mem t a)
    a✝ : t.Finite
    ht' : LT.lt t.encard Top.top
    ⊢ LT.lt (Insert.insert a t).encard Top.top
  -/
  rw [encard_insert_of_not_mem hat]
  /-
    α : Type u_1
    s : Set α
    h : s.Finite
    a : α
    t : Set α
    hat : Not (Membership.mem t a)
    a✝ : t.Finite
    ht' : LT.lt t.encard Top.top
    ⊢ LT.lt (HAdd.hAdd t.encard 1) Top.top
  -/
  exact lt_tsub_iff_right.1 ht'
  /-
    🎉 no goals
  -/


theorem Finite.encard_eq_coe (h : s.Finite) : s.encard = ENat.toNat s.encard :=
  (ENat.coe_toNat h.encard_lt_top.ne).symm


theorem Finite.exists_encard_eq_coe (h : s.Finite) : ∃ (n : ℕ), s.encard = n :=
  ⟨_, h.encard_eq_coe⟩


@[simp] theorem encard_lt_top_iff : s.encard < ⊤ ↔ s.Finite :=
  ⟨fun h ↦ by_contra fun h' ↦ h.ne (Infinite.encard_eq h'), Finite.encard_lt_top⟩


@[simp] theorem encard_eq_top_iff : s.encard = ⊤ ↔ s.Infinite := by
  /-
    α : Type u_1
    s : Set α
    ⊢ Iff (Eq s.encard Top.top) s.Infinite
  -/
  rw [← not_iff_not, ← Ne, ← lt_top_iff_ne_top, encard_lt_top_iff, not_infinite]
  /-
    🎉 no goals
  -/


alias ⟨_, encard_eq_top⟩ := encard_eq_top_iff


theorem encard_ne_top_iff : s.encard ≠ ⊤ ↔ s.Finite := by
  /-
    α : Type u_1
    s : Set α
    ⊢ Iff (Ne s.encard Top.top) s.Finite
  -/
  simp
  /-
    🎉 no goals
  -/


theorem finite_of_encard_le_coe {k : ℕ} (h : s.encard ≤ k) : s.Finite := by
  /-
    α : Type u_1
    s : Set α
    k : Nat
    h : LE.le s.encard ↑k
    ⊢ s.Finite
  -/
  rw [← encard_lt_top_iff]; exact h.trans_lt (WithTop.coe_lt_top _)
                            /-
                              🎉 no goals
                            -/


theorem finite_of_encard_eq_coe {k : ℕ} (h : s.encard = k) : s.Finite :=
  finite_of_encard_le_coe h.le


theorem encard_le_coe_iff {k : ℕ} : s.encard ≤ k ↔ s.Finite ∧ ∃ (n₀ : ℕ), s.encard = n₀ ∧ n₀ ≤ k :=
                                          /-
                                            α : Type u_1
                                            s : Set α
                                            k : Nat
                                            h : LE.le s.encard ↑k
                                            ⊢ Exists fun n₀ => And (Eq s.encard ↑n₀) (LE.le n₀ k)
                                          -/
  ⟨fun h ↦ ⟨finite_of_encard_le_coe h, by rwa [ENat.le_coe_iff] at h⟩,
                                          /-
                                            🎉 no goals
                                          -/
                              /-
                                α : Type u_1
                                s : Set α
                                k : Nat
                                x✝ : And s.Finite (Exists fun n₀ => And (Eq s.encard ↑n₀) (LE.le n₀ k))
                                left✝ : s.Finite
                                n₀ : Nat
                                hs : Eq s.encard ↑n₀
                                hle : LE.le n₀ k
                                ⊢ LE.le s.encard ↑k
                              -/
    fun ⟨_,⟨n₀,hs, hle⟩⟩ ↦ by rwa [hs, Nat.cast_le]⟩
                              /-
                                🎉 no goals
                              -/


theorem encard_le_card (h : s ⊆ t) : s.encard ≤ t.encard := by
  /-
    α : Type u_1
    s t : Set α
    h : HasSubset.Subset s t
    ⊢ LE.le s.encard t.encard
  -/
  rw [← union_diff_cancel h, encard_union_eq disjoint_sdiff_right]; exact le_self_add
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem encard_mono {α : Type*} : Monotone (encard : Set α → ℕ∞) :=
  fun _ _ ↦ encard_le_card


theorem encard_diff_add_encard_of_subset (h : s ⊆ t) : (t \ s).encard + s.encard = t.encard := by
  /-
    α : Type u_1
    s t : Set α
    h : HasSubset.Subset s t
    ⊢ Eq (HAdd.hAdd (SDiff.sdiff t s).encard s.encard) t.encard
  -/
  rw [← encard_union_eq disjoint_sdiff_left, diff_union_self, union_eq_self_of_subset_right h]
  /-
    🎉 no goals
  -/


@[simp] theorem one_le_encard_iff_nonempty : 1 ≤ s.encard ↔ s.Nonempty := by
  /-
    α : Type u_1
    s : Set α
    ⊢ Iff (LE.le 1 s.encard) s.Nonempty
  -/
  rw [nonempty_iff_ne_empty, Ne, ← encard_eq_zero, ENat.one_le_iff_ne_zero]
  /-
    🎉 no goals
  -/


theorem encard_diff_add_encard_inter (s t : Set α) :
    (s \ t).encard + (s ∩ t).encard = s.encard := by
  rw [← encard_union_eq (disjoint_of_subset_right inter_subset_right disjoint_sdiff_left),
    diff_union_inter]


theorem encard_union_add_encard_inter (s t : Set α) :
    (s ∪ t).encard + (s ∩ t).encard = s.encard + t.encard := by
  rw [← diff_union_self, encard_union_eq disjoint_sdiff_left, add_right_comm,
    encard_diff_add_encard_inter]


theorem encard_eq_encard_iff_encard_diff_eq_encard_diff (h : (s ∩ t).Finite) :
    s.encard = t.encard ↔ (s \ t).encard = (t \ s).encard := by
  rw [← encard_diff_add_encard_inter s t, ← encard_diff_add_encard_inter t s, inter_comm t s,
    WithTop.add_right_cancel_iff h.encard_lt_top.ne]


theorem encard_le_encard_iff_encard_diff_le_encard_diff (h : (s ∩ t).Finite) :
    s.encard ≤ t.encard ↔ (s \ t).encard ≤ (t \ s).encard := by
  rw [← encard_diff_add_encard_inter s t, ← encard_diff_add_encard_inter t s, inter_comm t s,
    WithTop.add_le_add_iff_right h.encard_lt_top.ne]


theorem encard_lt_encard_iff_encard_diff_lt_encard_diff (h : (s ∩ t).Finite) :
    s.encard < t.encard ↔ (s \ t).encard < (t \ s).encard := by
  rw [← encard_diff_add_encard_inter s t, ← encard_diff_add_encard_inter t s, inter_comm t s,
    WithTop.add_lt_add_iff_right h.encard_lt_top.ne]


theorem encard_union_le (s t : Set α) : (s ∪ t).encard ≤ s.encard + t.encard := by
  /-
    α : Type u_1
    s t : Set α
    ⊢ LE.le (Union.union s t).encard (HAdd.hAdd s.encard t.encard)
  -/
  rw [← encard_union_add_encard_inter]; exact le_self_add
                                        /-
                                          🎉 no goals
                                        -/


theorem finite_iff_finite_of_encard_eq_encard (h : s.encard = t.encard) : s.Finite ↔ t.Finite := by
  /-
    α : Type u_1
    s t : Set α
    h : Eq s.encard t.encard
    ⊢ Iff s.Finite t.Finite
  -/
  rw [← encard_lt_top_iff, ← encard_lt_top_iff, h]
  /-
    🎉 no goals
  -/


theorem infinite_iff_infinite_of_encard_eq_encard (h : s.encard = t.encard) :
                                  /-
                                    α : Type u_1
                                    s t : Set α
                                    h : Eq s.encard t.encard
                                    ⊢ Iff s.Infinite t.Infinite
                                  -/
    s.Infinite ↔ t.Infinite := by rw [← encard_eq_top_iff, h, encard_eq_top_iff]
                                  /-
                                    🎉 no goals
                                  -/


theorem Finite.finite_of_encard_le {s : Set α} {t : Set β} (hs : s.Finite)
    (h : t.encard ≤ s.encard) : t.Finite :=
  encard_lt_top_iff.1 (h.trans_lt hs.encard_lt_top)


theorem Finite.eq_of_subset_of_encard_le (ht : t.Finite) (hst : s ⊆ t) (hts : t.encard ≤ s.encard) :
    s = t := by
  /-
    α : Type u_1
    s t : Set α
    ht : t.Finite
    hst : HasSubset.Subset s t
    hts : LE.le t.encard s.encard
    ⊢ Eq s t
  -/
  rw [← zero_add (a := encard s), ← encard_diff_add_encard_of_subset hst] at hts
  /-
    α : Type u_1
    s t : Set α
    ht : t.Finite
    hst : HasSubset.Subset s t
    hts : LE.le (HAdd.hAdd (SDiff.sdiff t s).encard s.encard) (HAdd.hAdd 0 s.encard)
    ⊢ Eq s t
  -/
  have hdiff := WithTop.le_of_add_le_add_right (ht.subset hst).encard_lt_top.ne hts
  /-
    α : Type u_1
    s t : Set α
    ht : t.Finite
    hst : HasSubset.Subset s t
    hts : LE.le (HAdd.hAdd (SDiff.sdiff t s).encard s.encard) (HAdd.hAdd 0 s.encard)
    hdiff : LE.le (SDiff.sdiff t s).encard 0
    ⊢ Eq s t
  -/
  rw [nonpos_iff_eq_zero, encard_eq_zero, diff_eq_empty] at hdiff
  /-
    α : Type u_1
    s t : Set α
    ht : t.Finite
    hst : HasSubset.Subset s t
    hts : LE.le (HAdd.hAdd (SDiff.sdiff t s).encard s.encard) (HAdd.hAdd 0 s.encard)
    hdiff : HasSubset.Subset t s
    ⊢ Eq s t
  -/
  exact hst.antisymm hdiff
  /-
    🎉 no goals
  -/


theorem Finite.eq_of_subset_of_encard_le' (hs : s.Finite) (hst : s ⊆ t)
    (hts : t.encard ≤ s.encard) : s = t :=
  (hs.finite_of_encard_le hts).eq_of_subset_of_encard_le hst hts


theorem Finite.encard_lt_encard (ht : t.Finite) (h : s ⊂ t) : s.encard < t.encard :=
  (encard_mono h.subset).lt_of_ne (fun he ↦ h.ne (ht.eq_of_subset_of_encard_le h.subset he.symm.le))


theorem encard_strictMono [Finite α] : StrictMono (encard : Set α → ℕ∞) :=
  fun _ _ h ↦ (toFinite _).encard_lt_encard h


theorem encard_diff_add_encard (s t : Set α) : (s \ t).encard + t.encard = (s ∪ t).encard := by
  /-
    α : Type u_1
    s t : Set α
    ⊢ Eq (HAdd.hAdd (SDiff.sdiff s t).encard t.encard) (Union.union s t).encard
  -/
  rw [← encard_union_eq disjoint_sdiff_left, diff_union_self]
  /-
    🎉 no goals
  -/


theorem encard_le_encard_diff_add_encard (s t : Set α) : s.encard ≤ (s \ t).encard + t.encard :=
  (encard_mono subset_union_left).trans_eq (encard_diff_add_encard _ _).symm


theorem tsub_encard_le_encard_diff (s t : Set α) : s.encard - t.encard ≤ (s \ t).encard := by
  /-
    α : Type u_1
    s t : Set α
    ⊢ LE.le (HSub.hSub s.encard t.encard) (SDiff.sdiff s t).encard
  -/
  rw [tsub_le_iff_left, add_comm]; apply encard_le_encard_diff_add_encard
                                   /-
                                     🎉 no goals
                                   -/


theorem encard_add_encard_compl (s : Set α) : s.encard + sᶜ.encard = (univ : Set α).encard := by
  /-
    α : Type u_1
    s : Set α
    ⊢ Eq (HAdd.hAdd s.encard (HasCompl.compl s).encard) Set.univ.encard
  -/
  rw [← encard_union_eq disjoint_compl_right, union_compl_self]
  /-
    🎉 no goals
  -/


theorem encard_insert_le (s : Set α) (x : α) : (insert x s).encard ≤ s.encard + 1 := by
  /-
    α : Type u_1
    s : Set α
    x : α
    ⊢ LE.le (Insert.insert x s).encard (HAdd.hAdd s.encard 1)
  -/
  rw [← union_singleton, ← encard_singleton x]; apply encard_union_le
                                                /-
                                                  🎉 no goals
                                                -/


theorem encard_singleton_inter (s : Set α) (x : α) : ({x} ∩ s).encard ≤ 1 := by
  /-
    α : Type u_1
    s : Set α
    x : α
    ⊢ LE.le (Inter.inter (Singleton.singleton x) s).encard 1
  -/
  rw [← encard_singleton x]; exact encard_le_card inter_subset_left
                             /-
                               🎉 no goals
                             -/


theorem encard_diff_singleton_add_one (h : a ∈ s) :
    (s \ {a}).encard + 1 = s.encard := by
  /-
    α : Type u_1
    s : Set α
    a : α
    h : Membership.mem s a
    ⊢ Eq (HAdd.hAdd (SDiff.sdiff s (Singleton.singleton a)).encard 1) s.encard
  -/
  rw [← encard_insert_of_not_mem (fun h ↦ h.2 rfl), insert_diff_singleton, insert_eq_of_mem h]
  /-
    🎉 no goals
  -/


theorem encard_diff_singleton_of_mem (h : a ∈ s) :
    (s \ {a}).encard = s.encard - 1 := by
  rw [← encard_diff_singleton_add_one h, ← WithTop.add_right_cancel_iff WithTop.one_ne_top,
    tsub_add_cancel_of_le (self_le_add_left _ _)]


theorem encard_tsub_one_le_encard_diff_singleton (s : Set α) (x : α) :
    s.encard - 1 ≤ (s \ {x}).encard := by
  /-
    α : Type u_1
    s : Set α
    x : α
    ⊢ LE.le (HSub.hSub s.encard 1) (SDiff.sdiff s (Singleton.singleton x)).encard
  -/
  rw [← encard_singleton x]; apply tsub_encard_le_encard_diff
                             /-
                               🎉 no goals
                             -/


theorem encard_exchange (ha : a ∉ s) (hb : b ∈ s) : (insert a (s \ {b})).encard = s.encard := by
  /-
    α : Type u_1
    s : Set α
    a b : α
    ha : Not (Membership.mem s a)
    hb : Membership.mem s b
    ⊢ Eq (Insert.insert a (SDiff.sdiff s (Singleton.singleton b))).encard s.encard
  -/
  rw [encard_insert_of_not_mem, encard_diff_singleton_add_one hb]
  /-
    α : Type u_1
    s : Set α
    a b : α
    ha : Not (Membership.mem s a)
    hb : Membership.mem s b
    ⊢ Not (Membership.mem (SDiff.sdiff s (Singleton.singleton b)) a)
  -/
  simp_all only [not_true, mem_diff, mem_singleton_iff, false_and, not_false_eq_true]
  /-
    🎉 no goals
  -/


theorem encard_exchange' (ha : a ∉ s) (hb : b ∈ s) : (insert a s \ {b}).encard = s.encard := by
  /-
    α : Type u_1
    s : Set α
    a b : α
    ha : Not (Membership.mem s a)
    hb : Membership.mem s b
    ⊢ Eq (SDiff.sdiff (Insert.insert a s) (Singleton.singleton b)).encard s.encard
  -/
  rw [← insert_diff_singleton_comm (by rintro rfl; exact ha hb), encard_exchange ha hb]
  /-
    🎉 no goals
  -/


theorem encard_eq_add_one_iff {k : ℕ∞} :
    s.encard = k + 1 ↔ (∃ a t, ¬a ∈ t ∧ insert a t = s ∧ t.encard = k) := by
  /-
    α : Type u_1
    s : Set α
    k : ENat
    ⊢ Iff (Eq s.encard (HAdd.hAdd k 1)) (Exists fun a => Exists fun t => And (Not  …
  -/
  refine ⟨fun h ↦ ?_, ?_⟩
    /-
      case refine_1
      α : Type u_1
      s : Set α
      k : ENat
      h : Eq s.encard (HAdd.hAdd k 1)
      ⊢ Exists fun a => Exists fun t => And (Not (Membership.mem t a)) (And (Eq (Ins …
    -/
  · obtain ⟨a, ha⟩ := nonempty_of_encard_ne_zero (s := s) (by simp [h])
    /-
      case refine_1.intro
      α : Type u_1
      s : Set α
      k : ENat
      h : Eq s.encard (HAdd.hAdd k 1)
      a : α
      ha : Membership.mem s a
      ⊢ Exists fun a => Exists fun t => And (Not (Membership.mem t a)) (And (Eq (Ins …
    -/
    refine ⟨a, s \ {a}, fun h ↦ h.2 rfl, by rwa [insert_diff_singleton, insert_eq_of_mem], ?_⟩
    rw [← WithTop.add_right_cancel_iff WithTop.one_ne_top, ← h,
      encard_diff_singleton_add_one ha]
  /-
    case refine_2
    α : Type u_1
    s : Set α
    k : ENat
    ⊢ (Exists fun a => Exists fun t => And (Not (Membership.mem t a)) (And (Eq (In …
  -/
  rintro ⟨a, t, h, rfl, rfl⟩
  /-
    case refine_2.intro.intro.intro.intro
    α : Type u_1
    a : α
    t : Set α
    h : Not (Membership.mem t a)
    ⊢ Eq (Insert.insert a t).encard (HAdd.hAdd t.encard 1)
  -/
  rw [encard_insert_of_not_mem h]
  /-
    🎉 no goals
  -/


/-- Every set is either empty, infinite, or can have its `encard` reduced by a removal. Intended
  for well-founded induction on the value of `encard`. -/
theorem eq_empty_or_encard_eq_top_or_encard_diff_singleton_lt (s : Set α) :
    s = ∅ ∨ s.encard = ⊤ ∨ ∃ a ∈ s, (s \ {a}).encard < s.encard := by
  refine s.eq_empty_or_nonempty.elim Or.inl (Or.inr ∘ fun ⟨a,ha⟩ ↦
    (s.finite_or_infinite.elim (fun hfin ↦ Or.inr ⟨a, ha, ?_⟩) (Or.inl ∘ Infinite.encard_eq)))
  /-
    α : Type u_1
    s : Set α
    x✝ : s.Nonempty
    a : α
    ha : Membership.mem s a
    hfin : s.Finite
    ⊢ LT.lt (SDiff.sdiff s (Singleton.singleton a)).encard s.encard
  -/
  rw [← encard_diff_singleton_add_one ha]; nth_rw 1 [← add_zero (encard _)]
  /-
    α : Type u_1
    s : Set α
    x✝ : s.Nonempty
    a : α
    ha : Membership.mem s a
    hfin : s.Finite
    ⊢ LT.lt (HAdd.hAdd (SDiff.sdiff s (Singleton.singleton a)).encard 0) (HAdd.hAd …
  -/
  exact WithTop.add_lt_add_left (hfin.diff _).encard_lt_top.ne zero_lt_one
  /-
    🎉 no goals
  -/


theorem encard_pair {x y : α} (hne : x ≠ y) : ({x, y} : Set α).encard = 2 := by
  rw [encard_insert_of_not_mem (by simpa), ← one_add_one_eq_two,
    WithTop.add_right_cancel_iff WithTop.one_ne_top, encard_singleton]


theorem encard_eq_one : s.encard = 1 ↔ ∃ x, s = {x} := by
  /-
    α : Type u_1
    s : Set α
    ⊢ Iff (Eq s.encard 1) (Exists fun x => Eq s (Singleton.singleton x))
  -/
  refine ⟨fun h ↦ ?_, fun ⟨x, hx⟩ ↦ by rw [hx, encard_singleton]⟩
  /-
    α : Type u_1
    s : Set α
    h : Eq s.encard 1
    ⊢ Exists fun x => Eq s (Singleton.singleton x)
  -/
  obtain ⟨x, hx⟩ := nonempty_of_encard_ne_zero (s := s) (by rw [h]; simp)
  /-
    case intro
    α : Type u_1
    s : Set α
    h : Eq s.encard 1
    x : α
    hx : Membership.mem s x
    ⊢ Exists fun x => Eq s (Singleton.singleton x)
  -/
  exact ⟨x, ((finite_singleton x).eq_of_subset_of_encard_le' (by simpa) (by simp [h])).symm⟩
  /-
    🎉 no goals
  -/


theorem encard_le_one_iff_eq : s.encard ≤ 1 ↔ s = ∅ ∨ ∃ x, s = {x} := by
  rw [le_iff_lt_or_eq, lt_iff_not_le, ENat.one_le_iff_ne_zero, not_not, encard_eq_zero,
    encard_eq_one]


theorem encard_le_one_iff : s.encard ≤ 1 ↔ ∀ a b, a ∈ s → b ∈ s → a = b := by
  /-
    α : Type u_1
    s : Set α
    ⊢ Iff (LE.le s.encard 1) (∀ (a b : α), Membership.mem s a → Membership.mem s b …
  -/
  rw [encard_le_one_iff_eq, or_iff_not_imp_left, ← Ne, ← nonempty_iff_ne_empty]
  refine ⟨fun h a b has hbs ↦ ?_,
    fun h ⟨x, hx⟩ ↦ ⟨x, ((singleton_subset_iff.2 hx).antisymm' (fun y hy ↦ h _ _ hy hx))⟩⟩
  /-
    α : Type u_1
    s : Set α
    h : s.Nonempty → Exists fun x => Eq s (Singleton.singleton x)
    a b : α
    has : Membership.mem s a
    hbs : Membership.mem s b
    ⊢ Eq a b
  -/
  obtain ⟨x, rfl⟩ := h ⟨_, has⟩
  /-
    case intro
    α : Type u_1
    a b x : α
    h : (Singleton.singleton x).Nonempty → Exists fun x_1 => Eq (Singleton.singlet …
    has : Membership.mem (Singleton.singleton x) a
    hbs : Membership.mem (Singleton.singleton x) b
    ⊢ Eq a b
  -/
  rw [(has : a = x), (hbs : b = x)]
  /-
    🎉 no goals
  -/


theorem one_lt_encard_iff : 1 < s.encard ↔ ∃ a b, a ∈ s ∧ b ∈ s ∧ a ≠ b := by
  /-
    α : Type u_1
    s : Set α
    ⊢ Iff (LT.lt 1 s.encard) (Exists fun a => Exists fun b => And (Membership.mem  …
  -/
  rw [← not_iff_not, not_exists, not_lt, encard_le_one_iff]; aesop
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem exists_ne_of_one_lt_encard (h : 1 < s.encard) (a : α) : ∃ b ∈ s, b ≠ a := by
  /-
    α : Type u_1
    s : Set α
    h : LT.lt 1 s.encard
    a : α
    ⊢ Exists fun b => And (Membership.mem s b) (Ne b a)
  -/
  by_contra! h'
  /-
    α : Type u_1
    s : Set α
    h : LT.lt 1 s.encard
    a : α
    h' : ∀ (b : α), Membership.mem s b → Eq b a
    ⊢ False
  -/
  obtain ⟨b, b', hb, hb', hne⟩ := one_lt_encard_iff.1 h
  /-
    case intro.intro.intro.intro
    α : Type u_1
    s : Set α
    h : LT.lt 1 s.encard
    a : α
    h' : ∀ (b : α), Membership.mem s b → Eq b a
    b b' : α
    hb : Membership.mem s b
    hb' : Membership.mem s b'
    hne : Ne b b'
    ⊢ False
  -/
  apply hne
  /-
    case intro.intro.intro.intro
    α : Type u_1
    s : Set α
    h : LT.lt 1 s.encard
    a : α
    h' : ∀ (b : α), Membership.mem s b → Eq b a
    b b' : α
    hb : Membership.mem s b
    hb' : Membership.mem s b'
    hne : Ne b b'
    ⊢ Eq b b'
  -/
  rw [h' b hb, h' b' hb']
  /-
    🎉 no goals
  -/


theorem encard_eq_two : s.encard = 2 ↔ ∃ x y, x ≠ y ∧ s = {x, y} := by
  /-
    α : Type u_1
    s : Set α
    ⊢ Iff (Eq s.encard 2) (Exists fun x => Exists fun y => And (Ne x y) (Eq s (Ins …
  -/
  refine ⟨fun h ↦ ?_, fun ⟨x, y, hne, hs⟩ ↦ by rw [hs, encard_pair hne]⟩
  /-
    α : Type u_1
    s : Set α
    h : Eq s.encard 2
    ⊢ Exists fun x => Exists fun y => And (Ne x y) (Eq s (Insert.insert x (Singlet …
  -/
  obtain ⟨x, hx⟩ := nonempty_of_encard_ne_zero (s := s) (by rw [h]; simp)
  rw [← insert_eq_of_mem hx, ← insert_diff_singleton, encard_insert_of_not_mem (fun h ↦ h.2 rfl),
    ← one_add_one_eq_two, WithTop.add_right_cancel_iff (WithTop.one_ne_top), encard_eq_one] at h
  /-
    case intro
    α : Type u_1
    s : Set α
    x : α
    h : Exists fun x_1 => Eq (SDiff.sdiff s (Singleton.singleton x)) (Singleton.si …
    hx : Membership.mem s x
    ⊢ Exists fun x => Exists fun y => And (Ne x y) (Eq s (Insert.insert x (Singlet …
  -/
  obtain ⟨y, h⟩ := h
  /-
    case intro.intro
    α : Type u_1
    s : Set α
    x : α
    hx : Membership.mem s x
    y : α
    h : Eq (SDiff.sdiff s (Singleton.singleton x)) (Singleton.singleton y)
    ⊢ Exists fun x => Exists fun y => And (Ne x y) (Eq s (Insert.insert x (Singlet …
  -/
  refine ⟨x, y, by rintro rfl; exact (h.symm.subset rfl).2 rfl, ?_⟩
  /-
    case intro.intro
    α : Type u_1
    s : Set α
    x : α
    hx : Membership.mem s x
    y : α
    h : Eq (SDiff.sdiff s (Singleton.singleton x)) (Singleton.singleton y)
    ⊢ Eq s (Insert.insert x (Singleton.singleton y))
  -/
  rw [← h, insert_diff_singleton, insert_eq_of_mem hx]
  /-
    🎉 no goals
  -/


theorem encard_eq_three {α : Type u_1} {s : Set α} :
    encard s = 3 ↔ ∃ x y z, x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ s = {x, y, z} := by
  /-
    α : Type u_1
    s : Set α
    ⊢ Iff (Eq s.encard 3) (Exists fun x => Exists fun y => Exists fun z => And (Ne …
  -/
  refine ⟨fun h ↦ ?_, fun ⟨x, y, z, hxy, hyz, hxz, hs⟩ ↦ ?_⟩
    /-
      case refine_1
      α : Type u_1
      s : Set α
      h : Eq s.encard 3
      ⊢ Exists fun x => Exists fun y => Exists fun z => And (Ne x y) (And (Ne x z) ( …
    -/
  · obtain ⟨x, hx⟩ := nonempty_of_encard_ne_zero (s := s) (by rw [h]; simp)
    rw [← insert_eq_of_mem hx, ← insert_diff_singleton,
      encard_insert_of_not_mem (fun h ↦ h.2 rfl), (by exact rfl : (3 : ℕ∞) = 2 + 1),
      WithTop.add_right_cancel_iff WithTop.one_ne_top, encard_eq_two] at h
    /-
      case refine_1.intro
      α : Type u_1
      s : Set α
      x : α
      h : Exists fun x_1 => Exists fun y => And (Ne x_1 y) (Eq (SDiff.sdiff s (Singl …
      hx : Membership.mem s x
      ⊢ Exists fun x => Exists fun y => Exists fun z => And (Ne x y) (And (Ne x z) ( …
    -/
    obtain ⟨y, z, hne, hs⟩ := h
    /-
      case refine_1.intro.intro.intro.intro
      α : Type u_1
      s : Set α
      x : α
      hx : Membership.mem s x
      y z : α
      hne : Ne y z
      hs : Eq (SDiff.sdiff s (Singleton.singleton x)) (Insert.insert y (Singleton.si …
      ⊢ Exists fun x => Exists fun y => Exists fun z => And (Ne x y) (And (Ne x z) ( …
    -/
    refine ⟨x, y, z, ?_, ?_, hne, ?_⟩
      /-
        case refine_1.intro.intro.intro.intro.refine_1
        α : Type u_1
        s : Set α
        x : α
        hx : Membership.mem s x
        y z : α
        hne : Ne y z
        hs : Eq (SDiff.sdiff s (Singleton.singleton x)) (Insert.insert y (Singleton.si …
        ⊢ Ne x y
      -/
    · rintro rfl; exact (hs.symm.subset (Or.inl rfl)).2 rfl
                  /-
                    🎉 no goals
                  -/
      /-
        case refine_1.intro.intro.intro.intro.refine_2
        α : Type u_1
        s : Set α
        x : α
        hx : Membership.mem s x
        y z : α
        hne : Ne y z
        hs : Eq (SDiff.sdiff s (Singleton.singleton x)) (Insert.insert y (Singleton.si …
        ⊢ Ne x z
      -/
    · rintro rfl; exact (hs.symm.subset (Or.inr rfl)).2 rfl
                  /-
                    🎉 no goals
                  -/
    /-
      case refine_1.intro.intro.intro.intro.refine_3
      α : Type u_1
      s : Set α
      x : α
      hx : Membership.mem s x
      y z : α
      hne : Ne y z
      hs : Eq (SDiff.sdiff s (Singleton.singleton x)) (Insert.insert y (Singleton.si …
      ⊢ Eq s (Insert.insert x (Insert.insert y (Singleton.singleton z)))
    -/
    rw [← hs, insert_diff_singleton, insert_eq_of_mem hx]
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    α : Type u_1
    s : Set α
    x✝ : Exists fun x => Exists fun y => Exists fun z => And (Ne x y) (And (Ne x z …
    x y z : α
    hxy : Ne x y
    hyz : Ne x z
    hxz : Ne y z
    hs : Eq s (Insert.insert x (Insert.insert y (Singleton.singleton z)))
    ⊢ Eq s.encard 3
  -/
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
  rw [hs, encard_insert_of_not_mem, encard_insert_of_not_mem, encard_singleton] <;> aesop
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


theorem Nat.encard_range (k : ℕ) : {i | i < k}.encard = k := by
  /-
    k : Nat
    ⊢ Eq (setOf fun i => LT.lt i k).encard ↑k
  -/
  convert encard_coe_eq_coe_finsetCard (Finset.range k) using 1
    /-
      case h.e'_2
      k : Nat
      ⊢ Eq (setOf fun i => LT.lt i k).encard (↑(Finset.range k)).encard
    -/
  · rw [Finset.coe_range, Iio_def]
    /-
      🎉 no goals
    -/
  /-
    case h.e'_3
    k : Nat
    ⊢ Eq ↑k ↑(Finset.range k).card
  -/
  rw [Finset.card_range]
  /-
    🎉 no goals
  -/


theorem Finite.eq_insert_of_subset_of_encard_eq_succ (hs : s.Finite) (h : s ⊆ t)
    (hst : t.encard = s.encard + 1) : ∃ a, t = insert a s := by
  rw [← encard_diff_add_encard_of_subset h, add_comm,
    WithTop.add_left_cancel_iff hs.encard_lt_top.ne, encard_eq_one] at hst
  /-
    α : Type u_1
    s t : Set α
    hs : s.Finite
    h : HasSubset.Subset s t
    hst : Exists fun x => Eq (SDiff.sdiff t s) (Singleton.singleton x)
    ⊢ Exists fun a => Eq t (Insert.insert a s)
  -/
  obtain ⟨x, hx⟩ := hst; use x; rw [← diff_union_of_subset h, hx, singleton_union]
                                /-
                                  🎉 no goals
                                -/


theorem exists_subset_encard_eq {k : ℕ∞} (hk : k ≤ s.encard) : ∃ t, t ⊆ s ∧ t.encard = k := by
  /-
    α : Type u_1
    s : Set α
    k : ENat
    hk : LE.le k s.encard
    ⊢ Exists fun t => And (HasSubset.Subset t s) (Eq t.encard k)
  -/
  revert hk
  /-
    α : Type u_1
    s : Set α
    k : ENat
    ⊢ LE.le k s.encard → Exists fun t => And (HasSubset.Subset t s) (Eq t.encard k)
  -/
  refine ENat.nat_induction k (fun _ ↦ ⟨∅, empty_subset _, by simp⟩) (fun n IH hle ↦ ?_) ?_
    /-
      case refine_1
      α : Type u_1
      s : Set α
      k : ENat
      n : Nat
      IH : LE.le (↑n) s.encard → Exists fun t => And (HasSubset.Subset t s) (Eq t.en …
      hle : LE.le (↑n.succ) s.encard
      ⊢ Exists fun t => And (HasSubset.Subset t s) (Eq t.encard ↑n.succ)
    -/
  · obtain ⟨t₀, ht₀s, ht₀⟩ := IH (le_trans (by simp) hle)
    /-
      case refine_1.intro.intro
      α : Type u_1
      s : Set α
      k : ENat
      n : Nat
      IH : LE.le (↑n) s.encard → Exists fun t => And (HasSubset.Subset t s) (Eq t.en …
      hle : LE.le (↑n.succ) s.encard
      t₀ : Set α
      ht₀s : HasSubset.Subset t₀ s
      ht₀ : Eq t₀.encard ↑n
      ⊢ Exists fun t => And (HasSubset.Subset t s) (Eq t.encard ↑n.succ)
    -/
    simp only [Nat.cast_succ] at *
    have hne : t₀ ≠ s := by
      rintro rfl; rw [ht₀, ← Nat.cast_one, ← Nat.cast_add, Nat.cast_le] at hle; simp at hle
    /-
      case refine_1.intro.intro
      α : Type u_1
      s : Set α
      k : ENat
      n : Nat
      IH : LE.le (↑n) s.encard → Exists fun t => And (HasSubset.Subset t s) (Eq t.en …
      t₀ : Set α
      ht₀s : HasSubset.Subset t₀ s
      ht₀ : Eq t₀.encard ↑n
      hle : LE.le (HAdd.hAdd (↑n) 1) s.encard
      hne : Ne t₀ s
      ⊢ Exists fun t => And (HasSubset.Subset t s) (Eq t.encard (HAdd.hAdd (↑n) 1))
    -/
    obtain ⟨x, hx⟩ := exists_of_ssubset (ht₀s.ssubset_of_ne hne)
    /-
      case refine_1.intro.intro.intro
      α : Type u_1
      s : Set α
      k : ENat
      n : Nat
      IH : LE.le (↑n) s.encard → Exists fun t => And (HasSubset.Subset t s) (Eq t.en …
      t₀ : Set α
      ht₀s : HasSubset.Subset t₀ s
      ht₀ : Eq t₀.encard ↑n
      hle : LE.le (HAdd.hAdd (↑n) 1) s.encard
      hne : Ne t₀ s
      x : α
      hx : And (Membership.mem s x) (Not (Membership.mem t₀ x))
      ⊢ Exists fun t => And (HasSubset.Subset t s) (Eq t.encard (HAdd.hAdd (↑n) 1))
    -/
    exact ⟨insert x t₀, insert_subset hx.1 ht₀s, by rw [encard_insert_of_not_mem hx.2, ht₀]⟩
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    α : Type u_1
    s : Set α
    k : ENat
    ⊢ (∀ (n : Nat), LE.le (↑n) s.encard → Exists fun t => And (HasSubset.Subset t  …
  -/
  simp only [top_le_iff, encard_eq_top_iff]
  /-
    case refine_2
    α : Type u_1
    s : Set α
    k : ENat
    ⊢ (∀ (n : Nat), LE.le (↑n) s.encard → Exists fun t => And (HasSubset.Subset t  …
  -/
  exact fun _ hi ↦ ⟨s, Subset.rfl, hi⟩
  /-
    🎉 no goals
  -/


theorem exists_superset_subset_encard_eq {k : ℕ∞}
    (hst : s ⊆ t) (hsk : s.encard ≤ k) (hkt : k ≤ t.encard) :
    ∃ r, s ⊆ r ∧ r ⊆ t ∧ r.encard = k := by
  /-
    α : Type u_1
    s t : Set α
    k : ENat
    hst : HasSubset.Subset s t
    hsk : LE.le s.encard k
    hkt : LE.le k t.encard
    ⊢ Exists fun r => And (HasSubset.Subset s r) (And (HasSubset.Subset r t) (Eq r …
  -/
  obtain (hs | hs) := eq_or_ne s.encard ⊤
    /-
      case inl
      α : Type u_1
      s t : Set α
      k : ENat
      hst : HasSubset.Subset s t
      hsk : LE.le s.encard k
      hkt : LE.le k t.encard
      hs : Eq s.encard Top.top
      ⊢ Exists fun r => And (HasSubset.Subset s r) (And (HasSubset.Subset r t) (Eq r …
    -/
  · rw [hs, top_le_iff] at hsk; subst hsk; exact ⟨s, Subset.rfl, hst, hs⟩
                                           /-
                                             🎉 no goals
                                           -/
  /-
    case inr
    α : Type u_1
    s t : Set α
    k : ENat
    hst : HasSubset.Subset s t
    hsk : LE.le s.encard k
    hkt : LE.le k t.encard
    hs : Ne s.encard Top.top
    ⊢ Exists fun r => And (HasSubset.Subset s r) (And (HasSubset.Subset r t) (Eq r …
  -/
  obtain ⟨k, rfl⟩ := exists_add_of_le hsk
  /-
    case inr.intro
    α : Type u_1
    s t : Set α
    hst : HasSubset.Subset s t
    hs : Ne s.encard Top.top
    k : ENat
    hsk : LE.le s.encard (HAdd.hAdd s.encard k)
    hkt : LE.le (HAdd.hAdd s.encard k) t.encard
    ⊢ Exists fun r => And (HasSubset.Subset s r) (And (HasSubset.Subset r t) (Eq r …
  -/
  obtain ⟨k', hk'⟩ := exists_add_of_le hkt
  have hk : k ≤ encard (t \ s) := by
    rw [← encard_diff_add_encard_of_subset hst, add_comm] at hkt
    exact WithTop.le_of_add_le_add_right hs hkt
  /-
    case inr.intro.intro
    α : Type u_1
    s t : Set α
    hst : HasSubset.Subset s t
    hs : Ne s.encard Top.top
    k : ENat
    hsk : LE.le s.encard (HAdd.hAdd s.encard k)
    hkt : LE.le (HAdd.hAdd s.encard k) t.encard
    k' : ENat
    hk' : Eq t.encard (HAdd.hAdd (HAdd.hAdd s.encard k) k')
    hk : LE.le k (SDiff.sdiff t s).encard
    ⊢ Exists fun r => And (HasSubset.Subset s r) (And (HasSubset.Subset r t) (Eq r …
  -/
  obtain ⟨r', hr', rfl⟩ := exists_subset_encard_eq hk
  /-
    case inr.intro.intro.intro.intro
    α : Type u_1
    s t : Set α
    hst : HasSubset.Subset s t
    hs : Ne s.encard Top.top
    k' : ENat
    r' : Set α
    hr' : HasSubset.Subset r' (SDiff.sdiff t s)
    hsk : LE.le s.encard (HAdd.hAdd s.encard r'.encard)
    hkt : LE.le (HAdd.hAdd s.encard r'.encard) t.encard
    hk' : Eq t.encard (HAdd.hAdd (HAdd.hAdd s.encard r'.encard) k')
    hk : LE.le r'.encard (SDiff.sdiff t s).encard
    ⊢ Exists fun r => And (HasSubset.Subset s r) (And (HasSubset.Subset r t) (Eq r …
  -/
  refine ⟨s ∪ r', subset_union_left, union_subset hst (hr'.trans diff_subset), ?_⟩
  /-
    case inr.intro.intro.intro.intro
    α : Type u_1
    s t : Set α
    hst : HasSubset.Subset s t
    hs : Ne s.encard Top.top
    k' : ENat
    r' : Set α
    hr' : HasSubset.Subset r' (SDiff.sdiff t s)
    hsk : LE.le s.encard (HAdd.hAdd s.encard r'.encard)
    hkt : LE.le (HAdd.hAdd s.encard r'.encard) t.encard
    hk' : Eq t.encard (HAdd.hAdd (HAdd.hAdd s.encard r'.encard) k')
    hk : LE.le r'.encard (SDiff.sdiff t s).encard
    ⊢ Eq (Union.union s r').encard (HAdd.hAdd s.encard r'.encard)
  -/
  rw [encard_union_eq (disjoint_of_subset_right hr' disjoint_sdiff_right)]
  /-
    🎉 no goals
  -/


theorem InjOn.encard_image (h : InjOn f s) : (f '' s).encard = s.encard := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    f : α → β
    h : Set.InjOn f s
    ⊢ Eq (Set.image f s).encard s.encard
  -/
  rw [encard, ENat.card_image_of_injOn h, encard]
  /-
    🎉 no goals
  -/


theorem encard_congr (e : s ≃ t) : s.encard = t.encard := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    e : Equiv ↑s ↑t
    ⊢ Eq s.encard t.encard
  -/
  rw [← encard_univ_coe, ← encard_univ_coe t, encard_univ, encard_univ, ENat.card_congr e]
  /-
    🎉 no goals
  -/


theorem _root_.Function.Injective.encard_image (hf : f.Injective) (s : Set α) :
    (f '' s).encard = s.encard :=
  hf.injOn.encard_image


theorem _root_.Function.Embedding.encard_le (e : s ↪ t) : s.encard ≤ t.encard := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    e : Function.Embedding ↑s ↑t
    ⊢ LE.le s.encard t.encard
  -/
  rw [← encard_univ_coe, ← e.injective.encard_image, ← Subtype.coe_injective.encard_image]
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    e : Function.Embedding ↑s ↑t
    ⊢ LE.le (Set.image (fun a => ↑a) (Set.image (⇑e) Set.univ)).encard t.encard
  -/
  exact encard_mono (by simp)
  /-
    🎉 no goals
  -/


theorem encard_image_le (f : α → β) (s : Set α) : (f '' s).encard ≤ s.encard := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    s : Set α
    ⊢ LE.le (Set.image f s).encard s.encard
  -/
  obtain (h | h) := isEmpty_or_nonempty α
    /-
      case inl
      α : Type u_1
      β : Type u_2
      f : α → β
      s : Set α
      h : IsEmpty α
      ⊢ LE.le (Set.image f s).encard s.encard
    -/
  · rw [s.eq_empty_of_isEmpty]; simp
                                /-
                                  🎉 no goals
                                -/
  /-
    case inr
    α : Type u_1
    β : Type u_2
    f : α → β
    s : Set α
    h : Nonempty α
    ⊢ LE.le (Set.image f s).encard s.encard
  -/
  rw [← (f.invFunOn_injOn_image s).encard_image]
  /-
    case inr
    α : Type u_1
    β : Type u_2
    f : α → β
    s : Set α
    h : Nonempty α
    ⊢ LE.le (Set.image (Function.invFunOn f s) (Set.image f s)).encard s.encard
  -/
  apply encard_le_card
  /-
    case inr.h
    α : Type u_1
    β : Type u_2
    f : α → β
    s : Set α
    h : Nonempty α
    ⊢ HasSubset.Subset (Set.image (Function.invFunOn f s) (Set.image f s)) s
  -/
  exact f.invFunOn_image_image_subset s
  /-
    🎉 no goals
  -/


theorem Finite.injOn_of_encard_image_eq (hs : s.Finite) (h : (f '' s).encard = s.encard) :
    InjOn f s := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    f : α → β
    hs : s.Finite
    h : Eq (Set.image f s).encard s.encard
    ⊢ Set.InjOn f s
  -/
  obtain (h' | hne) := isEmpty_or_nonempty α
    /-
      case inl
      α : Type u_1
      β : Type u_2
      s : Set α
      f : α → β
      hs : s.Finite
      h : Eq (Set.image f s).encard s.encard
      h' : IsEmpty α
      ⊢ Set.InjOn f s
    -/
  · rw [s.eq_empty_of_isEmpty]; simp
                                /-
                                  🎉 no goals
                                -/
  /-
    case inr
    α : Type u_1
    β : Type u_2
    s : Set α
    f : α → β
    hs : s.Finite
    h : Eq (Set.image f s).encard s.encard
    hne : Nonempty α
    ⊢ Set.InjOn f s
  -/
  rw [← (f.invFunOn_injOn_image s).encard_image] at h
  /-
    case inr
    α : Type u_1
    β : Type u_2
    s : Set α
    f : α → β
    hs : s.Finite
    hne : Nonempty α
    h : Eq (Set.image (Function.invFunOn f s) (Set.image f s)).encard s.encard
    ⊢ Set.InjOn f s
  -/
  rw [injOn_iff_invFunOn_image_image_eq_self]
  /-
    case inr
    α : Type u_1
    β : Type u_2
    s : Set α
    f : α → β
    hs : s.Finite
    hne : Nonempty α
    h : Eq (Set.image (Function.invFunOn f s) (Set.image f s)).encard s.encard
    ⊢ Eq (Set.image (Function.invFunOn f s) (Set.image f s)) s
  -/
  exact hs.eq_of_subset_of_encard_le (f.invFunOn_image_image_subset s) h.symm.le
  /-
    🎉 no goals
  -/


theorem encard_preimage_of_injective_subset_range (hf : f.Injective) (ht : t ⊆ range f) :
    (f ⁻¹' t).encard = t.encard := by
  /-
    α : Type u_1
    β : Type u_2
    t : Set β
    f : α → β
    hf : Function.Injective f
    ht : HasSubset.Subset t (Set.range f)
    ⊢ Eq (Set.preimage f t).encard t.encard
  -/
  rw [← hf.encard_image, image_preimage_eq_inter_range, inter_eq_self_of_subset_left ht]
  /-
    🎉 no goals
  -/


theorem encard_le_encard_of_injOn (hf : MapsTo f s t) (f_inj : InjOn f s) :
    s.encard ≤ t.encard := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    f : α → β
    hf : Set.MapsTo f s t
    f_inj : Set.InjOn f s
    ⊢ LE.le s.encard t.encard
  -/
  rw [← f_inj.encard_image]; apply encard_le_card; rintro _ ⟨x, hx, rfl⟩; exact hf hx
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


theorem Finite.exists_injOn_of_encard_le [Nonempty β] {s : Set α} {t : Set β} (hs : s.Finite)
    (hle : s.encard ≤ t.encard) : ∃ (f : α → β), s ⊆ f ⁻¹' t ∧ InjOn f s := by
  classical
  obtain (rfl | h | ⟨a, has, -⟩) := s.eq_empty_or_encard_eq_top_or_encard_diff_singleton_lt
  · simp
  · exact (encard_ne_top_iff.mpr hs h).elim
  obtain ⟨b, hbt⟩ := encard_pos.1 ((encard_pos.2 ⟨_, has⟩).trans_le hle)
  have hle' : (s \ {a}).encard ≤ (t \ {b}).encard := by
    rwa [← WithTop.add_le_add_iff_right WithTop.one_ne_top,
    encard_diff_singleton_add_one has, encard_diff_singleton_add_one hbt]

  obtain ⟨f₀, hf₀s, hinj⟩ := exists_injOn_of_encard_le (hs.diff {a}) hle'
  simp only [preimage_diff, subset_def, mem_diff, mem_singleton_iff, mem_preimage, and_imp] at hf₀s

  use Function.update f₀ a b
  rw [← insert_eq_of_mem has, ← insert_diff_singleton, injOn_insert (fun h ↦ h.2 rfl)]
  simp only [mem_diff, mem_singleton_iff, not_true, and_false, insert_diff_singleton, subset_def,
    mem_insert_iff, mem_preimage, ne_eq, Function.update_apply, forall_eq_or_imp, ite_true, and_imp,
    mem_image, ite_eq_left_iff, not_exists, not_and, not_forall, exists_prop, and_iff_right hbt]

  refine ⟨?_, ?_, fun x hxs hxa ↦ ⟨hxa, (hf₀s x hxs hxa).2⟩⟩
  · rintro x hx; split_ifs with h
    · assumption
    · exact (hf₀s x hx h).1
  exact InjOn.congr hinj (fun x ⟨_, hxa⟩ ↦ by rwa [Function.update_of_ne])
termination_by encard s


theorem Finite.exists_bijOn_of_encard_eq [Nonempty β] (hs : s.Finite) (h : s.encard = t.encard) :
    ∃ (f : α → β), BijOn f s t := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    inst✝ : Nonempty β
    hs : s.Finite
    h : Eq s.encard t.encard
    ⊢ Exists fun f => Set.BijOn f s t
  -/
  obtain ⟨f, hf, hinj⟩ := hs.exists_injOn_of_encard_le h.le; use f
  /-
    case h
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    inst✝ : Nonempty β
    hs : s.Finite
    h : Eq s.encard t.encard
    f : α → β
    hf : HasSubset.Subset s (Set.preimage f t)
    hinj : Set.InjOn f s
    ⊢ Set.BijOn f s t
  -/
  convert hinj.bijOn_image
  rw [(hs.image f).eq_of_subset_of_encard_le' (image_subset_iff.mpr hf)
    (h.symm.trans hinj.encard_image.symm).le]


/-- A tactic (for use in default params) that applies `Set.toFinite` to synthesize a `Set.Finite`
  term. -/
syntax "toFinite_tac" : tactic


macro_rules
  | `(tactic| toFinite_tac) => `(tactic| apply Set.toFinite)


/-- A tactic useful for transferring proofs for `encard` to their corresponding `card` statements -/
syntax "to_encard_tac" : tactic


macro_rules
  | `(tactic| to_encard_tac) => `(tactic|
      simp only [← Nat.cast_le (α := ℕ∞), ← Nat.cast_inj (R := ℕ∞), Nat.cast_add, Nat.cast_one])



/-- The cardinality of `s : Set α` . Has the junk value `0` if `s` is infinite -/
noncomputable def ncard (s : Set α) : ℕ := ENat.toNat s.encard


theorem ncard_def (s : Set α) : s.ncard = ENat.toNat s.encard := rfl


theorem Finite.cast_ncard_eq (hs : s.Finite) : s.ncard = s.encard := by
  /-
    α : Type u_1
    s : Set α
    hs : s.Finite
    ⊢ Eq (↑s.ncard) s.encard
  -/
  rwa [ncard, ENat.coe_toNat_eq_self, ne_eq, encard_eq_top_iff, Set.Infinite, not_not]
  /-
    🎉 no goals
  -/


theorem Nat.card_coe_set_eq (s : Set α) : Nat.card s = s.ncard := by
  /-
    α : Type u_1
    s : Set α
    ⊢ Eq (Nat.card ↑s) s.ncard
  -/
  obtain (h | h) := s.finite_or_infinite
    /-
      case inl
      α : Type u_1
      s : Set α
      h : s.Finite
      ⊢ Eq (Nat.card ↑s) s.ncard
    -/
  · have := h.fintype
    rw [ncard, h.encard_eq_coe_toFinset_card, Nat.card_eq_fintype_card,
      toFinite_toFinset, toFinset_card, ENat.toNat_coe]
  /-
    case inr
    α : Type u_1
    s : Set α
    h : s.Infinite
    ⊢ Eq (Nat.card ↑s) s.ncard
  -/
  have := infinite_coe_iff.2 h
  /-
    case inr
    α : Type u_1
    s : Set α
    h : s.Infinite
    this : Infinite ↑s
    ⊢ Eq (Nat.card ↑s) s.ncard
  -/
  rw [ncard, h.encard_eq, Nat.card_eq_zero_of_infinite, ENat.toNat_top]
  /-
    🎉 no goals
  -/


theorem ncard_eq_toFinset_card (s : Set α) (hs : s.Finite := by toFinite_tac) :
    s.ncard = hs.toFinset.card := by
  rw [← Nat.card_coe_set_eq, @Nat.card_eq_fintype_card _ hs.fintype,
    @Finite.card_toFinset _ _ hs.fintype hs]


theorem ncard_eq_toFinset_card' (s : Set α) [Fintype s] :
    s.ncard = s.toFinset.card := by
  /-
    α : Type u_1
    s : Set α
    inst✝ : Fintype ↑s
    ⊢ Eq s.ncard s.toFinset.card
  -/
  simp [← Nat.card_coe_set_eq, Nat.card_eq_fintype_card]
  /-
    🎉 no goals
  -/


lemma cast_ncard {s : Set α} (hs : s.Finite) :
    (s.ncard : Cardinal) = Cardinal.mk s := @Nat.cast_card _ hs


theorem encard_le_coe_iff_finite_ncard_le {k : ℕ} : s.encard ≤ k ↔ s.Finite ∧ s.ncard ≤ k := by
  /-
    α : Type u_1
    s : Set α
    k : Nat
    ⊢ Iff (LE.le s.encard ↑k) (And s.Finite (LE.le s.ncard k))
  -/
  rw [encard_le_coe_iff, and_congr_right_iff]
  exact fun hfin ↦ ⟨fun ⟨n₀, hn₀, hle⟩ ↦ by rwa [ncard_def, hn₀, ENat.toNat_coe],
    fun h ↦ ⟨s.ncard, by rw [hfin.cast_ncard_eq], h⟩⟩


theorem Infinite.ncard (hs : s.Infinite) : s.ncard = 0 := by
  /-
    α : Type u_1
    s : Set α
    hs : s.Infinite
    ⊢ Eq s.ncard 0
  -/
  rw [← Nat.card_coe_set_eq, @Nat.card_eq_zero_of_infinite _ hs.to_subtype]
  /-
    🎉 no goals
  -/


@[gcongr]
theorem ncard_le_ncard (hst : s ⊆ t) (ht : t.Finite := by toFinite_tac) :
    s.ncard ≤ t.ncard := by
  /-
    α : Type u_1
    s t : Set α
    hst : HasSubset.Subset s t
    ht : autoParam t.Finite _auto✝
    ⊢ LE.le s.ncard t.ncard
  -/
  rw [← Nat.cast_le (α := ℕ∞), ht.cast_ncard_eq, (ht.subset hst).cast_ncard_eq]
  /-
    α : Type u_1
    s t : Set α
    hst : HasSubset.Subset s t
    ht : autoParam t.Finite _auto✝
    ⊢ LE.le s.encard t.encard
  -/
  exact encard_mono hst
  /-
    🎉 no goals
  -/


                                                                           /-
                                                                             α : Type u_1
                                                                             inst✝ : Finite α
                                                                             x✝¹ x✝ : Set α
                                                                             hst : HasSubset.Subset x✝¹ x✝
                                                                             ⊢ x✝.Finite
                                                                           -/
theorem ncard_mono [Finite α] : @Monotone (Set α) _ _ _ ncard := fun _ _ ↦ ncard_le_ncard
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


@[simp] theorem ncard_eq_zero (hs : s.Finite := by toFinite_tac) :
    s.ncard = 0 ↔ s = ∅ := by
  /-
    α : Type u_1
    s : Set α
    hs : autoParam s.Finite _auto✝
    ⊢ Iff (Eq s.ncard 0) (Eq s EmptyCollection.emptyCollection)
  -/
  rw [← Nat.cast_inj (R := ℕ∞), hs.cast_ncard_eq, Nat.cast_zero, encard_eq_zero]
  /-
    🎉 no goals
  -/


@[simp, norm_cast] theorem ncard_coe_Finset (s : Finset α) : (s : Set α).ncard = s.card := by
  /-
    α : Type u_1
    s : Finset α
    ⊢ Eq (↑s).ncard s.card
  -/
  rw [ncard_eq_toFinset_card _, Finset.finite_toSet_toFinset]
  /-
    🎉 no goals
  -/


theorem ncard_univ (α : Type*) : (univ : Set α).ncard = Nat.card α := by
  /-
    α : Type u_3
    ⊢ Eq Set.univ.ncard (Nat.card α)
  -/
  cases' finite_or_infinite α with h h
    /-
      case inl
      α : Type u_3
      h : Finite α
      ⊢ Eq Set.univ.ncard (Nat.card α)
    -/
  · have hft := Fintype.ofFinite α
    /-
      case inl
      α : Type u_3
      h : Finite α
      hft : Fintype α
      ⊢ Eq Set.univ.ncard (Nat.card α)
    -/
    rw [ncard_eq_toFinset_card, Finite.toFinset_univ, Finset.card_univ, Nat.card_eq_fintype_card]
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_3
    h : Infinite α
    ⊢ Eq Set.univ.ncard (Nat.card α)
  -/
  rw [Nat.card_eq_zero_of_infinite, Infinite.ncard]
  /-
    case inr
    α : Type u_3
    h : Infinite α
    ⊢ Set.univ.Infinite
  -/
  exact infinite_univ
  /-
    🎉 no goals
  -/


@[simp] theorem ncard_empty (α : Type*) : (∅ : Set α).ncard = 0 := by
  /-
    α : Type u_3
    ⊢ Eq EmptyCollection.emptyCollection.ncard 0
  -/
  rw [ncard_eq_zero]
  /-
    🎉 no goals
  -/


theorem ncard_pos (hs : s.Finite := by toFinite_tac) : 0 < s.ncard ↔ s.Nonempty := by
  /-
    α : Type u_1
    s : Set α
    hs : autoParam s.Finite _auto✝
    ⊢ Iff (LT.lt 0 s.ncard) s.Nonempty
  -/
  rw [pos_iff_ne_zero, Ne, ncard_eq_zero hs, nonempty_iff_ne_empty]
  /-
    🎉 no goals
  -/


protected alias ⟨_, Nonempty.ncard_pos⟩ := ncard_pos


theorem ncard_ne_zero_of_mem {a : α} (h : a ∈ s) (hs : s.Finite := by toFinite_tac) : s.ncard ≠ 0 :=
  ((ncard_pos hs).mpr ⟨a, h⟩).ne.symm


theorem finite_of_ncard_ne_zero (hs : s.ncard ≠ 0) : s.Finite :=
  s.finite_or_infinite.elim id fun h ↦ (hs h.ncard).elim


theorem finite_of_ncard_pos (hs : 0 < s.ncard) : s.Finite :=
  finite_of_ncard_ne_zero hs.ne.symm


theorem nonempty_of_ncard_ne_zero (hs : s.ncard ≠ 0) : s.Nonempty := by
  /-
    α : Type u_1
    s : Set α
    hs : Ne s.ncard 0
    ⊢ s.Nonempty
  -/
  rw [nonempty_iff_ne_empty]; rintro rfl; simp at hs
                                          /-
                                            🎉 no goals
                                          -/


@[simp] theorem ncard_singleton (a : α) : ({a} : Set α).ncard = 1 := by
  /-
    α : Type u_1
    a : α
    ⊢ Eq (Singleton.singleton a).ncard 1
  -/
  simp [ncard]
  /-
    🎉 no goals
  -/


theorem ncard_singleton_inter (a : α) (s : Set α) : ({a} ∩ s).ncard ≤ 1 := by
  /-
    α : Type u_1
    a : α
    s : Set α
    ⊢ LE.le (Inter.inter (Singleton.singleton a) s).ncard 1
  -/
  rw [← Nat.cast_le (α := ℕ∞), (toFinite _).cast_ncard_eq, Nat.cast_one]
  /-
    α : Type u_1
    a : α
    s : Set α
    ⊢ LE.le (Inter.inter (Singleton.singleton a) s).encard 1
  -/
  apply encard_singleton_inter
  /-
    🎉 no goals
  -/

@[simp] theorem ncard_insert_of_not_mem {a : α} (h : a ∉ s) (hs : s.Finite := by toFinite_tac) :
    (insert a s).ncard = s.ncard + 1 := by
  rw [← Nat.cast_inj (R := ℕ∞), (hs.insert a).cast_ncard_eq, Nat.cast_add, Nat.cast_one,
    hs.cast_ncard_eq, encard_insert_of_not_mem h]


theorem ncard_insert_of_mem {a : α} (h : a ∈ s) : ncard (insert a s) = s.ncard := by
  /-
    α : Type u_1
    s : Set α
    a : α
    h : Membership.mem s a
    ⊢ Eq (Insert.insert a s).ncard s.ncard
  -/
  rw [insert_eq_of_mem h]
  /-
    🎉 no goals
  -/


theorem ncard_insert_le (a : α) (s : Set α) : (insert a s).ncard ≤ s.ncard + 1 := by
  /-
    α : Type u_1
    a : α
    s : Set α
    ⊢ LE.le (Insert.insert a s).ncard (HAdd.hAdd s.ncard 1)
  -/
  obtain hs | hs := s.finite_or_infinite
    /-
      case inl
      α : Type u_1
      a : α
      s : Set α
      hs : s.Finite
      ⊢ LE.le (Insert.insert a s).ncard (HAdd.hAdd s.ncard 1)
    -/
  · to_encard_tac; rw [hs.cast_ncard_eq, (hs.insert _).cast_ncard_eq]; apply encard_insert_le
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
  /-
    case inr
    α : Type u_1
    a : α
    s : Set α
    hs : s.Infinite
    ⊢ LE.le (Insert.insert a s).ncard (HAdd.hAdd s.ncard 1)
  -/
  rw [(hs.mono (subset_insert a s)).ncard]
  /-
    case inr
    α : Type u_1
    a : α
    s : Set α
    hs : s.Infinite
    ⊢ LE.le 0 (HAdd.hAdd s.ncard 1)
  -/
  exact Nat.zero_le _
  /-
    🎉 no goals
  -/


theorem ncard_insert_eq_ite {a : α} [Decidable (a ∈ s)] (hs : s.Finite := by toFinite_tac) :
    ncard (insert a s) = if a ∈ s then s.ncard else s.ncard + 1 := by
  /-
    α : Type u_1
    s : Set α
    a : α
    inst✝ : Decidable (Membership.mem s a)
    hs : autoParam s.Finite _auto✝
    ⊢ Eq (Insert.insert a s).ncard (ite (Membership.mem s a) s.ncard (HAdd.hAdd s. …
  -/
  by_cases h : a ∈ s
    /-
      case pos
      α : Type u_1
      s : Set α
      a : α
      inst✝ : Decidable (Membership.mem s a)
      hs : autoParam s.Finite _auto✝
      h : Membership.mem s a
      ⊢ Eq (Insert.insert a s).ncard (ite (Membership.mem s a) s.ncard (HAdd.hAdd s. …
    -/
  · rw [ncard_insert_of_mem h, if_pos h]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      s : Set α
      a : α
      inst✝ : Decidable (Membership.mem s a)
      hs : autoParam s.Finite _auto✝
      h : Not (Membership.mem s a)
      ⊢ Eq (Insert.insert a s).ncard (ite (Membership.mem s a) s.ncard (HAdd.hAdd s. …
    -/
  · rw [ncard_insert_of_not_mem h hs, if_neg h]
    /-
      🎉 no goals
    -/


theorem ncard_le_ncard_insert (a : α) (s : Set α) : s.ncard ≤ (insert a s).ncard := by
  classical
  refine
    s.finite_or_infinite.elim (fun h ↦ ?_) (fun h ↦ by (rw [h.ncard]; exact Nat.zero_le _))
  rw [ncard_insert_eq_ite h]; split_ifs <;> simp


@[simp] theorem ncard_pair {a b : α} (h : a ≠ b) : ({a, b} : Set α).ncard = 2 := by
  /-
    α : Type u_1
    a b : α
    h : Ne a b
    ⊢ Eq (Insert.insert a (Singleton.singleton b)).ncard 2
  -/
  rw [ncard_insert_of_not_mem, ncard_singleton]; simpa
                                                 /-
                                                   🎉 no goals
                                                 -/


@[simp] theorem ncard_diff_singleton_add_one {a : α} (h : a ∈ s)
    (hs : s.Finite := by toFinite_tac) : (s \ {a}).ncard + 1 = s.ncard := by
  /-
    α : Type u_1
    s : Set α
    a : α
    h : Membership.mem s a
    hs : autoParam s.Finite _auto✝
    ⊢ Eq (HAdd.hAdd (SDiff.sdiff s (Singleton.singleton a)).ncard 1) s.ncard
  -/
  to_encard_tac; rw [hs.cast_ncard_eq, (hs.diff _).cast_ncard_eq,
    encard_diff_singleton_add_one h]


@[simp] theorem ncard_diff_singleton_of_mem {a : α} (h : a ∈ s) (hs : s.Finite := by toFinite_tac) :
    (s \ {a}).ncard = s.ncard - 1 :=
  eq_tsub_of_add_eq (ncard_diff_singleton_add_one h hs)


theorem ncard_diff_singleton_lt_of_mem {a : α} (h : a ∈ s) (hs : s.Finite := by toFinite_tac) :
    (s \ {a}).ncard < s.ncard := by
  /-
    α : Type u_1
    s : Set α
    a : α
    h : Membership.mem s a
    hs : autoParam s.Finite _auto✝
    ⊢ LT.lt (SDiff.sdiff s (Singleton.singleton a)).ncard s.ncard
  -/
  rw [← ncard_diff_singleton_add_one h hs]; apply lt_add_one
                                            /-
                                              🎉 no goals
                                            -/


theorem ncard_diff_singleton_le (s : Set α) (a : α) : (s \ {a}).ncard ≤ s.ncard := by
  /-
    α : Type u_1
    s : Set α
    a : α
    ⊢ LE.le (SDiff.sdiff s (Singleton.singleton a)).ncard s.ncard
  -/
  obtain hs | hs := s.finite_or_infinite
    /-
      case inl
      α : Type u_1
      s : Set α
      a : α
      hs : s.Finite
      ⊢ LE.le (SDiff.sdiff s (Singleton.singleton a)).ncard s.ncard
    -/
  · apply ncard_le_ncard diff_subset hs
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    s : Set α
    a : α
    hs : s.Infinite
    ⊢ LE.le (SDiff.sdiff s (Singleton.singleton a)).ncard s.ncard
  -/
  convert @zero_le ℕ _ _
  /-
    case h.e'_3
    α : Type u_1
    s : Set α
    a : α
    hs : s.Infinite
    ⊢ Eq (SDiff.sdiff s (Singleton.singleton a)).ncard 0
  -/
  exact (hs.diff (by simp : Set.Finite {a})).ncard
  /-
    🎉 no goals
  -/


theorem pred_ncard_le_ncard_diff_singleton (s : Set α) (a : α) : s.ncard - 1 ≤ (s \ {a}).ncard := by
  /-
    α : Type u_1
    s : Set α
    a : α
    ⊢ LE.le (HSub.hSub s.ncard 1) (SDiff.sdiff s (Singleton.singleton a)).ncard
  -/
  cases' s.finite_or_infinite with hs hs
    /-
      case inl
      α : Type u_1
      s : Set α
      a : α
      hs : s.Finite
      ⊢ LE.le (HSub.hSub s.ncard 1) (SDiff.sdiff s (Singleton.singleton a)).ncard
    -/
  · by_cases h : a ∈ s
      /-
        case pos
        α : Type u_1
        s : Set α
        a : α
        hs : s.Finite
        h : Membership.mem s a
        ⊢ LE.le (HSub.hSub s.ncard 1) (SDiff.sdiff s (Singleton.singleton a)).ncard
      -/
    · rw [ncard_diff_singleton_of_mem h hs]
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u_1
      s : Set α
      a : α
      hs : s.Finite
      h : Not (Membership.mem s a)
      ⊢ LE.le (HSub.hSub s.ncard 1) (SDiff.sdiff s (Singleton.singleton a)).ncard
    -/
    rw [diff_singleton_eq_self h]
    /-
      case neg
      α : Type u_1
      s : Set α
      a : α
      hs : s.Finite
      h : Not (Membership.mem s a)
      ⊢ LE.le (HSub.hSub s.ncard 1) s.ncard
    -/
    apply Nat.pred_le
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    s : Set α
    a : α
    hs : s.Infinite
    ⊢ LE.le (HSub.hSub s.ncard 1) (SDiff.sdiff s (Singleton.singleton a)).ncard
  -/
  convert Nat.zero_le _
  /-
    case h.e'_3
    α : Type u_1
    s : Set α
    a : α
    hs : s.Infinite
    ⊢ Eq (HSub.hSub s.ncard 1) 0
  -/
  rw [hs.ncard]
  /-
    🎉 no goals
  -/


theorem ncard_exchange {a b : α} (ha : a ∉ s) (hb : b ∈ s) : (insert a (s \ {b})).ncard = s.ncard :=
  congr_arg ENat.toNat <| encard_exchange ha hb


theorem ncard_exchange' {a b : α} (ha : a ∉ s) (hb : b ∈ s) :
    (insert a s \ {b}).ncard = s.ncard := by
  rw [← ncard_exchange ha hb, ← singleton_union, ← singleton_union, union_diff_distrib,
    @diff_singleton_eq_self _ b {a} fun h ↦ ha (by rwa [← mem_singleton_iff.mp h])]


lemma odd_card_insert_iff {a : α} (hs : s.Finite := by toFinite_tac) (ha : a ∉ s) :
    Odd (insert a s).ncard ↔ Even s.ncard := by
  /-
    α : Type u_1
    s : Set α
    a : α
    hs : autoParam s.Finite _auto✝
    ha : Not (Membership.mem s a)
    ⊢ Iff (Odd (Insert.insert a s).ncard) (Even s.ncard)
  -/
  rw [ncard_insert_of_not_mem ha hs, Nat.odd_add]
  /-
    α : Type u_1
    s : Set α
    a : α
    hs : autoParam s.Finite _auto✝
    ha : Not (Membership.mem s a)
    ⊢ Iff (Iff (Odd s.ncard) (Even 1)) (Even s.ncard)
  -/
  simp only [Nat.odd_add, ← Nat.not_even_iff_odd, Nat.not_even_one, iff_false, Decidable.not_not]
  /-
    🎉 no goals
  -/


lemma even_card_insert_iff {a : α} (hs : s.Finite := by toFinite_tac) (ha : a ∉ s) :
    Even (insert a s).ncard ↔ Odd s.ncard := by
  /-
    α : Type u_1
    s : Set α
    a : α
    hs : autoParam s.Finite _auto✝
    ha : Not (Membership.mem s a)
    ⊢ Iff (Even (Insert.insert a s).ncard) (Odd s.ncard)
  -/
  rw [ncard_insert_of_not_mem ha hs, Nat.even_add_one, Nat.not_even_iff_odd]
  /-
    🎉 no goals
  -/


theorem ncard_image_le (hs : s.Finite := by toFinite_tac) : (f '' s).ncard ≤ s.ncard := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    f : α → β
    hs : autoParam s.Finite _auto✝
    ⊢ LE.le (Set.image f s).ncard s.ncard
  -/
  to_encard_tac; rw [hs.cast_ncard_eq, (hs.image _).cast_ncard_eq]; apply encard_image_le
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem ncard_image_of_injOn (H : Set.InjOn f s) : (f '' s).ncard = s.ncard :=
  congr_arg ENat.toNat <| H.encard_image


theorem injOn_of_ncard_image_eq (h : (f '' s).ncard = s.ncard) (hs : s.Finite := by toFinite_tac) :
    Set.InjOn f s := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    f : α → β
    h : Eq (Set.image f s).ncard s.ncard
    hs : autoParam s.Finite _auto✝
    ⊢ Set.InjOn f s
  -/
  rw [← Nat.cast_inj (R := ℕ∞), hs.cast_ncard_eq, (hs.image _).cast_ncard_eq] at h
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    f : α → β
    h : Eq (Set.image f s).encard s.encard
    hs : autoParam s.Finite _auto✝
    ⊢ Set.InjOn f s
  -/
  exact hs.injOn_of_encard_image_eq h
  /-
    🎉 no goals
  -/


theorem ncard_image_iff (hs : s.Finite := by toFinite_tac) :
    (f '' s).ncard = s.ncard ↔ Set.InjOn f s :=
  ⟨fun h ↦ injOn_of_ncard_image_eq h hs, ncard_image_of_injOn⟩


theorem ncard_image_of_injective (s : Set α) (H : f.Injective) : (f '' s).ncard = s.ncard :=
  ncard_image_of_injOn fun _ _ _ _ h ↦ H h


theorem ncard_preimage_of_injective_subset_range {s : Set β} (H : f.Injective)
    (hs : s ⊆ Set.range f) :
    (f ⁻¹' s).ncard = s.ncard := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    s : Set β
    H : Function.Injective f
    hs : HasSubset.Subset s (Set.range f)
    ⊢ Eq (Set.preimage f s).ncard s.ncard
  -/
  rw [← ncard_image_of_injective _ H, image_preimage_eq_iff.mpr hs]
  /-
    🎉 no goals
  -/


theorem fiber_ncard_ne_zero_iff_mem_image {y : β} (hs : s.Finite := by toFinite_tac) :
    { x ∈ s | f x = y }.ncard ≠ 0 ↔ y ∈ f '' s := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    f : α → β
    y : β
    hs : autoParam s.Finite _auto✝
    ⊢ Iff (Ne (setOf fun x => And (Membership.mem s x) (Eq (f x) y)).ncard 0) (Mem …
  -/
  refine ⟨nonempty_of_ncard_ne_zero, ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    f : α → β
    y : β
    hs : autoParam s.Finite _auto✝
    ⊢ Membership.mem (Set.image f s) y → Ne (setOf fun x => And (Membership.mem s  …
  -/
  rintro ⟨z, hz, rfl⟩
  exact @ncard_ne_zero_of_mem _ ({ x ∈ s | f x = f z }) z (mem_sep hz rfl)
    (hs.subset (sep_subset _ _))


@[simp] theorem ncard_map (f : α ↪ β) : (f '' s).ncard = s.ncard :=
  ncard_image_of_injective _ f.inj'


@[simp] theorem ncard_subtype (P : α → Prop) (s : Set α) :
    { x : Subtype P | (x : α) ∈ s }.ncard = (s ∩ setOf P).ncard := by
  /-
    α : Type u_1
    P : α → Prop
    s : Set α
    ⊢ Eq (setOf fun x => Membership.mem s ↑x).ncard (Inter.inter s (setOf P)).ncard
  -/
  convert (ncard_image_of_injective _ (@Subtype.coe_injective _ P)).symm
  /-
    case h.e'_3.h.e'_2
    α : Type u_1
    P : α → Prop
    s : Set α
    ⊢ Eq (Inter.inter s (setOf P)) (Set.image (fun a => ↑a) (setOf fun x => Member …
  -/
  ext x
  /-
    case h.e'_3.h.e'_2.h
    α : Type u_1
    P : α → Prop
    s : Set α
    x : α
    ⊢ Iff (Membership.mem (Inter.inter s (setOf P)) x) (Membership.mem (Set.image  …
  -/
  simp [← and_assoc, exists_eq_right]
  /-
    🎉 no goals
  -/


theorem ncard_inter_le_ncard_left (s t : Set α) (hs : s.Finite := by toFinite_tac) :
    (s ∩ t).ncard ≤ s.ncard :=
  ncard_le_ncard inter_subset_left hs


theorem ncard_inter_le_ncard_right (s t : Set α) (ht : t.Finite := by toFinite_tac) :
    (s ∩ t).ncard ≤ t.ncard :=
  ncard_le_ncard inter_subset_right ht


theorem eq_of_subset_of_ncard_le (h : s ⊆ t) (h' : t.ncard ≤ s.ncard)
    (ht : t.Finite := by toFinite_tac) : s = t :=
  ht.eq_of_subset_of_encard_le h
        /-
          α : Type u_1
          s t : Set α
          h : HasSubset.Subset s t
          h' : LE.le t.ncard s.ncard
          ht : autoParam t.Finite _auto✝
          ⊢ LE.le t.encard s.encard
        -/
    (by rwa [← Nat.cast_le (α := ℕ∞), ht.cast_ncard_eq, (ht.subset h).cast_ncard_eq] at h')
        /-
          🎉 no goals
        -/


theorem subset_iff_eq_of_ncard_le (h : t.ncard ≤ s.ncard) (ht : t.Finite := by toFinite_tac) :
    s ⊆ t ↔ s = t :=
  ⟨fun hst ↦ eq_of_subset_of_ncard_le hst h ht, Eq.subset'⟩


theorem map_eq_of_subset {f : α ↪ α} (h : f '' s ⊆ s) (hs : s.Finite := by toFinite_tac) :
    f '' s = s :=
  eq_of_subset_of_ncard_le h (ncard_map _).ge hs


theorem sep_of_ncard_eq {a : α} {P : α → Prop} (h : { x ∈ s | P x }.ncard = s.ncard) (ha : a ∈ s)
    (hs : s.Finite := by toFinite_tac) : P a :=
                                                            /-
                                                              α : Type u_1
                                                              s : Set α
                                                              a : α
                                                              P : α → Prop
                                                              h : Eq (setOf fun x => And (Membership.mem s x) (P x)).ncard s.ncard
                                                              ha : Membership.mem s a
                                                              hs : autoParam s.Finite _auto✝
                                                              ⊢ HasSubset.Subset (setOf fun x => And (Membership.mem s x) (P x)) s
                                                            -/
  sep_eq_self_iff_mem_true.mp (eq_of_subset_of_ncard_le (by simp) h.symm.le hs) _ ha
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem ncard_lt_ncard (h : s ⊂ t) (ht : t.Finite := by toFinite_tac) :
    s.ncard < t.ncard := by
  /-
    α : Type u_1
    s t : Set α
    h : HasSSubset.SSubset s t
    ht : autoParam t.Finite _auto✝
    ⊢ LT.lt s.ncard t.ncard
  -/
  rw [← Nat.cast_lt (α := ℕ∞), ht.cast_ncard_eq, (ht.subset h.subset).cast_ncard_eq]
  /-
    α : Type u_1
    s t : Set α
    h : HasSSubset.SSubset s t
    ht : autoParam t.Finite _auto✝
    ⊢ LT.lt s.encard t.encard
  -/
  exact ht.encard_lt_encard h
  /-
    🎉 no goals
  -/


theorem ncard_strictMono [Finite α] : @StrictMono (Set α) _ _ _ ncard :=
              /-
                α : Type u_1
                inst✝ : Finite α
                x✝¹ x✝ : Set α
                h : LT.lt x✝¹ x✝
                ⊢ x✝.Finite
              -/
  fun _ _ h ↦ ncard_lt_ncard h
              /-
                🎉 no goals
              -/


theorem ncard_eq_of_bijective {n : ℕ} (f : ∀ i, i < n → α)
    (hf : ∀ a ∈ s, ∃ i, ∃ h : i < n, f i h = a) (hf' : ∀ (i) (h : i < n), f i h ∈ s)
    (f_inj : ∀ (i j) (hi : i < n) (hj : j < n), f i hi = f j hj → i = j) : s.ncard = n := by
  /-
    α : Type u_1
    s : Set α
    n : Nat
    f : (i : Nat) → LT.lt i n → α
    hf : ∀ (a : α), Membership.mem s a → Exists fun i => Exists fun h => Eq (f i h …
    hf' : ∀ (i : Nat) (h : LT.lt i n), Membership.mem s (f i h)
    f_inj : ∀ (i j : Nat) (hi : LT.lt i n) (hj : LT.lt j n), Eq (f i hi) (f j hj)  …
    ⊢ Eq s.ncard n
  -/
  let f' : Fin n → α := fun i ↦ f i.val i.is_lt
  suffices himage : s = f' '' Set.univ by
    rw [← Fintype.card_fin n, ← Nat.card_eq_fintype_card, ← Set.ncard_univ, himage]
    exact ncard_image_of_injOn <| fun i _hi j _hj h ↦ Fin.ext <| f_inj i.val j.val i.is_lt j.is_lt h
  /-
    α : Type u_1
    s : Set α
    n : Nat
    f : (i : Nat) → LT.lt i n → α
    hf : ∀ (a : α), Membership.mem s a → Exists fun i => Exists fun h => Eq (f i h …
    hf' : ∀ (i : Nat) (h : LT.lt i n), Membership.mem s (f i h)
    f_inj : ∀ (i j : Nat) (hi : LT.lt i n) (hj : LT.lt j n), Eq (f i hi) (f j hj)  …
    f' : Fin n → α := fun i => f ↑i ⋯
    ⊢ Eq s (Set.image f' Set.univ)
  -/
  ext x
  /-
    case h
    α : Type u_1
    s : Set α
    n : Nat
    f : (i : Nat) → LT.lt i n → α
    hf : ∀ (a : α), Membership.mem s a → Exists fun i => Exists fun h => Eq (f i h …
    hf' : ∀ (i : Nat) (h : LT.lt i n), Membership.mem s (f i h)
    f_inj : ∀ (i j : Nat) (hi : LT.lt i n) (hj : LT.lt j n), Eq (f i hi) (f j hj)  …
    f' : Fin n → α := fun i => f ↑i ⋯
    x : α
    ⊢ Iff (Membership.mem s x) (Membership.mem (Set.image f' Set.univ) x)
  -/
  simp only [image_univ, mem_range]
  /-
    case h
    α : Type u_1
    s : Set α
    n : Nat
    f : (i : Nat) → LT.lt i n → α
    hf : ∀ (a : α), Membership.mem s a → Exists fun i => Exists fun h => Eq (f i h …
    hf' : ∀ (i : Nat) (h : LT.lt i n), Membership.mem s (f i h)
    f_inj : ∀ (i j : Nat) (hi : LT.lt i n) (hj : LT.lt j n), Eq (f i hi) (f j hj)  …
    f' : Fin n → α := fun i => f ↑i ⋯
    x : α
    ⊢ Iff (Membership.mem s x) (Exists fun y => Eq (f' y) x)
  -/
  refine ⟨fun hx ↦ ?_, fun ⟨⟨i, hi⟩, hx⟩ ↦ hx ▸ hf' i hi⟩
  /-
    case h
    α : Type u_1
    s : Set α
    n : Nat
    f : (i : Nat) → LT.lt i n → α
    hf : ∀ (a : α), Membership.mem s a → Exists fun i => Exists fun h => Eq (f i h …
    hf' : ∀ (i : Nat) (h : LT.lt i n), Membership.mem s (f i h)
    f_inj : ∀ (i j : Nat) (hi : LT.lt i n) (hj : LT.lt j n), Eq (f i hi) (f j hj)  …
    f' : Fin n → α := fun i => f ↑i ⋯
    x : α
    hx : Membership.mem s x
    ⊢ Exists fun y => Eq (f' y) x
  -/
  obtain ⟨i, hi, rfl⟩ := hf x hx
  /-
    case h.intro.intro
    α : Type u_1
    s : Set α
    n : Nat
    f : (i : Nat) → LT.lt i n → α
    hf : ∀ (a : α), Membership.mem s a → Exists fun i => Exists fun h => Eq (f i h …
    hf' : ∀ (i : Nat) (h : LT.lt i n), Membership.mem s (f i h)
    f_inj : ∀ (i j : Nat) (hi : LT.lt i n) (hj : LT.lt j n), Eq (f i hi) (f j hj)  …
    f' : Fin n → α := fun i => f ↑i ⋯
    i : Nat
    hi : LT.lt i n
    hx : Membership.mem s (f i hi)
    ⊢ Exists fun y => Eq (f' y) (f i hi)
  -/
  use ⟨i, hi⟩
  /-
    🎉 no goals
  -/


theorem ncard_congr {t : Set β} (f : ∀ a ∈ s, β) (h₁ : ∀ a ha, f a ha ∈ t)
    (h₂ : ∀ a b ha hb, f a ha = f b hb → a = b) (h₃ : ∀ b ∈ t, ∃ a ha, f a ha = b) :
    s.ncard = t.ncard := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    f : (a : α) → Membership.mem s a → β
    h₁ : ∀ (a : α) (ha : Membership.mem s a), Membership.mem t (f a ha)
    h₂ : ∀ (a b : α) (ha : Membership.mem s a) (hb : Membership.mem s b), Eq (f a  …
    h₃ : ∀ (b : β), Membership.mem t b → Exists fun a => Exists fun ha => Eq (f a  …
    ⊢ Eq s.ncard t.ncard
  -/
  set f' : s → t := fun x ↦ ⟨f x.1 x.2, h₁ _ _⟩
  have hbij : f'.Bijective := by
    constructor
    · rintro ⟨x, hx⟩ ⟨y, hy⟩ hxy
      simp only [f', Subtype.mk.injEq] at hxy ⊢
      exact h₂ _ _ hx hy hxy
    rintro ⟨y, hy⟩
    obtain ⟨a, ha, rfl⟩ := h₃ y hy
    simp only [Subtype.mk.injEq, Subtype.exists]
    exact ⟨_, ha, rfl⟩
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    f : (a : α) → Membership.mem s a → β
    h₁ : ∀ (a : α) (ha : Membership.mem s a), Membership.mem t (f a ha)
    h₂ : ∀ (a b : α) (ha : Membership.mem s a) (hb : Membership.mem s b), Eq (f a  …
    h₃ : ∀ (b : β), Membership.mem t b → Exists fun a => Exists fun ha => Eq (f a  …
    f' : ↑s → ↑t := fun x => ⟨f ↑x ⋯, ⋯⟩
    hbij : Function.Bijective f'
    ⊢ Eq s.ncard t.ncard
  -/
  simp_rw [← Nat.card_coe_set_eq]
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    f : (a : α) → Membership.mem s a → β
    h₁ : ∀ (a : α) (ha : Membership.mem s a), Membership.mem t (f a ha)
    h₂ : ∀ (a b : α) (ha : Membership.mem s a) (hb : Membership.mem s b), Eq (f a  …
    h₃ : ∀ (b : β), Membership.mem t b → Exists fun a => Exists fun ha => Eq (f a  …
    f' : ↑s → ↑t := fun x => ⟨f ↑x ⋯, ⋯⟩
    hbij : Function.Bijective f'
    ⊢ Eq (Nat.card ↑s) (Nat.card ↑t)
  -/
  exact Nat.card_congr (Equiv.ofBijective f' hbij)
  /-
    🎉 no goals
  -/


theorem ncard_le_ncard_of_injOn {t : Set β} (f : α → β) (hf : ∀ a ∈ s, f a ∈ t) (f_inj : InjOn f s)
    (ht : t.Finite := by toFinite_tac) :
    s.ncard ≤ t.ncard := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    f : α → β
    hf : ∀ (a : α), Membership.mem s a → Membership.mem t (f a)
    f_inj : Set.InjOn f s
    ht : autoParam t.Finite _auto✝
    ⊢ LE.le s.ncard t.ncard
  -/
  have hle := encard_le_encard_of_injOn hf f_inj
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    f : α → β
    hf : ∀ (a : α), Membership.mem s a → Membership.mem t (f a)
    f_inj : Set.InjOn f s
    ht : autoParam t.Finite _auto✝
    hle : LE.le s.encard t.encard
    ⊢ LE.le s.ncard t.ncard
  -/
  to_encard_tac; rwa [ht.cast_ncard_eq, (ht.finite_of_encard_le hle).cast_ncard_eq]
                 /-
                   🎉 no goals
                 -/


theorem exists_ne_map_eq_of_ncard_lt_of_maps_to {t : Set β} (hc : t.ncard < s.ncard) {f : α → β}
    (hf : ∀ a ∈ s, f a ∈ t) (ht : t.Finite := by toFinite_tac) :
    ∃ x ∈ s, ∃ y ∈ s, x ≠ y ∧ f x = f y := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    hc : LT.lt t.ncard s.ncard
    f : α → β
    hf : ∀ (a : α), Membership.mem s a → Membership.mem t (f a)
    ht : autoParam t.Finite _auto✝
    ⊢ Exists fun x => And (Membership.mem s x) (Exists fun y => And (Membership.me …
  -/
  by_contra h'
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    hc : LT.lt t.ncard s.ncard
    f : α → β
    hf : ∀ (a : α), Membership.mem s a → Membership.mem t (f a)
    ht : autoParam t.Finite _auto✝
    h' : Not (Exists fun x => And (Membership.mem s x) (Exists fun y => And (Membe …
    ⊢ False
  -/
  simp only [Ne, exists_prop, not_exists, not_and, not_imp_not] at h'
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    hc : LT.lt t.ncard s.ncard
    f : α → β
    hf : ∀ (a : α), Membership.mem s a → Membership.mem t (f a)
    ht : autoParam t.Finite _auto✝
    h' : ∀ (x : α), Membership.mem s x → ∀ (x_1 : α), Membership.mem s x_1 → Eq (f …
    ⊢ False
  -/
  exact (ncard_le_ncard_of_injOn f hf h' ht).not_lt hc
  /-
    🎉 no goals
  -/


theorem le_ncard_of_inj_on_range {n : ℕ} (f : ℕ → α) (hf : ∀ i < n, f i ∈ s)
    (f_inj : ∀ i < n, ∀ j < n, f i = f j → i = j) (hs : s.Finite := by toFinite_tac) :
    n ≤ s.ncard := by
  /-
    α : Type u_1
    s : Set α
    n : Nat
    f : Nat → α
    hf : ∀ (i : Nat), LT.lt i n → Membership.mem s (f i)
    f_inj : ∀ (i : Nat), LT.lt i n → ∀ (j : Nat), LT.lt j n → Eq (f i) (f j) → Eq  …
    hs : autoParam s.Finite _auto✝
    ⊢ LE.le n s.ncard
  -/
  rw [ncard_eq_toFinset_card _ hs]
  /-
    α : Type u_1
    s : Set α
    n : Nat
    f : Nat → α
    hf : ∀ (i : Nat), LT.lt i n → Membership.mem s (f i)
    f_inj : ∀ (i : Nat), LT.lt i n → ∀ (j : Nat), LT.lt j n → Eq (f i) (f j) → Eq  …
    hs : autoParam s.Finite _auto✝
    ⊢ LE.le n (Set.Finite.toFinset hs).card
  -/
                                           /-
                                             🎉 no goals
                                           -/
  apply Finset.le_card_of_inj_on_range <;> simpa
                                           /-
                                             🎉 no goals
                                           -/


theorem surj_on_of_inj_on_of_ncard_le {t : Set β} (f : ∀ a ∈ s, β) (hf : ∀ a ha, f a ha ∈ t)
    (hinj : ∀ a₁ a₂ ha₁ ha₂, f a₁ ha₁ = f a₂ ha₂ → a₁ = a₂) (hst : t.ncard ≤ s.ncard)
    (ht : t.Finite := by toFinite_tac) :
    ∀ b ∈ t, ∃ a ha, b = f a ha := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    f : (a : α) → Membership.mem s a → β
    hf : ∀ (a : α) (ha : Membership.mem s a), Membership.mem t (f a ha)
    hinj : ∀ (a₁ a₂ : α) (ha₁ : Membership.mem s a₁) (ha₂ : Membership.mem s a₂),  …
    hst : LE.le t.ncard s.ncard
    ht : autoParam t.Finite _auto✝
    ⊢ ∀ (b : β), Membership.mem t b → Exists fun a => Exists fun ha => Eq b (f a ha)
  -/
  intro b hb
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    f : (a : α) → Membership.mem s a → β
    hf : ∀ (a : α) (ha : Membership.mem s a), Membership.mem t (f a ha)
    hinj : ∀ (a₁ a₂ : α) (ha₁ : Membership.mem s a₁) (ha₂ : Membership.mem s a₂),  …
    hst : LE.le t.ncard s.ncard
    ht : autoParam t.Finite _auto✝
    b : β
    hb : Membership.mem t b
    ⊢ Exists fun a => Exists fun ha => Eq b (f a ha)
  -/
  set f' : s → t := fun x ↦ ⟨f x.1 x.2, hf _ _⟩
  have finj : f'.Injective := by
    rintro ⟨x, hx⟩ ⟨y, hy⟩ hxy
    simp only [f', Subtype.mk.injEq] at hxy ⊢
    apply hinj _ _ hx hy hxy
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    f : (a : α) → Membership.mem s a → β
    hf : ∀ (a : α) (ha : Membership.mem s a), Membership.mem t (f a ha)
    hinj : ∀ (a₁ a₂ : α) (ha₁ : Membership.mem s a₁) (ha₂ : Membership.mem s a₂),  …
    hst : LE.le t.ncard s.ncard
    ht : autoParam t.Finite _auto✝
    b : β
    hb : Membership.mem t b
    f' : ↑s → ↑t := fun x => ⟨f ↑x ⋯, ⋯⟩
    finj : Function.Injective f'
    ⊢ Exists fun a => Exists fun ha => Eq b (f a ha)
  -/
  have hft := ht.fintype
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    f : (a : α) → Membership.mem s a → β
    hf : ∀ (a : α) (ha : Membership.mem s a), Membership.mem t (f a ha)
    hinj : ∀ (a₁ a₂ : α) (ha₁ : Membership.mem s a₁) (ha₂ : Membership.mem s a₂),  …
    hst : LE.le t.ncard s.ncard
    ht : autoParam t.Finite _auto✝
    b : β
    hb : Membership.mem t b
    f' : ↑s → ↑t := fun x => ⟨f ↑x ⋯, ⋯⟩
    finj : Function.Injective f'
    hft : Fintype ↑t
    ⊢ Exists fun a => Exists fun ha => Eq b (f a ha)
  -/
  have hft' := Fintype.ofInjective f' finj
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    f : (a : α) → Membership.mem s a → β
    hf : ∀ (a : α) (ha : Membership.mem s a), Membership.mem t (f a ha)
    hinj : ∀ (a₁ a₂ : α) (ha₁ : Membership.mem s a₁) (ha₂ : Membership.mem s a₂),  …
    hst : LE.le t.ncard s.ncard
    ht : autoParam t.Finite _auto✝
    b : β
    hb : Membership.mem t b
    f' : ↑s → ↑t := fun x => ⟨f ↑x ⋯, ⋯⟩
    finj : Function.Injective f'
    hft : Fintype ↑t
    hft' : Fintype ↑s
    ⊢ Exists fun a => Exists fun ha => Eq b (f a ha)
  -/
  set f'' : ∀ a, a ∈ s.toFinset → β := fun a h ↦ f a (by simpa using h)
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    f : (a : α) → Membership.mem s a → β
    hf : ∀ (a : α) (ha : Membership.mem s a), Membership.mem t (f a ha)
    hinj : ∀ (a₁ a₂ : α) (ha₁ : Membership.mem s a₁) (ha₂ : Membership.mem s a₂),  …
    hst : LE.le t.ncard s.ncard
    ht : autoParam t.Finite _auto✝
    b : β
    hb : Membership.mem t b
    f' : ↑s → ↑t := fun x => ⟨f ↑x ⋯, ⋯⟩
    finj : Function.Injective f'
    hft : Fintype ↑t
    hft' : Fintype ↑s
    f'' : (a : α) → Membership.mem s.toFinset a → β := fun a h => f a ⋯
    ⊢ Exists fun a => Exists fun ha => Eq b (f a ha)
  -/
  convert @Finset.surj_on_of_inj_on_of_card_le _ _ _ t.toFinset f'' _ _ _ _ (by simpa) using 1
    /-
      case h.e'_2
      α : Type u_1
      β : Type u_2
      s : Set α
      t : Set β
      f : (a : α) → Membership.mem s a → β
      hf : ∀ (a : α) (ha : Membership.mem s a), Membership.mem t (f a ha)
      hinj : ∀ (a₁ a₂ : α) (ha₁ : Membership.mem s a₁) (ha₂ : Membership.mem s a₂),  …
      hst : LE.le t.ncard s.ncard
      ht : autoParam t.Finite _auto✝
      b : β
      hb : Membership.mem t b
      f' : ↑s → ↑t := fun x => ⟨f ↑x ⋯, ⋯⟩
      finj : Function.Injective f'
      hft : Fintype ↑t
      hft' : Fintype ↑s
      f'' : (a : α) → Membership.mem s.toFinset a → β := fun a h => f a ⋯
      ⊢ Eq (fun a => Exists fun ha => Eq b (f a ha)) fun a => Exists fun ha => Eq b  …
    -/
  · simp [f'']
    /-
      🎉 no goals
    -/
    /-
      case convert_1
      α : Type u_1
      β : Type u_2
      s : Set α
      t : Set β
      f : (a : α) → Membership.mem s a → β
      hf : ∀ (a : α) (ha : Membership.mem s a), Membership.mem t (f a ha)
      hinj : ∀ (a₁ a₂ : α) (ha₁ : Membership.mem s a₁) (ha₂ : Membership.mem s a₂),  …
      hst : LE.le t.ncard s.ncard
      ht : autoParam t.Finite _auto✝
      b : β
      hb : Membership.mem t b
      f' : ↑s → ↑t := fun x => ⟨f ↑x ⋯, ⋯⟩
      finj : Function.Injective f'
      hft : Fintype ↑t
      hft' : Fintype ↑s
      f'' : (a : α) → Membership.mem s.toFinset a → β := fun a h => f a ⋯
      ⊢ ∀ (a : α) (ha : Membership.mem s.toFinset a), Membership.mem t.toFinset (f'' …
    -/
  · simp [f'', hf]
    /-
      🎉 no goals
    -/
    /-
      case convert_2
      α : Type u_1
      β : Type u_2
      s : Set α
      t : Set β
      f : (a : α) → Membership.mem s a → β
      hf : ∀ (a : α) (ha : Membership.mem s a), Membership.mem t (f a ha)
      hinj : ∀ (a₁ a₂ : α) (ha₁ : Membership.mem s a₁) (ha₂ : Membership.mem s a₂),  …
      hst : LE.le t.ncard s.ncard
      ht : autoParam t.Finite _auto✝
      b : β
      hb : Membership.mem t b
      f' : ↑s → ↑t := fun x => ⟨f ↑x ⋯, ⋯⟩
      finj : Function.Injective f'
      hft : Fintype ↑t
      hft' : Fintype ↑s
      f'' : (a : α) → Membership.mem s.toFinset a → β := fun a h => f a ⋯
      ⊢ ∀ (a₁ a₂ : α) (ha₁ : Membership.mem s.toFinset a₁) (ha₂ : Membership.mem s.t …
    -/
  · intros a₁ a₂ ha₁ ha₂ h
    /-
      case convert_2
      α : Type u_1
      β : Type u_2
      s : Set α
      t : Set β
      f : (a : α) → Membership.mem s a → β
      hf : ∀ (a : α) (ha : Membership.mem s a), Membership.mem t (f a ha)
      hinj : ∀ (a₁ a₂ : α) (ha₁ : Membership.mem s a₁) (ha₂ : Membership.mem s a₂),  …
      hst : LE.le t.ncard s.ncard
      ht : autoParam t.Finite _auto✝
      b : β
      hb : Membership.mem t b
      f' : ↑s → ↑t := fun x => ⟨f ↑x ⋯, ⋯⟩
      finj : Function.Injective f'
      hft : Fintype ↑t
      hft' : Fintype ↑s
      f'' : (a : α) → Membership.mem s.toFinset a → β := fun a h => f a ⋯
      a₁ a₂ : α
      ha₁ : Membership.mem s.toFinset a₁
      ha₂ : Membership.mem s.toFinset a₂
      h : Eq (f'' a₁ ha₁) (f'' a₂ ha₂)
      ⊢ Eq a₁ a₂
    -/
    rw [mem_toFinset] at ha₁ ha₂
    /-
      case convert_2
      α : Type u_1
      β : Type u_2
      s : Set α
      t : Set β
      f : (a : α) → Membership.mem s a → β
      hf : ∀ (a : α) (ha : Membership.mem s a), Membership.mem t (f a ha)
      hinj : ∀ (a₁ a₂ : α) (ha₁ : Membership.mem s a₁) (ha₂ : Membership.mem s a₂),  …
      hst : LE.le t.ncard s.ncard
      ht : autoParam t.Finite _auto✝
      b : β
      hb : Membership.mem t b
      f' : ↑s → ↑t := fun x => ⟨f ↑x ⋯, ⋯⟩
      finj : Function.Injective f'
      hft : Fintype ↑t
      hft' : Fintype ↑s
      f'' : (a : α) → Membership.mem s.toFinset a → β := fun a h => f a ⋯
      a₁ a₂ : α
      ha₁✝ : Membership.mem s.toFinset a₁
      ha₁ : Membership.mem s a₁
      ha₂✝ : Membership.mem s.toFinset a₂
      ha₂ : Membership.mem s a₂
      h : Eq (f'' a₁ ha₁✝) (f'' a₂ ha₂✝)
      ⊢ Eq a₁ a₂
    -/
    exact hinj _ _ ha₁ ha₂ h
    /-
      🎉 no goals
    -/
  /-
    case convert_3
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    f : (a : α) → Membership.mem s a → β
    hf : ∀ (a : α) (ha : Membership.mem s a), Membership.mem t (f a ha)
    hinj : ∀ (a₁ a₂ : α) (ha₁ : Membership.mem s a₁) (ha₂ : Membership.mem s a₂),  …
    hst : LE.le t.ncard s.ncard
    ht : autoParam t.Finite _auto✝
    b : β
    hb : Membership.mem t b
    f' : ↑s → ↑t := fun x => ⟨f ↑x ⋯, ⋯⟩
    finj : Function.Injective f'
    hft : Fintype ↑t
    hft' : Fintype ↑s
    f'' : (a : α) → Membership.mem s.toFinset a → β := fun a h => f a ⋯
    ⊢ LE.le t.toFinset.card s.toFinset.card
  -/
  rwa [← ncard_eq_toFinset_card', ← ncard_eq_toFinset_card']
  /-
    🎉 no goals
  -/


theorem inj_on_of_surj_on_of_ncard_le {t : Set β} (f : ∀ a ∈ s, β) (hf : ∀ a ha, f a ha ∈ t)
    (hsurj : ∀ b ∈ t, ∃ a ha, f a ha = b) (hst : s.ncard ≤ t.ncard) ⦃a₁⦄ (ha₁ : a₁ ∈ s) ⦃a₂⦄
    (ha₂ : a₂ ∈ s) (ha₁a₂ : f a₁ ha₁ = f a₂ ha₂) (hs : s.Finite := by toFinite_tac) :
    a₁ = a₂ := by
  classical
  set f' : s → t := fun x ↦ ⟨f x.1 x.2, hf _ _⟩
  have hsurj : f'.Surjective := by
    rintro ⟨y, hy⟩
    obtain ⟨a, ha, rfl⟩ := hsurj y hy
    simp only [Subtype.mk.injEq, Subtype.exists]
    exact ⟨_, ha, rfl⟩
  haveI := hs.fintype
  haveI := Fintype.ofSurjective _ hsurj
  set f'' : ∀ a, a ∈ s.toFinset → β := fun a h ↦ f a (by simpa using h)
  exact
    @Finset.inj_on_of_surj_on_of_card_le _ _ _ t.toFinset f''
      (fun a ha ↦ by { rw [mem_toFinset] at ha ⊢; exact hf a ha }) (by simpa)
      (by { rwa [← ncard_eq_toFinset_card', ← ncard_eq_toFinset_card'] }) a₁
      (by simpa) a₂ (by simpa) (by simpa)


@[simp] lemma ncard_graphOn (s : Set α) (f : α → β) : (s.graphOn f).ncard = s.ncard := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    f : α → β
    ⊢ Eq (Set.graphOn f s).ncard s.ncard
  -/
  rw [← ncard_image_of_injOn fst_injOn_graph, image_fst_graphOn]
  /-
    🎉 no goals
  -/


theorem ncard_union_add_ncard_inter (s t : Set α) (hs : s.Finite := by toFinite_tac)
    (ht : t.Finite := by toFinite_tac) : (s ∪ t).ncard + (s ∩ t).ncard = s.ncard + t.ncard := by
  /-
    α : Type u_1
    s t : Set α
    hs : autoParam s.Finite _auto✝
    ht : autoParam t.Finite _auto✝
    ⊢ Eq (HAdd.hAdd (Union.union s t).ncard (Inter.inter s t).ncard) (HAdd.hAdd s. …
  -/
  to_encard_tac; rw [hs.cast_ncard_eq, ht.cast_ncard_eq, (hs.union ht).cast_ncard_eq,
    (hs.subset inter_subset_left).cast_ncard_eq, encard_union_add_encard_inter]


theorem ncard_inter_add_ncard_union (s t : Set α) (hs : s.Finite := by toFinite_tac)
    (ht : t.Finite := by toFinite_tac) : (s ∩ t).ncard + (s ∪ t).ncard = s.ncard + t.ncard := by
  /-
    α : Type u_1
    s t : Set α
    hs : autoParam s.Finite _auto✝
    ht : autoParam t.Finite _auto✝
    ⊢ Eq (HAdd.hAdd (Inter.inter s t).ncard (Union.union s t).ncard) (HAdd.hAdd s. …
  -/
  rw [add_comm, ncard_union_add_ncard_inter _ _ hs ht]
  /-
    🎉 no goals
  -/


theorem ncard_union_le (s t : Set α) : (s ∪ t).ncard ≤ s.ncard + t.ncard := by
  /-
    α : Type u_1
    s t : Set α
    ⊢ LE.le (Union.union s t).ncard (HAdd.hAdd s.ncard t.ncard)
  -/
  obtain (h | h) := (s ∪ t).finite_or_infinite
    /-
      case inl
      α : Type u_1
      s t : Set α
      h : (Union.union s t).Finite
      ⊢ LE.le (Union.union s t).ncard (HAdd.hAdd s.ncard t.ncard)
    -/
  · to_encard_tac
    rw [h.cast_ncard_eq, (h.subset subset_union_left).cast_ncard_eq,
      (h.subset subset_union_right).cast_ncard_eq]
    /-
      case inl
      α : Type u_1
      s t : Set α
      h : (Union.union s t).Finite
      ⊢ LE.le (Union.union s t).encard (HAdd.hAdd s.encard t.encard)
    -/
    apply encard_union_le
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    s t : Set α
    h : (Union.union s t).Infinite
    ⊢ LE.le (Union.union s t).ncard (HAdd.hAdd s.ncard t.ncard)
  -/
  rw [h.ncard]
  /-
    case inr
    α : Type u_1
    s t : Set α
    h : (Union.union s t).Infinite
    ⊢ LE.le 0 (HAdd.hAdd s.ncard t.ncard)
  -/
  apply zero_le
  /-
    🎉 no goals
  -/


theorem ncard_union_eq (h : Disjoint s t) (hs : s.Finite := by toFinite_tac)
    (ht : t.Finite := by toFinite_tac) : (s ∪ t).ncard = s.ncard + t.ncard := by
  /-
    α : Type u_1
    s t : Set α
    h : Disjoint s t
    hs : autoParam s.Finite _auto✝
    ht : autoParam t.Finite _auto✝
    ⊢ Eq (Union.union s t).ncard (HAdd.hAdd s.ncard t.ncard)
  -/
  to_encard_tac
  /-
    α : Type u_1
    s t : Set α
    h : Disjoint s t
    hs : autoParam s.Finite _auto✝
    ht : autoParam t.Finite _auto✝
    ⊢ Eq (↑(Union.union s t).ncard) (HAdd.hAdd ↑s.ncard ↑t.ncard)
  -/
  rw [hs.cast_ncard_eq, ht.cast_ncard_eq, (hs.union ht).cast_ncard_eq, encard_union_eq h]
  /-
    🎉 no goals
  -/


theorem ncard_diff_add_ncard_of_subset (h : s ⊆ t) (ht : t.Finite := by toFinite_tac) :
    (t \ s).ncard + s.ncard = t.ncard := by
  /-
    α : Type u_1
    s t : Set α
    h : HasSubset.Subset s t
    ht : autoParam t.Finite _auto✝
    ⊢ Eq (HAdd.hAdd (SDiff.sdiff t s).ncard s.ncard) t.ncard
  -/
  to_encard_tac
  rw [ht.cast_ncard_eq, (ht.subset h).cast_ncard_eq, (ht.diff _).cast_ncard_eq,
    encard_diff_add_encard_of_subset h]


theorem ncard_diff (hst : s ⊆ t) (hs : s.Finite := by toFinite_tac) :
    (t \ s).ncard = t.ncard - s.ncard := by
  /-
    α : Type u_1
    s t : Set α
    hst : HasSubset.Subset s t
    hs : autoParam s.Finite _auto✝
    ⊢ Eq (SDiff.sdiff t s).ncard (HSub.hSub t.ncard s.ncard)
  -/
  obtain ht | ht := t.finite_or_infinite
    /-
      case inl
      α : Type u_1
      s t : Set α
      hst : HasSubset.Subset s t
      hs : autoParam s.Finite _auto✝
      ht : t.Finite
      ⊢ Eq (SDiff.sdiff t s).ncard (HSub.hSub t.ncard s.ncard)
    -/
  · rw [← ncard_diff_add_ncard_of_subset hst ht, add_tsub_cancel_right]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      s t : Set α
      hst : HasSubset.Subset s t
      hs : autoParam s.Finite _auto✝
      ht : t.Infinite
      ⊢ Eq (SDiff.sdiff t s).ncard (HSub.hSub t.ncard s.ncard)
    -/
  · rw [ht.ncard, Nat.zero_sub, (ht.diff hs).ncard]
    /-
      🎉 no goals
    -/


lemma cast_ncard_sdiff {R : Type*} [AddGroupWithOne R] (hst : s ⊆ t) (ht : t.Finite) :
    ((t \ s).ncard : R) = t.ncard - s.ncard := by
  /-
    α : Type u_1
    s t : Set α
    R : Type u_3
    inst✝ : AddGroupWithOne R
    hst : HasSubset.Subset s t
    ht : t.Finite
    ⊢ Eq (↑(SDiff.sdiff t s).ncard) (HSub.hSub ↑t.ncard ↑s.ncard)
  -/
  rw [ncard_diff hst (ht.subset hst), Nat.cast_sub (ncard_le_ncard hst ht)]
  /-
    🎉 no goals
  -/


theorem ncard_le_ncard_diff_add_ncard (s t : Set α) (ht : t.Finite := by toFinite_tac) :
    s.ncard ≤ (s \ t).ncard + t.ncard := by
  /-
    α : Type u_1
    s t : Set α
    ht : autoParam t.Finite _auto✝
    ⊢ LE.le s.ncard (HAdd.hAdd (SDiff.sdiff s t).ncard t.ncard)
  -/
  cases' s.finite_or_infinite with hs hs
    /-
      case inl
      α : Type u_1
      s t : Set α
      ht : autoParam t.Finite _auto✝
      hs : s.Finite
      ⊢ LE.le s.ncard (HAdd.hAdd (SDiff.sdiff s t).ncard t.ncard)
    -/
  · to_encard_tac
    /-
      case inl
      α : Type u_1
      s t : Set α
      ht : autoParam t.Finite _auto✝
      hs : s.Finite
      ⊢ LE.le (↑s.ncard) (HAdd.hAdd ↑(SDiff.sdiff s t).ncard ↑t.ncard)
    -/
    rw [ht.cast_ncard_eq, hs.cast_ncard_eq, (hs.diff _).cast_ncard_eq]
    /-
      case inl
      α : Type u_1
      s t : Set α
      ht : autoParam t.Finite _auto✝
      hs : s.Finite
      ⊢ LE.le s.encard (HAdd.hAdd (SDiff.sdiff s t).encard t.encard)
    -/
    apply encard_le_encard_diff_add_encard
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    s t : Set α
    ht : autoParam t.Finite _auto✝
    hs : s.Infinite
    ⊢ LE.le s.ncard (HAdd.hAdd (SDiff.sdiff s t).ncard t.ncard)
  -/
  convert Nat.zero_le _
  /-
    case h.e'_3
    α : Type u_1
    s t : Set α
    ht : autoParam t.Finite _auto✝
    hs : s.Infinite
    ⊢ Eq s.ncard 0
  -/
  rw [hs.ncard]
  /-
    🎉 no goals
  -/


theorem le_ncard_diff (s t : Set α) (hs : s.Finite := by toFinite_tac) :
    t.ncard - s.ncard ≤ (t \ s).ncard :=
                           /-
                             α : Type u_1
                             s t : Set α
                             hs : autoParam s.Finite _auto✝
                             ⊢ LE.le t.ncard (HAdd.hAdd s.ncard (SDiff.sdiff t s).ncard)
                           -/
  tsub_le_iff_left.mpr (by rw [add_comm]; apply ncard_le_ncard_diff_add_ncard _ _ hs)
                                          /-
                                            🎉 no goals
                                          -/


theorem ncard_diff_add_ncard (s t : Set α) (hs : s.Finite := by toFinite_tac)
  (ht : t.Finite := by toFinite_tac) :
    (s \ t).ncard + t.ncard = (s ∪ t).ncard := by
  /-
    α : Type u_1
    s t : Set α
    hs : autoParam s.Finite _auto✝
    ht : autoParam t.Finite _auto✝
    ⊢ Eq (HAdd.hAdd (SDiff.sdiff s t).ncard t.ncard) (Union.union s t).ncard
  -/
  rw [← ncard_union_eq disjoint_sdiff_left (hs.diff _) ht, diff_union_self]
  /-
    🎉 no goals
  -/


theorem diff_nonempty_of_ncard_lt_ncard (h : s.ncard < t.ncard) (hs : s.Finite := by toFinite_tac) :
    (t \ s).Nonempty := by
  /-
    α : Type u_1
    s t : Set α
    h : LT.lt s.ncard t.ncard
    hs : autoParam s.Finite _auto✝
    ⊢ (SDiff.sdiff t s).Nonempty
  -/
  rw [Set.nonempty_iff_ne_empty, Ne, diff_eq_empty]
  /-
    α : Type u_1
    s t : Set α
    h : LT.lt s.ncard t.ncard
    hs : autoParam s.Finite _auto✝
    ⊢ Not (HasSubset.Subset t s)
  -/
  exact fun h' ↦ h.not_le (ncard_le_ncard h' hs)
  /-
    🎉 no goals
  -/


theorem exists_mem_not_mem_of_ncard_lt_ncard (h : s.ncard < t.ncard)
    (hs : s.Finite := by toFinite_tac) : ∃ e, e ∈ t ∧ e ∉ s :=
  diff_nonempty_of_ncard_lt_ncard h hs


@[simp] theorem ncard_inter_add_ncard_diff_eq_ncard (s t : Set α)
    (hs : s.Finite := by toFinite_tac) : (s ∩ t).ncard + (s \ t).ncard = s.ncard := by
  rw [← ncard_union_eq (disjoint_of_subset_left inter_subset_right disjoint_sdiff_right)
    (hs.inter_of_left _) (hs.diff _), union_comm, diff_union_inter]


theorem ncard_eq_ncard_iff_ncard_diff_eq_ncard_diff (hs : s.Finite := by toFinite_tac)
    (ht : t.Finite := by toFinite_tac) : s.ncard = t.ncard ↔ (s \ t).ncard = (t \ s).ncard := by
  rw [← ncard_inter_add_ncard_diff_eq_ncard s t hs, ← ncard_inter_add_ncard_diff_eq_ncard t s ht,
    inter_comm, add_right_inj]


theorem ncard_le_ncard_iff_ncard_diff_le_ncard_diff (hs : s.Finite := by toFinite_tac)
    (ht : t.Finite := by toFinite_tac) : s.ncard ≤ t.ncard ↔ (s \ t).ncard ≤ (t \ s).ncard := by
  rw [← ncard_inter_add_ncard_diff_eq_ncard s t hs, ← ncard_inter_add_ncard_diff_eq_ncard t s ht,
    inter_comm, add_le_add_iff_left]


theorem ncard_lt_ncard_iff_ncard_diff_lt_ncard_diff (hs : s.Finite := by toFinite_tac)
    (ht : t.Finite := by toFinite_tac) : s.ncard < t.ncard ↔ (s \ t).ncard < (t \ s).ncard := by
  rw [← ncard_inter_add_ncard_diff_eq_ncard s t hs, ← ncard_inter_add_ncard_diff_eq_ncard t s ht,
    inter_comm, add_lt_add_iff_left]


theorem ncard_add_ncard_compl (s : Set α) (hs : s.Finite := by toFinite_tac)
    (hsc : sᶜ.Finite := by toFinite_tac) : s.ncard + sᶜ.ncard = Nat.card α := by
  /-
    α : Type u_1
    s : Set α
    hs : autoParam s.Finite _auto✝
    hsc : autoParam (HasCompl.compl s).Finite _auto✝
    ⊢ Eq (HAdd.hAdd s.ncard (HasCompl.compl s).ncard) (Nat.card α)
  -/
  rw [← ncard_univ, ← ncard_union_eq (@disjoint_compl_right _ _ s) hs hsc, union_compl_self]
  /-
    🎉 no goals
  -/


/-- Given a subset `s` of a set `t`, of sizes at most and at least `n` respectively, there exists a
set `u` of size `n` which is both a superset of `s` and a subset of `t`. -/
lemma exists_subsuperset_card_eq {n : ℕ} (hst : s ⊆ t) (hsn : s.ncard ≤ n) (hnt : n ≤ t.ncard) :
    ∃ u, s ⊆ u ∧ u ⊆ t ∧ u.ncard = n := by
  /-
    α : Type u_1
    s t : Set α
    n : Nat
    hst : HasSubset.Subset s t
    hsn : LE.le s.ncard n
    hnt : LE.le n t.ncard
    ⊢ Exists fun u => And (HasSubset.Subset s u) (And (HasSubset.Subset u t) (Eq u …
  -/
  obtain ht | ht := t.infinite_or_finite
    /-
      case inl
      α : Type u_1
      s t : Set α
      n : Nat
      hst : HasSubset.Subset s t
      hsn : LE.le s.ncard n
      hnt : LE.le n t.ncard
      ht : t.Infinite
      ⊢ Exists fun u => And (HasSubset.Subset s u) (And (HasSubset.Subset u t) (Eq u …
    -/
  · rw [ht.ncard, Nat.le_zero, ← ht.ncard] at hnt
    /-
      case inl
      α : Type u_1
      s t : Set α
      n : Nat
      hst : HasSubset.Subset s t
      hsn : LE.le s.ncard n
      hnt : Eq n t.ncard
      ht : t.Infinite
      ⊢ Exists fun u => And (HasSubset.Subset s u) (And (HasSubset.Subset u t) (Eq u …
    -/
    exact ⟨t, hst, Subset.rfl, hnt.symm⟩
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    s t : Set α
    n : Nat
    hst : HasSubset.Subset s t
    hsn : LE.le s.ncard n
    hnt : LE.le n t.ncard
    ht : t.Finite
    ⊢ Exists fun u => And (HasSubset.Subset s u) (And (HasSubset.Subset u t) (Eq u …
  -/
  lift s to Finset α using ht.subset hst
  /-
    case inr.intro
    α : Type u_1
    t : Set α
    n : Nat
    hnt : LE.le n t.ncard
    ht : t.Finite
    s : Finset α
    hst : HasSubset.Subset (↑s) t
    hsn : LE.le (↑s).ncard n
    ⊢ Exists fun u => And (HasSubset.Subset (↑s) u) (And (HasSubset.Subset u t) (E …
  -/
  lift t to Finset α using ht
  obtain ⟨u, hsu, hut, hu⟩ := Finset.exists_subsuperset_card_eq (mod_cast hst) (by simpa using hsn)
    (mod_cast hnt)
  /-
    case inr.intro.intro.intro.intro.intro
    α : Type u_1
    n : Nat
    s : Finset α
    hsn : LE.le (↑s).ncard n
    t : Finset α
    hnt : LE.le n (↑t).ncard
    hst : HasSubset.Subset ↑s ↑t
    u : Finset α
    hsu : HasSubset.Subset s u
    hut : HasSubset.Subset u t
    hu : Eq u.card n
    ⊢ Exists fun u => And (HasSubset.Subset (↑s) u) (And (HasSubset.Subset u ↑t) ( …
  -/
  exact ⟨u, mod_cast hsu, mod_cast hut, mod_cast hu⟩
  /-
    🎉 no goals
  -/


/-- We can shrink a set to any smaller size. -/
lemma exists_subset_card_eq {n : ℕ} (hns : n ≤ s.ncard) : ∃ t ⊆ s, t.ncard = n := by
  /-
    α : Type u_1
    s : Set α
    n : Nat
    hns : LE.le n s.ncard
    ⊢ Exists fun t => And (HasSubset.Subset t s) (Eq t.ncard n)
  -/
  simpa using exists_subsuperset_card_eq s.empty_subset (by simp) hns
  /-
    🎉 no goals
  -/


/-- Given a set `t` and a set `s` inside it, we can shrink `t` to any appropriate size, and keep `s`
    inside it. -/
@[deprecated exists_subsuperset_card_eq (since := "2024-06-24")]
theorem exists_intermediate_Set (i : ℕ) (h₁ : i + s.ncard ≤ t.ncard) (h₂ : s ⊆ t) :
    ∃ r : Set α, s ⊆ r ∧ r ⊆ t ∧ r.ncard = i + s.ncard :=
  exists_subsuperset_card_eq h₂ (Nat.le_add_left ..) h₁


@[deprecated exists_subsuperset_card_eq (since := "2024-06-24")]
theorem exists_intermediate_set' {m : ℕ} (hs : s.ncard ≤ m) (ht : m ≤ t.ncard) (h : s ⊆ t) :
    ∃ r : Set α, s ⊆ r ∧ r ⊆ t ∧ r.ncard = m := exists_subsuperset_card_eq h hs ht


/-- We can shrink `s` to any smaller size. -/
@[deprecated exists_subset_card_eq (since := "2024-06-23")]
theorem exists_smaller_set (s : Set α) (i : ℕ) (h₁ : i ≤ s.ncard) :
    ∃ t : Set α, t ⊆ s ∧ t.ncard = i := exists_subset_card_eq h₁


theorem Infinite.exists_subset_ncard_eq {s : Set α} (hs : s.Infinite) (k : ℕ) :
    ∃ t, t ⊆ s ∧ t.Finite ∧ t.ncard = k := by
  /-
    α : Type u_1
    s : Set α
    hs : s.Infinite
    k : Nat
    ⊢ Exists fun t => And (HasSubset.Subset t s) (And t.Finite (Eq t.ncard k))
  -/
  have := hs.to_subtype
  /-
    α : Type u_1
    s : Set α
    hs : s.Infinite
    k : Nat
    this : Infinite ↑s
    ⊢ Exists fun t => And (HasSubset.Subset t s) (And t.Finite (Eq t.ncard k))
  -/
  obtain ⟨t', -, rfl⟩ := @Infinite.exists_subset_card_eq s univ infinite_univ k
  /-
    case intro.intro
    α : Type u_1
    s : Set α
    hs : s.Infinite
    this : Infinite ↑s
    t' : Finset ↑s
    ⊢ Exists fun t => And (HasSubset.Subset t s) (And t.Finite (Eq t.ncard t'.card))
  -/
  refine ⟨Subtype.val '' (t' : Set s), by simp, Finite.image _ (by simp), ?_⟩
  /-
    case intro.intro
    α : Type u_1
    s : Set α
    hs : s.Infinite
    this : Infinite ↑s
    t' : Finset ↑s
    ⊢ Eq (Set.image Subtype.val ↑t').ncard t'.card
  -/
  rw [ncard_image_of_injective _ Subtype.coe_injective]
  /-
    case intro.intro
    α : Type u_1
    s : Set α
    hs : s.Infinite
    this : Infinite ↑s
    t' : Finset ↑s
    ⊢ Eq (↑t').ncard t'.card
  -/
  simp
  /-
    🎉 no goals
  -/


theorem Infinite.exists_superset_ncard_eq {s t : Set α} (ht : t.Infinite) (hst : s ⊆ t)
    (hs : s.Finite) {k : ℕ} (hsk : s.ncard ≤ k) : ∃ s', s ⊆ s' ∧ s' ⊆ t ∧ s'.ncard = k := by
  /-
    α : Type u_1
    s t : Set α
    ht : t.Infinite
    hst : HasSubset.Subset s t
    hs : s.Finite
    k : Nat
    hsk : LE.le s.ncard k
    ⊢ Exists fun s' => And (HasSubset.Subset s s') (And (HasSubset.Subset s' t) (E …
  -/
  obtain ⟨s₁, hs₁, hs₁fin, hs₁card⟩ := (ht.diff hs).exists_subset_ncard_eq (k - s.ncard)
  /-
    case intro.intro.intro
    α : Type u_1
    s t : Set α
    ht : t.Infinite
    hst : HasSubset.Subset s t
    hs : s.Finite
    k : Nat
    hsk : LE.le s.ncard k
    s₁ : Set α
    hs₁ : HasSubset.Subset s₁ (SDiff.sdiff t s)
    hs₁fin : s₁.Finite
    hs₁card : Eq s₁.ncard (HSub.hSub k s.ncard)
    ⊢ Exists fun s' => And (HasSubset.Subset s s') (And (HasSubset.Subset s' t) (E …
  -/
  refine ⟨s ∪ s₁, subset_union_left, union_subset hst (hs₁.trans diff_subset), ?_⟩
  rwa [ncard_union_eq (disjoint_of_subset_right hs₁ disjoint_sdiff_right) hs hs₁fin, hs₁card,
    add_tsub_cancel_of_le]


theorem exists_subset_or_subset_of_two_mul_lt_ncard {n : ℕ} (hst : 2 * n < (s ∪ t).ncard) :
    ∃ r : Set α, n < r.ncard ∧ (r ⊆ s ∨ r ⊆ t) := by
  classical
  have hu := finite_of_ncard_ne_zero ((Nat.zero_le _).trans_lt hst).ne.symm
  rw [ncard_eq_toFinset_card _ hu,
    Finite.toFinset_union (hu.subset subset_union_left)
      (hu.subset subset_union_right)] at hst
  obtain ⟨r', hnr', hr'⟩ := Finset.exists_subset_or_subset_of_two_mul_lt_card hst
  exact ⟨r', by simpa, by simpa using hr'⟩


@[simp] theorem ncard_eq_one : s.ncard = 1 ↔ ∃ a, s = {a} := by
  /-
    α : Type u_1
    s : Set α
    ⊢ Iff (Eq s.ncard 1) (Exists fun a => Eq s (Singleton.singleton a))
  -/
  refine ⟨fun h ↦ ?_, by rintro ⟨a, rfl⟩; rw [ncard_singleton]⟩
  /-
    α : Type u_1
    s : Set α
    h : Eq s.ncard 1
    ⊢ Exists fun a => Eq s (Singleton.singleton a)
  -/
  have hft := (finite_of_ncard_ne_zero (ne_zero_of_eq_one h)).fintype
  /-
    α : Type u_1
    s : Set α
    h : Eq s.ncard 1
    hft : Fintype ↑s
    ⊢ Exists fun a => Eq s (Singleton.singleton a)
  -/
  simp_rw [ncard_eq_toFinset_card', @Finset.card_eq_one _ (toFinset s)] at h
  /-
    α : Type u_1
    s : Set α
    hft : Fintype ↑s
    h : Exists fun a => Eq s.toFinset (Singleton.singleton a)
    ⊢ Exists fun a => Eq s (Singleton.singleton a)
  -/
  refine h.imp fun a ha ↦ ?_
  /-
    α : Type u_1
    s : Set α
    hft : Fintype ↑s
    h : Exists fun a => Eq s.toFinset (Singleton.singleton a)
    a : α
    ha : Eq s.toFinset (Singleton.singleton a)
    ⊢ Eq s (Singleton.singleton a)
  -/
  simp_rw [Set.ext_iff, mem_singleton_iff]
  /-
    α : Type u_1
    s : Set α
    hft : Fintype ↑s
    h : Exists fun a => Eq s.toFinset (Singleton.singleton a)
    a : α
    ha : Eq s.toFinset (Singleton.singleton a)
    ⊢ ∀ (x : α), Iff (Membership.mem s x) (Eq x a)
  -/
  simp only [Finset.ext_iff, mem_toFinset, Finset.mem_singleton] at ha
  /-
    α : Type u_1
    s : Set α
    hft : Fintype ↑s
    h : Exists fun a => Eq s.toFinset (Singleton.singleton a)
    a : α
    ha : ∀ (a_1 : α), Iff (Membership.mem s a_1) (Eq a_1 a)
    ⊢ ∀ (x : α), Iff (Membership.mem s x) (Eq x a)
  -/
  exact ha
  /-
    🎉 no goals
  -/


theorem exists_eq_insert_iff_ncard (hs : s.Finite := by toFinite_tac) :
    (∃ a ∉ s, insert a s = t) ↔ s ⊆ t ∧ s.ncard + 1 = t.ncard := by
  classical
  cases' t.finite_or_infinite with ht ht
  · rw [ncard_eq_toFinset_card _ hs, ncard_eq_toFinset_card _ ht,
      ← @Finite.toFinset_subset_toFinset _ _ _ hs ht, ← Finset.exists_eq_insert_iff]
    convert Iff.rfl using 2; simp only [Finite.mem_toFinset]
    ext x
    simp [Finset.ext_iff, Set.ext_iff]
  simp only [ht.ncard, exists_prop, add_eq_zero, and_false, iff_false, not_exists, not_and,
    reduceCtorEq]
  rintro x - rfl
  exact ht (hs.insert x)


theorem ncard_le_one (hs : s.Finite := by toFinite_tac) :
    s.ncard ≤ 1 ↔ ∀ a ∈ s, ∀ b ∈ s, a = b := by
  /-
    α : Type u_1
    s : Set α
    hs : autoParam s.Finite _auto✝
    ⊢ Iff (LE.le s.ncard 1) (∀ (a : α), Membership.mem s a → ∀ (b : α), Membership …
  -/
  simp_rw [ncard_eq_toFinset_card _ hs, Finset.card_le_one, Finite.mem_toFinset]
  /-
    🎉 no goals
  -/


theorem ncard_le_one_iff (hs : s.Finite := by toFinite_tac) :
    s.ncard ≤ 1 ↔ ∀ {a b}, a ∈ s → b ∈ s → a = b := by
  /-
    α : Type u_1
    s : Set α
    hs : autoParam s.Finite _auto✝
    ⊢ Iff (LE.le s.ncard 1) (∀ {a b : α}, Membership.mem s a → Membership.mem s b  …
  -/
  rw [ncard_le_one hs]
  /-
    α : Type u_1
    s : Set α
    hs : autoParam s.Finite _auto✝
    ⊢ Iff (∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem s b → Eq a b) …
  -/
  tauto
  /-
    🎉 no goals
  -/


theorem ncard_le_one_iff_eq (hs : s.Finite := by toFinite_tac) :
    s.ncard ≤ 1 ↔ s = ∅ ∨ ∃ a, s = {a} := by
  /-
    α : Type u_1
    s : Set α
    hs : autoParam s.Finite _auto✝
    ⊢ Iff (LE.le s.ncard 1) (Or (Eq s EmptyCollection.emptyCollection) (Exists fun …
  -/
  obtain rfl | ⟨x, hx⟩ := s.eq_empty_or_nonempty
    /-
      case inl
      α : Type u_1
      hs : autoParam EmptyCollection.emptyCollection.Finite _auto✝
      ⊢ Iff (LE.le EmptyCollection.emptyCollection.ncard 1) (Or (Eq EmptyCollection. …
    -/
  · exact iff_of_true (by simp) (Or.inl rfl)
    /-
      🎉 no goals
    -/
  /-
    case inr.intro
    α : Type u_1
    s : Set α
    hs : autoParam s.Finite _auto✝
    x : α
    hx : Membership.mem s x
    ⊢ Iff (LE.le s.ncard 1) (Or (Eq s EmptyCollection.emptyCollection) (Exists fun …
  -/
  rw [ncard_le_one_iff hs]
  /-
    case inr.intro
    α : Type u_1
    s : Set α
    hs : autoParam s.Finite _auto✝
    x : α
    hx : Membership.mem s x
    ⊢ Iff (∀ {a b : α}, Membership.mem s a → Membership.mem s b → Eq a b) (Or (Eq  …
  -/
  refine ⟨fun h ↦ Or.inr ⟨x, (singleton_subset_iff.mpr hx).antisymm' fun y hy ↦ h hy hx⟩, ?_⟩
  /-
    case inr.intro
    α : Type u_1
    s : Set α
    hs : autoParam s.Finite _auto✝
    x : α
    hx : Membership.mem s x
    ⊢ Or (Eq s EmptyCollection.emptyCollection) (Exists fun a => Eq s (Singleton.s …
  -/
  rintro (rfl | ⟨a, rfl⟩)
    /-
      case inr.intro.inl
      α : Type u_1
      x : α
      hs : autoParam EmptyCollection.emptyCollection.Finite _auto✝
      hx : Membership.mem EmptyCollection.emptyCollection x
      ⊢ ∀ {a b : α}, Membership.mem EmptyCollection.emptyCollection a → Membership.m …
    -/
  · exact (not_mem_empty _ hx).elim
    /-
      🎉 no goals
    -/
  /-
    case inr.intro.inr.intro
    α : Type u_1
    x a : α
    hs : autoParam (Singleton.singleton a).Finite _auto✝
    hx : Membership.mem (Singleton.singleton a) x
    ⊢ ∀ {a_1 b : α}, Membership.mem (Singleton.singleton a) a_1 → Membership.mem ( …
  -/
  simp_rw [mem_singleton_iff] at hx ⊢; subst hx
  /-
    case inr.intro.inr.intro
    α : Type u_1
    x : α
    hs : autoParam (Singleton.singleton x).Finite _auto✝
    ⊢ ∀ {a b : α}, Eq a x → Eq b x → Eq a b
  -/
  simp only [forall_eq_apply_imp_iff, imp_self, implies_true]
  /-
    🎉 no goals
  -/


theorem ncard_le_one_iff_subset_singleton [Nonempty α]
    (hs : s.Finite := by toFinite_tac) :
    s.ncard ≤ 1 ↔ ∃ x : α, s ⊆ {x} := by
  simp_rw [ncard_eq_toFinset_card _ hs, Finset.card_le_one_iff_subset_singleton,
    Finite.toFinset_subset, Finset.coe_singleton]


/-- A `Set` of a subsingleton type has cardinality at most one. -/
theorem ncard_le_one_of_subsingleton [Subsingleton α] (s : Set α) : s.ncard ≤ 1 := by
  /-
    α : Type u_1
    inst✝ : Subsingleton α
    s : Set α
    ⊢ LE.le s.ncard 1
  -/
  rw [ncard_eq_toFinset_card]
  /-
    α : Type u_1
    inst✝ : Subsingleton α
    s : Set α
    ⊢ LE.le ⋯.toFinset.card 1
  -/
  exact Finset.card_le_one_of_subsingleton _
  /-
    🎉 no goals
  -/


theorem one_lt_ncard (hs : s.Finite := by toFinite_tac) :
    1 < s.ncard ↔ ∃ a ∈ s, ∃ b ∈ s, a ≠ b := by
  /-
    α : Type u_1
    s : Set α
    hs : autoParam s.Finite _auto✝
    ⊢ Iff (LT.lt 1 s.ncard) (Exists fun a => And (Membership.mem s a) (Exists fun  …
  -/
  simp_rw [ncard_eq_toFinset_card _ hs, Finset.one_lt_card, Finite.mem_toFinset]
  /-
    🎉 no goals
  -/


theorem one_lt_ncard_iff (hs : s.Finite := by toFinite_tac) :
    1 < s.ncard ↔ ∃ a b, a ∈ s ∧ b ∈ s ∧ a ≠ b := by
  /-
    α : Type u_1
    s : Set α
    hs : autoParam s.Finite _auto✝
    ⊢ Iff (LT.lt 1 s.ncard) (Exists fun a => Exists fun b => And (Membership.mem s …
  -/
  rw [one_lt_ncard hs]
  /-
    α : Type u_1
    s : Set α
    hs : autoParam s.Finite _auto✝
    ⊢ Iff (Exists fun a => And (Membership.mem s a) (Exists fun b => And (Membersh …
  -/
  simp only [exists_prop, exists_and_left]
  /-
    🎉 no goals
  -/


lemma one_lt_ncard_of_nonempty_of_even (hs : Set.Finite s) (hn : Set.Nonempty s := by toFinite_tac)
    (he : Even (s.ncard)) : 1 < s.ncard := by
  /-
    α : Type u_1
    s : Set α
    hs : s.Finite
    hn : autoParam s.Nonempty _auto✝
    he : Even s.ncard
    ⊢ LT.lt 1 s.ncard
  -/
  rw [← Set.ncard_pos hs] at hn
  /-
    α : Type u_1
    s : Set α
    hs : s.Finite
    hn : autoParam (LT.lt 0 s.ncard) _auto✝
    he : Even s.ncard
    ⊢ LT.lt 1 s.ncard
  -/
  have : s.ncard ≠ 1 := fun h ↦ by simp [h] at he
  /-
    α : Type u_1
    s : Set α
    hs : s.Finite
    hn : autoParam (LT.lt 0 s.ncard) _auto✝
    he : Even s.ncard
    this : Ne s.ncard 1
    ⊢ LT.lt 1 s.ncard
  -/
  omega
  /-
    🎉 no goals
  -/


theorem two_lt_ncard_iff (hs : s.Finite := by toFinite_tac) :
    2 < s.ncard ↔ ∃ a b c, a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c := by
  /-
    α : Type u_1
    s : Set α
    hs : autoParam s.Finite _auto✝
    ⊢ Iff (LT.lt 2 s.ncard) (Exists fun a => Exists fun b => Exists fun c => And ( …
  -/
  simp_rw [ncard_eq_toFinset_card _ hs, Finset.two_lt_card_iff, Finite.mem_toFinset]
  /-
    🎉 no goals
  -/


theorem two_lt_ncard (hs : s.Finite := by toFinite_tac) :
    2 < s.ncard ↔ ∃ a ∈ s, ∃ b ∈ s, ∃ c ∈ s, a ≠ b ∧ a ≠ c ∧ b ≠ c := by
  /-
    α : Type u_1
    s : Set α
    hs : autoParam s.Finite _auto✝
    ⊢ Iff (LT.lt 2 s.ncard) (Exists fun a => And (Membership.mem s a) (Exists fun  …
  -/
  simp only [two_lt_ncard_iff hs, exists_and_left, exists_prop]
  /-
    🎉 no goals
  -/


theorem exists_ne_of_one_lt_ncard (hs : 1 < s.ncard) (a : α) : ∃ b, b ∈ s ∧ b ≠ a := by
  /-
    α : Type u_1
    s : Set α
    hs : LT.lt 1 s.ncard
    a : α
    ⊢ Exists fun b => And (Membership.mem s b) (Ne b a)
  -/
  have hsf := finite_of_ncard_ne_zero (zero_lt_one.trans hs).ne.symm
  /-
    α : Type u_1
    s : Set α
    hs : LT.lt 1 s.ncard
    a : α
    hsf : s.Finite
    ⊢ Exists fun b => And (Membership.mem s b) (Ne b a)
  -/
  rw [ncard_eq_toFinset_card _ hsf] at hs
  /-
    α : Type u_1
    s : Set α
    a : α
    hsf : s.Finite
    hs : LT.lt 1 hsf.toFinset.card
    ⊢ Exists fun b => And (Membership.mem s b) (Ne b a)
  -/
  simpa only [Finite.mem_toFinset] using Finset.exists_ne_of_one_lt_card hs a
  /-
    🎉 no goals
  -/


theorem eq_insert_of_ncard_eq_succ {n : ℕ} (h : s.ncard = n + 1) :
    ∃ a t, a ∉ t ∧ insert a t = s ∧ t.ncard = n := by
  classical
  have hsf := finite_of_ncard_pos (n.zero_lt_succ.trans_eq h.symm)
  rw [ncard_eq_toFinset_card _ hsf, Finset.card_eq_succ] at h
  obtain ⟨a, t, hat, hts, rfl⟩ := h
  simp only [Finset.ext_iff, Finset.mem_insert, Finite.mem_toFinset] at hts
  refine ⟨a, t, hat, ?_, ?_⟩
  · simp [Set.ext_iff, hts]
  · simp


theorem ncard_eq_succ {n : ℕ} (hs : s.Finite := by toFinite_tac) :
    s.ncard = n + 1 ↔ ∃ a t, a ∉ t ∧ insert a t = s ∧ t.ncard = n := by
  /-
    α : Type u_1
    s : Set α
    n : Nat
    hs : autoParam s.Finite _auto✝
    ⊢ Iff (Eq s.ncard (HAdd.hAdd n 1)) (Exists fun a => Exists fun t => And (Not ( …
  -/
  refine ⟨eq_insert_of_ncard_eq_succ, ?_⟩
  /-
    α : Type u_1
    s : Set α
    n : Nat
    hs : autoParam s.Finite _auto✝
    ⊢ (Exists fun a => Exists fun t => And (Not (Membership.mem t a)) (And (Eq (In …
  -/
  rintro ⟨a, t, hat, h, rfl⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    s : Set α
    hs : autoParam s.Finite _auto✝
    a : α
    t : Set α
    hat : Not (Membership.mem t a)
    h : Eq (Insert.insert a t) s
    ⊢ Eq s.ncard (HAdd.hAdd t.ncard 1)
  -/
  rw [← h, ncard_insert_of_not_mem hat (hs.subset ((subset_insert a t).trans_eq h))]
  /-
    🎉 no goals
  -/


theorem ncard_eq_two : s.ncard = 2 ↔ ∃ x y, x ≠ y ∧ s = {x, y} := by
  /-
    α : Type u_1
    s : Set α
    ⊢ Iff (Eq s.ncard 2) (Exists fun x => Exists fun y => And (Ne x y) (Eq s (Inse …
  -/
  rw [← encard_eq_two, ncard_def, ← Nat.cast_inj (R := ℕ∞), Nat.cast_ofNat]
  /-
    α : Type u_1
    s : Set α
    ⊢ Iff (Eq (↑s.encard.toNat) 2) (Eq s.encard 2)
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ ?_⟩
    /-
      case refine_1
      α : Type u_1
      s : Set α
      h : Eq (↑s.encard.toNat) 2
      ⊢ Eq s.encard 2
    -/
  · rwa [ENat.coe_toNat] at h; rintro h'; simp [h'] at h
                                          /-
                                            🎉 no goals
                                          -/
  /-
    case refine_2
    α : Type u_1
    s : Set α
    h : Eq s.encard 2
    ⊢ Eq (↑s.encard.toNat) 2
  -/
  simp [h]
  /-
    🎉 no goals
  -/


theorem ncard_eq_three : s.ncard = 3 ↔ ∃ x y z, x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ s = {x, y, z} := by
  /-
    α : Type u_1
    s : Set α
    ⊢ Iff (Eq s.ncard 3) (Exists fun x => Exists fun y => Exists fun z => And (Ne  …
  -/
  rw [← encard_eq_three, ncard_def, ← Nat.cast_inj (R := ℕ∞), Nat.cast_ofNat]
  /-
    α : Type u_1
    s : Set α
    ⊢ Iff (Eq (↑s.encard.toNat) 3) (Eq s.encard 3)
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ ?_⟩
    /-
      case refine_1
      α : Type u_1
      s : Set α
      h : Eq (↑s.encard.toNat) 3
      ⊢ Eq s.encard 3
    -/
  · rwa [ENat.coe_toNat] at h; rintro h'; simp [h'] at h
                                          /-
                                            🎉 no goals
                                          -/
  /-
    case refine_2
    α : Type u_1
    s : Set α
    h : Eq s.encard 3
    ⊢ Eq (↑s.encard.toNat) 3
  -/
  simp [h]
  /-
    🎉 no goals
  -/


