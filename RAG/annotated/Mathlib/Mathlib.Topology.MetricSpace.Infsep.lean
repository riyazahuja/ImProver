/-- The "extended infimum separation" of a set with an edist function. -/
noncomputable def einfsep [EDist α] (s : Set α) : ℝ≥0∞ :=
  ⨅ (x ∈ s) (y ∈ s) (_ : x ≠ y), edist x y


theorem le_einfsep_iff {d} :
    d ≤ s.einfsep ↔ ∀ x ∈ s, ∀ y ∈ s, x ≠ y → d ≤ edist x y := by
  /-
    α : Type u_1
    inst✝ : EDist α
    s : Set α
    d : ENNReal
    ⊢ Iff (LE.le d s.einfsep) (∀ (x : α), Membership.mem s x → ∀ (y : α), Membersh …
  -/
  simp_rw [einfsep, le_iInf_iff]
  /-
    🎉 no goals
  -/


theorem einfsep_zero : s.einfsep = 0 ↔ ∀ C > 0, ∃ x ∈ s, ∃ y ∈ s, x ≠ y ∧ edist x y < C := by
  /-
    α : Type u_1
    inst✝ : EDist α
    s : Set α
    ⊢ Iff (Eq s.einfsep 0) (∀ (C : ENNReal), GT.gt C 0 → Exists fun x => And (Memb …
  -/
  simp_rw [einfsep, ← _root_.bot_eq_zero, iInf_eq_bot, iInf_lt_iff, exists_prop]
  /-
    🎉 no goals
  -/


theorem einfsep_pos : 0 < s.einfsep ↔ ∃ C > 0, ∀ x ∈ s, ∀ y ∈ s, x ≠ y → C ≤ edist x y := by
  /-
    α : Type u_1
    inst✝ : EDist α
    s : Set α
    ⊢ Iff (LT.lt 0 s.einfsep) (Exists fun C => And (GT.gt C 0) (∀ (x : α), Members …
  -/
  rw [pos_iff_ne_zero, Ne, einfsep_zero]
  /-
    α : Type u_1
    inst✝ : EDist α
    s : Set α
    ⊢ Iff (Not (∀ (C : ENNReal), GT.gt C 0 → Exists fun x => And (Membership.mem s …
  -/
  simp only [not_forall, not_exists, not_lt, exists_prop, not_and]
  /-
    🎉 no goals
  -/


theorem einfsep_top :
    s.einfsep = ∞ ↔ ∀ x ∈ s, ∀ y ∈ s, x ≠ y → edist x y = ∞ := by
  /-
    α : Type u_1
    inst✝ : EDist α
    s : Set α
    ⊢ Iff (Eq s.einfsep Top.top) (∀ (x : α), Membership.mem s x → ∀ (y : α), Membe …
  -/
  simp_rw [einfsep, iInf_eq_top]
  /-
    🎉 no goals
  -/


theorem einfsep_lt_top :
    s.einfsep < ∞ ↔ ∃ x ∈ s, ∃ y ∈ s, x ≠ y ∧ edist x y < ∞ := by
  /-
    α : Type u_1
    inst✝ : EDist α
    s : Set α
    ⊢ Iff (LT.lt s.einfsep Top.top) (Exists fun x => And (Membership.mem s x) (Exi …
  -/
  simp_rw [einfsep, iInf_lt_iff, exists_prop]
  /-
    🎉 no goals
  -/


theorem einfsep_ne_top :
    s.einfsep ≠ ∞ ↔ ∃ x ∈ s, ∃ y ∈ s, x ≠ y ∧ edist x y ≠ ∞ := by
  /-
    α : Type u_1
    inst✝ : EDist α
    s : Set α
    ⊢ Iff (Ne s.einfsep Top.top) (Exists fun x => And (Membership.mem s x) (Exists …
  -/
  simp_rw [← lt_top_iff_ne_top, einfsep_lt_top]
  /-
    🎉 no goals
  -/


theorem einfsep_lt_iff {d} :
    s.einfsep < d ↔ ∃ x ∈ s, ∃ y ∈ s, x ≠ y ∧ edist x y < d := by
  /-
    α : Type u_1
    inst✝ : EDist α
    s : Set α
    d : ENNReal
    ⊢ Iff (LT.lt s.einfsep d) (Exists fun x => And (Membership.mem s x) (Exists fu …
  -/
  simp_rw [einfsep, iInf_lt_iff, exists_prop]
  /-
    🎉 no goals
  -/


theorem nontrivial_of_einfsep_lt_top (hs : s.einfsep < ∞) : s.Nontrivial := by
  /-
    α : Type u_1
    inst✝ : EDist α
    s : Set α
    hs : LT.lt s.einfsep Top.top
    ⊢ s.Nontrivial
  -/
  rcases einfsep_lt_top.1 hs with ⟨_, hx, _, hy, hxy, _⟩
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    inst✝ : EDist α
    s : Set α
    hs : LT.lt s.einfsep Top.top
    w✝¹ : α
    hx : Membership.mem s w✝¹
    w✝ : α
    hy : Membership.mem s w✝
    hxy : Ne w✝¹ w✝
    right✝ : LT.lt (EDist.edist w✝¹ w✝) Top.top
    ⊢ s.Nontrivial
  -/
  exact ⟨_, hx, _, hy, hxy⟩
  /-
    🎉 no goals
  -/


theorem nontrivial_of_einfsep_ne_top (hs : s.einfsep ≠ ∞) : s.Nontrivial :=
  nontrivial_of_einfsep_lt_top (lt_top_iff_ne_top.mpr hs)


theorem Subsingleton.einfsep (hs : s.Subsingleton) : s.einfsep = ∞ := by
  /-
    α : Type u_1
    inst✝ : EDist α
    s : Set α
    hs : s.Subsingleton
    ⊢ Eq s.einfsep Top.top
  -/
  rw [einfsep_top]
  /-
    α : Type u_1
    inst✝ : EDist α
    s : Set α
    hs : s.Subsingleton
    ⊢ ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem s y → Ne x y → Eq  …
  -/
  exact fun _ hx _ hy hxy => (hxy <| hs hx hy).elim
  /-
    🎉 no goals
  -/


theorem le_einfsep_image_iff {d} {f : β → α} {s : Set β} : d ≤ einfsep (f '' s)
    ↔ ∀ x ∈ s, ∀ y ∈ s, f x ≠ f y → d ≤ edist (f x) (f y) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : EDist α
    d : ENNReal
    f : β → α
    s : Set β
    ⊢ Iff (LE.le d (Set.image f s).einfsep) (∀ (x : β), Membership.mem s x → ∀ (y  …
  -/
  simp_rw [le_einfsep_iff, forall_mem_image]
  /-
    🎉 no goals
  -/


theorem le_edist_of_le_einfsep {d x} (hx : x ∈ s) {y} (hy : y ∈ s) (hxy : x ≠ y)
    (hd : d ≤ s.einfsep) : d ≤ edist x y :=
  le_einfsep_iff.1 hd x hx y hy hxy


theorem einfsep_le_edist_of_mem {x} (hx : x ∈ s) {y} (hy : y ∈ s) (hxy : x ≠ y) :
    s.einfsep ≤ edist x y :=
  le_edist_of_le_einfsep hx hy hxy le_rfl


theorem einfsep_le_of_mem_of_edist_le {d x} (hx : x ∈ s) {y} (hy : y ∈ s) (hxy : x ≠ y)
    (hxy' : edist x y ≤ d) : s.einfsep ≤ d :=
  le_trans (einfsep_le_edist_of_mem hx hy hxy) hxy'


theorem le_einfsep {d} (h : ∀ x ∈ s, ∀ y ∈ s, x ≠ y → d ≤ edist x y) : d ≤ s.einfsep :=
  le_einfsep_iff.2 h


@[simp]
theorem einfsep_empty : (∅ : Set α).einfsep = ∞ :=
  subsingleton_empty.einfsep


@[simp]
theorem einfsep_singleton : ({x} : Set α).einfsep = ∞ :=
  subsingleton_singleton.einfsep


theorem einfsep_iUnion_mem_option {ι : Type*} (o : Option ι) (s : ι → Set α) :
                                                          /-
                                                            α : Type u_1
                                                            inst✝ : EDist α
                                                            ι : Type u_3
                                                            o : Option ι
                                                            s : ι → Set α
                                                            ⊢ Eq (Set.iUnion fun i => Set.iUnion fun h => s i).einfsep (iInf fun i => iInf …
                                                          -/
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
    (⋃ i ∈ o, s i).einfsep = ⨅ i ∈ o, (s i).einfsep := by cases o <;> simp
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


theorem einfsep_anti (hst : s ⊆ t) : t.einfsep ≤ s.einfsep :=
  le_einfsep fun _x hx _y hy => einfsep_le_edist_of_mem (hst hx) (hst hy)


theorem einfsep_insert_le : (insert x s).einfsep ≤ ⨅ (y ∈ s) (_ : x ≠ y), edist x y := by
  /-
    α : Type u_1
    inst✝ : EDist α
    x : α
    s : Set α
    ⊢ LE.le (Insert.insert x s).einfsep (iInf fun y => iInf fun h => iInf fun x_1  …
  -/
  simp_rw [le_iInf_iff]
  /-
    α : Type u_1
    inst✝ : EDist α
    x : α
    s : Set α
    ⊢ ∀ (i : α), Membership.mem s i → Ne x i → LE.le (Insert.insert x s).einfsep ( …
  -/
  exact fun _ hy hxy => einfsep_le_edist_of_mem (mem_insert _ _) (mem_insert_of_mem _ hy) hxy
  /-
    🎉 no goals
  -/


theorem le_einfsep_pair : edist x y ⊓ edist y x ≤ ({x, y} : Set α).einfsep := by
  /-
    α : Type u_1
    inst✝ : EDist α
    x y : α
    ⊢ LE.le (Min.min (EDist.edist x y) (EDist.edist y x)) (Insert.insert x (Single …
  -/
  simp_rw [le_einfsep_iff, inf_le_iff, mem_insert_iff, mem_singleton_iff]
  /-
    α : Type u_1
    inst✝ : EDist α
    x y : α
    ⊢ ∀ (x_1 : α), Or (Eq x_1 x) (Eq x_1 y) → ∀ (y_1 : α), Or (Eq y_1 x) (Eq y_1 y …
  -/
                                              /-
                                                🎉 no goals
                                              -/
                                              /-
                                                🎉 no goals
                                              -/
  rintro a (rfl | rfl) b (rfl | rfl) hab <;> (try simp only [le_refl, true_or, or_true]) <;>
    /-
      case inl.inl
      α : Type u_1
      inst✝ : EDist α
      y b : α
      hab : Ne b b
      ⊢ Or (LE.le (EDist.edist b y) (EDist.edist b b)) (LE.le (EDist.edist y b) (EDi …
    -/
    /-
      🎉 no goals
    -/
    contradiction
    /-
      🎉 no goals
    -/


theorem einfsep_pair_le_left (hxy : x ≠ y) : ({x, y} : Set α).einfsep ≤ edist x y :=
  einfsep_le_edist_of_mem (mem_insert _ _) (mem_insert_of_mem _ (mem_singleton _)) hxy


theorem einfsep_pair_le_right (hxy : x ≠ y) : ({x, y} : Set α).einfsep ≤ edist y x := by
  /-
    α : Type u_1
    inst✝ : EDist α
    x y : α
    hxy : Ne x y
    ⊢ LE.le (Insert.insert x (Singleton.singleton y)).einfsep (EDist.edist y x)
  -/
  rw [pair_comm]; exact einfsep_pair_le_left hxy.symm
                  /-
                    🎉 no goals
                  -/


theorem einfsep_pair_eq_inf (hxy : x ≠ y) : ({x, y} : Set α).einfsep = edist x y ⊓ edist y x :=
  le_antisymm (le_inf (einfsep_pair_le_left hxy) (einfsep_pair_le_right hxy)) le_einfsep_pair


theorem einfsep_eq_iInf : s.einfsep = ⨅ d : s.offDiag, (uncurry edist) (d : α × α) := by
  /-
    α : Type u_1
    inst✝ : EDist α
    s : Set α
    ⊢ Eq s.einfsep (iInf fun d => Function.uncurry EDist.edist ↑d)
  -/
  refine eq_of_forall_le_iff fun _ => ?_
  simp_rw [le_einfsep_iff, le_iInf_iff, imp_forall_iff, SetCoe.forall, mem_offDiag,
    Prod.forall, uncurry_apply_pair, and_imp]


theorem einfsep_of_fintype [DecidableEq α] [Fintype s] :
    s.einfsep = s.offDiag.toFinset.inf (uncurry edist) := by
  /-
    α : Type u_1
    inst✝² : EDist α
    s : Set α
    inst✝¹ : DecidableEq α
    inst✝ : Fintype ↑s
    ⊢ Eq s.einfsep (s.offDiag.toFinset.inf (Function.uncurry EDist.edist))
  -/
  refine eq_of_forall_le_iff fun _ => ?_
  simp_rw [le_einfsep_iff, imp_forall_iff, Finset.le_inf_iff, mem_toFinset, mem_offDiag,
    Prod.forall, uncurry_apply_pair, and_imp]


theorem Finite.einfsep (hs : s.Finite) : s.einfsep = hs.offDiag.toFinset.inf (uncurry edist) := by
  /-
    α : Type u_1
    inst✝ : EDist α
    s : Set α
    hs : s.Finite
    ⊢ Eq s.einfsep (⋯.toFinset.inf (Function.uncurry EDist.edist))
  -/
  refine eq_of_forall_le_iff fun _ => ?_
  simp_rw [le_einfsep_iff, imp_forall_iff, Finset.le_inf_iff, Finite.mem_toFinset, mem_offDiag,
    Prod.forall, uncurry_apply_pair, and_imp]


theorem Finset.coe_einfsep [DecidableEq α] {s : Finset α} :
    (s : Set α).einfsep = s.offDiag.inf (uncurry edist) := by
  /-
    α : Type u_1
    inst✝¹ : EDist α
    inst✝ : DecidableEq α
    s : Finset α
    ⊢ Eq (↑s).einfsep (s.offDiag.inf (Function.uncurry EDist.edist))
  -/
  simp_rw [einfsep_of_fintype, ← Finset.coe_offDiag, Finset.toFinset_coe]
  /-
    🎉 no goals
  -/


theorem Nontrivial.einfsep_exists_of_finite [Finite s] (hs : s.Nontrivial) :
    ∃ x ∈ s, ∃ y ∈ s, x ≠ y ∧ s.einfsep = edist x y := by
  classical
    cases nonempty_fintype s
    simp_rw [einfsep_of_fintype]
    rcases Finset.exists_mem_eq_inf s.offDiag.toFinset (by simpa) (uncurry edist) with ⟨w, hxy, hed⟩
    simp_rw [mem_toFinset] at hxy
    exact ⟨w.fst, hxy.1, w.snd, hxy.2.1, hxy.2.2, hed⟩


theorem Finite.einfsep_exists_of_nontrivial (hsf : s.Finite) (hs : s.Nontrivial) :
    ∃ x ∈ s, ∃ y ∈ s, x ≠ y ∧ s.einfsep = edist x y :=
  letI := hsf.fintype
  hs.einfsep_exists_of_finite


theorem einfsep_pair (hxy : x ≠ y) : ({x, y} : Set α).einfsep = edist x y := by
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    x y : α
    hxy : Ne x y
    ⊢ Eq (Insert.insert x (Singleton.singleton y)).einfsep (EDist.edist x y)
  -/
  nth_rw 1 [← min_self (edist x y)]
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    x y : α
    hxy : Ne x y
    ⊢ Eq (Insert.insert x (Singleton.singleton y)).einfsep (Min.min (EDist.edist x …
  -/
  convert einfsep_pair_eq_inf hxy using 2
  /-
    case h.e'_3.h.e'_4
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    x y : α
    hxy : Ne x y
    ⊢ Eq (EDist.edist x y) (EDist.edist y x)
  -/
  rw [edist_comm]
  /-
    🎉 no goals
  -/


theorem einfsep_insert : einfsep (insert x s) =
    (⨅ (y ∈ s) (_ : x ≠ y), edist x y) ⊓ s.einfsep := by
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    x : α
    s : Set α
    ⊢ Eq (Insert.insert x s).einfsep (Min.min (iInf fun y => iInf fun h => iInf fu …
  -/
  refine le_antisymm (le_min einfsep_insert_le (einfsep_anti (subset_insert _ _))) ?_
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    x : α
    s : Set α
    ⊢ LE.le (Min.min (iInf fun y => iInf fun h => iInf fun x_1 => EDist.edist x y) …
  -/
  simp_rw [le_einfsep_iff, inf_le_iff, mem_insert_iff]
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    x : α
    s : Set α
    ⊢ ∀ (x_1 : α), Or (Eq x_1 x) (Membership.mem s x_1) → ∀ (y : α), Or (Eq y x) ( …
  -/
  rintro y (rfl | hy) z (rfl | hz) hyz
    /-
      case inl.inl
      α : Type u_1
      inst✝ : PseudoEMetricSpace α
      s : Set α
      z : α
      hyz : Ne z z
      ⊢ Or (LE.le (iInf fun y => iInf fun h => iInf fun x => EDist.edist z y) (EDist …
    -/
  · exact False.elim (hyz rfl)
    /-
      🎉 no goals
    -/
    /-
      case inl.inr
      α : Type u_1
      inst✝ : PseudoEMetricSpace α
      s : Set α
      y z : α
      hz : Membership.mem s z
      hyz : Ne y z
      ⊢ Or (LE.le (iInf fun y_1 => iInf fun h => iInf fun x => EDist.edist y y_1) (E …
    -/
  · exact Or.inl (iInf_le_of_le _ (iInf₂_le hz hyz))
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      α : Type u_1
      inst✝ : PseudoEMetricSpace α
      s : Set α
      y : α
      hy : Membership.mem s y
      z : α
      hyz : Ne y z
      ⊢ Or (LE.le (iInf fun y => iInf fun h => iInf fun x => EDist.edist z y) (EDist …
    -/
  · rw [edist_comm]
    /-
      case inr.inl
      α : Type u_1
      inst✝ : PseudoEMetricSpace α
      s : Set α
      y : α
      hy : Membership.mem s y
      z : α
      hyz : Ne y z
      ⊢ Or (LE.le (iInf fun y => iInf fun h => iInf fun x => EDist.edist z y) (EDist …
    -/
    exact Or.inl (iInf_le_of_le _ (iInf₂_le hy hyz.symm))
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      α : Type u_1
      inst✝ : PseudoEMetricSpace α
      x : α
      s : Set α
      y : α
      hy : Membership.mem s y
      z : α
      hz : Membership.mem s z
      hyz : Ne y z
      ⊢ Or (LE.le (iInf fun y => iInf fun h => iInf fun x_1 => EDist.edist x y) (EDi …
    -/
  · exact Or.inr (einfsep_le_edist_of_mem hy hz hyz)
    /-
      🎉 no goals
    -/


theorem einfsep_triple (hxy : x ≠ y) (hyz : y ≠ z) (hxz : x ≠ z) :
    einfsep ({x, y, z} : Set α) = edist x y ⊓ edist x z ⊓ edist y z := by
  simp_rw [einfsep_insert, iInf_insert, iInf_singleton, einfsep_singleton, inf_top_eq,
    ciInf_pos hxy, ciInf_pos hyz, ciInf_pos hxz]


theorem le_einfsep_pi_of_le {π : β → Type*} [Fintype β] [∀ b, PseudoEMetricSpace (π b)]
    {s : ∀ b : β, Set (π b)} {c : ℝ≥0∞} (h : ∀ b, c ≤ einfsep (s b)) :
    c ≤ einfsep (Set.pi univ s) := by
  /-
    β : Type u_2
    π : β → Type u_3
    inst✝¹ : Fintype β
    inst✝ : (b : β) → PseudoEMetricSpace (π b)
    s : (b : β) → Set (π b)
    c : ENNReal
    h : ∀ (b : β), LE.le c (s b).einfsep
    ⊢ LE.le c (Set.univ.pi s).einfsep
  -/
  refine le_einfsep fun x hx y hy hxy => ?_
  /-
    β : Type u_2
    π : β → Type u_3
    inst✝¹ : Fintype β
    inst✝ : (b : β) → PseudoEMetricSpace (π b)
    s : (b : β) → Set (π b)
    c : ENNReal
    h : ∀ (b : β), LE.le c (s b).einfsep
    x : (i : β) → π i
    hx : Membership.mem (Set.univ.pi s) x
    y : (i : β) → π i
    hy : Membership.mem (Set.univ.pi s) y
    hxy : Ne x y
    ⊢ LE.le c (EDist.edist x y)
  -/
  rw [mem_univ_pi] at hx hy
  /-
    β : Type u_2
    π : β → Type u_3
    inst✝¹ : Fintype β
    inst✝ : (b : β) → PseudoEMetricSpace (π b)
    s : (b : β) → Set (π b)
    c : ENNReal
    h : ∀ (b : β), LE.le c (s b).einfsep
    x : (i : β) → π i
    hx : ∀ (i : β), Membership.mem (s i) (x i)
    y : (i : β) → π i
    hy : ∀ (i : β), Membership.mem (s i) (y i)
    hxy : Ne x y
    ⊢ LE.le c (EDist.edist x y)
  -/
  rcases Function.ne_iff.mp hxy with ⟨i, hi⟩
  /-
    case intro
    β : Type u_2
    π : β → Type u_3
    inst✝¹ : Fintype β
    inst✝ : (b : β) → PseudoEMetricSpace (π b)
    s : (b : β) → Set (π b)
    c : ENNReal
    h : ∀ (b : β), LE.le c (s b).einfsep
    x : (i : β) → π i
    hx : ∀ (i : β), Membership.mem (s i) (x i)
    y : (i : β) → π i
    hy : ∀ (i : β), Membership.mem (s i) (y i)
    hxy : Ne x y
    i : β
    hi : Ne (x i) (y i)
    ⊢ LE.le c (EDist.edist x y)
  -/
  exact le_trans (le_einfsep_iff.1 (h i) _ (hx _) _ (hy _) hi) (edist_le_pi_edist _ _ i)
  /-
    🎉 no goals
  -/


theorem subsingleton_of_einfsep_eq_top (hs : s.einfsep = ∞) : s.Subsingleton := by
  /-
    α : Type u_1
    inst✝ : PseudoMetricSpace α
    s : Set α
    hs : Eq s.einfsep Top.top
    ⊢ s.Subsingleton
  -/
  rw [einfsep_top] at hs
  /-
    α : Type u_1
    inst✝ : PseudoMetricSpace α
    s : Set α
    hs : ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem s y → Ne x y →  …
    ⊢ s.Subsingleton
  -/
  exact fun _ hx _ hy => of_not_not fun hxy => edist_ne_top _ _ (hs _ hx _ hy hxy)
  /-
    🎉 no goals
  -/


theorem einfsep_eq_top_iff : s.einfsep = ∞ ↔ s.Subsingleton :=
  ⟨subsingleton_of_einfsep_eq_top, Subsingleton.einfsep⟩


theorem Nontrivial.einfsep_ne_top (hs : s.Nontrivial) : s.einfsep ≠ ∞ := by
  /-
    α : Type u_1
    inst✝ : PseudoMetricSpace α
    s : Set α
    hs : s.Nontrivial
    ⊢ Ne s.einfsep Top.top
  -/
  contrapose! hs
  /-
    α : Type u_1
    inst✝ : PseudoMetricSpace α
    s : Set α
    hs : Eq s.einfsep Top.top
    ⊢ Not s.Nontrivial
  -/
  rw [not_nontrivial_iff]
  /-
    α : Type u_1
    inst✝ : PseudoMetricSpace α
    s : Set α
    hs : Eq s.einfsep Top.top
    ⊢ s.Subsingleton
  -/
  exact subsingleton_of_einfsep_eq_top hs
  /-
    🎉 no goals
  -/


theorem Nontrivial.einfsep_lt_top (hs : s.Nontrivial) : s.einfsep < ∞ := by
  /-
    α : Type u_1
    inst✝ : PseudoMetricSpace α
    s : Set α
    hs : s.Nontrivial
    ⊢ LT.lt s.einfsep Top.top
  -/
  rw [lt_top_iff_ne_top]
  /-
    α : Type u_1
    inst✝ : PseudoMetricSpace α
    s : Set α
    hs : s.Nontrivial
    ⊢ Ne s.einfsep Top.top
  -/
  exact hs.einfsep_ne_top
  /-
    🎉 no goals
  -/


theorem einfsep_lt_top_iff : s.einfsep < ∞ ↔ s.Nontrivial :=
  ⟨nontrivial_of_einfsep_lt_top, Nontrivial.einfsep_lt_top⟩


theorem einfsep_ne_top_iff : s.einfsep ≠ ∞ ↔ s.Nontrivial :=
  ⟨nontrivial_of_einfsep_ne_top, Nontrivial.einfsep_ne_top⟩


theorem le_einfsep_of_forall_dist_le {d} (h : ∀ x ∈ s, ∀ y ∈ s, x ≠ y → d ≤ dist x y) :
    ENNReal.ofReal d ≤ s.einfsep :=
  le_einfsep fun x hx y hy hxy => (edist_dist x y).symm ▸ ENNReal.ofReal_le_ofReal (h x hx y hy hxy)


theorem einfsep_pos_of_finite [Finite s] : 0 < s.einfsep := by
  /-
    α : Type u_1
    inst✝¹ : EMetricSpace α
    s : Set α
    inst✝ : Finite ↑s
    ⊢ LT.lt 0 s.einfsep
  -/
  cases nonempty_fintype s
  /-
    case intro
    α : Type u_1
    inst✝¹ : EMetricSpace α
    s : Set α
    inst✝ : Finite ↑s
    val✝ : Fintype ↑s
    ⊢ LT.lt 0 s.einfsep
  -/
  by_cases hs : s.Nontrivial
    /-
      case pos
      α : Type u_1
      inst✝¹ : EMetricSpace α
      s : Set α
      inst✝ : Finite ↑s
      val✝ : Fintype ↑s
      hs : s.Nontrivial
      ⊢ LT.lt 0 s.einfsep
    -/
  · rcases hs.einfsep_exists_of_finite with ⟨x, _hx, y, _hy, hxy, hxy'⟩
    /-
      case pos.intro.intro.intro.intro.intro
      α : Type u_1
      inst✝¹ : EMetricSpace α
      s : Set α
      inst✝ : Finite ↑s
      val✝ : Fintype ↑s
      hs : s.Nontrivial
      x : α
      _hx : Membership.mem s x
      y : α
      _hy : Membership.mem s y
      hxy : Ne x y
      hxy' : Eq s.einfsep (EDist.edist x y)
      ⊢ LT.lt 0 s.einfsep
    -/
    exact hxy'.symm ▸ edist_pos.2 hxy
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝¹ : EMetricSpace α
      s : Set α
      inst✝ : Finite ↑s
      val✝ : Fintype ↑s
      hs : Not s.Nontrivial
      ⊢ LT.lt 0 s.einfsep
    -/
  · rw [not_nontrivial_iff] at hs
    /-
      case neg
      α : Type u_1
      inst✝¹ : EMetricSpace α
      s : Set α
      inst✝ : Finite ↑s
      val✝ : Fintype ↑s
      hs : s.Subsingleton
      ⊢ LT.lt 0 s.einfsep
    -/
    exact hs.einfsep.symm ▸ WithTop.top_pos
    /-
      🎉 no goals
    -/


theorem relatively_discrete_of_finite [Finite s] :
    ∃ C > 0, ∀ x ∈ s, ∀ y ∈ s, x ≠ y → C ≤ edist x y := by
  /-
    α : Type u_1
    inst✝¹ : EMetricSpace α
    s : Set α
    inst✝ : Finite ↑s
    ⊢ Exists fun C => And (GT.gt C 0) (∀ (x : α), Membership.mem s x → ∀ (y : α),  …
  -/
  rw [← einfsep_pos]
  /-
    α : Type u_1
    inst✝¹ : EMetricSpace α
    s : Set α
    inst✝ : Finite ↑s
    ⊢ LT.lt 0 s.einfsep
  -/
  exact einfsep_pos_of_finite
  /-
    🎉 no goals
  -/


theorem Finite.einfsep_pos (hs : s.Finite) : 0 < s.einfsep :=
  letI := hs.fintype
  einfsep_pos_of_finite


theorem Finite.relatively_discrete (hs : s.Finite) :
    ∃ C > 0, ∀ x ∈ s, ∀ y ∈ s, x ≠ y → C ≤ edist x y :=
  letI := hs.fintype
  relatively_discrete_of_finite


/-- The "infimum separation" of a set with an edist function. -/
noncomputable def infsep [EDist α] (s : Set α) : ℝ :=
  ENNReal.toReal s.einfsep


theorem infsep_zero : s.infsep = 0 ↔ s.einfsep = 0 ∨ s.einfsep = ∞ := by
  /-
    α : Type u_1
    inst✝ : EDist α
    s : Set α
    ⊢ Iff (Eq s.infsep 0) (Or (Eq s.einfsep 0) (Eq s.einfsep Top.top))
  -/
  rw [infsep, ENNReal.toReal_eq_zero_iff]
  /-
    🎉 no goals
  -/


theorem infsep_nonneg : 0 ≤ s.infsep :=
  ENNReal.toReal_nonneg


theorem infsep_pos : 0 < s.infsep ↔ 0 < s.einfsep ∧ s.einfsep < ∞ := by
  /-
    α : Type u_1
    inst✝ : EDist α
    s : Set α
    ⊢ Iff (LT.lt 0 s.infsep) (And (LT.lt 0 s.einfsep) (LT.lt s.einfsep Top.top))
  -/
  simp_rw [infsep, ENNReal.toReal_pos_iff]
  /-
    🎉 no goals
  -/


theorem Subsingleton.infsep_zero (hs : s.Subsingleton) : s.infsep = 0 :=
  Set.infsep_zero.mpr <| Or.inr hs.einfsep


theorem nontrivial_of_infsep_pos (hs : 0 < s.infsep) : s.Nontrivial := by
  /-
    α : Type u_1
    inst✝ : EDist α
    s : Set α
    hs : LT.lt 0 s.infsep
    ⊢ s.Nontrivial
  -/
  contrapose hs
  /-
    α : Type u_1
    inst✝ : EDist α
    s : Set α
    hs : Not s.Nontrivial
    ⊢ Not (LT.lt 0 s.infsep)
  -/
  rw [not_nontrivial_iff] at hs
  /-
    α : Type u_1
    inst✝ : EDist α
    s : Set α
    hs : s.Subsingleton
    ⊢ Not (LT.lt 0 s.infsep)
  -/
  exact hs.infsep_zero ▸ lt_irrefl _
  /-
    🎉 no goals
  -/


theorem infsep_empty : (∅ : Set α).infsep = 0 :=
  subsingleton_empty.infsep_zero


theorem infsep_singleton : ({x} : Set α).infsep = 0 :=
  subsingleton_singleton.infsep_zero


theorem infsep_pair_le_toReal_inf (hxy : x ≠ y) :
    ({x, y} : Set α).infsep ≤ (edist x y ⊓ edist y x).toReal := by
  /-
    α : Type u_1
    inst✝ : EDist α
    x y : α
    hxy : Ne x y
    ⊢ LE.le (Insert.insert x (Singleton.singleton y)).infsep (Min.min (EDist.edist …
  -/
  simp_rw [infsep, einfsep_pair_eq_inf hxy]
  /-
    α : Type u_1
    inst✝ : EDist α
    x y : α
    hxy : Ne x y
    ⊢ LE.le (Min.min (EDist.edist x y) (EDist.edist y x)).toReal (Min.min (EDist.e …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem infsep_pair_eq_toReal : ({x, y} : Set α).infsep = (edist x y).toReal := by
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    x y : α
    ⊢ Eq (Insert.insert x (Singleton.singleton y)).infsep (EDist.edist x y).toReal
  -/
  by_cases hxy : x = y
    /-
      case pos
      α : Type u_1
      inst✝ : PseudoEMetricSpace α
      x y : α
      hxy : Eq x y
      ⊢ Eq (Insert.insert x (Singleton.singleton y)).infsep (EDist.edist x y).toReal
    -/
  · rw [hxy]
    /-
      case pos
      α : Type u_1
      inst✝ : PseudoEMetricSpace α
      x y : α
      hxy : Eq x y
      ⊢ Eq (Insert.insert y (Singleton.singleton y)).infsep (EDist.edist y y).toReal
    -/
    simp only [infsep_singleton, pair_eq_singleton, edist_self, ENNReal.zero_toReal]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝ : PseudoEMetricSpace α
      x y : α
      hxy : Not (Eq x y)
      ⊢ Eq (Insert.insert x (Singleton.singleton y)).infsep (EDist.edist x y).toReal
    -/
  · rw [infsep, einfsep_pair hxy]
    /-
      🎉 no goals
    -/


theorem Nontrivial.le_infsep_iff {d} (hs : s.Nontrivial) :
    d ≤ s.infsep ↔ ∀ x ∈ s, ∀ y ∈ s, x ≠ y → d ≤ dist x y := by
  simp_rw [infsep, ← ENNReal.ofReal_le_iff_le_toReal hs.einfsep_ne_top, le_einfsep_iff, edist_dist,
    ENNReal.ofReal_le_ofReal_iff dist_nonneg]


theorem Nontrivial.infsep_lt_iff {d} (hs : s.Nontrivial) :
    s.infsep < d ↔ ∃ x ∈ s, ∃ y ∈ s, x ≠ y ∧ dist x y < d := by
  /-
    α : Type u_1
    inst✝ : PseudoMetricSpace α
    s : Set α
    d : Real
    hs : s.Nontrivial
    ⊢ Iff (LT.lt s.infsep d) (Exists fun x => And (Membership.mem s x) (Exists fun …
  -/
  rw [← not_iff_not]
  /-
    α : Type u_1
    inst✝ : PseudoMetricSpace α
    s : Set α
    d : Real
    hs : s.Nontrivial
    ⊢ Iff (Not (LT.lt s.infsep d)) (Not (Exists fun x => And (Membership.mem s x)  …
  -/
  push_neg
  /-
    α : Type u_1
    inst✝ : PseudoMetricSpace α
    s : Set α
    d : Real
    hs : s.Nontrivial
    ⊢ Iff (LE.le d s.infsep) (∀ (x : α), Membership.mem s x → ∀ (y : α), Membershi …
  -/
  exact hs.le_infsep_iff
  /-
    🎉 no goals
  -/


theorem Nontrivial.le_infsep {d} (hs : s.Nontrivial)
    (h : ∀ x ∈ s, ∀ y ∈ s, x ≠ y → d ≤ dist x y) : d ≤ s.infsep :=
  hs.le_infsep_iff.2 h


theorem le_edist_of_le_infsep {d x} (hx : x ∈ s) {y} (hy : y ∈ s) (hxy : x ≠ y)
    (hd : d ≤ s.infsep) : d ≤ dist x y := by
  /-
    α : Type u_1
    inst✝ : PseudoMetricSpace α
    s : Set α
    d : Real
    x : α
    hx : Membership.mem s x
    y : α
    hy : Membership.mem s y
    hxy : Ne x y
    hd : LE.le d s.infsep
    ⊢ LE.le d (Dist.dist x y)
  -/
  by_cases hs : s.Nontrivial
    /-
      case pos
      α : Type u_1
      inst✝ : PseudoMetricSpace α
      s : Set α
      d : Real
      x : α
      hx : Membership.mem s x
      y : α
      hy : Membership.mem s y
      hxy : Ne x y
      hd : LE.le d s.infsep
      hs : s.Nontrivial
      ⊢ LE.le d (Dist.dist x y)
    -/
  · exact hs.le_infsep_iff.1 hd x hx y hy hxy
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝ : PseudoMetricSpace α
      s : Set α
      d : Real
      x : α
      hx : Membership.mem s x
      y : α
      hy : Membership.mem s y
      hxy : Ne x y
      hd : LE.le d s.infsep
      hs : Not s.Nontrivial
      ⊢ LE.le d (Dist.dist x y)
    -/
  · rw [not_nontrivial_iff] at hs
    /-
      case neg
      α : Type u_1
      inst✝ : PseudoMetricSpace α
      s : Set α
      d : Real
      x : α
      hx : Membership.mem s x
      y : α
      hy : Membership.mem s y
      hxy : Ne x y
      hd : LE.le d s.infsep
      hs : s.Subsingleton
      ⊢ LE.le d (Dist.dist x y)
    -/
    rw [hs.infsep_zero] at hd
    /-
      case neg
      α : Type u_1
      inst✝ : PseudoMetricSpace α
      s : Set α
      d : Real
      x : α
      hx : Membership.mem s x
      y : α
      hy : Membership.mem s y
      hxy : Ne x y
      hd : LE.le d 0
      hs : s.Subsingleton
      ⊢ LE.le d (Dist.dist x y)
    -/
    exact le_trans hd dist_nonneg
    /-
      🎉 no goals
    -/


theorem infsep_le_dist_of_mem (hx : x ∈ s) (hy : y ∈ s) (hxy : x ≠ y) : s.infsep ≤ dist x y :=
  le_edist_of_le_infsep hx hy hxy le_rfl


theorem infsep_le_of_mem_of_edist_le {d x} (hx : x ∈ s) {y} (hy : y ∈ s) (hxy : x ≠ y)
    (hxy' : dist x y ≤ d) : s.infsep ≤ d :=
  le_trans (infsep_le_dist_of_mem hx hy hxy) hxy'


theorem infsep_pair : ({x, y} : Set α).infsep = dist x y := by
  /-
    α : Type u_1
    inst✝ : PseudoMetricSpace α
    x y : α
    ⊢ Eq (Insert.insert x (Singleton.singleton y)).infsep (Dist.dist x y)
  -/
  rw [infsep_pair_eq_toReal, edist_dist]
  /-
    α : Type u_1
    inst✝ : PseudoMetricSpace α
    x y : α
    ⊢ Eq (ENNReal.ofReal (Dist.dist x y)).toReal (Dist.dist x y)
  -/
  exact ENNReal.toReal_ofReal dist_nonneg
  /-
    🎉 no goals
  -/


theorem infsep_triple (hxy : x ≠ y) (hyz : y ≠ z) (hxz : x ≠ z) :
    ({x, y, z} : Set α).infsep = dist x y ⊓ dist x z ⊓ dist y z := by
  simp only [infsep, einfsep_triple hxy hyz hxz, ENNReal.toReal_inf, edist_ne_top x y,
    edist_ne_top x z, edist_ne_top y z, dist_edist, Ne, inf_eq_top_iff, and_self_iff,
    not_false_iff]


theorem Nontrivial.infsep_anti (hs : s.Nontrivial) (hst : s ⊆ t) : t.infsep ≤ s.infsep :=
  ENNReal.toReal_mono hs.einfsep_ne_top (einfsep_anti hst)


theorem infsep_eq_iInf [Decidable s.Nontrivial] :
    s.infsep = if s.Nontrivial then ⨅ d : s.offDiag, (uncurry dist) (d : α × α) else 0 := by
  /-
    α : Type u_1
    inst✝¹ : PseudoMetricSpace α
    s : Set α
    inst✝ : Decidable s.Nontrivial
    ⊢ Eq s.infsep (ite s.Nontrivial (iInf fun d => Function.uncurry Dist.dist ↑d) 0)
  -/
  split_ifs with hs
  · have hb : BddBelow (uncurry dist '' s.offDiag) := by
      refine ⟨0, fun d h => ?_⟩
      simp_rw [mem_image, Prod.exists, uncurry_apply_pair] at h
      rcases h with ⟨_, _, _, rfl⟩
      exact dist_nonneg
    /-
      case pos
      α : Type u_1
      inst✝¹ : PseudoMetricSpace α
      s : Set α
      inst✝ : Decidable s.Nontrivial
      hs : s.Nontrivial
      hb : BddBelow (Set.image (Function.uncurry Dist.dist) s.offDiag)
      ⊢ Eq s.infsep (iInf fun d => Function.uncurry Dist.dist ↑d)
    -/
    refine eq_of_forall_le_iff fun _ => ?_
    simp_rw [hs.le_infsep_iff, le_ciInf_set_iff (offDiag_nonempty.mpr hs) hb, imp_forall_iff,
      mem_offDiag, Prod.forall, uncurry_apply_pair, and_imp]
    /-
      case neg
      α : Type u_1
      inst✝¹ : PseudoMetricSpace α
      s : Set α
      inst✝ : Decidable s.Nontrivial
      hs : Not s.Nontrivial
      ⊢ Eq s.infsep 0
    -/
  · exact (not_nontrivial_iff.mp hs).infsep_zero
    /-
      🎉 no goals
    -/


theorem Nontrivial.infsep_eq_iInf (hs : s.Nontrivial) :
    s.infsep = ⨅ d : s.offDiag, (uncurry dist) (d : α × α) := by
  /-
    α : Type u_1
    inst✝ : PseudoMetricSpace α
    s : Set α
    hs : s.Nontrivial
    ⊢ Eq s.infsep (iInf fun d => Function.uncurry Dist.dist ↑d)
  -/
  classical rw [Set.infsep_eq_iInf, if_pos hs]
  /-
    🎉 no goals
  -/


theorem infsep_of_fintype [Decidable s.Nontrivial] [DecidableEq α] [Fintype s] : s.infsep =
                                                          /-
                                                            α : Type u_1
                                                            β : Type u_2
                                                            inst✝³ : PseudoMetricSpace α
                                                            x y z : α
                                                            s t : Set α
                                                            inst✝² : Decidable s.Nontrivial
                                                            inst✝¹ : DecidableEq α
                                                            inst✝ : Fintype ↑s
                                                            hs : s.Nontrivial
                                                            ⊢ s.offDiag.toFinset.Nonempty
                                                          -/
    if hs : s.Nontrivial then s.offDiag.toFinset.inf' (by simpa) (uncurry dist) else 0 := by
                                                          /-
                                                            🎉 no goals
                                                          -/
  /-
    α : Type u_1
    inst✝³ : PseudoMetricSpace α
    s : Set α
    inst✝² : Decidable s.Nontrivial
    inst✝¹ : DecidableEq α
    inst✝ : Fintype ↑s
    ⊢ Eq s.infsep (dite s.Nontrivial (fun hs => s.offDiag.toFinset.inf' ⋯ (Functio …
  -/
  split_ifs with hs
    /-
      case pos
      α : Type u_1
      inst✝³ : PseudoMetricSpace α
      s : Set α
      inst✝² : Decidable s.Nontrivial
      inst✝¹ : DecidableEq α
      inst✝ : Fintype ↑s
      hs : s.Nontrivial
      ⊢ Eq s.infsep (s.offDiag.toFinset.inf' ⋯ (Function.uncurry Dist.dist))
    -/
  · refine eq_of_forall_le_iff fun _ => ?_
    simp_rw [hs.le_infsep_iff, imp_forall_iff, Finset.le_inf'_iff, mem_toFinset, mem_offDiag,
      Prod.forall, uncurry_apply_pair, and_imp]
    /-
      case neg
      α : Type u_1
      inst✝³ : PseudoMetricSpace α
      s : Set α
      inst✝² : Decidable s.Nontrivial
      inst✝¹ : DecidableEq α
      inst✝ : Fintype ↑s
      hs : Not s.Nontrivial
      ⊢ Eq s.infsep 0
    -/
  · rw [not_nontrivial_iff] at hs
    /-
      case neg
      α : Type u_1
      inst✝³ : PseudoMetricSpace α
      s : Set α
      inst✝² : Decidable s.Nontrivial
      inst✝¹ : DecidableEq α
      inst✝ : Fintype ↑s
      hs : s.Subsingleton
      ⊢ Eq s.infsep 0
    -/
    exact hs.infsep_zero
    /-
      🎉 no goals
    -/


theorem Nontrivial.infsep_of_fintype [DecidableEq α] [Fintype s] (hs : s.Nontrivial) :
                                           /-
                                             α : Type u_1
                                             β : Type u_2
                                             inst✝² : PseudoMetricSpace α
                                             x y z : α
                                             s t : Set α
                                             inst✝¹ : DecidableEq α
                                             inst✝ : Fintype ↑s
                                             hs : s.Nontrivial
                                             ⊢ s.offDiag.toFinset.Nonempty
                                           -/
    s.infsep = s.offDiag.toFinset.inf' (by simpa) (uncurry dist) := by
                                           /-
                                             🎉 no goals
                                           -/
  /-
    α : Type u_1
    inst✝² : PseudoMetricSpace α
    s : Set α
    inst✝¹ : DecidableEq α
    inst✝ : Fintype ↑s
    hs : s.Nontrivial
    ⊢ Eq s.infsep (s.offDiag.toFinset.inf' ⋯ (Function.uncurry Dist.dist))
  -/
  classical rw [Set.infsep_of_fintype, dif_pos hs]
  /-
    🎉 no goals
  -/


theorem Finite.infsep [Decidable s.Nontrivial] (hsf : s.Finite) :
    s.infsep =
                                                              /-
                                                                α : Type u_1
                                                                β : Type u_2
                                                                inst✝¹ : PseudoMetricSpace α
                                                                x y z : α
                                                                s t : Set α
                                                                inst✝ : Decidable s.Nontrivial
                                                                hsf : s.Finite
                                                                hs : s.Nontrivial
                                                                ⊢ ⋯.toFinset.Nonempty
                                                              -/
      if hs : s.Nontrivial then hsf.offDiag.toFinset.inf' (by simpa) (uncurry dist) else 0 := by
                                                              /-
                                                                🎉 no goals
                                                              -/
  /-
    α : Type u_1
    inst✝¹ : PseudoMetricSpace α
    s : Set α
    inst✝ : Decidable s.Nontrivial
    hsf : s.Finite
    ⊢ Eq s.infsep (dite s.Nontrivial (fun hs => ⋯.toFinset.inf' ⋯ (Function.uncurr …
  -/
  split_ifs with hs
    /-
      case pos
      α : Type u_1
      inst✝¹ : PseudoMetricSpace α
      s : Set α
      inst✝ : Decidable s.Nontrivial
      hsf : s.Finite
      hs : s.Nontrivial
      ⊢ Eq s.infsep (⋯.toFinset.inf' ⋯ (Function.uncurry Dist.dist))
    -/
  · refine eq_of_forall_le_iff fun _ => ?_
    simp_rw [hs.le_infsep_iff, imp_forall_iff, Finset.le_inf'_iff, Finite.mem_toFinset,
      mem_offDiag, Prod.forall, uncurry_apply_pair, and_imp]
    /-
      case neg
      α : Type u_1
      inst✝¹ : PseudoMetricSpace α
      s : Set α
      inst✝ : Decidable s.Nontrivial
      hsf : s.Finite
      hs : Not s.Nontrivial
      ⊢ Eq s.infsep 0
    -/
  · rw [not_nontrivial_iff] at hs
    /-
      case neg
      α : Type u_1
      inst✝¹ : PseudoMetricSpace α
      s : Set α
      inst✝ : Decidable s.Nontrivial
      hsf : s.Finite
      hs : s.Subsingleton
      ⊢ Eq s.infsep 0
    -/
    exact hs.infsep_zero
    /-
      🎉 no goals
    -/


theorem Finite.infsep_of_nontrivial (hsf : s.Finite) (hs : s.Nontrivial) :
                                             /-
                                               α : Type u_1
                                               β : Type u_2
                                               inst✝ : PseudoMetricSpace α
                                               x y z : α
                                               s t : Set α
                                               hsf : s.Finite
                                               hs : s.Nontrivial
                                               ⊢ ⋯.toFinset.Nonempty
                                             -/
    s.infsep = hsf.offDiag.toFinset.inf' (by simpa) (uncurry dist) := by
                                             /-
                                               🎉 no goals
                                             -/
  /-
    α : Type u_1
    inst✝ : PseudoMetricSpace α
    s : Set α
    hsf : s.Finite
    hs : s.Nontrivial
    ⊢ Eq s.infsep (⋯.toFinset.inf' ⋯ (Function.uncurry Dist.dist))
  -/
  classical simp_rw [hsf.infsep, dif_pos hs]
  /-
    🎉 no goals
  -/


theorem _root_.Finset.coe_infsep [DecidableEq α] (s : Finset α) : (s : Set α).infsep =
    if hs : s.offDiag.Nonempty then s.offDiag.inf' hs (uncurry dist) else 0 := by
  have H : (s : Set α).Nontrivial ↔ s.offDiag.Nonempty := by
    rw [← Set.offDiag_nonempty, ← Finset.coe_offDiag, Finset.coe_nonempty]
  /-
    α : Type u_1
    inst✝¹ : PseudoMetricSpace α
    inst✝ : DecidableEq α
    s : Finset α
    H : Iff (↑s).Nontrivial s.offDiag.Nonempty
    ⊢ Eq (↑s).infsep (dite s.offDiag.Nonempty (fun hs => s.offDiag.inf' hs (Functi …
  -/
  split_ifs with hs
    /-
      case pos
      α : Type u_1
      inst✝¹ : PseudoMetricSpace α
      inst✝ : DecidableEq α
      s : Finset α
      H : Iff (↑s).Nontrivial s.offDiag.Nonempty
      hs : s.offDiag.Nonempty
      ⊢ Eq (↑s).infsep (s.offDiag.inf' hs (Function.uncurry Dist.dist))
    -/
  · simp_rw [(H.mpr hs).infsep_of_fintype, ← Finset.coe_offDiag, Finset.toFinset_coe]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝¹ : PseudoMetricSpace α
      inst✝ : DecidableEq α
      s : Finset α
      H : Iff (↑s).Nontrivial s.offDiag.Nonempty
      hs : Not s.offDiag.Nonempty
      ⊢ Eq (↑s).infsep 0
    -/
  · exact (not_nontrivial_iff.mp (H.mp.mt hs)).infsep_zero
    /-
      🎉 no goals
    -/


theorem _root_.Finset.coe_infsep_of_offDiag_nonempty [DecidableEq α] {s : Finset α}
    (hs : s.offDiag.Nonempty) : (s : Set α).infsep = s.offDiag.inf' hs (uncurry dist) := by
  /-
    α : Type u_1
    inst✝¹ : PseudoMetricSpace α
    inst✝ : DecidableEq α
    s : Finset α
    hs : s.offDiag.Nonempty
    ⊢ Eq (↑s).infsep (s.offDiag.inf' hs (Function.uncurry Dist.dist))
  -/
  rw [Finset.coe_infsep, dif_pos hs]
  /-
    🎉 no goals
  -/


theorem _root_.Finset.coe_infsep_of_offDiag_empty
    [DecidableEq α] {s : Finset α} (hs : s.offDiag = ∅) : (s : Set α).infsep = 0 := by
  /-
    α : Type u_1
    inst✝¹ : PseudoMetricSpace α
    inst✝ : DecidableEq α
    s : Finset α
    hs : Eq s.offDiag EmptyCollection.emptyCollection
    ⊢ Eq (↑s).infsep 0
  -/
  rw [← Finset.not_nonempty_iff_eq_empty] at hs
  /-
    α : Type u_1
    inst✝¹ : PseudoMetricSpace α
    inst✝ : DecidableEq α
    s : Finset α
    hs : Not s.offDiag.Nonempty
    ⊢ Eq (↑s).infsep 0
  -/
  rw [Finset.coe_infsep, dif_neg hs]
  /-
    🎉 no goals
  -/


theorem Nontrivial.infsep_exists_of_finite [Finite s] (hs : s.Nontrivial) :
    ∃ x ∈ s, ∃ y ∈ s, x ≠ y ∧ s.infsep = dist x y := by
  classical
    cases nonempty_fintype s
    simp_rw [hs.infsep_of_fintype]
    rcases Finset.exists_mem_eq_inf' (s := s.offDiag.toFinset) (by simpa) (uncurry dist) with
      ⟨w, hxy, hed⟩
    simp_rw [mem_toFinset] at hxy
    exact ⟨w.fst, hxy.1, w.snd, hxy.2.1, hxy.2.2, hed⟩


theorem Finite.infsep_exists_of_nontrivial (hsf : s.Finite) (hs : s.Nontrivial) :
    ∃ x ∈ s, ∃ y ∈ s, x ≠ y ∧ s.infsep = dist x y :=
  letI := hsf.fintype
  hs.infsep_exists_of_finite


theorem infsep_zero_iff_subsingleton_of_finite [Finite s] : s.infsep = 0 ↔ s.Subsingleton := by
  /-
    α : Type u_1
    inst✝¹ : MetricSpace α
    s : Set α
    inst✝ : Finite ↑s
    ⊢ Iff (Eq s.infsep 0) s.Subsingleton
  -/
  rw [infsep_zero, einfsep_eq_top_iff, or_iff_right_iff_imp]
  /-
    α : Type u_1
    inst✝¹ : MetricSpace α
    s : Set α
    inst✝ : Finite ↑s
    ⊢ Eq s.einfsep 0 → s.Subsingleton
  -/
  exact fun H => (einfsep_pos_of_finite.ne' H).elim
  /-
    🎉 no goals
  -/


theorem infsep_pos_iff_nontrivial_of_finite [Finite s] : 0 < s.infsep ↔ s.Nontrivial := by
  /-
    α : Type u_1
    inst✝¹ : MetricSpace α
    s : Set α
    inst✝ : Finite ↑s
    ⊢ Iff (LT.lt 0 s.infsep) s.Nontrivial
  -/
  rw [infsep_pos, einfsep_lt_top_iff, and_iff_right_iff_imp]
  /-
    α : Type u_1
    inst✝¹ : MetricSpace α
    s : Set α
    inst✝ : Finite ↑s
    ⊢ s.Nontrivial → LT.lt 0 s.einfsep
  -/
  exact fun _ => einfsep_pos_of_finite
  /-
    🎉 no goals
  -/


theorem Finite.infsep_zero_iff_subsingleton (hs : s.Finite) : s.infsep = 0 ↔ s.Subsingleton :=
  letI := hs.fintype
  infsep_zero_iff_subsingleton_of_finite


theorem Finite.infsep_pos_iff_nontrivial (hs : s.Finite) : 0 < s.infsep ↔ s.Nontrivial :=
  letI := hs.fintype
  infsep_pos_iff_nontrivial_of_finite


theorem _root_.Finset.infsep_zero_iff_subsingleton (s : Finset α) :
    (s : Set α).infsep = 0 ↔ (s : Set α).Subsingleton :=
  infsep_zero_iff_subsingleton_of_finite


theorem _root_.Finset.infsep_pos_iff_nontrivial (s : Finset α) :
    0 < (s : Set α).infsep ↔ (s : Set α).Nontrivial :=
  infsep_pos_iff_nontrivial_of_finite


