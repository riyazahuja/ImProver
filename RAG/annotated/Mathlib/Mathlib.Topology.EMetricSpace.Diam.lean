/-- The diameter of a set in a pseudoemetric space, named `EMetric.diam` -/
noncomputable def diam (s : Set α) :=
  ⨆ (x ∈ s) (y ∈ s), edist x y


theorem diam_eq_sSup (s : Set α) : diam s = sSup (image2 edist s s) := sSup_image2.symm


theorem diam_le_iff {d : ℝ≥0∞} : diam s ≤ d ↔ ∀ x ∈ s, ∀ y ∈ s, edist x y ≤ d := by
  /-
    α : Type u_1
    s : Set α
    inst✝ : PseudoEMetricSpace α
    d : ENNReal
    ⊢ Iff (LE.le (EMetric.diam s) d) (∀ (x : α), Membership.mem s x → ∀ (y : α), M …
  -/
  simp only [diam, iSup_le_iff]
  /-
    🎉 no goals
  -/


theorem diam_image_le_iff {d : ℝ≥0∞} {f : β → α} {s : Set β} :
    diam (f '' s) ≤ d ↔ ∀ x ∈ s, ∀ y ∈ s, edist (f x) (f y) ≤ d := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : PseudoEMetricSpace α
    d : ENNReal
    f : β → α
    s : Set β
    ⊢ Iff (LE.le (EMetric.diam (Set.image f s)) d) (∀ (x : β), Membership.mem s x  …
  -/
  simp only [diam_le_iff, forall_mem_image]
  /-
    🎉 no goals
  -/


theorem edist_le_of_diam_le {d} (hx : x ∈ s) (hy : y ∈ s) (hd : diam s ≤ d) : edist x y ≤ d :=
  diam_le_iff.1 hd x hx y hy


/-- If two points belong to some set, their edistance is bounded by the diameter of the set -/
theorem edist_le_diam_of_mem (hx : x ∈ s) (hy : y ∈ s) : edist x y ≤ diam s :=
  edist_le_of_diam_le hx hy le_rfl


/-- If the distance between any two points in a set is bounded by some constant, this constant
bounds the diameter. -/
theorem diam_le {d : ℝ≥0∞} (h : ∀ x ∈ s, ∀ y ∈ s, edist x y ≤ d) : diam s ≤ d :=
  diam_le_iff.2 h


/-- The diameter of a subsingleton vanishes. -/
theorem diam_subsingleton (hs : s.Subsingleton) : diam s = 0 :=
  nonpos_iff_eq_zero.1 <| diam_le fun _x hx y hy => (hs hx hy).symm ▸ edist_self y ▸ le_rfl


/-- The diameter of the empty set vanishes -/
@[simp]
theorem diam_empty : diam (∅ : Set α) = 0 :=
  diam_subsingleton subsingleton_empty


/-- The diameter of a singleton vanishes -/
@[simp]
theorem diam_singleton : diam ({x} : Set α) = 0 :=
  diam_subsingleton subsingleton_singleton


@[to_additive (attr := simp)]
theorem diam_one [One α] : diam (1 : Set α) = 0 :=
  diam_singleton


theorem diam_iUnion_mem_option {ι : Type*} (o : Option ι) (s : ι → Set α) :
                                                    /-
                                                      α : Type u_1
                                                      inst✝ : PseudoEMetricSpace α
                                                      ι : Type u_3
                                                      o : Option ι
                                                      s : ι → Set α
                                                      ⊢ Eq (EMetric.diam (Set.iUnion fun i => Set.iUnion fun h => s i)) (iSup fun i  …
                                                    -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
    diam (⋃ i ∈ o, s i) = ⨆ i ∈ o, diam (s i) := by cases o <;> simp
                                                                /-
                                                                  🎉 no goals
                                                                -/


theorem diam_insert : diam (insert x s) = max (⨆ y ∈ s, edist x y) (diam s) :=
  eq_of_forall_ge_iff fun d => by
    simp only [diam_le_iff, forall_mem_insert, edist_self, edist_comm x, max_le_iff, iSup_le_iff,
      zero_le, true_and, forall_and, and_self_iff, ← and_assoc]


theorem diam_pair : diam ({x, y} : Set α) = edist x y := by
  /-
    α : Type u_1
    x y : α
    inst✝ : PseudoEMetricSpace α
    ⊢ Eq (EMetric.diam (Insert.insert x (Singleton.singleton y))) (EDist.edist x y)
  -/
  simp only [iSup_singleton, diam_insert, diam_singleton, ENNReal.max_zero_right]
  /-
    🎉 no goals
  -/


theorem diam_triple : diam ({x, y, z} : Set α) = max (max (edist x y) (edist x z)) (edist y z) := by
  /-
    α : Type u_1
    x y z : α
    inst✝ : PseudoEMetricSpace α
    ⊢ Eq (EMetric.diam (Insert.insert x (Insert.insert y (Singleton.singleton z))) …
  -/
  simp only [diam_insert, iSup_insert, iSup_singleton, diam_singleton, ENNReal.max_zero_right]
  /-
    🎉 no goals
  -/


/-- The diameter is monotonous with respect to inclusion -/
@[gcongr]
theorem diam_mono {s t : Set α} (h : s ⊆ t) : diam s ≤ diam t :=
  diam_le fun _x hx _y hy => edist_le_diam_of_mem (h hx) (h hy)


/-- The diameter of a union is controlled by the diameter of the sets, and the edistance
between two points in the sets. -/
theorem diam_union {t : Set α} (xs : x ∈ s) (yt : y ∈ t) :
    diam (s ∪ t) ≤ diam s + edist x y + diam t := by
  have A : ∀ a ∈ s, ∀ b ∈ t, edist a b ≤ diam s + edist x y + diam t := fun a ha b hb =>
    calc
      edist a b ≤ edist a x + edist x y + edist y b := edist_triangle4 _ _ _ _
      _ ≤ diam s + edist x y + diam t :=
        add_le_add (add_le_add (edist_le_diam_of_mem ha xs) le_rfl) (edist_le_diam_of_mem yt hb)
  /-
    α : Type u_1
    s : Set α
    x y : α
    inst✝ : PseudoEMetricSpace α
    t : Set α
    xs : Membership.mem s x
    yt : Membership.mem t y
    A : ∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem t b → LE.le (EDi …
    ⊢ LE.le (EMetric.diam (Union.union s t)) (HAdd.hAdd (HAdd.hAdd (EMetric.diam s …
  -/
  refine diam_le fun a ha b hb => ?_
  /-
    α : Type u_1
    s : Set α
    x y : α
    inst✝ : PseudoEMetricSpace α
    t : Set α
    xs : Membership.mem s x
    yt : Membership.mem t y
    A : ∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem t b → LE.le (EDi …
    a : α
    ha : Membership.mem (Union.union s t) a
    b : α
    hb : Membership.mem (Union.union s t) b
    ⊢ LE.le (EDist.edist a b) (HAdd.hAdd (HAdd.hAdd (EMetric.diam s) (EDist.edist  …
  -/
  cases' (mem_union _ _ _).1 ha with h'a h'a <;> cases' (mem_union _ _ _).1 hb with h'b h'b
  · calc
      edist a b ≤ diam s := edist_le_diam_of_mem h'a h'b
      _ ≤ diam s + (edist x y + diam t) := le_self_add
      _ = diam s + edist x y + diam t := (add_assoc _ _ _).symm
    /-
      case inl.inr
      α : Type u_1
      s : Set α
      x y : α
      inst✝ : PseudoEMetricSpace α
      t : Set α
      xs : Membership.mem s x
      yt : Membership.mem t y
      A : ∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem t b → LE.le (EDi …
      a : α
      ha : Membership.mem (Union.union s t) a
      b : α
      hb : Membership.mem (Union.union s t) b
      h'a : Membership.mem s a
      h'b : Membership.mem t b
      ⊢ LE.le (EDist.edist a b) (HAdd.hAdd (HAdd.hAdd (EMetric.diam s) (EDist.edist  …
    -/
  · exact A a h'a b h'b
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      α : Type u_1
      s : Set α
      x y : α
      inst✝ : PseudoEMetricSpace α
      t : Set α
      xs : Membership.mem s x
      yt : Membership.mem t y
      A : ∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem t b → LE.le (EDi …
      a : α
      ha : Membership.mem (Union.union s t) a
      b : α
      hb : Membership.mem (Union.union s t) b
      h'a : Membership.mem t a
      h'b : Membership.mem s b
      ⊢ LE.le (EDist.edist a b) (HAdd.hAdd (HAdd.hAdd (EMetric.diam s) (EDist.edist  …
    -/
  · have Z := A b h'b a h'a
    /-
      case inr.inl
      α : Type u_1
      s : Set α
      x y : α
      inst✝ : PseudoEMetricSpace α
      t : Set α
      xs : Membership.mem s x
      yt : Membership.mem t y
      A : ∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem t b → LE.le (EDi …
      a : α
      ha : Membership.mem (Union.union s t) a
      b : α
      hb : Membership.mem (Union.union s t) b
      h'a : Membership.mem t a
      h'b : Membership.mem s b
      Z : LE.le (EDist.edist b a) (HAdd.hAdd (HAdd.hAdd (EMetric.diam s) (EDist.edis …
      ⊢ LE.le (EDist.edist a b) (HAdd.hAdd (HAdd.hAdd (EMetric.diam s) (EDist.edist  …
    -/
    rwa [edist_comm] at Z
    /-
      🎉 no goals
    -/
  · calc
      edist a b ≤ diam t := edist_le_diam_of_mem h'a h'b
      _ ≤ diam s + edist x y + diam t := le_add_self


theorem diam_union' {t : Set α} (h : (s ∩ t).Nonempty) : diam (s ∪ t) ≤ diam s + diam t := by
  /-
    α : Type u_1
    s : Set α
    inst✝ : PseudoEMetricSpace α
    t : Set α
    h : (Inter.inter s t).Nonempty
    ⊢ LE.le (EMetric.diam (Union.union s t)) (HAdd.hAdd (EMetric.diam s) (EMetric. …
  -/
  let ⟨x, ⟨xs, xt⟩⟩ := h
  /-
    α : Type u_1
    s : Set α
    inst✝ : PseudoEMetricSpace α
    t : Set α
    h : (Inter.inter s t).Nonempty
    x : α
    xs : Membership.mem s x
    xt : Membership.mem t x
    ⊢ LE.le (EMetric.diam (Union.union s t)) (HAdd.hAdd (EMetric.diam s) (EMetric. …
  -/
  simpa using diam_union xs xt
  /-
    🎉 no goals
  -/


theorem diam_closedBall {r : ℝ≥0∞} : diam (closedBall x r) ≤ 2 * r :=
  diam_le fun a ha b hb =>
    calc
      edist a b ≤ edist a x + edist b x := edist_triangle_right _ _ _
      _ ≤ r + r := add_le_add ha hb
      _ = 2 * r := (two_mul r).symm


theorem diam_ball {r : ℝ≥0∞} : diam (ball x r) ≤ 2 * r :=
  le_trans (diam_mono ball_subset_closedBall) diam_closedBall


theorem diam_pi_le_of_le {π : β → Type*} [Fintype β] [∀ b, PseudoEMetricSpace (π b)]
    {s : ∀ b : β, Set (π b)} {c : ℝ≥0∞} (h : ∀ b, diam (s b) ≤ c) : diam (Set.pi univ s) ≤ c := by
  /-
    β : Type u_2
    π : β → Type u_3
    inst✝¹ : Fintype β
    inst✝ : (b : β) → PseudoEMetricSpace (π b)
    s : (b : β) → Set (π b)
    c : ENNReal
    h : ∀ (b : β), LE.le (EMetric.diam (s b)) c
    ⊢ LE.le (EMetric.diam (Set.univ.pi s)) c
  -/
  refine diam_le fun x hx y hy => edist_pi_le_iff.mpr ?_
  /-
    β : Type u_2
    π : β → Type u_3
    inst✝¹ : Fintype β
    inst✝ : (b : β) → PseudoEMetricSpace (π b)
    s : (b : β) → Set (π b)
    c : ENNReal
    h : ∀ (b : β), LE.le (EMetric.diam (s b)) c
    x : (i : β) → π i
    hx : Membership.mem (Set.univ.pi s) x
    y : (i : β) → π i
    hy : Membership.mem (Set.univ.pi s) y
    ⊢ ∀ (b : β), LE.le (EDist.edist (x b) (y b)) c
  -/
  rw [mem_univ_pi] at hx hy
  /-
    β : Type u_2
    π : β → Type u_3
    inst✝¹ : Fintype β
    inst✝ : (b : β) → PseudoEMetricSpace (π b)
    s : (b : β) → Set (π b)
    c : ENNReal
    h : ∀ (b : β), LE.le (EMetric.diam (s b)) c
    x : (i : β) → π i
    hx : ∀ (i : β), Membership.mem (s i) (x i)
    y : (i : β) → π i
    hy : ∀ (i : β), Membership.mem (s i) (y i)
    ⊢ ∀ (b : β), LE.le (EDist.edist (x b) (y b)) c
  -/
  exact fun b => diam_le_iff.1 (h b) (x b) (hx b) (y b) (hy b)
  /-
    🎉 no goals
  -/


theorem diam_eq_zero_iff : diam s = 0 ↔ s.Subsingleton :=
  ⟨fun h _x hx _y hy => edist_le_zero.1 <| h ▸ edist_le_diam_of_mem hx hy, diam_subsingleton⟩


theorem diam_pos_iff : 0 < diam s ↔ s.Nontrivial := by
  /-
    β : Type u_2
    inst✝ : EMetricSpace β
    s : Set β
    ⊢ Iff (LT.lt 0 (EMetric.diam s)) s.Nontrivial
  -/
  simp only [pos_iff_ne_zero, Ne, diam_eq_zero_iff, Set.not_subsingleton_iff]
  /-
    🎉 no goals
  -/


theorem diam_pos_iff' : 0 < diam s ↔ ∃ x ∈ s, ∃ y ∈ s, x ≠ y := by
  /-
    β : Type u_2
    inst✝ : EMetricSpace β
    s : Set β
    ⊢ Iff (LT.lt 0 (EMetric.diam s)) (Exists fun x => And (Membership.mem s x) (Ex …
  -/
  simp only [diam_pos_iff, Set.Nontrivial, exists_prop]
  /-
    🎉 no goals
  -/


