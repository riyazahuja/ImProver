/-- A *trail* is a walk with no repeating edges. -/
@[mk_iff isTrail_def]
structure IsTrail {u v : V} (p : G.Walk u v) : Prop where
  edges_nodup : p.edges.Nodup


/-- A *path* is a walk with no repeating vertices.
Use `SimpleGraph.Walk.IsPath.mk'` for a simpler constructor. -/
structure IsPath {u v : V} (p : G.Walk u v) extends IsTrail p : Prop where
  support_nodup : p.support.Nodup

-- Porting note: used to use `extends to_trail : is_trail p` in structure

protected lemma IsPath.isTrail {p : Walk G u v}(h : IsPath p) : IsTrail p := h.toIsTrail


/-- A *circuit* at `u : V` is a nonempty trail beginning and ending at `u`. -/
@[mk_iff isCircuit_def]
structure IsCircuit {u : V} (p : G.Walk u u) extends IsTrail p : Prop where
  ne_nil : p ≠ nil

-- Porting note: used to use `extends to_trail : is_trail p` in structure

protected lemma IsCircuit.isTrail {p : Walk G u u} (h : IsCircuit p) : IsTrail p := h.toIsTrail


/-- A *cycle* at `u : V` is a circuit at `u` whose only repeating vertex
is `u` (which appears exactly twice). -/
structure IsCycle {u : V} (p : G.Walk u u) extends IsCircuit p : Prop where
  support_nodup : p.support.tail.Nodup

-- Porting note: used to use `extends to_circuit : is_circuit p` in structure

protected lemma IsCycle.isCircuit {p : Walk G u u} (h : IsCycle p) : IsCircuit p := h.toIsCircuit


@[simp]
theorem isTrail_copy {u v u' v'} (p : G.Walk u v) (hu : u = u') (hv : v = v') :
    (p.copy hu hv).IsTrail ↔ p.IsTrail := by
  /-
    V : Type u
    G : SimpleGraph V
    u v u' v' : V
    p : G.Walk u v
    hu : Eq u u'
    hv : Eq v v'
    ⊢ Iff (p.copy hu hv).IsTrail p.IsTrail
  -/
  subst_vars
  /-
    V : Type u
    G : SimpleGraph V
    u' v' : V
    p : G.Walk u' v'
    ⊢ Iff (p.copy ⋯ ⋯).IsTrail p.IsTrail
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem IsPath.mk' {u v : V} {p : G.Walk u v} (h : p.support.Nodup) : p.IsPath :=
  ⟨⟨edges_nodup_of_support_nodup h⟩, h⟩


theorem isPath_def {u v : V} (p : G.Walk u v) : p.IsPath ↔ p.support.Nodup :=
  ⟨IsPath.support_nodup, IsPath.mk'⟩


@[simp]
theorem isPath_copy {u v u' v'} (p : G.Walk u v) (hu : u = u') (hv : v = v') :
    (p.copy hu hv).IsPath ↔ p.IsPath := by
  /-
    V : Type u
    G : SimpleGraph V
    u v u' v' : V
    p : G.Walk u v
    hu : Eq u u'
    hv : Eq v v'
    ⊢ Iff (p.copy hu hv).IsPath p.IsPath
  -/
  subst_vars
  /-
    V : Type u
    G : SimpleGraph V
    u' v' : V
    p : G.Walk u' v'
    ⊢ Iff (p.copy ⋯ ⋯).IsPath p.IsPath
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem isCircuit_copy {u u'} (p : G.Walk u u) (hu : u = u') :
    (p.copy hu hu).IsCircuit ↔ p.IsCircuit := by
  /-
    V : Type u
    G : SimpleGraph V
    u u' : V
    p : G.Walk u u
    hu : Eq u u'
    ⊢ Iff (p.copy hu hu).IsCircuit p.IsCircuit
  -/
  subst_vars
  /-
    V : Type u
    G : SimpleGraph V
    u' : V
    p : G.Walk u' u'
    ⊢ Iff (p.copy ⋯ ⋯).IsCircuit p.IsCircuit
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma IsCircuit.not_nil {p : G.Walk v v} (hp : IsCircuit p) : ¬ p.Nil := (hp.ne_nil ·.eq_nil)


theorem isCycle_def {u : V} (p : G.Walk u u) :
    p.IsCycle ↔ p.IsTrail ∧ p ≠ nil ∧ p.support.tail.Nodup :=
  Iff.intro (fun h => ⟨h.1.1, h.1.2, h.2⟩) fun h => ⟨⟨h.1, h.2.1⟩, h.2.2⟩


@[simp]
theorem isCycle_copy {u u'} (p : G.Walk u u) (hu : u = u') :
    (p.copy hu hu).IsCycle ↔ p.IsCycle := by
  /-
    V : Type u
    G : SimpleGraph V
    u u' : V
    p : G.Walk u u
    hu : Eq u u'
    ⊢ Iff (p.copy hu hu).IsCycle p.IsCycle
  -/
  subst_vars
  /-
    V : Type u
    G : SimpleGraph V
    u' : V
    p : G.Walk u' u'
    ⊢ Iff (p.copy ⋯ ⋯).IsCycle p.IsCycle
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma IsCycle.not_nil {p : G.Walk v v} (hp : IsCycle p) : ¬ p.Nil := (hp.ne_nil ·.eq_nil)


@[simp]
theorem IsTrail.nil {u : V} : (nil : G.Walk u u).IsTrail :=
      /-
        V : Type u
        G : SimpleGraph V
        u : V
        ⊢ SimpleGraph.Walk.nil.edges.Nodup
      -/
  ⟨by simp [edges]⟩
      /-
        🎉 no goals
      -/


theorem IsTrail.of_cons {u v w : V} {h : G.Adj u v} {p : G.Walk v w} :
                                         /-
                                           V : Type u
                                           G : SimpleGraph V
                                           u v w : V
                                           h : G.Adj u v
                                           p : G.Walk v w
                                           ⊢ (SimpleGraph.Walk.cons h p).IsTrail → p.IsTrail
                                         -/
    (cons h p).IsTrail → p.IsTrail := by simp [isTrail_def]
                                         /-
                                           🎉 no goals
                                         -/


@[simp]
theorem cons_isTrail_iff {u v w : V} (h : G.Adj u v) (p : G.Walk v w) :
                                                             /-
                                                               V : Type u
                                                               G : SimpleGraph V
                                                               u v w : V
                                                               h : G.Adj u v
                                                               p : G.Walk v w
                                                               ⊢ Iff (SimpleGraph.Walk.cons h p).IsTrail (And p.IsTrail (Not (Membership.mem  …
                                                             -/
    (cons h p).IsTrail ↔ p.IsTrail ∧ s(u, v) ∉ p.edges := by simp [isTrail_def, and_comm]
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem IsTrail.reverse {u v : V} (p : G.Walk u v) (h : p.IsTrail) : p.reverse.IsTrail := by
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    p : G.Walk u v
    h : p.IsTrail
    ⊢ p.reverse.IsTrail
  -/
  simpa [isTrail_def] using h
  /-
    🎉 no goals
  -/


@[simp]
theorem reverse_isTrail_iff {u v : V} (p : G.Walk u v) : p.reverse.IsTrail ↔ p.IsTrail := by
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    p : G.Walk u v
    ⊢ Iff p.reverse.IsTrail p.IsTrail
  -/
  constructor <;>
      /-
        case mp
        V : Type u
        G : SimpleGraph V
        u v : V
        p : G.Walk u v
        ⊢ p.reverse.IsTrail → p.IsTrail
      -/
      /-
        case mp
        V : Type u
        G : SimpleGraph V
        u v : V
        p : G.Walk u v
        h : p.reverse.IsTrail
        ⊢ p.IsTrail
      -/
      /-
        case h.e'_5
        V : Type u
        G : SimpleGraph V
        u v : V
        p : G.Walk u v
        h : p.reverse.IsTrail
        ⊢ Eq p p.reverse.reverse
      -/
      /-
        🎉 no goals
      -/
      convert h.reverse _
      /-
        🎉 no goals
      -/
      /-
        🎉 no goals
      -/
      try rw [reverse_reverse]
      /-
        🎉 no goals
      -/


theorem IsTrail.of_append_left {u v w : V} {p : G.Walk u v} {q : G.Walk v w}
    (h : (p.append q).IsTrail) : p.IsTrail := by
  /-
    V : Type u
    G : SimpleGraph V
    u v w : V
    p : G.Walk u v
    q : G.Walk v w
    h : (p.append q).IsTrail
    ⊢ p.IsTrail
  -/
  rw [isTrail_def, edges_append, List.nodup_append] at h
  /-
    V : Type u
    G : SimpleGraph V
    u v w : V
    p : G.Walk u v
    q : G.Walk v w
    h : And p.edges.Nodup (And q.edges.Nodup (p.edges.Disjoint q.edges))
    ⊢ p.IsTrail
  -/
  exact ⟨h.1⟩
  /-
    🎉 no goals
  -/


theorem IsTrail.of_append_right {u v w : V} {p : G.Walk u v} {q : G.Walk v w}
    (h : (p.append q).IsTrail) : q.IsTrail := by
  /-
    V : Type u
    G : SimpleGraph V
    u v w : V
    p : G.Walk u v
    q : G.Walk v w
    h : (p.append q).IsTrail
    ⊢ q.IsTrail
  -/
  rw [isTrail_def, edges_append, List.nodup_append] at h
  /-
    V : Type u
    G : SimpleGraph V
    u v w : V
    p : G.Walk u v
    q : G.Walk v w
    h : And p.edges.Nodup (And q.edges.Nodup (p.edges.Disjoint q.edges))
    ⊢ q.IsTrail
  -/
  exact ⟨h.2.1⟩
  /-
    🎉 no goals
  -/


theorem IsTrail.count_edges_le_one [DecidableEq V] {u v : V} {p : G.Walk u v} (h : p.IsTrail)
    (e : Sym2 V) : p.edges.count e ≤ 1 :=
  List.nodup_iff_count_le_one.mp h.edges_nodup e


theorem IsTrail.count_edges_eq_one [DecidableEq V] {u v : V} {p : G.Walk u v} (h : p.IsTrail)
    {e : Sym2 V} (he : e ∈ p.edges) : p.edges.count e = 1 :=
  List.count_eq_one_of_mem h.edges_nodup he


theorem IsTrail.length_le_card_edgeFinset [Fintype G.edgeSet] {u v : V}
    {w : G.Walk u v} (h : w.IsTrail) : w.length ≤ G.edgeFinset.card := by
  classical
  let edges := w.edges.toFinset
  have : edges.card = w.length := length_edges _ ▸ List.toFinset_card_of_nodup h.edges_nodup
  rw [← this]
  have : edges ⊆ G.edgeFinset := by
    intro e h
    refine mem_edgeFinset.mpr ?_
    apply w.edges_subset_edgeSet
    simpa [edges] using h
  exact Finset.card_le_card this


                                                             /-
                                                               V : Type u
                                                               G : SimpleGraph V
                                                               u : V
                                                               ⊢ SimpleGraph.Walk.nil.IsPath
                                                             -/
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
theorem IsPath.nil {u : V} : (nil : G.Walk u u).IsPath := by constructor <;> simp
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


theorem IsPath.of_cons {u v w : V} {h : G.Adj u v} {p : G.Walk v w} :
                                       /-
                                         V : Type u
                                         G : SimpleGraph V
                                         u v w : V
                                         h : G.Adj u v
                                         p : G.Walk v w
                                         ⊢ (SimpleGraph.Walk.cons h p).IsPath → p.IsPath
                                       -/
    (cons h p).IsPath → p.IsPath := by simp [isPath_def]
                                       /-
                                         🎉 no goals
                                       -/


@[simp]
theorem cons_isPath_iff {u v w : V} (h : G.Adj u v) (p : G.Walk v w) :
    (cons h p).IsPath ↔ p.IsPath ∧ u ∉ p.support := by
  /-
    V : Type u
    G : SimpleGraph V
    u v w : V
    h : G.Adj u v
    p : G.Walk v w
    ⊢ Iff (SimpleGraph.Walk.cons h p).IsPath (And p.IsPath (Not (Membership.mem p. …
  -/
                  /-
                    🎉 no goals
                  -/
  constructor <;> simp +contextual [isPath_def]
                  /-
                    🎉 no goals
                  -/


protected lemma IsPath.cons {p : Walk G v w} (hp : p.IsPath) (hu : u ∉ p.support) {h : G.Adj u v} :
    (cons h p).IsPath :=
  (cons_isPath_iff _ _).2 ⟨hp, hu⟩


@[simp]
theorem isPath_iff_eq_nil {u : V} (p : G.Walk u u) : p.IsPath ↔ p = nil := by
  /-
    V : Type u
    G : SimpleGraph V
    u : V
    p : G.Walk u u
    ⊢ Iff p.IsPath (Eq p SimpleGraph.Walk.nil)
  -/
              /-
                🎉 no goals
              -/
  cases p <;> simp [IsPath.nil]
              /-
                🎉 no goals
              -/


theorem IsPath.reverse {u v : V} {p : G.Walk u v} (h : p.IsPath) : p.reverse.IsPath := by
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    p : G.Walk u v
    h : p.IsPath
    ⊢ p.reverse.IsPath
  -/
  simpa [isPath_def] using h
  /-
    🎉 no goals
  -/


@[simp]
theorem isPath_reverse_iff {u v : V} (p : G.Walk u v) : p.reverse.IsPath ↔ p.IsPath := by
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    p : G.Walk u v
    ⊢ Iff p.reverse.IsPath p.IsPath
  -/
                              /-
                                🎉 no goals
                              -/
  constructor <;> intro h <;> convert h.reverse; simp
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem IsPath.of_append_left {u v w : V} {p : G.Walk u v} {q : G.Walk v w} :
    (p.append q).IsPath → p.IsPath := by
  /-
    V : Type u
    G : SimpleGraph V
    u v w : V
    p : G.Walk u v
    q : G.Walk v w
    ⊢ (p.append q).IsPath → p.IsPath
  -/
  simp only [isPath_def, support_append]
  /-
    V : Type u
    G : SimpleGraph V
    u v w : V
    p : G.Walk u v
    q : G.Walk v w
    ⊢ (HAppend.hAppend p.support q.support.tail).Nodup → p.support.Nodup
  -/
  exact List.Nodup.of_append_left
  /-
    🎉 no goals
  -/


theorem IsPath.of_append_right {u v w : V} {p : G.Walk u v} {q : G.Walk v w}
    (h : (p.append q).IsPath) : q.IsPath := by
  /-
    V : Type u
    G : SimpleGraph V
    u v w : V
    p : G.Walk u v
    q : G.Walk v w
    h : (p.append q).IsPath
    ⊢ q.IsPath
  -/
  rw [← isPath_reverse_iff] at h ⊢
  /-
    V : Type u
    G : SimpleGraph V
    u v w : V
    p : G.Walk u v
    q : G.Walk v w
    h : (p.append q).reverse.IsPath
    ⊢ q.reverse.IsPath
  -/
  rw [reverse_append] at h
  /-
    V : Type u
    G : SimpleGraph V
    u v w : V
    p : G.Walk u v
    q : G.Walk v w
    h : (q.reverse.append p.reverse).IsPath
    ⊢ q.reverse.IsPath
  -/
  apply h.of_append_left
  /-
    🎉 no goals
  -/


@[simp]
theorem IsCycle.not_of_nil {u : V} : ¬(nil : G.Walk u u).IsCycle := fun h => h.ne_nil rfl


lemma IsCycle.ne_bot : ∀ {p : G.Walk u u}, p.IsCycle → G ≠ ⊥
                  /-
                    V : Type u
                    G : SimpleGraph V
                    u : V
                    hp : SimpleGraph.Walk.nil.IsCycle
                    ⊢ Ne G Bot.bot
                  -/
  | nil, hp => by cases hp.ne_nil rfl
                  /-
                    🎉 no goals
                  -/
                       /-
                         V : Type u
                         G : SimpleGraph V
                         u v✝ : V
                         h : G.Adj u v✝
                         p✝ : G.Walk v✝ u
                         hp : (SimpleGraph.Walk.cons h p✝).IsCycle
                         ⊢ Ne G Bot.bot
                       -/
  | cons h _, hp => by rintro rfl; exact h
                                   /-
                                     🎉 no goals
                                   -/


lemma IsCycle.three_le_length {v : V} {p : G.Walk v v} (hp : p.IsCycle) : 3 ≤ p.length := by
  /-
    V : Type u
    G : SimpleGraph V
    v : V
    p : G.Walk v v
    hp : p.IsCycle
    ⊢ LE.le 3 p.length
  -/
  have ⟨⟨hp, hp'⟩, _⟩ := hp
  match p with
  | .nil => simp at hp'
  | .cons h .nil => simp at h
  | .cons _ (.cons _ .nil) => simp at hp
  | .cons _ (.cons _ (.cons _ _)) => simp_rw [SimpleGraph.Walk.length_cons]; omega


theorem cons_isCycle_iff {u v : V} (p : G.Walk v u) (h : G.Adj u v) :
    (Walk.cons h p).IsCycle ↔ p.IsPath ∧ ¬s(u, v) ∈ p.edges := by
  simp only [Walk.isCycle_def, Walk.isPath_def, Walk.isTrail_def, edges_cons, List.nodup_cons,
    support_cons, List.tail_cons]
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    p : G.Walk v u
    h : G.Adj u v
    ⊢ Iff (And (And (Not (Membership.mem p.edges (Sym2.mk { fst := u, snd := v })) …
  -/
  have : p.support.Nodup → p.edges.Nodup := edges_nodup_of_support_nodup
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    p : G.Walk v u
    h : G.Adj u v
    this : p.support.Nodup → p.edges.Nodup
    ⊢ Iff (And (And (Not (Membership.mem p.edges (Sym2.mk { fst := u, snd := v })) …
  -/
  tauto
  /-
    🎉 no goals
  -/


protected lemma IsCycle.reverse {p : G.Walk u u} (h : p.IsCycle) : p.reverse.IsCycle := by
  /-
    V : Type u
    G : SimpleGraph V
    u : V
    p : G.Walk u u
    h : p.IsCycle
    ⊢ p.reverse.IsCycle
  -/
  simp only [Walk.isCycle_def, nodup_tail_support_reverse] at h ⊢
  /-
    V : Type u
    G : SimpleGraph V
    u : V
    p : G.Walk u u
    h : And p.IsTrail (And (Ne p SimpleGraph.Walk.nil) p.support.tail.Nodup)
    ⊢ And p.reverse.IsTrail (And (Ne p.reverse SimpleGraph.Walk.nil) p.support.tai …
  -/
  exact ⟨h.1.reverse, fun h' ↦ h.2.1 (by simp_all [← Walk.length_eq_zero_iff]), h.2.2⟩
  /-
    🎉 no goals
  -/


@[simp]
lemma isCycle_reverse {p : G.Walk u u} : p.reverse.IsCycle ↔ p.IsCycle where
             /-
               V : Type u
               G : SimpleGraph V
               u : V
               p : G.Walk u u
               h : p.reverse.IsCycle
               ⊢ p.IsCycle
             -/
  mp h := by simpa using h.reverse
             /-
               🎉 no goals
             -/
  mpr := .reverse


lemma IsPath.tail {p : G.Walk u v} (hp : p.IsPath) (hp' : ¬ p.Nil) : p.tail.IsPath := by
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    p : G.Walk u v
    hp : p.IsPath
    hp' : Not p.Nil
    ⊢ p.tail.IsPath
  -/
  rw [Walk.isPath_def] at hp ⊢
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    p : G.Walk u v
    hp : p.support.Nodup
    hp' : Not p.Nil
    ⊢ p.tail.support.Nodup
  -/
  rw [← cons_support_tail _ hp', List.nodup_cons] at hp
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    p : G.Walk u v
    hp : And (Not (Membership.mem p.tail.support u)) p.tail.support.Nodup
    hp' : Not p.Nil
    ⊢ p.tail.support.Nodup
  -/
  exact hp.2
  /-
    🎉 no goals
  -/


instance [DecidableEq V] {u v : V} (p : G.Walk u v) : Decidable p.IsPath := by
  /-
    V : Type u
    V' : Type v
    V'' : Type w
    G : SimpleGraph V
    G' : SimpleGraph V'
    G'' : SimpleGraph V''
    u✝ v✝ w : V
    inst✝ : DecidableEq V
    u v : V
    p : G.Walk u v
    ⊢ Decidable p.IsPath
  -/
  rw [isPath_def]
  /-
    V : Type u
    V' : Type v
    V'' : Type w
    G : SimpleGraph V
    G' : SimpleGraph V'
    G'' : SimpleGraph V''
    u✝ v✝ w : V
    inst✝ : DecidableEq V
    u v : V
    p : G.Walk u v
    ⊢ Decidable p.support.Nodup
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem IsPath.length_lt [Fintype V] {u v : V} {p : G.Walk u v} (hp : p.IsPath) :
    p.length < Fintype.card V := by
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : Fintype V
    u v : V
    p : G.Walk u v
    hp : p.IsPath
    ⊢ LT.lt p.length (Fintype.card V)
  -/
  rw [Nat.lt_iff_add_one_le, ← length_support]
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : Fintype V
    u v : V
    p : G.Walk u v
    hp : p.IsPath
    ⊢ LE.le p.support.length (Fintype.card V)
  -/
  exact hp.support_nodup.length_le_card
  /-
    🎉 no goals
  -/


protected theorem IsTrail.takeUntil {u v w : V} {p : G.Walk v w} (hc : p.IsTrail)
    (h : u ∈ p.support) : (p.takeUntil u h).IsTrail :=
                                                    /-
                                                      V : Type u
                                                      G : SimpleGraph V
                                                      inst✝ : DecidableEq V
                                                      u v w : V
                                                      p : G.Walk v w
                                                      hc : p.IsTrail
                                                      h : Membership.mem p.support u
                                                      ⊢ ((p.takeUntil u h).append (p.dropUntil u h)).IsTrail
                                                    -/
  IsTrail.of_append_left (q := p.dropUntil u h) (by rwa [← take_spec _ h] at hc)
                                                    /-
                                                      🎉 no goals
                                                    -/


protected theorem IsTrail.dropUntil {u v w : V} {p : G.Walk v w} (hc : p.IsTrail)
    (h : u ∈ p.support) : (p.dropUntil u h).IsTrail :=
  IsTrail.of_append_right (p := p.takeUntil u h) (q := p.dropUntil u h)
        /-
          V : Type u
          G : SimpleGraph V
          inst✝ : DecidableEq V
          u v w : V
          p : G.Walk v w
          hc : p.IsTrail
          h : Membership.mem p.support u
          ⊢ ((p.takeUntil u h).append (p.dropUntil u h)).IsTrail
        -/
    (by rwa [← take_spec _ h] at hc)
        /-
          🎉 no goals
        -/


protected theorem IsPath.takeUntil {u v w : V} {p : G.Walk v w} (hc : p.IsPath)
    (h : u ∈ p.support) : (p.takeUntil u h).IsPath :=
                                                   /-
                                                     V : Type u
                                                     G : SimpleGraph V
                                                     inst✝ : DecidableEq V
                                                     u v w : V
                                                     p : G.Walk v w
                                                     hc : p.IsPath
                                                     h : Membership.mem p.support u
                                                     ⊢ ((p.takeUntil u h).append (p.dropUntil u h)).IsPath
                                                   -/
  IsPath.of_append_left (q := p.dropUntil u h) (by rwa [← take_spec _ h] at hc)
                                                   /-
                                                     🎉 no goals
                                                   -/

-- Porting note: p was previously accidentally an explicit argument

protected theorem IsPath.dropUntil {u v w : V} {p : G.Walk v w} (hc : p.IsPath)
    (h : u ∈ p.support) : (p.dropUntil u h).IsPath :=
  IsPath.of_append_right (p := p.takeUntil u h) (q := p.dropUntil u h)
        /-
          V : Type u
          G : SimpleGraph V
          inst✝ : DecidableEq V
          u v w : V
          p : G.Walk v w
          hc : p.IsPath
          h : Membership.mem p.support u
          ⊢ ((p.takeUntil u h).append (p.dropUntil u h)).IsPath
        -/
    (by rwa [← take_spec _ h] at hc)
        /-
          🎉 no goals
        -/


protected theorem IsTrail.rotate {u v : V} {c : G.Walk v v} (hc : c.IsTrail) (h : u ∈ c.support) :
    (c.rotate h).IsTrail := by
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v : V
    c : G.Walk v v
    hc : c.IsTrail
    h : Membership.mem c.support u
    ⊢ (c.rotate h).IsTrail
  -/
  rw [isTrail_def, (c.rotate_edges h).perm.nodup_iff]
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v : V
    c : G.Walk v v
    hc : c.IsTrail
    h : Membership.mem c.support u
    ⊢ c.edges.Nodup
  -/
  exact hc.edges_nodup
  /-
    🎉 no goals
  -/


protected theorem IsCircuit.rotate {u v : V} {c : G.Walk v v} (hc : c.IsCircuit)
    (h : u ∈ c.support) : (c.rotate h).IsCircuit := by
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v : V
    c : G.Walk v v
    hc : c.IsCircuit
    h : Membership.mem c.support u
    ⊢ (c.rotate h).IsCircuit
  -/
  refine ⟨hc.isTrail.rotate _, ?_⟩
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v : V
    c : G.Walk v v
    hc : c.IsCircuit
    h : Membership.mem c.support u
    ⊢ Ne (c.rotate h) SimpleGraph.Walk.nil
  -/
  cases c
    /-
      case nil
      V : Type u
      G : SimpleGraph V
      inst✝ : DecidableEq V
      u v : V
      hc : SimpleGraph.Walk.nil.IsCircuit
      h : Membership.mem SimpleGraph.Walk.nil.support u
      ⊢ Ne (SimpleGraph.Walk.nil.rotate h) SimpleGraph.Walk.nil
    -/
  · exact (hc.ne_nil rfl).elim
    /-
      🎉 no goals
    -/
    /-
      case cons
      V : Type u
      G : SimpleGraph V
      inst✝ : DecidableEq V
      u v v✝ : V
      h✝ : G.Adj v v✝
      p✝ : G.Walk v✝ v
      hc : (SimpleGraph.Walk.cons h✝ p✝).IsCircuit
      h : Membership.mem (SimpleGraph.Walk.cons h✝ p✝).support u
      ⊢ Ne ((SimpleGraph.Walk.cons h✝ p✝).rotate h) SimpleGraph.Walk.nil
    -/
  · intro hn
    /-
      case cons
      V : Type u
      G : SimpleGraph V
      inst✝ : DecidableEq V
      u v v✝ : V
      h✝ : G.Adj v v✝
      p✝ : G.Walk v✝ v
      hc : (SimpleGraph.Walk.cons h✝ p✝).IsCircuit
      h : Membership.mem (SimpleGraph.Walk.cons h✝ p✝).support u
      hn : Eq ((SimpleGraph.Walk.cons h✝ p✝).rotate h) SimpleGraph.Walk.nil
      ⊢ False
    -/
    have hn' := congr_arg length hn
    /-
      case cons
      V : Type u
      G : SimpleGraph V
      inst✝ : DecidableEq V
      u v v✝ : V
      h✝ : G.Adj v v✝
      p✝ : G.Walk v✝ v
      hc : (SimpleGraph.Walk.cons h✝ p✝).IsCircuit
      h : Membership.mem (SimpleGraph.Walk.cons h✝ p✝).support u
      hn : Eq ((SimpleGraph.Walk.cons h✝ p✝).rotate h) SimpleGraph.Walk.nil
      hn' : Eq ((SimpleGraph.Walk.cons h✝ p✝).rotate h).length SimpleGraph.Walk.nil. …
      ⊢ False
    -/
    rw [rotate, length_append, add_comm, ← length_append, take_spec] at hn'
    /-
      case cons
      V : Type u
      G : SimpleGraph V
      inst✝ : DecidableEq V
      u v v✝ : V
      h✝ : G.Adj v v✝
      p✝ : G.Walk v✝ v
      hc : (SimpleGraph.Walk.cons h✝ p✝).IsCircuit
      h : Membership.mem (SimpleGraph.Walk.cons h✝ p✝).support u
      hn : Eq ((SimpleGraph.Walk.cons h✝ p✝).rotate h) SimpleGraph.Walk.nil
      hn' : Eq (SimpleGraph.Walk.cons h✝ p✝).length SimpleGraph.Walk.nil.length
      ⊢ False
    -/
    simp at hn'
    /-
      🎉 no goals
    -/


protected theorem IsCycle.rotate {u v : V} {c : G.Walk v v} (hc : c.IsCycle) (h : u ∈ c.support) :
    (c.rotate h).IsCycle := by
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v : V
    c : G.Walk v v
    hc : c.IsCycle
    h : Membership.mem c.support u
    ⊢ (c.rotate h).IsCycle
  -/
  refine ⟨hc.isCircuit.rotate _, ?_⟩
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v : V
    c : G.Walk v v
    hc : c.IsCycle
    h : Membership.mem c.support u
    ⊢ (c.rotate h).support.tail.Nodup
  -/
  rw [List.IsRotated.nodup_iff (support_rotate _ _)]
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v : V
    c : G.Walk v v
    hc : c.IsCycle
    h : Membership.mem c.support u
    ⊢ c.support.tail.Nodup
  -/
  exact hc.support_nodup
  /-
    🎉 no goals
  -/


/-- The type for paths between two vertices. -/
abbrev Path (u v : V) := { p : G.Walk u v // p.IsPath }


@[simp]
protected theorem isPath {u v : V} (p : G.Path u v) : (p : G.Walk u v).IsPath := p.property


@[simp]
protected theorem isTrail {u v : V} (p : G.Path u v) : (p : G.Walk u v).IsTrail :=
  p.property.isTrail


/-- The length-0 path at a vertex. -/
@[refl, simps]
protected def nil {u : V} : G.Path u u :=
  ⟨Walk.nil, Walk.IsPath.nil⟩


/-- The length-1 path between a pair of adjacent vertices. -/
@[simps]
def singleton {u v : V} (h : G.Adj u v) : G.Path u v :=
                            /-
                              V : Type u
                              V' : Type v
                              V'' : Type w
                              G : SimpleGraph V
                              G' : SimpleGraph V'
                              G'' : SimpleGraph V''
                              u v : V
                              h : G.Adj u v
                              ⊢ (SimpleGraph.Walk.cons h SimpleGraph.Walk.nil).IsPath
                            -/
  ⟨Walk.cons h Walk.nil, by simp [h.ne]⟩
                            /-
                              🎉 no goals
                            -/


theorem mk'_mem_edges_singleton {u v : V} (h : G.Adj u v) :
                                                     /-
                                                       V : Type u
                                                       G : SimpleGraph V
                                                       u v : V
                                                       h : G.Adj u v
                                                       ⊢ Membership.mem (↑(SimpleGraph.Path.singleton h)).edges (Sym2.mk { fst := u,  …
                                                     -/
    s(u, v) ∈ (singleton h : G.Walk u v).edges := by simp [singleton]
                                                     /-
                                                       🎉 no goals
                                                     -/


/-- The reverse of a path is another path.  See also `SimpleGraph.Walk.reverse`. -/
@[symm, simps]
def reverse {u v : V} (p : G.Path u v) : G.Path v u :=
  ⟨Walk.reverse p, p.property.reverse⟩


theorem count_support_eq_one [DecidableEq V] {u v w : V} {p : G.Path u v}
    (hw : w ∈ (p : G.Walk u v).support) : (p : G.Walk u v).support.count w = 1 :=
  List.count_eq_one_of_mem p.property.support_nodup hw


theorem count_edges_eq_one [DecidableEq V] {u v : V} {p : G.Path u v} (e : Sym2 V)
    (hw : e ∈ (p : G.Walk u v).edges) : (p : G.Walk u v).edges.count e = 1 :=
  List.count_eq_one_of_mem p.property.isTrail.edges_nodup hw


@[simp]
theorem nodup_support {u v : V} (p : G.Path u v) : (p : G.Walk u v).support.Nodup :=
  (Walk.isPath_def _).mp p.property


theorem loop_eq {v : V} (p : G.Path v v) : p = Path.nil := by
  /-
    V : Type u
    G : SimpleGraph V
    v : V
    p : G.Path v v
    ⊢ Eq p SimpleGraph.Path.nil
  -/
  obtain ⟨_ | _, h⟩ := p
    /-
      case mk.nil
      V : Type u
      G : SimpleGraph V
      v : V
      h : SimpleGraph.Walk.nil.IsPath
      ⊢ Eq ⟨SimpleGraph.Walk.nil, h⟩ SimpleGraph.Path.nil
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case mk.cons
      V : Type u
      G : SimpleGraph V
      v v✝ : V
      h✝ : G.Adj v v✝
      p✝ : G.Walk v✝ v
      h : (SimpleGraph.Walk.cons h✝ p✝).IsPath
      ⊢ Eq ⟨SimpleGraph.Walk.cons h✝ p✝, h⟩ SimpleGraph.Path.nil
    -/
  · simp at h
    /-
      🎉 no goals
    -/


theorem not_mem_edges_of_loop {v : V} {e : Sym2 V} {p : G.Path v v} :
                                      /-
                                        V : Type u
                                        G : SimpleGraph V
                                        v : V
                                        e : Sym2 V
                                        p : G.Path v v
                                        ⊢ Not (Membership.mem (↑p).edges e)
                                      -/
    ¬e ∈ (p : G.Walk v v).edges := by simp [p.loop_eq]
                                      /-
                                        🎉 no goals
                                      -/


theorem cons_isCycle {u v : V} (p : G.Path v u) (h : G.Adj u v)
    (he : ¬s(u, v) ∈ (p : G.Walk v u).edges) : (Walk.cons h ↑p).IsCycle := by
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    p : G.Path v u
    h : G.Adj u v
    he : Not (Membership.mem (↑p).edges (Sym2.mk { fst := u, snd := v }))
    ⊢ (SimpleGraph.Walk.cons h ↑p).IsCycle
  -/
  simp [Walk.isCycle_def, Walk.cons_isTrail_iff, he]
  /-
    🎉 no goals
  -/


/-- Given a walk, produces a walk from it by bypassing subwalks between repeated vertices.
The result is a path, as shown in `SimpleGraph.Walk.bypass_isPath`.
This is packaged up in `SimpleGraph.Walk.toPath`. -/
def bypass {u v : V} : G.Walk u v → G.Walk u v
  | nil => nil
  | cons ha p =>
    let p' := p.bypass
    if hs : u ∈ p'.support then
      p'.dropUntil u hs
    else
      cons ha p'


@[simp]
theorem bypass_copy {u v u' v'} (p : G.Walk u v) (hu : u = u') (hv : v = v') :
    (p.copy hu hv).bypass = p.bypass.copy hu hv := by
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v u' v' : V
    p : G.Walk u v
    hu : Eq u u'
    hv : Eq v v'
    ⊢ Eq (p.copy hu hv).bypass (p.bypass.copy hu hv)
  -/
  subst_vars
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u' v' : V
    p : G.Walk u' v'
    ⊢ Eq (p.copy ⋯ ⋯).bypass (p.bypass.copy ⋯ ⋯)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem bypass_isPath {u v : V} (p : G.Walk u v) : p.bypass.IsPath := by
  induction p with
  | nil => simp!
  | cons _ p' ih =>
    simp only [bypass]
    split_ifs with hs
    · exact ih.dropUntil hs
    · simp [*, cons_isPath_iff]


theorem length_bypass_le {u v : V} (p : G.Walk u v) : p.bypass.length ≤ p.length := by
  induction p with
  | nil => rfl
  | cons _ _ ih =>
    simp only [bypass]
    split_ifs
    · trans
      · apply length_dropUntil_le
      rw [length_cons]
      omega
    · rw [length_cons, length_cons]
      exact Nat.add_le_add_right ih 1


lemma bypass_eq_self_of_length_le {u v : V} (p : G.Walk u v) (h : p.length ≤ p.bypass.length) :
    p.bypass = p := by
  induction p with
  | nil => rfl
  | cons h p ih =>
    simp only [Walk.bypass]
    split_ifs with hb
    · exfalso
      simp only [hb, Walk.bypass, Walk.length_cons, dif_pos] at h
      apply Nat.not_succ_le_self p.length
      calc p.length + 1
        _ ≤ (p.bypass.dropUntil _ _).length := h
        _ ≤ p.bypass.length := Walk.length_dropUntil_le p.bypass hb
        _ ≤ p.length := Walk.length_bypass_le _
    · simp only [hb, Walk.bypass, Walk.length_cons, not_false_iff, dif_neg,
        Nat.add_le_add_iff_right] at h
      rw [ih h]


/-- Given a walk, produces a path with the same endpoints using `SimpleGraph.Walk.bypass`. -/
def toPath {u v : V} (p : G.Walk u v) : G.Path u v :=
  ⟨p.bypass, p.bypass_isPath⟩


theorem support_bypass_subset {u v : V} (p : G.Walk u v) : p.bypass.support ⊆ p.support := by
  induction p with
  | nil => simp!
  | cons _ _ ih =>
    simp! only
    split_ifs
    · apply List.Subset.trans (support_dropUntil_subset _ _)
      apply List.subset_cons_of_subset
      assumption
    · rw [support_cons]
      apply List.cons_subset_cons
      assumption


theorem support_toPath_subset {u v : V} (p : G.Walk u v) :
    (p.toPath : G.Walk u v).support ⊆ p.support :=
  support_bypass_subset _


theorem darts_bypass_subset {u v : V} (p : G.Walk u v) : p.bypass.darts ⊆ p.darts := by
  induction p with
  | nil => simp!
  | cons _ _ ih =>
    simp! only
    split_ifs
    · apply List.Subset.trans (darts_dropUntil_subset _ _)
      apply List.subset_cons_of_subset _ ih
    · rw [darts_cons]
      exact List.cons_subset_cons _ ih


theorem edges_bypass_subset {u v : V} (p : G.Walk u v) : p.bypass.edges ⊆ p.edges :=
  List.map_subset _ p.darts_bypass_subset


theorem darts_toPath_subset {u v : V} (p : G.Walk u v) : (p.toPath : G.Walk u v).darts ⊆ p.darts :=
  darts_bypass_subset _


theorem edges_toPath_subset {u v : V} (p : G.Walk u v) : (p.toPath : G.Walk u v).edges ⊆ p.edges :=
  edges_bypass_subset _


theorem map_isPath_of_injective (hinj : Function.Injective f) (hp : p.IsPath) :
    (p.map f).IsPath := by
  induction p with
  | nil => simp
  | cons _ _ ih =>
    rw [Walk.cons_isPath_iff] at hp
    simp only [map_cons, cons_isPath_iff, ih hp.1, support_map, List.mem_map, not_exists, not_and,
      true_and]
    intro x hx hf
    cases hinj hf
    exact hp.2 hx


protected theorem IsPath.of_map {f : G →g G'} (hp : (p.map f).IsPath) : p.IsPath := by
  induction p with
  | nil => simp
  | cons _ _ ih =>
    rw [map_cons, Walk.cons_isPath_iff, support_map] at hp
    rw [Walk.cons_isPath_iff]
    cases' hp with hp1 hp2
    refine ⟨ih hp1, ?_⟩
    contrapose! hp2
    exact List.mem_map_of_mem f hp2


theorem map_isPath_iff_of_injective (hinj : Function.Injective f) : (p.map f).IsPath ↔ p.IsPath :=
  ⟨IsPath.of_map, map_isPath_of_injective hinj⟩


theorem map_isTrail_iff_of_injective (hinj : Function.Injective f) :
    (p.map f).IsTrail ↔ p.IsTrail := by
  induction p with
  | nil => simp
  | cons _ _ ih =>
    rw [map_cons, cons_isTrail_iff, ih, cons_isTrail_iff]
    apply and_congr_right'
    rw [← Sym2.map_pair_eq, edges_map, ← List.mem_map_of_injective (Sym2.map.injective hinj)]


alias ⟨_, map_isTrail_of_injective⟩ := map_isTrail_iff_of_injective


theorem map_isCycle_iff_of_injective {p : G.Walk u u} (hinj : Function.Injective f) :
    (p.map f).IsCycle ↔ p.IsCycle := by
  rw [isCycle_def, isCycle_def, map_isTrail_iff_of_injective hinj, Ne, map_eq_nil_iff,
    support_map, ← List.map_tail, List.nodup_map_iff hinj]


alias ⟨_, IsCycle.map⟩ := map_isCycle_iff_of_injective


@[simp]
theorem mapLe_isTrail {G G' : SimpleGraph V} (h : G ≤ G') {u v : V} {p : G.Walk u v} :
    (p.mapLe h).IsTrail ↔ p.IsTrail :=
  map_isTrail_iff_of_injective Function.injective_id


alias ⟨IsTrail.of_mapLe, IsTrail.mapLe⟩ := mapLe_isTrail


@[simp]
theorem mapLe_isPath {G G' : SimpleGraph V} (h : G ≤ G') {u v : V} {p : G.Walk u v} :
    (p.mapLe h).IsPath ↔ p.IsPath :=
  map_isPath_iff_of_injective Function.injective_id


alias ⟨IsPath.of_mapLe, IsPath.mapLe⟩ := mapLe_isPath


@[simp]
theorem mapLe_isCycle {G G' : SimpleGraph V} (h : G ≤ G') {u : V} {p : G.Walk u u} :
    (p.mapLe h).IsCycle ↔ p.IsCycle :=
  map_isCycle_iff_of_injective Function.injective_id


alias ⟨IsCycle.of_mapLe, IsCycle.mapLe⟩ := mapLe_isCycle


/-- Given an injective graph homomorphism, map paths to paths. -/
@[simps]
protected def map (f : G →g G') (hinj : Function.Injective f) {u v : V} (p : G.Path u v) :
    G'.Path (f u) (f v) :=
  ⟨Walk.map f p, Walk.map_isPath_of_injective hinj p.2⟩


theorem map_injective {f : G →g G'} (hinj : Function.Injective f) (u v : V) :
    Function.Injective (Path.map f hinj : G.Path u v → G'.Path (f u) (f v)) := by
  /-
    V : Type u
    V' : Type v
    G : SimpleGraph V
    G' : SimpleGraph V'
    f : G.Hom G'
    hinj : Function.Injective ⇑f
    u v : V
    ⊢ Function.Injective (SimpleGraph.Path.map f hinj)
  -/
  rintro ⟨p, hp⟩ ⟨p', hp'⟩ h
  /-
    case mk.mk
    V : Type u
    V' : Type v
    G : SimpleGraph V
    G' : SimpleGraph V'
    f : G.Hom G'
    hinj : Function.Injective ⇑f
    u v : V
    p : G.Walk u v
    hp : p.IsPath
    p' : G.Walk u v
    hp' : p'.IsPath
    h : Eq (SimpleGraph.Path.map f hinj ⟨p, hp⟩) (SimpleGraph.Path.map f hinj ⟨p', …
    ⊢ Eq ⟨p, hp⟩ ⟨p', hp'⟩
  -/
  simp only [Path.map, Subtype.coe_mk, Subtype.mk.injEq] at h
  /-
    case mk.mk
    V : Type u
    V' : Type v
    G : SimpleGraph V
    G' : SimpleGraph V'
    f : G.Hom G'
    hinj : Function.Injective ⇑f
    u v : V
    p : G.Walk u v
    hp : p.IsPath
    p' : G.Walk u v
    hp' : p'.IsPath
    h : Eq (SimpleGraph.Walk.map f p) (SimpleGraph.Walk.map f p')
    ⊢ Eq ⟨p, hp⟩ ⟨p', hp'⟩
  -/
  simp [Walk.map_injective_of_injective hinj u v h]
  /-
    🎉 no goals
  -/


/-- Given a graph embedding, map paths to paths. -/
@[simps!]
protected def mapEmbedding (f : G ↪g G') {u v : V} (p : G.Path u v) : G'.Path (f u) (f v) :=
  Path.map f.toHom f.injective p


theorem mapEmbedding_injective (f : G ↪g G') (u v : V) :
    Function.Injective (Path.mapEmbedding f : G.Path u v → G'.Path (f u) (f v)) :=
  map_injective f.injective u v


protected theorem IsPath.transfer (hp) (pp : p.IsPath) :
    (p.transfer H hp).IsPath := by
  induction p with
  | nil => simp
  | cons _ _ ih =>
    simp only [Walk.transfer, cons_isPath_iff, support_transfer _ ] at pp ⊢
    exact ⟨ih _ pp.1, pp.2⟩


protected theorem IsCycle.transfer {q : G.Walk u u} (qc : q.IsCycle) (hq) :
    (q.transfer H hq).IsCycle := by
  cases q with
  | nil => simp at qc
  | cons _ q =>
    simp only [edges_cons, List.find?, List.mem_cons, forall_eq_or_imp, mem_edgeSet] at hq
    simp only [Walk.transfer, cons_isCycle_iff, edges_transfer q hq.2] at qc ⊢
    exact ⟨qc.1.transfer hq.2, qc.2⟩


protected theorem IsPath.toDeleteEdges (s : Set (Sym2 V))
    {p : G.Walk v w} (h : p.IsPath) (hp) : (p.toDeleteEdges s hp).IsPath :=
  h.transfer _


protected theorem IsCycle.toDeleteEdges (s : Set (Sym2 V))
    {p : G.Walk v v} (h : p.IsCycle) (hp) : (p.toDeleteEdges s hp).IsCycle :=
  h.transfer _


@[simp]
theorem toDeleteEdges_copy {v u u' v' : V} (s : Set (Sym2 V))
    (p : G.Walk u v) (hu : u = u') (hv : v = v') (h) :
    (p.copy hu hv).toDeleteEdges s h =
                             /-
                               V : Type u
                               V' : Type v
                               V'' : Type w
                               G : SimpleGraph V
                               G' : SimpleGraph V'
                               G'' : SimpleGraph V''
                               v✝ w v u u' v' : V
                               s : Set (Sym2 V)
                               p : G.Walk u v
                               hu : Eq u u'
                               hv : Eq v v'
                               h : ∀ (e : Sym2 V), Membership.mem (p.copy hu hv).edges e → Not (Membership.me …
                               ⊢ ∀ (e : Sym2 V), Membership.mem p.edges e → Not (Membership.mem s e)
                             -/
      (p.toDeleteEdges s (by subst_vars; exact h)).copy hu hv := by
                                         /-
                                           🎉 no goals
                                         -/
  /-
    V : Type u
    G : SimpleGraph V
    v u u' v' : V
    s : Set (Sym2 V)
    p : G.Walk u v
    hu : Eq u u'
    hv : Eq v v'
    h : ∀ (e : Sym2 V), Membership.mem (p.copy hu hv).edges e → Not (Membership.me …
    ⊢ Eq (SimpleGraph.Walk.toDeleteEdges s (p.copy hu hv) h) ((SimpleGraph.Walk.to …
  -/
  subst_vars
  /-
    V : Type u
    G : SimpleGraph V
    u' v' : V
    s : Set (Sym2 V)
    p : G.Walk u' v'
    h : ∀ (e : Sym2 V), Membership.mem (p.copy ⋯ ⋯).edges e → Not (Membership.mem  …
    ⊢ Eq (SimpleGraph.Walk.toDeleteEdges s (p.copy ⋯ ⋯) h) ((SimpleGraph.Walk.toDe …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Two vertices are *reachable* if there is a walk between them.
This is equivalent to `Relation.ReflTransGen` of `G.Adj`.
See `SimpleGraph.reachable_iff_reflTransGen`. -/
def Reachable (u v : V) : Prop := Nonempty (G.Walk u v)


theorem reachable_iff_nonempty_univ {u v : V} :
    G.Reachable u v ↔ (Set.univ : Set (G.Walk u v)).Nonempty :=
  Set.nonempty_iff_univ_nonempty


lemma not_reachable_iff_isEmpty_walk {u v : V} : ¬G.Reachable u v ↔ IsEmpty (G.Walk u v) :=
  not_nonempty_iff


protected theorem Reachable.elim {p : Prop} {u v : V} (h : G.Reachable u v)
    (hp : G.Walk u v → p) : p :=
  Nonempty.elim h hp


protected theorem Reachable.elim_path {p : Prop} {u v : V} (h : G.Reachable u v)
                                    /-
                                      V : Type u
                                      G : SimpleGraph V
                                      p : Prop
                                      u v : V
                                      h : G.Reachable u v
                                      hp : G.Path u v → p
                                      ⊢ p
                                    -/
    (hp : G.Path u v → p) : p := by classical exact h.elim fun q => hp q.toPath
                                    /-
                                      🎉 no goals
                                    -/


protected theorem Walk.reachable {G : SimpleGraph V} {u v : V} (p : G.Walk u v) : G.Reachable u v :=
  ⟨p⟩


protected theorem Adj.reachable {u v : V} (h : G.Adj u v) : G.Reachable u v :=
  h.toWalk.reachable


@[refl]
protected theorem Reachable.refl (u : V) : G.Reachable u u := ⟨Walk.nil⟩


protected theorem Reachable.rfl {u : V} : G.Reachable u u := Reachable.refl _


@[symm]
protected theorem Reachable.symm {u v : V} (huv : G.Reachable u v) : G.Reachable v u :=
  huv.elim fun p => ⟨p.reverse⟩


theorem reachable_comm {u v : V} : G.Reachable u v ↔ G.Reachable v u :=
  ⟨Reachable.symm, Reachable.symm⟩


@[trans]
protected theorem Reachable.trans {u v w : V} (huv : G.Reachable u v) (hvw : G.Reachable v w) :
    G.Reachable u w :=
  huv.elim fun puv => hvw.elim fun pvw => ⟨puv.append pvw⟩


theorem reachable_iff_reflTransGen (u v : V) :
    G.Reachable u v ↔ Relation.ReflTransGen G.Adj u v := by
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    ⊢ Iff (G.Reachable u v) (Relation.ReflTransGen G.Adj u v)
  -/
  constructor
    /-
      case mp
      V : Type u
      G : SimpleGraph V
      u v : V
      ⊢ G.Reachable u v → Relation.ReflTransGen G.Adj u v
    -/
  · rintro ⟨h⟩
    induction h with
    | nil => rfl
    | cons h' _ ih => exact (Relation.ReflTransGen.single h').trans ih
    /-
      case mpr
      V : Type u
      G : SimpleGraph V
      u v : V
      ⊢ Relation.ReflTransGen G.Adj u v → G.Reachable u v
    -/
  · intro h
    induction h with
    | refl => rfl
    | tail _ ha hr => exact Reachable.trans hr ⟨Walk.cons ha Walk.nil⟩


protected theorem Reachable.map {u v : V} {G : SimpleGraph V} {G' : SimpleGraph V'} (f : G →g G')
    (h : G.Reachable u v) : G'.Reachable (f u) (f v) :=
  h.elim fun p => ⟨p.map f⟩


@[mono]
protected lemma Reachable.mono {u v : V} {G G' : SimpleGraph V}
    (h : G ≤ G') (Guv : G.Reachable u v) : G'.Reachable u v :=
  Guv.map (SimpleGraph.Hom.mapSpanningSubgraphs h)


theorem Iso.reachable_iff {G : SimpleGraph V} {G' : SimpleGraph V'} {φ : G ≃g G'} {u v : V} :
    G'.Reachable (φ u) (φ v) ↔ G.Reachable u v :=
  ⟨fun r => φ.left_inv u ▸ φ.left_inv v ▸ r.map φ.symm.toHom, Reachable.map φ.toHom⟩


theorem Iso.symm_apply_reachable {G : SimpleGraph V} {G' : SimpleGraph V'} {φ : G ≃g G'} {u : V}
    {v : V'} : G.Reachable (φ.symm v) u ↔ G'.Reachable v (φ u) := by
  /-
    V : Type u
    V' : Type v
    G : SimpleGraph V
    G' : SimpleGraph V'
    φ : G.Iso G'
    u : V
    v : V'
    ⊢ Iff (G.Reachable (φ.symm v) u) (G'.Reachable v (φ u))
  -/
  rw [← Iso.reachable_iff, RelIso.apply_symm_apply]
  /-
    🎉 no goals
  -/


theorem reachable_is_equivalence : Equivalence G.Reachable :=
  Equivalence.mk (@Reachable.refl _ G) (@Reachable.symm _ G) (@Reachable.trans _ G)


/-- Distinct vertices are not reachable in the empty graph. -/
@[simp]
lemma reachable_bot {u v : V} : (⊥ : SimpleGraph V).Reachable u v ↔ u = v :=
  ⟨fun h ↦ h.elim fun p ↦ match p with | .nil => rfl, fun h ↦ h ▸ .rfl⟩


/-- The equivalence relation on vertices given by `SimpleGraph.Reachable`. -/
def reachableSetoid : Setoid V := Setoid.mk _ G.reachable_is_equivalence


/-- A graph is preconnected if every pair of vertices is reachable from one another. -/
def Preconnected : Prop := ∀ u v : V, G.Reachable u v


theorem Preconnected.map {G : SimpleGraph V} {H : SimpleGraph V'} (f : G →g H) (hf : Surjective f)
    (hG : G.Preconnected) : H.Preconnected :=
  hf.forall₂.2 fun _ _ => Nonempty.map (Walk.map _) <| hG _ _


@[mono]
protected lemma Preconnected.mono  {G G' : SimpleGraph V} (h : G ≤ G') (hG : G.Preconnected) :
    G'.Preconnected := fun u v => (hG u v).mono h


lemma bot_preconnected_iff_subsingleton : (⊥ : SimpleGraph V).Preconnected ↔ Subsingleton V := by
  /-
    V : Type u
    ⊢ Iff Bot.bot.Preconnected (Subsingleton V)
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ by simpa [subsingleton_iff, ← reachable_bot] using h⟩
  /-
    V : Type u
    h : Bot.bot.Preconnected
    ⊢ Subsingleton V
  -/
  contrapose h
  /-
    V : Type u
    h : Not (Subsingleton V)
    ⊢ Not Bot.bot.Preconnected
  -/
  simp [nontrivial_iff.mp <| not_subsingleton_iff_nontrivial.mp h, Preconnected, reachable_bot, h]
  /-
    🎉 no goals
  -/


lemma bot_preconnected [Subsingleton V] : (⊥ : SimpleGraph V).Preconnected :=
  bot_preconnected_iff_subsingleton.mpr ‹_›


lemma bot_not_preconnected [Nontrivial V] : ¬(⊥ : SimpleGraph V).Preconnected :=
  bot_preconnected_iff_subsingleton.not.mpr <| not_subsingleton_iff_nontrivial.mpr ‹_›


lemma top_preconnected : (⊤ : SimpleGraph V).Preconnected := fun x y => by
  /-
    V : Type u
    x y : V
    ⊢ Top.top.Reachable x y
  -/
  if h : x = y then rw [h] else exact Adj.reachable h
  /-
    🎉 no goals
  -/


theorem Iso.preconnected_iff {G : SimpleGraph V} {H : SimpleGraph V'} (e : G ≃g H) :
    G.Preconnected ↔ H.Preconnected :=
  ⟨Preconnected.map e.toHom e.toEquiv.surjective,
    Preconnected.map e.symm.toHom e.symm.toEquiv.surjective⟩


/-- A graph is connected if it's preconnected and contains at least one vertex.
This follows the convention observed by mathlib that something is connected iff it has
exactly one connected component.

There is a `CoeFun` instance so that `h u v` can be used instead of `h.Preconnected u v`. -/
@[mk_iff]
structure Connected : Prop where
  protected preconnected : G.Preconnected
  protected [nonempty : Nonempty V]


lemma connected_iff_exists_forall_reachable : G.Connected ↔ ∃ v, ∀ w, G.Reachable v w := by
  /-
    V : Type u
    G : SimpleGraph V
    ⊢ Iff G.Connected (Exists fun v => ∀ (w : V), G.Reachable v w)
  -/
  rw [connected_iff]
  /-
    V : Type u
    G : SimpleGraph V
    ⊢ Iff (And G.Preconnected (Nonempty V)) (Exists fun v => ∀ (w : V), G.Reachabl …
  -/
  constructor
    /-
      case mp
      V : Type u
      G : SimpleGraph V
      ⊢ And G.Preconnected (Nonempty V) → Exists fun v => ∀ (w : V), G.Reachable v w
    -/
  · rintro ⟨hp, ⟨v⟩⟩
    /-
      case mp.intro.intro
      V : Type u
      G : SimpleGraph V
      hp : G.Preconnected
      v : V
      ⊢ Exists fun v => ∀ (w : V), G.Reachable v w
    -/
    exact ⟨v, fun w => hp v w⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      V : Type u
      G : SimpleGraph V
      ⊢ (Exists fun v => ∀ (w : V), G.Reachable v w) → And G.Preconnected (Nonempty V)
    -/
  · rintro ⟨v, h⟩
    /-
      case mpr.intro
      V : Type u
      G : SimpleGraph V
      v : V
      h : ∀ (w : V), G.Reachable v w
      ⊢ And G.Preconnected (Nonempty V)
    -/
    exact ⟨fun u w => (h u).symm.trans (h w), ⟨v⟩⟩
    /-
      🎉 no goals
    -/


instance : CoeFun G.Connected fun _ => ∀ u v : V, G.Reachable u v := ⟨fun h => h.preconnected⟩


theorem Connected.map {G : SimpleGraph V} {H : SimpleGraph V'} (f : G →g H) (hf : Surjective f)
    (hG : G.Connected) : H.Connected :=
  haveI := hG.nonempty.map f
  ⟨hG.preconnected.map f hf⟩


@[mono]
protected lemma Connected.mono {G G' : SimpleGraph V} (h : G ≤ G')
    (hG : G.Connected) : G'.Connected where
  preconnected := hG.preconnected.mono h
  nonempty := hG.nonempty


lemma bot_not_connected [Nontrivial V] : ¬(⊥ : SimpleGraph V).Connected := by
  /-
    V : Type u
    inst✝ : Nontrivial V
    ⊢ Not Bot.bot.Connected
  -/
  simp [bot_not_preconnected, connected_iff, ‹_›]
  /-
    🎉 no goals
  -/


lemma top_connected [Nonempty V] : (⊤ : SimpleGraph V).Connected where
  preconnected := top_preconnected


theorem Iso.connected_iff {G : SimpleGraph V} {H : SimpleGraph V'} (e : G ≃g H) :
    G.Connected ↔ H.Connected :=
  ⟨Connected.map e.toHom e.toEquiv.surjective, Connected.map e.symm.toHom e.symm.toEquiv.surjective⟩


/-- The quotient of `V` by the `SimpleGraph.Reachable` relation gives the connected
components of a graph. -/
def ConnectedComponent := Quot G.Reachable


/-- Gives the connected component containing a particular vertex. -/
def connectedComponentMk (v : V) : G.ConnectedComponent := Quot.mk G.Reachable v


@[simps]
instance inhabited [Inhabited V] : Inhabited G.ConnectedComponent :=
  ⟨G.connectedComponentMk default⟩


instance isEmpty [IsEmpty V] : IsEmpty (ConnectedComponent G) := by
  /-
    V : Type u
    V' : Type v
    V'' : Type w
    G : SimpleGraph V
    G' : SimpleGraph V'
    G'' : SimpleGraph V''
    inst✝ : IsEmpty V
    ⊢ IsEmpty G.ConnectedComponent
  -/
  by_contra! hc
  /-
    V : Type u
    V' : Type v
    V'' : Type w
    G : SimpleGraph V
    G' : SimpleGraph V'
    G'' : SimpleGraph V''
    inst✝ : IsEmpty V
    hc : Not (IsEmpty G.ConnectedComponent)
    ⊢ False
  -/
  rw [@not_isEmpty_iff] at hc
  /-
    V : Type u
    V' : Type v
    V'' : Type w
    G : SimpleGraph V
    G' : SimpleGraph V'
    G'' : SimpleGraph V''
    inst✝ : IsEmpty V
    hc : Nonempty G.ConnectedComponent
    ⊢ False
  -/
  obtain ⟨v, _⟩ := (Classical.inhabited_of_nonempty hc).default.exists_rep
  /-
    case intro
    V : Type u
    V' : Type v
    V'' : Type w
    G : SimpleGraph V
    G' : SimpleGraph V'
    G'' : SimpleGraph V''
    inst✝ : IsEmpty V
    hc : Nonempty G.ConnectedComponent
    v : V
    h✝ : Eq (Quot.mk G.Reachable v) Inhabited.default
    ⊢ False
  -/
  exact IsEmpty.false v
  /-
    🎉 no goals
  -/


@[elab_as_elim]
protected theorem ind {β : G.ConnectedComponent → Prop}
    (h : ∀ v : V, β (G.connectedComponentMk v)) (c : G.ConnectedComponent) : β c :=
  Quot.ind h c


@[elab_as_elim]
protected theorem ind₂ {β : G.ConnectedComponent → G.ConnectedComponent → Prop}
    (h : ∀ v w : V, β (G.connectedComponentMk v) (G.connectedComponentMk w))
    (c d : G.ConnectedComponent) : β c d :=
  Quot.induction_on₂ c d h


protected theorem sound {v w : V} :
    G.Reachable v w → G.connectedComponentMk v = G.connectedComponentMk w :=
  Quot.sound


protected theorem exact {v w : V} :
    G.connectedComponentMk v = G.connectedComponentMk w → G.Reachable v w :=
  @Quotient.exact _ G.reachableSetoid _ _


@[simp]
protected theorem eq {v w : V} :
    G.connectedComponentMk v = G.connectedComponentMk w ↔ G.Reachable v w :=
  @Quotient.eq' _ G.reachableSetoid _ _


theorem connectedComponentMk_eq_of_adj {v w : V} (a : G.Adj v w) :
    G.connectedComponentMk v = G.connectedComponentMk w :=
  ConnectedComponent.sound a.reachable


/-- The `ConnectedComponent` specialization of `Quot.lift`. Provides the stronger
assumption that the vertices are connected by a path. -/
protected def lift {β : Sort*} (f : V → β)
    (h : ∀ (v w : V) (p : G.Walk v w), p.IsPath → f v = f w) : G.ConnectedComponent → β :=
  Quot.lift f fun v w (h' : G.Reachable v w) => h'.elim_path fun hp => h v w hp hp.2


@[simp]
protected theorem lift_mk {β : Sort*} {f : V → β}
    {h : ∀ (v w : V) (p : G.Walk v w), p.IsPath → f v = f w} {v : V} :
    ConnectedComponent.lift f h (G.connectedComponentMk v) = f v :=
  rfl


protected theorem «exists» {p : G.ConnectedComponent → Prop} :
    (∃ c : G.ConnectedComponent, p c) ↔ ∃ v, p (G.connectedComponentMk v) :=
  Quot.mk_surjective.exists


protected theorem «forall» {p : G.ConnectedComponent → Prop} :
    (∀ c : G.ConnectedComponent, p c) ↔ ∀ v, p (G.connectedComponentMk v) :=
  Quot.mk_surjective.forall


theorem _root_.SimpleGraph.Preconnected.subsingleton_connectedComponent (h : G.Preconnected) :
    Subsingleton G.ConnectedComponent :=
  ⟨ConnectedComponent.ind₂ fun v w => ConnectedComponent.sound (h v w)⟩


/-- This is `Quot.recOn` specialized to connected components.
For convenience, it strengthens the assumptions in the hypothesis
to provide a path between the vertices. -/
@[elab_as_elim]
def recOn
    {motive : G.ConnectedComponent → Sort*}
    (c : G.ConnectedComponent)
    (f : (v : V) → motive (G.connectedComponentMk v))
    (h : ∀ (u v : V) (p : G.Walk u v) (_ : p.IsPath),
      ConnectedComponent.sound p.reachable ▸ f u = f v) :
    motive c :=
  Quot.recOn c f fun u v r => r.elim_path fun p => h u v p p.2


/-- The map on connected components induced by a graph homomorphism. -/
def map (φ : G →g G') (C : G.ConnectedComponent) : G'.ConnectedComponent :=
  C.lift (fun v => G'.connectedComponentMk (φ v)) fun _ _ p _ =>
    ConnectedComponent.eq.mpr (p.map φ).reachable


@[simp]
theorem map_mk (φ : G →g G') (v : V) :
    (G.connectedComponentMk v).map φ = G'.connectedComponentMk (φ v) :=
  rfl


@[simp]
theorem map_id (C : ConnectedComponent G) : C.map Hom.id = C := by
  /-
    V : Type u
    G : SimpleGraph V
    C : G.ConnectedComponent
    ⊢ Eq (SimpleGraph.ConnectedComponent.map SimpleGraph.Hom.id C) C
  -/
  refine C.ind ?_
  /-
    V : Type u
    G : SimpleGraph V
    C : G.ConnectedComponent
    ⊢ ∀ (v : V), Eq (SimpleGraph.ConnectedComponent.map SimpleGraph.Hom.id (G.conn …
  -/
  exact fun _ => rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem map_comp (C : G.ConnectedComponent) (φ : G →g G') (ψ : G' →g G'') :
    (C.map φ).map ψ = C.map (ψ.comp φ) := by
  /-
    V : Type u
    V' : Type v
    V'' : Type w
    G : SimpleGraph V
    G' : SimpleGraph V'
    G'' : SimpleGraph V''
    C : G.ConnectedComponent
    φ : G.Hom G'
    ψ : G'.Hom G''
    ⊢ Eq (SimpleGraph.ConnectedComponent.map ψ (SimpleGraph.ConnectedComponent.map …
  -/
  refine C.ind ?_
  /-
    V : Type u
    V' : Type v
    V'' : Type w
    G : SimpleGraph V
    G' : SimpleGraph V'
    G'' : SimpleGraph V''
    C : G.ConnectedComponent
    φ : G.Hom G'
    ψ : G'.Hom G''
    ⊢ ∀ (v : V), Eq (SimpleGraph.ConnectedComponent.map ψ (SimpleGraph.ConnectedCo …
  -/
  exact fun _ => rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem iso_image_comp_eq_map_iff_eq_comp {C : G.ConnectedComponent} :
    G'.connectedComponentMk (φ v) = C.map ↑(↑φ : G ↪g G') ↔ G.connectedComponentMk v = C := by
  /-
    V : Type u
    V' : Type v
    G : SimpleGraph V
    G' : SimpleGraph V'
    φ : G.Iso G'
    v : V
    C : G.ConnectedComponent
    ⊢ Iff (Eq (G'.connectedComponentMk (φ v)) (SimpleGraph.ConnectedComponent.map  …
  -/
  refine C.ind fun u => ?_
  simp only [Iso.reachable_iff, ConnectedComponent.map_mk, RelEmbedding.coe_toRelHom,
    RelIso.coe_toRelEmbedding, ConnectedComponent.eq]


@[simp]
theorem iso_inv_image_comp_eq_iff_eq_map {C : G.ConnectedComponent} :
    G.connectedComponentMk (φ.symm v') = C ↔ G'.connectedComponentMk v' = C.map φ := by
  /-
    V : Type u
    V' : Type v
    G : SimpleGraph V
    G' : SimpleGraph V'
    φ : G.Iso G'
    v' : V'
    C : G.ConnectedComponent
    ⊢ Iff (Eq (G.connectedComponentMk (φ.symm v')) C) (Eq (G'.connectedComponentMk …
  -/
  refine C.ind fun u => ?_
  simp only [Iso.symm_apply_reachable, ConnectedComponent.eq, ConnectedComponent.map_mk,
    RelEmbedding.coe_toRelHom, RelIso.coe_toRelEmbedding]


/-- An isomorphism of graphs induces a bijection of connected components. -/
@[simps]
def connectedComponentEquiv (φ : G ≃g G') : G.ConnectedComponent ≃ G'.ConnectedComponent where
  toFun := ConnectedComponent.map φ
  invFun := ConnectedComponent.map φ.symm
  left_inv C := ConnectedComponent.ind
    (fun v => congr_arg G.connectedComponentMk (Equiv.left_inv φ.toEquiv v)) C
  right_inv C := ConnectedComponent.ind
    (fun v => congr_arg G'.connectedComponentMk (Equiv.right_inv φ.toEquiv v)) C


@[simp]
theorem connectedComponentEquiv_refl :
    (Iso.refl : G ≃g G).connectedComponentEquiv = Equiv.refl _ := by
  /-
    V : Type u
    G : SimpleGraph V
    ⊢ Eq SimpleGraph.Iso.refl.connectedComponentEquiv (Equiv.refl G.ConnectedCompo …
  -/
  ext ⟨v⟩
  /-
    case H.mk
    V : Type u
    G : SimpleGraph V
    x✝ : G.ConnectedComponent
    v : V
    ⊢ Eq (SimpleGraph.Iso.refl.connectedComponentEquiv (Quot.mk G.Reachable v)) (( …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem connectedComponentEquiv_symm (φ : G ≃g G') :
    φ.symm.connectedComponentEquiv = φ.connectedComponentEquiv.symm := by
  /-
    V : Type u
    V' : Type v
    G : SimpleGraph V
    G' : SimpleGraph V'
    φ : G.Iso G'
    ⊢ Eq φ.symm.connectedComponentEquiv φ.connectedComponentEquiv.symm
  -/
  ext ⟨_⟩
  /-
    case H.mk
    V : Type u
    V' : Type v
    G : SimpleGraph V
    G' : SimpleGraph V'
    φ : G.Iso G'
    x✝ : G'.ConnectedComponent
    a✝ : V'
    ⊢ Eq (φ.symm.connectedComponentEquiv (Quot.mk G'.Reachable a✝)) (φ.connectedCo …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem connectedComponentEquiv_trans (φ : G ≃g G') (φ' : G' ≃g G'') :
    connectedComponentEquiv (φ.trans φ') =
    φ.connectedComponentEquiv.trans φ'.connectedComponentEquiv := by
  /-
    V : Type u
    V' : Type v
    V'' : Type w
    G : SimpleGraph V
    G' : SimpleGraph V'
    G'' : SimpleGraph V''
    φ : G.Iso G'
    φ' : G'.Iso G''
    ⊢ Eq (SimpleGraph.Iso.connectedComponentEquiv (RelIso.trans φ φ')) (φ.connecte …
  -/
  ext ⟨_⟩
  /-
    case H.mk
    V : Type u
    V' : Type v
    V'' : Type w
    G : SimpleGraph V
    G' : SimpleGraph V'
    G'' : SimpleGraph V''
    φ : G.Iso G'
    φ' : G'.Iso G''
    x✝ : G.ConnectedComponent
    a✝ : V
    ⊢ Eq ((SimpleGraph.Iso.connectedComponentEquiv (RelIso.trans φ φ')) (Quot.mk G …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The set of vertices in a connected component of a graph. -/
def supp (C : G.ConnectedComponent) :=
  { v | G.connectedComponentMk v = C }


@[ext]
theorem supp_injective :
    Function.Injective (ConnectedComponent.supp : G.ConnectedComponent → Set V) := by
  /-
    V : Type u
    G : SimpleGraph V
    ⊢ Function.Injective SimpleGraph.ConnectedComponent.supp
  -/
  refine ConnectedComponent.ind₂ ?_
  /-
    V : Type u
    G : SimpleGraph V
    ⊢ ∀ (v w : V), Eq (G.connectedComponentMk v).supp (G.connectedComponentMk w).s …
  -/
  intro v w
  /-
    V : Type u
    G : SimpleGraph V
    v w : V
    ⊢ Eq (G.connectedComponentMk v).supp (G.connectedComponentMk w).supp → Eq (G.c …
  -/
  simp only [ConnectedComponent.supp, Set.ext_iff, ConnectedComponent.eq, Set.mem_setOf_eq]
  /-
    V : Type u
    G : SimpleGraph V
    v w : V
    ⊢ (∀ (x : V), Iff (G.Reachable x v) (G.Reachable x w)) → G.Reachable v w
  -/
  intro h
  /-
    V : Type u
    G : SimpleGraph V
    v w : V
    h : ∀ (x : V), Iff (G.Reachable x v) (G.Reachable x w)
    ⊢ G.Reachable v w
  -/
  rw [reachable_comm, h]
  /-
    🎉 no goals
  -/


@[simp]
theorem supp_inj {C D : G.ConnectedComponent} : C.supp = D.supp ↔ C = D :=
  ConnectedComponent.supp_injective.eq_iff


instance : SetLike G.ConnectedComponent V where
  coe := ConnectedComponent.supp
  coe_injective' := ConnectedComponent.supp_injective


@[simp]
theorem mem_supp_iff (C : G.ConnectedComponent) (v : V) :
    v ∈ C.supp ↔ G.connectedComponentMk v = C :=
  Iff.rfl


theorem connectedComponentMk_mem {v : V} : v ∈ G.connectedComponentMk v :=
  rfl


/-- The equivalence between connected components, induced by an isomorphism of graphs,
itself defines an equivalence on the supports of each connected component.
-/
def isoEquivSupp (φ : G ≃g G') (C : G.ConnectedComponent) :
    C.supp ≃ (φ.connectedComponentEquiv C).supp where
  toFun v := ⟨φ v, ConnectedComponent.iso_image_comp_eq_map_iff_eq_comp.mpr v.prop⟩
  invFun v' := ⟨φ.symm v', ConnectedComponent.iso_inv_image_comp_eq_iff_eq_map.mpr v'.prop⟩
  left_inv v := Subtype.ext_val (φ.toEquiv.left_inv ↑v)
  right_inv v := Subtype.ext_val (φ.toEquiv.right_inv ↑v)


lemma mem_coe_supp_of_adj {v w : V} {H : Subgraph G} {c : ConnectedComponent H.coe}
    (hv : v ∈ (↑) '' (c : Set H.verts)) (hw : w ∈ H.verts)
    (hadj : H.Adj v w) : w ∈ (↑) '' (c : Set H.verts):= by
  /-
    V : Type u
    G : SimpleGraph V
    v w : V
    H : G.Subgraph
    c : H.coe.ConnectedComponent
    hv : Membership.mem (Set.image Subtype.val ↑c) v
    hw : Membership.mem H.verts w
    hadj : H.Adj v w
    ⊢ Membership.mem (Set.image Subtype.val ↑c) w
  -/
  obtain ⟨_, h⟩ := hv
  /-
    case intro
    V : Type u
    G : SimpleGraph V
    v w : V
    H : G.Subgraph
    c : H.coe.ConnectedComponent
    hw : Membership.mem H.verts w
    hadj : H.Adj v w
    w✝ : ↑H.verts
    h : And (Membership.mem (↑c) w✝) (Eq (↑w✝) v)
    ⊢ Membership.mem (Set.image Subtype.val ↑c) w
  -/
  use ⟨w, hw⟩
  /-
    case h
    V : Type u
    G : SimpleGraph V
    v w : V
    H : G.Subgraph
    c : H.coe.ConnectedComponent
    hw : Membership.mem H.verts w
    hadj : H.Adj v w
    w✝ : ↑H.verts
    h : And (Membership.mem (↑c) w✝) (Eq (↑w✝) v)
    ⊢ And (Membership.mem ↑c ⟨w, hw⟩) (Eq (↑⟨w, hw⟩) w)
  -/
  rw [← (mem_supp_iff _ _).mp h.1]
  /-
    case h
    V : Type u
    G : SimpleGraph V
    v w : V
    H : G.Subgraph
    c : H.coe.ConnectedComponent
    hw : Membership.mem H.verts w
    hadj : H.Adj v w
    w✝ : ↑H.verts
    h : And (Membership.mem (↑c) w✝) (Eq (↑w✝) v)
    ⊢ And (Membership.mem ↑(H.coe.connectedComponentMk w✝) ⟨w, hw⟩) (Eq (↑⟨w, hw⟩) …
  -/
  exact ⟨connectedComponentMk_eq_of_adj <| Subgraph.Adj.coe <| h.2 ▸ hadj.symm, rfl⟩
  /-
    🎉 no goals
  -/


lemma connectedComponentMk_supp_subset_supp {G'} {v : V} (h : G ≤ G') (c' : G'.ConnectedComponent)
    (hc' : v ∈ c'.supp) : (G.connectedComponentMk v).supp ⊆ c'.supp := by
  /-
    V : Type u
    G G' : SimpleGraph V
    v : V
    h : LE.le G G'
    c' : G'.ConnectedComponent
    hc' : Membership.mem c'.supp v
    ⊢ HasSubset.Subset (G.connectedComponentMk v).supp c'.supp
  -/
  intro v' hv'
  /-
    V : Type u
    G G' : SimpleGraph V
    v : V
    h : LE.le G G'
    c' : G'.ConnectedComponent
    hc' : Membership.mem c'.supp v
    v' : V
    hv' : Membership.mem (G.connectedComponentMk v).supp v'
    ⊢ Membership.mem c'.supp v'
  -/
  simp only [mem_supp_iff, ConnectedComponent.eq] at hv' ⊢
  /-
    V : Type u
    G G' : SimpleGraph V
    v : V
    h : LE.le G G'
    c' : G'.ConnectedComponent
    hc' : Membership.mem c'.supp v
    v' : V
    hv' : G.Reachable v' v
    ⊢ Eq (G'.connectedComponentMk v') c'
  -/
  rw [ConnectedComponent.sound (hv'.mono h)]
  /-
    V : Type u
    G G' : SimpleGraph V
    v : V
    h : LE.le G G'
    c' : G'.ConnectedComponent
    hc' : Membership.mem c'.supp v
    v' : V
    hv' : G.Reachable v' v
    ⊢ Eq (G'.connectedComponentMk v) c'
  -/
  exact hc'
  /-
    🎉 no goals
  -/


lemma biUnion_supp_eq_supp {G G' : SimpleGraph V} (h : G ≤ G') (c' : ConnectedComponent G') :
    ⋃ (c : ConnectedComponent G) (_ : c.supp ⊆ c'.supp), c.supp = c'.supp := by
  /-
    V : Type u
    G G' : SimpleGraph V
    h : LE.le G G'
    c' : G'.ConnectedComponent
    ⊢ Eq (Set.iUnion fun c => Set.iUnion fun x => c.supp) c'.supp
  -/
  ext v
  /-
    case h
    V : Type u
    G G' : SimpleGraph V
    h : LE.le G G'
    c' : G'.ConnectedComponent
    v : V
    ⊢ Iff (Membership.mem (Set.iUnion fun c => Set.iUnion fun x => c.supp) v) (Mem …
  -/
  simp_rw [Set.mem_iUnion]
  /-
    case h
    V : Type u
    G G' : SimpleGraph V
    h : LE.le G G'
    c' : G'.ConnectedComponent
    v : V
    ⊢ Iff (Exists fun i => Exists fun i_1 => Membership.mem i.supp v) (Membership. …
  -/
  refine ⟨fun ⟨_, ⟨hi, hi'⟩⟩ ↦ hi hi', ?_⟩
  /-
    case h
    V : Type u
    G G' : SimpleGraph V
    h : LE.le G G'
    c' : G'.ConnectedComponent
    v : V
    ⊢ Membership.mem c'.supp v → Exists fun i => Exists fun i_1 => Membership.mem  …
  -/
  intro hv
  /-
    case h
    V : Type u
    G G' : SimpleGraph V
    h : LE.le G G'
    c' : G'.ConnectedComponent
    v : V
    hv : Membership.mem c'.supp v
    ⊢ Exists fun i => Exists fun i_1 => Membership.mem i.supp v
  -/
  use G.connectedComponentMk v
  /-
    case h
    V : Type u
    G G' : SimpleGraph V
    h : LE.le G G'
    c' : G'.ConnectedComponent
    v : V
    hv : Membership.mem c'.supp v
    ⊢ Exists fun i => Membership.mem (G.connectedComponentMk v).supp v
  -/
  use c'.connectedComponentMk_supp_subset_supp h hv
  /-
    case h
    V : Type u
    G G' : SimpleGraph V
    h : LE.le G G'
    c' : G'.ConnectedComponent
    v : V
    hv : Membership.mem c'.supp v
    ⊢ Membership.mem (G.connectedComponentMk v).supp v
  -/
  simp only [mem_supp_iff]
  /-
    🎉 no goals
  -/


lemma top_supp_eq_univ (c : ConnectedComponent (⊤ : SimpleGraph V)) :
    c.supp = (Set.univ : Set V) := by
  /-
    V : Type u
    c : Top.top.ConnectedComponent
    ⊢ Eq c.supp Set.univ
  -/
  have ⟨w, hw⟩ := c.exists_rep
  /-
    V : Type u
    c : Top.top.ConnectedComponent
    w : V
    hw : Eq (Quot.mk Top.top.Reachable w) c
    ⊢ Eq c.supp Set.univ
  -/
  ext v
  /-
    case h
    V : Type u
    c : Top.top.ConnectedComponent
    w : V
    hw : Eq (Quot.mk Top.top.Reachable w) c
    v : V
    ⊢ Iff (Membership.mem c.supp v) (Membership.mem Set.univ v)
  -/
  simp only [Set.mem_univ, iff_true, mem_supp_iff, ← hw]
  /-
    case h
    V : Type u
    c : Top.top.ConnectedComponent
    w : V
    hw : Eq (Quot.mk Top.top.Reachable w) c
    v : V
    ⊢ Eq (Top.top.connectedComponentMk v) (Quot.mk Top.top.Reachable w)
  -/
  apply SimpleGraph.ConnectedComponent.sound
  /-
    case h.a
    V : Type u
    c : Top.top.ConnectedComponent
    w : V
    hw : Eq (Quot.mk Top.top.Reachable w) c
    v : V
    ⊢ Top.top.Reachable v w
  -/
  exact (@SimpleGraph.top_connected V (Nonempty.intro v)).preconnected v w
  /-
    🎉 no goals
  -/


lemma pairwise_disjoint_supp_connectedComponent (G : SimpleGraph V) :
    Pairwise fun c c' : ConnectedComponent G ↦ Disjoint c.supp c'.supp := by
  /-
    V : Type u
    G : SimpleGraph V
    ⊢ Pairwise fun c c' => Disjoint c.supp c'.supp
  -/
  simp_rw [Set.disjoint_left]
  /-
    V : Type u
    G : SimpleGraph V
    ⊢ Pairwise fun c c' => ∀ ⦃a : V⦄, Membership.mem c.supp a → Not (Membership.me …
  -/
  intro _ _ h a hsx hsy
  /-
    V : Type u
    G : SimpleGraph V
    i✝ j✝ : G.ConnectedComponent
    h : Ne i✝ j✝
    a : V
    hsx : Membership.mem i✝.supp a
    hsy : Membership.mem j✝.supp a
    ⊢ False
  -/
  rw [ConnectedComponent.mem_supp_iff] at hsx hsy
  /-
    V : Type u
    G : SimpleGraph V
    i✝ j✝ : G.ConnectedComponent
    h : Ne i✝ j✝
    a : V
    hsx : Eq (G.connectedComponentMk a) i✝
    hsy : Eq (G.connectedComponentMk a) j✝
    ⊢ False
  -/
  rw [hsx] at hsy
  /-
    V : Type u
    G : SimpleGraph V
    i✝ j✝ : G.ConnectedComponent
    h : Ne i✝ j✝
    a : V
    hsx : Eq (G.connectedComponentMk a) i✝
    hsy : Eq i✝ j✝
    ⊢ False
  -/
  exact h hsy
  /-
    🎉 no goals
  -/

-- TODO: Extract as lemma about general equivalence relation

lemma iUnion_connectedComponentSupp (G : SimpleGraph V) :
    ⋃ c : G.ConnectedComponent, c.supp = Set.univ := by
  /-
    V : Type u
    G : SimpleGraph V
    ⊢ Eq (Set.iUnion fun c => c.supp) Set.univ
  -/
  refine Set.eq_univ_of_forall fun v ↦ ⟨G.connectedComponentMk v, ?_⟩
  /-
    V : Type u
    G : SimpleGraph V
    v : V
    ⊢ And (Membership.mem (Set.range fun c => c.supp) ↑(G.connectedComponentMk v)) …
  -/
  simp only [Set.mem_range, SetLike.mem_coe]
  /-
    V : Type u
    G : SimpleGraph V
    v : V
    ⊢ And (Exists fun y => Eq y.supp ↑(G.connectedComponentMk v)) (Membership.mem  …
  -/
  exact ⟨by use G.connectedComponentMk v; exact rfl, rfl⟩
  /-
    🎉 no goals
  -/


theorem Preconnected.set_univ_walk_nonempty (hconn : G.Preconnected) (u v : V) :
    (Set.univ : Set (G.Walk u v)).Nonempty := by
  /-
    V : Type u
    G : SimpleGraph V
    hconn : G.Preconnected
    u v : V
    ⊢ Set.univ.Nonempty
  -/
  rw [← Set.nonempty_iff_univ_nonempty]
  /-
    V : Type u
    G : SimpleGraph V
    hconn : G.Preconnected
    u v : V
    ⊢ Nonempty (G.Walk u v)
  -/
  exact hconn u v
  /-
    🎉 no goals
  -/


theorem Connected.set_univ_walk_nonempty (hconn : G.Connected) (u v : V) :
    (Set.univ : Set (G.Walk u v)).Nonempty :=
  hconn.preconnected.set_univ_walk_nonempty u v


/-- An edge of a graph is a *bridge* if, after removing it, its incident vertices
are no longer reachable from one another. -/
def IsBridge (G : SimpleGraph V) (e : Sym2 V) : Prop :=
  e ∈ G.edgeSet ∧
                                                                   /-
                                                                     V : Type u
                                                                     V' : Type v
                                                                     V'' : Type w
                                                                     G✝ : SimpleGraph V
                                                                     G' : SimpleGraph V'
                                                                     G'' : SimpleGraph V''
                                                                     G : SimpleGraph V
                                                                     e : Sym2 V
                                                                     ⊢ ∀ (a₁ a₂ : V), Eq ((fun v w => Not ((SDiff.sdiff G (SimpleGraph.fromEdgeSet  …
                                                                   -/
    Sym2.lift ⟨fun v w => ¬(G \ fromEdgeSet {e}).Reachable v w, by simp [reachable_comm]⟩ e
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


theorem isBridge_iff {u v : V} :
    G.IsBridge s(u, v) ↔ G.Adj u v ∧ ¬(G \ fromEdgeSet {s(u, v)}).Reachable u v := Iff.rfl


theorem reachable_delete_edges_iff_exists_walk {v w : V} :
    (G \ fromEdgeSet {s(v, w)}).Reachable v w ↔ ∃ p : G.Walk v w, ¬s(v, w) ∈ p.edges := by
  /-
    V : Type u
    G : SimpleGraph V
    v w : V
    ⊢ Iff ((SDiff.sdiff G (SimpleGraph.fromEdgeSet (Singleton.singleton (Sym2.mk { …
  -/
  constructor
    /-
      case mp
      V : Type u
      G : SimpleGraph V
      v w : V
      ⊢ (SDiff.sdiff G (SimpleGraph.fromEdgeSet (Singleton.singleton (Sym2.mk { fst  …
    -/
  · rintro ⟨p⟩
    /-
      case mp.intro
      V : Type u
      G : SimpleGraph V
      v w : V
      p : (SDiff.sdiff G (SimpleGraph.fromEdgeSet (Singleton.singleton (Sym2.mk { fs …
      ⊢ Exists fun p => Not (Membership.mem p.edges (Sym2.mk { fst := v, snd := w }))
    -/
    use p.map (Hom.mapSpanningSubgraphs (by simp))
    /-
      case h
      V : Type u
      G : SimpleGraph V
      v w : V
      p : (SDiff.sdiff G (SimpleGraph.fromEdgeSet (Singleton.singleton (Sym2.mk { fs …
      ⊢ Not (Membership.mem (SimpleGraph.Walk.map (SimpleGraph.Hom.mapSpanningSubgra …
    -/
    simp_rw [Walk.edges_map, List.mem_map, Hom.mapSpanningSubgraphs_apply, Sym2.map_id', id]
    /-
      case h
      V : Type u
      G : SimpleGraph V
      v w : V
      p : (SDiff.sdiff G (SimpleGraph.fromEdgeSet (Singleton.singleton (Sym2.mk { fs …
      ⊢ Not (Exists fun a => And (Membership.mem p.edges a) (Eq a (Sym2.mk { fst :=  …
    -/
    rintro ⟨e, h, rfl⟩
    /-
      case h.intro.intro
      V : Type u
      G : SimpleGraph V
      v w : V
      p : (SDiff.sdiff G (SimpleGraph.fromEdgeSet (Singleton.singleton (Sym2.mk { fs …
      h : Membership.mem p.edges (Sym2.mk { fst := v, snd := w })
      ⊢ False
    -/
    simpa using p.edges_subset_edgeSet h
    /-
      🎉 no goals
    -/
    /-
      case mpr
      V : Type u
      G : SimpleGraph V
      v w : V
      ⊢ (Exists fun p => Not (Membership.mem p.edges (Sym2.mk { fst := v, snd := w } …
    -/
  · rintro ⟨p, h⟩
    /-
      case mpr.intro
      V : Type u
      G : SimpleGraph V
      v w : V
      p : G.Walk v w
      h : Not (Membership.mem p.edges (Sym2.mk { fst := v, snd := w }))
      ⊢ (SDiff.sdiff G (SimpleGraph.fromEdgeSet (Singleton.singleton (Sym2.mk { fst  …
    -/
    refine ⟨p.transfer _ fun e ep => ?_⟩
    simp only [edgeSet_sdiff, edgeSet_fromEdgeSet, edgeSet_sdiff_sdiff_isDiag, Set.mem_diff,
      Set.mem_singleton_iff]
    /-
      case mpr.intro
      V : Type u
      G : SimpleGraph V
      v w : V
      p : G.Walk v w
      h : Not (Membership.mem p.edges (Sym2.mk { fst := v, snd := w }))
      e : Sym2 V
      ep : Membership.mem p.edges e
      ⊢ And (Membership.mem G.edgeSet e) (Not (Eq e (Sym2.mk { fst := v, snd := w })))
    -/
    exact ⟨p.edges_subset_edgeSet ep, fun h' => h (h' ▸ ep)⟩
    /-
      🎉 no goals
    -/


theorem isBridge_iff_adj_and_forall_walk_mem_edges {v w : V} :
    G.IsBridge s(v, w) ↔ G.Adj v w ∧ ∀ p : G.Walk v w, s(v, w) ∈ p.edges := by
  /-
    V : Type u
    G : SimpleGraph V
    v w : V
    ⊢ Iff (G.IsBridge (Sym2.mk { fst := v, snd := w })) (And (G.Adj v w) (∀ (p : G …
  -/
  rw [isBridge_iff, and_congr_right']
  /-
    V : Type u
    G : SimpleGraph V
    v w : V
    ⊢ Iff (Not ((SDiff.sdiff G (SimpleGraph.fromEdgeSet (Singleton.singleton (Sym2 …
  -/
  rw [reachable_delete_edges_iff_exists_walk, not_exists_not]
  /-
    🎉 no goals
  -/


theorem reachable_deleteEdges_iff_exists_cycle.aux [DecidableEq V] {u v w : V}
    (hb : ∀ p : G.Walk v w, s(v, w) ∈ p.edges) (c : G.Walk u u) (hc : c.IsTrail)
    (he : s(v, w) ∈ c.edges)
    (hw : w ∈ (c.takeUntil v (c.fst_mem_support_of_mem_edges he)).support) : False := by
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v w : V
    hb : ∀ (p : G.Walk v w), Membership.mem p.edges (Sym2.mk { fst := v, snd := w })
    c : G.Walk u u
    hc : c.IsTrail
    he : Membership.mem c.edges (Sym2.mk { fst := v, snd := w })
    hw : Membership.mem (c.takeUntil v ⋯).support w
    ⊢ False
  -/
  have hv := c.fst_mem_support_of_mem_edges he
  -- decompose c into
  --      puw     pwv     pvu
  --   u ----> w ----> v ----> u
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v w : V
    hb : ∀ (p : G.Walk v w), Membership.mem p.edges (Sym2.mk { fst := v, snd := w })
    c : G.Walk u u
    hc : c.IsTrail
    he : Membership.mem c.edges (Sym2.mk { fst := v, snd := w })
    hw : Membership.mem (c.takeUntil v ⋯).support w
    hv : Membership.mem c.support v
    ⊢ False
  -/
  let puw := (c.takeUntil v hv).takeUntil w hw
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v w : V
    hb : ∀ (p : G.Walk v w), Membership.mem p.edges (Sym2.mk { fst := v, snd := w })
    c : G.Walk u u
    hc : c.IsTrail
    he : Membership.mem c.edges (Sym2.mk { fst := v, snd := w })
    hw : Membership.mem (c.takeUntil v ⋯).support w
    hv : Membership.mem c.support v
    puw : G.Walk u w := (c.takeUntil v hv).takeUntil w hw
    ⊢ False
  -/
  let pwv := (c.takeUntil v hv).dropUntil w hw
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v w : V
    hb : ∀ (p : G.Walk v w), Membership.mem p.edges (Sym2.mk { fst := v, snd := w })
    c : G.Walk u u
    hc : c.IsTrail
    he : Membership.mem c.edges (Sym2.mk { fst := v, snd := w })
    hw : Membership.mem (c.takeUntil v ⋯).support w
    hv : Membership.mem c.support v
    puw : G.Walk u w := (c.takeUntil v hv).takeUntil w hw
    pwv : G.Walk w v := (c.takeUntil v hv).dropUntil w hw
    ⊢ False
  -/
  let pvu := c.dropUntil v hv
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v w : V
    hb : ∀ (p : G.Walk v w), Membership.mem p.edges (Sym2.mk { fst := v, snd := w })
    c : G.Walk u u
    hc : c.IsTrail
    he : Membership.mem c.edges (Sym2.mk { fst := v, snd := w })
    hw : Membership.mem (c.takeUntil v ⋯).support w
    hv : Membership.mem c.support v
    puw : G.Walk u w := (c.takeUntil v hv).takeUntil w hw
    pwv : G.Walk w v := (c.takeUntil v hv).dropUntil w hw
    pvu : G.Walk v u := c.dropUntil v hv
    ⊢ False
  -/
  have : c = (puw.append pwv).append pvu := by simp [puw, pwv, pvu]
  -- We have two walks from v to w
  --      pvu     puw
  --   v ----> u ----> w
  --   |               ^
  --    `-------------'
  --      pwv.reverse
  -- so they both contain the edge s(v, w), but that's a contradiction since c is a trail.
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v w : V
    hb : ∀ (p : G.Walk v w), Membership.mem p.edges (Sym2.mk { fst := v, snd := w })
    c : G.Walk u u
    hc : c.IsTrail
    he : Membership.mem c.edges (Sym2.mk { fst := v, snd := w })
    hw : Membership.mem (c.takeUntil v ⋯).support w
    hv : Membership.mem c.support v
    puw : G.Walk u w := (c.takeUntil v hv).takeUntil w hw
    pwv : G.Walk w v := (c.takeUntil v hv).dropUntil w hw
    pvu : G.Walk v u := c.dropUntil v hv
    this : Eq c ((puw.append pwv).append pvu)
    ⊢ False
  -/
  have hbq := hb (pvu.append puw)
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v w : V
    hb : ∀ (p : G.Walk v w), Membership.mem p.edges (Sym2.mk { fst := v, snd := w })
    c : G.Walk u u
    hc : c.IsTrail
    he : Membership.mem c.edges (Sym2.mk { fst := v, snd := w })
    hw : Membership.mem (c.takeUntil v ⋯).support w
    hv : Membership.mem c.support v
    puw : G.Walk u w := (c.takeUntil v hv).takeUntil w hw
    pwv : G.Walk w v := (c.takeUntil v hv).dropUntil w hw
    pvu : G.Walk v u := c.dropUntil v hv
    this : Eq c ((puw.append pwv).append pvu)
    hbq : Membership.mem (pvu.append puw).edges (Sym2.mk { fst := v, snd := w })
    ⊢ False
  -/
  have hpq' := hb pwv.reverse
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v w : V
    hb : ∀ (p : G.Walk v w), Membership.mem p.edges (Sym2.mk { fst := v, snd := w })
    c : G.Walk u u
    hc : c.IsTrail
    he : Membership.mem c.edges (Sym2.mk { fst := v, snd := w })
    hw : Membership.mem (c.takeUntil v ⋯).support w
    hv : Membership.mem c.support v
    puw : G.Walk u w := (c.takeUntil v hv).takeUntil w hw
    pwv : G.Walk w v := (c.takeUntil v hv).dropUntil w hw
    pvu : G.Walk v u := c.dropUntil v hv
    this : Eq c ((puw.append pwv).append pvu)
    hbq : Membership.mem (pvu.append puw).edges (Sym2.mk { fst := v, snd := w })
    hpq' : Membership.mem pwv.reverse.edges (Sym2.mk { fst := v, snd := w })
    ⊢ False
  -/
  rw [Walk.edges_reverse, List.mem_reverse] at hpq'
  rw [Walk.isTrail_def, this, Walk.edges_append, Walk.edges_append, List.nodup_append_comm,
    ← List.append_assoc, ← Walk.edges_append] at hc
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v w : V
    hb : ∀ (p : G.Walk v w), Membership.mem p.edges (Sym2.mk { fst := v, snd := w })
    c : G.Walk u u
    he : Membership.mem c.edges (Sym2.mk { fst := v, snd := w })
    hw : Membership.mem (c.takeUntil v ⋯).support w
    hv : Membership.mem c.support v
    puw : G.Walk u w := (c.takeUntil v hv).takeUntil w hw
    pwv : G.Walk w v := (c.takeUntil v hv).dropUntil w hw
    pvu : G.Walk v u := c.dropUntil v hv
    hc : (HAppend.hAppend (pvu.append puw).edges pwv.edges).Nodup
    this : Eq c ((puw.append pwv).append pvu)
    hbq : Membership.mem (pvu.append puw).edges (Sym2.mk { fst := v, snd := w })
    hpq' : Membership.mem pwv.edges (Sym2.mk { fst := v, snd := w })
    ⊢ False
  -/
  exact List.disjoint_of_nodup_append hc hbq hpq'
  /-
    🎉 no goals
  -/

-- Porting note: the unused variable checker helped eliminate a good amount of this proof (!)

theorem adj_and_reachable_delete_edges_iff_exists_cycle {v w : V} :
    G.Adj v w ∧ (G \ fromEdgeSet {s(v, w)}).Reachable v w ↔
      ∃ (u : V) (p : G.Walk u u), p.IsCycle ∧ s(v, w) ∈ p.edges := by
  classical
  rw [reachable_delete_edges_iff_exists_walk]
  constructor
  · rintro ⟨h, p, hp⟩
    refine ⟨w, Walk.cons h.symm p.toPath, ?_, ?_⟩
    · apply Path.cons_isCycle
      rw [Sym2.eq_swap]
      intro h
      cases hp (Walk.edges_toPath_subset p h)
    · simp only [Sym2.eq_swap, Walk.edges_cons, List.mem_cons, eq_self_iff_true, true_or]
  · rintro ⟨u, c, hc, he⟩
    refine ⟨c.adj_of_mem_edges he, ?_⟩
    by_contra! hb
    have hb' : ∀ p : G.Walk w v, s(w, v) ∈ p.edges := by
      intro p
      simpa [Sym2.eq_swap] using hb p.reverse
    have hvc : v ∈ c.support := Walk.fst_mem_support_of_mem_edges c he
    refine reachable_deleteEdges_iff_exists_cycle.aux hb' (c.rotate hvc) (hc.isTrail.rotate hvc)
      ?_ (Walk.start_mem_support _)
    rwa [(Walk.rotate_edges c hvc).mem_iff, Sym2.eq_swap]


theorem isBridge_iff_adj_and_forall_cycle_not_mem {v w : V} : G.IsBridge s(v, w) ↔
    G.Adj v w ∧ ∀ ⦃u : V⦄ (p : G.Walk u u), p.IsCycle → s(v, w) ∉ p.edges := by
  /-
    V : Type u
    G : SimpleGraph V
    v w : V
    ⊢ Iff (G.IsBridge (Sym2.mk { fst := v, snd := w })) (And (G.Adj v w) (∀ ⦃u : V …
  -/
  rw [isBridge_iff, and_congr_right_iff]
  /-
    V : Type u
    G : SimpleGraph V
    v w : V
    ⊢ G.Adj v w → Iff (Not ((SDiff.sdiff G (SimpleGraph.fromEdgeSet (Singleton.sin …
  -/
  intro h
  /-
    V : Type u
    G : SimpleGraph V
    v w : V
    h : G.Adj v w
    ⊢ Iff (Not ((SDiff.sdiff G (SimpleGraph.fromEdgeSet (Singleton.singleton (Sym2 …
  -/
  rw [← not_iff_not]
  /-
    V : Type u
    G : SimpleGraph V
    v w : V
    h : G.Adj v w
    ⊢ Iff (Not (Not ((SDiff.sdiff G (SimpleGraph.fromEdgeSet (Singleton.singleton  …
  -/
  push_neg
  /-
    V : Type u
    G : SimpleGraph V
    v w : V
    h : G.Adj v w
    ⊢ Iff ((SDiff.sdiff G (SimpleGraph.fromEdgeSet (Singleton.singleton (Sym2.mk { …
  -/
  rw [← adj_and_reachable_delete_edges_iff_exists_cycle]
  /-
    V : Type u
    G : SimpleGraph V
    v w : V
    h : G.Adj v w
    ⊢ Iff ((SDiff.sdiff G (SimpleGraph.fromEdgeSet (Singleton.singleton (Sym2.mk { …
  -/
  simp only [h, true_and]
  /-
    🎉 no goals
  -/


theorem isBridge_iff_mem_and_forall_cycle_not_mem {e : Sym2 V} :
    G.IsBridge e ↔ e ∈ G.edgeSet ∧ ∀ ⦃u : V⦄ (p : G.Walk u u), p.IsCycle → e ∉ p.edges :=
  Sym2.ind (fun _ _ => isBridge_iff_adj_and_forall_cycle_not_mem) e


